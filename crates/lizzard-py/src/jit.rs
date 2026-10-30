//! Running functions compiled by `lizzard-jit`.
//!
//! A function is compiled once it is warm (unless `LIZZARD_JIT=0`): on its
//! `LIZZARD_JIT_WARM`-th call (2 by default), or when one of its loops has
//! run `LIZZARD_JIT_LOOPS` iterations in the interpreter, which then
//! switches to the compiled code in the middle of the loop. By then the
//! interpreter's caches have seen the types, attributes, globals and callees
//! the code works with, and the compiler specializes on them (`specs`).
//!
//! Compiled code works on the same register slots as the interpreter, and
//! hands a frame back to it when a specialization does not hold (`BAIL`),
//! so the two mix freely; generators run compiled too. Instructions it does
//! not compile inline call `step` below, which runs them with the
//! interpreter's own code.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use lizzard_jit::{CallGuard, CallSpec, Entry, Helpers, Layout, Num, Quick, Spec, BAIL, ERROR, EXIT, NEXT, TAKEN, YIELD, YIELD_INT};
use lizzard_py_front::ir::{Const, Instr, ParamKind};
use pyo3::ffi;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::frame::Regs;
use crate::interp::{lizzard_function_type, run_interp, step, Exit, FuncRt, LizzardFunction};
use crate::ops::*;
use crate::rt::Ck;

/// A function's compiled code and what decides when to compile it.
#[derive(Default)]
pub struct JitCell {
    /// The entry point (0 until compiled).
    entry: AtomicUsize,
    calls: AtomicU32,
    loops: AtomicU32,
    /// Bail-outs of the current code; many recompile it with what the
    /// caches see by then (at most `MAX_COMPILES` times).
    bails: AtomicU32,
    compiles: AtomicU32,
    /// It cannot be compiled.
    failed: AtomicBool,
    /// See `lizzard_jit::Unit::misses`.
    misses: AtomicU32,
    /// Bail-outs made it recompile: no unboxed registers across jumps.
    bail_recompiled: AtomicBool,
    /// How many instructions the current code has specs for.
    nspecs: AtomicU32,
    lock: Mutex<()>,
    /// What the code refers to (callee descriptions), alive as long as the
    /// function.
    deps: Mutex<Vec<Box<CallDesc>>>,
}

const MAX_COMPILES: u32 = 4;
const BAILS_TO_RECOMPILE: u32 = 1000;

/// A function compiled code calls directly (`Helpers::enter`, or its
/// code with `target` for the callee's context).
pub struct CallDesc {
    func: Py<LizzardFunction>,
    target: Target<'static, 'static>,
}

impl CallDesc {
    fn new(func: Py<LizzardFunction>) -> Box<CallDesc> {
        // SAFETY: the function object (on Python's heap, alive while the
        // description holds it) does not move.
        let lf: &'static LizzardFunction = unsafe { &*(func.get() as *const LizzardFunction) };
        let py = unsafe { Python::assume_attached() };
        let target = Target { py, f: &lf.rt, globals: lf.globals.bind(py), closure: &lf.closure };
        Box::new(CallDesc { func, target })
    }
}

// SAFETY: the target refers to the function object the description keeps;
// it is only used by threads attached to the interpreter.
unsafe impl Send for CallDesc {}
unsafe impl Sync for CallDesc {}

/// The function a frame runs, for compiled code's helpers.
pub(crate) struct Target<'a, 'py> {
    py: Python<'py>,
    f: &'a FuncRt,
    globals: &'a Bound<'py, PyDict>,
    closure: &'a [Py<PyAny>],
}

/// What compiled code's helpers need, behind its opaque `ctx` pointer: the
/// fields compiled code reads or writes (`lizzard_jit::CTX_TID`, ...), all
/// one machine word, so that compiled code builds a callee's context
/// itself.
#[repr(C)]
pub(crate) struct Ctx<'a, 'py> {
    tid: usize,
    /// The value `return` returns (a new reference).
    ret: *mut ffi::PyObject,
    slots: *mut u64,
    target: *const Target<'a, 'py>,
    /// How `step` left the function (a yield).
    exit: Option<Box<Exit<'py>>>,
    /// The caller takes a yielded int as it is (`YIELD_INT`).
    want_int: usize,
}

const _: () = assert!(std::mem::size_of::<Ctx>() == lizzard_jit::CTX_SIZE as usize);

impl<'a, 'py> Ctx<'a, 'py> {
    fn new(target: &Target<'a, 'py>, slots: *mut u64) -> Self {
        Ctx { tid: crate::rc::tid(), ret: std::ptr::null_mut(), slots, target, exit: None, want_int: 0 }
    }

    fn t(&self) -> &'a Target<'a, 'py> {
        // SAFETY: the target outlives the frame's code.
        unsafe { &*self.target }
    }

    /// A view of the frame's registers.
    fn regs(&self) -> Regs {
        // SAFETY: the frame's slots, one per register of `f`; nothing else
        // uses them while a helper runs.
        unsafe { Regs::window(self.slots, &self.t().f.objs) }
    }
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Whether functions are compiled: `LIZZARD_JIT=0` (or `off`) runs
/// everything in the interpreter.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("LIZZARD_JIT").as_deref(), Ok("0") | Ok("off")))
}

fn warm_calls() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| env_u32("LIZZARD_JIT_WARM", 2))
}

fn warm_loops() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| env_u32("LIZZARD_JIT_LOOPS", 200))
}

#[inline]
pub(crate) fn loaded(f: &FuncRt) -> Option<Entry> {
    let e = f.jit.entry.load(Ordering::Acquire);
    // SAFETY: only compiled entries are stored.
    (e != 0).then(|| unsafe { std::mem::transmute::<usize, Entry>(e) })
}

/// The function's compiled code for a call: compiled on the call that makes
/// it warm. None while it runs in the interpreter.
#[inline]
pub fn entry(py: Python<'_>, f: &FuncRt) -> Option<Entry> {
    if let Some(e) = loaded(f) {
        return Some(e);
    }
    if !enabled() || f.jit.failed.load(Ordering::Relaxed) {
        return None;
    }
    if f.jit.calls.fetch_add(1, Ordering::Relaxed) + 1 < warm_calls() {
        return None;
    }
    compile(py, f)
}

/// The compiled code to continue a loop in, once the loop is warm.
#[inline]
pub fn osr(py: Python<'_>, f: &FuncRt) -> Option<Entry> {
    if let Some(e) = loaded(f) {
        return Some(e);
    }
    if !enabled() || f.jit.failed.load(Ordering::Relaxed) {
        return None;
    }
    if f.jit.loops.fetch_add(1, Ordering::Relaxed) + 1 < warm_loops() {
        return None;
    }
    compile(py, f)
}

/// The code bailed out: count it, and after many, recompile with what the
/// caches see now.
fn bailed(f: &FuncRt) {
    if f.jit.bails.fetch_add(1, Ordering::Relaxed) + 1 >= BAILS_TO_RECOMPILE && f.jit.compiles.load(Ordering::Relaxed) < MAX_COMPILES {
        if debug() {
            eprintln!("lizzard: {} bails out often: recompiling", f.ir.name);
        }
        f.jit.bails.store(0, Ordering::Relaxed);
        f.jit.bail_recompiled.store(true, Ordering::Relaxed);
        f.jit.calls.store(0, Ordering::Relaxed);
        f.jit.loops.store(0, Ordering::Relaxed);
        // The old code stays valid for frames running it.
        f.jit.entry.store(0, Ordering::Release);
    }
}

fn debug() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LIZZARD_JIT_DEBUG").is_some())
}

fn compile(py: Python<'_>, f: &FuncRt) -> Option<Entry> {
    // Another thread compiling it: run interpreted meanwhile (waiting here
    // while attached could deadlock with that thread).
    let _g = f.jit.lock.try_lock().ok()?;
    if let Some(e) = loaded(f) {
        return Some(e);
    }
    f.jit.compiles.fetch_add(1, Ordering::Relaxed);
    let ir = &f.ir;
    if debug() {
        eprintln!("lizzard: compiling {}", ir.name);
    }
    let (rc, dealloc, merge_zero, dec_shared) = crate::rc::jit_model();
    let a = |p: *const ()| p as usize;
    let helpers = Helpers {
        step: a(lz_step as *const ()),
        catch: a(lz_catch as *const ()),
        safepoint: a(lz_safepoint as *const ()),
        load_attr: a(lz_load_attr as *const ()),
        store_attr: a(lz_store_attr as *const ()),
        truthy: a(lz_truthy as *const ()),
        binop: a(lz_binop as *const ()),
        compare: a(lz_compare as *const ()),
        compare_bool: a(lz_compare_bool as *const ()),
        call: a(lz_call as *const ()),
        for_iter: a(lz_for_iter as *const ()),
        for_iter_int: a(lz_for_iter_int as *const ()),
        check: a(lz_check as *const ()),
        load_global: a(lz_load_global as *const ()),
        enter: a(lz_enter as *const ()),
        incref: a(ffi::Py_IncRef as *const ()),
        decref: a(ffi::Py_DecRef as *const ()),
        dealloc,
        merge_zero,
        dec_shared,
        float_new: a(ffi::PyFloat_FromDouble as *const ()),
        long_new: a(ffi::PyLong_FromLongLong as *const ()),
        long_value: a(lz_long_value as *const ()),
        missed: a(lz_missed as *const ()),
        init_returned: a(lz_init_returned as *const ()),
        finish: a(lz_finish as *const ()),
        vectorcall: a(ffi::PyObject_Vectorcall as *const ()),
        enter_recursive: a(ffi::Py_EnterRecursiveCall as *const ()),
        leave_recursive: a(ffi::Py_LeaveRecursiveCall as *const ()),
        recursion_msg: c" while calling a Lizzard function".as_ptr() as usize,
        tail_depth: a(lz_tail_depth as *const ()),
        resume: a(lz_resume as *const ()),
        take_exit: a(lz_take_exit as *const ()),
        unwind: a(lz_unwind as *const ()),
    };
    let head = std::mem::size_of::<ffi::PyObject>() as i32;
    let layout = Layout {
        ob_type: std::mem::offset_of!(ffi::PyObject, ob_type) as i32,
        tp_version: std::mem::offset_of!(ffi::PyTypeObject, tp_version_tag) as i32,
        rc,
        numbers: fast_paths_on(),
        float_type: &raw mut ffi::PyFloat_Type as usize,
        long_type: &raw mut ffi::PyLong_Type as usize,
        fval: head,
        long_tag: head,
        long_digit: head + 8,
        // SAFETY: the singletons.
        none: unsafe { ffi::Py_None() } as usize,
        true_: unsafe { ffi::Py_True() } as usize,
        false_: unsafe { ffi::Py_False() } as usize,
        epoch: crate::globals::epoch_addr(),
        small_ints: small_ints(),
        list_iter: list_iter(),
    };
    let mut deps = f.jit.deps.try_lock().ok()?;
    let unit = unit(py, f, &mut deps, 0);
    f.jit.nspecs.store(count(&unit.specs), Ordering::Relaxed);
    f.jit.misses.store(0, Ordering::Relaxed);
    match lizzard_jit::compile(&unit, &helpers, &layout) {
        Ok(e) => {
            f.jit.entry.store(e as usize, Ordering::Release);
            Some(e)
        }
        Err(e) => {
            // `LIZZARD_JIT_DEBUG=strict`: the tests' setting.
            if std::env::var("LIZZARD_JIT_DEBUG").as_deref() == Ok("strict") {
                panic!("lizzard: cannot compile {}: {e}", ir.name);
            }
            if debug() {
                eprintln!("lizzard: cannot compile {}: {e}", ir.name);
            }
            f.jit.failed.store(true, Ordering::Relaxed);
            None
        }
    }
}

/// Whether constructing an instance of `tp` may take one about to be
/// released (`lizzard_jit::Reuse`): no finalizer, a known weak reference
/// list, and an `__init__` (`init`) that only assigns inline-value
/// attributes of the instance, the same ones each time.
fn reuse(tp: *mut ffi::PyTypeObject, init: &lizzard_jit::Unit) -> Option<lizzard_jit::Reuse> {
    // SAFETY: a class (the cache keeps it alive).
    unsafe {
        if (*tp).tp_finalize.is_some() || (*tp).tp_del.is_some() {
            return None;
        }
    }
    let weakref = weakref_offset(tp)?;
    let ir = init.ir;
    let mut values = None;
    let mut order = Vec::new();
    let mut members = Vec::new();
    for (pc, i) in ir.code.iter().enumerate() {
        if !matches!(i, Instr::Return { .. }) && i.successors(pc as u32) != [pc as u32 + 1] {
            return None;
        }
        if !i.reads().contains(&0) {
            continue;
        }
        if let (Instr::StoreAttr { obj: 0, val, .. }, Spec::Member { off, .. }) = (i, &init.specs[pc]) {
            if *val == 0 || values.is_some() {
                return None;
            }
            if !members.contains(off) {
                members.push(*off);
            }
            continue;
        }
        let (Instr::StoreAttr { obj: 0, val, .. }, Spec::Inline { off, values: v, .. }) = (i, &init.specs[pc]) else { return None };
        if *val == 0 || values.is_some_and(|x| x != *v) || !members.is_empty() {
            return None;
        }
        values = Some(*v);
        let ix = ((off - v - 8) / 8) as u8;
        if !order.contains(&ix) {
            order.push(ix);
        }
    }
    if !members.is_empty() {
        // Every slot of the class (its base is `object`) is assigned.
        // SAFETY: a class.
        let all = unsafe {
            if (*tp).tp_base != &raw mut ffi::PyBaseObject_Type || (*tp).tp_members.is_null() {
                return None;
            }
            let mut n = 0;
            let mut m = (*tp).tp_members;
            while !(*m).name.is_null() {
                if (*m).type_code != ffi::Py_T_OBJECT_EX || !members.contains(&((*m).offset as i32)) {
                    return None;
                }
                n += 1;
                m = m.add(1);
            }
            n
        };
        return (all == members.len()).then_some(lizzard_jit::Reuse { weakref, values: None, order });
    }
    Some(lizzard_jit::Reuse { weakref, values: Some(values?), order })
}

fn count(specs: &[Spec]) -> u32 {
    specs.iter().filter(|s| !matches!(s, Spec::None | Spec::Check(_))).count() as u32
}

/// Sites compiled without specs keep running their generic paths: if the
/// caches know more by now, recompile (once the function is warm again).
extern "C" fn lz_missed(ctx_: *mut c_void) {
    let ctx = ctx(ctx_);
    let f = ctx.t().f;
    if f.jit.compiles.load(Ordering::Relaxed) >= MAX_COMPILES {
        return;
    }
    let mut deps = Vec::new();
    let now = count(&specs(ctx.t().py, f, &mut deps, 1));
    if now > f.jit.nspecs.load(Ordering::Relaxed) {
        if debug() {
            eprintln!("lizzard: {} has more cache data: recompiling", f.ir.name);
        }
        f.jit.calls.store(0, Ordering::Relaxed);
        f.jit.loops.store(0, Ordering::Relaxed);
        f.jit.entry.store(0, Ordering::Release);
    }
}

/// The layout of list iterators, if CPython's agrees with what compiled
/// code reads (checked once on a list being iterated).
/// The list iterator's type, if compiled code reads list iterators.
pub(crate) fn list_iter_type() -> Option<usize> {
    list_iter().map(|l| l.tp)
}

fn list_iter() -> Option<lizzard_jit::ListIter> {
    static LAYOUT: OnceLock<Option<lizzard_jit::ListIter>> = OnceLock::new();
    *LAYOUT.get_or_init(|| {
        let head = std::mem::size_of::<ffi::PyObject>();
        let l = lizzard_jit::ListIter { tp: 0, index: head as i32, seq: head as i32 + 8, size: head as i32, items: head as i32 + 8, end_index: false };
        // SAFETY: attached (compiling); the objects are released again.
        unsafe {
            let a = ffi::PyLong_FromLongLong(1000);
            let b = ffi::PyLong_FromLongLong(2000);
            let list = ffi::PyList_New(0);
            ffi::PyList_Append(list, a);
            ffi::PyList_Append(list, b);
            let it = ffi::PyObject_GetIter(list);
            let first = ffi::PyIter_Next(it);
            let at = |o: *mut ffi::PyObject, off: i32| *((o as *mut u8).add(off as usize) as *const usize);
            let ok = !first.is_null()
                && first == a
                && at(it, l.index) == 1
                && at(it, l.seq) == list as usize
                && at(list, l.size) == 2
                && *(at(list, l.items) as *const *mut ffi::PyObject).add(1) == b;
            // How the end is marked: no list, or (free threading) no index.
            let second = ffi::PyIter_Next(it);
            let past = ffi::PyIter_Next(it);
            let end_index = at(it, l.seq) != 0;
            let ok = ok && second == b && past.is_null() && ffi::PyErr_Occurred().is_null() && (if end_index { at(it, l.index) == usize::MAX && at(it, l.seq) == list as usize } else { true });
            let tp = ffi::Py_TYPE(it) as usize;
            for o in [first, second, it, list, a, b] {
                if !o.is_null() {
                    ffi::Py_DecRef(o);
                }
            }
            ok.then_some(lizzard_jit::ListIter { tp, end_index, ..l })
        }
    })
}

/// Where CPython keeps its preallocated small ints (-5..=256), if they are
/// immortal objects at evenly spaced addresses (checked once).
fn small_ints() -> Option<lizzard_jit::SmallInts> {
    static SMALL: OnceLock<Option<lizzard_jit::SmallInts>> = OnceLock::new();
    *SMALL.get_or_init(|| {
        let (min, max) = (-5i64, 256i64);
        // SAFETY: attached (compiling); the ints are released again.
        unsafe {
            let at = |v: i64| {
                let p = ffi::PyLong_FromLongLong(v);
                let immortal = !p.is_null() && ffi::Py_REFCNT(p) > (1 << 29);
                if !p.is_null() {
                    ffi::Py_DecRef(p);
                }
                immortal.then_some(p as i64)
            };
            let first = at(min)?;
            let stride = at(min + 1)? - first;
            if stride <= 0 {
                return None;
            }
            for v in min..=max {
                if at(v)? != first + (v - min) * stride {
                    return None;
                }
            }
            Some(lizzard_jit::SmallInts { first: first as usize, stride, min, count: max - min + 1 })
        }
    })
}

/// A check by type pointers, for compiled code.
fn quick(py: Python<'_>, ck: &Ck) -> Quick {
    // SAFETY: the type objects.
    unsafe {
        match ck {
            Ck::Any => Quick::Any,
            Ck::Int => Quick::Type(&raw mut ffi::PyLong_Type as usize),
            Ck::Float => Quick::FloatOrInt,
            Ck::Bool => Quick::Bool,
            Ck::Str => Quick::Type(&raw mut ffi::PyUnicode_Type as usize),
            Ck::None => Quick::Is(ffi::Py_None() as usize),
            Ck::Tuple => Quick::Type(&raw mut ffi::PyTuple_Type as usize),
            // The check keeps the class alive.
            Ck::Class(c) => Quick::Type(c.bind(py).as_ptr() as usize),
            _ => Quick::Slow,
        }
    }
}

/// What compiling `f` needs, with what its caches saw; `depth` 0 for the
/// function compiled, 1 for one inlined into it (which inlines nothing).
fn unit<'a>(py: Python<'_>, f: &'a FuncRt, deps: &mut Vec<Box<CallDesc>>, depth: u32) -> lizzard_jit::Unit<'a> {
    let ir = &f.ir;
    let nums = ir
        .consts
        .iter()
        .map(|c| match c {
            Const::Float(x) => Num::F64(*x),
            Const::Int(i) => Num::I64(*i),
            _ => Num::Other,
        })
        .collect();
    lizzard_jit::Unit {
        ir,
        specs: specs(py, f, deps, depth),
        consts: f.consts.iter().map(|c| c.as_ptr() as usize).collect(),
        nums,
        names: f.names.iter().map(|n| n.as_ptr() as usize).collect(),
        ics: f.ics.iter().map(|c| c as *const AttrIc as usize).collect(),
        borrow_params: f.borrow_params,
        misses: &f.jit.misses as *const AtomicU32 as usize,
        speculate: !f.jit.bail_recompiled.load(Ordering::Relaxed),
    }
}

/// What the caches saw, per instruction.
fn specs<'a>(py: Python<'_>, f: &'a FuncRt, deps: &mut Vec<Box<CallDesc>>, depth: u32) -> Vec<Spec<'a>> {
    let ir = &f.ir;
    let mut out = Vec::with_capacity(ir.code.len());
    // The global each register was last loaded from, in straight-line code.
    let mut global_of: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (pc, i) in ir.code.iter().enumerate() {
        let spec = match i {
            Instr::LoadAttr { .. } => attr_spec(&f.ics[pc], false),
            Instr::StoreAttr { .. } => attr_spec(&f.ics[pc], true),
            Instr::LoadGlobal { .. } => match crate::globals::spec(&f.gics[pc]) {
                Some((epoch, v)) => Spec::Global { epoch, value: v as usize },
                None => Spec::None,
            },
            Instr::Check { check, .. } => Spec::Check(quick(py, f.ck(py, *check))),
            Instr::ForIter { .. } => Spec::Iter(f.iters[pc].load(std::sync::atomic::Ordering::Relaxed)),
            // The global still this function: compiled code loops.
            Instr::JumpIfNotSelf { func, .. } => match global_of.get(func) {
                // SAFETY: a valid object (the global's).
                Some(&callee) if unsafe { ffi::Py_TYPE(callee as *mut ffi::PyObject) } == lizzard_function_type() => {
                    // SAFETY: the type says what it is.
                    let lf = unsafe { Bound::from_borrowed_ptr(py, callee as *mut ffi::PyObject).cast_into_unchecked::<LizzardFunction>() };
                    let lf = lf.get();
                    if std::ptr::eq(&*lf.rt, f) && lf.closure.is_empty() { Spec::SelfFunc(callee) } else { Spec::None }
                }
                _ => Spec::None,
            },
            Instr::Call { func, args, kwnames, direct, .. } if kwnames.is_empty() => match global_of.get(func) {
                // SAFETY: a valid object (the global's).
                Some(&callee) if unsafe { ffi::Py_TYPE(callee as *mut ffi::PyObject) } == &raw mut ffi::PyType_Type => match ctor_spec(&f.ctics[pc]) {
                    // A class constructed by allocating and `__init__`.
                    Some((tp, version, init)) if tp as usize == callee => match call_spec(py, f, pc, init, args.len() + 1, 0, deps, depth) {
                        // SAFETY: a class.
                        Some((_, spec)) => {
                            let reuse = spec.inline.as_deref().and_then(|init| reuse(tp, init));
                            Spec::Call(CallSpec { guard: CallGuard::Ctor { tp: tp as usize, version, alloc: unsafe { (*tp).tp_alloc }.map_or(0, |a| a as *const () as usize), reuse }, ..spec })
                        }
                        None => Spec::None,
                    },
                    _ => Spec::None,
                },
                Some(&callee) => {
                    let skip = direct.as_ref().map_or(0, |d| d.skip);
                    call_spec(py, f, pc, callee as *mut ffi::PyObject, args.len(), skip, deps, depth).map_or(Spec::None, |(_, spec)| Spec::Call(CallSpec { guard: CallGuard::Func(callee), ..spec }))
                }
                None => Spec::None,
            },
            Instr::CallMethod { args, kwnames, direct, .. } if kwnames.is_empty() => match method_spec(&f.mics[pc]) {
                Some((tp, version, func, keys, nentries, valid)) => {
                    let skip = direct.as_ref().map_or(0, |d| d.skip);
                    let guard = CallGuard::Method { tp, version, keys, nentries, valid };
                    match call_spec(py, f, pc, func, args.len() + 1, skip, deps, depth) {
                        Some((_, spec)) => Spec::Call(CallSpec { guard, ..spec }),
                        // A builtin method (alive while the type's version
                        // holds): called with the arguments directly.
                        // SAFETY: a valid object.
                        None if unsafe { ffi::Py_TYPE(func) } == &raw mut ffi::PyMethodDescr_Type => {
                            let n = args.len() + 1;
                            Spec::Call(CallSpec { guard, desc: func as usize, nregs: n as u32, checks: vec![Quick::Any; n], inline: None, native: true, entry: 0, target: 0 })
                        }
                        None => Spec::None,
                    }
                }
                None => Spec::None,
            },
            _ => Spec::None,
        };
        // Which registers hold a global's (cached) value.
        if let (Instr::LoadGlobal { dst, .. }, Spec::Global { value, .. }) = (i, &spec) {
            global_of.insert(*dst, *value);
        } else {
            for w in i.writes() {
                global_of.remove(&w);
            }
        }
        if i.successors(pc as u32) != [pc as u32 + 1] {
            global_of.clear();
        }
        out.push(spec);
    }
    out
}

/// A direct call of `callee` (if it is a Lizzard function compiled code can
/// enter: plain positional parameters, `nargs` of them, not a generator)
/// from site `pc`; its guard is left for the caller to fill in.
#[allow(clippy::too_many_arguments)]
fn call_spec<'a>(py: Python<'_>, site: &FuncRt, pc: usize, callee: *mut ffi::PyObject, nargs: usize, skip: u64, deps: &mut Vec<Box<CallDesc>>, depth: u32) -> Option<((), CallSpec<'a>)> {
    // SAFETY: a valid object (kept alive by its cache's owner).
    if unsafe { ffi::Py_TYPE(callee) } != lizzard_function_type() {
        return None;
    }
    // SAFETY: the type says what it is.
    let func: Py<LizzardFunction> = unsafe { Bound::from_borrowed_ptr(py, callee).cast_into_unchecked::<LizzardFunction>().unbind() };
    let rt = func.get().rt.clone();
    let ir = &rt.ir;
    if !rt.positional || ir.generator || ir.params.len() != nargs || ir.params.iter().any(|p| !matches!(p.kind, ParamKind::PosOnly | ParamKind::PosOrKw)) {
        return None;
    }
    // The checks the site's static types prove, when it is the function
    // the checker resolved.
    let proven = match site.cics[pc].get() {
        Some(known) if Arc::ptr_eq(known, &rt) => skip,
        _ => 0,
    };
    let checks = ir.params.iter().enumerate().map(|(i, p)| if proven & (1 << i) != 0 { Quick::Any } else { quick(py, rt.ck(py, p.check)) }).collect();
    let nregs = ir.reprs.len() as u32;
    // SAFETY: the description keeps the function alive as long as the
    // caller's code (`deps`), longer than compiling it.
    let callee_rt: &'a FuncRt = unsafe { &*Arc::as_ptr(&rt) };
    let desc = CallDesc::new(func);
    let addr = &*desc as *const CallDesc as usize;
    // Called directly when it never writes its parameters (they are the
    // caller's).
    let (entry, target) = if callee_rt.borrow_params { (&callee_rt.jit.entry as *const AtomicUsize as usize, &desc.target as *const Target as usize) } else { (0, 0) };
    deps.push(desc);
    // A small callee, warm (it ran), may be compiled into the caller.
    // Two levels at most, the second one for tiny callees (an `__init__`).
    let small = match depth {
        0 => callee_rt.ir.code.len() <= lizzard_jit::INLINE_MAX,
        1 => callee_rt.ir.code.len() <= lizzard_jit::INLINE_MAX_NESTED,
        _ => false,
    };
    let inline = (small && callee_rt.jit.calls.load(Ordering::Relaxed) + callee_rt.jit.compiles.load(Ordering::Relaxed) > 0)
        .then(|| Box::new(unit(py, callee_rt, deps, depth + 1)))
        .filter(|u| lizzard_jit::inlinable(u));
    Some(((), CallSpec { guard: CallGuard::Func(0), desc: addr, nregs, checks, inline, native: false, entry, target }))
}

/// Run a frame from `pc` in compiled code until it returns or yields, as
/// `interp::exec` does; a bail-out continues in the interpreter. With
/// `pending`, the instruction at `pc` raises it.
#[allow(clippy::too_many_arguments)]
pub fn exec<'py>(
    py: Python<'py>,
    entry: Entry,
    f: &FuncRt,
    globals: &Bound<'py, PyDict>,
    closure: &[Py<PyAny>],
    regs: &mut Regs,
    pc: &mut usize,
    pending: Option<PyErr>,
) -> PyResult<Exit<'py>> {
    let ran = run(py, entry, f, globals, closure, regs, pc, pending);
    settle(py, f, globals, closure, regs, pc, ran)
}

/// How a run of compiled code ends the call: as it stopped, or the
/// interpreter continues it.
pub(crate) fn settle<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, closure: &[Py<PyAny>], regs: &mut Regs, pc: &mut usize, ran: Ran<'py>) -> PyResult<Exit<'py>> {
    match ran {
        Ran::Exit(e) => Ok(e),
        Ran::Bail => run_interp(py, f, globals, closure, regs, pc, None),
        Ran::Err(e) => {
            let ir = &f.ir;
            let line = ir.lines.get(*pc).copied().unwrap_or(ir.line);
            let _ = e.add_note(py, format!("lizzard: {}:{line} in {}", ir.file, ir.name));
            Err(e)
        }
        Ran::YieldInt { .. } => unreachable!("an int yielded to a caller that takes objects"),
    }
}

/// How compiled code stopped.
pub enum Ran<'py> {
    Exit(Exit<'py>),
    /// A generator yielded an int as it is (asked for: `run_with`).
    YieldInt { value: i64, dst: u32 },
    /// The interpreter continues at the pc.
    Bail,
    /// An exception no handler of the function caught.
    Err(PyErr),
}

/// Run compiled code on a frame from `*pc` (left where it stopped).
#[allow(clippy::too_many_arguments)]
pub fn run<'py>(py: Python<'py>, entry: Entry, f: &FuncRt, globals: &Bound<'py, PyDict>, closure: &[Py<PyAny>], regs: &mut Regs, pc: &mut usize, pending: Option<PyErr>) -> Ran<'py> {
    run_with(py, entry, f, globals, closure, regs, pc, pending, false)
}

/// `run`; `want_int`: a `yield` of an unboxed int gives `Ran::YieldInt`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn run_with<'py>(py: Python<'py>, entry: Entry, f: &FuncRt, globals: &Bound<'py, PyDict>, closure: &[Py<PyAny>], regs: &mut Regs, pc: &mut usize, pending: Option<PyErr>, want_int: bool) -> Ran<'py> {
    let slots = regs.as_mut_ptr();
    let target = Target { py, f, globals, closure };
    let mut ctx = Ctx::new(&target, slots);
    ctx.want_int = want_int as usize;
    let throw = pending.is_some();
    if let Some(e) = pending {
        e.restore(py);
    }
    let mut p = *pc as u32;
    // SAFETY: `entry` was compiled from `f.ir`; `slots` has a slot for each
    // of its registers, and `ctx` outlives the call.
    let status = unsafe { entry(&mut ctx as *mut Ctx as *mut c_void, slots, &mut p, throw as u32) };
    *pc = p as usize;
    match status {
        // SAFETY: compiled code stored a new reference.
        EXIT if !ctx.ret.is_null() => Ran::Exit(Exit::Return(unsafe { Bound::from_owned_ptr(py, ctx.ret) })),
        EXIT => Ran::Exit(*ctx.exit.take().expect("compiled code exits with a value")),
        YIELD => {
            let Instr::Yield { dst, .. } = f.ir.code[*pc] else { unreachable!("a yield's status at another instruction") };
            // SAFETY: compiled code stored a new reference.
            Ran::Exit(Exit::Yield { value: unsafe { Bound::from_owned_ptr(py, ctx.ret) }, dst })
        }
        YIELD_INT => {
            let Instr::Yield { dst, .. } = f.ir.code[*pc] else { unreachable!("a yield's status at another instruction") };
            Ran::YieldInt { value: ctx.ret as i64, dst }
        }
        BAIL => {
            bailed(f);
            Ran::Bail
        }
        _ => Ran::Err(PyErr::fetch(py)),
    }
}

fn ctx<'c>(ctx: *mut c_void) -> &'c mut Ctx<'c, 'c> {
    // SAFETY: compiled code passes the `Ctx` it was given; nothing else
    // uses it while a helper runs.
    unsafe { &mut *(ctx as *mut Ctx<'c, 'c>) }
}

/// Execute instruction `pc` for compiled code.
extern "C" fn lz_step(ctx_: *mut c_void, _slots: *mut u64, pc: u32) -> u32 {
    let ctx = ctx(ctx_);
    let mut regs = ctx.regs();
    let py = ctx.t().py;
    let mut next = pc as usize;
    let r = catch_unwind(AssertUnwindSafe(|| step(py, ctx.t().f, ctx.t().globals, ctx.t().closure, &mut regs, &mut next)));
    match r {
        Ok(Ok(None)) if next == pc as usize + 1 => NEXT,
        Ok(Ok(None)) => TAKEN,
        Ok(Ok(Some(exit))) => {
            ctx.exit = Some(Box::new(exit));
            EXIT
        }
        Ok(Err(e)) => {
            e.restore(py);
            ERROR
        }
        Err(_) => {
            PanicException::new_err("panic in a Lizzard instruction").restore(py);
            ERROR
        }
    }
}

/// A frame's `d`th self tail call (`interp::tail_depth`).
extern "C" fn lz_tail_depth(d: i64) -> u32 {
    match crate::interp::tail_depth(d) {
        Ok(()) => NEXT,
        Err(e) => {
            // SAFETY: compiled code runs attached.
            e.restore(unsafe { Python::assume_attached() });
            ERROR
        }
    }
}

/// Move the exception being raised into register `reg`, for a handler.
extern "C" fn lz_catch(ctx_: *mut c_void, _slots: *mut u64, reg: u32) {
    let ctx = ctx(ctx_);
    let mut regs = ctx.regs();
    let e = PyErr::fetch(ctx.t().py);
    regs.set_obj(reg, e.into_value(ctx.t().py).into_bound(ctx.t().py).into_any());
    crate::interp::release_deferred();
}

/// At a loop back-edge, now and then: run signal handlers (Ctrl-C raises
/// KeyboardInterrupt), and detach for a moment so that a stop-the-world
/// pause (the free-threaded GC) can happen.
extern "C" fn lz_safepoint(ctx_: *mut c_void) -> u32 {
    let py = ctx(ctx_).t().py;
    if let Err(e) = py.check_signals() {
        e.restore(py);
        return ERROR;
    }
    py.detach(|| ());
    NEXT
}

/// `Call` or `CallMethod` at `pc`.
extern "C" fn lz_call(ctx_: *mut c_void, _slots: *mut u64, pc: u32) -> u32 {
    let ctx = ctx(ctx_);
    let mut regs = ctx.regs();
    let (py, f) = (ctx.t().py, ctx.t().f);
    let pc = pc as usize;
    let r = match &f.ir.code[pc] {
        Instr::Call { dst, .. } | Instr::CallMethod { dst, .. } => crate::interp::call_at(py, f, pc, &regs).map(|v| (*dst, v)),
        _ => unreachable!("lz_call on another instruction"),
    };
    match r {
        Ok((dst, v)) => {
            regs.set_obj(dst, v);
            NEXT
        }
        Err(e) => {
            e.restore(py);
            ERROR
        }
    }
}

/// `LoadGlobal` at `pc`.
extern "C" fn lz_load_global(ctx_: *mut c_void, _slots: *mut u64, dst: u32, pc: u32) -> u32 {
    let ctx = ctx(ctx_);
    let mut regs = ctx.regs();
    let (py, f) = (ctx.t().py, ctx.t().f);
    let Instr::LoadGlobal { name, .. } = &f.ir.code[pc as usize] else { unreachable!("lz_load_global on another instruction") };
    let r = crate::interp::builtins_dict(py).and_then(|b| crate::globals::load(py, &f.gics[pc as usize], ctx.t().globals, b, f.names[*name as usize].bind(py)));
    match r {
        Ok(v) => {
            regs.set_obj(dst, v);
            NEXT
        }
        Err(e) => {
            e.restore(py);
            ERROR
        }
    }
}

/// `Check`: type pointers decide most; the full check (and its message)
/// is `step`'s.
extern "C" fn lz_check(ctx_: *mut c_void, slots: *mut u64, reg: u32, idx: u32, pc: u32) -> u32 {
    unsafe {
        let o = *slots.add(reg as usize) as *mut ffi::PyObject;
        let c = ctx(ctx_);
        if !o.is_null() && crate::rt::quick(o, c.t().f.ck(c.t().py, idx)) {
            return NEXT;
        }
        lz_step(ctx_, slots, pc)
    }
}

/// `ForIter`: `NEXT` with the next item in `dst`, `TAKEN` at the end.
extern "C" fn lz_for_iter(ctx_: *mut c_void, slots: *mut u64, dst: u32, iter: u32, pc: u32) -> u32 {
    unsafe {
        let it = slot(slots, iter);
        if it.is_null() {
            return ERROR;
        }
        let t = ctx(ctx_).t();
        crate::interp::note_iter(t.f, pc as usize, it);
        // A Lizzard generator resumes directly (no StopIteration).
        if ffi::Py_TYPE(it) == crate::generator::type_ptr() {
            return crate::generator::for_iter(t.py, slots, dst, it);
        }
        lz_for_iter_py(slots, dst, it)
    }
}

/// `lz_for_iter`, where the loop keeps its item unboxed: a Lizzard
/// generator's int may come as it is, in `*out` (`NEXT_INT`).
extern "C" fn lz_for_iter_int(ctx_: *mut c_void, slots: *mut u64, dst: u32, iter: u32, pc: u32, out: *mut i64) -> u32 {
    unsafe {
        let it = slot(slots, iter);
        if !it.is_null() && ffi::Py_TYPE(it) == crate::generator::type_ptr() {
            let t = ctx(ctx_).t();
            crate::interp::note_iter(t.f, pc as usize, it);
            return crate::generator::for_iter_int(t.py, slots, dst, it, out);
        }
        lz_for_iter(ctx_, slots, dst, iter, pc)
    }
}

/// The value of an exact int, if it fits in an `i64`.
extern "C" fn lz_long_value(o: *mut ffi::PyObject, out: *mut i64) -> u32 {
    unsafe {
        if ffi::Py_TYPE(o) != &raw mut ffi::PyLong_Type {
            return 0;
        }
        let mut overflow = 0;
        let v = ffi::PyLong_AsLongLongAndOverflow(o, &mut overflow);
        if overflow != 0 || (v == -1 && !ffi::PyErr_Occurred().is_null()) {
            ffi::PyErr_Clear();
            return 0;
        }
        *out = v;
        1
    }
}

/// The interpreter runs the rest of an inlined call of `desc`'s function
/// from `pc` on `frame`, its parameters borrowed; then the frame is
/// released. Its result (a new reference), or null with an exception set.
extern "C" fn lz_finish(ctx_: *mut c_void, frame: *mut u64, desc: *const CallDesc, pc: u32) -> *mut ffi::PyObject {
    let py = ctx(ctx_).t().py;
    // SAFETY: the description lives as long as the caller's code.
    let lf = unsafe { (*desc).func.get() };
    let rt = &*lf.rt;
    // The frame owns its parameters from now on.
    for i in 0..rt.ir.params.len() {
        // SAFETY: bound parameters.
        unsafe { crate::rc::incref(*frame.add(i) as *mut ffi::PyObject) };
    }
    // SAFETY: the frame has a slot per register of `rt`.
    let mut regs = unsafe { Regs::window(frame, &rt.objs) };
    let mut pc = pc as usize;
    let r = run_interp(py, rt, lf.globals.bind(py), &lf.closure, &mut regs, &mut pc, None);
    // SAFETY: the frame's slots.
    crate::frame::release_slots(unsafe { std::slice::from_raw_parts_mut(frame, rt.objs.len()) }, &rt.objs);
    match r {
        Ok(Exit::Return(v)) => v.into_ptr(),
        Ok(_) => unreachable!("yield in a function that is not a generator"),
        Err(e) => {
            e.restore(py);
            std::ptr::null_mut()
        }
    }
}

/// A function compiled code called directly bailed out at `pc`: the
/// interpreter finishes it (as `lz_finish`).
extern "C" fn lz_resume(ctx_: *mut c_void, frame: *mut u64, desc: *const CallDesc, pc: u32) -> *mut ffi::PyObject {
    // SAFETY: the description lives as long as the caller's code.
    bailed(unsafe { &(*desc).func.get().rt });
    lz_finish(ctx_, frame, desc, pc)
}

/// A function compiled code called directly returned through `step`: its
/// value, and the frame released (but the borrowed parameters).
extern "C" fn lz_take_exit(callee: *mut c_void, desc: *const CallDesc) -> *mut ffi::PyObject {
    let c = ctx(callee);
    // SAFETY: the description lives as long as the caller's code.
    let rt = unsafe { &*(*desc).func.get().rt };
    let n = rt.ir.params.len();
    // SAFETY: the frame's slots.
    let slots = unsafe { std::slice::from_raw_parts_mut(c.slots, rt.objs.len()) };
    crate::frame::release_slots(&mut slots[n..], &rt.objs[n..]);
    match c.exit.take().map(|e| *e) {
        Some(Exit::Return(v)) => v.into_ptr(),
        _ => unreachable!("a function that is not a generator exits by returning"),
    }
}

/// An exception raised at `pc` leaves an inlined call: note where, and
/// release the frame but its borrowed parameters.
extern "C" fn lz_unwind(frame: *mut u64, desc: *const CallDesc, pc: u32) {
    // SAFETY: compiled code runs attached.
    let py = unsafe { Python::assume_attached() };
    // SAFETY: the description lives as long as the caller's code.
    let rt = unsafe { &*(*desc).func.get().rt };
    let ir = &rt.ir;
    let e = PyErr::fetch(py);
    let line = ir.lines.get(pc as usize).copied().unwrap_or(ir.line);
    let _ = e.add_note(py, format!("lizzard: {}:{line} in {}", ir.file, ir.name));
    let n = ir.params.len();
    // SAFETY: the frame's slots.
    let slots = unsafe { std::slice::from_raw_parts_mut(frame, rt.objs.len()) };
    crate::frame::release_slots(&mut slots[n..], &rt.objs[n..]);
    e.restore(py);
}

/// `__init__` returned `r` (a new reference), not None.
extern "C" fn lz_init_returned(r: *mut ffi::PyObject) {
    // SAFETY: compiled code runs attached; `r` is a new reference.
    let py = unsafe { Python::assume_attached() };
    let v = unsafe { Bound::from_owned_ptr(py, r) };
    if let Err(e) = crate::interp::init_returned(&v) {
        e.restore(py);
    }
}

/// Call the function `desc` describes, its arguments (borrowed from the
/// caller) at the start of `frame` and its other registers unbound: run it
/// compiled when it is, else in the interpreter. Its result (a new
/// reference), or null with an exception set. The frame is released.
extern "C" fn lz_enter(ctx_: *mut c_void, frame: *mut u64, desc: *const CallDesc) -> *mut ffi::PyObject {
    let caller = ctx(ctx_);
    let py = caller.t().py;
    // SAFETY: the description lives as long as the caller's code.
    let lf = unsafe { (*desc).func.get() };
    let rt = &*lf.rt;
    let globals = lf.globals.bind(py);
    let closure = &lf.closure[..];
    let nparams = rt.ir.params.len();
    // SAFETY: attached; the message is a C string.
    if unsafe { ffi::Py_EnterRecursiveCall(c" while calling a Lizzard function".as_ptr()) } != 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: the frame has a slot per register of `rt`.
    let mut regs = unsafe { Regs::window(frame, &rt.objs) };
    // The parameters' references: the caller's, as long as compiled code
    // runs a function that never writes them; the frame's otherwise.
    let mut borrowed = rt.borrow_params;
    let own = |borrowed: &mut bool| {
        if *borrowed {
            for i in 0..nparams {
                // SAFETY: bound parameters.
                unsafe { crate::rc::incref(*frame.add(i) as *mut ffi::PyObject) };
            }
            *borrowed = false;
        }
    };
    if !borrowed {
        borrowed = true;
        own(&mut borrowed);
    }
    let mut pc = 0usize;
    let r = match entry(py, rt) {
        Some(e) => match run(py, e, rt, globals, closure, &mut regs, &mut pc, None) {
            Ran::Exit(Exit::Return(v)) => Ok(v),
            Ran::Exit(_) | Ran::YieldInt { .. } => unreachable!("yield in a function that is not a generator"),
            Ran::Bail => {
                own(&mut borrowed);
                run_interp(py, rt, globals, closure, &mut regs, &mut pc, None).map(|e| match e {
                    Exit::Return(v) => v,
                    _ => unreachable!("yield in a function that is not a generator"),
                })
            }
            Ran::Err(e) => {
                let ir = &rt.ir;
                let line = ir.lines.get(pc).copied().unwrap_or(ir.line);
                let _ = e.add_note(py, format!("lizzard: {}:{line} in {}", ir.file, ir.name));
                Err(e)
            }
        },
        None => {
            own(&mut borrowed);
            run_interp(py, rt, globals, closure, &mut regs, &mut pc, None).map(|e| match e {
                Exit::Return(v) => v,
                _ => unreachable!("yield in a function that is not a generator"),
            })
        }
    };
    // SAFETY: the frame's slots.
    let slots = unsafe { std::slice::from_raw_parts_mut(frame, rt.objs.len()) };
    let from = if borrowed { nparams } else { 0 };
    crate::frame::release_slots(&mut slots[from..], &rt.objs[from..]);
    // SAFETY: entered above.
    unsafe { ffi::Py_LeaveRecursiveCall() };
    match r {
        Ok(v) => v.into_ptr(),
        Err(e) => {
            e.restore(py);
            std::ptr::null_mut()
        }
    }
}
