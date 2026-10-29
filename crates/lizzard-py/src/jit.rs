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

use lizzard_jit::{CallGuard, CallSpec, Entry, Helpers, Layout, Num, Quick, Spec, BAIL, ERROR, EXIT, NEXT, TAKEN};
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
    /// How many instructions the current code has specs for.
    nspecs: AtomicU32,
    lock: Mutex<()>,
    /// What the code refers to (callee descriptions), alive as long as the
    /// function.
    deps: Mutex<Vec<Box<CallDesc>>>,
}

const MAX_COMPILES: u32 = 4;
const BAILS_TO_RECOMPILE: u32 = 1000;

/// A function compiled code calls directly (`Helpers::enter`).
pub struct CallDesc {
    func: Py<LizzardFunction>,
}

/// What compiled code's helpers need, behind its opaque `ctx` pointer. The
/// first two fields are at `lizzard_jit::CTX_TID` and `CTX_RET`.
#[repr(C)]
pub(crate) struct Ctx<'a, 'py> {
    tid: usize,
    /// The value `return` returns (a new reference).
    ret: *mut ffi::PyObject,
    py: Python<'py>,
    f: &'a FuncRt,
    globals: &'a Bound<'py, PyDict>,
    closure: &'a [Py<PyAny>],
    slots: *mut u64,
    exit: Option<Exit<'py>>,
}

impl<'a, 'py> Ctx<'a, 'py> {
    fn new(py: Python<'py>, f: &'a FuncRt, globals: &'a Bound<'py, PyDict>, closure: &'a [Py<PyAny>], slots: *mut u64) -> Self {
        Ctx { tid: crate::rc::tid(), ret: std::ptr::null_mut(), py, f, globals, closure, slots, exit: None }
    }

    /// A view of the frame's registers.
    fn regs(&self) -> Regs {
        // SAFETY: the frame's slots, one per register of `f`; nothing else
        // uses them while a helper runs.
        unsafe { Regs::window(self.slots, &self.f.objs) }
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
fn loaded(f: &FuncRt) -> Option<Entry> {
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
    };
    let consts: Vec<usize> = f.consts.iter().map(|c| c.as_ptr() as usize).collect();
    let nums: Vec<Num> = ir
        .consts
        .iter()
        .map(|c| match c {
            Const::Float(x) => Num::F64(*x),
            Const::Int(i) => Num::I64(*i),
            _ => Num::Other,
        })
        .collect();
    let names: Vec<usize> = f.names.iter().map(|n| n.as_ptr() as usize).collect();
    let ics: Vec<usize> = f.ics.iter().map(|c| c as *const AttrIc as usize).collect();
    let mut deps = f.jit.deps.try_lock().ok()?;
    let specs = specs(py, f, &mut deps);
    f.jit.nspecs.store(count(&specs), Ordering::Relaxed);
    f.jit.misses.store(0, Ordering::Relaxed);
    let misses = &f.jit.misses as *const AtomicU32 as usize;
    let unit = lizzard_jit::Unit { ir, specs: &specs, consts: &consts, nums: &nums, names: &names, ics: &ics, borrow_params: f.borrow_params, misses };
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

fn count(specs: &[Spec]) -> u32 {
    specs.iter().filter(|s| !matches!(s, Spec::None | Spec::Check(_))).count() as u32
}

/// Sites compiled without specs keep running their generic paths: if the
/// caches know more by now, recompile (once the function is warm again).
extern "C" fn lz_missed(ctx_: *mut c_void) {
    let ctx = ctx(ctx_);
    let f = ctx.f;
    if f.jit.compiles.load(Ordering::Relaxed) >= MAX_COMPILES {
        return;
    }
    let mut deps = Vec::new();
    let now = count(&specs(ctx.py, f, &mut deps));
    if now > f.jit.nspecs.load(Ordering::Relaxed) {
        if debug() {
            eprintln!("lizzard: {} has more cache data: recompiling", f.ir.name);
        }
        f.jit.calls.store(0, Ordering::Relaxed);
        f.jit.loops.store(0, Ordering::Relaxed);
        f.jit.entry.store(0, Ordering::Release);
    }
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

/// What the caches saw, per instruction.
fn specs(py: Python<'_>, f: &FuncRt, deps: &mut Vec<Box<CallDesc>>) -> Vec<Spec> {
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
            Instr::Call { func, args, kwnames, direct, .. } if kwnames.is_empty() => match global_of.get(func) {
                // SAFETY: a valid object (the global's).
                Some(&callee) if unsafe { ffi::Py_TYPE(callee as *mut ffi::PyObject) } == &raw mut ffi::PyType_Type => match ctor_spec(&f.ctics[pc]) {
                    // A class constructed by allocating and `__init__`.
                    Some((tp, version, init)) if tp as usize == callee => match call_spec(py, f, pc, init, args.len() + 1, 0, deps) {
                        // SAFETY: a class.
                        Some((_, spec)) => Spec::Call(CallSpec { guard: CallGuard::Ctor { tp: tp as usize, version, alloc: unsafe { (*tp).tp_alloc }.map_or(0, |a| a as *const () as usize) }, ..spec }),
                        None => Spec::None,
                    },
                    _ => Spec::None,
                },
                Some(&callee) => {
                    let skip = direct.as_ref().map_or(0, |d| d.skip);
                    call_spec(py, f, pc, callee as *mut ffi::PyObject, args.len(), skip, deps).map_or(Spec::None, |(_, spec)| Spec::Call(CallSpec { guard: CallGuard::Func(callee), ..spec }))
                }
                None => Spec::None,
            },
            Instr::CallMethod { args, kwnames, direct, .. } if kwnames.is_empty() => match method_spec(&f.mics[pc]) {
                Some((tp, version, func, keys, nentries, valid)) => {
                    let skip = direct.as_ref().map_or(0, |d| d.skip);
                    match call_spec(py, f, pc, func, args.len() + 1, skip, deps) {
                        Some((_, spec)) => Spec::Call(CallSpec { guard: CallGuard::Method { tp, version, keys, nentries, valid }, ..spec }),
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
fn call_spec(py: Python<'_>, site: &FuncRt, pc: usize, callee: *mut ffi::PyObject, nargs: usize, skip: u64, deps: &mut Vec<Box<CallDesc>>) -> Option<((), CallSpec)> {
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
    let desc = Box::new(CallDesc { func });
    let addr = &*desc as *const CallDesc as usize;
    deps.push(desc);
    Some(((), CallSpec { guard: CallGuard::Func(0), desc: addr, nregs: ir.reprs.len() as u32, checks }))
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
    match run(py, entry, f, globals, closure, regs, pc, pending) {
        Ran::Exit(e) => Ok(e),
        Ran::Bail => run_interp(py, f, globals, closure, regs, pc, None),
        Ran::Err(e) => {
            let ir = &f.ir;
            let line = ir.lines.get(*pc).copied().unwrap_or(ir.line);
            let _ = e.add_note(py, format!("lizzard: {}:{line} in {}", ir.file, ir.name));
            Err(e)
        }
    }
}

/// How compiled code stopped.
pub enum Ran<'py> {
    Exit(Exit<'py>),
    /// The interpreter continues at the pc.
    Bail,
    /// An exception no handler of the function caught.
    Err(PyErr),
}

/// Run compiled code on a frame from `*pc` (left where it stopped).
#[allow(clippy::too_many_arguments)]
pub fn run<'py>(py: Python<'py>, entry: Entry, f: &FuncRt, globals: &Bound<'py, PyDict>, closure: &[Py<PyAny>], regs: &mut Regs, pc: &mut usize, pending: Option<PyErr>) -> Ran<'py> {
    let slots = regs.as_mut_ptr();
    let mut ctx = Ctx::new(py, f, globals, closure, slots);
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
        EXIT => Ran::Exit(ctx.exit.take().expect("compiled code exits with a value")),
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
    let py = ctx.py;
    let mut next = pc as usize;
    let r = catch_unwind(AssertUnwindSafe(|| step(py, ctx.f, ctx.globals, ctx.closure, &mut regs, &mut next)));
    match r {
        Ok(Ok(None)) if next == pc as usize + 1 => NEXT,
        Ok(Ok(None)) => TAKEN,
        Ok(Ok(Some(exit))) => {
            ctx.exit = Some(exit);
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

/// Move the exception being raised into register `reg`, for a handler.
extern "C" fn lz_catch(ctx_: *mut c_void, _slots: *mut u64, reg: u32) {
    let ctx = ctx(ctx_);
    let mut regs = ctx.regs();
    let e = PyErr::fetch(ctx.py);
    regs.set_obj(reg, e.into_value(ctx.py).into_bound(ctx.py).into_any());
}

/// At a loop back-edge, now and then: run signal handlers (Ctrl-C raises
/// KeyboardInterrupt), and detach for a moment so that a stop-the-world
/// pause (the free-threaded GC) can happen.
extern "C" fn lz_safepoint(ctx_: *mut c_void) -> u32 {
    let py = ctx(ctx_).py;
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
    let (py, f) = (ctx.py, ctx.f);
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
    let (py, f) = (ctx.py, ctx.f);
    let Instr::LoadGlobal { name, .. } = &f.ir.code[pc as usize] else { unreachable!("lz_load_global on another instruction") };
    let r = crate::interp::builtins_dict(py).and_then(|b| crate::globals::load(py, &f.gics[pc as usize], ctx.globals, b, f.names[*name as usize].bind(py)));
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
        if !o.is_null() && crate::rt::quick(o, c.f.ck(c.py, idx)) {
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
        // A Lizzard generator resumes directly (no StopIteration): `step`.
        if ffi::Py_TYPE(it) == crate::generator::type_ptr() {
            return lz_step(ctx_, slots, pc);
        }
        lz_for_iter_py(slots, dst, it)
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
    let py = caller.py;
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
            Ran::Exit(_) => unreachable!("yield in a function that is not a generator"),
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
