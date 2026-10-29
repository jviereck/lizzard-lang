//! Running functions compiled by `lizzard-jit`.
//!
//! A function is compiled when it first runs (unless `LIZZARD_JIT=0`), and
//! from then on its frames run as native code, generators included: the
//! code works on the same register slots and enters and leaves where the
//! interpreter does. Instructions it does not compile inline call `step`
//! below, which runs them with the interpreter's own code.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

use lizzard_jit::{Entry, Helpers, ERROR, EXIT, NEXT, TAKEN};
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::frame::Regs;
use crate::interp::{step, Exit, FuncRt};
use lizzard_py_front::ir::Instr;

/// What compiled code's helpers need, behind its opaque `ctx` pointer.
struct Ctx<'a, 'py> {
    py: Python<'py>,
    f: &'a FuncRt,
    globals: &'a Bound<'py, PyDict>,
    closure: &'a [Py<PyAny>],
    regs: *mut Regs,
    exit: Option<Exit<'py>>,
    /// The value `return` returns (a new reference), from `lz_return`.
    ret: *mut ffi::PyObject,
}

/// Whether functions are compiled: `LIZZARD_JIT=0` (or `off`) runs
/// everything in the interpreter.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("LIZZARD_JIT").as_deref(), Ok("0") | Ok("off")))
}

/// The function's compiled code, compiling it on first use (None if it
/// cannot be compiled: it then stays interpreted).
pub fn code(f: &FuncRt) -> Option<Entry> {
    *f.jit.get_or_init(|| {
        let a = |p: *const ()| p as usize;
        let helpers = Helpers {
            step: a(lz_step as *const ()),
            catch: a(lz_catch as *const ()),
            safepoint: a(lz_safepoint as *const ()),
            load_attr: a(lz_load_attr as *const ()),
            store_attr: a(lz_store_attr as *const ()),
            konst: a(lz_const as *const ()),
            mov: a(lz_move as *const ()),
            box_i64: a(lz_box_i64 as *const ()),
            ret: a(lz_return as *const ()),
            call: a(lz_call as *const ()),
            for_iter: a(lz_for_iter as *const ()),
            check: a(lz_check as *const ()),
            truthy: a(lz_truthy as *const ()),
            binop: a(lz_binop as *const ()),
            compare: a(lz_compare as *const ()),
        };
        // The function keeps its constants and names alive as long as the
        // code can run.
        let consts: Vec<usize> = f.consts.iter().map(|c| c.as_ptr() as usize).collect();
        let names: Vec<usize> = f.names.iter().map(|n| n.as_ptr() as usize).collect();
        let ics: Vec<usize> = f.ics.iter().map(|c| c as *const AttrIc as usize).collect();
        match lizzard_jit::compile(&f.ir, &helpers, &lizzard_jit::Objects { consts: &consts, names: &names, ics: &ics }) {
            Ok(e) => Some(e),
            Err(e) => {
                if std::env::var_os("LIZZARD_JIT_DEBUG").is_some() {
                    eprintln!("lizzard: cannot compile {}: {e}", f.ir.name);
                }
                None
            }
        }
    })
}

/// Run a frame from `pc` in compiled code until it returns or yields, as
/// `interp::exec` does. With `pending`, the instruction at `pc` raises it.
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
    let slots = regs.as_mut_ptr();
    let mut ctx = Ctx { py, f, globals, closure, regs: regs as *mut Regs, exit: None, ret: std::ptr::null_mut() };
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
        // SAFETY: `lz_return` stored a new reference.
        EXIT if !ctx.ret.is_null() => Ok(Exit::Return(unsafe { Bound::from_owned_ptr(py, ctx.ret) })),
        EXIT => Ok(ctx.exit.take().expect("compiled code exits with a value")),
        _ => {
            let e = PyErr::fetch(py);
            let ir = &f.ir;
            let line = ir.lines.get(*pc).copied().unwrap_or(ir.line);
            let _ = e.add_note(py, format!("lizzard: {}:{line} in {}", ir.file, ir.name));
            Err(e)
        }
    }
}

/// Execute instruction `pc` for compiled code.
extern "C" fn lz_step(ctx: *mut c_void, _slots: *mut u64, pc: u32) -> u32 {
    // SAFETY: compiled code passes the `Ctx` that `exec` gave it; nothing
    // else uses the frame while the helper runs.
    let ctx = unsafe { &mut *(ctx as *mut Ctx<'_, '_>) };
    let regs = unsafe { &mut *ctx.regs };
    let py = ctx.py;
    let mut next = pc as usize;
    let r = catch_unwind(AssertUnwindSafe(|| step(py, ctx.f, ctx.globals, ctx.closure, regs, &mut next)));
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
extern "C" fn lz_catch(ctx: *mut c_void, _slots: *mut u64, reg: u32) {
    // SAFETY: as in `lz_step`.
    let ctx = unsafe { &mut *(ctx as *mut Ctx<'_, '_>) };
    let regs = unsafe { &mut *ctx.regs };
    let e = PyErr::fetch(ctx.py);
    regs.set_obj(reg, e.into_value(ctx.py).into_bound(ctx.py).into_any());
}

/// At a loop back-edge, now and then: run signal handlers (Ctrl-C raises
/// KeyboardInterrupt), and detach for a moment so that a stop-the-world
/// pause (the free-threaded GC) can happen.
extern "C" fn lz_safepoint(ctx: *mut c_void) -> u32 {
    // SAFETY: as in `lz_step`.
    let ctx = unsafe { &mut *(ctx as *mut Ctx<'_, '_>) };
    let py = ctx.py;
    if let Err(e) = py.check_signals() {
        e.restore(py);
        return ERROR;
    }
    py.detach(|| ());
    NEXT
}

// ---- Raw helpers for the hot object instructions ----
//
// Compiled code calls these with the instruction's operands as immediates
// (register numbers, interned names, constants): no decoding, no PyO3
// wrappers. They return `NEXT`, or `ERROR` with an exception set.

use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use pyo3::ffi;

/// The object in slot `r` (borrowed), or null with UnboundLocalError set.
unsafe fn slot(slots: *mut u64, r: u32) -> *mut ffi::PyObject {
    let p = unsafe { *slots.add(r as usize) } as *mut ffi::PyObject;
    if p.is_null() {
        unsafe { ffi::PyErr_SetString(ffi::PyExc_UnboundLocalError, c"local variable referenced before assignment".as_ptr()) };
    }
    p
}

/// Store a new reference in slot `r`, releasing the old one.
unsafe fn put(slots: *mut u64, r: u32, v: *mut ffi::PyObject) {
    let s = unsafe { slots.add(r as usize) };
    let old = unsafe { std::mem::replace(&mut *s, v as u64) } as *mut ffi::PyObject;
    if !old.is_null() {
        unsafe { ffi::Py_DecRef(old) };
    }
}

/// Store a result: a new reference, or null when an exception is set.
unsafe fn result(slots: *mut u64, r: u32, v: *mut ffi::PyObject) -> u32 {
    if v.is_null() {
        return ERROR;
    }
    unsafe { put(slots, r, v) };
    NEXT
}

/// An attribute site's inline cache: the slot (`__slots__` member) an
/// attribute was found in, for one type at one version. A type's version
/// tag changes whenever it or a base class is modified, so a hit needs no
/// lookup: the attribute is read or written at its offset. Instances
/// without `__slots__` keep their attributes in CPython's internal storage,
/// which only the generic path reaches.
#[derive(Default)]
pub struct AttrIc {
    entry: AtomicPtr<IcEntry>,
    misses: AtomicU32,
}

struct IcEntry {
    tp: *mut ffi::PyTypeObject,
    version: u32,
    member: *mut ffi::PyMemberDef,
}

/// A site that misses this often stops looking (it sees many types).
const IC_MAX_MISSES: u32 = 8;

unsafe extern "C" {
    /// CPython 3.14+: the next item (1), the end (0), or an error (-1).
    fn PyIter_NextItem(iter: *mut ffi::PyObject, item: *mut *mut ffi::PyObject) -> std::ffi::c_int;
    /// CPython 3.12+: give a type a version tag (0 if it cannot have one).
    fn PyUnstable_Type_AssignVersionTag(tp: *mut ffi::PyTypeObject) -> std::ffi::c_int;
}

/// The `__slots__` member `name` of `o`'s type, from the cache or found and
/// cached; None to take the generic path.
unsafe fn member(ic: &AttrIc, o: *mut ffi::PyObject, name: *mut ffi::PyObject) -> Option<*mut ffi::PyMemberDef> {
    unsafe {
        let tp = ffi::Py_TYPE(o);
        let e = ic.entry.load(Ordering::Acquire);
        if !e.is_null() && (*e).tp == tp && (*e).version == (*tp).tp_version_tag {
            return Some((*e).member);
        }
        if ic.misses.fetch_add(1, Ordering::Relaxed) >= IC_MAX_MISSES || PyUnstable_Type_AssignVersionTag(tp) == 0 {
            return None;
        }
        let version = (*tp).tp_version_tag;
        let mut d = std::ptr::null_mut();
        // On the type, a member descriptor is the descriptor itself.
        if ffi::PyObject_GetOptionalAttr(tp as *mut ffi::PyObject, name, &mut d) <= 0 {
            ffi::PyErr_Clear();
            return None;
        }
        let found = (ffi::Py_TYPE(d) == &raw mut ffi::PyMemberDescr_Type).then(|| (*(d as *mut ffi::PyMemberDescrObject)).d_member);
        ffi::Py_DecRef(d);
        let m = found?;
        // Entries are small and a site replaces its entry at most
        // `IC_MAX_MISSES` times, so replaced ones are not freed (another
        // thread may still be reading one).
        let new = Box::into_raw(Box::new(IcEntry { tp, version, member: m }));
        ic.entry.store(new, Ordering::Release);
        Some(m)
    }
}

pub extern "C" fn lz_load_attr(slots: *mut u64, dst: u32, obj: u32, name: *mut ffi::PyObject, ic: *const AttrIc) -> u32 {
    unsafe {
        let o = slot(slots, obj);
        if o.is_null() {
            return ERROR;
        }
        let v = match member(&*ic, o, name) {
            Some(m) => ffi::PyMember_GetOne(o as *const std::ffi::c_char, m),
            None => ffi::PyObject_GetAttr(o, name),
        };
        result(slots, dst, v)
    }
}

pub extern "C" fn lz_store_attr(slots: *mut u64, obj: u32, val: u32, name: *mut ffi::PyObject, ic: *const AttrIc) -> u32 {
    unsafe {
        let (o, v) = (slot(slots, obj), slot(slots, val));
        if o.is_null() || v.is_null() {
            return ERROR;
        }
        let r = match member(&*ic, o, name) {
            Some(m) => ffi::PyMember_SetOne(o as *mut std::ffi::c_char, m, v),
            None => ffi::PyObject_SetAttr(o, name, v),
        };
        if r < 0 {
            return ERROR;
        }
        NEXT
    }
}

/// `dst = const` (a new reference to an object the function keeps).
pub extern "C" fn lz_const(slots: *mut u64, dst: u32, obj: *mut ffi::PyObject) {
    unsafe {
        ffi::Py_IncRef(obj);
        put(slots, dst, obj);
    }
}

pub extern "C" fn lz_move(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        ffi::Py_IncRef(o);
        put(slots, dst, o);
        NEXT
    }
}

/// `dst = int(src)` from an `I64` slot.
pub extern "C" fn lz_box_i64(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let v = *slots.add(src as usize) as i64;
        result(slots, dst, ffi::PyLong_FromLongLong(v))
    }
}

/// `dst: Bool = bool(src)`
pub extern "C" fn lz_truthy(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        let t = if o == ffi::Py_True() {
            1
        } else if o == ffi::Py_False() || o == ffi::Py_None() {
            0
        } else {
            ffi::PyObject_IsTrue(o)
        };
        if t < 0 {
            return ERROR;
        }
        *slots.add(dst as usize) = t as u64;
        NEXT
    }
}

/// `a op b`; `op` is the `BinOp` code, plus 16 for the in-place form.
pub extern "C" fn lz_binop(slots: *mut u64, dst: u32, a: u32, b: u32, op: u32) -> u32 {
    unsafe {
        let (x, y) = (slot(slots, a), slot(slots, b));
        if x.is_null() || y.is_null() {
            return ERROR;
        }
        let v = match op {
            0 => ffi::PyNumber_Add(x, y),
            1 => ffi::PyNumber_Subtract(x, y),
            2 => ffi::PyNumber_Multiply(x, y),
            3 => ffi::PyNumber_TrueDivide(x, y),
            4 => ffi::PyNumber_FloorDivide(x, y),
            5 => ffi::PyNumber_Remainder(x, y),
            6 => ffi::PyNumber_Power(x, y, ffi::Py_None()),
            7 => ffi::PyNumber_MatrixMultiply(x, y),
            8 => ffi::PyNumber_Or(x, y),
            9 => ffi::PyNumber_And(x, y),
            10 => ffi::PyNumber_Xor(x, y),
            11 => ffi::PyNumber_Lshift(x, y),
            12 => ffi::PyNumber_Rshift(x, y),
            16 => ffi::PyNumber_InPlaceAdd(x, y),
            17 => ffi::PyNumber_InPlaceSubtract(x, y),
            18 => ffi::PyNumber_InPlaceMultiply(x, y),
            19 => ffi::PyNumber_InPlaceTrueDivide(x, y),
            20 => ffi::PyNumber_InPlaceFloorDivide(x, y),
            21 => ffi::PyNumber_InPlaceRemainder(x, y),
            22 => ffi::PyNumber_InPlacePower(x, y, ffi::Py_None()),
            23 => ffi::PyNumber_InPlaceMatrixMultiply(x, y),
            24 => ffi::PyNumber_InPlaceOr(x, y),
            25 => ffi::PyNumber_InPlaceAnd(x, y),
            26 => ffi::PyNumber_InPlaceXor(x, y),
            27 => ffi::PyNumber_InPlaceLshift(x, y),
            _ => ffi::PyNumber_InPlaceRshift(x, y),
        };
        result(slots, dst, v)
    }
}

/// `a op b` for `==`, `!=`, `<`, `<=`, `>`, `>=` (`op` is CPython's
/// `Py_EQ`... code).
pub extern "C" fn lz_compare(slots: *mut u64, dst: u32, a: u32, b: u32, op: u32) -> u32 {
    unsafe {
        let (x, y) = (slot(slots, a), slot(slots, b));
        if x.is_null() || y.is_null() {
            return ERROR;
        }
        result(slots, dst, ffi::PyObject_RichCompare(x, y, op as i32))
    }
}

/// `return src`: hand the value to `exec` (status `NEXT`, then compiled
/// code exits).
pub extern "C" fn lz_return(ctx: *mut c_void, slots: *mut u64, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        ffi::Py_IncRef(o);
        (*(ctx as *mut Ctx<'_, '_>)).ret = o;
        NEXT
    }
}

/// `Call` or `CallMethod` at `pc`.
pub extern "C" fn lz_call(ctx: *mut c_void, _slots: *mut u64, pc: u32) -> u32 {
    // SAFETY: as in `lz_step`.
    let ctx = unsafe { &mut *(ctx as *mut Ctx<'_, '_>) };
    let regs = unsafe { &mut *ctx.regs };
    let (py, f) = (ctx.py, ctx.f);
    let pc = pc as usize;
    let r = match &f.ir.code[pc] {
        Instr::Call { dst, func, args, .. } => regs.obj(py, *func).map(|o| o.as_ptr()).and_then(|callee| Ok((*dst, crate::interp::vector_call(py, regs, None, callee, args, f.kwnames(pc))?))),
        Instr::CallMethod { dst, obj, name, args, .. } => regs.obj(py, *obj).map(|o| o.as_ptr()).and_then(|o| Ok((*dst, crate::interp::vector_call(py, regs, Some(o), f.names[*name as usize].as_ptr(), args, f.kwnames(pc))?))),
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

/// `ForIter`: `NEXT` with the next item in `dst`, `TAKEN` at the end.
pub extern "C" fn lz_for_iter(ctx: *mut c_void, slots: *mut u64, dst: u32, iter: u32, pc: u32) -> u32 {
    unsafe {
        let it = slot(slots, iter);
        if it.is_null() {
            return ERROR;
        }
        // A Lizzard generator resumes directly (no StopIteration): `step`.
        if ffi::Py_TYPE(it) == crate::generator::type_ptr() {
            return lz_step(ctx, slots, pc);
        }
        let mut item = std::ptr::null_mut();
        match PyIter_NextItem(it, &mut item) {
            1 => {
                put(slots, dst, item);
                NEXT
            }
            0 => TAKEN,
            _ => ERROR,
        }
    }
}

/// `Check`: type pointers decide most; the full check (and its message)
/// is `step`'s.
pub extern "C" fn lz_check(ctx: *mut c_void, slots: *mut u64, reg: u32, idx: u32, pc: u32) -> u32 {
    unsafe {
        let o = *slots.add(reg as usize) as *mut ffi::PyObject;
        let c = &*(ctx as *mut Ctx<'_, '_>);
        if !o.is_null() && crate::rt::quick(o, c.f.ck(c.py, idx)) {
            return NEXT;
        }
        lz_step(ctx, slots, pc)
    }
}
