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
use pyo3::ffi;

use crate::ops::*;

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
            compare_bool: a(lz_compare_bool as *const ()),
            take: a(lz_take as *const ()),
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

/// `return src`: hand the value to `exec` (status `NEXT`, then compiled
/// code exits).
pub extern "C" fn lz_return(ctx: *mut c_void, slots: *mut u64, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        crate::rc::incref(o);
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
        Instr::Call { dst, .. } | Instr::CallMethod { dst, .. } => crate::interp::call_at(py, f, pc, regs).map(|v| (*dst, v)),
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
        lz_for_iter_py(slots, dst, it)
    }
}
