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

/// What compiled code's helpers need, behind its opaque `ctx` pointer.
struct Ctx<'a, 'py> {
    py: Python<'py>,
    f: &'a FuncRt,
    globals: &'a Bound<'py, PyDict>,
    closure: &'a [Py<PyAny>],
    regs: *mut Regs,
    exit: Option<Exit<'py>>,
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
        let helpers = Helpers { step: lz_step as *const () as usize, catch: lz_catch as *const () as usize, safepoint: lz_safepoint as *const () as usize };
        match lizzard_jit::compile(&f.ir, &helpers) {
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
    let mut ctx = Ctx { py, f, globals, closure, regs: regs as *mut Regs, exit: None };
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
