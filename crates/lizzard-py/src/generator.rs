//! Generators: a suspended frame owned by a Python object.
//!
//! The model is the original VM's (`lizzard-vm/src/generator.rs`): the
//! generator owns its frame (registers and the pc of the `yield` it is
//! suspended at). Resuming writes the sent value into the yield's
//! destination register and enters `exec` after the `yield`; `throw` enters
//! at the `yield` with the exception raised there, so the function's own
//! handlers see it. An atomic "running" flag stops a second thread (or a
//! re-entrant call) from resuming it while it runs: holding it is what
//! gives access to the frame, which is resumed in place. A compiled
//! function needs only the same entries and exits.
//!
//! The object implements CPython's generator protocol (`__next__`, `send`,
//! `throw`, `close`) and is registered as a `collections.abc.Generator`, so
//! Python code drives it like any generator, and `yield from` works in both
//! directions.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

use pyo3::exceptions::{PyBaseException, PyGeneratorExit, PyRuntimeError, PyStopIteration, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use crate::frame::Regs;
use crate::interp::{exec, Exit, FuncRt};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Status {
    Created,
    Suspended,
    Done,
}

impl Status {
    fn from(v: u8) -> Status {
        match v {
            0 => Status::Created,
            1 => Status::Suspended,
            _ => Status::Done,
        }
    }
}

struct Frame {
    regs: Regs,
    /// The `yield` / `yield from` it is suspended at (0 before it starts).
    pc: usize,
    /// Where the value sent to that `yield` goes.
    dst: u32,
    /// The iterator a `yield from` delegates to.
    delegate: Option<Py<PyAny>>,
    /// The exception it was handling when it yielded (in an `except` block):
    /// the caller's is restored meanwhile.
    handled: Option<Py<PyAny>>,
}


/// How a generator is resumed.
pub enum Op<'py> {
    Send(Bound<'py, PyAny>),
    Throw(PyErr),
}

/// The next item for a `for` loop (`next_direct`).
pub(crate) enum Next<'py> {
    Value(Bound<'py, PyAny>),
    /// An int it yielded as it is.
    Int(i64),
    /// It returned (the value).
    End(Bound<'py, PyAny>),
    /// Not this way: `resume`.
    Slow,
}

/// What resuming an iterator gave: a value, or its return value.
pub enum Sent<'py> {
    Next(Bound<'py, PyAny>),
    Return(Bound<'py, PyAny>),
}

static TYPE: std::sync::atomic::AtomicPtr<ffi::PyTypeObject> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// `LizzardGenerator`'s type object (set when the module initializes).
pub fn type_ptr() -> *mut ffi::PyTypeObject {
    TYPE.load(Ordering::Relaxed)
}

pub fn init_type(py: Python<'_>) {
    TYPE.store(py.get_type::<LizzardGenerator>().as_type_ptr(), Ordering::Relaxed);
}

/// The generator a Lizzard generator function returns.
#[pyclass(frozen, module = "lizzard._lizzard")]
pub struct LizzardGenerator {
    rt: Arc<FuncRt>,
    globals: Py<PyDict>,
    closure: Vec<Py<PyAny>>,
    status: AtomicU8,
    /// None once it is done. Only who set `running` touches it.
    frame: UnsafeCell<Option<Frame>>,
    running: AtomicBool,
}

// SAFETY: `frame` is used only by the thread that set `running` (or by
// `Drop`, which has the generator to itself).
unsafe impl Sync for LizzardGenerator {}

impl LizzardGenerator {
    /// A generator over a new frame with the (checked) arguments.
    pub fn new(rt: Arc<FuncRt>, globals: Py<PyDict>, closure: Vec<Py<PyAny>>, args: Vec<Bound<'_, PyAny>>) -> LizzardGenerator {
        let mut regs = Regs::new(&rt.objs);
        for (i, a) in args.into_iter().enumerate() {
            regs.set_obj(i as u32, a);
        }
        let frame = Frame { regs, pc: 0, dst: 0, delegate: None, handled: None };
        LizzardGenerator { rt, globals, closure, status: AtomicU8::new(Status::Created as u8), frame: UnsafeCell::new(Some(frame)), running: AtomicBool::new(false) }
    }

    fn status(&self) -> Status {
        Status::from(self.status.load(Ordering::Acquire))
    }

    fn set_status(&self, s: Status) {
        self.status.store(s as u8, Ordering::Release);
    }

    /// Done: its frame released (which may run Python code; `running` is
    /// still set).
    fn finish(&self, slot: &mut Option<Frame>) {
        self.set_status(Status::Done);
        drop(slot.take());
    }

    /// Resume with a value or an exception; the next value it yields, or the
    /// value it returned.
    pub fn resume<'py>(&self, py: Python<'py>, op: Op<'py>) -> PyResult<Sent<'py>> {
        if self.running.swap(true, Ordering::Acquire) {
            return Err(PyValueError::new_err("generator already executing"));
        }
        let r = self.resume_inner(py, op);
        self.running.store(false, Ordering::Release);
        r
    }

    fn resume_inner<'py>(&self, py: Python<'py>, op: Op<'py>) -> PyResult<Sent<'py>> {
        // SAFETY: this thread set `running`: the frame is its own.
        let slot = unsafe { &mut *self.frame.get() };
        let status = self.status();
        let frame = match (status, slot.as_mut()) {
            (Status::Created, Some(f)) => match op {
                // Never started: it is done without running.
                Op::Throw(e) => {
                    self.finish(slot);
                    return Err(e);
                }
                Op::Send(v) if !v.is_none() => return Err(PyTypeError::new_err("can't send non-None value to a just-started generator")),
                Op::Send(_) => f,
            },
            (Status::Suspended, Some(f)) => f,
            _ => {
                self.finish(slot);
                return match op {
                    Op::Send(_) => Ok(Sent::Return(py.None().into_bound(py))),
                    Op::Throw(e) => Err(e),
                };
            }
        };
        let mut pending = None;
        if status == Status::Suspended {
            match frame.delegate.take() {
                // `yield from`: the iterator handles it until it finishes.
                Some(d) => {
                    let d = d.into_bound(py);
                    let r = match op {
                        Op::Send(v) => send_to(&d, &v),
                        Op::Throw(e) => throw_to(py, &d, e),
                    };
                    match r {
                        Ok(Sent::Next(v)) => {
                            frame.delegate = Some(d.unbind());
                            return Ok(Sent::Next(v));
                        }
                        Ok(Sent::Return(v)) => {
                            frame.regs.set_obj(frame.dst, v);
                            frame.pc += 1;
                        }
                        Err(e) => pending = Some(e),
                    }
                }
                None => match op {
                    Op::Send(v) => {
                        frame.regs.set_obj(frame.dst, v);
                        frame.pc += 1;
                    }
                    Op::Throw(e) => pending = Some(e),
                },
            }
        }
        let mut pc = frame.pc;
        self.set_status(Status::Suspended);
        // The generator's own handled exception while it runs, the caller's
        // otherwise (put back only if the generator changed it). Only its
        // own `except` blocks can leave one when it yields: functions it
        // calls restore theirs.
        let r;
        let own_handled;
        if self.rt.sets_handled || frame.handled.is_some() {
            let outer = handled(py);
            let own = frame.handled.take();
            if let Some(h) = &own {
                set_handled(h.bind(py));
            }
            r = exec(py, &self.rt, self.globals.bind(py), &self.closure, &mut frame.regs, &mut pc, pending);
            let inner = handled(py);
            if own.is_some() || !inner.is(&outer) {
                set_handled(&outer);
            }
            own_handled = (!inner.is(&outer)).then(|| inner.unbind());
        } else {
            r = exec(py, &self.rt, self.globals.bind(py), &self.closure, &mut frame.regs, &mut pc, pending);
            own_handled = None;
        }
        self.after(py, slot, pc, r, own_handled)
    }

    /// The end of a run of its code (`pc` where it stopped): suspended at a
    /// yield (with the handled exception it keeps, if any), or done.
    fn after<'py>(&self, py: Python<'py>, slot: &mut Option<Frame>, pc: usize, r: PyResult<Exit<'py>>, handled: Option<Py<PyAny>>) -> PyResult<Sent<'py>> {
        let suspend = |slot: &mut Option<Frame>, value: Bound<'py, PyAny>, dst: u32, delegate: Option<Bound<'py, PyAny>>| {
            let frame = slot.as_mut().expect("a running generator's frame");
            frame.pc = pc;
            frame.dst = dst;
            frame.delegate = delegate.map(Bound::unbind);
            frame.handled = handled;
            Sent::Next(value)
        };
        match r {
            Ok(Exit::Yield { value, dst }) => Ok(suspend(slot, value, dst, None)),
            Ok(Exit::YieldFrom { value, dst, iter }) => Ok(suspend(slot, value, dst, Some(iter))),
            Ok(Exit::Return(v)) => {
                self.finish(slot);
                Ok(Sent::Return(v))
            }
            Err(e) => {
                self.finish(slot);
                // PEP 479: a StopIteration escaping the body would end the
                // caller's loop silently.
                if e.is_instance_of::<PyStopIteration>(py) {
                    let re = PyRuntimeError::new_err("generator raised StopIteration");
                    re.set_cause(py, Some(e));
                    return Err(re);
                }
                Err(e)
            }
        }
    }

    /// The next item for a `for` loop, from the generator's compiled code
    /// entered directly (no `exec` and its layers): when it is compiled,
    /// not running, not delegating (`yield from`) and keeps no handled
    /// exception; else `Slow` (`resume` does it). `want_int`: an int it
    /// yields unboxed comes as it is.
    #[inline(always)]
    pub(crate) fn next_direct<'py>(&self, py: Python<'py>, want_int: bool) -> PyResult<Next<'py>> {
        let Some(entry) = crate::jit::loaded(&self.rt) else { return Ok(Next::Slow) };
        if self.rt.sets_handled || self.running.swap(true, Ordering::Acquire) {
            return Ok(Next::Slow);
        }
        let r = self.next_running(py, entry, want_int);
        self.running.store(false, Ordering::Release);
        r
    }

    #[inline(always)]
    fn next_running<'py>(&self, py: Python<'py>, entry: lizzard_jit::Entry, want_int: bool) -> PyResult<Next<'py>> {
        // SAFETY: this thread set `running`: the frame is its own.
        let slot = unsafe { &mut *self.frame.get() };
        let status = self.status();
        let frame = match (status, slot.as_mut()) {
            (Status::Created | Status::Suspended, Some(f)) if f.handled.is_none() => f,
            _ => return Ok(Next::Slow),
        };
        let mut pending = None;
        let mut pc = match (status, &frame.delegate) {
            // `yield from` another Lizzard generator: its next item, the
            // same way (still running, as Python's are); its end continues
            // here with its return value, its exception is raised at the
            // `yield from`.
            (Status::Suspended, Some(d)) => {
                let d = d.bind(py).clone();
                match delegate_next(py, d, want_int) {
                    Ok(Next::End(v)) => {
                        frame.delegate = None;
                        frame.regs.set_obj(frame.dst, v);
                        frame.pc + 1
                    }
                    Err(e) => {
                        frame.delegate = None;
                        pending = Some(e);
                        frame.pc
                    }
                    r => return r,
                }
            }
            (Status::Suspended, None) => {
                // `next()` sends None to the `yield`.
                frame.regs.set_obj(frame.dst, py.None().into_bound(py));
                frame.pc + 1
            }
            _ => 0,
        };
        self.set_status(Status::Suspended);
        let globals = self.globals.bind(py);
        let ran = crate::jit::run_with(py, entry, &self.rt, globals, &self.closure, &mut frame.regs, &mut pc, pending, want_int);
        let r = match ran {
            crate::jit::Ran::Exit(Exit::Yield { value, dst }) => {
                frame.pc = pc;
                frame.dst = dst;
                return Ok(Next::Value(value));
            }
            crate::jit::Ran::YieldInt { value, dst } => {
                frame.pc = pc;
                frame.dst = dst;
                return Ok(Next::Int(value));
            }
            ran => crate::jit::settle(py, &self.rt, globals, &self.closure, &mut frame.regs, &mut pc, ran),
        };
        Ok(match self.after(py, slot, pc, r, None)? {
            Sent::Next(v) => Next::Value(v),
            Sent::Return(v) => Next::End(v),
        })
    }

    fn close_gen(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.status() != Status::Suspended {
            if !self.running.swap(true, Ordering::Acquire) {
                // SAFETY: `running` was just set by this thread.
                self.finish(unsafe { &mut *self.frame.get() });
                self.running.store(false, Ordering::Release);
            }
            return Ok(py.None());
        }
        match self.resume(py, Op::Throw(PyGeneratorExit::new_err(()))) {
            Ok(Sent::Next(_)) => Err(PyRuntimeError::new_err("generator ignored GeneratorExit")),
            Ok(Sent::Return(v)) => Ok(v.unbind()),
            Err(e) if e.is_instance_of::<PyGeneratorExit>(py) || e.is_instance_of::<PyStopIteration>(py) => Ok(py.None()),
            Err(e) => Err(e),
        }
    }
}

/// The next item of the iterator a `yield from` delegates to (`next_direct`
/// of a Lizzard generator; `Slow` for others). Kept out of line: the
/// recursion would stop the compiler from inlining `next_direct` into the
/// loop helpers.
#[inline(never)]
fn delegate_next<'py>(py: Python<'py>, d: Bound<'py, PyAny>, want_int: bool) -> PyResult<Next<'py>> {
    // SAFETY: a type pointer comparison on a valid object.
    if unsafe { ffi::Py_TYPE(d.as_ptr()) } != type_ptr() {
        return Ok(Next::Slow);
    }
    // SAFETY: the type says what it is.
    let inner = unsafe { d.cast_into_unchecked::<LizzardGenerator>() };
    inner.get().next_direct(py, want_int)
}

/// `ForIter` over Lizzard generator `it` (as `ops::lz_for_iter_py` over
/// other iterators): `NEXT` with the next item in `dst`, `TAKEN` at the end,
/// `ERROR` with the exception set.
///
/// # Safety
/// `it` is a LizzardGenerator; `slots` has register `dst`.
pub(crate) unsafe fn for_iter(py: Python<'_>, slots: *mut u64, dst: u32, it: *mut ffi::PyObject) -> u32 {
    // SAFETY: as required.
    unsafe { for_iter_with(py, slots, dst, it, None) }
}

/// `for_iter`, an int yielded as it is going to `*out` (`NEXT_INT`).
///
/// # Safety
/// As `for_iter`; `out` is writable.
pub(crate) unsafe fn for_iter_int(py: Python<'_>, slots: *mut u64, dst: u32, it: *mut ffi::PyObject, out: *mut i64) -> u32 {
    // SAFETY: as required.
    unsafe { for_iter_with(py, slots, dst, it, Some(out)) }
}

unsafe fn for_iter_with(py: Python<'_>, slots: *mut u64, dst: u32, it: *mut ffi::PyObject, out: Option<*mut i64>) -> u32 {
    use lizzard_jit::{ERROR, NEXT, NEXT_INT, TAKEN};
    // SAFETY: as required.
    let g = unsafe { Bound::from_borrowed_ptr(py, it).cast_into_unchecked::<LizzardGenerator>() };
    let g = g.get();
    let r = match g.next_direct(py, out.is_some()) {
        Ok(Next::Slow) => match g.resume(py, Op::Send(py.None().into_bound(py))) {
            Ok(Sent::Next(v)) => Ok(Next::Value(v)),
            Ok(Sent::Return(v)) => Ok(Next::End(v)),
            Err(e) => Err(e),
        },
        r => r,
    };
    match r {
        Ok(Next::Value(v)) => {
            // SAFETY: a new reference for the slot.
            unsafe { crate::ops::put(slots, dst, v.into_ptr()) };
            NEXT
        }
        Ok(Next::Int(x)) => {
            // SAFETY: asked for with `out`.
            unsafe { *out.expect("an int for a loop that takes one") = x };
            NEXT_INT
        }
        Ok(_) => TAKEN,
        Err(e) => {
            e.restore(py);
            ERROR
        }
    }
}

fn stop_iteration(v: Bound<'_, PyAny>) -> PyErr {
    PyStopIteration::new_err((v.unbind(),))
}

#[pymethods]
impl LizzardGenerator {
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self.resume(py, Op::Send(py.None().into_bound(py)))? {
            Sent::Next(v) => Ok(Some(v.unbind())),
            Sent::Return(v) if v.is_none() => Ok(None),
            Sent::Return(v) => Err(stop_iteration(v)),
        }
    }

    fn send(&self, py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match self.resume(py, Op::Send(value))? {
            Sent::Next(v) => Ok(v.unbind()),
            Sent::Return(v) => Err(stop_iteration(v)),
        }
    }

    #[pyo3(signature = (typ, val=None, tb=None))]
    fn throw(&self, py: Python<'_>, typ: Bound<'_, PyAny>, val: Option<Bound<'_, PyAny>>, tb: Option<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if val.is_some() || tb.is_some() {
            let w = py.get_type::<pyo3::exceptions::PyDeprecationWarning>();
            PyErr::warn(py, &w, c"the (type, exc, tb) signature of throw() is deprecated, use the single-arg signature instead.", 1)?;
        }
        let exc = make_exception(&typ, val.filter(|v| !v.is_none()), tb.filter(|t| !t.is_none()))?;
        match self.resume(py, Op::Throw(PyErr::from_value(exc)))? {
            Sent::Next(v) => Ok(v.unbind()),
            Sent::Return(v) => Err(stop_iteration(v)),
        }
    }

    fn close(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.close_gen(py)
    }

    #[getter]
    fn gi_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    #[getter]
    fn gi_suspended(&self) -> bool {
        self.status() == Status::Suspended && !self.running.load(Ordering::Acquire)
    }

    #[getter]
    fn __name__(&self) -> &str {
        self.rt.ir.name.rsplit('.').next().unwrap_or(&self.rt.ir.name)
    }

    #[getter]
    fn __qualname__(&self) -> &str {
        &self.rt.ir.name
    }

    fn __repr__(slf: &Bound<'_, Self>) -> String {
        format!("<Lizzard generator object {} at {:#x}>", slf.get().rt.ir.name, slf.as_ptr() as usize)
    }
}

/// An unfinished generator is closed when it is freed, so its `finally`
/// blocks run, as CPython does.
impl Drop for LizzardGenerator {
    fn drop(&mut self) {
        let suspended = self.status() == Status::Suspended;
        if !suspended {
            return;
        }
        Python::attach(|py| {
            let saved = PyErr::take(py);
            if let Err(e) = self.close_gen(py) {
                e.restore(py);
                let msg = std::ffi::CString::new(format!("Exception ignored while closing generator <Lizzard generator object {}>", self.rt.ir.name)).unwrap_or_default();
                // SAFETY: reports and clears the exception just restored; `%s`
                // takes a C string.
                unsafe { PyErr_FormatUnraisable(c"%s".as_ptr(), msg.as_ptr()) };
            }
            if let Some(e) = saved {
                e.restore(py);
            }
        });
    }
}

unsafe extern "C" {
    /// CPython 3.13+: report the current exception as unraisable, with a
    /// formatted message.
    fn PyErr_FormatUnraisable(format: *const std::ffi::c_char, ...);
}

/// The exception `throw(typ, val, tb)` raises.
fn make_exception<'py>(typ: &Bound<'py, PyAny>, val: Option<Bound<'py, PyAny>>, tb: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let exc = if let Ok(t) = typ.cast::<PyType>() {
        if !t.is_subclass_of::<PyBaseException>()? {
            return Err(PyTypeError::new_err(format!("exceptions must be classes or instances deriving from BaseException, not {}", t.name()?)));
        }
        match val {
            None => t.call0()?,
            Some(v) if v.is_instance(t)? => v,
            Some(v) => match v.cast::<PyTuple>() {
                Ok(args) => t.call1(args.clone())?,
                Err(_) => t.call1((v,))?,
            },
        }
    } else if typ.is_instance_of::<PyBaseException>() {
        if val.is_some() {
            return Err(PyTypeError::new_err("instance exception may not have a separate value"));
        }
        typ.clone()
    } else {
        return Err(PyTypeError::new_err(format!(
            "exceptions must be classes or instances deriving from BaseException, not {}",
            typ.get_type().name()?
        )));
    };
    match tb {
        Some(tb) => exc.call_method1("with_traceback", (tb,)),
        None => Ok(exc),
    }
}

/// Send `v` to an iterator, as `yield from` does: `next()` for None,
/// `send(v)` otherwise; a Lizzard generator is resumed directly.
pub fn send_to<'py>(it: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>) -> PyResult<Sent<'py>> {
    let py = it.py();
    if let Ok(g) = it.cast::<LizzardGenerator>() {
        return g.get().resume(py, Op::Send(v.clone()));
    }
    let mut out: *mut ffi::PyObject = std::ptr::null_mut();
    // SAFETY: valid objects; `out` receives a new reference unless it fails.
    match unsafe { ffi::PyIter_Send(it.as_ptr(), v.as_ptr(), &mut out) } {
        ffi::PySendResult::PYGEN_NEXT => Ok(Sent::Next(unsafe { Bound::from_owned_ptr(py, out) })),
        ffi::PySendResult::PYGEN_RETURN => Ok(Sent::Return(unsafe { Bound::from_owned_ptr(py, out) })),
        ffi::PySendResult::PYGEN_ERROR => Err(PyErr::fetch(py)),
    }
}

/// Throw an exception into a `yield from`'s iterator, as CPython does:
/// GeneratorExit closes it and is then raised in the delegating generator;
/// an iterator without `throw` gets it raised in the delegating generator.
fn throw_to<'py>(py: Python<'py>, it: &Bound<'py, PyAny>, e: PyErr) -> PyResult<Sent<'py>> {
    if e.is_instance_of::<PyGeneratorExit>(py) {
        if let Ok(close) = it.getattr("close") {
            close.call0()?;
        }
        return Err(e);
    }
    if let Ok(g) = it.cast::<LizzardGenerator>() {
        return g.get().resume(py, Op::Throw(e));
    }
    let Ok(throw) = it.getattr("throw") else { return Err(e) };
    match throw.call1((e.value(py),)) {
        Ok(v) => Ok(Sent::Next(v)),
        Err(err) if err.is_instance_of::<PyStopIteration>(py) => Ok(Sent::Return(err.value(py).getattr("value")?)),
        Err(err) => Err(err),
    }
}

/// The exception being handled (`sys.exception()`), or None.
fn handled(py: Python<'_>) -> Bound<'_, PyAny> {
    // SAFETY: returns a new reference or NULL.
    let p = unsafe { ffi::PyErr_GetHandledException() };
    if p.is_null() { py.None().into_bound(py) } else { unsafe { Bound::from_owned_ptr(py, p) } }
}

fn set_handled(e: &Bound<'_, PyAny>) {
    // SAFETY: a valid object (None clears it); the reference is not stolen.
    unsafe { ffi::PyErr_SetHandledException(e.as_ptr()) };
}
