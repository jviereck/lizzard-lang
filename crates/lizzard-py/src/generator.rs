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
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
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

/// What resuming it takes, where compiled code finds it (`lizzard_jit::GEN_*`
/// offsets from `Hot`, at `hot_offset()` in the object): a compiled `for`
/// resumes a plainly suspended generator itself. All but the flags are
/// used only by the thread that set `running`.
#[repr(C)]
pub(crate) struct Hot {
    running: AtomicU8,
    status: AtomicU8,
    /// 1: frame present, no `yield from` going on, no handled exception kept,
    /// and code that never sets one.
    plain: AtomicU8,
    _pad: u8,
    /// The `yield` / `yield from` it is suspended at (0 before it starts).
    pc: UnsafeCell<u32>,
    /// Where a value sent to that `yield` goes; `NO_SEND` if its code never
    /// reads it.
    dst: UnsafeCell<u32>,
    _pad2: u32,
    /// The frame's slots, once it ran in place (null before, and when done).
    slots: UnsafeCell<*mut u64>,
    /// Its `jit::Target`, with `slots`.
    target: UnsafeCell<*const c_void>,
    /// Its function's compiled entry (`JitCell::entry`).
    entry: *const AtomicUsize,
    /// Per pc: `FuncRt::yield_dst`.
    yield_dst: *const u32,
}

const _: () = {
    assert!(std::mem::offset_of!(Hot, running) == lizzard_jit::GEN_RUNNING as usize);
    assert!(std::mem::offset_of!(Hot, status) == lizzard_jit::GEN_STATUS as usize);
    assert!(std::mem::offset_of!(Hot, plain) == lizzard_jit::GEN_PLAIN as usize);
    assert!(std::mem::offset_of!(Hot, pc) == lizzard_jit::GEN_PC as usize);
    assert!(std::mem::offset_of!(Hot, dst) == lizzard_jit::GEN_DST as usize);
    assert!(std::mem::offset_of!(Hot, slots) == lizzard_jit::GEN_SLOTS as usize);
    assert!(std::mem::offset_of!(Hot, target) == lizzard_jit::GEN_TARGET as usize);
    assert!(std::mem::offset_of!(Hot, entry) == lizzard_jit::GEN_ENTRY as usize);
    assert!(std::mem::offset_of!(Hot, yield_dst) == lizzard_jit::GEN_YIELD_DST as usize);
    assert!(Status::Suspended as u8 == lizzard_jit::GEN_SUSPENDED);
};

/// Where `Hot` is in a generator object (known once one exists).
static HOT_OFFSET: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn hot_offset() -> Option<usize> {
    Some(HOT_OFFSET.load(Ordering::Relaxed)).filter(|&o| o != 0)
}

/// The generator a Lizzard generator function returns.
#[pyclass(frozen, module = "lizzard._lizzard")]
pub struct LizzardGenerator {
    rt: Arc<FuncRt>,
    globals: Py<PyDict>,
    closure: Vec<Py<PyAny>>,
    hot: Hot,
    /// None once it is done. Only who set `running` touches it.
    frame: UnsafeCell<Option<Frame>>,
    /// Its `jit::Target`, made where the generator lives (`Hot::target`).
    target: UnsafeCell<MaybeUninit<crate::jit::Target<'static, 'static>>>,
}

// SAFETY: `frame`, `target` and `hot`'s plain fields are used only by the
// thread that set `running` (or by `Drop`, which has the generator to
// itself).
unsafe impl Sync for LizzardGenerator {}
// SAFETY: as for `Sync`; the raw pointers point into the generator and its
// `FuncRt`, which it keeps.
unsafe impl Send for LizzardGenerator {}

impl LizzardGenerator {
    /// A generator over a new frame with the (checked) arguments.
    pub fn new(rt: Arc<FuncRt>, globals: Py<PyDict>, closure: Vec<Py<PyAny>>, args: Vec<Bound<'_, PyAny>>) -> LizzardGenerator {
        let mut regs = Regs::new(&rt.objs);
        for (i, a) in args.into_iter().enumerate() {
            regs.set_obj(i as u32, a);
        }
        let frame = Frame { regs, delegate: None, handled: None };
        let hot = Hot {
            running: AtomicU8::new(0),
            status: AtomicU8::new(Status::Created as u8),
            plain: AtomicU8::new(0),
            _pad: 0,
            pc: UnsafeCell::new(0),
            dst: UnsafeCell::new(NO_SEND),
            _pad2: 0,
            slots: UnsafeCell::new(std::ptr::null_mut()),
            target: UnsafeCell::new(std::ptr::null()),
            entry: rt.jit.entry_cell(),
            yield_dst: rt.yield_dst.as_ptr(),
        };
        LizzardGenerator { rt, globals, closure, hot, frame: UnsafeCell::new(Some(frame)), target: UnsafeCell::new(MaybeUninit::uninit()) }
    }

    /// The object for `g` (and where its `Hot` is, the first time).
    pub fn create(py: Python<'_>, g: LizzardGenerator) -> PyResult<Py<LizzardGenerator>> {
        let obj = Py::new(py, g)?;
        if HOT_OFFSET.load(Ordering::Relaxed) == 0 {
            let off = &obj.get().hot as *const Hot as usize - obj.as_ptr() as usize;
            HOT_OFFSET.store(off, Ordering::Relaxed);
        }
        Ok(obj)
    }

    fn status(&self) -> Status {
        Status::from(self.hot.status.load(Ordering::Acquire))
    }

    fn set_status(&self, s: Status) {
        self.hot.status.store(s as u8, Ordering::Release);
    }

    /// The generator object `o` (borrowed: no reference taken).
    ///
    /// # Safety
    /// `o` is a LizzardGenerator, alive while the result is used.
    unsafe fn of<'a>(o: *mut ffi::PyObject) -> &'a LizzardGenerator {
        let at = o as usize + HOT_OFFSET.load(Ordering::Relaxed) - std::mem::offset_of!(LizzardGenerator, hot);
        // SAFETY: as required (`HOT_OFFSET` is set once one exists).
        unsafe { &*(at as *const LizzardGenerator) }
    }

    /// Take `running` (false: it is already). With plain stores when no
    /// other thread can reach the generator (`rc::unique`), else atomically.
    fn start(&self) -> bool {
        let off = HOT_OFFSET.load(Ordering::Relaxed);
        let obj = (&self.hot as *const Hot as usize - off) as *mut ffi::PyObject;
        // SAFETY: the object holding this generator (`off` is known once one
        // exists, and generators exist only as objects).
        if off != 0 && unsafe { crate::rc::unique(obj) } {
            if self.hot.running.load(Ordering::Relaxed) != 0 {
                return false;
            }
            self.hot.running.store(1, Ordering::Relaxed);
            return true;
        }
        self.hot.running.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok()
    }

    fn stop(&self) {
        self.hot.running.store(0, Ordering::Release);
    }

    /// The `yield` it is suspended at, and where a value sent to it goes.
    fn at(&self) -> (usize, u32) {
        // SAFETY: the caller set `running`.
        unsafe { (*self.hot.pc.get() as usize, *self.hot.dst.get()) }
    }

    /// Suspended at `pc` (a `yield` or `yield from`).
    fn suspend_at(&self, pc: usize, frame: &Frame) {
        // SAFETY: the caller set `running`.
        unsafe {
            *self.hot.pc.get() = pc as u32;
            *self.hot.dst.get() = self.rt.yield_dst.get(pc).copied().unwrap_or(NO_SEND);
        }
        let plain = !self.rt.sets_handled && frame.delegate.is_none() && frame.handled.is_none();
        self.hot.plain.store(plain as u8, Ordering::Release);
    }

    /// Done: its frame released (which may run Python code; `running` is
    /// still set).
    fn finish(&self, slot: &mut Option<Frame>) {
        self.hot.plain.store(0, Ordering::Release);
        self.set_status(Status::Done);
        // SAFETY: the caller set `running`.
        unsafe { *self.hot.slots.get() = std::ptr::null_mut() };
        drop(slot.take());
    }

    /// The frame's slots and its target, made the first time it runs in
    /// place (the frame and the object do not move).
    fn place(&self, py: Python<'_>, frame: &mut Frame) -> (*mut u64, *const c_void) {
        // SAFETY: the caller set `running`.
        unsafe {
            if (*self.hot.slots.get()).is_null() {
                let t = crate::jit::Target::new(py, &self.rt, self.globals.bind(py), &self.closure);
                // SAFETY: it refers to the generator's own fields, which
                // live as long as it does.
                let t: crate::jit::Target<'static, 'static> = std::mem::transmute(t);
                (*self.target.get()).write(t);
                *self.hot.target.get() = (*self.target.get()).as_ptr() as *const c_void;
                *self.hot.slots.get() = frame.regs.as_mut_ptr();
            }
            (*self.hot.slots.get(), *self.hot.target.get())
        }
    }

    /// Resume with a value or an exception; the next value it yields, or the
    /// value it returned.
    pub fn resume<'py>(&self, py: Python<'py>, op: Op<'py>) -> PyResult<Sent<'py>> {
        if !self.start() {
            return Err(PyValueError::new_err("generator already executing"));
        }
        let r = self.resume_inner(py, op);
        self.stop();
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
        let mut pc = if status == Status::Suspended { self.at().0 } else { 0 };
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
                            send(frame, self.at().1, v);
                            pc += 1;
                        }
                        Err(e) => pending = Some(e),
                    }
                }
                None => match op {
                    Op::Send(v) => {
                        send(frame, self.at().1, v);
                        pc += 1;
                    }
                    Op::Throw(e) => pending = Some(e),
                },
            }
        }
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
        let suspend = |slot: &mut Option<Frame>, value: Bound<'py, PyAny>, delegate: Option<Bound<'py, PyAny>>| {
            let frame = slot.as_mut().expect("a running generator's frame");
            frame.delegate = delegate.map(Bound::unbind);
            frame.handled = handled;
            self.suspend_at(pc, frame);
            Sent::Next(value)
        };
        match r {
            Ok(Exit::Yield { value, .. }) => Ok(suspend(slot, value, None)),
            Ok(Exit::YieldFrom { value, iter, .. }) => Ok(suspend(slot, value, Some(iter))),
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
    /// not running, and keeps no handled exception (code that sets one
    /// goes through `resume`); else `Slow` (`resume` does it). `want_int`:
    /// an int it yields unboxed comes as it is.
    #[inline(always)]
    pub(crate) fn next_direct<'py>(&self, py: Python<'py>, want_int: bool) -> PyResult<Next<'py>> {
        let Some(entry) = crate::jit::loaded(&self.rt) else { return Ok(Next::Slow) };
        if self.rt.sets_handled || !self.start() {
            return Ok(Next::Slow);
        }
        let r = self.next_running(py, entry, want_int);
        self.stop();
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
        let (at, dst) = self.at();
        let mut pending = None;
        let pc = match (status, &frame.delegate) {
            // `yield from` another Lizzard generator: its next item, the
            // same way (still running, as Python's are); its end continues
            // here with its return value, its exception is raised at the
            // `yield from`.
            (Status::Suspended, Some(d)) => {
                // (Borrowed: the outer frame keeps it while the outer runs.)
                match delegate_next(py, d.as_ptr(), want_int) {
                    Ok(Next::End(v)) => {
                        frame.delegate = None;
                        send(frame, dst, v);
                        at + 1
                    }
                    Err(e) => {
                        frame.delegate = None;
                        pending = Some(e);
                        at
                    }
                    r => return r,
                }
            }
            (Status::Suspended, None) => {
                // `next()` sends None to the `yield` (if its code reads it).
                if dst != NO_SEND {
                    frame.regs.set_none(dst);
                }
                at + 1
            }
            _ => 0,
        };
        self.set_status(Status::Suspended);
        let (slots, target) = self.place(py, frame);
        let throw = pending.is_some();
        if let Some(e) = pending {
            e.restore(py);
        }
        let mut ctx = crate::jit::Ctx::raw(target, slots, want_int);
        let mut p = pc as u32;
        // SAFETY: the entry was compiled from this function; the frame has
        // its slots; `ctx` outlives the call.
        let st = unsafe { entry(&mut ctx as *mut _ as *mut c_void, slots, &mut p, throw as u32) };
        match st {
            lizzard_jit::YIELD => {
                self.suspend_at(p as usize, frame);
                // SAFETY: compiled code stored a new reference.
                Ok(Next::Value(unsafe { Bound::from_owned_ptr(py, ctx.ret()) }))
            }
            lizzard_jit::YIELD_INT => {
                self.suspend_at(p as usize, frame);
                Ok(Next::Int(ctx.ret() as i64))
            }
            st => self.ended(py, slot, &mut ctx, st, p as usize),
        }
    }

    /// Its code stopped other than at a plain `yield` (`st`, at `pc`):
    /// returned, raised, bailed out, or left through `step`.
    #[cold]
    fn ended<'py>(&self, py: Python<'py>, slot: &mut Option<Frame>, ctx: &mut crate::jit::Ctx<'_, 'py>, st: u32, pc: usize) -> PyResult<Next<'py>> {
        let mut pc = pc;
        let frame = slot.as_mut().expect("a running generator's frame");
        let ran = crate::jit::ran(py, &self.rt, ctx, st);
        let r = crate::jit::settle(py, &self.rt, self.globals.bind(py), &self.closure, &mut frame.regs, &mut pc, ran);
        Ok(match self.after(py, slot, pc, r, None)? {
            Sent::Next(v) => Next::Value(v),
            Sent::Return(v) => Next::End(v),
        })
    }

    fn close_gen(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.status() != Status::Suspended {
            if self.start() {
                // SAFETY: `running` was just set by this thread.
                self.finish(unsafe { &mut *self.frame.get() });
                self.stop();
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

/// A sent value for register `dst` (dropped if its code never reads it).
fn send(frame: &mut Frame, dst: u32, v: Bound<'_, PyAny>) {
    if dst != NO_SEND {
        frame.regs.set_obj(dst, v);
    }
}

/// `Hot::dst` of a `yield` whose value is never read.
pub(crate) const NO_SEND: u32 = lizzard_jit::NO_SEND;

/// A compiled `for` resumed generator `it` itself and its code stopped
/// other than at a plain `yield` (`st`, at `*pc`, its context `ctx`):
/// `for_iter_int`'s status, its item in `dst` or `*out`. `running` is set.
///
/// # Safety
/// As `for_iter_int`; `ctx` is the context compiled code made.
pub(crate) unsafe fn resumed(py: Python<'_>, it: *mut ffi::PyObject, ctx: *mut c_void, st: u32, pc: u32, slots: *mut u64, dst: u32, out: *mut i64) -> u32 {
    // SAFETY: as required (the loop's register keeps it).
    let g = unsafe { LizzardGenerator::of(it) };
    // SAFETY: compiled code set `running`.
    let slot = unsafe { &mut *g.frame.get() };
    let ctx = unsafe { &mut *(ctx as *mut crate::jit::Ctx<'_, '_>) };
    let r = g.ended(py, slot, ctx, st, pc as usize);
    g.stop();
    // SAFETY: as required.
    unsafe { item(py, r, slots, dst, (!out.is_null()).then_some(out)) }
}

/// The next item of the iterator a `yield from` delegates to (`next_direct`
/// of a Lizzard generator; `Slow` for others). Kept out of line: the
/// recursion would stop the compiler from inlining `next_direct` into the
/// loop helpers.
#[inline(never)]
fn delegate_next<'py>(py: Python<'py>, d: *mut ffi::PyObject, want_int: bool) -> PyResult<Next<'py>> {
    // SAFETY: a type pointer comparison on a valid object.
    if unsafe { ffi::Py_TYPE(d) } != type_ptr() {
        return Ok(Next::Slow);
    }
    // SAFETY: the type says what it is; the caller keeps it.
    unsafe { LizzardGenerator::of(d) }.next_direct(py, want_int)
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
    // SAFETY: as required (the loop's register keeps it).
    let g = unsafe { LizzardGenerator::of(it) };
    let r = match g.next_direct(py, out.is_some()) {
        Ok(Next::Slow) => match g.resume(py, Op::Send(py.None().into_bound(py))) {
            Ok(Sent::Next(v)) => Ok(Next::Value(v)),
            Ok(Sent::Return(v)) => Ok(Next::End(v)),
            Err(e) => Err(e),
        },
        r => r,
    };
    // SAFETY: as required.
    unsafe { item(py, r, slots, dst, out) }
}

/// A `for`'s next item: in `dst` (`NEXT`), an int in `*out` (`NEXT_INT`),
/// the end (`TAKEN`), or `ERROR` with the exception set.
unsafe fn item(py: Python<'_>, r: PyResult<Next<'_>>, slots: *mut u64, dst: u32, out: Option<*mut i64>) -> u32 {
    use lizzard_jit::{ERROR, NEXT, NEXT_INT, TAKEN};
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
        self.hot.running.load(Ordering::Acquire) != 0
    }

    #[getter]
    fn gi_suspended(&self) -> bool {
        self.status() == Status::Suspended && self.hot.running.load(Ordering::Acquire) == 0
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
