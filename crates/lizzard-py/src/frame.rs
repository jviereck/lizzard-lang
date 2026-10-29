//! A function's registers, as the interpreter, generators and compiled code
//! all see them: one 8-byte slot per register. An `Obj` slot holds an owned
//! `PyObject*` (null while unbound), an `I64` slot an `i64`, an `F32` slot
//! an `f32` in its low four bytes, and a `Bool` slot 0 or 1. Compiled code
//! reads and writes the slots directly; the frame owns the objects and
//! releases them when it is dropped.
//!
//! Small frames keep their slots inline (a call's frame lives on the Rust
//! stack: no allocation), larger ones on the heap. A window views slots
//! someone else keeps (the interpreter's stack of callee frames), and owns
//! nothing: the interpreter releases those frames itself.

use pyo3::exceptions::PyUnboundLocalError;
use pyo3::ffi;
use pyo3::prelude::*;

const _: () = assert!(std::mem::size_of::<*mut ffi::PyObject>() == std::mem::size_of::<u64>());

const INLINE: usize = 24;

pub struct Regs {
    store: Store,
    len: usize,
    /// Which registers hold objects: the function's `FuncRt::objs`, which
    /// outlives its frames (a generator holds its `FuncRt`).
    objs: *const [bool],
}

enum Store {
    Inline([u64; INLINE]),
    Heap(Vec<u64>),
    /// A window's slots, kept elsewhere.
    External(*mut u64),
}

// SAFETY: the objects are owned references, released with the thread
// attached (`Drop`); free-threaded CPython objects may move between threads.
unsafe impl Send for Regs {}

pub fn unbound() -> PyErr {
    PyUnboundLocalError::new_err("local variable referenced before assignment")
}

impl Regs {
    /// A frame for a function with registers `objs` (which must outlive it).
    pub fn new(objs: &[bool]) -> Regs {
        let len = objs.len();
        let store = if len > INLINE { Store::Heap(vec![0; len]) } else { Store::Inline([0; INLINE]) };
        Regs { store, len, objs: objs as *const [bool] }
    }

    /// A view of `objs.len()` slots at `slots`, kept elsewhere.
    ///
    /// # Safety
    /// The slots stay valid and are only reached through the view while it
    /// is used.
    pub unsafe fn window(slots: *mut u64, objs: &[bool]) -> Regs {
        Regs { store: Store::External(slots), len: objs.len(), objs: objs as *const [bool] }
    }

    fn objs(&self) -> &[bool] {
        // SAFETY: see `objs`.
        unsafe { &*self.objs }
    }

    fn slots(&self) -> &[u64] {
        match &self.store {
            // SAFETY: see `window`.
            Store::External(p) => unsafe { std::slice::from_raw_parts(*p, self.len) },
            Store::Inline(a) => &a[..self.len],
            Store::Heap(v) => &v[..self.len],
        }
    }

    fn slots_mut(&mut self) -> &mut [u64] {
        match &mut self.store {
            // SAFETY: see `window`.
            Store::External(p) => unsafe { std::slice::from_raw_parts_mut(*p, self.len) },
            Store::Inline(a) => &mut a[..self.len],
            Store::Heap(v) => &mut v[..self.len],
        }
    }

    /// The slots, for compiled code and the interpreter's fast paths. The
    /// pointer is valid until the frame moves.
    pub fn as_mut_ptr(&mut self) -> *mut u64 {
        self.slots_mut().as_mut_ptr()
    }

    /// Store a new reference without releasing anything (an unbound slot).
    pub fn init_obj(&mut self, r: u32, v: *mut ffi::PyObject) {
        self.slots_mut()[r as usize] = v as u64;
    }

    /// Take the object out of register `r` (a new reference, or null).
    pub fn take_obj(&mut self, r: u32) -> *mut ffi::PyObject {
        std::mem::take(&mut self.slots_mut()[r as usize]) as *mut ffi::PyObject
    }

    /// Release the objects now, with the thread attached (`Drop` would
    /// first have to make sure it is).
    pub fn release(&mut self, _py: Python<'_>) {
        let objs = self.objs;
        // SAFETY: see `objs`.
        release_slots(self.slots_mut(), unsafe { &*objs });
        // Nothing left for `Drop`.
        self.len = 0;
    }


    fn ptr(&self, r: u32) -> &*mut ffi::PyObject {
        debug_assert!(self.objs()[r as usize], "register {r} is not an object");
        // SAFETY: an object slot holds a `*mut PyObject` of the same size.
        unsafe { &*(&self.slots()[r as usize] as *const u64 as *const *mut ffi::PyObject) }
    }

    /// The object in register `r` (UnboundLocalError while unbound).
    pub fn obj<'a, 'py>(&'a self, py: Python<'py>, r: u32) -> PyResult<&'a Bound<'py, PyAny>> {
        let p = self.ptr(r);
        if p.is_null() {
            return Err(unbound());
        }
        // SAFETY: a non-null object slot holds a reference the frame owns,
        // valid for as long as the frame is borrowed.
        Ok(unsafe { Bound::ref_from_ptr(py, p) })
    }

    pub fn is_bound(&self, r: u32) -> bool {
        !self.objs()[r as usize] || !self.ptr(r).is_null()
    }

    /// Store an object, releasing the one it replaces.
    pub fn set_obj(&mut self, r: u32, v: Bound<'_, PyAny>) {
        let old = std::mem::replace(&mut self.slots_mut()[r as usize], v.into_ptr() as u64) as *mut ffi::PyObject;
        if !old.is_null() {
            // SAFETY: the frame owned the old reference; the thread is
            // attached (`v` proves it).
            unsafe { crate::rc::decref(old) };
        }
    }

    /// Make register `r` unbound again.
    pub fn clear(&mut self, r: u32) {
        let old = std::mem::replace(&mut self.slots_mut()[r as usize], 0);
        if self.objs()[r as usize] && old != 0 {
            // SAFETY: as in `set_obj`; callers are attached.
            unsafe { crate::rc::decref(old as *mut ffi::PyObject) };
        }
    }

    /// `dst = src` for a register of any representation.
    pub fn copy(&mut self, py: Python<'_>, dst: u32, src: u32) -> PyResult<()> {
        if self.objs()[src as usize] {
            let v = self.obj(py, src)?.clone();
            self.set_obj(dst, v);
        } else {
            let v = self.slots()[src as usize];
            self.slots_mut()[dst as usize] = v;
        }
        Ok(())
    }

    pub fn i64(&self, r: u32) -> i64 {
        self.slots()[r as usize] as i64
    }

    pub fn set_i64(&mut self, r: u32, v: i64) {
        self.slots_mut()[r as usize] = v as u64;
    }

    pub fn f32(&self, r: u32) -> f32 {
        f32::from_bits(self.slots()[r as usize] as u32)
    }

    pub fn set_f32(&mut self, r: u32, v: f32) {
        self.slots_mut()[r as usize] = v.to_bits() as u64;
    }

    pub fn bool(&self, r: u32) -> bool {
        self.slots()[r as usize] != 0
    }

    pub fn set_bool(&mut self, r: u32, v: bool) {
        self.slots_mut()[r as usize] = v as u64;
    }
}

/// Release the objects in `slots` (`objs` says which hold objects), leaving
/// them unbound. The thread is attached.
pub fn release_slots(slots: &mut [u64], objs: &[bool]) {
    for (s, &o) in slots.iter_mut().zip(objs) {
        if o && *s != 0 {
            // SAFETY: the frame owns these references; attached.
            unsafe { crate::rc::decref(std::mem::take(s) as *mut ffi::PyObject) };
        }
    }
}

impl Drop for Regs {
    fn drop(&mut self) {
        if let Store::External(_) = self.store {
            return;
        }
        let objs = self.objs;
        // SAFETY: see `objs`.
        let objs = unsafe { &*objs };
        if self.slots().iter().zip(objs).any(|(s, &o)| o && *s != 0) {
            Python::attach(|py| self.release(py));
        }
    }
}
