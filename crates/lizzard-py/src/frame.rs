//! A function's registers, as the interpreter, generators and compiled code
//! all see them: one 8-byte slot per register. An `Obj` slot holds an owned
//! `PyObject*` (null while unbound), an `I64` slot an `i64`, an `F32` slot
//! an `f32` in its low four bytes, and a `Bool` slot 0 or 1. Compiled code
//! reads and writes the slots directly; the frame owns the objects and
//! releases them when it is dropped.

use std::sync::Arc;

use pyo3::exceptions::PyUnboundLocalError;
use pyo3::ffi;
use pyo3::prelude::*;

const _: () = assert!(std::mem::size_of::<*mut ffi::PyObject>() == std::mem::size_of::<u64>());

pub struct Regs {
    slots: Vec<u64>,
    /// Which registers hold objects.
    objs: Arc<[bool]>,
}

// SAFETY: the objects are owned references, released with the thread
// attached (`Drop`); free-threaded CPython objects may move between threads.
unsafe impl Send for Regs {}

pub fn unbound() -> PyErr {
    PyUnboundLocalError::new_err("local variable referenced before assignment")
}

impl Regs {
    pub fn new(objs: Arc<[bool]>) -> Regs {
        let mut slots = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        slots.resize(objs.len(), 0);
        Regs { slots, objs }
    }

    /// The slots, for compiled code.
    pub fn as_mut_ptr(&mut self) -> *mut u64 {
        self.slots.as_mut_ptr()
    }

    fn ptr(&self, r: u32) -> &*mut ffi::PyObject {
        debug_assert!(self.objs[r as usize], "register {r} is not an object");
        // SAFETY: an object slot holds a `*mut PyObject` of the same size.
        unsafe { &*(&self.slots[r as usize] as *const u64 as *const *mut ffi::PyObject) }
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
        !self.objs[r as usize] || !self.ptr(r).is_null()
    }

    /// Store an object, releasing the one it replaces.
    pub fn set_obj(&mut self, r: u32, v: Bound<'_, PyAny>) {
        let old = std::mem::replace(&mut self.slots[r as usize], v.into_ptr() as u64) as *mut ffi::PyObject;
        if !old.is_null() {
            // SAFETY: the frame owned the old reference; the thread is
            // attached (`v` proves it).
            unsafe { ffi::Py_DecRef(old) };
        }
    }

    /// Make register `r` unbound again.
    pub fn clear(&mut self, r: u32) {
        let old = std::mem::replace(&mut self.slots[r as usize], 0);
        if self.objs[r as usize] && old != 0 {
            // SAFETY: as in `set_obj`; callers are attached.
            unsafe { ffi::Py_DecRef(old as *mut ffi::PyObject) };
        }
    }

    /// `dst = src` for a register of any representation.
    pub fn copy(&mut self, py: Python<'_>, dst: u32, src: u32) -> PyResult<()> {
        if self.objs[src as usize] {
            let v = self.obj(py, src)?.clone();
            self.set_obj(dst, v);
        } else {
            self.slots[dst as usize] = self.slots[src as usize];
        }
        Ok(())
    }

    pub fn i64(&self, r: u32) -> i64 {
        self.slots[r as usize] as i64
    }

    pub fn set_i64(&mut self, r: u32, v: i64) {
        self.slots[r as usize] = v as u64;
    }

    pub fn f32(&self, r: u32) -> f32 {
        f32::from_bits(self.slots[r as usize] as u32)
    }

    pub fn set_f32(&mut self, r: u32, v: f32) {
        self.slots[r as usize] = v.to_bits() as u64;
    }

    pub fn bool(&self, r: u32) -> bool {
        self.slots[r as usize] != 0
    }

    pub fn set_bool(&mut self, r: u32, v: bool) {
        self.slots[r as usize] = v as u64;
    }
}

thread_local! {
    /// Slot buffers of finished frames, reused so that a call allocates
    /// nothing.
    static POOL: std::cell::RefCell<Vec<Vec<u64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for Regs {
    fn drop(&mut self) {
        if self.slots.iter().zip(self.objs.iter()).any(|(s, o)| *o && *s != 0) {
            Python::attach(|_| {
                for (s, o) in self.slots.iter_mut().zip(self.objs.iter()) {
                    if *o && *s != 0 {
                        // SAFETY: the frame owns these references; attached.
                        unsafe { ffi::Py_DecRef(std::mem::take(s) as *mut ffi::PyObject) };
                    }
                }
            });
        }
        let mut slots = std::mem::take(&mut self.slots);
        slots.clear();
        let _ = POOL.try_with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < 64 {
                p.push(slots);
            }
        });
    }
}
