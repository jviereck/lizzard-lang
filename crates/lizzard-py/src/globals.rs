//! Cached global and builtin lookups.
//!
//! A `LoadGlobal` site remembers the value it found. A dict watcher on the
//! module's globals and on the builtins bumps `EPOCH` on any change to
//! either, so a remembered value is good while the epoch is the one it was
//! found at. The cache is shared by the threads running the function:
//! readers take no lock (the epoch and value are read under a sequence
//! counter that writers, serialized by a mutex, keep odd while they write),
//! and the value is a borrowed pointer taken with `PyUnstable_TryIncRef`:
//! after `PyUnstable_EnableTryIncRef`, CPython delays freeing the object's
//! memory, so a value another thread just replaced and released is either
//! still alive (the increment succeeds) or fails cleanly.

use std::ffi::c_int;
use std::sync::atomic::{fence, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use pyo3::exceptions::PyNameError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

static EPOCH: AtomicU64 = AtomicU64::new(1);
static WRITE: Mutex<()> = Mutex::new(());

unsafe extern "C" {
    /// CPython 3.14: increment the reference count unless it is zero.
    fn PyUnstable_TryIncRef(obj: *mut ffi::PyObject) -> c_int;
    fn PyUnstable_EnableTryIncRef(obj: *mut ffi::PyObject);
    fn PyDict_AddWatcher(cb: unsafe extern "C" fn(c_int, *mut ffi::PyObject, *mut ffi::PyObject, *mut ffi::PyObject) -> c_int) -> c_int;
}

unsafe extern "C" fn on_change(_event: c_int, _dict: *mut ffi::PyObject, _key: *mut ffi::PyObject, _value: *mut ffi::PyObject) -> c_int {
    EPOCH.fetch_add(1, Ordering::SeqCst);
    0
}

fn watcher() -> c_int {
    static ID: OnceLock<c_int> = OnceLock::new();
    // SAFETY: registers a callback that never fails; attached.
    *ID.get_or_init(|| unsafe { PyDict_AddWatcher(on_change) })
}

/// The address of the epoch, for compiled code.
pub fn epoch_addr() -> usize {
    &EPOCH as *const AtomicU64 as usize
}

/// What a site's cache holds: the epoch and the value (borrowed).
pub fn spec(ic: &GlobalIc) -> Option<(u64, *mut ffi::PyObject)> {
    let s = ic.seq.load(Ordering::Acquire);
    let (e, v) = (ic.epoch.load(Ordering::Relaxed), ic.value.load(Ordering::Relaxed));
    fence(Ordering::Acquire);
    (s & 1 == 0 && ic.seq.load(Ordering::Relaxed) == s && !v.is_null()).then_some((e, v))
}

#[derive(Default)]
pub struct GlobalIc {
    seq: AtomicU32,
    epoch: AtomicU64,
    value: AtomicPtr<ffi::PyObject>,
}

/// `name` in the module's globals, else in the builtins (NameError).
pub fn load<'py>(py: Python<'py>, ic: &GlobalIc, globals: &Bound<'py, PyDict>, builtins: &Bound<'py, PyDict>, name: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyAny>> {
    let s1 = ic.seq.load(Ordering::Acquire);
    if s1 & 1 == 0 {
        let e = ic.epoch.load(Ordering::Relaxed);
        let v = ic.value.load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        if ic.seq.load(Ordering::Relaxed) == s1 && e == EPOCH.load(Ordering::Acquire) && !v.is_null() {
            // SAFETY: see the module comment.
            if unsafe { PyUnstable_TryIncRef(v) } == 1 {
                return Ok(unsafe { Bound::from_owned_ptr(py, v) });
            }
        }
    }
    let id = watcher();
    // SAFETY: valid dicts; watching one again is harmless.
    unsafe {
        ffi::PyDict_Watch(id, globals.as_ptr());
        ffi::PyDict_Watch(id, builtins.as_ptr());
    }
    // The epoch before the lookup: a change during it invalidates the entry.
    let e = EPOCH.load(Ordering::SeqCst);
    let v = match globals.get_item(name)? {
        Some(v) => v,
        None => builtins.get_item(name)?.ok_or_else(|| PyNameError::new_err(format!("name '{name}' is not defined")))?,
    };
    // SAFETY: we hold a strong reference.
    unsafe { PyUnstable_EnableTryIncRef(v.as_ptr()) };
    if let Ok(_w) = WRITE.lock() {
        ic.seq.fetch_add(1, Ordering::AcqRel);
        ic.epoch.store(e, Ordering::Relaxed);
        ic.value.store(v.as_ptr(), Ordering::Relaxed);
        ic.seq.fetch_add(1, Ordering::Release);
    }
    Ok(v)
}
