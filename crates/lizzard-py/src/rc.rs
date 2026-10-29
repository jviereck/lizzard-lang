//! Reference counting inline, as CPython's own `Py_INCREF` / `Py_DECREF`
//! (`Include/refcount.h`, 3.14) do it, instead of calling libpython's
//! `Py_IncRef` / `Py_DecRef` for every operation.
//!
//! - GIL build: a 32-bit count in the low half of the first word; a negative
//!   value (as `i32`) marks an immortal object; `_Py_Dealloc` at zero.
//! - Free-threaded build: the thread that owns an object (`ob_tid` is its
//!   thread id: the thread pointer, `fs:0` on x86-64 Linux) changes the
//!   non-atomic local count, and calls `_Py_MergeZeroLocalRefcount` when it
//!   reaches zero; other threads add to the shared count atomically (in
//!   units of `1 << 2`) or call `_Py_DecRefShared`. A local count of
//!   `u32::MAX` marks an immortal object.
//!
//! `init` checks the layout against CPython's own counts at start-up; if it
//! does not match (or the platform has no known thread pointer), the
//! library functions are used.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(Py_GIL_DISABLED)]
use std::sync::atomic::AtomicIsize;

use pyo3::ffi;

static INLINE: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    #[cfg(Py_GIL_DISABLED)]
    fn _Py_MergeZeroLocalRefcount(o: *mut ffi::PyObject);
    #[cfg(Py_GIL_DISABLED)]
    fn _Py_DecRefShared(o: *mut ffi::PyObject);
    #[cfg(not(Py_GIL_DISABLED))]
    fn _Py_Dealloc(o: *mut ffi::PyObject);
}

/// CPython's `_Py_ThreadId()`: the thread pointer.
#[cfg(Py_GIL_DISABLED)]
#[inline(always)]
fn thread_id() -> Option<usize> {
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "freebsd")))]
    {
        let tid: usize;
        // SAFETY: reads the thread pointer, as CPython does.
        unsafe { std::arch::asm!("mov {}, qword ptr fs:[0]", out(reg) tid, options(nostack, readonly, preserves_flags)) };
        Some(tid)
    }
    #[cfg(target_arch = "aarch64")]
    {
        let tid: usize;
        // SAFETY: reads the thread pointer register, as CPython does.
        unsafe { std::arch::asm!("mrs {}, tpidr_el0", out(reg) tid, options(nostack, nomem, preserves_flags)) };
        Some(tid)
    }
    #[cfg(not(any(all(target_arch = "x86_64", any(target_os = "linux", target_os = "freebsd")), target_arch = "aarch64")))]
    {
        None
    }
}

#[cfg(Py_GIL_DISABLED)]
mod layout {
    pub const TID: usize = 0;
    pub const LOCAL: usize = 12;
    pub const SHARED: usize = 16;
    pub const SHARED_SHIFT: u32 = 2;
}

/// `Py_INCREF(o)`.
#[inline(always)]
pub unsafe fn incref(o: *mut ffi::PyObject) {
    if !INLINE.load(Ordering::Relaxed) {
        return unsafe { ffi::Py_IncRef(o) };
    }
    unsafe {
        let p = o as *mut u8;
        #[cfg(Py_GIL_DISABLED)]
        {
            let local = p.add(layout::LOCAL) as *mut u32;
            let new = (*local).wrapping_add(1);
            if new == 0 {
                return; // immortal
            }
            if *(p.add(layout::TID) as *const usize) == thread_id().unwrap_unchecked() {
                *local = new;
            } else {
                (*(p.add(layout::SHARED) as *const AtomicIsize)).fetch_add(1 << layout::SHARED_SHIFT, Ordering::Relaxed);
            }
        }
        #[cfg(not(Py_GIL_DISABLED))]
        {
            let cnt = p as *mut u32;
            if (*cnt as i32) < 0 {
                return; // immortal
            }
            *cnt += 1;
        }
    }
}

/// `Py_DECREF(o)`.
#[inline(always)]
pub unsafe fn decref(o: *mut ffi::PyObject) {
    if !INLINE.load(Ordering::Relaxed) {
        return unsafe { ffi::Py_DecRef(o) };
    }
    unsafe {
        let p = o as *mut u8;
        #[cfg(Py_GIL_DISABLED)]
        {
            let local = p.add(layout::LOCAL) as *mut u32;
            let l = *local;
            if l == u32::MAX {
                return; // immortal
            }
            if *(p.add(layout::TID) as *const usize) == thread_id().unwrap_unchecked() {
                *local = l - 1;
                if l == 1 {
                    _Py_MergeZeroLocalRefcount(o);
                }
            } else {
                _Py_DecRefShared(o);
            }
        }
        #[cfg(not(Py_GIL_DISABLED))]
        {
            let cnt = p as *mut u32;
            if (*cnt as i32) < 0 {
                return; // immortal
            }
            *cnt -= 1;
            if *cnt == 0 {
                _Py_Dealloc(o);
            }
        }
    }
}

/// Check the inline paths against CPython's own counts (`Py_REFCNT`) on a
/// fresh object and an immortal one, and turn them on if they agree.
pub fn init() {
    #[cfg(Py_GIL_DISABLED)]
    if thread_id().is_none() {
        return;
    }
    unsafe {
        let o = ffi::PyList_New(0);
        if o.is_null() {
            ffi::PyErr_Clear();
            return;
        }
        let before = ffi::Py_REFCNT(o);
        INLINE.store(true, Ordering::Relaxed);
        incref(o);
        incref(o);
        let up = ffi::Py_REFCNT(o);
        decref(o);
        let down = ffi::Py_REFCNT(o);
        let none = ffi::Py_None();
        let n0 = ffi::Py_REFCNT(none);
        incref(none);
        decref(none);
        let ok = up == before + 2 && down == before + 1 && ffi::Py_REFCNT(none) == n0;
        INLINE.store(ok, Ordering::Relaxed);
        // Back to one reference, then free it, with CPython's functions.
        ffi::Py_DecRef(o);
        ffi::Py_DecRef(o);
    }
}

/// Whether reference counting is inline (the start-up check passed).
pub fn inline_on() -> bool {
    INLINE.load(Ordering::Relaxed)
}
