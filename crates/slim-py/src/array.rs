//! Direct element access for 1-D float32 NumPy arrays.
//!
//! `ArrLoadF32` / `ArrStoreF32` read the array's fields (NumPy's public
//! `PyArrayObject_fields` layout) and access the element in its buffer when
//! the array is an exact `ndarray`, 1-D, float32 in native byte order and
//! aligned (and writeable, for stores). Anything else takes the generic path
//! through `__getitem__` / `__setitem__`, with the same result. The layout is
//! validated once against `__array_interface__`; on a mismatch the direct
//! path stays off.

use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

use crate::rt::resolve;

#[repr(C)]
struct ArrayFields {
    ob_base: ffi::PyObject,
    data: *mut u8,
    nd: c_int,
    dimensions: *mut isize,
    strides: *mut isize,
    base: *mut ffi::PyObject,
    descr: *mut Descr,
    flags: c_int,
}

#[repr(C)]
struct Descr {
    ob_base: ffi::PyObject,
    typeobj: *mut ffi::PyTypeObject,
    kind: c_char,
    type_: c_char,
    byteorder: c_char,
}

const ALIGNED: c_int = 0x0100;
const WRITEABLE: c_int = 0x0400;

pub static FAST: AtomicU64 = AtomicU64::new(0);
pub static GENERIC: AtomicU64 = AtomicU64::new(0);

struct NpTypes {
    ndarray: Py<PyAny>,
    float32: Py<PyAny>,
    /// The field layout matched `__array_interface__` for a probe array.
    direct: bool,
}

static NP: PyOnceLock<Option<NpTypes>> = PyOnceLock::new();

fn np(py: Python<'_>) -> Option<&NpTypes> {
    NP.get_or_init(py, || {
        let ndarray = resolve(py, "numpy.ndarray")?;
        let float32 = resolve(py, "numpy.float32")?;
        let direct = validate(py, &ndarray, &float32).unwrap_or(false);
        Some(NpTypes { ndarray: ndarray.unbind(), float32: float32.unbind(), direct })
    })
    .as_ref()
}

fn validate(py: Python<'_>, ndarray: &Bound<'_, PyAny>, float32: &Bound<'_, PyAny>) -> PyResult<bool> {
    let numpy = py.import("numpy")?;
    let kw = pyo3::types::PyDict::new(py);
    kw.set_item("dtype", float32)?;
    let probe = numpy.call_method("arange", (8,), Some(&kw))?.get_item(pyo3::types::PySlice::new(py, 1, 8, 3))?;
    let iface = probe.getattr("__array_interface__")?;
    let data: usize = iface.get_item("data")?.get_item(0)?.extract()?;
    let strides: (isize,) = probe.getattr("strides")?.extract()?;
    let shape: (isize,) = probe.getattr("shape")?.extract()?;
    let ok = unsafe {
        let p = probe.as_ptr();
        let a = &*(p as *const ArrayFields);
        ffi::Py_TYPE(p) == ndarray.as_ptr() as *mut ffi::PyTypeObject
            && a.data as usize == data
            && a.nd == 1
            && *a.dimensions == shape.0
            && *a.strides == strides.0
            && !a.descr.is_null()
            && (*a.descr).typeobj == float32.as_ptr() as *mut ffi::PyTypeObject
    };
    Ok(ok)
}

fn native(byteorder: c_char) -> bool {
    matches!(byteorder as u8, b'=' | b'|') || (cfg!(target_endian = "little") && byteorder as u8 == b'<') || (cfg!(target_endian = "big") && byteorder as u8 == b'>')
}

/// The element's address, if the direct path applies. Bounds are checked
/// here (negative indices count from the end, as in NumPy).
fn element(py: Python<'_>, arr: &Bound<'_, PyAny>, idx: i64, write: bool) -> PyResult<Option<*mut u8>> {
    let Some(np) = np(py) else { return Ok(None) };
    if !np.direct {
        return Ok(None);
    }
    unsafe {
        let p = arr.as_ptr();
        if ffi::Py_TYPE(p) != np.ndarray.as_ptr() as *mut ffi::PyTypeObject {
            return Ok(None);
        }
        let a = &*(p as *const ArrayFields);
        if a.nd != 1 || a.descr.is_null() {
            return Ok(None);
        }
        let d = &*a.descr;
        if d.typeobj != np.float32.as_ptr() as *mut ffi::PyTypeObject || !native(d.byteorder) {
            return Ok(None);
        }
        if a.flags & ALIGNED == 0 || (write && a.flags & WRITEABLE == 0) {
            return Ok(None);
        }
        let n = *a.dimensions as i64;
        let i = if idx < 0 { idx + n } else { idx };
        if i < 0 || i >= n {
            return Err(PyIndexError::new_err(format!("index {idx} is out of bounds for axis 0 with size {n}")));
        }
        Ok(Some(a.data.offset(i as isize * *a.strides)))
    }
}

fn float32<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    resolve(py, "numpy.float32").ok_or_else(|| PyTypeError::new_err("numpy is not available"))
}

pub fn box_f32<'py>(py: Python<'py>, v: f32) -> PyResult<Bound<'py, PyAny>> {
    float32(py)?.call1((v as f64,))
}

pub fn load_f32(py: Python<'_>, arr: &Bound<'_, PyAny>, idx: i64) -> PyResult<f32> {
    if let Some(p) = element(py, arr, idx, false)? {
        FAST.fetch_add(1, Ordering::Relaxed);
        // Aligned (checked above); relaxed atomics keep racing accesses from
        // other threads defined.
        let bits = unsafe { (*(p as *const AtomicU32)).load(Ordering::Relaxed) };
        return Ok(f32::from_bits(bits));
    }
    GENERIC.fetch_add(1, Ordering::Relaxed);
    let v = arr.get_item(idx)?;
    if !v.is_instance(&float32(py)?)? {
        let got = crate::rt::describe(py, &v)?;
        return Err(PyTypeError::new_err(format!("expected a float32 element, got {got}")));
    }
    Ok(v.extract::<f64>()? as f32)
}

pub fn store_f32(py: Python<'_>, arr: &Bound<'_, PyAny>, idx: i64, v: f32) -> PyResult<()> {
    if let Some(p) = element(py, arr, idx, true)? {
        FAST.fetch_add(1, Ordering::Relaxed);
        unsafe { (*(p as *const AtomicU32)).store(v.to_bits(), Ordering::Relaxed) };
        return Ok(());
    }
    GENERIC.fetch_add(1, Ordering::Relaxed);
    arr.set_item(idx, box_f32(py, v)?)
}
