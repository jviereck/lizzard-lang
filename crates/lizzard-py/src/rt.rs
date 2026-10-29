//! Runtime support: resolving qualified names to Python objects, and the
//! runtime type checks behind annotations.

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};
use lizzard_py_front::ir::RtCheck;

static CLASSES: Mutex<Option<HashMap<String, Option<Py<PyAny>>>>> = Mutex::new(None);

/// The object a qualified name (`numpy.float32`) refers to, or `None` if it
/// does not exist at runtime (some classes exist only in .pyi files).
/// Cached; the lock is never held while running Python code.
pub fn resolve<'py>(py: Python<'py>, qualname: &str) -> Option<Bound<'py, PyAny>> {
    if let Some(hit) = CLASSES.lock().unwrap().get_or_insert_with(HashMap::new).get(qualname) {
        return hit.as_ref().map(|o| o.bind(py).clone());
    }
    let parts: Vec<&str> = qualname.split('.').collect();
    let mut found = None;
    for i in (1..parts.len()).rev() {
        let Ok(module) = py.import(parts[..i].join(".")) else { continue };
        let mut obj = module.into_any();
        let mut ok = true;
        for attr in &parts[i..] {
            match obj.getattr(*attr) {
                Ok(o) => obj = o,
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            found = Some(obj);
        }
        break;
    }
    CLASSES.lock().unwrap().get_or_insert_with(HashMap::new).insert(qualname.to_string(), found.as_ref().map(|o| o.clone().unbind()));
    found
}

/// Whether `obj` passes `check`; on failure, a description of what it is.
pub fn check(py: Python<'_>, obj: &Bound<'_, PyAny>, check: &RtCheck) -> PyResult<Result<(), String>> {
    let ok = match check {
        RtCheck::Any => true,
        RtCheck::Int => obj.is_instance_of::<PyInt>(),
        RtCheck::Float => obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>(),
        RtCheck::Bool => obj.is_instance_of::<PyBool>(),
        RtCheck::Str => obj.is_instance_of::<PyString>(),
        RtCheck::None => obj.is_none(),
        RtCheck::Tuple => obj.is_instance_of::<PyTuple>(),
        RtCheck::Instance(q) => match resolve(py, q) {
            Some(cls) => obj.is_instance(&cls)?,
            None => true,
        },
        RtCheck::NdArray { scalar, ndim } => match resolve(py, "numpy.ndarray") {
            Some(nd) if obj.is_instance(&nd)? => {
                let scalar_ok = match scalar.as_deref().and_then(|s| resolve(py, s)) {
                    Some(cls) => obj.getattr("dtype")?.getattr("type")?.is(&cls),
                    None => true,
                };
                let ndim_ok = match ndim {
                    Some(n) => obj.getattr("ndim")?.extract::<u32>()? == *n,
                    None => true,
                };
                scalar_ok && ndim_ok
            }
            _ => false,
        },
        RtCheck::Callable => obj.is_callable(),
        RtCheck::Union(cs) => {
            let mut any = false;
            for c in cs {
                if self::check(py, obj, c)?.is_ok() {
                    any = true;
                    break;
                }
            }
            any
        }
    };
    if ok {
        return Ok(Ok(()));
    }
    Ok(Err(describe(py, obj)?))
}

/// What a value is, for error messages: its type, and for arrays their
/// dtype and number of dimensions.
pub fn describe(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Some(nd) = resolve(py, "numpy.ndarray") {
        if obj.is_instance(&nd)? {
            let dtype = obj.getattr("dtype")?.str()?.to_string();
            let ndim: u32 = obj.getattr("ndim")?.extract()?;
            return Ok(format!("ndarray with dtype {dtype} and ndim {ndim}"));
        }
    }
    Ok(obj.get_type().name()?.to_string())
}

/// A check with its classes looked up, for one check site: resolving a
/// qualified name on every check would dominate calls.
pub enum Ck {
    Any,
    Int,
    Float,
    Bool,
    Str,
    None,
    Tuple,
    Class(Py<PyAny>),
    Callable,
    Union(Vec<Ck>),
    /// Checked by `check` (NumPy arrays).
    Other(RtCheck),
}

pub fn prepare(py: Python<'_>, c: &RtCheck) -> Ck {
    match c {
        RtCheck::Any => Ck::Any,
        RtCheck::Int => Ck::Int,
        RtCheck::Float => Ck::Float,
        RtCheck::Bool => Ck::Bool,
        RtCheck::Str => Ck::Str,
        RtCheck::None => Ck::None,
        RtCheck::Tuple => Ck::Tuple,
        RtCheck::Instance(q) => match resolve(py, q) {
            Some(cls) => Ck::Class(cls.unbind()),
            None => Ck::Any,
        },
        RtCheck::Callable => Ck::Callable,
        RtCheck::Union(cs) => Ck::Union(cs.iter().map(|c| prepare(py, c)).collect()),
        c @ RtCheck::NdArray { .. } => Ck::Other(c.clone()),
    }
}

fn passes(py: Python<'_>, obj: &Bound<'_, PyAny>, ck: &Ck) -> PyResult<bool> {
    Ok(match ck {
        Ck::Any => true,
        Ck::Int => obj.is_instance_of::<PyInt>(),
        Ck::Float => obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>(),
        Ck::Bool => obj.is_instance_of::<PyBool>(),
        Ck::Str => obj.is_instance_of::<PyString>(),
        Ck::None => obj.is_none(),
        Ck::Tuple => obj.is_instance_of::<PyTuple>(),
        // The exact class first: a pointer comparison.
        Ck::Class(cls) => obj.get_type().is(cls.bind(py)) || obj.is_instance(cls.bind(py))?,
        Ck::Callable => obj.is_callable(),
        Ck::Union(cs) => {
            for c in cs {
                if passes(py, obj, c)? {
                    return Ok(true);
                }
            }
            false
        }
        Ck::Other(c) => check(py, obj, c)?.is_ok(),
    })
}

/// `check` for a prepared check.
pub fn check_prepared(py: Python<'_>, obj: &Bound<'_, PyAny>, ck: &Ck) -> PyResult<Result<(), String>> {
    if passes(py, obj, ck)? {
        return Ok(Ok(()));
    }
    Ok(Err(describe(py, obj)?))
}

/// The part of a check that type pointers decide: true when `o` passes for
/// sure; false when the full check must decide.
///
/// # Safety
/// `o` is a valid object.
pub unsafe fn quick(o: *mut pyo3::ffi::PyObject, ck: &Ck) -> bool {
    use pyo3::ffi;
    unsafe {
        match ck {
            Ck::Any => true,
            Ck::Int => ffi::PyLong_CheckExact(o) != 0,
            Ck::Float => ffi::PyFloat_CheckExact(o) != 0 || ffi::PyLong_CheckExact(o) != 0,
            Ck::Bool => o == ffi::Py_True() || o == ffi::Py_False(),
            Ck::Str => ffi::PyUnicode_CheckExact(o) != 0,
            Ck::None => o == ffi::Py_None(),
            Ck::Tuple => ffi::PyTuple_CheckExact(o) != 0,
            Ck::Class(c) => ffi::Py_TYPE(o) as *mut ffi::PyObject == c.as_ptr(),
            _ => false,
        }
    }
}
