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
