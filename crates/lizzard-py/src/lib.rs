//! `lizzard._lizzard`: compiles Lizzard modules (typed Python) and runs their
//! functions inside CPython.

mod array;
mod frame;
mod generator;
mod globals;
mod interp;
mod jit;
mod ops;
mod rt;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use interp::{FuncRt, LizzardFunction, Table};

/// The checked and lowered functions of one Lizzard module.
#[pyclass(frozen, module = "lizzard._lizzard")]
struct CompiledModule {
    funcs: Vec<Arc<FuncRt>>,
}

#[pymethods]
impl CompiledModule {
    fn function_names(&self) -> Vec<String> {
        self.funcs.iter().map(|f| f.ir.key.clone()).collect()
    }

    /// The function `name`, with `globals` as its module namespace.
    fn make_function(&self, py: Python<'_>, name: &str, globals: Bound<'_, PyDict>) -> PyResult<Py<LizzardFunction>> {
        let f = self.funcs.iter().find(|f| f.ir.key == name).ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_string()))?;
        let module = match globals.get_item("__name__")? {
            Some(n) => n.extract()?,
            None => String::new(),
        };
        LizzardFunction::create(py, LizzardFunction::new(f.clone(), globals.unbind(), module, Vec::new(), Vec::new()))
    }
}

fn paths(env: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<PathBuf>> {
    let v: Vec<String> = env.get_item(key)?.map(|v| v.extract()).transpose()?.unwrap_or_default();
    Ok(v.into_iter().map(PathBuf::from).collect())
}

/// Check and lower a Lizzard module. Returns `(module or None, diagnostics,
/// warnings)`; the module is `None` when there are diagnostics.
#[pyfunction]
fn compile_module<'py>(py: Python<'py>, source: &str, path: &str, name: &str, env: &Bound<'py, PyDict>) -> PyResult<(Option<CompiledModule>, Bound<'py, PyList>, Bound<'py, PyList>)> {
    let version: (u32, u32) = env.get_item("version")?.map(|v| v.extract()).transpose()?.unwrap_or((3, 14));
    let platform: String = env.get_item("platform")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
    let stdlib: Vec<String> = env.get_item("stdlib")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
    let pkg_version: Option<Py<PyAny>> = env.get_item("pkg_version")?.map(|v| v.unbind());
    let front_env = lizzard_py_front::Env {
        overlay_roots: paths(env, "overlay_roots")?,
        search_roots: paths(env, "search_roots")?,
        version,
        platform,
        stdlib: stdlib.into_iter().collect(),
        pkg_version: Box::new(move |pkg: &str| {
            let f = pkg_version.as_ref()?;
            Python::attach(|py| f.bind(py).call1((pkg,)).ok()?.extract::<Option<String>>().ok().flatten())
        }),
    };
    let compiled = lizzard_py_front::compile_module(front_env, name, path, source);
    let diags = PyList::new(py, &compiled.diagnostics)?;
    let warnings = PyList::new(py, &compiled.warnings)?;
    if !compiled.diagnostics.is_empty() {
        return Ok((None, diags, warnings));
    }
    if std::env::var_os("LIZZARD_DUMP_IR").is_some() {
        for f in &compiled.module.functions {
            eprintln!("== {} ({} registers: {:?})", f.name, f.reprs.len(), f.reprs);
            for (pc, i) in f.code.iter().enumerate() {
                eprintln!("{pc:4}  {i:?}");
            }
        }
    }
    let table: Table = Arc::default();
    let mut funcs = Vec::new();
    for f in compiled.module.functions {
        funcs.push(Arc::new(FuncRt::new(py, f, table.clone())?));
    }
    let _ = table.set(funcs.iter().map(|f| (f.ir.key.clone(), f.clone())).collect());
    Ok((Some(CompiledModule { funcs }), diags, warnings))
}

/// Counters of array element accesses: `fast` (direct buffer access) and
/// `generic` (through `__getitem__` / `__setitem__`).
#[pyfunction]
fn _stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("fast", array::FAST.load(Ordering::Relaxed))?;
    d.set_item("generic", array::GENERIC.load(Ordering::Relaxed))?;
    d.set_item("inline_values", ops::inline_values_on())?;
    d.set_item("fast_paths", ops::fast_paths_on())?;
    Ok(d)
}

#[pymodule(gil_used = false)]
fn _lizzard(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledModule>()?;
    m.add_class::<LizzardFunction>()?;
    m.add_class::<generator::LizzardGenerator>()?;
    generator::init_type(m.py());
    ops::init_fast_paths();
    ops::init_inline_values();
    m.add_function(wrap_pyfunction!(compile_module, m)?)?;
    m.add_function(wrap_pyfunction!(_stats, m)?)?;
    Ok(())
}
