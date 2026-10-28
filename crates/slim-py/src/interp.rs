//! The IR interpreter and slim function objects.

use std::sync::Arc;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNameError, PyTypeError, PyUnboundLocalError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString, PyTuple};
use slim_py_front::ir::{BinOp, CmpOp, Const, FuncIr, Instr, Repr};

use crate::array;
use crate::rt;

pub struct FuncRt {
    pub ir: FuncIr,
    consts: Vec<Py<PyAny>>,
    names: Vec<Py<PyString>>,
}

impl FuncRt {
    pub fn new(py: Python<'_>, ir: FuncIr) -> PyResult<FuncRt> {
        let mut consts = Vec::new();
        for c in &ir.consts {
            let o: Bound<'_, PyAny> = match c {
                Const::None => py.None().into_bound(py),
                Const::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
                Const::Int(i) => i.into_pyobject(py)?.into_any(),
                Const::BigInt(s) => py.import("builtins")?.getattr("int")?.call1((s.as_str(),))?,
                Const::Float(f) => f.into_pyobject(py)?.into_any(),
                Const::Str(s) => PyString::new(py, s).into_any(),
            };
            consts.push(o.unbind());
        }
        let names = ir.names.iter().map(|n| PyString::intern(py, n).unbind()).collect();
        Ok(FuncRt { ir, consts, names })
    }
}

/// A checked slim function, callable from Python.
#[pyclass(frozen, module = "slim._slim")]
pub struct SlimFunction {
    rt: Arc<FuncRt>,
    globals: Py<PyDict>,
    module: String,
}

impl SlimFunction {
    pub fn new(rt: Arc<FuncRt>, globals: Py<PyDict>, module: String) -> SlimFunction {
        SlimFunction { rt, globals, module }
    }
}

#[pymethods]
impl SlimFunction {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let ir = &self.rt.ir;
        let bound = bind_args(ir, args, kwargs)?;
        for (p, v) in ir.params.iter().zip(&bound) {
            let c = &ir.checks[p.check as usize];
            if let Err(got) = rt::check(py, v, &c.check)? {
                return Err(PyTypeError::new_err(format!("{}() argument '{}': expected {}, got {got}", ir.name, p.name, c.expected)));
            }
        }
        run(py, &self.rt, self.globals.bind(py), bound)
    }

    #[getter]
    fn __name__(&self) -> &str {
        &self.rt.ir.name
    }

    #[getter]
    fn __qualname__(&self) -> &str {
        &self.rt.ir.name
    }

    #[getter]
    fn __module__(&self) -> &str {
        &self.module
    }

    fn __repr__(&self) -> String {
        format!("<slim function {} at {}:{}>", self.rt.ir.name, self.rt.ir.file, self.rt.ir.line)
    }
}

/// Bind call arguments to parameters, with CPython's error messages.
fn bind_args<'py>(ir: &FuncIr, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let n = ir.params.len();
    let name = &ir.name;
    if args.len() > n {
        return Err(PyTypeError::new_err(format!("{name}() takes {n} positional arguments but {} were given", args.len())));
    }
    let mut slots: Vec<Option<Bound<'py, PyAny>>> = args.iter().map(Some).collect();
    slots.resize(n, None);
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let k: String = k.extract()?;
            match ir.params.iter().position(|p| p.name == k) {
                Some(i) if slots[i].is_some() => {
                    return Err(PyTypeError::new_err(format!("{name}() got multiple values for argument '{k}'")));
                }
                Some(i) => slots[i] = Some(v),
                None => return Err(PyTypeError::new_err(format!("{name}() got an unexpected keyword argument '{k}'"))),
            }
        }
    }
    let missing: Vec<String> = ir.params.iter().zip(&slots).filter(|(_, s)| s.is_none()).map(|(p, _)| format!("'{}'", p.name)).collect();
    if !missing.is_empty() {
        let s = if missing.len() == 1 { "" } else { "s" };
        return Err(PyTypeError::new_err(format!("{name}() missing {} required positional argument{s}: {}", missing.len(), missing.join(" and "))));
    }
    Ok(slots.into_iter().map(|s| s.unwrap()).collect())
}

enum V<'py> {
    Empty,
    O(Bound<'py, PyAny>),
    I(i64),
    F(f32),
    B(bool),
}

fn unbound() -> PyErr {
    PyUnboundLocalError::new_err("local variable referenced before assignment")
}

fn run<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, args: Vec<Bound<'py, PyAny>>) -> PyResult<Py<PyAny>> {
    let ir = &f.ir;
    let mut regs: Vec<V<'py>> = (0..ir.reprs.len()).map(|_| V::Empty).collect();
    for (i, a) in args.into_iter().enumerate() {
        regs[i] = V::O(a);
    }
    let mut pc = 0usize;
    loop {
        match step(py, f, globals, &mut regs, &mut pc) {
            Ok(Some(ret)) => return Ok(ret),
            Ok(None) => {}
            Err(e) => {
                let line = ir.lines.get(pc).copied().unwrap_or(ir.line);
                let _ = e.add_note(py, format!("slim: {}:{line} in {}", ir.file, ir.name));
                return Err(e);
            }
        }
    }
}

macro_rules! obj {
    ($regs:expr, $r:expr) => {
        match &$regs[$r as usize] {
            V::O(o) => o,
            V::Empty => return Err(unbound()),
            _ => unreachable!("register {} is not an object", $r),
        }
    };
}

macro_rules! scalar {
    ($regs:expr, $r:expr, $variant:ident) => {
        match &$regs[$r as usize] {
            V::$variant(x) => *x,
            V::Empty => return Err(unbound()),
            _ => unreachable!("register {} has the wrong representation", $r),
        }
    };
}

/// Execute one instruction; `Some` when the function returns.
fn step<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, regs: &mut [V<'py>], pc: &mut usize) -> PyResult<Option<Py<PyAny>>> {
    let ir = &f.ir;
    let instr = &ir.code[*pc];
    let mut next = *pc + 1;
    match instr {
        Instr::Const { dst, idx } => regs[*dst as usize] = V::O(f.consts[*idx as usize].bind(py).clone()),
        Instr::LoadGlobal { dst, name } => {
            let n = f.names[*name as usize].bind(py);
            let v = match globals.get_item(n)? {
                Some(v) => v,
                None => py.import("builtins")?.getattr(n).map_err(|_| PyNameError::new_err(format!("name '{n}' is not defined")))?,
            };
            regs[*dst as usize] = V::O(v);
        }
        Instr::Move { dst, src } => {
            let v = match &regs[*src as usize] {
                V::Empty => return Err(unbound()),
                V::O(o) => V::O(o.clone()),
                V::I(i) => V::I(*i),
                V::F(x) => V::F(*x),
                V::B(b) => V::B(*b),
            };
            regs[*dst as usize] = v;
        }
        Instr::LoadAttr { dst, obj, name } => {
            let v = obj!(regs, *obj).getattr(f.names[*name as usize].bind(py))?;
            regs[*dst as usize] = V::O(v);
        }
        Instr::Call { dst, func, args, kwnames } => {
            let npos = args.len() - kwnames.len();
            let mut pos = Vec::with_capacity(npos);
            for r in &args[..npos] {
                pos.push(obj!(regs, *r).clone());
            }
            let kwargs = if kwnames.is_empty() {
                None
            } else {
                let d = PyDict::new(py);
                for (r, n) in args[npos..].iter().zip(kwnames) {
                    d.set_item(f.names[*n as usize].bind(py), obj!(regs, *r))?;
                }
                Some(d)
            };
            let v = obj!(regs, *func).call(PyTuple::new(py, pos)?, kwargs.as_ref())?;
            regs[*dst as usize] = V::O(v);
        }
        Instr::BinOp { dst, op, a, b } => {
            let (x, y) = (obj!(regs, *a), obj!(regs, *b));
            let v = match op {
                BinOp::Add => x.add(y)?,
                BinOp::Sub => x.sub(y)?,
                BinOp::Mul => x.mul(y)?,
                BinOp::Div => x.div(y)?,
            };
            regs[*dst as usize] = V::O(v);
        }
        Instr::Compare { dst, op, a, b } => {
            let (x, y) = (obj!(regs, *a), obj!(regs, *b));
            let v = match op {
                CmpOp::Is => PyBool::new(py, x.is(y)).to_owned().into_any(),
                CmpOp::IsNot => PyBool::new(py, !x.is(y)).to_owned().into_any(),
                op => {
                    let o = match op {
                        CmpOp::Eq => CompareOp::Eq,
                        CmpOp::Ne => CompareOp::Ne,
                        CmpOp::Lt => CompareOp::Lt,
                        CmpOp::Le => CompareOp::Le,
                        CmpOp::Gt => CompareOp::Gt,
                        _ => CompareOp::Ge,
                    };
                    x.rich_compare(y, o)?
                }
            };
            regs[*dst as usize] = V::O(v);
        }
        Instr::GetItem { dst, obj, key } => {
            let v = obj!(regs, *obj).get_item(obj!(regs, *key))?;
            regs[*dst as usize] = V::O(v);
        }
        Instr::SetItem { obj, key, val } => obj!(regs, *obj).set_item(obj!(regs, *key), obj!(regs, *val))?,
        Instr::Truthy { dst, src } => {
            let b = obj!(regs, *src).is_truthy()?;
            regs[*dst as usize] = V::B(b);
        }
        Instr::Not { dst, src } => regs[*dst as usize] = V::B(!scalar!(regs, *src, B)),
        Instr::Jump { target } => next = *target as usize,
        Instr::JumpIfFalse { cond, target } => {
            if !scalar!(regs, *cond, B) {
                next = *target as usize;
            }
        }
        Instr::JumpIfTrue { cond, target } => {
            if scalar!(regs, *cond, B) {
                next = *target as usize;
            }
        }
        Instr::Raise { exc } => {
            let e = obj!(regs, *exc).clone();
            return Err(PyErr::from_value(e));
        }
        Instr::Return { src } => return Ok(Some(obj!(regs, *src).clone().unbind())),
        Instr::I64Const { dst, value } => regs[*dst as usize] = V::I(*value),
        Instr::I64Lt { dst, a, b } => regs[*dst as usize] = V::B(scalar!(regs, *a, I) < scalar!(regs, *b, I)),
        Instr::I64Add { dst, a, b } => regs[*dst as usize] = V::I(scalar!(regs, *a, I) + scalar!(regs, *b, I)),
        Instr::Len { dst, obj } => {
            let n = obj!(regs, *obj).len()?;
            regs[*dst as usize] = V::I(n as i64);
        }
        Instr::Box { dst, src } => {
            let v = match &regs[*src as usize] {
                V::Empty => return Err(unbound()),
                V::O(o) => o.clone(),
                V::I(i) => i.into_pyobject(py)?.into_any(),
                V::F(x) => array::box_f32(py, *x)?,
                V::B(b) => PyBool::new(py, *b).to_owned().into_any(),
            };
            regs[*dst as usize] = V::O(v);
        }
        Instr::Unbox { dst, src } => {
            let o = obj!(regs, *src);
            regs[*dst as usize] = match ir.reprs[*dst as usize] {
                Repr::I64 => V::I(o.extract::<i64>()?),
                Repr::F32 => V::F(o.extract::<f64>()? as f32),
                Repr::Bool => V::B(o.is_truthy()?),
                Repr::Obj => V::O(o.clone()),
            };
        }
        Instr::ArrLoadF32 { dst, arr, idx } => {
            let v = array::load_f32(py, obj!(regs, *arr), scalar!(regs, *idx, I))?;
            regs[*dst as usize] = V::F(v);
        }
        Instr::ArrStoreF32 { arr, idx, val } => array::store_f32(py, obj!(regs, *arr), scalar!(regs, *idx, I), scalar!(regs, *val, F))?,
        Instr::F32Bin { dst, op, a, b } => {
            let (x, y) = (scalar!(regs, *a, F), scalar!(regs, *b, F));
            regs[*dst as usize] = V::F(match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
            });
        }
        Instr::Check { reg, check } => {
            let c = &ir.checks[*check as usize];
            if let Err(got) = rt::check(py, obj!(regs, *reg), &c.check)? {
                let line = ir.lines[*pc];
                let origin = c.origin.as_ref().map(|o| format!(" (declared at {o})")).unwrap_or_default();
                return Err(PyTypeError::new_err(format!("{}:{line}: {} should be {}{origin}, got {got}", ir.file, c.what, c.expected)));
            }
        }
    }
    *pc = next;
    Ok(None)
}
