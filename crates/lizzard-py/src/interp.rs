//! The IR interpreter and Lizzard function objects.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNameError, PyTypeError, PyUnboundLocalError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::exceptions::PyBaseException;
use pyo3::types::{PyBool, PyDict, PyIterator, PyList, PySet, PyString, PyTuple, PyType};
use lizzard_py_front::ir::{BinOp, CmpOp, Const, FuncIr, Instr, ParamKind, Repr};

use crate::array;
use crate::generator::{self, LizzardGenerator, Sent};
use crate::rt;

/// A module's functions by key, for creating inner functions and lambdas.
pub type Table = Arc<OnceLock<HashMap<String, Arc<FuncRt>>>>;

pub struct FuncRt {
    pub ir: FuncIr,
    pub(crate) consts: Vec<Py<PyAny>>,
    names: Vec<Py<PyString>>,
    table: Table,
}

impl FuncRt {
    pub fn new(py: Python<'_>, ir: FuncIr, table: Table) -> PyResult<FuncRt> {
        let mut consts = Vec::new();
        for c in &ir.consts {
            let o: Bound<'_, PyAny> = match c {
                Const::None => py.None().into_bound(py),
                Const::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
                Const::Int(i) => i.into_pyobject(py)?.into_any(),
                Const::BigInt(s) => builtins(py)?.getattr("int")?.call1((s.as_str(),))?,
                Const::Float(f) => f.into_pyobject(py)?.into_any(),
                Const::Str(s) => PyString::new(py, s).into_any(),
            };
            consts.push(o.unbind());
        }
        let names = ir.names.iter().map(|n| PyString::intern(py, n).unbind()).collect();
        Ok(FuncRt { ir, consts, names, table })
    }
}

/// A checked Lizzard function, callable from Python.
#[pyclass(frozen, module = "lizzard._lizzard")]
pub struct LizzardFunction {
    rt: Arc<FuncRt>,
    globals: Py<PyDict>,
    module: String,
    /// Cells for the variables it captures (`ir.free`).
    closure: Vec<Py<PyAny>>,
    /// Defaults evaluated where it was created (parameters with
    /// `dynamic_default`, in order).
    defaults: Vec<Py<PyAny>>,
}

impl LizzardFunction {
    pub fn new(rt: Arc<FuncRt>, globals: Py<PyDict>, module: String, closure: Vec<Py<PyAny>>, defaults: Vec<Py<PyAny>>) -> LizzardFunction {
        LizzardFunction { rt, globals, module, closure, defaults }
    }
}

#[pymethods]
impl LizzardFunction {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let ir = &self.rt.ir;
        let bound = bind_args(ir, &self.rt.consts, &self.defaults, args, kwargs)?;
        check_args(py, ir, &bound)?;
        if ir.generator {
            // A generator function returns a generator over a new frame.
            let g = LizzardGenerator::new(self.rt.clone(), self.globals.clone_ref(py), self.closure.iter().map(|c| c.clone_ref(py)).collect(), bound);
            return Ok(Py::new(py, g)?.into_any());
        }
        run(py, &self.rt, self.globals.bind(py), bound, &self.closure)
    }

    /// Bind as a method when stored in a class, like a Python function.
    fn __get__<'py>(slf: Bound<'py, Self>, instance: Option<Bound<'py, PyAny>>, _owner: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        match instance {
            Some(obj) if !obj.is_none() => {
                static METHOD_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
                let mt = METHOD_TYPE.get_or_try_init(py, || py.import("types")?.getattr("MethodType").map(|m| m.unbind()))?;
                mt.bind(py).call1((slf, obj))
            }
            _ => Ok(slf.into_any()),
        }
    }

    #[getter]
    fn __name__(&self) -> &str {
        self.rt.ir.name.rsplit('.').next().unwrap_or(&self.rt.ir.name)
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
        format!("<Lizzard function {} at {}:{}>", self.rt.ir.name, self.rt.ir.file, self.rt.ir.line)
    }
}

/// Bind call arguments to parameters, with CPython's error messages.
fn bind_args<'py>(ir: &FuncIr, consts: &[Py<PyAny>], dynamic: &[Py<PyAny>], args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = args.py();
    let name = &ir.name;
    let err = |m: String| Err(PyTypeError::new_err(format!("{name}() {m}")));
    let positional: Vec<usize> = (0..ir.params.len()).filter(|&i| matches!(ir.params[i].kind, ParamKind::PosOnly | ParamKind::PosOrKw)).collect();
    let var_pos = ir.params.iter().position(|p| p.kind == ParamKind::VarPos);
    let var_kw = ir.params.iter().position(|p| p.kind == ParamKind::VarKw);
    let mut slots: Vec<Option<Bound<'py, PyAny>>> = vec![None; ir.params.len()];
    let given: Vec<Bound<'py, PyAny>> = args.iter().collect();
    for (i, a) in given.iter().enumerate().take(positional.len()) {
        slots[positional[i]] = Some(a.clone());
    }
    let extra = &given[given.len().min(positional.len())..];
    if let Some(vp) = var_pos {
        slots[vp] = Some(PyTuple::new(py, extra)?.into_any());
    }
    let kw_dict = match var_kw {
        Some(vk) => {
            let d = PyDict::new(py);
            slots[vk] = Some(d.clone().into_any());
            Some(d)
        }
        None => None,
    };
    let mut posonly_as_kw = Vec::new();
    let mut kwonly_given = 0;
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            match ir.params.iter().position(|p| p.name == key && matches!(p.kind, ParamKind::PosOrKw | ParamKind::KwOnly)) {
                Some(i) if slots[i].is_some() => return err(format!("got multiple values for argument '{key}'")),
                Some(i) => {
                    if ir.params[i].kind == ParamKind::KwOnly {
                        kwonly_given += 1;
                    }
                    slots[i] = Some(v);
                }
                None => match &kw_dict {
                    Some(d) => d.set_item(k, v)?,
                    None if ir.params.iter().any(|p| p.name == key && p.kind == ParamKind::PosOnly) => posonly_as_kw.push(format!("'{key}'")),
                    None => return err(format!("got an unexpected keyword argument '{key}'")),
                },
            }
        }
    }
    if !extra.is_empty() && var_pos.is_none() {
        let max = positional.len();
        let min = positional.iter().filter(|&&i| ir.params[i].default.is_none() && !ir.params[i].dynamic_default).count();
        let takes = if min == max { format!("{max} positional argument{}", plural(max)) } else { format!("from {min} to {max} positional arguments") };
        let n = given.len();
        let were = if kwonly_given > 0 {
            format!("{n} positional argument{} (and {kwonly_given} keyword-only argument{}) were given", plural(n), plural(kwonly_given))
        } else {
            format!("{n} {} given", if n == 1 { "was" } else { "were" })
        };
        return err(format!("takes {takes} but {were}"));
    }
    if !posonly_as_kw.is_empty() {
        return err(format!("got some positional-only arguments passed as keyword arguments: {}", posonly_as_kw.join(", ")));
    }
    let mut next_dynamic = 0;
    for (i, p) in ir.params.iter().enumerate() {
        if p.dynamic_default {
            if slots[i].is_none() {
                slots[i] = dynamic.get(next_dynamic).map(|d| d.bind(py).clone());
            }
            next_dynamic += 1;
        } else if slots[i].is_none() {
            if let Some(d) = p.default {
                slots[i] = Some(consts[d as usize].bind(py).clone());
            }
        }
    }
    for (kind, what) in [(ParamKind::PosOnly, "positional"), (ParamKind::KwOnly, "keyword-only")] {
        let missing: Vec<String> = ir
            .params
            .iter()
            .zip(&slots)
            .filter(|(p, s)| s.is_none() && (p.kind == kind || (kind == ParamKind::PosOnly && p.kind == ParamKind::PosOrKw)))
            .map(|(p, _)| format!("'{}'", p.name))
            .collect();
        if !missing.is_empty() {
            return err(format!("missing {} required {what} argument{}: {}", missing.len(), plural(missing.len()), english_list(&missing)));
        }
    }
    Ok(slots.into_iter().map(|s| s.unwrap()).collect())
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// `'a'`, `'a' and 'b'`, `'a', 'b', and 'c'`, as CPython lists names.
fn english_list(items: &[String]) -> String {
    match items {
        [a] => a.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
        [] => String::new(),
    }
}

/// Check bound arguments against the parameters' types (each item of
/// `*args`, each value of `**kwargs`).
fn check_args(py: Python<'_>, ir: &FuncIr, bound: &[Bound<'_, PyAny>]) -> PyResult<()> {
    for (p, v) in ir.params.iter().zip(bound) {
        let c = &ir.checks[p.check as usize];
        let fail = |what: String, got: String| Err(PyTypeError::new_err(format!("{}() argument {what}: expected {}, got {got}", ir.name, c.expected)));
        match p.kind {
            ParamKind::VarPos => {
                for (i, item) in v.try_iter()?.enumerate() {
                    if let Err(got) = rt::check(py, &item?, &c.check)? {
                        return fail(format!("'{}[{i}]'", p.name), got);
                    }
                }
            }
            ParamKind::VarKw => {
                for (k, item) in v.cast::<PyDict>().map_err(PyErr::from)?.iter() {
                    if let Err(got) = rt::check(py, &item, &c.check)? {
                        return fail(format!("'{}[{}]'", p.name, k.repr()?), got);
                    }
                }
            }
            _ => {
                if let Err(got) = rt::check(py, v, &c.check)? {
                    return fail(format!("'{}'", p.name), got);
                }
            }
        }
    }
    Ok(())
}

/// A register.
pub(crate) enum V<'py> {
    Empty,
    O(Bound<'py, PyAny>),
    I(i64),
    F(f32),
    B(bool),
}

/// How execution leaves a function's frame. The frame (registers and pc) is
/// left as it was, so a generator can resume it: these exits and the entry
/// of `exec` (a pc, with an exception raised there or not) are the
/// transitions a compiled function needs too.
pub(crate) enum Exit<'py> {
    Return(Bound<'py, PyAny>),
    /// `yield value` at the current pc; the sent value goes to `dst`.
    Yield { value: Bound<'py, PyAny>, dst: u32 },
    /// `yield from iter` at the current pc, which produced `value`; `iter`
    /// takes over until it finishes and its return value goes to `dst`.
    YieldFrom { value: Bound<'py, PyAny>, dst: u32, iter: Bound<'py, PyAny> },
}

fn unbound() -> PyErr {
    PyUnboundLocalError::new_err("local variable referenced before assignment")
}

fn run<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, args: Vec<Bound<'py, PyAny>>, closure: &[Py<PyAny>]) -> PyResult<Py<PyAny>> {
    let mut regs: Vec<V<'py>> = (0..f.ir.reprs.len()).map(|_| V::Empty).collect();
    for (i, a) in args.into_iter().enumerate() {
        regs[i] = V::O(a);
    }
    let mut pc = 0usize;
    match exec(py, f, globals, closure, &mut regs, &mut pc, None)? {
        Exit::Return(v) => Ok(v.unbind()),
        Exit::Yield { .. } | Exit::YieldFrom { .. } => unreachable!("yield in a function that is not a generator"),
    }
}

/// Run a frame from `pc` until it returns or yields. With `pending`, the
/// instruction at `pc` raises it (a generator's `throw`).
pub(crate) fn exec<'py>(
    py: Python<'py>,
    f: &FuncRt,
    globals: &Bound<'py, PyDict>,
    closure: &[Py<PyAny>],
    regs: &mut [V<'py>],
    pc: &mut usize,
    mut pending: Option<PyErr>,
) -> PyResult<Exit<'py>> {
    let ir = &f.ir;
    loop {
        let r = match pending.take() {
            Some(e) => Err(e),
            None => step(py, f, globals, closure, regs, pc),
        };
        match r {
            Ok(Some(exit)) => return Ok(exit),
            Ok(None) => {}
            Err(e) => {
                let pc = &mut *pc;
                // The innermost handler covering the instruction: regions
                // nest, and an inner handler is created after its outer one.
                let at = *pc as u32;
                let handler = ir.handlers.iter().rev().find(|h| h.ranges.iter().any(|(s, e)| *s <= at && at < *e));
                if let Some(h) = handler {
                    regs[h.exc as usize] = V::O(e.into_value(py).into_bound(py).into_any());
                    *pc = h.target as usize;
                    continue;
                }
                let line = ir.lines.get(*pc).copied().unwrap_or(ir.line);
                let _ = e.add_note(py, format!("lizzard: {}:{line} in {}", ir.file, ir.name));
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

/// The `builtins` module, found once (also usable while the interpreter
/// shuts down and imports no longer work).
fn builtins(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static BUILTINS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Ok(BUILTINS.get_or_try_init(py, || py.import("builtins").map(|m| m.into_any().unbind()))?.bind(py))
}

fn cell_get<'py>(py: Python<'py>, cell: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    // SAFETY: `cell` is a cell object; returns a new reference or NULL
    // (empty cell, no exception set).
    let p = unsafe { ffi::PyCell_Get(cell.as_ptr()) };
    if p.is_null() {
        return match PyErr::take(py) {
            Some(e) => Err(e),
            None => Ok(None),
        };
    }
    Ok(Some(unsafe { Bound::from_owned_ptr(py, p) }))
}

fn cell_set(cell: &Bound<'_, PyAny>, val: &Bound<'_, PyAny>) -> PyResult<()> {
    // SAFETY: `cell` is a cell object; the value is not stolen.
    if unsafe { ffi::PyCell_Set(cell.as_ptr(), val.as_ptr()) } < 0 {
        return Err(PyErr::fetch(cell.py()));
    }
    Ok(())
}

/// The exception `raise x` raises: `x`, or an instance of the class `x`.
fn exception_instance<'py>(x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let e = if x.is_instance_of::<PyType>() { x.call0()? } else { x.clone() };
    if !e.is_instance_of::<PyBaseException>() {
        return Err(PyTypeError::new_err("exceptions must derive from BaseException"));
    }
    Ok(e)
}

/// Execute one instruction; `Some` when the function returns.
fn step<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, closure: &[Py<PyAny>], regs: &mut [V<'py>], pc: &mut usize) -> PyResult<Option<Exit<'py>>> {
    let ir = &f.ir;
    let instr = &ir.code[*pc];
    let mut next = *pc + 1;
    match instr {
        Instr::Const { dst, idx } => regs[*dst as usize] = V::O(f.consts[*idx as usize].bind(py).clone()),
        Instr::LoadGlobal { dst, name } => {
            let n = f.names[*name as usize].bind(py);
            let v = match globals.get_item(n)? {
                Some(v) => v,
                None => builtins(py)?.getattr(n).map_err(|_| PyNameError::new_err(format!("name '{n}' is not defined")))?,
            };
            regs[*dst as usize] = V::O(v);
        }
        Instr::StoreGlobal { name, val } => globals.set_item(f.names[*name as usize].bind(py), obj!(regs, *val))?,
        Instr::MakeCell { dst } => {
            // SAFETY: NULL makes an empty cell; returns a new reference.
            let c = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyCell_New(std::ptr::null_mut()))? };
            regs[*dst as usize] = V::O(c);
        }
        Instr::LoadCell { dst, cell } => match cell_get(py, obj!(regs, *cell))? {
            Some(v) => regs[*dst as usize] = V::O(v),
            None => return Err(unbound()),
        },
        Instr::StoreCell { cell, val } => cell_set(obj!(regs, *cell), obj!(regs, *val))?,
        Instr::LoadFree { dst, idx } => match cell_get(py, closure[*idx as usize].bind(py))? {
            Some(v) => regs[*dst as usize] = V::O(v),
            None => {
                let n = &f.ir.free[*idx as usize];
                return Err(PyNameError::new_err(format!("cannot access free variable '{n}' where it is not associated with a value in enclosing scope")));
            }
        },
        Instr::StoreFree { idx, val } => cell_set(closure[*idx as usize].bind(py), obj!(regs, *val))?,
        Instr::LoadFreeCell { dst, idx } => regs[*dst as usize] = V::O(closure[*idx as usize].bind(py).clone()),
        Instr::MakeFunction { dst, key, closure: cells, defaults } => {
            let key = f.names[*key as usize].bind(py).to_cow()?;
            let rt = f.table.get().and_then(|t| t.get(key.as_ref())).cloned().expect("inner function compiled with its module");
            let module = match globals.get_item("__name__")? {
                Some(n) => n.extract()?,
                None => String::new(),
            };
            let mut cl = Vec::with_capacity(cells.len());
            for r in cells {
                cl.push(obj!(regs, *r).clone().unbind());
            }
            let mut dv = Vec::with_capacity(defaults.len());
            for r in defaults {
                dv.push(obj!(regs, *r).clone().unbind());
            }
            let func = Py::new(py, LizzardFunction::new(rt, globals.clone().unbind(), module, cl, dv))?;
            regs[*dst as usize] = V::O(func.into_bound(py).into_any());
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
        Instr::CallEx { dst, func, args, kwargs } => {
            let a = obj!(regs, *args).cast::<PyTuple>().map_err(PyErr::from)?.clone();
            let kw = match kwargs {
                Some(k) => Some(obj!(regs, *k).cast::<PyDict>().map_err(PyErr::from)?.clone()),
                None => None,
            };
            let v = obj!(regs, *func).call(a, kw.as_ref())?;
            regs[*dst as usize] = V::O(v);
        }
        Instr::ListExtend { list, iter } => {
            let l = obj!(regs, *list).cast::<PyList>().map_err(PyErr::from)?.clone();
            for item in obj!(regs, *iter).try_iter()? {
                l.append(item?)?;
            }
        }
        Instr::ListToTuple { dst, src } => {
            let t = obj!(regs, *src).cast::<PyList>().map_err(PyErr::from)?.to_tuple();
            regs[*dst as usize] = V::O(t.into_any());
        }
        Instr::DictMerge { dict, src, func } => {
            let d = obj!(regs, *dict).cast::<PyDict>().map_err(PyErr::from)?.clone();
            let src = obj!(regs, *src);
            let keys = src.call_method0("keys").map_err(|_| {
                PyTypeError::new_err(format!("argument after ** must be a mapping, not {}", src.get_type().name().map(|n| n.to_string()).unwrap_or_default()))
            })?;
            for k in keys.try_iter()? {
                let k = k?;
                if d.contains(&k)? {
                    let fname = obj!(regs, *func).getattr("__qualname__").and_then(|n| n.extract::<String>()).unwrap_or_else(|_| "function".into());
                    return Err(PyTypeError::new_err(format!("{fname}() got multiple values for keyword argument '{}'", k.str()?)));
                }
                d.set_item(&k, src.get_item(&k)?)?;
            }
        }
        Instr::BinOp { dst, op, a, b, inplace } => {
            let (x, y) = (obj!(regs, *a), obj!(regs, *b));
            let v = if *inplace && *op == BinOp::Pow {
                // SAFETY: valid objects in, a new reference or NULL out.
                unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyNumber_InPlacePower(x.as_ptr(), y.as_ptr(), ffi::Py_None()))? }
            } else if *inplace {
                let f = match op {
                    BinOp::Add => ffi::PyNumber_InPlaceAdd,
                    BinOp::Sub => ffi::PyNumber_InPlaceSubtract,
                    BinOp::Mul => ffi::PyNumber_InPlaceMultiply,
                    BinOp::Div => ffi::PyNumber_InPlaceTrueDivide,
                    BinOp::FloorDiv => ffi::PyNumber_InPlaceFloorDivide,
                    BinOp::Mod => ffi::PyNumber_InPlaceRemainder,
                    BinOp::BitOr => ffi::PyNumber_InPlaceOr,
                    BinOp::BitAnd => ffi::PyNumber_InPlaceAnd,
                    BinOp::BitXor => ffi::PyNumber_InPlaceXor,
                    BinOp::LShift => ffi::PyNumber_InPlaceLshift,
                    BinOp::RShift => ffi::PyNumber_InPlaceRshift,
                    BinOp::Pow => unreachable!(),
                };
                // SAFETY: both arguments are valid objects; the result is a
                // new reference or NULL with an exception set.
                unsafe { Bound::from_owned_ptr_or_err(py, f(x.as_ptr(), y.as_ptr()))? }
            } else {
                match op {
                    BinOp::Add => x.add(y)?,
                    BinOp::Sub => x.sub(y)?,
                    BinOp::Mul => x.mul(y)?,
                    BinOp::Div => x.div(y)?,
                    BinOp::FloorDiv => x.floor_div(y)?,
                    BinOp::Mod => x.rem(y)?,
                    BinOp::Pow => x.pow(y, py.None())?,
                    BinOp::BitOr => x.bitor(y)?,
                    BinOp::BitAnd => x.bitand(y)?,
                    BinOp::BitXor => x.bitxor(y)?,
                    BinOp::LShift => x.lshift(y)?,
                    BinOp::RShift => x.rshift(y)?,
                }
            };
            regs[*dst as usize] = V::O(v);
        }
        Instr::Neg { dst, src } => {
            let v = obj!(regs, *src).neg()?;
            regs[*dst as usize] = V::O(v);
        }
        Instr::Invert { dst, src } => {
            let v = obj!(regs, *src).bitnot()?;
            regs[*dst as usize] = V::O(v);
        }
        Instr::Pos { dst, src } => {
            let v = obj!(regs, *src).pos()?;
            regs[*dst as usize] = V::O(v);
        }
        Instr::BuildSet { dst, items } => {
            let set = PySet::empty(py)?;
            for r in items {
                set.add(obj!(regs, *r))?;
            }
            regs[*dst as usize] = V::O(set.into_any());
        }
        Instr::SetAdd { set, val } => {
            obj!(regs, *set).cast::<PySet>().map_err(PyErr::from)?.add(obj!(regs, *val))?;
        }
        Instr::StoreAttr { obj, name, val } => obj!(regs, *obj).setattr(f.names[*name as usize].bind(py), obj!(regs, *val))?,
        Instr::DelItem { obj, key } => obj!(regs, *obj).del_item(obj!(regs, *key))?,
        Instr::BuildList { dst, items } => {
            let mut v = Vec::with_capacity(items.len());
            for r in items {
                v.push(obj!(regs, *r).clone());
            }
            regs[*dst as usize] = V::O(PyList::new(py, v)?.into_any());
        }
        Instr::BuildTuple { dst, items } => {
            let mut v = Vec::with_capacity(items.len());
            for r in items {
                v.push(obj!(regs, *r).clone());
            }
            regs[*dst as usize] = V::O(PyTuple::new(py, v)?.into_any());
        }
        Instr::BuildDict { dst, keys, vals } => {
            let d = PyDict::new(py);
            for (k, v) in keys.iter().zip(vals) {
                d.set_item(obj!(regs, *k), obj!(regs, *v))?;
            }
            regs[*dst as usize] = V::O(d.into_any());
        }
        Instr::BuildSlice { dst, lower, upper, step } => {
            let (a, b, c) = (obj!(regs, *lower), obj!(regs, *upper), obj!(regs, *step));
            // SAFETY: valid objects in, a new reference or NULL out.
            let v = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PySlice_New(a.as_ptr(), b.as_ptr(), c.as_ptr()))? };
            regs[*dst as usize] = V::O(v);
        }
        Instr::ListAppend { list, val } => {
            let l = obj!(regs, *list).cast::<PyList>().map_err(PyErr::from)?;
            l.append(obj!(regs, *val))?;
        }
        Instr::GetIter { dst, obj } => {
            let it = obj!(regs, *obj).try_iter()?;
            regs[*dst as usize] = V::O(it.into_any());
        }
        Instr::ForIter { dst, iter, target } => {
            let it = obj!(regs, *iter);
            // A Lizzard generator is resumed directly, without an exception
            // at the end.
            if let Ok(g) = it.cast::<LizzardGenerator>() {
                match g.get().resume(py, generator::Op::Send(py.None().into_bound(py)))? {
                    Sent::Next(v) => regs[*dst as usize] = V::O(v),
                    Sent::Return(_) => next = *target as usize,
                }
            } else {
                let mut it = it.cast::<PyIterator>().map_err(PyErr::from)?.clone();
                match it.next() {
                    Some(v) => regs[*dst as usize] = V::O(v?),
                    None => next = *target as usize,
                }
            }
        }
        Instr::Unpack { src, dsts } => {
            let o = obj!(regs, *src).clone();
            let n = dsts.len();
            let known = if o.is_exact_instance_of::<PyTuple>() || o.is_exact_instance_of::<PyList>() || o.is_exact_instance_of::<PyDict>() { Some(o.len()?) } else { None };
            if let Some(k) = known {
                if k < n {
                    return Err(PyValueError::new_err(format!("not enough values to unpack (expected {n}, got {k})")));
                }
                if k > n {
                    return Err(PyValueError::new_err(format!("too many values to unpack (expected {n}, got {k})")));
                }
            }
            let mut it = o.try_iter()?;
            for (i, d) in dsts.iter().enumerate() {
                match it.next() {
                    Some(v) => regs[*d as usize] = V::O(v?),
                    None => return Err(PyValueError::new_err(format!("not enough values to unpack (expected {n}, got {i})"))),
                }
            }
            if let Some(extra) = it.next() {
                extra?;
                return Err(PyValueError::new_err(format!("too many values to unpack (expected {n})")));
            }
        }
        Instr::Format { dst, val, conv, spec } => {
            let v = obj!(regs, *val);
            let v = match conv {
                b's' => v.str()?.into_any(),
                b'r' => v.repr()?.into_any(),
                // SAFETY: a valid object in, a new reference or NULL out.
                b'a' => unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyObject_ASCII(v.as_ptr()))? },
                _ => v.clone(),
            };
            let spec = match spec {
                Some(r) => obj!(regs, *r).clone(),
                None => PyString::new(py, "").into_any(),
            };
            // SAFETY: as above.
            let s = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyObject_Format(v.as_ptr(), spec.as_ptr()))? };
            regs[*dst as usize] = V::O(s);
        }
        Instr::BuildString { dst, parts } => {
            let mut s = String::new();
            for r in parts {
                s.push_str(&obj!(regs, *r).cast::<PyString>().map_err(PyErr::from)?.to_cow()?);
            }
            regs[*dst as usize] = V::O(PyString::new(py, &s).into_any());
        }
        Instr::Compare { dst, op, a, b } => {
            let (x, y) = (obj!(regs, *a), obj!(regs, *b));
            let v = match op {
                CmpOp::Is => PyBool::new(py, x.is(y)).to_owned().into_any(),
                CmpOp::IsNot => PyBool::new(py, !x.is(y)).to_owned().into_any(),
                CmpOp::In => PyBool::new(py, y.contains(x)?).to_owned().into_any(),
                CmpOp::NotIn => PyBool::new(py, !y.contains(x)?).to_owned().into_any(),
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
        Instr::Raise { exc, cause } => {
            let e = exception_instance(obj!(regs, *exc))?;
            if let Some(c) = cause {
                let c = obj!(regs, *c);
                let c = if c.is_none() { c.clone() } else { exception_instance(c)? };
                // Also sets `__suppress_context__`.
                e.setattr("__cause__", c)?;
            }
            // SAFETY: valid objects; `PyErr_SetObject` chains the exception
            // being handled as `__context__`, as `raise` does.
            unsafe { ffi::PyErr_SetObject(e.get_type().as_ptr(), e.as_ptr()) };
            return Err(PyErr::fetch(py));
        }
        Instr::Reraise { exc } => return Err(PyErr::from_value(obj!(regs, *exc).clone())),
        Instr::ExcMatch { dst, exc, cls } => {
            let m = obj!(regs, *exc).is_instance(obj!(regs, *cls))?;
            regs[*dst as usize] = V::B(m);
        }
        Instr::GetHandled { dst } => {
            // SAFETY: returns a new reference or NULL.
            let p = unsafe { ffi::PyErr_GetHandledException() };
            let v = if p.is_null() { py.None().into_bound(py) } else { unsafe { Bound::from_owned_ptr(py, p) } };
            regs[*dst as usize] = V::O(v);
        }
        Instr::SetHandled { src } => {
            // SAFETY: a valid object (None clears it); the reference is not stolen.
            unsafe { ffi::PyErr_SetHandledException(obj!(regs, *src).as_ptr()) };
        }
        Instr::Unbind { reg } => regs[*reg as usize] = V::Empty,
        Instr::TypeOf { dst, src } => {
            let t = obj!(regs, *src).get_type().into_any();
            regs[*dst as usize] = V::O(t);
        }
        Instr::LoadSpecial { dst, obj, name } => {
            let o = obj!(regs, *obj);
            let t = o.get_type();
            let n = f.names[*name as usize].bind(py);
            let a = t.getattr(n).map_err(|_| PyTypeError::new_err(format!("'{}' object does not support the context manager protocol", t.name().map(|s| s.to_string()).unwrap_or_default())))?;
            // Bind it like a method lookup does.
            let v = match a.get_type().getattr("__get__") {
                Ok(get) => get.call1((&a, o, &t))?,
                Err(_) => a,
            };
            regs[*dst as usize] = V::O(v);
        }
        Instr::Return { src } => return Ok(Some(Exit::Return(obj!(regs, *src).clone()))),
        // Both leave `pc` at the instruction: the generator resumes after it,
        // or raises there.
        Instr::Yield { dst, src } => return Ok(Some(Exit::Yield { value: obj!(regs, *src).clone(), dst: *dst })),
        Instr::YieldFrom { dst, iter } => {
            let it = obj!(regs, *iter).clone();
            match generator::send_to(&it, &py.None().into_bound(py))? {
                Sent::Next(value) => return Ok(Some(Exit::YieldFrom { value, dst: *dst, iter: it })),
                Sent::Return(v) => regs[*dst as usize] = V::O(v),
            }
        }
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
                _ => unreachable!("float32 arithmetic is + - * / only"),
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
