//! The IR interpreter and Lizzard function objects.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNameError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::exceptions::PyBaseException;
use pyo3::types::{PyBool, PyDict, PyIterator, PyList, PySet, PyString, PyTuple, PyType};
use lizzard_py_front::ir::{BinOp, CmpOp, Const, FuncIr, Instr, ParamKind, Repr};

use crate::array;
use crate::frame::{unbound, Regs};
use crate::globals;
use crate::jit;
use crate::ops;
use crate::generator::{self, LizzardGenerator, Sent};
use crate::rt;

/// A module's functions by key, for creating inner functions and lambdas.
pub type Table = Arc<OnceLock<HashMap<String, Arc<FuncRt>>>>;

pub struct FuncRt {
    pub ir: FuncIr,
    /// Which registers hold objects.
    pub objs: Arc<[bool]>,
    /// All its parameters are plain positional ones.
    pub positional: bool,
    /// Its compiled code, once it has run (None if it cannot be compiled).
    pub jit: OnceLock<Option<lizzard_jit::Entry>>,
    pub(crate) consts: Vec<Py<PyAny>>,
    pub(crate) names: Vec<Py<PyString>>,
    /// The keyword names of each call with keywords, by pc, as the tuple
    /// vectorcall takes.
    kwnames: HashMap<usize, Py<PyTuple>>,
    /// `ir.checks`, prepared on first use.
    cks: Vec<PyOnceLock<rt::Ck>>,
    /// An inline cache per instruction (used by attribute loads and stores
    /// in compiled code).
    pub(crate) ics: Vec<ops::AttrIc>,
    /// A method cache per instruction (used by `CallMethod`).
    pub(crate) mics: Vec<ops::MethodIc>,
    /// A global-lookup cache per instruction (used by `LoadGlobal`).
    pub(crate) gics: Vec<globals::GlobalIc>,
    table: Table,
}

impl FuncRt {
    pub(crate) fn kwnames(&self, pc: usize) -> Option<&Py<PyTuple>> {
        self.kwnames.get(&pc)
    }

    /// Check `obj` against check `idx`; on failure, what it is.
    pub(crate) fn check(&self, py: Python<'_>, idx: u32, obj: &Bound<'_, PyAny>) -> PyResult<Result<(), String>> {
        let ck = self.ck(py, idx);
        // SAFETY: a valid object.
        if unsafe { rt::quick(obj.as_ptr(), ck) } {
            return Ok(Ok(()));
        }
        rt::check_prepared(py, obj, ck)
    }

    /// Check `idx`, prepared.
    pub(crate) fn ck(&self, py: Python<'_>, idx: u32) -> &rt::Ck {
        let cell = &self.cks[idx as usize];
        match cell.get(py) {
            Some(ck) => ck,
            None => cell.get_or_init(py, || rt::prepare(py, &self.ir.checks[idx as usize].check)),
        }
    }
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
        let names: Vec<Py<PyString>> = ir.names.iter().map(|n| PyString::intern(py, n).unbind()).collect();
        let mut kwnames = HashMap::new();
        for (pc, instr) in ir.code.iter().enumerate() {
            if let Instr::Call { kwnames: k, .. } | Instr::CallMethod { kwnames: k, .. } = instr {
                if !k.is_empty() {
                    kwnames.insert(pc, PyTuple::new(py, k.iter().map(|n| names[*n as usize].bind(py)))?.unbind());
                }
            }
        }
        let objs = ir.reprs.iter().map(|r| *r == Repr::Obj).collect();
        let cks = ir.checks.iter().map(|_| PyOnceLock::new()).collect();
        let ics = ir.code.iter().map(|_| ops::AttrIc::default()).collect();
        let gics = ir.code.iter().map(|_| globals::GlobalIc::default()).collect();
        let mics = ir.code.iter().map(|_| ops::MethodIc::default()).collect();
        let positional = ir.params.iter().all(|p| matches!(p.kind, ParamKind::PosOnly | ParamKind::PosOrKw));
        Ok(FuncRt { ir, objs, positional, jit: OnceLock::new(), consts, names, kwnames, cks, ics, mics, gics, table })
    }
}

/// A checked Lizzard function, callable from Python.
#[pyclass(frozen, module = "lizzard._lizzard")]
pub struct LizzardFunction {
    /// CPython calls it through this pointer (vectorcall; see `create`).
    vectorcall: ffi::vectorcallfunc,
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
        LizzardFunction { vectorcall: lz_vectorcall, rt, globals, module, closure, defaults }
    }

    /// The function object. The first one also makes the type support
    /// vectorcall (CPython calls `lz_vectorcall` with the arguments in an
    /// array: no tuple, no dict) and marks it a method descriptor, so that
    /// `obj.m(...)` calls it with `obj` first instead of creating a bound
    /// method, as for Python functions. PyO3 has no option for either, so
    /// the type is patched once, before any instance is used.
    pub fn create(py: Python<'_>, f: LizzardFunction) -> PyResult<Py<LizzardFunction>> {
        static PATCHED: OnceLock<()> = OnceLock::new();
        let obj = Py::new(py, f)?;
        PATCHED.get_or_init(|| {
            let field = &obj.get().vectorcall as *const ffi::vectorcallfunc as usize;
            let offset = (field - obj.as_ptr() as usize) as ffi::Py_ssize_t;
            let tp = py.get_type::<LizzardFunction>();
            let tp = tp.as_type_ptr();
            FUNCTION_TYPE.store(tp, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: the type object is ours; no instance has been called
            // yet. `offset` is where every instance keeps its pointer.
            unsafe {
                (*tp).tp_vectorcall_offset = offset;
                let add = ffi::Py_TPFLAGS_HAVE_VECTORCALL | ffi::Py_TPFLAGS_METHOD_DESCRIPTOR;
                #[cfg(Py_GIL_DISABLED)]
                (*tp).tp_flags.fetch_or(add, std::sync::atomic::Ordering::SeqCst);
                #[cfg(not(Py_GIL_DISABLED))]
                {
                    (*tp).tp_flags |= add;
                }
                ffi::PyType_Modified(tp);
            }
        });
        Ok(obj)
    }

    /// Call with checked arguments: run it, or create its generator.
    fn invoke(&self, py: Python<'_>, bound: Vec<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let ir = &self.rt.ir;
        check_args(py, &self.rt, &bound)?;
        if ir.generator {
            // A generator function returns a generator over a new frame.
            let g = LizzardGenerator::new(self.rt.clone(), self.globals.clone_ref(py), self.closure.iter().map(|c| c.clone_ref(py)).collect(), bound);
            return Ok(Py::new(py, g)?.into_any());
        }
        run(py, &self.rt, self.globals.bind(py), bound, &self.closure)
    }

    /// A vectorcall: positional arguments then keyword values, with their
    /// names in `kwnames` (a tuple, or null).
    fn call_vector(&self, py: Python<'_>, args: &[*mut ffi::PyObject], kwnames: *mut ffi::PyObject) -> PyResult<Py<PyAny>> {
        let ir = &self.rt.ir;
        // SAFETY: CPython passes valid borrowed references.
        let arg = |p: *mut ffi::PyObject| unsafe { Bound::from_borrowed_ptr(py, p) };
        let nkw = if kwnames.is_null() { 0 } else { unsafe { ffi::PyTuple_GET_SIZE(kwnames) as usize } };
        // The common case: exactly the positional parameters. They go
        // straight into the new frame, checked by type pointers first.
        if nkw == 0 && args.len() == ir.params.len() && self.rt.positional && !ir.generator {
            let rt = &self.rt;
            for (p, &a) in ir.params.iter().zip(args) {
                // SAFETY: a valid object.
                if !unsafe { rt::quick(a, rt.ck(py, p.check)) } {
                    if let Err(got) = rt.check(py, p.check, &arg(a))? {
                        let c = &ir.checks[p.check as usize];
                        return Err(PyTypeError::new_err(format!("{}() argument '{}': expected {}, got {got}", ir.name, p.name, c.expected)));
                    }
                }
            }
            let mut regs = Regs::new(&rt.objs);
            for (i, &a) in args.iter().enumerate() {
                // SAFETY: a new reference for the frame.
                unsafe { ffi::Py_IncRef(a) };
                regs.init_obj(i as u32, a);
            }
            return run_frame(py, rt, self.globals.bind(py), &self.closure, &mut regs);
        }
        let npos = args.len() - nkw;
        let pos = PyTuple::new(py, args[..npos].iter().map(|p| arg(*p)))?;
        let kwargs = if nkw == 0 {
            None
        } else {
            let d = PyDict::new(py);
            let names = unsafe { Bound::from_borrowed_ptr(py, kwnames) };
            for (i, n) in names.try_iter()?.enumerate() {
                d.set_item(n?, arg(args[npos + i]))?;
            }
            Some(d)
        };
        let bound = bind_args(ir, &self.rt.consts, &self.defaults, &pos, kwargs.as_ref())?;
        self.invoke(py, bound)
    }
}

static FUNCTION_TYPE: std::sync::atomic::AtomicPtr<ffi::PyTypeObject> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// `LizzardFunction`'s type object, once a function exists (null before).
fn lizzard_function_type() -> *mut ffi::PyTypeObject {
    FUNCTION_TYPE.load(std::sync::atomic::Ordering::Relaxed)
}

/// CPython's entry into a Lizzard function (vectorcall protocol).
unsafe extern "C" fn lz_vectorcall(callable: *mut ffi::PyObject, args: *const *mut ffi::PyObject, nargsf: usize, kwnames: *mut ffi::PyObject) -> *mut ffi::PyObject {
    // SAFETY: CPython calls this with the thread attached, `callable` a
    // LizzardFunction (the vectorcall slot is only set on that type), and
    // `nargs` arguments plus one per keyword name in `args`.
    let py = unsafe { Python::assume_attached() };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let f = unsafe { Bound::from_borrowed_ptr(py, callable) };
        let f = unsafe { f.cast_unchecked::<LizzardFunction>() };
        let nargs = unsafe { ffi::PyVectorcall_NARGS(nargsf) } as usize;
        let nkw = if kwnames.is_null() { 0 } else { unsafe { ffi::PyTuple_GET_SIZE(kwnames) as usize } };
        let all = if nargs + nkw == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(args, nargs + nkw) } };
        f.get().call_vector(py, all, kwnames)
    }));
    match r {
        Ok(Ok(v)) => v.into_ptr(),
        Ok(Err(e)) => {
            e.restore(py);
            std::ptr::null_mut()
        }
        Err(_) => {
            pyo3::panic::PanicException::new_err("panic in a Lizzard function").restore(py);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" {
    /// A bound method (`types.MethodType(func, obj)`).
    fn PyMethod_New(func: *mut ffi::PyObject, obj: *mut ffi::PyObject) -> *mut ffi::PyObject;
}

#[pymethods]
impl LizzardFunction {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        let bound = bind_args(&self.rt.ir, &self.rt.consts, &self.defaults, args, kwargs)?;
        self.invoke(py, bound)
    }

    /// Bind as a method when stored in a class, like a Python function.
    fn __get__<'py>(slf: Bound<'py, Self>, instance: Option<Bound<'py, PyAny>>, _owner: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        match instance {
            // SAFETY: valid objects in; a new reference or NULL out.
            Some(obj) if !obj.is_none() => unsafe { Bound::from_owned_ptr_or_err(py, PyMethod_New(slf.as_ptr(), obj.as_ptr())) },
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
fn check_args(py: Python<'_>, f: &FuncRt, bound: &[Bound<'_, PyAny>]) -> PyResult<()> {
    let ir = &f.ir;
    for (p, v) in ir.params.iter().zip(bound) {
        let c = &ir.checks[p.check as usize];
        let fail = |what: String, got: String| Err(PyTypeError::new_err(format!("{}() argument {what}: expected {}, got {got}", ir.name, c.expected)));
        match p.kind {
            ParamKind::VarPos => {
                for (i, item) in v.try_iter()?.enumerate() {
                    if let Err(got) = f.check(py, p.check, &item?)? {
                        return fail(format!("'{}[{i}]'", p.name), got);
                    }
                }
            }
            ParamKind::VarKw => {
                for (k, item) in v.cast::<PyDict>().map_err(PyErr::from)?.iter() {
                    if let Err(got) = f.check(py, p.check, &item)? {
                        return fail(format!("'{}[{}]'", p.name, k.repr()?), got);
                    }
                }
            }
            _ => {
                if let Err(got) = f.check(py, p.check, v)? {
                    return fail(format!("'{}'", p.name), got);
                }
            }
        }
    }
    Ok(())
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

fn run<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, args: Vec<Bound<'py, PyAny>>, closure: &[Py<PyAny>]) -> PyResult<Py<PyAny>> {
    let mut regs = Regs::new(&f.objs);
    for (i, a) in args.into_iter().enumerate() {
        regs.init_obj(i as u32, a.into_ptr());
    }
    run_frame(py, f, globals, closure, &mut regs)
}

/// Run a new frame (its arguments in place) to its return.
fn run_frame<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, closure: &[Py<PyAny>], regs: &mut Regs) -> PyResult<Py<PyAny>> {
    // Calls between Lizzard functions recurse on the C stack: CPython's
    // recursion limit keeps that bounded (RecursionError, not a crash).
    // SAFETY: attached; the message is a C string.
    if unsafe { ffi::Py_EnterRecursiveCall(c" while calling a Lizzard function".as_ptr()) } != 0 {
        return Err(PyErr::fetch(py));
    }
    let mut pc = 0usize;
    let r = exec(py, f, globals, closure, regs, &mut pc, None);
    unsafe { ffi::Py_LeaveRecursiveCall() };
    regs.release(py);
    match r? {
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
    regs: &mut Regs,
    pc: &mut usize,
    pending: Option<PyErr>,
) -> PyResult<Exit<'py>> {
    if jit::enabled() {
        if let Some(entry) = jit::code(f) {
            return jit::exec(py, entry, f, globals, closure, regs, pc, pending);
        }
    }
    run_interp(py, f, globals, closure, regs, pc, pending)
}

/// The interpreter. The common instructions run here on the raw slots,
/// with the same helpers compiled code calls (`ops`); the rest go to `step`.
/// The loop keeps its state in locals and handles exceptions only when one
/// is raised, so an instruction costs a dispatch and its own work.
fn run_interp<'py>(
    py: Python<'py>,
    f: &FuncRt,
    globals: &Bound<'py, PyDict>,
    closure: &[Py<PyAny>],
    regs: &mut Regs,
    pc: &mut usize,
    pending: Option<PyErr>,
) -> PyResult<Exit<'py>> {
    use lizzard_jit::{NEXT, TAKEN};
    let ir = &f.ir;
    let code = &ir.code[..];
    let slots = regs.as_mut_ptr();
    // SAFETY (for the raw slot accesses below): the front end emits register
    // numbers below `ir.reprs.len()`, the number of slots, and `step` never
    // reallocates them.
    let get = |r: u32| unsafe { *slots.add(r as usize) };
    let set = |r: u32, v: u64| unsafe { *slots.add(r as usize) = v };
    let mut p = *pc;
    let mut err = pending;
    loop {
        if let Some(e) = err.take() {
            // The innermost handler covering the instruction: regions nest,
            // and an inner handler is created after its outer one.
            let at = p as u32;
            match ir.handlers.iter().rev().find(|h| h.ranges.iter().any(|(s, e)| *s <= at && at < *e)) {
                Some(h) => {
                    regs.set_obj(h.exc, e.into_value(py).into_bound(py).into_any());
                    p = h.target as usize;
                }
                None => {
                    *pc = p;
                    let line = ir.lines.get(p).copied().unwrap_or(ir.line);
                    let _ = e.add_note(py, format!("lizzard: {}:{line} in {}", ir.file, ir.name));
                    return Err(e);
                }
            }
        }
        loop {
            let instr = &code[p];
            // `Some(status)` when handled here, None for `step`.
            let fast = match instr {
                Instr::Jump { target } => {
                    p = *target as usize;
                    continue;
                }
                Instr::JumpIfFalse { cond, target } => {
                    p = if get(*cond) == 0 { *target as usize } else { p + 1 };
                    continue;
                }
                Instr::JumpIfTrue { cond, target } => {
                    p = if get(*cond) != 0 { *target as usize } else { p + 1 };
                    continue;
                }
                Instr::I64Const { dst, value } => {
                    set(*dst, *value as u64);
                    Some(NEXT)
                }
                Instr::I64Lt { dst, a, b } => {
                    set(*dst, ((get(*a) as i64) < (get(*b) as i64)) as u64);
                    Some(NEXT)
                }
                Instr::I64Add { dst, a, b } => {
                    set(*dst, (get(*a) as i64).wrapping_add(get(*b) as i64) as u64);
                    Some(NEXT)
                }
                Instr::Not { dst, src } => {
                    set(*dst, get(*src) ^ 1);
                    Some(NEXT)
                }
                Instr::Const { dst, idx } => {
                    ops::lz_const(slots, *dst, f.consts[*idx as usize].as_ptr());
                    Some(NEXT)
                }
                Instr::Move { dst, src } if f.objs[*src as usize] => Some(ops::lz_move(slots, *dst, *src)),
                Instr::Move { dst, src } => {
                    set(*dst, get(*src));
                    Some(NEXT)
                }
                Instr::Take { dst, src } if f.objs[*src as usize] => Some(ops::lz_take(slots, *dst, *src)),
                Instr::Unbind { reg } => {
                    regs.clear(*reg);
                    Some(NEXT)
                }
                Instr::CompareBool { dst, op, a, b } if ops::rich_code(*op).is_some() => Some(ops::lz_compare_bool(slots, *dst, *a, *b, ops::rich_code(*op).unwrap())),
                Instr::LoadGlobal { dst, name } => Some(match builtins_dict(py).and_then(|b| globals::load(py, &f.gics[p], globals, b, f.names[*name as usize].bind(py))) {
                    Ok(v) => {
                        regs.set_obj(*dst, v);
                        NEXT
                    }
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }),
                Instr::LoadAttr { dst, obj, name } => Some(ops::lz_load_attr(slots, *dst, *obj, f.names[*name as usize].as_ptr(), &f.ics[p])),
                Instr::StoreAttr { obj, name, val } => Some(ops::lz_store_attr(slots, *obj, *val, f.names[*name as usize].as_ptr(), &f.ics[p])),
                Instr::BinOp { dst, op, a, b, inplace } => Some(ops::lz_binop(slots, *dst, *a, *b, *op as u32 + if *inplace { 16 } else { 0 })),
                Instr::Compare { dst, op, a, b } if ops::rich_code(*op).is_some() => Some(ops::lz_compare(slots, *dst, *a, *b, ops::rich_code(*op).unwrap())),
                Instr::Truthy { dst, src } => Some(ops::lz_truthy(slots, *dst, *src)),
                Instr::Box { dst, src } if ir.reprs[*src as usize] == Repr::I64 => Some(ops::lz_box_i64(slots, *dst, *src)),
                Instr::Check { reg, check } => {
                    let o = get(*reg) as *mut ffi::PyObject;
                    // SAFETY: a non-null object slot holds a valid object.
                    (!o.is_null() && unsafe { rt::quick(o, f.ck(py, *check)) }).then_some(NEXT)
                }
                // The frame ends: its reference is the return value.
                Instr::Return { src } if f.objs[*src as usize] && get(*src) != 0 => {
                    *pc = p;
                    let v = regs.take_obj(*src);
                    // SAFETY: an owned reference out of the frame.
                    return Ok(Exit::Return(unsafe { Bound::from_owned_ptr(py, v) }));
                }
                Instr::Call { dst, func, args, .. } => Some(match regs.obj(py, *func).map(|o| o.as_ptr()).and_then(|c| vector_call(py, regs, None, c, args, f.kwnames(p))) {
                    Ok(v) => {
                        regs.set_obj(*dst, v);
                        NEXT
                    }
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }),
                Instr::CallMethod { dst, obj, name, args, .. } => {
                    Some(match regs.obj(py, *obj).map(|o| o.as_ptr()).and_then(|o| vector_call(py, regs, Some((o, &f.mics[p])), f.names[*name as usize].as_ptr(), args, f.kwnames(p))) {
                        Ok(v) => {
                            regs.set_obj(*dst, v);
                            NEXT
                        }
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    })
                }
                Instr::ForIter { dst, iter, target } => {
                    let it = get(*iter) as *mut ffi::PyObject;
                    // SAFETY: a type pointer comparison on a valid object.
                    if it.is_null() || unsafe { ffi::Py_TYPE(it) } == generator::type_ptr() {
                        None
                    } else {
                        match ops::lz_for_iter_py(slots, *dst, it) {
                            TAKEN => {
                                p = *target as usize;
                                continue;
                            }
                            st => Some(st),
                        }
                    }
                }
                _ => None,
            };
            match fast {
                Some(NEXT) => p += 1,
                Some(_) => {
                    err = Some(PyErr::fetch(py));
                    break;
                }
                None => {
                    let mut q = p;
                    match step(py, f, globals, closure, regs, &mut q) {
                        Ok(None) => p = q,
                        Ok(Some(exit)) => {
                            *pc = q;
                            return Ok(exit);
                        }
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
            }
        }
    }
}

macro_rules! obj {
    ($py:expr, $regs:expr, $r:expr) => {
        $regs.obj($py, $r)?
    };
}

/// The `builtins` module, found once (also usable while the interpreter
/// shuts down and imports no longer work).
fn builtins(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static BUILTINS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Ok(BUILTINS.get_or_try_init(py, || py.import("builtins").map(|m| m.into_any().unbind()))?.bind(py))
}

/// The builtins module's namespace.
fn builtins_dict(py: Python<'_>) -> PyResult<&Bound<'_, PyDict>> {
    static DICT: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
    Ok(DICT.get_or_try_init(py, || builtins(py)?.getattr("__dict__")?.cast_into::<PyDict>().map(|d| d.unbind()).map_err(PyErr::from))?.bind(py))
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

/// Call `callee` with the objects in `args` (the last `kwnames.len()` are
/// keyword values), by vectorcall; with `slf`, call method `callee` (a name)
/// of `slf`.
pub(crate) fn vector_call<'py>(
    py: Python<'py>,
    regs: &Regs,
    slf: Option<(*mut ffi::PyObject, &ops::MethodIc)>,
    callee: *mut ffi::PyObject,
    args: &[u32],
    kwnames: Option<&Py<PyTuple>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut small = [std::ptr::null_mut(); 8];
    let mut big = Vec::new();
    let n = args.len() + slf.is_some() as usize;
    let argv: &mut [*mut ffi::PyObject] = if n <= small.len() {
        &mut small[..n]
    } else {
        big.resize(n, std::ptr::null_mut());
        &mut big
    };
    let mut k = 0;
    if let Some((s, _)) = slf {
        argv[0] = s;
        k = 1;
    }
    for r in args {
        argv[k] = regs.obj(py, *r)?.as_ptr();
        k += 1;
    }
    let kw = kwnames.map_or(std::ptr::null_mut(), |t| t.as_ptr());
    let nkw = kwnames.map_or(0, |t| t.bind(py).len());
    let nargs = n - nkw;
    // A method the call site's cache knows: call it with the object first,
    // as CPython does for a method descriptor.
    if let Some((o, ic)) = slf {
        // SAFETY: valid objects; a new reference to the method comes back.
        if let Some(m) = unsafe { ops::method(ic, o, callee) } {
            let m = unsafe { Bound::from_owned_ptr(py, m) };
            return call_object(py, m.as_ptr(), argv, nargs, kw);
        }
    }
    if slf.is_none() {
        return call_object(py, callee, argv, nargs, kw);
    }
    // SAFETY: `argv` holds borrowed references the frame keeps alive for the
    // call; `kw` is a tuple of the keyword names or NULL. A new reference or
    // NULL comes back.
    let r = unsafe { ffi::PyObject_VectorcallMethod(callee, argv.as_ptr(), nargs, kw) };
    unsafe { Bound::from_owned_ptr_or_err(py, r) }
}

/// Call `callee` with `argv` (`nargs` positional, then keyword values named
/// by `kw`): a Lizzard function directly, without CPython's vectorcall
/// dispatch and its panic guard, anything else by vectorcall.
fn call_object<'py>(py: Python<'py>, callee: *mut ffi::PyObject, argv: &[*mut ffi::PyObject], nargs: usize, kw: *mut ffi::PyObject) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: a type pointer comparison on a valid object.
    if unsafe { ffi::Py_TYPE(callee) } == lizzard_function_type() {
        // SAFETY: the type says what it is.
        let lf = unsafe { Bound::from_borrowed_ptr(py, callee).cast_into_unchecked::<LizzardFunction>() };
        return lf.get().call_vector(py, argv, kw).map(|v| v.into_bound(py));
    }
    // SAFETY: `argv` holds borrowed references alive for the call.
    unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyObject_Vectorcall(callee, argv.as_ptr(), nargs, kw)) }
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
pub(crate) fn step<'py>(py: Python<'py>, f: &FuncRt, globals: &Bound<'py, PyDict>, closure: &[Py<PyAny>], regs: &mut Regs, pc: &mut usize) -> PyResult<Option<Exit<'py>>> {
    let ir = &f.ir;
    let instr = &ir.code[*pc];
    let mut next = *pc + 1;
    match instr {
        Instr::Const { dst, idx } => regs.set_obj(*dst, f.consts[*idx as usize].bind(py).clone()),
        Instr::LoadGlobal { dst, name } => {
            let v = globals::load(py, &f.gics[*pc], globals, builtins_dict(py)?, f.names[*name as usize].bind(py))?;
            regs.set_obj(*dst, v);
        }
        Instr::StoreGlobal { name, val } => globals.set_item(f.names[*name as usize].bind(py), obj!(py, regs, *val))?,
        Instr::MakeCell { dst } => {
            // SAFETY: NULL makes an empty cell; returns a new reference.
            let c = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyCell_New(std::ptr::null_mut()))? };
            regs.set_obj(*dst, c);
        }
        Instr::LoadCell { dst, cell } => match cell_get(py, obj!(py, regs, *cell))? {
            Some(v) => regs.set_obj(*dst, v),
            None => return Err(unbound()),
        },
        Instr::StoreCell { cell, val } => cell_set(obj!(py, regs, *cell), obj!(py, regs, *val))?,
        Instr::LoadFree { dst, idx } => match cell_get(py, closure[*idx as usize].bind(py))? {
            Some(v) => regs.set_obj(*dst, v),
            None => {
                let n = &f.ir.free[*idx as usize];
                return Err(PyNameError::new_err(format!("cannot access free variable '{n}' where it is not associated with a value in enclosing scope")));
            }
        },
        Instr::StoreFree { idx, val } => cell_set(closure[*idx as usize].bind(py), obj!(py, regs, *val))?,
        Instr::LoadFreeCell { dst, idx } => regs.set_obj(*dst, closure[*idx as usize].bind(py).clone()),
        Instr::MakeFunction { dst, key, closure: cells, defaults } => {
            let key = f.names[*key as usize].bind(py).to_cow()?;
            let rt = f.table.get().and_then(|t| t.get(key.as_ref())).cloned().expect("inner function compiled with its module");
            let module = match globals.get_item("__name__")? {
                Some(n) => n.extract()?,
                None => String::new(),
            };
            let mut cl = Vec::with_capacity(cells.len());
            for r in cells {
                cl.push(obj!(py, regs, *r).clone().unbind());
            }
            let mut dv = Vec::with_capacity(defaults.len());
            for r in defaults {
                dv.push(obj!(py, regs, *r).clone().unbind());
            }
            let func = LizzardFunction::create(py, LizzardFunction::new(rt, globals.clone().unbind(), module, cl, dv))?;
            regs.set_obj(*dst, func.into_bound(py).into_any());
        }
        Instr::Move { dst, src } => {
            if !regs.is_bound(*src) {
                return Err(unbound());
            }
            regs.copy(py, *dst, *src)?;
        }
        Instr::Take { dst, src } => {
            if !regs.is_bound(*src) {
                return Err(unbound());
            }
            regs.copy(py, *dst, *src)?;
            regs.clear(*src);
        }
        Instr::CompareBool { dst, op, a, b } => {
            let (x, y) = (obj!(py, regs, *a), obj!(py, regs, *b));
            let t = match op {
                CmpOp::Is => x.is(y),
                CmpOp::IsNot => !x.is(y),
                CmpOp::In => y.contains(x)?,
                CmpOp::NotIn => !y.contains(x)?,
                op => {
                    let o = match op {
                        CmpOp::Eq => CompareOp::Eq,
                        CmpOp::Ne => CompareOp::Ne,
                        CmpOp::Lt => CompareOp::Lt,
                        CmpOp::Le => CompareOp::Le,
                        CmpOp::Gt => CompareOp::Gt,
                        _ => CompareOp::Ge,
                    };
                    x.rich_compare(y, o)?.is_truthy()?
                }
            };
            regs.set_bool(*dst, t);
        }
        Instr::LoadAttr { dst, obj, name } => {
            let v = obj!(py, regs, *obj).getattr(f.names[*name as usize].bind(py))?;
            regs.set_obj(*dst, v);
        }
        Instr::Call { dst, func, args, .. } => {
            let callee = obj!(py, regs, *func).as_ptr();
            let v = vector_call(py, regs, None, callee, args, f.kwnames.get(pc))?;
            regs.set_obj(*dst, v);
        }
        Instr::CallMethod { dst, obj, name, args, .. } => {
            let o = obj!(py, regs, *obj).as_ptr();
            let v = vector_call(py, regs, Some((o, &f.mics[*pc])), f.names[*name as usize].as_ptr(), args, f.kwnames.get(pc))?;
            regs.set_obj(*dst, v);
        }
        Instr::CallEx { dst, func, args, kwargs } => {
            let a = obj!(py, regs, *args).cast::<PyTuple>().map_err(PyErr::from)?.clone();
            let kw = match kwargs {
                Some(k) => Some(obj!(py, regs, *k).cast::<PyDict>().map_err(PyErr::from)?.clone()),
                None => None,
            };
            let v = obj!(py, regs, *func).call(a, kw.as_ref())?;
            regs.set_obj(*dst, v);
        }
        Instr::ListExtend { list, iter } => {
            let l = obj!(py, regs, *list).cast::<PyList>().map_err(PyErr::from)?.clone();
            for item in obj!(py, regs, *iter).try_iter()? {
                l.append(item?)?;
            }
        }
        Instr::ListToTuple { dst, src } => {
            let t = obj!(py, regs, *src).cast::<PyList>().map_err(PyErr::from)?.to_tuple();
            regs.set_obj(*dst, t.into_any());
        }
        Instr::DictMerge { dict, src, func } => {
            let d = obj!(py, regs, *dict).cast::<PyDict>().map_err(PyErr::from)?.clone();
            let src = obj!(py, regs, *src);
            let keys = src.call_method0("keys").map_err(|_| {
                PyTypeError::new_err(format!("argument after ** must be a mapping, not {}", src.get_type().name().map(|n| n.to_string()).unwrap_or_default()))
            })?;
            for k in keys.try_iter()? {
                let k = k?;
                if d.contains(&k)? {
                    let fname = obj!(py, regs, *func).getattr("__qualname__").and_then(|n| n.extract::<String>()).unwrap_or_else(|_| "function".into());
                    return Err(PyTypeError::new_err(format!("{fname}() got multiple values for keyword argument '{}'", k.str()?)));
                }
                d.set_item(&k, src.get_item(&k)?)?;
            }
        }
        Instr::BinOp { dst, op, a, b, inplace } => {
            let (x, y) = (obj!(py, regs, *a), obj!(py, regs, *b));
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
                    BinOp::MatMul => ffi::PyNumber_InPlaceMatrixMultiply,
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
                    BinOp::MatMul => x.matmul(y)?,
                    BinOp::BitOr => x.bitor(y)?,
                    BinOp::BitAnd => x.bitand(y)?,
                    BinOp::BitXor => x.bitxor(y)?,
                    BinOp::LShift => x.lshift(y)?,
                    BinOp::RShift => x.rshift(y)?,
                }
            };
            regs.set_obj(*dst, v);
        }
        Instr::Neg { dst, src } => {
            let v = obj!(py, regs, *src).neg()?;
            regs.set_obj(*dst, v);
        }
        Instr::Invert { dst, src } => {
            let v = obj!(py, regs, *src).bitnot()?;
            regs.set_obj(*dst, v);
        }
        Instr::Pos { dst, src } => {
            let v = obj!(py, regs, *src).pos()?;
            regs.set_obj(*dst, v);
        }
        Instr::BuildSet { dst, items } => {
            let set = PySet::empty(py)?;
            for r in items {
                set.add(obj!(py, regs, *r))?;
            }
            regs.set_obj(*dst, set.into_any());
        }
        Instr::SetAdd { set, val } => {
            obj!(py, regs, *set).cast::<PySet>().map_err(PyErr::from)?.add(obj!(py, regs, *val))?;
        }
        Instr::StoreAttr { obj, name, val } => obj!(py, regs, *obj).setattr(f.names[*name as usize].bind(py), obj!(py, regs, *val))?,
        Instr::DelItem { obj, key } => obj!(py, regs, *obj).del_item(obj!(py, regs, *key))?,
        Instr::DelAttr { obj, name } => obj!(py, regs, *obj).delattr(f.names[*name as usize].bind(py))?,
        Instr::BuildList { dst, items } => {
            let mut v = Vec::with_capacity(items.len());
            for r in items {
                v.push(obj!(py, regs, *r).clone());
            }
            regs.set_obj(*dst, PyList::new(py, v)?.into_any());
        }
        Instr::BuildTuple { dst, items } => {
            let mut v = Vec::with_capacity(items.len());
            for r in items {
                v.push(obj!(py, regs, *r).clone());
            }
            regs.set_obj(*dst, PyTuple::new(py, v)?.into_any());
        }
        Instr::BuildDict { dst, keys, vals } => {
            let d = PyDict::new(py);
            for (k, v) in keys.iter().zip(vals) {
                d.set_item(obj!(py, regs, *k), obj!(py, regs, *v))?;
            }
            regs.set_obj(*dst, d.into_any());
        }
        Instr::BuildSlice { dst, lower, upper, step } => {
            let (a, b, c) = (obj!(py, regs, *lower), obj!(py, regs, *upper), obj!(py, regs, *step));
            // SAFETY: valid objects in, a new reference or NULL out.
            let v = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PySlice_New(a.as_ptr(), b.as_ptr(), c.as_ptr()))? };
            regs.set_obj(*dst, v);
        }
        Instr::ListAppend { list, val } => {
            let l = obj!(py, regs, *list).cast::<PyList>().map_err(PyErr::from)?;
            l.append(obj!(py, regs, *val))?;
        }
        Instr::GetIter { dst, obj } => {
            let it = obj!(py, regs, *obj).try_iter()?;
            regs.set_obj(*dst, it.into_any());
        }
        Instr::ForIter { dst, iter, target } => {
            let it = obj!(py, regs, *iter);
            // A Lizzard generator is resumed directly, without an exception
            // at the end.
            if let Ok(g) = it.cast::<LizzardGenerator>() {
                match g.get().resume(py, generator::Op::Send(py.None().into_bound(py)))? {
                    Sent::Next(v) => regs.set_obj(*dst, v),
                    Sent::Return(_) => next = *target as usize,
                }
            } else {
                let mut it = it.cast::<PyIterator>().map_err(PyErr::from)?.clone();
                match it.next() {
                    Some(v) => regs.set_obj(*dst, v?),
                    None => next = *target as usize,
                }
            }
        }
        Instr::Unpack { src, dsts } => {
            let o = obj!(py, regs, *src).clone();
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
                    Some(v) => regs.set_obj(*d, v?),
                    None => return Err(PyValueError::new_err(format!("not enough values to unpack (expected {n}, got {i})"))),
                }
            }
            if let Some(extra) = it.next() {
                extra?;
                return Err(PyValueError::new_err(format!("too many values to unpack (expected {n})")));
            }
        }
        Instr::Format { dst, val, conv, spec } => {
            let v = obj!(py, regs, *val);
            let v = match conv {
                b's' => v.str()?.into_any(),
                b'r' => v.repr()?.into_any(),
                // SAFETY: a valid object in, a new reference or NULL out.
                b'a' => unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyObject_ASCII(v.as_ptr()))? },
                _ => v.clone(),
            };
            let spec = match spec {
                Some(r) => obj!(py, regs, *r).clone(),
                None => PyString::new(py, "").into_any(),
            };
            // SAFETY: as above.
            let s = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyObject_Format(v.as_ptr(), spec.as_ptr()))? };
            regs.set_obj(*dst, s);
        }
        Instr::BuildString { dst, parts } => {
            let mut s = String::new();
            for r in parts {
                s.push_str(&obj!(py, regs, *r).cast::<PyString>().map_err(PyErr::from)?.to_cow()?);
            }
            regs.set_obj(*dst, PyString::new(py, &s).into_any());
        }
        Instr::Compare { dst, op, a, b } => {
            let (x, y) = (obj!(py, regs, *a), obj!(py, regs, *b));
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
            regs.set_obj(*dst, v);
        }
        Instr::GetItem { dst, obj, key } => {
            let v = obj!(py, regs, *obj).get_item(obj!(py, regs, *key))?;
            regs.set_obj(*dst, v);
        }
        Instr::SetItem { obj, key, val } => obj!(py, regs, *obj).set_item(obj!(py, regs, *key), obj!(py, regs, *val))?,
        Instr::Truthy { dst, src } => {
            let b = obj!(py, regs, *src).is_truthy()?;
            regs.set_bool(*dst, b);
        }
        Instr::Not { dst, src } => regs.set_bool(*dst, !regs.bool(*src)),
        Instr::Jump { target } => next = *target as usize,
        Instr::JumpIfFalse { cond, target } => {
            if !regs.bool(*cond) {
                next = *target as usize;
            }
        }
        Instr::JumpIfTrue { cond, target } => {
            if regs.bool(*cond) {
                next = *target as usize;
            }
        }
        Instr::Raise { exc, cause } => {
            let e = exception_instance(obj!(py, regs, *exc))?;
            if let Some(c) = cause {
                let c = obj!(py, regs, *c);
                let c = if c.is_none() { c.clone() } else { exception_instance(c)? };
                // Also sets `__suppress_context__`.
                e.setattr("__cause__", c)?;
            }
            // SAFETY: valid objects; `PyErr_SetObject` chains the exception
            // being handled as `__context__`, as `raise` does.
            unsafe { ffi::PyErr_SetObject(e.get_type().as_ptr(), e.as_ptr()) };
            return Err(PyErr::fetch(py));
        }
        Instr::Reraise { exc } => return Err(PyErr::from_value(obj!(py, regs, *exc).clone())),
        Instr::ExcMatch { dst, exc, cls } => {
            let m = obj!(py, regs, *exc).is_instance(obj!(py, regs, *cls))?;
            regs.set_bool(*dst, m);
        }
        Instr::GetHandled { dst } => {
            // SAFETY: returns a new reference or NULL.
            let p = unsafe { ffi::PyErr_GetHandledException() };
            let v = if p.is_null() { py.None().into_bound(py) } else { unsafe { Bound::from_owned_ptr(py, p) } };
            regs.set_obj(*dst, v);
        }
        Instr::SetHandled { src } => {
            // SAFETY: a valid object (None clears it); the reference is not stolen.
            unsafe { ffi::PyErr_SetHandledException(obj!(py, regs, *src).as_ptr()) };
        }
        Instr::Unbind { reg } => regs.clear(*reg),
        Instr::TypeOf { dst, src } => {
            let t = obj!(py, regs, *src).get_type().into_any();
            regs.set_obj(*dst, t);
        }
        Instr::LoadSpecial { dst, obj, name } => {
            let o = obj!(py, regs, *obj);
            let t = o.get_type();
            let n = f.names[*name as usize].bind(py);
            let a = t.getattr(n).map_err(|_| PyTypeError::new_err(format!("'{}' object does not support the context manager protocol", t.name().map(|s| s.to_string()).unwrap_or_default())))?;
            // Bind it like a method lookup does.
            let v = match a.get_type().getattr("__get__") {
                Ok(get) => get.call1((&a, o, &t))?,
                Err(_) => a,
            };
            regs.set_obj(*dst, v);
        }
        Instr::Return { src } => return Ok(Some(Exit::Return(obj!(py, regs, *src).clone()))),
        // Both leave `pc` at the instruction: the generator resumes after it,
        // or raises there.
        Instr::Yield { dst, src } => return Ok(Some(Exit::Yield { value: obj!(py, regs, *src).clone(), dst: *dst })),
        Instr::YieldFrom { dst, iter } => {
            let it = obj!(py, regs, *iter).clone();
            match generator::send_to(&it, &py.None().into_bound(py))? {
                Sent::Next(value) => return Ok(Some(Exit::YieldFrom { value, dst: *dst, iter: it })),
                Sent::Return(v) => regs.set_obj(*dst, v),
            }
        }
        Instr::I64Const { dst, value } => regs.set_i64(*dst, *value),
        Instr::I64Lt { dst, a, b } => regs.set_bool(*dst, regs.i64(*a) < regs.i64(*b)),
        Instr::I64Add { dst, a, b } => regs.set_i64(*dst, regs.i64(*a) + regs.i64(*b)),
        Instr::Len { dst, obj } => {
            let n = obj!(py, regs, *obj).len()?;
            regs.set_i64(*dst, n as i64);
        }
        Instr::Box { dst, src } => {
            let v = match ir.reprs[*src as usize] {
                Repr::Obj => obj!(py, regs, *src).clone(),
                Repr::I64 => regs.i64(*src).into_pyobject(py)?.into_any(),
                Repr::F32 => array::box_f32(py, regs.f32(*src))?,
                Repr::Bool => PyBool::new(py, regs.bool(*src)).to_owned().into_any(),
            };
            regs.set_obj(*dst, v);
        }
        Instr::Unbox { dst, src } => {
            let o = obj!(py, regs, *src);
            match ir.reprs[*dst as usize] {
                Repr::I64 => {
                    let v = o.extract::<i64>()?;
                    regs.set_i64(*dst, v);
                }
                Repr::F32 => {
                    let v = o.extract::<f64>()? as f32;
                    regs.set_f32(*dst, v);
                }
                Repr::Bool => {
                    let v = o.is_truthy()?;
                    regs.set_bool(*dst, v);
                }
                Repr::Obj => {
                    let v = o.clone();
                    regs.set_obj(*dst, v);
                }
            }
        }
        Instr::ArrLoadF32 { dst, arr, idx } => {
            let v = array::load_f32(py, obj!(py, regs, *arr), regs.i64(*idx))?;
            regs.set_f32(*dst, v);
        }
        Instr::ArrStoreF32 { arr, idx, val } => array::store_f32(py, obj!(py, regs, *arr), regs.i64(*idx), regs.f32(*val))?,
        Instr::F32Bin { dst, op, a, b } => {
            let (x, y) = (regs.f32(*a), regs.f32(*b));
            let v = match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                _ => unreachable!("float32 arithmetic is + - * / only"),
            };
            regs.set_f32(*dst, v);
        }
        Instr::Check { reg, check } => {
            let c = &ir.checks[*check as usize];
            if let Err(got) = f.check(py, *check, obj!(py, regs, *reg))? {
                let line = ir.lines[*pc];
                let origin = c.origin.as_ref().map(|o| format!(" (declared at {o})")).unwrap_or_default();
                return Err(PyTypeError::new_err(format!("{}:{line}: {} should be {}{origin}, got {got}", ir.file, c.what, c.expected)));
            }
        }
    }
    *pc = next;
    Ok(None)
}
