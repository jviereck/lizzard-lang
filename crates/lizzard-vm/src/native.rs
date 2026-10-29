//! Native modules: Rust functions and classes callable from lizzard.
//!
//! This is the interpreter's side. Extensions build modules through the C
//! ABI of `lizzard-abi` (see `ext.rs`), with the binding API of `lizzard-ext`
//! (closures, conversions, class builders); statically embedded modules do
//! the same. A [`Module`] is built before a program is compiled. The
//! compiler sees each item's signature as lizzard types ([`NType`]), so calls
//! of natives are checked like calls of lizzard functions.
//!
//! A native class wraps a Rust value the interpreter does not look into:
//! an opaque pointer that the extension's drop function frees. Its mode
//! (from the Rust type's auto traits) decides how its methods may run:
//! - [`Mode::Sync`] (`T: Send + Sync`): shared calls run concurrently from
//!   any thread; exclusive (`&mut self`) calls are exclusive.
//! - [`Mode::Send`] (`T: Send`): every method call is exclusive.
//! - [`Mode::Local`] (any `T`): the lizzard class is not `Send` either, so
//!   instances stay on the thread that created them, which is also checked
//!   on every call (`ThreadError`), and are dropped there.
//!
//! A call that conflicts with a running one (an exclusive call while another
//! method runs) raises `BorrowError`, or waits for it to finish if the class
//! is locked.
//!
//! A generic class lists the instantiations it supports, each an ordinary
//! native class over its own Rust type (`Stack<i64>`, `Stack<String>`).
//! Lizzard sees one generic class `Stack<T>`; constructing `Stack<int>`
//! dispatches to the class registered for `int`, and other type arguments
//! raise `TypeError`.
//!
//! A module may define traits that its classes implement, as may lizzard
//! structs. A class may also implement the builtin `Ord`, `Num` and `Len`
//! through its dunder methods.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use lizzard_abi::{sig, CallFn, FreeFn, ReprFn};

use crate::array::Elem;
use crate::interp::{ExcInfo, Raised, Thread, VResult};
use crate::object::{Body, Obj};
use crate::rtype::{self, Ctor};
use crate::value::Value;

// ---------------------------------------------------------------- types

/// A lizzard type, as the compiler sees a native's parameters and result.
#[derive(Clone, Debug, PartialEq)]
pub enum NType {
    Int,
    Float,
    Bool,
    Str,
    Unit,
    Any,
    Function,
    List(Box<NType>),
    Dict(Box<NType>, Box<NType>),
    Option(Box<NType>),
    Array(Elem),
    /// A native class, by its class key (the Rust type it wraps).
    Class(u64),
}

impl NType {
    /// The runtime type of values of this type.
    pub fn rt(&self) -> Value {
        match self {
            NType::Int => rtype::int(),
            NType::Float => rtype::float(),
            NType::Bool => rtype::bool_(),
            NType::Str => rtype::str_(),
            NType::Unit => rtype::prim(Ctor::Unit),
            NType::Any => rtype::any(),
            NType::Function => rtype::local_of(rtype::as_type(rtype::prim(Ctor::Function)).unwrap()),
            NType::List(t) => rtype::list_of(t.rt()),
            NType::Dict(k, v) => rtype::dict_of(k.rt(), v.rt()),
            NType::Option(t) => rtype::option_of(t.rt()),
            NType::Array(e) => rtype::prim(Ctor::Array(Some(*e))),
            NType::Class(key) => match class_by_key(*key) {
                Some(c) => rtype::class_type(c),
                None => rtype::any(),
            },
        }
    }

    /// Whether this is `int`, `float` or `bool` (an unboxed scalar).
    pub fn is_scalar(&self) -> bool {
        matches!(self, NType::Int | NType::Float | NType::Bool)
    }

    /// Decodes a type descriptor (`lizzard_abi::sig`) at the start of `b`.
    pub(crate) fn decode(b: &mut &[u8]) -> Option<NType> {
        let (&tag, rest) = b.split_first()?;
        *b = rest;
        Some(match tag {
            sig::INT => NType::Int,
            sig::FLOAT => NType::Float,
            sig::BOOL => NType::Bool,
            sig::STR => NType::Str,
            sig::UNIT => NType::Unit,
            sig::ANY => NType::Any,
            sig::FUNCTION => NType::Function,
            sig::LIST => NType::List(Box::new(NType::decode(b)?)),
            sig::DICT => {
                let k = NType::decode(b)?;
                NType::Dict(Box::new(k), Box::new(NType::decode(b)?))
            }
            sig::OPTION => NType::Option(Box::new(NType::decode(b)?)),
            sig::ARRAY => {
                let (&e, rest) = b.split_first()?;
                *b = rest;
                NType::Array(elem_from_code(e)?)
            }
            sig::CLASS => {
                let (k, rest) = b.split_at_checked(8)?;
                *b = rest;
                NType::Class(u64::from_le_bytes(k.try_into().unwrap()))
            }
            _ => return None,
        })
    }

    /// Decodes a type list (a count, then the types).
    pub(crate) fn decode_list(b: &mut &[u8]) -> Option<Vec<NType>> {
        let (&n, rest) = b.split_first()?;
        *b = rest;
        (0..n).map(|_| NType::decode(b)).collect()
    }
}

/// The ABI code of an array element type.
pub(crate) fn elem_code(e: Elem) -> u8 {
    Elem::ALL.iter().position(|x| *x == e).unwrap() as u8
}

pub(crate) fn elem_from_code(c: u8) -> Option<Elem> {
    Elem::ALL.get(c as usize).copied()
}

/// Decodes a signature: (parameters, result).
pub(crate) fn decode_sig(mut b: &[u8]) -> Option<(Vec<NType>, NType)> {
    let params = NType::decode_list(&mut b)?;
    let ret = NType::decode(&mut b)?;
    b.is_empty().then_some((params, ret))
}

// ---------------------------------------------------------------- errors

/// An error raised in lizzard as an exception of class `kind`.
pub(crate) struct LizzardError {
    pub kind: String,
    pub msg: String,
    /// A lizzard exception raised by a callback, re-raised unchanged.
    pending: Option<PendingExc>,
}

struct PendingExc(ExcInfo);

impl Drop for PendingExc {
    fn drop(&mut self) {
        self.0.value.decref();
        self.0.cause.decref();
    }
}

impl LizzardError {
    pub fn new(kind: &str, msg: impl Into<String>) -> LizzardError {
        LizzardError { kind: kind.into(), msg: msg.into(), pending: None }
    }

    pub fn type_error(msg: impl Into<String>) -> LizzardError {
        LizzardError::new("TypeError", msg)
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Takes the thread's current exception.
    pub fn from_thread(th: &mut Thread) -> LizzardError {
        let e = th.exc.take().expect("raised without an exception");
        let kind = e.value.type_name();
        let msg = e.value.as_instance().and_then(|i| i.class().field_index("msg").map(|k| i.get_borrowed(k)));
        let msg = msg.map(crate::display::to_str).unwrap_or_default();
        LizzardError { kind, msg, pending: Some(PendingExc(e)) }
    }

    /// Raises the error in `th`.
    pub fn raise(mut self, th: &mut Thread) -> Raised {
        if let Some(p) = self.pending.take() {
            let p = std::mem::ManuallyDrop::new(p);
            // SAFETY: `p` is not used or dropped again.
            th.exc = Some(unsafe { std::ptr::read(&p.0) });
            return Raised;
        }
        let kind = if th.vm.builtins.has_class(&self.kind) { self.kind.as_str() } else { "RuntimeError" };
        th.raise(kind, std::mem::take(&mut self.msg))
    }
}

// ---------------------------------------------------------------- functions

/// A native function or method (an immortal object): an extension callback.
pub struct NativeFn {
    pub name: String,
    /// Parameter types (after `self` for a method).
    pub params: Vec<NType>,
    pub ret: NType,
    /// A method: the first argument is the object.
    pub has_self: bool,
    call: CallFn,
    data: *mut c_void,
    free: FreeFn,
}

// SAFETY: the extension's callback data is `Send + Sync` (the binding API
// requires it of the closures it wraps).
unsafe impl Send for NativeFn {}
unsafe impl Sync for NativeFn {}

impl Drop for NativeFn {
    fn drop(&mut self) {
        // SAFETY: the data was handed over with this free function.
        unsafe { (self.free)(self.data) }
    }
}

impl NativeFn {
    /// Takes ownership of an extension's function (its data is freed with
    /// it, also when the signature is malformed).
    ///
    /// # Safety
    /// `d` must be a valid descriptor.
    pub(crate) unsafe fn from_desc(d: &lizzard_abi::FnDesc) -> Result<NativeFn, String> {
        // SAFETY: valid for the call, by the caller's contract.
        let (name, desc) = unsafe { (d.name.as_str().to_string(), d.sig.as_bytes()) };
        let mut f = NativeFn { name, params: Vec::new(), ret: NType::Unit, has_self: d.has_self, call: d.call, data: d.data, free: d.free };
        let (params, ret) = decode_sig(desc).ok_or_else(|| format!("native `{}`: malformed signature", f.name))?;
        f.params = params;
        f.ret = ret;
        Ok(f)
    }

    /// For the JIT: the parameter and result types if all are scalars.
    pub fn scalar_sig(&self) -> Option<(&[NType], &NType)> {
        (!self.has_self && self.params.iter().all(NType::is_scalar) && self.ret.is_scalar()).then_some((&self.params, &self.ret))
    }
}

/// Calls a native with borrowed arguments (`self` first for a method).
pub fn call(th: &mut Thread, f: &NativeFn, args: &[Value]) -> VResult<Value> {
    let want = f.params.len() + f.has_self as usize;
    if args.len() != want {
        let n = f.params.len();
        return Err(th.raise(
            "TypeError",
            format!("{}() takes {n} argument{} but {} {} given", f.name, if n == 1 { "" } else { "s" }, args.len() - f.has_self as usize, if args.len() == 1 { "was" } else { "were" }),
        ));
    }
    if let Some(v) = args.iter().find(|v| v.is_dead()) {
        return Err(crate::helpers::dead_operand(th, *v).unwrap());
    }
    let mut out = Value::UNINIT.0;
    let ctx = th as *mut Thread as *mut c_void;
    // SAFETY: `args` are valid `Value`s (`repr(transparent)` over the raw
    // bits) for the call; the callback raises in `ctx` before failing.
    let ok = unsafe { (f.call)(f.data, ctx, args.as_ptr() as *const u64, args.len(), &mut out) };
    if ok {
        Ok(Value(out))
    } else {
        debug_assert!(th.exc.is_some(), "native {}() failed without raising", f.name);
        Err(Raised)
    }
}

// ---------------------------------------------------------------- classes

/// How a native class's methods may run (chosen from the Rust type's auto
/// traits at registration).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `Send + Sync`: shared calls run concurrently.
    Sync,
    /// `Send` only: every call is exclusive.
    Send,
    /// Not `Send`: local to the creating thread.
    Local,
}

/// An extension's formatter for a class's instances.
struct Repr {
    f: ReprFn,
    data: *mut c_void,
    free: Option<FreeFn>,
}

impl Drop for Repr {
    fn drop(&mut self) {
        if let Some(free) = self.free {
            // SAFETY: handed over with this free function.
            unsafe { free(self.data) }
        }
    }
}

/// A native class (an immortal object), or the family of a generic one.
pub struct NativeClass {
    pub name: String,
    pub module: String,
    /// Identifies the Rust type it wraps (0 for a family).
    pub key: u64,
    pub mode: Mode,
    /// Conflicting calls wait instead of raising `BorrowError`.
    pub locked: bool,
    pub init: Option<Value>,
    pub methods: HashMap<String, Value>,
    repr: Option<Repr>,
    /// Drops an instance's Rust value.
    drop: Option<FreeFn>,
    pub canonical: OnceLock<Value>,
    /// A generic family: its type parameters and the native classes of its
    /// supported instantiations (it has no init or methods of its own).
    pub tparams: Vec<String>,
    pub instances: Vec<Value>,
    /// An instantiation of a generic family: the family and the type
    /// arguments (its instances have type `family<args>`).
    pub family: OnceLock<Value>,
    pub type_args: Vec<NType>,
    /// Implemented traits of the module (`geom.Shape`), and builtin traits by
    /// name (`Ord`, `Num`, `Len`, with the traits they imply).
    pub traits: Vec<Value>,
    pub builtin_traits: Vec<&'static str>,
}

// SAFETY: the extension data (formatter, drop function) is `Send + Sync`.
unsafe impl Send for NativeClass {}
unsafe impl Sync for NativeClass {}

impl NativeClass {
    pub fn is_family(&self) -> bool {
        !self.tparams.is_empty()
    }

    /// The instantiation of this family for instance type `t` (`Stack<int>`).
    pub fn instance_for(&self, t: Value) -> Option<Value> {
        self.instances.iter().copied().find(|c| rtype::class_type(*c) == t)
    }

    /// Whether instances of this family with type arguments `args` are not
    /// `Send`. (Compares the arguments without interning the instance type,
    /// which this is part of computing.)
    pub fn local_for(&self, args: &[Value]) -> bool {
        self.instances.iter().any(|c| {
            let Some(Body::NativeClass(c)) = c.body() else { return false };
            c.mode == Mode::Local && c.type_args.len() == args.len() && c.type_args.iter().zip(args).all(|(n, a)| n.rt() == *a)
        })
    }

    /// The supported instantiations, as written (`Stack<int>, Stack<str>`).
    pub fn instance_names(&self) -> String {
        let names: Vec<String> = self.instances.iter().map(|c| format!("`{}`", rtype::name(rtype::class_type(*c)))).collect();
        names.join(", ")
    }

    /// Whether `obj` (an instance's class) is this class or one of its
    /// instantiations.
    pub fn covers(&self, this: Value, class: Value) -> bool {
        class == this || matches!(class.body(), Some(Body::NativeClass(c)) if c.family.get() == Some(&this))
    }
}

const EXCL: u32 = u32::MAX;

/// An instance of a native class.
pub struct NativeObj {
    pub class: Value,
    pub ty: Value,
    owner: std::thread::ThreadId,
    /// Number of running shared calls, or `EXCL` during an exclusive one.
    borrow: AtomicU32,
    /// Thread running the exclusive call (to detect re-entry).
    holder: AtomicU64,
    /// The extension's Rust value (null once released).
    data: UnsafeCell<*mut c_void>,
}

// SAFETY: access to the data follows the class's mode (see `enter`).
unsafe impl Send for NativeObj {}
unsafe impl Sync for NativeObj {}

fn thread_tag() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local!(static TAG: u64 = NEXT.fetch_add(1, Ordering::Relaxed));
    TAG.with(|t| *t)
}

impl NativeObj {
    pub fn class(&self) -> &'static NativeClass {
        match self.class.body() {
            Some(Body::NativeClass(c)) => c,
            _ => unreachable!(),
        }
    }

    /// Starts a call: returns the data and whether the access is exclusive
    /// (always for a `Send` class). End it with [`NativeObj::leave`].
    pub(crate) fn enter(&self, exclusive: bool) -> Result<(*mut c_void, bool), LizzardError> {
        let c = self.class();
        if c.mode == Mode::Local && std::thread::current().id() != self.owner {
            return Err(LizzardError::new("ThreadError", format!("`{}` object belongs to another thread (it is not `Send`)", c.name)));
        }
        let exclusive = exclusive || c.mode == Mode::Send;
        let me = thread_tag();
        loop {
            let cur = self.borrow.load(Ordering::Acquire);
            let free = if exclusive { cur == 0 } else { cur != EXCL };
            if free {
                let new = if exclusive { EXCL } else { cur + 1 };
                if self.borrow.compare_exchange_weak(cur, new, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                    if exclusive {
                        self.holder.store(me, Ordering::Relaxed);
                    }
                    // SAFETY: the data pointer is only replaced on release,
                    // when the object is unreachable.
                    return Ok((unsafe { *self.data.get() }, exclusive));
                }
                continue;
            }
            let reentrant = cur == EXCL && self.holder.load(Ordering::Relaxed) == me;
            if !c.locked || reentrant || c.mode == Mode::Local {
                let what = if cur == EXCL { "a call that needs it exclusively" } else { "another call" };
                return Err(LizzardError::new("BorrowError", format!("`{}` object is in use by {what}", c.name)));
            }
            // Wait without blocking memory reclamation on other threads.
            crate::rc::offline();
            std::thread::yield_now();
            crate::rc::online();
        }
    }

    /// Ends a call started by [`NativeObj::enter`].
    pub(crate) fn leave(&self, exclusive: bool) {
        if exclusive {
            self.holder.store(0, Ordering::Relaxed);
            self.borrow.store(0, Ordering::Release);
        } else {
            self.borrow.fetch_sub(1, Ordering::Release);
        }
    }

    /// Drops the Rust value (when the object is freed). A local object is
    /// only dropped on its owner thread: locality guarantees it is freed
    /// there, so anything else is a VM bug. Debug builds panic; release
    /// builds leak the value with a warning (dropping it would be unsound).
    pub(crate) fn release(&mut self) {
        let data = std::mem::replace(self.data.get_mut(), std::ptr::null_mut());
        if data.is_null() {
            return;
        }
        let c = self.class();
        if c.mode == Mode::Local && std::thread::current().id() != self.owner {
            let name = &c.name;
            if cfg!(debug_assertions) {
                panic!("native `{name}` object freed off its owner thread");
            }
            eprintln!("warning: native `{name}` object freed off its owner thread; leaked");
            return;
        }
        if let Some(drop) = c.drop {
            // SAFETY: the data was made for this class, whose drop function
            // frees it; it is not used again.
            unsafe { drop(data) }
        }
    }

    pub(crate) fn display(&self) -> String {
        let c = self.class();
        if let Some(r) = &c.repr {
            if let Ok((data, excl)) = self.enter(false) {
                let mut s = String::new();
                // SAFETY: a borrow is held on the data, and `s` outlives the call.
                let ok = unsafe { (r.f)(r.data, data, lizzard_abi::string_out(&mut s)) };
                self.leave(excl);
                if ok {
                    return s;
                }
            }
        }
        format!("<{} object>", rtype::name(self.ty))
    }
}

pub(crate) fn native_obj(v: Value) -> Option<&'static NativeObj> {
    match v.body() {
        Some(Body::Native(n)) => Some(n),
        _ => None,
    }
}

static REGISTRY: OnceLock<RwLock<HashMap<u64, Value>>> = OnceLock::new();
static NEXT_KEY: AtomicU64 = AtomicU64::new(1);

fn registry() -> &'static RwLock<HashMap<u64, Value>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// A new class key, unique in the process.
pub(crate) fn new_key() -> u64 {
    NEXT_KEY.fetch_add(1, Ordering::Relaxed)
}

/// The class last registered for key `key`.
pub fn class_by_key(key: u64) -> Option<Value> {
    registry().read().unwrap().get(&key).copied()
}

/// A new object of the class registered for `key`, owning `data`.
pub(crate) fn new_object(key: u64, data: *mut c_void) -> Option<Value> {
    let class = class_by_key(key)?;
    let obj = NativeObj {
        class,
        ty: rtype::class_type(class),
        owner: std::thread::current().id(),
        borrow: AtomicU32::new(0),
        holder: AtomicU64::new(0),
        data: UnsafeCell::new(data),
    };
    Some(Obj::alloc(Body::Native(Box::new(obj))))
}

// ---------------------------------------------------------------- modules

/// A native function's signature, for the compiler.
#[derive(Clone, Debug)]
pub struct FnSpec {
    pub name: String,
    pub params: Vec<NType>,
    pub ret: NType,
}

/// A native class, for the compiler.
#[derive(Clone, Debug)]
pub struct ClassSpec {
    pub name: String,
    /// The class key ([`NType::Class`]).
    pub key: u64,
    pub local: bool,
    pub init: Option<FnSpec>,
    pub methods: Vec<FnSpec>,
    /// A generic family: its type parameters and instantiations (type
    /// arguments, class).
    pub tparams: Vec<String>,
    pub instances: Vec<(Vec<NType>, ClassSpec)>,
    /// Implemented traits: the module's (`Shape`) and builtin ones (`Len`).
    pub traits: Vec<String>,
}

/// A trait defined by a native module, for the compiler. Every method is
/// required (`self` is not among the parameters).
#[derive(Clone, Debug)]
pub struct TraitSpec {
    pub name: String,
    pub methods: Vec<FnSpec>,
}

/// A native module's items, for the compiler.
#[derive(Clone, Debug, Default)]
pub struct ModuleSpec {
    pub name: String,
    pub functions: Vec<FnSpec>,
    pub classes: Vec<ClassSpec>,
    pub traits: Vec<TraitSpec>,
}

/// A module of native functions, classes and traits.
pub struct Module {
    pub name: String,
    functions: Vec<(String, Value)>,
    classes: Vec<(String, Value)>,
    traits: Traits,
}

/// A module's traits: signature and trait object.
type Traits = Vec<(TraitSpec, Value)>;

/// Builtin traits a native class may implement: (name, implied traits,
/// required methods).
const BUILTIN_TRAITS: &[(&str, &[&str], &[&str])] = &[
    ("Ord", &[], &["__lt__"]),
    ("Num", &["Ord"], &["__lt__", "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__", "__neg__"]),
    ("Len", &[], &["__len__"]),
];

/// Ids of trait objects defined by native modules (apart from the VM's).
static NEXT_TRAIT_ID: AtomicU32 = AtomicU32::new(1 << 30);

fn fn_spec(name: &str, v: Value) -> FnSpec {
    let Some(Body::NativeFn(f)) = v.body() else { unreachable!() };
    FnSpec { name: name.into(), params: f.params.clone(), ret: f.ret.clone() }
}

/// A native class's parts, before it is created.
pub(crate) struct ClassParts {
    pub name: String,
    pub key: u64,
    pub mode: Mode,
    pub locked: bool,
    pub type_args: Vec<NType>,
    pub drop: FreeFn,
    pub init: Option<NativeFn>,
    pub methods: Vec<NativeFn>,
    repr: Option<Repr>,
    pub implements: Vec<String>,
}

impl ClassParts {
    /// Takes ownership of an extension's class descriptor (all of its
    /// callback data, also on error).
    ///
    /// # Safety
    /// `d` must be a valid descriptor.
    pub(crate) unsafe fn from_desc(d: &lizzard_abi::ClassDesc) -> Result<ClassParts, String> {
        // SAFETY (whole function): the pointers are valid for the call, by
        // the caller's contract.
        let init = unsafe { d.init.as_ref() }.map(|f| unsafe { NativeFn::from_desc(f) });
        let methods: Vec<_> = unsafe { slice(d.methods, d.nmethods) }.iter().map(|f| unsafe { NativeFn::from_desc(f) }).collect();
        let repr = d.repr.map(|f| Repr { f, data: d.repr_data, free: d.repr_free });
        let name = unsafe { d.name.as_str() }.to_string();
        let implements = unsafe { slice(d.implements, d.nimplements) }.iter().map(|s| unsafe { s.as_str() }.to_string()).collect();
        let type_args = NType::decode_list(&mut unsafe { d.type_args.as_bytes() }).unwrap_or_default();
        let mode = match d.mode {
            lizzard_abi::MODE_SYNC => Mode::Sync,
            lizzard_abi::MODE_SEND => Mode::Send,
            lizzard_abi::MODE_LOCAL => Mode::Local,
            m => return Err(format!("native class `{name}`: unknown mode {m}")),
        };
        Ok(ClassParts {
            init: init.transpose()?,
            methods: methods.into_iter().collect::<Result<_, _>>()?,
            name,
            key: d.key,
            mode,
            locked: d.locked,
            type_args,
            drop: d.drop,
            repr,
            implements,
        })
    }
}

/// A slice from an ABI pointer and length.
///
/// # Safety
/// `p` must point to `n` valid elements (or `n` is 0).
pub(crate) unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        return &[];
    }
    // SAFETY: guaranteed by the caller.
    unsafe { std::slice::from_raw_parts(p, n) }
}

impl Module {
    pub fn new(name: &str) -> Module {
        Module { name: name.into(), functions: Vec::new(), classes: Vec::new(), traits: Vec::new() }
    }

    pub(crate) fn add_function(&mut self, f: NativeFn) {
        self.functions.push((f.name.clone(), Obj::alloc_immortal(Body::NativeFn(Box::new(f)))));
    }

    /// Adds a trait whose methods are all required, for classes of this
    /// module (defined after it) and lizzard structs to implement.
    pub(crate) fn add_trait(&mut self, name: &str, methods: Vec<FnSpec>) -> Result<(), String> {
        if self.traits.iter().any(|(t, _)| t.name == name) {
            return Err(format!("module `{}` defines trait `{name}` twice", self.name));
        }
        let tr = Obj::alloc_immortal(Body::Trait(Box::new(crate::object::Trait {
            name: name.into(),
            id: NEXT_TRAIT_ID.fetch_add(1, Ordering::Relaxed),
            nparams: 0,
            methods: methods.iter().map(|m| (m.name.clone(), None)).collect(),
            implies: Vec::new(),
            impls: Default::default(),
        })));
        self.traits.push((TraitSpec { name: name.into(), methods }, tr));
        Ok(())
    }

    pub(crate) fn add_class(&mut self, parts: ClassParts) -> Result<(), String> {
        let name = parts.name.clone();
        let class = new_class(&self.name, &self.traits, parts)?;
        self.classes.push((name, class));
        Ok(())
    }

    /// Adds a generic class `name<tparams>` with the given instantiations.
    pub(crate) fn add_generic(&mut self, name: &str, tparams: Vec<String>, instances: Vec<ClassParts>) -> Result<(), String> {
        if tparams.is_empty() {
            return Err(format!("generic class `{name}` needs type parameters"));
        }
        let mut seen: Vec<&[NType]> = Vec::new();
        for p in &instances {
            if p.type_args.len() != tparams.len() {
                return Err(format!("`{name}` takes {} type argument(s), got {}", tparams.len(), p.type_args.len()));
            }
            if seen.contains(&p.type_args.as_slice()) {
                return Err(format!("`{name}` instantiation {:?} is registered twice", p.type_args));
            }
            seen.push(&p.type_args);
        }
        let mut classes = Vec::new();
        for p in instances {
            classes.push(new_class(&self.name, &self.traits, p)?);
        }
        let family = Obj::alloc_immortal(Body::NativeClass(Box::new(NativeClass {
            name: name.into(),
            module: self.name.clone(),
            key: 0,
            mode: Mode::Sync,
            locked: false,
            init: None,
            methods: HashMap::new(),
            repr: None,
            drop: None,
            canonical: OnceLock::new(),
            tparams,
            instances: classes.clone(),
            family: OnceLock::new(),
            type_args: Vec::new(),
            traits: Vec::new(),
            builtin_traits: Vec::new(),
        })));
        for c in &classes {
            let Some(Body::NativeClass(c)) = c.body() else { unreachable!() };
            let _ = c.family.set(family);
        }
        self.classes.push((name.into(), family));
        Ok(())
    }

    /// The module's items as the compiler sees them.
    pub fn spec(&self) -> ModuleSpec {
        ModuleSpec {
            name: self.name.clone(),
            functions: self.functions.iter().map(|(n, v)| fn_spec(n, *v)).collect(),
            classes: self.classes.iter().map(|(n, v)| class_spec(n, *v)).collect(),
            traits: self.traits.iter().map(|(t, _)| t.clone()).collect(),
        }
    }

    /// The module's globals: (`module.item`, value).
    pub fn globals(&self) -> Vec<(String, Value)> {
        let traits = self.traits.iter().map(|(t, v)| (t.name.clone(), *v));
        self.functions.iter().chain(&self.classes).cloned().chain(traits).map(|(n, v)| (format!("{}.{n}", self.name), v)).collect()
    }
}

fn class_spec(name: &str, v: Value) -> ClassSpec {
    let Some(Body::NativeClass(c)) = v.body() else { unreachable!() };
    let mut methods: Vec<FnSpec> = c.methods.iter().map(|(m, f)| fn_spec(m, *f)).collect();
    methods.sort_by(|a, b| a.name.cmp(&b.name));
    ClassSpec {
        name: name.into(),
        key: c.key,
        local: c.mode == Mode::Local,
        init: c.init.map(|f| fn_spec("__init__", f)),
        methods,
        tparams: c.tparams.clone(),
        instances: c
            .instances
            .iter()
            .map(|i| {
                let Some(Body::NativeClass(ic)) = i.body() else { unreachable!() };
                (ic.type_args.clone(), class_spec(name, *i))
            })
            .collect(),
        traits: c
            .traits
            .iter()
            .map(|t| t.as_trait().unwrap().name.clone())
            .chain(c.builtin_traits.iter().map(|t| t.to_string()))
            .collect(),
    }
}

/// The traits named by `c.implements(...)`: the module's trait objects and
/// builtin trait names. An error if one is unknown or a method is missing
/// or has another signature.
fn class_traits(class: &str, names: &[String], traits: &Traits, methods: &HashMap<String, NativeFn>) -> Result<(Vec<Value>, Vec<&'static str>), String> {
    let (mut objs, mut builtins) = (Vec::new(), Vec::new());
    for n in names {
        if let Some((spec, v)) = traits.iter().find(|(t, _)| t.name == *n) {
            for m in &spec.methods {
                let Some(f) = methods.get(&m.name) else { return Err(format!("native class `{class}` implements `{n}` but does not define `{}`", m.name)) };
                if f.params != m.params || f.ret != m.ret {
                    return Err(format!(
                        "native class `{class}` implements `{n}`, but `{}` takes {:?} -> {:?} instead of {:?} -> {:?}",
                        m.name, f.params, f.ret, m.params, m.ret
                    ));
                }
            }
            objs.push(*v);
        } else if let Some((b, implies, required)) = BUILTIN_TRAITS.iter().find(|t| t.0 == n) {
            if let Some(m) = required.iter().find(|m| !methods.contains_key(**m)) {
                return Err(format!("native class `{class}` implements `{n}` but does not define `{m}`"));
            }
            for t in std::iter::once(b).chain(implies.iter()) {
                if !builtins.contains(t) {
                    builtins.push(*t);
                }
            }
        } else if n == "Send" {
            return Err(format!(
                "native class `{class}`: `Send` is not implemented explicitly; it follows the Rust type (`class` / `class_send` vs `class_not_send`)"
            ));
        } else {
            return Err(format!("native class `{class}` implements unknown trait `{n}` (define it with `trait_def` before the class, or use `Ord`, `Num`, `Len`)"));
        }
    }
    Ok((objs, builtins))
}

/// A new native class (registered as its key's class).
fn new_class(module: &str, traits: &Traits, p: ClassParts) -> Result<Value, String> {
    let mut methods = HashMap::new();
    for mut m in p.methods {
        // Shown as `Class.method`; a later definition replaces an earlier one.
        let short = std::mem::take(&mut m.name);
        m.name = format!("{}.{short}", p.name);
        methods.insert(short, m);
    }
    let (traits, builtin_traits) = class_traits(&p.name, &p.implements, traits, &methods)?;
    let class = Obj::alloc_immortal(Body::NativeClass(Box::new(NativeClass {
        name: p.name,
        module: module.into(),
        key: p.key,
        mode: p.mode,
        locked: p.locked,
        init: p.init.map(|f| Obj::alloc_immortal(Body::NativeFn(Box::new(f)))),
        methods: methods.into_iter().map(|(n, f)| (n, Obj::alloc_immortal(Body::NativeFn(Box::new(f))))).collect(),
        repr: p.repr,
        drop: Some(p.drop),
        canonical: OnceLock::new(),
        tparams: Vec::new(),
        instances: Vec::new(),
        family: OnceLock::new(),
        type_args: p.type_args,
        traits,
        builtin_traits,
    })));
    registry().write().unwrap().insert(p.key, class);
    Ok(class)
}
