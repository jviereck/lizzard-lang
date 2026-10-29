//! Rust bindings for lizzard: native functions, classes and traits.
//!
//! The same code serves both ways of providing a module:
//! - statically, linked into a host program (`lizzard::Runtime::module`);
//! - as an extension: a `cdylib` declared with [`module!`], which the
//!   interpreter loads at `import` (like a Python extension module).
//!
//! This crate does not link the interpreter: it reaches it only through
//! the C ABI of [`lizzard_abi`], so an extension works with any interpreter of
//! the same ABI version, whatever compiler built either.
//!
//! ```ignore
//! // Cargo.toml: [lib] name = "geom", crate-type = ["cdylib"]
//! lizzard_ext::module!(geom, |m| {
//!     m.function("dot", |a: ListRef<f64>, b: ListRef<f64>| -> f64 { ... });
//!     m.class::<Polygon>("Polygon", |c| { ... });
//! });
//! ```
//!
//! Values cross the boundary through [`FromLizzard`] (arguments, always
//! borrowed from the caller) and [`IntoLizzard`] (results). Handles such as
//! [`ListRef`] or [`Native`] hold a counted reference, so a native may keep
//! one; they are `!Send` and stay on the interpreter thread that created
//! them.
//!
//! A native class wraps a Rust value. The registration function decides how
//! its methods may run, from the Rust type's auto traits:
//! - [`Module::class`] (`T: Send + Sync`): `&self` methods run concurrently
//!   from any thread; `&mut self` methods are exclusive.
//! - [`Module::class_send`] (`T: Send`): every method call is exclusive.
//! - [`Module::class_not_send`] (any `T`): the Lizzard class is not `Send`
//!   either, so instances stay on the thread that created them, which is
//!   also checked on every call (`ThreadError`), and are dropped there.
//!
//! A call that conflicts with a running one (a `&mut self` method while
//! another method runs) raises `BorrowError`, or waits for it to finish if
//! the class is [`ClassBuilder::locked`].
//!
//! A generic class ([`Module::generic_class`]) lists the instantiations it
//! supports, each an ordinary native class over its own Rust type
//! (`Stack<i64>`, `Stack<String>`). Lizzard sees one generic class `Stack<T>`;
//! constructing `Stack<int>` dispatches to the Rust class registered for
//! `int`, and other type arguments raise `TypeError`.
//!
//! A module may define traits ([`Module::trait_def`]) that its classes
//! implement ([`ClassBuilder::implements`]), as may Lizzard structs. A class
//! may also implement the builtin `Ord`, `Num` and `Len` through its dunder
//! methods.

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub use lizzard_abi as abi;
use lizzard_abi::{sig, string_out, HostErr, RawValue, LizzardApi, Str, ABI_VERSION};

// ---------------------------------------------------------------- the interpreter

static API: OnceLock<&'static LizzardApi> = OnceLock::new();

/// Connects this crate to an interpreter. Called by the entry point of
/// [`module!`], and by a host program before it builds static modules.
pub fn install(api: &'static LizzardApi) -> Result<(), String> {
    if api.version != ABI_VERSION || api.size < std::mem::size_of::<LizzardApi>() {
        return Err(format!("the interpreter has Lizzard ABI version {}, this module was built for version {ABI_VERSION}", api.version));
    }
    let cur = API.get_or_init(|| api);
    if !std::ptr::eq(*cur, api) {
        return Err("a different Lizzard interpreter is already connected".into());
    }
    Ok(())
}

fn api() -> &'static LizzardApi {
    API.get().expect("lizzard-ext used before an interpreter was installed")
}

/// Reads a string the interpreter writes.
fn read(f: impl FnOnce(abi::StrOut)) -> String {
    let mut s = String::new();
    f(string_out(&mut s));
    s
}

/// A Lizzard value, borrowed for the duration of a native call (see
/// [`FromLizzard`]).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Value(RawValue);

impl Value {
    fn is_none(self) -> bool {
        self.0 == api().none
    }

    // Scalars are read and made inline, from the interpreter's
    // `ValueLayout` (like CPython's macros); the rest goes through calls.

    fn is_ptr(self, l: &abi::ValueLayout) -> bool {
        self.0 & l.ptr_mask == 0 && self.0 != 0
    }

    /// The payload of a boxed scalar of kind `kind`, at `offset`.
    fn boxed<T: Copy>(self, l: &abi::ValueLayout, kind: u64, offset: usize) -> Option<T> {
        if !self.is_ptr(l) {
            return None;
        }
        // SAFETY: a live object's first word is its (atomic) header, and a
        // boxed scalar's payload is immutable (see `ValueLayout`).
        unsafe {
            let header = (*(self.0 as *const AtomicU64)).load(Ordering::Relaxed);
            (header & l.kind_mask == kind).then(|| *((self.0 as usize + offset) as *const T))
        }
    }

    fn as_int(self) -> Option<i64> {
        let l = &api().layout;
        if self.0 & l.small_int_mask == l.small_int_tag {
            return Some((self.0 as i64) >> l.small_int_shift);
        }
        self.boxed(l, l.kind_int, l.int_offset)
    }

    fn as_float(self) -> Option<f64> {
        let l = &api().layout;
        self.boxed(l, l.kind_float, l.float_offset)
    }

    fn as_bool(self) -> Option<bool> {
        let a = api();
        match self.0 {
            v if v == a.true_ => Some(true),
            v if v == a.false_ => Some(false),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        let mut s = Str { ptr: std::ptr::null(), len: 0 };
        // SAFETY: the string lives as long as the (borrowed) value.
        unsafe { (api().as_str)(self.0, &mut s).then(|| s.as_str()) }
    }

    fn type_name(self) -> String {
        read(|o| unsafe { (api().type_name)(self.0, o) })
    }

    fn int(i: i64) -> Value {
        let a = api();
        let l = &a.layout;
        if (l.small_int_min..=l.small_int_max).contains(&i) {
            return Value(((i << l.small_int_shift) as u64) | l.small_int_tag);
        }
        Value(unsafe { (a.int_new)(i) })
    }

    fn float(f: f64) -> Value {
        Value(unsafe { (api().float_new)(f) })
    }

    fn bool(b: bool) -> Value {
        let a = api();
        Value(if b { a.true_ } else { a.false_ })
    }

    fn str(s: &str) -> Value {
        Value(unsafe { (api().str_new)(Str::new(s)) })
    }

    fn incref(self) {
        let a = api();
        if self.is_ptr(&a.layout) {
            unsafe { (a.incref)(self.0) }
        }
    }

    fn decref(self) {
        let a = api();
        if self.is_ptr(&a.layout) {
            unsafe { (a.decref)(self.0) }
        }
    }
}

/// Writes up to `cap` borrowed values; the interpreter returns their number.
fn snapshot(f: impl Fn(*mut RawValue, usize) -> usize) -> Vec<Value> {
    // Room for small containers, which then take a single call.
    let mut buf: Vec<RawValue> = Vec::with_capacity(16);
    loop {
        let n = f(buf.as_mut_ptr(), buf.capacity());
        if n <= buf.capacity() {
            // SAFETY: the interpreter wrote `n` values.
            unsafe { buf.set_len(n) };
            return buf.into_iter().map(Value).collect();
        }
        buf.reserve(n);
    }
}

// ---------------------------------------------------------------- types

/// An element type of `array<T>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
}

impl Elem {
    fn code(self) -> u8 {
        match self {
            Elem::I8 => sig::I8,
            Elem::I16 => sig::I16,
            Elem::I32 => sig::I32,
            Elem::I64 => sig::I64,
            Elem::U8 => sig::U8,
            Elem::U16 => sig::U16,
            Elem::U32 => sig::U32,
            Elem::U64 => sig::U64,
            Elem::F32 => sig::F32,
            Elem::F64 => sig::F64,
            Elem::Bool => sig::ELEM_BOOL,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Elem::F32 | Elem::F64)
    }
}

/// A Lizzard type, as the compiler sees a native's parameters and result.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// A native class, by the Rust type it wraps.
    Class(TypeId),
}

/// The interpreter's class keys of the Rust types of native classes.
fn class_key(id: TypeId) -> u64 {
    static KEYS: OnceLock<Mutex<HashMap<TypeId, u64>>> = OnceLock::new();
    thread_local!(static CACHE: RefCell<HashMap<TypeId, u64>> = RefCell::default());
    if let Some(k) = CACHE.with(|c| c.borrow().get(&id).copied()) {
        return k;
    }
    let k = *KEYS.get_or_init(Default::default).lock().unwrap().entry(id).or_insert_with(|| unsafe { (api().class_key_new)() });
    CACHE.with(|c| c.borrow_mut().insert(id, k));
    k
}

impl NType {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            NType::Int => out.push(sig::INT),
            NType::Float => out.push(sig::FLOAT),
            NType::Bool => out.push(sig::BOOL),
            NType::Str => out.push(sig::STR),
            NType::Unit => out.push(sig::UNIT),
            NType::Any => out.push(sig::ANY),
            NType::Function => out.push(sig::FUNCTION),
            NType::List(t) => {
                out.push(sig::LIST);
                t.encode(out);
            }
            NType::Dict(k, v) => {
                out.push(sig::DICT);
                k.encode(out);
                v.encode(out);
            }
            NType::Option(t) => {
                out.push(sig::OPTION);
                t.encode(out);
            }
            NType::Array(e) => out.extend([sig::ARRAY, e.code()]),
            NType::Class(id) => {
                out.push(sig::CLASS);
                out.extend(class_key(*id).to_le_bytes());
            }
        }
    }

    /// The runtime type of values of this type. Types without classes are
    /// cached (interned types live forever; a class may be registered again).
    fn rt(&self) -> RawValue {
        thread_local!(static CACHE: RefCell<HashMap<NType, RawValue>> = RefCell::default());
        let cached = !self.has_class();
        if cached {
            if let Some(t) = CACHE.with(|c| c.borrow().get(self).copied()) {
                return t;
            }
        }
        let mut d = Vec::new();
        self.encode(&mut d);
        let t = unsafe { (api().type_from_desc)(Str { ptr: d.as_ptr(), len: d.len() }) };
        if cached {
            CACHE.with(|c| c.borrow_mut().insert(self.clone(), t));
        }
        t
    }

    fn has_class(&self) -> bool {
        match self {
            NType::Class(_) => true,
            NType::List(t) | NType::Option(t) => t.has_class(),
            NType::Dict(k, v) => k.has_class() || v.has_class(),
            _ => false,
        }
    }

    fn name(&self) -> String {
        let t = self.rt();
        read(|o| unsafe { (api().type_display)(t, o) })
    }
}

/// Encodes a type list.
fn encode_list(ts: &[NType], out: &mut Vec<u8>) {
    out.push(u8::try_from(ts.len()).expect("too many parameters"));
    ts.iter().for_each(|t| t.encode(out));
}

fn encode_sig(params: &[NType], ret: &NType) -> Vec<u8> {
    let mut d = Vec::new();
    encode_list(params, &mut d);
    ret.encode(&mut d);
    d
}

/// A Rust type with a Lizzard counterpart.
pub trait LizzardType {
    fn ntype() -> NType;
}

/// Conversion of a (borrowed) Lizzard argument.
pub trait FromLizzard: LizzardType + Sized {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError>;
}

/// Conversion of a native result into an owned Lizzard value.
pub trait IntoLizzard {
    fn ntype() -> NType;
    fn into_lizzard(self) -> Result<Value, LizzardError>;
}

fn expected(t: NType, v: Value) -> LizzardError {
    LizzardError::type_error(format!("expected `{}`, got `{}`", t.name(), v.type_name()))
}

macro_rules! int_conv {
    ($($t:ty),*) => {$(
        impl LizzardType for $t {
            fn ntype() -> NType {
                NType::Int
            }
        }
        impl FromLizzard for $t {
            fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
                let i = v.as_int().ok_or_else(|| expected(NType::Int, v))?;
                <$t>::try_from(i).map_err(|_| LizzardError::new("OverflowError", format!("{i} does not fit in {}", stringify!($t))))
            }
        }
        impl IntoLizzard for $t {
            fn ntype() -> NType {
                NType::Int
            }
            fn into_lizzard(self) -> Result<Value, LizzardError> {
                i64::try_from(self)
                    .map(Value::int)
                    .map_err(|_| LizzardError::new("OverflowError", format!("{self} does not fit in an int")))
            }
        }
    )*};
}
int_conv!(i8, i16, i32, i64, u8, u16, u32, u64, isize, usize);

macro_rules! simple_conv {
    ($t:ty, $nt:expr, |$v:ident| $from:expr, |$s:ident| $into:expr) => {
        impl LizzardType for $t {
            fn ntype() -> NType {
                $nt
            }
        }
        impl FromLizzard for $t {
            fn from_lizzard($v: Value) -> Result<Self, LizzardError> {
                $from.ok_or_else(|| expected($nt, $v))
            }
        }
        impl IntoLizzard for $t {
            fn ntype() -> NType {
                $nt
            }
            fn into_lizzard(self) -> Result<Value, LizzardError> {
                let $s = self;
                Ok($into)
            }
        }
    };
}
simple_conv!(f64, NType::Float, |v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)), |s| Value::float(s));
simple_conv!(f32, NType::Float, |v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)).map(|f| f as f32), |s| Value::float(s as f64));
simple_conv!(bool, NType::Bool, |v| v.as_bool(), |s| Value::bool(s));
simple_conv!(String, NType::Str, |v| v.as_str().map(String::from), |s| Value::str(&s));

impl IntoLizzard for &str {
    fn ntype() -> NType {
        NType::Str
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(Value::str(self))
    }
}

impl IntoLizzard for () {
    fn ntype() -> NType {
        NType::Unit
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(Value(api().unit))
    }
}

impl<T: LizzardType> LizzardType for Option<T> {
    fn ntype() -> NType {
        NType::Option(Box::new(T::ntype()))
    }
}

impl<T: FromLizzard> FromLizzard for Option<T> {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        if v.is_none() {
            return Ok(None);
        }
        T::from_lizzard(v).map(Some)
    }
}

impl<T: IntoLizzard> IntoLizzard for Option<T> {
    fn ntype() -> NType {
        NType::Option(Box::new(T::ntype()))
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        match self {
            Some(t) => t.into_lizzard(),
            None => Ok(Value(api().none)),
        }
    }
}

impl<T: IntoLizzard> IntoLizzard for Result<T, LizzardError> {
    fn ntype() -> NType {
        T::ntype()
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        self?.into_lizzard()
    }
}

/// A hasher for `TypeId`s, which are hashes already.
#[derive(Default)]
struct IdHasher(u64);

impl std::hash::Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = self.0.rotate_left(8) ^ *b as u64;
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 ^= n;
    }
}

/// The runtime type of the Lizzard type of Rust type `id` (`ntype` gives it),
/// cached per thread so that conversions neither rebuild nor hash the
/// `NType`. Types with classes are not cached (a class may be registered
/// again).
fn rt_of(id: TypeId, ntype: fn() -> NType) -> RawValue {
    type Cache = HashMap<TypeId, Option<RawValue>, std::hash::BuildHasherDefault<IdHasher>>;
    thread_local!(static CACHE: RefCell<Cache> = RefCell::default());
    if let Some(Some(t)) = CACHE.with(|c| c.borrow().get(&id).copied()) {
        return t;
    }
    let nt = ntype();
    let t = nt.rt();
    CACHE.with(|c| c.borrow_mut().insert(id, (!nt.has_class()).then_some(t)));
    t
}

/// The runtime type of `T`'s Lizzard type.
fn rt<T: LizzardType + 'static>() -> RawValue {
    rt_of(TypeId::of::<T>(), T::ntype)
}

/// The value of a list or dict, checked against the exact runtime type of `T`.
fn typed_container<T: LizzardType + 'static>(v: Value) -> Result<(), LizzardError> {
    if unsafe { (api().has_type)(v.0, rt::<T>()) } {
        Ok(())
    } else {
        Err(expected(T::ntype(), v))
    }
}

fn list_snapshot(v: Value) -> Vec<Value> {
    snapshot(|out, cap| unsafe { (api().list_snapshot)(v.0, out, cap) })
}

impl<T: LizzardType> LizzardType for Vec<T> {
    fn ntype() -> NType {
        NType::List(Box::new(T::ntype()))
    }
}

impl<T: FromLizzard + 'static> FromLizzard for Vec<T> {
    /// Copies a `list<T>` (O(n)); `ListRef<T>` reads it in place.
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        typed_container::<Self>(v)?;
        list_snapshot(v).into_iter().map(T::from_lizzard).collect()
    }
}

/// A new list of runtime type `ty` taking `items`.
fn new_list(ty: RawValue, items: Vec<Value>) -> Result<Value, LizzardError> {
    let mut out = 0;
    let e = unsafe { (api().list_new)(ty, items.as_ptr() as *const RawValue, items.len(), &mut out) };
    LizzardError::check(e)?;
    Ok(Value(out))
}

impl<T: IntoLizzard + 'static> IntoLizzard for Vec<T> {
    fn ntype() -> NType {
        NType::List(Box::new(T::ntype()))
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        let mut items = Vec::with_capacity(self.len());
        for x in self {
            match x.into_lizzard() {
                Ok(v) => items.push(v),
                Err(e) => {
                    items.into_iter().for_each(Value::decref);
                    return Err(e);
                }
            }
        }
        new_list(rt_of(TypeId::of::<Self>(), <Self as IntoLizzard>::ntype), items)
    }
}

// ---------------------------------------------------------------- handles

/// A counted reference to a Lizzard value, valid on the thread that got it.
struct Handle {
    v: Value,
    _not_send: PhantomData<*const ()>,
}

impl Handle {
    fn borrow(v: Value) -> Handle {
        v.incref();
        Handle { v, _not_send: PhantomData }
    }

    fn owned(v: Value) -> Handle {
        Handle { v, _not_send: PhantomData }
    }

    fn into_value(self) -> Value {
        let v = self.v;
        std::mem::forget(self);
        v
    }
}

impl Clone for Handle {
    fn clone(&self) -> Handle {
        Handle::borrow(self.v)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.v.decref();
    }
}

/// Any Lizzard value except `None` (use `Option<Any>` to accept `None`).
#[derive(Clone)]
pub struct Any(Handle);

impl Any {
    pub fn type_name(&self) -> String {
        self.0.v.type_name()
    }
    pub fn as_int(&self) -> Option<i64> {
        self.0.v.as_int()
    }
    pub fn as_float(&self) -> Option<f64> {
        self.0.v.as_float()
    }
    pub fn as_str(&self) -> Option<String> {
        self.0.v.as_str().map(String::from)
    }
    /// The value as `str(x)` shows it.
    pub fn to_display(&self) -> String {
        read(|o| unsafe { (api().display)(self.0.v.0, o) })
    }
}

impl LizzardType for Any {
    fn ntype() -> NType {
        NType::Any
    }
}

impl FromLizzard for Any {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        if v.is_none() || unsafe { (api().is_dead)(v.0) } {
            return Err(expected(NType::Any, v));
        }
        Ok(Any(Handle::borrow(v)))
    }
}

impl IntoLizzard for Any {
    fn ntype() -> NType {
        NType::Any
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(self.0.into_value())
    }
}

/// A Lizzard function (or other callable), called with [`Ctx::call`].
#[derive(Clone)]
pub struct Func(Handle);

impl LizzardType for Func {
    fn ntype() -> NType {
        NType::Function
    }
}

impl FromLizzard for Func {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        // A local closure is fine: the handle is `!Send`, so it stays on
        // this thread (the parameter is typed `function`, which may hold
        // values that are not `Send`).
        if !unsafe { (api().is_function)(v.0) } {
            return Err(expected(NType::Function, v));
        }
        Ok(Func(Handle::borrow(v)))
    }
}

impl IntoLizzard for Func {
    fn ntype() -> NType {
        NType::Function
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(self.0.into_value())
    }
}

/// A `list<T>` read and written in place. Elements are converted one at a
/// time; stores are checked like Lizzard stores (types and locality).
pub struct ListRef<T> {
    h: Handle,
    _t: PhantomData<T>,
}

impl<T> Clone for ListRef<T> {
    fn clone(&self) -> Self {
        ListRef { h: self.h.clone(), _t: PhantomData }
    }
}

impl<T: FromLizzard + IntoLizzard + 'static> ListRef<T> {
    /// A new, empty `list<T>`.
    pub fn new() -> ListRef<T> {
        let v = new_list(rt::<Self>(), Vec::new()).expect("empty list");
        ListRef { h: Handle::owned(v), _t: PhantomData }
    }

    pub fn len(&self) -> usize {
        unsafe { (api().list_len)(self.h.v.0) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element `i`, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<T> {
        // The element is borrowed only while it is converted.
        let mut out = 0;
        unsafe { (api().list_get)(self.h.v.0, i, &mut out) }.then(|| T::from_lizzard(Value(out)).expect("list element of its declared type"))
    }

    /// All elements, converted.
    pub fn to_vec(&self) -> Vec<T> {
        list_snapshot(self.h.v).into_iter().map(|v| T::from_lizzard(v).expect("list element of its declared type")).collect()
    }

    pub fn push(&self, x: T) -> Result<(), LizzardError> {
        let v = x.into_lizzard()?;
        LizzardError::check(unsafe { (api().list_push)(self.h.v.0, v.0) })
    }

    /// Replaces element `i` (an `IndexError` past the end).
    pub fn set(&self, i: usize, x: T) -> Result<(), LizzardError> {
        let v = x.into_lizzard()?;
        LizzardError::check(unsafe { (api().list_set)(self.h.v.0, i, v.0) })
    }
}

impl<T: FromLizzard + IntoLizzard + 'static> Default for ListRef<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: LizzardType> LizzardType for ListRef<T> {
    fn ntype() -> NType {
        NType::List(Box::new(T::ntype()))
    }
}

impl<T: LizzardType + 'static> FromLizzard for ListRef<T> {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        typed_container::<Self>(v)?;
        Ok(ListRef { h: Handle::borrow(v), _t: PhantomData })
    }
}

impl<T: LizzardType> IntoLizzard for ListRef<T> {
    fn ntype() -> NType {
        <Self as LizzardType>::ntype()
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(self.h.into_value())
    }
}

/// A `dict<K, V>` read and written in place.
pub struct DictRef<K, V> {
    h: Handle,
    _t: PhantomData<(K, V)>,
}

impl<K, V> Clone for DictRef<K, V> {
    fn clone(&self) -> Self {
        DictRef { h: self.h.clone(), _t: PhantomData }
    }
}

impl<K: FromLizzard + IntoLizzard, V: FromLizzard + IntoLizzard> DictRef<K, V> {
    pub fn len(&self) -> usize {
        unsafe { (api().dict_len)(self.h.v.0) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, k: K) -> Result<Option<V>, LizzardError> {
        let kv = k.into_lizzard()?;
        let mut out = 0;
        let found = unsafe { (api().dict_get)(self.h.v.0, kv.0, &mut out) };
        let r = found.then(|| V::from_lizzard(Value(out)).expect("dict value of its declared type"));
        kv.decref();
        Ok(r)
    }

    pub fn insert(&self, k: K, v: V) -> Result<(), LizzardError> {
        let kv = k.into_lizzard()?;
        let vv = match v.into_lizzard() {
            Ok(v) => v,
            Err(e) => {
                kv.decref();
                return Err(e);
            }
        };
        LizzardError::check(unsafe { (api().dict_insert)(self.h.v.0, kv.0, vv.0) })
    }

    pub fn keys(&self) -> Vec<K> {
        let keys = snapshot(|out, cap| unsafe { (api().dict_keys)(self.h.v.0, out, cap) });
        keys.into_iter().map(|k| K::from_lizzard(k).expect("dict key of its declared type")).collect()
    }
}

impl<K: LizzardType, V: LizzardType> LizzardType for DictRef<K, V> {
    fn ntype() -> NType {
        NType::Dict(Box::new(K::ntype()), Box::new(V::ntype()))
    }
}

impl<K: LizzardType + 'static, V: LizzardType + 'static> FromLizzard for DictRef<K, V> {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        typed_container::<Self>(v)?;
        Ok(DictRef { h: Handle::borrow(v), _t: PhantomData })
    }
}

impl<K: LizzardType, V: LizzardType> IntoLizzard for DictRef<K, V> {
    fn ntype() -> NType {
        <Self as LizzardType>::ntype()
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(self.h.into_value())
    }
}

/// An element type of `array<T>`.
pub trait ArrayElem: Copy {
    const ELEM: Elem;
    fn from_scalar(s: abi::Scalar) -> Self;
    fn to_value(self) -> Value;
}

macro_rules! array_elem {
    ($($t:ty => $e:ident),*) => {$(
        impl ArrayElem for $t {
            const ELEM: Elem = Elem::$e;
            fn from_scalar(s: abi::Scalar) -> Self {
                match s.tag {
                    abi::SCALAR_FLOAT => s.float as $t,
                    _ => s.bits as i64 as $t,
                }
            }
            fn to_value(self) -> Value {
                if Elem::$e.is_float() {
                    Value::float(self as f64)
                } else {
                    Value::int(self as i64)
                }
            }
        }
    )*};
}
array_elem!(i8 => I8, i16 => I16, i32 => I32, i64 => I64, u8 => U8, u16 => U16, u32 => U32, u64 => U64, f32 => F32, f64 => F64);

impl ArrayElem for bool {
    const ELEM: Elem = Elem::Bool;
    fn from_scalar(s: abi::Scalar) -> Self {
        s.tag == abi::SCALAR_BOOL && s.bits != 0
    }
    fn to_value(self) -> Value {
        Value::bool(self)
    }
}

/// An `array<T>` accessed in place (bounds-checked, element by element).
pub struct ArrayRef<T> {
    h: Handle,
    _t: PhantomData<T>,
}

impl<T: ArrayElem> ArrayRef<T> {
    pub fn len(&self) -> usize {
        unsafe { (api().array_len)(self.h.v.0) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<T> {
        let mut s = abi::Scalar { tag: 0, bits: 0, float: 0.0 };
        unsafe { (api().array_get)(self.h.v.0, i, &mut s) }.then(|| T::from_scalar(s))
    }

    pub fn set(&self, i: usize, x: T) -> Result<(), LizzardError> {
        let v = x.to_value();
        let e = unsafe { (api().array_set)(self.h.v.0, i, v.0) };
        v.decref();
        LizzardError::check(e)
    }

    pub fn to_vec(&self) -> Vec<T> {
        (0..self.len()).filter_map(|i| self.get(i)).collect()
    }
}

impl<T: ArrayElem> LizzardType for ArrayRef<T> {
    fn ntype() -> NType {
        NType::Array(T::ELEM)
    }
}

impl<T: ArrayElem> FromLizzard for ArrayRef<T> {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        let mut e = 0;
        if unsafe { (api().array_elem)(v.0, &mut e) } && e == T::ELEM.code() {
            Ok(ArrayRef { h: Handle::borrow(v), _t: PhantomData })
        } else {
            Err(expected(NType::Array(T::ELEM), v))
        }
    }
}

impl<T: ArrayElem> IntoLizzard for ArrayRef<T> {
    fn ntype() -> NType {
        NType::Array(T::ELEM)
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(self.h.into_value())
    }
}

// ---------------------------------------------------------------- errors

/// An error raised in Lizzard as an exception of class `kind`.
pub struct LizzardError {
    pub kind: String,
    pub msg: String,
    /// A Lizzard exception raised by a callback, re-raised unchanged.
    pending: Option<Pending>,
}

/// The interpreter's error carrying a Lizzard exception.
struct Pending(usize);

impl Pending {
    fn take(self) -> HostErr {
        let e = self.0 as HostErr;
        std::mem::forget(self);
        e
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        unsafe { (api().err_free)(self.0 as HostErr) }
    }
}

impl LizzardError {
    pub fn new(kind: &str, msg: impl Into<String>) -> LizzardError {
        LizzardError { kind: kind.into(), msg: msg.into(), pending: None }
    }

    pub fn type_error(msg: impl Into<String>) -> LizzardError {
        LizzardError::new("TypeError", msg)
    }

    pub fn value_error(msg: impl Into<String>) -> LizzardError {
        LizzardError::new("ValueError", msg)
    }

    pub fn runtime(msg: impl Into<String>) -> LizzardError {
        LizzardError::new("RuntimeError", msg)
    }

    /// Takes an interpreter error.
    fn from_host(e: HostErr) -> LizzardError {
        let (mut kind, mut msg) = (String::new(), String::new());
        let pending = unsafe { (api().err_read)(e, string_out(&mut kind), string_out(&mut msg)) };
        if pending {
            return LizzardError { kind, msg, pending: Some(Pending(e as usize)) };
        }
        unsafe { (api().err_free)(e) };
        LizzardError { kind, msg, pending: None }
    }

    fn check(e: HostErr) -> Result<(), LizzardError> {
        if e.is_null() {
            Ok(())
        } else {
            Err(LizzardError::from_host(e))
        }
    }

    /// Raises the error in `ctx`.
    fn raise(self, ctx: abi::Ctx) {
        match self.pending {
            Some(p) => unsafe { (api().err_raise)(ctx, p.take()) },
            None => unsafe { (api().raise)(ctx, Str::new(&self.kind), Str::new(&self.msg)) },
        }
    }

    fn context(mut self, what: &str) -> LizzardError {
        if self.pending.is_none() {
            self.msg = format!("{what}: {}", self.msg);
        }
        self
    }
}

impl std::fmt::Display for LizzardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
}

impl std::fmt::Debug for LizzardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl From<String> for LizzardError {
    fn from(msg: String) -> LizzardError {
        LizzardError::runtime(msg)
    }
}

impl From<&str> for LizzardError {
    fn from(msg: &str) -> LizzardError {
        LizzardError::runtime(msg)
    }
}

// ---------------------------------------------------------------- context

/// The interpreter, as seen by a native call.
pub struct Ctx {
    ctx: abi::Ctx,
    _not_send: PhantomData<*const ()>,
}

impl Ctx {
    /// Calls a Lizzard function. Arguments are passed as new references.
    pub fn call<R: FromLizzard>(&mut self, f: &Func, args: impl IntoArgs) -> Result<R, LizzardError> {
        let vals = args.into_values()?;
        let mut out = 0;
        let e = unsafe { (api().call)(self.ctx, f.0.v.0, vals.as_ptr() as *const RawValue, vals.len(), &mut out) };
        vals.into_iter().for_each(Value::decref);
        LizzardError::check(e)?;
        let v = Value(out);
        let r = R::from_lizzard(v);
        v.decref();
        r
    }

    /// Runs `f` without access to Lizzard values (they cannot be captured: they
    /// are not `Send`), letting other threads reclaim memory meanwhile. Use
    /// it around long or blocking work.
    pub fn offline<R: Send>(&mut self, f: impl FnOnce() -> R + Send) -> R {
        struct Online;
        impl Drop for Online {
            fn drop(&mut self) {
                unsafe { (api().online)() }
            }
        }
        unsafe { (api().offline)() };
        let _on = Online;
        f()
    }
}

/// Arguments of [`Ctx::call`]: a tuple of convertible values.
pub trait IntoArgs {
    fn into_values(self) -> Result<Vec<Value>, LizzardError>;
}

macro_rules! into_args {
    ($($A:ident),*) => {
        impl<$($A: IntoLizzard),*> IntoArgs for ($($A,)*) {
            #[allow(non_snake_case, unused_mut)]
            fn into_values(self) -> Result<Vec<Value>, LizzardError> {
                let ($($A,)*) = self;
                let mut out: Vec<Value> = Vec::new();
                $(
                    match $A.into_lizzard() {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            out.into_iter().for_each(Value::decref);
                            return Err(e);
                        }
                    }
                )*
                Ok(out)
            }
        }
    };
}
into_args!();
into_args!(A1);
into_args!(A1, A2);
into_args!(A1, A2, A3);
into_args!(A1, A2, A3, A4);

// ---------------------------------------------------------------- functions

type Body = Box<dyn Fn(&mut Ctx, &[Value]) -> Result<Value, LizzardError> + Send + Sync>;

/// A native function or method, before it is given to the interpreter.
pub struct NativeFn {
    name: String,
    /// Parameter types (after `self` for a method).
    params: Vec<NType>,
    ret: NType,
    /// A method: the first argument is the object.
    has_self: bool,
    body: Box<FnBody>,
}

/// Storage a function descriptor borrows (the strings' heap buffers, which
/// stay put when the tuple moves): the name and the encoded signature.
type FnKeep = (String, Vec<u8>);

/// What the interpreter calls: the shown name (for panics) and the closure.
struct FnBody {
    shown: String,
    f: Body,
}

fn panic_msg(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown panic".into())
}

unsafe extern "C" fn call_fn(data: *const c_void, ctx: abi::Ctx, args: *const RawValue, n: usize, out: *mut RawValue) -> bool {
    // SAFETY: `data` is the `FnBody` of `NativeFn::desc`; the interpreter
    // passes `n` borrowed values (`Value` is `repr(transparent)`).
    let body = unsafe { &*(data as *const FnBody) };
    let args = if n == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(args as *const Value, n) } };
    let mut c = Ctx { ctx, _not_send: PhantomData };
    match catch_unwind(AssertUnwindSafe(|| (body.f)(&mut c, args))) {
        Ok(Ok(v)) => {
            unsafe { *out = v.0 };
            true
        }
        Ok(Err(e)) => {
            e.raise(ctx);
            false
        }
        Err(p) => {
            LizzardError::runtime(format!("native panic in {}(): {}", body.shown, panic_msg(&*p))).raise(ctx);
            false
        }
    }
}

unsafe extern "C" fn free_fn(data: *mut c_void) {
    // SAFETY: made by `Box::into_raw` in `NativeFn::desc`.
    drop(unsafe { Box::from_raw(data as *mut FnBody) })
}

impl NativeFn {
    fn new(name: &str, shown: String, params: Vec<NType>, ret: NType, has_self: bool, f: Body) -> NativeFn {
        NativeFn { name: name.into(), params, ret, has_self, body: Box::new(FnBody { shown, f }) }
    }

    /// The descriptor (handing over the closure) and the storage it borrows.
    fn desc(self) -> (abi::FnDesc, FnKeep) {
        let keep = (self.name, encode_sig(&self.params, &self.ret));
        let d = abi::FnDesc {
            name: Str::new(&keep.0),
            sig: Str { ptr: keep.1.as_ptr(), len: keep.1.len() },
            has_self: self.has_self,
            call: call_fn,
            data: Box::into_raw(self.body) as *mut c_void,
            free: free_fn,
        };
        (d, keep)
    }
}

fn arg<A: FromLizzard>(vals: &[Value], i: usize, name: &str) -> Result<A, LizzardError> {
    A::from_lizzard(vals[i]).map_err(|e| e.context(&format!("argument {} of {name}()", i + 1)))
}

/// A Rust closure usable as a native function: `Fn(A1, ..) -> R`, or
/// `Fn(&mut Ctx, A1, ..) -> R` (the `WithCtx` form).
pub trait IntoNative<Args>: Send + Sync + 'static {
    fn into_native(self, name: &str) -> NativeFn;
}

/// Marks the closure forms that take a [`Ctx`] first.
pub struct WithCtx<T>(PhantomData<T>);

macro_rules! impl_native {
    ($($A:ident $i:tt),*) => {
        impl<F, R, $($A,)*> IntoNative<($($A,)*)> for F
        where
            F: Fn($($A),*) -> R + Send + Sync + 'static,
            R: IntoLizzard,
            $($A: FromLizzard,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_native(self, name: &str) -> NativeFn {
                let n = name.to_string();
                NativeFn::new(name, n.clone(), vec![$($A::ntype()),*], R::ntype(), false, Box::new(move |_ctx, vals| {
                    $(let $A: $A = arg(vals, $i, &n)?;)*
                    self($($A),*).into_lizzard()
                }))
            }
        }

        impl<F, R, $($A,)*> IntoNative<WithCtx<($($A,)*)>> for F
        where
            F: Fn(&mut Ctx, $($A),*) -> R + Send + Sync + 'static,
            R: IntoLizzard,
            $($A: FromLizzard,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_native(self, name: &str) -> NativeFn {
                let n = name.to_string();
                NativeFn::new(name, n.clone(), vec![$($A::ntype()),*], R::ntype(), false, Box::new(move |ctx, vals| {
                    $(let $A: $A = arg(vals, $i, &n)?;)*
                    self(ctx, $($A),*).into_lizzard()
                }))
            }
        }
    };
}
impl_native!();
impl_native!(A1 0);
impl_native!(A1 0, A2 1);
impl_native!(A1 0, A2 1, A3 2);
impl_native!(A1 0, A2 1, A3 2, A4 3);
impl_native!(A1 0, A2 1, A3 2, A4 3, A5 4);
impl_native!(A1 0, A2 1, A3 2, A4 3, A5 4, A6 5);
impl_native!(A1 0, A2 1, A3 2, A4 3, A5 4, A6 5, A7 6);
impl_native!(A1 0, A2 1, A3 2, A4 3, A5 4, A6 5, A7 6, A8 7);

// ---------------------------------------------------------------- classes

/// A counted reference to an instance of the native class wrapping `T`.
pub struct Native<T> {
    h: Handle,
    _t: PhantomData<T>,
}

impl<T> Clone for Native<T> {
    fn clone(&self) -> Self {
        Native { h: self.h.clone(), _t: PhantomData }
    }
}

/// A running method call's access to a native object.
struct Borrow {
    v: Value,
    exclusive: bool,
}

impl Drop for Borrow {
    fn drop(&mut self) {
        unsafe { (api().obj_leave)(self.v.0, self.exclusive) }
    }
}

/// Starts a call on native object `v` (whose class wraps the caller's `T`).
fn enter(v: Value, exclusive: bool) -> Result<(*mut c_void, Borrow), LizzardError> {
    let (mut data, mut excl) = (std::ptr::null_mut(), false);
    LizzardError::check(unsafe { (api().obj_enter)(v.0, exclusive, &mut data, &mut excl) })?;
    Ok((data, Borrow { v, exclusive: excl }))
}

fn is_native<T: 'static>(v: Value) -> bool {
    let k = unsafe { (api().obj_key)(v.0) };
    k != 0 && k == class_key(TypeId::of::<T>())
}

impl<T: 'static> Native<T> {
    /// Wraps `t` in a new object of its registered class.
    pub fn new(t: T) -> Result<Native<T>, LizzardError> {
        let data = Box::into_raw(Box::new(t)) as *mut c_void;
        let mut out = 0;
        if !unsafe { (api().obj_new)(class_key(TypeId::of::<T>()), data, &mut out) } {
            // SAFETY: not taken by the interpreter.
            drop(unsafe { Box::from_raw(data as *mut T) });
            return Err(LizzardError::runtime(format!("no native class is registered for `{}`", std::any::type_name::<T>())));
        }
        Ok(Native { h: Handle::owned(Value(out)), _t: PhantomData })
    }

    /// Runs `f` with shared access (`BorrowError` while an exclusive call runs).
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> Result<R, LizzardError> {
        let (data, _b) = enter(self.h.v, false)?;
        // SAFETY: the object's class wraps `T` (checked on conversion), and a
        // borrow is held.
        Ok(f(unsafe { &*(data as *const T) }))
    }

    /// Runs `f` with exclusive access (`BorrowError` while another call runs).
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> Result<R, LizzardError> {
        let (data, _b) = enter(self.h.v, true)?;
        // SAFETY: as for `with`, with an exclusive borrow.
        Ok(f(unsafe { &mut *(data as *mut T) }))
    }
}

impl<T: 'static> LizzardType for Native<T> {
    fn ntype() -> NType {
        NType::Class(TypeId::of::<T>())
    }
}

impl<T: 'static> FromLizzard for Native<T> {
    fn from_lizzard(v: Value) -> Result<Self, LizzardError> {
        if is_native::<T>(v) {
            Ok(Native { h: Handle::borrow(v), _t: PhantomData })
        } else {
            Err(expected(<Self as LizzardType>::ntype(), v))
        }
    }
}

impl<T: 'static> IntoLizzard for Native<T> {
    fn ntype() -> NType {
        NType::Class(TypeId::of::<T>())
    }
    fn into_lizzard(self) -> Result<Value, LizzardError> {
        Ok(self.h.into_value())
    }
}

/// The type arguments of a generic class's instantiation: a [`LizzardType`]
/// for one parameter, a tuple of them for several.
pub trait TypeArgs {
    fn ntypes() -> Vec<NType>;
}

impl<T: LizzardType> TypeArgs for T {
    fn ntypes() -> Vec<NType> {
        vec![T::ntype()]
    }
}

impl<A: LizzardType, B: LizzardType> TypeArgs for (A, B) {
    fn ntypes() -> Vec<NType> {
        vec![A::ntype(), B::ntype()]
    }
}

impl<A: LizzardType, B: LizzardType, C: LizzardType> TypeArgs for (A, B, C) {
    fn ntypes() -> Vec<NType> {
        vec![A::ntype(), B::ntype(), C::ntype()]
    }
}

/// Result of a constructor: `T` or `Result<T, LizzardError>`.
pub trait CtorOut<T> {
    fn into_result(self) -> Result<T, LizzardError>;
}

impl<T> CtorOut<T> for T {
    fn into_result(self) -> Result<T, LizzardError> {
        Ok(self)
    }
}

impl<T> CtorOut<T> for Result<T, LizzardError> {
    fn into_result(self) -> Result<T, LizzardError> {
        self
    }
}

/// A Rust closure usable as a constructor of `T`.
pub trait IntoCtor<T, Args>: Send + Sync + 'static {
    fn into_ctor(self, name: &str) -> NativeFn;
}

/// A Rust closure usable as a `&self` method of `T`.
pub trait IntoMethod<T, Args>: Send + Sync + 'static {
    fn into_method(self, class: &str, name: &str) -> NativeFn;
}

/// A Rust closure usable as a `&mut self` method of `T`.
pub trait IntoMethodMut<T, Args>: Send + Sync + 'static {
    fn into_method(self, class: &str, name: &str) -> NativeFn;
}

/// The receiver of a method of the class with key `key`.
fn self_obj(v: Value, key: u64, what: &str) -> Result<Value, LizzardError> {
    if unsafe { (api().obj_key)(v.0) } == key {
        Ok(v)
    } else {
        Err(LizzardError::type_error(format!("{what}() called on a `{}`", v.type_name())))
    }
}

macro_rules! impl_class_fns {
    ($($A:ident $i:tt),*) => {
        impl<F, O, T, $($A,)*> IntoCtor<T, ($($A,)*)> for F
        where
            F: Fn($($A),*) -> O + Send + Sync + 'static,
            O: CtorOut<T>,
            T: 'static,
            $($A: FromLizzard,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_ctor(self, name: &str) -> NativeFn {
                let n = name.to_string();
                NativeFn::new(name, n.clone(), vec![$($A::ntype()),*], NType::Class(TypeId::of::<T>()), false, Box::new(move |_ctx, vals| {
                    $(let $A: $A = arg(vals, $i, &n)?;)*
                    Native::new(self($($A),*).into_result()?)?.into_lizzard()
                }))
            }
        }

        impl<F, R, T, $($A,)*> IntoMethod<T, ($($A,)*)> for F
        where
            F: Fn(&T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoLizzard,
            T: 'static,
            $($A: FromLizzard,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                let key = class_key(TypeId::of::<T>());
                NativeFn::new(name, n.clone(), vec![$($A::ntype()),*], R::ntype(), true, Box::new(move |_ctx, vals| {
                    let o = self_obj(vals[0], key, &n)?;
                    let rest = &vals[1..];
                    $(let $A: $A = arg(rest, $i, &n)?;)*
                    let (data, b) = enter(o, false)?;
                    // SAFETY: the class wraps `T`, and a borrow is held.
                    let r = self(unsafe { &*(data as *const T) }, $($A),*);
                    drop(b);
                    r.into_lizzard()
                }))
            }
        }

        impl<F, R, T, $($A,)*> IntoMethod<T, WithCtx<($($A,)*)>> for F
        where
            F: Fn(&mut Ctx, &T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoLizzard,
            T: 'static,
            $($A: FromLizzard,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                let key = class_key(TypeId::of::<T>());
                NativeFn::new(name, n.clone(), vec![$($A::ntype()),*], R::ntype(), true, Box::new(move |ctx, vals| {
                    let o = self_obj(vals[0], key, &n)?;
                    let rest = &vals[1..];
                    $(let $A: $A = arg(rest, $i, &n)?;)*
                    let (data, b) = enter(o, false)?;
                    // SAFETY: as above.
                    let r = self(ctx, unsafe { &*(data as *const T) }, $($A),*);
                    drop(b);
                    r.into_lizzard()
                }))
            }
        }

        impl<F, R, T, $($A,)*> IntoMethodMut<T, ($($A,)*)> for F
        where
            F: Fn(&mut T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoLizzard,
            T: 'static,
            $($A: FromLizzard,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                let key = class_key(TypeId::of::<T>());
                NativeFn::new(name, n.clone(), vec![$($A::ntype()),*], R::ntype(), true, Box::new(move |_ctx, vals| {
                    let o = self_obj(vals[0], key, &n)?;
                    let rest = &vals[1..];
                    $(let $A: $A = arg(rest, $i, &n)?;)*
                    let (data, b) = enter(o, true)?;
                    // SAFETY: the class wraps `T`, and an exclusive borrow is held.
                    let r = self(unsafe { &mut *(data as *mut T) }, $($A),*);
                    drop(b);
                    r.into_lizzard()
                }))
            }
        }

        impl<F, R, T, $($A,)*> IntoMethodMut<T, WithCtx<($($A,)*)>> for F
        where
            F: Fn(&mut Ctx, &mut T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoLizzard,
            T: 'static,
            $($A: FromLizzard,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                let key = class_key(TypeId::of::<T>());
                NativeFn::new(name, n.clone(), vec![$($A::ntype()),*], R::ntype(), true, Box::new(move |ctx, vals| {
                    let o = self_obj(vals[0], key, &n)?;
                    let rest = &vals[1..];
                    $(let $A: $A = arg(rest, $i, &n)?;)*
                    let (data, b) = enter(o, true)?;
                    // SAFETY: as above.
                    let r = self(ctx, unsafe { &mut *(data as *mut T) }, $($A),*);
                    drop(b);
                    r.into_lizzard()
                }))
            }
        }
    };
}
impl_class_fns!();
impl_class_fns!(A1 0);
impl_class_fns!(A1 0, A2 1);
impl_class_fns!(A1 0, A2 1, A3 2);
impl_class_fns!(A1 0, A2 1, A3 2, A4 3);
impl_class_fns!(A1 0, A2 1, A3 2, A4 3, A5 4);
impl_class_fns!(A1 0, A2 1, A3 2, A4 3, A5 4, A6 5);

// ---------------------------------------------------------------- modules

/// A module of native functions, classes and traits, built in the
/// interpreter as it is defined.
pub struct Module {
    raw: *mut c_void,
}

/// Panics with an error of the interpreter's module building.
fn built(ok: bool, err: String) {
    if !ok {
        panic!("{err}");
    }
}

impl Module {
    /// Builds module `name` in the installed interpreter; returns the
    /// interpreter's module (for its `module_from_raw`).
    pub fn build(name: &str, f: impl FnOnce(&mut Module)) -> *mut c_void {
        let mut m = Module { raw: unsafe { (api().module_new)(Str::new(name)) } };
        f(&mut m);
        std::mem::replace(&mut m.raw, std::ptr::null_mut())
    }

    /// Adds a function: a closure `Fn(A1, ..) -> R` or `Fn(&mut Ctx, A1, ..) -> R`.
    pub fn function<Args, F: IntoNative<Args>>(&mut self, name: &str, f: F) -> &mut Self {
        let (d, _keep) = f.into_native(name).desc();
        let mut err = String::new();
        built(unsafe { (api().module_function)(self.raw, &d, string_out(&mut err)) }, err);
        self
    }

    /// Adds a class whose methods may run concurrently (`T: Send + Sync`).
    pub fn class<T: Send + Sync + 'static>(&mut self, name: &str, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add_class(name, abi::MODE_SYNC, build)
    }

    /// Adds a class whose calls are exclusive (`T: Send`).
    pub fn class_send<T: Send + 'static>(&mut self, name: &str, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add_class(name, abi::MODE_SEND, build)
    }

    /// Adds a class over a Rust type that is not `Send` (any `T`): the Lizzard
    /// class is not `Send` either, so instances stay on their thread.
    pub fn class_not_send<T: 'static>(&mut self, name: &str, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add_class(name, abi::MODE_LOCAL, build)
    }

    /// Adds a trait whose methods are all required, for classes of this
    /// module (defined after it) and Lizzard structs to implement:
    ///
    /// ```ignore
    /// m.trait_def("Shape", |t| {
    ///     t.method::<(), f64>("area");
    ///     t.method::<(f64,), ()>("scale");
    /// });
    /// ```
    pub fn trait_def(&mut self, name: &str, build: impl FnOnce(&mut TraitBuilder)) -> &mut Self {
        let mut t = TraitBuilder { methods: Vec::new() };
        build(&mut t);
        let sigs: Vec<Vec<u8>> = t.methods.iter().map(|(_, p, r)| encode_sig(p, r)).collect();
        let descs: Vec<abi::MethodSig> =
            t.methods.iter().zip(&sigs).map(|((n, _, _), s)| abi::MethodSig { name: Str::new(n), sig: Str { ptr: s.as_ptr(), len: s.len() } }).collect();
        let mut err = String::new();
        built(unsafe { (api().module_trait)(self.raw, Str::new(name), descs.as_ptr(), descs.len(), string_out(&mut err)) }, err);
        self
    }

    fn add_class<T: 'static>(&mut self, name: &str, mode: u8, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        let parts = class_parts(name, mode, Vec::new(), build);
        let (d, _keep) = parts.desc();
        let mut err = String::new();
        built(unsafe { (api().module_class)(self.raw, &d, string_out(&mut err)) }, err);
        self
    }

    /// Adds a generic class `name<tparams>` whose supported instantiations
    /// are registered by `build`, each over its own Rust type:
    ///
    /// ```ignore
    /// m.generic_class("Stack", &["T"], |g| {
    ///     g.class::<i64, _>(stack::<i64>);       // Stack<int>
    ///     g.class::<String, _>(stack::<String>); // Stack<str>
    /// });
    /// ```
    ///
    /// Constructing another instantiation from Lizzard is a compile error when
    /// the type arguments are known statically, and a `TypeError` otherwise.
    pub fn generic_class(&mut self, name: &str, tparams: &[&str], build: impl FnOnce(&mut GenericBuilder)) -> &mut Self {
        assert!(!tparams.is_empty(), "generic class `{name}` needs type parameters");
        let mut g = GenericBuilder { name: name.into(), arity: tparams.len(), instances: Vec::new() };
        build(&mut g);
        let (descs, _keep): (Vec<_>, Vec<_>) = g.instances.into_iter().map(ClassParts::desc).unzip();
        let tps: Vec<Str> = tparams.iter().map(|t| Str::new(t)).collect();
        let mut err = String::new();
        let ok = unsafe { (api().module_generic)(self.raw, Str::new(name), tps.as_ptr(), tps.len(), descs.as_ptr(), descs.len(), string_out(&mut err)) };
        built(ok, err);
        self
    }
}

impl Drop for Module {
    /// A module abandoned while being built (by a panic).
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (api().module_free)(self.raw) }
        }
    }
}

/// A class's parts, before it is given to the interpreter.
struct ClassParts {
    name: String,
    key: u64,
    mode: u8,
    locked: bool,
    type_args: Vec<NType>,
    drop: abi::FreeFn,
    init: Option<NativeFn>,
    methods: Vec<NativeFn>,
    repr: Option<Box<ReprBody>>,
    implements: Vec<String>,
}

/// Storage a class descriptor borrows.
struct ClassKeep {
    _name: String,
    _type_args: Vec<u8>,
    _init: Option<(Box<abi::FnDesc>, FnKeep)>,
    _methods: Vec<abi::FnDesc>,
    _method_keep: Vec<FnKeep>,
    _implements: Vec<String>,
    _implements_strs: Vec<Str>,
}

type ReprBody = Box<dyn Fn(*const c_void) -> String + Send + Sync>;

unsafe extern "C" fn repr_fn(data: *const c_void, obj: *const c_void, out: abi::StrOut) -> bool {
    // SAFETY: `data` is the `ReprBody` of `ClassParts::desc`.
    let f = unsafe { &*(data as *const ReprBody) };
    match catch_unwind(AssertUnwindSafe(|| f(obj))) {
        Ok(s) => {
            unsafe { abi::write_str(out, &s) };
            true
        }
        Err(_) => false,
    }
}

unsafe extern "C" fn free_repr(data: *mut c_void) {
    drop(unsafe { Box::from_raw(data as *mut ReprBody) })
}

unsafe extern "C" fn drop_value<T>(data: *mut c_void) {
    // SAFETY: made by `Native::new` for the class of `T`.
    let b = unsafe { Box::from_raw(data as *mut T) };
    if let Err(p) = catch_unwind(AssertUnwindSafe(move || drop(b))) {
        eprintln!("warning: panic while dropping a native `{}`: {}", std::any::type_name::<T>(), panic_msg(&*p));
    }
}

impl ClassParts {
    fn desc(self) -> (abi::ClassDesc, ClassKeep) {
        let mut type_args = Vec::new();
        if !self.type_args.is_empty() {
            encode_list(&self.type_args, &mut type_args);
        }
        let init = self.init.map(|f| {
            let (d, k) = f.desc();
            (Box::new(d), k)
        });
        let (methods, method_keep): (Vec<_>, Vec<_>) = self.methods.into_iter().map(NativeFn::desc).unzip();
        let implements_strs: Vec<Str> = self.implements.iter().map(|s| Str::new(s)).collect();
        let (repr, repr_data, repr_free) = match self.repr {
            Some(r) => (Some(repr_fn as abi::ReprFn), Box::into_raw(r) as *mut c_void, Some(free_repr as abi::FreeFn)),
            None => (None, std::ptr::null_mut(), None),
        };
        let d = abi::ClassDesc {
            name: Str::new(&self.name),
            key: self.key,
            mode: self.mode,
            locked: self.locked,
            type_args: Str { ptr: type_args.as_ptr(), len: type_args.len() },
            drop: self.drop,
            init: init.as_ref().map_or(std::ptr::null(), |(d, _)| &**d as *const abi::FnDesc),
            methods: methods.as_ptr(),
            nmethods: methods.len(),
            repr,
            repr_data,
            repr_free,
            implements: implements_strs.as_ptr(),
            nimplements: implements_strs.len(),
        };
        let keep = ClassKeep {
            _name: self.name,
            _type_args: type_args,
            _init: init,
            _methods: methods,
            _method_keep: method_keep,
            _implements: self.implements,
            _implements_strs: implements_strs,
        };
        (d, keep)
    }
}

/// A class over `T`, built by `build`.
fn class_parts<T: 'static>(name: &str, mode: u8, type_args: Vec<NType>, build: impl FnOnce(&mut ClassBuilder<T>)) -> ClassParts {
    let mut b = ClassBuilder::<T> { name: name.into(), locked: false, init: None, methods: Vec::new(), repr: None, implements: Vec::new(), _t: PhantomData };
    build(&mut b);
    ClassParts {
        name: b.name,
        key: class_key(TypeId::of::<T>()),
        mode,
        locked: b.locked,
        type_args,
        drop: drop_value::<T>,
        init: b.init,
        methods: b.methods,
        repr: b.repr,
        implements: b.implements,
    }
}

/// Registers the instantiations of a generic class (see
/// [`Module::generic_class`]). `A` gives the type arguments ([`TypeArgs`]);
/// the Rust type's auto traits choose the mode as for [`Module::class`].
pub struct GenericBuilder {
    name: String,
    arity: usize,
    instances: Vec<ClassParts>,
}

impl GenericBuilder {
    /// An instantiation whose methods may run concurrently (`T: Send + Sync`).
    pub fn class<A: TypeArgs, T: Send + Sync + 'static>(&mut self, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add::<A, T>(abi::MODE_SYNC, build)
    }

    /// An instantiation whose calls are exclusive (`T: Send`).
    pub fn class_send<A: TypeArgs, T: Send + 'static>(&mut self, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add::<A, T>(abi::MODE_SEND, build)
    }

    /// An instantiation over a Rust type that is not `Send`: its instances
    /// are not `Send` either.
    pub fn class_not_send<A: TypeArgs, T: 'static>(&mut self, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add::<A, T>(abi::MODE_LOCAL, build)
    }

    fn add<A: TypeArgs, T: 'static>(&mut self, mode: u8, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        let args = A::ntypes();
        assert!(args.len() == self.arity, "`{}` takes {} type argument(s), got {} for `{}`", self.name, self.arity, args.len(), std::any::type_name::<T>());
        let dup = self.instances.iter().any(|c| c.type_args == args);
        assert!(!dup, "`{}` instantiation {args:?} is registered twice", self.name);
        let name = self.name.clone();
        self.instances.push(class_parts(&name, mode, args, build));
        self
    }
}

/// Builds a native class (see [`Module::class`]).
pub struct ClassBuilder<T> {
    name: String,
    locked: bool,
    init: Option<NativeFn>,
    methods: Vec<NativeFn>,
    repr: Option<Box<ReprBody>>,
    implements: Vec<String>,
    _t: PhantomData<fn() -> T>,
}

/// Builds a native trait (see [`Module::trait_def`]).
pub struct TraitBuilder {
    methods: Vec<(String, Vec<NType>, NType)>,
}

impl TraitBuilder {
    /// A required method taking arguments `A` (a tuple: `()`, `(f64,)`, ...)
    /// and returning `R`.
    pub fn method<A: ArgTypes, R: IntoLizzard>(&mut self, name: &str) -> &mut Self {
        self.methods.push((name.into(), A::ntypes(), R::ntype()));
        self
    }
}

/// The parameter types of a native trait method: a tuple of [`LizzardType`]s.
pub trait ArgTypes {
    fn ntypes() -> Vec<NType>;
}

macro_rules! impl_arg_types {
    ($($A:ident),*) => {
        impl<$($A: LizzardType),*> ArgTypes for ($($A,)*) {
            fn ntypes() -> Vec<NType> {
                vec![$($A::ntype()),*]
            }
        }
    };
}
impl_arg_types!();
impl_arg_types!(A1);
impl_arg_types!(A1, A2);
impl_arg_types!(A1, A2, A3);
impl_arg_types!(A1, A2, A3, A4);

impl<T: 'static> ClassBuilder<T> {
    /// The constructor, called as `Name(args)`: `Fn(A1, ..) -> T` (or
    /// `-> Result<T, LizzardError>`).
    pub fn init<Args, F: IntoCtor<T, Args>>(&mut self, f: F) -> &mut Self {
        self.init = Some(f.into_ctor(&self.name));
        self
    }

    /// A method taking `&T`: `Fn(&T, A1, ..) -> R` or `Fn(&mut Ctx, &T, A1, ..) -> R`.
    pub fn method<Args, F: IntoMethod<T, Args>>(&mut self, name: &str, f: F) -> &mut Self {
        let m = f.into_method(&self.name, name);
        self.add_method(m)
    }

    /// A method taking `&mut T` (exclusive): `Fn(&mut T, A1, ..) -> R` or
    /// `Fn(&mut Ctx, &mut T, A1, ..) -> R`.
    pub fn method_mut<Args, F: IntoMethodMut<T, Args>>(&mut self, name: &str, f: F) -> &mut Self {
        let m = f.into_method(&self.name, name);
        self.add_method(m)
    }

    fn add_method(&mut self, m: NativeFn) -> &mut Self {
        // A later definition replaces an earlier one.
        self.methods.retain(|x| x.name != m.name);
        self.methods.push(m);
        self
    }

    /// How instances are shown by `print` and `str`.
    pub fn repr(&mut self, f: impl Fn(&T) -> String + Send + Sync + 'static) -> &mut Self {
        // SAFETY: called with the object's `T` while a borrow is held.
        let body: ReprBody = Box::new(move |d| f(unsafe { &*(d as *const T) }));
        self.repr = Some(Box::new(body));
        self
    }

    /// Implements trait `name`: a trait of this module defined earlier
    /// ([`Module::trait_def`]), or the builtin `Ord` (`__lt__`), `Num` (the
    /// arithmetic dunders and `__lt__`) or `Len` (`__len__`). The methods are
    /// checked when the class is added (a mismatch panics). `Send` is not
    /// implemented this way: it follows the Rust type.
    pub fn implements(&mut self, name: &str) -> &mut Self {
        self.implements.push(name.into());
        self
    }

    /// Conflicting calls wait for each other instead of raising `BorrowError`
    /// (not for local classes: waiting on the same thread cannot end).
    pub fn locked(&mut self) -> &mut Self {
        self.locked = true;
        self
    }
}

// ---------------------------------------------------------------- extensions

/// Declares the entry point of an extension module (a `cdylib` whose
/// library name is the module's name):
///
/// ```ignore
/// lizzard_ext::module!(geom, |m| {
///     m.function("dot", |a: f64, b: f64| a * b);
/// });
/// ```
///
/// `import geom` then finds `libgeom.so` (`libgeom.dylib`, `geom.dll`) in
/// the importing script's directory or on `LIZZARD_PATH`.
#[macro_export]
macro_rules! module {
    ($name:ident, $build:expr) => {
        #[no_mangle]
        pub extern "C" fn lizzard_ext_abi_version() -> u32 {
            $crate::abi::ABI_VERSION
        }

        /// # Safety
        /// Called by the interpreter with its function table.
        #[no_mangle]
        pub unsafe extern "C" fn lizzard_ext_init(api: *const $crate::abi::LizzardApi, err: $crate::abi::StrOut) -> *mut ::std::ffi::c_void {
            unsafe { $crate::init(api, stringify!($name), $build, err) }
        }
    };
}

/// The body of an extension's `lizzard_ext_init` (see [`module!`]).
///
/// # Safety
/// `api` must be the interpreter's function table, which lives as long as
/// the process, and `err` a valid sink.
#[doc(hidden)]
pub unsafe fn init(api: *const LizzardApi, name: &str, build: impl FnOnce(&mut Module), err: abi::StrOut) -> *mut c_void {
    let fail = |msg: String| {
        unsafe { abi::write_str(err, &msg) };
        std::ptr::null_mut()
    };
    // SAFETY: guaranteed by the caller.
    let api = unsafe { &*api };
    if api.version != ABI_VERSION {
        return fail(format!("module `{name}` was built for Lizzard ABI version {ABI_VERSION}, the interpreter has version {}", api.version));
    }
    if let Err(e) = install(api) {
        return fail(e);
    }
    match catch_unwind(AssertUnwindSafe(|| Module::build(name, build))) {
        Ok(m) => m,
        Err(p) => fail(format!("module `{name}` failed to initialize: {}", panic_msg(&*p))),
    }
}
