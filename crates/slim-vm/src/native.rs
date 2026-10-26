//! Native modules: Rust functions and classes callable from slim.
//!
//! A [`Module`] is built before a program is compiled. The compiler sees each
//! item's signature as slim types ([`NType`]), so calls of natives are checked
//! like calls of slim functions. Values cross the boundary through
//! [`FromSlim`] (arguments, always borrowed from the caller) and [`IntoSlim`]
//! (results). Handles such as [`ListRef`] or [`Native`] hold a counted
//! reference, so a native may keep one; they are `!Send` and stay on the
//! interpreter thread that created them.
//!
//! A native class wraps a Rust value. The registration function decides how
//! its methods may run, from the Rust type's auto traits:
//! - [`Module::class`] (`T: Send + Sync`): `&self` methods run concurrently
//!   from any thread; `&mut self` methods are exclusive.
//! - [`Module::class_send`] (`T: Send`): every method call is exclusive.
//! - [`Module::class_not_send`] (any `T`): the slim class is not `Send`
//!   either, so instances stay on the thread that created them, which is
//!   also checked on every call (`ThreadError`), and are dropped there.
//!
//! A call that conflicts with a running one (a `&mut self` method while
//! another method runs) raises `BorrowError`, or waits for it to finish if
//! the class is [`ClassBuilder::locked`].
//!
//! A generic class ([`Module::generic_class`]) lists the instantiations it
//! supports, each an ordinary native class over its own Rust type
//! (`Stack<i64>`, `Stack<String>`). Slim sees one generic class `Stack<T>`;
//! constructing `Stack<int>` dispatches to the Rust class registered for
//! `int`, and other type arguments raise `TypeError`.
//!
//! A module may define traits ([`Module::trait_def`]) that its classes
//! implement ([`ClassBuilder::implements`]), as may slim classes. A class
//! may also implement the builtin `Ord`, `Num` and `Len` through its dunder
//! methods. Programs add their own traits to native classes with
//! `impl Trait for Class`.

use std::any::TypeId;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use crate::array::{Elem, Scalar};
use crate::interp::{ExcInfo, Raised, Thread, VResult};
use crate::object::{Body, Obj};
use crate::rtype::{self, Ctor};
use crate::value::Value;

// ---------------------------------------------------------------- types

/// A slim type, as the compiler sees a native's parameters and result.
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
    /// A native class, by the Rust type it wraps.
    Class(TypeId),
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
            NType::Class(id) => match class_by_type(*id) {
                Some(c) => rtype::class_type(c),
                None => rtype::any(),
            },
        }
    }

    /// Whether this is `int`, `float` or `bool` (an unboxed scalar).
    pub fn is_scalar(&self) -> bool {
        matches!(self, NType::Int | NType::Float | NType::Bool)
    }
}

/// A Rust type with a slim counterpart.
pub trait SlimType {
    fn ntype() -> NType;
}

/// Conversion of a (borrowed) slim argument.
pub trait FromSlim: SlimType + Sized {
    fn from_slim(v: Value) -> Result<Self, SlimError>;
}

/// Conversion of a native result into an owned slim value.
pub trait IntoSlim {
    fn ntype() -> NType;
    fn into_slim(self) -> Result<Value, SlimError>;
}

fn expected(t: NType, v: Value) -> SlimError {
    SlimError::type_error(format!("expected `{}`, got `{}`", rtype::name(t.rt()), crate::types::value_type_name(v)))
}

macro_rules! int_conv {
    ($($t:ty),*) => {$(
        impl SlimType for $t {
            fn ntype() -> NType {
                NType::Int
            }
        }
        impl FromSlim for $t {
            fn from_slim(v: Value) -> Result<Self, SlimError> {
                let i = v.as_int().ok_or_else(|| expected(NType::Int, v))?;
                <$t>::try_from(i).map_err(|_| SlimError::new("OverflowError", format!("{i} does not fit in {}", stringify!($t))))
            }
        }
        impl IntoSlim for $t {
            fn ntype() -> NType {
                NType::Int
            }
            fn into_slim(self) -> Result<Value, SlimError> {
                i64::try_from(self)
                    .map(Value::from_i64)
                    .map_err(|_| SlimError::new("OverflowError", format!("{self} does not fit in an int")))
            }
        }
    )*};
}
int_conv!(i8, i16, i32, i64, u8, u16, u32, u64, isize, usize);

macro_rules! simple_conv {
    ($t:ty, $nt:expr, |$v:ident| $from:expr, |$s:ident| $into:expr) => {
        impl SlimType for $t {
            fn ntype() -> NType {
                $nt
            }
        }
        impl FromSlim for $t {
            fn from_slim($v: Value) -> Result<Self, SlimError> {
                $from.ok_or_else(|| expected($nt, $v))
            }
        }
        impl IntoSlim for $t {
            fn ntype() -> NType {
                $nt
            }
            fn into_slim(self) -> Result<Value, SlimError> {
                let $s = self;
                Ok($into)
            }
        }
    };
}
simple_conv!(f64, NType::Float, |v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)), |s| Value::float(s));
simple_conv!(f32, NType::Float, |v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)).map(|f| f as f32), |s| Value::float(s as f64));
simple_conv!(bool, NType::Bool, |v| v.as_bool(), |s| Value::bool(s));
simple_conv!(String, NType::Str, |v| v.as_str().map(String::from), |s| Value::str(s));

impl IntoSlim for &str {
    fn ntype() -> NType {
        NType::Str
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        Ok(Value::str(self))
    }
}

impl IntoSlim for () {
    fn ntype() -> NType {
        NType::Unit
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        Ok(Value::UNIT)
    }
}

impl<T: SlimType> SlimType for Option<T> {
    fn ntype() -> NType {
        NType::Option(Box::new(T::ntype()))
    }
}

impl<T: FromSlim> FromSlim for Option<T> {
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        if v == Value::NONE {
            return Ok(None);
        }
        T::from_slim(v).map(Some)
    }
}

impl<T: IntoSlim> IntoSlim for Option<T> {
    fn ntype() -> NType {
        NType::Option(Box::new(T::ntype()))
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        match self {
            Some(t) => t.into_slim(),
            None => Ok(Value::NONE),
        }
    }
}

impl<T: IntoSlim> IntoSlim for Result<T, SlimError> {
    fn ntype() -> NType {
        T::ntype()
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        self?.into_slim()
    }
}

/// The value of a list or dict, checked against its exact runtime type.
fn typed_container(v: Value, t: &NType) -> Result<(), SlimError> {
    match rtype::container_type(v) {
        Some(ct) if ct.value() == t.rt() => Ok(()),
        _ => Err(expected(t.clone(), v)),
    }
}

impl<T: SlimType> SlimType for Vec<T> {
    fn ntype() -> NType {
        NType::List(Box::new(T::ntype()))
    }
}

impl<T: FromSlim> FromSlim for Vec<T> {
    /// Copies a `list<T>` (O(n)); `ListRef<T>` reads it in place.
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        typed_container(v, &<Self as SlimType>::ntype())?;
        let Some(Body::List(l)) = v.body() else { unreachable!() };
        l.snapshot().into_iter().map(T::from_slim).collect()
    }
}

impl<T: IntoSlim> IntoSlim for Vec<T> {
    fn ntype() -> NType {
        NType::List(Box::new(T::ntype()))
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        let mut items = Vec::with_capacity(self.len());
        for x in self {
            match x.into_slim() {
                Ok(v) => items.push(v),
                Err(e) => {
                    items.into_iter().for_each(|v| v.decref());
                    return Err(e);
                }
            }
        }
        let ty = <Self as IntoSlim>::ntype().rt();
        if items.iter().any(|v| v.is_local()) && !rtype::as_type(ty).unwrap().local {
            items.into_iter().for_each(|v| v.decref());
            return Err(SlimError::type_error("cannot put values that are not `Send` in a `Send` list"));
        }
        Ok(Obj::list(ty, items))
    }
}

// ---------------------------------------------------------------- handles

/// A counted reference to a slim value, valid on the thread that got it.
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

/// Any slim value except `None` (use `Option<Any>` to accept `None`).
#[derive(Clone)]
pub struct Any(Handle);

impl Any {
    pub fn type_name(&self) -> String {
        crate::types::value_type_name(self.0.v)
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
        crate::display::to_str(self.0.v)
    }
}

impl SlimType for Any {
    fn ntype() -> NType {
        NType::Any
    }
}

impl FromSlim for Any {
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        if v == Value::NONE || v.is_dead() {
            return Err(expected(NType::Any, v));
        }
        Ok(Any(Handle::borrow(v)))
    }
}

impl IntoSlim for Any {
    fn ntype() -> NType {
        NType::Any
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        Ok(self.0.into_value())
    }
}

/// A slim function (or other callable), called with [`Ctx::call`].
#[derive(Clone)]
pub struct Func(Handle);

impl SlimType for Func {
    fn ntype() -> NType {
        NType::Function
    }
}

impl FromSlim for Func {
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        // A local closure is fine: the handle is `!Send`, so it stays on
        // this thread (the parameter is typed `function`, which may hold values that are not `Send`).
        if !rtype::has_shape(v, rtype::as_type(rtype::prim(Ctor::Function)).unwrap()) {
            return Err(expected(NType::Function, v));
        }
        Ok(Func(Handle::borrow(v)))
    }
}

impl IntoSlim for Func {
    fn ntype() -> NType {
        NType::Function
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        Ok(self.0.into_value())
    }
}

/// A `list<T>` read and written in place. Elements are converted one at a
/// time; stores are checked like slim stores (types and locality).
pub struct ListRef<T> {
    h: Handle,
    _t: PhantomData<T>,
}

impl<T> Clone for ListRef<T> {
    fn clone(&self) -> Self {
        ListRef { h: self.h.clone(), _t: PhantomData }
    }
}

impl<T: FromSlim + IntoSlim> ListRef<T> {
    /// A new, empty `list<T>`.
    pub fn new() -> ListRef<T> {
        ListRef { h: Handle::owned(Obj::list(<Self as SlimType>::ntype().rt(), Vec::new())), _t: PhantomData }
    }

    fn list(&self) -> &'static crate::list::ListObj {
        match self.h.v.body() {
            Some(Body::List(l)) => l,
            _ => unreachable!(),
        }
    }

    pub fn len(&self) -> usize {
        self.list().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element `i`, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<T> {
        // The element is borrowed only while it is converted.
        self.list().get(i).map(|v| T::from_slim(v).expect("list element of its declared type"))
    }

    /// All elements, converted.
    pub fn to_vec(&self) -> Vec<T> {
        self.list().snapshot().into_iter().map(|v| T::from_slim(v).expect("list element of its declared type")).collect()
    }

    pub fn push(&self, x: T) -> Result<(), SlimError> {
        let v = self.admit(x)?;
        self.list().push(v);
        Ok(())
    }

    /// Replaces element `i` (an `IndexError` past the end).
    pub fn set(&self, i: usize, x: T) -> Result<(), SlimError> {
        let v = self.admit(x)?;
        match self.list().write(|w| (i < w.len()).then(|| w.set(i, v))) {
            Some(old) => {
                crate::rc::defer_decref(old);
                Ok(())
            }
            None => {
                v.decref();
                Err(SlimError::new("IndexError", format!("list index {i} out of range")))
            }
        }
    }

    fn admit(&self, x: T) -> Result<Value, SlimError> {
        let v = x.into_slim()?;
        let lt = rtype::as_type(self.list().ty).unwrap();
        let et = rtype::as_type(rtype::elem_of(lt)).unwrap();
        rtype::admit(v, et).ok_or_else(|| {
            let e = SlimError::type_error(format!("cannot store `{}` in `{}`", crate::types::value_type_name(v), lt.name));
            v.decref();
            e
        })
    }
}

impl<T: FromSlim + IntoSlim> Default for ListRef<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SlimType> SlimType for ListRef<T> {
    fn ntype() -> NType {
        NType::List(Box::new(T::ntype()))
    }
}

impl<T: SlimType> FromSlim for ListRef<T> {
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        typed_container(v, &<Self as SlimType>::ntype())?;
        Ok(ListRef { h: Handle::borrow(v), _t: PhantomData })
    }
}

impl<T: SlimType> IntoSlim for ListRef<T> {
    fn ntype() -> NType {
        <Self as SlimType>::ntype()
    }
    fn into_slim(self) -> Result<Value, SlimError> {
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

impl<K: FromSlim + IntoSlim, V: FromSlim + IntoSlim> DictRef<K, V> {
    fn dict(&self) -> &'static crate::dict::Dict {
        match self.h.v.body() {
            Some(Body::Dict(d)) => d,
            _ => unreachable!(),
        }
    }

    pub fn len(&self) -> usize {
        self.dict().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, k: K) -> Result<Option<V>, SlimError> {
        let kv = k.into_slim()?;
        let r = self.dict().get(kv).map(|v| V::from_slim(v).expect("dict value of its declared type"));
        kv.decref();
        Ok(r)
    }

    pub fn insert(&self, k: K, v: V) -> Result<(), SlimError> {
        let dt = rtype::as_type(self.dict().ty).unwrap();
        let (kt, vt) = crate::types::dict_kv(dt);
        let kv = k.into_slim()?;
        let vv = match v.into_slim() {
            Ok(v) => v,
            Err(e) => {
                kv.decref();
                return Err(e);
            }
        };
        let (Some(kv), Some(vv)) = (rtype::admit(kv, kt), rtype::admit(vv, vt)) else {
            kv.decref();
            vv.decref();
            return Err(SlimError::type_error(format!("cannot store this entry in `{}`", dt.name)));
        };
        match self.dict().insert(kv, vv) {
            Ok(Some((dup, old))) => {
                dup.decref();
                crate::rc::defer_decref(old);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(()) => Err(SlimError::type_error("unhashable key")),
        }
    }

    pub fn keys(&self) -> Vec<K> {
        self.dict().entries().into_iter().map(|(k, _)| K::from_slim(k).expect("dict key of its declared type")).collect()
    }
}

impl<K: SlimType, V: SlimType> SlimType for DictRef<K, V> {
    fn ntype() -> NType {
        NType::Dict(Box::new(K::ntype()), Box::new(V::ntype()))
    }
}

impl<K: SlimType, V: SlimType> FromSlim for DictRef<K, V> {
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        typed_container(v, &<Self as SlimType>::ntype())?;
        Ok(DictRef { h: Handle::borrow(v), _t: PhantomData })
    }
}

impl<K: SlimType, V: SlimType> IntoSlim for DictRef<K, V> {
    fn ntype() -> NType {
        <Self as SlimType>::ntype()
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        Ok(self.h.into_value())
    }
}

/// An element type of `array<T>`.
pub trait ArrayElem: Copy {
    const ELEM: Elem;
    fn from_scalar(s: Scalar) -> Self;
    fn to_value(self) -> Value;
}

macro_rules! array_elem {
    ($($t:ty => $e:ident),*) => {$(
        impl ArrayElem for $t {
            const ELEM: Elem = Elem::$e;
            fn from_scalar(s: Scalar) -> Self {
                match s {
                    Scalar::Int(i) => i as $t,
                    Scalar::Float(f) => f as $t,
                    Scalar::Bool(b) => b as u8 as $t,
                }
            }
            fn to_value(self) -> Value {
                if Elem::$e.is_float() {
                    Value::float(self as f64)
                } else {
                    Value::from_i64(self as i64)
                }
            }
        }
    )*};
}
array_elem!(i8 => I8, i16 => I16, i32 => I32, i64 => I64, u8 => U8, u16 => U16, u32 => U32, u64 => U64, f32 => F32, f64 => F64);

impl ArrayElem for bool {
    const ELEM: Elem = Elem::Bool;
    fn from_scalar(s: Scalar) -> Self {
        matches!(s, Scalar::Bool(true))
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
    fn arr(&self) -> &'static crate::array::ArrayObj {
        self.h.v.as_array().unwrap()
    }

    pub fn len(&self) -> usize {
        self.arr().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<T> {
        self.arr().get_num(i).map(T::from_scalar)
    }

    pub fn set(&self, i: usize, x: T) -> Result<(), SlimError> {
        let a = self.arr();
        if i >= a.len() {
            return Err(SlimError::new("IndexError", format!("array index {i} out of range")));
        }
        let v = x.to_value();
        let r = a.set(i, v);
        v.decref();
        r.map_err(|_| SlimError::new("OverflowError", format!("value out of range for `{}`", T::ELEM.name())))
    }

    pub fn to_vec(&self) -> Vec<T> {
        (0..self.len()).filter_map(|i| self.get(i)).collect()
    }
}

impl<T: ArrayElem> SlimType for ArrayRef<T> {
    fn ntype() -> NType {
        NType::Array(T::ELEM)
    }
}

impl<T: ArrayElem> FromSlim for ArrayRef<T> {
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        match v.as_array() {
            Some(a) if a.elem == T::ELEM => Ok(ArrayRef { h: Handle::borrow(v), _t: PhantomData }),
            _ => Err(expected(NType::Array(T::ELEM), v)),
        }
    }
}

impl<T: ArrayElem> IntoSlim for ArrayRef<T> {
    fn ntype() -> NType {
        NType::Array(T::ELEM)
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        Ok(self.h.into_value())
    }
}

// ---------------------------------------------------------------- errors

/// An error raised in slim as an exception of class `kind`.
pub struct SlimError {
    pub kind: String,
    pub msg: String,
    /// A slim exception raised by a callback, re-raised unchanged.
    pending: Option<PendingExc>,
}

struct PendingExc(ExcInfo);

impl Drop for PendingExc {
    fn drop(&mut self) {
        self.0.value.decref();
        self.0.cause.decref();
    }
}

impl SlimError {
    pub fn new(kind: &str, msg: impl Into<String>) -> SlimError {
        SlimError { kind: kind.into(), msg: msg.into(), pending: None }
    }

    pub fn type_error(msg: impl Into<String>) -> SlimError {
        SlimError::new("TypeError", msg)
    }

    pub fn value_error(msg: impl Into<String>) -> SlimError {
        SlimError::new("ValueError", msg)
    }

    pub fn runtime(msg: impl Into<String>) -> SlimError {
        SlimError::new("RuntimeError", msg)
    }

    /// Takes the thread's current exception.
    fn from_thread(th: &mut Thread) -> SlimError {
        let e = th.exc.take().expect("raised without an exception");
        let kind = e.value.type_name();
        let msg = e.value.as_instance().and_then(|i| i.class().field_index("msg").map(|k| i.get_borrowed(k)));
        let msg = msg.map(crate::display::to_str).unwrap_or_default();
        SlimError { kind, msg, pending: Some(PendingExc(e)) }
    }

    /// Raises the error in `th`.
    fn raise(mut self, th: &mut Thread) -> Raised {
        if let Some(p) = self.pending.take() {
            let p = std::mem::ManuallyDrop::new(p);
            // SAFETY: `p` is not used or dropped again.
            th.exc = Some(unsafe { std::ptr::read(&p.0) });
            return Raised;
        }
        let kind = if th.vm.builtins.has_class(&self.kind) { self.kind.as_str() } else { "RuntimeError" };
        th.raise(kind, std::mem::take(&mut self.msg))
    }

    fn context(mut self, what: &str) -> SlimError {
        if self.pending.is_none() {
            self.msg = format!("{what}: {}", self.msg);
        }
        self
    }
}

impl std::fmt::Display for SlimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
}

impl std::fmt::Debug for SlimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl From<String> for SlimError {
    fn from(msg: String) -> SlimError {
        SlimError::runtime(msg)
    }
}

impl From<&str> for SlimError {
    fn from(msg: &str) -> SlimError {
        SlimError::runtime(msg)
    }
}

// ---------------------------------------------------------------- context

/// The interpreter, as seen by a native call.
pub struct Ctx {
    th: *mut Thread,
    _not_send: PhantomData<*const ()>,
}

impl Ctx {
    fn th(&mut self) -> &mut Thread {
        // SAFETY: a `Ctx` only exists during a native call on this thread,
        // whose `&mut Thread` it was made from.
        unsafe { &mut *self.th }
    }

    /// Calls a slim function. Arguments are passed as new references.
    pub fn call<R: FromSlim>(&mut self, f: &Func, args: impl IntoArgs) -> Result<R, SlimError> {
        let vals = args.into_values()?;
        let th = self.th();
        let r = crate::helpers::call_simple(th, f.0.v, None, &vals);
        vals.into_iter().for_each(|v| v.decref());
        match r {
            Ok(v) => {
                let out = R::from_slim(v);
                v.decref();
                out
            }
            Err(Raised) => Err(SlimError::from_thread(th)),
        }
    }

    /// Runs `f` without access to slim values (they cannot be captured: they
    /// are not `Send`), letting other threads reclaim memory meanwhile. Use
    /// it around long or blocking work.
    pub fn offline<R: Send>(&mut self, f: impl FnOnce() -> R + Send) -> R {
        crate::rc::offline();
        let r = f();
        crate::rc::online();
        r
    }
}

/// Arguments of [`Ctx::call`]: a tuple of convertible values.
pub trait IntoArgs {
    fn into_values(self) -> Result<Vec<Value>, SlimError>;
}

macro_rules! into_args {
    ($($A:ident),*) => {
        impl<$($A: IntoSlim),*> IntoArgs for ($($A,)*) {
            #[allow(non_snake_case, unused_mut)]
            fn into_values(self) -> Result<Vec<Value>, SlimError> {
                let ($($A,)*) = self;
                let mut out: Vec<Value> = Vec::new();
                $(
                    match $A.into_slim() {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            out.into_iter().for_each(|v| v.decref());
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

type NativeBody = Box<dyn Fn(&mut Ctx, &[Value]) -> Result<Value, SlimError> + Send + Sync>;
/// Formats a native object for `str()`/`print`.
type ReprFn = Box<dyn Fn(&dyn std::any::Any) -> String + Send + Sync>;

/// A native function or method (an immortal object).
pub struct NativeFn {
    pub name: String,
    /// Parameter types (after `self` for a method).
    pub params: Vec<NType>,
    pub ret: NType,
    /// A method: the first argument is the object.
    pub has_self: bool,
    f: NativeBody,
}

impl NativeFn {
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
    let mut ctx = Ctx { th: th as *mut Thread, _not_send: PhantomData };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (f.f)(&mut ctx, args)));
    match r {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.raise(th)),
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            Err(th.raise("RuntimeError", format!("native panic in {}(): {msg}", f.name)))
        }
    }
}

fn arg<A: FromSlim>(vals: &[Value], i: usize, name: &str) -> Result<A, SlimError> {
    A::from_slim(vals[i]).map_err(|e| e.context(&format!("argument {} of {name}()", i + 1)))
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
            R: IntoSlim,
            $($A: FromSlim,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_native(self, name: &str) -> NativeFn {
                let n = name.to_string();
                NativeFn {
                    name: name.into(),
                    params: vec![$($A::ntype()),*],
                    ret: R::ntype(),
                    has_self: false,
                    f: Box::new(move |_ctx, vals| {
                        $(let $A: $A = arg(vals, $i, &n)?;)*
                        self($($A),*).into_slim()
                    }),
                }
            }
        }

        impl<F, R, $($A,)*> IntoNative<WithCtx<($($A,)*)>> for F
        where
            F: Fn(&mut Ctx, $($A),*) -> R + Send + Sync + 'static,
            R: IntoSlim,
            $($A: FromSlim,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_native(self, name: &str) -> NativeFn {
                let n = name.to_string();
                NativeFn {
                    name: name.into(),
                    params: vec![$($A::ntype()),*],
                    ret: R::ntype(),
                    has_self: false,
                    f: Box::new(move |ctx, vals| {
                        $(let $A: $A = arg(vals, $i, &n)?;)*
                        self(ctx, $($A),*).into_slim()
                    }),
                }
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

/// A native class (an immortal object), or the family of a generic one.
pub struct NativeClass {
    pub name: String,
    pub module: String,
    pub type_id: TypeId,
    pub mode: Mode,
    /// Conflicting calls wait instead of raising `BorrowError`.
    pub locked: bool,
    pub init: Option<Value>,
    pub methods: HashMap<String, Value>,
    repr: Option<ReprFn>,
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

/// Marks generic families, which wrap no Rust type.
struct Family;

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
    data: UnsafeCell<Option<Box<dyn std::any::Any>>>,
}

fn thread_tag() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local!(static TAG: u64 = NEXT.fetch_add(1, Ordering::Relaxed));
    TAG.with(|t| *t)
}

/// A running method call's access to a native object.
struct Borrow<'a> {
    obj: &'a NativeObj,
    exclusive: bool,
}

impl Drop for Borrow<'_> {
    fn drop(&mut self) {
        if self.exclusive {
            self.obj.holder.store(0, Ordering::Relaxed);
            self.obj.borrow.store(0, Ordering::Release);
        } else {
            self.obj.borrow.fetch_sub(1, Ordering::Release);
        }
    }
}

impl NativeObj {
    pub fn class(&self) -> &'static NativeClass {
        match self.class.body() {
            Some(Body::NativeClass(c)) => c,
            _ => unreachable!(),
        }
    }

    fn enter(&self, exclusive: bool) -> Result<Borrow<'_>, SlimError> {
        let c = self.class();
        if c.mode == Mode::Local && std::thread::current().id() != self.owner {
            return Err(SlimError::new("ThreadError", format!("`{}` object belongs to another thread (it is not `Send`)", c.name)));
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
                    return Ok(Borrow { obj: self, exclusive });
                }
                continue;
            }
            let reentrant = cur == EXCL && self.holder.load(Ordering::Relaxed) == me;
            if !c.locked || reentrant || c.mode == Mode::Local {
                let what = if cur == EXCL { "a call that needs it exclusively" } else { "another call" };
                return Err(SlimError::new("BorrowError", format!("`{}` object is in use by {what}", c.name)));
            }
            // Wait without blocking memory reclamation on other threads.
            crate::rc::offline();
            std::thread::yield_now();
            crate::rc::online();
        }
    }

    fn data<T: 'static>(&self, _b: &Borrow<'_>) -> &T {
        // SAFETY: a borrow is held; shared borrows only coexist for `Sync`
        // types (or on the owner thread for local ones).
        unsafe { (*self.data.get()).as_ref().expect("live native object").downcast_ref::<T>().expect("native type") }
    }

    #[allow(clippy::mut_from_ref)]
    fn data_mut<T: 'static>(&self, b: &Borrow<'_>) -> &mut T {
        debug_assert!(b.exclusive);
        // SAFETY: an exclusive borrow is held.
        unsafe { (*self.data.get()).as_mut().expect("live native object").downcast_mut::<T>().expect("native type") }
    }

    /// Drops the Rust value (when the object is freed). A local object is
    /// only dropped on its owner thread: locality guarantees it is freed
    /// there, so anything else is a VM bug. Debug builds panic; release
    /// builds leak the value with a warning (dropping it would be unsound).
    pub(crate) fn release(&mut self) {
        let data = self.data.get_mut().take();
        if self.class().mode == Mode::Local && std::thread::current().id() != self.owner {
            // Forgotten first, so a panic cannot drop it on this thread.
            std::mem::forget(data);
            let name = &self.class().name;
            if cfg!(debug_assertions) {
                panic!("native `{name}` object freed off its owner thread");
            }
            eprintln!("warning: native `{name}` object freed off its owner thread; leaked");
        }
    }

    pub(crate) fn display(&self) -> String {
        let c = self.class();
        if let Some(r) = &c.repr {
            if let Ok(b) = self.enter(false) {
                // SAFETY: see `data`.
                let d = unsafe { (*self.data.get()).as_ref().map(|d| r(&**d)) };
                drop(b);
                if let Some(s) = d {
                    return s;
                }
            }
        }
        format!("<{} object>", rtype::name(self.ty))
    }
}

fn native_obj(v: Value) -> Option<&'static NativeObj> {
    match v.body() {
        Some(Body::Native(n)) => Some(n),
        _ => None,
    }
}

static REGISTRY: OnceLock<RwLock<HashMap<TypeId, Value>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<TypeId, Value>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The class registered for Rust type `id`.
pub fn class_by_type(id: TypeId) -> Option<Value> {
    registry().read().unwrap().get(&id).copied()
}

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

impl<T: 'static> Native<T> {
    /// Wraps `t` in a new object of its registered class.
    pub fn new(t: T) -> Result<Native<T>, SlimError> {
        let class = class_by_type(TypeId::of::<T>())
            .ok_or_else(|| SlimError::runtime(format!("no native class is registered for `{}`", std::any::type_name::<T>())))?;
        let obj = NativeObj {
            class,
            ty: rtype::class_type(class),
            owner: std::thread::current().id(),
            borrow: AtomicU32::new(0),
            holder: AtomicU64::new(0),
            data: UnsafeCell::new(Some(Box::new(t))),
        };
        Ok(Native { h: Handle::owned(Obj::alloc(Body::Native(obj))), _t: PhantomData })
    }

    fn obj(&self) -> &'static NativeObj {
        native_obj(self.h.v).unwrap()
    }

    /// Runs `f` with shared access (`BorrowError` while an exclusive call runs).
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> Result<R, SlimError> {
        let o = self.obj();
        let b = o.enter(false)?;
        Ok(f(o.data::<T>(&b)))
    }

    /// Runs `f` with exclusive access (`BorrowError` while another call runs).
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> Result<R, SlimError> {
        let o = self.obj();
        let b = o.enter(true)?;
        Ok(f(o.data_mut::<T>(&b)))
    }
}

impl<T: 'static> SlimType for Native<T> {
    fn ntype() -> NType {
        NType::Class(TypeId::of::<T>())
    }
}

impl<T: 'static> FromSlim for Native<T> {
    fn from_slim(v: Value) -> Result<Self, SlimError> {
        match native_obj(v) {
            Some(o) if o.class().type_id == TypeId::of::<T>() => Ok(Native { h: Handle::borrow(v), _t: PhantomData }),
            _ => Err(expected(<Self as SlimType>::ntype(), v)),
        }
    }
}

impl<T: 'static> IntoSlim for Native<T> {
    fn ntype() -> NType {
        NType::Class(TypeId::of::<T>())
    }
    fn into_slim(self) -> Result<Value, SlimError> {
        Ok(self.h.into_value())
    }
}

/// The type arguments of a generic class's instantiation: a [`SlimType`]
/// for one parameter, a tuple of them for several.
pub trait TypeArgs {
    fn ntypes() -> Vec<NType>;
}

impl<T: SlimType> TypeArgs for T {
    fn ntypes() -> Vec<NType> {
        vec![T::ntype()]
    }
}

impl<A: SlimType, B: SlimType> TypeArgs for (A, B) {
    fn ntypes() -> Vec<NType> {
        vec![A::ntype(), B::ntype()]
    }
}

impl<A: SlimType, B: SlimType, C: SlimType> TypeArgs for (A, B, C) {
    fn ntypes() -> Vec<NType> {
        vec![A::ntype(), B::ntype(), C::ntype()]
    }
}

/// Result of a constructor: `T` or `Result<T, SlimError>`.
pub trait CtorOut<T> {
    fn into_result(self) -> Result<T, SlimError>;
}

impl<T> CtorOut<T> for T {
    fn into_result(self) -> Result<T, SlimError> {
        Ok(self)
    }
}

impl<T> CtorOut<T> for Result<T, SlimError> {
    fn into_result(self) -> Result<T, SlimError> {
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

fn self_obj<T: 'static>(v: Value, what: &str) -> Result<&'static NativeObj, SlimError> {
    match native_obj(v) {
        Some(o) if o.class().type_id == TypeId::of::<T>() => Ok(o),
        _ => Err(SlimError::type_error(format!("{what}() called on a `{}`", crate::types::value_type_name(v)))),
    }
}

macro_rules! impl_class_fns {
    ($($A:ident $i:tt),*) => {
        impl<F, O, T, $($A,)*> IntoCtor<T, ($($A,)*)> for F
        where
            F: Fn($($A),*) -> O + Send + Sync + 'static,
            O: CtorOut<T>,
            T: 'static,
            $($A: FromSlim,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_ctor(self, name: &str) -> NativeFn {
                let n = name.to_string();
                NativeFn {
                    name: name.into(),
                    params: vec![$($A::ntype()),*],
                    ret: NType::Class(TypeId::of::<T>()),
                    has_self: false,
                    f: Box::new(move |_ctx, vals| {
                        $(let $A: $A = arg(vals, $i, &n)?;)*
                        Native::new(self($($A),*).into_result()?)?.into_slim()
                    }),
                }
            }
        }

        impl<F, R, T, $($A,)*> IntoMethod<T, ($($A,)*)> for F
        where
            F: Fn(&T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoSlim,
            T: 'static,
            $($A: FromSlim,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                NativeFn {
                    name: n.clone(),
                    params: vec![$($A::ntype()),*],
                    ret: R::ntype(),
                    has_self: true,
                    f: Box::new(move |_ctx, vals| {
                        let o = self_obj::<T>(vals[0], &n)?;
                        let rest = &vals[1..];
                        $(let $A: $A = arg(rest, $i, &n)?;)*
                        let b = o.enter(false)?;
                        let r = self(o.data::<T>(&b), $($A),*);
                        drop(b);
                        r.into_slim()
                    }),
                }
            }
        }

        impl<F, R, T, $($A,)*> IntoMethod<T, WithCtx<($($A,)*)>> for F
        where
            F: Fn(&mut Ctx, &T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoSlim,
            T: 'static,
            $($A: FromSlim,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                NativeFn {
                    name: n.clone(),
                    params: vec![$($A::ntype()),*],
                    ret: R::ntype(),
                    has_self: true,
                    f: Box::new(move |ctx, vals| {
                        let o = self_obj::<T>(vals[0], &n)?;
                        let rest = &vals[1..];
                        $(let $A: $A = arg(rest, $i, &n)?;)*
                        let b = o.enter(false)?;
                        let r = self(ctx, o.data::<T>(&b), $($A),*);
                        drop(b);
                        r.into_slim()
                    }),
                }
            }
        }

        impl<F, R, T, $($A,)*> IntoMethodMut<T, ($($A,)*)> for F
        where
            F: Fn(&mut T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoSlim,
            T: 'static,
            $($A: FromSlim,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                NativeFn {
                    name: n.clone(),
                    params: vec![$($A::ntype()),*],
                    ret: R::ntype(),
                    has_self: true,
                    f: Box::new(move |_ctx, vals| {
                        let o = self_obj::<T>(vals[0], &n)?;
                        let rest = &vals[1..];
                        $(let $A: $A = arg(rest, $i, &n)?;)*
                        let b = o.enter(true)?;
                        let r = self(o.data_mut::<T>(&b), $($A),*);
                        drop(b);
                        r.into_slim()
                    }),
                }
            }
        }

        impl<F, R, T, $($A,)*> IntoMethodMut<T, WithCtx<($($A,)*)>> for F
        where
            F: Fn(&mut Ctx, &mut T, $($A),*) -> R + Send + Sync + 'static,
            R: IntoSlim,
            T: 'static,
            $($A: FromSlim,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn into_method(self, class: &str, name: &str) -> NativeFn {
                let n = format!("{class}.{name}");
                NativeFn {
                    name: n.clone(),
                    params: vec![$($A::ntype()),*],
                    ret: R::ntype(),
                    has_self: true,
                    f: Box::new(move |ctx, vals| {
                        let o = self_obj::<T>(vals[0], &n)?;
                        let rest = &vals[1..];
                        $(let $A: $A = arg(rest, $i, &n)?;)*
                        let b = o.enter(true)?;
                        let r = self(ctx, o.data_mut::<T>(&b), $($A),*);
                        drop(b);
                        r.into_slim()
                    }),
                }
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
    pub type_id: TypeId,
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

impl Module {
    pub fn new(name: &str) -> Module {
        Module { name: name.into(), functions: Vec::new(), classes: Vec::new(), traits: Vec::new() }
    }

    /// Adds a function: a closure `Fn(A1, ..) -> R` or `Fn(&mut Ctx, A1, ..) -> R`.
    pub fn function<Args, F: IntoNative<Args>>(&mut self, name: &str, f: F) -> &mut Self {
        let nf = f.into_native(name);
        self.functions.push((name.into(), Obj::alloc_immortal(Body::NativeFn(nf))));
        self
    }

    /// Adds a class whose methods may run concurrently (`T: Send + Sync`).
    pub fn class<T: Send + Sync + 'static>(&mut self, name: &str, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add_class(name, Mode::Sync, build)
    }

    /// Adds a class whose calls are exclusive (`T: Send`).
    pub fn class_send<T: Send + 'static>(&mut self, name: &str, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add_class(name, Mode::Send, build)
    }

    /// Adds a class over a Rust type that is not `Send` (any `T`): the slim
    /// class is not `Send` either, so instances stay on their thread.
    pub fn class_not_send<T: 'static>(&mut self, name: &str, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add_class(name, Mode::Local, build)
    }

    /// Adds a trait whose methods are all required, for classes of this
    /// module (defined after it) and slim classes to implement:
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
        let tr = Obj::alloc_immortal(Body::Trait(crate::object::Trait {
            name: name.into(),
            id: NEXT_TRAIT_ID.fetch_add(1, Ordering::Relaxed),
            nparams: 0,
            methods: t.methods.iter().map(|m| (m.name.clone(), None)).collect(),
            implies: Vec::new(),
            native_impls: Default::default(),
        }));
        self.traits.push((TraitSpec { name: name.into(), methods: t.methods }, tr));
        self
    }

    fn add_class<T: 'static>(&mut self, name: &str, mode: Mode, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        let class = new_class(&self.name, &self.traits, name, mode, Vec::new(), build);
        self.classes.push((name.into(), class));
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
    /// Constructing another instantiation from slim is a compile error when
    /// the type arguments are known statically, and a `TypeError` otherwise.
    pub fn generic_class(&mut self, name: &str, tparams: &[&str], build: impl FnOnce(&mut GenericBuilder)) -> &mut Self {
        assert!(!tparams.is_empty(), "generic class `{name}` needs type parameters");
        let mut g = GenericBuilder { module: self.name.clone(), traits: self.traits.clone(), name: name.into(), arity: tparams.len(), instances: Vec::new() };
        build(&mut g);
        let family = Obj::alloc_immortal(Body::NativeClass(NativeClass {
            name: name.into(),
            module: self.name.clone(),
            type_id: TypeId::of::<Family>(),
            mode: Mode::Sync,
            locked: false,
            init: None,
            methods: HashMap::new(),
            repr: None,
            canonical: OnceLock::new(),
            tparams: tparams.iter().map(|t| t.to_string()).collect(),
            instances: g.instances.clone(),
            family: OnceLock::new(),
            type_args: Vec::new(),
            traits: Vec::new(),
            builtin_traits: Vec::new(),
        }));
        for c in &g.instances {
            let Some(Body::NativeClass(c)) = c.body() else { unreachable!() };
            let _ = c.family.set(family);
        }
        self.classes.push((name.into(), family));
        self
    }

    /// The module's items as the compiler sees them.
    pub fn spec(&self) -> ModuleSpec {
        ModuleSpec {
            name: self.name.clone(),
            functions: self.functions.iter().map(|(n, v)| fn_spec(n, *v)).collect(),
            classes: self
                .classes
                .iter()
                .map(|(n, v)| class_spec(n, *v))
                .collect(),
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
        type_id: c.type_id,
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
/// builtin trait names. Panics if one is unknown or a method is missing or
/// has another signature.
fn class_traits(class: &str, names: &[String], traits: &Traits, methods: &HashMap<String, NativeFn>) -> (Vec<Value>, Vec<&'static str>) {
    let (mut objs, mut builtins) = (Vec::new(), Vec::new());
    for n in names {
        if let Some((spec, v)) = traits.iter().find(|(t, _)| t.name == *n) {
            for m in &spec.methods {
                let Some(f) = methods.get(&m.name) else { panic!("native class `{class}` implements `{n}` but does not define `{}`", m.name) };
                assert!(
                    f.params == m.params && f.ret == m.ret,
                    "native class `{class}` implements `{n}`, but `{}` takes {:?} -> {:?} instead of {:?} -> {:?}",
                    m.name,
                    f.params,
                    f.ret,
                    m.params,
                    m.ret
                );
            }
            objs.push(*v);
        } else if let Some((b, implies, required)) = BUILTIN_TRAITS.iter().find(|t| t.0 == n) {
            for m in *required {
                assert!(methods.contains_key(*m), "native class `{class}` implements `{n}` but does not define `{m}`");
            }
            for t in std::iter::once(b).chain(implies.iter()) {
                if !builtins.contains(t) {
                    builtins.push(*t);
                }
            }
        } else if n == "Send" {
            panic!("native class `{class}`: `Send` is not implemented explicitly; it follows the Rust type (`class` / `class_send` vs `class_not_send`)");
        } else {
            panic!("native class `{class}` implements unknown trait `{n}` (define it with `trait_def` before the class, or use `Ord`, `Num`, `Len`)");
        }
    }
    (objs, builtins)
}

/// A new native class over `T` (registered as `T`'s class).
fn new_class<T: 'static>(module: &str, traits: &Traits, name: &str, mode: Mode, type_args: Vec<NType>, build: impl FnOnce(&mut ClassBuilder<T>)) -> Value {
    let mut b = ClassBuilder::<T> { name: name.into(), locked: false, init: None, methods: HashMap::new(), repr: None, implements: Vec::new(), _t: PhantomData };
    build(&mut b);
    let (traits, builtin_traits) = class_traits(name, &b.implements, traits, &b.methods);
    let class = Obj::alloc_immortal(Body::NativeClass(NativeClass {
        name: name.into(),
        module: module.into(),
        type_id: TypeId::of::<T>(),
        mode,
        locked: b.locked,
        init: b.init.map(|f| Obj::alloc_immortal(Body::NativeFn(f))),
        methods: b.methods.into_iter().map(|(n, f)| (n, Obj::alloc_immortal(Body::NativeFn(f)))).collect(),
        repr: b.repr,
        canonical: OnceLock::new(),
        tparams: Vec::new(),
        instances: Vec::new(),
        family: OnceLock::new(),
        type_args,
        traits,
        builtin_traits,
    }));
    registry().write().unwrap().insert(TypeId::of::<T>(), class);
    class
}

/// Registers the instantiations of a generic class (see
/// [`Module::generic_class`]). `A` gives the type arguments ([`TypeArgs`]);
/// the Rust type's auto traits choose the mode as for [`Module::class`].
pub struct GenericBuilder {
    module: String,
    traits: Traits,
    name: String,
    arity: usize,
    instances: Vec<Value>,
}

impl GenericBuilder {
    /// An instantiation whose methods may run concurrently (`T: Send + Sync`).
    pub fn class<A: TypeArgs, T: Send + Sync + 'static>(&mut self, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add::<A, T>(Mode::Sync, build)
    }

    /// An instantiation whose calls are exclusive (`T: Send`).
    pub fn class_send<A: TypeArgs, T: Send + 'static>(&mut self, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add::<A, T>(Mode::Send, build)
    }

    /// An instantiation over a Rust type that is not `Send`: its instances
    /// are not `Send` either.
    pub fn class_not_send<A: TypeArgs, T: 'static>(&mut self, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        self.add::<A, T>(Mode::Local, build)
    }

    fn add<A: TypeArgs, T: 'static>(&mut self, mode: Mode, build: impl FnOnce(&mut ClassBuilder<T>)) -> &mut Self {
        let args = A::ntypes();
        assert!(args.len() == self.arity, "`{}` takes {} type argument(s), got {} for `{}`", self.name, self.arity, args.len(), std::any::type_name::<T>());
        let dup = self.instances.iter().any(|c| matches!(c.body(), Some(Body::NativeClass(c)) if c.type_args == args));
        assert!(!dup, "`{}` instantiation {args:?} is registered twice", self.name);
        let c = new_class(&self.module, &self.traits, &self.name, mode, args, build);
        self.instances.push(c);
        self
    }
}

/// Builds a native class (see [`Module::class`]).
pub struct ClassBuilder<T> {
    name: String,
    locked: bool,
    init: Option<NativeFn>,
    methods: HashMap<String, NativeFn>,
    repr: Option<ReprFn>,
    implements: Vec<String>,
    _t: PhantomData<fn() -> T>,
}

/// Builds a native trait (see [`Module::trait_def`]).
pub struct TraitBuilder {
    methods: Vec<FnSpec>,
}

impl TraitBuilder {
    /// A required method taking arguments `A` (a tuple: `()`, `(f64,)`, ...)
    /// and returning `R`.
    pub fn method<A: ArgTypes, R: IntoSlim>(&mut self, name: &str) -> &mut Self {
        self.methods.push(FnSpec { name: name.into(), params: A::ntypes(), ret: R::ntype() });
        self
    }
}

/// The parameter types of a native trait method: a tuple of [`SlimType`]s.
pub trait ArgTypes {
    fn ntypes() -> Vec<NType>;
}

macro_rules! impl_arg_types {
    ($($A:ident),*) => {
        impl<$($A: SlimType),*> ArgTypes for ($($A,)*) {
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
    /// `-> Result<T, SlimError>`).
    pub fn init<Args, F: IntoCtor<T, Args>>(&mut self, f: F) -> &mut Self {
        self.init = Some(f.into_ctor(&self.name));
        self
    }

    /// A method taking `&T`: `Fn(&T, A1, ..) -> R` or `Fn(&mut Ctx, &T, A1, ..) -> R`.
    pub fn method<Args, F: IntoMethod<T, Args>>(&mut self, name: &str, f: F) -> &mut Self {
        let m = f.into_method(&self.name, name);
        self.methods.insert(name.into(), m);
        self
    }

    /// A method taking `&mut T` (exclusive): `Fn(&mut T, A1, ..) -> R` or
    /// `Fn(&mut Ctx, &mut T, A1, ..) -> R`.
    pub fn method_mut<Args, F: IntoMethodMut<T, Args>>(&mut self, name: &str, f: F) -> &mut Self {
        let m = f.into_method(&self.name, name);
        self.methods.insert(name.into(), m);
        self
    }

    /// How instances are shown by `print` and `str`.
    pub fn repr(&mut self, f: impl Fn(&T) -> String + Send + Sync + 'static) -> &mut Self {
        self.repr = Some(Box::new(move |d| f(d.downcast_ref::<T>().expect("native type"))));
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
