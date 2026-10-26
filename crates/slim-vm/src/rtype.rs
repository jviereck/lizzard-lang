//! Runtime types: interned, immortal type objects.
//!
//! Every container carries the full type it was created with (`list<int>`,
//! `dict<str, list<int>>`), so generics are never erased. Types are
//! hash-consed: two types are equal exactly when their values are, which
//! makes the common checks a pointer compare. Types are immortal, so they are
//! copied freely without reference counting.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::array::Elem;
use crate::object::{Body, Obj};
use crate::value::Value;

/// A type constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ctor {
    /// Any value except `None`.
    Any,
    Int,
    Float,
    Bool,
    BigInt,
    Str,
    Unit,
    /// `tuple` (any tuple) or `tuple<A, B, ...>`.
    Tuple,
    /// `list` (any list) or `list<T>`.
    List,
    /// `dict` (any dict) or `dict<K, V>`.
    Dict,
    /// `Option<T>`: `None` or a `T`.
    Option,
    /// `array` (any element kind) or `array<T>`.
    Array(Option<Elem>),
    Function,
    /// `Iterator` (any iterator) or `Iterator<T>`.
    Iterator,
    /// `Channel` (any channel) or `Channel<T>`.
    Channel,
    Lock,
    Thread,
    /// Instances of a class (the immortal class object).
    Class(Value),
    /// Values implementing a trait (the immortal trait object).
    Trait(Value),
}

pub struct RtType {
    pub ctor: Ctor,
    /// Type arguments (type values). Empty for scalars and for a bare
    /// container type, which stands for any container of that kind.
    pub args: Box<[Value]>,
    pub name: String,
    /// The type's own value (set when it is interned).
    this: OnceLock<Value>,
    list_of: OnceLock<Value>,
    option_of: OnceLock<Value>,
}

impl RtType {
    /// The type as a value.
    pub fn value(&self) -> Value {
        *self.this.get().expect("interned type")
    }

    /// The `i`th type argument.
    pub fn arg(&self, i: usize) -> &'static RtType {
        as_type(self.args[i]).expect("type argument is a type")
    }

    pub fn is_bare(&self) -> bool {
        self.args.is_empty()
    }
}

type Table = RwLock<HashMap<(Ctor, Box<[u64]>), Value>>;

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The unique type value for `ctor<args>`.
pub fn intern(ctor: Ctor, args: &[Value]) -> Value {
    let key = (ctor, args.iter().map(|a| a.0).collect::<Box<[u64]>>());
    if let Some(v) = table().read().unwrap().get(&key) {
        return *v;
    }
    let mut t = table().write().unwrap();
    if let Some(v) = t.get(&key) {
        return *v;
    }
    let v = Obj::alloc_immortal(Body::Type(RtType {
        ctor,
        args: args.into(),
        name: type_name(ctor, args),
        this: OnceLock::new(),
        list_of: OnceLock::new(),
        option_of: OnceLock::new(),
    }));
    let _ = as_type(v).unwrap().this.set(v);
    t.insert(key, v);
    v
}

fn type_name(ctor: Ctor, args: &[Value]) -> String {
    let base = match ctor {
        Ctor::Any => "any",
        Ctor::Int => "int",
        Ctor::Float => "float",
        Ctor::Bool => "bool",
        Ctor::BigInt => "bigint",
        Ctor::Str => "str",
        Ctor::Unit => return "()".into(),
        Ctor::Tuple => "tuple",
        Ctor::List => "list",
        Ctor::Dict => "dict",
        Ctor::Option => "Option",
        Ctor::Array(None) => "array",
        Ctor::Array(Some(e)) => return format!("array<{}>", e.name()),
        Ctor::Function => "function",
        Ctor::Iterator => "Iterator",
        Ctor::Channel => "Channel",
        Ctor::Lock => "Lock",
        Ctor::Thread => "Thread",
        Ctor::Class(c) => return c.as_class().map(|c| c.name.clone()).unwrap_or_default(),
        Ctor::Trait(t) => return t.as_trait().map(|t| t.name.clone()).unwrap_or_default(),
    };
    if args.is_empty() {
        return base.into();
    }
    let parts: Vec<&str> = args.iter().map(|a| as_type(*a).map(|t| t.name.as_str()).unwrap_or("?")).collect();
    format!("{base}<{}>", parts.join(", "))
}

pub fn as_type(v: Value) -> Option<&'static RtType> {
    match v.body() {
        Some(Body::Type(t)) => Some(t),
        _ => None,
    }
}

/// The name of a type value.
pub fn name(t: Value) -> &'static str {
    as_type(t).map(|t| t.name.as_str()).unwrap_or("?")
}

pub fn prim(c: Ctor) -> Value {
    intern(c, &[])
}

/// `list<elem>`.
pub fn list_of(elem: Value) -> Value {
    *as_type(elem).unwrap().list_of.get_or_init(|| intern(Ctor::List, &[elem]))
}

/// `Option<inner>`.
pub fn option_of(inner: Value) -> Value {
    *as_type(inner).unwrap().option_of.get_or_init(|| intern(Ctor::Option, &[inner]))
}

pub fn dict_of(k: Value, v: Value) -> Value {
    intern(Ctor::Dict, &[k, v])
}

pub fn tuple_of(items: &[Value]) -> Value {
    intern(Ctor::Tuple, items)
}

pub fn iterator_of(item: Value) -> Value {
    intern(Ctor::Iterator, &[item])
}

macro_rules! cached_prims {
    ($($f:ident => $c:expr),* $(,)?) => {
        $(
            pub fn $f() -> Value {
                static V: OnceLock<Value> = OnceLock::new();
                *V.get_or_init(|| prim($c))
            }
        )*
    };
}

cached_prims! {
    any => Ctor::Any,
    int => Ctor::Int,
    float => Ctor::Float,
    bool_ => Ctor::Bool,
    str_ => Ctor::Str,
}

/// The element (`list`), value (`dict`) or item (`Iterator`, `Channel`) type
/// of a container type, or `any` for a bare one.
pub fn elem_of(t: &RtType) -> Value {
    match (t.ctor, t.args.len()) {
        (Ctor::List | Ctor::Iterator | Ctor::Channel, 1) => t.args[0],
        (Ctor::Dict, 2) => t.args[1],
        _ => any(),
    }
}

/// The scalar type of array elements of kind `e`.
pub fn elem_scalar(e: Elem) -> Value {
    match e {
        Elem::F32 | Elem::F64 => float(),
        Elem::Bool => bool_(),
        _ => int(),
    }
}

// ---------------------------------------------------------------- reified types of values

/// The type value a container carries, if it is one.
pub fn container_ty(v: Value) -> Option<Value> {
    Some(match v.body()? {
        Body::List(l) => l.ty,
        Body::Dict(d) => d.ty,
        Body::Iter(it) => it.ty,
        Body::Generator(g) => g.ty,
        Body::Channel(c) => c.ty,
        _ => return None,
    })
}

/// The full type of a container value, if it carries one.
pub fn container_type(v: Value) -> Option<&'static RtType> {
    container_ty(v).and_then(as_type)
}

/// The type of the items produced by iterating `v` (`any` if unknown).
pub fn item_type(v: Value) -> Value {
    if let Some(t) = container_type(v) {
        return match t.ctor {
            Ctor::Dict if t.args.len() == 2 => t.args[0],
            _ => elem_of(t),
        };
    }
    match v.body() {
        Some(Body::Str(_)) => str_(),
        Some(Body::Array(a)) => elem_scalar(a.elem),
        Some(Body::Tuple(items)) => {
            // A tuple of one element type iterates as that type.
            let tys: Vec<Value> = items.iter().map(|x| type_of(*x)).collect();
            match tys.first() {
                Some(f) if tys.iter().all(|t| t == f) => *f,
                _ => any(),
            }
        }
        _ => any(),
    }
}

/// The exact runtime type of a value (for inference and display).
pub fn type_of(v: Value) -> Value {
    if v.is_small_int() {
        return int();
    }
    match v {
        Value::NONE => return option_of(any()),
        Value::UNIT => return prim(Ctor::Unit),
        Value::TRUE | Value::FALSE => return bool_(),
        _ => {}
    }
    if let Some(t) = container_ty(v) {
        return t;
    }
    match v.body() {
        Some(Body::Int(_)) => int(),
        Some(Body::Float(_)) => float(),
        Some(Body::BigInt(_)) => prim(Ctor::BigInt),
        Some(Body::Str(_)) => str_(),
        Some(Body::Tuple(items)) => {
            let tys: Vec<Value> = items.iter().map(|x| type_of(*x)).collect();
            tuple_of(&tys)
        }
        Some(Body::Array(a)) => prim(Ctor::Array(Some(a.elem))),
        Some(Body::Instance(i)) => prim(Ctor::Class(i.class)),
        Some(Body::Function(_) | Body::Builtin(_) | Body::BoundMethod { .. }) => prim(Ctor::Function),
        Some(Body::Lock(_)) => prim(Ctor::Lock),
        Some(Body::Thread(_)) => prim(Ctor::Thread),
        _ => any(),
    }
}

// ---------------------------------------------------------------- checks

/// Whether `v` is a value of type `t`.
///
/// Container types are invariant: a `list<int>` is only a `list<int>`, never
/// a `list<float>` or `list<any>`; a bare `list` accepts every list. The
/// numeric tower lets an `int` stand where a `float` is expected.
pub fn has_type(v: Value, t: &RtType) -> bool {
    if v.is_dead() {
        return false;
    }
    match t.ctor {
        Ctor::Any => v != Value::NONE,
        Ctor::Int => v.as_int().is_some(),
        Ctor::Float => v.as_float().is_some() || v.as_int().is_some(),
        Ctor::Bool => v.as_bool().is_some(),
        Ctor::BigInt => matches!(v.body(), Some(Body::BigInt(_))) || v.as_int().is_some(),
        Ctor::Str => v.as_str().is_some(),
        Ctor::Unit => v == Value::UNIT,
        Ctor::Option => v == Value::NONE || has_type(v, t.arg(0)),
        Ctor::Tuple => match v.body() {
            Some(Body::Tuple(items)) => {
                t.is_bare() || (items.len() == t.args.len() && items.iter().enumerate().all(|(i, x)| has_type(*x, t.arg(i))))
            }
            _ => false,
        },
        Ctor::List => matches!(v.body(), Some(Body::List(l)) if t.is_bare() || same(l.ty, t)),
        Ctor::Dict => matches!(v.body(), Some(Body::Dict(d)) if t.is_bare() || same(d.ty, t)),
        Ctor::Channel => matches!(v.body(), Some(Body::Channel(c)) if t.is_bare() || same(c.ty, t)),
        Ctor::Iterator => match v.body() {
            Some(Body::Iter(it)) => t.is_bare() || same(it.ty, t),
            Some(Body::Generator(g)) => t.is_bare() || same(g.ty, t),
            // A user iterator's item type is not known (generic traits come later).
            Some(Body::Instance(i)) => {
                (t.is_bare() || t.arg(0).ctor == Ctor::Any)
                    && i.class().traits.iter().any(|tr| tr.as_trait().is_some_and(|tr| tr.name == "Iterator"))
            }
            _ => false,
        },
        Ctor::Array(e) => v.as_array().is_some_and(|a| e.is_none_or(|e| a.elem == e)),
        Ctor::Function => {
            matches!(v.body(), Some(Body::Function(_) | Body::Builtin(_) | Body::BoundMethod { .. } | Body::Class(_) | Body::Type(_)))
        }
        Ctor::Lock => matches!(v.body(), Some(Body::Lock(_))),
        Ctor::Thread => matches!(v.body(), Some(Body::Thread(_))),
        Ctor::Class(c) => v.as_instance().is_some_and(|i| i.class == c),
        Ctor::Trait(tr) => match v.body() {
            Some(Body::Instance(i)) => i.class().implements(tr),
            Some(Body::Generator(_) | Body::Iter(_)) => tr.as_trait().is_some_and(|t| t.name == "Iterator"),
            _ => false,
        },
    }
}

#[inline]
fn same(a: Value, t: &RtType) -> bool {
    as_type(a).is_some_and(|a| std::ptr::eq(a, t))
}

/// Converts an owned value for storage as an element of type `t`: checks it
/// and coerces an `int` stored as a `float`. Returns `None` (leaving `v`
/// untouched) if the value does not have the type.
pub fn admit(v: Value, t: &RtType) -> Option<Value> {
    if !has_type(v, t) {
        return None;
    }
    let float_slot = match t.ctor {
        Ctor::Float => true,
        Ctor::Option => t.arg(0).ctor == Ctor::Float,
        _ => false,
    };
    if float_slot {
        if let Some(i) = v.as_int() {
            v.decref();
            return Some(Value::float(i as f64));
        }
    }
    Some(v)
}
