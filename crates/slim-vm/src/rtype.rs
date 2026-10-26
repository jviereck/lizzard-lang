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
use crate::code::TypeDesc;
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
    /// The type is not `Send`: its values are local (confined to the thread
    /// that created them). True for native classes over non-`Send` Rust
    /// types, classes with a non-`Send` field, types with a non-`Send`
    /// argument, and erased types not written `+ Send`.
    pub local: bool,
    /// The type's own value (set when it is interned).
    this: OnceLock<Value>,
    /// The `+ Send` version of an erased type (`any + Send` for `any`).
    base: OnceLock<Value>,
    list_of: OnceLock<Value>,
    option_of: OnceLock<Value>,
    /// For a class type: the implemented trait types (`Iterable<int>`).
    impls: OnceLock<Box<[Value]>>,
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

type Table = RwLock<HashMap<(Ctor, Box<[u64]>, bool), Value>>;

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The unique type value for `ctor<args>`.
pub fn intern(ctor: Ctor, args: &[Value]) -> Value {
    intern_with(ctor, args, false)
}

/// Kinds of values that can be local (objects that hold references).
fn can_be_local(c: Ctor) -> bool {
    matches!(c, Ctor::Any | Ctor::List | Ctor::Dict | Ctor::Option | Ctor::Tuple | Ctor::Iterator | Ctor::Function | Ctor::Class(_) | Ctor::Trait(_))
}

/// Locality implied by the parts of a type: a local type argument, or a
/// class whose instances are always local.
fn implied_local(ctor: Ctor, args: &[Value]) -> bool {
    args.iter().any(|a| as_type(*a).is_some_and(|t| t.local))
        || matches!(ctor, Ctor::Class(c) if match c.body() {
            Some(Body::Class(c)) => c.always_local,
            Some(Body::NativeClass(c)) if c.is_family() => c.local_for(args),
            Some(Body::NativeClass(c)) => c.mode == crate::native::Mode::Local,
            _ => false,
        })
}

/// Types that do not show what their values hold. As annotations they may
/// hold values that are not `Send` (local values) unless written `+ Send`;
/// other types are local by their parts.
pub fn erased(c: Ctor) -> bool {
    matches!(c, Ctor::Any | Ctor::Function | Ctor::Iterator | Ctor::Trait(_))
}

/// `ctor<args>`, local if implied by its parts, or if `explicit` on an
/// erased type (`any` as opposed to `any + Send`).
pub fn intern_with(ctor: Ctor, args: &[Value], explicit: bool) -> Value {
    let implied = can_be_local(ctor) && implied_local(ctor, args);
    let explicit = explicit && erased(ctor);
    let local = explicit || implied;
    let key = (ctor, args.iter().map(|a| a.0).collect::<Box<[u64]>>(), local);
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
        name: type_name(ctor, args, erased(ctor) && !local),
        local,
        this: OnceLock::new(),
        base: OnceLock::new(),
        list_of: OnceLock::new(),
        option_of: OnceLock::new(),
        impls: OnceLock::new(),
    }));
    let _ = as_type(v).unwrap().this.set(v);
    t.insert(key, v);
    v
}

/// The version of an erased type that may hold local values (`any` for
/// `any + Send`); `t` itself for any other type.
pub fn local_of(t: &RtType) -> Value {
    intern_with(t.ctor, &t.args, true)
}

/// An erased type restricted to `Send` values (`any + Send` for `any`).
fn base(t: &RtType) -> &'static RtType {
    let v = *t.base.get_or_init(|| if t.local && erased(t.ctor) { intern_with(t.ctor, &t.args, false) } else { t.value() });
    as_type(v).unwrap()
}

/// `show_send`: an erased type restricted to `Send` values (`any + Send`).
/// Traits are shown plainly (their type values are also used as bounds).
fn type_name(ctor: Ctor, args: &[Value], show_send: bool) -> String {
    let n = type_name_plain(ctor, args);
    if !show_send || matches!(ctor, Ctor::Trait(_)) {
        return n;
    }
    format!("{n} + Send")
}

fn type_name_plain(ctor: Ctor, args: &[Value]) -> String {
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
        Ctor::Class(c) => match c.body() {
            Some(Body::Class(c)) => c.name.as_str(),
            Some(Body::NativeClass(c)) => c.name.as_str(),
            _ => "",
        },
        Ctor::Trait(t) => t.as_trait().map(|t| t.name.as_str()).unwrap_or_default(),
    };
    if args.is_empty() {
        return base.into();
    }
    let parts: Vec<&str> = args.iter().map(|a| as_type(*a).map(|t| t.name.as_str()).unwrap_or("?")).collect();
    format!("{base}<{}>", parts.join(", "))
}

/// Resolves a type whose classes and traits are already resolved, with type
/// parameters taken from `env`. `None` if it names an unresolved global.
pub fn subst(d: &TypeDesc, env: &[Value]) -> Option<Value> {
    Some(match d {
        TypeDesc::Builtin(c, args) => {
            let args = args.iter().map(|a| subst(a, env)).collect::<Option<Vec<_>>>()?;
            intern(*c, &args)
        }
        TypeDesc::Param(k, _) => *env.get(*k as usize)?,
        TypeDesc::Resolved(v) => *v,
        TypeDesc::Local(t) => local_of(as_type(subst(t, env)?).unwrap()),
        TypeDesc::Global(..) => return None,
    })
}

/// The type of instances of a non-generic class, or of an instantiation of
/// a generic native class (`Stack<int>`).
pub fn class_type(c: Value) -> Value {
    match c.body() {
        Some(Body::Class(k)) => *k.canonical.get_or_init(|| intern(Ctor::Class(c), &[])),
        Some(Body::NativeClass(k)) => *k.canonical.get_or_init(|| match k.family.get() {
            Some(f) => intern(Ctor::Class(*f), &k.type_args.iter().map(|t| t.rt()).collect::<Vec<_>>()),
            None => intern(Ctor::Class(c), &[]),
        }),
        _ => panic!("not a class"),
    }
}

/// The trait types a class type implements, with its type arguments
/// substituted (`Stack<int>` implements `Iterable<int>`).
pub fn impls_of(t: &RtType) -> &[Value] {
    t.impls.get_or_init(|| {
        let Ctor::Class(c) = t.ctor else { return Box::new([]) };
        let Some(class) = c.as_class() else { return Box::new([]) };
        class
            .trait_args
            .iter()
            .filter_map(|(tr, args)| {
                let args = args.iter().map(|a| subst(a, &t.args)).collect::<Option<Vec<_>>>()?;
                Some(intern(Ctor::Trait(*tr), &args))
            })
            .collect()
    })
}

/// Whether type `t` satisfies trait type `tr` (a bound such as `Ord` or
/// `Iterable<int>`). The builtin traits are structural; user traits are
/// satisfied by classes that implement them with the same type arguments.
pub fn satisfies(t: &RtType, tr: &RtType) -> bool {
    let tr = base(tr);
    let Ctor::Trait(trv) = tr.ctor else { return false };
    let Some(trait_) = trv.as_trait() else { return false };
    let class = match t.ctor {
        Ctor::Class(c) => c.as_class(),
        _ => None,
    };
    let native = native_class_of(t);
    // A native class implements a trait of its module or a builtin one
    // (`c.implements(..)`), or one of the program's (`impl T for C`).
    let native_impl = |name: &str| {
        native.is_some_and(|(_, nc)| nc.builtin_traits.contains(&name) || nc.traits.iter().any(|x| x.as_trait().is_some_and(|x| x.name == name)))
    };
    let script_impl = || native.is_some_and(|(v, _)| trait_.native_impls.read().unwrap().contains(&v));
    let implements = |name: &str| {
        class.is_some_and(|c| c.traits.iter().any(|x| x.as_trait().is_some_and(|x| x.name == name))) || native_impl(name) || script_impl()
    };
    // Tuples (and `Option`s, for hashing) qualify when all their parts do.
    let parts = || !t.is_bare() && t.args.iter().all(|a| satisfies(as_type(*a).unwrap(), tr));
    match trait_.name.as_str() {
        "Send" => !t.local,
        "Ord" => match t.ctor {
            Ctor::Int | Ctor::Float | Ctor::BigInt | Ctor::Bool | Ctor::Str => true,
            Ctor::Tuple => parts(),
            _ => implements("Ord") || implements("Num"),
        },
        "Num" => matches!(t.ctor, Ctor::Int | Ctor::Float | Ctor::BigInt) || implements("Num"),
        "Hash" => match t.ctor {
            Ctor::List | Ctor::Dict | Ctor::Array(_) | Ctor::Any => false,
            Ctor::Tuple | Ctor::Option => parts(),
            _ => true,
        },
        "Len" => {
            matches!(t.ctor, Ctor::Str | Ctor::List | Ctor::Dict | Ctor::Array(_) | Ctor::Tuple | Ctor::Channel) || implements("Len")
        }
        "Iterable" | "Iterator" if class.is_none() => {
            let item = match t.ctor {
                Ctor::Iterator => Some(elem_of(t)),
                _ if trait_.name == "Iterator" => None,
                Ctor::List | Ctor::Channel => Some(elem_of(t)),
                Ctor::Dict if t.args.len() == 2 => Some(t.args[0]),
                Ctor::Str => Some(str_()),
                Ctor::Array(Some(e)) => Some(elem_scalar(e)),
                Ctor::Tuple => t.args.first().copied().filter(|f| t.args.iter().all(|a| a == f)),
                _ => None,
            };
            item.is_some_and(|item| tr.is_bare() || tr.args[0] == item)
        }
        _ if native.is_some() => native.is_some_and(|(_, nc)| nc.traits.contains(&trv)) || script_impl(),
        _ => match class {
            Some(c) => {
                // A class implementing `Iterator<T>` also provides `Iterable<T>`.
                let provides = |i: &Value| {
                    *i == tr.value()
                        || (trait_.name == "Iterable"
                            && as_type(*i).is_some_and(|i| {
                                i.args == tr.args && matches!(i.ctor, Ctor::Trait(x) if x.as_trait().is_some_and(|x| x.name == "Iterator"))
                            }))
                };
                let declared = c.implements(trv) || (trait_.name == "Iterable" && implements("Iterator"));
                declared && (tr.is_bare() || impls_of(t).iter().any(provides))
            }
            None => matches!(t.ctor, Ctor::Trait(x) if x == trv && (tr.is_bare() || t.args == tr.args)),
        },
    }
}

/// The native class of instances of type `t`, if it is one: for a generic
/// native class, the instantiation for `t`'s type arguments.
fn native_class_of(t: &RtType) -> Option<(Value, &'static crate::native::NativeClass)> {
    let Ctor::Class(c) = t.ctor else { return None };
    let Some(Body::NativeClass(nc)) = c.body() else { return None };
    let v = if nc.is_family() { nc.instance_for(t.value())? } else { c };
    match v.body() {
        Some(Body::NativeClass(k)) => Some((v, k)),
        _ => None,
    }
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

/// `any`: any value except `None`, possibly not `Send`.
pub fn any() -> Value {
    static V: OnceLock<Value> = OnceLock::new();
    *V.get_or_init(|| local_of(as_type(any_send()).unwrap()))
}

cached_prims! {
    any_send => Ctor::Any,
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
        Some(Body::Instance(i)) => i.ty,
        Some(Body::Native(n)) => n.ty,
        Some(Body::Function(_) | Body::Builtin(_) | Body::BoundMethod { .. } | Body::Instantiated { .. }) => {
            let f = prim(Ctor::Function);
            if v.is_local() {
                local_of(as_type(f).unwrap())
            } else {
                f
            }
        }
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
    has_shape(v, t) && local_ok(v, t)
}

/// `has_type` without the locality check (for error messages).
pub fn has_shape(v: Value, t: &RtType) -> bool {
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
            // A user iterator: its `impl Iterator<T>` gives the item type.
            Some(Body::Instance(i)) => {
                let is_iter = |tr: &Value| tr.as_trait().is_some_and(|tr| tr.name == "Iterator");
                i.class().traits.iter().any(is_iter)
                    && (t.is_bare()
                        || t.arg(0).ctor == Ctor::Any
                        || impls_of(as_type(i.ty).unwrap())
                            .iter()
                            .filter_map(|x| as_type(*x))
                            .any(|x| matches!(x.ctor, Ctor::Trait(tr) if is_iter(&tr)) && x.args.first() == t.args.first()))
            }
            _ => false,
        },
        Ctor::Array(e) => v.as_array().is_some_and(|a| e.is_none_or(|e| a.elem == e)),
        Ctor::Function => matches!(
            v.body(),
            Some(Body::Function(_) | Body::Builtin(_) | Body::BoundMethod { .. } | Body::Instantiated { .. } | Body::Class(_) | Body::Type(_))
        ),
        Ctor::Lock => matches!(v.body(), Some(Body::Lock(_))),
        Ctor::Thread => matches!(v.body(), Some(Body::Thread(_))),
        Ctor::Class(c) => match v.body() {
            Some(Body::Instance(i)) => i.class == c && (t.is_bare() || same(i.ty, t)),
            Some(Body::Native(n)) => n.class == c || (n.class().family.get() == Some(&c) && (t.is_bare() || n.ty == t.value())),
            _ => false,
        },
        Ctor::Trait(tr) => match v.body() {
            Some(Body::Instance(i)) if t.is_bare() => i.class().implements(tr),
            Some(Body::Generator(_) | Body::Iter(_)) if t.is_bare() => tr.as_trait().is_some_and(|t| t.name == "Iterator" || t.name == "Iterable"),
            _ => v != Value::NONE && satisfies(as_type(type_of(v)).unwrap(), t),
        },
    }
}

/// The type of a value, `a`, is `t`; an erased `X` also accepts an
/// `X + Send`.
#[inline]
fn same(a: Value, t: &RtType) -> bool {
    as_type(a).is_some_and(|a| std::ptr::eq(a, t) || (t.local && std::ptr::eq(base(a), base(t))))
}

/// Locality allows `v` where `t` is expected: an erased type restricted to
/// `Send` values (`any + Send`) does not accept a local value.
/// (Other types are local exactly when their values are.)
#[inline]
pub fn local_ok(v: Value, t: &RtType) -> bool {
    t.local || !erased(t.ctor) || !v.is_local()
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
