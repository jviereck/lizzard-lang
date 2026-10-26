//! Static types used for ownership decisions and annotation checks.
//!
//! The compiler only needs to know whether a value is a copy type or a move
//! type; when it cannot tell (`Unknown`) it emits dynamically checked
//! instructions instead. Container types mirror the runtime types exactly:
//! a static `list<int>` is a `list<int>` at runtime (containers are
//! invariant), while `list<?>` (element `Unknown`) is "some list".

use slim_syntax::ast::TypeExpr;
use slim_vm::array::Elem;
use slim_vm::code::{Ctor, TypeDesc};

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Unknown,
    /// An explicit `any` type argument (`list<any>`). A value read from such
    /// a container is untyped (`Unknown`).
    Any,
    Int,
    Float,
    Bool,
    BigInt,
    Str,
    Unit,
    /// The type of the literal `None` (only valid as an `Option`).
    NoneT,
    /// `tuple<A, B>`, or `None` for an untyped tuple.
    Tuple(Option<Vec<Ty>>),
    /// `list<T>`; `Unknown` for some list (a bare `list`).
    List(Box<Ty>),
    /// `dict<K, V>`.
    Dict(Box<Ty>, Box<Ty>),
    Func,
    /// A module-level `def` (immortal, copied freely).
    StaticFn,
    /// `Iterator<T>`: generators and iterators.
    Iter(Box<Ty>),
    /// `Channel<T>`.
    Chan(Box<Ty>),
    /// An instance of a class, with its type arguments (empty for a
    /// non-generic class, or for "some instance" of a generic one).
    Instance(String, Vec<Ty>),
    /// A value implementing a trait (`Iterable<int>`).
    Trait(String, Vec<Ty>),
    /// A type parameter in a generic class, trait or function.
    Param(String),
    /// A builtin object type (`Lock`, `Thread`): a move type.
    Opaque(String),
    /// A class object (immortal, copied freely).
    Class(String),
    Option(Box<Ty>),
    /// `array` (element type unknown) or `array<T>`.
    Array(Option<Elem>),
    /// A type used as a value (`list::<int>`); calling it builds that type.
    TypeVal(Box<Ty>),
    /// `T + local`: values confined to their thread. Only declared types
    /// (and the expected types derived from them) carry it; the static type
    /// of a value never does (parameters and variables ignore locality).
    Local(Box<Ty>),
}

/// The slim type of an array element read.
pub fn elem_ty(e: Elem) -> Ty {
    match e {
        Elem::F32 | Elem::F64 => Ty::Float,
        Elem::Bool => Ty::Bool,
        _ => Ty::Int,
    }
}

/// A type argument as the static type of a value read from a container.
fn read_ty(t: &Ty) -> Ty {
    match t {
        Ty::Any => Ty::Unknown,
        t => t.clone(),
    }
}

impl Ty {
    /// The type with all locality removed.
    pub fn unlocal(&self) -> Ty {
        let b = |t: &Ty| Box::new(t.unlocal());
        let v = |ts: &[Ty]| ts.iter().map(Ty::unlocal).collect::<Vec<_>>();
        match self {
            Ty::Local(t) => t.unlocal(),
            Ty::List(t) => Ty::List(b(t)),
            Ty::Dict(k, x) => Ty::Dict(b(k), b(x)),
            Ty::Iter(t) => Ty::Iter(b(t)),
            Ty::Chan(t) => Ty::Chan(b(t)),
            Ty::Option(t) => Ty::Option(b(t)),
            Ty::TypeVal(t) => Ty::TypeVal(b(t)),
            Ty::Tuple(Some(ts)) => Ty::Tuple(Some(v(ts))),
            Ty::Instance(c, a) => Ty::Instance(c.clone(), v(a)),
            Ty::Trait(c, a) => Ty::Trait(c.clone(), v(a)),
            t => t.clone(),
        }
    }

    /// Whether `+ local` appears in the type.
    pub fn mentions_local(&self) -> bool {
        match self {
            Ty::Local(_) => true,
            Ty::List(t) | Ty::Iter(t) | Ty::Chan(t) | Ty::Option(t) | Ty::TypeVal(t) => t.mentions_local(),
            Ty::Dict(k, v) => k.mentions_local() || v.mentions_local(),
            Ty::Tuple(Some(ts)) | Ty::Instance(_, ts) | Ty::Trait(_, ts) => ts.iter().any(Ty::mentions_local),
            _ => false,
        }
    }

    /// Values of this type are never local objects (scalars and strings).
    pub fn never_local(&self) -> bool {
        match self {
            Ty::Int | Ty::Float | Ty::Bool | Ty::BigInt | Ty::Str | Ty::Unit | Ty::NoneT | Ty::Array(_) => true,
            Ty::Option(t) => t.never_local(),
            Ty::Tuple(Some(ts)) => ts.iter().all(Ty::never_local),
            _ => false,
        }
    }

    /// Replaces type parameters bound in `map` (others stay).
    pub fn subst(&self, map: &dyn Fn(&str) -> Option<Ty>) -> Ty {
        let b = |t: &Ty| Box::new(t.subst(map));
        let v = |ts: &[Ty]| ts.iter().map(|t| t.subst(map)).collect::<Vec<_>>();
        match self {
            Ty::Param(n) => map(n).unwrap_or_else(|| self.clone()),
            Ty::Local(t) => Ty::Local(b(t)),
            Ty::List(t) => Ty::List(b(t)),
            Ty::Dict(k, x) => Ty::Dict(b(k), b(x)),
            Ty::Iter(t) => Ty::Iter(b(t)),
            Ty::Chan(t) => Ty::Chan(b(t)),
            Ty::Option(t) => Ty::Option(b(t)),
            Ty::TypeVal(t) => Ty::TypeVal(b(t)),
            Ty::Tuple(Some(ts)) => Ty::Tuple(Some(v(ts))),
            Ty::Instance(c, a) => Ty::Instance(c.clone(), v(a)),
            Ty::Trait(c, a) => Ty::Trait(c.clone(), v(a)),
            t => t.clone(),
        }
    }

    pub fn list(t: Ty) -> Ty {
        Ty::List(Box::new(t))
    }

    pub fn dict(k: Ty, v: Ty) -> Ty {
        Ty::Dict(Box::new(k), Box::new(v))
    }

    pub fn iter(t: Ty) -> Ty {
        Ty::Iter(Box::new(t))
    }

    /// The element type produced by indexing (a borrow) or `None` if unknown.
    pub fn elem(&self) -> Option<Ty> {
        let t = match self {
            Ty::Local(t) => return t.elem().map(|t| t.unlocal()),
            Ty::List(t) => read_ty(t),
            Ty::Dict(_, v) => read_ty(v),
            Ty::Array(Some(e)) => elem_ty(*e),
            Ty::Str => Ty::Str,
            _ => return None,
        };
        (t != Ty::Unknown).then_some(t)
    }

    /// The type of the items a `for` loop over this type produces.
    pub fn item(&self) -> Ty {
        match self {
            Ty::Local(t) => t.item().unlocal(),
            Ty::Dict(k, _) => read_ty(k),
            Ty::Iter(t) | Ty::Chan(t) => read_ty(t),
            Ty::Tuple(Some(ts)) => match ts.first() {
                Some(f) if ts.iter().all(|t| t == f) => read_ty(f),
                _ => Ty::Unknown,
            },
            _ => self.elem().unwrap_or(Ty::Unknown),
        }
    }

    /// `Some(true)` for copy types, `Some(false)` for move types.
    pub fn is_copy(&self) -> Option<bool> {
        match self {
            Ty::Unknown | Ty::Any | Ty::Param(_) => None,
            Ty::Int | Ty::Float | Ty::Bool | Ty::BigInt | Ty::Unit | Ty::NoneT | Ty::Class(_) | Ty::StaticFn | Ty::TypeVal(_) => {
                Some(true)
            }
            Ty::Option(t) | Ty::Local(t) => t.is_copy(),
            _ => Some(false),
        }
    }

    pub fn is_move(&self) -> bool {
        self.is_copy() == Some(false)
    }

    /// Fully known: no `Unknown` anywhere (so the runtime type is exactly this).
    pub fn is_precise(&self) -> bool {
        match self {
            Ty::Unknown => false,
            Ty::Tuple(None) => false,
            Ty::Tuple(Some(ts)) => ts.iter().all(Ty::is_precise),
            Ty::List(t) | Ty::Iter(t) | Ty::Chan(t) | Ty::Option(t) => **t != Ty::Unknown && t.is_precise(),
            Ty::Dict(k, v) => k.is_precise() && v.is_precise(),
            Ty::Instance(_, args) | Ty::Trait(_, args) => args.iter().all(Ty::is_precise),
            Ty::Local(t) => t.is_precise(),
            _ => true,
        }
    }

    pub fn join(&self, other: &Ty) -> Ty {
        if self == other {
            return self.clone();
        }
        if self.mentions_local() || other.mentions_local() {
            return self.unlocal().join(&other.unlocal());
        }
        match (self, other) {
            (Ty::NoneT, Ty::Option(_)) => other.clone(),
            (Ty::Option(_), Ty::NoneT) => self.clone(),
            (Ty::NoneT, t) | (t, Ty::NoneT) if *t != Ty::Unknown => Ty::Option(Box::new(t.clone())),
            (Ty::Option(a), b) | (b, Ty::Option(a)) if **a == *b => Ty::Option(a.clone()),
            (Ty::Int, Ty::Float) | (Ty::Float, Ty::Int) => Ty::Float,
            // Invariant containers: different element types join to "some list".
            (Ty::List(_), Ty::List(_)) => Ty::list(Ty::Unknown),
            (Ty::Dict(..), Ty::Dict(..)) => Ty::dict(Ty::Unknown, Ty::Unknown),
            (Ty::Iter(_), Ty::Iter(_)) => Ty::iter(Ty::Unknown),
            (Ty::Chan(_), Ty::Chan(_)) => Ty::Chan(Box::new(Ty::Unknown)),
            (Ty::Tuple(Some(a)), Ty::Tuple(Some(b))) if a.len() == b.len() => {
                Ty::Tuple(Some(a.iter().zip(b).map(|(x, y)| x.join(y)).collect()))
            }
            (Ty::Tuple(_), Ty::Tuple(_)) => Ty::Tuple(None),
            _ => Ty::Unknown,
        }
    }

    pub fn name(&self) -> String {
        let args = |base: &str, ts: &[&Ty]| -> String {
            if ts.iter().any(|t| **t == Ty::Unknown) {
                base.to_string()
            } else {
                format!("{base}<{}>", ts.iter().map(|t| t.name()).collect::<Vec<_>>().join(", "))
            }
        };
        match self {
            Ty::Unknown => "?".into(),
            Ty::Any => "any".into(),
            Ty::Int => "int".into(),
            Ty::Float => "float".into(),
            Ty::Bool => "bool".into(),
            Ty::BigInt => "bigint".into(),
            Ty::Str => "str".into(),
            Ty::Unit => "()".into(),
            Ty::NoneT => "None".into(),
            Ty::Tuple(None) => "tuple".into(),
            Ty::Tuple(Some(ts)) => args("tuple", &ts.iter().collect::<Vec<_>>()),
            Ty::List(t) => args("list", &[t]),
            Ty::Dict(k, v) => args("dict", &[k, v]),
            Ty::Func | Ty::StaticFn => "function".into(),
            Ty::Iter(t) => args("Iterator", &[t]),
            Ty::Chan(t) => args("Channel", &[t]),
            Ty::Instance(c, a) | Ty::Trait(c, a) if !a.is_empty() => args(c, &a.iter().collect::<Vec<_>>()),
            Ty::Instance(c, _) | Ty::Trait(c, _) | Ty::Opaque(c) | Ty::Param(c) => c.clone(),
            Ty::Class(c) => format!("class {c}"),
            Ty::Option(t) => format!("Option<{}>", t.name()),
            Ty::Array(None) => "array".into(),
            Ty::Array(Some(e)) => format!("array<{}>", e.name()),
            Ty::TypeVal(t) => format!("type {}", t.name()),
            Ty::Local(t) => {
                let n = t.name();
                match n.strip_suffix('>') {
                    Some(head) => format!("{head} + local>"),
                    None => format!("{n}<+local>"),
                }
            }
        }
    }
}

/// What a name in a type annotation refers to.
pub enum TypeName {
    /// A class with this many type parameters.
    Class(usize),
    /// A trait with this many type parameters.
    Trait(usize),
    /// A type parameter, found at this index of the environment.
    Param(u16),
}

/// Converts an annotation into a static type and a runtime descriptor.
/// `lookup` resolves user class/trait names to their global slot.
pub fn resolve(
    t: &TypeExpr,
    lookup: &dyn Fn(&str) -> Option<(u32, TypeName)>,
) -> Result<(Ty, TypeDesc), String> {
    resolve_in(t, lookup, false)
}

/// `nested`: `t` is a type argument (where `any` is a type of its own).
fn resolve_in(
    t: &TypeExpr,
    lookup: &dyn Fn(&str) -> Option<(u32, TypeName)>,
    nested: bool,
) -> Result<(Ty, TypeDesc), String> {
    match t {
        TypeExpr::Unit(_) => Ok((Ty::Unit, TypeDesc::prim(Ctor::Unit))),
        TypeExpr::Local(inner) => {
            let (ty, d) = resolve_in(inner, lookup, nested)?;
            match &ty {
                Ty::List(_) | Ty::Dict(..) | Ty::Iter(_) | Ty::Tuple(_) | Ty::Instance(..) | Ty::Trait(..) | Ty::Func | Ty::Unknown | Ty::Any => {
                    Ok((Ty::Local(Box::new(ty)), TypeDesc::Local(Box::new(d))))
                }
                Ty::Chan(_) => Err("channels connect threads and cannot be local".into()),
                other => Err(format!("`{}` values cannot be local (only objects that hold references can)", other.name())),
            }
        }
        TypeExpr::Named(n, args, _) => {
            let arg = |i: usize| resolve_in(&args[i], lookup, true);
            let no_args = |ty: Ty, c: Ctor| -> Result<(Ty, TypeDesc), String> {
                if !args.is_empty() {
                    return Err(format!("`{n}` does not take type arguments"));
                }
                Ok((ty, TypeDesc::prim(c)))
            };
            match n.as_str() {
                "int" => no_args(Ty::Int, Ctor::Int),
                "float" => no_args(Ty::Float, Ctor::Float),
                "bool" => no_args(Ty::Bool, Ctor::Bool),
                "bigint" => no_args(Ty::BigInt, Ctor::BigInt),
                "str" => no_args(Ty::Str, Ctor::Str),
                "tuple" => {
                    if args.is_empty() {
                        return Ok((Ty::Tuple(None), TypeDesc::prim(Ctor::Tuple)));
                    }
                    let (tys, ds): (Vec<Ty>, Vec<TypeDesc>) = (0..args.len()).map(arg).collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
                    Ok((Ty::Tuple(Some(tys)), TypeDesc::Builtin(Ctor::Tuple, ds)))
                }
                "list" | "Iterator" | "Generator" | "Channel" => {
                    let (c, wrap): (Ctor, fn(Box<Ty>) -> Ty) = match n.as_str() {
                        "list" => (Ctor::List, Ty::List),
                        "Channel" => (Ctor::Channel, Ty::Chan),
                        _ => (Ctor::Iterator, Ty::Iter),
                    };
                    let name = if c == Ctor::Iterator { "Iterator" } else { n.as_str() };
                    match args.len() {
                        0 => Ok((wrap(Box::new(Ty::Unknown)), TypeDesc::prim(c))),
                        1 => {
                            let (ty, d) = arg(0)?;
                            Ok((wrap(Box::new(ty)), TypeDesc::Builtin(c, vec![d])))
                        }
                        _ => Err(format!("`{name}` takes one element type: `{name}<T>`")),
                    }
                }
                "dict" => match args.len() {
                    0 => Ok((Ty::dict(Ty::Unknown, Ty::Unknown), TypeDesc::prim(Ctor::Dict))),
                    2 => {
                        let ((k, kd), (v, vd)) = (arg(0)?, arg(1)?);
                        Ok((Ty::dict(k, v), TypeDesc::Builtin(Ctor::Dict, vec![kd, vd])))
                    }
                    _ => Err("`dict` takes a key and a value type: `dict<K, V>`".into()),
                },
                "function" | "Callable" | "Fn" => no_args(Ty::Func, Ctor::Function),
                "any" | "Any" => {
                    let (ty, d) = no_args(Ty::Unknown, Ctor::Any)?;
                    Ok((if nested { Ty::Any } else { ty }, d))
                }
                "Lock" => no_args(Ty::Opaque(n.clone()), Ctor::Lock),
                "Thread" => no_args(Ty::Opaque(n.clone()), Ctor::Thread),
                "array" => match args.as_slice() {
                    [] => Ok((Ty::Array(None), TypeDesc::prim(Ctor::Array(None)))),
                    [TypeExpr::Named(e, eargs, _)] if eargs.is_empty() && Elem::from_name(e).is_some() => {
                        let e = Elem::from_name(e);
                        Ok((Ty::Array(e), TypeDesc::prim(Ctor::Array(e))))
                    }
                    _ => Err("`array` takes one element type: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64 or bool".into()),
                },
                "Option" => {
                    if args.len() != 1 {
                        return Err("`Option` takes exactly one type argument: `Option<T>`".into());
                    }
                    let (inner, d) = arg(0)?;
                    if matches!(inner, Ty::Option(_)) {
                        return Err("nested `Option<Option<T>>` is not supported".into());
                    }
                    Ok((Ty::Option(Box::new(inner)), TypeDesc::option(d)))
                }
                "None" | "NoneType" => Err("`None` is not a type; use `Option<T>`".into()),
                "Optional" => Err("use `Option<T>` instead of `Optional`".into()),
                other => {
                    let found = lookup(other).ok_or_else(|| format!("unknown type `{other}`"))?;
                    let arity = match found.1 {
                        TypeName::Param(k) => {
                            if !args.is_empty() {
                                return Err(format!("type parameter `{other}` does not take type arguments"));
                            }
                            return Ok((Ty::Param(other.to_string()), TypeDesc::Param(k, other.to_string())));
                        }
                        TypeName::Class(n) | TypeName::Trait(n) => n,
                    };
                    // No arguments on a generic type: any instantiation.
                    if !args.is_empty() && args.len() != arity {
                        return Err(match arity {
                            0 => format!("`{other}` does not take type arguments"),
                            n => format!("`{other}` takes {n} type argument{}", if n == 1 { "" } else { "s" }),
                        });
                    }
                    let (tys, ds): (Vec<Ty>, Vec<TypeDesc>) =
                        (0..args.len()).map(arg).collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
                    let d = TypeDesc::Global(found.0, other.to_string(), ds);
                    Ok(match found.1 {
                        TypeName::Class(_) => (Ty::Instance(other.to_string(), tys), d),
                        _ => (Ty::Trait(other.to_string(), tys), d),
                    })
                }
            }
        }
    }
}
