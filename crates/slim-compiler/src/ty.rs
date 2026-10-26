//! Static types used for ownership decisions and annotation checks.
//!
//! The compiler only needs to know whether a value is a copy type or a move
//! type; when it cannot tell (`Unknown`) it emits dynamically checked
//! instructions instead.

use slim_vm::array::Elem;
use slim_syntax::ast::TypeExpr;
use slim_vm::code::TypeDesc;

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Unknown,
    Int,
    Float,
    Bool,
    BigInt,
    Str,
    Unit,
    /// The type of the literal `None` (only valid as an `Option`).
    NoneT,
    Tuple,
    List,
    Dict,
    Func,
    /// A module-level `def` (immortal, copied freely).
    StaticFn,
    Generator,
    Iter,
    Instance(String),
    Trait(String),
    /// A builtin object type (`Lock`, `Channel`, `Thread`): a move type.
    Opaque(String),
    /// A class object (immortal, copied freely).
    Class(String),
    Option(Box<Ty>),
    /// `array` (element type unknown) or `array<T>`.
    Array(Option<Elem>),
}

/// The slim type of an array element read.
pub fn elem_ty(e: Elem) -> Ty {
    match e {
        Elem::F32 | Elem::F64 => Ty::Float,
        Elem::Bool => Ty::Bool,
        _ => Ty::Int,
    }
}

impl Ty {
    /// `Some(true)` for copy types, `Some(false)` for move types.
    pub fn is_copy(&self) -> Option<bool> {
        match self {
            Ty::Unknown => None,
            Ty::Int | Ty::Float | Ty::Bool | Ty::BigInt | Ty::Unit | Ty::NoneT | Ty::Class(_) | Ty::StaticFn => Some(true),
            Ty::Option(t) => t.is_copy(),
            _ => Some(false),
        }
    }

    pub fn is_move(&self) -> bool {
        self.is_copy() == Some(false)
    }

    pub fn join(&self, other: &Ty) -> Ty {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Ty::NoneT, Ty::Option(_)) => other.clone(),
            (Ty::Option(_), Ty::NoneT) => self.clone(),
            (Ty::NoneT, t) | (t, Ty::NoneT) if *t != Ty::Unknown => Ty::Option(Box::new(t.clone())),
            (Ty::Option(a), b) | (b, Ty::Option(a)) if **a == *b => Ty::Option(a.clone()),
            (Ty::Int, Ty::Float) | (Ty::Float, Ty::Int) => Ty::Float,
            _ => Ty::Unknown,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Ty::Unknown => "?".into(),
            Ty::Int => "int".into(),
            Ty::Float => "float".into(),
            Ty::Bool => "bool".into(),
            Ty::BigInt => "bigint".into(),
            Ty::Str => "str".into(),
            Ty::Unit => "()".into(),
            Ty::NoneT => "None".into(),
            Ty::Tuple => "tuple".into(),
            Ty::List => "list".into(),
            Ty::Dict => "dict".into(),
            Ty::Func | Ty::StaticFn => "function".into(),
            Ty::Generator => "generator".into(),
            Ty::Iter => "iterator".into(),
            Ty::Instance(c) | Ty::Trait(c) | Ty::Opaque(c) => c.clone(),
            Ty::Class(c) => format!("class {c}"),
            Ty::Option(t) => format!("Option<{}>", t.name()),
            Ty::Array(None) => "array".into(),
            Ty::Array(Some(e)) => format!("array<{}>", e.name()),
        }
    }
}

/// What a name in a type annotation refers to.
pub enum TypeName {
    Class,
    Trait,
}

/// Converts an annotation into a static type and a runtime descriptor.
/// `lookup` resolves user class/trait names to their global slot.
pub fn resolve(
    t: &TypeExpr,
    lookup: &dyn Fn(&str) -> Option<(u32, TypeName)>,
) -> Result<(Ty, TypeDesc), String> {
    match t {
        TypeExpr::Unit(_) => Ok((Ty::Unit, TypeDesc::Unit)),
        TypeExpr::Named(n, args, _) => {
            let no_args = |ty: Ty, d: TypeDesc| -> Result<(Ty, TypeDesc), String> {
                Ok((ty, d))
            };
            match n.as_str() {
                "int" => no_args(Ty::Int, TypeDesc::Int),
                "float" => no_args(Ty::Float, TypeDesc::Float),
                "bool" => no_args(Ty::Bool, TypeDesc::Bool),
                "bigint" => no_args(Ty::BigInt, TypeDesc::BigInt),
                "str" => no_args(Ty::Str, TypeDesc::Str),
                "tuple" => no_args(Ty::Tuple, TypeDesc::Tuple),
                "list" => no_args(Ty::List, TypeDesc::List),
                "dict" => no_args(Ty::Dict, TypeDesc::Dict),
                "function" | "Callable" | "Fn" => no_args(Ty::Func, TypeDesc::Function),
                "any" | "Any" => no_args(Ty::Unknown, TypeDesc::Any),
                "Lock" | "Channel" | "Thread" => no_args(Ty::Opaque(n.clone()), TypeDesc::Builtin(n.clone())),
                "array" => match args.as_slice() {
                    [] => Ok((Ty::Array(None), TypeDesc::Array(None))),
                    [TypeExpr::Named(e, eargs, _)] if eargs.is_empty() && Elem::from_name(e).is_some() => {
                        let e = Elem::from_name(e);
                        Ok((Ty::Array(e), TypeDesc::Array(e)))
                    }
                    _ => Err("`array` takes one element type: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64 or bool".into()),
                },
                "Option" => {
                    if args.len() != 1 {
                        return Err("`Option` takes exactly one type argument: `Option<T>`".into());
                    }
                    let (inner, d) = resolve(&args[0], lookup)?;
                    if matches!(inner, Ty::Option(_)) {
                        return Err("nested `Option<Option<T>>` is not supported".into());
                    }
                    Ok((Ty::Option(Box::new(inner)), TypeDesc::Option(Box::new(d))))
                }
                "None" | "NoneType" => Err("`None` is not a type; use `Option<T>`".into()),
                "Optional" => Err("use `Option<T>` instead of `Optional`".into()),
                "Iterator" | "Generator" => match lookup(n) {
                    Some((idx, _)) => Ok((Ty::Iter, TypeDesc::Global(idx, n.clone()))),
                    None => Ok((Ty::Iter, TypeDesc::Any)),
                },
                other => match lookup(other) {
                    Some((idx, TypeName::Class)) => Ok((Ty::Instance(other.to_string()), TypeDesc::Global(idx, other.to_string()))),
                    Some((idx, TypeName::Trait)) => Ok((Ty::Trait(other.to_string()), TypeDesc::Global(idx, other.to_string()))),
                    None => Err(format!("unknown type `{other}`")),
                },
            }
        }
    }
}
