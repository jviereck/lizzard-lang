//! AST → register bytecode, with static ownership tracking.
//!
//! Every local has a flow-sensitive ownership state (live, moved, ...) and a
//! static type. Moves of values with a known move type are checked at compile
//! time; when the type is unknown the compiler emits dynamically checked
//! instructions (`MoveDyn`, `OwnDyn`, `CheckLive`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lizzard_syntax::ast::*;
use lizzard_syntax::{Error, Span};
use lizzard_vm::code::{
    Arg, ArgList, ClassSpec, Code, Ctor, ExcEntry, FieldSpec, ImplSpec, ParamInfo, Reg, RetKind, TraitSpec, TypeDesc,
    TypeParamInfo,
};
use lizzard_vm::native::{FnSpec, ModuleSpec, NType};
use lizzard_vm::object::{Body, Obj};
use lizzard_vm::ops::{BinKind, FOp, Instr, StoreMode, UnKind};
use lizzard_vm::Value;

use crate::scan;
use crate::ty::{elem_ty, resolve, Ty, TypeName};

type R<T> = Result<T, Error>;

fn own_err(msg: impl Into<String>, span: Span) -> Error {
    Error::new("OwnershipError", msg, span)
}

fn type_err(msg: impl Into<String>, span: Span) -> Error {
    Error::new("TypeError", msg, span)
}

fn name_err(msg: impl Into<String>, span: Span) -> Error {
    Error::new("NameError", msg, span)
}

fn syn_err(msg: impl Into<String>, span: Span) -> Error {
    Error::syntax(msg, span)
}

// ---------------------------------------------------------------- module info

#[derive(Clone, Debug, PartialEq)]
pub struct ParamSig {
    pub name: String,
    pub keep: bool,
    pub ty: Ty,
}

/// A type parameter with its bounds (`Ty::Trait`s).
#[derive(Clone, Debug, PartialEq)]
pub struct TParamSig {
    pub name: String,
    pub bounds: Vec<Ty>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sig {
    /// The function's own type parameters (hidden leading arguments).
    pub tparams: Vec<TParamSig>,
    pub params: Vec<ParamSig>,
    pub ret: Ty,
}

impl Sig {
    fn subst(&self, map: &dyn Fn(&str) -> Option<Ty>) -> Sig {
        Sig {
            tparams: self.tparams.iter().map(|t| TParamSig { name: t.name.clone(), bounds: t.bounds.iter().map(|b| b.subst(map)).collect() }).collect(),
            params: self.params.iter().map(|p| ParamSig { ty: p.ty.subst(map), ..p.clone() }).collect(),
            ret: self.ret.subst(map),
        }
    }
}

pub struct ClassInfo {
    /// The class is not `Send` (a native class over a non-`Send` Rust type,
    /// or a class with a non-`Send` field): instances stay on their thread.
    pub local: bool,
    pub tparams: Vec<TParamSig>,
    pub fields: Vec<(String, Ty, TypeDesc)>,
    /// Inherent methods (and a native class's Rust methods).
    pub methods: HashMap<String, Sig>,
    /// Methods of trait impls, including inherited defaults: name ->
    /// (trait, signature). Two traits may define the same name.
    pub trait_methods: HashMap<String, Vec<(String, Sig)>>,
    /// Implemented traits with their type arguments (in terms of the class's
    /// type parameters), including the traits they imply.
    pub impls: Vec<(String, Vec<Ty>)>,
    /// A generic native class: its supported instantiations (type
    /// arguments, the instantiation's class). The family itself has no
    /// methods; see `ModInfo::class_info`.
    pub instances: Vec<(Vec<Ty>, ClassInfo)>,
    /// Defined in Rust (no struct literal, no inherent `impl`).
    pub native: bool,
    /// Implemented traits as the runtime class records them: (trait, type
    /// arguments in terms of the struct's type parameters).
    pub trait_descs: Vec<(String, Vec<TypeDesc>)>,
}

/// How `x.name()` resolves on a struct, as in Rust: an inherent method
/// first, else the single trait method of that name.
enum MethodRef {
    Inherent(Sig),
    Trait(String, Sig),
    /// Several traits define it: a qualified call (`Trait::name(x)`) is needed.
    Ambiguous(Vec<String>),
}

impl ClassInfo {
    fn lookup(&self, name: &str) -> Option<MethodRef> {
        if let Some(s) = self.methods.get(name) {
            return Some(MethodRef::Inherent(s.clone()));
        }
        match self.trait_methods.get(name).map(|v| v.as_slice()) {
            Some([(t, s)]) => Some(MethodRef::Trait(t.clone(), s.clone())),
            Some(v) if !v.is_empty() => {
                let mut ts: Vec<String> = v.iter().map(|(t, _)| t.clone()).collect();
                ts.sort();
                Some(MethodRef::Ambiguous(ts))
            }
            _ => None,
        }
    }

    fn trait_method(&self, tr: &str, name: &str) -> Option<Sig> {
        self.trait_methods.get(name)?.iter().find(|(t, _)| t == tr).map(|(_, s)| s.clone())
    }
}

/// A method of a struct from one of its `impl` blocks, for code generation.
#[derive(Clone)]
pub struct ImplMethod {
    /// The trait of a trait impl; `None` for an inherent method.
    pub tr: Option<String>,
    pub def: FuncDef,
    /// The impl's names for the struct's type parameters, in order
    /// (`impl<U> Stack<U>` names the first one `U`).
    pub tparams: Vec<String>,
}

/// An impl that does not cover a whole struct of the program (see
/// `code::ImplSpec`): for a builtin type (`impl Describe for int`), a native
/// class, a specialized struct (`impl Describe for Box<int>`) or with extra
/// bounds (`impl<T: Ord> Ord for Box<T>`). It applies to the types its
/// target matches whose type arguments satisfy its bounds.
pub struct ImplInfo {
    /// The impl's type parameters with their bounds.
    pub tparams: Vec<TParamSig>,
    /// The trait and its arguments (`None` for an inherent impl).
    pub tr: Option<(String, Vec<Ty>)>,
    pub target: Ty,
    /// The target with `Param(k)` for the impl's `k`th type parameter.
    pub target_desc: TypeDesc,
    pub bound_descs: Vec<(u16, TypeDesc)>,
    pub trait_descs: Vec<TypeDesc>,
    /// Methods defined by the impl and, for a trait impl, the defaults it
    /// inherits.
    pub methods: HashMap<String, Sig>,
    pub defs: Vec<FuncDef>,
    /// `impl Describe for int`, for messages.
    pub shown: String,
    pub span: Span,
}

/// Matches type `t` against impl target `pat`, binding the target's type
/// parameters in `binds`. Unknown parts of `t` may match anything.
fn match_ty(pat: &Ty, t: &Ty, binds: &mut HashMap<String, Ty>) -> bool {
    let t = t.peel();
    match (pat.peel(), t) {
        (Ty::Param(n), _) => match binds.get(n) {
            Some(b) => b == t || *b == Ty::Unknown || *t == Ty::Unknown,
            None => {
                binds.insert(n.clone(), t.clone());
                true
            }
        },
        (_, Ty::Unknown) => true,
        (Ty::List(a), Ty::List(b)) | (Ty::Iter(a), Ty::Iter(b)) | (Ty::Chan(a), Ty::Chan(b)) | (Ty::Option(a), Ty::Option(b)) => match_ty(a, b, binds),
        (Ty::Dict(k1, v1), Ty::Dict(k2, v2)) => match_ty(k1, k2, binds) && match_ty(v1, v2, binds),
        (Ty::Tuple(Some(a)), Ty::Tuple(Some(b))) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| match_ty(x, y, binds)),
        (Ty::Tuple(Some(_)), Ty::Tuple(None)) => true,
        (Ty::Array(a), Ty::Array(b)) => a == b || b.is_none(),
        (Ty::Instance(c1, a1), Ty::Instance(c2, a2)) => {
            c1 == c2 && (a2.is_empty() || (a1.len() == a2.len() && a1.iter().zip(a2).all(|(x, y)| match_ty(x, y, binds))))
        }
        (a, b) => a == b,
    }
}

/// Whether two impl targets can describe the same type (type parameters
/// match anything; bounds are ignored, as in Rust).
fn overlaps(a: &Ty, b: &Ty) -> bool {
    match (a.peel(), b.peel()) {
        (Ty::Param(_), _) | (_, Ty::Param(_)) => true,
        (Ty::List(x), Ty::List(y)) | (Ty::Iter(x), Ty::Iter(y)) | (Ty::Chan(x), Ty::Chan(y)) | (Ty::Option(x), Ty::Option(y)) => overlaps(x, y),
        (Ty::Dict(k1, v1), Ty::Dict(k2, v2)) => overlaps(k1, k2) && overlaps(v1, v2),
        (Ty::Tuple(Some(x)), Ty::Tuple(Some(y))) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| overlaps(p, q)),
        (Ty::Instance(c1, x), Ty::Instance(c2, y)) => {
            c1 == c2 && (x.is_empty() || y.is_empty() || (x.len() == y.len() && x.iter().zip(y).all(|(p, q)| overlaps(p, q))))
        }
        (p, q) => p == q,
    }
}

/// Whether `t` mentions type parameter `n`.
fn mentions(t: &Ty, n: &str) -> bool {
    let found = std::cell::Cell::new(false);
    t.subst(&|x| {
        if x == n {
            found.set(true);
        }
        None
    });
    found.get()
}

/// The builtin method table of values of static type `t` (see
/// `builtins::has_type_method`), if `t` is a builtin type.
fn builtin_table(t: &Ty) -> Option<&'static str> {
    Some(match t {
        Ty::Int | Ty::Float | Ty::Bool | Ty::BigInt => "",
        Ty::Str => "str",
        Ty::List(_) => "list",
        Ty::Tuple(_) => "tuple",
        Ty::Dict(..) => "dict",
        Ty::Array(_) => "array",
        Ty::Chan(_) => "channel",
        Ty::Opaque(n) if n == "Lock" => "lock",
        Ty::Opaque(n) if n == "Thread" => "thread",
        _ => return None,
    })
}

pub struct TraitInfo {
    pub tparams: Vec<TParamSig>,
    pub methods: HashMap<String, Sig>,
    /// Defined by a native module: every method is required, and only Rust
    /// classes (`c.implements`) or Lizzard structs (`impl T for S`) implement it.
    pub native: bool,
}

/// Type parameters in scope: name -> environment index (a class type
/// argument index for fields, a register in code).
type Scope = [(String, u16)];

/// Builtin traits: (name, type parameters, implied traits, methods).
const BUILTIN_TRAITS: &[(&str, usize, &[&str], &[&str])] = &[
    ("Exception", 0, &[], &[]),
    ("LookupError", 0, &["Exception"], &[]),
    ("ArithmeticError", 0, &["Exception"], &[]),
    ("Iterable", 1, &[], &["__iter__"]),
    ("Iterator", 1, &["Iterable"], &["__next__"]),
    ("Ord", 0, &[], &["__lt__"]),
    ("Num", 0, &["Ord"], &["__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__", "__neg__"]),
    ("Hash", 0, &[], &[]),
    ("Len", 0, &[], &["__len__"]),
    // `__drop__` runs when a value is freed.
    ("Drop", 0, &[], &["__drop__"]),
    // Automatic, as in Rust: every type that is not local (thread-confined)
    // is `Send`. Only used as a bound.
    ("Send", 0, &[], &[]),
];

/// `d` (an erased type restricted to `Send`) as written: `any + Send`.
fn send_name(d: &Ty) -> String {
    let n = d.name();
    if n.ends_with(" + Send") {
        n
    } else {
        format!("{n} + Send")
    }
}

/// Whether the bounds of a type parameter include `Send`.
fn has_send(bounds: &[Ty]) -> bool {
    bounds.iter().any(|b| matches!(b, Ty::Trait(n, _) if n == "Send"))
}

fn builtin_trait(n: &str) -> Option<&'static (&'static str, usize, &'static [&'static str], &'static [&'static str])> {
    BUILTIN_TRAITS.iter().find(|t| t.0 == n)
}

pub struct ModInfo {
    /// Methods of each struct from its `impl` blocks.
    pub impl_methods: HashMap<String, Vec<ImplMethod>>,
    /// Associated functions as module functions named `Type::name`, with
    /// the impl's type parameters first.
    pub assoc_fns: Vec<FuncDef>,
    /// Impls matched against types (see `ImplInfo`).
    pub impls: Vec<ImplInfo>,
    pub gindex: HashMap<String, u32>,
    pub module_vars: HashSet<String>,
    pub funcs: HashMap<String, Sig>,
    pub classes: HashMap<String, ClassInfo>,
    pub traits: HashMap<String, TraitInfo>,
    pub builtin_classes: HashSet<String>,
    pub builtin_traits: HashSet<String>,
    pub filename: Arc<str>,
}

impl ModInfo {
    /// The class of `c<args>`: for a generic native class, the instantiation
    /// for these type arguments (`None` if they are not known statically or
    /// not supported: its members are then looked up at runtime).
    fn class_info(&self, c: &str, args: &[Ty]) -> Option<&ClassInfo> {
        let ci = self.classes.get(c)?;
        if ci.instances.is_empty() {
            return Some(ci);
        }
        ci.instances.iter().find(|(a, _)| a == args).map(|(_, i)| i)
    }

    fn type_lookup(&self, n: &str, scope: &Scope) -> Option<(u32, TypeName)> {
        if let Some((_, k)) = scope.iter().find(|(s, _)| s == n) {
            return Some((0, TypeName::Param(*k)));
        }
        let idx = *self.gindex.get(n)?;
        if let Some(ci) = self.classes.get(n) {
            Some((idx, TypeName::Class(ci.tparams.len())))
        } else if self.builtin_classes.contains(n) {
            Some((idx, TypeName::Class(0)))
        } else if let Some(t) = self.traits.get(n) {
            Some((idx, TypeName::Trait(t.tparams.len())))
        } else if self.builtin_traits.contains(n) {
            Some((idx, TypeName::Trait(builtin_trait(n).map_or(0, |t| t.1))))
        } else {
            None
        }
    }

    fn resolve_scoped(&self, t: &TypeExpr, scope: &Scope) -> R<(Ty, TypeDesc)> {
        resolve(t, &|n| self.type_lookup(n, scope)).map_err(|m| type_err(m, t.span()))
    }

    /// Resolves the bounds of type parameters `tps` (with `scope` in view).
    fn tparam_sigs(&self, tps: &[TypeParam], scope: &Scope) -> R<Vec<TParamSig>> {
        let mut out = Vec::new();
        for tp in tps {
            if scope.iter().filter(|(n, _)| *n == tp.name).count() > 1 || self.gindex.contains_key(&tp.name) && self.type_lookup(&tp.name, &[]).is_some() {
                return Err(type_err(format!("type parameter `{}` shadows another type", tp.name), tp.span));
            }
            let mut bounds = Vec::new();
            for b in &tp.bounds {
                if matches!(b, TypeExpr::Named(n, _, _) if n == "Local") {
                    return Err(type_err(format!("there is no `Local` trait: as in Rust, `{}: Send` is the bound for types that can cross threads", tp.name), b.span()));
                }
                // A trait used as a bound is a trait, not a value type.
                let ty = self.resolve_scoped(b, scope)?.0.peel().clone();
                if !matches!(ty, Ty::Trait(..)) {
                    return Err(type_err(format!("bound `{}` of `{}` is not a trait", ty.name(), tp.name), b.span()));
                }
                bounds.push(ty);
            }
            out.push(TParamSig { name: tp.name.clone(), bounds });
        }
        Ok(out)
    }

    /// `outer`: type parameters of the enclosing class or trait.
    fn sig_of(&self, f: &FuncDef, skip_self: bool, outer: &[TParamSig]) -> R<Sig> {
        let scope: Vec<(String, u16)> =
            outer.iter().map(|t| t.name.clone()).chain(f.tparams.iter().map(|t| t.name.clone())).map(|n| (n, 0)).collect();
        for tp in &f.tparams {
            if outer.iter().any(|o| o.name == tp.name) {
                return Err(type_err(format!("type parameter `{}` shadows the enclosing one", tp.name), tp.span));
            }
        }
        let tparams = self.tparam_sigs(&f.tparams, &scope)?;
        let mut params = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            if skip_self && i == 0 {
                continue;
            }
            let ty = match &p.ty {
                Some(t) => self.resolve_scoped(t, &scope)?.0,
                None => Ty::Unknown,
            };
            params.push(ParamSig { name: p.name.clone(), keep: p.keep, ty });
        }
        let ret = match &f.ret {
            Some(t) => self.resolve_scoped(t, &scope)?.0,
            None => Ty::Unknown,
        };
        Ok(Sig { tparams, params, ret })
    }

    /// Type parameters of a trait (empty for builtin and unknown traits).
    fn trait_tparams(&self, t: &str) -> Vec<TParamSig> {
        self.traits.get(t).map(|t| t.tparams.clone()).unwrap_or_default()
    }

    /// The traits a class instance implements, with arguments substituted
    /// (including those of registered impls that may apply).
    fn impls_of(&self, c: &str, args: &[Ty], pb: &dyn Fn(&str) -> Vec<Ty>) -> Vec<(String, Vec<Ty>)> {
        if self.builtin_classes.contains(c) {
            return vec![("Exception".into(), Vec::new())];
        }
        let Some(ci) = self.class_info(c, args).or_else(|| self.classes.get(c)) else { return Vec::new() };
        let map = class_map(ci, args);
        let mut out: Vec<(String, Vec<Ty>)> = ci.impls.iter().map(|(t, a)| (t.clone(), a.iter().map(|x| x.subst(&map)).collect())).collect();
        out.extend(self.registry_traits(&Ty::Instance(c.into(), args.to_vec()), pb));
        out
    }

    /// The registered impls that may apply to values of type `t` (target
    /// matched, bounds not violated), with their type parameters bound
    /// (`Unknown` where `t` does not tell).
    fn applicable(&self, t: &Ty, pb: &dyn Fn(&str) -> Vec<Ty>) -> Vec<(&ImplInfo, HashMap<String, Ty>)> {
        let mut out = Vec::new();
        for im in &self.impls {
            let mut b = HashMap::new();
            if !match_ty(&im.target, t, &mut b) {
                continue;
            }
            let ok = {
                let map = impl_map(im, &b);
                im.tparams.iter().all(|tp| {
                    let a = b.get(&tp.name).cloned().unwrap_or(Ty::Unknown);
                    tp.bounds.iter().all(|bd| self.satisfies(&a, &bd.subst(&map), pb))
                })
            };
            if ok {
                out.push((im, b));
            }
        }
        out
    }

    /// Methods named `name` of registered impls that may apply to `t`:
    /// (trait, signature), the trait `None` for an inherent impl.
    fn registry_methods(&self, t: &Ty, name: &str, pb: &dyn Fn(&str) -> Vec<Ty>) -> Vec<(Option<String>, Sig)> {
        self.applicable(t, pb)
            .into_iter()
            .filter_map(|(im, b)| {
                let s = im.methods.get(name)?;
                Some((im.tr.as_ref().map(|t| t.0.clone()), s.subst(&impl_map(im, &b))))
            })
            .collect()
    }

    /// The traits registered impls provide for `t`, with the traits they
    /// imply.
    fn registry_traits(&self, t: &Ty, pb: &dyn Fn(&str) -> Vec<Ty>) -> Vec<(String, Vec<Ty>)> {
        let mut out = Vec::new();
        for (im, b) in self.applicable(t, pb) {
            let Some((tr, args)) = &im.tr else { continue };
            let args: Vec<Ty> = args.iter().map(|a| a.subst(&impl_map(im, &b))).collect();
            if let Some((_, _, implies, _)) = builtin_trait(tr) {
                for i in implies.iter() {
                    let a = if builtin_trait(i).is_some_and(|x| x.1 == args.len()) { args.clone() } else { Vec::new() };
                    out.push((i.to_string(), a));
                }
            }
            out.push((tr.clone(), args));
        }
        out
    }

    /// Whether a registered impl of struct (or native class) `head` defines
    /// method `n`.
    fn has_registry_method(&self, head: &str, n: &str) -> bool {
        self.impls.iter().any(|im| matches!(&im.target, Ty::Instance(c, _) if c == head) && im.methods.contains_key(n))
    }

    /// Whether `t` implements trait `bound` (`Ty::Trait`). `pb` gives the
    /// bounds of type parameters in scope. Unknown types pass (the runtime
    /// checks them).
    fn satisfies(&self, t: &Ty, bound: &Ty, pb: &dyn Fn(&str) -> Vec<Ty>) -> bool {
        let Ty::Trait(b, bargs) = bound else { return true };
        if b == "Send" {
            return !self.is_local(t, &|n| !has_send(&pb(n)));
        }
        if let Ty::Local(inner) = t {
            return self.satisfies(inner, bound, pb);
        }
        let args_match = |got: &[Ty]| bargs.is_empty() || got.len() == bargs.len() && got.iter().zip(bargs).all(|(x, y)| ty_compatible(y, x) && ty_compatible(x, y));
        let item = |it: Ty| args_match(&[it]);
        match t {
            Ty::Unknown | Ty::Any => true,
            Ty::Param(n) => pb(n).iter().any(|pbound| self.trait_implies(pbound, b, bargs)),
            Ty::Instance(c, cargs) => self.impls_of(c, cargs, pb).iter().any(|(tn, ta)| tn == b && args_match(ta)),
            Ty::Trait(tn, ta) => self.trait_implies(&Ty::Trait(tn.clone(), ta.clone()), b, bargs),
            _ if self.registry_traits(t, pb).iter().any(|(tn, ta)| tn == b && args_match(ta)) => true,
            _ => match b.as_str() {
                "Ord" => match t {
                    Ty::Int | Ty::Float | Ty::BigInt | Ty::Bool | Ty::Str | Ty::Tuple(None) => true,
                    Ty::Tuple(Some(ts)) => ts.iter().all(|x| self.satisfies(x, bound, pb)),
                    _ => false,
                },
                "Num" => matches!(t, Ty::Int | Ty::Float | Ty::BigInt),
                "Hash" => match t {
                    Ty::List(_) | Ty::Dict(..) | Ty::Array(_) => false,
                    Ty::Tuple(Some(ts)) => ts.iter().all(|x| self.satisfies(x, bound, pb)),
                    Ty::Option(x) => self.satisfies(x, bound, pb),
                    _ => true,
                },
                "Len" => matches!(t, Ty::Str | Ty::List(_) | Ty::Dict(..) | Ty::Array(_) | Ty::Tuple(_) | Ty::Chan(_)),
                "Iterable" => match t {
                    Ty::List(x) | Ty::Iter(x) | Ty::Dict(x, _) => **x == Ty::Unknown || item((**x).clone()),
                    Ty::Str => item(Ty::Str),
                    Ty::Array(Some(e)) => item(elem_ty(*e)),
                    Ty::Array(None) | Ty::Tuple(_) => true,
                    _ => false,
                },
                "Iterator" => match t {
                    Ty::Iter(x) => **x == Ty::Unknown || item((**x).clone()),
                    _ => false,
                },
                _ => false,
            },
        }
    }

    /// Checks that the type arguments in `t` satisfy their bounds, that dict
    /// keys are hashable, and that an erased type written `+ Send` has only
    /// `Send` arguments (`Iterator<Counter> + Send` needs a `Send` `Counter`).
    fn check_wf(&self, t: &Ty, pb: &dyn Fn(&str) -> Vec<Ty>, span: Span) -> R<()> {
        let args: &[Ty] = match t {
            Ty::Iter(x) => std::slice::from_ref(&**x),
            Ty::Trait(_, xs) => xs,
            _ => &[],
        };
        if let Some(a) = args.iter().find(|a| self.is_local(a, &|n| !has_send(&pb(n)))) {
            return Err(type_err(
                match a {
                    Ty::Param(n) => format!("`{} + Send` needs `{n}: Send`", t.name()),
                    _ => format!("`{} + Send`: `{}` is not `Send`", t.name(), a.name()),
                },
                span,
            ));
        }
        self.check_wf_parts(t, pb, span)
    }

    fn check_wf_parts(&self, t: &Ty, pb: &dyn Fn(&str) -> Vec<Ty>, span: Span) -> R<()> {
        match t {
            Ty::Local(x) => self.check_wf_parts(x, pb, span),
            Ty::Chan(x) if self.is_local(x, &|n| !has_send(&pb(n))) => Err(type_err(
                match &**x {
                    Ty::Param(n) => format!("`{}`: channel items must be `Send`, and `{n}` may not be; add a `{n}: Send` bound", t.name()),
                    _ => format!("`{}`: channel items must be `Send`, and `{}` is not", t.name(), x.name()),
                },
                span,
            )),
            Ty::Instance(c, args)
                if self.classes.get(c).is_some_and(|ci| !ci.instances.is_empty() && ci.tparams.len() == args.len())
                    && args.iter().all(Ty::is_concrete)
                    && self.class_info(c, args).is_none() =>
            {
                let ci = &self.classes[c];
                let names: Vec<String> = ci.instances.iter().map(|(a, _)| format!("`{}`", Ty::Instance(c.clone(), a.clone()).name())).collect();
                Err(type_err(format!("`{}` is not available: `{c}` supports {}", t.name(), names.join(", ")), span))
            }
            Ty::Instance(c, args) | Ty::Trait(c, args) => {
                let tps = match self.classes.get(c) {
                    Some(ci) => ci.tparams.clone(),
                    None => self.trait_tparams(c),
                };
                let map = tparam_map(&tps, args);
                for (tp, a) in tps.iter().zip(args) {
                    for b in &tp.bounds {
                        let b = b.subst(&map);
                        if !self.satisfies(a, &b, pb) {
                            return Err(type_err(
                                format!("`{}` does not implement `{}` (required by `{}` of `{c}`)", a.name(), b.name(), tp.name),
                                span,
                            ));
                        }
                    }
                }
                args.iter().try_for_each(|a| self.check_wf(a, pb, span))
            }
            Ty::Dict(k, v) => {
                if !self.satisfies(k, &Ty::Trait("Hash".into(), Vec::new()), pb) {
                    return Err(type_err(format!("`{}` cannot be a dict key: it does not implement `Hash`", k.name()), span));
                }
                self.check_wf(k, pb, span)?;
                self.check_wf(v, pb, span)
            }
            Ty::List(x) | Ty::Iter(x) | Ty::Chan(x) | Ty::Option(x) | Ty::TypeVal(x) => self.check_wf(x, pb, span),
            Ty::Tuple(Some(ts)) => ts.iter().try_for_each(|a| self.check_wf(a, pb, span)),
            _ => Ok(()),
        }
    }

    /// Whether values of type `t` may not be `Send` ("local": confined to
    /// their thread), as far as the static type shows: a native class over a
    /// non-`Send` Rust type, a class with such a field, a type with such an
    /// argument, an erased type not written `+ Send`, or a type parameter for
    /// which `param_local` holds. Untyped values (`Unknown`) are checked at
    /// runtime instead.
    fn is_local(&self, t: &Ty, param_local: &dyn Fn(&str) -> bool) -> bool {
        let rec = |t: &Ty| self.is_local(t, param_local);
        match t {
            Ty::Local(_) => true,
            Ty::Param(n) => param_local(n),
            Ty::Instance(c, args) => {
                args.iter().any(rec)
                    || match (self.classes.get(c), self.class_info(c, args)) {
                        (_, Some(ci)) => ci.local,
                        // A generic native class with unknown arguments: any
                        // instantiation.
                        (Some(ci), None) => ci.instances.iter().any(|(_, i)| i.local),
                        (None, None) => false,
                    }
            }
            Ty::List(x) | Ty::Iter(x) | Ty::Option(x) | Ty::TypeVal(x) => rec(x),
            Ty::Dict(k, v) => rec(k) || rec(v),
            Ty::Tuple(Some(ts)) | Ty::Trait(_, ts) => ts.iter().any(rec),
            _ => false,
        }
    }

    /// Whether a bound `have` (`Ty::Trait`) provides trait `want<wargs>`.
    fn trait_implies(&self, have: &Ty, want: &str, wargs: &[Ty]) -> bool {
        let Ty::Trait(h, hargs) = have else { return false };
        if h == want {
            return wargs.is_empty() || hargs == wargs || hargs.is_empty();
        }
        builtin_trait(h).is_some_and(|(_, _, implies, _)| implies.iter().any(|i| *i == want && (wargs.is_empty() || hargs == wargs)))
    }

    /// The methods a value of type `t` is statically known to have, from a
    /// trait type. `None` when nothing is known.
    fn trait_methods(&self, t: &str, args: &[Ty]) -> Option<HashMap<String, Sig>> {
        self.trait_methods_for(t, args, Ty::Unknown)
    }

    /// Like `trait_methods`, with `Self` standing for `self_ty` (`Unknown`
    /// when the implementing type is not known).
    fn trait_methods_for(&self, t: &str, args: &[Ty], self_ty: Ty) -> Option<HashMap<String, Sig>> {
        let ti = self.traits.get(t)?;
        let map = tparam_map(&ti.tparams, args);
        let map = |n: &str| if n == "Self" { Some(self_ty.clone()) } else { map(n) };
        Some(ti.methods.iter().map(|(n, s)| (n.clone(), s.subst(&map))).collect())
    }
}

/// Substitution of an impl's type parameters by their bindings `b`.
fn impl_map<'a>(im: &'a ImplInfo, b: &'a HashMap<String, Ty>) -> impl Fn(&str) -> Option<Ty> + 'a {
    move |n| im.tparams.iter().any(|t| t.name == n).then(|| b.get(n).cloned().unwrap_or(Ty::Unknown))
}

/// Substitution of a class's type parameters by `args` (`Unknown` when the
/// instance's arguments are not known).
fn class_map<'a>(ci: &'a ClassInfo, args: &'a [Ty]) -> impl Fn(&str) -> Option<Ty> + 'a {
    tparam_map(&ci.tparams, args)
}

fn tparam_map<'a>(tps: &'a [TParamSig], args: &'a [Ty]) -> impl Fn(&str) -> Option<Ty> + 'a {
    move |n| tps.iter().position(|t| t.name == n).map(|k| args.get(k).cloned().unwrap_or(Ty::Unknown))
}

/// Compiles a parsed module. Returns the module code and the global names.
pub fn compile_module(module: &Module, filename: &str) -> R<(Arc<Code>, Vec<String>)> {
    compile_module_with(module, filename, &[])
}

/// The static type of a native's parameter or result. `Any` and `Func` are
/// `any` and `function`, which may hold values that are not `Send`: handles
/// are `!Send`, so the Rust side cannot move them to another thread either.
fn ntype_ty(t: &NType, classes: &NativeNames) -> Ty {
    let b = |t: &NType| Box::new(ntype_ty(t, classes));
    match t {
        NType::Int => Ty::Int,
        NType::Float => Ty::Float,
        NType::Bool => Ty::Bool,
        NType::Str => Ty::Str,
        NType::Unit => Ty::Unit,
        NType::Any => Ty::Local(Box::new(Ty::Any)),
        NType::Function => Ty::Local(Box::new(Ty::Func)),
        NType::List(x) => Ty::List(b(x)),
        NType::Dict(k, v) => Ty::Dict(b(k), b(v)),
        NType::Option(x) => Ty::Option(b(x)),
        NType::Array(e) => Ty::Array(Some(*e)),
        NType::Class(id) => match classes.get(id) {
            Some((n, args)) => Ty::Instance(n.clone(), args.iter().map(|a| ntype_ty(a, classes)).collect()),
            None => Ty::Unknown,
        },
    }
}

/// The Lizzard name and type arguments of the native class over each Rust
/// type, by class key (`geom.Stack`, `[int]` for an instantiation of a
/// generic class).
type NativeNames = HashMap<u64, (String, Vec<NType>)>;

fn native_class_info(c: &lizzard_vm::native::ClassSpec, m: &ModuleSpec, names: &NativeNames) -> ClassInfo {
    let mut methods: HashMap<String, Sig> = c.methods.iter().map(|f| (f.name.clone(), native_sig(f, names))).collect();
    if let Some(init) = &c.init {
        methods.insert("__init__".into(), native_sig(init, names));
    }
    let instances: Vec<(Vec<Ty>, ClassInfo)> =
        c.instances.iter().map(|(a, i)| (a.iter().map(|t| ntype_ty(t, names)).collect(), native_class_info(i, m, names))).collect();
    let tparams = c.tparams.iter().map(|n| TParamSig { name: n.clone(), bounds: Vec::new() }).collect();
    // The module's traits are qualified (`geom.Shape`); builtin ones are not.
    let mut impls: Vec<(String, Vec<Ty>)> = c
        .traits
        .iter()
        .map(|t| (if m.traits.iter().any(|x| x.name == *t) { format!("{}.{t}", m.name) } else { t.clone() }, Vec::new()))
        .collect();
    if let Some((_, first)) = instances.first() {
        // A generic family implements what every instantiation does.
        impls = first.impls.iter().filter(|t| instances.iter().all(|(_, i)| i.impls.contains(t))).cloned().collect();
    }
    ClassInfo { local: c.local, tparams, fields: Vec::new(), methods, trait_methods: HashMap::new(), impls, instances, native: true, trait_descs: Vec::new() }
}

fn native_sig(f: &FnSpec, classes: &NativeNames) -> Sig {
    let params = f
        .params
        .iter()
        .enumerate()
        .map(|(i, t)| ParamSig { name: format!("#{}", i + 1), keep: false, ty: ntype_ty(t, classes) })
        .collect();
    Sig { tparams: Vec::new(), params, ret: ntype_ty(&f.ret, classes) }
}

/// Compiles a module that may import the native modules `natives`.
/// Checks the top level of a program: only items (functions, structs,
/// traits, impls, imports), and a `main` entry point taking nothing or the
/// arguments (`args: list<str>`) and returning `()` or an exit status `int`.
fn check_program(module: &Module) -> R<()> {
    let mut main = None;
    for s in &module.body.stmts {
        match &s.kind {
            StmtKind::Def(f) => {
                if f.name == "main" {
                    main = Some(f);
                }
            }
            StmtKind::Struct(_) | StmtKind::Trait(_) | StmtKind::Impl(_) | StmtKind::Import { .. } | StmtKind::FromImport { .. } | StmtKind::Pass => {}
            // A docstring.
            StmtKind::Expr(Expr { kind: ExprKind::Str(_), .. }) => {}
            _ => return Err(syn_err("statements must be inside a function: the program starts at `main` (`def main() -> ():`)", s.span)),
        }
    }
    let Some(f) = main else { return Err(name_err("the program has no `main` function: define `def main() -> ():`", Span::default())) };
    let named = |t: &TypeExpr, n: &str, args: &[&str]| {
        matches!(t, TypeExpr::Named(m, a, _) if m == n && a.len() == args.len() && a.iter().zip(args).all(|(x, y)| matches!(x, TypeExpr::Named(xn, xa, _) if xn == y && xa.is_empty())))
    };
    if !f.tparams.is_empty() {
        return Err(type_err("`main` cannot be generic", f.span));
    }
    match f.params.as_slice() {
        [] => {}
        [p] if !p.keep && p.ty.as_ref().is_some_and(|t| named(t, "list", &["str"])) => {}
        _ => return Err(type_err("`main` takes no parameters or the program's arguments: `def main(args: list<str>)`", f.span)),
    }
    match &f.ret {
        Some(TypeExpr::Unit(_)) => {}
        Some(t) if named(t, "int", &[]) => {}
        _ => return Err(type_err("`main` must return `()` or an exit status `int`", f.span)),
    }
    if f.body.as_ref().is_some_and(scan::has_yield) {
        return Err(type_err("`main` cannot be a generator", f.span));
    }
    Ok(())
}

pub fn compile_module_with(module: &Module, filename: &str, natives: &[ModuleSpec]) -> R<(Arc<Code>, Vec<String>)> {
    check_program(module)?;
    let mut module = module.clone();
    let used = crate::imports::resolve(&mut module, natives)?;
    let module = &module;
    let mut globals: Vec<String> = lizzard_vm::builtins::names().into_iter().map(String::from).collect();
    for m in &used {
        for n in m.functions.iter().map(|f| &f.name).chain(m.classes.iter().map(|c| &c.name)).chain(m.traits.iter().map(|t| &t.name)) {
            globals.push(format!("{}.{n}", m.name));
        }
    }
    let builtin_names: HashSet<String> = globals.iter().cloned().collect();
    // The program's items: functions, structs, traits, associated functions.
    let mut module_vars = Vec::new();
    scan::assigned_names(&module.body, &mut module_vars);
    for n in &module_vars {
        if !globals.contains(n) {
            globals.push(n.clone());
        }
    }
    let gindex: HashMap<String, u32> = globals.iter().enumerate().map(|(i, n)| (n.clone(), i as u32)).collect();
    let builtin_classes: HashSet<String> = lizzard_vm::builtins::names()
        .into_iter()
        .filter(|n| n.ends_with("Error") || *n == "StopIteration" || *n == "GeneratorExit")
        .filter(|n| *n != "LookupError" && *n != "ArithmeticError")
        .map(String::from)
        .collect();
    let builtin_traits: HashSet<String> = BUILTIN_TRAITS.iter().map(|t| t.0.to_string()).collect();
    let mut info = ModInfo {
        impl_methods: HashMap::new(),
        assoc_fns: Vec::new(),
        impls: Vec::new(),
        gindex,
        module_vars: module_vars.iter().cloned().collect(),
        funcs: HashMap::new(),
        classes: HashMap::new(),
        traits: HashMap::new(),
        builtin_classes,
        builtin_traits,
        filename: filename.into(),
    };
    // Native classes and functions of the imported modules.
    let mut native_names: NativeNames = HashMap::new();
    for m in &used {
        for c in &m.classes {
            let n = format!("{}.{}", m.name, c.name);
            for (args, i) in &c.instances {
                native_names.insert(i.key, (n.clone(), args.clone()));
            }
            native_names.insert(c.key, (n, Vec::new()));
        }
    }
    for m in &used {
        for t in &m.traits {
            let methods = t.methods.iter().map(|f| (f.name.clone(), native_sig(f, &native_names))).collect();
            info.traits.insert(format!("{}.{}", m.name, t.name), TraitInfo { tparams: Vec::new(), methods, native: true });
        }
        for c in &m.classes {
            info.classes.insert(format!("{}.{}", m.name, c.name), native_class_info(c, m, &native_names));
        }
        for f in &m.functions {
            info.funcs.insert(format!("{}.{}", m.name, f.name), native_sig(f, &native_names));
        }
    }
    let names = |tps: &[TypeParam]| tps.iter().map(|t| TParamSig { name: t.name.clone(), bounds: Vec::new() }).collect::<Vec<_>>();
    let scope_of = |tps: &[TypeParam]| tps.iter().enumerate().map(|(k, t)| (t.name.clone(), k as u16)).collect::<Vec<_>>();
    // Register struct and trait names (and arities) first so annotations can
    // refer to them.
    for s in &module.body.stmts {
        match &s.kind {
            StmtKind::Struct(c) => {
                if builtin_names.contains(&c.name) {
                    return Err(syn_err(format!("cannot redefine builtin `{}`", c.name), c.span));
                }
                if info.classes.contains_key(&c.name) {
                    return Err(syn_err(format!("struct `{}` is defined twice", c.name), c.span));
                }
                let ci = ClassInfo {
                    local: false,
                    tparams: names(&c.tparams),
                    fields: Vec::new(),
                    methods: HashMap::new(),
                    trait_methods: HashMap::new(),
                    impls: Vec::new(),
                    instances: Vec::new(),
                    native: false,
                    trait_descs: Vec::new(),
                };
                info.classes.insert(c.name.clone(), ci);
            }
            StmtKind::Trait(t) => {
                if builtin_names.contains(&t.name) {
                    return Err(syn_err(format!("cannot redefine builtin `{}`", t.name), t.span));
                }
                info.traits.insert(t.name.clone(), TraitInfo { tparams: names(&t.tparams), methods: HashMap::new(), native: false });
            }
            _ => {}
        }
    }
    // Bounds of struct and trait type parameters.
    for s in &module.body.stmts {
        match &s.kind {
            StmtKind::Struct(c) => {
                let tps = info.tparam_sigs(&c.tparams, &scope_of(&c.tparams))?;
                info.classes.get_mut(&c.name).unwrap().tparams = tps;
            }
            StmtKind::Trait(t) => {
                let tps = info.tparam_sigs(&t.tparams, &scope_of(&t.tparams))?;
                info.traits.get_mut(&t.name).unwrap().tparams = tps;
            }
            _ => {}
        }
    }
    for s in &module.body.stmts {
        match &s.kind {
            StmtKind::Def(f) => {
                let sig = info.sig_of(f, false, &[])?;
                info.funcs.insert(f.name.clone(), sig);
            }
            StmtKind::Trait(t) => {
                // `Self` in a trait's signatures is the implementing type.
                let mut tps = info.trait_tparams(&t.name);
                tps.push(TParamSig { name: "Self".into(), bounds: Vec::new() });
                let mut ms = HashMap::new();
                for m in &t.methods {
                    check_method_typing(m)?;
                    ms.insert(m.name.clone(), info.sig_of(m, true, &tps)?);
                }
                info.traits.get_mut(&t.name).unwrap().methods = ms;
            }
            StmtKind::Struct(c) => {
                let scope = scope_of(&c.tparams);
                let mut fields = Vec::new();
                for f in &c.fields {
                    if fields.iter().any(|(n, _, _): &(String, Ty, TypeDesc)| *n == f.name) {
                        return Err(syn_err(format!("duplicate field `{}`", f.name), f.span));
                    }
                    let (ty, d) = info.resolve_scoped(&f.ty, &scope)?;
                    fields.push((f.name.clone(), ty, d));
                }
                info.classes.get_mut(&c.name).unwrap().fields = fields;
            }
            _ => {}
        }
    }
    // `impl` blocks: methods, associated functions and implemented traits.
    for s in &module.body.stmts {
        if let StmtKind::Impl(i) = &s.kind {
            collect_impl(&mut info, module, i)?;
        }
    }
    check_coherence(&info)?;
    // `Send`, as in Rust: a struct is not `Send` (it is local) when a field
    // type is not (to a fixpoint: fields may name later structs). Native
    // classes over non-`Send` Rust types are the roots. Within the struct, its
    // own type parameters count as `Send`: the type arguments decide
    // (`Box<Counter>` is not `Send` when `Counter` is not).
    loop {
        let found: Vec<String> = info
            .classes
            .iter()
            .filter(|(_, ci)| !ci.local && ci.fields.iter().any(|(_, t, _)| info.is_local(t, &|_| false)))
            .map(|(n, _)| n.clone())
            .collect();
        if found.is_empty() {
            break;
        }
        for n in found {
            info.classes.get_mut(&n).unwrap().local = true;
        }
    }
    // Trait default methods become available on implementing structs, with
    // the trait's type parameters replaced by the impl's arguments.
    for s in &module.body.stmts {
        if let StmtKind::Struct(c) = &s.kind {
            let ci = &info.classes[&c.name];
            let self_ty = Ty::Instance(c.name.clone(), ci.tparams.iter().map(|t| Ty::Param(t.name.clone())).collect());
            for (tname, targs) in ci.impls.clone() {
                if let Some(tm) = info.trait_methods_for(&tname, &targs, self_ty.clone()) {
                    let ci = info.classes.get_mut(&c.name).unwrap();
                    for (n, sig) in tm {
                        let e = ci.trait_methods.entry(n).or_default();
                        if !e.iter().any(|(t, _)| *t == tname) {
                            e.push((tname.clone(), sig));
                        }
                    }
                }
            }
        }
    }
    let mut ctx = FnCtx::new(&info, "<module>".into(), true);
    ctx.state = Some(HashMap::new());
    ctx.hoist_items(module)?;
    ctx.block(&module.body)?;
    let r = ctx.tmp_owned();
    ctx.emit(Instr::LoadImm { dst: r, bits: Value::UNIT.0 as i32 });
    ctx.emit(Instr::Return { src: r });
    let code = ctx.finish(Vec::new(), RetKind::Any, false, false);
    Ok((Arc::new(code), globals))
}

/// Renames type names in a type expression (`T` -> `U`).
fn rename_texpr(t: &TypeExpr, map: &dyn Fn(&str) -> Option<String>) -> TypeExpr {
    match t {
        TypeExpr::Unit(s) => TypeExpr::Unit(*s),
        TypeExpr::Send(x) => TypeExpr::Send(Box::new(rename_texpr(x, map))),
        TypeExpr::Named(n, args, sp) => {
            let n = if args.is_empty() { map(n).unwrap_or_else(|| n.clone()) } else { n.clone() };
            TypeExpr::Named(n, args.iter().map(|a| rename_texpr(a, map)).collect(), *sp)
        }
    }
}

/// Records an `impl` block. An impl of a whole struct of the program
/// (`impl<A, B> S<A, B>` with the struct's own bounds) adds inherent methods
/// and associated functions, or implements a trait, for the struct itself;
/// any other impl is matched against types (`collect_registry_impl`).
fn collect_impl(info: &mut ModInfo, module: &Module, i: &ImplDef) -> R<()> {
    if let TypeExpr::Named(tn, _, tspan) = &i.target {
        if info.builtin_classes.contains(tn) {
            return Err(type_err(format!("cannot implement methods or traits for builtin class `{tn}`"), *tspan));
        }
        if info.traits.contains_key(tn) || builtin_trait(tn).is_some() {
            return Err(type_err(format!("`{tn}` is a trait; write `impl {tn} for Type:`"), *tspan));
        }
        if let Some(pnames) = merged_params(info, i)? {
            return collect_struct_impl(info, module, i, tn, pnames);
        }
    }
    collect_registry_impl(info, module, i)
}

/// For an impl of a whole struct of the program (`impl<A, B> S<A, B>`,
/// distinct parameters, no bounds beyond the struct's): the impl's names for
/// the struct's type parameters.
fn merged_params(info: &ModInfo, i: &ImplDef) -> R<Option<Vec<String>>> {
    let TypeExpr::Named(tn, targs, _) = &i.target else { return Ok(None) };
    let Some(ci) = info.classes.get(tn).filter(|c| !c.native) else { return Ok(None) };
    let ctps = &ci.tparams;
    let pnames: Vec<String> = targs
        .iter()
        .filter_map(|a| match a {
            TypeExpr::Named(n, xs, _) if xs.is_empty() && i.tparams.iter().any(|t| t.name == *n) => Some(n.clone()),
            _ => None,
        })
        .collect();
    let distinct = pnames.iter().collect::<HashSet<_>>().len() == pnames.len();
    if targs.len() != ctps.len() || i.tparams.len() != ctps.len() || pnames.len() != ctps.len() || !distinct {
        return Ok(None);
    }
    let to_class = |n: &str| pnames.iter().position(|p| p == n).map(|k| Ty::Param(ctps[k].name.clone()));
    let scope: Vec<(String, u16)> = pnames.iter().enumerate().map(|(k, n)| (n.clone(), k as u16)).collect();
    for tp in info.tparam_sigs(&i.tparams, &scope)? {
        let k = pnames.iter().position(|p| *p == tp.name).unwrap();
        if tp.bounds.iter().any(|b| !ctps[k].bounds.contains(&b.subst(&to_class))) {
            return Ok(None);
        }
    }
    Ok(Some(pnames))
}

fn collect_struct_impl(info: &mut ModInfo, module: &Module, i: &ImplDef, tn: &String, pnames: Vec<String>) -> R<()> {
    let ci = &info.classes[tn];
    let ctps = ci.tparams.clone();
    // Impl names -> struct names, and the impl's scope (struct positions).
    let to_class = |n: &str| pnames.iter().position(|p| p == n).map(|k| Ty::Param(ctps[k].name.clone()));
    let scope: Vec<(String, u16)> = pnames.iter().enumerate().map(|(k, n)| (n.clone(), k as u16)).collect();
    // The impl's parameters with the struct's bounds, in impl names.
    let to_impl = |n: &str| ctps.iter().position(|t| t.name == n).map(|k| pnames[k].clone());
    let sdef = module.body.stmts.iter().find_map(|s| match &s.kind {
        StmtKind::Struct(d) if d.name == *tn => Some(d),
        _ => None,
    });
    let impl_tparams: Vec<TypeParam> = sdef
        .map(|d| {
            d.tparams
                .iter()
                .enumerate()
                .map(|(k, tp)| TypeParam { name: pnames[k].clone(), bounds: tp.bounds.iter().map(|b| rename_texpr(b, &to_impl)).collect(), span: tp.span })
                .collect()
        })
        .unwrap_or_default();
    let outer: Vec<TParamSig> = pnames.iter().enumerate().map(|(k, n)| TParamSig { name: n.clone(), bounds: ctps[k].bounds.iter().map(|b| b.subst(&|x| to_impl(x).map(Ty::Param))).collect() }).collect();
    let self_ty = Ty::Instance(tn.clone(), ctps.iter().map(|t| Ty::Param(t.name.clone())).collect());
    let fields: Vec<String> = ci.fields.iter().map(|f| f.0.clone()).collect();
    let mut sigs: Vec<(String, Sig, Span)> = Vec::new();
    for m in &i.methods {
        let is_method = m.params.first().is_some_and(|p| p.name == "self");
        if m.name == "__init__" {
            return Err(type_err(
                format!("structs have no `__init__`: define `def new(...) -> Self:` returning `{tn} {{ .. }}` and call `{tn}::new(...)`"),
                m.span,
            ));
        }
        if !is_method {
            if let Some((tr, _, _)) = &i.tr {
                return Err(type_err(format!("`{}` in `impl {tr} for {tn}` must take `self`: traits have no associated functions", m.name), m.span));
            }
            // An associated function: a module function `Type::name` whose
            // type parameters are the impl's, then its own.
            let mut def = m.clone();
            def.name = format!("{tn}::{}", m.name);
            def.tparams = impl_tparams.iter().cloned().chain(m.tparams.iter().cloned()).collect();
            if info.funcs.contains_key(&def.name) {
                return Err(type_err(format!("`{}` is defined twice", def.name), m.span));
            }
            let sig = info.sig_of(&def, false, &[])?;
            info.funcs.insert(def.name.clone(), sig);
            info.assoc_fns.push(def);
            continue;
        }
        check_method_typing(m)?;
        if fields.contains(&m.name) {
            return Err(syn_err(format!("`{}` is both a field and a method of `{tn}`", m.name), m.span));
        }
        if m.name == "__drop__" && !matches!(&i.tr, Some((t, _, _)) if t == "Drop") {
            return Err(type_err(format!("a destructor is defined by implementing `Drop`: `impl Drop for {tn}:`"), m.span));
        }
        let sig = info.sig_of(m, true, &outer)?.subst(&to_class);
        let twice = sigs.iter().any(|(n, _, _)| *n == m.name) || (i.tr.is_none() && info.classes[tn].methods.contains_key(&m.name));
        if twice {
            return Err(type_err(format!("method `{}` of `{tn}` is defined twice", m.name), m.span));
        }
        sigs.push((m.name.clone(), sig, m.span));
    }
    if let Some((tr, trargs, trspan)) = &i.tr {
        check_trait_impl(info, module, tn, &self_ty, tr, trargs, *trspan, &scope, &to_class, &sigs, i.span)?;
    }
    let ci = info.classes.get_mut(tn).unwrap();
    let tr = i.tr.as_ref().map(|t| t.0.clone());
    for (n, sig, _) in sigs {
        match &tr {
            Some(t) => ci.trait_methods.entry(n).or_default().push((t.clone(), sig)),
            None => {
                ci.methods.insert(n, sig);
            }
        }
    }
    let methods = i.methods.iter().filter(|m| m.params.first().is_some_and(|p| p.name == "self")).cloned();
    info.impl_methods.entry(tn.clone()).or_default().extend(methods.map(|def| ImplMethod { tr: tr.clone(), def, tparams: pnames.clone() }));
    Ok(())
}

/// Checks `impl tr<trargs> for tn` (methods `sigs`, in the struct's names)
/// and records the trait (and the traits it implies) on the struct.
#[allow(clippy::too_many_arguments)]
fn check_trait_impl(
    info: &mut ModInfo,
    module: &Module,
    tn: &str,
    self_ty: &Ty,
    tr: &str,
    trargs: &[TypeExpr],
    trspan: Span,
    scope: &[(String, u16)],
    to_class: &dyn Fn(&str) -> Option<Ty>,
    sigs: &[(String, Sig, Span)],
    span: Span,
) -> R<()> {
    let (args, descs) = trait_header(info, tn, tr, trargs, trspan, scope, to_class)?;
    if info.classes[tn].impls.iter().any(|(t, _)| t == tr) {
        return Err(type_err(format!("`{tn}` implements `{tr}` twice"), trspan));
    }
    check_trait_methods(info, module, tn, self_ty, tr, &args, sigs, span, &|_| None)?;
    let ci = info.classes.get_mut(tn).unwrap();
    ci.impls.push((tr.to_string(), args.clone()));
    ci.trait_descs.push((tr.to_string(), descs));
    if let Some((_, _, implies, _)) = builtin_trait(tr) {
        for i in implies.iter() {
            if !ci.impls.iter().any(|(t, _)| t == i) {
                let a = if builtin_trait(i).is_some_and(|b| b.1 == args.len()) { args.clone() } else { Vec::new() };
                ci.impls.push((i.to_string(), a));
            }
        }
    }
    Ok(())
}

/// The trait of `impl tr<trargs> for tn`: checks that it is one that can be
/// implemented and resolves its arguments (static and runtime forms).
fn trait_header(
    info: &ModInfo,
    tn: &str,
    tr: &str,
    trargs: &[TypeExpr],
    trspan: Span,
    scope: &[(String, u16)],
    to_class: &dyn Fn(&str) -> Option<Ty>,
) -> R<(Vec<Ty>, Vec<TypeDesc>)> {
    if tr == "Send" {
        return Err(type_err(format!("`Send` is automatic: `{tn}` is `Send` when all of its fields are"), trspan));
    }
    if tr == "Local" {
        return Err(type_err("there is no `Local` trait: as in Rust, a struct is `Send` unless a field is not (e.g. an `any` field)", trspan));
    }
    let arity = match (info.traits.get(tr), builtin_trait(tr)) {
        (Some(t), _) => t.tparams.len(),
        (None, Some(b)) => b.1,
        (None, None) => {
            if info.classes.contains_key(tr) || info.builtin_classes.contains(tr) {
                return Err(type_err(format!("`{tr}` is a struct, not a trait; structs cannot inherit (use a field and delegate by hand)"), trspan));
            }
            return Err(name_err(format!("unknown trait `{tr}`"), trspan));
        }
    };
    if trargs.len() != arity {
        return Err(type_err(
            match arity {
                0 => format!("trait `{tr}` does not take type arguments"),
                n => format!("trait `{tr}` takes {n} type argument{}: `impl {tr}<...> for {tn}`", if n == 1 { "" } else { "s" }),
            },
            trspan,
        ));
    }
    let mut args = Vec::new();
    let mut descs = Vec::new();
    for a in trargs {
        let (t, d) = info.resolve_scoped(a, scope)?;
        args.push(t.subst(to_class));
        descs.push(d);
    }
    Ok((args, descs))
}

/// Checks the methods `sigs` of `impl tr<args> for tn` against the trait:
/// every required one defined (or `provided` by a native target's Rust
/// class), with the trait's signature, and nothing the trait lacks.
#[allow(clippy::too_many_arguments)]
fn check_trait_methods(
    info: &ModInfo,
    module: &Module,
    tn: &str,
    self_ty: &Ty,
    tr: &str,
    args: &[Ty],
    sigs: &[(String, Sig, Span)],
    span: Span,
    provided: &dyn Fn(&str) -> Option<Sig>,
) -> R<()> {
    let shown = if args.is_empty() { tr.to_string() } else { format!("{tr}<{}>", args.iter().map(|a| a.name()).collect::<Vec<_>>().join(", ")) };
    let what = format!("`impl {shown} for {tn}`");
    let show = |s: &Sig| format!("({}) -> {}", s.params.iter().map(|p| p.ty.name()).collect::<Vec<_>>().join(", "), s.ret.name());
    let same = |have: &Sig, w: &Sig| have.params.len() == w.params.len() && have.params.iter().zip(&w.params).all(|(h, x)| h.ty == x.ty) && ty_compatible(&w.ret, &have.ret);
    match (info.traits.get(tr), builtin_trait(tr)) {
        (Some(ti), _) => {
            let want = info.trait_methods_for(tr, args, self_ty.clone()).unwrap_or_default();
            let tdef = module.body.stmts.iter().find_map(|s| match &s.kind {
                StmtKind::Trait(t) if t.name == tr => Some(t),
                _ => None,
            });
            let mut required: Vec<&String> = match tdef {
                Some(d) => ti.methods.keys().filter(|n| d.methods.iter().any(|m| m.name == **n && m.body.is_none())).collect(),
                None => ti.methods.keys().collect(),
            };
            required.sort();
            for n in required {
                if sigs.iter().any(|(m, _, _)| m == n) {
                    continue;
                }
                let Some(have) = provided(n) else {
                    return Err(type_err(format!("{what}: `{tn}` does not define required method `{n}`"), span));
                };
                if !same(&have, &want[n]) {
                    return Err(type_err(format!("{what}: `{tn}.{n}` is `{}`, but `{tr}` needs `{}`", show(&have), show(&want[n])), span));
                }
            }
            for (n, have, mspan) in sigs {
                let Some(w) = want.get(n) else {
                    return Err(type_err(format!("{what}: `{n}` is not a method of trait `{tr}`"), *mspan));
                };
                if !same(have, w) {
                    return Err(type_err(format!("{what}: `{n}` is `{}`, but `{tr}` needs `{}`", show(have), show(w)), *mspan));
                }
            }
        }
        (None, Some((_, _, implies, methods))) => {
            // The trait's own methods are required; those of the traits it
            // implies (`Num` implies `Ord`) may be defined here too.
            let allowed: Vec<&str> = methods.iter().chain(implies.iter().filter_map(|i| builtin_trait(i)).flat_map(|t| t.3.iter())).copied().collect();
            for m in methods.iter() {
                if !sigs.iter().any(|(n, _, _)| n == m) && provided(m).is_none() {
                    return Err(type_err(format!("{what}: `{tn}` does not define required method `{m}`"), span));
                }
            }
            if let Some((n, _, mspan)) = sigs.iter().find(|(n, _, _)| !allowed.contains(&n.as_str())) {
                return Err(type_err(format!("{what}: `{n}` is not a method of trait `{tr}`; define it in `impl {tn}:`"), *mspan));
            }
        }
        (None, None) => unreachable!(),
    }
    Ok(())
}

/// Records an impl that is matched against types (see `ImplInfo`), after
/// the orphan rule: a trait impl needs the trait or the target to be the
/// program's, and an inherent impl a struct of the program.
fn collect_registry_impl(info: &mut ModInfo, module: &Module, i: &ImplDef) -> R<()> {
    let tspan = i.target.span();
    let scope: Vec<(String, u16)> = i.tparams.iter().enumerate().map(|(k, t)| (t.name.clone(), k as u16)).collect();
    let tparams = info.tparam_sigs(&i.tparams, &scope)?;
    let (target, target_desc) = info.resolve_scoped(&i.target, &scope)?;
    let tn = target.name();
    match &target {
        Ty::Int | Ty::Float | Ty::Bool | Ty::BigInt | Ty::Str | Ty::List(_) | Ty::Dict(..) | Ty::Tuple(Some(_)) | Ty::Array(_) | Ty::Chan(_) | Ty::Opaque(_) | Ty::Instance(..) => {}
        Ty::Param(_) => return Err(type_err(format!("blanket impls (`impl<{tn}> Trait for {tn}`) are not supported; implement the trait for each type"), tspan)),
        Ty::Option(_) => return Err(type_err(format!("`{tn}` cannot be an impl target: an `Option` value is its inner value or `None`"), tspan)),
        _ => {
            return Err(type_err(
                format!("`{tn}` cannot be an impl target: implement traits for concrete types, not erased ones (`any`, `function`, `Iterator`, traits)"),
                tspan,
            ))
        }
    }
    if let Some(tp) = tparams.iter().find(|tp| !mentions(&target, &tp.name)) {
        return Err(type_err(format!("type parameter `{}` is not used by the impl's target `{tn}`", tp.name), tspan));
    }
    let pb = |n: &str| tparams.iter().find(|t| t.name == n).map(|t| t.bounds.clone()).unwrap_or_default();
    info.check_wf(&target, &pb, tspan)?;
    let class = match &target {
        Ty::Instance(c, _) => info.classes.get(c),
        _ => None,
    };
    let own_struct = class.is_some_and(|c| !c.native);
    let native = class.is_some_and(|c| c.native);
    let tr: Option<(String, Vec<Ty>, Vec<TypeDesc>)> = match &i.tr {
        None if native => {
            return Err(type_err(format!("`{tn}` is a native class: its methods are defined in Rust; add yours through a trait (`impl MyTrait for {tn}`)"), tspan))
        }
        None if !own_struct => {
            return Err(type_err(
                format!("inherent impls are only allowed for this program's structs; for `{tn}`, define a trait and implement it (`impl MyTrait for {tn}:`)"),
                tspan,
            ))
        }
        None => None,
        Some((tr, trargs, trspan)) => {
            let (args, descs) = trait_header(info, &tn, tr, trargs, *trspan, &scope, &|_| None)?;
            let own_trait = info.traits.get(tr).is_some_and(|t| !t.native);
            if !own_trait && !own_struct {
                let msg = if native && info.traits.get(tr).is_some_and(|t| t.native) {
                    format!("`{tr}` is a native trait: its implementations for native classes are declared in Rust (`c.implements(..)`)")
                } else if native {
                    format!("`impl {tr} for {tn}`: both are foreign (orphan rule: the trait or the type must be defined in this program); declare it in Rust (`c.implements(\"{tr}\")`)")
                } else {
                    format!("`impl {tr} for {tn}`: both are foreign (orphan rule: the trait or the type must be defined in this program)")
                };
                return Err(type_err(msg, *trspan));
            }
            let exception = tr == "Exception" || builtin_trait(tr).is_some_and(|b| b.2.contains(&"Exception"));
            if tr == "Drop" || exception {
                return Err(type_err(format!("`{tr}` can only be implemented for a whole struct (`impl<..> {tr} for S<..>`)"), *trspan));
            }
            Some((tr.clone(), args, descs))
        }
    };
    let fields: Vec<String> = class.map(|c| c.fields.iter().map(|f| f.0.clone()).collect()).unwrap_or_default();
    let mut sigs: Vec<(String, Sig, Span)> = Vec::new();
    for m in &i.methods {
        if m.params.first().is_none_or(|p| p.name != "self") {
            let msg = match &tr {
                Some((t, _, _)) => format!("`{}` in `impl {t} for {tn}` must take `self`: traits have no associated functions", m.name),
                None => format!("`{}`: associated functions need an impl of the whole struct (`impl<..> S<..>:`), not `impl {tn}`", m.name),
            };
            return Err(type_err(msg, m.span));
        }
        check_method_typing(m)?;
        if fields.contains(&m.name) {
            return Err(syn_err(format!("`{}` is both a field and a method of `{tn}`", m.name), m.span));
        }
        if m.name == "__drop__" {
            return Err(type_err("a destructor is defined by implementing `Drop` for the whole struct", m.span));
        }
        if sigs.iter().any(|(n, _, _)| *n == m.name) {
            return Err(type_err(format!("method `{}` of `{tn}` is defined twice", m.name), m.span));
        }
        sigs.push((m.name.clone(), info.sig_of(m, true, &tparams)?, m.span));
    }
    // A native target's Rust methods implement the trait's methods (each
    // instantiation of a generic class must have them).
    let rust_sig = |n: &str| -> Option<Sig> {
        let Ty::Instance(c, args) = &target else { return None };
        let ci = info.classes.get(c).filter(|c| c.native)?;
        if let Some(k) = info.class_info(c, args) {
            return k.methods.get(n).cloned();
        }
        let all: Vec<&ClassInfo> = ci.instances.iter().map(|(_, k)| k).collect();
        if all.is_empty() || !all.iter().all(|k| k.methods.contains_key(n)) {
            return None;
        }
        all[0].methods.get(n).cloned()
    };
    let mut methods: HashMap<String, Sig> = sigs.iter().map(|(n, s, _)| (n.clone(), s.clone())).collect();
    if let Some((t, args, _)) = &tr {
        check_trait_methods(info, module, &tn, &target, t, args, &sigs, i.span, &rust_sig)?;
        for (n, s) in info.trait_methods_for(t, args, target.clone()).unwrap_or_default() {
            if !methods.contains_key(&n) && rust_sig(&n).is_none() {
                methods.insert(n, s);
            }
        }
    }
    let mut bound_descs = Vec::new();
    for (k, tp) in i.tparams.iter().enumerate() {
        for b in &tp.bounds {
            let d = match info.resolve_scoped(b, &scope)?.1 {
                TypeDesc::Local(d) => *d,
                d => d,
            };
            bound_descs.push((k as u16, d));
        }
    }
    let shown = match &tr {
        Some((t, args, _)) => format!("impl {} for {tn}", Ty::Trait(t.clone(), args.clone()).name()),
        None => format!("impl {tn}"),
    };
    let (tr, trait_descs) = match tr {
        Some((t, a, d)) => (Some((t, a)), d),
        None => (None, Vec::new()),
    };
    info.impls.push(ImplInfo { tparams, tr, target, target_desc, bound_descs, trait_descs, methods, defs: i.methods.clone(), shown, span: i.span });
    Ok(())
}

/// Coherence, as in Rust: no two impls of a trait for overlapping types
/// (bounds are ignored), and no method defined twice for overlapping types.
fn check_coherence(info: &ModInfo) -> R<()> {
    for (k, im) in info.impls.iter().enumerate() {
        let head = match &im.target {
            Ty::Instance(c, _) => info.classes.get(c).map(|ci| (c, ci)),
            _ => None,
        };
        match &im.tr {
            Some((tr, _)) => {
                if let Some(o) = info.impls[..k].iter().find(|o| o.tr.as_ref().is_some_and(|(t, _)| t == tr) && overlaps(&im.target, &o.target)) {
                    return Err(type_err(format!("conflicting implementations of `{tr}`: `{}` overlaps `{}`", im.shown, o.shown), im.span));
                }
                if let Some((c, _)) = head.filter(|(_, ci)| ci.impls.iter().any(|(t, _)| t == tr)) {
                    return Err(type_err(format!("conflicting implementations of `{tr}`: `{}` overlaps the implementation for every `{c}`", im.shown), im.span));
                }
            }
            None => {
                for n in im.defs.iter().map(|d| &d.name) {
                    if let Some((c, _)) = head.filter(|(_, ci)| ci.methods.contains_key(n)) {
                        return Err(type_err(format!("method `{n}` is defined twice: in `{}` and in the impl for every `{c}`", im.shown), im.span));
                    }
                    let twice = info.impls[..k].iter().find(|o| o.tr.is_none() && overlaps(&im.target, &o.target) && o.defs.iter().any(|d| d.name == *n));
                    if let Some(o) = twice {
                        return Err(type_err(format!("method `{n}` is defined twice, in `{}` and `{}` (their targets overlap)", im.shown, o.shown), im.span));
                    }
                }
            }
        }
    }
    Ok(())
}

fn check_method_typing(m: &FuncDef) -> R<()> {
    match m.params.first() {
        Some(p) if p.name == "self" => {
            if p.ty.is_some() {
                return Err(syn_err("`self` must not be annotated", p.span));
            }
        }
        _ => return Err(syn_err(format!("method `{}` must take `self` as its first parameter", m.name), m.span)),
    }
    for p in &m.params[1..] {
        if p.ty.is_none() {
            return Err(syn_err(
                format!("parameter `{}` of method `{}` needs a type annotation (all method parameters must be typed)", p.name, m.name),
                p.span,
            ));
        }
    }
    if m.ret.is_none() {
        return Err(syn_err(
            format!("method `{}` needs a return type (use `-> ()` if it returns nothing)", m.name),
            m.span,
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- ownership state

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Uninit,
    Live,
    Moved,
    MaybeMoved,
    /// Possibly moved by an operation whose ownership effect is only known at
    /// runtime; uses are checked dynamically.
    DynMoved,
    MaybeUninit,
}

#[derive(Clone, Debug, PartialEq)]
struct Var {
    st: St,
    ty: Ty,
}

type State = HashMap<String, Var>;

fn join_st(a: St, b: St) -> St {
    use St::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Live, Moved) | (Moved, Live) | (MaybeMoved, _) | (_, MaybeMoved) => MaybeMoved,
        (DynMoved, Moved) | (Moved, DynMoved) => MaybeMoved,
        (DynMoved, _) | (_, DynMoved) => DynMoved,
        (Uninit, Moved) | (Moved, Uninit) => Moved,
        _ => MaybeUninit,
    }
}

fn join_states(a: &State, b: &State) -> State {
    let mut out = State::new();
    for k in a.keys().chain(b.keys()) {
        if out.contains_key(k) {
            continue;
        }
        if k.starts_with('@') {
            // Narrowed field paths survive a join only if narrowed on both sides.
            if let (Some(x), Some(y)) = (a.get(k), b.get(k)) {
                if x == y {
                    out.insert(k.clone(), x.clone());
                }
            }
            continue;
        }
        let x = a.get(k).cloned().unwrap_or(Var { st: St::Uninit, ty: Ty::Unknown });
        let y = b.get(k).cloned().unwrap_or(Var { st: St::Uninit, ty: Ty::Unknown });
        let ty = if x.st == St::Uninit {
            y.ty.clone()
        } else if y.st == St::Uninit {
            x.ty.clone()
        } else {
            x.ty.join(&y.ty)
        };
        out.insert(k.clone(), Var { st: join_st(x.st, y.st), ty });
    }
    out
}

fn join_opt(a: Option<State>, b: Option<State>) -> Option<State> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => Some(join_states(&a, &b)),
    }
}

// ---------------------------------------------------------------- function context

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LocalKind {
    Owned,
    /// A non-`keep` parameter: borrowed for the whole call.
    Borrowed,
}

struct Local {
    reg: Reg,
    kind: LocalKind,
    /// Declared annotation: (static type, runtime descriptor).
    decl: Option<(Ty, TypeDesc)>,
}

#[derive(Clone, Debug, PartialEq)]
enum Own {
    /// A temporary that owns its value.
    Owned,
    /// The register of a named local (or a module variable), borrowed.
    Local(String),
    /// A borrow of a parameter, field, element, global or capture.
    Borrowed,
}

#[derive(Clone, Debug)]
struct Res {
    reg: Reg,
    own: Own,
    ty: Ty,
    /// Local register this borrow derives from (`p` for `p.a.b`).
    root: Option<Reg>,
    /// Human description for error messages.
    desc: String,
}

struct LoopCtx {
    head: u32,
    breaks: Vec<usize>,
    continues: Vec<usize>,
    break_state: Option<State>,
    cont_state: Option<State>,
    finally_depth: usize,
}

struct FnCtx<'m> {
    m: &'m ModInfo,
    name: String,
    instrs: Vec<Instr>,
    /// See `Code::entry`.
    entry: u32,
    /// The expression being compiled is assigned to a float local (see
    /// `float_local_dest`); taken by the next `binary`.
    raw_dest: bool,
    lines: Vec<u32>,
    consts: Vec<Value>,
    names: Vec<String>,
    arglists: Vec<ArgList>,
    trait_calls: Vec<(u32, u32)>,
    types: Vec<TypeDesc>,
    codes: Vec<Arc<Code>>,
    classes: Vec<ClassSpec>,
    traits: Vec<TraitSpec>,
    impls: Vec<ImplSpec>,
    exc_table: Vec<ExcEntry>,
    owned: Vec<bool>,
    reg_names: Vec<Option<String>>,

    locals: HashMap<String, Local>,
    captures: Vec<(String, Ty)>,
    state: Option<State>,
    is_module: bool,
    owned_pool: Vec<Reg>,
    owned_used: usize,
    borrow_pool: Vec<Reg>,
    borrow_used: usize,
    loops: Vec<LoopCtx>,
    finally_stack: Vec<Block>,
    ret: RetKind,
    ret_ty: Ty,
    ret_desc: u32,
    is_generator: bool,
    line: u32,
    heap_borrows: usize,
    stmt_drops: Vec<Reg>,
    moved_log: Vec<String>,
    class_name: Option<String>,
    handler_exc: Vec<Reg>,
    /// Static types of call argument registers (for `Some(x)` typing).
    reg_ty: HashMap<Reg, Ty>,
    /// The type the context expects of the expression being compiled next
    /// (types empty literals and `list()`/`dict()`/`Channel()` calls).
    expect: Option<Ty>,
    /// Static types of the values this generator yields.
    yield_tys: Vec<Ty>,
    /// The declared item type of this generator (`-> Iterator<T>`).
    yield_decl: Option<Ty>,
    /// This generator may hold values that are not `Send` (it is not
    /// declared `-> Iterator<T> + Send`).
    gen_local: bool,
    /// Type parameters in scope, with the registers holding their types.
    tscope: Vec<TPar>,
    /// This function's own type parameters (hidden leading parameters).
    code_tparams: Vec<TypeParamInfo>,
    /// Static types of the arguments of the last `call_args`.
    last_arg_tys: Vec<Ty>,
}

/// A type parameter in scope: its name, the register holding its runtime
/// type, and its bounds.
#[derive(Clone, Debug)]
struct TPar {
    name: String,
    reg: Reg,
    bounds: Vec<Ty>,
}

impl<'m> FnCtx<'m> {
    fn new(m: &'m ModInfo, name: String, is_module: bool) -> FnCtx<'m> {
        FnCtx {
            m,
            name,
            instrs: Vec::new(),
            entry: 0,
            raw_dest: false,
            lines: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            arglists: Vec::new(),
            trait_calls: Vec::new(),
            types: Vec::new(),
            codes: Vec::new(),
            classes: Vec::new(),
            traits: Vec::new(),
            impls: Vec::new(),
            exc_table: Vec::new(),
            owned: Vec::new(),
            reg_names: Vec::new(),
            locals: HashMap::new(),
            captures: Vec::new(),
            state: None,
            is_module,
            owned_pool: Vec::new(),
            owned_used: 0,
            borrow_pool: Vec::new(),
            borrow_used: 0,
            loops: Vec::new(),
            finally_stack: Vec::new(),
            ret: RetKind::Any,
            ret_ty: Ty::Unknown,
            ret_desc: 0,
            is_generator: false,
            line: 1,
            heap_borrows: 0,
            stmt_drops: Vec::new(),
            moved_log: Vec::new(),
            class_name: None,
            handler_exc: Vec::new(),
            reg_ty: HashMap::new(),
            expect: None,
            yield_tys: Vec::new(),
            gen_local: false,
            yield_decl: None,
            tscope: Vec::new(),
            code_tparams: Vec::new(),
            last_arg_tys: Vec::new(),
        }
    }

    fn finish(mut self, params: Vec<ParamInfo>, ret: RetKind, is_generator: bool, is_method: bool) -> Code {
        crate::peephole::run(&mut self.instrs, &mut self.lines, &mut self.exc_table, &mut self.entry, &self.reg_names);
        let yield_ty = is_generator.then(|| {
            let t = match self.yield_decl.clone() {
                Some(t) => t,
                None => {
                    let tys = std::mem::take(&mut self.yield_tys);
                    let any = Ty::Local(Box::new(Ty::Any));
                    if tys.contains(&Ty::Unknown) {
                        any
                    } else {
                        match join_all(tys) {
                            Ty::NoneT => Ty::Option(Box::new(any)),
                            t => t,
                        }
                    }
                }
            };
            let d = self.elem_desc(&t);
            self.type_idx(d)
        });
        let type_cache = (0..self.types.len()).map(|_| std::sync::OnceLock::new()).collect();
        let ics = (0..self.instrs.len()).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
        Code {
            name: self.name,
            filename: self.m.filename.clone(),
            params,
            ret,
            nregs: self.owned.len() as u16,
            owned: self.owned,
            reg_names: self.reg_names,
            instrs: self.instrs,
            entry: self.entry,
            lines: self.lines,
            consts: self.consts,
            names: self.names,
            arglists: self.arglists,
            trait_calls: self.trait_calls,
            types: self.types,
            type_cache,
            ics,
            tparams: self.code_tparams,
            yield_ty,
            codes: self.codes,
            classes: self.classes,
            traits: self.traits,
            impls: self.impls,
            exc_table: self.exc_table,
            ncaptures: self.captures.len() as u16,
            is_generator,
            is_method,
        }
    }

    // ------------------------------------------------------------ emission

    fn emit(&mut self, i: Instr) -> usize {
        self.instrs.push(i);
        self.lines.push(self.line);
        self.instrs.len() - 1
    }

    fn here(&self) -> u32 {
        self.instrs.len() as u32
    }

    fn patch(&mut self, at: usize, target: u32) {
        match &mut self.instrs[at] {
            Instr::Jump { target: t }
            | Instr::JumpIfFalse { target: t, .. }
            | Instr::JumpIfTrue { target: t, .. }
            | Instr::MatchException { target: t, .. } => *t = target,
            Instr::ForIter { end, .. } => *end = target,
            other => panic!("cannot patch {other:?}"),
        }
    }

    /// Some jump or handler goes to `pc`.
    fn is_target(&self, pc: u32) -> bool {
        self.exc_table.iter().any(|e| e.start == pc || e.end == pc || e.handler == pc)
            || self.instrs.iter().any(|i| match *i {
                Instr::Jump { target }
                | Instr::Loop { target }
                | Instr::JumpIfFalse { target, .. }
                | Instr::JumpIfTrue { target, .. }
                | Instr::MatchException { target, .. }
                | Instr::CmpJump { target, .. } => target == pc,
                Instr::ForIter { end, .. } | Instr::ForIterSet { end, .. } => end == pc,
                _ => false,
            })
    }

    fn jump(&mut self) -> usize {
        self.emit(Instr::Jump { target: u32::MAX })
    }

    fn konst(&mut self, v: Value) -> u32 {
        if let Some(i) = self.consts.iter().position(|c| *c == v) {
            return i as u32;
        }
        self.consts.push(v);
        (self.consts.len() - 1) as u32
    }

    fn str_const(&mut self, s: &str) -> u32 {
        if let Some(i) = self.consts.iter().position(|c| c.as_str() == Some(s)) {
            return i as u32;
        }
        let v = Obj::alloc_immortal(Body::Str(s.into()));
        self.consts.push(v);
        (self.consts.len() - 1) as u32
    }

    fn name_idx(&mut self, s: &str) -> u32 {
        if let Some(i) = self.names.iter().position(|n| n == s) {
            return i as u32;
        }
        self.names.push(s.to_string());
        (self.names.len() - 1) as u32
    }

    fn type_idx(&mut self, t: TypeDesc) -> u32 {
        if let Some(i) = self.types.iter().position(|x| *x == t) {
            return i as u32;
        }
        self.types.push(t);
        (self.types.len() - 1) as u32
    }

    fn arglist(&mut self, al: ArgList) -> u32 {
        self.arglists.push(al);
        (self.arglists.len() - 1) as u32
    }

    fn new_reg(&mut self, owned: bool, name: Option<String>) -> Reg {
        self.owned.push(owned);
        self.reg_names.push(name);
        (self.owned.len() - 1) as Reg
    }

    fn tmp_owned(&mut self) -> Reg {
        if self.owned_used < self.owned_pool.len() {
            self.owned_used += 1;
            return self.owned_pool[self.owned_used - 1];
        }
        let r = self.new_reg(true, None);
        self.owned_pool.push(r);
        self.owned_used += 1;
        r
    }

    fn tmp_borrow(&mut self) -> Reg {
        if self.borrow_used < self.borrow_pool.len() {
            self.borrow_used += 1;
            return self.borrow_pool[self.borrow_used - 1];
        }
        let r = self.new_reg(false, None);
        self.borrow_pool.push(r);
        self.borrow_used += 1;
        r
    }

    fn load_imm(&mut self, dst: Reg, v: Value) {
        self.emit(Instr::LoadImm { dst, bits: v.0 as i64 as i32 });
    }

    // ------------------------------------------------------------ types

    fn scope(&self) -> Vec<(String, u16)> {
        self.tscope.iter().map(|t| (t.name.clone(), t.reg)).collect()
    }

    /// The bounds of type parameter `n` (empty if it is not in scope).
    fn pbounds(&self, n: &str) -> Vec<Ty> {
        self.tscope.iter().find(|t| t.name == n).map(|t| t.bounds.clone()).unwrap_or_default()
    }

    /// Resolves an annotation with the type parameters in scope, checking
    /// that its type arguments satisfy their bounds.
    fn resolve_type(&self, t: &TypeExpr) -> R<(Ty, TypeDesc)> {
        let r = self.m.resolve_scoped(t, &self.scope())?;
        self.m.check_wf(&r.0, &|n| self.pbounds(n), t.span())?;
        Ok(r)
    }

    fn satisfies(&self, t: &Ty, bound: &Ty) -> bool {
        self.m.satisfies(t, bound, &|n| self.pbounds(n))
    }

    /// Whether a value of static type `v` may be used where `d` is expected
    /// (trait types are checked against the traits `v` implements, and a
    /// `Send`-restricted erased type does not take a value that may not be
    /// `Send`).
    fn compatible(&self, d: &Ty, v: &Ty) -> bool {
        self.local_mismatch(d, v).is_none() && self.shape_compatible(d, v)
    }

    /// `compatible` without the locality check.
    fn shape_compatible(&self, d: &Ty, v: &Ty) -> bool {
        match (d.peel(), v.peel()) {
            (Ty::Trait(..), v) if !matches!(v, Ty::Unknown | Ty::Any | Ty::NoneT) => self.satisfies(v, d.peel()),
            (Ty::Option(a), Ty::Option(b)) => self.shape_compatible(a, b),
            (Ty::Option(a), b) if !matches!(b, Ty::NoneT) => self.shape_compatible(a, b),
            (d, v) => ty_compatible(d, v),
        }
    }

    /// Whether values of static type `t` may not be `Send` (an unbounded
    /// type parameter may not be; `T: Send` excludes it).
    fn may_be_local(&self, t: &Ty) -> bool {
        self.m.is_local(t, &|n| !has_send(&self.pbounds(n)))
    }

    /// Why a value of type `v` cannot go where `d` is expected for reasons of
    /// `Send`: `d` is (or contains, through `Option` and tuples) an erased
    /// type written `+ Send`, and `v` may not be `Send`.
    fn local_mismatch(&self, d: &Ty, v: &Ty) -> Option<String> {
        match (d, v) {
            (Ty::Option(a), Ty::Option(b)) => self.local_mismatch(a, b),
            (Ty::Option(a), b) => self.local_mismatch(a, b),
            (Ty::Tuple(Some(xs)), Ty::Tuple(Some(ys))) => xs.iter().zip(ys).find_map(|(x, y)| self.local_mismatch(x, y)),
            (Ty::Any | Ty::Func | Ty::Iter(_) | Ty::Trait(..), _) if !self.may_be_local(d) && self.may_be_local(v) => Some(match v {
                Ty::Param(n) => format!("`{n}` may not be `Send` and cannot be used as `{}`; add a `{n}: Send` bound", send_name(d)),
                Ty::Local(_) => format!("a `{}` may hold values that are not `Send` and cannot be used as `{}`", v.name(), send_name(d)),
                _ => format!("`{}` is not `Send` and cannot be used as `{}`", v.name(), send_name(d)),
            }),
            _ => None,
        }
    }

    /// The error message for a value of type `v` where `d` is expected:
    /// the locality reason if there is one, else `msg`.
    fn explain(&self, d: &Ty, v: &Ty, msg: String) -> String {
        self.local_mismatch(d, v).unwrap_or(msg)
    }

    /// The type arguments with which a value of type `t` provides trait `b`
    /// (`list<int>` provides `Iterable<int>`).
    fn trait_args_of(&self, t: &Ty, b: &str) -> Option<Vec<Ty>> {
        let iter_like = b == "Iterable" || b == "Iterator";
        match t {
            Ty::Instance(c, args) => self.m.impls_of(c, args, &|n| self.pbounds(n)).into_iter().find(|(n, _)| n == b).map(|(_, a)| a),
            Ty::Trait(n, args) if n == b || (n == "Iterator" && b == "Iterable") => Some(args.clone()),
            Ty::Param(n) => self.pbounds(n).into_iter().find_map(|x| match x {
                Ty::Trait(bn, args) if bn == b || (bn == "Iterator" && b == "Iterable") => Some(args),
                _ => None,
            }),
            Ty::Iter(x) if iter_like => Some(vec![(**x).clone()]),
            Ty::List(x) | Ty::Dict(x, _) | Ty::Chan(x) if b == "Iterable" => Some(vec![(**x).clone()]),
            Ty::Str if b == "Iterable" => Some(vec![Ty::Str]),
            Ty::Array(Some(e)) if b == "Iterable" => Some(vec![elem_ty(*e)]),
            _ => self.m.registry_traits(t, &|n| self.pbounds(n)).into_iter().find(|(n, _)| n == b).map(|(_, a)| a),
        }
    }

    /// The type of the items a `for` loop over a `t` produces.
    fn item_ty(&self, t: &Ty, span: Span) -> R<Ty> {
        match t {
            Ty::Param(n) => match self.trait_args_of(t, "Iterable") {
                Some(a) => Ok(a.into_iter().next().unwrap_or(Ty::Unknown)),
                None => Err(type_err(format!("iterating over `{n}` requires a bound `{n}: Iterable<...>`"), span)),
            },
            Ty::Instance(..) | Ty::Trait(..) => {
                Ok(self.trait_args_of(t, "Iterable").and_then(|a| a.into_iter().next()).map(|t| t.as_value()).unwrap_or(Ty::Unknown))
            }
            _ => Ok(t.item()),
        }
    }

    /// Requires that a value of type parameter type `t` implements `tr` for
    /// operation `what` (strict bounds in generic code).
    fn require(&self, t: &Ty, tr: &str, what: &str, span: Span) -> R<()> {
        if let Ty::Param(n) = t {
            if !self.satisfies(t, &Ty::Trait(tr.into(), Vec::new())) {
                return Err(type_err(format!("{what} on `{n}` requires a bound `{n}: {tr}`"), span));
            }
        }
        Ok(())
    }

    /// Binds type parameters `tps` in `out` by matching parameter type `p`
    /// against argument type `a`.
    fn unify(&self, p: &Ty, a: &Ty, tps: &[TParamSig], out: &mut [Option<Ty>]) {
        match (p, a) {
            (Ty::Local(x), _) => self.unify(x, a.peel(), tps, out),
            (_, Ty::Local(y)) if !matches!(p, Ty::Param(_)) => self.unify(p, y, tps, out),
            (Ty::Param(n), _) => {
                if let Some(k) = tps.iter().position(|t| t.name == *n) {
                    if out[k].is_none() && a.is_precise() && !matches!(a, Ty::NoneT | Ty::Any) {
                        out[k] = Some(a.clone());
                    }
                }
            }
            (Ty::List(x), Ty::List(y)) | (Ty::Iter(x), Ty::Iter(y)) | (Ty::Chan(x), Ty::Chan(y)) | (Ty::Option(x), Ty::Option(y)) => {
                self.unify(x, y, tps, out)
            }
            (Ty::Option(x), y) => self.unify(x, y, tps, out),
            (Ty::Dict(k1, v1), Ty::Dict(k2, v2)) => {
                self.unify(k1, k2, tps, out);
                self.unify(v1, v2, tps, out);
            }
            (Ty::Tuple(Some(xs)), Ty::Tuple(Some(ys))) if xs.len() == ys.len() => {
                for (x, y) in xs.iter().zip(ys) {
                    self.unify(x, y, tps, out);
                }
            }
            (Ty::Instance(c1, xs), Ty::Instance(c2, ys)) if c1 == c2 && xs.len() == ys.len() => {
                for (x, y) in xs.iter().zip(ys) {
                    self.unify(x, y, tps, out);
                }
            }
            (Ty::Trait(b, xs), _) if !xs.is_empty() => {
                if let Some(ys) = self.trait_args_of(a, b) {
                    for (x, y) in xs.iter().zip(&ys) {
                        self.unify(x, y, tps, out);
                    }
                }
            }
            _ => {}
        }
    }

    /// Compiles the arguments of a call of generic `sig` and infers its type
    /// parameters from `explicit` (a turbofish), the expected result type and
    /// the argument types. Returns the arguments, the type arguments (`None`
    /// where not inferred) and the result type.
    #[allow(clippy::too_many_arguments)]
    fn generic_call_args(
        &mut self,
        sig: &Sig,
        explicit: Option<Vec<Ty>>,
        expect: Option<&Ty>,
        args: &[Expr],
        recv_root: Option<Reg>,
        what: &str,
        span: Span,
    ) -> R<(ArgList, Vec<Option<Ty>>, Ty)> {
        let tps = &sig.tparams;
        let mut known: Vec<Option<Ty>> = match explicit {
            Some(v) => v.into_iter().map(Some).chain(std::iter::repeat(None)).take(tps.len()).collect(),
            None => vec![None; tps.len()],
        };
        if let Some(e) = expect {
            self.unify(&sig.ret, e, tps, &mut known);
        }
        let full = |known: &[Option<Ty>]| {
            let k = known.to_vec();
            move |n: &str| tps.iter().position(|t| t.name == n).map(|i| k[i].clone().unwrap_or(Ty::Unknown))
        };
        let pre = sig.subst(&full(&known));
        let keep_of = |i: usize| pre.params.get(i).map(|p| p.keep);
        let ty_of = |i: usize| pre.params.get(i).map(|p| (p.name.clone(), p.ty.clone()));
        let al = self.call_args(args, &keep_of, &ty_of, recv_root, span)?;
        let arg_tys = std::mem::take(&mut self.last_arg_tys);
        for (p, a) in sig.params.iter().zip(&arg_tys) {
            self.unify(&p.ty, a, tps, &mut known);
        }
        let post = sig.subst(&full(&known));
        for ((p, a), e) in post.params.iter().zip(&arg_tys).zip(args) {
            if !self.compatible(&p.ty, a) {
                return Err(type_err(self.explain(&p.ty, a, format!("argument `{}` expects `{}`, got `{}`", p.name, p.ty.name(), a.name())), e.span));
            }
        }
        for (k, tp) in post.tparams.iter().enumerate() {
            if let Some(t) = &known[k] {
                for b in &tp.bounds {
                    if !self.satisfies(t, b) {
                        return Err(type_err(
                            format!("`{}` does not implement `{}` (required by `{}` of `{what}`)", t.name(), b.name(), tp.name),
                            span,
                        ));
                    }
                }
            }
        }
        self.last_arg_tys = arg_tys;
        Ok((al, known, post.ret))
    }

    /// Loads the runtime types `tys` into fresh registers.
    fn make_types(&mut self, tys: &[Ty]) -> Option<Vec<Reg>> {
        let ds: Vec<TypeDesc> = tys.iter().map(|t| self.ty_desc(t)).collect::<Option<_>>()?;
        Some(
            ds.into_iter()
                .map(|d| {
                    let dst = self.tmp_borrow();
                    let ti = self.type_idx(d);
                    self.emit(Instr::MakeType { dst, ty: ti });
                    dst
                })
                .collect(),
        )
    }

    // ------------------------------------------------------------ state helpers

    fn st(&self, name: &str) -> Option<&Var> {
        self.state.as_ref().and_then(|s| s.get(name))
    }

    fn set_var(&mut self, name: &str, st: St, ty: Ty) {
        if let Some(s) = &mut self.state {
            s.insert(name.to_string(), Var { st, ty });
        }
    }

    fn set_st(&mut self, name: &str, st: St) {
        if let Some(s) = &mut self.state {
            if let Some(v) = s.get_mut(name) {
                v.st = st;
            } else {
                s.insert(name.to_string(), Var { st, ty: Ty::Unknown });
            }
        }
        if st == St::Moved || st == St::DynMoved {
            self.moved_log.push(name.to_string());
        }
    }

    fn is_global_name(&self, name: &str) -> bool {
        if self.is_module {
            return self.m.gindex.contains_key(name);
        }
        !self.locals.contains_key(name)
            && !self.captures.iter().any(|(n, _)| n == name)
            && self.m.gindex.contains_key(name)
    }

    /// Checks that a local (or module variable) may be used here.
    fn check_use(&mut self, name: &str, reg: Option<Reg>, span: Span) -> R<()> {
        let Some(v) = self.st(name).cloned() else {
            return Ok(());
        };
        match v.st {
            St::Live => Ok(()),
            St::Moved => Err(own_err(format!("`{name}` was moved and can no longer be used; use `{name}++` before moving it to keep a reference"), span)),
            St::MaybeMoved => Err(own_err(format!("`{name}` may have been moved on some path and can no longer be used"), span)),
            St::Uninit => Err(name_err(format!("`{name}` is used before it is assigned"), span)),
            St::DynMoved | St::MaybeUninit => {
                if let Some(r) = reg {
                    self.emit(Instr::CheckLive { reg: r });
                }
                Ok(())
            }
        }
    }

    // ------------------------------------------------------------ blocks and statements

    fn block(&mut self, b: &Block) -> R<()> {
        for s in &b.stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    /// Compiles a block whose last line is its value, moved into `dst`.
    fn block_value(&mut self, b: &Block, dst: Reg) -> R<Ty> {
        self.block_value_expect(b, dst, None)
    }

    /// `block_value` where the context expects the value to have type `expect`.
    fn block_value_expect(&mut self, b: &Block, dst: Reg, expect: Option<Ty>) -> R<Ty> {
        let n = b.stmts.len();
        if n == 0 {
            self.load_imm(dst, Value::UNIT);
            return Ok(Ty::Unit);
        }
        for s in &b.stmts[..n - 1] {
            self.stmt(s)?;
        }
        let last = &b.stmts[n - 1];
        match &last.kind {
            StmtKind::Expr(e) => {
                self.line = last.span.line;
                let mark = self.mark();
                self.expect = expect;
                let r = self.value_into(e, dst)?;
                self.end_stmt(mark);
                Ok(r)
            }
            _ => {
                self.stmt(last)?;
                if self.state.is_some() {
                    self.load_imm(dst, Value::UNIT);
                }
                Ok(Ty::Unit)
            }
        }
    }

    fn mark(&mut self) -> (usize, usize) {
        self.heap_borrows = 0;
        (self.owned_used, self.borrow_used)
    }

    fn end_stmt(&mut self, mark: (usize, usize)) {
        for r in std::mem::take(&mut self.stmt_drops) {
            self.emit(Instr::Drop { reg: r });
        }
        self.owned_used = mark.0;
        self.borrow_used = mark.1;
        self.heap_borrows = 0;
    }

    fn stmt(&mut self, s: &Stmt) -> R<()> {
        self.line = s.span.line;
        if self.state.is_none() {
            // Unreachable code after return/break/raise: still compile it so
            // errors are reported, with a fresh state.
            self.state = Some(State::new());
        }
        let mark = self.mark();
        let saved_drops = std::mem::take(&mut self.stmt_drops);
        let r = self.stmt_inner(s);
        self.end_stmt(mark);
        self.stmt_drops = saved_drops;
        r
    }

    fn stmt_inner(&mut self, s: &Stmt) -> R<()> {
        match &s.kind {
            StmtKind::Expr(e) => {
                match &e.kind {
                    ExprKind::If { .. } | ExprKind::Try(_) => {
                        let dst = self.tmp_owned();
                        self.value_into(e, dst)?;
                        if self.state.is_some() {
                            self.emit(Instr::Drop { reg: dst });
                        }
                    }
                    ExprKind::Ellipsis => {}
                    _ => {
                        let r = self.expr(e)?;
                        // A `()` result is an immediate: nothing to release.
                        if r.ty != Ty::Unit {
                            self.discard(&r);
                        }
                    }
                }
                Ok(())
            }
            StmtKind::Pass => Ok(()),
            StmtKind::Assign { target, value } => self.assign(target, value, s.span, false),
            StmtKind::AnnAssign { target, ann, value } => {
                let ExprKind::Name(n) = &target.kind else {
                    return Err(syn_err("only simple names can be annotated", target.span));
                };
                let (ty, d) = self.resolve_type(ann)?;
                if let Some(l) = self.locals.get_mut(n) {
                    l.decl = Some((ty.clone(), d.clone()));
                }
                match value {
                    Some(v) => self.assign(target, v, s.span, false),
                    None => Ok(()),
                }
            }
            StmtKind::AugAssign { target, op, value } => self.aug_assign(target, *op, value, s.span),
            StmtKind::Move { targets, value } => self.move_chain(targets, value, s.span),
            StmtKind::Return(v) => self.return_stmt(v.as_ref(), s.span),
            StmtKind::While { cond, body } => self.while_stmt(cond, body),
            StmtKind::For { target, iter, body } => self.for_stmt(target, iter, body, s.span),
            StmtKind::Break => self.break_continue(true, s.span),
            StmtKind::Continue => self.break_continue(false, s.span),
            StmtKind::Def(f) => self.def_stmt(f),
            // Items are defined at module start (see `hoist_items`).
            StmtKind::Struct(c) => self.item_here(&format!("struct `{}`", c.name), c.span),
            StmtKind::Trait(t) => self.item_here(&format!("trait `{}`", t.name), t.span),
            StmtKind::Impl(i) => self.item_here("`impl` blocks", i.span),
            StmtKind::Raise { exc, cause } => self.raise_stmt(exc.as_ref(), cause.as_ref(), s.span),
            // Resolved before compiling (see `imports`).
            StmtKind::Import { .. } | StmtKind::FromImport { .. } => Ok(()),
            StmtKind::Del(targets) => {
                for t in targets {
                    self.del_target(t)?;
                }
                Ok(())
            }
            StmtKind::With { ctx, name, body } => self.with_stmt(ctx, name.as_deref(), body, s.span),
            StmtKind::Assert { cond, msg } => {
                // `assert c, m` == `if not c: raise AssertionError(m)`
                let fails = self.cond_jump(cond, true)?;
                let raise = self.jump();
                let h = self.here();
                for at in fails {
                    self.patch(at, h);
                }
                let _ = raise;
                let skip = self.jump();
                let raise_at = self.here();
                self.patch(raise, raise_at);
                let before = self.state.clone();
                let args: Vec<Expr> = msg.iter().cloned().collect();
                let call = Expr {
                    kind: ExprKind::Call(Box::new(Expr { kind: ExprKind::Name("AssertionError".into()), span: s.span }), args),
                    span: s.span,
                };
                self.raise_stmt(Some(&call), None, s.span)?;
                self.state = before;
                let h = self.here();
                self.patch(skip, h);
                Ok(())
            }
        }
    }

    /// `with ctx as name: body` ==
    /// `t = ctx++; name = t.__enter__(); try: body finally: t.__exit__()`
    fn with_stmt(&mut self, ctx: &Expr, name: Option<&str>, body: &Block, span: Span) -> R<()> {
        let hidden = format!("<with {}>", self.instrs.len());
        let reg = self.new_reg(true, Some(hidden.clone()));
        self.locals.insert(hidden.clone(), Local { reg, kind: LocalKind::Owned, decl: None });
        let r = self.expr(ctx)?;
        let ty = r.ty.clone();
        let o = if r.own == Own::Owned {
            r
        } else {
            let dst = self.tmp_owned();
            self.emit(Instr::Dup { dst, src: r.reg });
            Res { reg: dst, own: Own::Owned, ty: ty.clone(), root: None, desc: "value".into() }
        };
        self.emit(Instr::Mov { dst: reg, src: o.reg });
        self.set_var(&hidden, St::Live, ty);
        let hname = || Box::new(Expr { kind: ExprKind::Name(hidden.clone()), span });
        let method = |m: &str| Expr {
            kind: ExprKind::Call(Box::new(Expr { kind: ExprKind::Attr(hname(), m.into()), span }), Vec::new()),
            span,
        };
        let enter = method("__enter__");
        match name {
            Some(n) => {
                let target = Expr { kind: ExprKind::Name(n.into()), span };
                self.assign(&target, &enter, span, false)?;
            }
            None => {
                let r = self.expr(&enter)?;
                self.discard(&r);
            }
        }
        let fin = Block { stmts: vec![Stmt { kind: StmtKind::Expr(method("__exit__")), span }] };
        let t = TryExpr { body: body.clone(), handlers: Vec::new(), else_: None, finally: Some(fin) };
        let dst = self.tmp_owned();
        self.try_expr(&t, Some(dst))?;
        if self.state.is_some() {
            self.emit(Instr::Drop { reg: dst });
            self.emit(Instr::Drop { reg });
        }
        self.set_st(&hidden, St::Uninit);
        Ok(())
    }

    // ------------------------------------------------------------ expressions

    fn discard(&mut self, r: &Res) {
        if r.own == Own::Owned {
            self.emit(Instr::Drop { reg: r.reg });
        }
    }

    /// Evaluates `e` and moves its (owned) value into `dst`.
    fn value_into(&mut self, e: &Expr, dst: Reg) -> R<Ty> {
        match &e.kind {
            ExprKind::If { branches, else_ } => {
                self.expect = None;
                self.if_expr(branches, else_.as_ref(), Some(dst))
            }
            ExprKind::Try(t) => {
                self.expect = None;
                self.try_expr(t, Some(dst))
            }
            _ => {
                let r = self.expr_owned(e, "value")?;
                self.emit(Instr::Mov { dst, src: r.reg });
                Ok(r.ty)
            }
        }
    }

    /// Evaluates `e` into an owned temporary. Named locals are moved; borrows
    /// of copy types are copied; borrows of move types are an error.
    fn expr_owned(&mut self, e: &Expr, what: &str) -> R<Res> {
        if let ExprKind::If { branches, else_ } = &e.kind {
            self.expect = None;
            let dst = self.tmp_owned();
            let ty = self.if_expr(branches, else_.as_ref(), Some(dst))?;
            return Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() });
        }
        let r = self.expr(e)?;
        self.to_owned(r, e.span, what)
    }

    fn to_owned(&mut self, r: Res, span: Span, what: &str) -> R<Res> {
        match &r.own {
            Own::Owned => Ok(r),
            Own::Local(name) => {
                let name = name.clone();
                self.move_local(&name, r, span)
            }
            Own::Borrowed => {
                let dst = self.tmp_owned();
                match r.ty.is_copy() {
                    Some(true) => {
                        self.emit(Instr::Dup { dst, src: r.reg });
                    }
                    Some(false) => {
                        return Err(own_err(
                            format!(
                                "cannot move {} (a borrowed `{}`) into {what}; use `{}++` to create a new reference{}",
                                r.desc,
                                r.ty.name(),
                                short(&r.desc),
                                if r.desc.starts_with("field") { ", or swap it out with `x <- obj.field <- new`" } else { "" }
                            ),
                            span,
                        ))
                    }
                    None => {
                        let w = self.name_idx(&r.desc);
                        self.emit(Instr::OwnDyn { dst, src: r.reg, what: w });
                    }
                }
                Ok(Res { reg: dst, own: Own::Owned, ty: r.ty, root: None, desc: "value".into() })
            }
        }
    }

    /// Moves a named local into a fresh owned temporary.
    fn move_local(&mut self, name: &str, r: Res, span: Span) -> R<Res> {
        let dst = self.tmp_owned();
        if self.is_module || self.is_global_name(name) {
            // A module variable: move it out of its global slot.
            let idx = self.m.gindex[name];
            match r.ty.is_copy() {
                Some(true) => {
                    self.emit(Instr::Dup { dst, src: r.reg });
                }
                Some(false) => {
                    self.emit(Instr::TakeGlobal { dst, idx });
                    self.set_st(name, St::Moved);
                }
                None => {
                    self.emit(Instr::TakeGlobal { dst, idx });
                    self.set_st(name, St::DynMoved);
                }
            }
            return Ok(Res { reg: dst, own: Own::Owned, ty: r.ty, root: None, desc: "value".into() });
        }
        let local = &self.locals[name];
        if local.kind == LocalKind::Borrowed {
            let r2 = Res { own: Own::Borrowed, desc: format!("parameter `{name}`"), ..r };
            return self.to_owned(r2, span, "this place");
        }
        let src = local.reg;
        match r.ty.is_copy() {
            Some(true) => {
                self.emit(Instr::Dup { dst, src });
            }
            Some(false) => {
                self.emit(Instr::Mov { dst, src });
                self.set_st(name, St::Moved);
            }
            None => {
                self.emit(Instr::MoveDyn { dst, src });
                self.set_st(name, St::DynMoved);
            }
        }
        Ok(Res { reg: dst, own: Own::Owned, ty: r.ty, root: None, desc: "value".into() })
    }

    fn name_expr(&mut self, n: &str, span: Span) -> R<Res> {
        if let Some(l) = self.locals.get(n) {
            let (reg, kind) = (l.reg, l.kind);
            let decl_ty = l.decl.as_ref().map(|d| d.0.as_value());
            self.check_use(n, Some(reg), span)?;
            let ty = self.st(n).map(|v| v.ty.clone()).or(decl_ty).unwrap_or(Ty::Unknown);
            let own = if kind == LocalKind::Borrowed { Own::Borrowed } else { Own::Local(n.to_string()) };
            let desc = if kind == LocalKind::Borrowed { format!("parameter `{n}`") } else { format!("`{n}`") };
            return Ok(Res { reg, own, ty, root: Some(reg), desc });
        }
        if let Some(i) = self.captures.iter().position(|(c, _)| c == n) {
            let ty = self.captures[i].1.clone();
            let dst = self.tmp_borrow();
            self.emit(Instr::LoadCapture { dst, idx: i as u32 });
            return Ok(Res { reg: dst, own: Own::Borrowed, ty, root: None, desc: format!("captured `{n}`") });
        }
        if let Some(&idx) = self.m.gindex.get(n) {
            let dst = self.tmp_borrow();
            self.emit(Instr::LoadGlobal { dst, idx });
            let ty = self.global_ty(n);
            if self.is_module && self.m.module_vars.contains(n) {
                self.check_use(n, None, span)?;
                let ty = self.st(n).map(|v| v.ty.clone()).unwrap_or(ty);
                return Ok(Res { reg: dst, own: Own::Local(n.to_string()), ty, root: None, desc: format!("`{n}`") });
            }
            // Builtins, functions, classes and traits are immortal: borrowing
            // them never needs an epoch pin.
            let immortal = !self.m.module_vars.contains(n)
                || self.m.funcs.contains_key(n)
                || self.m.classes.contains_key(n)
                || self.m.traits.contains_key(n);
            if !immortal {
                self.heap_borrows += 1;
            }
            return Ok(Res { reg: dst, own: Own::Borrowed, ty, root: None, desc: format!("global `{n}`") });
        }
        Err(name_err(format!("name `{n}` is not defined"), span))
    }

    fn global_ty(&self, n: &str) -> Ty {
        if self.m.classes.contains_key(n) || self.m.builtin_classes.contains(n) {
            return Ty::Class(n.to_string());
        }
        if self.m.funcs.contains_key(n) {
            return Ty::StaticFn;
        }
        Ty::Unknown
    }

    /// Compiles `e` where the context expects a value of type `expect`.
    fn expr_expect(&mut self, e: &Expr, expect: Option<Ty>) -> R<Res> {
        self.expect = expect;
        let r = self.expr(e);
        self.expect = None;
        r
    }

    fn expr(&mut self, e: &Expr) -> R<Res> {
        let owned = |reg, ty| Res { reg, own: Own::Owned, ty, root: None, desc: "value".into() };
        // The expected type applies to this expression only, not to its parts.
        let expect = self.expect.take();
        match &e.kind {
            ExprKind::Name(n) => self.name_expr(n, e.span),
            ExprKind::Int(v) => {
                let dst = self.tmp_owned();
                let i = i64::try_from(*v).map_err(|_| type_err("integer literal does not fit in a 64-bit int; use bigint(\"...\")", e.span))?;
                self.load_int(dst, i);
                Ok(owned(dst, Ty::Int))
            }
            ExprKind::Float(f) => {
                let dst = self.tmp_owned();
                let c = self.konst(Obj::alloc_immortal(Body::Float(*f)));
                self.emit(Instr::LoadConst { dst, idx: c });
                Ok(owned(dst, Ty::Float))
            }
            ExprKind::Str(s) => {
                let dst = self.tmp_owned();
                let c = self.str_const(s);
                self.emit(Instr::LoadConst { dst, idx: c });
                Ok(owned(dst, Ty::Str))
            }
            ExprKind::Bool(b) => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::bool(*b));
                Ok(owned(dst, Ty::Bool))
            }
            ExprKind::NoneLit => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::NONE);
                Ok(owned(dst, Ty::NoneT))
            }
            ExprKind::Unit => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::UNIT);
                Ok(owned(dst, Ty::Unit))
            }
            ExprKind::Ellipsis => Err(syn_err("`...` is only allowed as the body of a trait method", e.span)),
            ExprKind::Tuple(items) => {
                let ctx: Option<Vec<Ty>> = match &expect {
                    Some(Ty::Tuple(Some(ts))) if ts.len() == items.len() => Some(ts.clone()),
                    _ => None,
                };
                let mut args = Vec::new();
                let mut item_tys = Vec::new();
                for (i, it) in items.iter().enumerate() {
                    self.expect = ctx.as_ref().map(|ts| ts[i].clone());
                    let r = self.expr_owned(it, "a tuple")?;
                    item_tys.push(r.ty.clone());
                    args.push(Arg::new(r.reg, true, false, None));
                }
                let al = self.arglist(ArgList { args, pins: false, fast: false });
                let dst = self.tmp_owned();
                self.emit(Instr::BuildTuple { dst, args: al });
                Ok(owned(dst, Ty::Tuple(Some(item_tys))))
            }
            ExprKind::List(items) => {
                let ctx = match &expect {
                    Some(Ty::List(t)) if **t != Ty::Unknown => Some((**t).clone()),
                    _ => None,
                };
                let mut args = Vec::new();
                let mut tys = Vec::new();
                for it in items {
                    self.expect = ctx.clone();
                    let r = self.expr_owned(it, "a list")?;
                    tys.push((r.ty.clone(), it.span));
                    args.push(Arg::new(r.reg, true, false, None));
                }
                let elem = self.literal_elem(ctx, &tys, "list element", e.span)?;
                let d = TypeDesc::Builtin(Ctor::List, vec![self.elem_desc(&elem)]);
                let ti = self.type_idx(d);
                let al = self.arglist(ArgList { args, pins: false, fast: false });
                let dst = self.tmp_owned();
                self.emit(Instr::BuildList { dst, args: al, ty: ti });
                Ok(owned(dst, Ty::list(elem)))
            }
            ExprKind::Dict(items) => {
                let (kctx, vctx) = match &expect {
                    Some(Ty::Dict(k, v)) => (
                        (**k != Ty::Unknown).then(|| (**k).clone()),
                        (**v != Ty::Unknown).then(|| (**v).clone()),
                    ),
                    _ => (None, None),
                };
                let mut args = Vec::new();
                let (mut ktys, mut vtys) = (Vec::new(), Vec::new());
                for (k, v) in items {
                    self.expect = kctx.clone();
                    let kr = self.expr_owned(k, "a dict key")?;
                    ktys.push((kr.ty.clone(), k.span));
                    args.push(Arg::new(kr.reg, true, false, None));
                    self.expect = vctx.clone();
                    let vr = self.expr_owned(v, "a dict")?;
                    vtys.push((vr.ty.clone(), v.span));
                    args.push(Arg::new(vr.reg, true, false, None));
                }
                let kt = self.literal_elem(kctx, &ktys, "dict key", e.span)?;
                let vt = self.literal_elem(vctx, &vtys, "dict value", e.span)?;
                self.m.check_wf(&Ty::dict(kt.clone(), vt.clone()), &|n| self.pbounds(n), e.span)?;
                let d = TypeDesc::Builtin(Ctor::Dict, vec![self.elem_desc(&kt), self.elem_desc(&vt)]);
                let ti = self.type_idx(d);
                let al = self.arglist(ArgList { args, pins: false, fast: false });
                let dst = self.tmp_owned();
                self.emit(Instr::BuildDict { dst, args: al, ty: ti });
                Ok(owned(dst, Ty::dict(kt, vt)))
            }
            ExprKind::TypeApp(base, targs) => {
                let ExprKind::Name(n) = &base.kind else {
                    return Err(syn_err("type arguments (`::<...>`) follow a type name", e.span));
                };
                let generic = self.static_sig(base).filter(|(s, _)| !s.tparams.is_empty()).map(|s| s.0);
                let is_class = self.is_global_name(n) && self.m.classes.contains_key(n);
                if generic.is_some() || is_class {
                    let tys = match &generic {
                        Some(sig) => {
                            let tys = self.explicit_targs(targs, sig.tparams.len(), n, e.span)?;
                            for (tp, t) in sig.tparams.iter().zip(&tys) {
                                for b in &tp.bounds {
                                    let b = b.subst(&tparam_map(&sig.tparams, &tys));
                                    if !self.satisfies(t, &b) {
                                        return Err(type_err(
                                            format!("`{}` does not implement `{}` (required by `{}` of `{n}`)", t.name(), b.name(), tp.name),
                                            e.span,
                                        ));
                                    }
                                }
                            }
                            tys
                        }
                        None if targs.is_empty() => Vec::new(),
                        None => return Err(type_err(format!("`{n}` does not take type arguments"), e.span)),
                    };
                    if is_class {
                        // `Stack::<int>`: the class type.
                        let ty = Ty::Instance(n.clone(), tys);
                        let d = self.ty_desc(&ty).ok_or_else(|| type_err(format!("cannot instantiate `{}` here", ty.name()), e.span))?;
                        let dst = self.tmp_owned();
                        let ti = self.type_idx(d);
                        self.emit(Instr::MakeType { dst, ty: ti });
                        return Ok(owned(dst, Ty::TypeVal(Box::new(ty))));
                    }
                    // `first::<str>`: the function with its type arguments bound.
                    let f = self.expr(base)?;
                    let regs = self.make_types(&tys).ok_or_else(|| type_err("cannot instantiate this function here", e.span))?;
                    let al = self.arglist(ArgList { args: regs.into_iter().map(|r| Arg::new(r, false, false, None)).collect(), pins: false, fast: false });
                    let dst = self.tmp_owned();
                    self.emit(Instr::Instantiate { dst, func: f.reg, targs: al });
                    return Ok(owned(dst, Ty::Func));
                }
                if !matches!(n.as_str(), "list" | "dict" | "Channel") || !self.is_global_name(n) || self.m.module_vars.contains(n) {
                    return Err(type_err(format!("`{n}` does not take type arguments"), e.span));
                }
                let te = TypeExpr::Named(n.clone(), targs.clone(), e.span);
                let (ty, d) = self.resolve_type(&te)?;
                let dst = self.tmp_owned();
                let ti = self.type_idx(d);
                self.emit(Instr::MakeType { dst, ty: ti });
                Ok(owned(dst, Ty::TypeVal(Box::new(ty))))
            }
            ExprKind::Attr(obj, name) => self.attr_expr(obj, name, e.span),
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                if let Ty::Param(n) = &o.ty {
                    return Err(type_err(format!("cannot index a value of type parameter `{n}`"), e.span));
                }
                let i = self.expr(idx)?;
                if let Ty::Array(e) = o.ty {
                    // Array elements are scalars: read an owned copy.
                    let dst = self.tmp_owned();
                    self.emit(Instr::LoadElem { dst, obj: o.reg, idx: i.reg });
                    self.discard(&i);
                    self.discard(&o);
                    return Ok(owned(dst, e.map(elem_ty).unwrap_or(Ty::Unknown)));
                }
                let dst = self.tmp_borrow();
                self.emit(Instr::LoadIndex { dst, obj: o.reg, idx: i.reg });
                self.discard(&i);
                if o.own == Own::Owned {
                    self.stmt_drops.push(o.reg);
                }
                self.heap_borrows += 1;
                let ty = match (&o.ty, &idx.kind) {
                    (Ty::Tuple(Some(ts)), ExprKind::Int(k)) if (*k as usize) < ts.len() => ts[*k as usize].clone(),
                    _ => o.ty.elem().unwrap_or(Ty::Unknown),
                };
                Ok(Res { reg: dst, own: Own::Borrowed, ty, root: o.root, desc: "an element".into() })
            }
            ExprKind::Call(f, args) => self.call_expr(f, args, expect, e.span),
            ExprKind::StructLit { ty, fields } => self.struct_lit(ty, fields, expect, e.span),
            ExprKind::Path(base, n) => {
                if matches!(base.kind, ExprKind::TypeApp(..)) {
                    return Err(type_err(format!("type arguments on `{n}` need a call: `...::{n}(...)`"), e.span));
                }
                let g = self.path_global(base, n, e.span)?;
                self.expr(&Expr { kind: ExprKind::Name(g), span: e.span })
            }
            ExprKind::Dup(inner) => {
                let r = self.expr(inner)?;
                if r.own == Own::Owned {
                    return Ok(r);
                }
                if let Own::Local(n) = &r.own {
                    let n = n.clone();
                    self.check_use(&n, Some(r.reg), e.span)?;
                }
                let dst = self.tmp_owned();
                self.emit(Instr::Dup { dst, src: r.reg });
                Ok(owned(dst, r.ty))
            }
            ExprKind::Unary(op, x) => {
                let r = self.expr(x)?;
                if let Ty::Param(n) = &r.ty {
                    match op {
                        UnOp::Not => {}
                        UnOp::Neg | UnOp::Pos => self.require(&r.ty, "Num", &format!("unary `{}`", if *op == UnOp::Neg { "-" } else { "+" }), e.span)?,
                        UnOp::Invert => return Err(type_err(format!("`~` is not supported on type parameter `{n}`"), e.span)),
                    }
                }
                let dst = self.tmp_owned();
                let ty = match op {
                    UnOp::Not => {
                        self.emit(Instr::Not { dst, src: r.reg });
                        Ty::Bool
                    }
                    UnOp::Neg | UnOp::Pos | UnOp::Invert => {
                        let k = match op {
                            UnOp::Neg => UnKind::Neg,
                            UnOp::Pos => UnKind::Pos,
                            _ => UnKind::Invert,
                        };
                        self.emit(Instr::Unary { dst, op: k, src: r.reg });
                        match r.ty {
                            Ty::Int | Ty::Float | Ty::BigInt | Ty::Param(_) => r.ty.clone(),
                            _ => Ty::Unknown,
                        }
                    }
                };
                self.discard(&r);
                Ok(owned(dst, ty))
            }
            ExprKind::Binary(op, a, b) => self.binary(*op, a, b, e.span),
            ExprKind::Compare(..) | ExprKind::And(..) | ExprKind::Or(..) => {
                let dst = self.tmp_owned();
                let f = self.cond_jump(e, false)?;
                self.load_imm(dst, Value::TRUE);
                let j = self.jump();
                for at in f {
                    let h = self.here();
                    self.patch(at, h);
                }
                self.load_imm(dst, Value::FALSE);
                let h = self.here();
                self.patch(j, h);
                Ok(owned(dst, Ty::Bool))
            }
            ExprKind::Ternary { cond, then, else_ } => {
                let dst = self.tmp_owned();
                let f = self.cond_jump(cond, false)?;
                let before = self.state.clone();
                let t1 = self.value_into(then, dst)?;
                let after_then = self.state.take();
                let j = self.jump();
                let h = self.here();
                for at in f {
                    self.patch(at, h);
                }
                self.state = before;
                let t2 = self.value_into(else_, dst)?;
                let h = self.here();
                self.patch(j, h);
                self.state = join_opt(after_then, self.state.take());
                Ok(owned(dst, t1.join(&t2)))
            }
            ExprKind::If { branches, else_ } => {
                let dst = self.tmp_owned();
                let ty = self.if_expr(branches, else_.as_ref(), Some(dst))?;
                Ok(owned(dst, ty))
            }
            ExprKind::Try(t) => {
                let dst = self.tmp_owned();
                let ty = self.try_expr(t, Some(dst))?;
                Ok(owned(dst, ty))
            }
            ExprKind::Lambda(f) => {
                let dst = self.tmp_owned();
                let ty = self.make_closure(f, dst)?;
                Ok(owned(dst, ty))
            }
            ExprKind::Yield(v) => {
                if !self.is_generator {
                    return Err(syn_err("`yield` outside of a function", e.span));
                }
                if self.heap_borrows > 0 {
                    return Err(own_err("`yield` cannot appear in an expression that borrows a field, element or global", e.span));
                }
                let (src, ty) = match v {
                    Some(v) => {
                        self.expect = self.yield_decl.clone();
                        let r = self.expr_owned(v, "`yield`")?;
                        (r.reg, r.ty)
                    }
                    None => {
                        let r = self.tmp_owned();
                        self.load_imm(r, Value::UNIT);
                        (r, Ty::Unit)
                    }
                };
                self.check_held(e.span)?;
                if let Some(d) = &self.yield_decl {
                    if !self.compatible(d, &ty) {
                        return Err(type_err(self.explain(d, &ty, format!("`{}` yields `{}`, not `{}`", self.name, d.name(), ty.name())), e.span));
                    }
                }
                self.yield_tys.push(ty);
                let dst = self.tmp_owned();
                self.emit(Instr::Yield { dst, src });
                Ok(owned(dst, Ty::Unknown))
            }
            ExprKind::YieldFrom(sub) => {
                if !self.is_generator {
                    return Err(syn_err("`yield from` outside of a function", e.span));
                }
                self.check_held(e.span)?;
                let s = self.expr(sub)?;
                let item = self.item_ty(&s.ty, sub.span)?;
                if let Some(d) = &self.yield_decl {
                    if !self.compatible(d, &item) {
                        return Err(type_err(self.explain(d, &item, format!("`{}` yields `{}`, not `{}`", self.name, d.name(), item.name())), e.span));
                    }
                }
                self.yield_tys.push(item);
                let it = self.new_reg(true, Some("<yield from>".into()));
                self.emit(Instr::GetIter { dst: it, src: s.reg });
                self.discard(&s);
                let head = self.here();
                let item = self.tmp_owned();
                let fi = self.emit(Instr::ForIter { dst: item, iter: it, end: u32::MAX });
                let sent = self.tmp_owned();
                self.emit(Instr::Yield { dst: sent, src: item });
                self.emit(Instr::Drop { reg: sent });
                self.emit(Instr::Loop { target: head });
                let h = self.here();
                self.patch(fi, h);
                self.emit(Instr::Drop { reg: it });
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::UNIT);
                Ok(owned(dst, Ty::Unit))
            }
        }
    }

    /// The element type of a list or dict literal: the context's, else
    /// inferred from the items (`any` if some item's type is unknown).
    /// `what` names the part ("list element", "dict key", "dict value").
    fn literal_elem(&self, ctx: Option<Ty>, items: &[(Ty, Span)], what: &str, span: Span) -> R<Ty> {
        let (lit, example) = if what.starts_with("list") { ("[]", "xs: list<int> = []") } else { ("{}", "d: dict<str, int> = {}") };
        if let Some(t) = ctx {
            for (it, sp) in items {
                if !self.compatible(&t, it) {
                    return Err(type_err(self.explain(&t, it, format!("cannot use a `{}` as a {what} of type `{}`", it.name(), t.name())), *sp));
                }
            }
            return Ok(t);
        }
        if items.is_empty() {
            return Err(type_err(format!("cannot infer the type of `{lit}`; annotate it (e.g. `{example}`)"), span));
        }
        if items.iter().any(|(t, _)| *t == Ty::Unknown) {
            return Ok(Ty::Local(Box::new(Ty::Any)));
        }
        match join_all(items.iter().map(|(t, _)| t.clone()).collect()) {
            Ty::Unknown => {
                let mut names: Vec<String> = items.iter().map(|(t, _)| format!("`{}`", t.name())).collect();
                names.dedup();
                let hint = if lit == "[]" { "xs: list<any> = [...]" } else { "d: dict<str, any> = {...}" };
                Err(type_err(format!("mixed {what} types ({}); annotate it (e.g. `{hint}`)", names.join(", ")), span))
            }
            Ty::NoneT => Err(type_err(format!("cannot infer a {what} type from `None` alone; annotate it (e.g. `Option<int>`)"), span)),
            t => Ok(t),
        }
    }

    fn load_int(&mut self, dst: Reg, i: i64) {
        if (-(1i64 << 29)..(1i64 << 29)).contains(&i) {
            self.load_imm(dst, Value::int_unchecked(i));
        } else {
            let v = match Value::small_int(i) {
                Some(v) => v,
                None => Obj::alloc_immortal(Body::Int(i)),
            };
            let c = self.konst(v);
            self.emit(Instr::LoadConst { dst, idx: c });
        }
    }

    /// `x = <float arithmetic>` to a float local that already holds a box:
    /// the result is computed raw (`bind` then reuses the box).
    fn float_local_dest(&self, target: &Expr, value: &Expr) -> bool {
        match (&target.kind, &value.kind) {
            (ExprKind::Name(n), ExprKind::Binary(op, ..)) => {
                float_op(*op).is_some() && self.locals.contains_key(n) && self.st(n).is_some_and(|v| v.ty == Ty::Float && !matches!(v.st, St::Uninit | St::Moved))
            }
            _ => false,
        }
    }

    /// `a op b`.
    fn binary(&mut self, op: BinOp, a: &Expr, b: &Expr, span: Span) -> R<Res> {
        let owned = |reg, ty| Res { reg, own: Own::Owned, ty, root: None, desc: "value".into() };
        let raw_dest = std::mem::take(&mut self.raw_dest);
        if let Some(fop) = float_op(op) {
            // A float arithmetic tree: raw f64 temporaries, one box.
            let x = self.arith(a)?;
            let y = self.arith(b)?;
            if (raw_dest || x.is_raw() || y.is_raw()) && self.float_pair(&x, &y) {
                let r = self.farith(fop, x, y)?;
                let dst = self.tmp_owned();
                self.emit(Instr::BoxF { dst, src: r });
                return Ok(owned(dst, Ty::Float));
            }
            let x = self.arith_value(x)?;
            let y = self.arith_value(y)?;
            return self.binop(op, x, y, span);
        }
        let x = self.expr(a)?;
        let y = self.expr(b)?;
        self.binop(op, x, y, span)
    }


    /// `x op y` on values.
    fn binop(&mut self, op: BinOp, x: Res, y: Res, span: Span) -> R<Res> {
        let owned = |reg, ty| Res { reg, own: Own::Owned, ty, root: None, desc: "value".into() };
        let pty = self.param_binop(op, &x.ty, &y.ty, span)?;
        let dst = self.tmp_owned();
        let ins = match op {
            BinOp::Add => Instr::Add { dst, a: x.reg, b: y.reg },
            BinOp::Sub => Instr::Sub { dst, a: x.reg, b: y.reg },
            BinOp::Mul => Instr::Mul { dst, a: x.reg, b: y.reg },
            _ => {
                let k = match op {
                    BinOp::Div => BinKind::Div,
                    BinOp::FloorDiv => BinKind::FloorDiv,
                    BinOp::Mod => BinKind::Mod,
                    BinOp::Pow => BinKind::Pow,
                    BinOp::BitAnd => BinKind::BitAnd,
                    BinOp::BitOr => BinKind::BitOr,
                    BinOp::BitXor => BinKind::BitXor,
                    BinOp::Shl => BinKind::Shl,
                    _ => BinKind::Shr,
                };
                Instr::Binary { dst, op: k, a: x.reg, b: y.reg }
            }
        };
        self.emit(ins);
        let ty = pty.unwrap_or_else(|| binop_ty(op, &x.ty, &y.ty));
        self.discard(&y);
        self.discard(&x);
        Ok(owned(dst, ty))
    }

    /// Compiles an operand of float arithmetic: raw when it is itself a
    /// float arithmetic node or a `float` field, a literal is left for the
    /// parent to load either way.
    fn arith<'e>(&mut self, e: &'e Expr) -> R<Arith<'e>> {
        match &e.kind {
            ExprKind::Float(_) => Ok(Arith::Lit(e)),
            ExprKind::Binary(op, a, b) if float_op(*op).is_some() => {
                let x = self.arith(a)?;
                let y = self.arith(b)?;
                if self.float_pair(&x, &y) {
                    return Ok(Arith::Raw(self.farith(float_op(*op).unwrap(), x, y)?));
                }
                let x = self.arith_value(x)?;
                let y = self.arith_value(y)?;
                Ok(Arith::Val(self.binop(*op, x, y, e.span)?))
            }
            ExprKind::Attr(obj, name) => {
                let o = self.expr(obj)?;
                let raw_field = match o.ty.peel() {
                    Ty::Instance(c, _) => self.m.classes.get(c).is_some_and(|ci| !ci.native && ci.fields.iter().any(|(n, t, _)| n == name && *t == Ty::Float)),
                    _ => false,
                };
                if raw_field {
                    let dst = self.tmp_borrow();
                    let n = self.name_idx(name);
                    self.emit(Instr::LoadFieldF { dst, obj: o.reg, name: n });
                    if o.own == Own::Owned {
                        self.stmt_drops.push(o.reg);
                    }
                    return Ok(Arith::Raw(dst));
                }
                Ok(Arith::Val(self.attr_of(o, obj, name, e.span)?))
            }
            _ => Ok(Arith::Val(self.expr(e)?)),
        }
    }

    /// Both operands are numbers and at least one is a float: the result is a float.
    fn float_pair(&self, x: &Arith, y: &Arith) -> bool {
        let t = |a: &Arith| match a {
            Arith::Val(r) => r.ty.peel().clone(),
            _ => Ty::Float,
        };
        let (a, b) = (t(x), t(y));
        matches!((&a, &b), (Ty::Float | Ty::Int, Ty::Float | Ty::Int)) && (a == Ty::Float || b == Ty::Float)
    }

    /// The raw register of an operand.
    fn raw(&mut self, a: Arith) -> Reg {
        match a {
            Arith::Raw(r) => r,
            Arith::Lit(e) => {
                let ExprKind::Float(f) = e.kind else { unreachable!() };
                let dst = self.tmp_borrow();
                let bits = f.to_bits();
                self.emit(Instr::LoadRawF { dst, lo: bits as u32, hi: (bits >> 32) as u32 });
                dst
            }
            Arith::Val(r) => {
                let dst = self.tmp_borrow();
                self.emit(Instr::UnboxF { dst, src: r.reg });
                self.discard(&r);
                dst
            }
        }
    }

    /// An operand as a value (boxing a raw one).
    fn arith_value(&mut self, a: Arith) -> R<Res> {
        match a {
            Arith::Val(r) => Ok(r),
            Arith::Lit(e) => self.expr(e),
            Arith::Raw(r) => {
                let dst = self.tmp_owned();
                self.emit(Instr::BoxF { dst, src: r });
                Ok(Res { reg: dst, own: Own::Owned, ty: Ty::Float, root: None, desc: "value".into() })
            }
        }
    }

    fn farith(&mut self, op: FOp, x: Arith, y: Arith) -> R<Reg> {
        let a = self.raw(x);
        let b = self.raw(y);
        let dst = self.tmp_borrow();
        self.emit(Instr::FArith { dst, op, a, b });
        Ok(dst)
    }

    fn attr_expr(&mut self, obj: &Expr, name: &str, span: Span) -> R<Res> {
        let o = self.expr(obj)?;
        self.attr_of(o, obj, name, span)
    }

    /// `o.name`, `o` being the compiled `obj`.
    fn attr_of(&mut self, o: Res, obj: &Expr, name: &str, span: Span) -> R<Res> {
        self.option_check(&o, &format!("accessing `.{name}`"), span)?;
        let mut ty = Ty::Unknown;
        if let Ty::Instance(c, args) = &o.ty {
            if let Some(ci) = self.m.class_info(c, args) {
                if let Some((_, fty, _)) = ci.fields.iter().find(|(n, _, _)| n == name) {
                    ty = fty.subst(&class_map(ci, args)).as_value();
                } else if ci.lookup(name).is_some() || !self.m.registry_methods(&o.ty, name, &|n| self.pbounds(n)).is_empty() {
                    ty = self.func_ty(self.may_be_local(&o.ty));
                } else {
                    return Err(Error::new("AttributeError", format!("`{c}` has no field or method `{name}`"), span));
                }
            }
        }
        if let Ty::Param(n) = &o.ty {
            if self.param_method(n, name).is_empty() {
                return Err(type_err(param_field_msg(n, name), span));
            }
            ty = self.func_ty(self.may_be_local(&o.ty));
        }
        if ty == Ty::Unknown && name == "msg" {
            if let Ty::Instance(c, _) = &o.ty {
                if self.m.builtin_classes.contains(c) {
                    ty = Ty::Str;
                }
            }
        }
        if let ExprKind::Name(base) = &obj.kind {
            if let Some(v) = self.st(&format!("@{base}.{name}")) {
                ty = v.ty.clone();
            }
        }
        let dst = self.tmp_borrow();
        let n = self.name_idx(name);
        self.emit(Instr::LoadField { dst, obj: o.reg, name: n });
        if o.own == Own::Owned {
            self.stmt_drops.push(o.reg);
        }
        self.heap_borrows += 1;
        Ok(Res { reg: dst, own: Own::Borrowed, ty, root: o.root, desc: format!("field `{name}`") })
    }

    // ------------------------------------------------------------ conditions

    /// Emits a conditional jump; returns jump sites taken when `e` is
    /// `!when_true`... i.e. with `when_true = false`, the returned jumps are
    /// taken when the condition is false.
    fn cond_jump(&mut self, e: &Expr, jump_if: bool) -> R<Vec<usize>> {
        match &e.kind {
            ExprKind::Unary(UnOp::Not, x) => self.cond_jump(x, !jump_if),
            ExprKind::And(a, b) if !jump_if => {
                let mut j = self.cond_jump(a, false)?;
                j.extend(self.cond_jump(b, false)?);
                Ok(j)
            }
            ExprKind::Or(a, b) if jump_if => {
                let mut j = self.cond_jump(a, true)?;
                j.extend(self.cond_jump(b, true)?);
                Ok(j)
            }
            ExprKind::And(a, b) => {
                // jump if (a and b): skip when a is false.
                let skip = self.cond_jump(a, false)?;
                let j = self.cond_jump(b, true)?;
                let h = self.here();
                for at in skip {
                    self.patch(at, h);
                }
                Ok(j)
            }
            ExprKind::Or(a, b) => {
                // jump if !(a or b): skip when a is true.
                let skip = self.cond_jump(a, true)?;
                let j = self.cond_jump(b, false)?;
                let h = self.here();
                for at in skip {
                    self.patch(at, h);
                }
                Ok(j)
            }
            ExprKind::Compare(_, rest) if rest.len() > 1 => {
                let t = self.compare_chain(e)?;
                let j = if jump_if {
                    self.emit(Instr::JumpIfTrue { cond: t, target: u32::MAX })
                } else {
                    self.emit(Instr::JumpIfFalse { cond: t, target: u32::MAX })
                };
                Ok(vec![j])
            }
            ExprKind::Compare(first, rest) => {
                let (op, rhs) = &rest[0];
                let narrowing = self.narrowing_of(first, *op, rhs);
                let l = self.expr(first)?;
                let r = self.expr(rhs)?;
                self.check_cmp(*op, &l.ty, &r.ty, e.span)?;
                let t = self.tmp_owned();
                self.emit_cmp(*op, t, l.reg, r.reg);
                self.discard(&r);
                self.discard(&l);
                let j = if jump_if {
                    self.emit(Instr::JumpIfTrue { cond: t, target: u32::MAX })
                } else {
                    self.emit(Instr::JumpIfFalse { cond: t, target: u32::MAX })
                };
                // `x is not None` narrows x on the fall-through path.
                if let Some((name, is_not_none)) = narrowing {
                    if is_not_none != jump_if {
                        self.narrow(&name);
                    }
                }
                Ok(vec![j])
            }
            _ => {
                let r = self.expr(e)?;
                let cond = if r.own == Own::Owned && r.ty != Ty::Bool {
                    // Reduce to a bool first so the value is dropped on both paths.
                    let t = self.tmp_owned();
                    let t2 = self.tmp_owned();
                    self.emit(Instr::Not { dst: t, src: r.reg });
                    self.emit(Instr::Drop { reg: r.reg });
                    self.emit(Instr::Not { dst: t2, src: t });
                    t2
                } else {
                    r.reg
                };
                let j = if jump_if {
                    self.emit(Instr::JumpIfTrue { cond, target: u32::MAX })
                } else {
                    self.emit(Instr::JumpIfFalse { cond, target: u32::MAX })
                };
                Ok(vec![j])
            }
        }
    }

    /// `a < b < c` as a bool value (operands evaluated lazily, each dropped).
    fn compare_chain(&mut self, e: &Expr) -> R<Reg> {
        let ExprKind::Compare(first, rest) = &e.kind else { unreachable!() };
        let dst = self.tmp_owned();
        let mut left = self.expr(first)?;
        let mut stubs: Vec<(usize, Option<Reg>)> = Vec::new();
        for (i, (op, rhs)) in rest.iter().enumerate() {
            let right = self.expr(rhs)?;
            self.check_cmp(*op, &left.ty, &right.ty, rhs.span)?;
            self.emit_cmp(*op, dst, left.reg, right.reg);
            self.discard(&left);
            if i + 1 < rest.len() {
                let j = self.emit(Instr::JumpIfFalse { cond: dst, target: u32::MAX });
                stubs.push((j, (right.own == Own::Owned).then_some(right.reg)));
            }
            left = right;
        }
        self.discard(&left);
        let done = self.jump();
        let mut to_end = vec![done];
        for (j, owned) in stubs {
            let h = self.here();
            self.patch(j, h);
            if let Some(r) = owned {
                self.emit(Instr::Drop { reg: r });
            }
            to_end.push(self.jump());
        }
        let h = self.here();
        for j in to_end {
            self.patch(j, h);
        }
        Ok(dst)
    }

    /// Comparisons of type-parameter values need an `Ord` bound (`==`,
    /// `!=` and `is` work on every value).
    fn check_cmp(&self, op: CmpOp, a: &Ty, b: &Ty, span: Span) -> R<()> {
        let sym = match op {
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::In | CmpOp::NotIn => {
                if let Ty::Param(n) = b {
                    return Err(type_err(format!("`in` is not supported on type parameter `{n}`"), span));
                }
                return Ok(());
            }
            _ => return Ok(()),
        };
        let p = match (a, b) {
            (Ty::Param(n), _) | (_, Ty::Param(n)) => n,
            _ => return Ok(()),
        };
        if a != b && !matches!(a, Ty::Unknown) && !matches!(b, Ty::Unknown) {
            return Err(type_err(format!("`{sym}` between `{}` and `{}`: both operands must be `{p}`", a.name(), b.name()), span));
        }
        let t = if matches!(a, Ty::Param(_)) { a } else { b };
        self.require(t, "Ord", &format!("`{sym}`"), span)
    }

    /// Arithmetic on type-parameter values: `T op T -> T` with `T: Num`.
    /// `None` when neither operand is a type parameter.
    fn param_binop(&self, op: BinOp, a: &Ty, b: &Ty, span: Span) -> R<Option<Ty>> {
        let p = match (a, b) {
            (Ty::Param(n), _) | (_, Ty::Param(n)) => n.clone(),
            _ => return Ok(None),
        };
        let sym = op.symbol();
        if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr) {
            return Err(type_err(format!("`{sym}` is not supported on type parameter `{p}`"), span));
        }
        if a != b {
            return Err(type_err(format!("`{sym}` between `{}` and `{}`: both operands must be `{p}`", a.name(), b.name()), span));
        }
        self.require(a, "Num", &format!("`{sym}`"), span)?;
        // `/` of ints is a float.
        Ok(Some(if op == BinOp::Div { Ty::Unknown } else { a.clone() }))
    }

    fn emit_cmp(&mut self, op: CmpOp, dst: Reg, a: Reg, b: Reg) {
        let i = match op {
            CmpOp::Eq => Instr::Eq { dst, a, b },
            CmpOp::Ne => Instr::Ne { dst, a, b },
            CmpOp::Lt => Instr::Lt { dst, a, b },
            CmpOp::Le => Instr::Le { dst, a, b },
            CmpOp::Gt => Instr::Gt { dst, a, b },
            CmpOp::Ge => Instr::Ge { dst, a, b },
            CmpOp::Is => Instr::Is { dst, a, b },
            CmpOp::IsNot => Instr::IsNot { dst, a, b },
            CmpOp::In => Instr::In { dst, a, b },
            CmpOp::NotIn => Instr::NotIn { dst, a, b },
        };
        self.emit(i);
    }

    /// For `x is None` / `x is not None` on a local: (name, is_not_none).
    fn narrowing_of(&self, l: &Expr, op: CmpOp, r: &Expr) -> Option<(String, bool)> {
        if !matches!(r.kind, ExprKind::NoneLit) {
            return None;
        }
        let n = match &l.kind {
            ExprKind::Name(n) => n.clone(),
            ExprKind::Attr(o, f) => match &o.kind {
                ExprKind::Name(base) => format!("@{base}.{f}"),
                _ => return None,
            },
            _ => return None,
        };
        match op {
            CmpOp::IsNot => Some((n.clone(), true)),
            CmpOp::Is => Some((n.clone(), false)),
            _ => None,
        }
    }

    /// Records that a local is known not to be `None` on the current path.
    fn narrow(&mut self, name: &str) {
        if let Some(path) = name.strip_prefix('@') {
            // A field path: remember the narrowed type of `base.field`.
            let (base, field) = path.split_once('.').unwrap();
            let base_ty = self.local_ty(base);
            if let Ok(Ty::Option(inner)) = self.field_ty(&base_ty, field, Span::default()) {
                self.set_var(name, St::Live, *inner);
            }
            return;
        }
        if let Some(s) = &mut self.state {
            if let Some(v) = s.get_mut(name) {
                if let Ty::Option(inner) = &v.ty {
                    v.ty = (**inner).clone();
                }
            }
        }
    }

    fn local_ty(&self, n: &str) -> Ty {
        if let Some(v) = self.st(n) {
            return v.ty.clone();
        }
        self.locals.get(n).and_then(|l| l.decl.as_ref().map(|d| d.0.clone())).unwrap_or(Ty::Unknown)
    }

    /// Forgets narrowing of `base.field` paths after stores.
    fn unnarrow(&mut self, base: &str, field: Option<&str>) {
        if let Some(s) = &mut self.state {
            let prefix = match field {
                Some(f) => format!("@{base}.{f}"),
                None => format!("@{base}."),
            };
            s.retain(|k, _| !k.starts_with(&prefix));
        }
    }

    fn option_check(&self, r: &Res, what: &str, span: Span) -> R<()> {
        if let Ty::Option(inner) = &r.ty {
            return Err(type_err(
                format!(
                    "{} is `Option<{}>` and may be `None`; narrow it first (`if x is not None:`) or use `.some_or(default)` before {what}",
                    r.desc,
                    inner.name()
                ),
                span,
            ));
        }
        Ok(())
    }

    // ------------------------------------------------------------ if / while / for

    fn if_expr(&mut self, branches: &[(Expr, Block)], else_: Option<&Block>, dst: Option<Reg>) -> R<Ty> {
        let mut end_jumps = Vec::new();
        let mut out_state: Option<State> = None;
        let mut ty: Option<Ty> = None;
        for (cond, body) in branches {
            let before = self.state.clone();
            let f = self.cond_jump(cond, false)?;
            let false_state = {
                // The false path does not see narrowing applied for the true path.
                let mut s = self.state.clone();
                if let (Some(s), Some(b)) = (&mut s, &before) {
                    for (k, v) in b {
                        if let Some(cur) = s.get_mut(k) {
                            if cur.st == v.st {
                                cur.ty = v.ty.clone();
                            }
                        }
                    }
                }
                s
            };
            let t = match dst {
                Some(d) => self.block_value(body, d)?,
                None => {
                    self.block(body)?;
                    Ty::Unit
                }
            };
            if self.state.is_some() {
                ty = Some(ty.map_or(t.clone(), |x| x.join(&t)));
            }
            out_state = join_opt(out_state, self.state.take());
            end_jumps.push(self.jump());
            let h = self.here();
            for at in f {
                self.patch(at, h);
            }
            self.state = false_state;
            // `if x is None: ... else:` narrows x in the else branch.
            if let ExprKind::Compare(l, rest) = &cond.kind {
                if rest.len() == 1 {
                    if let Some((n, false)) = self.narrowing_of(l, rest[0].0, &rest[0].1) {
                        self.narrow(&n);
                    }
                }
            }
        }
        match else_ {
            Some(b) => {
                let t = match dst {
                    Some(d) => self.block_value(b, d)?,
                    None => {
                        self.block(b)?;
                        Ty::Unit
                    }
                };
                if self.state.is_some() {
                    ty = Some(ty.map_or(t.clone(), |x| x.join(&t)));
                }
            }
            None => {
                if let Some(d) = dst {
                    self.load_imm(d, Value::UNIT);
                }
                ty = Some(ty.map_or(Ty::Unit, |x| x.join(&Ty::Unit)));
            }
        }
        out_state = join_opt(out_state, self.state.take());
        let h = self.here();
        for at in end_jumps {
            self.patch(at, h);
        }
        self.state = out_state;
        Ok(ty.unwrap_or(Ty::Unknown))
    }

    /// A local the loop body assigns may hold the previous iteration's value
    /// at the head, so its reassignment must drop (a no-op on `UNINIT`/`MOVED`).
    fn widen_loop_head(&mut self, body: &Block, target: Option<&Expr>) {
        let mut names = Vec::new();
        scan::assigned_names(body, &mut names);
        if let Some(t) = target {
            let mut wrapped = Block::default();
            wrapped.stmts.push(Stmt {
                kind: StmtKind::Assign { target: t.clone(), value: t.clone() },
                span: t.span,
            });
            scan::assigned_names(&wrapped, &mut names);
        }
        for n in names {
            if !self.locals.contains_key(&n) {
                continue;
            }
            match self.st(&n).map(|v| v.st) {
                None | Some(St::Uninit) => {
                    let ty = self.st(&n).map(|v| v.ty.clone()).unwrap_or(Ty::Unknown);
                    self.set_var(&n, St::MaybeUninit, ty);
                }
                Some(St::Moved) => self.set_st(&n, St::MaybeMoved),
                _ => {}
            }
        }
    }

    fn while_stmt(&mut self, cond: &Expr, body: &Block) -> R<()> {
        self.widen_loop_head(body, None);
        let entry = self.state.clone();
        let head = self.here();
        let mark = self.mark();
        let exits = self.cond_jump(cond, false)?;
        self.end_stmt(mark);
        let after_cond = self.state.clone();
        self.loops.push(LoopCtx {
            head,
            breaks: Vec::new(),
            continues: Vec::new(),
            break_state: None,
            cont_state: None,
            finally_depth: self.finally_stack.len(),
        });
        self.block(body)?;
        let lp = self.loops.pop().unwrap();
        let end_body = join_opt(self.state.take(), lp.cont_state.clone());
        self.check_loop_moves(&entry, &end_body, body)?;
        let back = self.here();
        self.emit(Instr::Loop { target: head });
        for at in lp.continues {
            self.patch(at, back);
        }
        let end = self.here();
        for at in exits.into_iter().chain(lp.breaks) {
            self.patch(at, end);
        }
        let mut st = join_opt(after_cond, end_body);
        st = join_opt(st, lp.break_state);
        self.state = st;
        Ok(())
    }

    /// A value moved inside a loop body cannot be used in the next iteration.
    fn check_loop_moves(&mut self, entry: &Option<State>, end: &Option<State>, body: &Block) -> R<()> {
        let (Some(entry), Some(end)) = (entry, end) else { return Ok(()) };
        let mut used = HashSet::new();
        scan::used_names(body, &mut used);
        for (name, v) in end {
            let was_live = entry.get(name).is_some_and(|e| e.st == St::Live);
            if was_live && matches!(v.st, St::Moved | St::MaybeMoved) && used.contains(name) {
                return Err(own_err(
                    format!("`{name}` is moved inside the loop and would be used again in the next iteration; use `{name}++` or reassign it before the loop repeats"),
                    body.stmts.first().map(|s| s.span).unwrap_or_default(),
                ));
            }
        }
        Ok(())
    }

    fn for_stmt(&mut self, target: &Expr, iter: &Expr, body: &Block, span: Span) -> R<()> {
        // `for i in range(...)` with literal-sign steps compiles to a counter loop.
        if let ExprKind::Call(f, args) = &iter.kind {
            if let ExprKind::Name(fname) = &f.kind {
                if fname == "range" && !self.locals.contains_key("range") && !self.m.module_vars.contains("range") && (1..=3).contains(&args.len()) {
                    let step = match args.get(2) {
                        None => Some(1),
                        Some(Expr { kind: ExprKind::Int(v), .. }) => Some(*v as i64),
                        Some(Expr { kind: ExprKind::Unary(UnOp::Neg, inner), .. }) => match inner.kind {
                            ExprKind::Int(v) => Some(-(v as i64)),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(step) = step {
                        if step != 0 {
                            return self.for_range(target, args, step, body, span);
                        }
                    }
                }
            }
        }
        let it_reg = self.new_reg(true, Some("<iter>".into()));
        let item_ty;
        {
            let mark = self.mark();
            let src = self.expr(iter)?;
            item_ty = self.item_ty(&src.ty, iter.span)?;
            self.emit(Instr::GetIter { dst: it_reg, src: src.reg });
            self.discard(&src);
            self.end_stmt(mark);
        }
        self.widen_loop_head(body, Some(target));
        let entry = self.state.clone();
        let head = self.here();
        let item = self.tmp_owned();
        let fi = self.emit(Instr::ForIter { dst: item, iter: it_reg, end: u32::MAX });
        self.assign_owned(target, Res { reg: item, own: Own::Owned, ty: item_ty, root: None, desc: "value".into() }, span)?;
        self.owned_used -= 1;
        let after_head = self.state.clone();
        self.loops.push(LoopCtx {
            head,
            breaks: Vec::new(),
            continues: Vec::new(),
            break_state: None,
            cont_state: None,
            finally_depth: self.finally_stack.len(),
        });
        self.block(body)?;
        let lp = self.loops.pop().unwrap();
        let end_body = join_opt(self.state.take(), lp.cont_state.clone());
        self.check_loop_moves(&entry, &end_body, body)?;
        let back = self.here();
        self.emit(Instr::Loop { target: head });
        for at in lp.continues {
            self.patch(at, back);
        }
        let end = self.here();
        self.patch(fi, end);
        for at in lp.breaks {
            self.patch(at, end);
        }
        self.emit(Instr::Drop { reg: it_reg });
        let st = join_opt(join_opt(entry, after_head), join_opt(end_body, lp.break_state));
        self.state = st;
        Ok(())
    }

    fn for_range(&mut self, target: &Expr, args: &[Expr], step: i64, body: &Block, span: Span) -> R<()> {
        let ExprKind::Name(var) = &target.kind else {
            return Err(syn_err("`for ... in range(...)` needs a simple loop variable", target.span));
        };
        let counter = self.new_reg(true, Some("<range i>".into()));
        let end = self.new_reg(true, Some("<range end>".into()));
        let step_reg = self.new_reg(true, Some("<range step>".into()));
        {
            let mark = self.mark();
            let (start, stop) = if args.len() == 1 { (None, &args[0]) } else { (Some(&args[0]), &args[1]) };
            match start {
                Some(s) => {
                    let r = self.expr_owned(s, "range")?;
                    self.emit(Instr::Mov { dst: counter, src: r.reg });
                }
                None => self.load_int(counter, 0),
            }
            let r = self.expr_owned(stop, "range")?;
            self.emit(Instr::Mov { dst: end, src: r.reg });
            self.load_int(step_reg, step);
            // Non-int bounds are rejected by the type check.
            let ti = self.type_idx(TypeDesc::prim(Ctor::Int));
            let w = self.name_idx("range() argument");
            self.emit(Instr::CheckType { reg: counter, ty: ti, what: w });
            self.emit(Instr::CheckType { reg: end, ty: ti, what: w });
            self.end_stmt(mark);
        }
        self.widen_loop_head(body, None);
        let entry = self.state.clone();
        let head = self.here();
        let t = self.tmp_owned();
        if step > 0 {
            self.emit(Instr::Lt { dst: t, a: counter, b: end });
        } else {
            self.emit(Instr::Gt { dst: t, a: counter, b: end });
        }
        let exit = self.emit(Instr::JumpIfFalse { cond: t, target: u32::MAX });
        let item = self.tmp_owned();
        self.emit(Instr::Dup { dst: item, src: counter });
        self.assign_owned(target, Res { reg: item, own: Own::Owned, ty: Ty::Int, root: None, desc: "value".into() }, span)?;
        self.owned_used -= 2;
        let _ = var;
        let after_head = self.state.clone();
        self.loops.push(LoopCtx {
            head: u32::MAX,
            breaks: Vec::new(),
            continues: Vec::new(),
            break_state: None,
            cont_state: None,
            finally_depth: self.finally_stack.len(),
        });
        self.block(body)?;
        let lp = self.loops.pop().unwrap();
        let end_body = join_opt(self.state.take(), lp.cont_state.clone());
        self.check_loop_moves(&entry, &end_body, body)?;
        let incr = self.here();
        for at in lp.continues {
            self.patch(at, incr);
        }
        let t2 = self.tmp_owned();
        self.emit(Instr::Add { dst: t2, a: counter, b: step_reg });
        self.emit(Instr::Mov { dst: counter, src: t2 });
        self.owned_used -= 1;
        self.emit(Instr::Loop { target: head });
        let out = self.here();
        self.patch(exit, out);
        for at in lp.breaks {
            self.patch(at, out);
        }
        let st = join_opt(join_opt(entry, after_head), join_opt(end_body, lp.break_state));
        self.state = st;
        Ok(())
    }

    fn break_continue(&mut self, is_break: bool, span: Span) -> R<()> {
        let Some(lp) = self.loops.last() else {
            return Err(syn_err(if is_break { "`break` outside loop" } else { "`continue` outside loop" }, span));
        };
        let depth = lp.finally_depth;
        let finallys: Vec<Block> = self.finally_stack[depth..].iter().rev().cloned().collect();
        for f in &finallys {
            self.block(f)?;
        }
        let j = self.jump();
        let st = self.state.take();
        let lp = self.loops.last_mut().unwrap();
        if is_break {
            lp.breaks.push(j);
            lp.break_state = join_opt(lp.break_state.take(), st);
        } else {
            if lp.head != u32::MAX {
                // `while`/`for` loops patch continues to their head.
            }
            lp.continues.push(j);
            lp.cont_state = join_opt(lp.cont_state.take(), st);
        }
        Ok(())
    }

    // ------------------------------------------------------------ assignment

    fn assign(&mut self, target: &Expr, value: &Expr, span: Span, _is_move: bool) -> R<()> {
        self.raw_dest = self.float_local_dest(target, value);
        match &target.kind {
            ExprKind::Name(_) | ExprKind::Tuple(_) => {
                let expect = self.target_ty(target);
                let v = match &value.kind {
                    ExprKind::If { .. } | ExprKind::Try(_) => {
                        let dst = self.tmp_owned();
                        let ty = self.value_into(value, dst)?;
                        Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() }
                    }
                    _ => {
                        let r = self.expr_expect(value, expect)?;
                        self.to_owned(r, value.span, "a variable")?
                    }
                };
                self.assign_owned(target, v, span)
            }
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                let fty = self.field_ty(&o.ty, field, target.span)?;
                if fty.is_move() {
                    return Err(own_err(
                        format!("field `{field}` holds a move type (`{}`); use `<-` to move a value in", fty.name()),
                        span,
                    ));
                }
                let v = self.expr(value)?;
                if v.ty.is_move() {
                    return Err(own_err(
                        format!("`=` can only store copy types into a field; use `<-` to move this `{}` in", v.ty.name()),
                        span,
                    ));
                }
                self.check_field_store(&fty, field, &v.ty, value.span)?;
                let v = self.to_owned(v, value.span, "a field")?;
                let n = self.name_idx(field);
                self.emit(Instr::StoreField { obj: o.reg, name: n, src: v.reg, mode: StoreMode::Copy });
                if let ExprKind::Name(base) = &obj.kind {
                    self.unnarrow(base, Some(field));
                }
                self.discard(&o);
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                let v = self.expr(value)?;
                if v.ty.is_move() {
                    return Err(own_err(
                        format!("`=` can only store copy types into an element; use `<-` to move this `{}` in", v.ty.name()),
                        span,
                    ));
                }
                if let Ty::Array(Some(e)) = o.ty {
                    let ok = match (elem_ty(e), &v.ty) {
                        (_, Ty::Unknown) => true,
                        (Ty::Float, Ty::Int | Ty::Float) => true,
                        (t, vt) => t == *vt,
                    };
                    if !ok {
                        return Err(type_err(format!("cannot store a `{}` in an `array<{}>`", v.ty.name(), e.name()), value.span));
                    }
                }
                let v = self.to_owned(v, value.span, "an element")?;
                let proven = self.check_elem_store(&o.ty, &i, &v, value.span)?;
                self.emit(Instr::StoreIndex { obj: o.reg, idx: i.reg, src: v.reg, mode: StoreMode::Copy, proven });
                self.discard(&i);
                self.discard(&o);
                Ok(())
            }
            _ => Err(syn_err("invalid assignment target", target.span)),
        }
    }

    /// The declared type of an assignment target, if known statically: a
    /// declared variable, a field of a typed local, or an element of one.
    fn target_ty(&self, t: &Expr) -> Option<Ty> {
        match &t.kind {
            ExprKind::Name(n) => self.locals.get(n).and_then(|l| l.decl.as_ref().map(|d| d.0.clone())),
            ExprKind::Attr(o, f) => match &o.kind {
                ExprKind::Name(b) => self.field_ty(&self.local_ty(b), f, Span::default()).ok().filter(|t| *t != Ty::Unknown),
                _ => None,
            },
            ExprKind::Index(o, _) => match &o.kind {
                ExprKind::Name(b) => match self.local_ty(b) {
                    Ty::List(t) | Ty::Dict(_, t) if *t != Ty::Unknown => Some(*t),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    fn field_ty(&self, obj_ty: &Ty, field: &str, span: Span) -> R<Ty> {
        match obj_ty {
            Ty::Instance(c, args) => {
                if let Some(ci) = self.m.class_info(c, args) {
                    return match ci.fields.iter().find(|(n, _, _)| n == field) {
                        Some((_, t, _)) => Ok(t.subst(&class_map(ci, args))),
                        None => Err(Error::new("AttributeError", format!("`{c}` has no field `{field}`"), span)),
                    };
                }
                Ok(Ty::Unknown)
            }
            Ty::Param(n) => Err(type_err(param_field_msg(n, field), span)),
            _ => Ok(Ty::Unknown),
        }
    }

    /// A statically wrong store into a typed field is an error (the VM
    /// checks every store).
    fn check_field_store(&self, fty: &Ty, field: &str, v: &Ty, span: Span) -> R<()> {
        if !self.compatible(fty, v) {
            return Err(type_err(self.explain(fty, v, format!("cannot store a `{}` in field `{field}` of type `{}`", v.name(), fty.name())), span));
        }
        Ok(())
    }

    /// Stores an owned value into a name or tuple target.
    fn assign_owned(&mut self, target: &Expr, v: Res, span: Span) -> R<()> {
        match &target.kind {
            ExprKind::Name(n) => self.bind(n, v, target.span),
            ExprKind::Tuple(items) => {
                let mut dsts = Vec::new();
                for _ in items {
                    let r = self.tmp_owned();
                    dsts.push(Arg::new(r, true, false, None));
                }
                let al = self.arglist(ArgList { args: dsts.clone(), pins: false, fast: false });
                self.emit(Instr::Unpack { src: v.reg, dsts: al });
                let tys: Vec<Ty> = match &v.ty {
                    Ty::Tuple(Some(ts)) if ts.len() == items.len() => ts.clone(),
                    _ => vec![Ty::Unknown; items.len()],
                };
                for ((item, d), ty) in items.iter().zip(dsts).zip(tys) {
                    let r = Res { reg: d.reg, own: Own::Owned, ty, root: None, desc: "value".into() };
                    self.assign_owned(item, r, span)?;
                }
                Ok(())
            }
            ExprKind::Attr(..) | ExprKind::Index(..) => {
                Err(syn_err("only names can be bound here", target.span))
            }
            _ => Err(syn_err("invalid assignment target", target.span)),
        }
    }

    /// Binds an owned temporary to a local.
    fn bind(&mut self, n: &str, v: Res, span: Span) -> R<()> {
        if self.captures.iter().any(|(c, _)| c == n) {
            return Err(syn_err(format!("cannot assign to captured variable `{n}`"), span));
        }
        let l = &self.locals[n];
        if l.kind == LocalKind::Borrowed {
            return Err(own_err(format!("cannot assign to borrowed parameter `{n}` (declare it `+ keep` to own it)"), span));
        }
        let decl = l.decl.clone();
        if let Some((dty, _)) = &decl {
            if !self.compatible(dty, &v.ty) {
                return Err(type_err(self.explain(dty, &v.ty, format!("`{n}` is declared `{}` but assigned a `{}`", dty.name(), v.ty.name())), span));
            }
        }
        let ty = match &decl {
            Some((dty, _)) => dty.as_value(),
            None => v.ty.clone(),
        };
        let reg = self.locals[n].reg;
        if let Some((_, d)) = decl.as_ref().filter(|(dty, _)| !self.proves(dty, &v.ty)) {
            let ti = self.type_idx(d.clone());
            let w = self.name_idx(&format!("variable `{n}`"));
            self.emit(Instr::CheckType { reg: v.reg, ty: ti, what: w });
        }
        let prev = self.st(n).map(|x| x.st).unwrap_or(St::Uninit);
        let reassign = !matches!(prev, St::Uninit | St::Moved);
        match self.instrs.last() {
            // `x = <float arithmetic>`: the raw result replaces the old box.
            Some(&Instr::BoxF { dst, src }) if reassign && dst == v.reg && v.ty == Ty::Float && !self.is_target(self.here()) => {
                self.instrs.pop();
                self.lines.pop();
                self.emit(Instr::StoreRawF { dst: reg, src });
            }
            _ => {
                if reassign {
                    self.emit(Instr::Drop { reg });
                }
                self.emit(Instr::Mov { dst: reg, src: v.reg });
            }
        }
        self.set_var(n, St::Live, ty);
        self.unnarrow(n, None);
        Ok(())
    }

    fn aug_assign(&mut self, target: &Expr, op: BinOp, value: &Expr, span: Span) -> R<()> {
        let bin = Expr { kind: ExprKind::Binary(op, Box::new(target.clone()), Box::new(value.clone())), span };
        match &target.kind {
            ExprKind::Name(_) => {
                self.raw_dest = self.float_local_dest(target, &bin);
                let r = self.expr(&bin)?;
                self.assign_owned(target, r, span)
            }
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                let r = self.expr(&bin)?;
                let n = self.name_idx(field);
                self.emit(Instr::StoreField { obj: o.reg, name: n, src: r.reg, mode: StoreMode::Move });
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                let r = self.expr(&bin)?;
                let proven = self.check_elem_store(&o.ty, &i, &r, span)?;
                self.emit(Instr::StoreIndex { obj: o.reg, idx: i.reg, src: r.reg, mode: StoreMode::Move, proven });
                self.discard(&i);
                Ok(())
            }
            _ => Err(syn_err("invalid augmented assignment target", target.span)),
        }
    }

    /// `t1 <- t2 <- ... <- value`
    fn move_chain(&mut self, targets: &[Expr], value: &Expr, span: Span) -> R<()> {
        self.expect = targets.last().and_then(|t| self.target_ty(t));
        let mut cur = match &value.kind {
            ExprKind::If { .. } | ExprKind::Try(_) => {
                let dst = self.tmp_owned();
                let ty = self.value_into(value, dst)?;
                Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() }
            }
            _ => self.expr_owned(value, "this place")?,
        };
        // Swap through the inner targets from right to left.
        for t in targets[1..].iter().rev() {
            cur = self.swap_into(t, cur, span)?;
        }
        let t = &targets[0];
        match &t.kind {
            ExprKind::Name(_) => self.assign_owned(t, cur, span),
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                let fty = self.field_ty(&o.ty, field, t.span)?;
                self.check_field_store(&fty, field, &cur.ty, value.span)?;
                let n = self.name_idx(field);
                self.emit(Instr::StoreField { obj: o.reg, name: n, src: cur.reg, mode: StoreMode::Move });
                if let ExprKind::Name(base) = &obj.kind {
                    self.unnarrow(base, Some(field));
                }
                self.discard(&o);
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                let proven = self.check_elem_store(&o.ty, &i, &cur, span)?;
                self.emit(Instr::StoreIndex { obj: o.reg, idx: i.reg, src: cur.reg, mode: StoreMode::Move, proven });
                self.discard(&i);
                self.discard(&o);
                Ok(())
            }
            _ => Err(syn_err("invalid move target", t.span)),
        }
    }

    /// Checks a store of `v` at key/index `k` of a container of type
    /// `container`: a statically wrong type is an error. The VM checks every
    /// store against the container's runtime element type; returns true if
    /// the static types prove the check would pass, so it can be skipped.
    fn check_elem_store(&mut self, container: &Ty, k: &Res, v: &Res, span: Span) -> R<bool> {
        let (key, elem) = match container {
            Ty::Param(n) => return Err(type_err(format!("cannot index a value of type parameter `{n}`"), span)),
            Ty::List(t) => (Ty::Unknown, (**t).clone()),
            Ty::Dict(k, v) => ((**k).clone(), (**v).clone()),
            _ => return Ok(false),
        };
        for (want, got, what) in [(&key, k, "key"), (&elem, v, "element")] {
            if !self.compatible(want, &got.ty) {
                return Err(type_err(
                    self.explain(want, &got.ty, format!("cannot store `{}` as {what} of `{}`", got.ty.name(), container.name())),
                    span,
                ));
            }
        }
        // Containers are invariant and `Send` is part of the type, so an exact
        // static match proves the store.
        Ok(matches!(container, Ty::List(_)) && elem == v.ty && elem.is_precise())
    }

    /// The runtime descriptor that checks values of static type `t`
    /// (`None` when nothing is known about it).
    fn ty_desc(&self, t: &Ty) -> Option<TypeDesc> {
        let prim = |c| Some(TypeDesc::prim(c));
        match t {
            Ty::Unknown | Ty::NoneT | Ty::Class(_) | Ty::TypeVal(_) => None,
            Ty::Any => prim(Ctor::Any),
            Ty::Int => prim(Ctor::Int),
            Ty::Float => prim(Ctor::Float),
            Ty::Bool => prim(Ctor::Bool),
            Ty::BigInt => prim(Ctor::BigInt),
            Ty::Str => prim(Ctor::Str),
            Ty::Unit => prim(Ctor::Unit),
            Ty::Tuple(None) => prim(Ctor::Tuple),
            Ty::Tuple(Some(ts)) => Some(self.app(Ctor::Tuple, ts.iter())),
            Ty::List(e) => Some(self.app(Ctor::List, [&**e])),
            Ty::Dict(k, v) => Some(self.app(Ctor::Dict, [&**k, &**v])),
            Ty::Iter(e) => Some(self.app(Ctor::Iterator, [&**e])),
            Ty::Chan(e) => Some(self.app(Ctor::Channel, [&**e])),
            Ty::Array(e) => prim(Ctor::Array(*e)),
            Ty::Func | Ty::StaticFn => prim(Ctor::Function),
            Ty::Opaque(n) => prim(if n == "Lock" { Ctor::Lock } else { Ctor::Thread }),
            Ty::Instance(c, args) | Ty::Trait(c, args) => {
                let idx = *self.m.gindex.get(c)?;
                // Unknown type arguments: any instantiation.
                let ds: Option<Vec<TypeDesc>> =
                    args.iter().map(|a| if *a == Ty::Unknown { None } else { self.ty_desc(a) }).collect();
                Some(TypeDesc::Global(idx, c.clone(), ds.unwrap_or_default()))
            }
            Ty::Param(n) => self.tscope.iter().find(|t| t.name == *n).map(|t| TypeDesc::Param(t.reg, n.clone())),
            Ty::Local(t) => Some(TypeDesc::Local(Box::new(self.ty_desc(t)?))),
            Ty::Option(inner) => Some(TypeDesc::option(self.ty_desc(inner)?)),
        }
    }

    /// `c<args>` if every argument is known, else the bare `c`.
    fn app<'a>(&self, c: Ctor, args: impl IntoIterator<Item = &'a Ty>) -> TypeDesc {
        let ds: Option<Vec<TypeDesc>> =
            args.into_iter().map(|a| if *a == Ty::Unknown { None } else { self.ty_desc(a) }).collect();
        match ds {
            Some(ds) => TypeDesc::Builtin(c, ds),
            None => TypeDesc::prim(c),
        }
    }

    /// The runtime element type for a container created with static element
    /// type `t` (`any` when unknown).
    fn elem_desc(&self, t: &Ty) -> TypeDesc {
        self.ty_desc(t).unwrap_or_else(TypeDesc::any)
    }

    /// Stores owned `v` into `t`, returning the previous value (owned).
    fn swap_into(&mut self, t: &Expr, v: Res, span: Span) -> R<Res> {
        let dst = self.tmp_owned();
        match &t.kind {
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                let fty = self.field_ty(&o.ty, field, t.span)?;
                self.check_field_store(&fty, field, &v.ty, span)?;
                let n = self.name_idx(field);
                self.emit(Instr::SwapField { dst, obj: o.reg, name: n, src: v.reg });
                if let ExprKind::Name(base) = &obj.kind {
                    self.unnarrow(base, Some(field));
                }
                self.discard(&o);
                Ok(Res { reg: dst, own: Own::Owned, ty: fty.as_value(), root: None, desc: "value".into() })
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                self.check_elem_store(&o.ty, &i, &v, span)?;
                self.emit(Instr::SwapIndex { dst, obj: o.reg, idx: i.reg, src: v.reg });
                self.discard(&i);
                self.discard(&o);
                let ty = o.ty.elem().unwrap_or(Ty::Unknown);
                Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() })
            }
            ExprKind::Name(n) => {
                // `x <- y <- v`: x gets y's old value, y gets v.
                let old = self.name_expr(n, t.span)?;
                let old = self.to_owned(old, t.span, "a variable")?;
                self.emit(Instr::Mov { dst, src: old.reg });
                self.bind(n, v, span)?;
                Ok(Res { reg: dst, own: Own::Owned, ty: old.ty, root: None, desc: "value".into() })
            }
            _ => Err(syn_err("invalid swap target", t.span)),
        }
    }

    fn del_target(&mut self, t: &Expr) -> R<()> {
        match &t.kind {
            ExprKind::Name(n) => {
                let Some(l) = self.locals.get(n) else {
                    return Err(syn_err("`del` of a global is not supported", t.span));
                };
                if l.kind == LocalKind::Borrowed {
                    return Err(own_err(format!("cannot `del` borrowed parameter `{n}`"), t.span));
                }
                let reg = l.reg;
                self.emit(Instr::Drop { reg });
                self.set_st(n, St::Uninit);
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                let al = self.arglist(ArgList { args: vec![Arg::new(i.reg, false, false, None)], pins: false, fast: false });
                let n = self.name_idx("pop");
                let dst = self.tmp_owned();
                self.emit(Instr::CallMethod { dst, recv: o.reg, name: n, args: al });
                self.emit(Instr::Drop { reg: dst });
                self.discard(&i);
                self.discard(&o);
                Ok(())
            }
            _ => Err(syn_err("invalid `del` target", t.span)),
        }
    }

    // ------------------------------------------------------------ calls

    /// Static signature of a call target, if known, with its result type.
    /// A generic class's constructor has the class's type parameters.
    fn static_sig(&self, f: &Expr) -> Option<(Sig, Ty)> {
        match &f.kind {
            ExprKind::Name(n) if self.is_global_name(n) => {
                if let Some(s) = self.m.funcs.get(n) {
                    return Some((s.clone(), s.ret.clone()));
                }
                if let Some(ci) = self.m.classes.get(n) {
                    let mut sig =
                        ci.methods.get("__init__").cloned().unwrap_or(Sig { tparams: Vec::new(), params: Vec::new(), ret: Ty::Unit });
                    sig.tparams = ci.tparams.clone();
                    let ret = Ty::Instance(n.clone(), ci.tparams.iter().map(|t| Ty::Param(t.name.clone())).collect());
                    sig.ret = ret.clone();
                    return Some((sig, ret));
                }
                if self.m.builtin_classes.contains(n) {
                    let params = vec![ParamSig { name: "msg".into(), keep: false, ty: Ty::Unknown }];
                    return Some((Sig { tparams: Vec::new(), params, ret: Ty::Unknown }, Ty::Instance(n.clone(), Vec::new())));
                }
                None
            }
            _ => None,
        }
    }

    fn builtin_call_info(&self, f: &Expr) -> Option<(u64, String)> {
        if let ExprKind::Name(n) = &f.kind {
            if self.is_global_name(n) && !self.m.module_vars.contains(n) {
                return lizzard_vm::builtins::keep_mask(n).map(|m| (m, n.clone()));
            }
        }
        None
    }

    /// Resolves turbofish type arguments for a generic with `n` parameters.
    fn explicit_targs(&self, targs: &[TypeExpr], n: usize, what: &str, span: Span) -> R<Vec<Ty>> {
        if targs.len() != n {
            return Err(type_err(
                format!("`{what}` takes {n} type argument{} but {} {} given", if n == 1 { "" } else { "s" }, targs.len(), if targs.len() == 1 { "was" } else { "were" }),
                span,
            ));
        }
        targs.iter().map(|t| self.resolve_type(t).map(|r| r.0)).collect()
    }

    /// The global of `base::n`: an associated function `Type::n`.
    fn path_global(&self, base: &Expr, n: &str, span: Span) -> R<String> {
        let ExprKind::Name(tn) = &base.kind else {
            return Err(syn_err("expected a type name before `::`", span));
        };
        if tn == "Self" {
            return Err(type_err("`Self` is only available inside `impl` blocks", span));
        }
        let g = format!("{tn}::{n}");
        if self.m.funcs.contains_key(&g) {
            return Ok(g);
        }
        match self.m.classes.get(tn) {
            Some(ci) if ci.lookup(n).is_some() || self.m.has_registry_method(tn, n) => {
                Err(type_err(format!("`{n}` is a method of `{tn}`, not an associated function: call it on a value (`x.{n}(...)`)"), span))
            }
            Some(_) => Err(Error::new("AttributeError", format!("`{tn}` has no associated function `{n}`"), span)),
            None if self.m.traits.contains_key(tn) || builtin_trait(tn).is_some() => {
                Err(type_err(format!("`{tn}::{n}` is a trait method: call it with a receiver (`{tn}::{n}(x, ..)` or `x.{n}()`)"), span))
            }
            None => Err(name_err(format!("`{tn}` is not a struct"), span)),
        }
    }

    /// `Name { field: value, .. }`: every field once, in any order. The type
    /// arguments come from `Name::<T>`, the expected type, or the values.
    fn struct_lit(&mut self, base: &Expr, fields: &[(String, Expr, Span)], expect: Option<Ty>, span: Span) -> R<Res> {
        let (name, turbofish) = match &base.kind {
            ExprKind::Name(n) => (n.clone(), None),
            ExprKind::TypeApp(b, ts) => match &b.kind {
                ExprKind::Name(n) => (n.clone(), Some(ts)),
                _ => return Err(syn_err("expected a struct name before `{`", span)),
            },
            _ => return Err(syn_err("expected a struct name before `{`", span)),
        };
        if name == "Self" {
            return Err(type_err("`Self { .. }` is only available inside `impl` blocks", span));
        }
        let Some(ci) = self.m.classes.get(&name) else {
            if self.m.builtin_classes.contains(&name) {
                return Err(type_err(format!("`{name}` is a builtin class: create it with `{name}(\"message\")`"), span));
            }
            return Err(name_err(format!("unknown struct `{name}`"), span));
        };
        if ci.native {
            return Err(type_err(format!("`{name}` is a native class: create it by calling it (`{name}(...)`)"), span));
        }
        let mut seen: Vec<&str> = Vec::new();
        for (f, _, fspan) in fields {
            if !ci.fields.iter().any(|(n, _, _)| n == f) {
                return Err(Error::new("AttributeError", format!("struct `{name}` has no field `{f}`"), *fspan));
            }
            if seen.contains(&f.as_str()) {
                return Err(syn_err(format!("field `{f}` is given twice"), *fspan));
            }
            seen.push(f);
        }
        let missing: Vec<String> = ci.fields.iter().filter(|(n, _, _)| !seen.contains(&n.as_str())).map(|(n, _, _)| format!("`{n}`")).collect();
        if !missing.is_empty() {
            return Err(type_err(format!("missing field{} {} in `{name} {{ .. }}`", if missing.len() == 1 { "" } else { "s" }, missing.join(", ")), span));
        }
        let tps = ci.tparams.clone();
        let decl: Vec<(String, Ty)> = ci.fields.iter().map(|(n, t, _)| (n.clone(), t.clone())).collect();
        let mut known: Vec<Option<Ty>> = match turbofish {
            Some(ts) => self.explicit_targs(ts, tps.len(), &name, span)?.into_iter().map(Some).collect(),
            None => vec![None; tps.len()],
        };
        if turbofish.is_none() {
            if let Some(Ty::Instance(c, a)) = expect.as_ref().map(|t| t.peel()) {
                if *c == name && a.len() == tps.len() {
                    known = a.iter().map(|t| (*t != Ty::Unknown).then(|| t.clone())).collect();
                }
            }
        }
        let sub = |known: &[Option<Ty>], t: &Ty| {
            t.subst(&|p| tps.iter().position(|x| x.name == p).map(|i| known[i].clone().unwrap_or(Ty::Unknown)))
        };
        // Values in source order, owned (moved in).
        let mut vals: Vec<(usize, Res, Span, bool)> = Vec::new();
        for (f, v, fspan) in fields {
            let k = decl.iter().position(|(n, _)| n == f).unwrap();
            let pre = sub(&known, &decl[k].1);
            self.expect = (pre != Ty::Unknown).then_some(pre);
            let mut r = self.expr_owned(v, &format!("field `{f}`"))?;
            self.unify(&decl[k].1, &r.ty, &tps, &mut known);
            // A float result for a `float` field is stored raw, unboxed.
            let mut raw = false;
            if let Some(&Instr::BoxF { dst, src }) = self.instrs.last() {
                if dst == r.reg && decl[k].1 == Ty::Float && !self.is_target(self.here()) {
                    self.instrs.pop();
                    self.lines.pop();
                    r.reg = src;
                    raw = true;
                }
            }
            vals.push((k, r, *fspan, raw));
        }
        if known.iter().any(|k| k.is_none()) {
            let names: Vec<&str> = tps.iter().map(|t| t.name.as_str()).collect();
            return Err(type_err(
                format!("cannot infer the type arguments of `{name}`; write `{name}::<{}> {{ .. }}` or annotate the variable", names.join(", ")),
                span,
            ));
        }
        for (k, r, fspan, _) in &vals {
            let fty = sub(&known, &decl[*k].1);
            if !self.compatible(&fty, &r.ty) {
                let f = &decl[*k].0;
                return Err(type_err(self.explain(&fty, &r.ty, format!("field `{f}` of `{name}` expects `{}`, got `{}`", fty.name(), r.ty.name())), *fspan));
            }
        }
        let t = Ty::Instance(name.clone(), known.into_iter().map(|k| k.unwrap()).collect());
        self.m.check_wf(&t, &|n| self.pbounds(n), span)?;
        let Some(d) = self.ty_desc(&t) else {
            return Err(type_err(format!("cannot build `{}` here", t.name()), span));
        };
        vals.sort_by_key(|(k, _, _, _)| *k);
        let args = vals.iter().map(|(_, r, _, raw)| Arg { raw: *raw, ..Arg::new(r.reg, !raw, false, None) }).collect();
        let ti = self.type_idx(d);
        let al = self.arglist(ArgList { args, pins: false, fast: false });
        let dst = self.tmp_owned();
        self.emit(Instr::BuildStruct { dst, args: al, ty: ti });
        Ok(Res { reg: dst, own: Own::Owned, ty: t, root: None, desc: "value".into() })
    }

    fn call_expr(&mut self, f: &Expr, args: &[Expr], expect: Option<Ty>, span: Span) -> R<Res> {
        if let ExprKind::Attr(recv, mname) = &f.kind {
            return self.method_call(recv, mname, args, expect, span);
        }
        // `Type::f(...)` / `Type::<T>::f(...)`: the module function `Type::f`
        // (its leading type parameters are the impl's).
        if let ExprKind::Path(base, n) = &f.kind {
            // `Trait::m(x, ..)` / `Type::m(x, ..)`: a method with `x` as `self`.
            if let ExprKind::Name(tn) = &base.kind {
                let is_trait = self.m.traits.contains_key(tn);
                let is_method = !is_trait
                    && !self.m.funcs.contains_key(&format!("{tn}::{n}"))
                    && self.m.classes.get(tn).is_some_and(|ci| ci.lookup(n).is_some() || self.m.has_registry_method(tn, n));
                if is_trait || is_method {
                    let Some((recv, rest)) = args.split_first() else {
                        return Err(type_err(format!("`{tn}::{n}` needs the receiver as its first argument: `{tn}::{n}(x, ..)`"), span));
                    };
                    return self.method_call_q(recv, n, rest, expect, span, is_trait.then_some(tn.as_str()));
                }
            }
            let (b, targs) = match &base.kind {
                ExprKind::TypeApp(b, ts) => (&**b, Some(ts)),
                _ => (&**base, None),
            };
            let g = Expr { kind: ExprKind::Name(self.path_global(b, n, f.span)?), span: f.span };
            let callee = match targs {
                Some(ts) => Expr { kind: ExprKind::TypeApp(Box::new(g), ts.clone()), span: f.span },
                None => g,
            };
            return self.call_expr(&callee, args, expect, span);
        }
        let named = match &f.kind {
            ExprKind::TypeApp(b, _) => &b.kind,
            k => k,
        };
        if let ExprKind::Name(n) = named {
            if self.is_global_name(n) && self.m.classes.get(n).is_some_and(|c| !c.native) {
                return Err(type_err(
                    format!("`{n}` is a struct: create it with a literal `{n} {{ .. }}` or an associated function (`{n}::new(...)`)"),
                    span,
                ));
            }
        }
        // `name::<types>(...)` of a user class or function: the signature is
        // the name's; the type arguments are explicit.
        let (base, turbofish) = match &f.kind {
            ExprKind::TypeApp(b, ts)
                if matches!(&b.kind, ExprKind::Name(n) if self.is_global_name(n) && (self.m.classes.contains_key(n) || self.m.funcs.contains_key(n))) =>
            {
                (&**b, Some(ts))
            }
            _ => (f, None),
        };
        let sig = self.static_sig(base);
        if let Some((s, _)) = &sig {
            if !s.tparams.is_empty() {
                return self.generic_call(base, s.clone(), turbofish.map(|v| v.as_slice()), args, expect, span);
            }
        }
        let mut builtin = self.builtin_call_info(f);
        // `list()`, `dict()`, `Channel()` build the type the context expects.
        let ctor_ty = match (&builtin, expect.as_ref()) {
            (Some((_, n)), Some(Ty::List(e))) if n == "list" && **e != Ty::Unknown => expect.clone(),
            (Some((_, n)), Some(Ty::Dict(k, v))) if n == "dict" && (**k != Ty::Unknown && **v != Ty::Unknown) => expect.clone(),
            (Some((_, n)), Some(Ty::Chan(e))) if n == "Channel" && **e != Ty::Unknown => expect.clone(),
            _ => None,
        };
        if ctor_ty.is_none() {
            if let Some((_, n)) = &builtin {
                let what = match (n.as_str(), args.len()) {
                    ("list", 0) => Some(("list()", "xs: list<int> = list()` or `list::<int>()")),
                    ("dict", 0) => Some(("dict()", "d: dict<str, int> = dict()` or `dict::<str, int>()")),
                    ("Channel", 0) => Some(("Channel()", "ch: Channel<int> = Channel()` or `Channel::<int>()")),
                    _ => None,
                };
                if let Some((call, hint)) = what {
                    return Err(type_err(format!("cannot infer the element type of `{call}`; write `{hint}`"), span));
                }
            }
        }
        let callee = match &ctor_ty {
            Some(t) => {
                builtin = None;
                let d = self.ty_desc(t).expect("container type");
                let dst = self.tmp_owned();
                let ti = self.type_idx(d);
                self.emit(Instr::MakeType { dst, ty: ti });
                Res { reg: dst, own: Own::Owned, ty: Ty::TypeVal(Box::new(t.clone())), root: None, desc: "value".into() }
            }
            None => self.expr(f)?,
        };
        if let Ty::Param(n) = &callee.ty {
            return Err(type_err(format!("cannot call a value of type parameter `{n}`"), span));
        }
        if let Some((s, _)) = &sig {
            if s.params.len() != args.len() && !matches!(&f.kind, ExprKind::Name(n) if self.m.builtin_classes.contains(n)) {
                return Err(type_err(
                    format!("`{}` takes {} argument{} but {} {} given", callee_name(f), s.params.len(), if s.params.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "was" } else { "were" }),
                    span,
                ));
            }
        }
        let keep_of = |i: usize| -> Option<bool> {
            if let Some((s, _)) = &sig {
                return s.params.get(i).map(|p| p.keep);
            }
            if let Some((mask, _)) = &builtin {
                return Some(mask & (1 << i) != 0);
            }
            None
        };
        let arg_ty = |i: usize| -> Option<(String, Ty)> {
            sig.as_ref().and_then(|(s, _)| s.params.get(i)).map(|p| (p.name.clone(), p.ty.clone()))
        };
        let mut al = self.call_args(args, &keep_of, &arg_ty, None, span)?;
        let arg_tys = std::mem::take(&mut self.last_arg_tys);
        if let (Some((s, _)), ExprKind::Name(n), None) = (&sig, &base.kind, &builtin) {
            al.fast = self.m.funcs.contains_key(n)
                && s.tparams.is_empty()
                && s.params.len() == arg_tys.len()
                && s.params.iter().zip(&arg_tys).all(|(p, t)| !p.keep && (p.ty == Ty::Unknown || self.proves(&p.ty, t)));
        }
        if let Some((_, n)) = &builtin {
            self.check_builtin_bounds(n, &arg_tys, span)?;
            if n == "spawn" {
                self.check_spawn(args, &arg_tys)?;
            }
        }
        let dst = self.tmp_owned();
        let a = self.arglist(al.clone());
        self.emit(Instr::Call { dst, func: callee.reg, args: a });
        self.after_call(&al);
        self.discard(&callee);
        let mut ty = match &callee.ty {
            Ty::TypeVal(t) => (**t).clone(),
            _ => sig.map(|s| s.1.as_value()).unwrap_or(Ty::Unknown),
        };
        if let Some((_, n)) = &builtin {
            let arg = |i: usize| arg_tys.get(i).cloned().unwrap_or(Ty::Unknown);
            let arg_item = |i: usize| self.item_ty(&arg(i), span).unwrap_or(Ty::Unknown);
            ty = match n.as_str() {
                "str" | "repr" | "chr" => Ty::Str,
                "int" | "len" | "ord" | "hash" | "rc_run" | "rc_pending" | "rc_live" => Ty::Int,
                "float" => Ty::Float,
                "bool" | "isinstance" => Ty::Bool,
                "bigint" => Ty::BigInt,
                "list" | "sorted" => Ty::list(arg_item(0)),
                "tuple" => Ty::Tuple(None),
                "dict" => Ty::dict(Ty::Unknown, Ty::Unknown),
                "range" => Ty::iter(Ty::Int),
                "iter" => Ty::iter(arg_item(0)),
                "enumerate" => Ty::iter(Ty::Tuple(Some(vec![Ty::Int, arg_item(0)]))),
                "zip" => Ty::iter(Ty::Tuple(Some(vec![arg_item(0), arg_item(1)]))),
                "Channel" => Ty::Chan(Box::new(Ty::Unknown)),
                "Lock" => Ty::Opaque("Lock".into()),
                "spawn" => Ty::Opaque("Thread".into()),
                "array" => Ty::Array(match args.first().map(|a| &a.kind) {
                    Some(ExprKind::Name(n)) if !self.locals.contains_key(n) && !self.m.module_vars.contains(n) => {
                        lizzard_vm::array::Elem::from_name(n)
                    }
                    _ => None,
                }),
                "print" | "rc_auto" => Ty::Unit,
                "Some" => match arg(0) {
                    Ty::Unknown => Ty::Unknown,
                    t => Ty::Option(Box::new(t)),
                },
                "min" | "max" if args.len() == 1 => match arg_item(0) {
                    t @ Ty::Param(_) => t,
                    _ => Ty::Unknown,
                },
                "min" | "max" if arg_tys.iter().all(|t| matches!(t, Ty::Param(_)) && *t == arg_tys[0]) => arg(0),
                _ => Ty::Unknown,
            };
        }
        self.reg_ty.clear();
        Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() })
    }

    /// Builtins that need a bound on type-parameter arguments.
    /// `spawn(f, args...)`: everything that crosses to the new thread is
    /// `Send`, including `f`'s result (which `join` hands back).
    fn check_spawn(&self, args: &[Expr], arg_tys: &[Ty]) -> R<()> {
        for (i, (a, t)) in args.iter().zip(arg_tys).enumerate() {
            if self.may_be_local(t) {
                let what = if i == 0 { "the function".to_string() } else { format!("argument {i}") };
                let msg = match t {
                    Ty::Param(n) => format!("spawn(): {what} has type `{n}`, which may not be `Send`; add a `{n}: Send` bound"),
                    _ => format!("spawn(): {what} is not `Send` (`{}`) and cannot be sent to another thread", t.name()),
                };
                return Err(type_err(msg, a.span));
            }
        }
        if let Some((sig, _)) = args.first().and_then(|f| self.static_sig(f)) {
            // The callee's own type parameters are bound by the arguments.
            if self.m.is_local(&sig.ret, &|_| false) {
                return Err(type_err(
                    format!("spawn(): `{}` returns `{}`, which is not `Send` and cannot leave its thread", callee_name(&args[0]), sig.ret.name()),
                    args[0].span,
                ));
            }
        }
        Ok(())
    }

    fn check_builtin_bounds(&self, n: &str, arg_tys: &[Ty], span: Span) -> R<()> {
        let item = |i: usize| arg_tys.get(i).map(|t| self.item_ty(t, span)).transpose().map(|t| t.unwrap_or(Ty::Unknown));
        match n {
            "len" => self.require(arg_tys.first().unwrap_or(&Ty::Unknown), "Len", "`len()`", span),
            "hash" => self.require(arg_tys.first().unwrap_or(&Ty::Unknown), "Hash", "`hash()`", span),
            "sorted" => self.require(&item(0)?, "Ord", "`sorted()`", span),
            "sum" => self.require(&item(0)?, "Num", "`sum()`", span),
            "min" | "max" if arg_tys.len() == 1 => self.require(&item(0)?, "Ord", &format!("`{n}()`"), span),
            "min" | "max" => arg_tys.iter().try_for_each(|t| self.require(t, "Ord", &format!("`{n}()`"), span)),
            _ => Ok(()),
        }
    }

    /// A call of a generic function or a generic class's constructor:
    /// infers the type arguments and passes them (as hidden leading
    /// arguments of a function, or as the instantiated class type).
    #[allow(clippy::too_many_arguments)]
    fn generic_call(
        &mut self,
        f: &Expr,
        sig: Sig,
        turbofish: Option<&[TypeExpr]>,
        args: &[Expr],
        expect: Option<Ty>,
        span: Span,
    ) -> R<Res> {
        let ExprKind::Name(name) = &f.kind else { unreachable!() };
        let class = self.m.classes.contains_key(name);
        if class && !self.m.classes[name].instances.is_empty() {
            return self.native_generic_call(name, turbofish, args, expect, span);
        }
        if sig.params.len() != args.len() {
            return Err(type_err(
                format!("`{name}()` takes {} argument{} but {} {} given", sig.params.len(), if sig.params.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "was" } else { "were" }),
                span,
            ));
        }
        let explicit = match turbofish {
            // `Stack::<int>::new()` gives the impl's type parameters; the
            // function's own are inferred.
            Some(ts) if name.contains("::") && ts.len() < sig.tparams.len() => {
                Some(ts.iter().map(|t| self.resolve_type(t).map(|r| r.0)).collect::<R<Vec<_>>>()?)
            }
            Some(ts) => Some(self.explicit_targs(ts, sig.tparams.len(), name, span)?),
            None => None,
        };
        // A function value is loaded before its arguments are evaluated.
        let callee = if class { None } else { Some(self.expr(f)?) };
        let what = if class { name.clone() } else { format!("{name}()") };
        let (mut al, targs, ret) = self.generic_call_args(&sig, explicit, expect.as_ref(), args, None, &what, span)?;
        let known: Option<Vec<Ty>> = targs.iter().cloned().collect();
        let callee = match callee {
            Some(c) => {
                // Statically known type arguments are passed; otherwise the
                // callee infers them from the arguments at runtime.
                if let Some(regs) = known.as_ref().and_then(|k| self.make_types(k)) {
                    let mut full: Vec<Arg> = regs.into_iter().map(|r| Arg::new(r, false, false, None)).collect();
                    full.append(&mut al.args);
                    al.args = full;
                }
                c
            }
            None => {
                let ci = &self.m.classes[name];
                let Some(k) = known else {
                    let names: Vec<&str> = ci.tparams.iter().map(|t| t.name.as_str()).collect();
                    return Err(type_err(
                        format!("cannot infer the type arguments of `{name}`; write `{name}::<{}>(...)` or annotate the variable", names.join(", ")),
                        span,
                    ));
                };
                let t = Ty::Instance(name.clone(), k);
                let Some(d) = self.ty_desc(&t) else {
                    return Err(type_err(format!("cannot instantiate `{}` here", t.name()), span));
                };
                let dst = self.tmp_owned();
                let ti = self.type_idx(d);
                self.emit(Instr::MakeType { dst, ty: ti });
                Res { reg: dst, own: Own::Owned, ty: Ty::TypeVal(Box::new(t)), root: None, desc: "value".into() }
            }
        };
        let dst = self.tmp_owned();
        let a = self.arglist(al.clone());
        self.emit(Instr::Call { dst, func: callee.reg, args: a });
        self.after_call(&al);
        self.discard(&callee);
        self.reg_ty.clear();
        let ty = match callee.ty {
            Ty::TypeVal(t) => *t,
            _ => ret.as_value(),
        };
        Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() })
    }

    /// `Name::<args>(...)` of a generic native class (or with the type
    /// arguments from the expected type): checked against the instantiation's
    /// constructor when the arguments are known; otherwise the runtime picks
    /// the instantiation (`TypeError` if there is none).
    fn native_generic_call(&mut self, name: &str, turbofish: Option<&[TypeExpr]>, args: &[Expr], expect: Option<Ty>, span: Span) -> R<Res> {
        let n = self.m.classes[name].tparams.len();
        let targs = match turbofish {
            Some(ts) => self.explicit_targs(ts, n, name, span)?,
            None => match expect.as_ref().map(|t| t.peel()) {
                Some(Ty::Instance(c, a)) if c == name && a.len() == n => a.clone(),
                _ => {
                    let names: Vec<&str> = self.m.classes[name].tparams.iter().map(|t| t.name.as_str()).collect();
                    return Err(type_err(
                        format!("cannot infer the type arguments of `{name}`; write `{name}::<{}>(...)` or annotate the variable", names.join(", ")),
                        span,
                    ));
                }
            },
        };
        let t = Ty::Instance(name.into(), targs);
        self.m.check_wf(&t, &|n| self.pbounds(n), span)?;
        let Some(d) = self.ty_desc(&t) else {
            return Err(type_err(format!("cannot instantiate `{}` here", t.name()), span));
        };
        let init = match &t {
            Ty::Instance(c, a) => self.m.class_info(c, a).map(|ci| ci.methods.get("__init__").cloned()),
            _ => None,
        };
        let sig = match init {
            Some(Some(s)) => Some(s),
            Some(None) => return Err(type_err(format!("cannot create `{}` objects from lizzard", t.name()), span)),
            None => None,
        };
        if let Some(s) = &sig {
            if s.params.len() != args.len() {
                return Err(type_err(
                    format!("`{}()` takes {} argument{} but {} {} given", t.name(), s.params.len(), if s.params.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "was" } else { "were" }),
                    span,
                ));
            }
        }
        let callee = self.tmp_owned();
        let ti = self.type_idx(d);
        self.emit(Instr::MakeType { dst: callee, ty: ti });
        // Native parameters are borrowed.
        let al = self.call_args(args, &|_| Some(false), &|i| sig.as_ref().and_then(|s| s.params.get(i)).map(|p| (p.name.clone(), p.ty.clone())), None, span)?;
        let dst = self.tmp_owned();
        let a = self.arglist(al.clone());
        self.emit(Instr::Call { dst, func: callee, args: a });
        self.after_call(&al);
        self.emit(Instr::Drop { reg: callee });
        self.reg_ty.clear();
        Ok(Res { reg: dst, own: Own::Owned, ty: t, root: None, desc: "value".into() })
    }

    /// Evaluates call arguments. `keep_of(i)` is the static keep flag of
    /// parameter i, if known. `recv_root` is the receiver's root local.
    fn call_args(
        &mut self,
        args: &[Expr],
        keep_of: &dyn Fn(usize) -> Option<bool>,
        ty_of: &dyn Fn(usize) -> Option<(String, Ty)>,
        recv_root: Option<Reg>,
        span: Span,
    ) -> R<ArgList> {
        let mut out = Vec::new();
        let mut arg_tys = Vec::new();
        let mut moved_locals: Vec<(String, Reg, Span)> = Vec::new();
        let mut borrowed_roots: Vec<Reg> = recv_root.into_iter().collect();
        for (i, a) in args.iter().enumerate() {
            let keep = keep_of(i);
            let param = ty_of(i);
            let r = self.expr_expect(a, param.as_ref().map(|p| p.1.clone()))?;
            // An unnamed parameter only gives context (its store is checked later).
            if let Some((pname, pty)) = param.as_ref().filter(|p| !p.0.is_empty()) {
                if !self.compatible(pty, &r.ty) {
                    return Err(type_err(
                        self.explain(pty, &r.ty, format!("argument `{pname}` expects `{}`, got `{}`", pty.name(), r.ty.name())),
                        a.span,
                    ));
                }
            }
            self.reg_ty.insert(r.reg, r.ty.clone());
            arg_tys.push(r.ty.clone());
            match &r.own {
                Own::Owned => out.push(Arg::new(r.reg, true, false, None)),
                Own::Local(n) => {
                    let n = n.clone();
                    let module_var = self.is_module || self.is_global_name(&n);
                    match keep {
                        Some(true) => {
                            // Static move into the callee.
                            let o = self.move_local(&n, r.clone(), a.span)?;
                            if !module_var {
                                moved_locals.push((n.clone(), self.locals[&n].reg, a.span));
                            }
                            out.push(Arg::new(o.reg, true, false, None));
                        }
                        Some(false) => {
                            if !module_var {
                                borrowed_roots.push(r.reg);
                            }
                            out.push(Arg::new(r.reg, false, !module_var, if module_var { None } else { Some(r.reg) }));
                        }
                        None => {
                            if module_var {
                                // Unknown callee: pass a borrow of the module
                                // variable; a `keep` parameter moves it out of
                                // its slot at runtime.
                                let mut arg = Arg::new(r.reg, false, false, None);
                                arg.global = self.m.gindex[&n];
                                out.push(arg);
                                if r.ty.is_copy() != Some(true) {
                                    self.set_st(&n, St::DynMoved);
                                }
                            } else {
                                out.push(Arg::new(r.reg, true, true, None));
                                if r.ty.is_copy() != Some(true) {
                                    self.set_st(&n, St::DynMoved);
                                }
                            }
                        }
                    }
                }
                Own::Borrowed => {
                    if keep == Some(true) {
                        let o = self.to_owned(r, a.span, "a `keep` parameter")?;
                        out.push(Arg::new(o.reg, true, false, None));
                    } else {
                        if let Some(root) = r.root {
                            borrowed_roots.push(root);
                        }
                        out.push(Arg::new(r.reg, false, false, r.root));
                    }
                }
            }
        }
        for (n, reg, s) in &moved_locals {
            if borrowed_roots.contains(reg) {
                return Err(own_err(format!("`{n}` is moved into this call while the same call also borrows it; pass `{n}++` instead"), *s));
            }
        }
        let _ = span;
        self.last_arg_tys = arg_tys;
        Ok(ArgList { args: out, pins: self.heap_borrows > 0, fast: false })
    }

    fn after_call(&mut self, al: &ArgList) {
        for a in &al.args {
            if a.owned && !a.local {
                self.emit(Instr::Drop { reg: a.reg });
            }
        }
    }

    /// The bounds of type parameter `n` that provide method `mname`: (trait,
    /// signature) for a user trait's method, (trait, `None`) for a builtin
    /// trait's.
    fn param_method(&self, n: &str, mname: &str) -> Vec<(String, Option<Sig>)> {
        let mut out = Vec::new();
        for b in self.pbounds(n) {
            let Ty::Trait(bn, bargs) = &b else { continue };
            if let Some(ms) = self.m.trait_methods_for(bn, bargs, Ty::Param(n.into())) {
                if let Some(s) = ms.get(mname) {
                    out.push((bn.clone(), Some(s.clone())));
                }
            } else if let Some((_, _, implies, methods)) = builtin_trait(bn) {
                let implied = implies.iter().filter_map(|i| builtin_trait(i)).any(|t| t.3.contains(&mname));
                if methods.contains(&mname) || implied {
                    out.push((bn.clone(), None));
                }
            }
        }
        out
    }

    /// Emits a call of method `mname` on `recv`: by name, or as the method
    /// of trait `qual` (`Trait::name(x)`).
    fn emit_method_call(&mut self, dst: Reg, recv: Reg, mname: &str, args: u32, qual: Option<&str>) {
        let name = self.name_idx(mname);
        match qual {
            Some(t) => {
                let tc = (self.m.gindex[t], name);
                let call = match self.trait_calls.iter().position(|x| *x == tc) {
                    Some(i) => i as u32,
                    None => {
                        self.trait_calls.push(tc);
                        (self.trait_calls.len() - 1) as u32
                    }
                };
                self.emit(Instr::CallTraitMethod { dst, recv, call, args });
            }
            None => {
                self.emit(Instr::CallMethod { dst, recv, name, args });
            }
        }
    }

    fn method_call(&mut self, recv: &Expr, mname: &str, args: &[Expr], expect: Option<Ty>, span: Span) -> R<Res> {
        self.method_call_q(recv, mname, args, expect, span, None)
    }

    /// A method call; `force`: the trait of a qualified call
    /// (`Trait::name(x, ..)`).
    fn method_call_q(&mut self, recv: &Expr, mname: &str, args: &[Expr], expect: Option<Ty>, span: Span, force: Option<&str>) -> R<Res> {
        let r = self.expr(recv)?;
        if mname != "some_or" {
            self.option_check(&r, &format!("calling `.{mname}()`"), span)?;
        }
        let mut sig: Option<Sig> = None;
        // An inherent method of a non-generic Lizzard struct: `ArgList::fast` may apply.
        let mut plain = false;
        // The trait whose method this is: the call is qualified with it.
        let mut qual: Option<String> = force.map(String::from);
        let ambiguous = |c: &str, ts: &[String]| {
            let names: Vec<String> = ts.iter().map(|t| format!("`{t}::{mname}`")).collect();
            type_err(format!("`{mname}` is ambiguous for `{c}`: {}; call it as `{}::{mname}(x)`", names.join(" or "), ts[0]), span)
        };
        // Trait methods of registered impls that may apply, one entry per
        // trait (`None` when those impls' signatures differ).
        let by_trait = |reg: &[(Option<String>, Sig)], mut ts: Vec<(String, Option<Sig>)>| {
            for (t, s) in reg {
                let Some(t) = t else { continue };
                match ts.iter_mut().find(|(x, _)| x == t) {
                    Some((_, have)) if have.as_ref() != Some(s) => *have = None,
                    Some(_) => {}
                    None => ts.push((t.clone(), Some(s.clone()))),
                }
            }
            ts
        };
        let recv_ty = r.ty.peel().clone();
        match &recv_ty {
            Ty::Instance(c, cargs) => {
                if let Some(ci) = self.m.class_info(c, cargs) {
                    let map = class_map(ci, cargs);
                    let reg = self.m.registry_methods(&recv_ty, mname, &|n| self.pbounds(n));
                    match (force, ci.lookup(mname)) {
                        (Some(t), _) => match ci.trait_method(t, mname).map(|s| s.subst(&map)) {
                            Some(s) => sig = Some(s),
                            None if reg.iter().any(|(x, _)| x.as_deref() == Some(t)) => {
                                sig = by_trait(&reg, Vec::new()).into_iter().find(|(x, _)| x == t).and_then(|(_, s)| s)
                            }
                            // A native class's Rust method implements its traits.
                            None if ci.native && ci.methods.contains_key(mname) => sig = ci.methods.get(mname).map(|s| s.subst(&map)),
                            None => return Err(type_err(format!("`{c}` has no method `{t}::{mname}`"), span)),
                        },
                        (None, Some(MethodRef::Inherent(s))) => {
                            sig = Some(s.subst(&map));
                            plain = ci.tparams.is_empty() && !ci.native;
                        }
                        (None, found) => {
                            if let Some((_, s)) = reg.iter().find(|(t, _)| t.is_none()) {
                                // A conditional inherent impl (`impl<T: Ord> Stack<T>`).
                                sig = Some(s.clone());
                            } else {
                                let own: Vec<(String, Option<Sig>)> = match found {
                                    Some(MethodRef::Trait(t, s)) => vec![(t, Some(s.subst(&map)))],
                                    Some(MethodRef::Ambiguous(ts)) => ts.into_iter().map(|t| (t, None)).collect(),
                                    _ => Vec::new(),
                                };
                                let mut ts = by_trait(&reg, own);
                                match ts.len() {
                                    0 => {
                                        if !ci.fields.iter().any(|(n, _, _)| n == mname) {
                                            return Err(Error::new("AttributeError", format!("`{}` has no method `{mname}`", recv_ty.name()), span));
                                        }
                                    }
                                    1 => {
                                        let (t, s) = ts.pop().unwrap();
                                        sig = s;
                                        qual = Some(t);
                                    }
                                    _ => {
                                        let mut names: Vec<String> = ts.into_iter().map(|(t, _)| t).collect();
                                        names.sort();
                                        return Err(ambiguous(c, &names));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ty::Trait(t, targs) => {
                if let Some(tm) = self.m.trait_methods(t, targs) {
                    match tm.get(mname) {
                        Some(s) => {
                            sig = Some(s.clone());
                            qual = Some(t.clone());
                        }
                        None => return Err(Error::new("AttributeError", format!("trait `{t}` has no method `{mname}`"), span)),
                    }
                }
            }
            Ty::Param(n) => {
                let mut found = self.param_method(n, mname);
                if let Some(t) = force {
                    found.retain(|(b, _)| b == t);
                }
                match found.as_slice() {
                    [] => {
                        return Err(type_err(
                            format!("type parameter `{n}` has no method `{mname}`; add a bound whose trait provides it (`{n}: SomeTrait`)"),
                            span,
                        ))
                    }
                    [(t, s)] => {
                        sig = s.clone();
                        // Builtin traits' methods are found by name.
                        if s.is_some() {
                            qual = Some(t.clone());
                        }
                    }
                    many => {
                        let ts: Vec<String> = many.iter().map(|(t, _)| t.clone()).collect();
                        return Err(ambiguous(n, &ts));
                    }
                }
            }
            // Builtin types: their own methods, else trait methods of
            // registered impls (`impl Describe for int`).
            t => {
                if builtin_table(t).is_some_and(|tb| !lizzard_vm::builtins::has_type_method(tb, mname)) {
                    let reg = self.m.registry_methods(t, mname, &|n| self.pbounds(n));
                    let mut ts = by_trait(&reg, Vec::new());
                    if let Some(f) = force {
                        ts.retain(|(x, _)| x == f);
                        if ts.is_empty() && t.is_concrete() {
                            return Err(type_err(format!("`{}` has no method `{f}::{mname}`", t.name()), span));
                        }
                    }
                    match ts.len() {
                        0 => {}
                        1 => {
                            let (tr, s) = ts.pop().unwrap();
                            sig = s;
                            qual = Some(tr);
                        }
                        _ => {
                            let mut names: Vec<String> = ts.into_iter().map(|(t, _)| t).collect();
                            names.sort();
                            return Err(ambiguous(&t.name(), &names));
                        }
                    }
                }
            }
        }
        if qual.as_ref().is_some_and(|t| !self.m.gindex.contains_key(t)) {
            qual = None;
        }
        if let Some(s) = sig.clone().filter(|s| !s.tparams.is_empty()) {
            // A generic method: infer its own type arguments.
            if s.params.len() != args.len() {
                return Err(type_err(
                    format!("`{mname}()` takes {} argument{} but {} {} given", s.params.len(), if s.params.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "was" } else { "were" }),
                    span,
                ));
            }
            let what = format!("{mname}()");
            let (mut al, targs, ret) = self.generic_call_args(&s, None, expect.as_ref(), args, r.root, &what, span)?;
            if let Some(regs) = targs.iter().cloned().collect::<Option<Vec<Ty>>>().and_then(|k| self.make_types(&k)) {
                let mut full: Vec<Arg> = regs.into_iter().map(|r| Arg::new(r, false, false, None)).collect();
                full.append(&mut al.args);
                al.args = full;
            }
            let dst = self.tmp_owned();
            let a = self.arglist(al.clone());
            self.emit_method_call(dst, r.reg, mname, a, qual.as_deref());
            self.after_call(&al);
            if r.own == Own::Owned {
                self.emit(Instr::Drop { reg: r.reg });
            }
            self.reg_ty.clear();
            return Ok(Res { reg: dst, own: Own::Owned, ty: ret.as_value(), root: None, desc: "value".into() });
        }
        if let Some(s) = &sig {
            if s.params.len() != args.len() {
                return Err(type_err(
                    format!("`{mname}()` takes {} argument{} but {} {} given", s.params.len(), if s.params.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "was" } else { "were" }),
                    span,
                ));
            }
        }
        let builtin_mask = match &r.ty {
            Ty::Str | Ty::List(_) | Ty::Dict(..) | Ty::Tuple(_) | Ty::Iter(_) | Ty::Chan(_) | Ty::Array(_) => {
                lizzard_vm::builtins::method_keep_mask(mname)
            }
            _ => None,
        };
        let any_mask = if mname == "some_or" { Some(1u64 << 1) } else { None };
        let keep_of = |i: usize| -> Option<bool> {
            if let Some(s) = &sig {
                return s.params.get(i).map(|p| p.keep);
            }
            if let Some(m) = builtin_mask.or(any_mask) {
                return Some(m & (1 << (i + 1)) != 0);
            }
            None
        };
        let elem_arg = match (mname, &r.ty) {
            ("append", Ty::List(t)) => Some((0, (**t).clone())),
            ("insert", Ty::List(t)) => Some((1, (**t).clone())),
            ("send", Ty::Chan(t)) => Some((0, (**t).clone())),
            _ => None,
        };
        let arg_ty = |i: usize| -> Option<(String, Ty)> {
            if let Some(s) = &sig {
                return s.params.get(i).map(|p| (p.name.clone(), p.ty.clone()));
            }
            elem_arg.as_ref().filter(|(k, _)| *k == i).map(|(_, t)| (String::new(), t.clone()))
        };
        let mut al = self.call_args(args, &keep_of, &arg_ty, r.root, span)?;
        if let (true, Some(s)) = (plain, &sig) {
            al.fast = s.tparams.is_empty()
                && s.params.iter().zip(&self.last_arg_tys).all(|(p, t)| !p.keep && (p.ty == Ty::Unknown || self.proves(&p.ty, t)));
        }
        if let ("append" | "insert", Ty::List(_)) = (mname, &r.ty) {
            if let Some(arg) = al.args.last() {
                let ty = self.last_arg_tys.last().cloned().unwrap_or(Ty::Unknown);
                let v = Res { reg: arg.reg, own: Own::Owned, ty, root: None, desc: "value".into() };
                let none = Res { reg: arg.reg, own: Own::Owned, ty: Ty::Unknown, root: None, desc: String::new() };
                self.check_elem_store(&r.ty, &none, &v, span)?;
            }
        }
        let dst = self.tmp_owned();
        let a = self.arglist(al.clone());
        self.emit_method_call(dst, r.reg, mname, a, qual.as_deref());
        self.after_call(&al);
        if r.own == Own::Owned {
            self.emit(Instr::Drop { reg: r.reg });
        }
        let ty = match (&sig, mname, &r.ty) {
            (Some(s), _, _) => s.ret.as_value(),
            (None, "upper" | "lower" | "strip" | "lstrip" | "rstrip" | "replace" | "join", Ty::Str) => Ty::Str,
            (None, "split", Ty::Str) => Ty::list(Ty::Str),
            (None, "append" | "insert" | "extend" | "clear" | "reverse" | "sort", Ty::List(_)) => Ty::Unit,
            (None, "clear", Ty::Dict(..)) => Ty::Unit,
            (None, "keys", Ty::Dict(k, _)) => Ty::List(k.clone()),
            (None, "values", Ty::Dict(_, v)) => Ty::List(v.clone()),
            (None, "items", Ty::Dict(k, v)) => Ty::list(Ty::Tuple(Some(vec![(**k).clone(), (**v).clone()]))),
            (None, "pop" | "first" | "last", Ty::List(t)) if **t != Ty::Unknown && !matches!(**t, Ty::Option(_)) => {
                Ty::Option(t.clone())
            }
            (None, "copy", Ty::List(_) | Ty::Dict(..)) => r.ty.clone(),
            (None, "copy" | "view", Ty::Array(_)) => r.ty.clone(),
            (None, "to_list", Ty::Array(e)) => Ty::list(e.map(elem_ty).unwrap_or(Ty::Unknown)),
            (None, "some_or", Ty::Option(inner)) => (**inner).clone(),
            _ => Ty::Unknown,
        };
        self.reg_ty.clear();
        Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() })
    }

    // ------------------------------------------------------------ return / raise

    fn return_stmt(&mut self, v: Option<&Expr>, span: Span) -> R<()> {
        if self.is_module {
            return Err(syn_err("`return` outside function", span));
        }
        let (r, ty) = match v {
            Some(e) if self.is_generator => {
                let r = self.expr_owned(e, "the return value")?;
                (r.reg, r.ty)
            }
            Some(e) => {
                let dst = self.tmp_owned();
                self.expect = (self.ret_ty != Ty::Unknown).then(|| self.ret_ty.clone());
                let ty = self.value_into(e, dst)?;
                (dst, ty)
            }
            None => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::UNIT);
                (dst, Ty::Unit)
            }
        };
        self.emit_return(r, &ty);
        Ok(())
    }

    /// Whether every value of static type `got` passes the runtime check of
    /// the declared type `want` (so the check can be left out).
    fn proves(&self, want: &Ty, got: &Ty) -> bool {
        match (want, got) {
            (Ty::Int, Ty::Int) | (Ty::Float, Ty::Float) | (Ty::Bool, Ty::Bool) | (Ty::Str, Ty::Str) | (Ty::BigInt, Ty::BigInt) | (Ty::Unit, Ty::Unit) => {
                true
            }
            (Ty::Instance(a, x), Ty::Instance(b, y)) => {
                a == b && x.is_empty() && y.is_empty() && self.m.classes.get(a).is_some_and(|c| c.tparams.is_empty() && !c.native)
            }
            _ => false,
        }
    }

    /// Returns `r`, whose static type is `ty`.
    fn emit_return(&mut self, r: Reg, ty: &Ty) {
        let finallys: Vec<Block> = self.finally_stack.iter().rev().cloned().collect();
        let saved = std::mem::take(&mut self.finally_stack);
        for f in &finallys {
            let _ = self.block(f);
        }
        self.finally_stack = saved;
        if !self.is_generator {
            match self.ret.clone() {
                RetKind::Typed(_) if self.proves(&self.ret_ty, ty) => {}
                RetKind::Unit if *ty == Ty::Unit => {}
                RetKind::Typed(d) => {
                    let ti = self.type_idx(d);
                    self.emit(Instr::CheckType { reg: r, ty: ti, what: self.ret_desc });
                }
                RetKind::Unit => {
                    let ti = self.type_idx(TypeDesc::prim(Ctor::Unit));
                    self.emit(Instr::CheckType { reg: r, ty: ti, what: self.ret_desc });
                }
                RetKind::Any => {}
            }
        }
        self.emit(Instr::Return { src: r });
        self.state = None;
    }

    fn raise_stmt(&mut self, exc: Option<&Expr>, cause: Option<&Expr>, span: Span) -> R<()> {
        match exc {
            None => {
                let Some(&r) = self.handler_exc.last() else {
                    return Err(syn_err("bare `raise` is only allowed inside an `except` block", span));
                };
                self.emit(Instr::Reraise { exc: r });
            }
            Some(e) => {
                let r = self.expr_owned(e, "`raise`")?;
                let c = match cause {
                    Some(c) => self.expr_owned(c, "`raise ... from`")?.reg,
                    None => Reg::MAX,
                };
                self.emit(Instr::Raise { exc: r.reg, cause: c });
            }
        }
        self.state = None;
        Ok(())
    }

    // ------------------------------------------------------------ try

    fn try_expr(&mut self, t: &TryExpr, dst: Option<Reg>) -> R<Ty> {
        let dst = match dst {
            Some(d) => d,
            None => self.tmp_owned(),
        };
        let entry = self.state.clone();
        let log_start = self.moved_log.len();
        if let Some(f) = &t.finally {
            self.finally_stack.push(f.clone());
        }
        let start = self.here();
        // Temporaries from here on are only live inside the body; one that
        // owns a value when the body raises is dropped by the handler.
        let pool_from = self.owned_used;
        let mut ty = self.block_value(&t.body, dst)?;
        let body_end = self.here();
        let mut end_jumps = Vec::new();
        // `else` runs only when the body did not raise (outside the handler range).
        let mut normal_state = self.state.take();
        let else_jump = self.jump();
        let handler_pc = self.here();
        // Handlers see every variable moved in the body as possibly moved.
        let mut hstate = join_opt(entry.clone(), normal_state.clone());
        if let Some(s) = &mut hstate {
            for n in &self.moved_log[log_start..] {
                if let Some(v) = s.get_mut(n) {
                    if v.st == St::Live {
                        v.st = St::DynMoved;
                    }
                }
            }
        }
        let mut handlers_state: Option<State> = None;
        let fin_start;
        if !t.handlers.is_empty() {
            self.exc_table.push(ExcEntry { start, end: body_end, handler: handler_pc });
            self.state = hstate.clone();
            let e = self.new_reg(true, Some("<exception>".into()));
            self.emit(Instr::TakeException { dst: e });
            self.drop_temps_from(pool_from);
            let mut matches = Vec::new();
            let mut catch_all = None;
            for (i, h) in t.handlers.iter().enumerate() {
                if h.types.is_empty() {
                    if i != t.handlers.len() - 1 {
                        return Err(syn_err("a bare `except:` must be the last handler", h.span));
                    }
                    catch_all = Some(i);
                    continue;
                }
                for ty in &h.types {
                    let (_, d) = self.resolve_type(ty)?;
                    let ti = self.type_idx(d);
                    matches.push((i, self.emit(Instr::MatchException { exc: e, ty: ti, target: u32::MAX })));
                }
            }
            let mut targets = vec![u32::MAX; t.handlers.len()];
            match catch_all {
                Some(i) => {
                    let j = self.jump();
                    matches.push((i, j));
                }
                None => {
                    self.emit(Instr::Reraise { exc: e });
                }
            }
            for (i, h) in t.handlers.iter().enumerate() {
                self.state = hstate.clone();
                targets[i] = self.here();
                let exc_reg = match &h.name {
                    Some(n) => {
                        let ety = match h.types.as_slice() {
                            [TypeExpr::Named(n, _, _)] if self.m.classes.contains_key(n) || self.m.builtin_classes.contains(n) => Ty::Instance(n.clone(), Vec::new()),
                            _ => Ty::Unknown,
                        };
                        let t = self.tmp_owned();
                        self.emit(Instr::Dup { dst: t, src: e });
                        self.bind(n, Res { reg: t, own: Own::Owned, ty: ety, root: None, desc: "value".into() }, h.span)?;
                        self.owned_used -= 1;
                        e
                    }
                    None => e,
                };
                self.handler_exc.push(exc_reg);
                let hty = self.block_value(&h.body, dst);
                self.handler_exc.pop();
                let hty = hty?;
                if self.state.is_some() {
                    ty = ty.join(&hty);
                    if let Some(n) = &h.name {
                        if let Some(l) = self.locals.get(n) {
                            let reg = l.reg;
                            self.emit(Instr::Drop { reg });
                            self.set_st(n, St::Uninit);
                        }
                    }
                    self.emit(Instr::Drop { reg: e });
                }
                handlers_state = join_opt(handlers_state, self.state.take());
                end_jumps.push(self.jump());
            }
            for (i, at) in matches {
                self.patch(at, targets[i]);
            }
        }
        fin_start = self.here();
        let h = self.here();
        self.patch(else_jump, h);
        self.state = normal_state.take();
        if let Some(b) = &t.else_ {
            ty = self.block_value(b, dst)?;
        }
        let after_else = self.state.take();
        let finally_block = if t.finally.is_some() { self.finally_stack.pop() } else { None };
        let normal_end = self.here();
        let h = self.here();
        for at in end_jumps.drain(..) {
            self.patch(at, h);
        }
        self.state = join_opt(after_else, handlers_state);
        if let Some(f) = &finally_block {
            // Normal path: run the finally body inline.
            self.block(f)?;
            let skip = self.jump();
            // Exceptional path: any exception from the body, handlers or else.
            let fin_handler = self.here();
            self.exc_table.push(ExcEntry { start, end: normal_end, handler: fin_handler });
            let saved = self.state.clone();
            let mut st = join_opt(entry, saved.clone());
            if let Some(s) = &mut st {
                for n in &self.moved_log[log_start..] {
                    if let Some(v) = s.get_mut(n) {
                        if v.st == St::Live {
                            v.st = St::DynMoved;
                        }
                    }
                }
            }
            self.state = st;
            let e2 = self.new_reg(true, Some("<exception>".into()));
            self.emit(Instr::TakeException { dst: e2 });
            self.drop_temps_from(pool_from);
            self.block(f)?;
            self.emit(Instr::Reraise { exc: e2 });
            let h = self.here();
            self.patch(skip, h);
            self.state = saved;
        }
        let _ = fin_start;
        Ok(ty)
    }

    /// Drops the owned temporaries `pool[from..]` at an exception handler.
    /// A temporary that is not live holds `UNINIT` or `MOVED`, so this only
    /// releases values that were in flight when the exception was raised.
    fn drop_temps_from(&mut self, from: usize) {
        for r in self.owned_pool[from..].to_vec() {
            self.emit(Instr::Drop { reg: r });
        }
    }

    // ------------------------------------------------------------ definitions

    fn def_stmt(&mut self, f: &FuncDef) -> R<()> {
        let dst = self.tmp_owned();
        if self.is_module {
            let code = compile_function(self.m, f, Owner::None, &[], &[])?;
            self.codes.push(Arc::new(code));
            let ci = (self.codes.len() - 1) as u32;
            let al = self.arglist(ArgList::default());
            self.emit(Instr::MakeFunction { dst, code: ci, captures: al });
            let idx = self.m.gindex[&f.name];
            self.emit(Instr::StoreGlobal { idx, src: dst });
            self.set_var(&f.name, St::Live, Ty::StaticFn);
            return Ok(());
        }
        let ty = self.make_closure(f, dst)?;
        let r = Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() };
        self.bind(&f.name, r, f.span)
    }

    /// A generator declared `-> Iterator<T> + Send` may be resumed on another
    /// thread, so it cannot hold a value that may not be `Send` (a live
    /// variable or a capture) across a `yield`.
    fn check_held(&self, span: Span) -> R<()> {
        if self.gen_local {
            return Ok(());
        }
        let live = |st: &St| matches!(st, St::Live | St::MaybeMoved | St::DynMoved | St::MaybeUninit);
        let mut held: Vec<(&String, &Ty)> = self
            .state
            .iter()
            .flat_map(|s| s.iter())
            .filter(|(_, v)| live(&v.st) && self.may_be_local(&v.ty))
            .map(|(n, v)| (n, &v.ty))
            .chain(self.captures.iter().filter(|(_, t)| self.may_be_local(t)).map(|(n, t)| (n, t)))
            .collect();
        held.sort_by(|a, b| a.0.cmp(b.0));
        let Some((n, t)) = held.first() else { return Ok(()) };
        Err(type_err(
            format!(
                "generator `{}` is declared `Send` but holds `{}` (`{n}`), which is not `Send`, across a `yield`; declare it `-> {}`",
                self.name,
                t.name(),
                self.ret_ty.name()
            ),
            span,
        ))
    }

    /// `function + Send`, or `function` for a closure or bound method that
    /// holds a value that may not be `Send`.
    fn func_ty(&self, local: bool) -> Ty {
        if local {
            Ty::Local(Box::new(Ty::Func))
        } else {
            Ty::Func
        }
    }

    /// Compiles a nested function or lambda; captured locals move into it.
    /// Returns its type (`function` if it captures a value that may not be
    /// `Send`, else `function + Send`).
    fn make_closure(&mut self, f: &FuncDef, dst: Reg) -> R<Ty> {
        let body = f.body.as_ref().unwrap();
        let mut used = HashSet::new();
        scan::used_names(body, &mut used);
        let mut own_names: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
        scan::assigned_names(body, &mut own_names);
        let mut caps: Vec<(String, Ty)> = Vec::new();
        let mut cap_args = Vec::new();
        let mut names: Vec<&String> = used.iter().collect();
        names.sort();
        for n in names {
            if own_names.contains(n) {
                continue;
            }
            if let Some(l) = self.locals.get(n) {
                let (reg, kind) = (l.reg, l.kind);
                self.check_use(n, Some(reg), f.span)?;
                let ty = self.st(n).map(|v| v.ty.clone()).unwrap_or(Ty::Unknown);
                let r = Res {
                    reg,
                    own: if kind == LocalKind::Borrowed { Own::Borrowed } else { Own::Local(n.clone()) },
                    ty: ty.clone(),
                    root: Some(reg),
                    desc: format!("`{n}`"),
                };
                let o = self.to_owned(r, f.span, "a closure")?;
                cap_args.push(Arg::new(o.reg, true, false, None));
                caps.push((n.clone(), ty));
            } else if let Some((_, ty)) = self.captures.iter().find(|(c, _)| c == n).cloned() {
                let r = self.name_expr(n, f.span)?;
                let o = self.to_owned(r, f.span, "a closure")?;
                cap_args.push(Arg::new(o.reg, true, false, None));
                caps.push((n.clone(), ty));
            }
        }
        // Type parameters in scope are captured (type values are immortal).
        let mut outer = Vec::new();
        for tp in self.tscope.clone() {
            let o = self.tmp_owned();
            self.emit(Instr::Dup { dst: o, src: tp.reg });
            cap_args.push(Arg::new(o, true, false, None));
            caps.push((format!("<{}>", tp.name), Ty::Unknown));
            outer.push(TParamSig { name: tp.name.clone(), bounds: tp.bounds.clone() });
        }
        let local = caps.iter().any(|(_, t)| self.may_be_local(t));
        let code = compile_function(self.m, f, Owner::None, &caps, &outer)?;
        self.codes.push(Arc::new(code));
        let ci = (self.codes.len() - 1) as u32;
        let al = self.arglist(ArgList { args: cap_args, pins: false, fast: false });
        self.emit(Instr::MakeFunction { dst, code: ci, captures: al });
        Ok(self.func_ty(local))
    }

    /// Items are order-independent, as in Rust: traits, then structs (with
    /// the methods of their `impl` blocks), then associated functions and
    /// registered impls are defined before the module's statements run.
    fn hoist_items(&mut self, module: &Module) -> R<()> {
        for s in &module.body.stmts {
            if let StmtKind::Trait(t) = &s.kind {
                self.trait_stmt(t)?;
            }
        }
        for s in &module.body.stmts {
            if let StmtKind::Struct(c) = &s.kind {
                self.struct_stmt(c)?;
            }
        }
        for f in &self.m.assoc_fns {
            self.def_stmt(f)?;
        }
        let m = self.m;
        for im in &m.impls {
            let mut methods = Vec::new();
            for d in &im.defs {
                let code = compile_function(m, d, Owner::Impl(im), &[], &[])?;
                self.codes.push(Arc::new(code));
                methods.push((d.name.clone(), (self.codes.len() - 1) as u32));
            }
            self.impls.push(ImplSpec {
                name: im.shown.clone(),
                target: im.target_desc.clone(),
                nparams: im.tparams.len(),
                bounds: im.bound_descs.clone(),
                tr: im.tr.as_ref().map(|(t, _)| m.gindex[t]),
                trait_args: im.trait_descs.clone(),
                methods,
            });
            self.emit(Instr::MakeImpl { spec: (self.impls.len() - 1) as u32 });
        }
        Ok(())
    }

    /// A struct, trait or impl statement: already defined at module start;
    /// an error elsewhere.
    fn item_here(&self, what: &str, span: Span) -> R<()> {
        if self.is_module {
            return Ok(());
        }
        Err(syn_err(format!("{what} must be defined at module level"), span))
    }

    fn struct_stmt(&mut self, c: &StructDef) -> R<()> {
        let info = &self.m.classes[&c.name];
        let pb = |n: &str| info.tparams.iter().find(|t| t.name == n).map(|t| t.bounds.clone()).unwrap_or_default();
        for ((_, ty, _), f) in info.fields.iter().zip(&c.fields) {
            self.m.check_wf(ty, &pb, f.span)?;
        }
        for (_, args) in &info.impls {
            args.iter().try_for_each(|a| self.m.check_wf(a, &pb, c.span))?;
        }
        let mut methods = Vec::new();
        let mut trait_methods = Vec::new();
        for im in self.m.impl_methods.get(&c.name).into_iter().flatten() {
            let code = compile_function(self.m, &im.def, Owner::Class(&c.name, &im.tparams), &[], &[])?;
            self.codes.push(Arc::new(code));
            let ci = (self.codes.len() - 1) as u32;
            match &im.tr {
                Some(t) => trait_methods.push((self.m.gindex[t], im.def.name.clone(), ci)),
                None => methods.push((im.def.name.clone(), ci)),
            }
        }
        let fields = info.fields.iter().map(|(n, _, d)| FieldSpec { name: n.clone(), ty: d.clone() }).collect();
        let traits = info.trait_descs.iter().map(|(t, ds)| (self.m.gindex[t], t.clone(), ds.clone())).collect();
        let tparams = c.tparams.iter().map(|t| t.name.clone()).collect();
        let local = info.local;
        self.classes.push(ClassSpec { name: c.name.clone(), tparams, fields, methods, trait_methods, traits, local });
        let dst = self.tmp_owned();
        self.emit(Instr::MakeClass { dst, spec: (self.classes.len() - 1) as u32 });
        let idx = self.m.gindex[&c.name];
        self.emit(Instr::StoreGlobal { idx, src: dst });
        self.set_var(&c.name, St::Live, Ty::Class(c.name.clone()));
        Ok(())
    }

    fn trait_stmt(&mut self, t: &TraitDef) -> R<()> {
        if !self.is_module {
            return Err(syn_err("traits must be defined at module level", t.span));
        }
        let mut methods = Vec::new();
        for m in &t.methods {
            let default = match &m.body {
                Some(_) => {
                    let code = compile_function(self.m, m, Owner::Trait(&t.name), &[], &[])?;
                    self.codes.push(Arc::new(code));
                    Some((self.codes.len() - 1) as u32)
                }
                None => None,
            };
            methods.push((m.name.clone(), default));
        }
        self.traits.push(TraitSpec { name: t.name.clone(), nparams: t.tparams.len(), methods });
        let dst = self.tmp_owned();
        self.emit(Instr::MakeTrait { dst, spec: (self.traits.len() - 1) as u32 });
        let idx = self.m.gindex[&t.name];
        self.emit(Instr::StoreGlobal { idx, src: dst });
        self.set_var(&t.name, St::Live, Ty::Unknown);
        Ok(())
    }
}

/// Whether a value of static type `v` may be stored in a slot declared `d`.
fn join_all(tys: Vec<Ty>) -> Ty {
    let mut it = tys.into_iter();
    match it.next() {
        Some(first) => it.fold(first, |a, b| a.join(&b)),
        None => Ty::Unknown,
    }
}

fn ty_compatible(d: &Ty, v: &Ty) -> bool {
    // `X` holds an `X + Send`; whether a value may not be `Send` is checked by
    // `FnCtx::compatible`.
    if matches!(d, Ty::Local(_)) || matches!(v, Ty::Local(_)) {
        return ty_compatible(d.peel(), v.peel());
    }
    // An `Unknown` element type is "some container" (or not inferred): the
    // runtime check decides. Known element types must match exactly.
    let inv = |a: &Ty, b: &Ty| a == b || *a == Ty::Unknown || *b == Ty::Unknown;
    match (d, v) {
        (_, Ty::NoneT) => matches!(d, Ty::Option(_) | Ty::Unknown),
        (Ty::Unknown | Ty::Any, _) | (_, Ty::Unknown | Ty::Any) => true,
        (Ty::List(a), Ty::List(b)) | (Ty::Iter(a), Ty::Iter(b)) | (Ty::Chan(a), Ty::Chan(b)) => inv(a, b),
        (Ty::Dict(k1, v1), Ty::Dict(k2, v2)) => inv(k1, k2) && inv(v1, v2),
        (Ty::Tuple(None), Ty::Tuple(_)) | (Ty::Tuple(_), Ty::Tuple(None)) => true,
        (Ty::Tuple(Some(a)), Ty::Tuple(Some(b))) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| ty_compatible(x, y)),
        (Ty::Array(None), Ty::Array(_)) | (Ty::Array(_), Ty::Array(None)) => true,
        (Ty::Option(a), Ty::Option(b)) => ty_compatible(a, b),
        (Ty::Option(a), b) => ty_compatible(a, b),
        (Ty::Float, Ty::Int) => true,
        (Ty::BigInt, Ty::Int) => true,
        // Trait types are checked by `FnCtx::compatible` (and at runtime).
        (Ty::Trait(..), _) => true,
        // A user class implementing `Iterator` (checked at runtime).
        (Ty::Iter(_), Ty::Instance(..) | Ty::Trait(..)) => true,
        (Ty::Instance(c1, a), Ty::Instance(c2, b)) => {
            c1 == c2 && (a.is_empty() || b.is_empty() || (a.len() == b.len() && a.iter().zip(b).all(|(x, y)| inv(x, y))))
        }
        (Ty::Func, Ty::StaticFn | Ty::Class(_) | Ty::TypeVal(_)) | (Ty::StaticFn, Ty::Func) => true,
        (a, b) => a == b,
    }
}

fn param_field_msg(n: &str, field: &str) -> String {
    format!("`.{field}` on a value of type parameter `{n}`: its fields are not known (only methods of its bounds can be used)")
}

fn short(desc: &str) -> String {
    // "field `x`" -> "obj.x"-style hint; "parameter `p`" -> "p"
    match desc.split('`').nth(1) {
        Some(n) if desc.starts_with("field") => format!("obj.{n}"),
        Some(n) => n.to_string(),
        None => "value".into(),
    }
}

fn callee_name(f: &Expr) -> String {
    match &f.kind {
        ExprKind::Name(n) => format!("{n}()"),
        _ => "function".into(),
    }
}

/// An operand of float arithmetic (see `FnCtx::arith`).
enum Arith<'e> {
    Val(Res),
    /// A register holding raw f64 bits (non-owned, never used as a value).
    Raw(Reg),
    Lit(&'e Expr),
}

impl Arith<'_> {
    fn is_raw(&self) -> bool {
        matches!(self, Arith::Raw(_))
    }
}

fn float_op(op: BinOp) -> Option<FOp> {
    match op {
        BinOp::Add => Some(FOp::Add),
        BinOp::Sub => Some(FOp::Sub),
        BinOp::Mul => Some(FOp::Mul),
        BinOp::Div => Some(FOp::Div),
        _ => None,
    }
}

fn binop_ty(op: BinOp, a: &Ty, b: &Ty) -> Ty {
    use Ty::*;
    match (op, a, b) {
        (BinOp::Div, Int | Float, Int | Float) => Float,
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow, Int, Int) => Int,
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow, Float, Int | Float)
        | (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow, Int, Float) => Float,
        (BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr, Int, Int) => Int,
        (_, BigInt, Int | BigInt) | (_, Int, BigInt) => BigInt,
        (BinOp::Add, Str, Str) | (BinOp::Mul, Str, Int) => Str,
        // The result has the left operand's type.
        (BinOp::Add, List(x), List(_)) => List(x.clone()),
        (BinOp::Mul, List(x), Int) | (BinOp::Mul, Int, List(x)) => List(x.clone()),
        (BinOp::Add, Tuple(Some(x)), Tuple(Some(y))) => Tuple(Some(x.iter().chain(y).cloned().collect())),
        (BinOp::Add, Tuple(_), Tuple(_)) => Tuple(None),
        _ => Unknown,
    }
}

/// What a function is defined in.
#[derive(Clone, Copy)]
pub enum Owner<'a> {
    None,
    /// A method of a struct, with the impl's names for its type parameters.
    Class(&'a str, &'a [String]),
    /// A default method of a trait.
    Trait(&'a str),
    /// A method of a registered impl (`impl Describe for int`).
    Impl(&'a ImplInfo),
}

/// Compiles a function, method or closure (`captures`). `outer`: the
/// enclosing type parameters of a closure (captured as `<T>`).
pub fn compile_function(m: &ModInfo, f: &FuncDef, owner: Owner, captures: &[(String, Ty)], outer: &[TParamSig]) -> R<Code> {
    let class_name = match owner {
        Owner::Class(c, _) => Some(c),
        _ => None,
    };
    let name = match owner {
        Owner::Class(c, _) | Owner::Trait(c) => format!("{c}.{}", f.name),
        Owner::Impl(im) => format!("{}.{}", im.target.name(), f.name),
        Owner::None => f.name.clone(),
    };
    let in_impl = matches!(owner, Owner::Impl(_));
    let mut ctx = FnCtx::new(m, name, false);
    ctx.line = f.span.line;
    ctx.class_name = class_name.map(String::from);
    ctx.captures = captures.to_vec();
    let body = f.body.as_ref().unwrap();
    ctx.is_generator = scan::has_yield(body);
    let mut state = State::new();
    let mut params = Vec::new();
    let mut checks = Vec::new();
    let has_self = f.params.first().is_some_and(|p| p.name == "self") && (class_name.is_some() || in_impl || ctx_is_trait_method(f));
    // Registers: `self`, the hidden type parameters, then the parameters.
    let mut preg = Vec::new();
    if has_self {
        preg.push(ctx.new_reg(f.params[0].keep, Some("self".into())));
    }
    let own_tregs: Vec<Reg> = f.tparams.iter().map(|t| ctx.new_reg(false, Some(format!("<{}>", t.name)))).collect();
    for p in &f.params[has_self as usize..] {
        preg.push(ctx.new_reg(p.keep, Some(p.name.clone())));
    }
    // Type parameters of the enclosing class/trait or closure scope.
    let owner_tps: Vec<TParamSig> = match owner {
        Owner::Class(c, names) => m.classes[c].tparams.iter().zip(names).map(|(t, n)| TParamSig { name: n.clone(), bounds: t.bounds.clone() }).collect(),
        Owner::Trait(t) => m.trait_tparams(t),
        Owner::Impl(im) => im.tparams.clone(),
        Owner::None => outer.to_vec(),
    };
    for (k, tp) in owner_tps.iter().enumerate() {
        if f.tparams.iter().any(|t| t.name == tp.name) {
            return Err(type_err(format!("type parameter `{}` shadows the enclosing one", tp.name), f.span));
        }
        let reg = ctx.new_reg(false, Some(format!("<{}>", tp.name)));
        match owner {
            Owner::Class(..) => {
                ctx.emit(Instr::LoadTypeArg { dst: reg, obj: 0, k: k as u32 });
            }
            Owner::Trait(t) => {
                let tr = m.gindex[t];
                ctx.emit(Instr::LoadTraitArg { dst: reg, obj: 0, tr, k: k as u32 });
            }
            Owner::Impl(im) => {
                let ty = ctx.type_idx(im.target_desc.clone());
                ctx.emit(Instr::LoadImplArg { dst: reg, obj: 0, ty, k: k as u32 });
            }
            Owner::None => {
                let idx = captures.iter().position(|(n, _)| *n == format!("<{}>", tp.name)).expect("captured type parameter") as u32;
                ctx.emit(Instr::LoadCapture { dst: reg, idx });
            }
        }
        ctx.tscope.push(TPar { name: tp.name.clone(), reg, bounds: tp.bounds.clone() });
    }
    // `Self` in a trait's default method: the type of `self`.
    if let (Owner::Trait(t), true) = (owner, has_self) {
        let reg = ctx.new_reg(false, Some("<Self>".into()));
        ctx.emit(Instr::LoadTypeOf { dst: reg, obj: 0 });
        let bound = Ty::Trait(t.to_string(), owner_tps.iter().map(|p| Ty::Param(p.name.clone())).collect());
        ctx.tscope.push(TPar { name: "Self".into(), reg, bounds: vec![bound] });
    }
    for (tp, reg) in f.tparams.iter().zip(&own_tregs) {
        ctx.tscope.push(TPar { name: tp.name.clone(), reg: *reg, bounds: Vec::new() });
    }
    // Bounds of the function's own type parameters (they may mention each other).
    let sigs = m.tparam_sigs(&f.tparams, &ctx.scope())?;
    for (tp, sig) in f.tparams.iter().zip(&sigs) {
        let mut descs = Vec::new();
        for b in &tp.bounds {
            descs.push(match ctx.resolve_type(b)?.1 {
                TypeDesc::Local(d) => *d,
                d => d,
            });
        }
        ctx.tscope.iter_mut().find(|t| t.name == tp.name).unwrap().bounds = sig.bounds.clone();
        ctx.code_tparams.push(TypeParamInfo { name: tp.name.clone(), bounds: descs });
    }
    let self_ty = |own: &[TParamSig]| own.iter().map(|t| Ty::Param(t.name.clone())).collect::<Vec<_>>();
    for (i, p) in f.params.iter().enumerate() {
        let is_self = i == 0 && has_self;
        let (ty, desc) = match (&p.ty, is_self) {
            (_, true) => match owner {
                Owner::Class(c, _) => (Ty::Instance(c.to_string(), self_ty(&owner_tps)), None),
                // Any implementing class, `Send` or not.
                Owner::Trait(t) => (Ty::Local(Box::new(Ty::Trait(t.to_string(), self_ty(&owner_tps)))), None),
                Owner::Impl(im) => (im.target.clone(), None),
                Owner::None => (Ty::Unknown, None),
            },
            (Some(t), _) => {
                let (ty, d) = ctx.resolve_type(t)?;
                (ty, Some(d))
            }
            (None, _) => (Ty::Unknown, None),
        };
        if ctx.is_generator && !p.keep && ty.is_copy() != Some(true) && !is_self {
            return Err(own_err(
                format!("generator `{}` outlives its call: move-type parameter `{}` must be declared `+ keep`", f.name, p.name),
                p.span,
            ));
        }
        let kind = if p.keep { LocalKind::Owned } else { LocalKind::Borrowed };
        let reg = preg[i];
        ctx.locals.insert(p.name.clone(), Local { reg, kind, decl: desc.clone().map(|d| (ty.clone(), d)) });
        state.insert(p.name.clone(), Var { st: St::Live, ty: ty.as_value() });
        if let Some(d) = &desc {
            checks.push((reg, d.clone(), p.name.clone()));
        }
        params.push(ParamInfo { name: p.name.clone(), keep: p.keep, ty: desc });
    }
    // The hidden type parameters follow `self`.
    for (k, tp) in f.tparams.iter().enumerate() {
        params.insert(has_self as usize + k, ParamInfo { name: format!("<{}>", tp.name), keep: false, ty: None });
    }
    let mut local_names = Vec::new();
    scan::assigned_names(body, &mut local_names);
    for n in local_names {
        if ctx.locals.contains_key(&n) {
            continue;
        }
        let reg = ctx.new_reg(true, Some(n.clone()));
        ctx.locals.insert(n, Local { reg, kind: LocalKind::Owned, decl: None });
    }
    ctx.state = Some(state);
    let fname = if f.is_lambda { "lambda".to_string() } else { format!("`{}()`", ctx.name) };
    let first = ctx.here();
    for (reg, d, pname) in checks {
        let ti = ctx.type_idx(d);
        let w = ctx.name_idx(&format!("argument `{pname}` of {fname}"));
        ctx.emit(Instr::CheckType { reg, ty: ti, what: w });
    }
    if first == 0 {
        ctx.entry = ctx.here();
    }
    let (ret, ret_ty) = match &f.ret {
        Some(TypeExpr::Unit(_)) => (RetKind::Unit, Ty::Unit),
        Some(t) => {
            let (ty, d) = ctx.resolve_type(t)?;
            (RetKind::Typed(d), ty)
        }
        None => (RetKind::Any, Ty::Unknown),
    };
    if ctx.is_generator {
        // Unless declared `-> Iterator<T> + Send`, a generator may hold
        // values that are not `Send`.
        ctx.gen_local = ret_ty == Ty::Unknown || ctx.may_be_local(&ret_ty);
        match ret_ty.peel() {
            Ty::Iter(t) => ctx.yield_decl = (**t != Ty::Unknown).then(|| (**t).clone()),
            Ty::Unknown => {}
            Ty::Trait(t, a) if t == "Iterator" => ctx.yield_decl = a.first().cloned().filter(|t| *t != Ty::Unknown),
            other => {
                return Err(type_err(
                    format!("generator `{}` returns an `Iterator<T>`, not `{}`", f.name, other.name()),
                    f.span,
                ))
            }
        }
    }
    ctx.ret = if ctx.is_generator { RetKind::Any } else { ret.clone() };
    ctx.ret_ty = ret_ty;
    ctx.ret_desc = ctx.name_idx(&format!("return value of {fname}"));
    ctx.line = f.span.line;
    let dst = ctx.tmp_owned();
    if ctx.is_generator {
        ctx.block(body)?;
        if ctx.state.is_some() {
            ctx.load_imm(dst, Value::UNIT);
            ctx.emit(Instr::Return { src: dst });
        }
    } else if ret == RetKind::Unit {
        // `-> ()`: the last line's value is discarded.
        ctx.block(body)?;
        if ctx.state.is_some() {
            ctx.load_imm(dst, Value::UNIT);
            ctx.emit(Instr::Return { src: dst });
        }
    } else {
        let expect = (ctx.ret_ty != Ty::Unknown).then(|| ctx.ret_ty.clone());
        let ty = ctx.block_value_expect(body, dst, expect)?;
        if ctx.state.is_some() {
            ctx.emit_return(dst, &ty);
        }
    }
    let is_method = class_name.is_some() || in_impl;
    Ok(ctx.finish(params, ret, ctx_is_gen(body), is_method))
}

fn ctx_is_gen(b: &Block) -> bool {
    scan::has_yield(b)
}

fn ctx_is_trait_method(f: &FuncDef) -> bool {
    f.params.first().is_some_and(|p| p.name == "self" && p.ty.is_none()) && !f.is_lambda
}
