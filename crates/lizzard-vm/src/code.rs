//! Compiled code objects.

use std::fmt;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use crate::ops::Instr;
pub use crate::rtype::Ctor;
use crate::value::Value;

pub type Reg = u16;

/// A type as written in the program, resolved to a runtime type value the
/// first time it is used (`types::code_type`).
#[derive(Clone, Debug, PartialEq)]
pub enum TypeDesc {
    /// A builtin type constructor and its arguments (none for scalars and
    /// for a bare container type, which accepts any container of its kind).
    Builtin(Ctor, Vec<TypeDesc>),
    /// A class or trait bound to a module-level name, with type arguments.
    Global(u32, String, Vec<TypeDesc>),
    /// A type parameter: an index into the environment the type is resolved
    /// in (a register in code; the instance's type arguments in a class).
    Param(u16, String),
    /// An already resolved type value.
    Resolved(Value),
    /// An erased type (`any`, `function`, `Iterator<T>`, a trait) that may
    /// hold values that are not `Send`: what those annotations mean unless
    /// written `X + Send` (the inner type).
    Local(Box<TypeDesc>),
}

impl TypeDesc {
    pub fn prim(c: Ctor) -> TypeDesc {
        TypeDesc::Builtin(c, Vec::new())
    }

    /// `any`: any value, possibly not `Send`.
    pub fn any() -> TypeDesc {
        TypeDesc::Local(Box::new(TypeDesc::prim(Ctor::Any)))
    }

    pub fn option(t: TypeDesc) -> TypeDesc {
        TypeDesc::Builtin(Ctor::Option, vec![t])
    }

    pub fn is(&self, c: Ctor) -> bool {
        matches!(self, TypeDesc::Builtin(x, a) if *x == c && a.is_empty())
    }

    /// Mentions a type parameter (so it depends on the environment).
    pub fn has_params(&self) -> bool {
        match self {
            TypeDesc::Param(..) => true,
            TypeDesc::Resolved(_) => false,
            TypeDesc::Builtin(_, a) | TypeDesc::Global(_, _, a) => a.iter().any(TypeDesc::has_params),
            TypeDesc::Local(t) => t.has_params(),
        }
    }
}

impl fmt::Display for TypeDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeDesc::Param(_, n) => f.write_str(n),
            // Erased types as written: `any`, and `any + Send` unwrapped.
            TypeDesc::Local(t) => match &**t {
                TypeDesc::Builtin(Ctor::Any, a) if a.is_empty() => f.write_str("any"),
                TypeDesc::Builtin(Ctor::Function, a) if a.is_empty() => f.write_str("function"),
                t => write!(f, "{t}"),
            },
            TypeDesc::Resolved(v) => f.write_str(crate::rtype::name(*v)),
            TypeDesc::Global(_, n, args) => {
                f.write_str(n)?;
                if !args.is_empty() {
                    let parts: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                    write!(f, "<{}>", parts.join(", "))?;
                }
                Ok(())
            }
            TypeDesc::Builtin(c, args) => {
                let base = match c {
                    Ctor::Any => "any + Send",
                    Ctor::Int => "int",
                    Ctor::Float => "float",
                    Ctor::Bool => "bool",
                    Ctor::BigInt => "bigint",
                    Ctor::Str => "str",
                    Ctor::Unit => "()",
                    Ctor::Tuple => "tuple",
                    Ctor::List => "list",
                    Ctor::Dict => "dict",
                    Ctor::Option => "Option",
                    Ctor::Array(None) => "array",
                    Ctor::Array(Some(e)) => return write!(f, "array<{}>", e.name()),
                    Ctor::Function => "function + Send",
                    Ctor::Iterator => "Iterator",
                    Ctor::Channel => "Channel",
                    Ctor::Lock => "Lock",
                    Ctor::Thread => "Thread",
                    Ctor::Class(v) => v.as_class().map(|c| c.name.as_str()).unwrap_or("?"),
                    Ctor::Trait(v) => v.as_trait().map(|t| t.name.as_str()).unwrap_or("?"),
                };
                f.write_str(base)?;
                if !args.is_empty() {
                    let parts: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                    write!(f, "<{}>", parts.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParamInfo {
    pub name: String,
    pub keep: bool,
    pub ty: Option<TypeDesc>,
}

/// A type parameter of a generic function: a hidden leading parameter that
/// holds a type value, with the traits it must implement.
#[derive(Clone, Debug)]
pub struct TypeParamInfo {
    pub name: String,
    pub bounds: Vec<TypeDesc>,
}

#[derive(Clone, Copy, Debug)]
pub struct Arg {
    pub reg: Reg,
    /// The caller owns the value in `reg` (a temporary or an owned local).
    pub owned: bool,
    /// `reg` is a named local of the caller (so a move is visible to it).
    pub local: bool,
    /// For borrowed arguments derived from a caller local (`p` in `p.a`), the
    /// local's register: that local may not be moved by the same call.
    pub root: Option<Reg>,
    /// Module variable slot this borrowed argument was loaded from, or u32::MAX.
    pub global: u32,
    /// `reg` holds raw f64 bits for a `float` field (`BuildStruct`).
    pub raw: bool,
}

impl Arg {
    pub fn new(reg: Reg, owned: bool, local: bool, root: Option<Reg>) -> Arg {
        Arg { reg, owned, local, root, global: u32::MAX, raw: false }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArgList {
    pub args: Vec<Arg>,
    /// Some argument is a borrow of a field/element/global: the call pins the
    /// thread's epoch so the borrowed object outlives the call.
    pub pins: bool,
    /// The callee is a known, non-generic lizzard function without `keep`
    /// parameters, and the compiler proved the arguments' types: the call
    /// skips the generic argument handling and the parameter checks.
    pub fast: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ExcEntry {
    pub start: u32,
    pub end: u32,
    pub handler: u32,
}

#[derive(Clone, Debug)]
pub struct FieldSpec {
    pub name: String,
    pub ty: TypeDesc,
}

#[derive(Clone, Debug)]
pub struct ClassSpec {
    pub name: String,
    /// Type parameter names (`Param(k)` in field and trait types).
    pub tparams: Vec<String>,
    pub fields: Vec<FieldSpec>,
    /// Inherent methods: (method name, index into `Code::codes`).
    pub methods: Vec<(String, u32)>,
    /// Methods of trait impls: (trait global slot, method name, code index).
    pub trait_methods: Vec<(u32, String, u32)>,
    /// Implemented traits: global slot, name and type arguments.
    pub traits: Vec<(u32, String, Vec<TypeDesc>)>,
    /// Instances are local (the class is not `Send`): a field's type is not
    /// `Send`.
    pub local: bool,
}

/// An impl that does not cover a whole struct of the program: for a builtin
/// type (`impl Describe for int`), a native class, a specialized struct
/// (`impl Describe for Box<int>`) or with extra bounds (`impl<T: Ord> Ord
/// for Box<T>`). It is registered at runtime and matched against the type of
/// a value.
#[derive(Clone, Debug)]
pub struct ImplSpec {
    /// `impl Describe for int`, for messages.
    pub name: String,
    /// The target type; `Param(k)` is the impl's `k`th type parameter.
    pub target: TypeDesc,
    pub nparams: usize,
    /// Bounds of the impl's type parameters: (parameter, trait type).
    pub bounds: Vec<(u16, TypeDesc)>,
    /// The trait's global slot (`None` for an inherent impl).
    pub tr: Option<u32>,
    pub trait_args: Vec<TypeDesc>,
    /// (method name, index into `Code::codes`)
    pub methods: Vec<(String, u32)>,
}

#[derive(Clone, Debug)]
pub struct TraitSpec {
    pub name: String,
    pub nparams: usize,
    /// (method name, default implementation as index into `Code::codes`)
    pub methods: Vec<(String, Option<u32>)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RetKind {
    /// Untyped: any value.
    Any,
    /// `-> ()`: the implicit last-line value is discarded.
    Unit,
    Typed(TypeDesc),
}

pub struct Code {
    pub name: String,
    pub filename: Arc<str>,
    pub params: Vec<ParamInfo>,
    pub ret: RetKind,
    pub nregs: u16,
    /// Registers dropped when the frame exits (locals, keep params, owned temps).
    pub owned: Vec<bool>,
    pub reg_names: Vec<Option<String>>,
    pub instrs: Vec<Instr>,
    /// First instruction after the parameter checks (where `ArgList::fast`
    /// calls start).
    pub entry: u32,
    pub lines: Vec<u32>,
    pub consts: Vec<Value>,
    pub names: Vec<String>,
    pub arglists: Vec<ArgList>,
    /// `CallTraitMethod` targets: (trait global slot, index into `names`).
    pub trait_calls: Vec<(u32, u32)>,
    pub types: Vec<TypeDesc>,
    /// Runtime type values of `types`, filled in on first use (for types
    /// that do not mention type parameters).
    pub type_cache: Box<[OnceLock<Value>]>,
    /// Inline caches, one per instruction (0 = empty): field accesses keep
    /// `(class id + 1) << 32 | slot`, method calls `(class id + 1) << 48 |
    /// method`. Classes are immutable, so an entry never goes stale.
    pub ics: Box<[AtomicU64]>,
    /// Hidden leading parameters (after `self`) that hold type arguments.
    pub tparams: Vec<TypeParamInfo>,
    /// Item type of a generator function (`-> Iterator<T>`).
    pub yield_ty: Option<u32>,
    pub codes: Vec<Arc<Code>>,
    pub classes: Vec<ClassSpec>,
    pub traits: Vec<TraitSpec>,
    pub impls: Vec<ImplSpec>,
    pub exc_table: Vec<ExcEntry>,
    pub ncaptures: u16,
    pub is_generator: bool,
    pub is_method: bool,
}

impl Code {
    pub fn line_at(&self, pc: u32) -> u32 {
        self.lines.get(pc as usize).copied().unwrap_or(0)
    }

    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        self.disassemble_into(&mut out);
        out
    }

    fn disassemble_into(&self, out: &mut String) {
        use std::fmt::Write;
        let _ = writeln!(
            out,
            "== {} (params={}, regs={}, ret={:?}) ==",
            self.name,
            self.params.len(),
            self.nregs,
            self.ret
        );
        for (i, r) in self.reg_names.iter().enumerate() {
            if let Some(n) = r {
                let _ = writeln!(out, "  ; r{i} = {n}{}", if self.owned[i] { "" } else { " (borrowed)" });
            }
        }
        let mut last_line = 0;
        for (pc, ins) in self.instrs.iter().enumerate() {
            let line = self.lines[pc];
            let l = if line != last_line { format!("{line:>4}") } else { "    ".into() };
            last_line = line;
            let _ = writeln!(out, "{l} {pc:>5}  {}", ins.render(self));
        }
        for e in &self.exc_table {
            let _ = writeln!(out, "  exception: [{}, {}) -> {}", e.start, e.end, e.handler);
        }
        for c in &self.codes {
            out.push('\n');
            c.disassemble_into(out);
        }
    }
}
