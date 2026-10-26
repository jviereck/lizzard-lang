//! Compiled code objects.

use std::fmt;
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
    /// A class or trait bound to a module-level name.
    Global(u32, String),
}

impl TypeDesc {
    pub fn prim(c: Ctor) -> TypeDesc {
        TypeDesc::Builtin(c, Vec::new())
    }

    pub fn any() -> TypeDesc {
        TypeDesc::prim(Ctor::Any)
    }

    pub fn option(t: TypeDesc) -> TypeDesc {
        TypeDesc::Builtin(Ctor::Option, vec![t])
    }

    pub fn is(&self, c: Ctor) -> bool {
        matches!(self, TypeDesc::Builtin(x, a) if *x == c && a.is_empty())
    }
}

impl fmt::Display for TypeDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeDesc::Global(_, n) => f.write_str(n),
            TypeDesc::Builtin(c, args) => {
                let base = match c {
                    Ctor::Any => "any",
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
                    Ctor::Function => "function",
                    Ctor::Iterator => "Iterator",
                    Ctor::Channel => "Channel",
                    Ctor::Lock => "Lock",
                    Ctor::Thread => "Thread",
                    Ctor::Class(_) | Ctor::Trait(_) => "?",
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
}

impl Arg {
    pub fn new(reg: Reg, owned: bool, local: bool, root: Option<Reg>) -> Arg {
        Arg { reg, owned, local, root, global: u32::MAX }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArgList {
    pub args: Vec<Arg>,
    /// Some argument is a borrow of a field/element/global: the call pins the
    /// thread's epoch so the borrowed object outlives the call.
    pub pins: bool,
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
    pub fields: Vec<FieldSpec>,
    /// (method name, index into `Code::codes`)
    pub methods: Vec<(String, u32)>,
    /// Global slots of the implemented traits.
    pub traits: Vec<(u32, String)>,
}

#[derive(Clone, Debug)]
pub struct TraitSpec {
    pub name: String,
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
    pub lines: Vec<u32>,
    pub consts: Vec<Value>,
    pub names: Vec<String>,
    pub arglists: Vec<ArgList>,
    pub types: Vec<TypeDesc>,
    /// Runtime type values of `types`, filled in on first use.
    pub type_cache: Box<[OnceLock<Value>]>,
    /// Item type of a generator function (`-> Iterator<T>`).
    pub yield_ty: Option<u32>,
    pub codes: Vec<Arc<Code>>,
    pub classes: Vec<ClassSpec>,
    pub traits: Vec<TraitSpec>,
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
