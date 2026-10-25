//! Compiled code objects.

use std::fmt;
use std::sync::Arc;

use crate::ops::Instr;
use crate::value::Value;

pub type Reg = u16;

/// Runtime type descriptor used by annotation checks.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeDesc {
    Any,
    Int,
    Float,
    Bool,
    BigInt,
    Str,
    Unit,
    Tuple,
    List,
    Dict,
    Function,
    /// A class or trait bound to a module-level name (resolved at check time).
    Global(u32, String),
    Option(Box<TypeDesc>),
}

impl fmt::Display for TypeDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeDesc::Any => f.write_str("any"),
            TypeDesc::Int => f.write_str("int"),
            TypeDesc::Float => f.write_str("float"),
            TypeDesc::Bool => f.write_str("bool"),
            TypeDesc::BigInt => f.write_str("bigint"),
            TypeDesc::Str => f.write_str("str"),
            TypeDesc::Unit => f.write_str("()"),
            TypeDesc::Tuple => f.write_str("tuple"),
            TypeDesc::List => f.write_str("list"),
            TypeDesc::Dict => f.write_str("dict"),
            TypeDesc::Function => f.write_str("function"),
            TypeDesc::Global(_, n) => f.write_str(n),
            TypeDesc::Option(t) => write!(f, "Option<{t}>"),
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
