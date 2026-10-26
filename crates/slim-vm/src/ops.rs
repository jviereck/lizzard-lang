//! The opcode table. Each opcode is declared once together with its
//! semantics function; the macro generates the instruction enum, the
//! disassembler and the backend-generic dispatcher.

use crate::backend::{Backend, Flow};
use crate::code::{Code, Reg};
use crate::sem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BinKind {
    Div,
    FloorDiv,
    Mod,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum UnKind {
    Neg,
    Pos,
    Invert,
}

/// How a value is stored into a field, element or global.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StoreMode {
    /// `x.f = v`: only copy types; `v` is borrowed and copied.
    Copy,
    /// `x.f <- v`: `v` is owned and moved in.
    Move,
    /// Statically unknown: copy types are copied, others must be owned and are moved.
    Dyn,
}

macro_rules! opcodes {
    ($( $(#[$doc:meta])* $name:ident { $($field:ident : $fty:ty),* } => $sem:path ;)*) => {
        #[derive(Clone, Copy, Debug)]
        pub enum Instr {
            $( $(#[$doc])* $name { $($field: $fty),* } ),*
        }

        impl Instr {
            pub fn name(&self) -> &'static str {
                match self { $( Instr::$name { .. } => stringify!($name) ),* }
            }

            /// Operand list for the disassembler.
            #[allow(unused_mut, unused_variables)]
            pub fn operands(&self) -> Vec<(&'static str, String)> {
                match self {
                    $( Instr::$name { $($field),* } => {
                        let mut v = Vec::new();
                        $( v.push((stringify!($field), format!("{:?}", $field))); )*
                        v
                    } ),*
                }
            }
        }

        /// Executes (or, for a code-generating backend, emits) one instruction.
        #[inline(always)]
        pub fn dispatch<B: Backend>(b: &mut B, ins: Instr) -> Flow<()> {
            match ins {
                $( Instr::$name { $($field),* } => $sem(b, $($field),*) ),*
            }
        }
    };
}

opcodes! {
    /// dst = raw immediate (small ints and specials)
    LoadImm { dst: Reg, bits: i32 } => sem::load_imm;
    /// dst = constant (constants are immortal)
    LoadConst { dst: Reg, idx: u32 } => sem::load_const;
    /// dst = src; src = MOVED (ownership transfer, no RC)
    Mov { dst: Reg, src: Reg } => sem::mov;
    /// dst = src as a borrow (no RC; dst is a borrowed register)
    Borrow { dst: Reg, src: Reg } => sem::borrow;
    /// dst = src++ (new counted reference)
    Dup { dst: Reg, src: Reg } => sem::dup;
    /// Move of a statically untyped local: copies copy types, moves others.
    MoveDyn { dst: Reg, src: Reg } => sem::move_dyn;
    /// Take ownership of a borrowed value: copies copy types, raises otherwise.
    OwnDyn { dst: Reg, src: Reg, what: u32 } => sem::own_dyn;
    /// Drops an owned register.
    Drop { reg: Reg } => sem::drop;
    /// Raises if a register holds a moved / uninitialized value.
    CheckLive { reg: Reg } => sem::check_live;

    Add { dst: Reg, a: Reg, b: Reg } => sem::add;
    Sub { dst: Reg, a: Reg, b: Reg } => sem::sub;
    Mul { dst: Reg, a: Reg, b: Reg } => sem::mul;
    Binary { dst: Reg, op: BinKind, a: Reg, b: Reg } => sem::binary;
    Unary { dst: Reg, op: UnKind, src: Reg } => sem::unary;
    Not { dst: Reg, src: Reg } => sem::not;
    Lt { dst: Reg, a: Reg, b: Reg } => sem::lt;
    Le { dst: Reg, a: Reg, b: Reg } => sem::le;
    Gt { dst: Reg, a: Reg, b: Reg } => sem::gt;
    Ge { dst: Reg, a: Reg, b: Reg } => sem::ge;
    Eq { dst: Reg, a: Reg, b: Reg } => sem::eq;
    Ne { dst: Reg, a: Reg, b: Reg } => sem::ne;
    Is { dst: Reg, a: Reg, b: Reg } => sem::is;
    IsNot { dst: Reg, a: Reg, b: Reg } => sem::is_not;
    In { dst: Reg, a: Reg, b: Reg } => sem::contains;
    NotIn { dst: Reg, a: Reg, b: Reg } => sem::not_contains;

    Jump { target: u32 } => sem::jump;
    /// Loop back-edge: a safepoint, then a jump.
    Loop { target: u32 } => sem::loop_;
    JumpIfFalse { cond: Reg, target: u32 } => sem::jump_if_false;
    JumpIfTrue { cond: Reg, target: u32 } => sem::jump_if_true;

    /// dst = borrow of a global
    LoadGlobal { dst: Reg, idx: u32 } => sem::load_global;
    /// dst = global moved out (global becomes MOVED)
    TakeGlobal { dst: Reg, idx: u32 } => sem::take_global;
    /// global = src (owned, moved in); the old value is released
    StoreGlobal { idx: u32, src: Reg } => sem::store_global;
    /// dst = borrow of a captured variable
    LoadCapture { dst: Reg, idx: u32 } => sem::load_capture;

    Call { dst: Reg, func: Reg, args: u32 } => sem::call;
    CallMethod { dst: Reg, recv: Reg, name: u32, args: u32 } => sem::call_method;
    Return { src: Reg } => sem::ret;
    Yield { dst: Reg, src: Reg } => sem::yield_;

    MakeFunction { dst: Reg, code: u32, captures: u32 } => sem::make_function;
    MakeClass { dst: Reg, spec: u32 } => sem::make_class;
    MakeTrait { dst: Reg, spec: u32 } => sem::make_trait;
    CheckType { reg: Reg, ty: u32, what: u32 } => sem::check_type;

    /// dst = borrow of obj.name
    LoadField { dst: Reg, obj: Reg, name: u32 } => sem::load_field;
    StoreField { obj: Reg, name: u32, src: Reg, mode: StoreMode } => sem::store_field;
    /// dst = old obj.name (owned); obj.name = src (moved in)
    SwapField { dst: Reg, obj: Reg, name: u32, src: Reg } => sem::swap_field;
    /// dst = borrow of obj[idx]
    LoadIndex { dst: Reg, obj: Reg, idx: Reg } => sem::load_index;
    LoadElem { dst: Reg, obj: Reg, idx: Reg } => sem::load_elem;
    StoreIndex { obj: Reg, idx: Reg, src: Reg, mode: StoreMode } => sem::store_index;
    SwapIndex { dst: Reg, obj: Reg, idx: Reg, src: Reg } => sem::swap_index;

    BuildTuple { dst: Reg, args: u32 } => sem::build_tuple;
    BuildList { dst: Reg, args: u32 } => sem::build_list;
    BuildDict { dst: Reg, args: u32 } => sem::build_dict;
    /// Unpacks the owned sequence in `src` into the registers of arglist `dsts`.
    Unpack { src: Reg, dsts: u32 } => sem::unpack;

    /// Raises the exception in `exc` (owned); `cause` is u16::MAX for none.
    Raise { exc: Reg, cause: Reg } => sem::raise;
    /// Re-raises the exception currently being handled.
    Reraise { exc: Reg } => sem::reraise;
    /// In a handler: dst = the pending exception (owned), clears it.
    TakeException { dst: Reg } => sem::take_exception;
    /// Jumps to `target` if the exception in `exc` matches the type in `ty`.
    MatchException { exc: Reg, ty: u32, target: u32 } => sem::match_exception;

    /// dst = iter(src)
    GetIter { dst: Reg, src: Reg } => sem::get_iter;
    /// dst = next(iter) or jump to `end` when exhausted
    ForIter { dst: Reg, iter: Reg, end: u32 } => sem::for_iter;

    Nop {} => sem::nop;
}

impl Instr {
    pub fn render(&self, code: &Code) -> String {
        let mut s = format!("{:<14}", self.name());
        let ops = self.operands();
        let parts: Vec<String> = ops
            .iter()
            .map(|(k, v)| {
                let reg = matches!(*k, "dst" | "src" | "a" | "b" | "reg" | "obj" | "func" | "recv" | "cond" | "exc" | "iter" | "idx")
                    && !matches!(self, Instr::LoadConst { .. } | Instr::LoadGlobal { .. } | Instr::StoreGlobal { .. } | Instr::TakeGlobal { .. } | Instr::LoadCapture { .. });
                if reg || (*k == "idx" && matches!(self, Instr::LoadIndex { .. } | Instr::LoadElem { .. } | Instr::StoreIndex { .. } | Instr::SwapIndex { .. })) {
                    let r: usize = v.parse().unwrap_or(0);
                    match code.reg_names.get(r).and_then(|n| n.as_ref()) {
                        Some(n) => format!("r{v}({n})"),
                        None => format!("r{v}"),
                    }
                } else {
                    format!("{k}={v}")
                }
            })
            .collect();
        s.push_str(&parts.join(", "));
        match self {
            Instr::LoadConst { idx, .. } => {
                s.push_str(&format!("    ; {}", crate::display::repr(code.consts[*idx as usize])))
            }
            Instr::LoadImm { bits, .. } => {
                s.push_str(&format!("    ; {:?}", crate::value::Value(*bits as i64 as u64)))
            }
            Instr::LoadField { name, .. }
            | Instr::StoreField { name, .. }
            | Instr::SwapField { name, .. }
            | Instr::CallMethod { name, .. } => s.push_str(&format!("    ; .{}", code.names[*name as usize])),
            Instr::Call { args, .. } | Instr::BuildTuple { args, .. } | Instr::BuildList { args, .. } => {
                let al = &code.arglists[*args as usize];
                let a: Vec<String> = al
                    .args
                    .iter()
                    .map(|a| format!("r{}{}", a.reg, if a.owned { "" } else { "&" }))
                    .collect();
                s.push_str(&format!("    ; ({}){}", a.join(", "), if al.pins { " pin" } else { "" }));
            }
            _ => {}
        }
        s
    }
}
