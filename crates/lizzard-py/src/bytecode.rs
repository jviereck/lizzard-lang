//! The interpreter's own encoding of a function: one fixed-size record per
//! IR instruction (same pcs, so handler ranges, lines and generator resume
//! points carry over), with
//! - operands resolved when the function is loaded: register numbers, and
//!   indices into the function's constants, names, checks and caches,
//!   validated here so that the interpreter indexes without bounds checks;
//! - variants decided once instead of per execution (object or scalar
//!   moves, boxing from `i64`, the rich comparison's code);
//! - superinstructions for common pairs: an operation and the check of its
//!   result, a comparison and the branch on it. The fused record sits at
//!   the first pc and the second instruction keeps its own record, so a
//!   jump to it still works; when the second half needs more than the fast
//!   path (a check that fails, say), execution continues at its own pc, so
//!   exceptions see the right instruction.
//!
//! Everything else is `Slow`: `step` runs the IR instruction.

use lizzard_py_front::ir::{CmpOp, FuncIr, Instr, Repr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Opc {
    Slow,
    Jump,
    JumpIfFalse,
    JumpIfTrue,
    I64Const,
    I64Lt,
    I64Add,
    Not,
    Const,
    MoveObj,
    MoveScalar,
    TakeObj,
    UnbindObj,
    Nop,
    LoadAttr,
    LoadAttrCheck,
    StoreAttr,
    LoadGlobal,
    BinOp,
    BinOpCheck,
    Compare,
    CompareBool,
    CompareBoolJump,
    Truthy,
    BoxI64,
    Check,
    Return,
    ForIter,
    Call,
}

/// One instruction. The meaning of `x`..`w` and `sub` depends on `op`; see
/// `lower`.
#[derive(Clone, Copy, Debug)]
pub struct Op {
    pub op: Opc,
    pub sub: u8,
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}

fn op(op: Opc, x: u32, y: u32, z: u32) -> Op {
    Op { op, sub: 0, x, y, z, w: 0 }
}

/// CPython's `Py_LT`, `Py_LE`, `Py_EQ`, `Py_NE`, `Py_GT`, `Py_GE`.
fn rich(op: CmpOp) -> Option<u8> {
    Some(match op {
        CmpOp::Lt => 0,
        CmpOp::Le => 1,
        CmpOp::Eq => 2,
        CmpOp::Ne => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
        _ => return None,
    })
}

/// Encode `ir`. An instruction whose operands are out of range (which the
/// front end does not produce) stays `Slow`, where indexing is checked.
pub fn lower(ir: &FuncIr) -> Vec<Op> {
    let n = ir.code.len();
    let nregs = ir.reprs.len();
    let obj = |r: u32| ir.reprs[r as usize] == Repr::Obj;
    let valid = |i: &Instr| {
        i.reads().iter().chain(i.writes().iter()).all(|&r| (r as usize) < nregs)
            && match i {
                Instr::Jump { target } | Instr::JumpIfFalse { target, .. } | Instr::JumpIfTrue { target, .. } | Instr::ForIter { target, .. } | Instr::JumpIfNotSelf { target, .. } => (*target as usize) < n,
                _ => true,
            }
            && match i {
                Instr::Const { idx, .. } => (*idx as usize) < ir.consts.len(),
                Instr::LoadAttr { name, .. } | Instr::StoreAttr { name, .. } | Instr::LoadGlobal { name, .. } => (*name as usize) < ir.names.len(),
                Instr::Check { check, .. } => (*check as usize) < ir.checks.len(),
                _ => true,
            }
    };
    let mut out = Vec::with_capacity(n);
    for (pc, instr) in ir.code.iter().enumerate() {
        if !valid(instr) {
            out.push(op(Opc::Slow, 0, 0, 0));
            continue;
        }
        // The next instruction, for fusing (only if it is valid too).
        let next = ir.code.get(pc + 1).filter(|i| valid(i));
        let check_of = |dst: u32| match next {
            Some(Instr::Check { reg, check }) if *reg == dst => Some(*check),
            _ => None,
        };
        let o = match instr {
            Instr::Jump { target } => op(Opc::Jump, *target, 0, 0),
            Instr::JumpIfFalse { cond, target } => op(Opc::JumpIfFalse, *cond, *target, 0),
            Instr::JumpIfTrue { cond, target } => op(Opc::JumpIfTrue, *cond, *target, 0),
            Instr::I64Const { dst, value } => op(Opc::I64Const, *dst, *value as u64 as u32, (*value as u64 >> 32) as u32),
            Instr::I64Lt { dst, a, b } => op(Opc::I64Lt, *dst, *a, *b),
            Instr::I64Add { dst, a, b } => op(Opc::I64Add, *dst, *a, *b),
            Instr::Not { dst, src } => op(Opc::Not, *dst, *src, 0),
            Instr::Const { dst, idx } => op(Opc::Const, *dst, *idx, 0),
            Instr::Move { dst, src } if obj(*src) && obj(*dst) => op(Opc::MoveObj, *dst, *src, 0),
            Instr::Move { dst, src } | Instr::Take { dst, src } if !obj(*src) && !obj(*dst) => op(Opc::MoveScalar, *dst, *src, 0),
            Instr::Take { dst, src } if obj(*src) && obj(*dst) => op(Opc::TakeObj, *dst, *src, 0),
            Instr::Unbind { reg } if obj(*reg) => op(Opc::UnbindObj, *reg, 0, 0),
            Instr::Unbind { .. } => op(Opc::Nop, 0, 0, 0),
            Instr::LoadAttr { dst, obj, name } => match check_of(*dst) {
                Some(c) => Op { op: Opc::LoadAttrCheck, sub: 0, x: *dst, y: *obj, z: *name, w: c },
                None => op(Opc::LoadAttr, *dst, *obj, *name),
            },
            Instr::StoreAttr { obj, name, val } => op(Opc::StoreAttr, *obj, *val, *name),
            Instr::LoadGlobal { dst, name } => op(Opc::LoadGlobal, *dst, *name, 0),
            Instr::BinOp { dst, op: b, a, b: bb, inplace } => {
                let sub = *b as u8 + if *inplace { 16 } else { 0 };
                match check_of(*dst) {
                    Some(c) => Op { op: Opc::BinOpCheck, sub, x: *dst, y: *a, z: *bb, w: c },
                    None => Op { op: Opc::BinOp, sub, x: *dst, y: *a, z: *bb, w: 0 },
                }
            }
            Instr::Compare { dst, op: c, a, b } if rich(*c).is_some() => Op { op: Opc::Compare, sub: rich(*c).unwrap(), x: *dst, y: *a, z: *b, w: 0 },
            Instr::CompareBool { dst, op: c, a, b } if rich(*c).is_some() => {
                let sub = rich(*c).unwrap();
                match next {
                    Some(Instr::JumpIfFalse { cond, target }) if cond == dst => Op { op: Opc::CompareBoolJump, sub, x: *dst, y: *a, z: *b, w: *target },
                    _ => Op { op: Opc::CompareBool, sub, x: *dst, y: *a, z: *b, w: 0 },
                }
            }
            Instr::Truthy { dst, src } => op(Opc::Truthy, *dst, *src, 0),
            Instr::Box { dst, src } if ir.reprs[*src as usize] == Repr::I64 => op(Opc::BoxI64, *dst, *src, 0),
            Instr::Check { reg, check } => op(Opc::Check, *reg, *check, 0),
            Instr::Return { src } if obj(*src) => op(Opc::Return, *src, 0, 0),
            Instr::ForIter { dst, iter, target } => op(Opc::ForIter, *dst, *iter, *target),
            Instr::Call { dst, .. } | Instr::CallMethod { dst, .. } => op(Opc::Call, *dst, 0, 0),
            _ => op(Opc::Slow, 0, 0, 0),
        };
        out.push(o);
    }
    out
}
