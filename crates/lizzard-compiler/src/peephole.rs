//! Peephole pass over a finished function body:
//! - drops of temporaries that hold an immediate or a constant (small ints,
//!   `()`, `None`, literals: nothing to release) are removed;
//! - `op tmp, ..; Mov dst, tmp` becomes `op dst, ..` for ops that read their
//!   operands before writing `dst`;
//! - a comparison followed by a conditional jump on its result becomes one
//!   `CmpJump` (which still writes the result).
//!
//! Both only apply inside a basic block, so no jump target is affected.

use lizzard_vm::code::{ExcEntry, Reg};
use lizzard_vm::ops::{CmpKind, Instr};

pub fn run(instrs: &mut Vec<Instr>, lines: &mut Vec<u32>, exc_table: &mut [ExcEntry], entry: &mut u32, reg_names: &[Option<String>]) {
    let n = instrs.len();
    let mut is_target = vec![false; n + 1];
    for i in instrs.iter() {
        if let Some(t) = jump_target(i) {
            is_target[t as usize] = true;
        }
    }
    for e in exc_table.iter() {
        for pc in [e.start, e.end, e.handler] {
            is_target[pc as usize] = true;
        }
    }
    is_target[*entry as usize] = true;
    let temp = |r: Reg| reg_names.get(r as usize).is_none_or(|n| n.is_none());

    let mut keep = vec![true; n];
    // Registers known to hold an immediate or constant at this point.
    let mut imm: Vec<Reg> = Vec::new();
    for pc in 0..n {
        if !keep[pc] {
            continue;
        }
        if is_target[pc] {
            imm.clear();
        }
        let i = instrs[pc];
        if let (Some((op, dst, a, b)), Some(&next)) = (compare(i), instrs.get(pc + 1)) {
            let jump = match next {
                Instr::JumpIfFalse { cond, target } if cond == dst => Some((target, false)),
                Instr::JumpIfTrue { cond, target } if cond == dst => Some((target, true)),
                _ => None,
            };
            if let (Some((target, jump_if)), false) = (jump, is_target[pc + 1]) {
                instrs[pc] = Instr::CmpJump { dst, op, a, b, target, jump_if };
                keep[pc + 1] = false;
                imm.retain(|&r| r != dst);
                continue;
            }
        }
        // `op tmp; Drop d; Mov d, tmp` for a loop variable `d`.
        if let (Some(&Instr::Drop { reg: d }), Some(&Instr::Mov { dst, src })) = (instrs.get(pc + 1), instrs.get(pc + 2)) {
            if dst == d && temp(src) && !is_target[pc + 1] && !is_target[pc + 2] {
                let fused = match i {
                    // `Dup` cannot raise, so the old value may go first.
                    Instr::Dup { dst: t, src: s } if t == src && s != d => Some((Instr::Drop { reg: d }, Some(Instr::Dup { dst: d, src: s }))),
                    Instr::ForIter { dst: t, iter, end } if t == src && iter != d => Some((Instr::ForIterSet { dst: d, iter, end }, None)),
                    _ => None,
                };
                if let Some((first, second)) = fused {
                    instrs[pc] = first;
                    match second {
                        Some(s) => instrs[pc + 1] = s,
                        None => keep[pc + 1] = false,
                    }
                    keep[pc + 2] = false;
                    imm.retain(|&r| r != d && r != src);
                    continue;
                }
            }
        }
        match i {
            Instr::Drop { reg } if temp(reg) && imm.contains(&reg) => {
                keep[pc] = false;
                continue;
            }
            Instr::Mov { dst, src } if temp(src) && pc > 0 && keep[pc - 1] && !is_target[pc] => {
                if let Some(fused) = with_dst(instrs[pc - 1], src, dst) {
                    instrs[pc - 1] = fused;
                    keep[pc] = false;
                    let was_imm = imm.contains(&src);
                    imm.retain(|&r| r != src && r != dst);
                    if was_imm {
                        imm.push(dst);
                    }
                    continue;
                }
            }
            _ => {}
        }
        match i {
            Instr::LoadImm { dst, .. } | Instr::LoadConst { dst, .. } => {
                imm.retain(|&r| r != dst);
                imm.push(dst);
            }
            Instr::Mov { dst, src } => {
                let was_imm = imm.contains(&src);
                imm.retain(|&r| r != src && r != dst);
                if was_imm {
                    imm.push(dst);
                }
            }
            Instr::Unpack { .. } => imm.clear(),
            _ => {
                if let Some(d) = dst_of(&i) {
                    imm.retain(|&r| r != d);
                }
            }
        }
    }
    if keep.iter().all(|&k| k) {
        return;
    }

    // New index of each old pc (removed instructions map to the next kept one).
    let mut map = vec![0u32; n + 1];
    let mut k = 0u32;
    for pc in 0..n {
        map[pc] = k;
        k += keep[pc] as u32;
    }
    map[n] = k;
    let mut out = Vec::with_capacity(k as usize);
    let mut out_lines = Vec::with_capacity(k as usize);
    for pc in 0..n {
        if keep[pc] {
            out.push(retarget(instrs[pc], &map));
            out_lines.push(lines[pc]);
        }
    }
    *instrs = out;
    *lines = out_lines;
    *entry = map[*entry as usize];
    for e in exc_table.iter_mut() {
        e.start = map[e.start as usize];
        e.end = map[e.end as usize];
        e.handler = map[e.handler as usize];
    }
}

fn jump_target(i: &Instr) -> Option<u32> {
    match *i {
        Instr::Jump { target }
        | Instr::Loop { target }
        | Instr::JumpIfFalse { target, .. }
        | Instr::JumpIfTrue { target, .. }
        | Instr::MatchException { target, .. }
        | Instr::CmpJump { target, .. } => Some(target),
        Instr::ForIter { end, .. } | Instr::ForIterSet { end, .. } => Some(end),
        _ => None,
    }
}

fn retarget(i: Instr, map: &[u32]) -> Instr {
    let m = |t: u32| map[t as usize];
    match i {
        Instr::Jump { target } => Instr::Jump { target: m(target) },
        Instr::Loop { target } => Instr::Loop { target: m(target) },
        Instr::JumpIfFalse { cond, target } => Instr::JumpIfFalse { cond, target: m(target) },
        Instr::JumpIfTrue { cond, target } => Instr::JumpIfTrue { cond, target: m(target) },
        Instr::MatchException { exc, ty, target } => Instr::MatchException { exc, ty, target: m(target) },
        Instr::CmpJump { dst, op, a, b, target, jump_if } => Instr::CmpJump { dst, op, a, b, target: m(target), jump_if },
        Instr::ForIter { dst, iter, end } => Instr::ForIter { dst, iter, end: m(end) },
        Instr::ForIterSet { dst, iter, end } => Instr::ForIterSet { dst, iter, end: m(end) },
        other => other,
    }
}

/// `i` with its result written to `dst` instead of `tmp`, if `i` is a
/// simple op (reads all operands, then writes only `dst`) producing `tmp`.
fn with_dst(i: Instr, tmp: Reg, dst: Reg) -> Option<Instr> {
    use Instr::*;
    Some(match i {
        LoadImm { dst: d, bits } if d == tmp => LoadImm { dst, bits },
        LoadConst { dst: d, idx } if d == tmp => LoadConst { dst, idx },
        Dup { dst: d, src } if d == tmp => Dup { dst, src },
        // Writes `dst` only when it does not jump (as the `Mov` it absorbs).
        ForIter { dst: d, iter, end } if d == tmp && iter != dst => ForIter { dst, iter, end },
        BoxF { dst: d, src } if d == tmp => BoxF { dst, src },
        Add { dst: d, a, b } if d == tmp => Add { dst, a, b },
        Sub { dst: d, a, b } if d == tmp => Sub { dst, a, b },
        Mul { dst: d, a, b } if d == tmp => Mul { dst, a, b },
        Binary { dst: d, op, a, b } if d == tmp => Binary { dst, op, a, b },
        Unary { dst: d, op, src } if d == tmp => Unary { dst, op, src },
        Not { dst: d, src } if d == tmp => Not { dst, src },
        Lt { dst: d, a, b } if d == tmp => Lt { dst, a, b },
        Le { dst: d, a, b } if d == tmp => Le { dst, a, b },
        Gt { dst: d, a, b } if d == tmp => Gt { dst, a, b },
        Ge { dst: d, a, b } if d == tmp => Ge { dst, a, b },
        Eq { dst: d, a, b } if d == tmp => Eq { dst, a, b },
        Ne { dst: d, a, b } if d == tmp => Ne { dst, a, b },
        _ => return None,
    })
}

/// A comparison `(op, dst, a, b)`.
fn compare(i: Instr) -> Option<(CmpKind, Reg, Reg, Reg)> {
    Some(match i {
        Instr::Lt { dst, a, b } => (CmpKind::Lt, dst, a, b),
        Instr::Le { dst, a, b } => (CmpKind::Le, dst, a, b),
        Instr::Gt { dst, a, b } => (CmpKind::Gt, dst, a, b),
        Instr::Ge { dst, a, b } => (CmpKind::Ge, dst, a, b),
        Instr::Eq { dst, a, b } => (CmpKind::Eq, dst, a, b),
        Instr::Ne { dst, a, b } => (CmpKind::Ne, dst, a, b),
        _ => return None,
    })
}

/// The register an instruction writes (besides `Unpack`, which writes several).
fn dst_of(i: &Instr) -> Option<Reg> {
    i.operands().into_iter().find(|(f, _)| *f == "dst").and_then(|(_, v)| v.parse().ok())
}
