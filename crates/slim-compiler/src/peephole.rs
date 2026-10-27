//! Peephole pass over a finished function body:
//! - drops of temporaries that hold an immediate or a constant (small ints,
//!   `()`, `None`, literals: nothing to release) are removed;
//! - `op tmp, ..; Mov dst, tmp` becomes `op dst, ..` for ops that read their
//!   operands before writing `dst`.
//!
//! Both only apply inside a basic block, so no jump target is affected.

use slim_vm::code::{ExcEntry, Reg};
use slim_vm::ops::Instr;

pub fn run(instrs: &mut Vec<Instr>, lines: &mut Vec<u32>, exc_table: &mut [ExcEntry], entry: &mut u32, reg_names: &[Option<String>]) {
    let n = instrs.len();
    let mut target = vec![false; n + 1];
    for i in instrs.iter() {
        if let Some(t) = jump_target(i) {
            target[t as usize] = true;
        }
    }
    for e in exc_table.iter() {
        for pc in [e.start, e.end, e.handler] {
            target[pc as usize] = true;
        }
    }
    target[*entry as usize] = true;
    let temp = |r: Reg| reg_names.get(r as usize).is_none_or(|n| n.is_none());

    let mut keep = vec![true; n];
    // Registers known to hold an immediate or constant at this point.
    let mut imm: Vec<Reg> = Vec::new();
    for pc in 0..n {
        if target[pc] {
            imm.clear();
        }
        let i = instrs[pc];
        match i {
            Instr::Drop { reg } if temp(reg) && imm.contains(&reg) => {
                keep[pc] = false;
                continue;
            }
            Instr::Mov { dst, src } if temp(src) && pc > 0 && keep[pc - 1] && !target[pc] => {
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
        | Instr::MatchException { target, .. } => Some(target),
        Instr::ForIter { end, .. } => Some(end),
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
        Instr::ForIter { dst, iter, end } => Instr::ForIter { dst, iter, end: m(end) },
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

/// The register an instruction writes (besides `Unpack`, which writes several).
fn dst_of(i: &Instr) -> Option<Reg> {
    i.operands().into_iter().find(|(f, _)| *f == "dst").and_then(|(_, v)| v.parse().ok())
}
