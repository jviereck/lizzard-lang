//! Opcode semantics, written once and generic over the `Backend`.
//!
//! Each function receives its operands as plain Rust values (registers,
//! indices); only runtime data flows through `B::V`. Fast paths are expressed
//! with backend operations, slow paths call runtime helpers.

use crate::backend::{Backend, Cond, Flow};
use crate::code::Reg;
use crate::helpers as h;
use crate::ops::{BinKind, StoreMode, UnKind};
use crate::value::Value;

pub fn nop<B: Backend>(_b: &mut B) -> Flow<()> {
    Ok(())
}

// ---------------------------------------------------------------- moves & RC

pub fn load_imm<B: Backend>(b: &mut B, dst: Reg, bits: i32) -> Flow<()> {
    let v = b.imm(bits as i64 as u64);
    b.set_reg(dst, v);
    Ok(())
}

pub fn load_const<B: Backend>(b: &mut B, dst: Reg, idx: u32) -> Flow<()> {
    let v = b.code_const(idx);
    b.set_reg(dst, v);
    Ok(())
}

pub fn mov<B: Backend>(b: &mut B, dst: Reg, src: Reg) -> Flow<()> {
    let v = b.reg(src);
    b.set_reg(dst, v);
    let moved = b.imm(Value::MOVED.0);
    b.set_reg(src, moved);
    Ok(())
}

pub fn borrow<B: Backend>(b: &mut B, dst: Reg, src: Reg) -> Flow<()> {
    let v = b.reg(src);
    b.set_reg(dst, v);
    Ok(())
}

fn is_dead<B: Backend>(b: &mut B, v: B::V) -> B::B {
    let moved = b.imm(Value::MOVED.0);
    let uninit = b.imm(Value::UNINIT.0);
    let m = b.eq_word(v, moved);
    let u = b.eq_word(v, uninit);
    b.or(m, u)
}

pub fn dup<B: Backend>(b: &mut B, dst: Reg, src: Reg) -> Flow<()> {
    let v = b.reg(src);
    let dead = is_dead(b, v);
    b.if_unit(dead, |b| b.frame_op(h::f_dead, src as u32, 0, 0, 0), |_| Ok(()))?;
    b.incref(v);
    b.set_reg(dst, v);
    Ok(())
}

pub fn move_dyn<B: Backend>(b: &mut B, dst: Reg, src: Reg) -> Flow<()> {
    b.frame_op(h::f_move_dyn, dst as u32, src as u32, 0, 0)
}

pub fn own_dyn<B: Backend>(b: &mut B, dst: Reg, src: Reg, what: u32) -> Flow<()> {
    b.frame_op(h::f_own_dyn, dst as u32, src as u32, what, 0)
}

pub fn drop<B: Backend>(b: &mut B, reg: Reg) -> Flow<()> {
    let v = b.reg(reg);
    b.decref(v);
    let u = b.imm(Value::UNINIT.0);
    b.set_reg(reg, u);
    Ok(())
}

pub fn check_live<B: Backend>(b: &mut B, reg: Reg) -> Flow<()> {
    let v = b.reg(reg);
    let dead = is_dead(b, v);
    b.if_unit(dead, |b| b.frame_op(h::f_dead, reg as u32, 0, 0, 0), |_| Ok(()))
}

// ---------------------------------------------------------------- arithmetic

fn both_small<B: Backend>(b: &mut B, x: B::V, y: B::V) -> B::B {
    let a = b.is_small_int(x);
    let c = b.is_small_int(y);
    b.and(a, c)
}

/// Tagged small ints are `2i + 1`, so `x + y - 1` is the tagged sum.
pub fn add<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let ints = both_small(b, x, y);
    let r = b.if_v(
        ints,
        |b| {
            let y1 = b.add_word(y, -1);
            let (s, ovf) = b.add_ovf(x, y1);
            b.if_v(ovf, |b| b.helper2(h::add, x, y), |_| Ok(s))
        },
        |b| b.helper2(h::add, x, y),
    )?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn sub<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let ints = both_small(b, x, y);
    let r = b.if_v(
        ints,
        |b| {
            let (d, ovf) = b.sub_ovf(x, y);
            b.if_v(ovf, |b| b.helper2(h::sub, x, y), |b| Ok(b.add_word(d, 1)))
        },
        |b| b.helper2(h::sub, x, y),
    )?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn mul<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let ints = both_small(b, x, y);
    let r = b.if_v(
        ints,
        |b| {
            let x2 = b.add_word(x, -1);
            let yi = b.sar(y, 1);
            let (p, ovf) = b.mul_ovf(x2, yi);
            b.if_v(ovf, |b| b.helper2(h::mul, x, y), |b| Ok(b.add_word(p, 1)))
        },
        |b| b.helper2(h::mul, x, y),
    )?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn binary<B: Backend>(b: &mut B, dst: Reg, op: BinKind, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let f = match op {
        BinKind::Div => h::div,
        BinKind::FloorDiv => h::floordiv,
        BinKind::Mod => h::modulo,
        BinKind::Pow => h::pow,
        BinKind::BitAnd => h::bitand,
        BinKind::BitOr => h::bitor,
        BinKind::BitXor => h::bitxor,
        BinKind::Shl => h::shl,
        BinKind::Shr => h::shr,
    };
    let r = b.helper2(f, x, y)?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn unary<B: Backend>(b: &mut B, dst: Reg, op: UnKind, src: Reg) -> Flow<()> {
    let x = b.reg(src);
    let f = match op {
        UnKind::Neg => h::neg,
        UnKind::Pos => h::pos,
        UnKind::Invert => h::invert,
    };
    let r = b.helper1(f, x)?;
    b.set_reg(dst, r);
    Ok(())
}

fn truthy<B: Backend>(b: &mut B, v: B::V) -> Flow<B::B> {
    let t = b.imm(Value::TRUE.0);
    let f = b.imm(Value::FALSE.0);
    let is_t = b.eq_word(v, t);
    let is_f = b.eq_word(v, f);
    let is_bool = b.or(is_t, is_f);
    let r = b.if_v(is_bool, |_| Ok(v), |b| b.helper1(h::truthy, v))?;
    Ok(b.eq_word(r, t))
}

pub fn not<B: Backend>(b: &mut B, dst: Reg, src: Reg) -> Flow<()> {
    let v = b.reg(src);
    let t = truthy(b, v)?;
    let n = b.not(t);
    let r = b.bool_value(n);
    b.set_reg(dst, r);
    Ok(())
}

fn compare<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg, c: Cond, slow: crate::backend::Helper2) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let ints = both_small(b, x, y);
    let r = b.if_v(
        ints,
        |b| {
            let c = b.cmp(c, x, y);
            Ok(b.bool_value(c))
        },
        |b| b.helper2(slow, x, y),
    )?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn lt<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    compare(b, dst, a, bb, Cond::Lt, h::lt)
}

pub fn le<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    compare(b, dst, a, bb, Cond::Le, h::le)
}

pub fn gt<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    compare(b, dst, a, bb, Cond::Gt, h::gt)
}

pub fn ge<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    compare(b, dst, a, bb, Cond::Ge, h::ge)
}

pub fn eq<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let ints = both_small(b, x, y);
    let r = b.if_v(
        ints,
        |b| {
            let c = b.eq_word(x, y);
            Ok(b.bool_value(c))
        },
        |b| b.helper2(h::eq, x, y),
    )?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn ne<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let ints = both_small(b, x, y);
    let r = b.if_v(
        ints,
        |b| {
            let c = b.eq_word(x, y);
            let n = b.not(c);
            Ok(b.bool_value(n))
        },
        |b| b.helper2(h::ne, x, y),
    )?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn is<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let c = b.eq_word(x, y);
    let r = b.bool_value(c);
    b.set_reg(dst, r);
    Ok(())
}

pub fn is_not<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let c = b.eq_word(x, y);
    let n = b.not(c);
    let r = b.bool_value(n);
    b.set_reg(dst, r);
    Ok(())
}

pub fn contains<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let r = b.helper2(h::contains, x, y)?;
    b.set_reg(dst, r);
    Ok(())
}

pub fn not_contains<B: Backend>(b: &mut B, dst: Reg, a: Reg, bb: Reg) -> Flow<()> {
    let x = b.reg(a);
    let y = b.reg(bb);
    let r = b.helper2(h::not_contains, x, y)?;
    b.set_reg(dst, r);
    Ok(())
}

// ---------------------------------------------------------------- control

pub fn jump<B: Backend>(b: &mut B, target: u32) -> Flow<()> {
    b.jump(target)
}

pub fn loop_<B: Backend>(b: &mut B, target: u32) -> Flow<()> {
    b.frame_op(h::f_safepoint, 0, 0, 0, 0)?;
    b.jump(target)
}

pub fn jump_if_false<B: Backend>(b: &mut B, cond: Reg, target: u32) -> Flow<()> {
    let v = b.reg(cond);
    let t = truthy(b, v)?;
    let f = b.not(t);
    b.branch(f, target)
}

pub fn jump_if_true<B: Backend>(b: &mut B, cond: Reg, target: u32) -> Flow<()> {
    let v = b.reg(cond);
    let t = truthy(b, v)?;
    b.branch(t, target)
}

// ---------------------------------------------------------------- globals, calls

pub fn load_global<B: Backend>(b: &mut B, dst: Reg, idx: u32) -> Flow<()> {
    b.frame_op(h::f_load_global, dst as u32, idx, 0, 0)
}

pub fn take_global<B: Backend>(b: &mut B, dst: Reg, idx: u32) -> Flow<()> {
    b.frame_op(h::f_take_global, dst as u32, idx, 0, 0)
}

pub fn store_global<B: Backend>(b: &mut B, idx: u32, src: Reg) -> Flow<()> {
    b.frame_op(h::f_store_global, idx, src as u32, 0, 0)
}

pub fn load_capture<B: Backend>(b: &mut B, dst: Reg, idx: u32) -> Flow<()> {
    b.frame_op(h::f_load_capture, dst as u32, idx, 0, 0)
}

pub fn call<B: Backend>(b: &mut B, dst: Reg, func: Reg, args: u32) -> Flow<()> {
    b.frame_op(h::f_call, dst as u32, func as u32, args, 0)
}

pub fn call_method<B: Backend>(b: &mut B, dst: Reg, recv: Reg, name: u32, args: u32) -> Flow<()> {
    b.frame_op(h::f_call_method, dst as u32, recv as u32, name, args)
}

pub fn ret<B: Backend>(b: &mut B, src: Reg) -> Flow<()> {
    let v = b.reg(src);
    let moved = b.imm(Value::MOVED.0);
    b.set_reg(src, moved);
    b.frame_op(h::f_drop_frame, 0, 0, 0, 0)?;
    b.ret(v)
}

pub fn yield_<B: Backend>(b: &mut B, dst: Reg, src: Reg) -> Flow<()> {
    b.frame_op(h::f_yield, dst as u32, src as u32, 0, 0)
}

pub fn make_function<B: Backend>(b: &mut B, dst: Reg, code: u32, captures: u32) -> Flow<()> {
    b.frame_op(h::f_make_function, dst as u32, code, captures, 0)
}

pub fn make_class<B: Backend>(b: &mut B, dst: Reg, spec: u32) -> Flow<()> {
    b.frame_op(h::f_make_class, dst as u32, spec, 0, 0)
}

pub fn make_trait<B: Backend>(b: &mut B, dst: Reg, spec: u32) -> Flow<()> {
    b.frame_op(h::f_make_trait, dst as u32, spec, 0, 0)
}

pub fn call_trait_method<B: Backend>(b: &mut B, dst: Reg, recv: Reg, call: u32, args: u32) -> Flow<()> {
    b.frame_op(h::f_call_trait_method, dst as u32, recv as u32, call, args)
}

pub fn make_impl<B: Backend>(b: &mut B, spec: u32) -> Flow<()> {
    b.frame_op(h::f_make_impl, spec, 0, 0, 0)
}

pub fn load_impl_arg<B: Backend>(b: &mut B, dst: Reg, obj: Reg, ty: u32, k: u32) -> Flow<()> {
    b.frame_op(h::f_load_impl_arg, dst as u32, obj as u32, ty, k)
}

pub fn check_type<B: Backend>(b: &mut B, reg: Reg, ty: u32, what: u32) -> Flow<()> {
    b.frame_op(h::f_check_type, reg as u32, ty, what, 0)
}

pub fn make_type<B: Backend>(b: &mut B, dst: Reg, ty: u32) -> Flow<()> {
    b.frame_op(h::f_make_type, dst as u32, ty, 0, 0)
}

pub fn load_type_of<B: Backend>(b: &mut B, dst: Reg, obj: Reg) -> Flow<()> {
    b.frame_op(h::f_load_type_of, dst as u32, obj as u32, 0, 0)
}

pub fn load_type_arg<B: Backend>(b: &mut B, dst: Reg, obj: Reg, k: u32) -> Flow<()> {
    b.frame_op(h::f_load_type_arg, dst as u32, obj as u32, k, 0)
}

pub fn load_trait_arg<B: Backend>(b: &mut B, dst: Reg, obj: Reg, tr: u32, k: u32) -> Flow<()> {
    b.frame_op(h::f_load_trait_arg, dst as u32, obj as u32, tr, k)
}

pub fn instantiate<B: Backend>(b: &mut B, dst: Reg, func: Reg, targs: u32) -> Flow<()> {
    b.frame_op(h::f_instantiate, dst as u32, func as u32, targs, 0)
}

// ---------------------------------------------------------------- fields, elements

pub fn load_field<B: Backend>(b: &mut B, dst: Reg, obj: Reg, name: u32) -> Flow<()> {
    b.frame_op(h::f_load_field, dst as u32, obj as u32, name, 0)
}

pub fn store_field<B: Backend>(b: &mut B, obj: Reg, name: u32, src: Reg, mode: StoreMode) -> Flow<()> {
    b.frame_op(h::f_store_field, obj as u32, name, src as u32, mode as u32)
}

pub fn swap_field<B: Backend>(b: &mut B, dst: Reg, obj: Reg, name: u32, src: Reg) -> Flow<()> {
    b.frame_op(h::f_swap_field, dst as u32, obj as u32, name, src as u32)
}

pub fn load_index<B: Backend>(b: &mut B, dst: Reg, obj: Reg, idx: Reg) -> Flow<()> {
    b.frame_op(h::f_load_index, dst as u32, obj as u32, idx as u32, 0)
}

/// `a[i]` as an owned copy-type value (array elements).
pub fn load_elem<B: Backend>(b: &mut B, dst: Reg, obj: Reg, idx: Reg) -> Flow<()> {
    b.frame_op(h::f_load_elem, dst as u32, obj as u32, idx as u32, 0)
}

pub fn store_index<B: Backend>(b: &mut B, obj: Reg, idx: Reg, src: Reg, mode: StoreMode, proven: bool) -> Flow<()> {
    let m = mode as u32 | if proven { h::STORE_PROVEN } else { 0 };
    b.frame_op(h::f_store_index, obj as u32, idx as u32, src as u32, m)
}

pub fn swap_index<B: Backend>(b: &mut B, dst: Reg, obj: Reg, idx: Reg, src: Reg) -> Flow<()> {
    b.frame_op(h::f_swap_index, dst as u32, obj as u32, idx as u32, src as u32)
}

pub fn build_tuple<B: Backend>(b: &mut B, dst: Reg, args: u32) -> Flow<()> {
    b.frame_op(h::f_build_tuple, dst as u32, args, 0, 0)
}

pub fn build_struct<B: Backend>(b: &mut B, dst: Reg, args: u32, ty: u32) -> Flow<()> {
    b.frame_op(h::f_build_struct, dst as u32, args, ty, 0)
}

pub fn build_list<B: Backend>(b: &mut B, dst: Reg, args: u32, ty: u32) -> Flow<()> {
    b.frame_op(h::f_build_list, dst as u32, args, ty, 0)
}

pub fn build_dict<B: Backend>(b: &mut B, dst: Reg, args: u32, ty: u32) -> Flow<()> {
    b.frame_op(h::f_build_dict, dst as u32, args, ty, 0)
}

pub fn unpack<B: Backend>(b: &mut B, src: Reg, dsts: u32) -> Flow<()> {
    b.frame_op(h::f_unpack, src as u32, dsts, 0, 0)
}

// ---------------------------------------------------------------- exceptions

pub fn raise<B: Backend>(b: &mut B, exc: Reg, cause: Reg) -> Flow<()> {
    b.frame_op(h::f_raise, exc as u32, cause as u32, 0, 0)
}

pub fn reraise<B: Backend>(b: &mut B, exc: Reg) -> Flow<()> {
    b.frame_op(h::f_reraise, exc as u32, 0, 0, 0)
}

pub fn take_exception<B: Backend>(b: &mut B, dst: Reg) -> Flow<()> {
    b.frame_op(h::f_take_exception, dst as u32, 0, 0, 0)
}

pub fn match_exception<B: Backend>(b: &mut B, exc: Reg, ty: u32, target: u32) -> Flow<()> {
    b.frame_op(h::f_match_exception, exc as u32, ty, target, 0)
}

// ---------------------------------------------------------------- iteration

pub fn get_iter<B: Backend>(b: &mut B, dst: Reg, src: Reg) -> Flow<()> {
    b.frame_op(h::f_get_iter, dst as u32, src as u32, 0, 0)
}

pub fn for_iter<B: Backend>(b: &mut B, dst: Reg, iter: Reg, end: u32) -> Flow<()> {
    b.frame_op(h::f_for_iter, dst as u32, iter as u32, end, 0)
}
