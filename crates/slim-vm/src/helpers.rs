//! Runtime helpers: slow paths called by opcode semantics.
//!
//! Value helpers (`Helper1`/`Helper2`) return `Value::UNINIT` when they raise.
//! Frame helpers return 0 (ok), 1 (raised), 2 (returned) or 3 (yielded).

use crate::array::{ArrayObj, Scalar, StoreError};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

use crate::code::{Arg, ArgList, ClassSpec, Code, Ctor, Reg, TypeDesc};
use crate::display;
use crate::interp::{run_code, run_code_at, Frame, Raised, Thread, VResult};
use crate::list::ListObj;
use crate::object::{Body, Class, Field, Function, Instance, Obj, SlotKind, Trait};
use crate::ops::StoreMode;
use crate::rc;
use crate::rtype;
use crate::types;
use crate::value::Value;

const OK: u32 = 0;
const RAISED: u32 = 1;

#[inline]
fn vres(r: VResult<Value>) -> Value {
    r.unwrap_or(Value::UNINIT)
}

#[inline]
fn fres(r: VResult<()>) -> u32 {
    match r {
        Ok(()) => OK,
        Err(Raised) => RAISED,
    }
}

// ---------------------------------------------------------------- numbers

enum Num {
    Int(i64),
    Big(BigInt),
    Float(f64),
}

fn num(v: Value) -> Option<Num> {
    if let Some(i) = v.as_small_int() {
        return Some(Num::Int(i));
    }
    match v.body()? {
        Body::Int(i) => Some(Num::Int(*i)),
        Body::BigInt(b) => Some(Num::Big(b.clone())),
        Body::Float(f) => Some(Num::Float(*f)),
        _ => None,
    }
}

fn big_value(b: BigInt) -> Value {
    Obj::alloc(Body::BigInt(b))
}

fn to_f64(n: &Num) -> f64 {
    match n {
        Num::Int(i) => *i as f64,
        Num::Big(b) => b.to_f64().unwrap_or(f64::INFINITY),
        Num::Float(f) => *f,
    }
}

fn to_big(n: &Num) -> BigInt {
    match n {
        Num::Int(i) => BigInt::from(*i),
        Num::Big(b) => b.clone(),
        Num::Float(f) => BigInt::from(*f as i64),
    }
}

pub(crate) fn dead_operand(th: &mut Thread, v: Value) -> Option<Raised> {
    match v {
        Value::MOVED => Some(th.raise("OwnershipError", "use of a moved value")),
        Value::UNINIT => Some(th.raise("NameError", "use of an uninitialized value")),
        _ => None,
    }
}

fn binop_type_error(th: &mut Thread, op: &str, a: Value, b: Value) -> Raised {
    if let Some(r) = dead_operand(th, a).or_else(|| dead_operand(th, b)) {
        return r;
    }
    th.raise(
        "TypeError",
        format!("unsupported operand type(s) for {op}: '{}' and '{}'", a.type_name(), b.type_name()),
    )
}

/// `a op b` on an instance through its operator method (`__add__`, ...),
/// if it has one.
fn dunder_binop(th: &mut Thread, op: &str, a: Value, b: Value) -> Option<VResult<Value>> {
    let name = match op {
        "+" => "__add__",
        "-" => "__sub__",
        "*" => "__mul__",
        "/" => "__truediv__",
        "//" => "__floordiv__",
        "%" => "__mod__",
        "**" => "__pow__",
        _ => return None,
    };
    let m = dunder(th, a, name)?;
    Some(call_simple(th, m, Some(a), &[b]))
}

fn overflow(th: &mut Thread, op: &str) -> Raised {
    th.raise("OverflowError", format!("integer overflow in `{op}` (int is 64-bit; use bigint)"))
}

/// Arithmetic on two numbers with int/bigint/float promotion.
fn arith(
    th: &mut Thread,
    op: &str,
    a: Value,
    b: Value,
    int: fn(i64, i64) -> Option<i64>,
    big: fn(&BigInt, &BigInt) -> BigInt,
    float: fn(f64, f64) -> f64,
) -> VResult<Value> {
    let (Some(x), Some(y)) = (num(a), num(b)) else {
        if let Some(r) = dunder_binop(th, op, a, b) {
            return r;
        }
        return Err(binop_type_error(th, op, a, b));
    };
    match (&x, &y) {
        (Num::Int(i), Num::Int(j)) => match int(*i, *j) {
            Some(r) => Ok(Value::from_i64(r)),
            None => Err(overflow(th, op)),
        },
        (Num::Float(_), _) | (_, Num::Float(_)) => Ok(Value::float(float(to_f64(&x), to_f64(&y)))),
        _ => Ok(big_value(big(&to_big(&x), &to_big(&y)))),
    }
}

pub extern "C" fn add(th: &mut Thread, a: Value, b: Value) -> Value {
    if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
        let mut s = String::with_capacity(x.len() + y.len());
        s.push_str(x);
        s.push_str(y);
        return Value::str(s);
    }
    match (a.body(), b.body()) {
        (Some(Body::List(x)), Some(Body::List(y))) => {
            // The result has the left operand's type; the right operand's
            // elements are checked against it unless the types are equal.
            let mut out: Vec<Value> = x.snapshot();
            out.extend(y.snapshot());
            out.iter().for_each(|v| v.incref());
            if x.ty == y.ty {
                return Obj::list(x.ty, out);
            }
            return vres(types::new_list(th, x.ty, out));
        }
        (Some(Body::Tuple(x)), Some(Body::Tuple(y))) => {
            let out: Vec<Value> = x.iter().chain(y.iter()).copied().collect();
            out.iter().for_each(|v| v.incref());
            return Obj::alloc(Body::Tuple(out.into_boxed_slice()));
        }
        _ => {}
    }
    vres(arith(th, "+", a, b, i64::checked_add, |x, y| x + y, |x, y| x + y))
}

pub extern "C" fn sub(th: &mut Thread, a: Value, b: Value) -> Value {
    vres(arith(th, "-", a, b, i64::checked_sub, |x, y| x - y, |x, y| x - y))
}

pub extern "C" fn mul(th: &mut Thread, a: Value, b: Value) -> Value {
    // Sequence repetition.
    let rep = |s: Value, n: Value| -> Option<Value> {
        let n = n.as_int()?.max(0) as usize;
        if let Some(st) = s.as_str() {
            return Some(Value::str(st.repeat(n)));
        }
        if let Some(Body::List(l)) = s.body() {
            let items = l.snapshot();
            let mut out = Vec::with_capacity(items.len() * n);
            for _ in 0..n {
                out.extend(items.iter().copied());
            }
            out.iter().for_each(|v| v.incref());
            return Some(Obj::list(l.ty, out));
        }
        None
    };
    if let Some(r) = rep(a, b).or_else(|| rep(b, a)) {
        return r;
    }
    vres(arith(th, "*", a, b, i64::checked_mul, |x, y| x * y, |x, y| x * y))
}

pub extern "C" fn div(th: &mut Thread, a: Value, b: Value) -> Value {
    let (Some(x), Some(y)) = (num(a), num(b)) else {
        if let Some(r) = dunder_binop(th, "/", a, b) {
            return vres(r);
        }
        return vres(Err(binop_type_error(th, "/", a, b)));
    };
    let d = to_f64(&y);
    if d == 0.0 {
        return vres(Err(th.raise("ZeroDivisionError", "division by zero")));
    }
    Value::float(to_f64(&x) / d)
}

fn int_zero(n: &Num) -> bool {
    match n {
        Num::Int(i) => *i == 0,
        Num::Big(b) => b.is_zero(),
        Num::Float(f) => *f == 0.0,
    }
}

pub extern "C" fn floordiv(th: &mut Thread, a: Value, b: Value) -> Value {
    if let Some(y) = num(b) {
        if int_zero(&y) && num(a).is_some() {
            return vres(Err(th.raise("ZeroDivisionError", "integer division or modulo by zero")));
        }
    }
    vres(arith(
        th,
        "//",
        a,
        b,
        |x, y| {
            let q = x.checked_div(y)?;
            Some(if (x % y != 0) && ((x < 0) != (y < 0)) { q - 1 } else { q })
        },
        |x, y| {
            use num_integer::Integer;
            x.div_floor(y)
        },
        |x, y| (x / y).floor(),
    ))
}

pub extern "C" fn modulo(th: &mut Thread, a: Value, b: Value) -> Value {
    if let Some(y) = num(b) {
        if int_zero(&y) && num(a).is_some() {
            return vres(Err(th.raise("ZeroDivisionError", "integer division or modulo by zero")));
        }
    }
    vres(arith(
        th,
        "%",
        a,
        b,
        |x, y| {
            let r = x.checked_rem(y)?;
            Some(if r != 0 && ((r < 0) != (y < 0)) { r + y } else { r })
        },
        |x, y| {
            use num_integer::Integer;
            x.mod_floor(y)
        },
        |x, y| {
            let r = x % y;
            if r != 0.0 && ((r < 0.0) != (y < 0.0)) {
                r + y
            } else {
                r
            }
        },
    ))
}

pub extern "C" fn pow(th: &mut Thread, a: Value, b: Value) -> Value {
    let (Some(x), Some(y)) = (num(a), num(b)) else {
        if let Some(r) = dunder_binop(th, "**", a, b) {
            return vres(r);
        }
        return vres(Err(binop_type_error(th, "**", a, b)));
    };
    match (&x, &y) {
        (Num::Int(i), Num::Int(j)) if *j >= 0 => {
            let Ok(e) = u32::try_from(*j) else { return vres(Err(overflow(th, "**"))) };
            match i.checked_pow(e) {
                Some(r) => Value::from_i64(r),
                None => vres(Err(overflow(th, "**"))),
            }
        }
        (Num::Big(bx), Num::Int(j)) if *j >= 0 => big_value(bx.pow(*j as u32)),
        _ => Value::float(to_f64(&x).powf(to_f64(&y))),
    }
}

fn int_bitop(th: &mut Thread, op: &str, a: Value, b: Value, f: fn(i64, i64) -> Option<i64>) -> Value {
    if let (Some(x), Some(y)) = (a.as_bool(), b.as_bool()) {
        let r = f(x as i64, y as i64).unwrap_or(0);
        return Value::bool(r != 0);
    }
    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match f(x, y) {
            Some(r) => Value::from_i64(r),
            None => vres(Err(overflow(th, op))),
        },
        _ => vres(Err(binop_type_error(th, op, a, b))),
    }
}

pub extern "C" fn bitand(th: &mut Thread, a: Value, b: Value) -> Value {
    int_bitop(th, "&", a, b, |x, y| Some(x & y))
}

pub extern "C" fn bitor(th: &mut Thread, a: Value, b: Value) -> Value {
    int_bitop(th, "|", a, b, |x, y| Some(x | y))
}

pub extern "C" fn bitxor(th: &mut Thread, a: Value, b: Value) -> Value {
    int_bitop(th, "^", a, b, |x, y| Some(x ^ y))
}

pub extern "C" fn shl(th: &mut Thread, a: Value, b: Value) -> Value {
    int_bitop(th, "<<", a, b, |x, y| {
        if !(0..64).contains(&y) {
            return None;
        }
        let r = x.checked_shl(y as u32)?;
        (r >> y == x).then_some(r)
    })
}

pub extern "C" fn shr(th: &mut Thread, a: Value, b: Value) -> Value {
    int_bitop(th, ">>", a, b, |x, y| if y < 0 { None } else { Some(x >> y.min(63)) })
}

pub extern "C" fn neg(th: &mut Thread, a: Value) -> Value {
    match num(a) {
        Some(Num::Int(i)) => match i.checked_neg() {
            Some(r) => Value::from_i64(r),
            None => vres(Err(overflow(th, "-"))),
        },
        Some(Num::Big(b)) => big_value(-b),
        Some(Num::Float(f)) => Value::float(-f),
        None => match dunder(th, a, "__neg__") {
            Some(m) => vres(call_simple(th, m, Some(a), &[])),
            None => vres(Err(unary_error(th, "-", a))),
        },
    }
}

pub extern "C" fn pos(th: &mut Thread, a: Value) -> Value {
    if num(a).is_some() {
        a.incref();
        return a;
    }
    vres(Err(unary_error(th, "+", a)))
}

pub extern "C" fn invert(th: &mut Thread, a: Value) -> Value {
    match a.as_int() {
        Some(i) => Value::from_i64(!i),
        None => vres(Err(unary_error(th, "~", a))),
    }
}

fn unary_error(th: &mut Thread, op: &str, a: Value) -> Raised {
    if let Some(r) = dead_operand(th, a) {
        return r;
    }
    th.raise("TypeError", format!("bad operand type for unary {op}: '{}'", a.type_name()))
}

pub fn is_truthy(th: &mut Thread, v: Value) -> VResult<bool> {
    if let Some(i) = v.as_small_int() {
        return Ok(i != 0);
    }
    match v {
        Value::TRUE => return Ok(true),
        Value::FALSE | Value::NONE | Value::UNIT => return Ok(false),
        Value::MOVED | Value::UNINIT => return Err(dead_operand(th, v).unwrap()),
        _ => {}
    }
    Ok(match v.body() {
        Some(Body::Int(i)) => *i != 0,
        Some(Body::Float(f)) => *f != 0.0,
        Some(Body::BigInt(b)) => !b.is_zero(),
        Some(Body::Str(s)) => !s.is_empty(),
        Some(Body::Tuple(t)) => !t.is_empty(),
        Some(Body::List(l)) => !l.is_empty(),
        Some(Body::Array(a)) => !a.is_empty(),
        Some(Body::Dict(d)) => !d.is_empty(),
        _ => true,
    })
}

pub extern "C" fn truthy(th: &mut Thread, v: Value) -> Value {
    match is_truthy(th, v) {
        Ok(b) => Value::bool(b),
        Err(_) => Value::UNINIT,
    }
}

// ---------------------------------------------------------------- comparisons

pub fn values_eq(a: Value, b: Value) -> bool {
    if a == b {
        return a.as_float().is_none_or(|f| !f.is_nan());
    }
    if let (Some(x), Some(y)) = (num(a), num(b)) {
        return match (&x, &y) {
            (Num::Int(i), Num::Int(j)) => i == j,
            (Num::Float(_), _) | (_, Num::Float(_)) => to_f64(&x) == to_f64(&y),
            _ => to_big(&x) == to_big(&y),
        };
    }
    match (a.body(), b.body()) {
        (Some(Body::Str(x)), Some(Body::Str(y))) => x == y,
        (Some(Body::Tuple(x)), Some(Body::Tuple(y))) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| values_eq(*p, *q))
        }
        (Some(Body::List(x)), Some(Body::List(y))) => {
            if std::ptr::eq(x, y) {
                return true;
            }
            let (x, y) = (x.snapshot(), y.snapshot());
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| values_eq(*p, *q))
        }
        (Some(Body::Array(x)), Some(Body::Array(y))) => {
            x.len() == y.len()
                && (0..x.len()).all(|i| match (x.get_num(i), y.get_num(i)) {
                    (Some(Scalar::Int(p)), Some(Scalar::Int(q))) => p == q,
                    (Some(Scalar::Bool(p)), Some(Scalar::Bool(q))) => p == q,
                    (Some(Scalar::Float(p)), Some(Scalar::Float(q))) => p == q,
                    (Some(Scalar::Float(p)), Some(Scalar::Int(q))) | (Some(Scalar::Int(q)), Some(Scalar::Float(p))) => p == q as f64,
                    _ => false,
                })
        }
        _ => false,
    }
}

/// Ordering for `<` and friends: numbers, strings, tuples and lists.
fn order(th: &mut Thread, op: &str, a: Value, b: Value) -> VResult<std::cmp::Ordering> {
    use std::cmp::Ordering as O;
    if let (Some(x), Some(y)) = (num(a), num(b)) {
        return Ok(match (&x, &y) {
            (Num::Int(i), Num::Int(j)) => i.cmp(j),
            (Num::Float(_), _) | (_, Num::Float(_)) => to_f64(&x).partial_cmp(&to_f64(&y)).unwrap_or(O::Less),
            _ => to_big(&x).cmp(&to_big(&y)),
        });
    }
    match (a.body(), b.body()) {
        (Some(Body::Str(x)), Some(Body::Str(y))) => Ok(x.cmp(y)),
        (Some(Body::Tuple(x)), Some(Body::Tuple(y))) => seq_order(th, op, x, y),
        (Some(Body::List(x)), Some(Body::List(y))) => {
            let (x, y) = (x.snapshot(), y.snapshot());
            seq_order(th, op, &x, &y)
        }
        // Instances of a class implementing `Ord` compare with `__lt__`.
        (Some(Body::Instance(_) | Body::Native(_)), _) if dunder(th, a, "__lt__").is_some() => {
            let m = dunder(th, a, "__lt__").unwrap();
            let less = call_simple(th, m, Some(a), &[b])?;
            let is_less = is_truthy(th, less);
            less.decref();
            if is_less? {
                return Ok(O::Less);
            }
            let greater = match dunder(th, b, "__lt__") {
                Some(m2) => {
                    let r = call_simple(th, m2, Some(b), &[a])?;
                    let g = is_truthy(th, r);
                    r.decref();
                    g?
                }
                None => false,
            };
            Ok(if greater { O::Greater } else { O::Equal })
        }
        _ => {
            if let Some(r) = dead_operand(th, a).or_else(|| dead_operand(th, b)) {
                return Err(r);
            }
            Err(th.raise(
                "TypeError",
                format!("'{op}' not supported between instances of '{}' and '{}'", a.type_name(), b.type_name()),
            ))
        }
    }
}

fn seq_order(th: &mut Thread, op: &str, x: &[Value], y: &[Value]) -> VResult<std::cmp::Ordering> {
    for (p, q) in x.iter().zip(y.iter()) {
        if !values_eq(*p, *q) {
            return order(th, op, *p, *q);
        }
    }
    Ok(x.len().cmp(&y.len()))
}

pub extern "C" fn lt(th: &mut Thread, a: Value, b: Value) -> Value {
    vres(order(th, "<", a, b).map(|o| Value::bool(o.is_lt())))
}

pub extern "C" fn le(th: &mut Thread, a: Value, b: Value) -> Value {
    vres(order(th, "<=", a, b).map(|o| Value::bool(o.is_le())))
}

pub extern "C" fn gt(th: &mut Thread, a: Value, b: Value) -> Value {
    vres(order(th, ">", a, b).map(|o| Value::bool(o.is_gt())))
}

pub extern "C" fn ge(th: &mut Thread, a: Value, b: Value) -> Value {
    vres(order(th, ">=", a, b).map(|o| Value::bool(o.is_ge())))
}

pub extern "C" fn eq(th: &mut Thread, a: Value, b: Value) -> Value {
    if let Some(r) = dead_operand(th, a).or_else(|| dead_operand(th, b)) {
        let _ = r;
        return Value::UNINIT;
    }
    Value::bool(values_eq(a, b))
}

pub extern "C" fn ne(th: &mut Thread, a: Value, b: Value) -> Value {
    if let Some(r) = dead_operand(th, a).or_else(|| dead_operand(th, b)) {
        let _ = r;
        return Value::UNINIT;
    }
    Value::bool(!values_eq(a, b))
}

fn contains_impl(th: &mut Thread, item: Value, container: Value) -> VResult<bool> {
    match container.body() {
        Some(Body::Str(s)) => match item.as_str() {
            Some(sub) => Ok(s.contains(sub)),
            None => Err(th.raise("TypeError", "'in <string>' requires string as left operand")),
        },
        Some(Body::Tuple(t)) => Ok(t.iter().any(|v| values_eq(*v, item))),
        Some(Body::List(l)) => {
            Ok(l.snapshot().iter().any(|v| values_eq(*v, item)))
        }
        Some(Body::Dict(d)) => Ok(d.get(item).is_some()),
        Some(Body::Array(a)) => Ok((0..a.len()).any(|i| match a.get(i) {
            Some(Ok(v)) => {
                let eq = values_eq(v, item);
                v.decref();
                eq
            }
            _ => false,
        })),
        _ => Err(th.raise("TypeError", format!("argument of type '{}' is not iterable", container.type_name()))),
    }
}

pub extern "C" fn contains(th: &mut Thread, a: Value, b: Value) -> Value {
    vres(contains_impl(th, a, b).map(Value::bool))
}

pub extern "C" fn not_contains(th: &mut Thread, a: Value, b: Value) -> Value {
    vres(contains_impl(th, a, b).map(|r| Value::bool(!r)))
}

// ---------------------------------------------------------------- ownership

fn reg_desc(code: &Code, r: Reg) -> String {
    match code.reg_names.get(r as usize).and_then(|n| n.as_deref()) {
        Some(n) => format!("`{n}`"),
        None => "value".into(),
    }
}

fn raise_dead(th: &mut Thread, frame: &Frame, r: Reg) -> Raised {
    let v = frame.reg(r);
    let what = reg_desc(frame.code(), r);
    if v == Value::MOVED {
        th.raise("OwnershipError", format!("{what} was moved and can no longer be used"))
    } else {
        th.raise("NameError", format!("{what} is used before it is assigned"))
    }
}

pub extern "C" fn f_dead(th: &mut Thread, frame: &mut Frame, r: u32, _: u32, _: u32, _: u32) -> u32 {
    raise_dead(th, frame, r as Reg);
    RAISED
}

pub extern "C" fn f_move_dyn(th: &mut Thread, frame: &mut Frame, dst: u32, src: u32, _: u32, _: u32) -> u32 {
    let v = frame.reg(src as Reg);
    if v.is_dead() {
        raise_dead(th, frame, src as Reg);
        return RAISED;
    }
    if v.is_copy() {
        v.incref();
    } else {
        frame.set(src as Reg, Value::MOVED);
    }
    frame.set(dst as Reg, v);
    OK
}

pub extern "C" fn f_own_dyn(th: &mut Thread, frame: &mut Frame, dst: u32, src: u32, what: u32, _: u32) -> u32 {
    let v = frame.reg(src as Reg);
    if v.is_dead() {
        raise_dead(th, frame, src as Reg);
        return RAISED;
    }
    if !v.is_copy() {
        let what = &frame.code().names[what as usize];
        th.raise(
            "OwnershipError",
            format!("cannot take ownership of {what} (a borrowed {}); use `++` to create a new reference", v.type_name()),
        );
        return RAISED;
    }
    v.incref();
    frame.set(dst as Reg, v);
    OK
}

// ---------------------------------------------------------------- globals

pub extern "C" fn f_load_global(th: &mut Thread, frame: &mut Frame, dst: u32, idx: u32, _: u32, _: u32) -> u32 {
    let v = th.vm.global(idx);
    if v.is_dead() {
        let name = &th.vm.global_names[idx as usize];
        if v == Value::MOVED {
            th.raise("OwnershipError", format!("global `{name}` was moved and can no longer be used"));
        } else {
            th.raise("NameError", format!("name '{name}' is not defined"));
        }
        return RAISED;
    }
    frame.set(dst as Reg, v);
    OK
}

pub extern "C" fn f_take_global(th: &mut Thread, frame: &mut Frame, dst: u32, idx: u32, _: u32, _: u32) -> u32 {
    let slot = &th.vm.globals[idx as usize];
    let v = Value(slot.load(Ordering::Acquire));
    if v.is_dead() {
        return f_load_global(th, frame, dst, idx, 0, 0);
    }
    if v.is_copy() {
        v.incref();
    } else {
        slot.store(Value::MOVED.0, Ordering::Release);
    }
    frame.set(dst as Reg, v);
    OK
}

pub extern "C" fn f_store_global(th: &mut Thread, frame: &mut Frame, idx: u32, src: u32, _: u32, _: u32) -> u32 {
    let v = frame.reg(src as Reg);
    if v.is_local() {
        // Module variables are shared by all threads.
        th.raise(
            "TypeError",
            format!(
                "cannot store `{}`, which is not `Send`, in module variable `{}` (module variables are shared between threads)",
                types::value_type_name(v),
                th.vm.global_names[idx as usize]
            ),
        );
        return RAISED;
    }
    frame.set(src as Reg, Value::MOVED);
    let old = Value(th.vm.globals[idx as usize].swap(v.0, Ordering::AcqRel));
    rc::defer_decref(old);
    OK
}

pub extern "C" fn f_load_capture(_th: &mut Thread, frame: &mut Frame, dst: u32, idx: u32, _: u32, _: u32) -> u32 {
    let v = unsafe { *frame.captures.add(idx as usize) };
    frame.set(dst as Reg, v);
    OK
}

// ---------------------------------------------------------------- calls

/// One argument as seen by the callee.
#[derive(Clone, Copy)]
pub struct CallArg {
    pub v: Value,
    /// Module variable the value was borrowed from (a `keep` parameter moves it out).
    pub global: Option<u32>,
    /// Caller owns the value (temporary or owned local).
    pub owned: bool,
    /// Caller register the value came from (for ownership transfer).
    pub reg: Option<Reg>,
    pub root: Option<Reg>,
}

impl CallArg {
    pub fn borrowed(v: Value) -> CallArg {
        CallArg { v, owned: false, reg: None, root: None, global: None }
    }

    pub fn owned(v: Value) -> CallArg {
        CallArg { v, owned: true, reg: None, root: None, global: None }
    }
}

/// Registers of a callee with at most this many live in a buffer on the
/// native stack (the interpreter threads have large stacks).
const INLINE_REGS: usize = 24;

const INLINE_ARGS: usize = 6;

/// A call's arguments, inline for short argument lists.
struct ArgBuf {
    len: usize,
    inline: [CallArg; INLINE_ARGS],
    heap: Vec<CallArg>,
}

impl std::ops::Deref for ArgBuf {
    type Target = [CallArg];
    fn deref(&self) -> &[CallArg] {
        if self.len <= INLINE_ARGS {
            &self.inline[..self.len]
        } else {
            &self.heap
        }
    }
}

fn collect_args(frame: &Frame, args: &[Arg]) -> ArgBuf {
    let arg = |a: &Arg| CallArg {
        v: frame.reg(a.reg),
        owned: a.owned,
        reg: Some(a.reg),
        root: a.root,
        global: (a.global != u32::MAX).then_some(a.global),
    };
    let mut b = ArgBuf { len: args.len(), inline: [CallArg::borrowed(Value::UNINIT); INLINE_ARGS], heap: Vec::new() };
    if args.len() <= INLINE_ARGS {
        for (i, a) in args.iter().enumerate() {
            b.inline[i] = arg(a);
        }
    } else {
        b.heap = args.iter().map(arg).collect();
    }
    b
}

pub fn callable_name(f: Value) -> String {
    match f.body() {
        Some(Body::Function(func)) => format!("{}()", func.code.name),
        Some(Body::Builtin(b)) => format!("{}()", b.name),
        Some(Body::Class(c)) => format!("{}()", c.name),
        Some(Body::BoundMethod { func, .. }) => callable_name(*func),
        _ => "function".into(),
    }
}

/// Moves or copies an argument into a `keep` parameter. Returns the owned
/// value for the callee.
fn take_for_keep(
    th: &mut Thread,
    caller: Option<&mut Frame>,
    args: &[CallArg],
    i: usize,
    recv_reg: Option<Reg>,
    pname: &str,
    fname: &str,
) -> VResult<Value> {
    let a = args[i];
    if a.v.is_dead() {
        return Err(dead_operand(th, a.v).unwrap());
    }
    if a.owned && a.v.is_copy() && a.reg.is_some() {
        // Copy types are copied, the caller's value stays usable.
        a.v.incref();
        return Ok(a.v);
    }
    if a.owned {
        if let (Some(r), Some(frame)) = (a.reg, caller) {
            // Moving a local into a call that also borrows it would let the
            // callee free what it borrows.
            let conflict = recv_reg == Some(r)
                || args
                    .iter()
                    .enumerate()
                    .any(|(j, b)| j != i && (b.reg == Some(r) && !b.owned || b.root == Some(r)));
            if conflict {
                let what = reg_desc(frame.code(), r);
                return Err(th.raise(
                    "OwnershipError",
                    format!("{what} is moved into `{fname}` while the same call also borrows it; pass `{}++` instead", what.trim_matches('`')),
                ));
            }
            frame.set(r, Value::MOVED);
        }
        return Ok(a.v);
    }
    if a.v.is_copy() {
        a.v.incref();
        return Ok(a.v);
    }
    if let Some(g) = a.global {
        // A module variable passed to a `keep` parameter moves out of its slot.
        let slot = &th.vm.globals[g as usize];
        if slot.compare_exchange(a.v.0, Value::MOVED.0, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            return Ok(a.v);
        }
    }
    Err(th.raise(
        "OwnershipError",
        format!(
            "parameter `{pname}` of `{fname}` takes ownership (`keep`), but a borrowed {} was passed; use `++` to pass a new reference",
            a.v.type_name()
        ),
    ))
}

/// Calls `f` with `self_val` (borrowed) and `args`.
pub fn invoke(
    th: &mut Thread,
    mut caller: Option<&mut Frame>,
    f: Value,
    self_val: Option<Value>,
    recv_reg: Option<Reg>,
    args: &[CallArg],
) -> VResult<Value> {
    match f.body() {
        Some(Body::Function(func)) => {
            let code = &func.code;
            let nself = self_val.is_some() as usize;
            let nt = code.tparams.len();
            let expected = code.params.len();
            // Type arguments are hidden leading parameters; a dynamic call
            // passes none and they are inferred from the arguments.
            let off = if nt > 0 && args.len() + nself + nt == expected { nt } else { 0 };
            if args.len() + nself + off != expected {
                let given = args.len().saturating_sub(nt - off);
                let want = expected - nself - nt;
                return Err(th.raise(
                    "TypeError",
                    format!("{} takes {want} argument{} but {given} {} given", callable_name(f), if want == 1 { "" } else { "s" }, if given == 1 { "was" } else { "were" }),
                ));
            }
            let n = code.nregs as usize;
            let mut inline = [Value::UNINIT; INLINE_REGS];
            let mut heap = Vec::new();
            let regs: &mut [Value] = if n <= INLINE_REGS && !code.is_generator {
                &mut inline[..n]
            } else {
                heap = vec![Value::UNINIT; n];
                &mut heap
            };
            if let Some(s) = self_val {
                regs[0] = s;
            }
            for (i, _) in args.iter().enumerate() {
                let p = &code.params[i + nself + off];
                regs[i + nself + off] = if p.keep {
                    take_for_keep(th, caller.as_deref_mut(), args, i, recv_reg, &p.name, &code.name)?
                } else {
                    let v = args[i].v;
                    if v.is_dead() {
                        return Err(dead_operand(th, v).unwrap());
                    }
                    v
                };
            }
            if nt > 0 {
                if let Err(e) = bind_type_args(th, code, regs, nself) {
                    // Values moved into `keep` parameters are released.
                    for (i, p) in code.params.iter().enumerate() {
                        if p.keep {
                            regs[i].decref();
                        }
                    }
                    return Err(e);
                }
            }
            if code.is_generator {
                return crate::generator::create(th, f, heap);
            }
            run_code(th, code, regs, func.captures.as_ptr())
        }
        Some(Body::Builtin(b)) => {
            let nargs = args.len() + self_val.is_some() as usize;
            if nargs < b.min_args || nargs > b.max_args {
                let msg = if b.min_args == b.max_args {
                    format!("{}() takes {} argument{} ({} given)", b.name, b.min_args, if b.min_args == 1 { "" } else { "s" }, nargs)
                } else {
                    format!("{}() takes {} to {} arguments ({} given)", b.name, b.min_args, b.max_args, nargs)
                };
                return Err(th.raise("TypeError", msg));
            }
            let mut vals = Vec::with_capacity(nargs);
            if let Some(s) = self_val {
                vals.push(s);
            }
            for i in 0..args.len() {
                let pi = i + self_val.is_some() as usize;
                if b.keep_mask & (1 << pi) != 0 {
                    vals.push(take_for_keep(th, caller.as_deref_mut(), args, i, recv_reg, &format!("#{pi}"), b.name)?);
                } else {
                    if args[i].v.is_dead() {
                        return Err(dead_operand(th, args[i].v).unwrap());
                    }
                    vals.push(args[i].v);
                }
            }
            (b.f)(th, &vals)
        }
        Some(Body::Class(c)) => {
            if !c.tparams.is_empty() {
                return Err(th.raise(
                    "TypeError",
                    format!("cannot infer the type arguments of `{}`; write `{}::<{}>(...)`", c.name, c.name, c.tparams.join(", ")),
                ));
            }
            construct(th, caller, f, c, rtype::class_type(f), args)
        }
        Some(Body::Type(t)) if matches!(t.ctor, Ctor::Class(c) if c.as_class().is_some()) => {
            let Ctor::Class(cls) = t.ctor else { unreachable!() };
            let c = cls.as_class().unwrap();
            if t.args.len() != c.tparams.len() {
                return Err(th.raise("TypeError", format!("cannot infer the type arguments of `{}`", t.name)));
            }
            construct(th, caller, cls, c, t.value(), args)
        }
        Some(Body::NativeFn(nf)) => {
            let mut vals = Vec::with_capacity(args.len() + 1);
            vals.extend(self_val);
            vals.extend(args.iter().map(|a| a.v));
            crate::native::call(th, nf, &vals)
        }
        Some(Body::NativeClass(c)) if c.is_family() => Err(th.raise(
            "TypeError",
            format!("cannot infer the type arguments of `{}`; write `{}::<{}>(...)`", c.name, c.name, c.tparams.join(", ")),
        )),
        Some(Body::NativeClass(c)) => match c.init {
            Some(init) => invoke(th, caller, init, None, None, args),
            None => Err(th.raise("TypeError", format!("cannot create `{}` objects from slim", c.name))),
        },
        // `Stack::<int>(...)` of a generic native class: the instantiation
        // registered for these type arguments.
        Some(Body::Type(t)) if matches!(t.ctor, Ctor::Class(c) if matches!(c.body(), Some(Body::NativeClass(_)))) => {
            let Ctor::Class(fv) = t.ctor else { unreachable!() };
            let Some(Body::NativeClass(fam)) = fv.body() else { unreachable!() };
            if !fam.is_family() {
                return invoke(th, caller, fv, None, None, args);
            }
            match fam.instance_for(t.value()) {
                Some(c) => invoke(th, caller, c, None, None, args),
                None => Err(th.raise("TypeError", format!("`{}` is not available: `{}` supports {}", t.name, fam.name, fam.instance_names()))),
            }
        }
        Some(Body::Instantiated { func, targs }) => {
            let mut full: Vec<CallArg> = targs.iter().map(|t| CallArg::borrowed(*t)).collect();
            full.extend_from_slice(args);
            invoke(th, caller, *func, self_val, recv_reg, &full)
        }
        Some(Body::Type(t)) => {
            let vals: Vec<Value> = args.iter().map(|a| a.v).collect();
            if let Some(v) = vals.iter().find(|v| v.is_dead()) {
                return Err(dead_operand(th, *v).unwrap());
            }
            crate::builtins::construct_type(th, t, &vals)
        }
        Some(Body::BoundMethod { recv, func }) => invoke(th, caller, *func, Some(*recv), None, args),
        _ => {
            if let Some(r) = dead_operand(th, f) {
                return Err(r);
            }
            Err(th.raise("TypeError", format!("'{}' object is not callable", f.type_name())))
        }
    }
}

/// Infers the type arguments a call left out (`UNINIT`) from the runtime
/// types of the arguments, then checks every type argument's bounds.
fn bind_type_args(th: &mut Thread, code: &Code, regs: &mut [Value], nself: usize) -> VResult<()> {
    let nt = code.tparams.len();
    if regs[nself..nself + nt].contains(&Value::UNINIT) {
        for (j, p) in code.params.iter().enumerate().skip(nself + nt) {
            if let (Some(d), v) = (&p.ty, regs[j]) {
                if v != Value::NONE {
                    unify(th, d, rtype::as_type(rtype::type_of(v)).unwrap(), regs);
                }
            }
        }
    }
    for (k, tp) in code.tparams.iter().enumerate() {
        let t = regs[nself + k];
        if t == Value::UNINIT {
            let names: Vec<&str> = code.tparams.iter().map(|t| t.name.as_str()).collect();
            return Err(th.raise(
                "TypeError",
                format!("cannot infer type argument `{}` of `{}()`; write `{}::<{}>(...)`", tp.name, code.name, code.name, names.join(", ")),
            ));
        }
        for b in &tp.bounds {
            let bt = match types::materialize(th, b, &|r| regs[r as usize]) {
                Ok(bt) => rtype::as_type(bt).unwrap(),
                Err(n) => return Err(th.raise("NameError", format!("type `{n}` is used before it is defined"))),
            };
            if !rtype::satisfies(rtype::as_type(t).unwrap(), bt) {
                return Err(th.raise(
                    "TypeError",
                    format!("`{}` does not implement `{}` (required by `{}` of `{}()`)", rtype::name(t), bt.name, tp.name, code.name),
                ));
            }
        }
    }
    Ok(())
}

/// Binds unbound type parameters in `d` by matching it against type `t`.
fn unify(th: &Thread, d: &TypeDesc, t: &'static rtype::RtType, regs: &mut [Value]) {
    match d {
        TypeDesc::Param(r, _) => {
            if regs[*r as usize] == Value::UNINIT {
                regs[*r as usize] = t.value();
            }
        }
        TypeDesc::Local(inner) => unify(th, inner, t, regs),
        TypeDesc::Builtin(Ctor::Option, args) if t.ctor != Ctor::Option => unify(th, &args[0], t, regs),
        TypeDesc::Builtin(c, args) => {
            if *c == t.ctor && args.len() == t.args.len() {
                for (a, ta) in args.iter().zip(t.args.iter()) {
                    unify(th, a, rtype::as_type(*ta).unwrap(), regs);
                }
            }
        }
        TypeDesc::Global(idx, _, args) if !args.is_empty() => {
            let g = th.vm.global(*idx);
            let target = match t.ctor {
                Ctor::Class(c) if c == g => Some(t),
                // A class implementing the generic trait: use its impl.
                Ctor::Class(_) => rtype::impls_of(t)
                    .iter()
                    .map(|i| rtype::as_type(*i).unwrap())
                    .find(|i| matches!(i.ctor, Ctor::Trait(x) if x == g)),
                _ => None,
            };
            match target {
                Some(tt) if tt.args.len() == args.len() => {
                    for (a, ta) in args.iter().zip(tt.args.iter()) {
                        unify(th, a, rtype::as_type(*ta).unwrap(), regs);
                    }
                }
                _ => {
                    // `Iterable<T>` / `Iterator<T>` over a builtin container.
                    let builtin_iter = g.as_trait().is_some_and(|tr| tr.name == "Iterable" || tr.name == "Iterator");
                    if builtin_iter && args.len() == 1 && !matches!(t.ctor, Ctor::Class(_)) {
                        let item = match t.ctor {
                            Ctor::Dict if t.args.len() == 2 => t.args[0],
                            Ctor::Str => rtype::str_(),
                            Ctor::Array(Some(e)) => rtype::elem_scalar(e),
                            _ => rtype::elem_of(t),
                        };
                        unify(th, &args[0], rtype::as_type(item).unwrap(), regs);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Calls with borrowed arguments (from Rust code).
pub fn call_simple(th: &mut Thread, f: Value, self_val: Option<Value>, args: &[Value]) -> VResult<Value> {
    let args: Vec<CallArg> = args.iter().map(|v| CallArg::borrowed(*v)).collect();
    invoke(th, None, f, self_val, None, &args)
}

pub fn new_exception(class: Value, msg: Value) -> Value {
    let c = class.as_class().unwrap();
    let slots: Vec<AtomicU64> = c.fields.iter().map(|_| AtomicU64::new(Value::UNINIT.0)).collect();
    let e = Obj::alloc(Body::Instance(Instance { class, ty: rtype::class_type(class), slots: slots.into_boxed_slice() }));
    if let Some(i) = c.field_index("msg") {
        e.as_instance().unwrap().slots[i].store(msg.0, Ordering::Release);
    } else {
        msg.decref();
    }
    e
}

/// Creates an instance of type `ty` (a type of class `c`) and runs `__init__`.
fn construct(th: &mut Thread, caller: Option<&mut Frame>, cls: Value, c: &'static Class, ty: Value, args: &[CallArg]) -> VResult<Value> {
    if c.is_exception && !c.methods.contains_key("__init__") && c.field_index("msg").is_some() && c.fields.len() == 1 {
        // Builtin-style exception: optional message.
        if args.len() > 1 {
            return Err(th.raise("TypeError", format!("{}() takes at most 1 argument", c.name)));
        }
        let msg = match args.first() {
            Some(a) => match a.v.as_str() {
                Some(_) => {
                    a.v.incref();
                    a.v
                }
                None => Value::str(display::to_str(a.v)),
            },
            None => Value::str(""),
        };
        return Ok(new_exception(cls, msg));
    }
    let _ = caller;
    // A struct without fields (`raise Done`); others need a literal.
    if c.fields.is_empty() && args.is_empty() {
        return Ok(Obj::alloc(Body::Instance(Instance { class: cls, ty, slots: Box::new([]) })));
    }
    Err(th.raise("TypeError", format!("`{}` is a struct: create it with a literal `{} {{ .. }}` or an associated function", c.name, c.name)))
}

pub extern "C" fn f_call(th: &mut Thread, frame: &mut Frame, dst: u32, func: u32, args: u32, _: u32) -> u32 {
    let code = frame.code();
    let al = &code.arglists[args as usize];
    let f = frame.reg(func as Reg);
    if al.fast {
        if let Some(status) = call_fast(th, frame, f, None, al, dst) {
            return status;
        }
    }
    let cargs = collect_args(frame, &al.args);
    if al.pins {
        rc::pin();
    }
    let r = invoke(th, Some(frame), f, None, None, &cargs);
    if al.pins {
        rc::unpin();
    }
    match r {
        Ok(v) => {
            frame.set(dst as Reg, v);
            OK
        }
        Err(Raised) => RAISED,
    }
}

/// An `ArgList::fast` call of `f` (with `recv` as `self` for a method):
/// the arguments' types are proven, so they are passed as they are and the
/// callee starts after its parameter checks. `None` if `f` is not a plain
/// slim function of that arity.
fn call_fast(th: &mut Thread, frame: &mut Frame, f: Value, recv: Option<Value>, al: &ArgList, dst: u32) -> Option<u32> {
    let Some(Body::Function(func)) = f.body() else { return None };
    let code = &func.code;
    let n = code.nregs as usize;
    let nself = recv.is_some() as usize;
    if n > INLINE_REGS || code.is_generator || code.params.len() != al.args.len() + nself || code.params.first().is_some_and(|p| p.keep) {
        return None;
    }
    let mut regs = [Value::UNINIT; INLINE_REGS];
    if let Some(r) = recv {
        regs[0] = r;
    }
    for (i, a) in al.args.iter().enumerate() {
        let v = frame.reg(a.reg);
        if v.is_dead() {
            raise_dead(th, frame, a.reg);
            return Some(RAISED);
        }
        regs[nself + i] = v;
    }
    if al.pins {
        rc::pin();
    }
    let r = run_code_at(th, code, &mut regs[..n], func.captures.as_ptr(), code.entry);
    if al.pins {
        rc::unpin();
    }
    Some(match r {
        Ok(v) => {
            frame.set(dst as Reg, v);
            OK
        }
        Err(Raised) => RAISED,
    })
}

/// Finds the method for `recv.name()`, as in Rust: an inherent method (of
/// the struct, the native class or the builtin type), then one of a
/// registered inherent impl, then the only trait method of that name (of the
/// struct's impls or registered ones). `Err` lists the traits when several
/// define it (empty when there is none).
pub fn find_method(th: &Thread, recv: Value, name: &str) -> Result<Value, Vec<Value>> {
    let class = recv.as_instance().map(|i| i.class());
    let own = match recv.body() {
        Some(Body::Instance(i)) => i.class().methods.get(name).copied(),
        Some(Body::Native(n)) => n.class().methods.get(name).copied(),
        _ => crate::builtins::type_method(recv, name),
    };
    if let Some(m) = own {
        return Ok(m);
    }
    let mut traits: Vec<(Value, Value)> = class.and_then(|c| c.trait_methods.get(name)).cloned().unwrap_or_default();
    let registry = th.vm.impls.read().unwrap();
    if let Some(impls) = registry.get(name).filter(|_| !recv.is_dead() && recv != Value::NONE) {
        let t = rtype::as_type(rtype::type_of(recv)).unwrap();
        for i in impls.iter().filter(|i| i.bind(t).is_some()) {
            let m = i.methods[name];
            match i.trait_ {
                None => return Ok(m),
                Some(tr) if !traits.iter().any(|(x, _)| *x == tr) => traits.push((tr, m)),
                Some(_) => {}
            }
        }
    }
    match traits.as_slice() {
        [(_, m)] => Ok(*m),
        _ => Err(traits.iter().map(|(t, _)| *t).collect()),
    }
}

/// `find_method`, without telling an ambiguous name from a missing one.
pub fn lookup_method(th: &Thread, recv: Value, name: &str) -> Option<Value> {
    find_method(th, recv, name).ok()
}

/// The method trait `tr` provides for `recv` (`Trait::name(recv)`): from the
/// struct's impl, a registered impl, or the type itself (a struct
/// implementing a trait that implies `tr`, a native class's Rust method, a
/// builtin type's method).
pub fn trait_method_for(th: &Thread, recv: Value, tr: Value, name: &str) -> Option<Value> {
    if let Some(m) = recv.as_instance().and_then(|i| i.class().trait_method(tr, name)) {
        return Some(m);
    }
    if let Some(t) = tr.as_trait() {
        let impls = t.impls.read().unwrap();
        if !impls.is_empty() && !recv.is_dead() && recv != Value::NONE {
            let ty = rtype::as_type(rtype::type_of(recv)).unwrap();
            if let Some(i) = impls.iter().find(|i| i.methods.contains_key(name) && i.bind(ty).is_some()) {
                return Some(i.methods[name]);
            }
        }
    }
    match recv.body() {
        Some(Body::Instance(i)) => i.class().implements(tr).then(|| lookup_method(th, recv, name)).flatten(),
        Some(Body::Native(n)) => n.class().methods.get(name).copied(),
        _ => crate::builtins::type_method(recv, name),
    }
}

/// An operator method (`__add__`, `__lt__`, `__len__`) of an instance or a
/// native object.
pub fn dunder(th: &Thread, v: Value, name: &str) -> Option<Value> {
    match v.body() {
        Some(Body::Instance(_) | Body::Native(_)) => lookup_method(th, v, name),
        _ => None,
    }
}

pub extern "C" fn f_call_method(th: &mut Thread, frame: &mut Frame, dst: u32, recv: u32, name: u32, args: u32) -> u32 {
    let code = frame.code();
    let al = &code.arglists[args as usize];
    let r = frame.reg(recv as Reg);
    let mname = &code.names[name as usize];
    if r.is_dead() {
        raise_dead(th, frame, recv as Reg);
        return RAISED;
    }
    let found = match r.as_instance().and_then(|inst| cached_method(frame, inst, mname)) {
        Some(m) => Ok(m),
        None => find_method(th, r, mname),
    };
    if let (true, Ok(m)) = (al.fast, &found) {
        if let Some(status) = call_fast(th, frame, *m, Some(r), al, dst) {
            return status;
        }
    }
    let cargs = collect_args(frame, &al.args);
    if al.pins {
        rc::pin();
    }
    let result = match found {
        Ok(m) => invoke(th, Some(frame), m, Some(r), Some(recv as Reg), &cargs),
        Err(traits) => {
            // A callable stored in a field: `self.callback(x)`.
            let field = r.as_instance().and_then(|inst| inst.class().field_index(mname).map(|i| inst.get_borrowed(i)));
            match field {
                Some(f) => invoke(th, Some(frame), f, None, None, &cargs),
                None if traits.len() > 1 => Err(ambiguous_method(th, r, mname, &traits)),
                None => Err(no_attribute(th, r, mname)),
            }
        }
    };
    if al.pins {
        rc::unpin();
    }
    match result {
        Ok(v) => {
            frame.set(dst as Reg, v);
            OK
        }
        Err(Raised) => RAISED,
    }
}

fn ambiguous_method(th: &mut Thread, v: Value, name: &str, traits: &[Value]) -> Raised {
    let names: Vec<String> = traits.iter().filter_map(|t| t.as_trait()).map(|t| format!("`{}::{name}`", t.name)).collect();
    let first = traits.first().and_then(|t| t.as_trait()).map(|t| t.name.as_str()).unwrap_or("Trait");
    th.raise(
        "TypeError",
        format!("`{name}` is ambiguous for `{}`: {}; call it as `{first}::{name}(x)`", v.type_name(), names.join(" or ")),
    )
}

/// `Trait::name(recv, args)`: the method trait `tr` provides for `recv`.
pub extern "C" fn f_call_trait_method(th: &mut Thread, frame: &mut Frame, dst: u32, recv: u32, call: u32, args: u32) -> u32 {
    let code = frame.code();
    let (tg, name) = code.trait_calls[call as usize];
    let mname = &code.names[name as usize];
    let al = &code.arglists[args as usize];
    let r = frame.reg(recv as Reg);
    let cargs = collect_args(frame, &al.args);
    if r.is_dead() {
        raise_dead(th, frame, recv as Reg);
        return RAISED;
    }
    let trv = th.vm.global(tg);
    let m = trait_method_for(th, r, trv, mname);
    let Some(m) = m else {
        let t = trv.as_trait().map(|t| t.name.clone()).unwrap_or_default();
        th.raise("TypeError", format!("`{}` has no method `{t}::{mname}`", r.type_name()));
        return RAISED;
    };
    if al.pins {
        rc::pin();
    }
    let result = invoke(th, Some(frame), m, Some(r), Some(recv as Reg), &cargs);
    if al.pins {
        rc::unpin();
    }
    match result {
        Ok(v) => {
            frame.set(dst as Reg, v);
            OK
        }
        Err(Raised) => RAISED,
    }
}

fn no_attribute(th: &mut Thread, v: Value, name: &str) -> Raised {
    if let Some(r) = dead_operand(th, v) {
        return r;
    }
    th.raise("AttributeError", format!("'{}' object has no attribute '{name}'", v.type_name()))
}

pub extern "C" fn f_safepoint(th: &mut Thread, _frame: &mut Frame, _: u32, _: u32, _: u32, _: u32) -> u32 {
    if rc::safepoint_pending() > th.rc_threshold && th.rc_auto {
        th.rc_run(Some(th.rc_threshold));
    }
    OK
}

pub extern "C" fn f_drop_frame(_th: &mut Thread, frame: &mut Frame, _: u32, _: u32, _: u32, _: u32) -> u32 {
    frame.drop_owned();
    OK
}

pub extern "C" fn f_yield(th: &mut Thread, frame: &mut Frame, dst: u32, src: u32, _: u32, _: u32) -> u32 {
    crate::generator::yield_op(th, frame, dst as Reg, src as Reg)
}

// ---------------------------------------------------------------- definitions

pub extern "C" fn f_make_function(_th: &mut Thread, frame: &mut Frame, dst: u32, code: u32, captures: u32, _: u32) -> u32 {
    let c = frame.code();
    let fcode = c.codes[code as usize].clone();
    let al = &c.arglists[captures as usize];
    if al.args.is_empty() && frame.captures.is_null() && c.name == "<module>" {
        // Module-level functions live forever.
        let v = Obj::alloc_immortal(Body::Function(Function { code: fcode, captures: Box::new([]) }));
        frame.set(dst as Reg, v);
        return OK;
    }
    let caps: Vec<Value> = al
        .args
        .iter()
        .map(|a| {
            let v = frame.reg(a.reg);
            frame.set(a.reg, Value::MOVED);
            v
        })
        .collect();
    let v = Obj::alloc(Body::Function(Function { code: fcode, captures: caps.into_boxed_slice() }));
    frame.set(dst as Reg, v);
    OK
}

fn resolve_global(th: &Thread, idx: u32) -> Value {
    th.vm.global(idx)
}

pub extern "C" fn f_make_trait(th: &mut Thread, frame: &mut Frame, dst: u32, spec: u32, _: u32, _: u32) -> u32 {
    let c = frame.code();
    let s = &c.traits[spec as usize];
    let methods = s
        .methods
        .iter()
        .map(|(n, d)| {
            let def = d.map(|i| Obj::alloc_immortal(Body::Function(Function { code: c.codes[i as usize].clone(), captures: Box::new([]) })));
            (n.clone(), def)
        })
        .collect();
    let id = th.vm.next_class_id.fetch_add(1, Ordering::Relaxed);
    let t = Obj::alloc_immortal(Body::Trait(Box::new(Trait { name: s.name.clone(), id, nparams: s.nparams, methods, implies: Vec::new(), impls: Default::default() })));
    frame.set(dst as Reg, t);
    OK
}

/// Registers `impls[spec]`: under its trait and the traits that trait
/// implies (for bounds and `Trait::name(x)`), and by method name (for
/// `x.name()`). A trait impl inherits the trait's default methods, except
/// those a native target's Rust class defines.
pub extern "C" fn f_make_impl(th: &mut Thread, frame: &mut Frame, spec: u32, _: u32, _: u32, _: u32) -> u32 {
    let c = frame.code();
    let s = &c.impls[spec as usize];
    let resolve = |d: &TypeDesc| types::resolve_globals(th, d);
    let parts = (|| -> Result<_, String> {
        let target = resolve(&s.target)?;
        let bounds = s.bounds.iter().map(|(k, b)| Ok((*k, resolve(b)?))).collect::<Result<Vec<_>, String>>()?;
        let trait_args = s.trait_args.iter().map(&resolve).collect::<Result<Vec<_>, _>>()?;
        Ok((target, bounds, trait_args))
    })();
    let (target, bounds, trait_args) = match parts {
        Ok(p) => p,
        Err(n) => {
            th.raise("NameError", format!("type `{n}` is used before it is defined"));
            return RAISED;
        }
    };
    let mut methods: HashMap<String, Value> = s
        .methods
        .iter()
        .map(|(n, ci)| (n.clone(), Obj::alloc_immortal(Body::Function(Function { code: c.codes[*ci as usize].clone(), captures: Box::new([]) }))))
        .collect();
    let trait_ = s.tr.map(|g| th.vm.global(g));
    let mut traits = Vec::new();
    if let Some(t) = trait_.and_then(|t| t.as_trait()) {
        traits.push(trait_.unwrap());
        traits.extend(t.implies.iter().copied());
    }
    // Rust methods of a native target implement the trait's methods.
    let rust: Vec<String> = match &target {
        TypeDesc::Builtin(Ctor::Class(k), _) => match k.body() {
            Some(Body::NativeClass(nc)) => {
                let all = if nc.is_family() { nc.instances.clone() } else { vec![*k] };
                all.iter().filter_map(|i| match i.body() {
                    Some(Body::NativeClass(x)) => Some(x.methods.keys().cloned().collect::<Vec<_>>()),
                    _ => None,
                }).flatten().collect()
            }
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    for x in &traits {
        for (m, d) in &x.as_trait().unwrap().methods {
            if let (Some(d), false) = (d, rust.contains(m)) {
                methods.entry(m.clone()).or_insert(*d);
            }
        }
    }
    let imp: &'static rtype::RtImpl =
        Box::leak(Box::new(rtype::RtImpl { name: s.name.clone(), target, nparams: s.nparams, bounds, trait_, trait_args, methods }));
    for x in &traits {
        x.as_trait().unwrap().impls.write().unwrap().push(imp);
    }
    let mut by_name = th.vm.impls.write().unwrap();
    for m in imp.methods.keys() {
        by_name.entry(m.clone()).or_default().push(imp);
    }
    OK
}

/// Type parameter `k` of the impl whose target is `code.types[ty]`, as the
/// type of `o` matches it.
pub fn impl_arg(th: &mut Thread, code: &Code, o: Value, ty: u32, k: u32) -> VResult<Value> {
    let d = match types::resolve_globals(th, &code.types[ty as usize]) {
        Ok(d) => d,
        Err(n) => return Err(th.raise("NameError", format!("type `{n}` is used before it is defined"))),
    };
    let t = rtype::as_type(rtype::type_of(o)).unwrap();
    let mut env = vec![Value::UNINIT; k as usize + 1];
    fn grow(d: &TypeDesc, n: &mut usize) {
        match d {
            TypeDesc::Param(k, _) => *n = (*n).max(*k as usize + 1),
            TypeDesc::Builtin(_, a) | TypeDesc::Global(_, _, a) => a.iter().for_each(|x| grow(x, n)),
            TypeDesc::Local(x) => grow(x, n),
            TypeDesc::Resolved(_) => {}
        }
    }
    let mut n = env.len();
    grow(&d, &mut n);
    env.resize(n, Value::UNINIT);
    if !rtype::match_desc(&d, t, &mut env) || env[k as usize] == Value::UNINIT {
        return Err(th.raise("TypeError", format!("`{}` does not match the impl's target `{}`", t.name, code.types[ty as usize])));
    }
    Ok(env[k as usize])
}

pub extern "C" fn f_load_impl_arg(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, ty: u32, k: u32) -> u32 {
    match impl_arg(th, frame.code(), frame.reg(obj as Reg), ty, k) {
        Ok(a) => {
            frame.set(dst as Reg, a);
            OK
        }
        Err(Raised) => RAISED,
    }
}

pub fn slot_kind(ty: &TypeDesc) -> SlotKind {
    if ty.is(Ctor::Int) {
        SlotKind::RawInt
    } else if ty.is(Ctor::Float) {
        SlotKind::RawFloat
    } else {
        SlotKind::Value
    }
}

pub fn build_class(th: &mut Thread, s: &ClassSpec, codes: &[Arc<Code>]) -> VResult<Value> {
    let mut traits = Vec::new();
    let mut trait_args = Vec::new();
    for (idx, tname, targs) in &s.traits {
        let t = resolve_global(th, *idx);
        let Some(tr) = t.as_trait() else {
            return Err(th.raise("TypeError", format!("struct `{}`: `{tname}` is not a trait", s.name)));
        };
        if !targs.is_empty() {
            let mut args = Vec::new();
            for a in targs {
                match types::pre_resolve(th, a) {
                    Ok(a) => args.push(a),
                    Err(n) => {
                        return Err(th.raise("NameError", format!("struct `{}`: type `{n}` is used before it is defined", s.name)))
                    }
                }
            }
            trait_args.push((t, args));
        }
        if !traits.contains(&t) {
            traits.push(t);
        }
        for imp in &tr.implies {
            if !traits.contains(imp) {
                traits.push(*imp);
            }
        }
    }
    let func = |ci: u32| Obj::alloc_immortal(Body::Function(Function { code: codes[ci as usize].clone(), captures: Box::new([]) }));
    let mut methods: HashMap<String, Value> = HashMap::new();
    for (name, ci) in &s.methods {
        methods.insert(name.clone(), func(*ci));
    }
    let mut trait_methods: HashMap<String, Vec<(Value, Value)>> = HashMap::new();
    for (tg, name, ci) in &s.trait_methods {
        trait_methods.entry(name.clone()).or_default().push((resolve_global(th, *tg), func(*ci)));
    }
    for t in &traits {
        let tr = t.as_trait().unwrap();
        for (mname, default) in &tr.methods {
            let entry = trait_methods.entry(mname.clone()).or_default();
            if entry.iter().any(|(x, _)| x == t) {
                continue;
            }
            match default {
                Some(d) => entry.push((*t, *d)),
                // Defined by the impl of a trait implying this one (`Num`
                // provides `Ord`'s `__lt__`).
                None if !entry.is_empty() || methods.contains_key(mname) => {}
                None => {
                    return Err(th.raise(
                        "TypeError",
                        format!("struct `{}` implements `{}` but does not define required method `{mname}`", s.name, tr.name),
                    ))
                }
            }
        }
    }
    let exc_trait = th.vm.builtins.exception;
    let is_exception = traits.contains(&exc_trait);
    let fields = s
        .fields
        .iter()
        .map(|f| Field::new(f.name.clone(), f.ty.clone(), slot_kind(&f.ty)))
        .collect();
    trait_methods.retain(|_, v| !v.is_empty());
    let always_local = s.local;
    let id = th.vm.next_class_id.fetch_add(1, Ordering::Relaxed);
    Ok(Obj::alloc_immortal(Body::Class(Box::new(Class {
        name: s.name.clone(),
        id,
        tparams: s.tparams.clone(),
        fields,
        methods,
        trait_methods,
        traits,
        trait_args,
        is_exception,
        always_local,
        canonical: std::sync::OnceLock::new(),
    }))))
}

pub extern "C" fn f_make_class(th: &mut Thread, frame: &mut Frame, dst: u32, spec: u32, _: u32, _: u32) -> u32 {
    let c = frame.code();
    match build_class(th, &c.classes[spec as usize], &c.codes) {
        Ok(v) => {
            frame.set(dst as Reg, v);
            OK
        }
        Err(Raised) => RAISED,
    }
}

/// Type argument `k` of the type of `o` (a generic class instance).
pub fn type_arg(th: &mut Thread, o: Value, k: u32) -> VResult<Value> {
    let t = rtype::as_type(rtype::type_of(o)).unwrap();
    match t.args.get(k as usize) {
        Some(a) => Ok(*a),
        None => Err(th.raise("TypeError", format!("`{}` has no type argument {k}", t.name))),
    }
}

/// Type argument `k` of trait `globals[tr]` as implemented by `o`.
pub fn trait_arg(th: &mut Thread, o: Value, tr: u32, k: u32) -> VResult<Value> {
    let trv = th.vm.global(tr);
    let t = rtype::as_type(rtype::type_of(o)).unwrap();
    let registered = || {
        let tr = trv.as_trait()?;
        let impls = tr.impls.read().unwrap();
        impls.iter().filter(|i| i.trait_ == Some(trv)).find_map(|i| i.trait_type(&i.bind(t)?))
    };
    let found = rtype::impls_of(t)
        .iter()
        .copied()
        .find(|i| rtype::as_type(*i).unwrap().ctor == Ctor::Trait(trv))
        .or_else(registered)
        .and_then(|i| rtype::as_type(i).unwrap().args.get(k as usize).copied());
    match found {
        Some(a) => Ok(a),
        None => Err(th.raise(
            "TypeError",
            format!("`{}` does not implement a generic `{}`", t.name, rtype::name(rtype::prim(Ctor::Trait(trv)))),
        )),
    }
}

/// dst = the runtime type of the value in `obj`.
pub extern "C" fn f_load_type_of(_th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, _: u32, _: u32) -> u32 {
    let t = rtype::type_of(frame.reg(obj as Reg));
    frame.set(dst as Reg, t);
    OK
}

/// dst = type argument `k` of the instance in `obj` (in a generic method).
pub extern "C" fn f_load_type_arg(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, k: u32, _: u32) -> u32 {
    match type_arg(th, frame.reg(obj as Reg), k) {
        Ok(a) => {
            frame.set(dst as Reg, a);
            OK
        }
        Err(Raised) => RAISED,
    }
}

/// dst = type argument `k` of trait `globals[tr]` as implemented by the
/// instance in `obj` (in a default method of a generic trait).
pub extern "C" fn f_load_trait_arg(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, tr: u32, k: u32) -> u32 {
    match trait_arg(th, frame.reg(obj as Reg), tr, k) {
        Ok(a) => {
            frame.set(dst as Reg, a);
            OK
        }
        Err(Raised) => RAISED,
    }
}

/// dst = `func::<targs>`: a generic function with its type arguments.
pub extern "C" fn f_instantiate(_th: &mut Thread, frame: &mut Frame, dst: u32, func: u32, targs: u32, _: u32) -> u32 {
    let f = frame.reg(func as Reg);
    let targs: Box<[Value]> = frame.code().arglists[targs as usize].args.iter().map(|a| frame.reg(a.reg)).collect();
    f.incref();
    frame.set(dst as Reg, Obj::alloc(Body::Instantiated { func: f, targs }));
    OK
}

/// dst = the runtime type `code.types[ty]` (immortal).
pub extern "C" fn f_make_type(th: &mut Thread, frame: &mut Frame, dst: u32, ty: u32, _: u32, _: u32) -> u32 {
    match types::frame_type(th, frame, ty) {
        Ok(t) => {
            frame.set(dst as Reg, t.value());
            OK
        }
        Err(Raised) => RAISED,
    }
}

pub extern "C" fn f_check_type(th: &mut Thread, frame: &mut Frame, reg: u32, ty: u32, what: u32, _: u32) -> u32 {
    let code = frame.code();
    let v = frame.reg(reg as Reg);
    if v.is_dead() {
        raise_dead(th, frame, reg as Reg);
        return RAISED;
    }
    let t = match types::frame_type(th, frame, ty) {
        Ok(t) => t,
        Err(Raised) => return RAISED,
    };
    if rtype::has_type(v, t) {
        return OK;
    }
    let what = &code.names[what as usize];
    th.raise("TypeError", format!("{what} expected `{}`, got `{}`", t.name, types::value_type_name(v)));
    RAISED
}

// ---------------------------------------------------------------- fields

/// The slot of field `name` of `inst`, through the inline cache of the
/// current instruction.
#[inline]
fn cached_field(frame: &Frame, inst: &Instance, name: &str) -> Option<usize> {
    let class = inst.class();
    let ic = &frame.code().ics[frame.cur_pc as usize];
    let tag = (class.id as u64 + 1) << 32;
    let k = ic.load(Ordering::Relaxed);
    if k & !0xFFFF_FFFF == tag {
        return Some(k as u32 as usize);
    }
    let i = class.field_index(name)?;
    ic.store(tag | i as u64, Ordering::Relaxed);
    Some(i)
}

/// The inherent method `name` of `inst`'s class, through the inline cache
/// of the current instruction (`find_method` looks inherent methods up
/// first, so this is its result whenever it is `Some`).
#[inline]
fn cached_method(frame: &Frame, inst: &Instance, name: &str) -> Option<Value> {
    const PTR: u64 = (1 << 48) - 1;
    let class = inst.class();
    let ic = &frame.code().ics[frame.cur_pc as usize];
    let tag = (class.id as u64 + 1) << 48;
    let k = ic.load(Ordering::Relaxed);
    if k != 0 && k & !PTR == tag {
        return Some(Value(k & PTR));
    }
    let m = *class.methods.get(name)?;
    if class.id < 0xFFFF && m.0 & !PTR == 0 {
        ic.store(tag | m.0, Ordering::Relaxed);
    }
    Some(m)
}

fn field_slot(th: &mut Thread, frame: &Frame, obj: Value, name: &str) -> VResult<(&'static Instance, usize)> {
    let Some(inst) = obj.as_instance() else {
        return Err(no_attribute(th, obj, name));
    };
    match cached_field(frame, inst, name) {
        Some(i) => Ok((inst, i)),
        None => Err(th.raise("AttributeError", format!("'{}' object has no field '{name}'", inst.class().name))),
    }
}

pub extern "C" fn f_load_field(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, name: u32, _: u32) -> u32 {
    let o = frame.reg(obj as Reg);
    let n = &frame.code().names[name as usize];
    if o.is_dead() {
        raise_dead(th, frame, obj as Reg);
        return RAISED;
    }
    if let Some(inst) = o.as_instance() {
        let class = inst.class();
        if let Some(i) = cached_field(frame, inst, n) {
            let v = inst.get_borrowed(i);
            if v == Value::UNINIT {
                th.raise("AttributeError", format!("field `{n}` of '{}' is not initialized yet", class.name));
                return RAISED;
            }
            frame.set(dst as Reg, v);
            return OK;
        }
    }
    if let Some(m) = lookup_method(th, o, n) {
        o.incref();
        let bm = rc::autorelease(Obj::alloc(Body::BoundMethod { recv: o, func: m }));
        frame.set(dst as Reg, bm);
        return OK;
    }
    no_attribute(th, o, n);
    RAISED
}

pub extern "C" fn f_load_field_f(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, name: u32, _: u32) -> u32 {
    let o = frame.reg(obj as Reg);
    let n = &frame.code().names[name as usize];
    if o.is_dead() {
        raise_dead(th, frame, obj as Reg);
        return RAISED;
    }
    let r = (|| {
        let (inst, i) = field_slot(th, frame, o, n)?;
        let raw = inst.slots[i].load(Ordering::Acquire);
        match inst.class().fields[i].kind {
            SlotKind::RawFloat => Ok(raw),
            SlotKind::RawInt => Ok((raw as i64 as f64).to_bits()),
            SlotKind::Value => raw_float(th, Value(raw)),
        }
    })();
    match r {
        Ok(bits) => {
            frame.set(dst as Reg, Value(bits));
            OK
        }
        Err(Raised) => RAISED,
    }
}

/// The f64 bits of an int or float value.
fn raw_float(th: &mut Thread, v: Value) -> VResult<u64> {
    if let Some(i) = v.as_small_int() {
        return Ok((i as f64).to_bits());
    }
    match v.body() {
        Some(Body::Float(f)) => Ok(f.to_bits()),
        Some(Body::Int(i)) => Ok((*i as f64).to_bits()),
        _ => Err(th.raise("TypeError", format!("expected a float, got `{}`", types::value_type_name(v)))),
    }
}

pub extern "C" fn f_unbox_f(th: &mut Thread, frame: &mut Frame, dst: u32, src: u32, _: u32, _: u32) -> u32 {
    let v = frame.reg(src as Reg);
    if v.is_dead() {
        raise_dead(th, frame, src as Reg);
        return RAISED;
    }
    match raw_float(th, v) {
        Ok(bits) => {
            frame.set(dst as Reg, Value(bits));
            OK
        }
        Err(Raised) => RAISED,
    }
}

pub extern "C" fn f_store_raw_f(_th: &mut Thread, frame: &mut Frame, dst: u32, src: u32, _: u32, _: u32) -> u32 {
    let old = frame.reg(dst as Reg);
    let f = f64::from_bits(frame.reg(src as Reg).0);
    if !Obj::reuse_float(old, f) {
        old.decref();
        frame.set(dst as Reg, Value::float(f));
    }
    OK
}

pub extern "C" fn box_f(_th: &mut Thread, raw: Value) -> Value {
    Value::float(f64::from_bits(raw.0))
}

pub extern "C" fn f_float_div_zero(th: &mut Thread, _frame: &mut Frame, _: u32, _: u32, _: u32, _: u32) -> u32 {
    th.raise("ZeroDivisionError", "division by zero");
    RAISED
}

/// Converts an owned value for storage in a slot of the given kind.
fn to_slot(th: &mut Thread, v: Value, kind: SlotKind) -> VResult<u64> {
    match kind {
        SlotKind::Value => Ok(v.0),
        SlotKind::RawInt => {
            let i = v.as_int().ok_or_else(|| th.raise("TypeError", "expected int"))?;
            v.decref();
            Ok(i as u64)
        }
        SlotKind::RawFloat => {
            let f = v.as_float().or_else(|| v.as_int().map(|i| i as f64)).ok_or_else(|| th.raise("TypeError", "expected float"))?;
            v.decref();
            Ok(f.to_bits())
        }
    }
}

fn check_store(th: &mut Thread, inst: &Instance, i: usize, v: Value, mode: StoreMode) -> VResult<()> {
    let class = inst.class();
    let f = &class.fields[i];
    if v.is_dead() {
        return Err(dead_operand(th, v).unwrap());
    }
    let t = types::field_type(th, inst, i)?;
    if !rtype::has_type(v, t) {
        let msg = if rtype::has_shape(v, t) {
            format!("cannot store `{}`, which is not `Send`, in field `{}.{}` of type `{}`", types::value_type_name(v), class.name, f.name, t.name)
        } else {
            format!("field `{}.{}` expected `{}`, got `{}`", class.name, f.name, t.name, types::value_type_name(v))
        };
        return Err(th.raise("TypeError", msg));
    }
    if mode == StoreMode::Copy && !v.is_copy() {
        return Err(th.raise(
            "OwnershipError",
            format!("`=` can only store copy types into field `{}`; use `<-` to move a {} in", f.name, v.type_name()),
        ));
    }
    Ok(())
}

pub extern "C" fn f_store_field(th: &mut Thread, frame: &mut Frame, obj: u32, name: u32, src: u32, mode: u32) -> u32 {
    let mode = match mode {
        0 => StoreMode::Copy,
        1 => StoreMode::Move,
        _ => StoreMode::Dyn,
    };
    let o = frame.reg(obj as Reg);
    let v = frame.reg(src as Reg);
    let n = &frame.code().names[name as usize];
    let r = (|| {
        let (inst, i) = field_slot(th, frame, o, n)?;
        check_store(th, inst, i, v, mode)?;
        frame.set(src as Reg, Value::MOVED);
        let kind = inst.class().fields[i].kind;
        let raw = to_slot(th, v, kind)?;
        let old = inst.slots[i].swap(raw, Ordering::AcqRel);
        if kind == SlotKind::Value {
            rc::defer_decref(Value(old));
        }
        Ok(())
    })();
    fres(r)
}

pub extern "C" fn f_swap_field(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, name: u32, src: u32) -> u32 {
    let o = frame.reg(obj as Reg);
    let v = frame.reg(src as Reg);
    let n = &frame.code().names[name as usize];
    let r = (|| {
        let (inst, i) = field_slot(th, frame, o, n)?;
        check_store(th, inst, i, v, StoreMode::Move)?;
        let kind = inst.class().fields[i].kind;
        if kind == SlotKind::Value && inst.slots[i].load(Ordering::Acquire) == Value::UNINIT.0 {
            return Err(th.raise("AttributeError", format!("cannot swap out field `{n}`: it is not initialized yet")));
        }
        frame.set(src as Reg, Value::MOVED);
        let raw = to_slot(th, v, kind)?;
        let old = inst.slots[i].swap(raw, Ordering::AcqRel);
        let old = match kind {
            SlotKind::Value => Value(old),
            SlotKind::RawInt => Value::from_i64(old as i64),
            SlotKind::RawFloat => Value::float(f64::from_bits(old)),
        };
        frame.set(dst as Reg, old);
        Ok(())
    })();
    fres(r)
}

// ---------------------------------------------------------------- elements

/// Checks an index against `len`. Negative indexes are errors (no
/// wrap-around), so this is a single unsigned compare.
pub(crate) fn index_of(th: &mut Thread, idx: Value, len: usize, what: &str) -> VResult<usize> {
    let i = int_index(th, idx)?;
    if (i as u64) >= len as u64 {
        return Err(index_error(th, i, what, len));
    }
    Ok(i as usize)
}

pub(crate) fn int_index(th: &mut Thread, idx: Value) -> VResult<i64> {
    idx.as_int()
        .ok_or_else(|| th.raise("TypeError", format!("indices must be integers, not '{}'", idx.type_name())))
}

pub(crate) fn index_error(th: &mut Thread, i: i64, what: &str, len: usize) -> Raised {
    th.raise("IndexError", format!("index {i} out of range for {what} of length {len}"))
}

/// Element `k` of an array, owned.
fn array_get(th: &mut Thread, a: &ArrayObj, k: Value) -> VResult<Value> {
    let i = index_of(th, k, a.len(), "array")?;
    match a.get(i).unwrap() {
        Ok(v) => Ok(v),
        Err(()) => Err(th.raise("OverflowError", format!("u64 element {} does not fit in an int", a.bits(i).unwrap()))),
    }
}

/// Stores scalar `v` at `a[k]`.
fn array_set(th: &mut Thread, a: &ArrayObj, k: Value, v: Value) -> VResult<()> {
    let i = index_of(th, k, a.len(), "array")?;
    a.set(i, v).map_err(|e| match e {
        StoreError::Type => th.raise(
            "TypeError",
            format!("cannot store a '{}' in an array of {}", v.type_name(), a.elem.name()),
        ),
        StoreError::Overflow => {
            th.raise("OverflowError", format!("{} does not fit in {}", crate::display::repr(v), a.elem.name()))
        }
    })
}

/// `a[i]` as an owned value (copy types): arrays produce a fresh scalar;
/// other containers yield a copy of a copy-type element.
pub extern "C" fn f_load_elem(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, idx: u32, _: u32) -> u32 {
    let o = frame.reg(obj as Reg);
    let k = frame.reg(idx as Reg);
    let r = match o.body() {
        Some(Body::Array(a)) => array_get(th, a, k),
        _ => {
            let r = f_load_index(th, frame, dst, obj, idx, 0);
            if r != OK {
                return r;
            }
            let v = frame.reg(dst as Reg);
            if v.is_copy() {
                v.incref();
                Ok(v)
            } else {
                frame.set(dst as Reg, Value::UNINIT);
                Err(th.raise(
                    "OwnershipError",
                    format!("cannot take ownership of an element (a borrowed {}); use `++` to create a new reference", v.type_name()),
                ))
            }
        }
    };
    match r {
        Ok(v) => {
            frame.set(dst as Reg, v);
            OK
        }
        Err(Raised) => RAISED,
    }
}

pub extern "C" fn f_load_index(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, idx: u32, _: u32) -> u32 {
    let o = frame.reg(obj as Reg);
    let k = frame.reg(idx as Reg);
    let r = (|| -> VResult<Value> {
        match o.body() {
            // A boxed scalar lives as long as the borrow can.
            Some(Body::Array(a)) => array_get(th, a, k).map(rc::autorelease),
            Some(Body::List(l)) => {
                let i = int_index(th, k)?;
                match usize::try_from(i).ok().and_then(|i| l.get(i)) {
                    Some(v) => Ok(v),
                    None => Err(index_error(th, i, "list", l.len())),
                }
            }
            Some(Body::Tuple(t)) => {
                let i = index_of(th, k, t.len(), "tuple")?;
                Ok(t[i])
            }
            Some(Body::Str(s)) => {
                let chars: Vec<char> = s.chars().collect();
                let i = index_of(th, k, chars.len(), "str")?;
                Ok(rc::autorelease(Value::str(chars[i].to_string())))
            }
            _ => crate::builtins::index_other(th, o, k),
        }
    })();
    match r {
        Ok(v) => {
            frame.set(dst as Reg, v);
            OK
        }
        Err(Raised) => RAISED,
    }
}

/// Set in a store's mode operand when the compiler proved the stored value's
/// type equals the container's element type.
pub const STORE_PROVEN: u32 = 1 << 8;

/// Checks (and coerces) an owned value for list `l`, unless `proven`.
fn list_admit(th: &mut Thread, l: &ListObj, v: Value, proven: bool) -> VResult<Value> {
    if proven {
        return Ok(v);
    }
    let t = rtype::as_type(l.ty).unwrap();
    types::admit(th, v, rtype::as_type(rtype::elem_of(t)).unwrap(), t)
}

pub extern "C" fn f_store_index(th: &mut Thread, frame: &mut Frame, obj: u32, idx: u32, src: u32, mode: u32) -> u32 {
    let proven = mode & STORE_PROVEN != 0;
    let mode = mode & 0xff;
    let o = frame.reg(obj as Reg);
    let k = frame.reg(idx as Reg);
    let v = frame.reg(src as Reg);
    let r = (|| -> VResult<()> {
        if v.is_dead() {
            return Err(dead_operand(th, v).unwrap());
        }
        if mode == StoreMode::Copy as u32 && !v.is_copy() {
            return Err(th.raise("OwnershipError", format!("`=` can only store copy types into an element; use `<-` to move a {} in", v.type_name())));
        }
        match o.body() {
            Some(Body::Array(a)) => {
                // The scalar is copied into the array; the operand is consumed.
                array_set(th, a, k, v)?;
                frame.set(src as Reg, Value::MOVED);
                v.decref();
                Ok(())
            }
            Some(Body::List(l)) => {
                let i = int_index(th, k)?;
                let v = list_admit(th, l, v, proven)?;
                frame.set(src as Reg, Value::MOVED);
                let old = l.write(|w| ((i as u64) < w.len() as u64).then(|| w.set(i as usize, v)));
                match old {
                    Some(old) => {
                        rc::defer_decref(old);
                        Ok(())
                    }
                    None => {
                        frame.set(src as Reg, v);
                        Err(index_error(th, i, "list", l.len()))
                    }
                }
            }
            _ => {
                if crate::builtins::store_index_other(th, o, k, v)? {
                    frame.set(src as Reg, Value::MOVED);
                    Ok(())
                } else {
                    Err(th.raise("TypeError", format!("'{}' object does not support item assignment", o.type_name())))
                }
            }
        }
    })();
    fres(r)
}

pub extern "C" fn f_swap_index(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, idx: u32, src: u32) -> u32 {
    let o = frame.reg(obj as Reg);
    let k = frame.reg(idx as Reg);
    let v = frame.reg(src as Reg);
    let r = (|| -> VResult<()> {
        match o.body() {
            Some(Body::Array(a)) => {
                let old = array_get(th, a, k)?;
                if let Err(e) = array_set(th, a, k, v) {
                    old.decref();
                    return Err(e);
                }
                frame.set(src as Reg, Value::MOVED);
                v.decref();
                frame.set(dst as Reg, old);
                Ok(())
            }
            Some(Body::List(l)) => {
                let i = int_index(th, k)?;
                let v = list_admit(th, l, v, false)?;
                frame.set(src as Reg, Value::MOVED);
                let old = l.write(|w| ((i as u64) < w.len() as u64).then(|| w.set(i as usize, v)));
                match old {
                    Some(old) => {
                        frame.set(dst as Reg, old);
                        Ok(())
                    }
                    None => {
                        frame.set(src as Reg, v);
                        Err(index_error(th, i, "list", l.len()))
                    }
                }
            }
            _ => match crate::builtins::swap_index_other(th, o, k, v)? {
                Some(old) => {
                    frame.set(src as Reg, Value::MOVED);
                    frame.set(dst as Reg, old);
                    Ok(())
                }
                None => Err(th.raise("TypeError", format!("'{}' object does not support item assignment", o.type_name()))),
            },
        }
    })();
    fres(r)
}

fn take_args(frame: &mut Frame, args: u32) -> Vec<Value> {
    let code = frame.code();
    code.arglists[args as usize]
        .args
        .iter()
        .map(|a| {
            let v = frame.reg(a.reg);
            frame.set(a.reg, Value::MOVED);
            v
        })
        .collect()
}

pub extern "C" fn f_build_tuple(_th: &mut Thread, frame: &mut Frame, dst: u32, args: u32, _: u32, _: u32) -> u32 {
    let items = take_args(frame, args);
    frame.set(dst as Reg, Obj::alloc(Body::Tuple(items.into_boxed_slice())));
    OK
}

pub extern "C" fn f_build_list(th: &mut Thread, frame: &mut Frame, dst: u32, args: u32, ty: u32, _: u32) -> u32 {
    let t = match types::frame_type(th, frame, ty) {
        Ok(t) => t.value(),
        Err(Raised) => return RAISED,
    };
    let items = take_args(frame, args);
    match types::new_list(th, t, items) {
        Ok(l) => {
            frame.set(dst as Reg, l);
            OK
        }
        Err(Raised) => RAISED,
    }
}

/// A struct value from owned field values (declared order), each checked
/// against its field type. On an error every value is released.
pub extern "C" fn f_build_struct(th: &mut Thread, frame: &mut Frame, dst: u32, args: u32, ty: u32, _: u32) -> u32 {
    let t = match types::frame_type(th, frame, ty) {
        Ok(t) => t,
        Err(Raised) => return RAISED,
    };
    let mut vals = take_args(frame, args);
    let Ctor::Class(cls) = t.ctor else { unreachable!("struct type") };
    let c = cls.as_class().expect("struct class");
    let slots: Vec<AtomicU64> = c.fields.iter().map(|f| AtomicU64::new(if f.kind == SlotKind::Value { Value::UNINIT.0 } else { 0 })).collect();
    let inst = Obj::alloc(Body::Instance(Instance { class: cls, ty: t.value(), slots: slots.into_boxed_slice() }));
    let i = inst.as_instance().unwrap();
    // Raw floats go straight into float slots (boxed for any other slot).
    for (k, a) in frame.code().arglists[args as usize].args.iter().enumerate() {
        if a.raw {
            if c.fields[k].kind == SlotKind::RawFloat {
                i.slots[k].store(vals[k].0, Ordering::Release);
                vals[k] = Value::UNINIT;
            } else {
                vals[k] = Value::float(f64::from_bits(vals[k].0));
            }
        }
    }
    let mut it = vals.into_iter().enumerate().filter(|(_, v)| *v != Value::UNINIT);
    while let Some((k, v)) = it.next() {
        let stored = check_store(th, i, k, v, StoreMode::Move).and_then(|_| to_slot(th, v, c.fields[k].kind));
        match stored {
            Ok(raw) => i.slots[k].store(raw, Ordering::Release),
            Err(Raised) => {
                v.decref();
                it.for_each(|(_, v)| v.decref());
                inst.decref();
                return RAISED;
            }
        }
    }
    frame.set(dst as Reg, inst);
    OK
}

pub extern "C" fn f_build_dict(th: &mut Thread, frame: &mut Frame, dst: u32, args: u32, ty: u32, _: u32) -> u32 {
    let t = match types::frame_type(th, frame, ty) {
        Ok(t) => t.value(),
        Err(Raised) => return RAISED,
    };
    let items = take_args(frame, args);
    match crate::builtins::build_dict(th, t, items) {
        Ok(d) => {
            frame.set(dst as Reg, d);
            OK
        }
        Err(Raised) => RAISED,
    }
}

pub extern "C" fn f_unpack(th: &mut Thread, frame: &mut Frame, src: u32, dsts: u32, _: u32, _: u32) -> u32 {
    let v = frame.reg(src as Reg);
    let code = frame.code();
    let targets = &code.arglists[dsts as usize].args;
    let items: Vec<Value> = match v.body() {
        Some(Body::Tuple(t)) => t.to_vec(),
        Some(Body::List(l)) => l.snapshot(),
        _ => match crate::builtins::collect(th, v) {
            Ok(items) => {
                // `collect` returned owned items; release our extra references below.
                items.iter().for_each(|x| x.decref());
                items
            }
            Err(Raised) => return RAISED,
        },
    };
    if items.len() != targets.len() {
        th.raise(
            "ValueError",
            format!("cannot unpack {} values into {} names", items.len(), targets.len()),
        );
        return RAISED;
    }
    for (a, x) in targets.iter().zip(items) {
        x.incref();
        frame.set(a.reg, x);
    }
    frame.set(src as Reg, Value::MOVED);
    v.decref();
    OK
}

// ---------------------------------------------------------------- exceptions

pub fn implements_exception(th: &Thread, v: Value) -> bool {
    v.as_instance().is_some_and(|i| i.class().is_exception) || {
        let _ = th;
        false
    }
}

pub extern "C" fn f_raise(th: &mut Thread, frame: &mut Frame, exc: u32, cause: u32, _: u32, _: u32) -> u32 {
    let mut v = frame.reg(exc as Reg);
    frame.set(exc as Reg, Value::MOVED);
    if let Some(c) = v.as_class() {
        if !c.is_exception {
            th.raise("TypeError", format!("cannot raise `{}`: it does not implement `Exception`", c.name));
            return RAISED;
        }
        match call_simple(th, v, None, &[]) {
            Ok(e) => v = e,
            Err(Raised) => return RAISED,
        }
    }
    if !implements_exception(th, v) {
        let t = v.type_name();
        v.decref();
        th.raise("TypeError", format!("cannot raise a `{t}`: exceptions must implement `Exception`"));
        return RAISED;
    }
    let c = if cause as Reg == Reg::MAX {
        Value::UNINIT
    } else {
        let c = frame.reg(cause as Reg);
        frame.set(cause as Reg, Value::MOVED);
        c
    };
    th.set_exception(v, c);
    RAISED
}

pub extern "C" fn f_reraise(th: &mut Thread, frame: &mut Frame, exc: u32, _: u32, _: u32, _: u32) -> u32 {
    let v = frame.reg(exc as Reg);
    frame.set(exc as Reg, Value::MOVED);
    if v.is_dead() {
        th.raise("RuntimeError", "no active exception to re-raise");
        return RAISED;
    }
    let tb = std::mem::take(&mut th.handled_tb);
    th.set_exception(v, Value::UNINIT);
    if let Some(e) = &mut th.exc {
        e.tb = tb;
    }
    RAISED
}

pub extern "C" fn f_take_exception(_th: &mut Thread, frame: &mut Frame, dst: u32, _: u32, _: u32, _: u32) -> u32 {
    let th = _th;
    let e = th.exc.take().expect("handler entered without a pending exception");
    e.cause.decref();
    th.handled_tb = e.tb;
    frame.set(dst as Reg, e.value);
    OK
}

pub extern "C" fn f_match_exception(th: &mut Thread, frame: &mut Frame, exc: u32, ty: u32, target: u32, _: u32) -> u32 {
    let v = frame.reg(exc as Reg);
    let t = match types::frame_type(th, frame, ty) {
        Ok(t) => t,
        Err(Raised) => return RAISED,
    };
    if rtype::has_type(v, t) {
        frame.pc = target;
    }
    OK
}

// ---------------------------------------------------------------- iteration

pub extern "C" fn f_get_iter(th: &mut Thread, frame: &mut Frame, dst: u32, src: u32, _: u32, _: u32) -> u32 {
    let v = frame.reg(src as Reg);
    match crate::builtins::get_iter(th, v) {
        Ok(it) => {
            frame.set(dst as Reg, it);
            OK
        }
        Err(Raised) => RAISED,
    }
}

pub extern "C" fn f_for_iter(th: &mut Thread, frame: &mut Frame, dst: u32, iter: u32, end: u32, _: u32) -> u32 {
    let it = frame.reg(iter as Reg);
    match crate::builtins::iter_next(th, it) {
        Ok(Some(v)) => {
            frame.set(dst as Reg, v);
            OK
        }
        Ok(None) => {
            frame.pc = end;
            OK
        }
        Err(Raised) => RAISED,
    }
}
