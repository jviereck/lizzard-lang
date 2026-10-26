//! Runtime helpers: slow paths called by opcode semantics.
//!
//! Value helpers (`Helper1`/`Helper2`) return `Value::UNINIT` when they raise.
//! Frame helpers return 0 (ok), 1 (raised), 2 (returned) or 3 (yielded).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

use crate::code::{Arg, ClassSpec, Code, Reg, TypeDesc};
use crate::display;
use crate::interp::{run_code, Frame, Raised, Thread, VResult};
use crate::object::{Body, Class, Field, Function, Instance, Obj, SlotKind, Trait};
use crate::ops::StoreMode;
use crate::rc;
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

fn dead_operand(th: &mut Thread, v: Value) -> Option<Raised> {
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
            let mut out: Vec<Value> = x.snapshot();
            out.extend(y.snapshot());
            out.iter().for_each(|v| v.incref());
            return Obj::list(out);
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
            return Some(Obj::list(out));
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
        None => vres(Err(unary_error(th, "-", a))),
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

fn collect_args(frame: &Frame, args: &[Arg]) -> Vec<CallArg> {
    args.iter()
        .map(|a| CallArg {
            v: frame.reg(a.reg),
            owned: a.owned,
            reg: Some(a.reg),
            root: a.root,
            global: (a.global != u32::MAX).then_some(a.global),
        })
        .collect()
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
            let expected = code.params.len();
            if args.len() + nself != expected {
                let given = args.len();
                let want = expected - nself;
                return Err(th.raise(
                    "TypeError",
                    format!("{} takes {want} argument{} but {given} {} given", callable_name(f), if want == 1 { "" } else { "s" }, if given == 1 { "was" } else { "were" }),
                ));
            }
            let mut regs = vec![Value::UNINIT; code.nregs as usize];
            if let Some(s) = self_val {
                regs[0] = s;
            }
            for (i, _) in args.iter().enumerate() {
                let p = &code.params[i + nself];
                regs[i + nself] = if p.keep {
                    take_for_keep(th, caller.as_deref_mut(), args, i, recv_reg, &p.name, &code.name)?
                } else {
                    let v = args[i].v;
                    if v.is_dead() {
                        return Err(dead_operand(th, v).unwrap());
                    }
                    v
                };
            }
            if code.is_generator {
                return crate::generator::create(th, f, regs);
            }
            run_code(th, code, &mut regs, func.captures.as_ptr())
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
        Some(Body::Class(c)) => construct(th, caller, f, c, args),
        Some(Body::BoundMethod { recv, func }) => invoke(th, caller, *func, Some(*recv), None, args),
        _ => {
            if let Some(r) = dead_operand(th, f) {
                return Err(r);
            }
            Err(th.raise("TypeError", format!("'{}' object is not callable", f.type_name())))
        }
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
    let e = Obj::alloc(Body::Instance(Instance { class, slots: slots.into_boxed_slice() }));
    if let Some(i) = c.field_index("msg") {
        e.as_instance().unwrap().slots[i].store(msg.0, Ordering::Release);
    } else {
        msg.decref();
    }
    e
}

fn construct(th: &mut Thread, caller: Option<&mut Frame>, cls: Value, c: &'static Class, args: &[CallArg]) -> VResult<Value> {
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
    let slots: Vec<AtomicU64> = c
        .fields
        .iter()
        .map(|f| AtomicU64::new(if f.kind == SlotKind::Value { Value::UNINIT.0 } else { 0 }))
        .collect();
    let inst = Obj::alloc(Body::Instance(Instance { class: cls, slots: slots.into_boxed_slice() }));
    match c.methods.get("__init__") {
        Some(&init) => match invoke(th, caller, init, Some(inst), None, args) {
            Ok(r) => r.decref(),
            Err(e) => {
                inst.decref();
                return Err(e);
            }
        },
        None => {
            if !args.is_empty() {
                inst.decref();
                return Err(th.raise("TypeError", format!("{}() takes no arguments", c.name)));
            }
        }
    }
    let i = inst.as_instance().unwrap();
    for (k, f) in c.fields.iter().enumerate() {
        if f.kind == SlotKind::Value && i.slots[k].load(Ordering::Acquire) == Value::UNINIT.0 {
            inst.decref();
            return Err(th.raise("AttributeError", format!("`{}.__init__` did not initialize field `{}`", c.name, f.name)));
        }
    }
    Ok(inst)
}

pub extern "C" fn f_call(th: &mut Thread, frame: &mut Frame, dst: u32, func: u32, args: u32, _: u32) -> u32 {
    let code = frame.code();
    let al = &code.arglists[args as usize];
    let f = frame.reg(func as Reg);
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

/// Finds a method for `recv.name`.
pub fn lookup_method(recv: Value, name: &str) -> Option<Value> {
    match recv.body() {
        Some(Body::Instance(inst)) => inst.class().methods.get(name).copied(),
        _ => crate::builtins::type_method(recv, name),
    }
}

pub extern "C" fn f_call_method(th: &mut Thread, frame: &mut Frame, dst: u32, recv: u32, name: u32, args: u32) -> u32 {
    let code = frame.code();
    let al = &code.arglists[args as usize];
    let r = frame.reg(recv as Reg);
    let mname = &code.names[name as usize];
    let cargs = collect_args(frame, &al.args);
    if r.is_dead() {
        raise_dead(th, frame, recv as Reg);
        return RAISED;
    }
    if al.pins {
        rc::pin();
    }
    let result = match lookup_method(r, mname) {
        Some(m) => invoke(th, Some(frame), m, Some(r), Some(recv as Reg), &cargs),
        None => {
            // A callable stored in a field: `self.callback(x)`.
            let field = r.as_instance().and_then(|inst| inst.class().field_index(mname).map(|i| inst.get_borrowed(i)));
            match field {
                Some(f) => invoke(th, Some(frame), f, None, None, &cargs),
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

fn no_attribute(th: &mut Thread, v: Value, name: &str) -> Raised {
    if let Some(r) = dead_operand(th, v) {
        return r;
    }
    th.raise("AttributeError", format!("'{}' object has no attribute '{name}'", v.type_name()))
}

pub extern "C" fn f_safepoint(_th: &mut Thread, _frame: &mut Frame, _: u32, _: u32, _: u32, _: u32) -> u32 {
    rc::safepoint();
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
    let t = Obj::alloc_immortal(Body::Trait(Trait { name: s.name.clone(), id, methods, implies: Vec::new() }));
    frame.set(dst as Reg, t);
    OK
}

pub fn slot_kind(ty: &TypeDesc) -> SlotKind {
    match ty {
        TypeDesc::Int => SlotKind::RawInt,
        TypeDesc::Float => SlotKind::RawFloat,
        _ => SlotKind::Value,
    }
}

pub fn build_class(th: &mut Thread, s: &ClassSpec, codes: &[Arc<Code>]) -> VResult<Value> {
    let mut traits = Vec::new();
    for (idx, tname) in &s.traits {
        let t = resolve_global(th, *idx);
        let Some(tr) = t.as_trait() else {
            return Err(th.raise("TypeError", format!("class `{}`: `{tname}` is not a trait", s.name)));
        };
        if !traits.contains(&t) {
            traits.push(t);
        }
        for imp in &tr.implies {
            if !traits.contains(imp) {
                traits.push(*imp);
            }
        }
    }
    let mut methods: HashMap<String, Value> = HashMap::new();
    for (name, ci) in &s.methods {
        let f = Obj::alloc_immortal(Body::Function(Function { code: codes[*ci as usize].clone(), captures: Box::new([]) }));
        methods.insert(name.clone(), f);
    }
    for t in &traits {
        let tr = t.as_trait().unwrap();
        for (mname, default) in &tr.methods {
            if methods.contains_key(mname) {
                continue;
            }
            match default {
                Some(d) => {
                    methods.insert(mname.clone(), *d);
                }
                None => {
                    return Err(th.raise(
                        "TypeError",
                        format!("class `{}` implements `{}` but does not define required method `{mname}`", s.name, tr.name),
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
        .map(|f| Field { name: f.name.clone(), ty: f.ty.clone(), kind: slot_kind(&f.ty) })
        .collect();
    let id = th.vm.next_class_id.fetch_add(1, Ordering::Relaxed);
    Ok(Obj::alloc_immortal(Body::Class(Class { name: s.name.clone(), id, fields, methods, traits, is_exception })))
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

pub extern "C" fn f_check_type(th: &mut Thread, frame: &mut Frame, reg: u32, ty: u32, what: u32, _: u32) -> u32 {
    let code = frame.code();
    let v = frame.reg(reg as Reg);
    let t = &code.types[ty as usize];
    if v.is_dead() {
        raise_dead(th, frame, reg as Reg);
        return RAISED;
    }
    if types::check(th, v, t) {
        return OK;
    }
    let what = &code.names[what as usize];
    th.raise("TypeError", format!("{what} expected `{t}`, got `{}`", v.type_name()));
    RAISED
}

// ---------------------------------------------------------------- fields

fn field_slot(th: &mut Thread, obj: Value, name: &str) -> VResult<(&'static Instance, usize)> {
    let Some(inst) = obj.as_instance() else {
        return Err(no_attribute(th, obj, name));
    };
    match inst.class().field_index(name) {
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
        if let Some(i) = class.field_index(n) {
            let v = inst.get_borrowed(i);
            if v == Value::UNINIT {
                th.raise("AttributeError", format!("field `{n}` of '{}' is not initialized yet", class.name));
                return RAISED;
            }
            frame.set(dst as Reg, v);
            return OK;
        }
    }
    if let Some(m) = lookup_method(o, n) {
        o.incref();
        let bm = rc::autorelease(Obj::alloc(Body::BoundMethod { recv: o, func: m }));
        frame.set(dst as Reg, bm);
        return OK;
    }
    no_attribute(th, o, n);
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
    if !types::check(th, v, &f.ty) {
        return Err(th.raise(
            "TypeError",
            format!("field `{}.{}` expected `{}`, got `{}`", class.name, f.name, f.ty, v.type_name()),
        ));
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
        let (inst, i) = field_slot(th, o, n)?;
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
        let (inst, i) = field_slot(th, o, n)?;
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

pub extern "C" fn f_load_index(th: &mut Thread, frame: &mut Frame, dst: u32, obj: u32, idx: u32, _: u32) -> u32 {
    let o = frame.reg(obj as Reg);
    let k = frame.reg(idx as Reg);
    let r = (|| -> VResult<Value> {
        match o.body() {
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

pub extern "C" fn f_store_index(th: &mut Thread, frame: &mut Frame, obj: u32, idx: u32, src: u32, mode: u32) -> u32 {
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
            Some(Body::List(l)) => {
                let i = int_index(th, k)?;
                let old = l.write(|w| ((i as u64) < w.len() as u64).then(|| w.set(i as usize, v)));
                match old {
                    Some(old) => {
                        frame.set(src as Reg, Value::MOVED);
                        rc::defer_decref(old);
                        Ok(())
                    }
                    None => Err(index_error(th, i, "list", l.len())),
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
            Some(Body::List(l)) => {
                let i = int_index(th, k)?;
                let old = l.write(|w| ((i as u64) < w.len() as u64).then(|| w.set(i as usize, v)));
                match old {
                    Some(old) => {
                        frame.set(src as Reg, Value::MOVED);
                        frame.set(dst as Reg, old);
                        Ok(())
                    }
                    None => Err(index_error(th, i, "list", l.len())),
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

pub extern "C" fn f_build_list(_th: &mut Thread, frame: &mut Frame, dst: u32, args: u32, _: u32, _: u32) -> u32 {
    let items = take_args(frame, args);
    frame.set(dst as Reg, Obj::list(items));
    OK
}

pub extern "C" fn f_build_dict(th: &mut Thread, frame: &mut Frame, dst: u32, args: u32, _: u32, _: u32) -> u32 {
    let items = take_args(frame, args);
    match crate::builtins::build_dict(th, items) {
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
    let t = &frame.code().types[ty as usize];
    if types::check(th, v, t) {
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
