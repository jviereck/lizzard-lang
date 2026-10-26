//! Builtin functions, builtin type methods, iterators and exception classes.
//!
//! Builtins receive borrowed arguments (owned for `keep` parameters) and
//! return owned values.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::code::TypeDesc;
use crate::dict::Dict;
use crate::display;
use crate::generator;
use crate::helpers::{self, values_eq};
use crate::interp::{Raised, Thread, VResult};
use crate::list::ListObj;
use crate::object::{Body, Builtin, BuiltinFn, Class, Field, Obj, SlotKind, Trait};
use crate::rc;
use crate::value::Value;

// ---------------------------------------------------------------- iterators

pub enum IterKind {
    /// Iterates a list by index (holds a reference to the list).
    List { list: Value, idx: usize },
    /// Owned items handed out one by one.
    Items { items: std::vec::IntoIter<Value> },
    Range { cur: i64, end: i64, step: i64 },
    /// Wraps a user object implementing `__next__`.
    Object { obj: Value },
}

impl IterKind {
    pub(crate) fn release(&mut self) {
        match self {
            IterKind::List { list, .. } => list.decref(),
            IterKind::Items { items } => items.by_ref().for_each(|v| v.decref()),
            IterKind::Range { .. } => {}
            IterKind::Object { obj } => obj.decref(),
        }
    }
}

fn new_iter(k: IterKind) -> Value {
    Obj::alloc(Body::Iter(Mutex::new(k)))
}

fn items_iter(items: Vec<Value>) -> Value {
    new_iter(IterKind::Items { items: items.into_iter() })
}

/// `iter(v)`: an owned iterator over a borrowed iterable.
pub fn get_iter(th: &mut Thread, v: Value) -> VResult<Value> {
    match v.body() {
        Some(Body::List(_)) => {
            v.incref();
            Ok(new_iter(IterKind::List { list: v, idx: 0 }))
        }
        Some(Body::Tuple(t)) => {
            let items: Vec<Value> = t.to_vec();
            items.iter().for_each(|x| x.incref());
            Ok(items_iter(items))
        }
        Some(Body::Str(s)) => Ok(items_iter(s.chars().map(|c| Value::str(c.to_string())).collect())),
        Some(Body::Dict(d)) => {
            let keys: Vec<Value> = d.entries().into_iter().map(|(k, _)| k).collect();
            keys.iter().for_each(|x| x.incref());
            Ok(items_iter(keys))
        }
        Some(Body::Iter(_)) | Some(Body::Generator(_)) | Some(Body::Channel(_)) => {
            v.incref();
            Ok(v)
        }
        Some(Body::Instance(inst)) => {
            let class = inst.class();
            if let Some(&m) = class.methods.get("__iter__") {
                let it = helpers::call_simple(th, m, Some(v), &[])?;
                if it.as_instance().is_some() {
                    return Ok(new_iter(IterKind::Object { obj: it }));
                }
                return Ok(it);
            }
            if class.methods.contains_key("__next__") {
                v.incref();
                return Ok(new_iter(IterKind::Object { obj: v }));
            }
            Err(th.raise("TypeError", format!("'{}' object is not iterable", class.name)))
        }
        _ => {
            if v.is_dead() {
                return Err(th.raise("OwnershipError", "use of a moved value"));
            }
            Err(th.raise("TypeError", format!("'{}' object is not iterable", v.type_name())))
        }
    }
}

fn is_stop_iteration(th: &Thread) -> bool {
    let stop = th.vm.builtins.class("StopIteration");
    th.exc.as_ref().is_some_and(|e| e.value.as_instance().is_some_and(|i| i.class == stop))
}

fn clear_exception(th: &mut Thread) {
    if let Some(e) = th.exc.take() {
        e.value.decref();
        e.cause.decref();
    }
}

/// Advances an iterator; returns an owned item or `None` when exhausted.
pub fn iter_next(th: &mut Thread, it: Value) -> VResult<Option<Value>> {
    match it.body() {
        Some(Body::Iter(m)) => {
            let mut k = m.lock().unwrap();
            match &mut *k {
                IterKind::List { list, idx } => {
                    let Some(Body::List(l)) = list.body() else { unreachable!() };
                    match l.get(*idx) {
                        Some(v) => {
                            v.incref();
                            *idx += 1;
                            Ok(Some(v))
                        }
                        None => Ok(None),
                    }
                }
                IterKind::Items { items } => Ok(items.next()),
                IterKind::Range { cur, end, step } => {
                    if (*step > 0 && *cur < *end) || (*step < 0 && *cur > *end) {
                        let v = *cur;
                        *cur += *step;
                        Ok(Some(Value::from_i64(v)))
                    } else {
                        Ok(None)
                    }
                }
                IterKind::Object { obj } => {
                    let obj = *obj;
                    drop(k);
                    let Some(&m) = obj.as_instance().and_then(|i| i.class().methods.get("__next__")) else {
                        return Err(th.raise("TypeError", format!("'{}' object is not an iterator", obj.type_name())));
                    };
                    match helpers::call_simple(th, m, Some(obj), &[]) {
                        Ok(v) => Ok(Some(v)),
                        Err(Raised) if is_stop_iteration(th) => {
                            clear_exception(th);
                            Ok(None)
                        }
                        Err(e) => Err(e),
                    }
                }
            }
        }
        Some(Body::Generator(_)) => generator::resume(th, it, Value::NONE, None),
        Some(Body::Channel(_)) => Ok(crate::threads::recv(it)),
        _ => Err(th.raise("TypeError", format!("'{}' object is not an iterator", it.type_name()))),
    }
}

/// Collects a borrowed iterable into owned items.
pub fn collect(th: &mut Thread, v: Value) -> VResult<Vec<Value>> {
    let it = get_iter(th, v)?;
    let mut out = Vec::new();
    loop {
        match iter_next(th, it) {
            Ok(Some(x)) => out.push(x),
            Ok(None) => break,
            Err(e) => {
                out.iter().for_each(|x| x.decref());
                it.decref();
                return Err(e);
            }
        }
    }
    it.decref();
    Ok(out)
}

// ---------------------------------------------------------------- dict and index helpers

fn unhashable(th: &mut Thread, k: Value) -> Raised {
    th.raise("TypeError", format!("unhashable type: '{}'", k.type_name()))
}

fn key_error(th: &mut Thread, k: Value) -> Raised {
    th.raise("KeyError", display::repr(k))
}

pub fn build_dict(th: &mut Thread, items: Vec<Value>) -> VResult<Value> {
    let d = Dict::default();
    let mut it = items.into_iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        match d.insert(k, v) {
            Ok(Some((k2, old))) => {
                k2.decref();
                old.decref();
            }
            Ok(None) => {}
            Err(()) => {
                let e = unhashable(th, k);
                k.decref();
                v.decref();
                it.for_each(|x| x.decref());
                for (k, v) in d.drain() {
                    k.decref();
                    v.decref();
                }
                return Err(e);
            }
        }
    }
    Ok(Obj::alloc(Body::Dict(d)))
}

/// `o[k]` for types other than list/tuple/str (borrowed result).
pub fn index_other(th: &mut Thread, o: Value, k: Value) -> VResult<Value> {
    match o.body() {
        Some(Body::Dict(d)) => match d.get(k) {
            Some(v) => Ok(v),
            None => Err(key_error(th, k)),
        },
        _ => {
            if o.is_dead() {
                return Err(th.raise("OwnershipError", "use of a moved value"));
            }
            Err(th.raise("TypeError", format!("'{}' object is not subscriptable", o.type_name())))
        }
    }
}

/// `o[k] = v` (v owned). Returns false if `o` does not support it. The key
/// is borrowed: a new reference is taken for new entries.
pub fn store_index_other(th: &mut Thread, o: Value, k: Value, v: Value) -> VResult<bool> {
    match o.body() {
        Some(Body::Dict(d)) => {
            if dict_hash_check(k).is_none() {
                return Err(unhashable(th, k));
            }
            k.incref();
            match d.insert(k, v) {
                Ok(Some((dup, old))) => {
                    dup.decref();
                    rc::defer_decref(old);
                }
                Ok(None) => {}
                Err(()) => {
                    k.decref();
                    return Err(unhashable(th, k));
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn dict_hash_check(k: Value) -> Option<u64> {
    crate::dict::hash_value(k)
}

pub fn swap_index_other(th: &mut Thread, o: Value, k: Value, v: Value) -> VResult<Option<Value>> {
    match o.body() {
        Some(Body::Dict(d)) => match d.swap(k, v) {
            Some(old) => Ok(Some(old)),
            None => Err(key_error(th, k)),
        },
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------- builtin functions

macro_rules! arg_int {
    ($th:expr, $v:expr, $what:expr) => {
        match $v.as_int() {
            Some(i) => i,
            None => return Err($th.raise("TypeError", format!("{} must be an int, not '{}'", $what, $v.type_name()))),
        }
    };
}

fn b_print(_th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let parts: Vec<String> = args.iter().map(|v| display::to_str(*v)).collect();
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", parts.join(" "));
    Ok(Value::UNIT)
}

fn b_str(_th: &mut Thread, args: &[Value]) -> VResult<Value> {
    match args.first() {
        None => Ok(Value::str("")),
        Some(v) if v.as_str().is_some() => {
            v.incref();
            Ok(*v)
        }
        Some(v) => Ok(Value::str(display::to_str(*v))),
    }
}

fn b_repr(_th: &mut Thread, args: &[Value]) -> VResult<Value> {
    Ok(Value::str(display::repr(args[0])))
}

fn b_int(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let v = args[0];
    if let Some(i) = v.as_int() {
        return Ok(Value::from_i64(i));
    }
    if let Some(b) = v.as_bool() {
        return Ok(Value::from_i64(b as i64));
    }
    match v.body() {
        Some(Body::Float(f)) => {
            if !f.is_finite() || f.abs() >= 9.223372036854775807e18 {
                return Err(th.raise("OverflowError", "float does not fit in int"));
            }
            Ok(Value::from_i64(*f as i64))
        }
        Some(Body::BigInt(b)) => match b.to_i64() {
            Some(i) => Ok(Value::from_i64(i)),
            None => Err(th.raise("OverflowError", "bigint does not fit in int")),
        },
        Some(Body::Str(s)) => match s.trim().replace('_', "").parse::<i64>() {
            Ok(i) => Ok(Value::from_i64(i)),
            Err(_) => Err(th.raise("ValueError", format!("invalid literal for int(): {}", display::repr_str(s)))),
        },
        _ => Err(th.raise("TypeError", format!("int() argument must be a string or a number, not '{}'", v.type_name()))),
    }
}

fn b_float(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let v = args[0];
    if let Some(i) = v.as_int() {
        return Ok(Value::float(i as f64));
    }
    match v.body() {
        Some(Body::Float(_)) => {
            v.incref();
            Ok(v)
        }
        Some(Body::BigInt(b)) => Ok(Value::float(b.to_f64().unwrap_or(f64::INFINITY))),
        Some(Body::Str(s)) => match s.trim().parse::<f64>() {
            Ok(f) => Ok(Value::float(f)),
            Err(_) => Err(th.raise("ValueError", format!("could not convert string to float: {}", display::repr_str(s)))),
        },
        _ => Err(th.raise("TypeError", format!("float() argument must be a string or a number, not '{}'", v.type_name()))),
    }
}

fn b_bool(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    Ok(Value::bool(helpers::is_truthy(th, args[0])?))
}

fn b_bigint(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let v = args[0];
    if let Some(i) = v.as_int() {
        return Ok(Obj::alloc(Body::BigInt(BigInt::from(i))));
    }
    match v.body() {
        Some(Body::BigInt(_)) => {
            v.incref();
            Ok(v)
        }
        Some(Body::Str(s)) => match s.trim().replace('_', "").parse::<BigInt>() {
            Ok(b) => Ok(Obj::alloc(Body::BigInt(b))),
            Err(_) => Err(th.raise("ValueError", format!("invalid literal for bigint(): {}", display::repr_str(s)))),
        },
        _ => Err(th.raise("TypeError", format!("bigint() argument must be a string or an int, not '{}'", v.type_name()))),
    }
}

fn b_len(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let v = args[0];
    let n = match v.body() {
        Some(Body::Str(s)) => s.chars().count(),
        Some(Body::Tuple(t)) => t.len(),
        Some(Body::List(l)) => l.len(),
        Some(Body::Dict(d)) => d.len(),
        Some(Body::Channel(_)) => return crate::helpers::call_simple(th, type_method(v, "__len__").unwrap(), Some(v), &[]),
        Some(Body::Instance(inst)) if inst.class().methods.contains_key("__len__") => {
            let m = inst.class().methods["__len__"];
            return helpers::call_simple(th, m, Some(v), &[]);
        }
        _ => return Err(th.raise("TypeError", format!("object of type '{}' has no len()", v.type_name()))),
    };
    Ok(Value::from_i64(n as i64))
}

fn b_range(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let (start, end, step) = match args.len() {
        1 => (0, arg_int!(th, args[0], "range() argument"), 1),
        2 => (arg_int!(th, args[0], "range() argument"), arg_int!(th, args[1], "range() argument"), 1),
        _ => (
            arg_int!(th, args[0], "range() argument"),
            arg_int!(th, args[1], "range() argument"),
            arg_int!(th, args[2], "range() argument"),
        ),
    };
    if step == 0 {
        return Err(th.raise("ValueError", "range() arg 3 must not be zero"));
    }
    Ok(new_iter(IterKind::Range { cur: start, end, step }))
}

fn b_abs(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let v = args[0];
    if let Some(i) = v.as_int() {
        return match i.checked_abs() {
            Some(a) => Ok(Value::from_i64(a)),
            None => Err(th.raise("OverflowError", "integer overflow in abs()")),
        };
    }
    match v.body() {
        Some(Body::Float(f)) => Ok(Value::float(f.abs())),
        Some(Body::BigInt(b)) => Ok(Obj::alloc(Body::BigInt(if b.sign() == num_bigint::Sign::Minus { -b } else { b.clone() }))),
        _ => Err(th.raise("TypeError", format!("bad operand type for abs(): '{}'", v.type_name()))),
    }
}

fn extremum(th: &mut Thread, args: &[Value], want_less: bool, name: &str) -> VResult<Value> {
    let owned_items;
    let items: &[Value] = if args.len() == 1 {
        owned_items = collect(th, args[0])?;
        &owned_items
    } else {
        owned_items = Vec::new();
        let _ = &owned_items;
        args
    };
    let Some(&first) = items.first() else {
        return Err(th.raise("ValueError", format!("{name}() arg is an empty sequence")));
    };
    let mut best = first;
    for &x in &items[1..] {
        let lt = helpers::lt(th, x, best);
        if lt == Value::UNINIT {
            if args.len() == 1 {
                items.iter().for_each(|v| v.decref());
            }
            return Err(Raised);
        }
        let gt = helpers::lt(th, best, x);
        if (want_less && lt == Value::TRUE) || (!want_less && gt == Value::TRUE) {
            best = x;
        }
    }
    best.incref();
    if args.len() == 1 {
        items.iter().for_each(|v| v.decref());
    }
    Ok(best)
}

fn b_min(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    extremum(th, args, true, "min")
}

fn b_max(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    extremum(th, args, false, "max")
}

fn b_sum(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let items = collect(th, args[0])?;
    let mut acc = if args.len() > 1 {
        args[1].incref();
        args[1]
    } else {
        Value::int_unchecked(0)
    };
    let mut err = false;
    for &x in &items {
        let r = helpers::add(th, acc, x);
        acc.decref();
        if r == Value::UNINIT {
            err = true;
            acc = Value::UNIT;
            break;
        }
        acc = r;
    }
    items.iter().for_each(|v| v.decref());
    if err {
        Err(Raised)
    } else {
        Ok(acc)
    }
}

fn b_some(_th: &mut Thread, args: &[Value]) -> VResult<Value> {
    // `Some(v)` is represented as `v` itself; the parameter is `keep`.
    Ok(args[0])
}

fn b_rc_run(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let limit = match args.first() {
        Some(v) => Some(arg_int!(th, *v, "rc_run() limit").max(0) as usize),
        None => None,
    };
    let n = th.rc_run(limit);
    Ok(Value::from_i64(n as i64))
}

fn b_rc_pending(_th: &mut Thread, _args: &[Value]) -> VResult<Value> {
    Ok(Value::from_i64(rc::pending_count() as i64))
}

fn b_rc_auto(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    th.rc_auto = helpers::is_truthy(th, args[0])?;
    Ok(Value::UNIT)
}

fn b_rc_live(_th: &mut Thread, _args: &[Value]) -> VResult<Value> {
    Ok(Value::from_i64(rc::live_objects() as i64))
}

fn b_list(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let items = match args.first() {
        Some(v) => collect(th, *v)?,
        None => Vec::new(),
    };
    Ok(Obj::list(items))
}

fn b_tuple(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let items = match args.first() {
        Some(v) => collect(th, *v)?,
        None => Vec::new(),
    };
    Ok(Obj::alloc(Body::Tuple(items.into_boxed_slice())))
}

fn b_dict(_th: &mut Thread, _args: &[Value]) -> VResult<Value> {
    Ok(Obj::alloc(Body::Dict(Dict::default())))
}

fn b_isinstance(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let (v, t) = (args[0], args[1]);
    let r = match t.body() {
        Some(Body::Class(_)) => v.as_instance().is_some_and(|i| i.class == t),
        Some(Body::Trait(_)) => v.as_instance().is_some_and(|i| i.class().implements(t)),
        _ => return Err(th.raise("TypeError", "isinstance() arg 2 must be a class or trait")),
    };
    Ok(Value::bool(r))
}

fn b_iter(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    get_iter(th, args[0])
}

fn b_next(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    match iter_next(th, args[0])? {
        Some(v) => Ok(v),
        None => Err(th.raise("StopIteration", "")),
    }
}

fn b_sorted(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let mut items = collect(th, args[0])?;
    if let Err(e) = sort_values(th, &mut items) {
        items.iter().for_each(|v| v.decref());
        return Err(e);
    }
    Ok(Obj::list(items))
}

fn sort_values(th: &mut Thread, items: &mut [Value]) -> VResult<()> {
    let mut failed = false;
    items.sort_by(|a, b| {
        if failed {
            return std::cmp::Ordering::Equal;
        }
        match helpers::lt(th, *a, *b) {
            Value::TRUE => std::cmp::Ordering::Less,
            Value::FALSE => {
                if values_eq(*a, *b) {
                    std::cmp::Ordering::Equal
                } else {
                    std::cmp::Ordering::Greater
                }
            }
            _ => {
                failed = true;
                std::cmp::Ordering::Equal
            }
        }
    });
    if failed {
        Err(Raised)
    } else {
        Ok(())
    }
}

fn b_enumerate(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let items = collect(th, args[0])?;
    let out = items
        .into_iter()
        .enumerate()
        .map(|(i, v)| Obj::alloc(Body::Tuple(vec![Value::from_i64(i as i64), v].into_boxed_slice())))
        .collect();
    Ok(items_iter(out))
}

fn b_zip(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let a = collect(th, args[0])?;
    let b = match collect(th, args[1]) {
        Ok(b) => b,
        Err(e) => {
            a.iter().for_each(|v| v.decref());
            return Err(e);
        }
    };
    let n = a.len().min(b.len());
    let mut out = Vec::with_capacity(n);
    let (mut ai, mut bi) = (a.into_iter(), b.into_iter());
    for _ in 0..n {
        out.push(Obj::alloc(Body::Tuple(vec![ai.next().unwrap(), bi.next().unwrap()].into_boxed_slice())));
    }
    ai.for_each(|v| v.decref());
    bi.for_each(|v| v.decref());
    Ok(items_iter(out))
}

fn b_ord(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    match args[0].as_str().map(|s| s.chars().collect::<Vec<_>>()) {
        Some(c) if c.len() == 1 => Ok(Value::from_i64(c[0] as i64)),
        _ => Err(th.raise("TypeError", "ord() expected a character")),
    }
}

fn b_chr(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let i = arg_int!(th, args[0], "chr() argument");
    match u32::try_from(i).ok().and_then(char::from_u32) {
        Some(c) => Ok(Value::str(c.to_string())),
        None => Err(th.raise("ValueError", "chr() arg not in range")),
    }
}

fn b_hash(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    match crate::dict::hash_value(args[0]) {
        Some(h) => Ok(Value::from_i64(h as i64)),
        None => Err(unhashable(th, args[0])),
    }
}

fn b_spawn(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    crate::threads::spawn(th, args)
}

fn b_lock(_th: &mut Thread, _args: &[Value]) -> VResult<Value> {
    Ok(crate::threads::new_lock())
}

fn b_channel(_th: &mut Thread, _args: &[Value]) -> VResult<Value> {
    Ok(crate::threads::new_channel())
}

pub(crate) type BuiltinDefPub = BuiltinDef;

pub(crate) struct BuiltinDef {
    name: &'static str,
    f: BuiltinFn,
    min: usize,
    max: usize,
    keep: u64,
}

const fn def(name: &'static str, f: BuiltinFn, min: usize, max: usize, keep: u64) -> BuiltinDef {
    BuiltinDef { name, f, min, max, keep }
}

impl BuiltinDef {
    pub(crate) const fn new(name: &'static str, f: BuiltinFn, min: usize, max: usize, keep: u64) -> BuiltinDef {
        def(name, f, min, max, keep)
    }
}

const FUNCTIONS: &[BuiltinDef] = &[
    def("print", b_print, 0, 64, 0),
    def("str", b_str, 0, 1, 0),
    def("repr", b_repr, 1, 1, 0),
    def("int", b_int, 1, 1, 0),
    def("float", b_float, 1, 1, 0),
    def("bool", b_bool, 1, 1, 0),
    def("bigint", b_bigint, 1, 1, 0),
    def("len", b_len, 1, 1, 0),
    def("range", b_range, 1, 3, 0),
    def("abs", b_abs, 1, 1, 0),
    def("min", b_min, 1, 64, 0),
    def("max", b_max, 1, 64, 0),
    def("sum", b_sum, 1, 2, 0),
    def("Some", b_some, 1, 1, 1),
    def("rc_run", b_rc_run, 0, 1, 0),
    def("rc_pending", b_rc_pending, 0, 0, 0),
    def("rc_auto", b_rc_auto, 1, 1, 0),
    def("rc_live", b_rc_live, 0, 0, 0),
    def("list", b_list, 0, 1, 0),
    def("tuple", b_tuple, 0, 1, 0),
    def("dict", b_dict, 0, 0, 0),
    def("isinstance", b_isinstance, 2, 2, 0),
    def("iter", b_iter, 1, 1, 0),
    def("next", b_next, 1, 1, 0),
    def("sorted", b_sorted, 1, 1, 0),
    def("enumerate", b_enumerate, 1, 1, 0),
    def("zip", b_zip, 2, 2, 0),
    def("ord", b_ord, 1, 1, 0),
    def("chr", b_chr, 1, 1, 0),
    def("hash", b_hash, 1, 1, 0),
    // Threads: arguments to `spawn` move into the new thread.
    def("spawn", b_spawn, 1, 16, !1),
    def("Lock", b_lock, 0, 0, 0),
    def("Channel", b_channel, 0, 0, 0),
];

/// Builtin traits: (name, implied traits).
const TRAITS: &[(&str, &[&str])] = &[
    ("Exception", &[]),
    ("LookupError", &["Exception"]),
    ("ArithmeticError", &["Exception"]),
    ("Iterator", &[]),
];

/// Builtin exception classes: (name, implemented traits).
const EXCEPTIONS: &[(&str, &[&str])] = &[
    ("TypeError", &["Exception"]),
    ("ValueError", &["Exception"]),
    ("KeyError", &["LookupError", "Exception"]),
    ("IndexError", &["LookupError", "Exception"]),
    ("ZeroDivisionError", &["ArithmeticError", "Exception"]),
    ("OverflowError", &["ArithmeticError", "Exception"]),
    ("NameError", &["Exception"]),
    ("AttributeError", &["Exception"]),
    ("OwnershipError", &["Exception"]),
    ("RecursionError", &["Exception"]),
    ("StopIteration", &["Exception"]),
    ("RuntimeError", &["Exception"]),
    ("AssertionError", &["Exception"]),
    ("NotImplementedError", &["Exception"]),
    ("GeneratorExit", &["Exception"]),
];

/// Keep mask of a builtin function, for static ownership checking.
pub fn keep_mask(name: &str) -> Option<u64> {
    FUNCTIONS.iter().find(|d| d.name == name).map(|d| d.keep)
}

/// Keep mask of a builtin type method (bit 0 is `self`).
pub fn method_keep_mask(name: &str) -> Option<u64> {
    method_table().values().find_map(|m| m.get(name)).and_then(|v| match v.body() {
        Some(Body::Builtin(b)) => Some(b.keep_mask),
        _ => None,
    })
}

/// All names the compiler should treat as predefined globals.
pub fn names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = FUNCTIONS.iter().map(|d| d.name).collect();
    v.extend(TRAITS.iter().map(|t| t.0));
    v.extend(EXCEPTIONS.iter().map(|e| e.0));
    v
}

fn builtin_value(d: &BuiltinDef) -> Value {
    Obj::alloc_immortal(Body::Builtin(Builtin { name: d.name, f: d.f, min_args: d.min, max_args: d.max, keep_mask: d.keep }))
}

/// Creates the builtin objects: (name, value) pairs plus the exception table.
pub fn create(next_id: &std::sync::atomic::AtomicU32) -> (Vec<(&'static str, Value)>, crate::interp::Builtins) {
    use std::sync::atomic::Ordering;
    let mut all: Vec<(&'static str, Value)> = FUNCTIONS.iter().map(|d| (d.name, builtin_value(d))).collect();
    let mut traits: Vec<(&'static str, Value)> = Vec::new();
    for (name, implies) in TRAITS {
        let implies = implies.iter().map(|n| traits.iter().find(|(m, _)| m == n).unwrap().1).collect();
        let methods = if *name == "Iterator" { vec![("__next__".to_string(), None)] } else { Vec::new() };
        let t = Obj::alloc_immortal(Body::Trait(Trait {
            name: name.to_string(),
            id: next_id.fetch_add(1, Ordering::Relaxed),
            methods,
            implies,
        }));
        traits.push((name, t));
    }
    let mut classes = Vec::new();
    for (name, impls) in EXCEPTIONS {
        let tr = impls.iter().map(|n| traits.iter().find(|(m, _)| m == n).unwrap().1).collect();
        let c = Obj::alloc_immortal(Body::Class(Class {
            name: name.to_string(),
            id: next_id.fetch_add(1, Ordering::Relaxed),
            fields: vec![Field { name: "msg".into(), ty: TypeDesc::Str, kind: SlotKind::Value }],
            methods: HashMap::new(),
            traits: tr,
            is_exception: true,
        }));
        classes.push((*name, c));
    }
    let exception = traits[0].1;
    all.extend(traits.iter().copied());
    all.extend(classes.iter().copied());
    (all, crate::interp::Builtins { exception, classes })
}

// ---------------------------------------------------------------- builtin type methods

fn self_str(v: Value) -> &'static str {
    v.as_str().unwrap()
}

fn m_upper(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(Value::str(self_str(a[0]).to_uppercase()))
}

fn m_lower(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(Value::str(self_str(a[0]).to_lowercase()))
}

fn m_strip(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(Value::str(self_str(a[0]).trim()))
}

fn m_lstrip(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(Value::str(self_str(a[0]).trim_start()))
}

fn m_rstrip(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(Value::str(self_str(a[0]).trim_end()))
}

fn str_arg(th: &mut Thread, v: Value, what: &str) -> VResult<&'static str> {
    v.as_str().ok_or_else(|| th.raise("TypeError", format!("{what} must be a str, not '{}'", v.type_name())))
}

fn m_split(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let s = self_str(a[0]);
    let parts: Vec<Value> = match a.get(1) {
        Some(sep) => {
            let sep = str_arg(th, *sep, "separator")?;
            if sep.is_empty() {
                return Err(th.raise("ValueError", "empty separator"));
            }
            s.split(sep).map(Value::str).collect()
        }
        None => s.split_whitespace().map(Value::str).collect(),
    };
    Ok(Obj::list(parts))
}

fn m_join(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let sep = self_str(a[0]);
    let items = collect(th, a[1])?;
    let mut out = String::new();
    let mut bad = None;
    for (i, x) in items.iter().enumerate() {
        match x.as_str() {
            Some(s) => {
                if i > 0 {
                    out.push_str(sep);
                }
                out.push_str(s);
            }
            None => {
                bad = Some(x.type_name());
                break;
            }
        }
    }
    items.iter().for_each(|v| v.decref());
    if let Some(t) = bad {
        return Err(th.raise("TypeError", format!("join() expects str items, found '{t}'")));
    }
    Ok(Value::str(out))
}

fn m_startswith(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let p = str_arg(th, a[1], "prefix")?;
    Ok(Value::bool(self_str(a[0]).starts_with(p)))
}

fn m_endswith(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let p = str_arg(th, a[1], "suffix")?;
    Ok(Value::bool(self_str(a[0]).ends_with(p)))
}

fn m_find(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let s = self_str(a[0]);
    let p = str_arg(th, a[1], "substring")?;
    Ok(Value::from_i64(match s.find(p) {
        Some(b) => s[..b].chars().count() as i64,
        None => -1,
    }))
}

fn m_replace(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let from = str_arg(th, a[1], "old")?;
    let to = str_arg(th, a[2], "new")?;
    Ok(Value::str(self_str(a[0]).replace(from, to)))
}

fn m_str_count(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let p = str_arg(th, a[1], "substring")?;
    Ok(Value::from_i64(self_str(a[0]).matches(p).count() as i64))
}

fn m_isdigit(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let s = self_str(a[0]);
    Ok(Value::bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())))
}

fn m_isalpha(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let s = self_str(a[0]);
    Ok(Value::bool(!s.is_empty() && s.chars().all(|c| c.is_alphabetic())))
}

fn list_of(v: Value) -> &'static ListObj {
    match v.body() {
        Some(Body::List(l)) => l,
        _ => unreachable!(),
    }
}

fn m_append(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    list_of(a[0]).push(a[1]);
    Ok(Value::UNIT)
}

/// `l.pop()` / `l.pop(i)`: `Some(item)` (owned), or `None` if the list is
/// empty or `i` is out of range.
fn m_pop(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let i = match a.get(1) {
        Some(i) => Some(arg_int!(th, *i, "index")),
        None => None,
    };
    Ok(list_of(a[0])
        .write(|w| match i {
            None => w.pop(),
            Some(i) if (i as u64) < w.len() as u64 => Some(w.remove(i as usize)),
            Some(_) => None,
        })
        .unwrap_or(Value::NONE))
}

/// `l.first()` / `l.last()`: `Some(item)` (a new reference) or `None`.
fn m_first(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let v = list_of(a[0]).get(0).unwrap_or(Value::NONE);
    v.incref();
    Ok(v)
}

fn m_last(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let l = list_of(a[0]);
    let v = l.len().checked_sub(1).and_then(|i| l.get(i)).unwrap_or(Value::NONE);
    v.incref();
    Ok(v)
}

/// `l.insert(i, x)`: `i` may be at most the length (appends at the end).
fn m_insert(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let i = match a[1].as_int() {
        Some(i) => i,
        None => {
            a[2].decref();
            return Err(th.raise("TypeError", "list index must be an int"));
        }
    };
    let l = list_of(a[0]);
    let ok = l.write(|w| {
        if (i as u64) <= w.len() as u64 {
            w.insert(i as usize, a[2]);
            true
        } else {
            false
        }
    });
    if !ok {
        a[2].decref();
        return Err(helpers::index_error(th, i, "list", l.len()));
    }
    Ok(Value::UNIT)
}

fn m_extend(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let items = collect(th, a[1])?;
    list_of(a[0]).write(|w| w.extend(items));
    Ok(Value::UNIT)
}

fn m_clear(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    match a[0].body() {
        Some(Body::List(l)) => l.write(|w| w.clear()).into_iter().for_each(rc::defer_decref),
        Some(Body::Dict(d)) => {
            for (k, v) in d.drain() {
                rc::defer_decref(k);
                rc::defer_decref(v);
            }
        }
        _ => {}
    }
    Ok(Value::UNIT)
}

fn m_index(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let found = match a[0].body() {
        Some(Body::List(l)) => l.snapshot().iter().position(|v| values_eq(*v, a[1])),
        Some(Body::Tuple(t)) => t.iter().position(|v| values_eq(*v, a[1])),
        _ => None,
    };
    match found {
        Some(i) => Ok(Value::from_i64(i as i64)),
        None => Err(th.raise("ValueError", format!("{} is not in the sequence", display::repr(a[1])))),
    }
}

fn m_seq_count(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let n = match a[0].body() {
        Some(Body::List(l)) => l.snapshot().iter().filter(|v| values_eq(**v, a[1])).count(),
        Some(Body::Tuple(t)) => t.iter().filter(|v| values_eq(**v, a[1])).count(),
        _ => 0,
    };
    Ok(Value::from_i64(n as i64))
}

fn m_remove(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    // Equality is pure, so the search can run under the writer lock.
    let removed = list_of(a[0]).write(|w| {
        let i = (0..w.len()).find(|&i| values_eq(w.get(i), a[1]))?;
        Some(w.remove(i))
    });
    match removed {
        Some(v) => {
            rc::defer_decref(v);
            Ok(Value::UNIT)
        }
        None => Err(th.raise("ValueError", "list.remove(x): x not in list")),
    }
}

fn m_reverse(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    list_of(a[0]).write(|w| w.reverse());
    Ok(Value::UNIT)
}

/// Sorts outside the writer lock (comparisons may run user code), then
/// writes the order back if nothing changed the list meanwhile.
fn m_sort(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let l = list_of(a[0]);
    // References keep the items alive if the comparisons modify the list.
    let (mut items, version) = l.write(|w| {
        let items = w.items();
        items.iter().for_each(|v| v.incref());
        (items, w.final_version())
    });
    let r = sort_values(th, &mut items);
    let ok = r.is_ok()
        && l.write(|w| {
            if w.prior_version() != version || w.len() != items.len() {
                return false;
            }
            w.permute(&items);
            true
        });
    items.into_iter().for_each(rc::defer_decref);
    r?;
    if !ok {
        return Err(th.raise("RuntimeError", "list modified during sort"));
    }
    Ok(Value::UNIT)
}

fn m_list_copy(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let items = list_of(a[0]).snapshot();
    items.iter().for_each(|v| v.incref());
    Ok(Obj::list(items))
}

fn dict_of(v: Value) -> &'static Dict {
    match v.body() {
        Some(Body::Dict(d)) => d,
        _ => unreachable!(),
    }
}

fn m_dict_get(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let v = dict_of(a[0]).get(a[1]).or_else(|| a.get(2).copied()).unwrap_or(Value::NONE);
    v.incref();
    Ok(v)
}

fn m_keys(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let ks: Vec<Value> = dict_of(a[0]).entries().into_iter().map(|(k, _)| k).collect();
    ks.iter().for_each(|v| v.incref());
    Ok(Obj::list(ks))
}

fn m_values(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let vs: Vec<Value> = dict_of(a[0]).entries().into_iter().map(|(_, v)| v).collect();
    vs.iter().for_each(|v| v.incref());
    Ok(Obj::list(vs))
}

fn m_items(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let items = dict_of(a[0])
        .entries()
        .into_iter()
        .map(|(k, v)| {
            k.incref();
            v.incref();
            Obj::alloc(Body::Tuple(vec![k, v].into_boxed_slice()))
        })
        .collect();
    Ok(Obj::list(items))
}

fn m_dict_pop(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    match dict_of(a[0]).remove(a[1]) {
        Some((k, v)) => {
            rc::defer_decref(k);
            Ok(v)
        }
        None => match a.get(2) {
            Some(d) => {
                d.incref();
                Ok(*d)
            }
            None => Err(key_error(th, a[1])),
        },
    }
}

fn m_dict_copy(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let d = Dict::default();
    for (k, v) in dict_of(a[0]).entries() {
        k.incref();
        v.incref();
        let _ = d.insert(k, v);
    }
    Ok(Obj::alloc(Body::Dict(d)))
}

fn m_send(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    match generator::resume(th, a[0], a[1], None)? {
        Some(v) => Ok(v),
        None => Err(th.raise("StopIteration", "")),
    }
}

fn m_throw(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let mut e = a[1];
    if let Some(c) = e.as_class() {
        if c.is_exception {
            e = helpers::call_simple(th, e, None, &[])?;
        }
    }
    if !helpers::implements_exception(th, e) {
        e.decref();
        return Err(th.raise("TypeError", "throw() expects an exception"));
    }
    match generator::resume(th, a[0], Value::NONE, Some(e))? {
        Some(v) => Ok(v),
        None => Err(th.raise("StopIteration", "")),
    }
}

fn m_close(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let ge = th.vm.builtins.class("GeneratorExit");
    let e = helpers::new_exception(ge, Value::str(""));
    match generator::resume(th, a[0], Value::NONE, Some(e)) {
        Ok(Some(v)) => {
            v.decref();
            Err(th.raise("RuntimeError", "generator ignored GeneratorExit"))
        }
        Ok(None) => Ok(Value::UNIT),
        Err(Raised) => {
            let is_exit = th.exc.as_ref().is_some_and(|x| x.value.as_instance().is_some_and(|i| i.class == ge));
            if is_exit || is_stop_iteration(th) {
                clear_exception(th);
                Ok(Value::UNIT)
            } else {
                Err(Raised)
            }
        }
    }
}

/// `opt.some_or(default)`: the value if it is not `None`, else `default`.
fn m_some_or(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    if a[0] == Value::NONE {
        Ok(a[1])
    } else {
        a[1].decref();
        a[0].incref();
        Ok(a[0])
    }
}

type MethodTable = HashMap<&'static str, HashMap<&'static str, Value>>;

fn method_table() -> &'static MethodTable {
    static TABLE: OnceLock<MethodTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mk = |defs: &[BuiltinDef]| -> HashMap<&'static str, Value> {
            defs.iter().map(|d| (d.name, builtin_value(d))).collect()
        };
        let mut t = HashMap::new();
        t.insert(
            "str",
            mk(&[
                def("upper", m_upper, 1, 1, 0),
                def("lower", m_lower, 1, 1, 0),
                def("strip", m_strip, 1, 1, 0),
                def("lstrip", m_lstrip, 1, 1, 0),
                def("rstrip", m_rstrip, 1, 1, 0),
                def("split", m_split, 1, 2, 0),
                def("join", m_join, 2, 2, 0),
                def("startswith", m_startswith, 2, 2, 0),
                def("endswith", m_endswith, 2, 2, 0),
                def("find", m_find, 2, 2, 0),
                def("replace", m_replace, 3, 3, 0),
                def("count", m_str_count, 2, 2, 0),
                def("isdigit", m_isdigit, 1, 1, 0),
                def("isalpha", m_isalpha, 1, 1, 0),
            ]),
        );
        t.insert(
            "list",
            mk(&[
                def("append", m_append, 2, 2, 1 << 1),
                def("pop", m_pop, 1, 2, 0),
                def("first", m_first, 1, 1, 0),
                def("last", m_last, 1, 1, 0),
                def("insert", m_insert, 3, 3, 1 << 2),
                def("extend", m_extend, 2, 2, 0),
                def("clear", m_clear, 1, 1, 0),
                def("index", m_index, 2, 2, 0),
                def("count", m_seq_count, 2, 2, 0),
                def("remove", m_remove, 2, 2, 0),
                def("reverse", m_reverse, 1, 1, 0),
                def("sort", m_sort, 1, 1, 0),
                def("copy", m_list_copy, 1, 1, 0),
            ]),
        );
        t.insert("tuple", mk(&[def("index", m_index, 2, 2, 0), def("count", m_seq_count, 2, 2, 0)]));
        t.insert(
            "dict",
            mk(&[
                def("get", m_dict_get, 2, 3, 0),
                def("keys", m_keys, 1, 1, 0),
                def("values", m_values, 1, 1, 0),
                def("items", m_items, 1, 1, 0),
                def("pop", m_dict_pop, 2, 3, 0),
                def("clear", m_clear, 1, 1, 0),
                def("copy", m_dict_copy, 1, 1, 0),
            ]),
        );
        t.insert(
            "generator",
            mk(&[def("send", m_send, 2, 2, 1 << 1), def("throw", m_throw, 2, 2, 1 << 1), def("close", m_close, 1, 1, 0)]),
        );
        t.insert("any", mk(&[def("some_or", m_some_or, 2, 2, 1 << 1)]));
        for (ty, defs) in crate::threads::methods() {
            t.insert(ty, mk(&defs));
        }
        t
    })
}

/// Method `name` of a builtin type (or of every value, like `some_or`).
pub fn type_method(recv: Value, name: &str) -> Option<Value> {
    let ty = match recv.body() {
        Some(Body::Str(_)) => "str",
        Some(Body::List(_)) => "list",
        Some(Body::Tuple(_)) => "tuple",
        Some(Body::Dict(_)) => "dict",
        Some(Body::Generator(_)) => "generator",
        Some(Body::Thread(_)) => "thread",
        Some(Body::Lock(_)) => "lock",
        Some(Body::Channel(_)) => "channel",
        _ => "",
    };
    let t = method_table();
    t.get(ty).and_then(|m| m.get(name)).or_else(|| t["any"].get(name)).copied()
}
