//! `str()` / `repr()` formatting.

use crate::array::{Elem, Scalar};
use crate::object::Body;
use crate::value::Value;

pub fn format_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == f.trunc() && f.abs() < 1e16 {
        return format!("{f:.1}");
    }
    let s = format!("{f}");
    if s.contains('e') || s.len() <= 17 {
        return s;
    }
    format!("{f:e}")
}

/// Shortest form that reads back as the same `f32`.
fn format_f32(f: f32) -> String {
    if !f.is_finite() || (f == f.trunc() && f.abs() < 1e7) {
        return format_float(f as f64);
    }
    format!("{f}")
}

fn fmt(v: Value, repr: bool, out: &mut String, depth: usize) {
    if depth > 50 {
        out.push_str("...");
        return;
    }
    if let Some(i) = v.as_small_int() {
        out.push_str(&i.to_string());
        return;
    }
    match v {
        Value::UNINIT => return out.push_str("<uninitialized>"),
        Value::MOVED => return out.push_str("<moved>"),
        Value::NONE => return out.push_str("None"),
        Value::UNIT => return out.push_str("()"),
        Value::TRUE => return out.push_str("True"),
        Value::FALSE => return out.push_str("False"),
        _ => {}
    }
    let Some(body) = v.body() else {
        out.push_str("<invalid>");
        return;
    };
    match body {
        Body::Int(i) => out.push_str(&i.to_string()),
        Body::Float(f) => out.push_str(&format_float(*f)),
        Body::BigInt(b) => out.push_str(&b.to_string()),
        Body::Str(s) => {
            if repr {
                out.push_str(&repr_str(s));
            } else {
                out.push_str(s);
            }
        }
        Body::Tuple(items) => {
            out.push('(');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                fmt(*x, true, out, depth + 1);
            }
            if items.len() == 1 {
                out.push(',');
            }
            out.push(')');
        }
        Body::List(items) => {
            let items = items.snapshot();
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                fmt(*x, true, out, depth + 1);
            }
            out.push(']');
        }
        Body::Function(f) => out.push_str(&format!("<function {}>", f.code.name)),
        Body::Builtin(b) => out.push_str(&format!("<builtin {}>", b.name)),
        Body::BoundMethod { func, .. } => {
            out.push_str("<bound method ");
            fmt(*func, false, out, depth + 1);
            out.push('>');
        }
        Body::Class(c) => out.push_str(&format!("<class {}>", c.name)),
        Body::NativeClass(c) => out.push_str(&format!("<class {}.{}>", c.module, c.name)),
        Body::NativeFn(f) => out.push_str(&format!("<native function {}>", f.name)),
        Body::Native(n) => out.push_str(&n.display()),
        Body::Instantiated { func, targs } => {
            out.push_str("<function ");
            if let Some(Body::Function(f)) = func.body() {
                out.push_str(&f.code.name);
            }
            let names: Vec<&str> = targs.iter().map(|t| crate::rtype::name(*t)).collect();
            out.push_str(&format!("::<{}>>", names.join(", ")));
        }
        Body::Dict(d) => {
            out.push('{');
            for (i, (k, v)) in d.entries().into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                fmt(k, true, out, depth + 1);
                out.push_str(": ");
                fmt(v, true, out, depth + 1);
            }
            out.push('}');
        }
        Body::Array(a) => {
            out.push_str("array(");
            out.push_str(a.elem.name());
            out.push_str(", [");
            for i in 0..a.len() {
                if i > 0 {
                    out.push_str(", ");
                }
                match a.get_num(i) {
                    Some(Scalar::Int(n)) => out.push_str(&n.to_string()),
                    Some(Scalar::Float(f)) if a.elem == Elem::F32 => out.push_str(&format_f32(f as f32)),
                    Some(Scalar::Float(f)) => out.push_str(&format_float(f)),
                    Some(Scalar::Bool(b)) => out.push_str(if b { "True" } else { "False" }),
                    None => {}
                }
            }
            out.push_str("])");
        }
        Body::ElemType(e) => out.push_str(e.name()),
        Body::Type(t) => out.push_str(&t.name),
        Body::Iter(_) => out.push_str("<iterator>"),
        Body::Thread(_) => out.push_str("<Thread>"),
        Body::Lock(_) => out.push_str("<Lock>"),
        Body::Channel(c) => out.push_str(&format!("<{}>", crate::rtype::name(c.ty))),
        Body::Generator(g) => out.push_str(&format!("<generator {}>", g.name())),
        Body::Trait(t) => out.push_str(&format!("<trait {}>", t.name)),
        Body::Instance(inst) => {
            let class = inst.class();
            if class.is_exception {
                // `str(e)` is the message; `repr(e)` is `Name('message')`.
                let msg = exception_message(v);
                if repr {
                    out.push_str(&format!("{}({})", class.name, repr_str(&msg)));
                } else {
                    out.push_str(&msg);
                }
                return;
            }
            out.push_str(&class.name);
            out.push('(');
            for (i, f) in class.fields.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&f.name);
                out.push('=');
                let fv = inst.get_owned(i);
                fmt(fv, true, out, depth + 1);
                fv.decref();
            }
            out.push(')');
        }
    }
}

/// The message of an exception: its `msg` field if it has one.
pub fn exception_message(v: Value) -> String {
    let Some(inst) = v.as_instance() else { return to_str(v) };
    match inst.class().field_index("msg") {
        Some(i) => {
            let v = inst.get_owned(i);
            let s = to_str(v);
            v.decref();
            s
        }
        None => String::new(),
    }
}

pub fn repr_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(q);
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\\' => out.push_str("\\\\"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

pub fn to_str(v: Value) -> String {
    let mut s = String::new();
    fmt(v, false, &mut s, 0);
    s
}

pub fn repr(v: Value) -> String {
    let mut s = String::new();
    fmt(v, true, &mut s, 0);
    s
}
