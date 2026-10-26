//! Runtime checks for type annotations.

use crate::code::TypeDesc;
use crate::interp::Thread;
use crate::object::Body;
use crate::value::Value;

pub fn check(th: &Thread, v: Value, t: &TypeDesc) -> bool {
    match t {
        TypeDesc::Any => v != Value::NONE,
        TypeDesc::Int => v.as_int().is_some(),
        // An int is accepted where a float is expected (numeric tower).
        TypeDesc::Float => v.as_float().is_some() || v.as_int().is_some(),
        TypeDesc::Bool => v.as_bool().is_some(),
        TypeDesc::BigInt => matches!(v.body(), Some(Body::BigInt(_))) || v.as_int().is_some(),
        TypeDesc::Str => v.as_str().is_some(),
        TypeDesc::Unit => v == Value::UNIT,
        TypeDesc::Tuple => matches!(v.body(), Some(Body::Tuple(_))),
        TypeDesc::List => matches!(v.body(), Some(Body::List(_))),
        TypeDesc::Dict => matches!(v.body(), Some(Body::Dict(_))),
        TypeDesc::Array(e) => v.as_array().is_some_and(|a| e.is_none_or(|e| a.elem == e)),
        TypeDesc::Function => matches!(
            v.body(),
            Some(Body::Function(_) | Body::Builtin(_) | Body::BoundMethod { .. } | Body::Class(_))
        ),
        TypeDesc::Builtin(n) => v.is_ptr() && v.type_name() == *n,
        TypeDesc::Option(inner) => v == Value::NONE || check(th, v, inner),
        TypeDesc::Global(idx, _) => {
            let target = th.vm.global(*idx);
            match target.body() {
                Some(Body::Class(_)) => v.as_instance().is_some_and(|i| i.class == target),
                Some(Body::Trait(_)) => match v.body() {
                    Some(Body::Instance(i)) => i.class().implements(target),
                    Some(Body::Generator(_)) => trait_named(target, "Iterator"),
                    _ => false,
                },
                _ => false,
            }
        }
    }
}

fn trait_named(t: Value, name: &str) -> bool {
    t.as_trait().is_some_and(|t| t.name == name)
}
