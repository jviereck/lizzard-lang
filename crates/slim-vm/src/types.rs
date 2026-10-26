//! Resolving written types to runtime type values, and type errors.

use crate::code::{Code, TypeDesc};
use crate::interp::{Thread, VResult};
use crate::object::{Body, Field};
use crate::rtype::{self, Ctor, RtType};
use crate::value::Value;

/// The runtime type for `d`, or the name of a class or trait that is not
/// defined yet.
pub fn materialize(th: &Thread, d: &TypeDesc) -> Result<Value, String> {
    match d {
        TypeDesc::Builtin(c, args) => {
            let args = args.iter().map(|a| materialize(th, a)).collect::<Result<Vec<_>, _>>()?;
            Ok(rtype::intern(*c, &args))
        }
        TypeDesc::Global(idx, n) => {
            let v = th.vm.global(*idx);
            match v.body() {
                Some(Body::Class(_)) => Ok(rtype::prim(Ctor::Class(v))),
                Some(Body::Trait(_)) => Ok(rtype::prim(Ctor::Trait(v))),
                _ => Err(n.clone()),
            }
        }
    }
}

fn undefined(th: &mut Thread, n: &str) -> crate::interp::Raised {
    th.raise("NameError", format!("type `{n}` is used before it is defined"))
}

/// The runtime type of `code.types[idx]` (cached in the code object).
pub fn code_type(th: &mut Thread, code: &Code, idx: u32) -> VResult<&'static RtType> {
    let slot = &code.type_cache[idx as usize];
    if let Some(v) = slot.get() {
        return Ok(rtype::as_type(*v).unwrap());
    }
    match materialize(th, &code.types[idx as usize]) {
        Ok(v) => Ok(rtype::as_type(*slot.get_or_init(|| v)).unwrap()),
        Err(n) => Err(undefined(th, &n)),
    }
}

/// The runtime type of a field (cached in the field).
pub fn field_type(th: &mut Thread, f: &Field) -> VResult<&'static RtType> {
    if let Some(v) = f.rt.get() {
        return Ok(rtype::as_type(*v).unwrap());
    }
    match materialize(th, &f.ty) {
        Ok(v) => Ok(rtype::as_type(*f.rt.get_or_init(|| v)).unwrap()),
        Err(n) => Err(undefined(th, &n)),
    }
}

/// The name of a value's type for error messages: the full reified type for
/// containers (`list<int>`).
pub fn value_type_name(v: Value) -> String {
    match v.body() {
        _ if rtype::container_type(v).is_some() => rtype::container_type(v).unwrap().name.clone(),
        Some(Body::Tuple(_)) => rtype::name(rtype::type_of(v)).to_string(),
        _ => v.type_name(),
    }
}

/// `TypeError` for storing `v` in a container of type `container`.
pub fn elem_error(th: &mut Thread, v: Value, container: &str) -> crate::interp::Raised {
    th.raise("TypeError", format!("cannot store `{}` in `{container}`", value_type_name(v)))
}

/// `TypeError` for using `k` as a key of a dict of type `container`.
pub fn key_type_error(th: &mut Thread, k: Value, container: &str) -> crate::interp::Raised {
    th.raise("TypeError", format!("cannot use `{}` as a key of `{container}`", value_type_name(k)))
}

/// Checks (and coerces) an owned value for storage as an element of type
/// `elem` in a container of type `container`. On error `v` is untouched.
pub fn admit(th: &mut Thread, v: Value, elem: &RtType, container: &RtType) -> VResult<Value> {
    rtype::admit(v, elem).ok_or_else(|| elem_error(th, v, &container.name))
}

/// Checks (and coerces) owned items as elements of type `elem`. On error
/// every item is released.
pub fn admit_all(th: &mut Thread, items: Vec<Value>, elem: &RtType, container: &RtType) -> VResult<Vec<Value>> {
    let mut out = Vec::with_capacity(items.len());
    let mut it = items.into_iter();
    while let Some(v) = it.next() {
        match rtype::admit(v, elem) {
            Some(v) => out.push(v),
            None => {
                let e = elem_error(th, v, &container.name);
                v.decref();
                out.into_iter().chain(it).for_each(|x| x.decref());
                return Err(e);
            }
        }
    }
    Ok(out)
}

/// A new list of type `ty` from owned items, each checked against the
/// element type. On error every item is released.
pub fn new_list(th: &mut Thread, ty: Value, items: Vec<Value>) -> VResult<Value> {
    let t = rtype::as_type(ty).unwrap();
    let items = admit_all(th, items, rtype::as_type(rtype::elem_of(t)).unwrap(), t)?;
    Ok(crate::object::Obj::list(ty, items))
}

/// The key and value types of a dict type.
pub fn dict_kv(t: &RtType) -> (&'static RtType, &'static RtType) {
    if t.args.len() == 2 {
        (t.arg(0), t.arg(1))
    } else {
        let a = rtype::as_type(rtype::any()).unwrap();
        (a, a)
    }
}
