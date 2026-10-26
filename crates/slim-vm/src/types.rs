//! Resolving written types to runtime type values, and type errors.

use crate::code::{Code, TypeDesc};
use crate::interp::{Thread, VResult};
use crate::object::{Body, Field};
use crate::rtype::{self, Ctor, RtType};
use crate::value::Value;

/// The runtime type for `d`, with type parameters taken from `env`. Fails
/// with the name of a class or trait that is not defined yet, or of a type
/// parameter that has no value.
pub fn materialize(th: &Thread, d: &TypeDesc, env: &dyn Fn(u16) -> Value) -> Result<Value, String> {
    match d {
        TypeDesc::Builtin(c, args) => {
            let args = args.iter().map(|a| materialize(th, a, env)).collect::<Result<Vec<_>, _>>()?;
            Ok(rtype::intern(*c, &args))
        }
        TypeDesc::Global(idx, n, args) => {
            let v = th.vm.global(*idx);
            let args = args.iter().map(|a| materialize(th, a, env)).collect::<Result<Vec<_>, _>>()?;
            match v.body() {
                Some(Body::Class(_)) => Ok(rtype::intern(Ctor::Class(v), &args)),
                Some(Body::Trait(_)) => Ok(rtype::intern(Ctor::Trait(v), &args)),
                _ => Err(n.clone()),
            }
        }
        TypeDesc::Param(k, n) => match env(*k) {
            Value::UNINIT => Err(n.clone()),
            v => Ok(v),
        },
        TypeDesc::Resolved(v) => Ok(*v),
        TypeDesc::Local(t) => Ok(rtype::local_of(rtype::as_type(materialize(th, t, env)?).unwrap())),
    }
}

fn no_env(_: u16) -> Value {
    Value::UNINIT
}

fn undefined(th: &mut Thread, n: &str) -> crate::interp::Raised {
    th.raise("NameError", format!("type `{n}` is used before it is defined"))
}

/// The runtime type of `code.types[idx]`, which mentions no type parameters
/// (cached in the code object).
pub fn code_type(th: &mut Thread, code: &Code, idx: u32) -> VResult<&'static RtType> {
    let slot = &code.type_cache[idx as usize];
    if let Some(v) = slot.get() {
        return Ok(rtype::as_type(*v).unwrap());
    }
    match materialize(th, &code.types[idx as usize], &no_env) {
        Ok(v) => Ok(rtype::as_type(*slot.get_or_init(|| v)).unwrap()),
        Err(n) => Err(undefined(th, &n)),
    }
}

/// The runtime type of `code.types[idx]` in a frame: type parameters are
/// read from the frame's type-argument registers.
pub fn frame_type(th: &mut Thread, frame: &crate::interp::Frame, idx: u32) -> VResult<&'static RtType> {
    let code = frame.code();
    let d = &code.types[idx as usize];
    if !d.has_params() {
        return code_type(th, code, idx);
    }
    match materialize(th, d, &|r| frame.reg(r)) {
        Ok(v) => Ok(rtype::as_type(v).unwrap()),
        Err(n) => Err(undefined(th, &n)),
    }
}

/// The runtime type of `code.types[idx]` with type parameters read from
/// `regs` (a frame that is not running yet).
pub fn regs_type(th: &mut Thread, code: &Code, idx: u32, regs: &[Value]) -> VResult<&'static RtType> {
    let d = &code.types[idx as usize];
    if !d.has_params() {
        return code_type(th, code, idx);
    }
    match materialize(th, d, &|r| regs[r as usize]) {
        Ok(v) => Ok(rtype::as_type(v).unwrap()),
        Err(n) => Err(undefined(th, &n)),
    }
}

/// The runtime type of field `idx` of an instance (cached in the field
/// unless it depends on the instance's type arguments).
pub fn field_type(th: &mut Thread, inst: &crate::object::Instance, idx: usize) -> VResult<&'static RtType> {
    let f: &Field = &inst.class().fields[idx];
    if let Some(v) = f.rt.get() {
        return Ok(rtype::as_type(*v).unwrap());
    }
    if f.ty.has_params() {
        let args = &rtype::as_type(inst.ty).unwrap().args;
        return match materialize(th, &f.ty, &|k| args.get(k as usize).copied().unwrap_or(Value::UNINIT)) {
            Ok(v) => Ok(rtype::as_type(v).unwrap()),
            Err(n) => Err(undefined(th, &n)),
        };
    }
    match materialize(th, &f.ty, &no_env) {
        Ok(v) => Ok(rtype::as_type(*f.rt.get_or_init(|| v)).unwrap()),
        Err(n) => Err(undefined(th, &n)),
    }
}

/// Resolves the classes and traits named in `d` now; type parameters stay.
pub fn pre_resolve(th: &Thread, d: &TypeDesc) -> Result<TypeDesc, String> {
    Ok(match d {
        TypeDesc::Builtin(c, args) => {
            TypeDesc::Builtin(*c, args.iter().map(|a| pre_resolve(th, a)).collect::<Result<_, _>>()?)
        }
        TypeDesc::Global(idx, n, args) => {
            let args = args.iter().map(|a| pre_resolve(th, a)).collect::<Result<_, _>>()?;
            let v = th.vm.global(*idx);
            let c = match v.body() {
                Some(Body::Class(_)) => Ctor::Class(v),
                Some(Body::Trait(_)) => Ctor::Trait(v),
                _ => return Err(n.clone()),
            };
            TypeDesc::Builtin(c, args)
        }
        TypeDesc::Local(t) => TypeDesc::Local(Box::new(pre_resolve(th, t)?)),
        other => other.clone(),
    })
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

/// `TypeError` for storing `v` as an element of type `elem` in a container
/// of type `container`.
pub fn elem_error(th: &mut Thread, v: Value, elem: &RtType, container: &str) -> crate::interp::Raised {
    if rtype::has_type(v, elem) && !rtype::local_ok(v, elem) {
        return th.raise("TypeError", format!("cannot store local `{}` in non-local `{container}`", value_type_name(v)));
    }
    th.raise("TypeError", format!("cannot store `{}` in `{container}`", value_type_name(v)))
}

/// `TypeError` for using `k` as a key of a dict of type `container`.
pub fn key_type_error(th: &mut Thread, k: Value, container: &str) -> crate::interp::Raised {
    th.raise("TypeError", format!("cannot use `{}` as a key of `{container}`", value_type_name(k)))
}

/// Checks (and coerces) an owned value for storage as an element of type
/// `elem` in a container of type `container`. On error `v` is untouched.
pub fn admit(th: &mut Thread, v: Value, elem: &RtType, container: &RtType) -> VResult<Value> {
    rtype::admit(v, elem).ok_or_else(|| elem_error(th, v, elem, &container.name))
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
                let e = elem_error(th, v, elem, &container.name);
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
