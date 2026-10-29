//! The interpreter's side of the extension ABI (`lizzard-abi`): the function
//! table handed to native modules, statically embedded or loaded from a
//! dynamic library.
//!
//! Every function trusts what the ABI promises (values are live, handles
//! were checked by the extension) but stays memory safe on values of the
//! wrong kind: a list function given another value acts on an empty list.
//! None may panic, since a panic cannot unwind into the extension.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

use lizzard_abi::{write_str, ClassDesc, Ctx, FnDesc, HostErr, MethodSig, RawValue, Scalar, LizzardApi, Str, StrOut, ValueLayout, ABI_VERSION};

use crate::array::Scalar as Num;
use crate::interp::Thread;
use crate::native::{self, decode_sig, slice, ClassParts, FnSpec, Module, NType, NativeFn, LizzardError};
use crate::object::{Body, Obj};
use crate::rtype::{self, Ctor};
use crate::value::Value;

/// The function table given to extensions.
pub fn api() -> &'static LizzardApi {
    static API: OnceLock<LizzardApi> = OnceLock::new();
    API.get_or_init(table)
}

/// How this build encodes scalars (see `ValueLayout`). The payload offsets
/// depend on the compiler's layout of `Body`, so they are measured on probe
/// objects.
fn layout() -> ValueLayout {
    fn offset(v: Value) -> usize {
        let o = v.obj().expect("probe object");
        let p = match &o.body {
            Body::Int(i) => i as *const i64 as usize,
            Body::Float(f) => f as *const f64 as usize,
            _ => unreachable!(),
        };
        p - o as *const Obj as usize
    }
    ValueLayout {
        small_int_mask: 1,
        small_int_tag: 1,
        small_int_shift: 1,
        small_int_min: crate::value::SMALL_INT_MIN,
        small_int_max: crate::value::SMALL_INT_MAX,
        ptr_mask: 7,
        kind_mask: crate::object::KIND_MASK,
        kind_int: crate::object::KIND_INT,
        kind_float: crate::object::KIND_FLOAT,
        int_offset: offset(Obj::alloc_immortal(Body::Int(i64::MAX))),
        float_offset: offset(Obj::alloc_immortal(Body::Float(0.0))),
    }
}

/// Takes a module built through [`api`] (`module_new`).
///
/// # Safety
/// `m` must come from `module_new` and not be used again.
pub unsafe fn module_from_raw(m: *mut c_void) -> Module {
    // SAFETY: guaranteed by the caller.
    *unsafe { Box::from_raw(m as *mut Module) }
}

fn herr(e: LizzardError) -> HostErr {
    Box::into_raw(Box::new(e)) as HostErr
}

/// # Safety
/// `ctx` must be the thread of the running native call.
unsafe fn thread<'a>(ctx: Ctx) -> &'a mut Thread {
    // SAFETY: guaranteed by the caller.
    unsafe { &mut *(ctx as *mut Thread) }
}

fn list_obj(v: RawValue) -> Option<&'static crate::list::ListObj> {
    match Value(v).body() {
        Some(Body::List(l)) => Some(l),
        _ => None,
    }
}

fn dict_obj(v: RawValue) -> Option<&'static crate::dict::Dict> {
    match Value(v).body() {
        Some(Body::Dict(d)) => Some(d),
        _ => None,
    }
}

/// Writes borrowed values to `out` (up to `cap`); returns their number.
///
/// # Safety
/// `out` must have room for `cap` values.
unsafe fn write_values(vals: impl ExactSizeIterator<Item = Value>, out: *mut RawValue, cap: usize) -> usize {
    let n = vals.len();
    for (i, v) in vals.take(cap).enumerate() {
        // SAFETY: `i < cap`.
        unsafe { *out.add(i) = v.0 }
    }
    n
}

/// Runs a module-building function, reporting a panic (an interpreter bug)
/// as an error instead of unwinding into the extension.
fn guarded(err: StrOut, f: impl FnOnce() -> Result<(), String>) -> bool {
    let r = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
        Err(format!("interpreter panic: {msg}"))
    });
    match r {
        Ok(()) => true,
        Err(e) => {
            // SAFETY: `err` is a valid sink, by the ABI.
            unsafe { write_str(err, &e) };
            false
        }
    }
}

// ---------------------------------------------------------------- values

unsafe extern "C" fn incref(v: RawValue) {
    Value(v).incref()
}

unsafe extern "C" fn decref(v: RawValue) {
    Value(v).decref()
}

unsafe extern "C" fn is_dead(v: RawValue) -> bool {
    Value(v).is_dead()
}

unsafe extern "C" fn int_new(i: i64) -> RawValue {
    Value::from_i64(i).0
}

unsafe extern "C" fn float_new(f: f64) -> RawValue {
    Value::float(f).0
}

unsafe extern "C" fn bool_new(b: bool) -> RawValue {
    Value::bool(b).0
}

unsafe extern "C" fn str_new(s: Str) -> RawValue {
    // SAFETY: valid for the call, by the ABI.
    Value::str(unsafe { s.as_str() }).0
}

unsafe extern "C" fn as_int(v: RawValue, out: *mut i64) -> bool {
    Value(v).as_int().map(|i| unsafe { *out = i }).is_some()
}

unsafe extern "C" fn as_float(v: RawValue, out: *mut f64) -> bool {
    Value(v).as_float().map(|f| unsafe { *out = f }).is_some()
}

unsafe extern "C" fn as_bool(v: RawValue, out: *mut bool) -> bool {
    Value(v).as_bool().map(|b| unsafe { *out = b }).is_some()
}

unsafe extern "C" fn as_str(v: RawValue, out: *mut Str) -> bool {
    Value(v).as_str().map(|s| unsafe { *out = Str::new(s) }).is_some()
}

unsafe extern "C" fn is_function(v: RawValue) -> bool {
    rtype::has_shape(Value(v), rtype::as_type(rtype::prim(Ctor::Function)).unwrap())
}

unsafe extern "C" fn type_name(v: RawValue, out: StrOut) {
    unsafe { write_str(out, &crate::types::value_type_name(Value(v))) }
}

unsafe extern "C" fn display(v: RawValue, out: StrOut) {
    unsafe { write_str(out, &crate::display::to_str(Value(v))) }
}

// ---------------------------------------------------------------- types

unsafe extern "C" fn type_from_desc(desc: Str) -> RawValue {
    let mut b = unsafe { desc.as_bytes() };
    match NType::decode(&mut b) {
        Some(t) if b.is_empty() => t.rt().0,
        _ => rtype::any().0,
    }
}

unsafe extern "C" fn type_display(ty: RawValue, out: StrOut) {
    unsafe { write_str(out, rtype::name(Value(ty))) }
}

unsafe extern "C" fn has_type(v: RawValue, ty: RawValue) -> bool {
    rtype::container_type(Value(v)).is_some_and(|ct| ct.value() == Value(ty))
}

// ---------------------------------------------------------------- containers

unsafe extern "C" fn list_new(ty: RawValue, items: *const RawValue, n: usize, out: *mut RawValue) -> HostErr {
    let items: Vec<Value> = unsafe { slice(items as *const Value, n) }.to_vec();
    let ty = Value(ty);
    let local_ok = rtype::as_type(ty).is_some_and(|t| t.local);
    if items.iter().any(|v| v.is_local()) && !local_ok {
        items.into_iter().for_each(|v| v.decref());
        return herr(LizzardError::type_error("cannot put values that are not `Send` in a `Send` list"));
    }
    unsafe { *out = Obj::list(ty, items).0 };
    std::ptr::null_mut()
}

unsafe extern "C" fn list_len(v: RawValue) -> usize {
    list_obj(v).map_or(0, |l| l.len())
}

unsafe extern "C" fn list_get(v: RawValue, i: usize, out: *mut RawValue) -> bool {
    list_obj(v).and_then(|l| l.get(i)).map(|x| unsafe { *out = x.0 }).is_some()
}

unsafe extern "C" fn list_snapshot(v: RawValue, out: *mut RawValue, cap: usize) -> usize {
    let items = list_obj(v).map(|l| l.snapshot()).unwrap_or_default();
    unsafe { write_values(items.into_iter(), out, cap) }
}

/// `x` checked like a Lizzard store into list `l` (consumed on error).
fn admit_elem(l: &crate::list::ListObj, x: Value) -> Result<Value, LizzardError> {
    let lt = rtype::as_type(l.ty).unwrap();
    let et = rtype::as_type(rtype::elem_of(lt)).unwrap();
    rtype::admit(x, et).ok_or_else(|| {
        let e = LizzardError::type_error(format!("cannot store `{}` in `{}`", crate::types::value_type_name(x), lt.name));
        x.decref();
        e
    })
}

unsafe extern "C" fn list_push(v: RawValue, item: RawValue) -> HostErr {
    let Some(l) = list_obj(v) else {
        Value(item).decref();
        return herr(LizzardError::type_error("not a list"));
    };
    match admit_elem(l, Value(item)) {
        Ok(x) => {
            l.push(x);
            std::ptr::null_mut()
        }
        Err(e) => herr(e),
    }
}

unsafe extern "C" fn list_set(v: RawValue, i: usize, item: RawValue) -> HostErr {
    let Some(l) = list_obj(v) else {
        Value(item).decref();
        return herr(LizzardError::type_error("not a list"));
    };
    let x = match admit_elem(l, Value(item)) {
        Ok(x) => x,
        Err(e) => return herr(e),
    };
    match l.write(|w| (i < w.len()).then(|| w.set(i, x))) {
        Some(old) => {
            crate::rc::defer_decref(old);
            std::ptr::null_mut()
        }
        None => {
            x.decref();
            herr(LizzardError::new("IndexError", format!("list index {i} out of range")))
        }
    }
}

unsafe extern "C" fn dict_len(d: RawValue) -> usize {
    dict_obj(d).map_or(0, |d| d.len())
}

unsafe extern "C" fn dict_get(d: RawValue, k: RawValue, out: *mut RawValue) -> bool {
    dict_obj(d).and_then(|d| d.get(Value(k))).map(|v| unsafe { *out = v.0 }).is_some()
}

unsafe extern "C" fn dict_insert(d: RawValue, k: RawValue, v: RawValue) -> HostErr {
    let (kv, vv) = (Value(k), Value(v));
    let Some(d) = dict_obj(d) else {
        kv.decref();
        vv.decref();
        return herr(LizzardError::type_error("not a dict"));
    };
    let dt = rtype::as_type(d.ty).unwrap();
    let (kt, vt) = crate::types::dict_kv(dt);
    let (Some(kv), Some(vv)) = (rtype::admit(kv, kt), rtype::admit(vv, vt)) else {
        kv.decref();
        vv.decref();
        return herr(LizzardError::type_error(format!("cannot store this entry in `{}`", dt.name)));
    };
    match d.insert(kv, vv) {
        Ok(Some((dup, old))) => {
            dup.decref();
            crate::rc::defer_decref(old);
            std::ptr::null_mut()
        }
        Ok(None) => std::ptr::null_mut(),
        Err(()) => herr(LizzardError::type_error("unhashable key")),
    }
}

unsafe extern "C" fn dict_keys(d: RawValue, out: *mut RawValue, cap: usize) -> usize {
    let keys: Vec<Value> = dict_obj(d).map(|d| d.entries().into_iter().map(|(k, _)| k).collect()).unwrap_or_default();
    unsafe { write_values(keys.into_iter(), out, cap) }
}

unsafe extern "C" fn array_elem(v: RawValue, out: *mut u8) -> bool {
    Value(v).as_array().map(|a| unsafe { *out = native::elem_code(a.elem) }).is_some()
}

unsafe extern "C" fn array_len(v: RawValue) -> usize {
    Value(v).as_array().map_or(0, |a| a.len())
}

unsafe extern "C" fn array_get(v: RawValue, i: usize, out: *mut Scalar) -> bool {
    let Some(n) = Value(v).as_array().and_then(|a| a.get_num(i)) else { return false };
    let s = match n {
        Num::Int(i) => Scalar { tag: lizzard_abi::SCALAR_INT, bits: i as u64, float: 0.0 },
        Num::Float(f) => Scalar { tag: lizzard_abi::SCALAR_FLOAT, bits: 0, float: f },
        Num::Bool(b) => Scalar { tag: lizzard_abi::SCALAR_BOOL, bits: b as u64, float: 0.0 },
    };
    unsafe { *out = s };
    true
}

unsafe extern "C" fn array_set(v: RawValue, i: usize, x: RawValue) -> HostErr {
    let Some(a) = Value(v).as_array() else { return herr(LizzardError::type_error("not an array")) };
    if i >= a.len() {
        return herr(LizzardError::new("IndexError", format!("array index {i} out of range")));
    }
    match a.set(i, Value(x)) {
        Ok(()) => std::ptr::null_mut(),
        Err(_) => herr(LizzardError::new("OverflowError", format!("value out of range for `{}`", a.elem.name()))),
    }
}

// ---------------------------------------------------------------- errors and calls

unsafe extern "C" fn err_read(e: HostErr, kind: StrOut, msg: StrOut) -> bool {
    let e = unsafe { &*(e as *const LizzardError) };
    unsafe {
        write_str(kind, &e.kind);
        write_str(msg, &e.msg);
    }
    e.is_pending()
}

unsafe extern "C" fn err_free(e: HostErr) {
    drop(unsafe { Box::from_raw(e as *mut LizzardError) })
}

unsafe extern "C" fn err_raise(ctx: Ctx, e: HostErr) {
    let e = unsafe { Box::from_raw(e as *mut LizzardError) };
    e.raise(unsafe { thread(ctx) });
}

unsafe extern "C" fn raise(ctx: Ctx, kind: Str, msg: Str) {
    let e = LizzardError::new(unsafe { kind.as_str() }, unsafe { msg.as_str() });
    e.raise(unsafe { thread(ctx) });
}

unsafe extern "C" fn call(ctx: Ctx, f: RawValue, args: *const RawValue, n: usize, out: *mut RawValue) -> HostErr {
    let th = unsafe { thread(ctx) };
    let args = unsafe { slice(args as *const Value, n) };
    match crate::helpers::call_simple(th, Value(f), None, args) {
        Ok(v) => {
            unsafe { *out = v.0 };
            std::ptr::null_mut()
        }
        Err(_) => herr(LizzardError::from_thread(th)),
    }
}

unsafe extern "C" fn offline() {
    crate::rc::offline()
}

unsafe extern "C" fn online() {
    crate::rc::online()
}

// ---------------------------------------------------------------- native objects

unsafe extern "C" fn class_key_new() -> u64 {
    native::new_key()
}

unsafe extern "C" fn obj_new(key: u64, data: *mut c_void, out: *mut RawValue) -> bool {
    native::new_object(key, data).map(|v| unsafe { *out = v.0 }).is_some()
}

unsafe extern "C" fn obj_key(v: RawValue) -> u64 {
    native::native_obj(Value(v)).map_or(0, |o| o.class().key)
}

unsafe extern "C" fn obj_enter(v: RawValue, exclusive: bool, data: *mut *mut c_void, exclusive_out: *mut bool) -> HostErr {
    let Some(o) = native::native_obj(Value(v)) else { return herr(LizzardError::type_error("not a native object")) };
    match o.enter(exclusive) {
        Ok((d, excl)) => {
            unsafe {
                *data = d;
                *exclusive_out = excl;
            }
            std::ptr::null_mut()
        }
        Err(e) => herr(e),
    }
}

unsafe extern "C" fn obj_leave(v: RawValue, exclusive: bool) {
    if let Some(o) = native::native_obj(Value(v)) {
        o.leave(exclusive)
    }
}

// ---------------------------------------------------------------- modules

unsafe extern "C" fn module_new(name: Str) -> *mut c_void {
    Box::into_raw(Box::new(Module::new(unsafe { name.as_str() }))) as *mut c_void
}

unsafe extern "C" fn module_free(m: *mut c_void) {
    drop(unsafe { Box::from_raw(m as *mut Module) })
}

unsafe fn module<'a>(m: *mut c_void) -> &'a mut Module {
    unsafe { &mut *(m as *mut Module) }
}

unsafe extern "C" fn module_function(m: *mut c_void, f: *const FnDesc, err: StrOut) -> bool {
    guarded(err, || {
        let f = unsafe { NativeFn::from_desc(&*f) }?;
        unsafe { module(m) }.add_function(f);
        Ok(())
    })
}

unsafe extern "C" fn module_trait(m: *mut c_void, name: Str, methods: *const MethodSig, n: usize, err: StrOut) -> bool {
    guarded(err, || {
        let name = unsafe { name.as_str() };
        let mut specs = Vec::new();
        for ms in unsafe { slice(methods, n) } {
            let mname = unsafe { ms.name.as_str() };
            let (params, ret) = decode_sig(unsafe { ms.sig.as_bytes() }).ok_or_else(|| format!("trait `{name}`: malformed signature of `{mname}`"))?;
            specs.push(FnSpec { name: mname.into(), params, ret });
        }
        unsafe { module(m) }.add_trait(name, specs)
    })
}

unsafe extern "C" fn module_class(m: *mut c_void, c: *const ClassDesc, err: StrOut) -> bool {
    guarded(err, || {
        let parts = unsafe { ClassParts::from_desc(&*c) }?;
        unsafe { module(m) }.add_class(parts)
    })
}

unsafe extern "C" fn module_generic(m: *mut c_void, name: Str, tparams: *const Str, ntparams: usize, classes: *const ClassDesc, n: usize, err: StrOut) -> bool {
    guarded(err, || {
        // Take every class first: their data is owned from here on.
        let parts: Vec<Result<ClassParts, String>> = unsafe { slice(classes, n) }.iter().map(|c| unsafe { ClassParts::from_desc(c) }).collect();
        let parts = parts.into_iter().collect::<Result<Vec<_>, _>>()?;
        let tparams = unsafe { slice(tparams, ntparams) }.iter().map(|t| unsafe { t.as_str() }.to_string()).collect();
        unsafe { module(m) }.add_generic(unsafe { name.as_str() }, tparams, parts)
    })
}

fn table() -> LizzardApi {
    LizzardApi {
        version: ABI_VERSION,
        size: std::mem::size_of::<LizzardApi>(),
        none: Value::NONE.0,
        unit: Value::UNIT.0,
        false_: Value::FALSE.0,
        true_: Value::TRUE.0,
        layout: layout(),
        incref,
        decref,
        is_dead,
        int_new,
        float_new,
        bool_new,
        str_new,
        as_int,
        as_float,
        as_bool,
        as_str,
        is_function,
        type_name,
        display,
        type_from_desc,
        type_display,
        has_type,
        list_new,
        list_len,
        list_get,
        list_snapshot,
        list_push,
        list_set,
        dict_len,
        dict_get,
        dict_insert,
        dict_keys,
        array_elem,
        array_len,
        array_get,
        array_set,
        err_read,
        err_free,
        err_raise,
        raise,
        call,
        offline,
        online,
        class_key_new,
        obj_new,
        obj_key,
        obj_enter,
        obj_leave,
        module_new,
        module_free,
        module_function,
        module_trait,
        module_class,
        module_generic,
    }
}
