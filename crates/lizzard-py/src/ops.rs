//! The hot instructions on raw register slots, shared by the interpreter
//! and compiled code: operands are register numbers and the objects the
//! function keeps (interned names, constants), and the work goes straight to
//! CPython's C API, without PyO3 wrappers. Each returns `NEXT` (`TAKEN` for
//! a jump), or `ERROR` with an exception set.

use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use lizzard_jit::{ERROR, NEXT, TAKEN};
use pyo3::ffi;


/// The object in slot `r` (borrowed), or null with UnboundLocalError set.
pub unsafe fn slot(slots: *mut u64, r: u32) -> *mut ffi::PyObject {
    let p = unsafe { *slots.add(r as usize) } as *mut ffi::PyObject;
    if p.is_null() {
        unsafe { ffi::PyErr_SetString(ffi::PyExc_UnboundLocalError, c"local variable referenced before assignment".as_ptr()) };
    }
    p
}

/// Store a new reference in slot `r`, releasing the old one.
pub unsafe fn put(slots: *mut u64, r: u32, v: *mut ffi::PyObject) {
    let s = unsafe { slots.add(r as usize) };
    let old = unsafe { std::mem::replace(&mut *s, v as u64) } as *mut ffi::PyObject;
    if !old.is_null() {
        unsafe { ffi::Py_DecRef(old) };
    }
}

/// Store a result: a new reference, or null when an exception is set.
unsafe fn result(slots: *mut u64, r: u32, v: *mut ffi::PyObject) -> u32 {
    if v.is_null() {
        return ERROR;
    }
    unsafe { put(slots, r, v) };
    NEXT
}

/// An attribute site's inline cache: the slot (`__slots__` member) an
/// attribute was found in, for one type at one version. A type's version
/// tag changes whenever it or a base class is modified, so a hit needs no
/// lookup: the attribute is read or written at its offset. Instances
/// without `__slots__` keep their attributes in CPython's internal storage,
/// which only the generic path reaches.
#[derive(Default)]
pub struct AttrIc {
    entry: AtomicPtr<IcEntry>,
    misses: AtomicU32,
}

struct IcEntry {
    tp: *mut ffi::PyTypeObject,
    version: u32,
    /// Null: the attribute is not a slot (the generic path).
    member: *mut ffi::PyMemberDef,
}

/// A site that misses this often stops looking (it sees many types).
const IC_MAX_MISSES: u32 = 8;

unsafe extern "C" {
    /// CPython 3.14+: the next item (1), the end (0), or an error (-1).
    fn PyIter_NextItem(iter: *mut ffi::PyObject, item: *mut *mut ffi::PyObject) -> std::ffi::c_int;
    /// CPython 3.12+: give a type a version tag (0 if it cannot have one).
    fn PyUnstable_Type_AssignVersionTag(tp: *mut ffi::PyTypeObject) -> std::ffi::c_int;
}

/// The `__slots__` member `name` of `o`'s type, from the cache or found and
/// cached; None to take the generic path.
unsafe fn member(ic: &AttrIc, o: *mut ffi::PyObject, name: *mut ffi::PyObject) -> Option<*mut ffi::PyMemberDef> {
    unsafe {
        let tp = ffi::Py_TYPE(o);
        let e = ic.entry.load(Ordering::Acquire);
        if !e.is_null() && (*e).tp == tp && (*e).version == (*tp).tp_version_tag {
            let m = (*e).member;
            return (!m.is_null()).then_some(m);
        }
        // A plain load first: a site that gave up does no atomic write.
        if ic.misses.load(Ordering::Relaxed) >= IC_MAX_MISSES {
            return None;
        }
        ic.misses.fetch_add(1, Ordering::Relaxed);
        if PyUnstable_Type_AssignVersionTag(tp) == 0 {
            return None;
        }
        let version = (*tp).tp_version_tag;
        let mut d = std::ptr::null_mut();
        // On the type, a member descriptor is the descriptor itself.
        let found = match ffi::PyObject_GetOptionalAttr(tp as *mut ffi::PyObject, name, &mut d) {
            r if r < 0 => {
                ffi::PyErr_Clear();
                return None;
            }
            0 => None,
            _ => {
                let m = (ffi::Py_TYPE(d) == &raw mut ffi::PyMemberDescr_Type).then(|| (*(d as *mut ffi::PyMemberDescrObject)).d_member);
                ffi::Py_DecRef(d);
                m
            }
        };
        // Not a slot is cached too, so that a monomorphic site of a class
        // without `__slots__` goes straight to the generic path.
        let m = found.unwrap_or(std::ptr::null_mut());
        // Entries are small and a site replaces its entry at most
        // `IC_MAX_MISSES` times, so replaced ones are not freed (another
        // thread may still be reading one).
        let new = Box::into_raw(Box::new(IcEntry { tp, version, member: m }));
        ic.entry.store(new, Ordering::Release);
        (!m.is_null()).then_some(m)
    }
}

pub extern "C" fn lz_load_attr(slots: *mut u64, dst: u32, obj: u32, name: *mut ffi::PyObject, ic: *const AttrIc) -> u32 {
    unsafe {
        let o = slot(slots, obj);
        if o.is_null() {
            return ERROR;
        }
        let v = match member(&*ic, o, name) {
            Some(m) => ffi::PyMember_GetOne(o as *const std::ffi::c_char, m),
            None => ffi::PyObject_GetAttr(o, name),
        };
        result(slots, dst, v)
    }
}

pub extern "C" fn lz_store_attr(slots: *mut u64, obj: u32, val: u32, name: *mut ffi::PyObject, ic: *const AttrIc) -> u32 {
    unsafe {
        let (o, v) = (slot(slots, obj), slot(slots, val));
        if o.is_null() || v.is_null() {
            return ERROR;
        }
        let r = match member(&*ic, o, name) {
            Some(m) => ffi::PyMember_SetOne(o as *mut std::ffi::c_char, m, v),
            None => ffi::PyObject_SetAttr(o, name, v),
        };
        if r < 0 {
            return ERROR;
        }
        NEXT
    }
}

/// `dst = const` (a new reference to an object the function keeps).
pub extern "C" fn lz_const(slots: *mut u64, dst: u32, obj: *mut ffi::PyObject) {
    unsafe {
        ffi::Py_IncRef(obj);
        put(slots, dst, obj);
    }
}

pub extern "C" fn lz_move(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        ffi::Py_IncRef(o);
        put(slots, dst, o);
        NEXT
    }
}

/// `dst = int(src)` from an `I64` slot.
pub extern "C" fn lz_box_i64(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let v = *slots.add(src as usize) as i64;
        result(slots, dst, ffi::PyLong_FromLongLong(v))
    }
}

/// `dst: Bool = bool(src)`
pub extern "C" fn lz_truthy(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        let t = if o == ffi::Py_True() {
            1
        } else if o == ffi::Py_False() || o == ffi::Py_None() {
            0
        } else {
            ffi::PyObject_IsTrue(o)
        };
        if t < 0 {
            return ERROR;
        }
        *slots.add(dst as usize) = t as u64;
        NEXT
    }
}

// ---- Exact int and float fast paths ----
//
// CPython's specializing interpreter computes `int + int`, `float * float`
// and comparisons of them in place; so do these helpers. A compact int
// (one 30-bit digit and a sign, `cpython/longintrepr.h`, what
// `PyUnstable_Long_CompactValue` reads) and a float's double are read
// directly; anything else (big ints, `bool`, subclasses, mixed types) takes
// the generic path. `init_fast_paths` checks the layout once at startup.

static FAST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const HEAD: usize = std::mem::size_of::<ffi::PyObject>();

/// The value of an exact, compact int.
#[inline]
unsafe fn small_int(o: *mut ffi::PyObject) -> Option<i64> {
    unsafe {
        if ffi::Py_TYPE(o) != &raw mut ffi::PyLong_Type {
            return None;
        }
        let tag = *((o as *const u8).add(HEAD) as *const usize);
        if tag >= (2 << 3) {
            return None;
        }
        let digit = *((o as *const u8).add(HEAD + 8) as *const u32) as i64;
        Some((1 - (tag & 3) as i64) * digit)
    }
}

/// The value of an exact float.
#[inline]
unsafe fn exact_float(o: *mut ffi::PyObject) -> Option<f64> {
    unsafe { (ffi::Py_TYPE(o) == &raw mut ffi::PyFloat_Type).then(|| *((o as *const u8).add(HEAD) as *const f64)) }
}

/// Check the layout these fast paths read against CPython's own answers,
/// and turn them on if it matches.
pub fn init_fast_paths() {
    unsafe {
        let mut ok = true;
        for v in [0i64, 1, -1, 7, -1234, (1 << 30) - 1, -((1 << 30) - 1)] {
            let o = ffi::PyLong_FromLongLong(v);
            ok &= small_int(o) == Some(v);
            ffi::Py_DecRef(o);
        }
        let big = ffi::PyLong_FromLongLong(1 << 40);
        ok &= small_int(big).is_none();
        ffi::Py_DecRef(big);
        let f = ffi::PyFloat_FromDouble(2.5);
        ok &= exact_float(f) == Some(2.5);
        ffi::Py_DecRef(f);
        FAST.store(ok, Ordering::Relaxed);
    }
}

/// `x op y` for two exact ints or floats, or None for the generic path.
#[inline]
unsafe fn fast_binop(x: *mut ffi::PyObject, y: *mut ffi::PyObject, op: u32) -> Option<*mut ffi::PyObject> {
    unsafe {
        if !FAST.load(Ordering::Relaxed) {
            return None;
        }
        if let (Some(a), Some(b)) = (small_int(x), small_int(y)) {
            // Compact ints are below 2**30: these cannot overflow i64.
            let v = match op & 15 {
                0 => a + b,
                1 => a - b,
                2 => a * b,
                _ => return None,
            };
            return Some(ffi::PyLong_FromLongLong(v));
        }
        if let (Some(a), Some(b)) = (exact_float(x), exact_float(y)) {
            let v = match op & 15 {
                0 => a + b,
                1 => a - b,
                2 => a * b,
                3 if b != 0.0 => a / b,
                _ => return None,
            };
            return Some(ffi::PyFloat_FromDouble(v));
        }
        None
    }
}

/// `x op y` (CPython's `Py_LT`... code) for two exact ints or floats.
#[inline]
unsafe fn fast_compare(x: *mut ffi::PyObject, y: *mut ffi::PyObject, op: u32) -> Option<bool> {
    unsafe {
        if !FAST.load(Ordering::Relaxed) {
            return None;
        }
        fn cmp<T: PartialOrd>(a: T, b: T, op: u32) -> bool {
            match op {
                0 => a < b,
                1 => a <= b,
                2 => a == b,
                3 => a != b,
                4 => a > b,
                _ => a >= b,
            }
        }
        if let (Some(a), Some(b)) = (small_int(x), small_int(y)) {
            return Some(cmp(a, b, op));
        }
        if let (Some(a), Some(b)) = (exact_float(x), exact_float(y)) {
            return Some(cmp(a, b, op));
        }
        None
    }
}

/// `a op b`; `op` is the `BinOp` code, plus 16 for the in-place form.
pub extern "C" fn lz_binop(slots: *mut u64, dst: u32, a: u32, b: u32, op: u32) -> u32 {
    unsafe {
        let (x, y) = (slot(slots, a), slot(slots, b));
        if x.is_null() || y.is_null() {
            return ERROR;
        }
        if let Some(v) = fast_binop(x, y, op) {
            return result(slots, dst, v);
        }
        let v = match op {
            0 => ffi::PyNumber_Add(x, y),
            1 => ffi::PyNumber_Subtract(x, y),
            2 => ffi::PyNumber_Multiply(x, y),
            3 => ffi::PyNumber_TrueDivide(x, y),
            4 => ffi::PyNumber_FloorDivide(x, y),
            5 => ffi::PyNumber_Remainder(x, y),
            6 => ffi::PyNumber_Power(x, y, ffi::Py_None()),
            7 => ffi::PyNumber_MatrixMultiply(x, y),
            8 => ffi::PyNumber_Or(x, y),
            9 => ffi::PyNumber_And(x, y),
            10 => ffi::PyNumber_Xor(x, y),
            11 => ffi::PyNumber_Lshift(x, y),
            12 => ffi::PyNumber_Rshift(x, y),
            16 => ffi::PyNumber_InPlaceAdd(x, y),
            17 => ffi::PyNumber_InPlaceSubtract(x, y),
            18 => ffi::PyNumber_InPlaceMultiply(x, y),
            19 => ffi::PyNumber_InPlaceTrueDivide(x, y),
            20 => ffi::PyNumber_InPlaceFloorDivide(x, y),
            21 => ffi::PyNumber_InPlaceRemainder(x, y),
            22 => ffi::PyNumber_InPlacePower(x, y, ffi::Py_None()),
            23 => ffi::PyNumber_InPlaceMatrixMultiply(x, y),
            24 => ffi::PyNumber_InPlaceOr(x, y),
            25 => ffi::PyNumber_InPlaceAnd(x, y),
            26 => ffi::PyNumber_InPlaceXor(x, y),
            27 => ffi::PyNumber_InPlaceLshift(x, y),
            _ => ffi::PyNumber_InPlaceRshift(x, y),
        };
        result(slots, dst, v)
    }
}

/// `a op b` for `==`, `!=`, `<`, `<=`, `>`, `>=` (`op` is CPython's
/// `Py_EQ`... code).
pub extern "C" fn lz_compare(slots: *mut u64, dst: u32, a: u32, b: u32, op: u32) -> u32 {
    unsafe {
        let (x, y) = (slot(slots, a), slot(slots, b));
        if x.is_null() || y.is_null() {
            return ERROR;
        }
        if let Some(t) = fast_compare(x, y, op) {
            let b = if t { ffi::Py_True() } else { ffi::Py_False() };
            ffi::Py_IncRef(b);
            return result(slots, dst, b);
        }
        result(slots, dst, ffi::PyObject_RichCompare(x, y, op as i32))
    }
}

/// `ForIter` over a Python iterator: `NEXT` with the next item in `dst`,
/// `TAKEN` at the end. (A Lizzard generator is resumed by `step`.)
pub extern "C" fn lz_for_iter_py(slots: *mut u64, dst: u32, it: *mut ffi::PyObject) -> u32 {
    unsafe {
        let mut item = std::ptr::null_mut();
        match PyIter_NextItem(it, &mut item) {
            1 => {
                put(slots, dst, item);
                NEXT
            }
            0 => TAKEN,
            _ => ERROR,
        }
    }
}

/// `dst = src`, leaving `src` unbound: the reference moves.
pub extern "C" fn lz_take(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        *slots.add(src as usize) = 0;
        put(slots, dst, o);
        NEXT
    }
}

/// `dst: Bool = bool(a op b)` for `==`, `!=`, `<`, `<=`, `>`, `>=`.
pub extern "C" fn lz_compare_bool(slots: *mut u64, dst: u32, a: u32, b: u32, op: u32) -> u32 {
    unsafe {
        let (x, y) = (slot(slots, a), slot(slots, b));
        if x.is_null() || y.is_null() {
            return ERROR;
        }
        if let Some(t) = fast_compare(x, y, op) {
            *slots.add(dst as usize) = t as u64;
            return NEXT;
        }
        let r = ffi::PyObject_RichCompare(x, y, op as i32);
        if r.is_null() {
            return ERROR;
        }
        // Builtin comparisons give True or False (immortal: no decref work).
        let t = if r == ffi::Py_True() {
            1
        } else if r == ffi::Py_False() {
            0
        } else {
            ffi::PyObject_IsTrue(r)
        };
        ffi::Py_DecRef(r);
        if t < 0 {
            return ERROR;
        }
        *slots.add(dst as usize) = t as u64;
        NEXT
    }
}

/// CPython's `Py_LT`... code of a rich comparison (None for `is`, `in`).
pub fn rich_code(op: lizzard_py_front::ir::CmpOp) -> Option<u32> {
    use lizzard_py_front::ir::CmpOp;
    Some(match op {
        CmpOp::Lt => 0,
        CmpOp::Le => 1,
        CmpOp::Eq => 2,
        CmpOp::Ne => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
        _ => return None,
    })
}
