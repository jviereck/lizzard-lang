//! The hot instructions on raw register slots, shared by the interpreter
//! and compiled code: operands are register numbers and the objects the
//! function keeps (interned names, constants), and the work goes straight to
//! CPython's C API, without PyO3 wrappers. Each returns `NEXT` (`TAKEN` for
//! a jump), or `ERROR` with an exception set.

use std::sync::atomic::{AtomicIsize, AtomicPtr, AtomicU32, Ordering};

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
        unsafe { crate::rc::decref(old) };
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

/// An attribute site's inline cache, keyed on the type and its version tag
/// (CPython changes it whenever the type or a base class is modified, so a
/// stale entry never hits). It remembers where the attribute lives:
/// - in a `__slots__` member (read and written with `PyMember_GetOne`/`SetOne`);
/// - in the instance's inline values: CPython 3.13+ keeps a plain class's
///   instance attributes in an array right after the object, in the order
///   of the type's shared keys, unless its `__dict__` was replaced. The
///   attribute is then at a fixed offset from the object, as for CPython's
///   own `LOAD_ATTR_INSTANCE_VALUE` / `STORE_ATTR_INSTANCE_VALUE`, with the
///   same guards (see `inline`);
/// - or nowhere it can reach (the generic path).
#[derive(Default)]
pub struct AttrIc {
    entry: AtomicPtr<IcEntry>,
    misses: AtomicU32,
}

#[derive(Clone, Copy)]
enum Where {
    Slot(*mut ffi::PyMemberDef),
    /// Byte offset of the value pointer from the object.
    Inline(usize),
    Generic,
}

struct IcEntry {
    tp: *mut ffi::PyTypeObject,
    version: u32,
    at: Where,
}

/// A site that misses this often stops looking (it sees many types).
const IC_MAX_MISSES: u32 = 8;

unsafe extern "C" {
    /// CPython 3.14+: the next item (1), the end (0), or an error (-1).
    fn PyIter_NextItem(iter: *mut ffi::PyObject, item: *mut *mut ffi::PyObject) -> std::ffi::c_int;
    /// CPython 3.12+: give a type a version tag (0 if it cannot have one).
    fn PyUnstable_Type_AssignVersionTag(tp: *mut ffi::PyTypeObject) -> std::ffi::c_int;
    /// The attribute `name` on the type or its bases (a new reference), or
    /// null.
    fn _PyType_LookupRef(tp: *mut ffi::PyTypeObject, name: *mut ffi::PyObject) -> *mut ffi::PyObject;
}

/// CPython 3.14's layout of instance inline values (`Include/internal/
/// pycore_dict.h`, `pycore_object.h`). `inline::init` checks it against
/// CPython's own answers at start-up and leaves the path off otherwise.
mod inline {
    use super::*;

    pub static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    pub const TPFLAGS_INLINE_VALUES: std::ffi::c_ulong = 1 << 2;
    pub const TPFLAGS_MANAGED_DICT: std::ffi::c_ulong = 1 << 4;
    /// `struct _dictvalues { uint8 capacity, size, embedded, valid; PyObject *values[]; }`
    pub const VALID: usize = 3;
    const VALUES: usize = 8;
    /// Where the managed `__dict__` pointer lives, before the object.
    #[cfg(Py_GIL_DISABLED)]
    pub const MANAGED_DICT: isize = -8;
    #[cfg(not(Py_GIL_DISABLED))]
    pub const MANAGED_DICT: isize = -24;
    /// The managed weak reference list (`Py_TPFLAGS_MANAGED_WEAKREF`), just
    /// before the dict pointer; checked at start-up (`WEAKREF_ON`).
    pub const MANAGED_WEAKREF: isize = MANAGED_DICT - 8;
    pub const TPFLAGS_MANAGED_WEAKREF: std::ffi::c_ulong = 1 << 3;
    pub static WEAKREF_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// `struct _dictkeysobject`: kind at 10, entry count at 24, indices at
    /// 32 (`1 << dk_log2_index_bytes` bytes), then `{key, value}` entries.
    const DK_LOG2_INDEX_BYTES: usize = 9;
    const DK_KIND: usize = 10;
    pub const DK_NENTRIES: usize = 24;
    const DK_INDICES: usize = 32;
    const DICT_KEYS_SPLIT: u8 = 2;

    #[inline]
    pub unsafe fn values(o: *mut ffi::PyObject) -> *mut u8 {
        unsafe { (o as *mut u8).offset((*ffi::Py_TYPE(o)).tp_basicsize) }
    }

    /// The inline values are in use (no detached `__dict__`).
    #[inline]
    pub unsafe fn valid(o: *mut ffi::PyObject) -> bool {
        unsafe { (*(values(o).add(VALID) as *const std::sync::atomic::AtomicU8)).load(Ordering::Acquire) != 0 }
    }

    /// No `__dict__` object has been created for it.
    #[inline]
    pub unsafe fn no_dict(o: *mut ffi::PyObject) -> bool {
        unsafe { (*((o as *mut u8).offset(MANAGED_DICT) as *const AtomicPtr<ffi::PyObject>)).load(Ordering::Acquire).is_null() }
    }

    /// Record a first assignment of the value at `off` in the insertion
    /// order (`_PyDictValues_AddToInsertionOrder`): the order array follows
    /// `values[capacity]`. False when the index or the array has no room.
    pub unsafe fn add_to_order(o: *mut ffi::PyObject, off: usize) -> bool {
        unsafe {
            let v = values(o);
            let capacity = *v as usize;
            let size = *v.add(1) as usize;
            let ix = (off - (*ffi::Py_TYPE(o)).tp_basicsize as usize - VALUES) / 8;
            if ix >= capacity || size >= capacity {
                return false;
            }
            *v.add(VALUES + 8 * capacity + size) = ix as u8;
            *v.add(1) = (size + 1) as u8;
            true
        }
    }

    /// The offset of `name`'s value from an instance of `tp`, if its
    /// instances keep `name` in inline values. The caller has checked that
    /// the type has no attribute `name` and uses the generic getattr.
    pub unsafe fn offset(tp: *mut ffi::PyTypeObject, name: *mut ffi::PyObject) -> Option<usize> {
        unsafe {
            let flags = ffi::PyType_GetFlags(tp);
            if flags & TPFLAGS_INLINE_VALUES == 0 || flags & TPFLAGS_MANAGED_DICT == 0 || flags & ffi::Py_TPFLAGS_HEAPTYPE == 0 {
                return None;
            }
            let keys = shared_keys(tp)?;
            // Shared keys only grow: an index, once there, stays.
            let i = index(keys, nentries(keys), name)?;
            Some((*tp).tp_basicsize as usize + VALUES + 8 * i)
        }
    }

    /// The shared keys of a heap type's instances.
    pub unsafe fn shared_keys(tp: *mut ffi::PyTypeObject) -> Option<*const u8> {
        unsafe {
            if ffi::PyType_GetFlags(tp) & ffi::Py_TPFLAGS_HEAPTYPE == 0 {
                return None;
            }
            let keys = (*(tp as *mut ffi::PyHeapTypeObject)).ht_cached_keys as *const u8;
            (!keys.is_null() && *keys.add(DK_KIND) == DICT_KEYS_SPLIT).then_some(keys)
        }
    }

    pub unsafe fn nentries(keys: *const u8) -> isize {
        unsafe { (*(keys.add(DK_NENTRIES) as *const AtomicIsize)).load(Ordering::Acquire) }
    }

    /// The index of `name` among the first `n` entries of the keys.
    pub unsafe fn index(keys: *const u8, n: isize, name: *mut ffi::PyObject) -> Option<usize> {
        unsafe {
            let entries = keys.add(DK_INDICES + (1usize << *keys.add(DK_LOG2_INDEX_BYTES))) as *const *mut ffi::PyObject;
            (0..n.max(0) as usize).find(|&i| {
                let k = *entries.add(2 * i);
                !k.is_null() && (k == name || ffi::PyUnicode_Compare(k, name) == 0)
            })
        }
    }

    /// Check the layout with a class made for it; on success, turn the
    /// inline-values path on.
    pub fn init() {
        unsafe {
            let code = c"class _LizzardProbe:\n    pass\n_p = _LizzardProbe()\n_p.a = 1\n_p.b = 'x'\n_p.c = 2.5\n";
            let globals = ffi::PyDict_New();
            let builtins = ffi::PyEval_GetBuiltins();
            ffi::PyDict_SetItemString(globals, c"__builtins__".as_ptr(), builtins);
            let r = ffi::PyRun_String(code.as_ptr(), ffi::Py_file_input, globals, globals);
            let mut ok = !r.is_null();
            if r.is_null() {
                ffi::PyErr_Clear();
            } else {
                crate::rc::decref(r);
            }
            let p = if ok { ffi::PyDict_GetItemString(globals, c"_p".as_ptr()) } else { std::ptr::null_mut() };
            ok &= !p.is_null();
            if ok {
                let tp = ffi::Py_TYPE(p);
                ok &= valid(p) && no_dict(p);
                for n in [c"a", c"b", c"c"] {
                    let name = ffi::PyUnicode_InternFromString(n.as_ptr());
                    let want = ffi::PyObject_GetAttr(p, name);
                    ok &= match offset(tp, name) {
                        Some(off) => *((p as *mut u8).add(off) as *const *mut ffi::PyObject) == want,
                        None => false,
                    };
                    crate::rc::decref(want);
                    crate::rc::decref(name);
                }
            }
            // First assignments: on a fresh instance, store `b` then `a`
            // the way compiled code does, and ask CPython what it sees.
            if ok {
                let q = ffi::PyObject_CallNoArgs(ffi::Py_TYPE(p) as *mut ffi::PyObject);
                ok &= !q.is_null();
                if ok {
                    for (n, v) in [(c"b", c"x"), (c"a", c"y")] {
                        let name = ffi::PyUnicode_InternFromString(n.as_ptr());
                        let val = ffi::PyUnicode_FromString(v.as_ptr());
                        ok &= match offset(ffi::Py_TYPE(q), name) {
                            Some(off) => store_value(q, off, val),
                            None => false,
                        };
                        crate::rc::decref(val);
                        crate::rc::decref(name);
                    }
                    ffi::PyDict_SetItemString(globals, c"_q".as_ptr(), q);
                    crate::rc::decref(q);
                    let check = c"list(vars(_q).items()) == [('b', 'x'), ('a', 'y')] and _q.a == 'y' and not hasattr(_q, 'c')";
                    let r = ffi::PyRun_String(check.as_ptr(), ffi::Py_eval_input, globals, globals);
                    ok &= !r.is_null() && r == ffi::Py_True();
                    if r.is_null() {
                        ffi::PyErr_Clear();
                    } else {
                        crate::rc::decref(r);
                    }
                }
            }
            // The weak reference list: null, then set by `weakref.ref`.
            if ok {
                let p = ffi::PyDict_GetItemString(globals, c"_p".as_ptr());
                let at = (p as *mut u8).offset(MANAGED_WEAKREF) as *const *mut ffi::PyObject;
                let flags = ffi::PyType_GetFlags(ffi::Py_TYPE(p));
                let before = (*at).is_null();
                let r = ffi::PyRun_String(c"import weakref as _wr\n_w = _wr.ref(_p)\n".as_ptr(), ffi::Py_file_input, globals, globals);
                let set = !r.is_null() && !(*at).is_null();
                if r.is_null() {
                    ffi::PyErr_Clear();
                } else {
                    crate::rc::decref(r);
                }
                WEAKREF_ON.store(flags & TPFLAGS_MANAGED_WEAKREF != 0 && before && set, Ordering::Relaxed);
            }
            crate::rc::decref(globals);
            ON.store(ok, Ordering::Relaxed);
        }
    }
}

pub use inline::init as init_inline_values;

/// Where an instance of `tp` keeps its weak reference list (an offset from
/// the object; None: it cannot have weak references), if known.
pub fn weakref_offset(tp: *mut ffi::PyTypeObject) -> Option<Option<i32>> {
    unsafe {
        if ffi::PyType_GetFlags(tp) & inline::TPFLAGS_MANAGED_WEAKREF != 0 {
            return inline::WEAKREF_ON.load(Ordering::Relaxed).then_some(Some(inline::MANAGED_WEAKREF as i32));
        }
        match (*tp).tp_weaklistoffset {
            0 => Some(None),
            off if off > 0 => Some(Some(off as i32)),
            _ => None,
        }
    }
}

/// Whether attributes in inline values are read directly (the layout check
/// at start-up passed).
pub fn inline_values_on() -> bool {
    inline::ON.load(Ordering::Relaxed)
}

/// Whether int and float operations take the fast paths.
pub fn fast_paths_on() -> bool {
    FAST.load(Ordering::Relaxed)
}

/// Where attribute `name` of `o` lives: from the cache, or found and cached.
unsafe fn lookup(ic: &AttrIc, o: *mut ffi::PyObject, name: *mut ffi::PyObject, store: bool) -> Where {
    unsafe {
        let tp = ffi::Py_TYPE(o);
        let e = ic.entry.load(Ordering::Acquire);
        if !e.is_null() && (*e).tp == tp && (*e).version == (*tp).tp_version_tag {
            return (*e).at;
        }
        // A plain load first: a site that gave up does no atomic write.
        if ic.misses.load(Ordering::Relaxed) >= IC_MAX_MISSES {
            return Where::Generic;
        }
        ic.misses.fetch_add(1, Ordering::Relaxed);
        if PyUnstable_Type_AssignVersionTag(tp) == 0 {
            return Where::Generic;
        }
        let version = (*tp).tp_version_tag;
        let d = _PyType_LookupRef(tp, name);
        let at = if d.is_null() {
            // Not on the type: an instance attribute. With the generic
            // getattr / setattr, it is in the inline values if anywhere.
            // (Addresses of libpython's exported functions: one each.)
            let generic = if store {
                (*tp).tp_setattro.map(|f| f as *const () as usize) == Some(ffi::PyObject_GenericSetAttr as *const () as usize)
            } else {
                (*tp).tp_getattro.map(|f| f as *const () as usize) == Some(ffi::PyObject_GenericGetAttr as *const () as usize)
            };
            match (generic && inline::ON.load(Ordering::Relaxed)).then(|| inline::offset(tp, name)).flatten() {
                Some(off) => Where::Inline(off),
                // The class keeps instance attributes inline but has not
                // seen this name yet (its first instance is being set up):
                // the generic path adds it to the shared keys, which do not
                // change the class's version. Look again next time (the
                // miss counts, so a name that never arrives stops this).
                None if generic && inline::ON.load(Ordering::Relaxed) && inline::shared_keys(tp).is_some() => return Where::Generic,
                None => Where::Generic,
            }
        } else {
            let m = (ffi::Py_TYPE(d) == &raw mut ffi::PyMemberDescr_Type).then(|| (*(d as *mut ffi::PyMemberDescrObject)).d_member);
            crate::rc::decref(d);
            m.map_or(Where::Generic, Where::Slot)
        };
        // Entries are small and a site replaces its entry at most
        // `IC_MAX_MISSES` times, so replaced ones are not freed (another
        // thread may still be reading one).
        let new = Box::into_raw(Box::new(IcEntry { tp, version, at }));
        ic.entry.store(new, Ordering::Release);
        at
    }
}

/// What compiled code may assume at an attribute site (`store`: a store):
/// the type and version its cache saw, and where the attribute lives.
pub fn attr_spec(ic: &AttrIc, store: bool) -> lizzard_jit::Spec<'static> {
    unsafe {
        let e = ic.entry.load(Ordering::Acquire);
        if e.is_null() {
            return lizzard_jit::Spec::None;
        }
        let e = &*e;
        let (tp, version) = (e.tp as usize, e.version);
        match e.at {
            Where::Inline(off) => lizzard_jit::Spec::Inline {
                tp,
                version,
                off: off as i32,
                values: (*e.tp).tp_basicsize as i32,
                dict: inline::MANAGED_DICT as i32,
            },
            Where::Slot(m) if (*m).type_code == ffi::Py_T_OBJECT_EX && (!store || (*m).flags & ffi::Py_READONLY == 0) => lizzard_jit::Spec::Member { tp, version, off: (*m).offset as i32 },
            _ => lizzard_jit::Spec::None,
        }
    }
}

/// What a method call site's cache saw: the type, its version, the method
/// (borrowed from the type's dict) and the unshadowed guard (the address of
/// the shared keys' entry count, or 0, the count, and the offset of the
/// values' valid byte).
pub fn method_spec(ic: &MethodIc) -> Option<(usize, u32, *mut ffi::PyObject, usize, i64, i32)> {
    unsafe {
        let e = ic.entry.load(Ordering::Acquire);
        if e.is_null() {
            return None;
        }
        let e = &*e;
        let keys = if e.keys.is_null() { 0 } else { e.keys as usize + inline::DK_NENTRIES };
        let valid = ((*e.tp).tp_basicsize as usize + inline::VALID) as i32;
        Some((e.tp as usize, e.version, e.func, keys, e.nentries as i64, valid))
    }
}

/// A method call site's cache: for one type and version, the method
/// (a method descriptor found on the type) and why no instance attribute
/// can shadow it, as CPython's `LOAD_ATTR_METHOD_NO_DICT` and
/// `LOAD_ATTR_METHOD_WITH_VALUES` check: the instances have no `__dict__`,
/// or their attributes are in inline values whose shared keys did not have
/// the name when it was looked up and have not grown since.
#[derive(Default)]
pub struct MethodIc {
    entry: AtomicPtr<MethodEntry>,
    misses: AtomicU32,
}

struct MethodEntry {
    tp: *mut ffi::PyTypeObject,
    version: u32,
    /// Borrowed from the type's dict (alive while the version holds), and
    /// enabled for `PyUnstable_TryIncRef`.
    func: *mut ffi::PyObject,
    /// For inline values: the shared keys and how many entries they had;
    /// null for instances without a `__dict__`.
    keys: *const u8,
    nentries: isize,
}

unsafe extern "C" {
    fn PyUnstable_EnableTryIncRef(obj: *mut ffi::PyObject);
    #[cfg(not(Py_GIL_DISABLED))]
    fn PyUnstable_TryIncRef(obj: *mut ffi::PyObject) -> std::ffi::c_int;
}

/// The method `name` of `o` (a new reference) when the cache can prove what
/// `o.name` is, else None (the caller looks it up the generic way).
pub unsafe fn method(ic: &MethodIc, o: *mut ffi::PyObject, name: *mut ffi::PyObject) -> Option<*mut ffi::PyObject> {
    unsafe {
        let tp = ffi::Py_TYPE(o);
        let e = ic.entry.load(Ordering::Acquire);
        if !e.is_null() && (*e).tp == tp && (*e).version == (*tp).tp_version_tag {
            let e = &*e;
            let unshadowed = e.keys.is_null() || (inline::valid(o) && inline::nentries(e.keys) == e.nentries);
            if unshadowed && PyUnstable_TryIncRef(e.func) == 1 {
                return Some(e.func);
            }
            return None;
        }
        if ic.misses.load(Ordering::Relaxed) >= IC_MAX_MISSES || !inline::ON.load(Ordering::Relaxed) {
            return None;
        }
        ic.misses.fetch_add(1, Ordering::Relaxed);
        if PyUnstable_Type_AssignVersionTag(tp) == 0 || (*tp).tp_getattro.map(|f| f as *const () as usize) != Some(ffi::PyObject_GenericGetAttr as *const () as usize) {
            return None;
        }
        let version = (*tp).tp_version_tag;
        let d = _PyType_LookupRef(tp, name);
        if d.is_null() {
            return None;
        }
        let entry = (|| {
            if ffi::PyType_GetFlags(ffi::Py_TYPE(d)) & ffi::Py_TPFLAGS_METHOD_DESCRIPTOR == 0 {
                return None;
            }
            let flags = ffi::PyType_GetFlags(tp);
            let (keys, nentries) = if flags & inline::TPFLAGS_MANAGED_DICT == 0 && (*tp).tp_dictoffset == 0 {
                (std::ptr::null(), 0)
            } else {
                let keys = inline::shared_keys(tp)?;
                if !inline::valid(o) {
                    return None;
                }
                let n = inline::nentries(keys);
                if inline::index(keys, n, name).is_some() {
                    return None;
                }
                (keys, n)
            };
            PyUnstable_EnableTryIncRef(d);
            Some(MethodEntry { tp, version, func: d, keys, nentries })
        })();
        let r = match entry {
            Some(m) => {
                crate::rc::incref(d);
                // As for attribute entries, replaced ones are not freed.
                ic.entry.store(Box::into_raw(Box::new(m)), Ordering::Release);
                Some(d)
            }
            None => None,
        };
        crate::rc::decref(d);
        r
    }
}

/// A construction site's cache (`C(args)` where `C` is a class), as
/// CPython's `CALL_ALLOC_AND_ENTER_INIT`: for one class and version,
/// creating an instance is allocating it and calling its `__init__`. That
/// holds when the class is a heap type, not abstract, with `object`'s
/// `__new__` (which only allocates) and allocator, and an `__init__` that is
/// a Python or Lizzard function (so its `tp_init` calls it). The caller
/// checks that the class's metaclass is `type` itself (no custom
/// `__call__`). Any change to the class or a base changes its version.
#[derive(Default)]
pub struct CtorIc {
    entry: AtomicPtr<CtorEntry>,
    misses: AtomicU32,
}

struct CtorEntry {
    tp: *mut ffi::PyTypeObject,
    version: u32,
    /// Borrowed from the class's dict (alive while the version holds), and
    /// enabled for `PyUnstable_TryIncRef`.
    init: *mut ffi::PyObject,
}

const TPFLAGS_IS_ABSTRACT: std::ffi::c_ulong = 1 << 20;

/// What a construction site's cache saw: the class, its version and its
/// `__init__` (borrowed from the class's dict).
pub fn ctor_spec(ic: &CtorIc) -> Option<(*mut ffi::PyTypeObject, u32, *mut ffi::PyObject)> {
    let e = ic.entry.load(Ordering::Acquire);
    // SAFETY: entries are never freed.
    (!e.is_null()).then(|| unsafe { ((*e).tp, (*e).version, (*e).init) })
}

/// The `__init__` (a new reference) when constructing an instance of `tp`
/// is allocating it and calling that; else None (the generic `type` call).
pub unsafe fn ctor(ic: &CtorIc, tp: *mut ffi::PyTypeObject, init_name: *mut ffi::PyObject, lizzard_function: *mut ffi::PyTypeObject) -> Option<*mut ffi::PyObject> {
    unsafe {
        let e = ic.entry.load(Ordering::Acquire);
        if !e.is_null() && (*e).tp == tp && (*e).version == (*tp).tp_version_tag {
            return (PyUnstable_TryIncRef((*e).init) == 1).then_some((*e).init);
        }
        if ic.misses.load(Ordering::Relaxed) >= IC_MAX_MISSES {
            return None;
        }
        ic.misses.fetch_add(1, Ordering::Relaxed);
        let flags = ffi::PyType_GetFlags(tp);
        let plain = flags & ffi::Py_TPFLAGS_HEAPTYPE != 0
            && flags & TPFLAGS_IS_ABSTRACT == 0
            && (*tp).tp_new.map(|f| f as *const () as usize) == ffi::PyBaseObject_Type.tp_new.map(|f| f as *const () as usize)
            && (*tp).tp_alloc.map(|f| f as *const () as usize) == Some(ffi::PyType_GenericAlloc as *const () as usize);
        if !plain || PyUnstable_Type_AssignVersionTag(tp) == 0 {
            return None;
        }
        let version = (*tp).tp_version_tag;
        let init = _PyType_LookupRef(tp, init_name);
        if init.is_null() {
            return None;
        }
        let it = ffi::Py_TYPE(init);
        if it != lizzard_function && it != &raw mut ffi::PyFunction_Type {
            ffi::Py_DecRef(init);
            return None;
        }
        PyUnstable_EnableTryIncRef(init);
        // As for the other caches, replaced entries are not freed.
        ic.entry.store(Box::into_raw(Box::new(CtorEntry { tp, version, init })), Ordering::Release);
        // `init` is the new reference returned; the entry's is borrowed from
        // the class's dict.
        Some(init)
    }
}

/// A new reference to the value at `slot` (an inline value of a live
/// object), or null. In the free-threaded build another thread may replace
/// and release it meanwhile: as CPython's `_Py_TryIncrefCompare`, take the
/// reference only if the count is not zero and the slot still holds it.
#[inline]
unsafe fn load_value(slot: *const AtomicPtr<ffi::PyObject>) -> *mut ffi::PyObject {
    unsafe {
        let v = (*slot).load(Ordering::Acquire);
        if v.is_null() {
            return v;
        }
        #[cfg(Py_GIL_DISABLED)]
        {
            if PyUnstable_TryIncRef(v) == 0 {
                return std::ptr::null_mut();
            }
            if (*slot).load(Ordering::Acquire) != v {
                crate::rc::decref(v);
                return std::ptr::null_mut();
            }
        }
        #[cfg(not(Py_GIL_DISABLED))]
        crate::rc::incref(v);
        v
    }
}

#[cfg(Py_GIL_DISABLED)]
unsafe extern "C" {
    fn PyUnstable_TryIncRef(obj: *mut ffi::PyObject) -> std::ffi::c_int;
    fn PyCriticalSection_Begin(c: *mut [usize; 2], op: *mut ffi::PyObject);
    fn PyCriticalSection_End(c: *mut [usize; 2]);
}

/// Store an inline value at `off`, as CPython's `STORE_ATTR_INSTANCE_VALUE`:
/// replace the existing value, or (a first assignment) set it and append its
/// index to the values' insertion order, which `vars()` and `__dict__`
/// follow. Returns false to take the generic path (a `__dict__` object
/// exists, the values are detached, or no room for the index).
#[inline]
unsafe fn store_value(o: *mut ffi::PyObject, off: usize, v: *mut ffi::PyObject) -> bool {
    unsafe {
        // As CPython's own setattr: under the object's critical section.
        #[cfg(Py_GIL_DISABLED)]
        let mut cs = [0usize; 2];
        #[cfg(Py_GIL_DISABLED)]
        PyCriticalSection_Begin(&mut cs, o);
        let slot = (o as *mut u8).add(off) as *const AtomicPtr<ffi::PyObject>;
        let old = (*slot).load(Ordering::Relaxed);
        let mut ok = inline::valid(o) && inline::no_dict(o);
        if ok && old.is_null() {
            ok = inline::add_to_order(o, off);
        }
        if ok {
            crate::rc::incref(v);
            (*slot).store(v, Ordering::Release);
        }
        #[cfg(Py_GIL_DISABLED)]
        PyCriticalSection_End(&mut cs);
        if ok && !old.is_null() {
            crate::rc::decref(old);
        }
        ok
    }
}

pub extern "C" fn lz_load_attr(slots: *mut u64, dst: u32, obj: u32, name: *mut ffi::PyObject, ic: *const AttrIc) -> u32 {
    unsafe {
        let o = slot(slots, obj);
        if o.is_null() {
            return ERROR;
        }
        let v = match lookup(&*ic, o, name, false) {
            Where::Slot(m) => ffi::PyMember_GetOne(o as *const std::ffi::c_char, m),
            Where::Inline(off) if inline::valid(o) => match load_value((o as *mut u8).add(off) as *const AtomicPtr<ffi::PyObject>) {
                // Unset (or being replaced): the generic path decides.
                v if v.is_null() => ffi::PyObject_GetAttr(o, name),
                v => v,
            },
            _ => ffi::PyObject_GetAttr(o, name),
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
        let r = match lookup(&*ic, o, name, true) {
            Where::Slot(m) => ffi::PyMember_SetOne(o as *mut std::ffi::c_char, m, v),
            Where::Inline(off) if store_value(o, off, v) => 0,
            _ => ffi::PyObject_SetAttr(o, name, v),
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
        crate::rc::incref(obj);
        put(slots, dst, obj);
    }
}

pub extern "C" fn lz_move(slots: *mut u64, dst: u32, src: u32) -> u32 {
    unsafe {
        let o = slot(slots, src);
        if o.is_null() {
            return ERROR;
        }
        crate::rc::incref(o);
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
            crate::rc::decref(o);
        }
        let big = ffi::PyLong_FromLongLong(1 << 40);
        ok &= small_int(big).is_none();
        crate::rc::decref(big);
        let f = ffi::PyFloat_FromDouble(2.5);
        ok &= exact_float(f) == Some(2.5);
        crate::rc::decref(f);
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
            crate::rc::incref(b);
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
        crate::rc::decref(r);
        if t < 0 {
            return ERROR;
        }
        *slots.add(dst as usize) = t as u64;
        NEXT
    }
}

