//! The C ABI between the Lizzard interpreter and native extension modules.
//!
//! An extension is a dynamic library (a Rust `cdylib` built with `lizzard-ext`)
//! that exports two functions:
//!
//! ```text
//! u32   lizzard_ext_abi_version(void);                       // ABI_VERSION it was built for
//! void *lizzard_ext_init(const LizzardApi *api, StrOut err);    // the module, or NULL (`err` gets why)
//! ```
//!
//! The interpreter checks the version, then calls `lizzard_ext_init`, which
//! builds the module through `api` and returns it. An extension never touches
//! interpreter memory directly: values are opaque [`RawValue`] bits that only
//! the functions of [`LizzardApi`] interpret, every object is freed by the side
//! that allocated it (the interpreter calls an extension's [`FreeFn`]s), and
//! strings cross as borrowed [`Str`]s that the receiver copies. So an
//! extension built by another compiler, or against another build of Lizzard,
//! works as long as the ABI version matches.
//!
//! Panics must not cross the boundary: the extension catches them in its
//! callbacks and reports them as exceptions or errors.
//!
//! Statically embedded modules (`lizzard::Runtime::module`) use the same ABI.

use std::ffi::c_void;

/// Bumped on every incompatible change of [`LizzardApi`], [`ValueLayout`] or
/// the descriptors.
pub const ABI_VERSION: u32 = 2;

/// A Lizzard value: opaque bits. Each function says whether it borrows or
/// takes (consumes) a value it is given, and whether a value it returns is
/// owned (the caller must `decref` it) or borrowed.
pub type RawValue = u64;

/// How scalars are encoded, so an extension can read them (and make the
/// immediate ones) without calling into the interpreter, like the macros of
/// CPython's C API. Everything else stays opaque.
///
/// - **Immediates:** `v & small_int_mask == small_int_tag` is a small int,
///   whose value is `(v as i64) >> small_int_shift`; an int in
///   `small_int_min..=small_int_max` is encoded as
///   `((i << small_int_shift) as u64) | small_int_tag`. `None`, `()`,
///   `False` and `True` are the constants of [`LizzardApi`].
/// - **Heap objects:** `v & ptr_mask == 0 && v != 0` is a pointer to an
///   object whose first word (read atomically: other threads update it) has
///   `kind_mask` bits equal to `kind_int` for a full-width `int` and to
///   `kind_float` for a `float`. Its `i64` or `f64` is at `int_offset` or
///   `float_offset` bytes from the pointer, and never changes.
/// - Values are borrowed for the duration of a call, as everywhere: reading
///   one is only valid while it is live.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ValueLayout {
    pub small_int_mask: u64,
    pub small_int_tag: u64,
    pub small_int_shift: u32,
    pub small_int_min: i64,
    pub small_int_max: i64,
    pub ptr_mask: u64,
    pub kind_mask: u64,
    pub kind_int: u64,
    pub kind_float: u64,
    pub int_offset: usize,
    pub float_offset: usize,
}

/// A borrowed UTF-8 string.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Str {
    pub ptr: *const u8,
    pub len: usize,
}

impl Str {
    pub fn new(s: &str) -> Str {
        Str { ptr: s.as_ptr(), len: s.len() }
    }

    /// # Safety
    /// `self` must point to `len` bytes of valid UTF-8 that outlive `'a`.
    pub unsafe fn as_str<'a>(self) -> &'a str {
        if self.len == 0 {
            return "";
        }
        // SAFETY: guaranteed by the caller.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.len)) }
    }

    /// # Safety
    /// As for [`Str::as_str`], for bytes of any content.
    pub unsafe fn as_bytes<'a>(self) -> &'a [u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: guaranteed by the caller.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Receives a string, which it copies before returning.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StrOut {
    pub sink: unsafe extern "C" fn(data: *mut c_void, s: Str),
    pub data: *mut c_void,
}

/// An error made by the interpreter (null: success). The receiver owns it:
/// it reads it with `err_read`, then frees it (`err_free`) or raises it
/// (`err_raise`).
pub type HostErr = *mut c_void;

/// The interpreter thread running a native call.
pub type Ctx = *mut c_void;

/// A native function: `args` are borrowed; on success the owned result is
/// stored in `out` and it returns true. On failure it has raised an
/// exception in `ctx` (`raise` or `err_raise`) and returns false.
pub type CallFn = unsafe extern "C" fn(data: *const c_void, ctx: Ctx, args: *const RawValue, nargs: usize, out: *mut RawValue) -> bool;

/// Frees extension data handed to the interpreter.
pub type FreeFn = unsafe extern "C" fn(data: *mut c_void);

/// Formats a native object (`obj`, its Rust value) for `str()`; false on failure.
pub type ReprFn = unsafe extern "C" fn(data: *const c_void, obj: *const c_void, out: StrOut) -> bool;

/// A native function or method. The interpreter takes ownership of `data`
/// (freed with `free`) when it is given the descriptor.
#[repr(C)]
pub struct FnDesc {
    pub name: Str,
    /// The signature: [`sig`] encoding.
    pub sig: Str,
    /// A method: the first argument is the object (not in `sig`).
    pub has_self: bool,
    pub call: CallFn,
    pub data: *mut c_void,
    pub free: FreeFn,
}

/// A trait method's signature ([`sig`] encoding, without `self`).
#[repr(C)]
pub struct MethodSig {
    pub name: Str,
    pub sig: Str,
}

/// How a native class's methods may run.
pub const MODE_SYNC: u8 = 0;
/// Every call is exclusive.
pub const MODE_SEND: u8 = 1;
/// Not `Send`: local to the creating thread.
pub const MODE_LOCAL: u8 = 2;

/// A native class. The interpreter takes ownership of the data of `init`,
/// `methods` and `repr` (also when it rejects the class).
#[repr(C)]
pub struct ClassDesc {
    pub name: Str,
    /// From `class_key_new`: identifies the Rust type it wraps.
    pub key: u64,
    pub mode: u8,
    pub locked: bool,
    /// An instantiation of a generic class: its type arguments ([`sig`]
    /// type list encoding; empty otherwise).
    pub type_args: Str,
    /// Drops an instance's Rust value.
    pub drop: FreeFn,
    /// Nullable.
    pub init: *const FnDesc,
    pub methods: *const FnDesc,
    pub nmethods: usize,
    pub repr: Option<ReprFn>,
    pub repr_data: *mut c_void,
    pub repr_free: Option<FreeFn>,
    pub implements: *const Str,
    pub nimplements: usize,
}

/// An `array<T>` element.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Scalar {
    pub tag: u8,
    /// An integer's low 64 bits.
    pub bits: u64,
    pub float: f64,
}

pub const SCALAR_INT: u8 = 0;
pub const SCALAR_FLOAT: u8 = 1;
pub const SCALAR_BOOL: u8 = 2;

/// Type and signature descriptors: a type is a tag byte followed by its
/// arguments; a type list is a count byte followed by the types; a
/// signature is the parameter list followed by the result type.
pub mod sig {
    pub const INT: u8 = 0;
    pub const FLOAT: u8 = 1;
    pub const BOOL: u8 = 2;
    pub const STR: u8 = 3;
    pub const UNIT: u8 = 4;
    pub const ANY: u8 = 5;
    pub const FUNCTION: u8 = 6;
    /// Element type follows.
    pub const LIST: u8 = 7;
    /// Key and value types follow.
    pub const DICT: u8 = 8;
    /// Payload type follows.
    pub const OPTION: u8 = 9;
    /// An element code byte follows (`I8`, ..., `BOOL`).
    pub const ARRAY: u8 = 10;
    /// A class key (`u64`, little endian) follows.
    pub const CLASS: u8 = 11;

    /// `array<T>` element codes.
    pub const I8: u8 = 0;
    pub const I16: u8 = 1;
    pub const I32: u8 = 2;
    pub const I64: u8 = 3;
    pub const U8: u8 = 4;
    pub const U16: u8 = 5;
    pub const U32: u8 = 6;
    pub const U64: u8 = 7;
    pub const F32: u8 = 8;
    pub const F64: u8 = 9;
    pub const ELEM_BOOL: u8 = 10;
}

/// The interpreter's functions, passed to `lizzard_ext_init`. All take and
/// return values on the calling interpreter thread.
#[repr(C)]
pub struct LizzardApi {
    pub version: u32,
    /// `size_of::<LizzardApi>()` of the interpreter.
    pub size: usize,
    pub none: RawValue,
    pub unit: RawValue,
    pub false_: RawValue,
    pub true_: RawValue,
    pub layout: ValueLayout,

    // ---- values
    /// Counted references; only heap objects have counts, so immediates
    /// need no call (`layout.ptr_mask`).
    pub incref: unsafe extern "C" fn(v: RawValue),
    pub decref: unsafe extern "C" fn(v: RawValue),
    /// A moved-out or deleted value.
    pub is_dead: unsafe extern "C" fn(v: RawValue) -> bool,
    /// Owned results.
    pub int_new: unsafe extern "C" fn(i: i64) -> RawValue,
    pub float_new: unsafe extern "C" fn(f: f64) -> RawValue,
    pub bool_new: unsafe extern "C" fn(b: bool) -> RawValue,
    pub str_new: unsafe extern "C" fn(s: Str) -> RawValue,
    pub as_int: unsafe extern "C" fn(v: RawValue, out: *mut i64) -> bool,
    /// `float` only.
    pub as_float: unsafe extern "C" fn(v: RawValue, out: *mut f64) -> bool,
    pub as_bool: unsafe extern "C" fn(v: RawValue, out: *mut bool) -> bool,
    /// Borrowed from `v`.
    pub as_str: unsafe extern "C" fn(v: RawValue, out: *mut Str) -> bool,
    pub is_function: unsafe extern "C" fn(v: RawValue) -> bool,
    /// The name of `v`'s type (`list<int>`).
    pub type_name: unsafe extern "C" fn(v: RawValue, out: StrOut),
    /// `str(v)`.
    pub display: unsafe extern "C" fn(v: RawValue, out: StrOut),

    // ---- types
    /// The (immortal) runtime type of a [`sig`] type descriptor.
    pub type_from_desc: unsafe extern "C" fn(desc: Str) -> RawValue,
    pub type_display: unsafe extern "C" fn(ty: RawValue, out: StrOut),
    /// Whether `v` is a list or dict of exactly type `ty`.
    pub has_type: unsafe extern "C" fn(v: RawValue, ty: RawValue) -> bool,

    // ---- lists, dicts, arrays (`v` borrowed)
    /// A new list of type `ty` taking `items` (consumed, also on error).
    pub list_new: unsafe extern "C" fn(ty: RawValue, items: *const RawValue, n: usize, out: *mut RawValue) -> HostErr,
    pub list_len: unsafe extern "C" fn(v: RawValue) -> usize,
    /// Borrowed element.
    pub list_get: unsafe extern "C" fn(v: RawValue, i: usize, out: *mut RawValue) -> bool,
    /// Borrowed elements: writes up to `cap` and returns the length.
    pub list_snapshot: unsafe extern "C" fn(v: RawValue, out: *mut RawValue, cap: usize) -> usize,
    /// Consumes `item`.
    pub list_push: unsafe extern "C" fn(v: RawValue, item: RawValue) -> HostErr,
    /// Consumes `item`.
    pub list_set: unsafe extern "C" fn(v: RawValue, i: usize, item: RawValue) -> HostErr,
    pub dict_len: unsafe extern "C" fn(d: RawValue) -> usize,
    /// `k` borrowed; borrowed result.
    pub dict_get: unsafe extern "C" fn(d: RawValue, k: RawValue, out: *mut RawValue) -> bool,
    /// Consumes `k` and `v`.
    pub dict_insert: unsafe extern "C" fn(d: RawValue, k: RawValue, v: RawValue) -> HostErr,
    /// Borrowed keys, as `list_snapshot`.
    pub dict_keys: unsafe extern "C" fn(d: RawValue, out: *mut RawValue, cap: usize) -> usize,
    /// Whether `v` is an array; its element code.
    pub array_elem: unsafe extern "C" fn(v: RawValue, out: *mut u8) -> bool,
    pub array_len: unsafe extern "C" fn(v: RawValue) -> usize,
    pub array_get: unsafe extern "C" fn(v: RawValue, i: usize, out: *mut Scalar) -> bool,
    /// `x` borrowed.
    pub array_set: unsafe extern "C" fn(v: RawValue, i: usize, x: RawValue) -> HostErr,

    // ---- errors and calls
    /// Copies an error's exception class and message; true if it carries a
    /// Lizzard exception (raised by a callback), which `err_raise` re-raises.
    pub err_read: unsafe extern "C" fn(e: HostErr, kind: StrOut, msg: StrOut) -> bool,
    pub err_free: unsafe extern "C" fn(e: HostErr),
    /// Raises (and consumes) `e`.
    pub err_raise: unsafe extern "C" fn(ctx: Ctx, e: HostErr),
    /// Raises an exception of builtin class `kind` (`RuntimeError` if there is none).
    pub raise: unsafe extern "C" fn(ctx: Ctx, kind: Str, msg: Str),
    /// Calls `f` with borrowed `args`; the owned result goes to `out`.
    pub call: unsafe extern "C" fn(ctx: Ctx, f: RawValue, args: *const RawValue, n: usize, out: *mut RawValue) -> HostErr,
    /// Lets other threads reclaim memory while this one holds no values.
    pub offline: unsafe extern "C" fn(),
    pub online: unsafe extern "C" fn(),

    // ---- native objects
    /// A new class key, unique in the process.
    pub class_key_new: unsafe extern "C" fn() -> u64,
    /// A new object of the class registered for `key`, owning `data`
    /// (freed with the class's `drop`). False if no class is registered;
    /// then `data` is not taken.
    pub obj_new: unsafe extern "C" fn(key: u64, data: *mut c_void, out: *mut RawValue) -> bool,
    /// The class key of a native object (0 for other values).
    pub obj_key: unsafe extern "C" fn(v: RawValue) -> u64,
    /// Starts a method call on a native object: its data, and whether the
    /// access is exclusive (also for a shared request on a `Send` class).
    /// `BorrowError` or `ThreadError` on a conflict.
    pub obj_enter: unsafe extern "C" fn(v: RawValue, exclusive: bool, data: *mut *mut c_void, exclusive_out: *mut bool) -> HostErr,
    /// Ends it (`exclusive` as returned by `obj_enter`).
    pub obj_leave: unsafe extern "C" fn(v: RawValue, exclusive: bool),

    // ---- modules
    pub module_new: unsafe extern "C" fn(name: Str) -> *mut c_void,
    pub module_free: unsafe extern "C" fn(m: *mut c_void),
    pub module_function: unsafe extern "C" fn(m: *mut c_void, f: *const FnDesc, err: StrOut) -> bool,
    pub module_trait: unsafe extern "C" fn(m: *mut c_void, name: Str, methods: *const MethodSig, n: usize, err: StrOut) -> bool,
    pub module_class: unsafe extern "C" fn(m: *mut c_void, c: *const ClassDesc, err: StrOut) -> bool,
    /// A generic class whose instantiations are `classes`.
    pub module_generic: unsafe extern "C" fn(
        m: *mut c_void,
        name: Str,
        tparams: *const Str,
        ntparams: usize,
        classes: *const ClassDesc,
        n: usize,
        err: StrOut,
    ) -> bool,
}

/// The extension's entry point (`lizzard_ext_init`).
pub type InitFn = unsafe extern "C" fn(api: *const LizzardApi, err: StrOut) -> *mut c_void;
/// The extension's ABI version (`lizzard_ext_abi_version`).
pub type VersionFn = unsafe extern "C" fn() -> u32;

/// A [`StrOut`] appending to a `String`.
pub fn string_out(s: &mut String) -> StrOut {
    unsafe extern "C" fn push(data: *mut c_void, s: Str) {
        // SAFETY: `data` is the `&mut String` of `string_out`, and `s` is
        // valid for the call.
        unsafe { (*(data as *mut String)).push_str(s.as_str()) }
    }
    StrOut { sink: push, data: s as *mut String as *mut c_void }
}

/// Writes `s` to `out`.
///
/// # Safety
/// `out` must be a valid sink.
pub unsafe fn write_str(out: StrOut, s: &str) {
    // SAFETY: guaranteed by the caller.
    unsafe { (out.sink)(out.data, Str::new(s)) }
}
