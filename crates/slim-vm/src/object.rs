//! Heap objects.

use std::collections::HashMap;
use std::sync::atomic::{fence, AtomicU64, Ordering};
use std::sync::Arc;

use num_bigint::BigInt;

use crate::code::{Code, TypeDesc};
use crate::interp::{Raised, Thread};
use crate::value::Value;

const IMMORTAL: u64 = 1 << 63;
/// Set by an increment from zero: a borrow of shared memory revived an object
/// that is queued for freeing, so `rc_run` must wait for a new grace period.
const REVIVED: u64 = 1 << 62;
/// The object has an entry in a pending-free queue.
const QUEUED: u64 = 1 << 61;
/// The object is confined to the thread that created it (its type is local,
/// or it is a closure or generator holding local values): it may
/// only be referenced from that thread's frames and from other local
/// objects. Set at creation and never changed.
const LOCAL: u64 = 1 << 60;
const FLAGS: u64 = IMMORTAL | REVIVED | QUEUED | LOCAL;
/// Count word of a freed object in poison mode (`SLIM_POISON=1`, debug builds).
#[cfg(debug_assertions)]
const POISONED: u64 = 0x0BAD_F00D_DEAD_0000;

/// What `rc_run` should do with a dequeued object.
pub(crate) enum FreeCheck {
    /// The count is zero and no borrow revived it: free it.
    Free,
    /// Revived from zero: wait for a new grace period.
    Requeue,
    /// Live again (owned by someone); its next drop to zero queues it anew.
    Skip,
}

#[repr(C)]
pub struct Obj {
    rc: AtomicU64,
    pub body: Body,
}

pub enum Body {
    /// Full-width i64 that does not fit the 63-bit immediate.
    Int(i64),
    Float(f64),
    BigInt(BigInt),
    Str(Box<str>),
    Tuple(Box<[Value]>),
    List(crate::list::ListObj),
    Function(Function),
    Builtin(Builtin),
    Class(Class),
    Trait(Trait),
    Instance(Instance),
    BoundMethod { recv: Value, func: Value },
    /// A generic function with its type arguments (`first::<int>`).
    Instantiated { func: Value, targs: Box<[Value]> },
    Dict(crate::dict::Dict),
    /// `array<T>`: fixed-size packed scalars.
    Array(crate::array::ArrayObj),
    /// An array element type (`f64`, `u8`, ...), as passed to `array(...)`.
    ElemType(crate::array::Elem),
    /// A runtime type (immortal, interned).
    Type(crate::rtype::RtType),
    Iter(crate::builtins::IterObj),
    Generator(crate::generator::Generator),
    Thread(crate::threads::ThreadHandle),
    Lock(crate::threads::LockObj),
    Channel(crate::threads::ChannelObj),
    /// A Rust function or method (immortal).
    NativeFn(crate::native::NativeFn),
    /// A Rust class (immortal).
    NativeClass(crate::native::NativeClass),
    /// An instance of a Rust class.
    Native(crate::native::NativeObj),
}

pub struct Function {
    pub code: Arc<Code>,
    pub captures: Box<[Value]>,
}

pub type BuiltinFn = fn(&mut Thread, &[Value]) -> Result<Value, Raised>;

pub struct Builtin {
    pub name: &'static str,
    pub f: BuiltinFn,
    pub min_args: usize,
    pub max_args: usize,
    /// Bit i set: parameter i is `keep` (takes ownership).
    pub keep_mask: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    /// A tagged `Value` (owned reference).
    Value,
    /// Raw i64 (field typed `int`).
    RawInt,
    /// Raw f64 bits (field typed `float`).
    RawFloat,
}

pub struct Field {
    pub name: String,
    pub ty: TypeDesc,
    /// The runtime type of `ty`, resolved on first use.
    pub rt: std::sync::OnceLock<Value>,
    pub kind: SlotKind,
}

impl Field {
    pub fn new(name: String, ty: TypeDesc, kind: SlotKind) -> Field {
        Field { name, ty, rt: std::sync::OnceLock::new(), kind }
    }
}

pub struct Class {
    pub name: String,
    pub id: u32,
    /// Type parameter names; field and trait types refer to them by index.
    pub tparams: Vec<String>,
    pub fields: Vec<Field>,
    /// Inherent methods.
    pub methods: HashMap<String, Value>,
    /// Methods of trait impls (and inherited defaults), by name: (trait,
    /// function). Two traits may define the same name.
    pub trait_methods: HashMap<String, Vec<(Value, Value)>>,
    /// Trait objects this class implements (including implied ones).
    pub traits: Vec<Value>,
    /// Type arguments of generic implemented traits (`impl Iterable<T>`).
    pub trait_args: Vec<(Value, Vec<TypeDesc>)>,
    pub is_exception: bool,
    /// The class is not `Send` (a field's type is not), so every instance is
    /// local to its thread.
    pub always_local: bool,
    /// The type of instances of a non-generic class.
    pub canonical: std::sync::OnceLock<Value>,
}

impl Class {
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    pub fn implements(&self, tr: Value) -> bool {
        self.traits.contains(&tr)
    }

    /// `x.name()` without a trait: the inherent method, else the only trait
    /// method of that name (`None` if there is none or it is ambiguous).
    pub fn method(&self, name: &str) -> Option<Value> {
        if let Some(m) = self.methods.get(name) {
            return Some(*m);
        }
        match self.trait_methods.get(name).map(|v| v.as_slice()) {
            Some([(_, m)]) => Some(*m),
            _ => None,
        }
    }

    /// `Trait::name(x)`: the method of trait `tr`.
    pub fn trait_method(&self, tr: Value, name: &str) -> Option<Value> {
        self.trait_methods.get(name)?.iter().find(|(t, _)| *t == tr).map(|(_, m)| *m)
    }

    /// The traits defining `name` when an unqualified call is ambiguous.
    pub fn ambiguous(&self, name: &str) -> Vec<Value> {
        match self.trait_methods.get(name) {
            Some(v) if v.len() > 1 && !self.methods.contains_key(name) => v.iter().map(|(t, _)| *t).collect(),
            _ => Vec::new(),
        }
    }
}

pub struct Trait {
    pub name: String,
    pub id: u32,
    /// Number of type parameters (`Iterable<T>` has one).
    pub nparams: usize,
    /// (name, default implementation)
    pub methods: Vec<(String, Option<Value>)>,
    /// Traits implied by this one (e.g. `KeyError`'s `LookupError` implies `Exception`).
    pub implies: Vec<Value>,
    /// Native classes the program implements this trait for
    /// (`impl Describe for Polygon`).
    pub native_impls: std::sync::RwLock<Vec<Value>>,
}

pub struct Instance {
    pub class: Value,
    /// The instance's type (`Stack<int>`), an immortal type value.
    pub ty: Value,
    pub slots: Box<[AtomicU64]>,
}

impl Instance {
    pub fn class(&self) -> &'static Class {
        match self.class.body() {
            Some(Body::Class(c)) => c,
            _ => unreachable!("instance class is not a class"),
        }
    }

    /// Reads a slot as an owned `Value` (raw numbers are boxed). The caller
    /// must release it.
    pub fn get_owned(&self, idx: usize) -> Value {
        let raw = self.slots[idx].load(Ordering::Acquire);
        match self.class().fields[idx].kind {
            SlotKind::Value => {
                let v = Value(raw);
                v.incref();
                v
            }
            SlotKind::RawInt => Value::from_i64(raw as i64),
            SlotKind::RawFloat => Value::float(f64::from_bits(raw)),
        }
    }

    /// Reads a slot as a borrow. Raw numbers are boxed and handed to the
    /// epoch system so the box lives as long as the borrow can.
    pub fn get_borrowed(&self, idx: usize) -> Value {
        let raw = self.slots[idx].load(Ordering::Acquire);
        match self.class().fields[idx].kind {
            SlotKind::Value => Value(raw),
            SlotKind::RawInt => match Value::small_int(raw as i64) {
                Some(v) => v,
                None => crate::rc::autorelease(Value::from_i64(raw as i64)),
            },
            SlotKind::RawFloat => crate::rc::autorelease(Value::float(f64::from_bits(raw))),
        }
    }
}

#[cfg(debug_assertions)]
fn poison_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SLIM_POISON").is_some_and(|v| v != "0"))
}

/// Whether a new object must be local: its type is local, or it references
/// a local object (no non-local object may reference a local one).
fn body_is_local(body: &Body) -> bool {
    let ty_local = |t: Value| crate::rtype::as_type(t).is_some_and(|t| t.local);
    match body {
        Body::Instance(i) => ty_local(i.ty),
        Body::List(l) => ty_local(l.ty),
        Body::Dict(d) => ty_local(d.ty),
        Body::Generator(g) => ty_local(g.ty),
        Body::Iter(it) => {
            ty_local(it.ty)
                || match &*it.k.lock().unwrap_or_else(|e| e.into_inner()) {
                    crate::builtins::IterKind::List { list, .. } => list.is_local(),
                    crate::builtins::IterKind::Items { items } => items.as_slice().iter().any(|v| v.is_local()),
                    crate::builtins::IterKind::Object { obj } => obj.is_local(),
                    _ => false,
                }
        }
        Body::Tuple(items) => items.iter().any(|v| v.is_local()),
        Body::Function(f) => f.captures.iter().any(|v| v.is_local()),
        Body::BoundMethod { recv, func } => recv.is_local() || func.is_local(),
        Body::Instantiated { func, .. } => func.is_local(),
        Body::Native(n) => ty_local(n.ty),
        _ => false,
    }
}

#[cfg(debug_assertions)]
#[inline(always)]
fn check_poison(w: u64) {
    if w == POISONED {
        panic!("use after free: refcount operation on a freed object");
    }
}

impl Obj {
    /// A new list of type `ty` owning `items` (already of its element type).
    pub fn list(ty: Value, items: Vec<Value>) -> Value {
        Obj::alloc(Body::List(crate::list::ListObj::new(ty, items)))
    }

    pub fn alloc(body: Body) -> Value {
        let rc = if body_is_local(&body) { 1 | LOCAL } else { 1 };
        let b = Box::new(Obj { rc: AtomicU64::new(rc), body });
        let p = Box::into_raw(b) as u64;
        debug_assert!(p & 7 == 0);
        crate::rc::note_alloc();
        Value(p)
    }

    pub fn alloc_immortal(body: Body) -> Value {
        let b = Box::new(Obj { rc: AtomicU64::new(IMMORTAL), body });
        Value(Box::into_raw(b) as u64)
    }

    /// Whether the object is confined to its creating thread.
    #[inline(always)]
    pub fn is_local(&self) -> bool {
        self.rc.load(Ordering::Relaxed) & LOCAL != 0
    }

    #[inline(always)]
    pub fn is_immortal(&self) -> bool {
        self.rc.load(Ordering::Relaxed) & IMMORTAL != 0
    }

    pub fn refcount(&self) -> u64 {
        self.rc.load(Ordering::Relaxed) & !FLAGS
    }

    #[inline(always)]
    pub fn incref(&self) {
        let old = self.rc.load(Ordering::Relaxed);
        if old & IMMORTAL != 0 {
            return;
        }
        #[cfg(debug_assertions)]
        check_poison(old);
        let old = self.rc.fetch_add(1, Ordering::Relaxed);
        if old & !FLAGS == 0 {
            self.rc.fetch_or(REVIVED, Ordering::Relaxed);
        }
    }

    /// Decrements the count; an object reaching zero is queued for `rc_run`.
    #[inline(always)]
    pub fn decref(&self) {
        let w = self.rc.load(Ordering::Relaxed);
        if w & IMMORTAL != 0 {
            return;
        }
        #[cfg(debug_assertions)]
        check_poison(w);
        let old = self.rc.fetch_sub(1, Ordering::Release);
        if old & !FLAGS == 1 {
            fence(Ordering::Acquire);
            if self.rc.fetch_or(QUEUED, Ordering::AcqRel) & QUEUED == 0 {
                crate::rc::enqueue_free(self as *const Obj as *mut Obj);
            }
        }
    }

    /// Decides the fate of an object popped from a pending-free queue whose
    /// grace period passed.
    pub(crate) fn free_check(&self) -> FreeCheck {
        loop {
            let w = self.rc.load(Ordering::Acquire);
            let (new, verdict) = if w & !FLAGS > 0 {
                (w & !(QUEUED | REVIVED), FreeCheck::Skip)
            } else if w & REVIVED != 0 {
                (w & !REVIVED, FreeCheck::Requeue)
            } else {
                return FreeCheck::Free;
            };
            if self.rc.compare_exchange(w, new, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return verdict;
            }
        }
    }

    /// Gives a zero-count object a temporary reference while its destructor
    /// runs, without marking it revived.
    pub(crate) fn hold_for_drop(&self) {
        self.rc.fetch_add(1, Ordering::Relaxed);
    }

    /// Drops the destructor's temporary reference. Returns true if the object
    /// can be freed; false if the destructor stored a new reference (the
    /// object is live again and no longer queued).
    pub(crate) fn release_after_drop(&self) -> bool {
        loop {
            let w = self.rc.load(Ordering::Acquire);
            let new = if w & !FLAGS == 1 { w & !REVIVED & !1 } else { (w - 1) & !(QUEUED | REVIVED) };
            if self.rc.compare_exchange(w, new, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return w & !FLAGS == 1;
            }
        }
    }

    /// Marks freed memory in poison mode; returns true if the memory must be
    /// leaked instead of returned to the allocator.
    #[cfg(debug_assertions)]
    pub(crate) fn poison(p: *mut Obj) -> bool {
        if !poison_enabled() {
            return false;
        }
        // SAFETY: the object is unreachable; its body was already released.
        unsafe {
            std::ptr::drop_in_place(&mut (*p).body);
            std::ptr::write(&mut (*p).body, Body::Int(0));
            (*p).rc.store(POISONED, Ordering::Relaxed);
        }
        true
    }

    /// Releases all references held by the object (called by `rc_run` just
    /// before the memory is returned).
    pub fn release_children(&mut self) {
        match &mut self.body {
            Body::Tuple(items) => items.iter().for_each(|v| v.decref()),
            Body::List(l) => l.take_all().into_iter().for_each(|v| v.decref()),
            Body::Function(f) => f.captures.iter().for_each(|v| v.decref()),
            Body::Instance(inst) => {
                let class = inst.class();
                for (i, f) in class.fields.iter().enumerate() {
                    if f.kind == SlotKind::Value {
                        Value(*inst.slots[i].get_mut()).decref();
                    }
                }
            }
            Body::BoundMethod { recv, func } => {
                recv.decref();
                func.decref();
            }
            Body::Instantiated { func, .. } => func.decref(),
            Body::Dict(d) => {
                for (k, v) in d.take_all() {
                    k.decref();
                    v.decref();
                }
            }
            Body::Iter(it) => it.k.get_mut().unwrap_or_else(|e| e.into_inner()).release(),
            Body::Generator(g) => g.release(),
            Body::Thread(t) => t.release(),
            Body::Channel(c) => c.release(),
            Body::Array(a) => a.release(),
            Body::Native(n) => n.release(),
            Body::Lock(_) | Body::ElemType(_) | Body::Type(_) | Body::NativeFn(_) | Body::NativeClass(_) => {}
            Body::Int(_)
            | Body::Float(_)
            | Body::BigInt(_)
            | Body::Str(_)
            | Body::Builtin(_)
            | Body::Class(_)
            | Body::Trait(_) => {}
        }
    }
}

impl Value {
    pub fn str(s: impl Into<Box<str>>) -> Value {
        Obj::alloc(Body::Str(s.into()))
    }

    pub fn as_str(self) -> Option<&'static str> {
        match self.body() {
            Some(Body::Str(s)) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(self) -> Option<&'static crate::array::ArrayObj> {
        match self.body() {
            Some(Body::Array(a)) => Some(a),
            _ => None,
        }
    }

    pub fn as_instance(self) -> Option<&'static Instance> {
        match self.body() {
            Some(Body::Instance(i)) => Some(i),
            _ => None,
        }
    }

    pub fn as_class(self) -> Option<&'static Class> {
        match self.body() {
            Some(Body::Class(c)) => Some(c),
            _ => None,
        }
    }

    pub fn as_trait(self) -> Option<&'static Trait> {
        match self.body() {
            Some(Body::Trait(t)) => Some(t),
            _ => None,
        }
    }

    pub fn type_name(self) -> String {
        if self.is_small_int() {
            return "int".into();
        }
        match self {
            Value::UNINIT => return "<uninitialized>".into(),
            Value::MOVED => return "<moved>".into(),
            Value::NONE => return "None".into(),
            Value::UNIT => return "()".into(),
            Value::TRUE | Value::FALSE => return "bool".into(),
            _ => {}
        }
        match self.body() {
            Some(Body::Int(_)) => "int",
            Some(Body::Float(_)) => "float",
            Some(Body::BigInt(_)) => "bigint",
            Some(Body::Str(_)) => "str",
            Some(Body::Tuple(_)) => "tuple",
            Some(Body::List(l)) => return crate::rtype::name(l.ty).to_string(),
            Some(Body::Function(_)) | Some(Body::Builtin(_)) | Some(Body::BoundMethod { .. }) | Some(Body::Instantiated { .. }) => {
                "function"
            }
            Some(Body::Class(_)) => "class",
            Some(Body::Trait(_)) => "trait",
            Some(Body::Instance(i)) => return crate::rtype::name(i.ty).to_string(),
            Some(Body::Native(n)) => return crate::rtype::name(n.ty).to_string(),
            Some(Body::NativeFn(_)) => "function",
            Some(Body::NativeClass(_)) => "class",
            Some(Body::Dict(d)) => return crate::rtype::name(d.ty).to_string(),
            Some(Body::Array(a)) => return format!("array<{}>", a.elem.name()),
            Some(Body::ElemType(_)) | Some(Body::Type(_)) => "type",
            Some(Body::Iter(_)) => "iterator",
            Some(Body::Generator(_)) => "generator",
            Some(Body::Thread(_)) => "Thread",
            Some(Body::Lock(_)) => "Lock",
            Some(Body::Channel(c)) => return crate::rtype::name(c.ty).to_string(),
            None => "<invalid>",
        }
        .into()
    }
}
