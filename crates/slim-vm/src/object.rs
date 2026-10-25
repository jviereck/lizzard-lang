//! Heap objects.

use std::collections::HashMap;
use std::sync::atomic::{fence, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use num_bigint::BigInt;

use crate::code::{Code, TypeDesc};
use crate::interp::{Raised, Thread};
use crate::value::Value;

const IMMORTAL: u64 = 1 << 63;

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
    List(Mutex<Vec<Value>>),
    Function(Function),
    Builtin(Builtin),
    Class(Class),
    Trait(Trait),
    Instance(Instance),
    BoundMethod { recv: Value, func: Value },
    Dict(Mutex<crate::dict::Dict>),
    Iter(Mutex<crate::builtins::IterKind>),
    Generator(crate::generator::Generator),
    Thread(crate::threads::ThreadHandle),
    Lock(crate::threads::LockObj),
    Channel(crate::threads::ChannelObj),
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
    pub kind: SlotKind,
}

pub struct Class {
    pub name: String,
    pub id: u32,
    pub fields: Vec<Field>,
    pub methods: HashMap<String, Value>,
    /// Trait objects this class implements (including implied ones).
    pub traits: Vec<Value>,
    pub is_exception: bool,
}

impl Class {
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    pub fn implements(&self, tr: Value) -> bool {
        self.traits.contains(&tr)
    }
}

pub struct Trait {
    pub name: String,
    pub id: u32,
    /// (name, default implementation)
    pub methods: Vec<(String, Option<Value>)>,
    /// Traits implied by this one (e.g. `KeyError`'s `LookupError` implies `Exception`).
    pub implies: Vec<Value>,
}

pub struct Instance {
    pub class: Value,
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

impl Obj {
    pub fn alloc(body: Body) -> Value {
        let b = Box::new(Obj { rc: AtomicU64::new(1), body });
        let p = Box::into_raw(b) as u64;
        debug_assert!(p & 7 == 0);
        crate::rc::note_alloc();
        Value(p)
    }

    pub fn alloc_immortal(body: Body) -> Value {
        let b = Box::new(Obj { rc: AtomicU64::new(IMMORTAL), body });
        Value(Box::into_raw(b) as u64)
    }

    #[inline(always)]
    pub fn is_immortal(&self) -> bool {
        self.rc.load(Ordering::Relaxed) & IMMORTAL != 0
    }

    pub(crate) fn set_rc(&self, n: u64) {
        self.rc.store(n, Ordering::Relaxed);
    }

    pub fn refcount(&self) -> u64 {
        self.rc.load(Ordering::Relaxed) & !IMMORTAL
    }

    #[inline(always)]
    pub fn incref(&self) {
        if !self.is_immortal() {
            self.rc.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrements the count; an object reaching zero is queued for `rc_run`.
    #[inline(always)]
    pub fn decref(&self) {
        if self.is_immortal() {
            return;
        }
        if self.rc.fetch_sub(1, Ordering::Release) == 1 {
            fence(Ordering::Acquire);
            crate::rc::enqueue_free(self as *const Obj as *mut Obj);
        }
    }

    /// Releases all references held by the object (called by `rc_run` just
    /// before the memory is returned).
    pub fn release_children(&mut self) {
        match &mut self.body {
            Body::Tuple(items) => items.iter().for_each(|v| v.decref()),
            Body::List(items) => {
                let items = items.get_mut().unwrap_or_else(|e| e.into_inner());
                items.drain(..).for_each(|v| v.decref());
            }
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
            Body::Dict(d) => {
                let d = d.get_mut().unwrap_or_else(|e| e.into_inner());
                for (k, v) in d.drain() {
                    k.decref();
                    v.decref();
                }
            }
            Body::Iter(it) => it.get_mut().unwrap_or_else(|e| e.into_inner()).release(),
            Body::Generator(g) => g.release(),
            Body::Thread(t) => t.release(),
            Body::Channel(c) => c.release(),
            Body::Lock(_) => {}
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
            Some(Body::List(_)) => "list",
            Some(Body::Function(_)) | Some(Body::Builtin(_)) | Some(Body::BoundMethod { .. }) => "function",
            Some(Body::Class(_)) => "class",
            Some(Body::Trait(_)) => "trait",
            Some(Body::Instance(i)) => return i.class().name.clone(),
            Some(Body::Dict(_)) => "dict",
            Some(Body::Iter(_)) => "iterator",
            Some(Body::Generator(_)) => "generator",
            Some(Body::Thread(_)) => "Thread",
            Some(Body::Lock(_)) => "Lock",
            Some(Body::Channel(_)) => "Channel",
            None => "<invalid>",
        }
        .into()
    }
}
