//! 64-bit tagged values.
//!
//! ```text
//! ....xxx1  small int (63-bit payload)
//! ....x000  pointer to a heap `Obj` (never 0)
//! ....x010  special immediate (payload << 3): MOVED, None, (), False, True
//! 0         UNINIT (also the "exception pending" sentinel of runtime helpers)
//! ```

use crate::object::{Body, Obj};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Value(pub u64);

const SPECIAL_TAG: u64 = 0b010;

const fn special(n: u64) -> Value {
    Value((n << 3) | SPECIAL_TAG)
}

pub const SMALL_INT_MIN: i64 = -(1 << 62);
pub const SMALL_INT_MAX: i64 = (1 << 62) - 1;

impl Value {
    pub const UNINIT: Value = Value(0);
    pub const MOVED: Value = special(1);
    pub const NONE: Value = special(2);
    pub const UNIT: Value = special(3);
    pub const FALSE: Value = special(4);
    pub const TRUE: Value = special(5);

    #[inline(always)]
    pub fn is_small_int(self) -> bool {
        self.0 & 1 == 1
    }

    #[inline(always)]
    pub fn is_ptr(self) -> bool {
        self.0 & 7 == 0 && self.0 != 0
    }

    #[inline(always)]
    pub fn small_int(i: i64) -> Option<Value> {
        if (SMALL_INT_MIN..=SMALL_INT_MAX).contains(&i) {
            Some(Value(((i << 1) | 1) as u64))
        } else {
            None
        }
    }

    /// Small int constructor for values known to be in range.
    #[inline(always)]
    pub const fn int_unchecked(i: i64) -> Value {
        Value(((i << 1) | 1) as u64)
    }

    #[inline(always)]
    pub fn as_small_int(self) -> Option<i64> {
        if self.is_small_int() {
            Some((self.0 as i64) >> 1)
        } else {
            None
        }
    }

    /// An `int` value: a small int immediate or a boxed full-width i64.
    pub fn from_i64(i: i64) -> Value {
        match Value::small_int(i) {
            Some(v) => v,
            None => Obj::alloc(Body::Int(i)),
        }
    }

    pub fn as_int(self) -> Option<i64> {
        if let Some(i) = self.as_small_int() {
            return Some(i);
        }
        match self.body() {
            Some(Body::Int(i)) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(self) -> Option<f64> {
        match self.body() {
            Some(Body::Float(f)) => Some(*f),
            _ => None,
        }
    }

    pub fn float(f: f64) -> Value {
        Obj::alloc(Body::Float(f))
    }

    #[inline(always)]
    pub fn bool(b: bool) -> Value {
        if b {
            Value::TRUE
        } else {
            Value::FALSE
        }
    }

    #[inline(always)]
    pub fn as_bool(self) -> Option<bool> {
        match self {
            Value::TRUE => Some(true),
            Value::FALSE => Some(false),
            _ => None,
        }
    }

    /// The heap object behind a pointer value.
    #[inline(always)]
    pub fn obj(self) -> Option<&'static Obj> {
        if self.is_ptr() {
            // SAFETY: pointer values always point to live objects while a
            // reference (owned or borrowed) to them exists; frees only happen
            // in `rc_run` after the grace period.
            Some(unsafe { &*(self.0 as *const Obj) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn body(self) -> Option<&'static Body> {
        self.obj().map(|o| &o.body)
    }

    #[inline(always)]
    pub fn incref(self) {
        if let Some(o) = self.obj() {
            o.incref();
        }
    }

    #[inline(always)]
    pub fn decref(self) {
        if let Some(o) = self.obj() {
            o.decref();
        }
    }

    /// Copy types are copied implicitly: int, bigint, float, bool, None, ().
    pub fn is_copy(self) -> bool {
        if !self.is_ptr() {
            return self != Value::MOVED && self != Value::UNINIT;
        }
        matches!(self.body(), Some(Body::Int(_) | Body::Float(_) | Body::BigInt(_)))
    }

    pub fn is_dead(self) -> bool {
        self == Value::MOVED || self == Value::UNINIT
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(i) = self.as_small_int() {
            return write!(f, "{i}");
        }
        match *self {
            Value::UNINIT => write!(f, "<uninit>"),
            Value::MOVED => write!(f, "<moved>"),
            Value::NONE => write!(f, "None"),
            Value::UNIT => write!(f, "()"),
            Value::TRUE => write!(f, "True"),
            Value::FALSE => write!(f, "False"),
            v => write!(f, "<obj {:#x} {}>", v.0, v.type_name()),
        }
    }
}
