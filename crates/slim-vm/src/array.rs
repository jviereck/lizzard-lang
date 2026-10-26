//! `array<T>`: fixed-size, packed raw memory with checked element access.
//!
//! The length and the data pointer never change, so readers and writers
//! need no lock: elements are accessed with relaxed atomics of the element's
//! width, which compile to plain loads and stores while keeping racing
//! accesses well defined. Elements are scalars (copy types), so an array
//! holds no counted references; a view holds one reference to its base.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};

use crate::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
}

impl Elem {
    pub const ALL: [Elem; 11] =
        [Elem::I8, Elem::I16, Elem::I32, Elem::I64, Elem::U8, Elem::U16, Elem::U32, Elem::U64, Elem::F32, Elem::F64, Elem::Bool];

    pub fn name(self) -> &'static str {
        match self {
            Elem::I8 => "i8",
            Elem::I16 => "i16",
            Elem::I32 => "i32",
            Elem::I64 => "i64",
            Elem::U8 => "u8",
            Elem::U16 => "u16",
            Elem::U32 => "u32",
            Elem::U64 => "u64",
            Elem::F32 => "f32",
            Elem::F64 => "f64",
            Elem::Bool => "bool",
        }
    }

    pub fn from_name(n: &str) -> Option<Elem> {
        Elem::ALL.into_iter().find(|e| e.name() == n)
    }

    pub fn size(self) -> usize {
        match self {
            Elem::I8 | Elem::U8 | Elem::Bool => 1,
            Elem::I16 | Elem::U16 => 2,
            Elem::I32 | Elem::U32 | Elem::F32 => 4,
            Elem::I64 | Elem::U64 | Elem::F64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Elem::F32 | Elem::F64)
    }

    /// Integer range of the element type.
    fn int_range(self) -> (i128, i128) {
        match self {
            Elem::I8 => (i8::MIN as i128, i8::MAX as i128),
            Elem::I16 => (i16::MIN as i128, i16::MAX as i128),
            Elem::I32 => (i32::MIN as i128, i32::MAX as i128),
            Elem::I64 => (i64::MIN as i128, i64::MAX as i128),
            Elem::U8 => (0, u8::MAX as i128),
            Elem::U16 => (0, u16::MAX as i128),
            Elem::U32 => (0, u32::MAX as i128),
            Elem::U64 => (0, u64::MAX as i128),
            _ => (0, 1),
        }
    }
}

/// Why a store was rejected.
pub enum StoreError {
    /// The value has the wrong type for the element kind.
    Type,
    /// The integer does not fit the element type.
    Overflow,
}

pub struct ArrayObj {
    pub elem: Elem,
    len: usize,
    ptr: *mut u8,
    /// `UNINIT` if the array owns its memory, else a counted reference to the
    /// array whose memory this view shares.
    base: Value,
}

// SAFETY: elements are only accessed through atomics; the memory lives as
// long as the owning array (views hold a reference to it).
unsafe impl Send for ArrayObj {}
unsafe impl Sync for ArrayObj {}

fn layout(elem: Elem, len: usize) -> Layout {
    Layout::from_size_align((len * elem.size()).max(8), 8).expect("array too large")
}

impl ArrayObj {
    /// A zero-filled array.
    pub fn zeroed(elem: Elem, len: usize) -> ArrayObj {
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout(elem, len)) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout(elem, len));
        }
        ArrayObj { elem, len, ptr, base: Value::UNINIT }
    }

    /// A view of `n` elements of `base` (an array value) starting at `start`;
    /// takes a new reference to the owner of the memory.
    pub fn view(base: Value, start: usize, n: usize) -> Option<ArrayObj> {
        let a = base.as_array()?;
        if start.checked_add(n)? > a.len {
            return None;
        }
        let owner = if a.base == Value::UNINIT { base } else { a.base };
        owner.incref();
        // SAFETY: `start + n <= len`, so the view stays inside the allocation.
        let ptr = unsafe { a.ptr.add(start * a.elem.size()) };
        Some(ArrayObj { elem: a.elem, len: n, ptr, base: owner })
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw bits of element `i` (zero-extended), `None` if out of bounds.
    #[inline(always)]
    pub fn bits(&self, i: usize) -> Option<u64> {
        if i >= self.len {
            return None;
        }
        // SAFETY: `i < len`; the pointer is aligned to the element size.
        unsafe {
            let p = self.ptr.add(i * self.elem.size());
            Some(match self.elem.size() {
                1 => (*(p as *const AtomicU8)).load(Relaxed) as u64,
                2 => (*(p as *const AtomicU16)).load(Relaxed) as u64,
                4 => (*(p as *const AtomicU32)).load(Relaxed) as u64,
                _ => (*(p as *const AtomicU64)).load(Relaxed),
            })
        }
    }

    #[inline(always)]
    fn set_bits(&self, i: usize, bits: u64) {
        debug_assert!(i < self.len);
        // SAFETY: callers check `i < len`.
        unsafe {
            let p = self.ptr.add(i * self.elem.size());
            match self.elem.size() {
                1 => (*(p as *const AtomicU8)).store(bits as u8, Relaxed),
                2 => (*(p as *const AtomicU16)).store(bits as u16, Relaxed),
                4 => (*(p as *const AtomicU32)).store(bits as u32, Relaxed),
                _ => (*(p as *const AtomicU64)).store(bits, Relaxed),
            }
        }
    }

    /// Element `i` as a number: `Ok(int)`, `Ok(float)` or `Ok(bool)`.
    pub fn get_num(&self, i: usize) -> Option<Scalar> {
        let b = self.bits(i)?;
        Some(match self.elem {
            Elem::I8 => Scalar::Int(b as u8 as i8 as i128),
            Elem::I16 => Scalar::Int(b as u16 as i16 as i128),
            Elem::I32 => Scalar::Int(b as u32 as i32 as i128),
            Elem::I64 => Scalar::Int(b as i64 as i128),
            Elem::U8 | Elem::U16 | Elem::U32 | Elem::U64 => Scalar::Int(b as i128),
            Elem::F32 => Scalar::Float(f32::from_bits(b as u32) as f64),
            Elem::F64 => Scalar::Float(f64::from_bits(b)),
            Elem::Bool => Scalar::Bool(b != 0),
        })
    }

    /// Element `i` as an owned value. `Err(())` if a `u64` element does not
    /// fit in an `int`.
    pub fn get(&self, i: usize) -> Option<Result<Value, ()>> {
        Some(match self.get_num(i)? {
            Scalar::Int(n) => i64::try_from(n).map(Value::from_i64).map_err(|_| ()),
            Scalar::Float(f) => Ok(Value::float(f)),
            Scalar::Bool(b) => Ok(Value::bool(b)),
        })
    }

    /// Encodes `v` for this element type.
    pub fn encode(&self, v: Value) -> Result<u64, StoreError> {
        if self.elem == Elem::Bool {
            return v.as_bool().map(|b| b as u64).ok_or(StoreError::Type);
        }
        if self.elem.is_float() {
            let f = v.as_float().or_else(|| v.as_int().map(|i| i as f64)).ok_or(StoreError::Type)?;
            return Ok(match self.elem {
                Elem::F32 => (f as f32).to_bits() as u64,
                _ => f.to_bits(),
            });
        }
        let i = v.as_int().ok_or(StoreError::Type)? as i128;
        let (lo, hi) = self.elem.int_range();
        if i < lo || i > hi {
            return Err(StoreError::Overflow);
        }
        Ok(i as u64)
    }

    /// Stores `v` at `i` (bounds already checked by the caller).
    pub fn set(&self, i: usize, v: Value) -> Result<(), StoreError> {
        let bits = self.encode(v)?;
        self.set_bits(i, bits);
        Ok(())
    }

    pub fn fill_bits(&self, bits: u64) {
        for i in 0..self.len {
            self.set_bits(i, bits);
        }
    }

    /// Copies all elements of `src` (same kind and length).
    pub fn copy_from(&self, src: &ArrayObj) {
        for i in 0..self.len {
            self.set_bits(i, src.bits(i).unwrap());
        }
    }

    /// Releases the reference a view holds on its base.
    pub(crate) fn release(&mut self) {
        if self.base != Value::UNINIT {
            std::mem::replace(&mut self.base, Value::UNINIT).decref();
            // A released view no longer points at anything (and owns nothing).
            self.len = 0;
            self.ptr = std::ptr::null_mut();
        }
    }

    pub(crate) fn owns_memory(&self) -> bool {
        self.base == Value::UNINIT && !self.ptr.is_null()
    }
}

impl Drop for ArrayObj {
    fn drop(&mut self) {
        if self.owns_memory() {
            // SAFETY: allocated by `zeroed` with this layout; views of this
            // array hold references to it, so none outlive it.
            unsafe { dealloc(self.ptr, layout(self.elem, self.len)) };
        }
    }
}

/// An element read as a number.
pub enum Scalar {
    Int(i128),
    Float(f64),
    Bool(bool),
}
