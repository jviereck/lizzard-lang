//! Growable list with lock-free reads.
//!
//! Readers (`get`, `len`, `snapshot`) never lock: they load the buffer and
//! length with Acquire and bounds-check against both, so a racing reader sees
//! an old, new or duplicated element but never memory out of bounds. Values
//! that leave the list are released through the epoch system (deferred
//! decrements, or ownership handed to the caller under the `rc_run` revival
//! rules), and replaced buffers are retired through `rc::retire`, so a
//! borrowed element or a stale buffer stays valid until the reader's next
//! safepoint.
//!
//! Writers serialize on a one-bit spin lock. A writer never runs user code
//! or blocks while holding it.

use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use crate::rc;
use crate::value::Value;

/// A buffer: a `usize` capacity followed by `cap` slots.
struct Buf;

fn layout(cap: usize) -> Layout {
    Layout::array::<u64>(cap + 1).expect("list too large")
}

fn new_buf(cap: usize) -> *mut Buf {
    // SAFETY: the layout is non-zero sized; slots are initialized below.
    unsafe {
        let p = alloc(layout(cap)) as *mut u64;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout(cap));
        }
        p.write(cap as u64);
        for i in 0..cap {
            p.add(1 + i).write(Value::UNINIT.0);
        }
        p as *mut Buf
    }
}

/// # Safety
/// `p` came from `new_buf` and is not used afterwards.
unsafe fn free_buf(p: usize) {
    let cap = *(p as *const u64) as usize;
    dealloc(p as *mut u8, layout(cap));
}

#[inline(always)]
fn cap_of(b: *mut Buf) -> usize {
    if b.is_null() {
        0
    } else {
        // SAFETY: a non-null buffer starts with its capacity.
        unsafe { *(b as *const u64) as usize }
    }
}

#[inline(always)]
fn slot<'a>(b: *mut Buf, i: usize) -> &'a AtomicU64 {
    // SAFETY: callers check `i < cap_of(b)`; slots are `AtomicU64`-sized.
    unsafe { &*((b as *const AtomicU64).add(1 + i)) }
}

pub struct ListObj {
    wlock: AtomicBool,
    len: AtomicUsize,
    buf: AtomicPtr<Buf>,
    /// Bumped at the start and end of every write (odd while one is in
    /// progress); lets `sort` detect concurrent modification.
    version: AtomicU64,
}

// SAFETY: all shared state is atomic; buffers are reclaimed via epochs.
unsafe impl Send for ListObj {}
unsafe impl Sync for ListObj {}

impl ListObj {
    /// A list owning `items`.
    pub fn new(items: Vec<Value>) -> ListObj {
        let l = ListObj {
            wlock: AtomicBool::new(false),
            len: AtomicUsize::new(0),
            buf: AtomicPtr::new(std::ptr::null_mut()),
            version: AtomicU64::new(0),
        };
        if !items.is_empty() {
            let b = new_buf(items.len());
            for (i, v) in items.iter().enumerate() {
                slot(b, i).store(v.0, Ordering::Relaxed);
            }
            l.buf.store(b, Ordering::Relaxed);
            l.len.store(items.len(), Ordering::Relaxed);
        }
        l
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrowed element `i`, or `None` if out of range.
    #[inline]
    pub fn get(&self, i: usize) -> Option<Value> {
        let b = self.buf.load(Ordering::Acquire);
        let n = self.len.load(Ordering::Acquire);
        if i >= n.min(cap_of(b)) {
            return None;
        }
        Some(Value(slot(b, i).load(Ordering::Acquire)))
    }

    /// Borrowed elements at one moment (racy under concurrent writes).
    pub fn snapshot(&self) -> Vec<Value> {
        let b = self.buf.load(Ordering::Acquire);
        let n = self.len.load(Ordering::Acquire).min(cap_of(b));
        (0..n).map(|i| Value(slot(b, i).load(Ordering::Acquire))).collect()
    }

    /// Runs `f` with exclusive write access.
    pub fn write<R>(&self, f: impl FnOnce(&mut ListWriter<'_>) -> R) -> R {
        let mut spins = 0u32;
        while self.wlock.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        struct Unlock<'a>(&'a ListObj);
        impl Drop for Unlock<'_> {
            fn drop(&mut self) {
                self.0.version.fetch_add(1, Ordering::AcqRel);
                self.0.wlock.store(false, Ordering::Release);
            }
        }
        // The version is odd while a write is in progress.
        let prior = self.version.fetch_add(1, Ordering::AcqRel);
        let _u = Unlock(self);
        f(&mut ListWriter { l: self, prior })
    }

    /// Appends an owned value.
    pub fn push(&self, v: Value) {
        self.write(|w| w.push(v))
    }

    /// Takes all elements (owned) out of an unreachable list.
    pub(crate) fn take_all(&mut self) -> Vec<Value> {
        let b = *self.buf.get_mut();
        let n = std::mem::take(self.len.get_mut());
        (0..n).map(|i| Value(slot(b, i).load(Ordering::Relaxed))).collect()
    }
}

impl Drop for ListObj {
    fn drop(&mut self) {
        let b = *self.buf.get_mut();
        if !b.is_null() {
            // SAFETY: the list is unreachable; its values were released by
            // `release_children`.
            unsafe { free_buf(b as usize) };
        }
    }
}

/// Write access to a list (holding its writer lock).
pub struct ListWriter<'a> {
    l: &'a ListObj,
    prior: u64,
}

impl ListWriter<'_> {
    /// The list's version before this write began.
    pub fn prior_version(&self) -> u64 {
        self.prior
    }

    /// The version the list will have once this write ends.
    pub fn final_version(&self) -> u64 {
        self.prior + 2
    }

    /// The elements (borrowed), consistent because writers are excluded.
    pub fn items(&self) -> Vec<Value> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    fn buf(&self) -> *mut Buf {
        self.l.buf.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        self.l.len.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Value {
        assert!(i < self.len());
        Value(slot(self.buf(), i).load(Ordering::Relaxed))
    }

    fn put(&self, i: usize, v: Value) {
        slot(self.buf(), i).store(v.0, Ordering::Release);
    }

    fn set_len(&self, n: usize) {
        self.l.len.store(n, Ordering::Release);
    }

    /// Makes room for at least `n` elements.
    fn reserve(&mut self, n: usize) {
        let old = self.buf();
        let cap = cap_of(old);
        if n <= cap {
            return;
        }
        let new_cap = n.max(cap * 2).max(4);
        let b = new_buf(new_cap);
        for i in 0..self.len() {
            slot(b, i).store(slot(old, i).load(Ordering::Relaxed), Ordering::Relaxed);
        }
        self.l.buf.store(b, Ordering::Release);
        if !old.is_null() {
            // SAFETY: the old buffer is unreachable for new readers; its values
            // now belong to the new buffer.
            unsafe { rc::retire(old as usize, free_buf) };
        }
    }

    /// Replaces element `i` (owned `v`), returning the old value (owned).
    pub fn set(&mut self, i: usize, v: Value) -> Value {
        assert!(i < self.len());
        Value(slot(self.buf(), i).swap(v.0, Ordering::AcqRel))
    }

    pub fn push(&mut self, v: Value) {
        let n = self.len();
        self.reserve(n + 1);
        self.put(n, v);
        self.set_len(n + 1);
    }

    pub fn pop(&mut self) -> Option<Value> {
        let n = self.len();
        if n == 0 {
            return None;
        }
        let v = self.get(n - 1);
        self.set_len(n - 1);
        Some(v)
    }

    /// Inserts at `i` (clamped to the length).
    pub fn insert(&mut self, i: usize, v: Value) {
        let n = self.len();
        let i = i.min(n);
        self.reserve(n + 1);
        // Shift from the end so slot `n` is filled before the length grows.
        for j in (i..n).rev() {
            self.put(j + 1, self.get(j));
        }
        self.put(i, v);
        self.set_len(n + 1);
    }

    /// Removes element `i`, returning it (owned).
    pub fn remove(&mut self, i: usize) -> Value {
        let n = self.len();
        let v = self.get(i);
        for j in i..n - 1 {
            self.put(j, self.get(j + 1));
        }
        self.set_len(n - 1);
        v
    }

    pub fn extend(&mut self, items: Vec<Value>) {
        let n = self.len();
        self.reserve(n + items.len());
        for (k, v) in items.iter().enumerate() {
            self.put(n + k, *v);
        }
        self.set_len(n + items.len());
    }

    /// Removes all elements, returning them (owned).
    pub fn clear(&mut self) -> Vec<Value> {
        let n = self.len();
        let out: Vec<Value> = (0..n).map(|i| self.get(i)).collect();
        self.set_len(0);
        out
    }

    pub fn reverse(&mut self) {
        let n = self.len();
        for i in 0..n / 2 {
            let (a, b) = (self.get(i), self.get(n - 1 - i));
            self.put(i, b);
            self.put(n - 1 - i, a);
        }
    }

    /// Overwrites the elements with a permutation of themselves.
    pub fn permute(&mut self, items: &[Value]) {
        assert_eq!(items.len(), self.len());
        for (i, v) in items.iter().enumerate() {
            self.put(i, *v);
        }
    }
}
