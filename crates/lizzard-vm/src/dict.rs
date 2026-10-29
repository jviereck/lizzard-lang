//! Insertion-ordered hash map keyed by values, with lock-free reads.
//!
//! Layout follows CPython's compact dict: an append-only entry array plus an
//! open-addressing index of entry numbers. Readers probe without locking:
//! entry fields are written before the index slot that publishes them
//! (Release), deleted entries have their key cleared to `UNINIT`, and a
//! replaced table is retired through the epoch system, so a racing reader sees
//! old or new data but never freed memory. Released keys and values go
//! through deferred decrements (or ownership handed to the caller under the
//! `rc_run` revival rules).
//!
//! Writers serialize on a one-bit spin lock and never run user code while
//! holding it (hashing and key equality are pure).

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crate::helpers::values_eq;
use crate::object::Body;
use crate::rc;
use crate::value::Value;

const EMPTY: u32 = u32::MAX;

struct Entry {
    hash: AtomicU64,
    key: AtomicU64,
    value: AtomicU64,
}

struct Table {
    /// Open-addressing index (power-of-two length, at most half full).
    index: Box<[AtomicU32]>,
    entries: Box<[Entry]>,
    /// Entries `0..used` have been published.
    used: AtomicUsize,
}

impl Table {
    fn new(cap: usize) -> Box<Table> {
        let cap = cap.max(8);
        let isize = (cap * 2).next_power_of_two();
        Box::new(Table {
            index: (0..isize).map(|_| AtomicU32::new(EMPTY)).collect(),
            entries: (0..cap)
                .map(|_| Entry { hash: AtomicU64::new(0), key: AtomicU64::new(0), value: AtomicU64::new(0) })
                .collect(),
            used: AtomicUsize::new(0),
        })
    }

    /// Entry number holding `k`, if any.
    fn find(&self, k: Value, h: u64) -> Option<usize> {
        let mask = self.index.len() - 1;
        let mut p = Probe::new(h, mask);
        loop {
            let i = p.slot;
            let ix = self.index[i].load(Ordering::Acquire);
            if ix == EMPTY {
                return None;
            }
            let e = &self.entries[ix as usize];
            if e.hash.load(Ordering::Relaxed) == h {
                let key = Value(e.key.load(Ordering::Acquire));
                if key != Value::UNINIT && (key == k || values_eq(key, k)) {
                    return Some(ix as usize);
                }
            }
            p.next(mask);
        }
    }

    /// Appends an entry; the caller ensures there is room.
    fn append(&self, h: u64, k: Value, v: Value) {
        let ix = self.used.load(Ordering::Relaxed);
        let e = &self.entries[ix];
        e.hash.store(h, Ordering::Relaxed);
        e.key.store(k.0, Ordering::Relaxed);
        e.value.store(v.0, Ordering::Relaxed);
        let mask = self.index.len() - 1;
        let mut p = Probe::new(h, mask);
        while self.index[p.slot].load(Ordering::Relaxed) != EMPTY {
            p.next(mask);
        }
        self.index[p.slot].store(ix as u32, Ordering::Release);
        self.used.store(ix + 1, Ordering::Release);
    }

    /// Live entries (borrowed) at one moment.
    fn live(&self) -> impl Iterator<Item = (u64, Value, Value)> + '_ {
        let n = self.used.load(Ordering::Acquire);
        self.entries[..n].iter().filter_map(|e| {
            let k = Value(e.key.load(Ordering::Acquire));
            let v = Value(e.value.load(Ordering::Acquire));
            (k != Value::UNINIT && v != Value::UNINIT).then(|| (e.hash.load(Ordering::Relaxed), k, v))
        })
    }
}

/// # Safety
/// `p` came from `Box::into_raw(Box<Table>)` and is not used afterwards.
/// CPython's probe sequence: starts at the low bits of the hash, then
/// mixes in the higher bits, so that identity hashes of ints (which keep
/// keys used in order close together in memory) do not cluster.
struct Probe {
    slot: usize,
    perturb: u64,
}

impl Probe {
    #[inline]
    fn new(h: u64, mask: usize) -> Probe {
        Probe { slot: h as usize & mask, perturb: h }
    }

    #[inline]
    fn next(&mut self, mask: usize) {
        self.perturb >>= 5;
        self.slot = (self.slot.wrapping_mul(5).wrapping_add(1).wrapping_add(self.perturb as usize)) & mask;
    }
}

unsafe fn free_table(p: usize) {
    drop(Box::from_raw(p as *mut Table));
}

pub struct Dict {
    /// The dict's type (`dict<K, V>`), an immortal type value.
    pub ty: Value,
    wlock: AtomicBool,
    table: AtomicPtr<Table>,
    len: AtomicUsize,
}

impl Dict {
    /// An empty dict of type `ty`.
    pub fn new(ty: Value) -> Dict {
        Dict { ty, wlock: AtomicBool::new(false), table: AtomicPtr::new(std::ptr::null_mut()), len: AtomicUsize::new(0) }
    }
}

impl Drop for Dict {
    fn drop(&mut self) {
        let t = *self.table.get_mut();
        if !t.is_null() {
            // SAFETY: the dict is unreachable; its entries were released.
            unsafe { free_table(t as usize) };
        }
    }
}

struct Unlock<'a>(&'a AtomicBool);

impl Drop for Unlock<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Hash of a key, or `None` if the value is not hashable (lists, dicts).
/// A fast, non-cryptographic hasher for strings and other keys (FxHash's
/// word step, finished with splitmix64's mixer). Ints hash to themselves,
/// as in CPython. Neither resists deliberately colliding keys.
struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.write_u64(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            self.write_u64(u64::from_le_bytes(w));
        }
    }

    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    #[inline]
    fn write_u8(&mut self, x: u8) {
        self.write_u64(x as u64);
    }

    #[inline]
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }

    #[inline]
    fn write_i64(&mut self, x: i64) {
        self.write_u64(x as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        mix(self.0)
    }
}

#[inline]
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub fn hash_value(v: Value) -> Option<u64> {
    if let Some(i) = v.as_small_int() {
        return Some(i as u64);
    }
    let mut h = FastHasher(0);
    if let Some(i) = v.as_int() {
        return Some(i as u64);
    }
    if let Some(b) = v.as_bool() {
        return Some(b as u64);
    }
    match v.body() {
        None => v.0.hash(&mut h),
        Some(Body::Float(f)) => {
            if f.fract() == 0.0 && f.abs() < 9.2e18 {
                // Equal to an int key: the same hash.
                return Some(*f as i64 as u64);
            } else {
                f.to_bits().hash(&mut h);
            }
        }
        Some(Body::BigInt(b)) => b.hash(&mut h),
        Some(Body::Str(s)) => s.hash(&mut h),
        Some(Body::Tuple(items)) => {
            for x in items.iter() {
                hash_value(*x)?.hash(&mut h);
            }
        }
        Some(Body::List(_) | Body::Dict(_)) => return None,
        Some(_) => v.0.hash(&mut h),
    }
    Some(h.finish())
}

impl Dict {
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn table(&self) -> Option<&Table> {
        let t = self.table.load(Ordering::Acquire);
        // SAFETY: tables are retired through the epoch system, so one loaded
        // now stays valid until this thread's next safepoint.
        (!t.is_null()).then(|| unsafe { &*t })
    }

    fn lock(&self) -> Unlock<'_> {
        let mut spins = 0u32;
        while self.wlock.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        Unlock(&self.wlock)
    }

    /// Borrowed value for `k` (lock-free).
    pub fn get(&self, k: Value) -> Option<Value> {
        let h = hash_value(k)?;
        let t = self.table()?;
        let i = t.find(k, h)?;
        let v = Value(t.entries[i].value.load(Ordering::Acquire));
        (v != Value::UNINIT).then_some(v)
    }

    /// Makes room for one more entry (writer lock held).
    fn reserve_one(&self) -> &Table {
        let old = self.table.load(Ordering::Relaxed);
        // SAFETY: only writers replace the table, and we hold the lock.
        if let Some(t) = (!old.is_null()).then(|| unsafe { &*old }) {
            if t.used.load(Ordering::Relaxed) < t.entries.len() {
                return t;
            }
        }
        let live = self.len.load(Ordering::Relaxed);
        let new = Table::new((live + 1) * 2);
        if !old.is_null() {
            // SAFETY: as above.
            for (h, k, v) in unsafe { &*old }.live() {
                new.append(h, k, v);
            }
        }
        let p = Box::into_raw(new);
        self.table.store(p, Ordering::Release);
        if !old.is_null() {
            // SAFETY: new readers see the new table; the values moved to it.
            unsafe { rc::retire(old as usize, free_table) };
        }
        // SAFETY: just published; freed only by a later retire.
        unsafe { &*p }
    }

    /// Inserts owned `k` and `v`. Returns the references the caller must
    /// release (the duplicate key and the replaced value).
    pub fn insert(&self, k: Value, v: Value) -> Result<Option<(Value, Value)>, ()> {
        let h = hash_value(k).ok_or(())?;
        let _g = self.lock();
        if let Some(t) = self.table() {
            if let Some(i) = t.find(k, h) {
                let old = Value(t.entries[i].value.swap(v.0, Ordering::AcqRel));
                return Ok(Some((k, old)));
            }
        }
        self.reserve_one().append(h, k, v);
        self.len.fetch_add(1, Ordering::AcqRel);
        Ok(None)
    }

    /// Replaces the value for an existing key, returning the old value.
    pub fn swap(&self, k: Value, v: Value) -> Option<Value> {
        let h = hash_value(k)?;
        let _g = self.lock();
        let t = self.table()?;
        let i = t.find(k, h)?;
        Some(Value(t.entries[i].value.swap(v.0, Ordering::AcqRel)))
    }

    /// Removes `k`, returning the owned (key, value).
    pub fn remove(&self, k: Value) -> Option<(Value, Value)> {
        let h = hash_value(k)?;
        let _g = self.lock();
        let t = self.table()?;
        let i = t.find(k, h)?;
        let e = &t.entries[i];
        let key = Value(e.key.swap(Value::UNINIT.0, Ordering::AcqRel));
        let val = Value(e.value.swap(Value::UNINIT.0, Ordering::AcqRel));
        self.len.fetch_sub(1, Ordering::AcqRel);
        Some((key, val))
    }

    /// Live (key, value) pairs, borrowed, at one moment (lock-free).
    pub fn entries(&self) -> Vec<(Value, Value)> {
        match self.table() {
            Some(t) => t.live().map(|(_, k, v)| (k, v)).collect(),
            None => Vec::new(),
        }
    }

    /// Removes all entries, returning the owned references.
    pub fn drain(&self) -> Vec<(Value, Value)> {
        let _g = self.lock();
        let old = self.table.swap(std::ptr::null_mut(), Ordering::AcqRel);
        self.len.store(0, Ordering::Release);
        if old.is_null() {
            return Vec::new();
        }
        // SAFETY: we hold the lock; readers may still see the old table.
        let out = unsafe { &*old }.live().map(|(_, k, v)| (k, v)).collect();
        // SAFETY: unpublished above; its values are now owned by the caller.
        unsafe { rc::retire(old as usize, free_table) };
        out
    }

    /// Takes all entries (owned) out of an unreachable dict.
    pub(crate) fn take_all(&mut self) -> Vec<(Value, Value)> {
        let t = std::mem::replace(self.table.get_mut(), std::ptr::null_mut());
        *self.len.get_mut() = 0;
        if t.is_null() {
            return Vec::new();
        }
        // SAFETY: unreachable; nobody else holds the table.
        let t = unsafe { Box::from_raw(t) };
        let out: Vec<(Value, Value)> = t.live().map(|(_, k, v)| (k, v)).collect();
        out
    }
}
