//! Insertion-ordered hash map keyed by values.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::helpers::values_eq;
use crate::object::Body;
use crate::value::Value;

#[derive(Default)]
pub struct Dict {
    entries: Vec<Option<(Value, Value)>>,
    index: HashMap<u64, Vec<usize>>,
    len: usize,
}

/// Hash of a key, or `None` if the value is not hashable (lists, dicts).
pub fn hash_value(v: Value) -> Option<u64> {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    if let Some(i) = v.as_int() {
        i.hash(&mut h);
        return Some(h.finish());
    }
    if let Some(b) = v.as_bool() {
        (b as i64).hash(&mut h);
        return Some(h.finish());
    }
    match v.body() {
        None => v.0.hash(&mut h),
        Some(Body::Float(f)) => {
            if f.fract() == 0.0 && f.abs() < 9.2e18 {
                (*f as i64).hash(&mut h);
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
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn find(&self, k: Value, h: u64) -> Option<usize> {
        self.index.get(&h)?.iter().copied().find(|&i| matches!(self.entries[i], Some((ek, _)) if values_eq(ek, k)))
    }

    /// Borrowed value for `k`.
    pub fn get(&self, k: Value) -> Option<Value> {
        let h = hash_value(k)?;
        self.find(k, h).map(|i| self.entries[i].unwrap().1)
    }

    /// Inserts owned `k` and `v`. Returns the references the caller must
    /// release (the replaced value and the duplicate key).
    pub fn insert(&mut self, k: Value, v: Value) -> Result<Option<(Value, Value)>, ()> {
        let h = hash_value(k).ok_or(())?;
        if let Some(i) = self.find(k, h) {
            let e = self.entries[i].as_mut().unwrap();
            let old = std::mem::replace(&mut e.1, v);
            return Ok(Some((k, old)));
        }
        self.entries.push(Some((k, v)));
        self.index.entry(h).or_default().push(self.entries.len() - 1);
        self.len += 1;
        Ok(None)
    }

    /// Replaces the value for an existing key, returning the old value.
    pub fn swap(&mut self, k: Value, v: Value) -> Option<Value> {
        let h = hash_value(k)?;
        let i = self.find(k, h)?;
        let e = self.entries[i].as_mut().unwrap();
        Some(std::mem::replace(&mut e.1, v))
    }

    /// Removes `k`, returning the owned (key, value).
    pub fn remove(&mut self, k: Value) -> Option<(Value, Value)> {
        let h = hash_value(k)?;
        let i = self.find(k, h)?;
        let e = self.entries[i].take();
        if let Some(v) = self.index.get_mut(&h) {
            v.retain(|&j| j != i);
        }
        self.len -= 1;
        if self.entries.len() > 32 && self.len < self.entries.len() / 4 {
            self.compact();
        }
        e
    }

    fn compact(&mut self) {
        let entries: Vec<_> = self.entries.drain(..).flatten().collect();
        self.index.clear();
        for (i, (k, _)) in entries.iter().enumerate() {
            self.index.entry(hash_value(*k).unwrap()).or_default().push(i);
        }
        self.entries = entries.into_iter().map(Some).collect();
    }

    pub fn entries(&self) -> impl Iterator<Item = (Value, Value)> + '_ {
        self.entries.iter().flatten().copied()
    }

    /// Removes all entries, returning the owned references.
    pub fn drain(&mut self) -> Vec<(Value, Value)> {
        self.index.clear();
        self.len = 0;
        self.entries.drain(..).flatten().collect()
    }
}
