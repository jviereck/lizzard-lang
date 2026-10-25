//! Reference-counting bookkeeping.
//!
//! A decrement never frees inline: objects whose count reaches zero are queued
//! on the current thread's pending-free list, and references released from
//! shared memory (fields, elements, globals) are queued as *deferred*
//! decrements that only apply once no borrow taken before them can still be
//! live (borrow pins). `Vm::rc_run` drains both queues at a time the program
//! chooses.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::Obj;
use crate::value::Value;

/// Global epoch, advanced whenever deferred decrements are queued.
static EPOCH: AtomicU64 = AtomicU64::new(1);
static LIVE_OBJECTS: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct RcState {
    pending: VecDeque<*mut Obj>,
    /// (epoch, value) of decrements waiting for the grace period.
    deferred: Vec<(u64, Value)>,
    /// Epochs at which currently active borrow pins were taken.
    pins: Vec<u64>,
}

thread_local! {
    static RC: RefCell<RcState> = RefCell::new(RcState::default());
}

pub(crate) fn note_alloc() {
    LIVE_OBJECTS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_free() {
    LIVE_OBJECTS.fetch_sub(1, Ordering::Relaxed);
}

/// Number of mortal heap objects currently allocated (for the RC audit).
pub fn live_objects() -> u64 {
    LIVE_OBJECTS.load(Ordering::Relaxed)
}

pub(crate) fn enqueue_free(p: *mut Obj) {
    let _ = RC.try_with(|rc| rc.borrow_mut().pending.push_back(p));
}

/// Releases a reference that was held in shared memory. The decrement is
/// applied by `rc_run` after every borrow pin older than now has ended.
pub fn defer_decref(v: Value) {
    if !v.is_ptr() {
        return;
    }
    if v.obj().is_some_and(|o| o.is_immortal()) {
        return;
    }
    let e = EPOCH.fetch_add(1, Ordering::Relaxed);
    let _ = RC.try_with(|rc| rc.borrow_mut().deferred.push((e, v)));
}

/// Hands a fresh object to the epoch system: it stays alive until the next
/// `rc_run` after all current borrows end. Used for values materialized for a
/// borrow (e.g. boxing a raw float field).
pub fn autorelease(v: Value) -> Value {
    defer_decref(v);
    v
}

/// Pins the current epoch: deferred decrements queued from now on are not
/// applied until the pin is released.
pub fn pin() {
    let e = EPOCH.load(Ordering::Relaxed);
    RC.with(|rc| rc.borrow_mut().pins.push(e));
}

pub fn unpin() {
    RC.with(|rc| {
        rc.borrow_mut().pins.pop();
    });
}

/// Removes and returns the deferred decrements whose grace period passed.
pub(crate) fn take_ready_deferred() -> Vec<Value> {
    RC.with(|rc| {
        let mut rc = rc.borrow_mut();
        let horizon = rc.pins.first().copied().unwrap_or(u64::MAX);
        if horizon == u64::MAX {
            return rc.deferred.drain(..).map(|(_, v)| v).collect();
        }
        let mut ready = Vec::new();
        rc.deferred.retain(|&(e, v)| {
            if e < horizon {
                ready.push(v);
                false
            } else {
                true
            }
        });
        ready
    })
}

pub(crate) fn pop_pending() -> Option<*mut Obj> {
    RC.with(|rc| rc.borrow_mut().pending.pop_front())
}

/// Objects queued for freeing plus deferred decrements.
pub fn pending_count() -> usize {
    RC.with(|rc| {
        let rc = rc.borrow();
        rc.pending.len() + rc.deferred.len()
    })
}

pub(crate) fn has_ready_work() -> bool {
    RC.with(|rc| {
        let rc = rc.borrow();
        !rc.pending.is_empty() || (!rc.deferred.is_empty() && rc.pins.is_empty())
    })
}
