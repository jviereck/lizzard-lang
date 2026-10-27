//! Reference-counting bookkeeping and epoch-based reclamation.
//!
//! A decrement never frees inline: objects whose count reaches zero are queued
//! on the current thread's pending-free list, and references released from
//! shared memory (fields, elements, globals) are queued as *deferred*
//! decrements. Both kinds of entries are stamped with the global epoch.
//! `Thread::rc_run` processes only entries older than the *horizon*: the
//! minimum, over all threads, of the epoch at each thread's last safepoint and
//! of its oldest borrow pin. A thread holds no unpinned borrows of shared
//! memory at a safepoint (function entry, loop back-edge), so no borrow can
//! observe an entry that is past the horizon.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::object::Obj;
use crate::value::Value;

static EPOCH: AtomicU64 = AtomicU64::new(1);
static LIVE_OBJECTS: AtomicU64 = AtomicU64::new(0);

const OFFLINE: u64 = u64::MAX;

/// Per-thread record visible to other threads.
pub struct ThreadRec {
    /// Epoch observed at the last safepoint, or `OFFLINE` while blocked.
    quiescent: AtomicU64,
    /// Oldest active borrow pin, or `u64::MAX`.
    min_pin: AtomicU64,
}

static THREADS: Mutex<Vec<Arc<ThreadRec>>> = Mutex::new(Vec::new());

/// Queue entries left behind by threads that exited.
struct Orphans {
    pending: Vec<(u64, usize)>,
    deferred: Vec<(u64, Value)>,
    retired: Vec<(u64, Retired)>,
}

static ORPHANS: Mutex<Orphans> =
    Mutex::new(Orphans { pending: Vec::new(), deferred: Vec::new(), retired: Vec::new() });

/// Raw memory (an old list buffer or dict table) that lock-free readers may
/// still be reading; freed without touching the values it contains.
pub struct Retired {
    ptr: usize,
    free: unsafe fn(usize),
}

struct RcState {
    rec: Arc<ThreadRec>,
    pending: VecDeque<(u64, *mut Obj)>,
    deferred: Vec<(u64, Value)>,
    retired: Vec<(u64, Retired)>,
    pins: Vec<u64>,
    cascade: Vec<*mut Obj>,
    in_cascade: bool,
}

impl RcState {
    fn new() -> RcState {
        let rec = Arc::new(ThreadRec {
            quiescent: AtomicU64::new(EPOCH.load(Ordering::SeqCst)),
            min_pin: AtomicU64::new(u64::MAX),
        });
        THREADS.lock().unwrap().push(rec.clone());
        RcState { rec, pending: VecDeque::new(), deferred: Vec::new(), retired: Vec::new(), pins: Vec::new(), cascade: Vec::new(), in_cascade: false }
    }
}

impl RcState {
    /// Leaves the epoch system: unregisters the thread (so it no longer
    /// holds back the horizon) and hands its queues to the orphan queue.
    fn hand_off(&mut self) {
        THREADS.lock().unwrap().retain(|r| !Arc::ptr_eq(r, &self.rec));
        let mut o = ORPHANS.lock().unwrap();
        o.pending.extend(self.pending.drain(..).map(|(e, p)| (e, p as usize)));
        o.pending.extend(self.cascade.drain(..).map(|p| (0, p as usize)));
        o.deferred.append(&mut self.deferred);
        o.retired.append(&mut self.retired);
    }
}

impl Drop for RcState {
    fn drop(&mut self) {
        self.hand_off();
    }
}

thread_local! {
    static RC: RefCell<RcState> = RefCell::new(RcState::new());
}

/// A spawned thread's last step: hands its queues to the orphan queue
/// before it reports that it finished. (The thread-local state is only
/// dropped later, when the OS thread exits, which shutdown cannot wait for.)
/// The thread must not touch slim values afterwards.
pub(crate) fn thread_exit() {
    RC.with(|rc| rc.borrow_mut().hand_off());
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

fn epoch() -> u64 {
    EPOCH.load(Ordering::SeqCst)
}

pub(crate) fn enqueue_free(p: *mut Obj) {
    let e = epoch();
    let _ = RC.try_with(|rc| {
        let mut rc = rc.borrow_mut();
        if rc.in_cascade {
            rc.cascade.push(p);
        } else {
            rc.pending.push_back((e, p));
        }
    });
}

/// Releases a reference that was held in shared memory; the decrement is
/// applied by `rc_run` once no borrow taken before now can be live.
pub fn defer_decref(v: Value) {
    if !v.is_ptr() || v.obj().is_some_and(|o| o.is_immortal()) {
        return;
    }
    let e = epoch();
    let _ = RC.try_with(|rc| rc.borrow_mut().deferred.push((e, v)));
}

/// Puts an object revived from zero back into the pending queue with a new
/// stamp, so borrows taken since it was first queued end before it is freed.
pub(crate) fn requeue(p: *mut Obj) {
    let e = epoch();
    RC.with(|rc| rc.borrow_mut().pending.push_back((e, p)));
}

/// Frees `ptr` with `free` once no lock-free reader can still hold it.
///
/// # Safety
/// `free(ptr)` must be sound to call once, from any thread, after every
/// borrow taken before now has ended.
pub unsafe fn retire(ptr: usize, free: unsafe fn(usize)) {
    let e = epoch();
    if RC.try_with(|rc| rc.borrow_mut().retired.push((e, Retired { ptr, free }))).is_err() {
        // The thread is exiting: hand the memory to the orphan queue.
        ORPHANS.lock().unwrap().retired.push((e, Retired { ptr, free }));
    }
}

/// Frees retired memory whose grace period passed.
pub(crate) fn free_retired(h: u64) {
    let mut ready = Vec::new();
    {
        let mut o = ORPHANS.lock().unwrap();
        o.retired.retain(|(e, r)| {
            if *e < h {
                ready.push(Retired { ptr: r.ptr, free: r.free });
                false
            } else {
                true
            }
        });
    }
    RC.with(|rc| {
        rc.borrow_mut().retired.retain(|(e, r)| {
            if *e < h {
                ready.push(Retired { ptr: r.ptr, free: r.free });
                false
            } else {
                true
            }
        })
    });
    for r in ready {
        // SAFETY: guaranteed by the caller of `retire`.
        unsafe { (r.free)(r.ptr) };
    }
}

/// Hands a fresh object to the epoch system: it stays alive until the next
/// `rc_run` after all current borrows end. Used for values materialized for a
/// borrow (e.g. boxing a raw float field).
pub fn autorelease(v: Value) -> Value {
    defer_decref(v);
    v
}

/// Marks a safepoint: the thread holds no unpinned borrows of shared memory.
#[inline]
pub fn safepoint() {
    let _ = RC.try_with(|rc| rc.borrow().rec.quiescent.store(epoch(), Ordering::SeqCst));
}

/// The thread is about to block (join, lock, channel): it cannot touch
/// shared memory until `online`.
pub fn offline() {
    let _ = RC.try_with(|rc| rc.borrow().rec.quiescent.store(OFFLINE, Ordering::SeqCst));
}

pub fn online() {
    safepoint();
}

/// Pins the epoch of the thread's last safepoint for the duration of a call
/// that receives borrows of shared memory.
pub fn pin() {
    RC.with(|rc| {
        let mut rc = rc.borrow_mut();
        let q = rc.rec.quiescent.load(Ordering::SeqCst);
        rc.pins.push(q);
        let m = rc.pins[0];
        rc.rec.min_pin.store(m, Ordering::SeqCst);
    });
}

pub fn unpin() {
    RC.with(|rc| {
        let mut rc = rc.borrow_mut();
        rc.pins.pop();
        let m = rc.pins.first().copied().unwrap_or(u64::MAX);
        rc.rec.min_pin.store(m, Ordering::SeqCst);
    });
}

/// Entries stamped before this epoch can be processed.
fn horizon() -> u64 {
    let threads = THREADS.lock().unwrap();
    threads
        .iter()
        .map(|r| r.quiescent.load(Ordering::SeqCst).min(r.min_pin.load(Ordering::SeqCst)))
        .min()
        .unwrap_or(u64::MAX)
}

/// Starts an `rc_run`: advances the epoch, marks a safepoint and returns
/// the horizon for this run.
pub(crate) fn begin_run() -> u64 {
    EPOCH.fetch_add(1, Ordering::SeqCst);
    safepoint();
    horizon()
}

/// Removes the deferred decrements whose grace period passed.
pub(crate) fn take_deferred(h: u64) -> Vec<Value> {
    let mut decs = Vec::new();
    {
        let mut o = ORPHANS.lock().unwrap();
        o.deferred.retain(|&(e, v)| {
            if e < h {
                decs.push(v);
                false
            } else {
                true
            }
        });
    }
    RC.with(|rc| {
        rc.borrow_mut().deferred.retain(|&(e, v)| {
            if e < h {
                decs.push(v);
                false
            } else {
                true
            }
        });
    });
    decs
}

/// While set, objects reaching zero are children of an object being freed:
/// they were only reachable through it and are freed in the same run.
pub(crate) fn set_cascade(on: bool) {
    RC.with(|rc| rc.borrow_mut().in_cascade = on);
}

/// Pops the next object that can be freed in a run with horizon `h`.
pub(crate) fn pop_ready(h: u64) -> Option<*mut Obj> {
    let local = RC.with(|rc| {
        let mut rc = rc.borrow_mut();
        if let Some(p) = rc.cascade.pop() {
            return Some(p);
        }
        match rc.pending.front() {
            Some(&(e, p)) if e < h => {
                rc.pending.pop_front();
                Some(p)
            }
            _ => None,
        }
    });
    if local.is_some() {
        return local;
    }
    let mut o = ORPHANS.lock().unwrap();
    let i = o.pending.iter().position(|&(e, _)| e < h)?;
    Some(o.pending.swap_remove(i).1 as *mut Obj)
}

/// Takes this thread's deferred decrements of local values, its queued local
/// objects, and every cascade entry (for `Thread::rc_free_local`).
pub(crate) fn take_local() -> (Vec<Value>, Vec<*mut Obj>) {
    RC.with(|rc| {
        let mut rc = rc.borrow_mut();
        let mut decs = Vec::new();
        rc.deferred.retain(|&(_, v)| {
            if v.is_local() {
                decs.push(v);
                false
            } else {
                true
            }
        });
        let mut objs: Vec<*mut Obj> = std::mem::take(&mut rc.cascade);
        rc.pending.retain(|&(_, p)| {
            // SAFETY: queued objects stay allocated until freed by `rc_run`.
            if unsafe { (*p).is_local() } {
                objs.push(p);
                false
            } else {
                true
            }
        });
        (decs, objs)
    })
}

/// Objects queued for freeing plus deferred decrements on this thread.
pub fn pending_count() -> usize {
    RC.with(|rc| {
        let rc = rc.borrow();
        rc.pending.len() + rc.deferred.len() + rc.retired.len()
    })
}

/// Total entries still queued anywhere (this thread and exited threads).
pub fn total_pending() -> usize {
    let o = ORPHANS.lock().unwrap();
    pending_count() + o.pending.len() + o.deferred.len() + o.retired.len()
}

pub(crate) fn has_ready_work() -> bool {
    pending_count() > 0
}
