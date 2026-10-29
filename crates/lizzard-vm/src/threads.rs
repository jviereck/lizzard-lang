//! Threads, locks and channels. There is no global interpreter lock: every
//! Lizzard thread runs on its own OS thread and shares the heap.
//!
//! Blocking operations (join, lock acquire, channel receive) take the thread
//! *offline* for the epoch system, so other threads can keep reclaiming
//! memory while it waits.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::helpers::{self, CallArg};
use crate::interp::{Raised, Thread, VResult};
use crate::object::{Body, Obj};
use crate::rc;
use crate::types;
use crate::value::Value;

/// Result of a finished thread: its return value or its exception (owned).
type Outcome = Result<Value, Value>;

pub struct ThreadHandle {
    state: Arc<(Mutex<Option<Outcome>>, Condvar)>,
    joined: Mutex<bool>,
}

impl ThreadHandle {
    pub(crate) fn release(&mut self) {
        if let Some(r) = self.state.0.lock().unwrap().take() {
            match r {
                Ok(v) | Err(v) => v.decref(),
            }
        }
    }
}

pub struct LockObj {
    locked: Mutex<bool>,
    cv: Condvar,
}

pub struct ChannelObj {
    /// `Channel<T>`, `T` being the element type.
    pub ty: Value,
    q: Mutex<(VecDeque<Value>, bool)>,
    cv: Condvar,
}

impl ChannelObj {
    pub(crate) fn release(&mut self) {
        let (q, _) = &mut *self.q.lock().unwrap();
        q.drain(..).for_each(|v| v.decref());
    }
}

static RUNNING: (Mutex<usize>, Condvar) = (Mutex::new(0), Condvar::new());
static SPAWNED: AtomicUsize = AtomicUsize::new(0);

struct SendValues(Value, Vec<Value>, Vec<bool>);
// SAFETY: values are tagged words; objects are shared through atomic
// reference counts and the epoch system.
unsafe impl Send for SendValues {}

/// Which parameters of a callable take ownership (`keep`).
fn keep_flags(f: Value, nargs: usize) -> Vec<bool> {
    match f.body() {
        Some(Body::Function(func)) => {
            // Hidden type parameters (`<T>`) are not passed by the caller.
            let ps: Vec<bool> = func.code.params.iter().filter(|p| !p.name.starts_with('<')).map(|p| p.keep).collect();
            (0..nargs).map(|i| ps.get(i).copied().unwrap_or(false)).collect()
        }
        Some(Body::Instantiated { func, .. }) => keep_flags(*func, nargs),
        Some(Body::NativeFn(_)) | Some(Body::NativeClass(_)) => vec![false; nargs],
        Some(Body::Builtin(b)) => (0..nargs).map(|i| b.keep_mask & (1 << i) != 0).collect(),
        Some(Body::BoundMethod { func, .. }) => {
            let mut k = keep_flags(*func, nargs + 1);
            k.remove(0);
            k
        }
        Some(Body::Class(c)) => match c.methods.get("__init__") {
            Some(init) => {
                let mut k = keep_flags(*init, nargs + 1);
                k.remove(0);
                k
            }
            None => vec![false; nargs],
        },
        _ => vec![false; nargs],
    }
}

/// `spawn(f, args...)`: runs `f(*args)` on a new thread. The arguments are
/// moved into the thread (`keep`); `f` is borrowed.
pub fn spawn(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    let f = args[0];
    let fargs: Vec<Value> = args[1..].to_vec();
    if !matches!(
        f.body(),
        Some(
            Body::Function(_)
                | Body::Builtin(_)
                | Body::BoundMethod { .. }
                | Body::Class(_)
                | Body::Instantiated { .. }
                | Body::Type(_)
                | Body::NativeFn(_)
                | Body::NativeClass(_)
        )
    ) {
        fargs.iter().for_each(|v| v.decref());
        return Err(th.raise("TypeError", format!("spawn() needs a function, got '{}'", f.type_name())));
    }
    // Local values never leave their thread.
    if let Some((i, v)) = std::iter::once(&f).chain(&fargs).enumerate().find(|(_, v)| v.is_local()) {
        let what = if i == 0 { "the function".to_string() } else { format!("argument {i}") };
        let msg = format!("spawn(): {what} is not `Send` (`{}`) and cannot be sent to another thread", types::value_type_name(*v));
        fargs.iter().for_each(|v| v.decref());
        return Err(th.raise("TypeError", msg));
    }
    f.incref();
    let keeps = keep_flags(f, fargs.len());
    let state: Arc<(Mutex<Option<Outcome>>, Condvar)> = Arc::new((Mutex::new(None), Condvar::new()));
    let st2 = state.clone();
    let vm = th.vm.clone();
    let payload = SendValues(f, fargs, keeps);
    *RUNNING.0.lock().unwrap() += 1;
    SPAWNED.fetch_add(1, Ordering::Relaxed);
    let spawned = std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
        let payload = payload;
        let SendValues(f, fargs, keeps) = payload;
        let mut t = Thread::new(vm);
        let cargs: Vec<CallArg> = fargs
            .iter()
            .zip(&keeps)
            .map(|(v, k)| if *k { CallArg::owned(*v) } else { CallArg::borrowed(*v) })
            .collect();
        let r = helpers::invoke(&mut t, None, f, None, None, &cargs);
        for (v, k) in fargs.iter().zip(&keeps) {
            if !k {
                v.decref();
            }
        }
        f.decref();
        let mut outcome = match r {
            Ok(v) => Ok(v),
            Err(Raised) => {
                let e = t.exc.take().expect("raised without exception");
                e.cause.decref();
                Err(e.value)
            }
        };
        // A local result or exception cannot leave the thread: it is freed
        // here and the joiner gets a `TypeError` instead.
        let leaving = match &outcome {
            Ok(v) | Err(v) => *v,
        };
        if leaving.is_local() {
            let what = if outcome.is_ok() { "result" } else { "exception" };
            let msg = format!("the thread's {what} is not `Send` (`{}`) and cannot leave its thread", types::value_type_name(leaving));
            leaving.decref();
            let _ = t.raise("TypeError", msg);
            let e = t.exc.take().expect("raised");
            e.cause.decref();
            outcome = Err(e.value);
        }
        *st2.0.lock().unwrap() = Some(outcome);
        st2.1.notify_all();
        // Free what this thread can; local objects are freed now (only this
        // thread may run their destructors); the rest is handed to the
        // orphan queue.
        t.rc_run(None);
        t.rc_free_local();
        drop(t);
        rc::thread_exit();
        let (n, cv) = &RUNNING;
        *n.lock().unwrap() -= 1;
        cv.notify_all();
    });
    if spawned.is_err() {
        *RUNNING.0.lock().unwrap() -= 1;
        return Err(th.raise("RuntimeError", "could not start thread"));
    }
    Ok(Obj::alloc(Body::Thread(ThreadHandle { state, joined: Mutex::new(false) })))
}

/// Blocks until every spawned thread finished (interpreter shutdown).
pub fn wait_all() {
    let (n, cv) = &RUNNING;
    let mut g = n.lock().unwrap();
    rc::offline();
    while *g > 0 {
        g = cv.wait(g).unwrap();
    }
    drop(g);
    rc::online();
}

fn handle_of(v: Value) -> &'static ThreadHandle {
    match v.body() {
        Some(Body::Thread(h)) => h,
        _ => unreachable!(),
    }
}

/// `t.join()`: waits for the thread and returns its result (owned), or
/// re-raises its exception.
fn m_join(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let h = handle_of(a[0]);
    {
        let mut j = h.joined.lock().unwrap();
        if *j {
            return Err(th.raise("RuntimeError", "thread already joined"));
        }
        *j = true;
    }
    let (m, cv) = &*h.state;
    let mut g = m.lock().unwrap();
    rc::offline();
    while g.is_none() {
        g = cv.wait(g).unwrap();
    }
    rc::online();
    match g.take().unwrap() {
        Ok(v) => Ok(v),
        Err(e) => {
            drop(g);
            th.set_exception(e, Value::UNINIT);
            Err(Raised)
        }
    }
}

fn m_is_alive(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let h = handle_of(a[0]);
    Ok(Value::bool(h.state.0.lock().unwrap().is_none() && !*h.joined.lock().unwrap()))
}

pub fn new_lock() -> Value {
    Obj::alloc(Body::Lock(LockObj { locked: Mutex::new(false), cv: Condvar::new() }))
}

fn lock_of(v: Value) -> &'static LockObj {
    match v.body() {
        Some(Body::Lock(l)) => l,
        _ => unreachable!(),
    }
}

fn m_acquire(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let l = lock_of(a[0]);
    let mut g = l.locked.lock().unwrap();
    if *g {
        rc::offline();
        while *g {
            g = l.cv.wait(g).unwrap();
        }
        rc::online();
    }
    *g = true;
    Ok(Value::UNIT)
}

fn m_release(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let l = lock_of(a[0]);
    let mut g = l.locked.lock().unwrap();
    if !*g {
        return Err(th.raise("RuntimeError", "release of an unlocked Lock"));
    }
    *g = false;
    l.cv.notify_one();
    Ok(Value::UNIT)
}

fn m_locked(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(Value::bool(*lock_of(a[0]).locked.lock().unwrap()))
}

/// A new channel of type `ty` (`Channel<T>`).
pub fn new_channel(ty: Value) -> Value {
    Obj::alloc(Body::Channel(Box::new(ChannelObj { ty, q: Mutex::new((VecDeque::new(), false)), cv: Condvar::new() })))
}

fn chan_of(v: Value) -> &'static ChannelObj {
    match v.body() {
        Some(Body::Channel(c)) => c,
        _ => unreachable!(),
    }
}

/// `ch.send(v)`: moves `v` into the channel.
fn m_send(th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let c = chan_of(a[0]);
    let t = crate::rtype::as_type(c.ty).unwrap();
    let v = match crate::types::admit(th, a[1], crate::rtype::as_type(crate::rtype::elem_of(t)).unwrap(), t) {
        Ok(v) => v,
        Err(e) => {
            a[1].decref();
            return Err(e);
        }
    };
    let mut g = c.q.lock().unwrap();
    if g.1 {
        v.decref();
        return Err(th.raise("RuntimeError", "send on a closed channel"));
    }
    g.0.push_back(v);
    c.cv.notify_one();
    Ok(Value::UNIT)
}

/// Receives the next value (owned); `None` once the channel is closed and empty.
pub fn recv(ch: Value) -> Option<Value> {
    let c = chan_of(ch);
    let mut g = c.q.lock().unwrap();
    if g.0.is_empty() && !g.1 {
        rc::offline();
        while g.0.is_empty() && !g.1 {
            g = c.cv.wait(g).unwrap();
        }
        rc::online();
    }
    g.0.pop_front()
}

fn m_recv(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(recv(a[0]).unwrap_or(Value::NONE))
}

fn m_try_recv(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(chan_of(a[0]).q.lock().unwrap().0.pop_front().unwrap_or(Value::NONE))
}

fn m_close(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    let c = chan_of(a[0]);
    c.q.lock().unwrap().1 = true;
    c.cv.notify_all();
    Ok(Value::UNIT)
}

fn m_chan_len(_th: &mut Thread, a: &[Value]) -> VResult<Value> {
    Ok(Value::from_i64(chan_of(a[0]).q.lock().unwrap().0.len() as i64))
}

pub(crate) fn methods() -> Vec<(&'static str, Vec<crate::builtins::BuiltinDefPub>)> {
    use crate::builtins::BuiltinDefPub as D;
    vec![
        ("thread", vec![D::new("join", m_join, 1, 1, 0), D::new("is_alive", m_is_alive, 1, 1, 0)]),
        (
            "lock",
            vec![
                D::new("acquire", m_acquire, 1, 1, 0),
                D::new("release", m_release, 1, 1, 0),
                D::new("locked", m_locked, 1, 1, 0),
                D::new("__enter__", m_acquire, 1, 1, 0),
                D::new("__exit__", m_release, 1, 1, 0),
            ],
        ),
        (
            "channel",
            vec![
                D::new("send", m_send, 2, 2, 1 << 1),
                D::new("recv", m_recv, 1, 1, 0),
                D::new("try_recv", m_try_recv, 1, 1, 0),
                D::new("close", m_close, 1, 1, 0),
                D::new("__len__", m_chan_len, 1, 1, 0),
            ],
        ),
    ]
}
