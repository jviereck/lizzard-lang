//! Lock-free readers racing writers on one list and one dict. Readers check
//! every borrowed element; with poisoning on, a freed element would read as
//! an int and fail the check (or trip the poison panic on a refcount op).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use slim_vm::dict::Dict;
use slim_vm::object::{Body, Obj};
use slim_vm::{rc, rtype, Thread, Value};

struct Shared(Value, Value);
// SAFETY: values are tagged words; the objects are shared via atomic counts.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

fn list_of(v: Value) -> &'static slim_vm::list::ListObj {
    match v.body() {
        Some(Body::List(l)) => l,
        _ => unreachable!(),
    }
}

fn dict_of(v: Value) -> &'static slim_vm::dict::Dict {
    match v.body() {
        Some(Body::Dict(d)) => d,
        _ => unreachable!(),
    }
}

#[test]
fn readers_never_see_freed_values() {
    std::env::set_var("SLIM_POISON", "1");
    let vm = slim_vm::new_vm(Vec::new());
    let shared = Arc::new(Shared(
        Obj::list(rtype::list_of(rtype::any()), Vec::new()),
        Obj::alloc(Body::Dict(Dict::new(rtype::dict_of(rtype::any(), rtype::any())))),
    ));
    let stop = Arc::new(AtomicBool::new(false));
    let keys: Vec<Value> = (0..32).map(|i| Value::str(format!("k{i}"))).collect();
    let keys = Arc::new(Shared(Obj::list(rtype::list_of(rtype::any()), keys), Value::UNIT));
    let mut handles = Vec::new();
    for t in 0..8 {
        let (sh, stop, keys, vm) = (shared.clone(), stop.clone(), keys.clone(), vm.clone());
        handles.push(std::thread::spawn(move || {
            let mut th = Thread::new(vm);
            let (l, d) = (list_of(sh.0), dict_of(sh.1));
            let ks = list_of(keys.0);
            let mut i = 0u64;
            while !stop.load(Ordering::Relaxed) {
                i += 1;
                if t % 2 == 0 {
                    // Writer.
                    l.push(Value::str(format!("item {i}")));
                    if i % 3 == 0 {
                        if let Some(v) = l.write(|w| w.pop()) {
                            v.decref();
                        }
                    }
                    if i % 5 == 0 {
                        let old = l.write(|w| (w.len() > 0).then(|| w.set(0, Value::str("replaced"))));
                        if let Some(old) = old {
                            rc::defer_decref(old);
                        }
                    }
                    if l.len() > 200 {
                        l.write(|w| w.clear()).into_iter().for_each(rc::defer_decref);
                    }
                    let k = ks.get((i % 32) as usize).unwrap();
                    k.incref();
                    match d.insert(k, Value::str(format!("v{i}"))) {
                        Ok(Some((dup, old))) => {
                            dup.decref();
                            rc::defer_decref(old);
                        }
                        Ok(None) => {}
                        Err(()) => unreachable!(),
                    }
                    if i % 4 == 0 {
                        if let Some((k, v)) = d.remove(ks.get((i * 7 % 32) as usize).unwrap()) {
                            rc::defer_decref(k);
                            v.decref();
                        }
                    }
                } else {
                    // Reader: borrowed reads only, checked.
                    for j in 0..l.len() {
                        if let Some(v) = l.get(j) {
                            assert!(v.as_str().is_some_and(|s| !s.is_empty()), "list element freed");
                        }
                    }
                    for (k, v) in d.entries() {
                        assert!(k.as_str().is_some() && v.as_str().is_some(), "dict entry freed");
                    }
                    let k = ks.get((i % 32) as usize).unwrap();
                    if let Some(v) = d.get(k) {
                        assert!(v.as_str().is_some(), "dict value freed");
                    }
                }
                // No borrows are held across this point.
                rc::safepoint();
                if i % 64 == 0 {
                    th.rc_run(None);
                }
            }
            th.rc_run(None);
        }));
    }
    std::thread::sleep(std::time::Duration::from_millis(1500));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
}
