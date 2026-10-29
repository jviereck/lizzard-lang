//! The lizzard virtual machine: values, objects, reference counting, the opcode
//! table with backend-generic semantics, and the interpreter.

pub mod array;
pub mod backend;
pub mod builtins;
pub mod code;
pub mod dict;
pub mod ext;
pub mod display;
pub mod generator;
pub mod helpers;
pub mod interp;
pub mod list;
pub mod native;
pub mod object;
pub mod ops;
pub mod rc;
pub mod rtype;
pub mod sem;
pub mod threads;
pub mod types;
pub mod value;

use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::Arc;

use object::{Body, Obj};

pub use code::Code;
pub use interp::{Thread, Vm};
pub use value::Value;

/// Creates a VM whose global table has the given names (builtins are filled in).
pub fn new_vm(global_names: Vec<String>) -> Arc<Vm> {
    new_vm_with(global_names, &[])
}

/// Like `new_vm`; `natives` gives the values of native module items
/// (`("geom.dot", f)`, see `native::Module::globals`).
pub fn new_vm_with(global_names: Vec<String>, natives: &[(String, Value)]) -> Arc<Vm> {
    let next_class_id = AtomicU32::new(1);
    let (defs, builtins) = builtins::create(&next_class_id);
    let globals: Vec<AtomicU64> = global_names
        .iter()
        .map(|n| {
            let v = defs
                .iter()
                .find(|(d, _)| d == n)
                .map(|(_, v)| *v)
                .or_else(|| natives.iter().find(|(d, _)| d == n).map(|(_, v)| *v))
                .unwrap_or(Value::UNINIT);
            AtomicU64::new(v.0)
        })
        .collect();
    Arc::new(Vm {
        globals: globals.into_boxed_slice(),
        global_names,
        builtins,
        next_class_id,
        recursion_limit: 3000,
        impls: Default::default(),
    })
}

/// Runs a program on the current thread: its module code, which defines its
/// items, then its `main` (the compiler checks that there is one), given
/// `args` if it takes them. Returns the exit status (`main`'s `int` result,
/// or 0), or the formatted traceback of an uncaught exception.
pub fn run_main(th: &mut Thread, code: &Arc<Code>, args: &[String]) -> Result<i64, String> {
    let mut regs = vec![Value::UNINIT; code.nregs as usize];
    let r = interp::run_code(th, code, &mut regs, std::ptr::null()).and_then(|v| {
        v.decref();
        let main = th.vm.global_names.iter().position(|n| n == "main").map(|i| Value(th.vm.globals[i].load(std::sync::atomic::Ordering::Acquire)));
        let Some(Body::Function(f)) = main.and_then(|m| m.body()) else {
            return Err(th.raise("NameError", "the program has no `main` function"));
        };
        let argv = (!f.code.params.is_empty()).then(|| {
            Obj::list(rtype::list_of(rtype::str_()), args.iter().map(|a| Value::str(a.as_str())).collect())
        });
        let r = helpers::call_simple(th, main.unwrap(), None, argv.as_slice());
        argv.into_iter().for_each(|a| a.decref());
        r
    });
    match r {
        Ok(v) => {
            let status = v.as_int().unwrap_or(0);
            v.decref();
            Ok(status)
        }
        Err(_) => {
            let msg = th.format_exception(&code.filename);
            if let Some(e) = th.exc.take() {
                e.value.decref();
                e.cause.decref();
            }
            Err(msg)
        }
    }
}

/// Releases all globals and drains the RC queues (interpreter shutdown).
pub fn shutdown(th: &mut Thread) {
    threads::wait_all();
    // Immortal globals (functions, classes, builtins) stay available to
    // destructors that run during shutdown.
    for g in th.vm.globals.iter() {
        let v = Value(g.load(std::sync::atomic::Ordering::Acquire));
        if v.obj().is_some_and(|o| !o.is_immortal()) {
            g.store(Value::UNINIT.0, std::sync::atomic::Ordering::Release);
            v.decref();
        }
    }
    loop {
        let n = th.rc_run(None);
        if n == 0 && rc::pending_count() == 0 {
            break;
        }
        if n == 0 {
            // Only deferred decrements with live pins remain; nothing to do.
            break;
        }
    }
}
