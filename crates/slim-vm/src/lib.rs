//! The slim virtual machine: values, objects, reference counting, the opcode
//! table with backend-generic semantics, and the interpreter.

pub mod array;
pub mod backend;
pub mod builtins;
pub mod code;
pub mod dict;
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
        native_methods: Default::default(),
    })
}

/// Runs a module's code on the current thread. On an uncaught exception the
/// formatted traceback is returned.
pub fn run_module(th: &mut Thread, code: &Arc<Code>) -> Result<(), String> {
    let mut regs = vec![Value::UNINIT; code.nregs as usize];
    match interp::run_code(th, code, &mut regs, std::ptr::null()) {
        Ok(v) => {
            v.decref();
            Ok(())
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
