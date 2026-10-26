//! Embedding slim: run programs that call Rust code.
//!
//! ```no_run
//! let mut rt = slim::Runtime::new();
//! rt.module("geom", |m| {
//!     m.function("dot", |a: slim::ListRef<f64>, b: slim::ListRef<f64>| -> f64 {
//!         a.to_vec().iter().zip(b.to_vec()).map(|(x, y)| x * y).sum()
//!     });
//! });
//! rt.run_source("from geom import dot\nprint(dot([1.0, 2.0], [3.0, 4.0]))\n", "main.slim").unwrap();
//! ```

pub use slim_vm::native::{
    Any, ArrayElem, ArrayRef, ClassBuilder, Ctx, DictRef, FromSlim, Func, GenericBuilder, IntoSlim, ListRef, Module, NType, Native, SlimError,
    SlimType, TypeArgs,
};

/// A slim interpreter with native modules.
#[derive(Default)]
pub struct Runtime {
    modules: Vec<Module>,
    /// Report objects still alive at exit (`--rc-audit`).
    pub rc_audit: bool,
}

impl Runtime {
    pub fn new() -> Runtime {
        Runtime::default()
    }

    /// Adds a native module that programs can `import`.
    pub fn module(&mut self, name: &str, build: impl FnOnce(&mut Module)) -> &mut Self {
        let mut m = Module::new(name);
        build(&mut m);
        self.modules.retain(|x| x.name != name);
        self.modules.push(m);
        self
    }

    /// Compiles `src` and returns its disassembly.
    pub fn disassemble(&self, src: &str, filename: &str) -> Result<String, String> {
        let specs: Vec<_> = self.modules.iter().map(Module::spec).collect();
        let (code, _) = slim_compiler::compile_source_with(src, filename, &specs).map_err(|e| e.render(filename, src))?;
        Ok(code.disassemble())
    }

    /// Runs a program on the current thread (which needs a large stack for
    /// deep recursion). Returns a rendered diagnostic or traceback on error.
    pub fn run_source(&self, src: &str, filename: &str) -> Result<(), String> {
        let specs: Vec<_> = self.modules.iter().map(Module::spec).collect();
        let (code, globals) = slim_compiler::compile_source_with(src, filename, &specs).map_err(|e| e.render(filename, src))?;
        let natives: Vec<_> = self.modules.iter().flat_map(Module::globals).collect();
        let vm = slim_vm::new_vm_with(globals, &natives);
        let mut th = slim_vm::Thread::new(vm);
        let result = slim_vm::run_module(&mut th, &code);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        slim_vm::shutdown(&mut th);
        if self.rc_audit {
            let live = slim_vm::rc::live_objects();
            if live != 0 {
                eprintln!("rc-audit: {live} object(s) still alive at exit (leaked or in a cycle)");
            } else {
                eprintln!("rc-audit: ok");
            }
        }
        result
    }

    /// Runs a program file (see `run_source`).
    pub fn run_file(&self, path: &str) -> Result<(), String> {
        let src = std::fs::read_to_string(path).map_err(|e| format!("slim: cannot read {path}: {e}"))?;
        self.run_source(&src, path)
    }
}
