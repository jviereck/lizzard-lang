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
//!
//! Native modules come from two places:
//! - [`Runtime::module`]: built by the host program, which links them in;
//! - extension libraries: a `cdylib` declared with [`module!`], loaded when
//!   a program imports a module that the runtime does not have. `import
//!   geom` loads `libgeom.so` (`libgeom.dylib` on macOS, `geom.dll` on
//!   Windows) from the program's directory or a directory of
//!   [`Runtime::search_path`] (initially `SLIM_PATH`).
//!
//! Both use the same binding API (the `slim-ext` crate, re-exported here)
//! over the same C ABI, so a module can be built either way.

use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use slim_ext::{
    module, Any, ArrayElem, ArrayRef, ClassBuilder, Ctx, DictRef, Elem, FromSlim, Func, GenericBuilder, IntoSlim, ListRef, Module, NType, Native,
    SlimError, SlimType, TypeArgs, Value,
};

mod load;

/// A slim interpreter with native modules.
pub struct Runtime {
    modules: Vec<slim_vm::native::Module>,
    /// Report objects still alive at exit (`--rc-audit`).
    pub rc_audit: bool,
    /// Directories searched for extension libraries, after the program's own.
    pub search_path: Vec<PathBuf>,
}

impl Default for Runtime {
    fn default() -> Runtime {
        let search_path = std::env::var_os("SLIM_PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
        Runtime { modules: Vec::new(), rc_audit: false, search_path }
    }
}

/// A compiled program and the values of the native items it may use.
struct Program {
    code: Arc<Code>,
    globals: Vec<String>,
    natives: Vec<(String, slim_vm::Value)>,
}

use slim_vm::Code;

impl Runtime {
    pub fn new() -> Runtime {
        Runtime::default()
    }

    /// Adds a native module that programs can `import`.
    pub fn module(&mut self, name: &str, build: impl FnOnce(&mut Module)) -> &mut Self {
        slim_ext::install(slim_vm::ext::api()).expect("slim-ext is connected to another interpreter");
        let raw = Module::build(name, build);
        // SAFETY: `raw` was just built through this interpreter's API.
        let m = unsafe { slim_vm::ext::module_from_raw(raw) };
        self.modules.retain(|x| x.name != name);
        self.modules.push(m);
        self
    }

    /// Parses and compiles `src`, loading the extension modules it imports.
    fn compile(&self, src: &str, filename: &str) -> Result<Program, String> {
        let ast = slim_syntax::parse(src).map_err(|e| e.render(filename, src))?;
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(d) = Path::new(filename).parent() {
            dirs.push(if d.as_os_str().is_empty() { PathBuf::from(".") } else { d.to_path_buf() });
        }
        dirs.extend(self.search_path.iter().cloned());
        let mut loaded = Vec::new();
        for name in slim_compiler::imported_modules(&ast) {
            if self.modules.iter().any(|m| m.name == name) {
                continue;
            }
            if let Some(m) = load::extension(&name, &dirs)? {
                loaded.push(m);
            }
        }
        let modules: Vec<&slim_vm::native::Module> = self.modules.iter().chain(&loaded).collect();
        let specs: Vec<_> = modules.iter().map(|m| m.spec()).collect();
        let (code, globals) = slim_compiler::compile_module_with(&ast, filename, &specs).map_err(|e| e.render(filename, src))?;
        let natives = modules.iter().flat_map(|m| m.globals()).collect();
        Ok(Program { code, globals, natives })
    }

    /// Compiles `src` and returns its disassembly.
    pub fn disassemble(&self, src: &str, filename: &str) -> Result<String, String> {
        Ok(self.compile(src, filename)?.code.disassemble())
    }

    /// Runs a program on the current thread (which needs a large stack for
    /// deep recursion). Returns a rendered diagnostic or traceback on error.
    pub fn run_source(&self, src: &str, filename: &str) -> Result<(), String> {
        let p = self.compile(src, filename)?;
        let vm = slim_vm::new_vm_with(p.globals, &p.natives);
        let mut th = slim_vm::Thread::new(vm);
        let result = slim_vm::run_module(&mut th, &p.code);
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
