//! The slim compiler: AST to register bytecode with ownership checking.

mod compiler;
mod scan;
mod ty;

pub use compiler::compile_module;

use std::sync::Arc;

use slim_vm::Code;

/// Parses and compiles source code. Returns the module code and global names,
/// or a rendered diagnostic.
pub fn compile_source(src: &str, filename: &str) -> Result<(Arc<Code>, Vec<String>), slim_syntax::Error> {
    let module = slim_syntax::parse(src)?;
    compile_module(&module, filename)
}
