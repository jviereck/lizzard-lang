//! The lizzard compiler: AST to register bytecode with ownership checking.

mod compiler;
mod imports;
mod peephole;
mod scan;
mod ty;

pub use compiler::{compile_module, compile_module_with};

use std::sync::Arc;

use lizzard_vm::Code;

/// Parses and compiles source code. Returns the module code and global names,
/// or a rendered diagnostic.
pub fn compile_source(src: &str, filename: &str) -> Result<(Arc<Code>, Vec<String>), lizzard_syntax::Error> {
    let module = lizzard_syntax::parse(src)?;
    compile_module(&module, filename)
}

/// Like `compile_source`, for a program that may import the native modules
/// `natives`.
pub fn compile_source_with(
    src: &str,
    filename: &str,
    natives: &[lizzard_vm::native::ModuleSpec],
) -> Result<(Arc<Code>, Vec<String>), lizzard_syntax::Error> {
    let module = lizzard_syntax::parse(src)?;
    compile_module_with(&module, filename, natives)
}

/// The modules a program imports (`import m`, `from m import x`), in order
/// of first import.
pub fn imported_modules(module: &lizzard_syntax::ast::Module) -> Vec<String> {
    use lizzard_syntax::ast::StmtKind;
    let mut out: Vec<String> = Vec::new();
    for s in &module.body.stmts {
        let (StmtKind::Import { module: m, .. } | StmtKind::FromImport { module: m, .. }) = &s.kind else { continue };
        if !out.contains(m) {
            out.push(m.clone());
        }
    }
    out
}
