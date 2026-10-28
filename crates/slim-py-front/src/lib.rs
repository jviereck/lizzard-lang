//! slim's front end for typed Python: definition files (`.pyi` overlays
//! over a package's own declarations), the type checker and lowering of
//! slim functions to the IR in `ir`.

pub mod check;
pub mod db;
pub mod ir;

pub use db::Env;

pub struct Compiled {
    pub module: ir::ModuleIr,
    /// `path:line:col: error: message`, in source order.
    pub diagnostics: Vec<String>,
    pub warnings: Vec<String>,
}

/// Check the slim module `name` at `path` and lower its functions.
pub fn compile_module(env: Env, name: &str, path: &str, source: &str) -> Compiled {
    let mut db = db::Db::new(env);
    let (m, body) = match db.add_user_module(name, path, source) {
        Ok(x) => x,
        Err(e) => return Compiled { module: ir::ModuleIr::default(), diagnostics: vec![e], warnings: db.warnings },
    };
    let functions = check::check_module(&mut db, m, &body);
    let mut diags = std::mem::take(&mut db.diags);
    diags.sort_by_key(|(f, off, _)| (*f, *off));
    diags.dedup();
    let diagnostics = diags
        .into_iter()
        .map(|(f, off, msg)| {
            let file = &db.files[f];
            let (l, c) = file.line_col(off);
            format!("{}:{l}:{c}: error: {msg}", file.path)
        })
        .collect();
    Compiled { module: ir::ModuleIr { functions }, diagnostics, warnings: db.warnings }
}
