//! `cargo run -p lizzard-py-front --example check -- FILE SITE_PACKAGES OVERLAY_ROOT`
//! Type-checks a Lizzard module and prints diagnostics and the lowered IR.
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src = std::fs::read_to_string(&args[1]).unwrap();
    let env = lizzard_py_front::Env {
        overlay_roots: vec![PathBuf::from(&args[3])],
        search_roots: vec![PathBuf::from(&args[2])],
        version: (3, 14),
        platform: "darwin".into(),
        stdlib: ["sys", "os", "abc", "collections", "types", "enum", "datetime", "ctypes", "warnings", "builtins", "array", "mmap", "decimal", "fractions", "numbers", "inspect", "contextlib", "functools", "re"].iter().map(|s| s.to_string()).collect(),
        pkg_version: Box::new(|_| Some("2.4.0".into())),
    };
    let c = lizzard_py_front::compile_module(env, "__main__", &args[1], &src);
    for w in &c.warnings {
        println!("warning: {w}");
    }
    for d in &c.diagnostics {
        println!("{d}");
    }
    if std::env::var("SHOW_IR").is_ok() {
        for f in &c.module.functions {
            println!("--- {} ({} checks)", f.name, f.checks.len());
            for (pc, i) in f.code.iter().enumerate() {
                println!("{pc:3} L{:<3} {i:?}", f.lines[pc]);
            }
            for ch in &f.checks {
                println!("    check {ch:?}");
            }
        }
    }
}
