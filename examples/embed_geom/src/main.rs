//! Embedding lizzard: the native `geom` module (`examples/geom_ext`), linked
//! into this program and registered statically, used from a lizzard script.
//! The same script runs with the `lizzard` interpreter, which loads the module
//! as an extension library instead.

use lizzard::Runtime;

fn main() {
    let mut rt = Runtime::new();
    rt.module("geom", geom::build);
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../geom_ext/geom.lz");
    match rt.run_file(script) {
        // `main`'s exit status (its low 8 bits, as the OS keeps them).
        Ok(status) => std::process::exit(status as u8 as i32),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
