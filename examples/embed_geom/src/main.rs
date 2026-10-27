//! Embedding slim: the native `geom` module (`examples/geom_ext`), linked
//! into this program and registered statically, used from a slim script.
//! The same script runs with the `slim` interpreter, which loads the module
//! as an extension library instead.

use slim::Runtime;

fn main() {
    let mut rt = Runtime::new();
    rt.module("geom", geom::build);
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../geom_ext/geom.slim");
    if let Err(e) = rt.run_file(script) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
