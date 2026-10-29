//! Extension modules: the `lizzard` interpreter loads `examples/geom_ext` (a
//! `cdylib`) at `import geom`, through the C ABI only.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Builds the extension (with the profile of this test) and returns the
/// directory holding the library.
fn build_extension() -> PathBuf {
    let dir = Path::new(env!("CARGO_BIN_EXE_lizzard")).parent().unwrap().to_path_buf();
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["build", "-q", "-p", "geom_ext"]).current_dir(root());
    if dir.file_name().is_some_and(|n| n == "release") {
        cmd.arg("--release");
    }
    assert!(cmd.status().expect("failed to run cargo").success(), "building geom_ext failed");
    dir
}

fn library(name: &str) -> String {
    format!("{}{name}{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX)
}

/// Runs `script` with `lizzard`; returns stdout+stderr and the exit code.
fn lizzard(script: &Path, lizzard_path: Option<&Path>) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_lizzard"));
    cmd.arg("--rc-audit").arg(script).env("LIZZARD_POISON", "1").env_remove("LIZZARD_PATH");
    if let Some(p) = lizzard_path {
        cmd.env("LIZZARD_PATH", p);
    }
    let out = cmd.output().expect("failed to run lizzard");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s.push_str(&format!("[exit {}]\n", out.status.code().unwrap_or(-1)));
    s
}

/// A fresh scratch directory.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lizzard-ext-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn extension_modules() {
    let lib_dir = build_extension();
    let script = root().join("examples/geom_ext/geom.py");

    // Found on LIZZARD_PATH; the same output as the statically embedded module.
    let expected = std::fs::read_to_string(root().join("examples/geom_ext/geom.out")).unwrap();
    assert_eq!(lizzard(&script, Some(&lib_dir)), expected);

    // Found next to the program.
    let d = scratch("local");
    std::fs::copy(lib_dir.join(library("geom")), d.join(library("geom"))).unwrap();
    std::fs::copy(&script, d.join("geom.py")).unwrap();
    assert_eq!(lizzard(&d.join("geom.py"), None), expected);

    // Not found: the compiler's error.
    let out = lizzard(&script, None);
    assert!(out.contains("NameError: no module named `geom`"), "{out}");

    // A library that is not an extension, and one for another module.
    let d = scratch("bad");
    std::fs::write(d.join(library("junk")), b"not a library").unwrap();
    std::fs::write(d.join("junk.py"), "import junk\n").unwrap();
    let out = lizzard(&d.join("junk.py"), None);
    assert!(out.contains("ImportError: cannot load module `junk`") && out.ends_with("[exit 1]\n"), "{out}");
    std::fs::copy(lib_dir.join(library("geom")), d.join(library("shapes"))).unwrap();
    std::fs::write(d.join("shapes.py"), "import shapes\n").unwrap();
    let out = lizzard(&d.join("shapes.py"), None);
    assert!(out.contains("cannot load module `shapes`") && out.contains("it defines module `geom`"), "{out}");
}

#[test]
fn abi_version_mismatch() {
    // SAFETY: a bitwise copy of the (plain-data) function table.
    let mut api: lizzard_abi::LizzardApi = unsafe { std::ptr::read(lizzard_vm::ext::api()) };
    api.version = lizzard_abi::ABI_VERSION + 1;
    let api: &'static lizzard_abi::LizzardApi = Box::leak(Box::new(api));
    let e = lizzard_ext::install(api).unwrap_err();
    assert!(e.contains(&format!("ABI version {}", lizzard_abi::ABI_VERSION + 1)), "{e}");
}
