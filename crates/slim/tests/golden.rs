//! Golden tests: every `tests/programs/*.slim` is run and its combined
//! stdout+stderr compared with the matching `.out` file.
//! Run with `BLESS=1` to (re)generate the expected output.

use std::path::{Path, PathBuf};
use std::process::Command;

fn programs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/programs")
}

fn run_program(name: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_slim"))
        .arg("--rc-audit")
        .arg(name)
        .env("SLIM_POISON", "1")
        .current_dir(programs_dir())
        .output()
        .expect("failed to run slim");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s.push_str(&format!("[exit {}]\n", out.status.code().unwrap_or(-1)));
    s
}

#[test]
fn golden_programs() {
    let bless = std::env::var("BLESS").is_ok();
    let mut entries: Vec<_> = std::fs::read_dir(programs_dir())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "slim"))
        .collect();
    entries.sort();
    let mut failures = Vec::new();
    for p in &entries {
        let name = p.file_name().unwrap().to_str().unwrap();
        let actual = run_program(name);
        let expected_path = p.with_extension("out");
        if bless {
            std::fs::write(&expected_path, &actual).unwrap();
            continue;
        }
        let expected = std::fs::read_to_string(&expected_path).unwrap_or_default();
        if actual != expected {
            failures.push(format!("--- {name}\nexpected:\n{expected}\nactual:\n{actual}"));
        }
    }
    assert!(failures.is_empty(), "{} golden test(s) failed:\n{}", failures.len(), failures.join("\n"));
    assert!(!entries.is_empty(), "no test programs found");
}
