//! The `slim` command: options, program arguments and exit status.

use std::process::Command;

#[test]
fn main_args_and_exit_status() {
    let dir = std::env::temp_dir().join(format!("slim-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("args.slim");
    std::fs::write(&prog, "def main(args: list<str>) -> int:\n    print(args)\n    len(args)\n").unwrap();
    // Options come before the file; everything after it is the program's.
    let out = Command::new(env!("CARGO_BIN_EXE_slim")).arg("--rc-audit").arg(&prog).args(["a", "b c", "--rc-audit"]).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "['a', 'b c', '--rc-audit']\n");
    assert_eq!(String::from_utf8_lossy(&out.stderr), "rc-audit: ok\n");
    assert_eq!(out.status.code(), Some(3));
    // `run` is optional.
    let out = Command::new(env!("CARGO_BIN_EXE_slim")).arg("run").arg(&prog).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[]\n");
    assert_eq!(out.status.code(), Some(0));
    // An unknown option before the file is a usage error.
    let out = Command::new(env!("CARGO_BIN_EXE_slim")).arg("--nope").arg(&prog).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}
