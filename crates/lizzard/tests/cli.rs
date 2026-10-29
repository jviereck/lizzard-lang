//! The `lizzard` command: options, program arguments and exit status.

use std::process::Command;

#[test]
fn main_args_and_exit_status() {
    let dir = std::env::temp_dir().join(format!("lizzard-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("args.lz");
    std::fs::write(&prog, "def main(args: list<str>) -> int:\n    print(args)\n    len(args)\n").unwrap();
    // Options come before the file; everything after it is the program's.
    let out = Command::new(env!("CARGO_BIN_EXE_lizzard")).arg("--rc-audit").arg(&prog).args(["a", "b c", "--rc-audit"]).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "['a', 'b c', '--rc-audit']\n");
    assert_eq!(String::from_utf8_lossy(&out.stderr), "rc-audit: ok\n");
    assert_eq!(out.status.code(), Some(3));
    // `run` is optional.
    let out = Command::new(env!("CARGO_BIN_EXE_lizzard")).arg("run").arg(&prog).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[]\n");
    assert_eq!(out.status.code(), Some(0));
    // An unknown option before the file is a usage error.
    let out = Command::new(env!("CARGO_BIN_EXE_lizzard")).arg("--nope").arg(&prog).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}
