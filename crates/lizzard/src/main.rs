use std::process::ExitCode;

const USAGE: &str = "usage: lizzard [run] [--dis] [--rc-audit] <file.py> [args...]

Runs the program's `main`; `main(args: list<str>)` receives the arguments after
the file, and `main() -> int` sets the exit status.
Extension modules (`import geom`) are loaded from libgeom.so / libgeom.dylib /
geom.dll in the program's directory or a directory on LIZZARD_PATH.";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "run") {
        args.remove(0);
    }
    let mut file = None;
    let mut dis = false;
    let mut audit = std::env::var("LIZZARD_RC_AUDIT").is_ok();
    let mut it = args.into_iter();
    // Options come before the file; everything after it is the program's.
    for a in it.by_ref() {
        match a.as_str() {
            "--dis" => dis = true,
            "--rc-audit" => audit = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            f if !f.starts_with('-') => {
                file = Some(f.to_string());
                break;
            }
            other => {
                eprintln!("unknown option `{other}`\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(file) = file else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let prog_args: Vec<String> = it.collect();
    // Run on a thread with a large stack: each Lizzard call recurses on the
    // native stack.
    let h = std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(move || run(&file, prog_args, dis, audit))
        .expect("failed to spawn interpreter thread");
    h.join().unwrap_or(ExitCode::from(101))
}

fn run(file: &str, args: Vec<String>, dis: bool, audit: bool) -> ExitCode {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("lizzard: cannot read {file}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut rt = lizzard::Runtime::new();
    rt.rc_audit = audit;
    rt.args = args;
    if dis {
        return match rt.disassemble(&src, file) {
            Ok(d) => {
                print!("{d}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(1)
            }
        };
    }
    let result = rt.run_source(&src, file);
    match result {
        // The low 8 bits, as the OS keeps them.
        Ok(status) => ExitCode::from(status as u8),
        Err(tb) => {
            eprintln!("{tb}");
            ExitCode::from(1)
        }
    }
}
