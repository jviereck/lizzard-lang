use std::process::ExitCode;

const USAGE: &str = "usage: slim [run] <file.slim> [--dis] [--rc-audit]

Extension modules (`import geom`) are loaded from libgeom.so / libgeom.dylib /
geom.dll in the program's directory or a directory on SLIM_PATH.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut file = None;
    let mut dis = false;
    let mut audit = std::env::var("SLIM_RC_AUDIT").is_ok();
    for a in &args {
        match a.as_str() {
            "run" if file.is_none() => {}
            "--dis" => dis = true,
            "--rc-audit" => audit = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            f if file.is_none() && !f.starts_with("--") => file = Some(f.to_string()),
            other => {
                eprintln!("unknown argument `{other}`\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(file) = file else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    // Run on a thread with a large stack: each slim call recurses on the
    // native stack.
    let h = std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(move || run(&file, dis, audit))
        .expect("failed to spawn interpreter thread");
    h.join().unwrap_or(ExitCode::from(101))
}

fn run(file: &str, dis: bool, audit: bool) -> ExitCode {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("slim: cannot read {file}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut rt = slim::Runtime::new();
    rt.rc_audit = audit;
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
        Ok(()) => ExitCode::SUCCESS,
        Err(tb) => {
            eprintln!("{tb}");
            ExitCode::from(1)
        }
    }
}
