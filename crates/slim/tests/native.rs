//! Rust bindings: native functions and classes called from slim.
//!
//! One test function runs the scenarios in order, so the live-object count
//! (process-wide) can be checked after each program.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use slim::{ArrayRef, Ctx, DictRef, Func, ListRef, Native, Runtime, SlimError};

type Log = Arc<Mutex<Vec<String>>>;

/// A runtime with the test modules; `log(s)` appends to the returned log.
fn runtime() -> (Runtime, Log) {
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new();
    let l = log.clone();
    rt.module("util", move |m| {
        let l = l.clone();
        m.function("log", move |s: String| l.lock().unwrap().push(s));
        m.function("add", |a: i64, b: i64| a + b);
        m.function("half", |x: f64| x / 2.0);
        m.function("small", |x: u8| x as i64 * 2);
        m.function("shout", |s: String| s.to_uppercase());
        m.function("maybe", |x: Option<i64>| x.map(|v| v + 1));
        m.function("squares", |n: i64| (0..n).map(|i| i * i).collect::<Vec<i64>>());
        m.function("total", |xs: Vec<f64>| xs.iter().sum::<f64>());
        m.function("scale", |xs: ListRef<f64>, k: f64| -> Result<(), SlimError> {
            for i in 0..xs.len() {
                let v = xs.get(i).unwrap();
                xs.set(i, v * k)?;
            }
            Ok(())
        });
        m.function("tally", |d: DictRef<String, i64>, key: String| -> Result<i64, SlimError> {
            let n = d.get(key.clone())?.unwrap_or(0) + 1;
            d.insert(key, n)?;
            Ok(n)
        });
        m.function("fill", |a: ArrayRef<f32>, v: f32| {
            for i in 0..a.len() {
                a.set(i, v).unwrap();
            }
        });
        m.function("checked_div", |a: i64, b: i64| -> Result<i64, SlimError> {
            if b == 0 {
                return Err(SlimError::new("ZeroDivisionError", "checked_div by zero"));
            }
            Ok(a / b)
        });
        m.function("boom", || -> i64 { panic!("kaboom") });
        m.function("apply", |ctx: &mut Ctx, f: Func, x: i64| -> Result<i64, SlimError> { ctx.call(&f, (x,)) });
        m.function("work_offline", |ctx: &mut Ctx, n: i64| -> i64 { ctx.offline(move || (0..n).sum()) });
    });
    rt.module("geom", |m| {
        m.class::<Point>("Point", |c| {
            c.init(|x: f64, y: f64| Point { x, y });
            c.method("norm2", |p: &Point| p.x * p.x + p.y * p.y);
            c.method("x", |p: &Point| p.x);
            c.method_mut("shift", |p: &mut Point, dx: f64| p.x += dx);
            c.repr(|p: &Point| format!("Point({}, {})", p.x, p.y));
        });
        m.function("origin", || Native::new(Point { x: 0.0, y: 0.0 }));
        m.function("dist2", |a: Native<Point>, b: Native<Point>| -> Result<f64, SlimError> {
            let (ax, ay) = a.with(|p| (p.x, p.y))?;
            let (bx, by) = b.with(|p| (p.x, p.y))?;
            Ok((ax - bx).powi(2) + (ay - by).powi(2))
        });
        // Send + Sync: shared calls from many threads.
        m.class::<Hits>("Hits", |c| {
            c.init(|| Hits(AtomicI64::new(0)));
            c.method("hit", |h: &Hits| h.0.fetch_add(1, Ordering::Relaxed) + 1);
            c.method("count", |h: &Hits| h.0.load(Ordering::Relaxed));
        });
        // Send only: every call is exclusive.
        m.class_send::<Acc>("Acc", |c| {
            c.init(|| Acc(0));
            c.method_mut("add", |a: &mut Acc, n: i64| {
                a.0 += n;
                a.0
            });
            c.method_mut("add_via", |ctx: &mut Ctx, a: &mut Acc, f: Func| -> Result<i64, SlimError> {
                // Calling back into slim while holding `&mut Acc`.
                let n: i64 = ctx.call(&f, ())?;
                a.0 += n;
                Ok(a.0)
            });
            c.method("get", |a: &Acc| a.0);
        });
        // Send only, waiting instead of raising on conflicts.
        m.class_send::<Acc2>("Locked", |c| {
            c.init(|| Acc2(0));
            c.method_mut("add", |a: &mut Acc2, n: i64| {
                let v = a.0;
                std::thread::yield_now();
                a.0 = v + n;
                a.0
            });
            c.method("get", |a: &Acc2| a.0);
            c.locked();
        });
        // Not Send: local to its thread.
        m.class_not_send::<Cell>("Cell", |c| {
            c.init(|v: i64| Cell(std::rc::Rc::new(std::cell::Cell::new(v))));
            c.method("get", |c: &Cell| c.0.get());
            c.method("set", |c: &Cell, v: i64| c.0.set(v));
        });
    });
    (rt, log)
}

struct Point {
    x: f64,
    y: f64,
}
struct Hits(AtomicI64);
struct Acc(i64);
struct Acc2(i64);
struct Cell(std::rc::Rc<std::cell::Cell<i64>>);

/// Runs `src`; returns the log, or the error text.
fn run(src: &str) -> Result<Vec<String>, String> {
    let (rt, log) = runtime();
    let before = slim_vm::rc::live_objects();
    let r = rt.run_source(src, "test.slim");
    let after = slim_vm::rc::live_objects();
    assert_eq!(before, after, "objects leaked by:\n{src}");
    r.map(|_| log.lock().unwrap().clone())
}

fn ok(src: &str) -> Vec<String> {
    match run(src) {
        Ok(l) => l,
        Err(e) => panic!("program failed:\n{e}\n---\n{src}"),
    }
}

fn err(src: &str) -> String {
    match run(src) {
        Ok(_) => panic!("program should fail:\n{src}"),
        Err(e) => e,
    }
}

#[test]
fn native_bindings() {
    // Freed objects are poisoned: a use after free panics.
    std::env::set_var("SLIM_POISON", "1");
    // Scalars, strings, options, conversions.
    let l = ok(r#"
from util import log, add, half, small, shout, maybe, squares, total
log(str(add(2, 3)))
log(str(half(5)))
log(shout("hi"))
log(str(maybe(None)) + " " + str(maybe(4)))
xs = squares(4)
xs.append(16)
log(str(xs))
log(str(total([1.5, 2.5])))
log(str(small(100)))
"#);
    assert_eq!(l, ["5", "2.5", "HI", "None 5", "[0, 1, 4, 9, 16]", "4.0", "200"]);

    // `import m` and qualified names.
    let l = ok("import util\nutil.log(str(util.add(1, 1)))\n");
    assert_eq!(l, ["2"]);

    // Static argument checks use the Rust signature.
    let e = err("from util import add\nadd(1, \"x\")\n");
    assert!(e.contains("argument `#2` expects `int`, got `str`"), "{e}");
    let e = err("from util import scale\nxs = [1, 2]\nscale(xs, 2.0)\n");
    assert!(e.contains("expects `list<float>`, got `list<int>`"), "{e}");
    let e = err("import nope\n");
    assert!(e.contains("no module named `nope`"), "{e}");

    // Range checks raise OverflowError; the runtime check covers `any`.
    let e = err("from util import small\ndef u(x: any) -> any:\n    x++\nsmall(u(300))\n");
    assert!(e.contains("OverflowError: argument 1 of small(): 300 does not fit in u8"), "{e}");

    // Typed handles read and write in place.
    let l = ok(r#"
from util import log, scale, tally, fill
xs = [1.0, 2.0]
scale(xs, 3.0)
log(str(xs))
d: dict<str, int> = {}
tally(d, "a")
tally(d, "a")
log(str(d))
a = array(f32, 3)
fill(a, 0.5)
log(str(a))
"#);
    assert_eq!(l, ["[3.0, 6.0]", "{'a': 2}", "array(f32, [0.5, 0.5, 0.5])"]);

    // `Result` errors and panics become exceptions.
    let l = ok(r#"
from util import log, checked_div, boom
try:
    checked_div(1, 0)
except ZeroDivisionError as e:
    log("caught " + str(e))
try:
    boom()
except RuntimeError as e:
    log(str(e))
"#);
    assert_eq!(l, ["caught checked_div by zero", "native panic in boom(): kaboom"]);

    // Callbacks; exceptions raised by them pass through the native unchanged.
    let l = ok(r#"
from util import log, apply, work_offline
log(str(apply(|x: int| x * 10, 4)))
def bad(x: int) -> int:
    raise ValueError("from slim")
try:
    apply(bad, 1)
except ValueError as e:
    log("callback: " + str(e))
log(str(work_offline(5)))
"#);
    assert_eq!(l, ["40", "callback: from slim", "10"]);

    // Classes: construction, methods, `Native<T>` arguments and results.
    let l = ok(r#"
from util import log
from geom import Point, origin, dist2
p = Point(3.0, 4.0)
log(str(p.norm2()))
p.shift(1.0)
log(str(p))
log(str(dist2(p, origin())))
def far(q: Point) -> bool:
    q.norm2() > 10.0
log(str(far(p)))
log(str(isinstance(p, Point)))
"#);
    assert_eq!(l, ["25.0", "Point(4, 4)", "32.0", "True", "True"]);
    let e = err("from geom import Point\np = Point(1.0, 2.0)\np.nope()\n");
    assert!(e.contains("has no method `nope`"), "{e}");

    // Send + Sync: shared calls from many threads.
    let l = ok(r#"
from util import log
from geom import Hits
def work(h: Hits, n: int) -> ():
    for i in range(n):
        h.hit()
h = Hits()
ts: list<Thread> = []
for i in range(4):
    ts.append(spawn(work, h++, 500))
for t in ts:
    t.join()
log(str(h.count()))
"#);
    assert_eq!(l, ["2000"]);

    // Send only: a re-entrant call conflicts with the running `&mut` call.
    let l = ok(r#"
from util import log
from geom import Acc
a = Acc()
log(str(a.add(2)))
log(str(a.add_via(|| 5)))
try:
    a.add_via(|| a.get())
except BorrowError as e:
    log("BorrowError: " + str(e))
log(str(a.get()))
"#);
    assert_eq!(l, ["2", "7", "BorrowError: `Acc` object is in use by a call that needs it exclusively", "7"]);

    // A locked class makes conflicting calls from other threads wait.
    let l = ok(r#"
from util import log
from geom import Locked
def work(a: Locked) -> ():
    for i in range(300):
        a.add(1)
a = Locked()
ts: list<Thread> = []
for i in range(4):
    ts.append(spawn(work, a++))
for t in ts:
    t.join()
log(str(a.get()))
"#);
    assert_eq!(l, ["1200"]);

    // Not Send: a binding over a non-`Send` Rust type is a non-`Send` slim
    // class, so instances never leave their thread. `Any`/`Func` handles are
    // `any`/`function`, which may hold values that are not `Send`.
    let l = ok(r#"
from util import log, apply
from geom import Cell
def main() -> ():
    c = Cell(1)
    c.set(c.get() + 1)
    log(str(c.get()))
    cells: list<Cell> = [Cell(5)]
    log(str(cells[0].get()))
    log(str(apply(|x: int| c.get() + x, 10)))
main()
"#);
    assert_eq!(l, ["2", "5", "12"]);
    let e = err("from geom import Cell\ndef show(x: Cell) -> int:\n    x.get()\ndef main() -> ():\n    spawn(show, Cell(1))\nmain()\n");
    assert!(e.contains("spawn(): argument 1 is not `Send` (`geom.Cell`) and cannot be sent to another thread"), "{e}");
    let e = err("from geom import Cell\nshared = Cell(1)\n");
    assert!(e.contains("cannot store `geom.Cell`, which is not `Send`, in module variable `shared`"), "{e}");
}
