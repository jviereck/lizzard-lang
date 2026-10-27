//! Rust bindings: native functions and classes called from slim.
//!
//! One test function runs the scenarios in order, so the live-object count
//! (process-wide) can be checked after each program.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use slim::{ArrayRef, ClassBuilder, Ctx, DictRef, FromSlim, Func, IntoSlim, ListRef, Native, Runtime, SlimError};

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
        // Generic: one Rust class per supported type argument. Handles
        // (`Native<Point>`) are not `Send`, so neither is `Stack<Point>`.
        m.generic_class("Stack", &["T"], |g| {
            g.class::<i64, _>(stack::<i64>);
            g.class::<String, _>(stack::<String>);
            g.class_not_send::<Native<Point>, _>(stack::<Native<Point>>);
        });
        // A Rust trait, implemented by a Rust class (with builtin `Len` and
        // `Ord` through dunders) and by slim classes.
        m.trait_def("Shape", |t| {
            t.method::<(), f64>("area");
            t.method::<(f64,), ()>("scale");
        });
        m.class::<Poly>("Polygon", |c| {
            c.init(|| Poly { points: Vec::new() });
            c.method_mut("add", |p: &mut Poly, x: f64, y: f64| p.points.push((x, y)));
            c.method("area", |p: &Poly| p.area());
            c.method_mut("scale", |p: &mut Poly, k: f64| p.points.iter_mut().for_each(|q| *q = (q.0 * k, q.1 * k)));
            c.method("__len__", |p: &Poly| p.points.len() as i64);
            c.method("__lt__", |p: &Poly, o: Native<Poly>| -> Result<bool, SlimError> { Ok(p.area() < o.with(|q| q.area())?) });
            c.implements("Shape");
            c.implements("Len");
            c.implements("Ord");
            c.repr(|p: &Poly| format!("Polygon({})", p.area()));
        });
        m.function("countdown", |n: i64| -> Result<Native<Stack<i64>>, SlimError> { Native::new(Stack { items: (1..=n).rev().collect() }) });
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
struct Stack<T> {
    items: Vec<T>,
}
struct Poly {
    points: Vec<(f64, f64)>,
}

impl Poly {
    fn area(&self) -> f64 {
        let n = self.points.len();
        let twice: f64 = (0..n).map(|i| self.points[i].0 * self.points[(i + 1) % n].1 - self.points[(i + 1) % n].0 * self.points[i].1).sum();
        twice.abs() / 2.0
    }
}

fn stack<T: FromSlim + IntoSlim + Clone + 'static>(c: &mut ClassBuilder<Stack<T>>) {
    c.init(|| Stack { items: Vec::new() });
    c.method_mut("push", |s: &mut Stack<T>, x: T| s.items.push(x));
    c.method_mut("pop", |s: &mut Stack<T>| s.items.pop());
    c.method("peek", |s: &Stack<T>| s.items.last().cloned());
    c.method("len", |s: &Stack<T>| s.items.len() as i64);
    c.method("__len__", |s: &Stack<T>| s.items.len() as i64);
    c.implements("Len");
    c.locked();
}

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


def main() -> ():
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
    let l = ok("import util\ndef main() -> ():\n    util.log(str(util.add(1, 1)))\n");
    assert_eq!(l, ["2"]);

    // Static argument checks use the Rust signature.
    let e = err("from util import add\ndef main() -> ():\n    add(1, \"x\")\n");
    assert!(e.contains("argument `#2` expects `int`, got `str`"), "{e}");
    let e = err("from util import scale\ndef main() -> ():\n    xs = [1, 2]\n    scale(xs, 2.0)\n");
    assert!(e.contains("expects `list<float>`, got `list<int>`"), "{e}");
    let e = err("import nope\ndef main() -> ():\n    pass\n");
    assert!(e.contains("no module named `nope`"), "{e}");

    // Range checks raise OverflowError; the runtime check covers `any`.
    let e = err("from util import small\ndef u(x: any) -> any:\n    x++\ndef main() -> ():\n    small(u(300))\n");
    assert!(e.contains("OverflowError: argument 1 of small(): 300 does not fit in u8"), "{e}");

    // Typed handles read and write in place.
    let l = ok(r#"
from util import log, scale, tally, fill


def main() -> ():
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


def main() -> ():
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
def bad(x: int) -> int:
    raise ValueError("from slim")


def main() -> ():
    log(str(apply(|x: int| x * 10, 4)))
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
def far(q: Point) -> bool:
    q.norm2() > 10.0


def main() -> ():
    p = Point(3.0, 4.0)
    log(str(p.norm2()))
    p.shift(1.0)
    log(str(p))
    log(str(dist2(p, origin())))
    log(str(far(p)))
    log(str(isinstance(p, Point)))
"#);
    assert_eq!(l, ["25.0", "Point(4, 4)", "32.0", "True", "True"]);
    let e = err("from geom import Point\ndef main() -> ():\n    p = Point(1.0, 2.0)\n    p.nope()\n");
    assert!(e.contains("has no method `nope`"), "{e}");

    // Send + Sync: shared calls from many threads.
    let l = ok(r#"
from util import log
from geom import Hits
def work(h: Hits, n: int) -> ():
    for i in range(n):
        h.hit()


def main() -> ():
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


def main() -> ():
    a = Acc()
    log(str(a.add(2)))
    log(str(a.add_via(|| 5)))
    b = a++
    try:
        a.add_via(|| b.get())
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


def main() -> ():
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
"#);
    assert_eq!(l, ["2", "5", "12"]);
    let e = err("from geom import Cell\ndef show(x: Cell) -> int:\n    x.get()\ndef main() -> ():\n    spawn(show, Cell(1))\n");
    assert!(e.contains("spawn(): argument 1 is not `Send` (`geom.Cell`) and cannot be sent to another thread"), "{e}");

    // Generic classes: each instantiation is its own Rust class.
    let l = ok(r#"
from util import log
from geom import Stack, Point, countdown
def main() -> ():
    s = Stack::<int>()
    s.push(1)
    s.push(2)
    log(str(s.len()) + " " + str(s.pop()) + " " + str(s.peek()))
    t: Stack<str> = Stack()
    t.push("a")
    log(t.pop().some_or("none") + " " + str(t.pop()))
    p = Stack::<Point>()
    p.push(Point(1.0, 2.0))
    log(str(p.peek()))
    log(str(isinstance(s, Stack)) + " " + str(isinstance(Point(0.0, 0.0), Stack)))
    c = countdown(3)
    log(str(c.pop()) + " " + str(c.len()))
    log(str(s))
"#);
    assert_eq!(l, ["2 2 1", "a None", "Point(1, 2)", "True False", "1 2", "<Stack<int> object>"]);

    // Type arguments only known at runtime pick the instantiation then.
    let l = ok(r#"
from util import log
from geom import Stack
def make<T>(x: T + keep) -> Stack<T>:
    s = Stack::<T>()
    s.push(x)
    s
def main() -> ():
    log(str(make(7).len()))
    log(str(make("x").peek()))
    try:
        make(1.5)
    except TypeError as e:
        log(str(e))
"#);
    assert_eq!(l, ["1", "x", "`Stack<float>` is not available: `Stack` supports `Stack<int>`, `Stack<str>`, `Stack<Point>`"]);

    // Statically known: checked at compile time.
    let e = err("from geom import Stack\ndef main() -> ():\n    s = Stack::<float>()\n");
    assert!(e.contains("`geom.Stack<float>` is not available: `geom.Stack` supports `geom.Stack<int>`, `geom.Stack<str>`, `geom.Stack<geom.Point>`"), "{e}");
    let e = err("from geom import Stack\ndef f(s: Stack<bool>) -> ():\n    s.len()\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`geom.Stack<bool>` is not available"), "{e}");
    let e = err("from geom import Stack\ndef main() -> ():\n    s = Stack::<int>()\n    s.push(\"x\")\n");
    assert!(e.contains("argument `#1` expects `int`, got `str`"), "{e}");
    let e = err("from geom import Stack\ndef main() -> ():\n    s = Stack()\n");
    assert!(e.contains("cannot infer the type arguments of `geom.Stack`"), "{e}");
    let e = err("from geom import Stack\ndef main() -> ():\n    s = Stack::<int>()\n    s.nope()\n");
    assert!(e.contains("has no method `nope`"), "{e}");

    // `Send` follows the instantiation: `Stack<int>` crosses threads,
    // `Stack<Point>` does not.
    let l = ok(r#"
from util import log
from geom import Stack
def work(s: Stack<int>, n: int) -> ():
    for i in range(n):
        s.push(i)
def main() -> ():
    s = Stack::<int>()
    ts: list<Thread> = []
    for i in range(4):
        ts.append(spawn(work, s++, 100))
    for t in ts:
        t.join()
    log(str(s.len()))
"#);
    assert_eq!(l, ["400"]);
    let e = err("from geom import Stack, Point\ndef show(s: Stack<Point>) -> int:\n    s.len()\ndef main() -> ():\n    spawn(show, Stack::<Point>())\n");
    assert!(e.contains("is not `Send` (`geom.Stack<geom.Point>`)"), "{e}");
    // Native traits: a Rust trait implemented by a Rust class and a struct;
    // a program's trait implemented for a Rust class; builtin traits.
    let l = ok(r#"
from util import log
from geom import Polygon, Shape, Stack
struct Square:
    side: float

impl Shape for Square:
    def area(self) -> float:
        self.side * self.side
    def scale(self, k: float) -> ():
        self.side = self.side * k

trait Describe:
    def area(self) -> float: ...
    def describe(self) -> str:
        "area " + str(self.area())

impl Describe for Polygon

def tri(k: float) -> Polygon:
    p = Polygon()
    p.add(0.0, 0.0)
    p.add(4.0 * k, 0.0)
    p.add(0.0, 3.0 * k)
    p

def total<T: Shape>(xs: list<T>) -> float:
    t = 0.0
    for x in xs:
        t = t + x.area()
    t

def size<T: Len>(x: T) -> int:
    len(x)

def main() -> ():
    shapes: list<Shape> = [tri(1.0), Square { side: 2.0 }]
    for s in shapes:
        s.scale(2.0)
        log(str(s.area()))
    log(str(total([tri(1.0), tri(2.0)])))
    log(tri(1.0).describe())
    d: Describe = tri(2.0)
    log(d.describe())
    log(str(len(tri(1.0))) + " " + str(tri(1.0) < tri(2.0)) + " " + str(tri(2.0) < tri(1.0)))
    log(str(sorted([tri(3.0), tri(1.0), tri(2.0)])))
    log(str(isinstance(tri(1.0), Shape)) + " " + str(isinstance(tri(1.0), Describe)))
    st = Stack::<int>()
    st.push(5)
    log(str(size(st)) + " " + str(size(tri(1.0))))
"#);
    assert_eq!(
        l,
        ["24.0", "16.0", "30.0", "area 6.0", "area 24.0", "3 True False", "[Polygon(6), Polygon(24), Polygon(54)]", "True True", "1 3"]
    );

    // A bound checked at runtime (the type argument is only known then).
    let l = ok(r#"
from util import log
from geom import Polygon, Shape
def area_of<T: Shape>(x: T) -> float:
    x.area()
def untyped(x):
    x++
def main() -> ():
    log(str(area_of(untyped(Polygon()))))
    try:
        area_of(untyped(5))
    except TypeError as e:
        log("TypeError")
"#);
    assert_eq!(l, ["0.0", "TypeError"]);

    let e = err("from geom import Shape\nstruct Sq\nimpl Shape for Sq:\n    def area(self) -> float:\n        1.0\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`impl geom.Shape for Sq`: `Sq` does not define required method `scale`"), "{e}");
    let e = err("from geom import Point\ntrait D:\n    def area(self) -> float: ...\nimpl D for Point\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`impl D for geom.Point`: `geom.Point` does not define required method `area`"), "{e}");
    let e = err("from geom import Point\ntrait D:\n    def x(self) -> int: ...\nimpl D for Point\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`geom.Point.x` is `() -> float`, but `D` needs `() -> int`"), "{e}");
    let e = err("from geom import Polygon\nimpl Polygon:\n    def f(self) -> int:\n        1\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`geom.Polygon` is a native class: its methods are defined in Rust"), "{e}");
    let e = err("from geom import Shape, Point\nimpl Shape for Point\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`geom.Shape` is a native trait"), "{e}");
    let e = err("from geom import Polygon\ntrait D:\n    def area(self) -> float: ...\ndef f() -> ():\n    impl D for Polygon\ndef main() -> ():\n    pass\n");
    assert!(e.contains("must be at module level"), "{e}");
    // Items are order-independent: an impl may come before its trait.
    let l = ok("from util import log\nfrom geom import Polygon\nimpl D for Polygon\ntrait D:\n    def area(self) -> float: ...\n    def d(self) -> str:\n        str(self.area())\ndef main() -> ():\n    log(Polygon().d())\n");
    assert_eq!(l, ["0.0"]);
    // A program's traits for native classes, with bodies; for every
    // instantiation of a generic one, or a single one.
    let l = ok(r#"
from util import log
from geom import Polygon, Stack
trait Describe:
    def area(self) -> float: ...
    def name(self) -> str:
        "a shape"
    def describe(self) -> str:
        self.name() + " of area " + str(self.area())
impl Describe for Polygon:
    def name(self) -> str:
        "polygon"
trait Count:
    def count(self) -> int: ...
impl<T> Count for Stack<T>:
    def count(self) -> int:
        self.len()
trait Top:
    def top(self) -> str: ...
impl Top for Stack<str>:
    def top(self) -> str:
        self.peek().some_or("empty")
def total<T: Count>(xs: list<T>) -> int:
    n = 0
    for x in xs:
        n = n + x.count()
    n
def main() -> ():
    p = Polygon()
    p.add(0.0, 0.0)
    p.add(4.0, 0.0)
    p.add(0.0, 3.0)
    log(p.describe() + ", " + str(Describe::area(p)))
    s = Stack::<int>()
    s.push(1)
    s.push(2)
    t = Stack::<str>()
    log(str(s.count()) + " " + str(Count::count(t)) + " " + t.top() + " " + str(total([s++, s++])))
    t.push("x")
    log(t.top() + " " + str(isinstance(s, Count)) + " " + str(isinstance(s, Top)) + " " + str(isinstance(t, Top)))
"#);
    assert_eq!(l, ["polygon of area 6.0, 6.0", "2 0 empty 4", "x True False True"]);
    let e = err("from geom import Stack\ntrait C:\n    def c(self) -> int: ...\nimpl<T> C for Stack<T>:\n    def c(self) -> int:\n        1\nimpl C for Stack<int>:\n    def c(self) -> int:\n        2\ndef main() -> ():\n    pass\n");
    assert!(e.contains("conflicting implementations of `C`: `impl C for geom.Stack<int>` overlaps `impl C for geom.Stack<T>`"), "{e}");
    let e = err("from geom import Stack\ntrait C:\n    def c(self) -> int: ...\nimpl C for Stack<float>:\n    def c(self) -> int:\n        1\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`geom.Stack<float>` is not available"), "{e}");
    let e = err("from geom import Polygon\nimpl Len for Polygon\ndef main() -> ():\n    pass\n");
    assert!(e.contains("`impl Len for geom.Polygon`: both are foreign"), "{e}");

    let e = err("from geom import Stack, Point\ndef untyped(x):\n    x++\ndef main() -> ():\n    xs: list<any + Send> = []\n    xs.append(untyped(Stack::<Point>()))\n");
    assert!(e.contains("cannot store `Stack<Point>`, which is not `Send`, in `list<any + Send>`"), "{e}");
}
