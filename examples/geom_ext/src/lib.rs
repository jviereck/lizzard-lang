//! The `geom` module: Rust functions, classes and a trait for Lizzard.
//!
//! Built two ways from this one definition:
//! - an extension library (`libgeom.so` / `libgeom.dylib`), which the
//!   `lizzard` interpreter loads when a program does `import geom`;
//! - a Rust library, which a host program links and registers statically
//!   (`examples/embed_geom`: `rt.module("geom", geom::build)`).

use std::sync::atomic::{AtomicI64, Ordering};

use lizzard_ext::{ClassBuilder, FromLizzard, IntoLizzard, ListRef, Module, LizzardError};

/// Shared between threads: methods take `&self`.
struct Counter(AtomicI64);

/// A plain Rust value: `&mut self` methods get exclusive access.
struct Polygon {
    points: Vec<(f64, f64)>,
}

impl Polygon {
    fn area(&self) -> f64 {
        let n = self.points.len();
        let twice: f64 = (0..n)
            .map(|i| {
                let (x1, y1) = self.points[i];
                let (x2, y2) = self.points[(i + 1) % n];
                x1 * y2 - x2 * y1
            })
            .sum();
        twice.abs() / 2.0
    }
}

/// A generic Rust type: Lizzard sees `Stack<T>` for the registered `T`s.
struct Stack<T> {
    items: Vec<T>,
}

fn stack<T: FromLizzard + IntoLizzard + Send + Sync + 'static>(c: &mut ClassBuilder<Stack<T>>) {
    c.init(|| Stack { items: Vec::new() });
    c.method_mut("push", |s: &mut Stack<T>, x: T| s.items.push(x));
    c.method_mut("pop", |s: &mut Stack<T>| s.items.pop());
    c.method("len", |s: &Stack<T>| s.items.len() as i64);
}

/// Defines the module's items.
pub fn build(m: &mut Module) {
    m.function("dot", |a: ListRef<f64>, b: ListRef<f64>| -> Result<f64, LizzardError> {
        if a.len() != b.len() {
            return Err(LizzardError::value_error(format!("dot(): lengths {} and {} differ", a.len(), b.len())));
        }
        Ok(a.to_vec().iter().zip(b.to_vec()).map(|(x, y)| x * y).sum())
    });
    m.class::<Counter>("Counter", |c| {
        c.init(|start: i64| Counter(AtomicI64::new(start)));
        c.method("inc", |c: &Counter| c.0.fetch_add(1, Ordering::Relaxed) + 1);
        c.method("get", |c: &Counter| c.0.load(Ordering::Relaxed));
    });
    // A trait for Rust and Lizzard classes alike.
    m.trait_def("Shape", |t| {
        t.method::<(), f64>("area");
    });
    m.class_send::<Polygon>("Polygon", |c| {
        c.init(|| Polygon { points: Vec::new() });
        c.method_mut("add", |p: &mut Polygon, x: f64, y: f64| p.points.push((x, y)));
        c.method("area", |p: &Polygon| p.area());
        c.method("__len__", |p: &Polygon| p.points.len() as i64);
        c.repr(|p: &Polygon| format!("Polygon({} points)", p.points.len()));
        c.implements("Shape");
        c.implements("Len");
    });
    m.generic_class("Stack", &["T"], |g| {
        g.class::<i64, _>(stack::<i64>);
        g.class::<String, _>(stack::<String>);
    });
}

lizzard_ext::module!(geom, build);
