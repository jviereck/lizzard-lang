//! Embedding slim: a native `geom` module used from a slim script.

use std::sync::atomic::{AtomicI64, Ordering};

use slim::{ListRef, Runtime, SlimError};

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

fn main() {
    let mut rt = Runtime::new();
    rt.module("geom", |m| {
        m.function("dot", |a: ListRef<f64>, b: ListRef<f64>| -> Result<f64, SlimError> {
            if a.len() != b.len() {
                return Err(SlimError::value_error(format!("dot(): lengths {} and {} differ", a.len(), b.len())));
            }
            Ok(a.to_vec().iter().zip(b.to_vec()).map(|(x, y)| x * y).sum())
        });
        m.class::<Counter>("Counter", |c| {
            c.init(|start: i64| Counter(AtomicI64::new(start)));
            c.method("inc", |c: &Counter| c.0.fetch_add(1, Ordering::Relaxed) + 1);
            c.method("get", |c: &Counter| c.0.load(Ordering::Relaxed));
        });
        m.class_send::<Polygon>("Polygon", |c| {
            c.init(|| Polygon { points: Vec::new() });
            c.method_mut("add", |p: &mut Polygon, x: f64, y: f64| p.points.push((x, y)));
            c.method("area", |p: &Polygon| p.area());
            c.repr(|p: &Polygon| format!("Polygon({} points)", p.points.len()));
        });
    });
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/geom.slim");
    if let Err(e) = rt.run_file(script) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
