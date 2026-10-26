//! Registration errors of native modules panic. (A separate test binary:
//! `native.rs` counts live objects process-wide.)

use slim::Runtime;

struct Point {
    x: f64,
}

#[test]
#[should_panic(expected = "native class `Bad` implements `Shape`, but `area` takes")]
fn native_trait_mismatch_panics() {
    let mut rt = Runtime::new();
    rt.module("bad", |m| {
        m.trait_def("Shape", |t| {
            t.method::<(), f64>("area");
        });
        m.class::<Point>("Bad", |c| {
            c.method("area", |p: &Point| p.x as i64);
            c.implements("Shape");
        });
    });
}
