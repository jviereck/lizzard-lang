//! Front-end tests against small fixture stubs (`fixtures/site` is the
//! package's own layer, `fixtures/overlay` Lizzard's overlay).

use std::path::PathBuf;

fn check_with(source: &str, version: Option<&'static str>) -> Vec<String> {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let env = lizzard_py_front::Env {
        overlay_roots: vec![fixtures.join("overlay")],
        search_roots: vec![fixtures.join("site")],
        version: (3, 14),
        platform: "darwin".into(),
        stdlib: ["sys"].iter().map(|s| s.to_string()).collect(),
        pkg_version: Box::new(move |_| version.map(String::from)),
    };
    let src = format!("# lizzard: module\nimport mini\n\n{source}");
    let c = lizzard_py_front::compile_module(env, "m", "m.py", &src);
    c.diagnostics.iter().map(|d| d.split(": error: ").nth(1).unwrap_or(d).to_string()).chain(c.warnings.iter().map(|w| format!("warning: {w}"))).collect()
}

fn check(source: &str) -> Vec<String> {
    check_with(source, Some("1.2.0"))
}

#[test]
fn covariant_and_invariant_parameters() {
    let d = check(
        "def f() -> None:\n    a: mini.Box[mini.scalar[mini.B64]] = mini.make_box()\n    b: mini.Cell[mini.scalar[mini.B64]] = mini.make_cell()\n",
    );
    assert_eq!(d, ["cannot assign Cell[f64] to `b` of type Cell[scalar[B64]]"]);
}

#[test]
fn class_with_builtin_base_is_a_float() {
    assert!(check("def f() -> float:\n    return mini.make_box().get()\n").is_empty());
}

#[test]
fn overlay_replaces_members_and_keeps_the_rest() {
    let d = check("def f() -> None:\n    a: str = mini.make_box().label()\n    b: str = mini.make_box().get()\n");
    assert_eq!(d, ["cannot assign int to `a` of type str", "cannot assign f64 to `b` of type str"]);
}

#[test]
fn aliases_display_by_name_and_compare_by_target() {
    let d = check("def f(x: mini.f32) -> mini.scalar[mini.B32]:\n    y: int = x\n    return x\n");
    assert_eq!(d, ["cannot assign f32 to `y` of type int"]);
}

#[test]
fn invariant_parameter_rejects_a_subclass_argument() {
    let d = check("def f(x: mini.scalar[mini.B32]) -> mini.scalar[mini.B64]:\n    return x\n");
    assert_eq!(d, ["`f` returns scalar[B64], but this returns f32"]);
}

#[test]
fn self_solves_to_the_receiver() {
    let d = check("def f() -> None:\n    b = mini.make_box()\n    c: int = b.same(b)\n");
    assert_eq!(d, ["cannot assign Box[f64] to `c` of type int"]);
}

#[test]
fn overloads_and_the_any_rule() {
    let d = check(
        "def f() -> None:\n    a: int = mini.make_box().pick(0)\n    b: int = mini.make_any().pick(0)\n",
    );
    // A precise receiver picks its overload; `Box[Any]` matches both
    // overloads with different results, so the call is `Any`.
    assert_eq!(d, ["cannot assign f64 to `a` of type int"]);
}

#[test]
fn version_conditions_are_evaluated() {
    assert_eq!(check("def f() -> int:\n    return mini.future()\n"), ["`f` returns int, but this returns str"]);
}

#[test]
fn overlay_with_wrong_version_is_ignored() {
    let d = check_with("def f() -> None:\n    a: str = mini.make_box().label()\n", Some("0.9"));
    assert_eq!(d.len(), 1);
    assert!(d[0].starts_with("warning: lizzard: ignoring") && d[0].contains("mini>=1.0"), "{d:?}");
}

#[test]
fn unsupported_constructs_are_reported() {
    let d = check("class C:\n    pass\n\ndef f(x: int | None) -> None:\n    pass\n");
    assert_eq!(d, ["classes are not supported by Lizzard yet", "union types are not supported by Lizzard yet"]);
}
