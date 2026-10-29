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
    let d = check("def g() -> None:\n    with g():\n        pass\n\ndef f(x: int | None) -> None:\n    pass\n");
    assert_eq!(d, ["this statement is not supported by Lizzard yet", "union types are not supported by Lizzard yet"]);
}

#[test]
fn class_attributes_are_declared_and_typed() {
    let d = check(
        "class P:\n    x: float\n\n    def __init__(self, x: float, tag: str) -> None:\n        self.x = x\n        self.tag: str = tag\n\n    def bad(self) -> None:\n        self.x = \"s\"\n        self.y = 1\n        self.tag = 2\n\ndef f(p: P) -> str:\n    n: float = p.x\n    return p.missing\n",
    );
    assert_eq!(
        d,
        [
            "cannot assign str to attribute `x` of type float",
            "P has no attribute `y`; declare it in the class body (`y: T`) or in `__init__` (`self.y: T = ...`)",
            "cannot assign int to attribute `tag` of type str",
            "P has no attribute `missing`",
        ]
    );
}

#[test]
fn constructors_check_init_arguments() {
    let d = check(
        "class P:\n    def __init__(self, x: int, y: int = 0) -> None:\n        self.x: int = x\n\nclass E:\n    pass\n\ndef f() -> None:\n    a = P(1)\n    b = P(1, 2)\n    c = P(\"s\")\n    e = E(1)\n    n: str = a.x\n",
    );
    assert_eq!(d, ["argument `x` of P(): expected int, got str", "E() takes no arguments", "cannot assign int to `n` of type str"]);
}

#[test]
fn subclasses_are_assignable_to_bases_not_the_reverse() {
    let d = check(
        "class A:\n    def f(self) -> int:\n        return 1\n\nclass B(A):\n    def g(self) -> int:\n        return self.f() + 1\n\ndef h(a: A, b: B) -> None:\n    x: A = b\n    y: B = a\n    a.g()\n    xs: list[A] = [b, b]\n",
    );
    assert_eq!(d, ["cannot assign A to `y` of type B", "A has no attribute `g`"]);
}

#[test]
fn overrides_must_be_compatible() {
    let d = check(
        "class A:\n    n: int = 0\n    def f(self, x: int) -> float:\n        return 1.0\n    def g(self) -> int:\n        return 1\n\nclass B(A):\n    n: str = \"\"\n    def f(self, x: float) -> int:\n        return 1\n    def g(self) -> str:\n        return \"\"\n\nclass C(A):\n    def f(self, x: bool) -> float:\n        return 1.0\n",
    );
    assert_eq!(
        d,
        [
            "`B.n` is declared as str, but `A.n` is int",
            "`B.g` returns str, but `A.g` returns int: an override must return a compatible type",
            "parameter `x` of `C.f` is bool, but `A.f` accepts int: an override must accept it",
        ]
    );
}

#[test]
fn super_finds_base_members() {
    let d = check(
        "class A:\n    def __init__(self, n: int) -> None:\n        self.n: int = n\n    def f(self) -> int:\n        return self.n\n\nclass B(A):\n    def __init__(self) -> None:\n        super().__init__(\"s\")\n    def f(self) -> int:\n        return super().f() + super().g()\n",
    );
    assert_eq!(d, ["argument `n` of __init__(): expected int, got str", "no base class of `B` has an attribute `g`"]);
}

#[test]
fn generic_classes_keep_their_type_parameters() {
    let d = check(
        "class Box[T]:\n    def __init__(self, item: T) -> None:\n        self.item: T = item\n    def get(self) -> T:\n        return self.item\n    def wrong(self) -> T:\n        return 1\n\ndef f() -> None:\n    b = Box(1)\n    s: str = b.get()\n    t: str = Box[str](\"x\").get()\n    u: int = b.item\n",
    );
    assert_eq!(d, ["`Box.wrong` returns T, but this returns int", "cannot assign int to `s` of type str"]);
}

#[test]
fn isinstance_narrows_a_local() {
    let d = check(
        "class A:\n    pass\n\nclass B(A):\n    def g(self) -> int:\n        return 1\n\ndef f(a: A) -> int:\n    if isinstance(a, B):\n        return a.g()\n    return a.g()\n",
    );
    assert_eq!(d, ["A has no attribute `g`"]);
}

#[test]
fn lists_dicts_and_tuples_are_typed() {
    let d = check(
        "def f() -> None:\n    xs = [1, 2]\n    xs.append(\"s\")\n    ys: list[float] = [1, 2]\n    zs: list[float] = xs\n    e = []\n    d = {\"a\": 1}\n    n: str = d[\"a\"]\n    d[1] = 2\n    t = (1, \"a\")\n    s: int = t[1]\n    u = t[2]\n    a, b, c = t\n    for k, v in d.items():\n        w: int = k\n",
    );
    assert_eq!(
        d,
        [
            "argument `object` of append(): expected int, got str",
            "cannot assign list[int] to `zs` of type list[float]",
            "cannot infer the type of an empty list; annotate it (`x: list[int] = []`)",
            "cannot assign int to `n` of type str",
            "argument `key` of __setitem__(): expected str, got int",
            "cannot assign str to `s` of type int",
            "index 2 is out of range for tuple[int, str]",
            "cannot unpack a tuple of 2 items into 3 targets",
            "cannot assign str to `w` of type int",
        ]
    );
}

#[test]
fn container_literals_take_the_expected_type() {
    // An annotation, a parameter or an attribute types the literal.
    let d = check(
        "class C:\n    def __init__(self) -> None:\n        self.xs: list[float] = []\n        self.m: dict[str, list[int]] = {}\n\ndef g(xs: list[float]) -> None:\n    pass\n\ndef f(c: C) -> None:\n    g([1, 2])\n    c.xs = [3]\n    c.m[\"a\"] = []\n    c.m.get(\"a\", []).append(1)\n    pairs: list[tuple[str, float]] = [(\"a\", 1)]\n",
    );
    assert!(d.is_empty(), "{d:?}");
}

#[test]
fn iteration_comprehensions_and_membership() {
    let d = check(
        "def f(xs: list[int], d: dict[str, int]) -> None:\n    for x in xs:\n        s: str = x\n    sq = [x * x for x in xs if x > 1]\n    t: list[str] = sq\n    inv = {v: k for k, v in d.items()}\n    u: dict[str, str] = inv\n    b: bool = 1 in xs\n    c = 1 in \"abc\"\n    for i in 5:\n        pass\n",
    );
    assert_eq!(
        d,
        [
            "cannot assign int to `s` of type str",
            "cannot assign list[int] to `t` of type list[str]",
            "cannot assign dict[int, str] to `u` of type dict[str, str]",
            "argument `value` of __contains__(): expected str, got int",
            "int is not iterable",
        ]
    );
}
