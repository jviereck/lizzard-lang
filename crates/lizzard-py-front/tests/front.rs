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
        stdlib: ["sys", "dataclasses"].iter().map(|s| s.to_string()).collect(),
        pkg_version: Box::new(move |_| version.map(String::from)),
    };
    let src = format!("# lizzard: module\nimport mini\nfrom dataclasses import dataclass, field\n\n{source}");
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
    let d = check("def g() -> None:\n    match 1:\n        case _:\n            pass\n\ndef h() -> None:\n    x = yield 1\n");
    assert_eq!(d, ["this statement is not supported by Lizzard yet", "this expression is not supported by Lizzard yet"]);
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

#[test]
fn optional_values_must_be_narrowed() {
    let d = check(
        "class N:\n    def __init__(self) -> None:\n        self.v: int = 1\n        self.next: N | None = None\n\ndef f(n: N, xs: list[int] | None, s: str | None) -> int:\n    a = n.next.v\n    b = len(xs)\n    c: int = s\n    if n.next is not None:\n        a = n.next.v\n    if xs is None:\n        return 0\n    if s:\n        a += len(s)\n    return a + len(xs)\n",
    );
    assert_eq!(
        d,
        [
            "cannot read attribute `v` of N | None: the value may be None (check `is not None` first)",
            "list[int] | None has no len()",
            "cannot assign str | None to `c` of type int",
        ]
    );
}

#[test]
fn narrowing_by_isinstance_and_or_and_loops() {
    let d = check(
        "class A:\n    def a(self) -> int:\n        return 1\n\nclass B:\n    def b(self) -> int:\n        return 2\n\ndef f(x: A | B | None, y: int | str) -> int:\n    if x is None or isinstance(x, A):\n        return 0\n    n: int = x.b()\n    if isinstance(y, str):\n        y = len(y)\n    m: int = y\n    z: int | None = None\n    while z is None or z < 3:\n        z = (z or 0) + 1\n    w: int = z\n    for i in range(3):\n        q: int = z\n        z = None\n    return n\n",
    );
    assert_eq!(d, ["cannot assign int | None to `q` of type int"]);
}

#[test]
fn unions_distribute_over_members() {
    let d = check(
        "class C:\n    def area(self) -> float:\n        return 1.0\n\nclass S:\n    def area(self) -> int:\n        return 1\n\ndef f(s: C | S, t: int | float, u: list[int] | list[str]) -> None:\n    a: float = s.area()\n    b: int = s.area()\n    c: float = t * 2\n    for x in u:\n        d: int = x\n    e = t if t else None\n    g: float = e\n",
    );
    assert_eq!(d, ["cannot assign float | int to `b` of type int", "cannot assign int | str to `d` of type int", "cannot assign int | float | None to `g` of type float"]);
}

#[test]
fn functions_read_and_write_module_globals() {
    let d = check(
        "count = 0\nnames: list[str] = []\nlimit: int\n\ndef f() -> str:\n    global count\n    count += 1\n    names.append(\"x\")\n    count = \"s\"\n    return names[0] + str(limit)\n\ndef g() -> int:\n    n = count\n    count = 2\n    return n\n\ndef h() -> None:\n    global missing\n",
    );
    assert_eq!(
        d,
        [
            "cannot assign str to the global `count` of type int",
            "local variable `count` is used before it is assigned",
            "`missing` must be assigned at the top level of the module to be used as a global",
        ]
    );
}

#[test]
fn exceptions_are_checked() {
    let d = check(
        "class NotExc:\n    pass\n\nclass Err(Exception):\n    def __init__(self, code: int) -> None:\n        self.code: int = code\n\ndef f() -> None:\n    raise\n\ndef g() -> int:\n    try:\n        return 1\n    except NotExc:\n        return 2\n    except (KeyError, Err) as e:\n        s: str = e\n        raise ValueError() from 5\n    except Err as e:\n        return e.code\n\ndef h() -> int:\n    try:\n        return 1\n    except ValueError:\n        pass\n\ndef k() -> None:\n    with 3:\n        pass\n",
    );
    assert_eq!(
        d,
        [
            "a bare `raise` is only allowed in an `except` block",
            "cannot catch NotExc: it does not derive from BaseException",
            "cannot assign KeyError | Err to `s` of type str",
            "an exception cause must derive from BaseException or be None, got int",
            "`h` may end without returning a value (declared to return int)",
            "int does not support the context manager protocol (`__enter__`/`__exit__`)",
        ]
    );
}

#[test]
fn sets_and_operators_are_typed() {
    let d = check(
        "def f(xs: list[int]) -> None:\n    s = {1, 2}\n    t: set[str] = {x for x in [\"a\"]}\n    u = s | t\n    s.add(\"x\")\n    a: int = 2 ** 3\n    b: int = 2 ** len(xs)\n    c: float = 2.0 ** 2\n    d: int = 5 & 3 | 1 << 2\n    e: bool = True ^ False\n    g = 1.5 | 2\n    h: int = ~5\n    k: bool = all([])\n",
    );
    assert_eq!(
        d,
        [
            "unsupported operand types for |: set[int] and set[str]",
            "argument `element` of add(): expected int, got str",
            "unsupported operand types for |: float and int",
        ]
    );
}

#[test]
fn variadic_and_keyword_only_parameters() {
    let d = check(
        "def f(a: int, *rest: str, flag: bool = False, **opts: float) -> None:\n    n: tuple[str, ...] = rest\n    m: dict[str, float] = opts\n    s: int = rest[0]\n\ndef g(x: int, *, y: int) -> None:\n    pass\n\ndef h(t: tuple[int, int], xs: list[int], d: dict[str, int]) -> None:\n    f(1, \"a\", \"b\", flag=True, w=1.5)\n    f(1, 2)\n    g(1, 2)\n    g(1)\n    g(*t)\n    g(*xs, **d)\n    g(1, **xs)\n",
    );
    assert_eq!(
        d,
        [
            "cannot assign str to `s` of type int",
            "argument `rest` of f(): expected str, got int",
            "g(): takes 1 positional arguments but 2 were given",
            "g(): missing keyword argument `y`",
            "g(): takes 1 positional arguments but 2 were given",
            "`**` needs a dict with str keys, got list[int]",
        ]
    );
}

#[test]
fn property_setters_check_assigned_values() {
    let d = check(
        "class P:\n    def __init__(self) -> None:\n        self._x: int = 0\n\n    @property\n    def x(self) -> int:\n        return self._x\n\n    @x.setter\n    def x(self, v: int) -> None:\n        self._x = v\n\n    @property\n    def y(self) -> int:\n        return 1\n\ndef f(p: P) -> None:\n    p.x = 3\n    p.x = \"s\"\n    p.y = 2\n    s: str = p.x\n",
    );
    assert_eq!(d, ["argument `v` of the setter of `x`: expected int, got str", "property `y` has no setter", "cannot assign int to `s` of type str"]);
}

#[test]
fn generic_functions_are_checked_in_their_body_and_at_calls() {
    let d = check(
        "from typing import TypeVar\nT = TypeVar(\"T\")\n\nclass Shape:\n    def area(self) -> float:\n        return 1.0\n\ndef first[U](xs: list[U]) -> U:\n    return xs[0]\n\ndef wrong[U](x: U) -> U:\n    return 1\n\ndef ident(x: T) -> T:\n    y: int = x\n    return x\n\ndef big[S: Shape](xs: list[S]) -> S:\n    a: float = xs[0].area()\n    return xs[0]\n\ndef f() -> None:\n    n: str = first([1])\n    s: Shape = big([Shape()])\n    big([1])\n",
    );
    assert_eq!(
        d,
        [
            "`wrong` returns U, but this returns int",
            "cannot assign T to `y` of type int",
            "cannot assign int to `n` of type str",
            "argument `xs` of big(): expected list[Shape], got list[int]",
        ]
    );
}

#[test]
fn closures_and_lambdas_are_typed() {
    let d = check(
        "from typing import Callable\n\ndef f(xs: list[str]) -> None:\n    count = 0\n\n    def g() -> None:\n        nonlocal missing\n\n    def bump(n: int) -> str:\n        nonlocal count\n        count += n\n        return count\n\n    k: Callable[[int], str] = lambda x: x + 1\n    h = lambda a: a\n    ok: Callable[[int], int] = lambda x, n=2: x * n\n    by_len = sorted(xs, key=lambda s: len(s))\n    m = max(xs, key=lambda s: s.upper())\n    wrong = sorted(xs, key=lambda s: s.nope)\n    run: Callable[[str], int] = len\n    run(5)\n",
    );
    assert_eq!(
        d,
        [
            "no binding for nonlocal `missing` found",
            "`f.<locals>.bump` returns str, but this returns int",
            "cannot assign Callable[[int], int] to `k` of type Callable[[int], str]",
            "cannot infer the parameter types of this lambda: use it where a `Callable[[...], ...]` is expected, or use a `def`",
            "str has no attribute `nope`",
            "argument 1: expected str, got int",
        ]
    );
}

#[test]
fn dataclasses_get_an_init_from_their_fields() {
    let d = check(
        "@dataclass\nclass P:\n    x: int\n    y: int = 0\n    tags: list[str] = field(default_factory=list)\n\n@dataclass\nclass Q(P):\n    z: str = \"\"\n\n@dataclass(frozen=True)\nclass F:\n    a: int\n\n@dataclass\nclass Bad:\n    a: int = 1\n    b: str\n\ndef f() -> None:\n    P(1)\n    P(1, 2, [\"a\"])\n    Q(1, z=\"s\")\n    P()\n    P(\"s\")\n    Q(1, 2, [], \"z\", 5)\n    s: str = P(1).x\n    F(1).a = 2\n",
    );
    assert_eq!(
        d,
        [
            "dataclass field `b` without a default cannot follow fields with defaults",
            "P(): missing argument `x`",
            "argument `x` of P(): expected int, got str",
            "Q(): takes 4 positional arguments but 5 were given",
            "cannot assign int to `s` of type str",
            "cannot assign to field `a` of frozen dataclass `F`",
        ]
    );
}
