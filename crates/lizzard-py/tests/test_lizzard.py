"""End-to-end tests of Lizzard's typed-Python slice: the NumPy example, the
float32 fast path, classes, containers, runtime checks and diagnostics.

Every program runs in a fresh subprocess, under Lizzard and under plain CPython.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXAMPLE = ROOT / "examples" / "numpy_add" / "add.py"
VEC32 = HERE / "programs" / "vec32.py"
CLASSES = HERE / "programs" / "classes.py"
CONTAINERS = HERE / "programs" / "containers.py"
UNIONS = HERE / "programs" / "unions.py"
GLOBALS = HERE / "programs" / "globals.py"
EXCEPTIONS = HERE / "programs" / "exceptions.py"
SETS = HERE / "programs" / "sets_and_ops.py"
ARGUMENTS = HERE / "programs" / "arguments.py"
GENERICS = HERE / "programs" / "generics.py"
CLOSURES = HERE / "programs" / "closures.py"
DATACLASSES = HERE / "programs" / "dataclasses_demo.py"
GENERATORS = HERE / "programs" / "generators.py"
OPERATORS = HERE / "programs" / "operators.py"
PROPERTIES = HERE / "programs" / "properties.py"
THREADS = HERE / "programs" / "threads.py"
PROTOCOLS = HERE / "programs" / "protocols.py"
LITERALS = HERE / "programs" / "literals.py"
SLOTS = HERE / "programs" / "slots.py"
REBINDING = HERE / "programs" / "rebinding.py"
NUMBERS = HERE / "programs" / "arith_edges.py"
INSTANCE_ATTRS = HERE / "programs" / "instance_attrs.py"


def run(*args, cwd=None, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([sys.executable, *args], cwd=cwd, env=e, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


class Scratch:
    """A temporary directory to write modules into."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        return self

    def __exit__(self, *_):
        self.tmp.cleanup()

    def write(self, name, text):
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
        return p


def without_last_line(path):
    return "".join(path.read_text().splitlines(keepends=True)[:-1])


class Example(unittest.TestCase):
    def test_last_line_is_a_static_error(self):
        code, out, err = run("-m", "lizzard", str(EXAMPLE))
        self.assertEqual(code, 1, err)
        self.assertEqual(out, "")
        lines = EXAMPLE.read_text().splitlines()
        self.assertIn(f"add.py:{len(lines)}:5: error: argument `b` of add()", err)
        self.assertIn("dtype[float32]", err)
        self.assertIn("dtype[float64]", err)

    def test_plain_cpython_runs_it_all(self):
        code, out, err = run(str(EXAMPLE))
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "[0.5 1.5 2.5 3.5 4.5]\n[0.5 1.5 2.5 3.5 4.5]\n")

    def test_without_last_line_matches_cpython(self):
        with Scratch() as s:
            p = s.write("add_ok.py", without_last_line(EXAMPLE))
            lizzard = run("-m", "lizzard", str(p))
            plain = run(str(p))
        self.assertEqual(lizzard[0], 0, lizzard[2])
        self.assertEqual(lizzard[1], plain[1])
        self.assertEqual(lizzard[1], "[0.5 1.5 2.5 3.5 4.5]\n[0.5 1.5 2.5 3.5 4.5]\n")

    def test_entry_checks_from_plain_python(self):
        with Scratch() as s:
            s.write("add_ok.py", without_last_line(EXAMPLE))
            s.write("main.py", """\
                import numpy as np
                import lizzard
                import add_ok

                x = np.arange(5, dtype=np.float32)
                print(type(add_ok.add).__name__)
                for f in (add_ok.add, add_ok.add_loop):
                    try:
                        f(x, x.astype(np.float64))
                    except TypeError as e:
                        print("TypeError:", e)
                m = np.ones((2, 2), np.float32)
                try:
                    add_ok.add_loop(m, m)
                except ValueError as e:
                    print("ValueError:", e)
                print(add_ok.add(b=x, a=x))
                """)
            code, out, err = run("main.py", cwd=s.path)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "LizzardFunction",
            "TypeError: add() argument 'b': expected ndarray[tuple[Any, ...], dtype[float32]], got ndarray with dtype float64 and ndim 1",
            "TypeError: add_loop() argument 'b': expected ndarray[tuple[Any, ...], dtype[float32]], got ndarray with dtype float64 and ndim 1",
            "ValueError: expected two 1-D arrays of the same length",
            "[0. 2. 4. 6. 8.]",
        ])


class Float32Path(unittest.TestCase):
    def test_output_matches_cpython(self):
        lizzard = run("-m", "lizzard", str(VEC32))
        plain = run(str(VEC32))
        self.assertEqual(lizzard[0], 0, lizzard[2])
        self.assertEqual(lizzard[1], plain[1])

    def test_direct_and_generic_access(self):
        code, out, err = run("-c", textwrap.dedent("""\
            import numpy as np
            import lizzard
            import vec32

            def stats():
                return tuple(lizzard.stats().values())

            rng = np.random.default_rng(7)
            x = rng.standard_normal(1000).astype(np.float32)
            y = rng.standard_normal(1000).astype(np.float32)
            s0 = stats()
            assert np.array_equal(vec32.add_loop(x, y), x + y)
            s1 = stats()
            assert np.array_equal(vec32.add_loop(x[::2], y[::2]), x[::2] + y[::2])
            s2 = stats()
            swapped = x.astype(">f4")
            assert np.array_equal(vec32.add_loop(swapped, swapped), x + x)
            s3 = stats()
            for k in (0.1, 3.0, 1 / 3):
                assert np.array_equal(vec32.scale(x, k).view(np.uint32), (x * k).view(np.uint32))
            assert vec32.scale(x, np.float64(0.1)).dtype == np.float32
            print(s1[0] - s0[0], s1[1] - s0[1])
            print(s2[0] - s1[0], s2[1] - s1[1])
            print(s3[0] - s2[0], s3[1] - s2[1])
            print(repr(vec32.get(x, -1)) == repr(x[-1]))
            try:
                vec32.get(x, 1000)
            except IndexError as e:
                print("IndexError:", e)
            """), cwd=VEC32.parent)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "3000 0",        # contiguous: 2 loads + 1 store per element, all direct
            "1500 0",        # strided views are direct too
            "0 3000",        # byte-swapped arrays take the generic path
            "True",
            "IndexError: index 1000 is out of bounds for axis 0 with size 1000",
        ])


class Programs(unittest.TestCase):
    def test_output_matches_cpython(self):
        for prog in (CLASSES, CONTAINERS, UNIONS, GLOBALS, EXCEPTIONS, SETS, ARGUMENTS, GENERICS, CLOSURES, DATACLASSES, GENERATORS, OPERATORS, PROPERTIES, THREADS, PROTOCOLS, LITERALS, SLOTS, REBINDING, NUMBERS, INSTANCE_ATTRS):
            with self.subTest(prog.name):
                plain = run(str(prog))
                self.assertEqual(plain[0], 0, plain[2])
                # Compiled (the default) and interpreted.
                for jit in ("1", "0"):
                    lizzard = run("-m", "lizzard", str(prog), env={"LIZZARD_JIT": jit})
                    self.assertEqual(lizzard[0], 0, lizzard[2])
                    self.assertEqual(lizzard[1], plain[1], f"LIZZARD_JIT={jit}")

    def test_fast_paths_are_on(self):
        # The layouts the int/float fast paths and inline attribute access
        # read passed their start-up checks on this CPython.
        code, out, err = run("-c", "import lizzard._lizzard as L; s = L._stats(); print(s['fast_paths'], s['inline_values'])")
        self.assertEqual((code, out), (0, "True True\n"), err)

    def test_deep_recursion_raises(self):
        # Lizzard calls recurse on the C stack: CPython's stack check turns
        # runaway recursion into RecursionError, in both engines.
        with Scratch() as s:
            p = s.write("m.py", """\
                # lizzard: module
                def down(n: int) -> int:
                    return down(n + 1)

                if __name__ == "__main__":
                    try:
                        down(0)
                    except RecursionError:
                        print("RecursionError")
                """)
            for jit in ("1", "0"):
                code, out, err = run("-m", "lizzard", str(p), env={"LIZZARD_JIT": jit})
                self.assertEqual((code, out), (0, "RecursionError\n"), err)

    def test_methods_are_compiled(self):
        code, out, err = run("-c", textwrap.dedent("""\
            import lizzard
            import classes as c

            print(type(c.Shape.__dict__["area"]).__name__, c.Circle.area.__qualname__, c.Circle.area.__name__)
            print(type(c.Rect.__dict__["square"].__func__).__name__)
            print(type(c.Vec.__dict__["zero"].__func__).__name__)
            print(type(c.Vec.__dict__["norm2"].fget).__name__)
            print(type(c.Stack.__dict__["push"]).__name__, type(c.total_area).__name__)
            """), cwd=CLASSES.parent)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "LizzardFunction Circle.area area",
            "LizzardFunction",
            "LizzardFunction",
            "LizzardFunction",
            "LizzardFunction LizzardFunction",
        ])

    def test_checks_at_the_boundary(self):
        code, out, err = run("-c", textwrap.dedent("""\
            import lizzard
            import classes as c
            import containers as k

            def attempt(f):
                try:
                    print("ok:", f())
                except Exception as e:
                    print(type(e).__name__ + ":", str(e).split(": ", 1)[-1] if ".py:" in str(e) else e)

            attempt(lambda: c.Circle("x"))
            circle = c.Circle(1.0)
            circle.r = "oops"
            attempt(lambda: circle.area())
            attempt(lambda: c.total_area([c.Circle(1.0), 3]))
            attempt(lambda: k.divmod_pair("a", 1))
            attempt(lambda: c.Savings("ann", 0.1).withdraw(500.0))

            # Python subclasses of Lizzard classes: compiled code calls their
            # overrides and checks what they return.
            class Loud(c.Dog):
                def sound(self):
                    return "WOOF"

            class Broken(c.Dog):
                def sound(self):
                    return 42

            attempt(lambda: Loud("max").speak())
            attempt(lambda: Broken("bug").speak())
            """), cwd=CLASSES.parent)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "TypeError: Circle.__init__() argument 'r': expected float, got str",
            "TypeError: self.r should be float, got str",
            "TypeError: item of shapes should be Shape, got int",
            "TypeError: divmod_pair() argument 'a': expected int, got str",
            "InsufficientFunds: need 400.00 more",
            "ok: max says WOOF",
            "TypeError: self.sound(...) should be str, got int",
        ])


class Arguments(unittest.TestCase):
    def test_binding_errors_match_cpython(self):
        probe = textwrap.dedent("""\
            import arguments as a
            for call in ["a.kw_only(1, 2, 3)", "a.kw_only(1)", "a.pos_only(1, a=2)", "a.describe()",
                         "a.kw_only(1, offset=1, zz=2)", "a.pos_only(1, 2, 3)", "a.describe('a', sep='x', **{'sep': 'y'})",
                         "a.Logger('x').log(1)"]:
                try:
                    print(eval(call))
                except TypeError as e:
                    print(e)
            """)
        lizzard = run("-c", "import lizzard\n" + probe, cwd=ARGUMENTS.parent)
        plain = run("-c", probe, cwd=ARGUMENTS.parent)
        self.assertEqual(lizzard[0], 0, lizzard[2])
        lines, plain_lines = lizzard[1].splitlines(), plain[1].splitlines()
        # The same errors, except that Lizzard checks the type of `*msgs`.
        self.assertEqual(lines[:-1], plain_lines[:-1])
        self.assertEqual(lines[-1], "Logger.log() argument 'msgs[0]': expected str, got int")


class Closures(unittest.TestCase):
    def test_inner_functions_and_lambdas_are_compiled(self):
        code, out, err = run("-c", textwrap.dedent("""\
            import lizzard
            import closures as c

            inc = c.make_counter(0)
            add = c.compose(c.double, c.double)
            print(type(inc).__name__, inc.__qualname__, type(add).__name__, add.__qualname__)
            try:
                add("x")
            except TypeError as e:
                print(e)
            """), cwd=CLOSURES.parent)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "LizzardFunction make_counter.<locals>.inc LizzardFunction compose.<locals>.<lambda>",
            "compose.<locals>.<lambda>() argument 'x': expected int, got str",
        ])


class Generators(unittest.TestCase):
    DRIVER = textwrap.dedent("""\
        import collections.abc, sys, gc
        import genedge as m

        def show(label, f):
            try:
                print(label, "->", repr(f()))
            except BaseException as e:
                print(label, "->", type(e).__name__, e, "| cause:", type(e.__cause__).__name__ if e.__cause__ else None)

        g = m.simple()
        print(isinstance(g, collections.abc.Generator), isinstance(g, collections.abc.Iterator), iter(g) is g)
        show("send to new", lambda: g.send(5))
        show("next", lambda: next(g))
        show("send 10", lambda: g.send(10))
        show("send 20", lambda: g.send(20))
        show("after end", lambda: next(g))
        show("throw after end", lambda: g.throw(ValueError("late")))
        show("pep479", lambda: list(m.leaks_stop()))
        s = m.stubborn(); next(s)
        show("ignored exit", lambda: s.close())
        c = m.closer(); next(c)
        show("close value", lambda: c.close())
        show("close new", lambda: m.closer().close())
        h = m.in_handler()
        print(next(h), "| caller sees", sys.exception())
        print(next(h), "| caller sees", sys.exception())
        print(next(h))
        t = m.thrower(); next(t)
        show("throw type+val", lambda: t.throw(ValueError, "v"))
        show("next after throw", lambda: next(t))
        t2 = m.thrower(); next(t2)
        show("throw instance", lambda: t2.throw(ValueError("a", "b")))
        show("throw unstarted", lambda: m.thrower().throw(KeyError("k")))
        show("throw non-exception", lambda: m.thrower().throw(42))
        show("deep", lambda: list(m.deep()))
        show("self ref", lambda: next(m.G))
        b = m.bad_finally(); next(b)
        show("close bad finally", lambda: b.close())
        b2 = m.bad_finally(); next(b2)
        del b2
        gc.collect()
        print("done")
        """)

    def test_protocol_matches_cpython(self):
        # The same driver over the module compiled by Lizzard, and imported
        # by plain CPython.
        lizzard = run("-c", "import lizzard\n" + self.DRIVER, cwd=GENERATORS.parent)
        plain = run("-c", self.DRIVER, cwd=GENERATORS.parent)
        self.assertEqual(lizzard[0], 0, lizzard[2])
        self.assertEqual(lizzard[1], plain[1])
        self.assertIn("generator raised StopIteration", lizzard[1])
        # Unfinished generators are closed when freed, and errors there are
        # reported, as CPython does.
        for err in (lizzard[2], plain[2]):
            self.assertIn("DeprecationWarning: the (type, exc, tb) signature of throw() is deprecated", err)
            self.assertIn("RuntimeError: finally failed", err)
            self.assertIn("RuntimeError: generator ignored GeneratorExit", err)
        self.assertEqual(lizzard[2].count("Exception ignored while closing generator <Lizzard generator object"), 2)

    def test_generators_are_lizzard_generators(self):
        code, out, err = run("-c", textwrap.dedent("""\
            import lizzard, inspect
            import generators as g
            it = g.count(3)
            print(type(it).__name__, it.__qualname__, it.gi_running, list(it))
            print(type(g.Tree.__iter__).__name__)
            try:
                g.count("x")
            except TypeError as e:
                print(e)
            """), cwd=GENERATORS.parent)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "LizzardGenerator count False [0, 1, 2]",
            "LizzardFunction",
            "count() argument 'n': expected int, got str",
        ])


class Properties(unittest.TestCase):
    def test_accessors_are_compiled_and_checked(self):
        code, out, err = run("-c", textwrap.dedent("""\
            import lizzard, properties
            a = properties.Account("x", 1)
            for attr in ("balance", "label"):
                p = getattr(type(a), attr)
                print(attr, type(p.fget).__name__, type(p.fset).__name__, type(p.fdel).__name__)
            try:
                a.balance = "lots"
            except TypeError as e:
                print(e)
            """), cwd=PROPERTIES.parent)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "balance LizzardFunction LizzardFunction LizzardFunction",
            "label LizzardFunction NoneType NoneType",
            "Account.balance() argument 'value': expected float, got str",
        ])


class Checker(unittest.TestCase):
    def check(self, source):
        with Scratch() as s:
            p = s.write("m.py", "# lizzard: module\n" + textwrap.dedent(source))
            return run("-m", "lizzard", str(p))

    def test_errors(self):
        code, _, err = self.check("""\
            import numpy as np

            def f(a: int) -> int:
                return a + undefined

            def g(a: int) -> str:
                return a

            def h(a) -> None:
                match a:
                    case _:
                        pass

            f("x")
            """)
        self.assertEqual(code, 1)
        lines = [l.split(": error: ", 1)[1] for l in err.splitlines()]
        self.assertEqual(lines, [
            "name `undefined` is not defined",
            "`g` returns str, but this returns int",
            "parameter `a` needs a type annotation",
            "this statement is not supported by Lizzard yet",
            "argument `a` of f(): expected int, got str",
        ])

    def test_class_errors(self):
        code, _, err = self.check("""\
            class Animal:
                def __init__(self, name: str) -> None:
                    self.name: str = name

                def speak(self) -> str:
                    return self.name

            class Dog(Animal):
                def speak(self) -> int:
                    return 1

                def rename(self) -> None:
                    self.nick = "x"

            Animal(1)
            """)
        self.assertEqual(code, 1)
        lines = [l.split(": error: ", 1)[1] for l in err.splitlines()]
        self.assertEqual(lines, [
            "`Dog.speak` returns int, but `Animal.speak` returns str: an override must return a compatible type",
            "Dog has no attribute `nick`; declare it in the class body (`nick: T`) or in `__init__` (`self.nick: T = ...`)",
            "argument `name` of Animal(): expected str, got int",
        ])

    def test_missing_module_and_attribute(self):
        code, _, err = self.check("""\
            import numpy as np
            import no_such_module

            def f() -> None:
                np.no_such_function()
            """)
        self.assertEqual(code, 1)
        self.assertIn("cannot find module `no_such_module`", err)
        self.assertIn("module `numpy` has no attribute `no_such_function`", err)

    def test_strict_attributes(self):
        # Every attribute must be declared: in the class, a base class, or
        # the definitions of a standard-library base such as threading.Thread.
        code, _, err = self.check("""\
            import queue
            import threading

            class Worker(threading.Thread):
                def __init__(self, n: int) -> None:
                    super().__init__(name=f"w{n}", daemon=True)
                    self.total: int = 0

                def run(self) -> None:
                    self.total = 1

                def typo(self) -> None:
                    self.totl = 1

            class Jobs(queue.Queue):
                def size(self) -> int:
                    return self.maxsize

            def f() -> None:
                Worker(1).strat()
                Jobs()
            """)
        self.assertEqual(code, 1)
        lines = [l.split(": error: ", 1)[1] for l in err.splitlines()]
        self.assertEqual(lines, [
            "Worker has no attribute `totl`; declare it in the class body (`totl: T`) or in `__init__` (`self.totl: T = ...`)",
            "Jobs has no attribute `maxsize` (its base class `queue.Queue` has no type definitions)",
            "Worker has no attribute `strat`",
            "cannot check the arguments of Jobs() (its base class `queue.Queue` has no type definitions)",
        ])

    def test_reveal_type(self):
        code, _, err = self.check("""\
            from typing import Literal

            def f(m: Literal["r", "w"]) -> None:
                if m == "r":
                    reveal_type(m)
                reveal_type([m])
            """)
        self.assertEqual(code, 0, err)
        lines = [l.split(": note: ", 1)[1] for l in err.splitlines()]
        self.assertEqual(lines, ["revealed type is `Literal['r']`", "revealed type is `list[Literal['r', 'w']]`"])

    def test_wrong_declaration_raises_at_runtime(self):
        with Scratch() as s:
            s.write("fakepkg/__init__.py", "def make():\n    return 'not an int'\n")
            stub = s.write("stubs/fakepkg/__init__.pyi", "def make() -> int: ...\n")
            p = s.write("m.py", """\
                # lizzard: module
                import fakepkg

                def f() -> int:
                    return fakepkg.make()

                if __name__ == "__main__":
                    print(f())
                """)
            code, _, err = run("-m", "lizzard", str(p), cwd=s.path, env={"LIZZARD_STUBS_PATH": str(s.path / "stubs")})
        self.assertEqual(code, 1)
        self.assertIn(f"fakepkg.make(...) should be int (declared at {stub}:1), got str", err)
        self.assertIn("lizzard: ", err)


@unittest.skipIf(sys._is_gil_enabled(), "free-threaded build only")
class FreeThreading(unittest.TestCase):
    def test_import_keeps_gil_disabled(self):
        code, out, err = run("-c", "import lizzard, sys; print(sys._is_gil_enabled())")
        self.assertEqual((code, out), (0, "False\n"), err)


if __name__ == "__main__":
    unittest.main()
