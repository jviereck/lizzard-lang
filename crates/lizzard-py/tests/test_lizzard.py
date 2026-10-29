"""End-to-end tests of Lizzard's typed-Python slice: the NumPy example, the
float32 fast path, runtime checks and diagnostics.

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
                while a:
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
