# Lizzard as typed Python: design outline

## Context
Lizzard pivots from its own Rust-flavoured language to **typed Python**. A Lizzard module is a normal `.py` file that runs unchanged under CPython. Lizzard adds three things:
1. a **static type checker** that requires annotations where compilation needs them;
2. **type definition files** for the Python modules it imports, so code using NumPy, the stdlib or any Python library stays well typed;
3. a **Cranelift JIT** that compiles the module's functions into native code running inside CPython 3.14t (free-threaded). Values are real `PyObject*`s, and memory is CPython's normal reference counting.

Everything Rust-inspired is dropped:
- `struct`/`trait`/`impl`, `Send`, coherence and orphan rules;
- the ownership syntax (`keep`, `++`, `<-`, moves);
- `Option`/`Some`, turbofish and Rust lambdas;
- implicit returns and the special `def main`.

Classes, inheritance, module globals and the top level all behave exactly as in Python. Reference-count elision and deferred cleanup are possible later optimizations, with no new syntax.

## 1. Principles
1. **Valid, runnable Python.** A Lizzard module has the same behaviour under plain CPython; Lizzard only checks and compiles it.
2. **Annotations are checked assertions.** A value entering an annotated position from untyped or Python code is checked. A mismatch raises `TypeError`. This is Lizzard's existing rule. Plain CPython runs no checks, so a program that passes them behaves identically in both modes.
3. **Checks where compiled code relies on a type.** Compiled code relies on a type, by unboxing or reading an attribute at a fixed offset, only after a check that guarantees it: an exact-type pointer comparison, `isinstance`, or a guard on the type's version tag. Everything else is a normal CPython operation, so a wrong type is a Python exception, never a memory error.
4. **Python semantics throughout:** aliasing on assignment, CPython reference counting and cycle GC, inheritance, mutable classes and globals.

## 2. Modules
```python
# lizzard: module
import sys
import numpy as np
import numpy.typing as npt

SCALE: float = 2.0                       # module globals work as in Python
cache: dict[str, int] = {}

class Body:
    x: float
    v: float
    def __init__(self, x: float, v: float) -> None:
        self.x = x
        self.v = v
    def step(self, dt: float) -> None:
        self.x += self.v * dt

def energy(a: npt.NDArray[np.float64]) -> float:
    return float(np.dot(a, a))

if __name__ == "__main__":
    bodies = [Body(0.0, 1.0) for _ in range(1000)]   # top-level locals as usual
    for b in bodies:
        b.step(0.01)
    print(energy(np.ones(10)), sys.argv[1:])
```
- **Opt-in:** the pragma comment `# lizzard: module` in the first lines marks a module. Other modules stay plain Python.
- **Top level** is unrestricted Python. CPython's own interpreter runs it once at import, after Lizzard has checked it statically. Lizzard compiles every function, method and lambda the module defines.
- **Entry point:** the usual `if __name__ == "__main__":` block. Run with `python app.py`, or with `python -m lizzard app.py` to compile the entry module itself, since modules are otherwise compiled only when imported.

## 3. Typing rules
- **Required annotations:**
  - function parameters and return types;
  - class attributes, declared in the class body or annotated on their first assignment in `__init__`;
  - module globals that compiled code reads, unless the type can be inferred from a literal or a constructor.

  Local variable types are inferred.
- **Supported typing features:**
  - builtins (`int`, `float`, `bool`, `str`, `bytes`, `list[T]`, `dict[K, V]`, `set[T]`, `tuple[...]`);
  - `T | None` with narrowing (`is None`, `isinstance`, `match`);
  - `Callable`, `Any` (dynamic), `Final`, `ClassVar`;
  - PEP 695 generics and `TypeVar`;
  - `Protocol` and ABCs for interfaces;
  - `cast` and `assert isinstance(...)`.
- **Generics are kept for precise checking, not erased.**
  - **Statically,** the checker tracks full generic types (PEP 695 classes and functions, `list[T]`, `dict[K, V]`, ...).
  - **Generic class instances** built by compiled code record their type arguments in `__orig_class__`, the same attribute `Box[int](...)` sets under CPython. A `Box[int]` passed where `Box[str]` is expected then fails at the boundary, even when it comes through untyped Python code.
  - **Generic functions** receive their type arguments as hidden parameters (as Lizzard does today), so checks inside them know the concrete `T`.
  - **Builtin containers** can't carry type arguments. A `list[float]` coming from Python therefore has its elements checked when they are read, an exact-type comparison.
- **No extra scalar types.** Standard Python types only:
  - `float` is a C double, unboxed in compiled code.
  - `int` keeps Python's unbounded semantics: compiled code uses an `i64` fast path and falls back to a boxed `PyLong` on overflow.
- **NumPy arrays are typed with NumPy's own typing:**
  - `npt.NDArray[np.float64]`, or the shape-typed `np.ndarray[tuple[int, int], np.dtype[np.float64]]` when the number of dimensions is known.
  - The dtype (and number of dimensions, if declared) is checked once at a boundary.
  - Element access compiles to bounds-checked loads through the array's strides, unboxed to NumPy's scalar types (`np.float64`, `np.int32`, ...).
- **Rejected in Lizzard modules** with a clear error, unless written behind `Any`:
  - dynamic constructs the checker can't type, such as `exec`, `eval`, `globals()[...]` writes and `setattr` with a computed name;
  - unannotated `def`s.

## 4. Classes
Classes are ordinary Python classes: inheritance (including from plain Python and extension classes such as `nn.Module`), properties, dunders, class and static methods, and monkey-patching all work.

How the compiler keeps them fast:
- **Attribute and method access** uses inline caches guarded by the class's version tag (`tp_version_tag`), which CPython changes whenever a type or one of its bases is modified. On a cache miss, or when an attribute is a property, the access goes through the normal `PyObject_GetAttr` path.
- **`__slots__` classes, or classes marked `@final`,** let the compiler load attributes at a fixed offset after an exact-type check, and call methods directly.
- **Exceptions** are normal exception classes.

## 5. Type definition files for Python imports
**Format.** Plain `.pyi` syntax with standard typing, including `@final` and NumPy's array types. Nothing lizzard-specific.

**Lookup order when a Lizzard module imports `m`:**
1. **Lizzard definition files** on `LIZZARD_STUBS_PATH`, or in a `lizzard-stubs/m/` package. These are authoritative.
2. **The package's own `.pyi` files** (`py.typed`).
3. **Bundled typeshed stubs** for the stdlib.
4. **No definitions:** the module is `Any`. Its results become typed only when annotated, e.g. `x: float = m.f()`, which checks the value on assignment.

**Checks at the boundary:**
- Return values and attribute reads from Python are checked against the declared type:
  - exact types by pointer comparison;
  - open classes with `isinstance`, cached per call site;
  - container elements when they are read, because Python code or another thread may mutate the container at any time.
- A wrong definition raises `TypeError` and names the declaration, for example `numpy.dot declared -> float at lizzard-stubs/numpy/__init__.pyi:123, got ndarray`.
- Calls into Python are ordinary vectorcall calls.

**Standard stub features:**
- `@overload` is resolved statically, falling back to `Any` when ambiguous.
- `TypeVar` and generics are supported.
- `ParamSpec`, `TypedDict`, `Literal` and `Protocol` are checked at the boundary or treated as `Any`.

**Tooling:**
- `lizzard stubgen m` bootstraps a definition file from runtime introspection plus any existing `.pyi`.
- Lizzard modules need no definition files: they are fully annotated Python that pyright and mypy read directly.

## 6. Mapping from current Lizzard
| current Lizzard | becomes |
|---|---|
| `struct` + `impl`, `class C(impl T)` | a normal `class` (use `__slots__`/`@final` for speed) |
| `trait`, bounds, `impl Trait for X` | `Protocol` / ABC, PEP 695 bounds (`def f[T: Sized](x: T)`) |
| `Send`, `+ keep`, `x++`, `<-`, moves | removed; Python aliasing and CPython reference counting |
| `rc_run()` | removed; CPython frees objects immediately |
| `list<int>`, `f::<int>()` | `list[int]`; explicit type arguments are dropped (inferred or annotated) |
| `Option<T>`, `Some(x)` | `T \| None` with narrowing |
| `\|x\| expr` | `lambda x: expr`, or a nested `def` |
| implicit return, block `if`/`try` expressions | `return`, ternary, statements |
| `def main(args)` returning an exit code; items only at the top level | the `__name__ == "__main__"` guard, top-level code as in Python |
| reified generics | kept: PEP 695 generics, type arguments recorded in `__orig_class__` and passed to generic functions |
| `int` (i64, raises `OverflowError`), raw arrays | Python's unbounded `int`; NumPy arrays |
| native classes via `lizzard-ext` | ordinary PyO3/C extension modules with definition files |

## 7. Implementation architecture
- **Parser (`crates/lizzard-syntax`):**
  - Replace the lexer and parser with a full Python grammar. The recommended option is ruff's `ruff_python_parser`, pinned as a git dependency.
  - Add a lowering pass into a typed-Python AST. Remove the Rust-style constructs.
- **Checker (`crates/lizzard-compiler`):**
  - Rebuild around Python's type system: classes with inheritance and the MRO, `Protocol`, unions and narrowing, globals, `Any`.
  - Reuse the expression and type inference parts of `compiler.rs`, and the reported-error style of the error goldens.
  - Remove trait, impl, coherence, `Send` and ownership checking.
  - Add the definition-file loader (new `stubs.rs`).
- **New `crates/lizzard-jit`:**
  - Lowers checked functions to Cranelift with unboxed scalars and inline caches keyed on version tags.
  - Uses CPython-style reference counting, with inline fast paths for the free-threaded object header.
  - Calls the C API for dynamic operations.
  - Safepoints at loop back-edges check CPython's eval-breaker flag, for stop-the-world GC and signals.
  - Global loads go through the module dict, cached with a dict watcher (`PyDict_AddWatcher`) that invalidates on rebinding.
- **New `crates/lizzard-py`:** a PyO3 extension module (`_lizzard`) providing:
  - the `sys.meta_path` import hook that checks and compiles marked modules;
  - functions as vectorcall objects: a checked entry for Python callers, and a direct native entry for compiled-to-compiled calls;
  - traceback entries for compiled frames;
  - an on-disk cache of compiled code, keyed by source hash and CPython version.
- **New `python/lizzard/`:** a tiny pure-Python package for `python -m lizzard` and the `lizzard check` / `lizzard stubgen` commands. Lizzard modules don't need to import it.
- **Retired:**
  - **`crates/lizzard-vm`:** the interpreter, `rc.rs`, Lizzard's own dict/list/str, traits and generics registry.
  - **`lizzard-abi`/`lizzard-ext`:** Rust extensions become ordinary Python extension modules.

  Ideas to carry over: the inline-cache layout, unboxed float arithmetic, compare-and-jump fusion, and the benchmark suite.

## 8. Milestones
1. **Python front end and checker.**
   - The new parser, the typed-Python checker with generics, the `lizzard` package, and `lizzard check`.
   - Port `tests/programs/*.py` to valid typed Python; CPython itself is the oracle for their `.out` files.
2. **Definition files.** The loader, boundary-check semantics, bundled typeshed, `lizzard stubgen`, and initial NumPy definitions.
3. **Import hook and JIT for the core.**
   - Trampolines (entry points), unboxed int/float code, control flow, calls, globals, generic type arguments.
   - C API fallback for everything dynamic.
   - Goldens run compiled and must match CPython's output.
4. **Classes:** attribute and method inline caches keyed on version tags, `__slots__`/`@final` fast paths, compiled methods.
5. **NumPy kernels:** dtype and dimension checks at boundaries for `npt.NDArray[...]` and shape-typed arrays, and compiled bounds-checked loops through the array's strides.
6. **Threads:** safepoints, cooperation with stop-the-world GC, and a free-threaded stress test.
7. **Later, optional:** an analysis that removes redundant reference-count operations, and deferred decrefs for realtime use.

## Verification
- **Dual-run goldens:** every `tests/programs/*.py` runs (a) under plain `python3.14t` and (b) compiled by Lizzard. Both must match the `.out` file.
- **Checker errors:** error goldens for checker diagnostics (missing annotations, type errors, narrowing, definition mismatches), run through `lizzard check`.
- **Boundary tests:** deliberately wrong definition files and monkey-patched classes must raise or take the generic path, never crash. Run under a debug CPython build and with `PYTHONMALLOC=debug`.
- **Reference-count audit:** `sys.gettotalrefcount` on a debug build before and after each golden, to catch leaks and over-decrefs in generated code.
- **Benchmarks:** the existing bench programs, ported, against CPython 3.14t (and numba for NumPy kernels), using the same best-of-5 script.

## Status: slice 1 (interpreter, no JIT)
A first end-to-end slice runs the float32 NumPy example (`examples/numpy_add/add.py`) from CPython.

**What exists:**
- **`crates/lizzard-py-front`** (pure Rust): ruff's parser (pinned to 0.16.8), the module and definition-file database with lazy name resolution and overlay merging (`db.rs`), and the checker that types and lowers functions to a register IR in one pass (`check.rs`, `ir.rs`). Fixture tests are in `tests/front.rs`.
- **`crates/lizzard-py`** (PyO3 extension `lizzard._lizzard`, also free-threaded):
  - the IR interpreter and `LizzardFunction`, with entry checks and CPython-style argument binding (`interp.rs`);
  - runtime checks and qualified-name resolution (`rt.rs`);
  - direct float32 element access with a layout check (`array.rs`).
- **The Python package** (`crates/lizzard-py/python/lizzard`): the import hook, `python -m lizzard`, and the NumPy overlay `stubs/numpy/__init__.pyi`.
- **Bundled builtins definitions** (`crates/lizzard-py-front/stubs/builtins.pyi`, compiled into the front end): `list`, `dict`, `set`, `tuple` and `str` methods, `Iterable`/`Iterator`/`Sequence`, `range`, and builtin functions (`isinstance`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `min`/`max` with `key=`, `sum`, `any`, `all`, `iter`, `next`, `open`, ...). They stand in for typeshed until it is bundled. `typing`/`collections.abc` names such as `Iterable`, `Sequence`, `Callable` map to them.
- **`crates/lizzard-py/dev.sh`** builds into `.venv-ft` (3.14t) and `.venv-gil` (3.14), each with its own target directory, and runs `crates/lizzard-py/tests`. Each program in `tests/programs/` (classes, containers, unions, globals, exceptions, sets and operators, arguments, generics, closures, dataclasses) must print exactly what plain CPython prints.

**Supported language subset:**
- imports;
- **functions:** annotated parameters of every kind (`/`, `*args: T`, keyword-only, `**kwargs: T`), literal defaults, generic functions (`def f[T: Bound](...)` and `TypeVar`s); calls with keywords, `*iterable` and `**mapping`;
- **inner functions, closures and lambdas** (see below);
- **module globals** read by functions, and assigned after `global x`;
- `type X = ...` aliases;
- **classes and dataclasses** (see below);
- **lists, dicts, sets and tuples:** displays, comprehensions, indexing and slicing, `del x[k]`, methods from the bundled definitions, `in` / `not in`, iteration and unpacking (`a, b = t`, `for k, v in d.items()`);
- **unions** (`T | None`, `Optional[T]`, `Union[...]`) with narrowing (see below);
- assignments (including annotated, attribute, subscript, tuple and augmented assignments; `+=` is in place, as in Python) and `x: T` declarations;
- `if`/`elif`/`else`, `for` over `range(...)` (unboxed) or any iterable, `while`, `break`, `continue`, `return`, `assert`;
- **exceptions:** `try`/`except`/`else`/`finally`, `raise`, `raise X from Y`, bare `raise`, `with` (see below);
- conditional expressions, f-strings, `+ - * / // % ** | & ^ << >>`, unary `- + ~`, single comparisons, `and`/`or`/`not`.

Top-level code is checked and then run by CPython. It is checked first, so that functions know the types of the globals it assigns.

**Classes.** A class's methods are compiled to `LizzardFunction`s, which bind like Python functions (`__get__`), so the class itself is an ordinary CPython class:
- Attributes are declared in the class body (`x: float`, optionally with a value) or annotated on their first assignment in `__init__` (`self.x: float = x`). Assigning an undeclared attribute is an error. Reads from the instance are checked at runtime, since Python code can set any attribute.
- `__init__` may omit its return annotation. A call of the class is checked against `__init__` (inherited ones too); a class without `__init__` takes no arguments, unless it derives from an exception.
- Single and multiple inheritance with CPython's MRO (C3), `super().name` in methods, `@staticmethod`, `@classmethod`, `@property` (getters), and dunders (`__add__`, `__eq__`, `__len__`, `__getitem__`, ...) used by operators.
- `@property` with an optional `@x.setter`: assigning the property is checked against the setter.
- Overrides must be compatible: the same arguments with parameter types at least as wide, and a return type at least as narrow. A redeclared attribute keeps its type.
- `@dataclass` (from `dataclasses`): the checker builds `__init__` from the annotated fields (inherited dataclass fields first; defaults from values, `field(default=...)` and `field(default_factory=...)`; `kw_only=True`, `init=False`). Assigning a field of a frozen dataclass is an error. CPython generates the methods at runtime.
- PEP 695 generic classes (`class Stack[T]:`), with explicit arguments (`Stack[int]()`) or inferred from `__init__` (`Pair("a", 1)` is `Pair[str, int]`). Inside the class, `T` and `Self` are *rigid*: only values of that type itself are assignable to them, so `return 5` in a method returning `T` is an error.
- Python code may subclass Lizzard classes. Compiled code calls the Python overrides and checks their results against the declared types.

**Unions and narrowing.** Attribute access, calls, operators and iteration on a union apply to every member; reading an attribute of `X | None` is an error until `None` is ruled out. Narrowing applies to locals and to `x.attr` (for a local `x`) and comes from `is None` / `is not None`, `isinstance` (also with a tuple of classes), truthiness, `not`, `and`/`or` (also between the operands of one condition), conditional expressions, comprehension conditions, `assert`, `while` conditions, early exits (`if x is None: return`) and assignments. Branches merge at joins; a loop forgets what its body assigns. Narrowed attributes are still checked when read, so a call that changes them in between raises a TypeError, never a crash.

**Exceptions.** A function has a handler table: instruction ranges, the register that receives the exception and where handling continues; the innermost handler wins. The exception being handled is CPython's own (`PyErr_SetHandledException`), so `__context__` chaining and `sys.exc_info()` behave as in Python and are restored when a handler ends, whichever way. `return`, `break` and `continue` run the enclosing `finally` bodies, handler restores and `__exit__(None, None, None)` calls first, outside the ranges their own statement protects. `except E as e` binds `e` for that clause only, typed by the caught classes; the builtin exception hierarchy is known. `with` looks `__enter__`/`__exit__` up on the type, and `__exit__` may suppress the exception.

**Inner functions, closures and lambdas.** A syntactic scope analysis finds the variables that inner functions capture. Those live in real CPython cells, passed to the inner function as its closure, so late binding and `nonlocal` work as in Python, across several levels and for recursive inner functions. Inner functions and lambdas compile to their own functions (qualified names like `f.<locals>.g`) and are created at runtime with their closure and their defaults, which are evaluated where they are defined (`lambda x, n=n: ...`). `Callable[[A, B], R]` types functions, bound methods, classes and lambdas (parameters contravariant, result covariant). A lambda's parameter types come from the expected `Callable`; in a call, the other arguments are checked first so the callee's type variables are known (`sorted(people, key=lambda p: p.age)`).

**Behaviour decisions:**
- **A type error in top-level code is a static error.** The example's last line therefore stops the import with a diagnostic. The runtime check is what Python callers see.
- **`a[i]` on `npt.NDArray` (unknown ndim) is `Any`**, by the typing spec's rule for overloads matched through `Any`. The unboxed float32 path needs a shape-typed array: `np.ndarray[tuple[int], np.dtype[np.float32]]`.
- **Container items are checked when read**, as planned: a `list[float]` from Python is only checked to be a `list` at the boundary, and `xs[i]` or a `for` loop item is checked against `float`.
- **Literals take the expected type.** `[1, 2]` is `list[int]`, but where a `list[float]` is expected (an annotation, a parameter, an attribute) it is a `list[float]`. Empty `[]` and `{}` need an expected type. Without one, mixed items join to their common base class, or `object`.
- **A local keeps the type of its first assignment.** Reusing a name for a value of another type is an error, as in mypy.
- **`dict.get(key)` is `Any`**; `get(key, default)` is `V`. (Typeshed has `V | None`; that would make every `get` need a check.)
- **Generator expressions are rejected** rather than compiled eagerly as lists, which would change when their items are computed. List comprehensions work.
- **Unions in third-party definition files are `Any`.** NumPy's stubs use them mostly for parameters built from protocols, which the checker does not model yet; real unions there would reject valid calls. Lizzard modules and the bundled builtins have real unions.
- **A type variable is fixed by its first argument.** `same(1, "a")` for `def same[T](a: T, b: T)` is an error rather than `T = int | str` (widening could be unsound with invariant containers).
- **A lambda needs an expected `Callable` type** for its parameters (an annotation, a parameter, `key=`); `f = lambda x: ...` alone is an error, like an unannotated `def`.
- **A name is local to a function if it is assigned anywhere in it**, as in Python: reading it before the assignment is an error rather than a read of the global.
- **Builtin exceptions' attributes other than `args` are `Any`** (`errno`, `filename`, ...).

**Findings:**
- **Overlays must use only public names.** NumPy 2.5 renamed the private TypeVar `_NBit1` to `_NBitT`, which silently degraded an overlay that used it. The overlay now uses `Self` and method-level type parameters. `Self` is a TypeVar bounded by its class, solved to the receiver's own type.
- **Statically typed `float` doesn't mean exactly `float`.** A value typed `float` may be an `np.float64` (a `float` subclass), which NumPy promotes where an exact Python float would be a weak scalar (NEP 50). So `float32 op float` is typed `floating`, not `float32`. Mixed float32/float arithmetic goes through NumPy; only float32-with-float32 is unboxed.
- **Aliases supply runtime classes.** A parameterized type gets its runtime class from the alias its module exports for it (`floating[_32Bit]` is `numpy.float32`). These aliases are resolved on demand when a runtime check needs one.

**Deferred:**
- checking that overlays only refine the base declarations;
- bundled typeshed (stdlib modules are `Any`), `Protocol`s (structural typing) and `Literal` types;
- generators (`yield`) and generator expressions, `async`, `match`, `except*`, the walrus operator, `del` of names and attributes, chained comparisons;
- decorators other than `@staticmethod`, `@classmethod`, `@property`/`.setter` and `@dataclass`, nested classes, classes inside functions, `__slots__`, metaclasses;
- `__orig_class__` checks for generic instances at the boundary (a `Stack[int]` is only checked to be a `Stack`);
- vectorcall entry points, the JIT, and safepoints;
- direct access for other dtypes and dimensions.
