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
- `Protocol` and `Literal` are modelled (below); `ParamSpec` and `TypedDict` are treated as `Any`.

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
- **Bundled builtins definitions** (`crates/lizzard-py-front/stubs/builtins.pyi`, compiled into the front end): `list`, `dict`, `set`, `tuple` and `str` methods, the standard protocols (`SupportsIndex`, `Sized`, `Hashable`, `Collection`, ...), `Iterable`/`Iterator`/`Sequence`, `range`, and builtin functions (`isinstance`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `min`/`max` with `key=`, `sum`, `any`, `all`, `iter`, `next`, `open`, ...). They stand in for typeshed until it is bundled. `typing`/`collections.abc` names such as `Iterable`, `Sequence`, `Callable` map to them.
- **`crates/lizzard-py/dev.sh`** builds into `.venv-ft` (3.14t) and `.venv-gil` (3.14), each with its own target directory, and runs `crates/lizzard-py/tests`. Each program in `tests/programs/` (classes, containers, unions, globals, exceptions, sets and operators, arguments, generics, closures, dataclasses, generators, operator overloading, properties) must print exactly what plain CPython prints.

**Supported language subset:**
- imports;
- **functions:** annotated parameters of every kind (`/`, `*args: T`, keyword-only, `**kwargs: T`), literal defaults, generic functions (`def f[T: Bound](...)` and `TypeVar`s); calls with keywords, `*iterable` and `**mapping`;
- **inner functions, closures and lambdas** (see below);
- **generators** (`yield`, `yield from`; see below);
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
- Single and multiple inheritance with CPython's MRO (C3), `super().name` in methods, `@staticmethod` and `@classmethod`.
- **Operator overloading** through dunders: arithmetic, bitwise and `@` operators with their reflected forms (`__radd__`, ...) and in-place forms (`a += b` uses `__iadd__` if defined), unary `-`/`+`/`~`, rich comparisons (a comparison uses the left operand's method, else the right one's reflected method, as Python does; ordering objects that define neither is an error, `==`/`!=` fall back to identity), chained comparisons, `__getitem__`/`__setitem__`/`__delitem__`, `__contains__`, `__iter__`, `__len__`, `__bool__`, `__call__`, and builtins that dispatch to dunders (`abs()`, `min`/`max`, `float()`, `hash()`, ...).
- **Properties:** `@property` with `@x.getter`, `@x.setter` and `@x.deleter`, each compiled. Reads use the getter, assignments are checked against the setter, `del obj.x` needs a deleter; a property without a setter is read-only. `del obj.attr` also deletes plain attributes.
- **`typing.Protocol`:** a class matches a protocol structurally, by the types of the members the protocol declares. Protocol-typed values are not `isinstance`-checked at runtime (Python allows that only for `@runtime_checkable` protocols), and a protocol's own methods stay Python functions.
  - **Builtin values match protocols too.** `int`, `float`, `bool`, `str`, tuples and the containers match by their builtins classes' methods, so `SupportsIndex`, `SupportsInt`, `SupportsFloat`, `SupportsAbs[T]`, `Sized`, `Hashable`, `Container`, `Collection[T]` and `Reversible[T]` (from `typing` or `collections.abc`) take what typeshed's do: `list` is not `Hashable` (its `__hash__` is None), nor is a class that defines `__eq__` without `__hash__`.
  - **Callback protocols:** a function, lambda or class matches a protocol whose only member is `__call__`, by its signature; a lambda takes its parameter types from the protocol.
  - **Recursive protocols** (`_NestedSequence[T]`, whose items are `T | _NestedSequence[T]`) are checked assuming the check in progress holds.
  - **Nominal first:** a class that names a protocol among its bases matches it as a subclass (`str` is a `Collection[str]` through `Sequence[str]`, although its `__contains__` takes only `str`).
  - Forward references in quotes (`list["Node"]`) are types.
- **`Literal` types.** `Literal["r", "w"]` is a union of literal types; its aliases (`type Mode = ...`, or `Mode = Literal[...]`) work as in mypy.
  - A literal expression has a literal type where one is expected (an annotation, a `Literal` parameter, a default, a return), and a literal argument picks `Literal` overloads (`load(path, "binary")`). Elsewhere it is its base type, and an unannotated variable takes the base type (`n = one()` is an `int`).
  - Literal types are subtypes of their base types; arithmetic, methods and comparisons use the base type.
  - `x == "r"` and `x != "r"` narrow a literal union to that member, or to the others.
  - At the boundary, a literal-typed value is checked against its base type only.
- **Unions in definition files are types** everywhere, now that their protocol and literal parameters are modelled. NumPy's own overloads then type `np.arange(10)` (`ndarray[tuple[int], dtype[intp]]`), `np.zeros((2, 3), dtype=np.float32)` and `np.empty_like(a)` precisely, so the overlay no longer declares them.
- **`reveal_type(x)`** reports x's static type as a note (`note: revealed type is ...`), as mypy does, without failing the check. Imported from `typing`, it is also called at runtime.
- Overrides must be compatible: the same arguments with parameter types at least as wide, and a return type at least as narrow. A redeclared attribute keeps its type.
- `@dataclass` (from `dataclasses`): the checker builds `__init__` from the annotated fields (inherited dataclass fields first; defaults from values, `field(default=...)` and `field(default_factory=...)`; `kw_only=True`, `init=False`, `order=True`). Assigning a field of a frozen dataclass is an error. CPython generates the methods at runtime.
- PEP 695 generic classes (`class Stack[T]:`), with explicit arguments (`Stack[int]()`) or inferred from `__init__` (`Pair("a", 1)` is `Pair[str, int]`). Inside the class, `T` and `Self` are *rigid*: only values of that type itself are assignable to them, so `return 5` in a method returning `T` is an error.
- Python code may subclass Lizzard classes. Compiled code calls the Python overrides and checks their results against the declared types.

**Unions and narrowing.** Attribute access, calls, operators and iteration on a union apply to every member; reading an attribute of `X | None` is an error until `None` is ruled out. Narrowing applies to locals and to `x.attr` (for a local `x`) and comes from `is None` / `is not None`, `isinstance` (also with a tuple of classes), truthiness, `not`, `and`/`or` (also between the operands of one condition), conditional expressions, comprehension conditions, `assert`, `while` conditions, early exits (`if x is None: return`) and assignments. Branches merge at joins; a loop forgets what its body assigns. Narrowed attributes are still checked when read, so a call that changes them in between raises a TypeError, never a crash.

**Exceptions.** A function has a handler table: instruction ranges, the register that receives the exception and where handling continues; the innermost handler wins. The exception being handled is CPython's own (`PyErr_SetHandledException`), so `__context__` chaining and `sys.exc_info()` behave as in Python and are restored when a handler ends, whichever way. `return`, `break` and `continue` run the enclosing `finally` bodies, handler restores and `__exit__(None, None, None)` calls first, outside the ranges their own statement protects. `except E as e` binds `e` for that clause only, typed by the caught classes; the builtin exception hierarchy is known. `with` looks `__enter__`/`__exit__` up on the type, and `__exit__` may suppress the exception.

**Inner functions, closures and lambdas.** A syntactic scope analysis finds the variables that inner functions capture. Those live in real CPython cells, passed to the inner function as its closure, so late binding and `nonlocal` work as in Python, across several levels and for recursive inner functions. Inner functions and lambdas compile to their own functions (qualified names like `f.<locals>.g`) and are created at runtime with their closure and their defaults, which are evaluated where they are defined (`lambda x, n=n: ...`). `Callable[[A, B], R]` types functions, bound methods, classes and lambdas (parameters contravariant, result covariant). A lambda's parameter types come from the expected `Callable`; in a call, the other arguments are checked first so the callee's type variables are known (`sorted(people, key=lambda p: p.age)`).

**Generators.** A generator function is annotated as returning `Iterator[T]`, `Iterable[T]` or `Generator[Y, S, R]`: its `yield`s are checked against `Y`, `yield` gives `S` (from `send`), and `return` gives `R` (`StopIteration.value`, and the value of `yield from` over it). Calling it checks the arguments and returns a `LizzardGenerator` that owns the function's frame.
- **The model is the original VM's** (`lizzard-vm/src/generator.rs`): the frame is the registers and the pc of the `yield` it is suspended at. Resuming writes the sent value into the yield's destination register and continues after it; `throw` continues *at* the `yield` with the exception raised there, so the function's own handler table applies. An atomic "running" flag makes a concurrent or re-entrant resume raise `ValueError: generator already executing`.
- **The interpreter's entries and exits are the JIT's.** `exec` enters a frame at any pc, optionally with an exception pending there, and leaves by `Return`, `Yield` or `YieldFrom` without unwinding the frame. A compiled function needs the same: an entry block that dispatches on pc 0 and the resume points (`Yield`/`YieldFrom` + 1, and the yields themselves for `throw`), with exits that store the pc. The generator object only moves registers between its owned frame and `exec`; it needs no change for compiled code.
- **CPython's protocol:** `__next__`, `send`, `throw` (also the deprecated three-argument form, with its warning), `close` (returning the generator's return value), `StopIteration` values, PEP 479 (a `StopIteration` escaping the body becomes `RuntimeError`), `gi_running`. It is registered as a `collections.abc.Generator`.
- **`yield from` works with any iterator**, Lizzard or Python: the generator object delegates `send`, `throw` and `close` to it as CPython does (`PyIter_Send`, `throw`, `close`), and resumes the frame with its return value. Python's `yield from` drives Lizzard generators the same way.
- **The handled exception is part of the frame.** A generator suspended in an `except` block keeps its `sys.exception()` and gives the caller back theirs while suspended.
- **Unfinished generators are closed when freed**, so their `finally` blocks run; an exception there is reported as unraisable, as CPython does.
- **`for` loops resume Lizzard generators directly**, without a `StopIteration` at the end.

**Strict attribute access.** Every attribute read, write, `del`, call and operator must be declared: in the class or a base class, in a definition file, or among `object`'s and `type`'s own attributes (`__class__`, `__dict__`, `__name__`, `__mro__`, ...). There is no fallback to `Any`. A class deriving from a class nothing defines (an opaque standard-library module's, say `queue.Queue`) can still be written, but using a member it does not declare itself, or constructing it, is an error that names the untyped base. Exception classes get the builtin exceptions' attributes (`args`, `__cause__`, `errno` for `OSError`, ...).
- **Standard-library definitions** live next to the bundled overlays (`python/lizzard/stubs/<module>.pyi`). A standard-library module with such a file is read from it alone; the others stay opaque. `threading` is the first: `Thread` (and subclasses of it), `Timer`, `Lock`, `RLock`, `Condition`, `Semaphore`, `BoundedSemaphore`, `Event`, `Barrier` and the module functions.

**JIT (`crates/lizzard-jit`, Cranelift).** Functions are compiled to native code once warm: on their second call (`LIZZARD_JIT_WARM`), or after 200 iterations of a loop (`LIZZARD_JIT_LOOPS`), which then continues in the compiled code (on-stack replacement at the loop header). `LIZZARD_JIT=0` keeps everything in the interpreter, `LIZZARD_JIT_DUMP=1` prints the Cranelift IR, `LIZZARD_JIT_DEBUG=1` reports compilations (`=strict`, the tests' setting, makes a function that cannot be compiled an error).
- **One frame for both engines.** Registers are 8-byte slots (`frame.rs`): an owned `PyObject*` (null while unbound), an `i64`, an `f32` or a bool. The interpreter, generators and compiled code all work on it, so a frame can move between the engines at any instruction boundary.
- **Compiled code** is `fn(ctx, regs, *pc, pending) -> status`, entered at 0, a loop header, or where a generator resumes. It does the control flow natively (jumps, exception handler dispatch, the exits), and statuses say how it stopped: returned or yielded, raised, or **bailed out** (`BAIL`: the interpreter continues the frame at `*pc`).
- **Specialized on what the caches saw.** By the time a function is warm, the interpreter's caches know, per instruction (`lizzard_jit::Spec`): an attribute's type, version tag and offset (inline values or `__slots__`), a global's value and the epoch it was read at, the Lizzard function or plain class a call reaches, the method a `CallMethod` finds and why nothing shadows it, and each check as type-pointer comparisons. Compiled code checks these with guards of a few machine instructions (type pointer, version tag, the values' valid byte) and does the work inline: an attribute load is a load at a fixed offset, a global load a comparison of the epoch. A failed guard takes the instruction's generic path (the runtime's helper, correct for anything), or, where the code after it relies on the specialization, bails out. A site compiled before its cache saw anything counts its generic executions, and after 1000 the function is recompiled if the caches know more by then (at most 4 compilations).
- **Virtual registers.** Within a block (a run of instructions nothing jumps into), compiled code may know a register's value without its slot holding it: an unboxed float or int (the checker records which `+ - * /`, `//`, `%` and bitwise operators have `float` or `int` operands, `FuncIr::hints`, so arithmetic on them is machine arithmetic with an exact-type guard per object operand; an int overflow, a division by zero or a non-float result takes the generic path, or bails out), a borrowed pointer (an attribute's value, a constant, a global: no reference taken), or both. A register is **materialized** (boxed, given its reference, stored in its slot) only where something needs the object: a helper, a jump to another block, an exception handler, a bail-out, a call argument, a store; and only if it is live there (a liveness pass with exception edges). Borrowed pointers get their reference before anything that could release what they point into: a store, a call, the release of a slot's old value. Scalar registers are Cranelift variables, written through to their slots.
- **Across jumps and loops.** A dataflow pass finds, per label, the registers that hold a float or an int compiled code made on every path into it; they stay unboxed across the jump as Cranelift block parameters (a loop accumulator never becomes an object). A jump whose value turns out not to be of that kind bails out at the target, and a loop entered from the interpreter (on-stack replacement) unboxes its slots or bails.
- **Speculation.** Until bail-outs force a recompile (after which the function is compiled without it), a failed attribute guard bails out instead of rejoining a slow path, so what it checked holds afterwards: an object's type, version and inline values are verified once and trusted until something can run Python code (a helper, a call); a `float` check guards an exact float and keeps it unboxed; an object operand of int arithmetic is unboxed once. An inlined method's receiver and a constructed instance inherit the call guard's verification, and inlined parameters what the caller knew (a constant stays a constant).
- **Reference counting is inline**, as CPython's headers do it (`rc.rs`'s model: the owner thread's local count or the atomic shared count in the free-threaded build), and avoided where the value's lifetime is known: borrowed reads, a computed value's new reference handed to the attribute it is stored in, a register's reference moved out by `return`.
- **Floats are updated in place**: storing a computed float over an attribute holding an exact float that nothing else references (one reference, owned by this thread) writes its value instead of allocating, as `self.x += v * dt` needs.
- **Direct calls.** A call that reaches a Lizzard function with plain positional parameters (a global function, a cached method, a class's `__init__` after allocating the instance) builds the callee's frame in place, its arguments borrowed from the caller when the callee never writes its parameters (`FuncRt::borrow_params`); the checks the call site proves are skipped and the others are type-pointer guards. Compiled code then calls the callee's compiled entry itself: the context is five machine words on the caller's stack (thread, return value, frame, the function's description, exit), around `Py_EnterRecursiveCall`; a compiled function's `return` releases the slots its code wrote, so the caller has nothing to clean up, and only a bail-out or an exception in the callee goes through a helper. A callee not compiled yet goes through `enter`, which runs it in the interpreter. A cached builtin method (`list.append`) is called with `PyObject_Vectorcall` directly. (A variant taking int parameters unboxed was tried and dropped: CPython's small ints are immortal and cached, so boxing them costs a few instructions and no allocation.)
- **Constructing in a dying instance.** `a = a.add(d)` allocates a `V` and then frees the old one. When the constructor's `__init__` is inlined and an instance of the class is about to be released (its only reference is a register that dies at the call: a receiver whose caller overwrites it), the construction takes that instance instead of allocating: at runtime it must have one reference, owned by this thread, no weak references, no finalizer in its class, and exactly the attributes `__init__` assigns, in the order it assigns them (or all the `__slots__` of a class deriving from `object`). Nothing can then tell it from a new instance; `__init__`'s stores replace its attributes, and its floats, referenced only there, are updated in place. A `V` and its two floats per iteration are neither allocated nor freed.
- **Inlining.** A small callee (at most 48 instructions, no loops, no handlers, parameters never written, every instruction one compiled code does itself given its specs) is compiled into its caller behind the call's guard, two levels deep (the second for callees of at most 16 instructions, such as an `__init__`): its registers are virtual like the caller's (a float argument stays unboxed across the call), and its frame is a stack slot that is zeroed and released only where it was written. Any slow path in the inlined body hands its frame to the interpreter, which finishes the call (`finish`); the caller continues in compiled code with the result. An inlined function whose every reachable `return` returns an int compiled code made (the kind analysis run with what the caller knows of the arguments) returns it unboxed: `fib(n - 1) + fib(n - 2)` adds machine ints, boxing nothing between the calls. A result the interpreter finished with is unboxed at the call, and a big int continues in the interpreter after it.
- **Frames built when they are handed over.** An argument the caller holds as an unboxed number (`n - 1`) is passed as it is: the inlined parameter's slot stays null, and the number is boxed into it only where the callee's code needs the object, or where its frame goes to the interpreter (`finish`, which takes its own references; the box is released after it, after an exception's `unwind`, and at the callee's end). `fib`'s arguments and `V(self.x + o.x, ...)`'s floats are never boxed on the normal path.
- **Slots written only on slow paths** are released only if one ran: a store from a cold block (a slow path, a bail-out, a big int result) sets a flag in the function's stack frame (cleared at every entry), and a return (or an inlined callee's end) checks it once instead of each such slot. Frames start null or are released by their owner (the interpreter's), so an unset flag means nothing to release.
- **Loop safepoints** save the loop's unboxed values and scalar variables to the stack around their (rare) call, so that the loop needs no call-preserved registers for them: `for i in range(n): t += i * i` keeps `i` in a register (21 instructions per iteration, from 27).
- **Self tail calls** (`return f(...)` in `f`, outside `try`, `with` and `except`, every argument's check proven by its type) run as a loop, in the interpreter and in compiled code. The front end gives such a function copies of its parameters (the parameters themselves stay as the caller passed them, so calls into it still borrow them). At the call, `JumpIfNotSelf` tests whether the name still means this function (its code, globals and closure; a rebound global is called as usual). If so, the arguments become the copies, the other locals are unbound as on entry, and the body starts over. Compiled code compares against the function object the global held when it was compiled, and keeps `int` and `float` parameters' copies unboxed across the jump, guarded at the entry (parameters start out as their declared kind when speculating). The calls are counted, and RecursionError comes at the recursion limit, as CPython's frames would raise it: output stays CPython's, just without the frames. (An exception's traceback notes show the function once, not once per call.) A tail-recursive sum of 900 ints, 2000 times: CPython 0.246 s, interpreter 0.240 s, JIT 0.016 s; over floats 0.307 s / 0.196 s / 0.014 s. `int` with `float` arithmetic (`n * 0.5`) is compiled as float arithmetic too.
- **`for` over a list** reads the list iterator's index and the list's items directly (layouts probed at start-up, like the end marker: no list, or with free threading, an index of -1) and marks the end as CPython does. While speculating, another iterable bails out, so the loop body continues in the same block and keeps what it knows of the item; a type check on it, and the move into the loop variable, carry its type on to the method guard, which then checks only the type's version. (Types other than exact floats and ints are forgotten wherever Python code may have run, as an instance's `__class__` can be assigned.)
- **Safepoints:** every 1024 loop back-edges, compiled code runs signal handlers and detaches briefly, so Ctrl-C and the free-threaded stop-the-world GC work in tight loops.
- **Other threads are trusted more than CPython does**: compiled code reads and writes attributes without the object's critical section, and a borrowed value may be one another thread replaces. Reference counts stay correct (shared counts are atomic).
- **Calls use vectorcall.** `LizzardFunction` implements CPython's vectorcall protocol and is a method descriptor (PyO3 offers neither, so the type is patched once at startup): CPython and compiled code call it with the arguments in an array, and `obj.m(...)` passes `obj` first instead of creating a bound method. `obj.m(args)` compiles to `CallMethod` (`PyObject_VectorcallMethod`, which looks `m` up through CPython's type cache) when the arguments are names and literals, so that looking `m` up after them changes nothing. Keyword names are tuples built once per call site, and a frame's slots come from a per-thread pool. The vectorcall entry only assumes the thread is attached, as CPython guarantees, and does not go through PyO3's attach (about as costly as the call itself). PyO3 therefore defers the releases of the references it drops there, a `PyErr`'s among them, to its next attach. Where exceptions are handled or leave a function, Lizzard runs those releases itself, so that a loop catching exceptions does not keep every one alive; and the runtime drops no exception it neither raises nor hands to a handler (a special method is bound through its type's `tp_descr_get`, not a `__get__` lookup that may fail).
- **Attribute inline caches** (`ops.rs`). Each attribute site caches, for one `(type, tp_version_tag)` (any change to the class or a base gives it a new version tag, so a stale entry never hits), where the attribute lives; a site that sees more than 8 types stops caching:
  - **in the instance's inline values**, CPython 3.13+'s storage for a plain class's instance attributes: an array right after the object, in the order of the type's shared keys. The attribute is at a fixed offset, read and written there directly, as CPython's own `LOAD_ATTR_INSTANCE_VALUE` / `STORE_ATTR_INSTANCE_VALUE` do and with their guards: the type has no attribute of that name and uses the generic getattr/setattr; on a read, the values are still valid (no detached `__dict__`) and a missing value takes the generic path (AttributeError, `__getattr__`); in the free-threaded build the reference is taken with `PyUnstable_TryIncRef` and a re-check of the slot (CPython's `_Py_TryIncrefCompare`); a store replaces an existing value only, under the object's critical section, with no `__dict__` object (first assignments take the generic path, which records the insertion order). This reads CPython-internal layouts (`Include/internal/pycore_dict.h`, `pycore_object.h`, 3.14): at start-up a probe class's attributes are read both ways, and the path stays off unless they agree (`lizzard._lizzard._stats()["inline_values"]`);
  - **in a `__slots__` member**, read and written with `PyMember_GetOne`/`SetOne`. The checker accepts `__slots__` and requires the attributes a class adds to be listed;
  - else nowhere it can reach: `PyObject_GetAttr`/`SetAttr`.
  The interpreter uses these entries through the helpers; compiled code turns them into guards and inline loads and stores.
- **Constructor caches.** A call site whose callee is a class caches, per `(class, version)`, that constructing an instance is allocating it and calling its `__init__`, as CPython's `CALL_ALLOC_AND_ENTER_INIT`: the class is a plain heap type whose metaclass is `type`, not abstract, with `object`'s `__new__` and allocator, and an `__init__` that is a Lizzard or Python function. The site then allocates (`tp_alloc`, all `object.__new__` does) and calls `__init__` with the instance first: a Lizzard `__init__` runs as a frame in the interpreter's loop, and the call's result is the instance (after checking that `__init__` returned None, with CPython's message). The checker resolves `C(args)` to `C.__init__`, so its proven argument checks are skipped too. Anything else (a custom `__new__`, a metaclass `__call__`, keyword arguments) takes `type.__call__`. Construction of `V(x, y)`: 4958 -> 4163 instructions per `a.add(d)` (CPython: 2008).
- **First assignments in inline values.** An attribute store whose name is in the class's shared keys but not yet set on the instance (the assignments in `__init__`) also writes the inline value directly, appending the attribute's index to the values' insertion order as CPython's `STORE_ATTR_INSTANCE_VALUE` does, so `vars()` and `__dict__` list attributes in assignment order. The start-up probe checks this too. A store site does not cache "generic" while the class's shared keys lack the name (they gain it from the first instance's generic store without changing the class's version).
- **Method caches.** A `CallMethod` site caches the method of one `(type, version)` when CPython's `LOAD_ATTR_METHOD_NO_DICT` / `..._WITH_VALUES` reasoning proves no instance attribute shadows it: the instances have no `__dict__`, or their attributes are in valid inline values whose shared keys lacked the name and have not grown since (keys only grow). The method (a method descriptor: a Lizzard or Python function, `list.append`, ...) is called with the object first; a Lizzard function directly.
- **Checks are prepared once per site** (the class of a check is resolved once, not by name on every call), and pass on an exact type-pointer comparison before trying `isinstance`.
- Cranelift is built optimized even in debug builds (workspace `Cargo.toml`); compiling then costs about a millisecond per function.
- **Benchmarks** (`crates/lizzard-py/bench/bench.py`, release build, best of 3, CPython 3.14t on x86-64; `/s` is the `__slots__` variant):

  | | CPython | interpreter | JIT |
  |---|---|---|---|
  | int loop (2M) | 0.104 s | 0.162 s | 0.004 s |
  | `fib(25)` | 0.008 s | 0.028 s | 0.003 s |
  | vector objects (500k `a.add(d)`) | 0.083 s | 0.180 s | 0.007 s |
  | same, `__slots__` | 0.076 s | 0.188 s | 0.007 s |
  | method calls updating attributes (500k, over a list) | 0.036 s | 0.090 s | 0.004 s |
  | same, `__slots__` | 0.035 s | 0.089 s | 0.004 s |
  | list build and scan (1M) | 0.102 s | 0.154 s | 0.071 s |
  | generator consumed by `for` (2M) | 0.156 s | 0.498 s | 0.373 s |
  | generator pipeline: filter, `yield from` (1M) | 0.085 s | 0.294 s | 0.298 s |

  In machine instructions (Valgrind, per unit, large runs; CPython / JIT): loop iteration 715 / 21, `b.step(0.01)` 938 / 70 (inlined), `fib(10)` 84.6k / 21.8k (a `fib` call about 478 / 123; two levels inlined, ints unboxed between them, arguments boxed only for the interpreter). What remains: the call's frame and context setup and the slots of the second inlined level, which its real calls write (`fib`); reloading the iterator at each loop header (facts about objects do not survive a back-edge yet); and CPython's own work: `append` (`lists`). Generators are the weak spot, slower than CPython: about 2000 instructions per item, and only some 400 of them the generator's compiled code. A `for` over a Lizzard generator is not compiled (a helper runs `step`, which resumes it), a resume locks the generator's state twice, moves its frame out and back and saves and restores the handled exception, entering the compiled code builds a new context, and a compiled `yield` goes through `step` too, which allocates its exit; and the list-iteration speculation bails out on a generator until the loop is recompiled without speculation.

**Interpreter performance.** `crates/lizzard-py/bench/count.sh` counts machine instructions per unit of work with Valgrind (deterministic, unlike timings). At first the interpreter needed 3.9x CPython's instructions per loop iteration, 5.0x per method call and 6.8x per `fib` call. The causes, measured, and what was done:
- **Dispatch was half the time.** Each instruction went through `step`, one large function returning `PyResult<Option<Exit>>`: about 45 machine instructions of overhead each. The interpreter loop now runs the common instructions itself on the raw slots, with the helpers compiled code uses (`ops.rs`), and handles exceptions only when one is raised; the rest still go through `step`.
- **The IR was twice as long as needed.** An assignment from a temporary was `Move` (incref) plus `Unbind` (decref): it is now `Take`, which moves the reference. A branch on a comparison was compare, check of the result's type, truthiness and branch: it is now `CompareBool` (only the truth matters, so neither the bool object nor its check is needed). `i * i` boxed `i` twice: the box is reused.
- **Globals and builtins were looked up in dicts every time** (a builtin missed the globals first). `LoadGlobal` sites now cache the value (`globals.rs`): a dict watcher on the module's globals and the builtins bumps an epoch on any change; readers take no lock (a sequence counter guards the entry) and take the value with `PyUnstable_TryIncRef`, CPython 3.14's API for reading an object another thread may release.
- **Calls were expensive:** CPython's vectorcall dispatch, a panic guard, a `Vec` of arguments, argument checks through PyO3, a frame from a thread-local pool, two scans to release it. A call between Lizzard functions is now direct (a type-pointer test), the arguments go straight into a frame on the Rust stack, checks are decided by type pointers when they can be, and `return` hands over its register's reference.
- **Int and float operations went through CPython's generic paths** (`PyObject_RichCompare` alone was about 160 instructions for `n < 2`). As in CPython's specializing interpreter, `+ - *` and comparisons of exact compact ints, and `+ - * /` and comparisons of exact floats, are computed in place (the layout read is checked against CPython's own results at start-up, else the fast paths stay off).
- **Recursion:** calls between Lizzard functions recurse on the C stack, and nothing bounded that: deep recursion crashed. Frames now enter CPython's recursion check, which raises RecursionError before the stack runs out (the interpreter crate is built with `opt-level = 1` in debug builds so that its frames fit CPython's margin).

  Now (instructions per unit, CPython / interpreter): loop iteration 577 / 1010 (1.75x), method call 936 / 1908 (2.0x), `fib(10)` 84k / 229k (2.7x). The four steps after the inline-values work, each measured:
  1. **Inline reference counting** (`rc.rs`): `Py_INCREF`/`Py_DECREF` as CPython's headers do them (the free-threaded build's owner-thread local count via the thread pointer, else the atomic shared count), checked against `Py_REFCNT` at start-up; and a liveness pass (with exception edges) that turns an object `Move` whose source is not read again into a `Take`. Small: most remaining reference counting is inside CPython (method call 2269 -> 2226).
  2. **Specialized call sites**: a call of a positional Lizzard function builds the callee's frame straight from the caller's registers, and the checker records (`Direct`) which parameter checks the arguments' static types prove; the site skips them when the function called is the one it resolved (a rebound global or an override gets every check). Method call 2226 -> 2133, `fib(10)` 256k -> 246k.
  3. **Compact bytecode** (`bytecode.rs`): fixed-size records with operands validated at load (unchecked indexing in the loop), variants decided once, and `BinOp+Check`, `LoadAttr+Check`, `CompareBool+JumpIfFalse` fused. Dispatch went from about 29 to about 16 machine instructions per instruction; loop iteration 1147 -> 1001, method call 2133 -> 1920, `fib(10)` 246k -> 226k.
  4. **Calls without C recursion**: the interpreter runs calls between interpreted Lizzard functions in one loop. A callee's registers are a window on a stack of slots the loop owns (room for the next frame is reserved before the caller's registers are read, so binding never moves them); a return or an exception leaving the callee pops it and continues in the caller. Recursion is bounded by `sys.getrecursionlimit()` with CPython's `RecursionError: maximum recursion depth exceeded`, not by the C stack. Its cost per call matches the native recursion it replaces (Rust's calls were already cheap), so the gain is depth and CPython's limit, not speed.
  What remained then:
  - **The JIT** used none of steps 2-4 (it ran the IR, recursing through `call_at`), so on call-heavy code it was slower than the interpreter. It has since been rewritten (above).
  - **Constructing instances** cost twice CPython's (`a.add(d)`, which builds a `V`: 4163 instructions against 2008), the allocation and its `__init__` frame. See below for what changed.
  - **Result checks** after operations whose operands are not exact builtins (a `float` attribute read from an instance is checked when read).
- Every test program runs compiled and interpreted, and both must match CPython.

**Behaviour decisions:**
- **A type error in top-level code is a static error.** The example's last line therefore stops the import with a diagnostic. The runtime check is what Python callers see.
- **`a[i]` on `npt.NDArray` (unknown ndim) is `Any`**, by the typing spec's rule for overloads matched through `Any`. The unboxed float32 path needs a shape-typed array: `np.ndarray[tuple[int], np.dtype[np.float32]]`.
- **Container items are checked when read**, as planned: a `list[float]` from Python is only checked to be a `list` at the boundary, and `xs[i]` or a `for` loop item is checked against `float`.
- **Literals take the expected type.** `[1, 2]` is `list[int]`, but where a `list[float]` is expected (an annotation, a parameter, an attribute) it is a `list[float]`. Empty `[]` and `{}` need an expected type. Without one, mixed items join to their common base class, or `object`.
- **A local keeps the type of its first assignment.** Reusing a name for a value of another type is an error, as in mypy.
- **`dict.get(key)` is `Any`**; `get(key, default)` is `V`. (Typeshed has `V | None`; that would make every `get` need a check.)
- **Generator expressions are rejected** rather than compiled eagerly as lists, which would change when their items are computed. List comprehensions work.
- **A type variable is fixed by its first argument.** `same(1, "a")` for `def same[T](a: T, b: T)` is an error rather than `T = int | str` (widening could be unsound with invariant containers).
- **A lambda needs an expected `Callable` type** for its parameters (an annotation, a parameter, `key=`); `f = lambda x: ...` alone is an error, like an unannotated `def`.
- **A name is local to a function if it is assigned anywhere in it**, as in Python: reading it before the assignment is an error rather than a read of the global.
- **Builtin exceptions' attributes are known by name but mostly `Any`** (`errno`, `filename`, `__cause__`, ...); `args` is a tuple and `__suppress_context__` a bool.
- **The value sent to a `yield` is checked against `S | None`.** `next()` sends None whatever the send type is, as typing allows.
- **Temporaries may keep objects alive until the function returns.** Assignment moves a temporary's reference into the variable, so `del x` frees an object at once, but other temporaries (a call's arguments, say) are only released when overwritten. A liveness pass that clears dead registers (needed by the JIT anyway) would make finalization as prompt as CPython's.
- **Compiled frames add no traceback entries** (they add a `lizzard:` note per frame instead), so tracebacks of errors in generators point at the caller.

**Findings:**
- **Overlays must use only public names.** NumPy 2.5 renamed the private TypeVar `_NBit1` to `_NBitT`, which silently degraded an overlay that used it. The overlay now uses `Self` and method-level type parameters. `Self` is a TypeVar bounded by its class, solved to the receiver's own type.
- **Statically typed `float` doesn't mean exactly `float`.** A value typed `float` may be an `np.float64` (a `float` subclass), which NumPy promotes where an exact Python float would be a weak scalar (NEP 50). So `float32 op float` is typed `floating`, not `float32`. Mixed float32/float arithmetic goes through NumPy; only float32-with-float32 is unboxed.
- **Aliases supply runtime classes.** A parameterized type gets its runtime class from the alias its module exports for it (`floating[_32Bit]` is `numpy.float32`). These aliases are resolved on demand when a runtime check needs one.

**Deferred:**
- checking that overlays only refine the base declarations;
- bundled typeshed (stdlib modules without a definition file are `Any`), `@runtime_checkable` protocols, `Literal` enum members, `ParamSpec` and `TypedDict`;
- `threading.local`, whose instances take arbitrary attributes;
- generator expressions, `async` (coroutines, async generators), `match`, `except*`, the walrus operator, `del` of globals;
- `gi_frame`, `gi_code` and `inspect.isgenerator()` for Lizzard generators;
- decorators other than `@staticmethod`, `@classmethod`, `@property`/`.setter` and `@dataclass`, nested classes, classes inside functions, `__slots__`, metaclasses;
- `__orig_class__` checks for generic instances at the boundary (a `Stack[int]` is only checked to be a `Stack`);
- caches for `__dict__` attributes; facts about objects (types, verified shapes) across loop iterations; inlining callees with loops or handlers; compiling calls to Python functions and builtins other than methods;
- direct access for other dtypes and dimensions.
