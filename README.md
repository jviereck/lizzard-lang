# Lizzard

**A JIT for typed Python that runs inside CPython, fully compatible with the Python code around it.**

A Lizzard module is an ordinary `.py` file with type annotations. It runs unchanged under plain CPython. Lizzard type-checks it and compiles its functions to native code (Cranelift). The compiled code runs *embedded in the CPython interpreter*:

- its values are real `PyObject*`s;
- memory is CPython's own reference counting and garbage collector;
- calls in and out are ordinary CPython calls.

So a Lizzard module imports, and is imported by, any Python code: the standard library, NumPy, C extensions, threads. Python code can subclass its classes, monkey-patch them and pass it anything. A wrong type at the boundary raises `TypeError`, never a crash. A program prints exactly what CPython prints for it; the test suite checks that against CPython for every test program.

```python
# lizzard: module
class Body:
    def __init__(self, x: float, v: float) -> None:
        self.x: float = x
        self.v: float = v

    def step(self, dt: float) -> None:
        self.x += self.v * dt


def simulate(n: int) -> float:
    bodies = [Body(float(i), 1.0) for i in range(100)]
    for _ in range(n // 100):
        for b in bodies:
            b.step(0.01)
    return bodies[0].x


if __name__ == "__main__":
    print(simulate(500_000))
```

```bash
python app.py                # plain CPython: same output
```

```bash
python -m lizzard app.py     # checked and compiled by Lizzard
```

Importing `lizzard` installs an import hook, so any module marked `# lizzard: module` is checked and compiled when it is imported.

## How it works

1. **Checker** (`crates/lizzard-py-front`). Python is parsed with ruff's parser and the module is type-checked. Every function, method and lambda is lowered to a register IR. Annotations are *checked assertions*: a value entering an annotated position from untyped code is checked at runtime, and where the checker proves a type no check is emitted.
2. **Interpreter** (`crates/lizzard-py`, a PyO3 extension). Functions start in a compact bytecode interpreter whose inline caches record what each instruction sees: attribute layouts, global values, call targets, iterator kinds.
3. **JIT** (`crates/lizzard-jit`). A function that becomes warm (its second call, or 200 iterations of a loop) is compiled with Cranelift, specialized on what the caches saw. Compiled code shares the interpreter's frame layout, so execution moves between the two at any instruction:
   - a loop that warms up continues in compiled code (on-stack replacement);
   - a failed guard *bails out* to the interpreter at that instruction, and too many bail-outs recompile the function without speculating.
4. Compiled code runs **inside CPython**, on the free-threaded 3.14t build or the regular 3.14 build. It cooperates with signals and the free-threaded stop-the-world GC through safepoints at loop back-edges.

`LIZZARD_JIT=0` runs only the interpreter, `LIZZARD_JIT_DEBUG=1` reports compilations, and `LIZZARD_JIT_DUMP=1` prints the generated Cranelift IR. `docs/rev02.md` describes the design and every optimization in detail.

## Supported Python so far

- **Functions:** annotated parameters of every kind (`/`, `*args: T`, keyword-only, `**kwargs: T`), defaults, keyword and `*`/`**` calls, PEP 695 generic functions and `TypeVar`s.
- **Inner functions, closures, lambdas:** real CPython cells, `nonlocal`, late binding, `Callable[...]`.
- **Classes:**
  - single and multiple inheritance (C3 MRO), `super()`, `@staticmethod`, `@classmethod`;
  - properties with getters, setters and deleters;
  - `__slots__`, `@dataclass`, PEP 695 generic classes;
  - operator overloading through dunders (arithmetic, reflected, in-place, comparisons, `__getitem__`, `__iter__`, `__call__`, ...);
  - Python code may subclass Lizzard classes, and overrides are checked.
- **Types:**
  - `int` (unbounded, with a machine-int fast path), `float`, `bool`, `str`, `bytes`, `list`, `dict`, `set`, `tuple`;
  - `T | None` and unions with flow-sensitive narrowing, `Literal`, `Protocol` (including callback and recursive protocols), `Any`, `type X = ...` aliases, `reveal_type`.
- **Statements:**
  - `if`, `for`, `while`, `break`, `continue`, `return`, `assert`, `del`;
  - assignments of every common form, including tuple unpacking and augmented assignment;
  - `global`, imports.
- **Exceptions:** `try`/`except`/`else`/`finally`, `raise ... from`, bare `raise`, `with`. Exception chaining and `sys.exc_info()` behave as in CPython.
- **Generators:** `yield`, `yield from` (with Lizzard or Python iterators), `send`, `throw`, `close`, PEP 479, and closing on free.
- **Expressions:** comprehensions (list, dict, set), f-strings, conditional expressions, chained comparisons, all arithmetic and bitwise operators.
- **Other modules:** stdlib and third-party modules typed through `.pyi` definition files: bundled builtins, `threading`, and NumPy's own annotations. NumPy `float32` arrays get direct element access.
- **Threads:** both the free-threaded (3.14t) and the GIL build of CPython 3.14.

**Not yet:** `match`, `async`, generator expressions, `for`/`while ... else`, `except*`, nested classes, class decorators other than `@dataclass`, metaclass arguments, starred assignment targets, relative imports. Dynamic constructs the checker cannot type are rejected unless written behind `Any`.

## How the JIT makes typed Python fast

The checker's types and the interpreter's caches let compiled code skip most of what CPython does dynamically, while keeping Python's semantics.

**Fewer type checks**
- **Checks the types prove are not emitted.** A call between Lizzard functions skips the parameter checks the arguments' static types prove.
- **Remaining checks are exact type-pointer comparisons** before any `isinstance`.
- **Speculation.** An object's type, version tag and attribute layout are verified once and trusted until something could run arbitrary Python code; a failed guard bails out to the interpreter instead of joining a slow path.
- **Attribute access at fixed offsets.** Reads and writes go straight to CPython's inline values or `__slots__`, guarded by the class's version tag, as CPython's own specializing interpreter does.

**Unboxed numbers**
- **Machine arithmetic.** `float` and `int` arithmetic compiles to machine arithmetic, with an overflow check falling back to Python ints.
- **Unboxed across loop iterations.** Loop variables and accumulators are carried unboxed in registers, so `for i in range(n): t += i * i` allocates nothing.
- **Unboxed across calls and generators.** Values stay unboxed across compiled-in calls (arguments and int results), from generators to the loops consuming them, and through `int`/`float` mixes.
- **Floats updated in place.** A float attribute nothing else references is overwritten in place (`self.x += v * dt` allocates nothing).

**Reference counting only where needed**
- **Inline.** Incref and decref follow CPython's free-threaded object header (owner-thread local count, atomic shared count), with immortal objects skipped.
- **Avoided where the lifetime is known:**
  - borrowed reads of attributes, globals and parameters;
  - a new reference moved into the slot it is stored in;
  - a register's reference handed to `return`.
- **Cold-path slots released only when needed.** Frame slots written only on slow paths are released only if a slow path ran (one flag check instead of one per slot).
- **No atomics when nothing else can reach the object.** A generator owned by the current thread and referenced once is resumed with plain loads and stores, as CPython's own specialized loops do.

**Lizzard → Lizzard calls**
- **Direct calls.** Compiled code calls compiled code natively, building the callee's frame in place with borrowed arguments.
- **Inlining.** Small callees, such as methods, `__init__`s and recursive functions, are compiled into the caller behind a guard, two levels deep. Their frames are only built if the interpreter needs them, on a bail-out.
- **Constructing in a dying instance.** `a = a.add(d)` builds the new object in the memory of the one being released, instead of allocating.
- **Self tail calls become loops.** `return f(...)` inside `f` jumps back to the top, still raising `RecursionError` at CPython's limit.
- **Generators resumed from compiled code.** A compiled `for` over a compiled generator calls the generator's code directly and gets each `yield` back without entering the runtime, including through `yield from`.

**Loops and globals**
- **List iteration inline.** `for` over a list reads the list iterator directly.
- **Cheap safepoints.** Safepoints cost one counter decrement per iteration, and the loop's values stay in registers.
- **Globals.** They are read through caches invalidated by CPython dict watchers.

## Performance

`crates/lizzard-py/bench/bench.py`, release build, best of 3, CPython 3.14.6 free-threaded, Intel Xeon @ 2.80 GHz:

| Benchmark | CPython | Lizzard interpreter | Lizzard JIT | JIT vs CPython |
|---|---|---|---|---|
| int loop, `t += i * i` (2M) | 0.103 s | 0.149 s | **0.004 s** | ~26× |
| `fib(25)` | 0.008 s | 0.028 s | **0.002 s** | ~4× |
| vector objects, `a = a.add(d)` (500k) | 0.081 s | 0.184 s | **0.008 s** | ~10× |
| same, `__slots__` | 0.074 s | 0.171 s | **0.006 s** | ~12× |
| method calls updating attributes, `b.step(dt)` over a list (500k) | 0.037 s | 0.083 s | **0.004 s** | ~9× |
| same, `__slots__` | 0.035 s | 0.083 s | **0.004 s** | ~9× |
| list build and scan (1M) | 0.095 s | 0.173 s | **0.062 s** | 1.5× |
| generator consumed by `for` (2M) | 0.152 s | 0.323 s | **0.040 s** | 3.8× |
| generator pipeline: filter + `yield from` (1M) | 0.084 s | 0.187 s | **0.063 s** | 1.3× |
| tail-recursive int sum (900 deep, ×2000) | 0.219 s | 0.133 s | **0.016 s** | 14× |
| tail-recursive float sum (900 deep, ×2000) | 0.245 s | 0.197 s | **0.014 s** | 18× |

Timings of a few milliseconds are close to the timer's resolution. Machine instruction counts (Valgrind, per unit of work) are exact:

| Unit of work | CPython | Lizzard JIT |
|---|---|---|
| one loop iteration | 715 | 21 |
| one `b.step(0.01)` call | 938 | 70 |
| `fib(10)` | 84.6k | 21.8k |
| one item of the `for`-over-generator benchmark | ~760 | ~310 |

The tail-recursion rows come from a separate script, not `bench.py`. The `list build and scan` row is dominated by CPython's own `list.append`, which Lizzard calls rather than replaces.

## Building and testing

Requirements: Rust, [uv](https://github.com/astral-sh/uv), CPython 3.14 / 3.14t.

```bash
crates/lizzard-py/dev.sh        # builds into .venv-ft (3.14t) and .venv-gil (3.14), runs the tests
```

```bash
crates/lizzard-py/dev.sh ft     # only the free-threaded build
```

For benchmarks, build in release mode first:

```bash
cd crates/lizzard-py && maturin develop --release
```

Then run the suite in each mode:

```bash
python bench/bench.py                         # CPython
```

```bash
LIZZARD_JIT=0 python -m lizzard bench/bench.py  # Lizzard interpreter
```

```bash
python -m lizzard bench/bench.py              # Lizzard JIT
```

The tests run every program in `crates/lizzard-py/tests/programs/` under plain CPython and under Lizzard (compiled, compiled eagerly before any cache is warm, and interpreted), and require identical output. `jit_paths.py`, `gen_paths.py` and `tail_calls.py` drive the compiled code's guards, bail-outs and fallbacks on purpose.

## Layout

| Path | What |
|---|---|
| `crates/lizzard-py-front` | parser, type checker and IR (pure Rust) |
| `crates/lizzard-py` | the CPython extension: import hook, interpreter, generators, runtime helpers for compiled code |
| `crates/lizzard-jit` | the Cranelift compiler |
| `crates/lizzard-py/python/lizzard` | the Python package (`python -m lizzard`, definition files) |
| `crates/lizzard-py/bench` | benchmarks and the instruction-count script |
| `docs/rev02.md` | design and implementation notes |
