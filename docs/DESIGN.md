# slim — Python-syntax toy language with a derived Cranelift JIT

## Context
Greenfield project in `/Users/jmeivier/dev/slim_lang` (empty; rustc 1.96). The goal is a research-style VM inspired by Deegen and Druid. **Each opcode's semantics is written once**, and from that single definition we get both the interpreter and a hot-function Cranelift JIT. The transitions between them (JIT entry, OSR, deopt, exception exits, generator resume) are generated rather than hand-written.

**Sequencing: language first, JIT later.**
- Milestones 1–6 build the complete language on the interpreter, so we can see how it feels.
- From day one, the interpreter's ops are written as generic `Backend` semantics, so the JIT (milestones 7–9) can be derived later without rewriting them.
- Nothing Cranelift-related is built until then.

**Refcounting is deterministic and user-scheduled, for realtime use.** Dropping the last reference never frees anything inline. Dead objects are queued, and `rc_run()` does the freeing and bookkeeping at a moment the program chooses.

Language: Python syntax with these differences:
- **Rust-style types**: data in `struct`, behaviour in `impl Type:` and `impl Trait for Type:` blocks, nominal **traits**, no inheritance (delegation is written by hand)
- mandatory typing for struct fields and methods, with annotations checked at runtime
- **explicit ownership**: moves, `++` for a new reference, and `+ keep` parameters, which keeps refcount traffic minimal
- Rust-style `Option<T>` with `Some(v)` / `None`. There's no `unwrap`: you either narrow the type or call `some_or(...)`.
- Rust-style lambdas `|x| expr` / `|x| { statements }`, where the multiline body sets its own indentation level
- **expression-oriented like Rust**: the last line of a block is its value and an implicit `return`, and `if`/`try` blocks produce values
- generators, Python-style exceptions, and real **no-GIL multithreading**

## Language surface (v1)
```python
trait Log:
    def log(self, msg: str) -> (): ...           # `...` body = required method
    def hello(self) -> ():                       # default method
        self.log("hello")

struct Console                                   # no fields

impl Log for Console:
    def log(self, msg: str) -> ():               # msg is borrowed (no `keep`)
        print(msg)

struct Service:
    out: Console                                 # all fields declared + typed
    last: Option<str>
    count: int                                   # stored as raw i64

impl Service:
    def new(out: Console + keep) -> Self:        # associated function (no `self`)
        Service { out, last: None, count: 0 }    # literal: every field; `out` moves in

impl Log for Service:
    def log(self, msg: str) -> ():               # hand-written delegation
        self.out.log(msg)                        # temp borrow of a field: fine
        old <- self.last <- Some(msg++)          # swap out old value, store new ref
        if old is not None:                      # narrows `old` to str
            print(old)
        self.count += 1                          # copy types use plain `=`

s = Service::new(Console {})                     # fresh object moved into keep param
name = "svc"
t = name++                                       # new reference; `name` still usable
u = name                                         # move; `name` dead from here on
label = s.last++.some_or("none")                 # no unwrap(); some_or consumes its receiver

def classify(n: int) -> str:
    if n > 100:
        return "huge"                            # early return still allowed
    size = if n > 10:                            # `if` is an expression
        "big"
    else:
        "small"
    size                                         # last line = implicit return

double = |x: int| x * 2
report = |n: int| {
    if n > 10:
        print("big")
    n                                            # lambda block value
}
```
- **Copy types:** `int`, `bigint`, `float`, `bool`, plus `None` and `()`. They are always copied and never touch reference counts.
  - `int` is a 64-bit integer, and overflow raises `OverflowError`.
  - `bigint` is a separate arbitrary-precision type.
- **Move types:** everything else, i.e. str, tuple, list, dict, instances, generators, exceptions, Thread, Lock and Channel.
- **`Option<T>`, written like Rust.** Its values are `Some(v)` and `None`, and `None` is valid only where an `Option` is expected. There's no `unwrap()`. You get at the value in one of two ways:
  - **Narrowing.** `if x is not None:` / `if x is None: ... else:` narrows `x` to `T` inside the branch, and so does `while x is not None:`. This works on locals and on field paths (`self.head`). Each narrowed use of a field re-checks the tag, so if a callee or another thread has cleared the field in the meantime you get an error, not a crash.
  - **`some_or`.** `opt.some_or(default)` consumes its receiver, which makes it a `keep` call. On a field you write `self.f++.some_or(d)`. On an `Option` of a copy type it simply copies.
- **Representation of `Option`.** `Some(v)` is represented as `v` itself, so it costs nothing, and `None` is an immediate. Nested `Option<Option<T>>` is a compile error in v1.
- **Generic types use angle brackets** in annotations: `list<int>`, `dict<str, int>`, `Option<Node>`.
- A function that returns nothing is declared `-> ()`.
- **Expression-oriented blocks, like Rust:**
  - The value of a block is the value of its last line, so the last line of a function or lambda body is an implicit `return`. Explicit `return` stays available for early exits.
  - `if/elif/else` is an expression whose value is the last line of the branch taken. An `if` without `else` has the value `()`.
  - `try/except/else` gives the value of the last line of the path taken, and `finally` doesn't change it.
  - `while`, `for`, assignments, `<-` and `del` all have the value `()`.
  - A compound expression can appear on the right-hand side of an assignment, or as the last line, using normal indented blocks: `x = if c:` followed by an indented block. The inline form `a if c else b` is also kept.
  - In a function declared `-> ()`, the last line's value is discarded (dropped), so you never have to write a trailing `()`. For any other return type, the implicit value is type-checked like an explicit `return`.
  - The implicit return is a move, like `return`: a last line of `self.name` (a move type) has to be written `self.name++`.
  - In a generator, the last line's value is not implicitly returned. Only an explicit `return x` sets `StopIteration.value`.
- **Lambdas:**
  - `|a, b: int| expr` returns `expr`. Writing `|x| { ... }` instead allows full statements, and its value is the last line. `return` can still be used for early exits.
  - A `{` directly after the parameter list opens a block, not a dict literal; write `|x| ({...})` to return a dict.
  - Inside the block the lexer pushes a nested indentation context. The first non-blank line after `{` sets the base indent, the usual INDENT/DEDENT rules apply relative to it, and `}` ends the context.
  - Parameters follow the same rules as functions: borrowed by default, with `+ keep` allowed. Captures move move-type values in (`++` a value first to keep it) and copy copy-type values.
  - A lambda is itself a move-type object.
- **Mandatory struct typing.**
  - Every field is declared in the `struct` with a type.
  - Every method annotates all its parameters except `self`, plus its return type.
  - A struct value is created by a literal that gives every field exactly once: `Point { x: 1.0, y }` (`y` is shorthand for `y: y`). There is no `__init__`; constructors are associated functions (`Point::new(..)`). Field values move into the struct, like `keep` arguments.
  - Annotations stay optional on free functions and locals.
- **Runtime checks.** Annotations are checked on parameters at call entry, on return values, on annotated locals, and on field stores; a mismatch raises `TypeError`.
  - `x: T` matches if `type(x) is T`, if T is a trait in `type(x)`'s `impl` list, or if T is `Option<U>` and x is `None` or matches U.
  - Generics are reified (see "Runtime types" below): `list<int>` checks the list's own type, `list` accepts any list.
- **Structs, impls and traits (10d).** As in Rust: `struct Name<T>:` declares the fields (or `struct Marker` none), `impl<T> Name<T>:` adds methods (first parameter `self`) and associated functions (no `self`, called `Name::f(..)` or `Name::<int>::f(..)`), and `impl<T> Trait<args> for Name<T>:` implements a trait. A marker impl has no body (`impl Exception for AppError`).
  - **Traits** are defined as before: `trait Shape:` with required methods (`...` body) and defaults (real bodies). `Self` in a trait's signatures is the implementing type; in a default method it is `self`'s runtime type. Traits are used in annotations (`s: Shape`, `list<Shape + Send>`), as bounds (`T: Shape`) and with `isinstance`.
  - **Checks.** An impl of a trait must define every required method with the trait's parameter types (after substituting `Self` and the trait's arguments) and a compatible return type, and nothing the trait doesn't declare; defaults are inherited. `Send` is never implemented (it follows the fields), and a struct cannot implement another struct.
  - **Methods** of several `impl` blocks are merged; a name defined twice (or both a field and a method) is an error. Operator dunders (`__add__`, `__lt__`, ...) may be inherent methods; `__drop__` only in `impl Drop for T`. Structs are sealed (fixed slots, no monkeypatching).
  - **Impls cover the whole struct**: `impl<A, B> Pair<A, B>` with the struct's own bounds (any names). Specialized or conditional impls are not supported yet.
  - **Items are order-independent.** Traits, structs (with their impls) and associated functions are defined at module start, before the module's statements run, so an impl may come before its trait.
  - Calling a struct (`Point(1.0, 2.0)`) is an error; `class` is a syntax error that explains the replacement.
- **Exceptions:** `try/except/else/finally`, `raise` (which moves the exception), `raise X from Y`, `except (A, B) as e` (`e` is an owned local). Exception groups such as `LookupError` and `ArithmeticError` are built-in traits. A user exception is a struct implementing `Exception` (a `msg: str` field is shown by `str(e)`): `raise AppError { msg: "bad", code: 7 }`; `raise Done` works for a struct without fields. Builtin exception classes are called: `ValueError("x")`.
- **Generators:** `yield` (which moves the value out), `yield from`, `.send()`, `.throw()`, `.close()`, `StopIteration`. `for` loops use `__iter__` / `__next__`; `__next__` returns an owned value, and a list iterator returns `++` of each element.
- **Threads:** `spawn(f, args...)` is a `keep` call, so objects move or `++` into the thread. Built-ins `Lock` and `Channel` (`send` is `keep`).

## Ownership model (compile-time checked, minimal refcounting)
| construct | meaning | RC work |
|---|---|---|
| object creation | owned by the destination, rc = 1 | none |
| `y = x` (move type) | move; `x` dead | none |
| `y = x++` | new reference; both usable | +1 |
| `foo.bar <- obj` | move into field; `obj` dead; old field value dropped | -1 (old, deferred) |
| `x <- foo.bar <- y` | swap: old value moves to `x`, `y` moves in | none (atomic swap) |
| `x = foo.bar` (move type) | **compile error**: use `++` or `<-` swap | — |
| `foo.bar.m()`, `f(foo.bar)` to a non-keep param | temporary borrow; `foo` stays fully usable | none |
| `f(obj)`, non-keep param | borrow; `obj` still usable | none |
| `f(obj)`, `p: T + keep` param | move into callee; `obj` dead | none |
| `f(obj++)` | caller keeps `obj`, callee gets a new ref | +1 |
| `return x` / last-line value / `yield x` | move out (owned locals only; borrowed values need `++`) | none |
| scope exit / reassignment of an owned local | drop | -1 |

**Rules enforced by the compiler**
- Locals are never borrows. A non-keep parameter is borrowed for the whole call, so it can't be moved or stored anywhere without `++`.
- Use-after-move of a local is a compile error. The analysis is flow-sensitive; a maybe-moved local can't be used, though it can still be dropped.
- Generator functions, nested functions and lambdas outlive the call, so their move-type parameters must be `+ keep`. Nested functions and lambdas capture by move.

**Dynamic calls.** When a callee isn't known statically, the call site can't tell whether a parameter is `keep`. This is the case for first-class function values; methods on typed receivers and module-level `def`s are resolved statically. For dynamic calls:
- the `Call` op carries an "owned args" mask
- the callee's keep mask is compared against it at runtime
- a moved register becomes `MOVED`, and a later use raises `OwnershipError`
- passing a borrowed value to a keep parameter raises `OwnershipError`

**Borrows freeze nothing; the only rule is that a borrow can't escape.** Memory safety comes from the runtime, so the compiler only checks that a borrow doesn't escape its expression or call:
- it can't be bound to a local, stored, returned, yielded or captured without `++`
- no `yield` inside an expression that holds a borrow

Everything else is allowed while a borrow of `p.a.b` or `c[i]` is live. That includes reading, writing or moving `p` and `p.a`, other borrows such as `p.a.m(p.b)`, `++`, and writes from aliases or other threads. There's no hidden RC traffic anywhere in the language.

**Why that's safe: every free is epoch-gated.**
1. Every queued item is stamped with the global epoch when it's enqueued. That covers objects whose count reached 0 (whether from a local drop, a moved parent being dropped, or a swap-out followed by a drop) and every deferred decrement of an overwritten field or element.
2. A call that receives a field or element borrow pins the thread's epoch for the duration of the call. That's one push/pop on a thread-local stack, published with a relaxed store.
3. Borrows that don't span a call are covered by QSBR, because the thread hasn't passed a safepoint yet.
4. `rc_run` processes only items older than every thread's oldest pin and last safepoint.

The result: a borrowed object can't be freed while any borrow of it is live, even if the parent is moved and dropped by a callee, or overwritten by another thread. The borrower keeps working with the object it borrowed, a snapshot.

**Cascades.** When `rc_run` frees an object, the children whose count reaches 0 are freed in the same run without waiting. They were reachable only through a parent whose grace period has already passed.

**Runtime representation**

### Scheduled cleanup: `rc_run()`
A decrement never frees anything inline; it only does bookkeeping.

**What a decrement does.** When a count reaches 0, the object goes onto the current thread's **pending-free queue**. This costs one atomic decrement plus one push. When a `<-` store overwrites a value in shared memory, the old value is pushed onto a **deferred-decrement queue**, stamped with the current epoch.

**What `rc_run()` does.** It does all the expensive work, at a time the program chooses:
1. It applies deferred decrements whose grace period has passed, meaning their epoch stamp is older than every thread's oldest borrow pin and last safepoint. Objects that reach 0 join the pending-free queue.
2. It pops the pending objects whose grace period has passed. For each one it runs the struct's optional `__drop__(self) -> ()` destructor (from `impl Drop for T`), decrements the object's children (which may queue more objects, handled in the same run), and returns the memory to the allocator.

**Built-ins**
| function | effect |
|---|---|
| `rc_run() -> int` | Drains the queues completely and returns the number of objects freed |
| `rc_run(limit) -> int` | Frees at most `limit` objects; the rest wait for the next call. This gives bounded pause time in realtime loops |
| `rc_pending() -> int` | Number of queued objects |
| `rc_auto(on: bool) -> ()` | Per-thread switch, on by default. When on, the VM calls `rc_run(N)` itself at function returns or loop back-edges once the pending count passes a threshold. Realtime threads turn it off and call `rc_run` explicitly, e.g. once per frame or tick |

**Further rules**
- **Exceptions in `__drop__`** are printed and ignored, as in Python.
- **Thread exit** drains that thread's queues.
- **Interpreter shutdown** runs `rc_run()` until both queues are empty.
- **Safepoints** cross the QSBR epoch automatically. That's a single store of the current epoch, not a free, so a realtime thread still controls when frees happen.
- **Why the grace period matters.** Epoch pins and QSBR together cover every live borrow, so a lock-free borrowed read on any thread never sees memory that `rc_run` has freed. That includes an `rc_run` called by the borrower's callee.
- **Single-threaded programs without pins.** If a program has one thread and no pins are live, the grace period has always passed, and `rc_run` frees everything immediately.
- Frames separate owned registers from borrow registers.
- Registers start out `UNINIT`, and a move sets the source register to `MOVED`.
- Dropping an `UNINIT` or `MOVED` register does nothing. Unwinding drops every owned register using a per-function bitmask, so no per-pc maps are needed.

## Architecture

### Cargo workspace
| crate | contents |
|---|---|
| `slim-syntax` | Indentation-aware lexer with an indent-context stack (lambda `{}` blocks), `<-`, `++`, `+ keep`, `<...>` generics in type position; recursive-descent parser, AST, spans/diagnostics |
| `slim-compiler` | Scope analysis (including lambda captures), **type/ownership checker** (moves, borrows, keep masks, definite assignment, `Option` narrowing), AST → register bytecode, exception tables, liveness, the borrow-escape check, and insertion of epoch pins |
| `slim-runtime` | `Value`, object header + atomic RC, QSBR, classes/traits, str/tuple/list/dict/bigint/boxed float, exceptions, `Thread` state, `extern "C"` helpers |
| `slim-vm` | `ops.rs` opcode table + **semantics written once**; `backend/mod.rs` trait; `backend/interp.rs`; dispatch loop; frames; generators |
| `slim-jit` | `backend/jit.rs` Cranelift backend; compile-queue thread; entry/exit generation; code installation |
| `slim` (bin) | CLI: `slim run file.slim [--jit=off\|auto\|always] [--dis]` |
| `tests/` | Golden `.slim` programs with `.out` files (including expected compile errors), run under every JIT mode |

Dependencies:
- `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native`
- `parking_lot`, `num-bigint`
- dev only: `loom`

### Opcode table (declared once)
A `macro_rules! opcodes { Add { dst: Reg, lhs: Reg, rhs: Reg, site: Site } ... }` in `slim-vm/src/ops.rs` generates:
- the `Op` enum
- the encoder and decoder
- the disassembler
- interpreter dispatch (`sem::add::<Interp>`)
- JIT dispatch (`sem::add::<Jit>`)

The ownership ops are ordinary opcodes, so the JIT sees every refcount operation explicitly:
- `Move`, `Dup` (`++`), `Drop`
- `StoreMove` (`<-`), `Swap` (`x <- f <- y`)
- `Pin` / `Unpin` (epoch pins around calls that receive a field or element borrow)

### Backend trait (the core idea)
```rust
type Flow<T> = Result<T, Exit>;           // Exit = Raise | Deopt | Return | Yield
pub trait Backend {
    type V: Copy; type Bool: Copy;
    fn reg(&mut self, r: Reg) -> Self::V;
    fn set_reg(&mut self, r: Reg, v: Self::V);            // JIT: write-through to frame
    fn konst(&mut self, bits: u64) -> Self::V;
    fn has_tag(&mut self, v: Self::V, t: Tag) -> Self::Bool;
    fn iadd_ovf(&mut self, a: Self::V, b: Self::V) -> (Self::V, Self::Bool);
    fn load_slot(&mut self, obj: Self::V, idx: Self::V) -> Self::V;   // atomic Acquire, borrowed
    fn swap_slot(&mut self, obj: Self::V, idx: Self::V, v: Self::V) -> Self::V;
    fn incref(&mut self, v: Self::V); fn decref(&mut self, v: Self::V);
    fn ic(&mut self, s: Site) -> (Self::V, Self::V);      // interp: atomic load; JIT: iconsts from snapshot
    fn record(&mut self, s: Site, v: Self::V);            // type feedback; JIT: no-op
    fn if_else<T: Merge<Self>>(&mut self, c: Self::Bool,
        t: impl FnOnce(&mut Self) -> Flow<T>, e: impl FnOnce(&mut Self) -> Flow<T>) -> Flow<T>;
    fn speculate<T: Merge<Self>>(&mut self, s: Site, c: Self::Bool,
        fast: impl FnOnce(&mut Self) -> Flow<T>, slow: impl FnOnce(&mut Self) -> Flow<T>) -> Flow<T>;
    fn call<const N: usize>(&mut self, h: Helper<N>, a: [Self::V; N]) -> Flow<Self::V>;
}
```
**How each backend implements it**
- **Interpreter backend.** `V = Value`. `if_else` and `speculate` run the closures directly, so after monomorphization each op compiles to code as good as a hand-written handler.
- **JIT backend.** `V = cranelift::Value`.
  - `if_else` emits blocks and merges results through block params or `Variable`s.
  - `Err` from a branch means that block is already terminated.
  - `speculate` emits guard + deopt when feedback says the slow path is cold, and a full if/else otherwise.
  - `call` emits a call to the `extern "C"` helper plus a check for the exception sentinel.

**Rules for writing semantics**
- **Static vs dynamic.** Operands and arg counts are plain Rust values; only runtime data is `B::V`. Loops over static counts unroll, and dynamic loops go into helpers.
- **Restartable ops.** Every guard and deopt comes before any side effect. A debug audit checks this.
- **Escape hatch.** Ops may be helper-only. Calls, generators, raise and containers start out that way.
- **Inline caches.** One `AtomicU64` per site packs `(class_id, slot/method idx)`. Classes are sealed, so ICs never need invalidating, and there's no bytecode quickening.
- **Free type facts from class typing.** Field stores are type-checked, so after an IC hit the type of a field load is known. Method parameters are checked on entry. The compiler passes these facts to the JIT, which skips the tag checks and works on raw i64/f64.

### Values and memory
- **Generic `Value` word: 64-bit pointer tagging.**
  - Pointers are 8-aligned with low bits `000`.
  - An `int` whose value fits in 63 bits is an immediate (low bit `1`); the rare full-width i64 is boxed. Arithmetic overflow past i64 raises `OverflowError`.
  - `bool`, `None`, `()`, `UNINIT` and `MOVED` are `x10` immediates. `Some(v)` is just `v`.
  - **A float is boxed when it is passed around as a generic Value.** Boxes are immutable, so copying one is a hidden incref; that's allowed because floats are a copy type.
- **Typed storage is raw.** Fields declared `int` or `float` store a raw i64/f64 in their slot, and JIT code keeps ints and floats unboxed in machine registers. Boxing happens only at generic boundaries such as untyped locals, containers and dynamic calls.
- **Object header:** atomic refcount with an immortal bit (classes, functions, code objects and string constants), plus a class pointer. Slots are `[AtomicU64]`: stores use Release, loads use Acquire.
- **Thread safety without a GIL.**
  - Deferred decrefs are gated by epochs (QSBR plus borrow pins), and only `rc_run` frees (see the cleanup section above).
  - Field reads are lock-free borrowed loads, and they're memory-safe even while other threads write.
  - Swaps use an atomic `swap` and need no deferral.
  - Pending queues are per-thread vectors, so enqueueing takes no lock.
  - list and dict take a per-object `parking_lot` lock in v1.
  - Biased refcounting is a later option.
- **Cycles leak in v1.** Ownership makes cycles less likely: to build one you have to write `++` explicitly.

### Frames, JIT transitions, exceptions, generators
- **Frame layout:** registers (owned and borrow sets) + pc + code pointer, shared by the interpreter and the JIT.
- **JIT function signature:** `extern "C" fn(*mut Thread, *mut Frame, entry_pc: u32) -> ExitStatus`.
- **Entries:** the entry block does a `br_table` over pc 0, loop headers (OSR) and yield-resume points, using the shared liveness analysis.
- **Exits:** registers are write-through, so an exit stores only the pc and returns `Return`, `Yield`, `Raise` (exception pending in `Thread`) or `Deopt`.
  - The interpreter resumes at that pc.
  - Unwinding always happens in the interpreter: it reads the exception table and drops owned registers using the bitmask.
  - Helpers never panic; they return sentinels.
- **Exception table:** zero-cost style. `finally` is duplicated on the normal and exceptional paths.
- **Generators:**
  - The generator object owns its frame on the heap; move-type parameters are `keep`.
  - Resume writes the sent value into the yield's destination register and enters at `resume_pc`.
  - `throw` and `close` resume in the interpreter.
  - An atomic "running" flag prevents two threads from resuming the same generator at once.
- **Hot detection:**
  - Relaxed atomic counters on function entry and loop back-edges.
  - Past a threshold, the function goes to the **background compiler thread**, which owns the `JITModule`.
  - Code is published with an `AtomicPtr` (Release), and retired code is leaked in v1.
  - There's a native-stack recursion limit.

## Milestones (each ends with all golden tests green)

### Phase A: the language, on the interpreter
1. **Front end and interpreter.**
   - Workspace, lexer/parser (including `<-`, `++`, `+ keep`, `-> ()`, `<...>` types), compiler, opcode macro, disassembler.
   - The `Backend` trait with only the interpreter implementation.
   - int, bool, `()`, functions, globals, if/while/for-range, `print`, str literals.
   - Expression-oriented blocks: implicit last-line return, `if` as an expression (including on the right-hand side of an assignment), and value-or-discard compilation modes in the code generator.
   - A golden-test harness that includes expected compile errors.
2. **Ownership and RC framework.**
   - The ownership checker for locals and parameters (move, `++`, keep, use-after-move errors).
   - `Move`/`Dup`/`Drop` ops, the per-thread pending-free and deferred-decref queues, `rc_run`/`rc_pending`/`rc_auto`, `__drop__`.
   - A debug refcount audit.
3. **Classes and typing.**
   - Classes with typed slots (raw i64/f64), `<-` stores, swap, the borrow-escape check, and epoch pins with grace-gated `rc_run`.
   - Method ICs and traits (`class C(impl T)`, required/default methods, completeness check).
   - `Option<T>`/`Some`/`None` with narrowing and `some_or`, definite assignment in `__init__`, and runtime annotation checks.
4. **Exceptions, core types and lambdas.**
   - Exception tables, raise/try/finally, unwinding drops, tracebacks.
   - str/tuple/list/dict/float/bigint.
   - Lambdas: the lexer indent-context stack, move captures, closure objects.
   - Dynamic-call ownership masks.
5. **Generators.** yield/send/throw/close, `yield from`, the iterator protocol, and heap generator frames.
6. **Threads.** `spawn`/`join`/`Lock`/`Channel`, QSBR epochs gating deferred decrefs, immortal objects. Add stress tests and `loom` tests for QSBR and `rc_run`.

**Checkpoint:** use the language and revise the syntax and semantics before starting on the JIT.

### Phase A+: interop groundwork (built after the checkpoint)
7. **Epoch soundness:** rc flag bits, `rc::retire`, poison mode.
8. **Lock-free-read `list` and `dict`;** no negative indexes.
9. **`array<T>`:** fixed-size packed scalars. **9b:** static element types.
9c. **Runtime types:** reified generics for containers. **9d:** user generics with bounds. **9e:** thread locality (redesigned in **9f**/**9g**: `Send` as in Rust).
10. **Rust bindings** (static embedding). **10b:** generic native classes. **10c:** native traits and `impl Trait for NativeClass`. **10d:** `struct` + `impl` blocks replace `class`.

### Phase B: the derived JIT (later)
11. **Spike, then the JIT backend.**
   - Spike `cranelift-jit` on aarch64-darwin: call an `extern "C"` helper, a `br_table` multi-entry function, and `sadd_overflow`.
   - The JIT `Backend` for the existing semantics, entering at pc 0 only.
   - `--jit=always` differential testing: every golden test runs in both modes, and outputs must match.
12. **Transitions.** OSR loop-header entries, generator resume entries, `Raise`/`Deopt` exits, the background compiler thread, and `AtomicPtr` installation.
13. **Speculation.** Feedback-driven `speculate` + deopt, typed unboxed float/int paths, and benchmarks (fib, nbody, a generator pipeline, a parallel sum). A guard on a container's runtime type makes its element loads typed; `array`'s length and data pointer never change, so bounds checks hoist out of loops.

## Biggest risks / mitigations
| risk | mitigation |
|---|---|
| Refcount bugs across exits | RC is explicit in the bytecode; write-through registers; a debug refcount audit; Miri on the runtime, without the JIT |
| Ownership-checker gaps, such as a borrow escaping its call or a missing pin | The escape check is expression-local and easy to test; a debug mode that poisons freed objects; stress tests where callees and other threads move, drop or overwrite the parents of live borrows |
| The Backend abstraction leaking | Helper-only escape hatch |
| Non-restartable ops | Guard-before-effect rule, plus a debug check |
| ARM memory ordering | Explicit Acquire/Release; loom |
| Scope | The differential harness stays green at every milestone |

## Verification
- `cargo test --workspace` runs:
  - unit tests for the lexer, parser, compiler and ownership checker
  - golden tests: every `tests/programs/*.slim` is checked against its `.out`. In Phase B the same tests also run under `--jit=always` and `--jit=auto` (low threshold).
  - RC tests:
    - with `rc_auto(False)`, objects stay alive until `rc_run()`
    - `rc_run(limit)` frees at most `limit` objects
    - `__drop__` order and output
    - `rc_pending()` counts
  - expression tests: implicit returns, `if`/`try` values, an `if` without `else` producing `()`, the discarded last value in a `-> ()` function, and `TypeError` on a wrongly typed implicit return
  - negative tests: use-after-move, `x = foo.bar` without `++`/`<-`, a borrow escaping (bound, stored, returned or yielded without `++`), moving a borrowed param, missing annotations, using an `Option` without narrowing, `None` assigned to a non-Option, lambda indentation errors, `TypeError`/`OverflowError`/`OwnershipError` messages
- A debug-build refcount audit at exit: with the immortals excluded, every non-leaked object must reach rc 0.
- `slim run --dis file.slim` lets you inspect bytecode, with every RC op visible.
- From M6, thread stress tests: N threads racing `<-` stores and borrowed reads on shared objects, and shared list/dict under Lock. `cargo +nightly test -Zsanitizer=thread` on the runtime.
- In Phase B, `SLIM_JIT_DUMP=1` dumps the Cranelift IR, and benchmarks in `benches/` compare interpreter and JIT time. The JIT should be several times faster on integer and float code.

## Implementation notes (Phase A as built)
Refinements made while building milestones 1–6. Where they conflict with the sections above, these take precedence.
- **Crates.** The runtime lives in `slim-vm` (no separate `slim-runtime` crate). The crates are `slim-syntax`, `slim-compiler`, `slim-vm` and `slim` (CLI). `slim-jit` arrives with Phase B.
- **Minimal freeze.** Borrows freeze nothing, with one exception: a local can't be moved into a call that also borrows that same local, e.g. `f(x, x)` where the first parameter is `keep`. This is checked statically, and at runtime for dynamic calls.
- **Epoch-gated frees.** Every free is stamped with an epoch, whether it comes from a refcount reaching 0 or a deferred decref. `rc_run` processes only entries older than the horizon, which is the minimum over threads of the last safepoint and the oldest pin.
  - Children freed while releasing a parent cascade in the same run.
  - Entries left by exited threads go to an orphan queue.
- **Safepoints:** function entry, loop back-edges and generator resume. Blocking calls (join, lock acquire, channel receive) take the thread offline.
- **Pins.** Only calls whose arguments include a heap borrow (a field, an element, or a borrowed module value) pin the epoch. Loads of immortal globals don't.
- **`some_or`** borrows its receiver and returns either the default or a new reference (`self++`).
- **`and` / `or`** always produce a `bool`.
- **Generators.** Generators driven by `for` / `next` are sent `None`.
- **New statements:** `with ctx [as name]:` (`__enter__` / `__exit__`) and `assert cond[, msg]`.
- **Loop heads.** A local assigned in a loop body counts as "maybe initialized" at the loop head, so reassigning it drops the previous iteration's value.
- **Module-level `def`s** are static functions (a copy type), so `f = g` is allowed.

## Implementation notes (milestones 7–10)
- **Refcount word flags.** Bit 60 is `LOCAL` (see "Thread locality" below). Bit 63 is `IMMORTAL`. Bit 62 is `REVIVED`, set by an increment from zero: a borrow of shared memory took a new reference to an object already queued for freeing. Bit 61 is `QUEUED`: the object has a pending-free entry. When an entry's grace period has passed, `rc_run` frees it only if the count is still zero and it was not revived. A revived object waits a new grace period; a live one is dropped from the queue and is queued again when it next reaches zero.
- **`rc::retire(ptr, free)`** frees raw memory (an old list buffer, an old dict table) after the same grace period as objects.
- **Poison mode** (`SLIM_POISON=1`, debug builds): freed objects are overwritten and leaked, and any later refcount operation on them panics. The golden tests run in this mode.
- **Containers without read locks.** `list` and `dict` reads (`l[i]`, `len`, `in`, iteration, lookups) take no lock. Writers serialize on a one-bit spin lock and never run user code while holding it. Values and buffers leave through the epoch system, so a racing reader may see an old, new or duplicated element, or get an `IndexError`, but never freed memory or memory out of bounds. `sort` compares outside the lock and raises `RuntimeError` if the list changed meanwhile. Dict iteration works on a snapshot.
- **No negative indexes** on any sequence: `0 <= i < len` or `IndexError`. `pop()` returns `Option<T>` (`None` when empty); `first()` / `last()` return `Option<T>` holding a new reference.
- **`array<T>`** (`i8`…`u64`, `f32`, `f64`, `bool`): fixed length, packed memory accessed with relaxed atomics, bounds-checked. Out-of-range integers raise `OverflowError`. `a.view(start, n)` shares memory and keeps its base alive.
- **Runtime types (9c).** Types are interned, immortal values (`rtype.rs`); equal types are the same pointer.
  - Every list, dict, iterator, generator and channel carries its full type (`list<int>`, `dict<str, list<int>>`, `Iterator<int>`, `Channel<str>`).
  - The VM checks every store into a container against its element type: index stores and swaps, `append`, `insert`, `extend`, dict stores, generator yields and channel sends. An `int` stored where a `float` is expected is converted. A store the compiler proves correct skips the check (`StoreIndex ... proven=true` in `--dis`). The check never depends on the compiler, so a compiler bug can only cost speed.
  - Annotation checks compare whole types and never scan elements.
  - **Invariance:** a `list<int>` is not a `list<float>` or a `list<any>`. A bare `list` (or `dict`, `tuple`, `Iterator`, `Channel`) accepts any of its kind. `any` accepts every value except `None`.
  - **Where types come from.** List and dict literals infer their element type from their items. An empty literal, `list()`, `dict()` and `Channel()` take it from the context: an annotation, a parameter, a field, a return type or an enclosing element type. Without context they are a compile error. Mixed literals need an annotation (`list<any>`).
  - **Types are values.** `list::<int>`, `dict::<str, int>` and `Channel::<str>` are type values; calling one builds an empty container or converts an iterable, checking each item. Builtins compute their result types from their inputs: `split` gives `list<str>`, `d.items()` gives `list<tuple<K, V>>`, and `sorted(xs)` and `list(xs)` take the item type of `xs`.
  - **Tuples** are typed structurally (`tuple<int, str>`); indexing with a literal and unpacking keep the item types.
  - **Generators** declare their item type with `-> Iterator<T>`; otherwise it is inferred from the `yield`s (`any` if unknown).
  - **Static argument checks.** Arguments are checked against the parameter types of statically known callees.
- **User generics (9d).** Structs, traits and functions take type parameters with trait bounds: `struct Stack<T>` with `impl<T> Iterable<T> for Stack<T>`, `trait Source<T>`, `def max_of<T: Ord + Hash>(xs: list<T>)`. Explicit type arguments in expressions use a turbofish: `Stack::<int>::new()`, `Stack::<int> { items: [] }`, `first::<str>`.
  - **Reified.** An instance carries its instantiated type (`Stack<int>`). Fields typed with a parameter are checked against the instance's type arguments, and annotations compare whole types (`Stack<int>` is not `Stack<str>`; a bare `Stack` accepts any).
  - **Generic functions and methods** receive their type arguments as hidden leading parameters (after `self`). The compiler passes them when it can infer them (from a turbofish, the arguments, or the expected result type). Otherwise the callee infers them at runtime from the arguments' types. Bounds are checked in both cases. A generic function used as a value infers its type arguments when it is called, or can be instantiated first (`f::<int>` binds them).
  - **Generic struct literals** need statically known type arguments: a turbofish, the context (`s: Stack<str> = Stack { items: [] }`) or inference from the field values (`Box { value: 5 }` is a `Box<int>`).
  - **Associated functions** are module functions named `Type::f` whose leading type parameters are the impl's, then their own; the explicit arguments of `Stack::<int>::new()` fill the impl's, and the rest are inferred like any generic call (`t: Stack<str> = Stack::new()`).
  - **Methods** of generic structs load the struct's type arguments from `self` at entry (`LoadTypeArg`); default methods of generic traits load the trait's arguments as implemented by `self`'s struct (`LoadTraitArg`) and `Self` with `LoadTypeOf`; closures capture the type parameters in scope. Bodies are compiled once (no monomorphization).
  - **Strict bounds.** In a generic body a value of type `T` can be moved, copied, stored in `T` slots, returned, compared with `==`/`is`, printed and passed where `T` or `any` is expected. Everything else needs a bound: `< <= > >=`, `sorted`, `min`, `max` need `Ord`; `+ - * / // % **` and unary `-` need `Num` (both operands `T`; `/` is untyped); `len` needs `Len`; `for` needs `Iterable<U>` (items are `U`); dict keys need `Hash`; method calls need a user trait bound that declares the method. Field access, indexing and calling a `T` are compile errors.
  - **Builtin traits:** `Ord` (numbers, `bool`, `str`, tuples of `Ord`; structs implementing `__lt__`), `Num` (implies `Ord`; numbers; structs implementing the arithmetic dunders), `Hash` (everything except `list`, `dict`, `array` and tuples containing them), `Len`, `Iterable<U>` and `Iterator<U>` (implies `Iterable<U>`), and `Drop` (`__drop__`). Structs implementing `Ord`/`Num` work with the operators through their dunders. User traits cannot reuse these names.
  - **Trait arguments** of an impl may refer to the impl's parameters (`impl<T> Iterable<T> for Stack<T>`). A struct implements a trait at most once.
  - **Known limitation:** `Option<T>` with `T = Option<U>` is not supported (the inner `None` cannot be told apart).
- **`Send` (9e, redesigned in 9f and 9g).** As in Rust, a type is `Send` (its values may cross threads) unless it holds something that isn't. A value of a non-`Send` type is *local*: it stays on the thread that created it. Internally, the compiler's `Ty::Local`, `RtType.local` and the rc `LOCAL` bit all mean "not `Send`".
  - **`Send` is inherited structurally.** It is an auto trait that cannot be implemented explicitly, and there is no `Local` trait.
    - A struct is not `Send` when a field type is not (computed to a fixpoint, since fields may name structs declared later).
    - A generic instantiation is not `Send` when a type argument is not: `Box<T>` is `Send` exactly when `T` is. The same goes for `list`, `dict`, `Option`, tuples and `Iterator<T>`.
    - The roots are native bindings over Rust types without `Send` (`class_not_send`) and the erased types below.
  - **Erased types follow Rust's `dyn Trait`.** `any`, `function`, `Iterator<T>` and trait types (`Shape`) may hold values that aren't `Send`. `X + Send` (`any + Send`, `function + Send`, `Iterator<int> + Send`, `Shape + Send`) restricts them to values that are. An `X + Send` can be used as an `X`, but not the other way round. Inside containers the two are different types (`list<any>` ≠ `list<any + Send>`).
    - Consequences:
      - `list<any>` is not `Send`, just as `Vec<Box<dyn Any>>` isn't, so sending one to a thread needs `list<any + Send>`;
      - a struct with an `any` field is not `Send`;
      - an unannotated generator yields `any`, so it is not `Send`.
    - A lambda or nested `def` is `function + Send` when every capture is `Send`, and `function` otherwise. The same holds for a bound method and its receiver. Module-level `def`s are `Send`.
    - A generator declared `-> Iterator<T> + Send` cannot hold a value that isn't `Send` (a live variable or a capture) across a `yield`. A plain `-> Iterator<T>` may hold anything.
    - `X + Send` needs `Send` arguments (`Iterator<Counter> + Send` is an error for a non-`Send` `Counter`). On non-erased types `+ Send` is an error, since they are `Send` by what they contain.
  - **`T: Send`.** An unbounded type parameter may not be `Send`. Spawning a `T`, `Channel<T>`, a module variable of type `T`, and storing a `T` as `any + Send` all need `T: Send`.
  - **Crossings need `Send`:** `spawn` arguments, the spawned function and its return type, channel items, module variables (like Rust statics), and a thread's result and escaping exception. These are checked statically wherever the types are known.
  - **Runtime.** The rc word's bit 60 is set at creation from the object's type, or for closures, bound methods and generators from what they hold, and never changes. It backstops values whose static type is unknown (such as the result of calling a `function` value):
    - `has_type` rejects a local value for an erased `+ Send` type;
    - `spawn` checks its arguments and function;
    - `join` checks the thread's result and exception;
    - module variable stores and channel sends are checked;
    - a `+ Send` generator checks its owned registers at creation and at each `yield`.
  - **Freeing.** A local object only reaches zero on its own thread. When a thread exits, its queued local objects are freed on that thread without waiting for a grace period: no other thread can reach them. So destructors of non-`Send` Rust values always run on their owner thread.
- **Rust bindings (10).** `crates/slim` is a library plus the CLI binary. `slim::Runtime` is the embedding API: `rt.module("geom", |m| ...)` registers a native module, then `rt.run_source`/`run_file` compiles and runs a program that can import it. `examples/embed_geom` is a complete example. The binding layer lives in `crates/slim-vm/src/native.rs`.
  - **Functions.** `m.function("dot", |a: ListRef<f64>, b: ListRef<f64>| ...)` accepts any `Fn` of up to 8 arguments. A first parameter `&mut Ctx` gives access to the interpreter: `ctx.call(&func, (args,))` calls a slim function or closure, and `ctx.offline(|| ...)` runs a `Send` closure with the thread marked offline, so a long computation or a blocking wait doesn't hold back `rc_run` on other threads. A result of `Result<T, SlimError>` raises on `Err`. `SlimError::new("ZeroDivisionError", msg)` names any builtin exception class (an unknown name becomes `RuntimeError`), and an exception raised by a callback passes through unchanged when the native returns its error. A panic becomes `RuntimeError: native panic in f(): <message>`.
  - **Conversions.** The `SlimType` trait maps each Rust type to the slim type the compiler sees. `FromSlim` converts arguments; `IntoSlim` converts results.
    - Scalars: `i8`–`i64`, `u8`–`u64` and `usize` are range-checked, so `300` passed as a `u8` raises `OverflowError`. `f64` and `f32` accept ints too. Also `bool`, `String`, `()` and `Option<T>`.
    - `Vec<T>` copies a `list<T>`.
    - `Any` holds an arbitrary value.
    - `Func` holds a callable.
    - `ListRef<T>`, `DictRef<K, V>` and `ArrayRef<T>` are typed handles that read and write a container in place. Their type is checked once, by comparing the container's reified type (O(1)), and writes go through the container's element check.
    - `Native<T>` holds an instance of a native class.
  - **Static signatures.** Calls of natives are checked against the Rust signature, like calls of slim functions: "argument `#2` expects `int`, got `str`". Parameters are named `#1`, `#2`, and so on. An argument typed `any` is checked at runtime instead. `NativeFn::scalar_sig` exposes all-scalar signatures, which the JIT can use to call natives directly.
  - **Ownership.** Native parameters always borrow; there is no `keep`. A handle (`ListRef`, `Func`, `Native<T>`, ...) holds its own counted reference, so a native may keep one past the call. Handles are `!Send`, so they stay on the interpreter thread that created them. That is why `Any` and `Func` are typed `any` and `function`, which may hold values that are not `Send`.
  - **Classes.** `m.class::<T>("Point", |c| { c.init(...); c.method(...); c.method_mut(...); c.repr(...); })`. `init` takes the constructor's arguments and returns `T` (or `Result<T, SlimError>`). `method` gets `&T`, and `method_mut` gets `&mut T`. Instances are slim objects of type `geom.Point`. They work with `isinstance` and in annotations (after `from geom import Point`), and their Rust value is dropped when the count reaches zero. Which registration function to use depends on `T`'s auto traits:

    | Registration | Requires | Behavior |
    |---|---|---|
    | `class` | `T: Send + Sync` | `&self` methods run concurrently from any thread; `&mut self` methods are exclusive |
    | `class_send` | `T: Send` | every call is exclusive |
    | `class_not_send` | any `T` | the slim class is not `Send` either: instances never leave their thread; each call also checks the owner thread (`ThreadError`) as a safety net |

    Access is tracked by an atomic borrow counter on the object. A call that conflicts with a running call, such as a re-entrant `&mut self` call from a callback, raises `BorrowError`. A class registered with `c.locked()` waits instead: it goes offline and yields until the conflicting call finishes. Waiting is only useful across threads; a re-entrant conflict on one thread would never finish.
  - **Generic classes (10b).** A generic Rust type is registered once per supported instantiation. `T` may be any `SlimType` (scalars, `String`, `Native<P>`, ...), and a tuple `(A, B)` gives several type arguments:

    ```rust
    fn stack<T: FromSlim + IntoSlim + Send + Sync + 'static>(c: &mut ClassBuilder<Stack<T>>) { c.init(..); c.method_mut("push", ..); }
    m.generic_class("Stack", &["T"], |g| {
        g.class::<i64, _>(stack::<i64>);                              // Stack<int>
        g.class::<String, _>(stack::<String>);                        // Stack<str>
        g.class_not_send::<Native<Point>, _>(stack::<Native<Point>>); // Stack<Point>: handles are !Send
    });
    ```

    - **Classes.** Each instantiation is an ordinary native class over its own Rust type, with its own methods and mode (`class`/`class_send`/`class_not_send`, chosen by the Rust type's auto traits as above). Slim sees one generic class `Stack<T>`, a family object that is the global `geom.Stack` and has no constructor or methods of its own.
    - **Types and checks.** An instance's runtime type is `Stack<int>` (`Ctor::Class(family)` with the type arguments), so `isinstance(s, Stack)`, annotations and runtime checks work as for slim generics. `Send` follows the instantiation: `Stack<int>` is `Send`, `Stack<Point>` is not.
    - **Construction.** `Stack::<int>()`, or `s: Stack<str> = Stack()` with the arguments taken from the expected type, dispatches to the instantiation for those arguments. When they are known statically, the call is checked against that instantiation's constructor and its methods' signatures, and an unsupported instantiation, written or annotated, is a compile error: "`geom.Stack<float>` is not available: `geom.Stack` supports `geom.Stack<int>`, `geom.Stack<str>`". Inside a generic function (`Stack::<T>()`), the runtime picks the instantiation and raises `TypeError` for an unsupported one; method calls on such a value are dynamic.
    - **Natives.** A native function may take or return `Native<Stack<i64>>`, which slim sees as `geom.Stack<int>`.
    - **Registration errors panic:** a wrong number of type arguments, or the same instantiation registered twice.
  - **Traits (10c).**
    - **Rust traits.** A module defines a trait whose methods are all required: `m.trait_def("Shape", |t| { t.method::<(), f64>("area"); t.method::<(f64,), ()>("scale"); })`. The argument types are a tuple (`()`, `(f64,)`, ...). Its classes, defined after it, implement it with `c.implements("Shape")`.
      - Slim imports it like a class (`from geom import Shape`) and uses it as any trait: `s: Shape`, `list<Shape + Send>`, `T: Shape`, `impl Shape for Square:`.
      - A struct must define every method.
    - **Builtin traits through dunders.** `c.implements("Ord" | "Num" | "Len")` with `__lt__`, the arithmetic dunders (`Num` implies `Ord`) or `__len__`. Operators, `sorted` and `len()` call a native object's dunder methods like an instance's. `Send` is never implemented this way: it follows the Rust type.
    - **Registration checks.** At registration, the class's methods are checked against the trait's signatures; a missing or mismatched method panics with the class, trait and method names.
    - **Program traits for native classes:** `impl Describe for Polygon`, without a body (the trait's default methods are used). It is checked at compile time:
      - every required method exists on the class, with the trait's parameter types and a compatible return type;
      - native classes get no inherent `impl` (their methods are defined in Rust);
      - the trait is not a native one (those are implemented in Rust) and is not generic.

      For a generic native class (`impl Len for Stack`), the impl covers every instantiation, and each instantiation is checked. At runtime, the statement (`ImplNative`) adds the class to the trait's `native_impls` and those of the traits it implies, which `rtype::satisfies` consults for bounds, `has_type` and `isinstance`. It also records the trait's default methods in `vm.native_methods`, which method calls on native objects fall back to, so `p.describe()` runs the slim default with `self` the native object.
  - **Imports.** `import geom [as g]` and `from geom import dot [as d], Point` are only allowed at module level. Before compiling, imports are resolved by an AST rewrite (`crates/slim-compiler/src/imports.rs`): every name an import binds (`dot`, `g.dot`) becomes the qualified global `geom.dot`, and the compiler registers the native signatures under those names. A function's parameters and locals shadow imported names. Redefining an imported name at module level is an error, and so is using a module as a value. Unknown modules and items are compile-time `NameError`s.
  - **Limitations.**
    - Rust traits have no default methods and no type parameters. Generic traits cannot be implemented for native classes from slim.
    - Natives cannot implement `Iterable`/`Iterator`.
    - `impl Trait for NativeClass` applies to the program that declares it: native trait objects are shared by programs, the program's traits are not.
    - Default methods of a program trait are reachable through method calls (`p.describe()`); `p.describe` without a call does not see them.
    - A generic native class supports only its registered instantiations: there is no generic Rust code behind `Stack<T>` for an arbitrary `T`, and each instantiation's values are converted (`FromSlim`/`IntoSlim`) when they cross into Rust.
    - Annotations need the class imported by name (`from m import C`); dotted type names (`m.C`) are not supported.
    - Error messages show qualified names such as `geom.Point`.
    - `ThreadError` cannot be reached from slim code, since a local value never reaches another thread; the check exists for misuse from Rust.
    - A local native object freed off its owner thread would be a VM bug. Debug builds panic; release builds leak the Rust value with a warning rather than drop it on the wrong thread.
