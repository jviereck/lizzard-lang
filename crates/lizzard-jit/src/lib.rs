//! Lizzard's Cranelift JIT: compiles a function's IR (`FuncIr`) to native
//! code that runs on the same frame as the interpreter.
//!
//! The frame is an array of 8-byte register slots (see `lizzard-py`'s
//! `frame.rs`). A function is compiled once it is warm, with what the
//! interpreter's caches saw (`Spec`, one per instruction): the type and
//! version of an attribute's object and where the attribute lives, the value
//! of a global, the function a call reaches. Compiled code checks each of
//! these with a guard of a few machine instructions and then does the work
//! inline; when a guard fails, the instruction takes its generic path (the
//! runtime's helper, which works for anything), or, where the code after it
//! depends on the specialization, the function *bails out*: compiled code
//! returns `BAIL` with the frame in the interpreter's form and the pc to
//! continue at, and the interpreter finishes the frame.
//!
//! Within a run of instructions without jumps into it (a block), registers
//! can be *virtual*: compiled code knows their value without the slot
//! holding it:
//! - a float or int (`IR hints` say which arithmetic is on floats or ints)
//!   kept unboxed in a machine register, boxed only if something needs the
//!   object;
//! - a borrowed pointer (an attribute's value, a constant, a global), whose
//!   reference is only taken if the register outlives what keeps the object
//!   alive.
//! Before a jump to another block, a helper call (which may read the slots
//! or run arbitrary code), an exception and a bail-out, the registers that
//! are live there are *materialized*: boxed or given their reference, and
//! stored in their slots. Scalar registers (`I64`, `Bool`) are Cranelift
//! variables, stored to their slots where something reads those (a helper,
//! the interpreter).
//!
//! A compiled function is
//! `extern "C" fn(ctx, regs: *mut u64, pc: *mut u32, pending: u32) -> u32`.
//! It starts at `*pc`: 0, a loop header (on-stack replacement: the
//! interpreter switches to compiled code in a hot loop), or where a
//! generator resumes (after a `yield`, or at the `yield` itself with an
//! exception pending, as `throw` raises it). It returns a status and leaves
//! `*pc` at the instruction that returned, yielded or raised, or (`BAIL`)
//! the one to continue at.
//!
//! This crate knows nothing about Python's API: the runtime passes the
//! helpers' addresses and the object layout (`Layout`), and `ctx` is opaque
//! here but for the two fields at `CTX_TID` and `CTX_RET`.

use std::collections::HashMap;
use std::ffi::c_void;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, AbiParam, AtomicRmwOp, Block, BlockArg, InstBuilder, MemFlagsData, SigRef, Signature, StackSlot, StackSlotData, StackSlotKind, TrapCode, Type, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module};
use lizzard_py_front::ir::{BinOp, CmpOp, FuncIr, Instr, Reg, Repr, RtCheck, HINT_FLOAT, HINT_INT};

/// The helper executed the instruction; continue with the next one.
pub const NEXT: u32 = 0;
/// The helper took the instruction's jump (`ForIter` at the end).
pub const TAKEN: u32 = 1;
/// The function returned or yielded (the runtime has the value).
pub const EXIT: u32 = 2;
/// An exception is set (in CPython's thread state).
pub const ERROR: u32 = 3;
/// Compiled code stopped: the interpreter continues the frame at `*pc`.
pub const BAIL: u32 = 4;
/// A generator yielded at `*pc` (a `Yield`): the value (a new reference)
/// is the context's `CTX_RET`.
pub const YIELD: u32 = 5;
/// The same, for an int yielded as it is (`CTX_WANT_INT`): `CTX_RET` holds
/// the `i64`.
pub const YIELD_INT: u32 = 6;
/// `for_iter_int`: the next item, an int, in its `out` (not in `dst`).
pub const NEXT_INT: u32 = 7;

/// A Lizzard generator as compiled code resumes it (`Spec::IterGen`): its
/// state at these offsets from its `Hot` (bytes: `running`, status, `plain`;
/// `u32`: pc, dst; pointers: slots, target, the entry's cell, the per-pc
/// destination table).
pub const GEN_RUNNING: i32 = 0;
pub const GEN_STATUS: i32 = 1;
pub const GEN_PLAIN: i32 = 2;
pub const GEN_PC: i32 = 4;
pub const GEN_DST: i32 = 8;
pub const GEN_SLOTS: i32 = 16;
pub const GEN_TARGET: i32 = 24;
pub const GEN_ENTRY: i32 = 32;
pub const GEN_YIELD_DST: i32 = 40;
/// The status of a suspended generator.
pub const GEN_SUSPENDED: u8 = 1;
/// The destination of a `yield` whose value is never read.
pub const NO_SEND: u32 = u32::MAX;

/// The context's thread id (free-threaded reference counting compares an
/// object's owner with it), and where `Return` leaves its value (a new
/// reference).
pub const CTX_TID: i32 = 0;
pub const CTX_RET: i32 = 8;
/// The frame's slots, the function's description for the helpers, and
/// where `step` leaves a yield (null on entry); the context's size.
pub const CTX_SLOTS: i32 = 16;
pub const CTX_TARGET: i32 = 24;
pub const CTX_EXIT: i32 = 32;
/// Non-zero: a `yield` of an unboxed int may hand over the int itself
/// (`YIELD_INT`).
pub const CTX_WANT_INT: i32 = 40;
pub const CTX_SIZE: i32 = 48;

pub type Entry = unsafe extern "C" fn(ctx: *mut c_void, regs: *mut u64, pc: *mut u32, pending: u32) -> u32;

/// How objects are laid out, as far as compiled code reads them.
#[derive(Clone, Copy)]
pub struct Layout {
    /// Offset of `ob_type` in an object, and of `tp_version_tag` in a type.
    pub ob_type: i32,
    pub tp_version: i32,
    pub rc: Rc,
    /// Exact ints and floats can be read directly (`fval`: a float's
    /// double; `long_tag` and `long_digit`: a compact int's tag and digit).
    pub numbers: bool,
    pub float_type: usize,
    pub long_type: usize,
    pub fval: i32,
    pub long_tag: i32,
    pub long_digit: i32,
    pub none: usize,
    pub true_: usize,
    pub false_: usize,
    /// The address of the `u64` that changes whenever a watched globals or
    /// builtins dict does.
    pub epoch: usize,
    /// CPython's preallocated small ints: `count` immortal objects for the
    /// values from `min`, the first at `first`, `stride` bytes apart.
    pub small_ints: Option<SmallInts>,
    /// The layout of list iterators, for `for` over a list.
    pub list_iter: Option<ListIter>,
}

/// `listiterobject { PyObject_HEAD; Py_ssize_t it_index; PyListObject
/// *it_seq; }` and `PyListObject`'s size and item array. Once exhausted,
/// `it_seq` is null, or with `end_index` (free threading), `it_index` is -1.
#[derive(Clone, Copy)]
pub struct ListIter {
    pub tp: usize,
    pub index: i32,
    pub seq: i32,
    pub size: i32,
    pub items: i32,
    pub end_index: bool,
}

#[derive(Clone, Copy)]
pub struct SmallInts {
    pub first: usize,
    pub stride: i64,
    pub min: i64,
    pub count: i64,
}

/// How reference counts change.
#[derive(Clone, Copy)]
pub enum Rc {
    /// Through `Helpers::incref` / `decref`.
    Calls,
    /// A 32-bit count at offset 0, immortal when negative (as `i32`).
    Gil,
    /// Free-threaded: the owner thread's id at `tid`, its 32-bit count at
    /// `local` (`u32::MAX`: immortal), other threads' count at `shared` (in
    /// units of 4).
    Ft { tid: i32, local: i32, shared: i32 },
}

/// The runtime functions compiled code calls, by address. Those taking
/// `(ctx, regs, ...)` and returning a status return `NEXT` or `ERROR`
/// unless said otherwise.
#[derive(Clone, Copy)]
pub struct Helpers {
    /// `step(ctx, regs, pc) -> status`: execute instruction `pc`
    /// (`NEXT`, `TAKEN`, `EXIT` or `ERROR`).
    pub step: usize,
    /// `catch(ctx, regs, reg)`: move the exception being raised into
    /// register `reg`, for a handler.
    pub catch: usize,
    /// `safepoint(ctx) -> status`: run pending signal handlers and let a
    /// stop-the-world pause happen. Called every `FUEL` loop iterations.
    pub safepoint: usize,
    /// `load_attr(regs, dst, obj, name, ic)`
    pub load_attr: usize,
    /// `store_attr(regs, obj, val, name, ic)`
    pub store_attr: usize,
    /// `truthy(regs, dst, src)`
    pub truthy: usize,
    /// `binop(regs, dst, a, b, op)`: `op` is the `BinOp`'s index, plus 16
    /// in place.
    pub binop: usize,
    /// `compare(regs, dst, a, b, op)` with CPython's `Py_LT`... codes.
    pub compare: usize,
    /// `compare_bool(regs, dst, a, b, op)`: the same, as a `Bool`.
    pub compare_bool: usize,
    /// `call(ctx, regs, pc)` for `Call` and `CallMethod`.
    pub call: usize,
    /// `for_iter(ctx, regs, dst, iter, pc)`: `NEXT`, `TAKEN` at the end, or
    /// `ERROR`.
    pub for_iter: usize,
    /// `for_iter_int(ctx, regs, dst, iter, pc, out)`: `for_iter`, but a
    /// Lizzard generator yielding an unboxed int gives it in `*out`
    /// (`NEXT_INT`).
    pub for_iter_int: usize,
    /// `gen_resumed(it, ctx, status, pc, regs, dst, out)`: compiled code
    /// resumed generator `it` (`ctx`: its context) and it stopped other
    /// than at a plain `yield` (`status` at `pc`): finishes that, releases
    /// `running`; `for_iter_int`'s status (`out` may be null).
    pub gen_resumed: usize,
    /// `check(ctx, regs, reg, check, pc)`
    pub check: usize,
    /// `load_global(ctx, regs, dst, pc)`
    pub load_global: usize,
    /// `enter(ctx, frame, desc) -> object`: call the function `desc`
    /// describes (`CallSpec::desc`) on `frame`, its registers with the
    /// arguments (borrowed) first and the others unbound; its result (a new
    /// reference), or null with an exception set. It releases the frame.
    pub enter: usize,
    /// `Py_IncRef`, `Py_DecRef` (`Rc::Calls`), `_Py_Dealloc` (`Rc::Gil`),
    /// `_Py_MergeZeroLocalRefcount` and `_Py_DecRefShared` (`Rc::Ft`).
    pub incref: usize,
    pub decref: usize,
    pub dealloc: usize,
    pub merge_zero: usize,
    pub dec_shared: usize,
    /// `PyFloat_FromDouble`, `PyLong_FromLongLong`.
    pub float_new: usize,
    pub long_new: usize,
    /// `long_value(obj, out: *mut i64) -> u32`: 1 with the value of an exact
    /// int that fits in an `i64`, else 0 (no exception).
    pub long_value: usize,
    /// `init_returned(obj)`: `__init__` returned `obj` (a new reference),
    /// not None: release it and raise TypeError.
    pub init_returned: usize,
    /// `PyObject_Vectorcall(callable, args, nargs, kwnames)`.
    pub vectorcall: usize,
    /// `Py_EnterRecursiveCall(where)` (`recursion_msg`) and
    /// `Py_LeaveRecursiveCall()`, around direct calls of compiled code.
    pub enter_recursive: usize,
    pub leave_recursive: usize,
    pub recursion_msg: usize,
    /// `tail_depth(d) -> status`: a frame's `d`th self tail call; `ERROR`
    /// (RecursionError) at the recursion limit.
    pub tail_depth: usize,
    /// `resume(ctx, frame, desc, pc) -> object`: a directly called
    /// function's code bailed out at `pc`: the interpreter finishes it, as
    /// `finish`.
    pub resume: usize,
    /// `take_exit(callee_ctx, desc) -> object`: a directly called function
    /// returned through `step`: its value, and the frame released.
    pub take_exit: usize,
    /// `finish(ctx, frame, desc, pc) -> object`: the interpreter runs the
    /// rest of an inlined call of `desc`'s function from `pc` on `frame`
    /// (its parameters borrowed) and releases the frame; the result (a new
    /// reference), or null with an exception set.
    pub finish: usize,
    /// `unwind(frame, desc, pc)`: an exception raised at `pc` leaves an
    /// inlined or directly called function: note where, and release the
    /// frame (but its borrowed parameters).
    pub unwind: usize,
    /// `missed(ctx)`: the generic paths of sites compiled without what
    /// their caches see (`Unit::misses`) ran `MISSES` times.
    pub missed: usize,
}

/// See `Helpers::missed`.
pub const MISSES: u32 = 1000;

/// A check by type pointers alone; `Slow` always asks the helper.
#[derive(Clone, Copy, Debug)]
pub enum Quick {
    Slow,
    Any,
    /// The exact type.
    Type(usize),
    /// An exact float or an exact int.
    FloatOrInt,
    /// That very object (None).
    Is(usize),
    /// True or False.
    Bool,
}

/// What the runtime's caches saw at an instruction.
#[derive(Clone, Debug)]
pub enum Spec<'a> {
    None,
    /// `LoadAttr` / `StoreAttr` of an instance attribute kept in inline
    /// values, on instances of type `tp` at `version`: the value pointer is
    /// at `off` from the object, in the values at `values` (CPython's
    /// `struct _dictvalues { uint8 capacity, size, embedded, valid; PyObject
    /// *values[capacity]; uint8 order[capacity]; }`), which are in use
    /// while `valid` is not 0; no `__dict__` exists while the pointer at
    /// `dict` is null.
    Inline { tp: usize, version: u32, off: i32, values: i32, dict: i32 },
    /// A `__slots__` member: an object pointer at `off` (null when unset).
    Member { tp: usize, version: u32, off: i32 },
    /// `LoadGlobal`: `value` while the epoch is `epoch`.
    Global { epoch: u64, value: usize },
    Check(Quick),
    /// `ForIter`: the iterators it met (`ITER_LIST`, ...; 0: none yet).
    Iter(u8),
    /// `ForIter` that met Lizzard generators (of type `tp`, their state at
    /// `hot` in the object, `GEN_*`): compiled code resumes a plainly
    /// suspended one itself.
    IterGen { hot: i32, tp: usize },
    /// `JumpIfNotSelf` of a function object (`func`) that runs this code:
    /// a self tail call loops when it is still that object.
    SelfFunc(usize),
    Call(CallSpec<'a>),
}

/// A call (`Call` or `CallMethod`) that reaches a Lizzard function compiled
/// code can enter directly.
#[derive(Clone, Debug)]
pub struct CallSpec<'a> {
    pub guard: CallGuard,
    /// For `Helpers::enter`.
    pub desc: usize,
    /// The callee's registers.
    pub nregs: u32,
    /// The checks of its parameters (the receiver first) a caller makes.
    pub checks: Vec<Quick>,
    /// The callee, to compile into the caller instead of calling it, if it
    /// qualifies (`inlinable`).
    pub inline: Option<Box<Unit<'a>>>,
    /// Not a Lizzard function: `desc` is a callable (a builtin method) that
    /// takes the arguments with `Helpers::vectorcall`, unchecked.
    pub native: bool,
    /// For a callee that never writes its parameters: the address of the
    /// word holding its compiled entry (0 while it has none), and its
    /// context's target (`CTX_TARGET`). Compiled code then calls it
    /// directly. 0 otherwise.
    pub entry: usize,
    pub target: usize,
}

/// When a construction may take an instance about to be released instead
/// of allocating one: an instance only one dead register references, with
/// no weak references (the list at `weakref`, if instances can have one)
/// whose attributes are exactly those `__init__` assigns, in the order it
/// assigns them (the inline values at `values` hold `order`; None: the
/// class keeps its attributes in `__slots__`, all of which `__init__`
/// assigns), so that it cannot be told from a new one. The class has no
/// finalizer.
#[derive(Clone, Debug)]
pub struct Reuse {
    pub weakref: Option<i32>,
    pub values: Option<i32>,
    pub order: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum CallGuard {
    /// `Call`: the function register holds this object.
    Func(usize),
    /// `Call` of class `tp` (at `version`) whose instances are created by
    /// allocating them (`alloc(tp, 0)`, its `tp_alloc`) and calling the
    /// `__init__` the call spec describes, the instance first.
    Ctor { tp: usize, version: u32, alloc: usize, reuse: Option<Reuse> },
    /// `CallMethod`: the receiver's type and its version, and why no
    /// instance attribute shadows the method: `keys` is null (no `__dict__`)
    /// or the instance's inline values are in use (the byte at `valid`) and
    /// the shared keys still have `nentries` entries (`keys` is the address
    /// of their count).
    Method { tp: usize, version: u32, keys: usize, nentries: i64, valid: i32 },
}

/// A constant's value, when it is a float or an int that fits in an `i64`.
#[derive(Clone, Copy, Debug)]
pub enum Num {
    Other,
    F64(f64),
    I64(i64),
}

/// What to compile.
#[derive(Clone, Debug)]
pub struct Unit<'a> {
    pub ir: &'a FuncIr,
    /// Per instruction.
    pub specs: Vec<Spec<'a>>,
    /// The constants (alive as long as the code) and their values.
    pub consts: Vec<usize>,
    pub nums: Vec<Num>,
    /// The interned names.
    pub names: Vec<usize>,
    /// Each instruction's attribute cache, for the helpers.
    pub ics: Vec<usize>,
    /// The parameters are never written: their slots may hold references
    /// the frame does not own (a direct call's arguments), so compiled code
    /// never releases them or moves them out (`Take` copies).
    pub borrow_params: bool,
    /// The address of a `u32` counting the generic executions of attribute
    /// accesses, global loads and calls compiled without a spec.
    pub misses: usize,
    /// Compile speculatively: registers that hold floats or ints at a label
    /// stay unboxed across jumps to it (`carried`), and a failed attribute
    /// guard bails out, so that what it checked holds after it. Off after
    /// bail-outs made the runtime recompile.
    pub speculate: bool,
}

/// What a register holds at an instruction, as far as compiled code
/// produces it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Not reached yet (the analysis' start).
    Top,
    Unbound,
    /// Some object.
    Obj,
    /// A float or an int compiled code made (it bails out otherwise).
    F,
    I,
    Mixed,
}

fn meet(a: Kind, b: Kind) -> Kind {
    match (a, b) {
        (Kind::Top, x) | (x, Kind::Top) => x,
        (x, y) if x == y => x,
        _ => Kind::Mixed,
    }
}

/// What each register holds before each instruction (`Kind`), from the
/// parameters' kinds (`params`; the others are objects).
fn kinds(u: &Unit, params: &[Kind]) -> Vec<Vec<Kind>> {
    let ir = u.ir;
    let n = ir.code.len();
    let nregs = ir.reprs.len();
    let obj = |r: Reg| ir.reprs[r as usize] == Repr::Obj;
    let mut at: Vec<Vec<Kind>> = vec![vec![Kind::Top; nregs]; n];
    if n == 0 {
        return at;
    }
    for (r, k) in at[0].iter_mut().enumerate() {
        // Without what a caller knows: what the parameter is declared
        // (speculatively; copies of it are unboxed or bail out).
        let declared = || match ir.checks.get(ir.params[r].check as usize).map(|c| &c.check) {
            Some(RtCheck::Int) => Kind::I,
            Some(RtCheck::Float) => Kind::F,
            _ => Kind::Obj,
        };
        *k = if r < ir.params.len() { params.get(r).copied().unwrap_or_else(declared) } else { Kind::Unbound };
    }
    let handler_of = |pc: u32| -> Vec<usize> { ir.handlers.iter().filter(|h| h.ranges.iter().any(|(s, e)| *s <= pc && pc < *e)).map(|h| h.target as usize).filter(|&t| t < n).collect() };
    let mut reached = vec![false; n];
    reached[0] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for pc in 0..n {
            let i = &ir.code[pc];
            let before = at[pc].clone();
            // Not reached (yet).
            if !reached[pc] {
                continue;
            }
            let mut st = before.clone();
            let w = |st: &mut Vec<Kind>, r: Reg, k: Kind| st[r as usize] = k;
            match i {
                Instr::Const { dst, idx } => w(&mut st, *dst, match u.nums[*idx as usize] {
                    Num::F64(_) => Kind::F,
                    Num::I64(_) => Kind::I,
                    Num::Other => Kind::Obj,
                }),
                Instr::BinOp { dst, op, .. } => {
                    let k = match ir.hints.get(pc).copied().unwrap_or(0) {
                        HINT_FLOAT if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) => Kind::F,
                        HINT_INT if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) => Kind::I,
                        _ => Kind::Obj,
                    };
                    w(&mut st, *dst, k);
                }
                Instr::Box { dst, src } if ir.reprs[*src as usize] == Repr::I64 => w(&mut st, *dst, Kind::I),
                // Items the checker knows are ints (speculatively: unboxed,
                // or the loop bails out).
                Instr::ForIter { dst, .. } if ir.hints.get(pc).copied() == Some(HINT_INT) => w(&mut st, *dst, Kind::I),
                Instr::Move { dst, src } if obj(*src) && obj(*dst) => w(&mut st, *dst, before[*src as usize]),
                Instr::Take { dst, src } if obj(*src) && obj(*dst) => {
                    w(&mut st, *dst, before[*src as usize]);
                    if !(u.borrow_params && (*src as usize) < ir.params.len()) {
                        w(&mut st, *src, Kind::Unbound);
                    }
                }
                Instr::Unbind { reg } => w(&mut st, *reg, Kind::Unbound),
                _ => {
                    for r in i.writes() {
                        w(&mut st, r, Kind::Obj);
                    }
                }
            }
            let mut flow = |t: usize, st: &[Kind], at: &mut Vec<Vec<Kind>>| {
                if !reached[t] {
                    reached[t] = true;
                    changed = true;
                }
                for r in 0..nregs {
                    let m = meet(at[t][r], st[r]);
                    if m != at[t][r] {
                        at[t][r] = m;
                        changed = true;
                    }
                }
            };
            for t in i.successors(pc as u32) {
                if (t as usize) < n {
                    flow(t as usize, &st, &mut at);
                }
            }
            for h in handler_of(pc as u32) {
                let mut hs = before.clone();
                for r in i.writes() {
                    hs[r as usize] = meet(hs[r as usize], Kind::Obj);
                }
                for hd in ir.handlers.iter().filter(|x| x.target as usize == h) {
                    hs[hd.exc as usize] = Kind::Obj;
                }
                flow(h, &hs, &mut at);
            }
        }
    }
    at
}

/// Per label, the registers that hold a float (`Kind::F`) or an int
/// (`Kind::I`) on every path into it and are live there: they stay unboxed
/// across the jumps (Cranelift block parameters). Not at the entry, handler
/// targets or after yields, which the runtime enters with the slots alone.
fn carried(u: &Unit, numbers: bool, labels: &[bool], live: &[Vec<u64>], params: &[Kind]) -> Vec<Vec<(Reg, Kind)>> {
    let ir = u.ir;
    let n = ir.code.len();
    let nregs = ir.reprs.len();
    let mut out = vec![Vec::new(); n];
    if !u.speculate || !numbers || n == 0 {
        return out;
    }
    let at = kinds(u, params);
    let mut excluded = vec![false; n];
    excluded[0] = true;
    for h in &ir.handlers {
        if (h.target as usize) < n {
            excluded[h.target as usize] = true;
        }
    }
    for (pc, i) in ir.code.iter().enumerate() {
        if matches!(i, Instr::Yield { .. } | Instr::YieldFrom { .. }) && pc + 1 < n {
            excluded[pc + 1] = true;
        }
    }
    for pc in 0..n {
        if !labels[pc] || excluded[pc] {
            continue;
        }
        // (An inlined function's parameters are known at every label.)
        for r in ir.params.len()..nregs {
            let k = at[pc][r];
            if matches!(k, Kind::F | Kind::I) && live[pc][r / 64] & (1 << (r % 64)) != 0 {
                out[pc].push((r as Reg, k));
            }
        }
    }
    out
}

/// Instructions an inlined function may have: those compiled code does
/// itself (given their specs), so that its slow paths are rare.
pub const INLINE_MAX: usize = 48;
/// The same, inlined into an inlined function (the second and last level).
pub const INLINE_MAX_NESTED: usize = 16;

/// Whether `u` can be compiled into its callers: small, straight-line or
/// branching forward only, no handlers, parameters never written, and
/// nothing that always takes a generic path.
pub fn inlinable(u: &Unit) -> bool {
    let ir = u.ir;
    if ir.generator || !ir.handlers.is_empty() || !u.borrow_params || ir.code.len() > INLINE_MAX || ir.reprs.len() > 64 {
        return false;
    }
    let obj = |r: Reg| ir.reprs[r as usize] == Repr::Obj;
    ir.code.iter().enumerate().all(|(pc, i)| {
        if i.successors(pc as u32).iter().any(|&t| t as usize <= pc) {
            return false;
        }
        let spec = &u.specs[pc];
        match i {
            Instr::Const { .. } | Instr::Move { .. } | Instr::Take { .. } | Instr::Unbind { .. } | Instr::Check { .. } | Instr::I64Const { .. } | Instr::I64Lt { .. } | Instr::I64Add { .. } | Instr::Jump { .. } => true,
            Instr::JumpIfFalse { cond, .. } | Instr::JumpIfTrue { cond, .. } => !obj(*cond),
            Instr::Not { src, dst } => !obj(*src) && !obj(*dst),
            Instr::Box { src, .. } => matches!(ir.reprs[*src as usize], Repr::I64 | Repr::Bool),
            Instr::LoadAttr { .. } | Instr::StoreAttr { .. } => matches!(spec, Spec::Inline { .. } | Spec::Member { .. }),
            Instr::LoadGlobal { .. } => matches!(spec, Spec::Global { .. }),
            Instr::Call { kwnames, .. } | Instr::CallMethod { kwnames, .. } => kwnames.is_empty() && matches!(spec, Spec::Call(_)),
            Instr::BinOp { op, .. } => match ir.hints.get(pc).copied().unwrap_or(0) {
                HINT_FLOAT => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div),
                HINT_INT => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor),
                _ => false,
            },
            Instr::CompareBool { dst, op, .. } => !obj(*dst) && (rich_code(*op).is_some() || matches!(op, CmpOp::Is | CmpOp::IsNot)),
            Instr::Truthy { dst, .. } => !obj(*dst),
            Instr::Return { src } => obj(*src),
            _ => false,
        }
    })
}

/// The registers direct calls in `u` (and in what it inlines) need.
fn frame_regs(u: &Unit) -> Option<u32> {
    u.specs
        .iter()
        .filter_map(|s| match s {
            Spec::Call(c) => Some(c.inline.as_deref().and_then(frame_regs).unwrap_or(0).max(c.nregs)),
            _ => None,
        })
        .max()
}

/// A block per label, with a parameter per carried register.
fn label_blocks(b: &mut FunctionBuilder, is_label: &[bool], carry: &[Vec<(Reg, Kind)>]) -> Vec<Option<Block>> {
    is_label
        .iter()
        .zip(carry)
        .map(|(&x, c)| {
            x.then(|| {
                let blk = b.create_block();
                for (_, k) in c {
                    b.append_block_param(blk, if *k == Kind::F { types::F64 } else { types::I64 });
                }
                blk
            })
        })
        .collect()
}

/// The instructions that start a block: jump targets, the instruction
/// after any that can branch, handler targets, entries.
/// `list_iter`: which `ForIter`s are compiled over lists, falling through
/// to the next instruction (no label of its own unless jumped to).
fn label_pcs(ir: &FuncIr, list_iter: impl std::ops::Fn(usize) -> bool) -> Vec<bool> {
    let n = ir.code.len();
    let mut is_label = vec![false; n];
    if n > 0 {
        is_label[0] = true;
    }
    for (pc, i) in ir.code.iter().enumerate() {
        if let Instr::ForIter { dst, target, .. } = i {
            if list_iter(pc) && ir.reprs[*dst as usize] == Repr::Obj {
                if (*target as usize) < n {
                    is_label[*target as usize] = true;
                }
                continue;
            }
        }
        let succ = i.successors(pc as u32);
        let jumps = matches!(i, Instr::Jump { .. } | Instr::JumpIfFalse { .. } | Instr::JumpIfTrue { .. } | Instr::ForIter { .. });
        if jumps || succ.len() > 1 || succ.first().is_some_and(|&t| t as usize != pc + 1) {
            for t in succ {
                if (t as usize) < n {
                    is_label[t as usize] = true;
                }
            }
        }
        if matches!(i, Instr::Yield { .. } | Instr::YieldFrom { .. }) && pc + 1 < n {
            is_label[pc + 1] = true;
        }
    }
    for h in &ir.handlers {
        if (h.target as usize) < n {
            is_label[h.target as usize] = true;
        }
    }
    is_label
}

/// What a `ForIter` has met (`Spec::Iter`): a list's iterator, a Lizzard
/// generator, or another (or several kinds).
pub const ITER_LIST: u8 = 1;
pub const ITER_GEN: u8 = 2;
pub const ITER_OTHER: u8 = 3;

/// Whether `ForIter` at `pc` is compiled over lists: only lists met it,
/// or nothing yet (a loop after the one compiled code was entered in).
fn over_lists(u: &Unit, l: &Layout, pc: usize) -> bool {
    l.list_iter.is_some() && matches!(u.specs[pc], Spec::Iter(0 | ITER_LIST))
}

/// Where the `dirty` flag is, in the fuel's stack slot (`Lower::fuel`).
const DIRTY: i32 = 8;

/// Loop iterations between safepoints.
const FUEL: i64 = 1024;

/// Compile `u`. The code lives as long as the process.
pub fn compile(u: &Unit, h: &Helpers, l: &Layout) -> Result<Entry, String> {
    let jb = JITBuilder::with_flags(&[("opt_level", "speed")], default_libcall_names()).map_err(|e| e.to_string())?;
    let mut module = JITModule::new(jb);
    let ptr = module.target_config().pointer_type();
    assert_eq!(ptr, types::I64, "Lizzard's JIT needs a 64-bit target");

    let mut sig = module.make_signature();
    sig.params.extend([AbiParam::new(ptr), AbiParam::new(ptr), AbiParam::new(ptr), AbiParam::new(types::I32)]);
    sig.returns.push(AbiParam::new(types::I32));
    let id = module.declare_anonymous_function(&sig).map_err(|e| e.to_string())?;

    let mut ctx = Context::new();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    {
        let b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        Lower::new(b, u, h, l, &module).lower();
    }
    if std::env::var_os("LIZZARD_JIT_DUMP").is_some() {
        eprintln!("; {}\n{}", u.ir.name, ctx.func.display());
    }
    let asm = std::env::var_os("LIZZARD_JIT_ASM").is_some();
    ctx.set_disasm(asm);
    module.define_function(id, &mut ctx).map_err(|e| format!("{e:?}"))?;
    if asm {
        if let Some(v) = ctx.compiled_code().and_then(|c| c.vcode.as_ref()) {
            eprintln!("; {} (machine code)\n{v}", u.ir.name);
        }
    }
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let code = module.get_finalized_function(id);
    // `LIZZARD_JIT_BYTES=dir`: the machine code, as `<name>@<address>.bin`.
    if let Some(dir) = std::env::var_os("LIZZARD_JIT_BYTES") {
        if let Some(len) = ctx.compiled_code().map(|c| c.code_buffer().len()) {
            // SAFETY: the function's code, `len` bytes.
            let bytes = unsafe { std::slice::from_raw_parts(code, len) };
            let name: String = u.ir.name.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}@{:x}.bin", code as usize)), bytes);
        }
    }
    // The module owns the code's memory; it stays mapped for the process.
    std::mem::forget(module);
    // SAFETY: the function was compiled with `Entry`'s signature.
    Ok(unsafe { std::mem::transmute::<*const u8, Entry>(code) })
}

/// The registers live before each instruction (a bit set per pc). An
/// instruction a handler covers may continue at the handler, before its
/// own writes: what the handler reads is live before it.
fn liveness(ir: &FuncIr) -> Vec<Vec<u64>> {
    let n = ir.code.len();
    let words = ir.reprs.len().div_ceil(64).max(1);
    let handler_of = |pc: u32| -> Vec<usize> { ir.handlers.iter().filter(|h| h.ranges.iter().any(|(s, e)| *s <= pc && pc < *e)).map(|h| h.target as usize).filter(|&t| t < n).collect() };
    let succ: Vec<Vec<usize>> = (0..n).map(|pc| ir.code[pc].successors(pc as u32).into_iter().map(|t| t as usize).filter(|&t| t < n).collect()).collect();
    let exc: Vec<Vec<usize>> = (0..n).map(|pc| handler_of(pc as u32)).collect();
    let reads: Vec<Vec<Reg>> = ir.code.iter().map(|i| i.reads()).collect();
    let writes: Vec<Vec<Reg>> = ir.code.iter().map(|i| i.writes()).collect();
    let mut live = vec![vec![0u64; words]; n];
    let mut changed = true;
    while changed {
        changed = false;
        for pc in (0..n).rev() {
            let mut out = vec![0u64; words];
            for &s in &succ[pc] {
                for w in 0..words {
                    out[w] |= live[s][w];
                }
            }
            for r in &writes[pc] {
                out[*r as usize / 64] &= !(1 << (*r % 64));
            }
            for r in &reads[pc] {
                out[*r as usize / 64] |= 1 << (*r % 64);
            }
            for &h in &exc[pc] {
                for w in 0..words {
                    out[w] |= live[h][w];
                }
            }
            if out != live[pc] {
                live[pc] = out;
                changed = true;
            }
        }
    }
    live
}

/// CPython's `Py_LT`, `Py_LE`, `Py_EQ`, `Py_NE`, `Py_GT`, `Py_GE`.
fn rich_code(op: CmpOp) -> Option<u32> {
    Some(match op {
        CmpOp::Lt => 0,
        CmpOp::Le => 1,
        CmpOp::Eq => 2,
        CmpOp::Ne => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
        _ => return None,
    })
}

fn int_cc(op: CmpOp) -> IntCC {
    match op {
        CmpOp::Lt => IntCC::SignedLessThan,
        CmpOp::Le => IntCC::SignedLessThanOrEqual,
        CmpOp::Eq => IntCC::Equal,
        CmpOp::Ne => IntCC::NotEqual,
        CmpOp::Gt => IntCC::SignedGreaterThan,
        _ => IntCC::SignedGreaterThanOrEqual,
    }
}

/// Python's float comparisons: false with a NaN, but for `!=`.
fn float_cc(op: CmpOp) -> FloatCC {
    match op {
        CmpOp::Lt => FloatCC::LessThan,
        CmpOp::Le => FloatCC::LessThanOrEqual,
        CmpOp::Eq => FloatCC::Equal,
        CmpOp::Ne => FloatCC::NotEqual,
        CmpOp::Gt => FloatCC::GreaterThan,
        _ => FloatCC::GreaterThanOrEqual,
    }
}

/// An unboxed number a register is known to hold.
#[derive(Clone, Copy)]
enum Unboxed {
    None,
    F64(Value),
    I64(Value),
}

/// What compiled code knows about an object register beyond its slot.
#[derive(Clone, Copy)]
struct Virt {
    /// A pointer to the object (never null).
    ptr: Option<Value>,
    /// The slot holds `ptr`, as its reference. Otherwise the slot holds a
    /// stale reference (or null) that the next store releases, and `ptr`,
    /// if any, is borrowed.
    in_slot: bool,
    num: Unboxed,
    /// The object's exact type, when known (from how compiled code made
    /// it: a guard's knowledge does not survive the join with its slow
    /// path).
    ty: Option<usize>,
    /// `ptr` is an object the code keeps alive (a constant, an immortal).
    konst: bool,
    /// `ty` is its type, and that type's version and the object's inline
    /// values were verified since the last code that could change them
    /// (anything that runs Python code: a helper, a call).
    checked: bool,
    /// `ptr` is a global's value (borrowed from the module's dict): only
    /// Python code can release it, not a slot's old reference.
    global: bool,
}

impl Virt {
    fn ptr(p: Value, in_slot: bool) -> Virt {
        Virt { ptr: Some(p), in_slot, num: Unboxed::None, ty: None, konst: false, checked: false, global: false }
    }
}

/// How an arithmetic operand looks: an unboxed number or an object.
#[derive(Clone, Copy)]
enum Operand {
    F64(Value),
    I64(Value),
    Obj(Value),
}

/// What compiling a function needs about it: the one being compiled, or
/// one inlined into it.
struct Fn<'a> {
    u: &'a Unit<'a>,
    ir: &'a FuncIr,
    /// A block per instruction that can be reached other than by falling
    /// through (a jump target, an entry, a handler).
    labels: Vec<Option<Block>>,
    /// Per handler: stores the exception and jumps to the handler.
    catch_blocks: Vec<Block>,
    /// Its register slots.
    regs: Value,
    /// The variable of each `I64` and `Bool` register.
    vars: Vec<Option<Variable>>,
    vals: HashMap<Reg, Virt>,
    live: Vec<Vec<u64>>,
    /// Per label, its unboxed registers (`carried`), in the order of the
    /// block's parameters.
    carry: Vec<Vec<(Reg, Kind)>>,
    nparams: u32,
    /// The current block has ended (a jump, a return).
    done: bool,
    /// Set when it is inlined.
    inline: Option<Inlined>,
    /// What is known at its first instruction (an inlined function's
    /// parameters).
    seed: Vec<(Reg, Virt)>,
    /// The object registers whose slots its code may have written (a
    /// return releases them).
    written: std::collections::BTreeSet<Reg>,
    /// Those written only in cold blocks (slow paths), which set the
    /// function's `dirty` flag: a return releases them only when it is set.
    cold_written: std::collections::BTreeSet<Reg>,
    /// How deeply it is inlined (0: the function compiled).
    depth: u32,
    /// Emitting a path that leaves an inlined function's code (to the
    /// interpreter): its slot writes do not count in `written`.
    cold: bool,
}

/// An inlined callee: its frame and where it continues.
struct Inlined {
    frame: StackSlot,
    /// For the helpers (`Helpers::finish`, `Helpers::unwind`).
    desc: usize,
    /// The caller's call instruction.
    call_pc: usize,
    /// Its result (a new reference) continues the caller here.
    exit: Block,
    /// An exception raised at the parameter's pc leaves it.
    fail: Block,
    /// The caller raises at its call (the frame is released).
    raised: Block,
    /// Per parameter: the caller's register is dead after the call and its
    /// slot holds the argument's reference (a candidate for `Reuse`).
    donors: Vec<bool>,
    /// The arguments: the parameters' slots get them only where the frame
    /// is handed over (`prepare_frame`). None: a number the caller had
    /// unboxed (a *lazy* parameter): its slot is null until something needs
    /// the object, and then owns it.
    args: Vec<Option<Value>>,
    /// The immortal constant every `return` so far returned (None: none
    /// yet; Some(None): not all the same one).
    ret_const: Option<Option<usize>>,
    /// Every return returns an int: the result goes back unboxed (the exit
    /// block's second parameter, its first null) but from the interpreter.
    ret_int: bool,
}

struct Lower<'a> {
    b: FunctionBuilder<'a>,
    f: Fn<'a>,
    h: &'a Helpers,
    l: &'a Layout,
    ptr: Type,
    call_conv: CallConv,
    sigs: HashMap<(Vec<Type>, Option<Type>), SigRef>,
    /// Leave with `*pc` = the parameter and status `EXIT`, `ERROR`, `BAIL`.
    exit: Block,
    fail: Block,
    bail: Block,
    /// A `yield`: `YIELD` at its pc.
    yielded: Block,
    /// A `yield` of an int as it is: `YIELD_INT` at its pc.
    yielded_int: Block,
    /// `return`: its value (a new reference) and pc.
    ret_block: Block,
    ctx: Value,
    pcp: Value,
    /// The thread's id (free-threaded reference counting).
    tid: Value,
    /// Loop iterations left until the next safepoint; at `DIRTY`, set
    /// (non-zero) when a cold block stores to a slot (`cold_written`).
    /// Frames start with their slots null, or are released by their owner
    /// (the interpreter's): unset, those slots need no release.
    fuel: StackSlot,
    /// Emitting a slow path (all of it, whatever blocks it makes).
    slow: bool,
    /// Where direct calls build the callee's frame.
    frame: Option<StackSlot>,
    frontend: cranelift_codegen::isa::TargetFrontendConfig,
}

const MF: MemFlagsData = MemFlagsData::trusted();

impl<'a> Lower<'a> {
    fn new(mut b: FunctionBuilder<'a>, u: &'a Unit<'a>, h: &'a Helpers, l: &'a Layout, module: &JITModule) -> Lower<'a> {
        let ir = u.ir;
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let ps = b.block_params(entry).to_vec();
        let ptr = module.target_config().pointer_type();

        let is_label = label_pcs(ir, |pc| u.speculate && over_lists(u, l, pc));
        let live = liveness(ir);
        let carry = carried(u, l.numbers, &is_label, &live, &[]);
        let labels = label_blocks(&mut b, &is_label, &carry);
        let catch_blocks = (0..ir.handlers.len()).map(|_| b.create_block()).collect();
        let mut leave = || {
            let blk = b.create_block();
            b.append_block_param(blk, types::I32);
            blk
        };
        let (exit, fail, bail, yielded, yielded_int) = (leave(), leave(), leave(), leave(), leave());
        let ret_block = b.create_block();
        b.append_block_param(ret_block, ptr);
        b.append_block_param(ret_block, types::I32);
        let fuel = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 3));
        let vars = ir.reprs.iter().map(|r| matches!(r, Repr::I64 | Repr::Bool).then(|| b.declare_var(types::I64))).collect();
        let tid = match l.rc {
            Rc::Ft { .. } => b.ins().load(ptr, MF, ps[0], CTX_TID),
            _ => b.ins().iconst(ptr, 0),
        };
        let max_frame = frame_regs(u);
        let seed = Vec::new();
        let frame = max_frame.map(|n| b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, n.max(1) * 8, 3)));
        let mut lw = Lower {
            b,
            f: Fn { u, ir, labels, catch_blocks, regs: ps[1], vars, vals: HashMap::new(), live, carry, nparams: ir.params.len() as u32, done: false, inline: None, seed, written: Default::default(), cold_written: Default::default(), depth: 0, cold: false },
            h,
            l,
            ptr,
            call_conv: module.isa().default_call_conv(),
            sigs: HashMap::new(),
            exit,
            fail,
            bail,
            yielded,
            yielded_int,
            ret_block,
            ctx: ps[0],
            pcp: ps[2],
            tid,
            fuel,
            slow: false,
            frame,
            frontend: module.target_config(),
        };
        lw.entry(ps[3]);
        lw
    }

    /// Define the scalar variables from their slots, then dispatch on `*pc`:
    /// 0, a loop header, after a `yield`, or (with an exception pending) at
    /// the `yield`.
    fn entry(&mut self, pending: Value) {
        for r in 0..self.f.vars.len() {
            if let Some(v) = self.f.vars[r] {
                let x = self.b.ins().load(types::I64, MF, self.f.regs, r as i32 * 8);
                self.b.def_var(v, x);
            }
        }
        let fuel0 = self.b.ins().iconst(types::I64, FUEL);
        self.b.ins().stack_store(self.ptr, fuel0, self.fuel, 0);
        let z = self.b.ins().iconst(types::I64, 0);
        self.b.ins().stack_store(self.ptr, z, self.fuel, DIRTY);
        let pc = self.b.ins().load(types::I32, MF, self.pcp, 0);
        let resume = self.b.create_block();
        let throw = self.b.create_block();
        let bad = self.b.create_block();
        self.b.ins().brif(pending, throw, &[], resume, &[]);

        let ir = self.f.ir;
        let yields: Vec<usize> = ir.code.iter().enumerate().filter(|(_, i)| matches!(i, Instr::Yield { .. } | Instr::YieldFrom { .. })).map(|(p, _)| p).collect();
        self.b.switch_to_block(resume);
        // 0, loop headers (targets of backward jumps), after yields.
        let mut at = std::collections::BTreeSet::new();
        at.insert(0usize);
        for (pc, i) in ir.code.iter().enumerate() {
            for t in i.successors(pc as u32) {
                if (t as usize) <= pc {
                    at.insert(t as usize);
                }
            }
        }
        at.extend(yields.iter().map(|p| p + 1));
        let mut sw = Switch::new();
        let mut unbox = Vec::new();
        for p in at {
            if let Some(Some(blk)) = self.f.labels.get(p) {
                if self.f.carry[p].is_empty() {
                    sw.set_entry(p as u128, *blk);
                } else {
                    // Its unboxed registers from their slots.
                    let pre = self.b.create_block();
                    sw.set_entry(p as u128, pre);
                    unbox.push((p, pre));
                }
            }
        }
        sw.emit(&mut self.b, pc, bad);
        for (p, pre) in unbox {
            self.b.switch_to_block(pre);
            let fallback = self.b.create_block();
            let args = self.carried_args(p, p, fallback);
            let blk = self.f.labels[p].unwrap();
            self.b.ins().jump(blk, &args);
            self.b.switch_to_block(fallback);
            self.f.vals.clear();
            self.bail_to(p);
        }

        self.b.switch_to_block(throw);
        let mut sw = Switch::new();
        for &p in &yields {
            let blk = self.b.create_block();
            sw.set_entry(p as u128, blk);
        }
        let entries: Vec<(u128, Block)> = sw.entries().iter().map(|(k, v)| (*k, *v)).collect();
        sw.emit(&mut self.b, pc, bad);
        for (p, blk) in entries {
            self.b.switch_to_block(blk);
            self.raise(p as usize);
        }

        self.b.switch_to_block(bad);
        self.b.ins().trap(TrapCode::unwrap_user(1));
    }

    fn lower(mut self) {
        self.body();
        if !self.f.done {
            self.b.ins().trap(TrapCode::unwrap_user(2));
        }
        for k in 0..self.f.ir.handlers.len() {
            let h = &self.f.ir.handlers[k];
            let blk = self.f.catch_blocks[k];
            self.b.switch_to_block(blk);
            let exc = self.b.ins().iconst(types::I32, h.exc as i64);
            self.f.written.insert(h.exc);
            self.call(self.h.catch, &[self.ctx, self.f.regs, exc], None);
            match self.f.labels.get(h.target as usize).copied().flatten() {
                Some(t) => {
                    self.b.ins().jump(t, &[]);
                }
                None => {
                    self.b.ins().trap(TrapCode::unwrap_user(3));
                }
            }
        }
        // A return releases the frame's references the code may have stored
        // (its owner releases whatever else it holds).
        self.b.switch_to_block(self.ret_block);
        let (p, pc) = (self.b.block_params(self.ret_block)[0], self.b.block_params(self.ret_block)[1]);
        let zero = self.b.ins().iconst(types::I64, 0);
        for r in self.f.written.clone() {
            if self.is_param(r) {
                continue;
            }
            let v = self.slot_load(r);
            self.b.ins().store(MF, zero, self.f.regs, r as i32 * 8);
            self.xdecref(v);
        }
        let cold: Vec<Reg> = self.f.cold_written.iter().copied().filter(|r| !self.f.written.contains(r) && !self.is_param(*r)).collect();
        let regs = self.f.regs;
        self.release_if_dirty(&cold, regs, true);
        self.b.ins().store(MF, p, self.ctx, CTX_RET);
        self.b.ins().jump(self.exit, &[BlockArg::Value(pc)]);
        for (blk, status) in [(self.exit, EXIT), (self.fail, ERROR), (self.bail, BAIL), (self.yielded, YIELD), (self.yielded_int, YIELD_INT)] {
            self.b.switch_to_block(blk);
            let pc = self.b.block_params(blk)[0];
            self.b.ins().store(MF, pc, self.pcp, 0);
            let s = self.b.ins().iconst(types::I32, status as i64);
            self.b.ins().return_(&[s]);
        }
        self.b.seal_all_blocks();
        let config = self.frontend;
        self.b.finalize(config);
    }

    /// The instructions of the current function, from its first label.
    fn body(&mut self) {
        let n = self.f.ir.code.len();
        self.f.done = true;
        for pc in 0..n {
            if let Some(blk) = self.f.labels[pc] {
                if !self.f.done {
                    self.goto(pc - 1, pc as u32);
                }
                self.b.switch_to_block(blk);
                self.f.vals.clear();
                // An inlined function's parameters never change: what is
                // known of them holds at every label (but verified shapes,
                // after code that may have run Python code). Their slots are
                // only written where the frame is handed over.
                if pc == 0 || self.f.inline.is_some() {
                    for (r, mut v) in self.f.seed.clone() {
                        if pc != 0 {
                            v.checked = false;
                            if v.ty.is_some_and(|t| t != self.l.float_type && t != self.l.long_type) {
                                v.ty = None;
                            }
                        }
                        self.f.vals.insert(r, v);
                    }
                }
                let params = self.b.block_params(blk).to_vec();
                for (&(r, k), v) in self.f.carry[pc].iter().zip(params) {
                    let (num, ty) = if k == Kind::F { (Unboxed::F64(v), self.l.float_type) } else { (Unboxed::I64(v), self.l.long_type) };
                    self.f.vals.insert(r, Virt { ptr: None, in_slot: false, num, ty: Some(ty), konst: false, checked: false, global: false });
                }
                self.f.done = false;
            } else if self.f.done {
                // Unreachable.
                continue;
            }
            self.instr(pc);
        }
    }

    // ---- Calls, slots, reference counts ----

    fn call(&mut self, f: usize, args: &[Value], ret: Option<Type>) -> Option<Value> {
        let tys: Vec<Type> = args.iter().map(|a| self.b.func.dfg.value_type(*a)).collect();
        let key = (tys, ret);
        let sig = match self.sigs.get(&key) {
            Some(s) => *s,
            None => {
                let mut s = Signature::new(self.call_conv);
                s.params.extend(key.0.iter().map(|t| AbiParam::new(*t)));
                if let Some(r) = ret {
                    s.returns.push(AbiParam::new(r));
                }
                let r = self.b.import_signature(s);
                self.sigs.insert(key, r);
                r
            }
        };
        let fp = self.b.ins().iconst(self.ptr, f as i64);
        let call = self.b.ins().call_indirect(sig, fp, args);
        ret.map(|_| self.b.inst_results(call)[0])
    }

    fn i32c(&mut self, v: u32) -> Value {
        self.b.ins().iconst(types::I32, v as i64)
    }

    fn pc_val(&mut self, pc: usize) -> Value {
        self.i32c(pc as u32)
    }

    fn slot_load(&mut self, r: Reg) -> Value {
        self.b.ins().load(types::I64, MF, self.f.regs, r as i32 * 8)
    }

    fn slot_store(&mut self, v: Value, r: Reg) {
        if self.f.ir.reprs[r as usize] == Repr::Obj && !self.f.cold {
            self.note_write(r);
        }
        self.b.ins().store(MF, v, self.f.regs, r as i32 * 8);
    }

    /// Object register `r`'s slot is written here: in a cold block, it
    /// sets the `dirty` flag.
    fn note_write(&mut self, r: Reg) {
        let cold = self.slow || self.b.current_block().is_some_and(|b| self.b.func.layout.is_cold(b));
        if !cold {
            self.f.written.insert(r);
            return;
        }
        self.f.cold_written.insert(r);
        let one = self.b.ins().iconst(types::I64, 1);
        self.b.ins().stack_store(self.ptr, one, self.fuel, DIRTY);
    }

    /// Release the slots of `regs` (null or owned), when `dirty` is set.
    fn release_if_dirty(&mut self, regs: &[Reg], frame: Value, zero: bool) {
        if regs.is_empty() {
            return;
        }
        let (yes, done) = (self.b.create_block(), self.b.create_block());
        let d = self.b.ins().stack_load(self.ptr, types::I64, self.fuel, DIRTY);
        self.b.ins().brif(d, yes, &[], done, &[]);
        self.b.set_cold_block(yes);
        self.b.switch_to_block(yes);
        let z = self.b.ins().iconst(types::I64, 0);
        for &r in regs {
            let v = self.b.ins().load(types::I64, MF, frame, r as i32 * 8);
            if zero {
                self.b.ins().store(MF, z, frame, r as i32 * 8);
            }
            self.xdecref(v);
        }
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    fn repr(&self, r: Reg) -> Repr {
        self.f.ir.reprs[r as usize]
    }

    fn is_param(&self, r: Reg) -> bool {
        r < self.f.nparams && self.f.u.borrow_params
    }

    /// An inlined function's parameter passed unboxed (`Inlined::args`).
    fn lazy_param(&self, r: Reg) -> bool {
        self.f.inline.as_ref().is_some_and(|i| i.args.get(r as usize).is_some_and(Option::is_none))
    }

    fn live_at(&self, pc: usize, r: Reg) -> bool {
        self.f.live.get(pc).is_some_and(|l| l[r as usize / 64] & (1 << (r % 64)) != 0)
    }

    /// Continue in a new block (returned after switching to it).
    fn cont(&mut self) -> Block {
        let blk = self.b.create_block();
        self.b.switch_to_block(blk);
        blk
    }

    /// `p` is an `iconst` of None, True or False: immortal.
    fn immortal(&self, p: Value) -> bool {
        self.const_of(p).is_some_and(|v| v == self.l.none || v == self.l.true_ || v == self.l.false_)
    }

    /// The value of `p` if it is an `iconst`.
    fn const_of(&self, p: Value) -> Option<usize> {
        use cranelift_codegen::ir::{InstructionData, Opcode, ValueDef};
        let ValueDef::Result(inst, _) = self.b.func.dfg.value_def(p) else { return None };
        match self.b.func.dfg.insts[inst] {
            InstructionData::UnaryImm { opcode: Opcode::Iconst, imm } => Some(imm.bits() as usize),
            _ => None,
        }
    }

    fn incref(&mut self, p: Value) {
        if self.immortal(p) {
            return;
        }
        match self.l.rc {
            Rc::Calls => {
                self.call(self.h.incref, &[p], None);
            }
            Rc::Gil => {
                let c = self.b.ins().load(types::I32, MF, p, 0);
                let done = self.b.create_block();
                let inc = self.b.create_block();
                let imm = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, c, 0);
                self.b.ins().brif(imm, done, &[], inc, &[]);
                self.b.switch_to_block(inc);
                let c1 = self.b.ins().iadd_imm_s(c, 1);
                self.b.ins().store(MF, c1, p, 0);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(done);
            }
            Rc::Ft { tid, local, shared } => {
                let c = self.b.ins().load(types::I32, MF, p, local);
                let c1 = self.b.ins().iadd_imm_s(c, 1);
                let (done, mortal, mine, other) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
                self.b.ins().brif(c1, mortal, &[], done, &[]);
                self.b.switch_to_block(mortal);
                let owner = self.b.ins().load(self.ptr, MF, p, tid);
                let own = self.b.ins().icmp(IntCC::Equal, owner, self.tid);
                self.b.ins().brif(own, mine, &[], other, &[]);
                self.b.switch_to_block(mine);
                self.b.ins().store(MF, c1, p, local);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(other);
                let four = self.b.ins().iconst(types::I64, 4);
                let addr = self.b.ins().iadd_imm_s(p, shared as i64);
                self.b.ins().atomic_rmw(types::I64, MemFlagsData::trusted(), AtomicRmwOp::Add, addr, four);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(done);
            }
        }
    }

    fn decref(&mut self, p: Value) {
        if self.immortal(p) {
            return;
        }
        match self.l.rc {
            Rc::Calls => {
                self.call(self.h.decref, &[p], None);
            }
            Rc::Gil => {
                let c = self.b.ins().load(types::I32, MF, p, 0);
                let (done, dec, free) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                let imm = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, c, 0);
                self.b.ins().brif(imm, done, &[], dec, &[]);
                self.b.switch_to_block(dec);
                let c1 = self.b.ins().iadd_imm_s(c, -1);
                self.b.ins().store(MF, c1, p, 0);
                self.b.ins().brif(c1, done, &[], free, &[]);
                self.b.switch_to_block(free);
                self.call(self.h.dealloc, &[p], None);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(done);
            }
            Rc::Ft { tid, local, .. } => {
                let c = self.b.ins().load(types::I32, MF, p, local);
                let (done, mortal, mine, other, merge) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
                let imm = self.b.ins().icmp_imm_s(IntCC::Equal, c, -1);
                self.b.ins().brif(imm, done, &[], mortal, &[]);
                self.b.switch_to_block(mortal);
                let owner = self.b.ins().load(self.ptr, MF, p, tid);
                let own = self.b.ins().icmp(IntCC::Equal, owner, self.tid);
                self.b.ins().brif(own, mine, &[], other, &[]);
                self.b.switch_to_block(mine);
                let c1 = self.b.ins().iadd_imm_s(c, -1);
                self.b.ins().store(MF, c1, p, local);
                self.b.ins().brif(c1, done, &[], merge, &[]);
                self.b.switch_to_block(merge);
                self.call(self.h.merge_zero, &[p], None);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(other);
                self.call(self.h.dec_shared, &[p], None);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(done);
            }
        }
    }

    fn xdecref(&mut self, p: Value) {
        let (dec, done) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(p, dec, &[], done, &[]);
        self.b.switch_to_block(dec);
        self.decref(p);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    /// Store reference `p` in slot `r`, releasing the old one.
    fn put(&mut self, r: Reg, p: Value) {
        let old = self.slot_load(r);
        self.slot_store(p, r);
        self.xdecref(old);
    }

    fn ob_type(&mut self, p: Value) -> Value {
        self.b.ins().load(self.ptr, MF, p, self.l.ob_type)
    }

    /// Continue at `ok` when `cond`, else at `slow`.
    fn guard(&mut self, cond: Value, slow: Block) {
        let ok = self.b.create_block();
        self.b.ins().brif(cond, ok, &[], slow, &[]);
        self.b.switch_to_block(ok);
    }

    fn guard_eq(&mut self, x: Value, y: i64, slow: Block) {
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, x, y);
        self.guard(c, slow);
    }

    /// The type of `p` is `tp` at `version`.
    fn guard_type(&mut self, p: Value, tp: usize, version: u32, slow: Block) {
        let t = self.ob_type(p);
        self.guard_eq(t, tp as i64, slow);
        let v = self.b.ins().load(types::I32, MF, t, self.l.tp_version);
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, v, version as i32 as i64);
        self.guard(c, slow);
    }

    /// `guard_type` for register `r`'s object `p`: its type's version
    /// alone when the type is known.
    fn guard_type_of(&mut self, r: Reg, p: Value, tp: usize, version: u32, slow: Block) {
        if self.f.vals.get(&r).is_some_and(|v| v.ty == Some(tp)) {
            let t = self.b.ins().iconst(self.ptr, tp as i64);
            let v = self.b.ins().load(types::I32, MF, t, self.l.tp_version);
            let c = self.b.ins().icmp_imm_s(IntCC::Equal, v, version as i32 as i64);
            return self.guard(c, slow);
        }
        self.guard_type(p, tp, version, slow);
    }

    // ---- Virtual registers ----

    /// Give register `r` its object in its slot. Allocation failure
    /// raises at `pc`. Storing releases the slot's stale reference, which
    /// may be what keeps a borrowed pointer's object alive: those registers
    /// get their references first.
    fn materialize(&mut self, r: Reg, pc: usize) {
        // (A global's value is not released by a slot's old value.)
        let mut regs: Vec<Reg> = self.borrowed().into_iter().filter(|x| !self.f.vals[x].global).collect();
        regs.push(r);
        self.materialize_many(regs, pc);
    }

    /// The registers holding borrowed pointers.
    fn borrowed(&self) -> Vec<Reg> {
        let mut regs: Vec<Reg> = self.f.vals.iter().filter(|(_, v)| !v.in_slot && !v.konst && v.ptr.is_some()).map(|(r, _)| *r).collect();
        regs.sort_unstable();
        regs
    }

    /// Materialize `regs`: all references taken before any slot's old
    /// reference is released.
    fn materialize_many(&mut self, mut regs: Vec<Reg>, pc: usize) {
        regs.sort_unstable();
        regs.dedup();
        let mut owned = Vec::new();
        for r in regs {
            let Some(v) = self.f.vals.get(&r).copied() else { continue };
            if v.in_slot {
                continue;
            }
            let p = match (v.ptr, v.num) {
                (Some(p), _) => {
                    self.incref(p);
                    p
                }
                (None, Unboxed::F64(x)) => self.boxed(self.h.float_new, x, pc),
                (None, Unboxed::I64(x)) => self.boxed(self.h.long_new, x, pc),
                (None, Unboxed::None) => unreachable!("a virtual register without a value"),
            };
            owned.push((r, p, v));
        }
        for (r, p, v) in owned {
            self.put(r, p);
            self.f.vals.insert(r, Virt { ptr: Some(p), in_slot: true, konst: false, ..v });
        }
    }

    /// `new(x)` (a new reference); null raises at `pc`.
    /// `PyLong_FromLongLong(x)`: a small int's preallocated object without
    /// a call. Null on failure.
    fn new_int(&mut self, x: Value) -> Value {
        let Some(si) = self.l.small_ints else {
            return self.call(self.h.long_new, &[x], Some(self.ptr)).unwrap();
        };
        let done = self.b.create_block();
        let r = self.b.append_block_param(done, self.ptr);
        let (small, big) = (self.b.create_block(), self.b.create_block());
        let i = self.b.ins().iadd_imm_s(x, -si.min);
        let c = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, i, si.count);
        self.b.ins().brif(c, small, &[], big, &[]);
        self.b.switch_to_block(small);
        let off = self.b.ins().imul_imm_s(i, si.stride);
        let p = self.b.ins().iadd_imm_s(off, si.first as i64);
        self.b.ins().jump(done, &[BlockArg::Value(p)]);
        self.b.switch_to_block(big);
        let p = self.call(self.h.long_new, &[x], Some(self.ptr)).unwrap();
        self.b.ins().jump(done, &[BlockArg::Value(p)]);
        self.b.switch_to_block(done);
        r
    }

    fn boxed(&mut self, new: usize, x: Value, pc: usize) -> Value {
        let p = if new == self.h.long_new { self.new_int(x) } else { self.call(new, &[x], Some(self.ptr)).unwrap() };
        let failed = self.b.create_block();
        let ok = self.b.create_block();
        self.b.ins().brif(p, ok, &[], failed, &[]);
        self.b.switch_to_block(failed);
        self.raise(pc);
        self.b.switch_to_block(ok);
        p
    }

    /// Materialize the virtual registers live before `pc` (what a helper
    /// executing it, a handler or the interpreter will read) and forget the
    /// others.
    fn flush(&mut self, pc: usize) {
        // The scalar registers live there, into their slots.
        for r in 0..self.f.vars.len() {
            if let Some(v) = self.f.vars[r] {
                if self.live_at(pc, r as Reg) {
                    let x = self.b.use_var(v);
                    self.slot_store(x, r as Reg);
                }
            }
        }
        let mut regs: Vec<Reg> = self.f.vals.keys().copied().collect();
        regs.sort_unstable();
        let mut live = Vec::new();
        for r in regs {
            if self.f.vals[&r].in_slot {
                continue;
            }
            if self.live_at(pc, r) {
                live.push(r);
            } else {
                self.f.vals.remove(&r);
            }
        }
        self.materialize_many(live, pc);
    }

    /// Before code that may release objects borrowed pointers point into
    /// (a store, a call): the registers live after it (at `live_pc`) or in
    /// `keep` get their references; the others are forgotten.
    fn secure_borrowed(&mut self, live_pc: usize, pc: usize, keep: &[Reg]) {
        self.secure_borrowed_but(live_pc, pc, keep, None);
    }

    /// `secure_borrowed`, leaving `spare` as it is.
    fn secure_borrowed_but(&mut self, live_pc: usize, pc: usize, keep: &[Reg], spare: Option<Reg>) {
        let mut list = Vec::new();
        for r in self.borrowed() {
            let v = self.f.vals[&r];
            if Some(r) == spare {
                continue;
            }
            if keep.contains(&r) {
                list.push(r);
            } else if !self.live_at(live_pc, r) {
                self.f.vals.remove(&r);
            } else if !matches!(v.num, Unboxed::None) {
                // The value is known: drop the pointer instead.
                self.f.vals.insert(r, Virt { ptr: None, ..v });
            } else {
                list.push(r);
            }
        }
        self.materialize_many(list, pc);
    }

    /// The object in register `r` (not null); unbound continues at `slow`.
    fn obj(&mut self, r: Reg, pc: usize, slow: Block) -> Value {
        if let Some(v) = self.f.vals.get(&r).copied() {
            if let Some(p) = v.ptr {
                return p;
            }
            self.materialize(r, pc);
            return self.f.vals[&r].ptr.unwrap();
        }
        let p = self.slot_load(r);
        let ok = self.b.create_block();
        self.b.ins().brif(p, ok, &[], slow, &[]);
        self.b.switch_to_block(ok);
        self.f.vals.insert(r, Virt::ptr(p, true));
        p
    }

    /// Register `r` as an arithmetic operand.
    fn operand(&mut self, r: Reg, pc: usize, slow: Block) -> Operand {
        match self.f.vals.get(&r).map(|v| v.num) {
            Some(Unboxed::F64(x)) => Operand::F64(x),
            Some(Unboxed::I64(x)) => Operand::I64(x),
            _ => Operand::Obj(self.obj(r, pc, slow)),
        }
    }

    /// Python code may have run (a helper, a call): types' versions and
    /// objects' inline values are to be verified again.
    fn clobber(&mut self) {
        let (f, i) = (self.l.float_type, self.l.long_type);
        for v in self.f.vals.values_mut() {
            v.checked = false;
            // Python code may assign an instance's `__class__` (not a
            // float's or an int's).
            if v.ty.is_some_and(|t| t != f && t != i) {
                v.ty = None;
            }
        }
    }

    /// Forget what is known about the registers instruction `pc` writes (a
    /// helper wrote them), and reload the scalar ones.
    fn wrote(&mut self, pc: usize) {
        for r in self.f.ir.code[pc].writes() {
            self.f.vals.remove(&r);
            if self.f.ir.reprs[r as usize] == Repr::Obj {
                self.note_write(r);
            }
            if let Some(v) = self.f.vars[r as usize] {
                let x = self.slot_load(r);
                self.b.def_var(v, x);
            }
        }
    }

    /// Write scalar register `r` (its slot is written when something reads
    /// it: `flush`).
    fn set_var(&mut self, r: Reg, x: Value) {
        let v = self.f.vars[r as usize].expect("a scalar register");
        self.b.def_var(v, x);
    }

    fn get_var(&mut self, r: Reg) -> Value {
        let v = self.f.vars[r as usize].expect("a scalar register");
        self.b.use_var(v)
    }

    // ---- Control flow ----

    /// An exception was raised by instruction `p` (the registers are
    /// materialized): its innermost handler, else leave the function.
    fn raise(&mut self, p: usize) {
        if self.f.inline.is_some() {
            self.prepare_frame(None);
            let fail = self.f.inline.as_ref().unwrap().fail;
            let pc = self.pc_val(p);
            self.b.ins().jump(fail, &[BlockArg::Value(pc)]);
            return;
        }
        let at = p as u32;
        match self.f.ir.handlers.iter().rposition(|h| h.ranges.iter().any(|(s, e)| *s <= at && at < *e)) {
            Some(k) => {
                self.b.ins().jump(self.f.catch_blocks[k], &[]);
            }
            None => {
                let pc = self.pc_val(p);
                self.b.ins().jump(self.fail, &[BlockArg::Value(pc)]);
            }
        }
    }

    /// Stop here: the interpreter continues at `pc`.
    fn bail_to(&mut self, pc: usize) {
        // The frame goes to the interpreter: what this path stores is not
        // the code's to release.
        let cold = std::mem::replace(&mut self.f.cold, true);
        self.flush(pc);
        self.f.cold = cold;
        let v = self.pc_val(pc);
        self.b.ins().jump(self.bail, &[BlockArg::Value(v)]);
    }

    /// The values of label `t`'s unboxed registers, from the current state
    /// (unboxing slots, at `pc`); a value of another type continues at
    /// `fallback`.
    fn carried_args(&mut self, t: usize, pc: usize, fallback: Block) -> Vec<BlockArg> {
        let carried = self.f.carry[t].clone();
        let mut args = Vec::new();
        for (r, k) in carried {
            let v = match (k, self.f.vals.get(&r).map(|v| v.num)) {
                (Kind::F, Some(Unboxed::F64(x))) | (Kind::I, Some(Unboxed::I64(x))) => x,
                _ => {
                    let p = self.obj(r, pc, fallback);
                    if k == Kind::F {
                        let ty = self.ob_type(p);
                        self.guard_eq(ty, self.l.float_type as i64, fallback);
                        self.b.ins().load(types::F64, MF, p, self.l.fval)
                    } else {
                        self.as_i64(p, fallback)
                    }
                }
            };
            args.push(BlockArg::Value(v));
        }
        args
    }

    /// Continue at `target` from instruction `from`: the registers live
    /// there materialized, but those it keeps unboxed, which are passed as
    /// they are (or, if one is not a number of its kind, the interpreter
    /// continues at `target`). A backward jump spends fuel and runs a
    /// safepoint when it is gone.
    fn goto(&mut self, from: usize, target: u32) {
        let t = target as usize;
        let start = self.f.vals.clone();
        let fallback = self.b.create_block();
        let args = self.carried_args(t, from, fallback);
        let keep: Vec<Reg> = self.f.carry[t].iter().map(|(r, _)| *r).collect();
        // The others, materialized.
        let mut regs: Vec<Reg> = self.f.vals.keys().copied().filter(|r| !keep.contains(r)).collect();
        regs.sort_unstable();
        let mut live = Vec::new();
        for r in regs {
            // (An inlined function's parameters are known at every label.)
            if !self.f.vals[&r].in_slot && self.live_at(t, r) && !self.lazy_param(r) {
                live.push(r);
            }
        }
        self.materialize_many(live, from);
        let carried: Vec<(Reg, Kind, Value)> = self.f.carry[t].iter().zip(&args).map(|(&(r, k), a)| (r, k, if let BlockArg::Value(v) = a { *v } else { unreachable!() })).collect();
        self.f.vals.clear();
        let blk = self.f.labels[t].expect("a jump to a label");
        if t > from {
            self.b.ins().jump(blk, &args);
        } else {
            // The fuel lives in memory: no register for it in loops.
            let fuel = self.b.ins().stack_load(self.ptr, types::I64, self.fuel, 0);
            let fuel = self.b.ins().iadd_imm_s(fuel, -1);
            self.b.ins().stack_store(self.ptr, fuel, self.fuel, 0);
            let check = self.b.create_block();
            self.b.set_cold_block(check);
            self.b.ins().brif(fuel, blk, &args, check, &[]);
            self.b.switch_to_block(check);
            // The loop's values cross the call in memory, so that the loop
            // itself needs no registers that survive calls for them.
            let vars: Vec<(Reg, Variable)> = (0..self.f.vars.len()).filter_map(|r| self.f.vars[r].filter(|_| self.live_at(t, r as Reg)).map(|v| (r as Reg, v))).collect();
            let n = (carried.len() + vars.len()) as u32;
            let save = self.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, n.max(1) * 8, 3));
            for (i, (_, _, v)) in carried.iter().enumerate() {
                self.b.ins().stack_store(self.ptr, *v, save, i as i32 * 8);
            }
            for (i, (_, var)) in vars.iter().enumerate() {
                let x = self.b.use_var(*var);
                self.b.ins().stack_store(self.ptr, x, save, (carried.len() + i) as i32 * 8);
            }
            let call = self.call(self.h.safepoint, &[self.ctx], Some(types::I32)).unwrap();
            let carried: Vec<(Reg, Kind, Value)> = carried
                .iter()
                .enumerate()
                .map(|(i, &(r, k, v))| {
                    let ty = self.b.func.dfg.value_type(v);
                    (r, k, self.b.ins().stack_load(self.ptr, ty, save, i as i32 * 8))
                })
                .collect();
            for (i, (_, var)) in vars.iter().enumerate() {
                let x = self.b.ins().stack_load(self.ptr, types::I64, save, (carried.len() + i) as i32 * 8);
                self.b.def_var(*var, x);
            }
            let args: Vec<BlockArg> = carried.iter().map(|&(_, _, v)| BlockArg::Value(v)).collect();
            let refuel = self.b.ins().iconst(types::I64, FUEL);
            self.b.ins().stack_store(self.ptr, refuel, self.fuel, 0);
            let failed = self.b.create_block();
            self.b.set_cold_block(failed);
            self.b.ins().brif(call, failed, &[], blk, &args);
            self.b.switch_to_block(failed);
            // The unboxed registers too, for a handler.
            for (r, k, v) in carried {
                let num = if k == Kind::F { Unboxed::F64(v) } else { Unboxed::I64(v) };
                self.f.vals.insert(r, Virt { ptr: None, in_slot: false, num, ty: None, konst: false, checked: false, global: false });
            }
            let was = std::mem::replace(&mut self.slow, true);
            self.flush(from);
            self.slow = was;
            self.f.vals.clear();
            self.raise(from);
        }
        self.b.set_cold_block(fallback);
        self.b.switch_to_block(fallback);
        self.f.vals = start;
        self.bail_to(t);
        self.f.vals.clear();
    }

    /// A branch from instruction `from`: to `yes` when `cond`, else `no`.
    fn branch(&mut self, from: usize, cond: Value, yes: u32, no: u32) {
        let saved = self.f.vals.clone();
        let (y, n) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(cond, y, &[], n, &[]);
        self.b.switch_to_block(y);
        self.goto(from, yes);
        self.b.switch_to_block(n);
        self.f.vals = saved;
        self.goto(from, no);
        self.f.done = true;
    }

    /// Continue with the next instruction.
    fn next(&mut self, pc: usize) {
        if pc + 1 >= self.f.ir.code.len() {
            self.b.ins().trap(TrapCode::unwrap_user(2));
            self.f.done = true;
        }
        // A label is entered by `lower`.
    }

    /// Check a helper's status: anything but `NEXT` raises at `pc`.
    fn check_status(&mut self, st: Value, pc: usize) {
        let (ok, bad) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(st, bad, &[], ok, &[]);
        self.b.switch_to_block(bad);
        self.raise(pc);
        self.b.switch_to_block(ok);
    }

    /// Run instruction `pc` with a helper `f(args)` returning a status
    /// (`after_status` decides), the registers materialized first.
    fn generic(&mut self, pc: usize) {
        if self.f.inline.is_some() {
            return self.deopt(pc);
        }
        self.clobber();
        self.flush(pc);
        let ir = self.f.ir;
        let i = &ir.code[pc];
        if matches!(self.f.u.specs[pc], Spec::None) && matches!(i, Instr::LoadAttr { .. } | Instr::StoreAttr { .. } | Instr::LoadGlobal { .. } | Instr::Call { .. } | Instr::CallMethod { .. }) {
            self.count_miss();
        }
        let regs = self.f.regs;
        let ctx = self.ctx;
        let ops = |s: &mut Self, xs: &[u32]| -> Vec<Value> { xs.iter().map(|x| s.i32c(*x)).collect() };
        let h = *self.h;
        match i {
            Instr::LoadAttr { dst, obj, name } => {
                let mut a = vec![regs];
                a.extend(ops(self, &[*dst, *obj]));
                a.push(self.b.ins().iconst(self.ptr, self.f.u.names[*name as usize] as i64));
                a.push(self.b.ins().iconst(self.ptr, self.f.u.ics[pc] as i64));
                let st = self.call(h.load_attr, &a, Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::StoreAttr { obj, name, val } => {
                let mut a = vec![regs];
                a.extend(ops(self, &[*obj, *val]));
                a.push(self.b.ins().iconst(self.ptr, self.f.u.names[*name as usize] as i64));
                a.push(self.b.ins().iconst(self.ptr, self.f.u.ics[pc] as i64));
                let st = self.call(h.store_attr, &a, Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::Truthy { dst, src } => {
                let mut a = vec![regs];
                a.extend(ops(self, &[*dst, *src]));
                let st = self.call(h.truthy, &a, Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::BinOp { dst, op, a: x, b: y, inplace } => {
                let mut a = vec![regs];
                a.extend(ops(self, &[*dst, *x, *y, *op as u32 + if *inplace { 16 } else { 0 }]));
                let st = self.call(h.binop, &a, Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::Compare { dst, op, a: x, b: y } if rich_code(*op).is_some() => {
                let mut a = vec![regs];
                a.extend(ops(self, &[*dst, *x, *y, rich_code(*op).unwrap()]));
                let st = self.call(h.compare, &a, Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::CompareBool { dst, op, a: x, b: y } if rich_code(*op).is_some() => {
                let mut a = vec![regs];
                a.extend(ops(self, &[*dst, *x, *y, rich_code(*op).unwrap()]));
                let st = self.call(h.compare_bool, &a, Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::Call { .. } | Instr::CallMethod { .. } => {
                let p = self.pc_val(pc);
                let st = self.call(h.call, &[ctx, regs, p], Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::LoadGlobal { dst, .. } => {
                let (d, p) = (self.i32c(*dst), self.pc_val(pc));
                let st = self.call(h.load_global, &[ctx, regs, d, p], Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::Check { reg, check } => {
                let (r, c, p) = (self.i32c(*reg), self.i32c(*check), self.pc_val(pc));
                let st = self.call(h.check, &[ctx, regs, r, c, p], Some(types::I32)).unwrap();
                self.check_status(st, pc);
            }
            Instr::ForIter { dst, iter, target } => {
                let (d, it, p) = (self.i32c(*dst), self.i32c(*iter), self.pc_val(pc));
                let st = self.call(h.for_iter, &[ctx, regs, d, it, p], Some(types::I32)).unwrap();
                self.wrote(pc);
                // NEXT continues, TAKEN jumps, ERROR raises.
                let (next, other, taken, raised) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
                self.b.ins().brif(st, other, &[], next, &[]);
                self.b.switch_to_block(other);
                let is_taken = self.b.ins().icmp_imm_u(IntCC::Equal, st, TAKEN as i64);
                self.b.ins().brif(is_taken, taken, &[], raised, &[]);
                self.b.switch_to_block(raised);
                self.raise(pc);
                self.b.switch_to_block(taken);
                self.goto(pc, *target);
                self.b.switch_to_block(next);
                self.goto(pc, pc as u32 + 1);
                self.f.done = true;
                return;
            }
            _ => {
                self.step(pc);
                return;
            }
        }
        self.wrote(pc);
    }

    /// Count a generic execution compiled without a spec; the `MISSES`th
    /// tells the runtime.
    fn count_miss(&mut self) {
        if self.f.inline.is_some() {
            return;
        }
        let addr = self.b.ins().iconst(self.ptr, self.f.u.misses as i64);
        let n = self.b.ins().load(types::I32, MF, addr, 0);
        let n = self.b.ins().iadd_imm_s(n, 1);
        self.b.ins().store(MF, n, addr, 0);
        let hit = self.b.ins().icmp_imm_s(IntCC::Equal, n, MISSES as i64);
        let (tell, done) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(hit, tell, &[], done, &[]);
        self.b.switch_to_block(tell);
        self.call(self.h.missed, &[self.ctx], None);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    /// Execute instruction `pc` with the `step` helper (registers
    /// materialized) and continue as it says.
    fn step(&mut self, pc: usize) {
        if self.f.inline.is_some() {
            return self.deopt(pc);
        }
        self.clobber();
        self.flush(pc);
        let p = self.pc_val(pc);
        let st = self.call(self.h.step, &[self.ctx, self.f.regs, p], Some(types::I32)).unwrap();
        let ir = self.f.ir;
        let i = &ir.code[pc];
        let other = self.b.create_block();
        let next = self.b.create_block();
        self.b.ins().brif(st, other, &[], next, &[]);
        self.b.switch_to_block(other);
        let raised = self.b.create_block();
        match i {
            Instr::ForIter { target, .. } | Instr::JumpIfNotSelf { target, .. } => {
                let taken = self.b.create_block();
                let is_taken = self.b.ins().icmp_imm_u(IntCC::Equal, st, TAKEN as i64);
                self.b.ins().brif(is_taken, taken, &[], raised, &[]);
                self.b.switch_to_block(taken);
                self.wrote(pc);
                self.goto(pc, *target);
            }
            Instr::Return { .. } | Instr::Yield { .. } | Instr::YieldFrom { .. } => {
                let exited = self.b.create_block();
                let is_exit = self.b.ins().icmp_imm_u(IntCC::Equal, st, EXIT as i64);
                self.b.ins().brif(is_exit, exited, &[], raised, &[]);
                self.b.switch_to_block(exited);
                let pcv = self.pc_val(pc);
                self.b.ins().jump(self.exit, &[BlockArg::Value(pcv)]);
            }
            _ => {
                self.b.ins().jump(raised, &[]);
            }
        }
        self.b.switch_to_block(raised);
        self.raise(pc);
        self.b.switch_to_block(next);
        self.wrote(pc);
        let succ = i.successors(pc as u32);
        if succ.is_empty() {
            self.b.ins().trap(TrapCode::unwrap_user(4));
            self.f.done = true;
        } else if succ.len() > 1 || succ[0] as usize != pc + 1 {
            self.goto(pc, pc as u32 + 1);
            self.f.done = true;
        } else {
            self.next(pc);
        }
    }

    /// In an inlined function: the interpreter runs the rest of the call
    /// from instruction `pc` (`Helpers::finish`), and the caller goes on
    /// with its result. Code emitted after this is unreachable.
    fn deopt(&mut self, pc: usize) {
        let cold = std::mem::replace(&mut self.f.cold, true);
        self.prepare_frame(Some(pc));
        self.flush(pc);
        self.f.cold = cold;
        let inl = self.f.inline.as_ref().unwrap();
        let (frame, desc, exit, call_pc) = (inl.frame, inl.desc, inl.exit, inl.call_pc);
        let raised = inl.raised;
        // The lazy parameters' objects: `finish` takes its own references
        // and releases the frame.
        let lazy: Vec<Value> = (0..inl.args.len() as Reg).filter(|&r| self.lazy_param(r)).collect::<Vec<_>>().into_iter().map(|r| self.slot_load(r)).collect();
        let fp = self.b.ins().stack_addr(self.ptr, frame, 0);
        let d = self.b.ins().iconst(self.ptr, desc as i64);
        let pcv = self.pc_val(pc);
        let r = self.call(self.h.finish, &[self.ctx, fp, d, pcv], Some(self.ptr)).unwrap();
        for p in lazy {
            self.decref(p);
        }
        let z = self.b.ins().iconst(types::I64, 0);
        self.b.ins().brif(r, exit, &[BlockArg::Value(r), BlockArg::Value(z)], raised, &[]);
        let _ = call_pc;
        self.cont();
    }

    /// Before an inlined function's frame is handed over (to the
    /// interpreter, or released after an exception): its parameters in their
    /// slots, and the object slots its code has not written so far null
    /// (instructions run in pc order: nothing emitted later ran yet).
    /// For the interpreter (`at`: it continues there), the lazy parameters
    /// are boxed into their slots (a failure raises at `at`); else (an
    /// exception) they are left as they are.
    fn prepare_frame(&mut self, at: Option<usize>) {
        let inl = self.f.inline.as_ref().unwrap();
        let regs = self.f.regs;
        let args = inl.args.clone();
        for (i, p) in args.iter().enumerate() {
            if let Some(p) = p {
                self.b.ins().store(MF, *p, regs, i as i32 * 8);
            }
        }
        let z = self.b.ins().iconst(types::I64, 0);
        for r in args.len()..self.f.ir.reprs.len() {
            if self.f.ir.reprs[r] == Repr::Obj && !self.f.written.contains(&(r as Reg)) && !self.f.cold_written.contains(&(r as Reg)) {
                self.b.ins().store(MF, z, regs, r as i32 * 8);
            }
        }
        let Some(pc) = at else { return };
        let lazy: Vec<Reg> = (0..args.len() as Reg).filter(|&r| args[r as usize].is_none() && self.f.vals.get(&r).is_some_and(|v| !v.in_slot)).collect();
        for r in lazy {
            let v = self.f.vals[&r];
            let p = match v.num {
                Unboxed::I64(x) => self.new_int(x),
                Unboxed::F64(x) => self.call(self.h.float_new, &[x], Some(self.ptr)).unwrap(),
                Unboxed::None => v.ptr.expect("a lazy parameter's value"),
            };
            if v.ptr.is_some() {
                self.incref(p);
            }
            let (ok, failed) = (self.b.create_block(), self.b.create_block());
            self.b.ins().brif(p, ok, &[], failed, &[]);
            self.b.switch_to_block(failed);
            let fail = self.f.inline.as_ref().unwrap().fail;
            let pcv = self.pc_val(pc);
            self.b.ins().jump(fail, &[BlockArg::Value(pcv)]);
            self.b.switch_to_block(ok);
            self.put(r, p);
            self.f.vals.insert(r, Virt { ptr: Some(p), in_slot: true, konst: false, ..v });
        }
    }

    /// Emit a speculative instruction's failure path: from the state at its
    /// start, the interpreter executes it (in an inlined function: finishes
    /// the call). The fast path's state stays: nothing joins it.
    fn bail_path(&mut self, blk: Block, start: HashMap<Reg, Virt>, pc: usize) {
        let fast = std::mem::replace(&mut self.f.vals, start);
        let done = self.f.done;
        self.b.set_cold_block(blk);
        self.b.switch_to_block(blk);
        let was = std::mem::replace(&mut self.slow, true);
        if self.f.inline.is_some() {
            self.deopt(pc);
        } else {
            self.bail_to(pc);
        }
        self.slow = was;
        self.f.vals = fast;
        self.f.done = done;
    }

    /// Whether register `r` is known to be an instance of `tp` with the
    /// type's version and its inline values verified.
    fn verified(&self, r: Reg, tp: usize) -> bool {
        self.f.vals.get(&r).is_some_and(|v| v.ty == Some(tp) && v.checked)
    }

    fn set_verified(&mut self, r: Reg, tp: usize) {
        if let Some(v) = self.f.vals.get_mut(&r) {
            v.ty = Some(tp);
            v.checked = true;
        }
    }

    /// Emit an instruction's slow path: from the state at the start of the
    /// instruction (`start`), the generic execution; then `then` for what
    /// follows it (a jump to the join with the fast path). What is known
    /// after the join is what was known at the start (values the fast path
    /// made do not dominate the join; what it materialized stays correct
    /// as not materialized), but for the instruction's writes, which the
    /// caller describes.
    fn slow_path(&mut self, slow: Block, start: HashMap<Reg, Virt>, pc: usize, then: impl FnOnce(&mut Self)) {
        self.f.vals = start.clone();
        let done = self.f.done;
        self.b.set_cold_block(slow);
        self.b.switch_to_block(slow);
        let was = std::mem::replace(&mut self.slow, true);
        if self.f.inline.is_some() {
            self.deopt(pc);
        } else {
            self.generic(pc);
            then(self);
        }
        self.slow = was;
        self.f.vals = start;
        for r in self.f.ir.code[pc].writes() {
            self.f.vals.remove(&r);
        }
        self.f.done = done;
    }

    // ---- Instructions ----

    fn instr(&mut self, pc: usize) {
        let ir = self.f.ir;
        let i = &ir.code[pc];
        match i {
            Instr::Jump { target } => {
                self.goto(pc, *target);
                self.f.done = true;
            }
            Instr::JumpIfFalse { cond, target } if self.f.vars[*cond as usize].is_some() => {
                let c = self.get_var(*cond);
                self.branch(pc, c, pc as u32 + 1, *target);
            }
            Instr::JumpIfTrue { cond, target } if self.f.vars[*cond as usize].is_some() => {
                let c = self.get_var(*cond);
                self.branch(pc, c, *target, pc as u32 + 1);
            }
            Instr::I64Const { dst, value } => {
                let v = self.b.ins().iconst(types::I64, *value);
                self.set_var(*dst, v);
                self.next(pc);
            }
            Instr::I64Lt { dst, a, b } => {
                let (x, y) = (self.get_var(*a), self.get_var(*b));
                let c = self.b.ins().icmp(IntCC::SignedLessThan, x, y);
                let c = self.b.ins().uextend(types::I64, c);
                self.set_var(*dst, c);
                self.next(pc);
            }
            Instr::I64Add { dst, a, b } => {
                let (x, y) = (self.get_var(*a), self.get_var(*b));
                let v = self.b.ins().iadd(x, y);
                self.set_var(*dst, v);
                self.next(pc);
            }
            Instr::Not { dst, src } if self.f.vars[*src as usize].is_some() && self.f.vars[*dst as usize].is_some() => {
                let v = self.get_var(*src);
                let v = self.b.ins().bxor_imm_u(v, 1);
                self.set_var(*dst, v);
                self.next(pc);
            }
            Instr::F32Bin { dst, op, a, b } if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) => {
                let x = self.b.ins().load(types::F32, MF, self.f.regs, *a as i32 * 8);
                let y = self.b.ins().load(types::F32, MF, self.f.regs, *b as i32 * 8);
                let v = match op {
                    BinOp::Add => self.b.ins().fadd(x, y),
                    BinOp::Sub => self.b.ins().fsub(x, y),
                    BinOp::Mul => self.b.ins().fmul(x, y),
                    _ => self.b.ins().fdiv(x, y),
                };
                self.b.ins().store(MF, v, self.f.regs, *dst as i32 * 8);
                self.next(pc);
            }
            Instr::Move { dst, src } | Instr::Take { dst, src } if self.f.vars[*src as usize].is_some() && self.f.vars[*dst as usize].is_some() => {
                let v = self.get_var(*src);
                self.set_var(*dst, v);
                self.next(pc);
            }
            Instr::Move { dst, src } | Instr::Take { dst, src } if self.repr(*src) == Repr::F32 && self.repr(*dst) == Repr::F32 => {
                let v = self.slot_load(*src);
                self.slot_store(v, *dst);
                self.next(pc);
            }
            Instr::Unbind { reg } if self.repr(*reg) != Repr::Obj => self.next(pc),
            Instr::Unbind { reg } => {
                self.f.vals.remove(reg);
                if !self.is_param(*reg) {
                    self.secure_borrowed(pc + 1, pc, &[]);
                    let old = self.slot_load(*reg);
                    let z = self.b.ins().iconst(types::I64, 0);
                    self.slot_store(z, *reg);
                    self.xdecref(old);
                }
                self.next(pc);
            }
            Instr::Const { dst, idx } => {
                let p = self.b.ins().iconst(self.ptr, self.f.u.consts[*idx as usize] as i64);
                let (num, ty) = match self.f.u.nums[*idx as usize] {
                    Num::F64(x) => (Unboxed::F64(self.b.ins().f64const(x)), Some(self.l.float_type)),
                    Num::I64(x) => (Unboxed::I64(self.b.ins().iconst(types::I64, x)), Some(self.l.long_type)),
                    Num::Other => (Unboxed::None, None),
                };
                self.f.vals.insert(*dst, Virt { ptr: Some(p), in_slot: false, num, ty, konst: true, checked: false, global: false });
                self.next(pc);
            }
            Instr::Box { dst, src } if self.repr(*src) == Repr::I64 && self.l.numbers => {
                let v = self.get_var(*src);
                self.f.vals.insert(*dst, Virt { ptr: None, in_slot: false, num: Unboxed::I64(v), ty: Some(self.l.long_type), konst: false, checked: false, global: false });
                self.next(pc);
            }
            Instr::Box { dst, src } if self.repr(*src) == Repr::Bool => {
                let v = self.get_var(*src);
                let (t, f) = (self.b.ins().iconst(self.ptr, self.l.true_ as i64), self.b.ins().iconst(self.ptr, self.l.false_ as i64));
                let p = self.b.ins().select(v, t, f);
                self.f.vals.insert(*dst, Virt { ptr: Some(p), in_slot: false, num: Unboxed::None, ty: None, konst: true, checked: false, global: false });
                self.next(pc);
            }
            Instr::Move { dst, src } if self.repr(*src) == Repr::Obj && self.repr(*dst) == Repr::Obj => self.move_obj(pc, *dst, *src, false),
            Instr::Take { dst, src } if self.repr(*src) == Repr::Obj && self.repr(*dst) == Repr::Obj => self.move_obj(pc, *dst, *src, true),
            Instr::LoadAttr { dst, obj, .. } => self.load_attr(pc, *dst, *obj),
            Instr::StoreAttr { obj, val, .. } => self.store_attr(pc, *obj, *val),
            Instr::LoadGlobal { dst, .. } => self.load_global(pc, *dst),
            Instr::Check { reg, .. } => self.check(pc, *reg),
            Instr::BinOp { dst, op, a, b, .. } => {
                let hint = ir.hints.get(pc).copied().unwrap_or(0);
                if hint == HINT_FLOAT && self.l.numbers && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
                    self.float_binop(pc, *dst, *op, *a, *b);
                } else if hint == HINT_INT && self.l.numbers && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) {
                    self.int_binop(pc, *dst, *op, *a, *b);
                } else {
                    self.generic(pc);
                    self.next(pc);
                }
            }
            Instr::CompareBool { dst, op, a, b } if self.f.vars[*dst as usize].is_some() && rich_code(*op).is_some() && self.l.numbers => self.compare_bool(pc, *dst, *op, *a, *b),
            Instr::CompareBool { dst, op: op @ (CmpOp::Is | CmpOp::IsNot), a, b } if self.f.vars[*dst as usize].is_some() => self.compare_is(pc, *dst, *op, *a, *b),
            Instr::Truthy { dst, src } if self.f.vars[*dst as usize].is_some() => self.truthy(pc, *dst, *src),
            Instr::Return { src } if self.repr(*src) == Repr::Obj => self.ret(pc, *src),
            Instr::Call { dst, func, args, kwnames, .. } if kwnames.is_empty() && matches!(self.f.u.specs[pc], Spec::Call(_)) => self.call_direct(pc, *dst, Some(*func), None, args),
            Instr::CallMethod { dst, obj, args, kwnames, .. } if kwnames.is_empty() && matches!(self.f.u.specs[pc], Spec::Call(_)) => self.call_direct(pc, *dst, None, Some(*obj), args),
            Instr::Truthy { .. } | Instr::Compare { .. } | Instr::CompareBool { .. } | Instr::Call { .. } | Instr::CallMethod { .. } => {
                self.generic(pc);
                self.next(pc);
            }
            Instr::Yield { src, .. } if self.f.inline.is_none() && self.repr(*src) == Repr::Obj => self.yield_value(pc, *src),
            Instr::JumpIfNotSelf { func, depth, target } => match self.f.u.specs[pc] {
                Spec::SelfFunc(f) if self.f.inline.is_none() && self.f.vars[*depth as usize].is_some() => self.jump_if_not_self(pc, *func, *depth, *target, f),
                _ => self.step(pc),
            },
            Instr::ForIter { dst, iter, target } if over_lists(self.f.u, self.l, pc) && self.repr(*dst) == Repr::Obj => self.for_iter_list(pc, *dst, *iter, *target),
            Instr::ForIter { dst, iter, target } if self.f.inline.is_none() && self.repr(*dst) == Repr::Obj => self.for_iter_call(pc, *dst, *iter, *target),
            Instr::ForIter { .. } => self.generic(pc),
            _ => self.step(pc),
        }
    }

    /// `Move` (`take`: `Take`) of an object.
    fn move_obj(&mut self, pc: usize, dst: Reg, src: Reg, take: bool) {
        let take = take && !self.is_param(src);
        if let Some(v) = self.f.vals.get(&src).copied().filter(|v| !v.in_slot) {
            // A value not in its slot moves as it is (a `Take` leaves the
            // source's stale slot to the next store).
            if take {
                self.f.vals.remove(&src);
            }
            self.f.vals.insert(dst, v);
            return self.next(pc);
        }
        // Storing releases `dst`'s old reference.
        self.secure_borrowed(pc + 1, pc, &[]);
        // Known: no slow path, and what is known moves along.
        if let Some(v @ Virt { ptr: Some(p), .. }) = self.f.vals.get(&src).copied() {
            if take {
                let z = self.b.ins().iconst(types::I64, 0);
                self.slot_store(z, src);
                self.f.vals.remove(&src);
            } else {
                self.incref(p);
            }
            self.put(dst, p);
            self.f.vals.insert(dst, Virt { in_slot: true, ..v });
            return self.next(pc);
        }
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let p = self.obj(src, pc, slow);
        let v = self.f.vals[&src];
        if take {
            let z = self.b.ins().iconst(types::I64, 0);
            self.slot_store(z, src);
            self.f.vals.remove(&src);
        } else {
            self.incref(p);
        }
        self.put(dst, p);
        self.f.vals.insert(dst, Virt { in_slot: true, ..v });
        let join = self.b.create_block();
        self.b.ins().jump(join, &[]);
        // Unbound: `step` raises.
        self.slow_path(slow, start, pc, |s| {
            s.b.ins().jump(join, &[]);
        });
        self.b.switch_to_block(join);
        self.next(pc);
    }

    fn load_attr(&mut self, pc: usize, dst: Reg, obj: Reg) {
        let (tp, version, off, valid) = match self.f.u.specs[pc] {
            Spec::Inline { tp, version, off, values, .. } => (tp, version, off, Some(values + 3)),
            Spec::Member { tp, version, off } => (tp, version, off, None),
            _ => {
                self.generic(pc);
                return self.next(pc);
            }
        };
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let p = self.obj(obj, pc, slow);
        if !self.verified(obj, tp) {
            self.guard_type_of(obj, p, tp, version, slow);
            if let Some(valid) = valid {
                let v = self.b.ins().load(types::I8, MF, p, valid);
                self.guard(v, slow);
            }
        }
        let v = self.b.ins().load(self.ptr, MF, p, off);
        if self.f.u.speculate {
            // Unset: the interpreter decides (AttributeError, `__getattr__`).
            self.guard(v, slow);
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            self.set_verified(obj, tp);
            self.f.vals.insert(dst, Virt::ptr(v, false));
            return self.next(pc);
        }
        let join = self.b.create_block();
        let x = self.b.append_block_param(join, self.ptr);
        let ok = self.b.create_block();
        self.b.ins().brif(v, ok, &[], slow, &[]);
        self.b.switch_to_block(ok);
        self.b.ins().jump(join, &[BlockArg::Value(v)]);
        self.slow_path(slow, start, pc, |s| {
            let v = s.slot_load(dst);
            s.b.ins().jump(join, &[BlockArg::Value(v)]);
        });
        self.b.switch_to_block(join);
        self.f.vals.insert(dst, Virt::ptr(x, false));
        self.next(pc);
    }

    fn store_attr(&mut self, pc: usize, obj: Reg, val: Reg) {
        let (tp, version, off, values, dict) = match self.f.u.specs[pc] {
            Spec::Inline { tp, version, off, values, dict } => (tp, version, off, Some(values), dict),
            Spec::Member { tp, version, off } => (tp, version, off, None, 0),
            _ => {
                self.generic(pc);
                return self.next(pc);
            }
        };
        // The store may release what borrowed pointers point to.
        self.secure_borrowed(pc + 1, pc, &[obj, val]);
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let p = self.obj(obj, pc, slow);
        if !self.verified(obj, tp) {
            self.guard_type_of(obj, p, tp, version, slow);
            if let Some(values) = values {
                let v = self.b.ins().load(types::I8, MF, p, values + 3);
                self.guard(v, slow);
                let d = self.b.ins().load(self.ptr, MF, p, dict);
                self.guard_eq(d, 0, slow);
            }
        }
        let old = self.b.ins().load(self.ptr, MF, p, off);
        // A first assignment of an inline value also appends its index to
        // the values' insertion order (`order[size++] = ix`), if there is
        // room (else: generic).
        let first = values.map(|values| {
            let ix = ((off - values - 8) / 8) as i64;
            let is_first = self.b.ins().icmp_imm_s(IntCC::Equal, old, 0);
            let (check, go) = (self.b.create_block(), self.b.create_block());
            self.b.ins().brif(is_first, check, &[], go, &[]);
            self.b.switch_to_block(check);
            let cap = self.b.ins().load(types::I8, MF, p, values);
            let size = self.b.ins().load(types::I8, MF, p, values + 1);
            let c1 = self.b.ins().icmp_imm_u(IntCC::UnsignedGreaterThan, cap, ix);
            self.guard(c1, slow);
            let c2 = self.b.ins().icmp(IntCC::UnsignedLessThan, size, cap);
            self.guard(c2, slow);
            self.b.ins().jump(go, &[]);
            self.b.switch_to_block(go);
            (is_first, values, ix)
        });
        let v = self.f.vals.get(&val).copied();
        let done = self.b.create_block();
        // The new reference to store, unless a float is updated in place:
        // one only the old value's slot referenced.
        let new = match v {
            Some(Virt { num: Unboxed::F64(x), in_slot: false, ptr: None, .. }) => {
                let (reuse, fresh, check) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                self.b.ins().brif(old, check, &[], fresh, &[]);
                self.b.switch_to_block(check);
                self.unique_float(old, reuse, fresh);
                self.b.switch_to_block(reuse);
                self.b.ins().store(MF, x, old, self.l.fval);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(fresh);
                // A new float, its reference handed to the attribute.
                let f = self.call(self.h.float_new, &[x], Some(self.ptr)).unwrap();
                self.guard(f, slow);
                f
            }
            Some(Virt { ptr: None, in_slot: false, num, .. }) => {
                let f = match num {
                    Unboxed::F64(x) => self.call(self.h.float_new, &[x], Some(self.ptr)).unwrap(),
                    Unboxed::I64(x) => self.new_int(x),
                    Unboxed::None => unreachable!(),
                };
                self.guard(f, slow);
                f
            }
            _ => {
                let x = self.obj(val, pc, slow);
                self.incref(x);
                x
            }
        };
        self.b.ins().store(MF, new, p, off);
        match first {
            Some((is_first, values, ix)) => {
                let (record, release) = (self.b.create_block(), self.b.create_block());
                self.b.ins().brif(is_first, record, &[], release, &[]);
                self.b.switch_to_block(record);
                let cap = self.b.ins().load(types::I8, MF, p, values);
                let size = self.b.ins().load(types::I8, MF, p, values + 1);
                let cap64 = self.b.ins().uextend(types::I64, cap);
                let size64 = self.b.ins().uextend(types::I64, size);
                let at = self.b.ins().ishl_imm_u(cap64, 3);
                let at = self.b.ins().iadd(at, size64);
                let at = self.b.ins().iadd(at, p);
                let ixv = self.b.ins().iconst(types::I8, ix);
                self.b.ins().store(MF, ixv, at, values + 8);
                let size1 = self.b.ins().iadd_imm_s(size, 1);
                self.b.ins().store(MF, size1, p, values + 1);
                self.b.ins().jump(done, &[]);
                self.b.switch_to_block(release);
                self.decref(old);
            }
            // A `__slots__` member may be unset.
            None => self.xdecref(old),
        }
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
        if self.f.u.speculate {
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            self.set_verified(obj, tp);
            return self.next(pc);
        }
        let join = self.b.create_block();
        self.b.ins().jump(join, &[]);
        self.slow_path(slow, start, pc, |s| {
            s.b.ins().jump(join, &[]);
        });
        self.b.switch_to_block(join);
        self.next(pc);
    }

    /// Continue at `yes` when `p` is an exact float only the current
    /// thread references once, else at `no`.
    fn unique_float(&mut self, p: Value, yes: Block, no: Block) {
        let t = self.ob_type(p);
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, t, self.l.float_type as i64);
        let next = self.b.create_block();
        self.b.ins().brif(c, next, &[], no, &[]);
        self.b.switch_to_block(next);
        match self.l.rc {
            Rc::Calls => {
                self.b.ins().jump(no, &[]);
            }
            Rc::Gil => {
                let c = self.b.ins().load(types::I32, MF, p, 0);
                let one = self.b.ins().icmp_imm_s(IntCC::Equal, c, 1);
                self.b.ins().brif(one, yes, &[], no, &[]);
            }
            Rc::Ft { tid, local, shared } => {
                let owner = self.b.ins().load(self.ptr, MF, p, tid);
                let own = self.b.ins().icmp(IntCC::Equal, owner, self.tid);
                let c = self.b.ins().load(types::I32, MF, p, local);
                let one = self.b.ins().icmp_imm_s(IntCC::Equal, c, 1);
                let s = self.b.ins().load(types::I64, MF, p, shared);
                let none = self.b.ins().icmp_imm_s(IntCC::Equal, s, 0);
                let a = self.b.ins().band(own, one);
                let a = self.b.ins().band(a, none);
                self.b.ins().brif(a, yes, &[], no, &[]);
            }
        }
    }

    fn load_global(&mut self, pc: usize, dst: Reg) {
        let Spec::Global { epoch, value } = self.f.u.specs[pc] else {
            self.generic(pc);
            return self.next(pc);
        };
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let addr = self.b.ins().iconst(self.ptr, self.l.epoch as i64);
        let e = self.b.ins().load(types::I64, MF, addr, 0);
        self.guard_eq(e, epoch as i64, slow);
        if self.f.u.speculate {
            // A changed global: the interpreter goes on.
            let v = self.b.ins().iconst(self.ptr, value as i64);
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            self.f.vals.insert(dst, Virt { global: true, ..Virt::ptr(v, false) });
            return self.next(pc);
        }
        let join = self.b.create_block();
        let x = self.b.append_block_param(join, self.ptr);
        let v = self.b.ins().iconst(self.ptr, value as i64);
        self.b.ins().jump(join, &[BlockArg::Value(v)]);
        self.slow_path(slow, start, pc, |s| {
            let v = s.slot_load(dst);
            s.b.ins().jump(join, &[BlockArg::Value(v)]);
        });
        self.b.switch_to_block(join);
        self.f.vals.insert(dst, Virt::ptr(x, false));
        self.next(pc);
    }

    /// Whether `q` passes for a value known to be `v`.
    fn known_pass(&self, v: Option<Virt>, q: Quick) -> bool {
        let Some(v) = v else { return false };
        let ty = match v.num {
            Unboxed::F64(_) => Some(self.l.float_type),
            Unboxed::I64(_) => Some(self.l.long_type),
            Unboxed::None => v.ty,
        };
        match (q, ty) {
            (Quick::Any, _) => true,
            (Quick::Is(o), _) => v.ptr.and_then(|p| self.const_of(p)) == Some(o),
            (Quick::Type(t), Some(u)) => t == u,
            (Quick::FloatOrInt, Some(u)) => u == self.l.float_type || u == self.l.long_type,
            _ => false,
        }
    }

    /// Emit `q` on `p`: continue when it passes, else at `slow`.
    fn quick(&mut self, p: Value, q: Quick, slow: Block) {
        match q {
            Quick::Any => {}
            Quick::Slow => {
                self.b.ins().jump(slow, &[]);
                self.cont();
            }
            Quick::Type(t) => {
                let ty = self.ob_type(p);
                self.guard_eq(ty, t as i64, slow);
            }
            Quick::Is(o) => self.guard_eq(p, o as i64, slow),
            Quick::Bool => {
                let t = self.b.ins().icmp_imm_s(IntCC::Equal, p, self.l.true_ as i64);
                let f = self.b.ins().icmp_imm_s(IntCC::Equal, p, self.l.false_ as i64);
                let c = self.b.ins().bor(t, f);
                self.guard(c, slow);
            }
            Quick::FloatOrInt => {
                let ty = self.ob_type(p);
                let f = self.b.ins().icmp_imm_s(IntCC::Equal, ty, self.l.float_type as i64);
                let i = self.b.ins().icmp_imm_s(IntCC::Equal, ty, self.l.long_type as i64);
                let c = self.b.ins().bor(f, i);
                self.guard(c, slow);
            }
        }
    }

    fn check(&mut self, pc: usize, reg: Reg) {
        let q = match self.f.u.specs[pc] {
            Spec::Check(q) => q,
            _ => Quick::Slow,
        };
        if self.repr(reg) != Repr::Obj {
            return self.step(pc);
        }
        if self.known_pass(self.f.vals.get(&reg).copied(), q) {
            return self.next(pc);
        }
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let p = self.obj(reg, pc, slow);
        if self.f.u.speculate && matches!(q, Quick::FloatOrInt) && self.l.numbers {
            // Speculatively an exact float (an int bails out), unboxed for
            // what follows.
            let t = self.ob_type(p);
            self.guard_eq(t, self.l.float_type as i64, slow);
            let x = self.b.ins().load(types::F64, MF, p, self.l.fval);
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            if let Some(v) = self.f.vals.get_mut(&reg) {
                v.num = Unboxed::F64(x);
                v.ty = Some(self.l.float_type);
            }
            return self.next(pc);
        }
        if let (true, Quick::Type(t)) = (self.f.u.speculate, q) {
            // Speculatively of that type (else bail out): known after.
            self.quick(p, q, slow);
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            if let Some(v) = self.f.vals.get_mut(&reg) {
                v.ty = Some(t);
            }
            return self.next(pc);
        }
        self.quick(p, q, slow);
        let join = self.b.create_block();
        self.b.ins().jump(join, &[]);
        self.slow_path(slow, start, pc, |s| {
            s.b.ins().jump(join, &[]);
        });
        self.b.switch_to_block(join);
        self.next(pc);
    }

    /// An object operand as a double and whether it was a float (else a
    /// compact int); anything else continues at `slow`.
    fn as_f64(&mut self, p: Value, slow: Block) -> (Value, Value) {
        let t = self.ob_type(p);
        let (fl, notf, int, join) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
        let x = self.b.append_block_param(join, types::F64);
        let isf = self.b.append_block_param(join, types::I8);
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, t, self.l.float_type as i64);
        self.b.ins().brif(c, fl, &[], notf, &[]);
        self.b.switch_to_block(fl);
        let v = self.b.ins().load(types::F64, MF, p, self.l.fval);
        let one = self.b.ins().iconst(types::I8, 1);
        self.b.ins().jump(join, &[BlockArg::Value(v), BlockArg::Value(one)]);
        self.b.switch_to_block(notf);
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, t, self.l.long_type as i64);
        self.b.ins().brif(c, int, &[], slow, &[]);
        self.b.switch_to_block(int);
        let i = self.compact_value(p, slow);
        let v = self.b.ins().fcvt_from_sint(types::F64, i);
        let zero = self.b.ins().iconst(types::I8, 0);
        self.b.ins().jump(join, &[BlockArg::Value(v), BlockArg::Value(zero)]);
        self.b.switch_to_block(join);
        (x, isf)
    }

    /// The value of an int (its type checked) when compact, else at `slow`.
    fn compact_value(&mut self, p: Value, slow: Block) -> Value {
        let tag = self.b.ins().load(types::I64, MF, p, self.l.long_tag);
        let c = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, tag, 16);
        self.guard(c, slow);
        let d = self.b.ins().load(types::I32, MF, p, self.l.long_digit);
        let d = self.b.ins().uextend(types::I64, d);
        // (1 - (tag & 3)) * digit
        let s = self.b.ins().band_imm_u(tag, 3);
        let one = self.b.ins().iconst(types::I64, 1);
        let s = self.b.ins().isub(one, s);
        self.b.ins().imul(s, d)
    }

    /// The value of an exact int that fits in an `i64`, else at `slow`.
    fn as_i64(&mut self, p: Value, slow: Block) -> Value {
        let t = self.ob_type(p);
        self.guard_eq(t, self.l.long_type as i64, slow);
        let tag = self.b.ins().load(types::I64, MF, p, self.l.long_tag);
        let (compact, big, join) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let x = self.b.append_block_param(join, types::I64);
        let c = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, tag, 16);
        self.b.ins().brif(c, compact, &[], big, &[]);
        self.b.switch_to_block(compact);
        let v = self.compact_value(p, slow);
        self.b.ins().jump(join, &[BlockArg::Value(v)]);
        self.b.switch_to_block(big);
        let ss = self.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
        let out = self.b.ins().stack_addr(self.ptr, ss, 0);
        let ok = self.call(self.h.long_value, &[p, out], Some(types::I32)).unwrap();
        self.guard(ok, slow);
        let v = self.b.ins().stack_load(self.ptr, types::I64, ss, 0);
        self.b.ins().jump(join, &[BlockArg::Value(v)]);
        self.b.switch_to_block(join);
        x
    }

    /// `a op b` for float operands (statically): a float result, kept
    /// unboxed. Two ints, or anything else, take the generic path; if its
    /// result is not a float, the interpreter continues.
    fn float_binop(&mut self, pc: usize, dst: Reg, op: BinOp, a: Reg, b: Reg) {
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let f = |s: &mut Self, r: Reg| -> (Value, Value) {
            match s.operand(r, pc, slow) {
                Operand::F64(x) => (x, s.b.ins().iconst(types::I8, 1)),
                Operand::I64(x) => (s.b.ins().fcvt_from_sint(types::F64, x), s.b.ins().iconst(types::I8, 0)),
                Operand::Obj(p) => s.as_f64(p, slow),
            }
        };
        let (x, fx) = f(self, a);
        let (y, fy) = f(self, b);
        let either = self.b.ins().bor(fx, fy);
        self.guard(either, slow);
        let v = match op {
            BinOp::Add => self.b.ins().fadd(x, y),
            BinOp::Sub => self.b.ins().fsub(x, y),
            BinOp::Mul => self.b.ins().fmul(x, y),
            _ => {
                let zero = self.b.ins().f64const(0.0);
                let nz = self.b.ins().fcmp(FloatCC::NotEqual, y, zero);
                self.guard(nz, slow);
                self.b.ins().fdiv(x, y)
            }
        };
        if self.f.u.speculate {
            // Anything else (two ints, another type, x / 0): the interpreter
            // goes on.
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            self.f.vals.insert(dst, Virt { ptr: None, in_slot: false, num: Unboxed::F64(v), ty: Some(self.l.float_type), konst: false, checked: false, global: false });
            return self.next(pc);
        }
        let join = self.b.create_block();
        let r = self.b.append_block_param(join, types::F64);
        self.b.ins().jump(join, &[BlockArg::Value(v)]);
        self.slow_path(slow, start, pc, |s| {
            // A float result joins; anything else: the interpreter goes on.
            let p = s.slot_load(dst);
            let t = s.ob_type(p);
            let (fl, other) = (s.b.create_block(), s.b.create_block());
            let c = s.b.ins().icmp_imm_s(IntCC::Equal, t, s.l.float_type as i64);
            s.b.ins().brif(c, fl, &[], other, &[]);
            s.b.switch_to_block(fl);
            let v = s.b.ins().load(types::F64, MF, p, s.l.fval);
            s.b.ins().jump(join, &[BlockArg::Value(v)]);
            s.b.switch_to_block(other);
            s.bail_to(pc + 1);
        });
        self.b.switch_to_block(join);
        self.f.vals.insert(dst, Virt { ptr: None, in_slot: false, num: Unboxed::F64(r), ty: Some(self.l.float_type), konst: false, checked: false, global: false });
        self.next(pc);
    }

    /// `a op b` for int operands (statically): an int result in an `i64`,
    /// kept unboxed; overflow and other cases take the generic path, and the
    /// interpreter continues if its result does not fit.
    /// Speculatively, object operands of int operations are ints: unbox
    /// `regs` that are objects (else bail out at `pc`) and keep the values
    /// on the registers for later uses.
    fn speculate_ints(&mut self, pc: usize, regs: &[Reg]) {
        if !self.f.u.speculate {
            return;
        }
        let todo: Vec<Reg> = regs.iter().copied().filter(|r| !matches!(self.f.vals.get(r).map(|v| v.num), Some(Unboxed::I64(_) | Unboxed::F64(_)))).collect();
        if todo.is_empty() {
            return;
        }
        let start = self.f.vals.clone();
        let bail = self.b.create_block();
        for r in todo {
            let p = self.obj(r, pc, bail);
            let x = self.as_i64(p, bail);
            if let Some(v) = self.f.vals.get_mut(&r) {
                v.num = Unboxed::I64(x);
                v.ty = Some(self.l.long_type);
            }
        }
        let cont = self.b.create_block();
        self.b.ins().jump(cont, &[]);
        self.bail_path(bail, start, pc);
        self.b.switch_to_block(cont);
    }

    fn int_binop(&mut self, pc: usize, dst: Reg, op: BinOp, a: Reg, b: Reg) {
        self.speculate_ints(pc, &[a, b]);
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let f = |s: &mut Self, r: Reg| -> Value {
            match s.operand(r, pc, slow) {
                Operand::I64(x) => x,
                Operand::F64(_) => {
                    s.b.ins().jump(slow, &[]);
                    s.cont();
                    s.b.ins().iconst(types::I64, 0)
                }
                Operand::Obj(p) => s.as_i64(p, slow),
            }
        };
        let x = f(self, a);
        let y = f(self, b);
        let v = match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul => {
                let (v, of) = match op {
                    BinOp::Add => self.b.ins().sadd_overflow(x, y),
                    BinOp::Sub => self.b.ins().ssub_overflow(x, y),
                    _ => self.b.ins().smul_overflow(x, y),
                };
                let ok = self.b.create_block();
                self.b.ins().brif(of, slow, &[], ok, &[]);
                self.b.switch_to_block(ok);
                v
            }
            BinOp::BitAnd => self.b.ins().band(x, y),
            BinOp::BitOr => self.b.ins().bor(x, y),
            BinOp::BitXor => self.b.ins().bxor(x, y),
            _ => {
                // Floor division and modulo with Python's signs; division by
                // zero and MIN // -1 are generic.
                let nz = self.b.ins().icmp_imm_s(IntCC::NotEqual, y, 0);
                self.guard(nz, slow);
                let m1 = self.b.ins().icmp_imm_s(IntCC::Equal, y, -1);
                let min = self.b.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
                let bad = self.b.ins().band(m1, min);
                let ok = self.b.create_block();
                self.b.ins().brif(bad, slow, &[], ok, &[]);
                self.b.switch_to_block(ok);
                let q = self.b.ins().sdiv(x, y);
                let r = self.b.ins().srem(x, y);
                // r != 0 and sign(r) != sign(y): q - 1, r + y.
                let rnz = self.b.ins().icmp_imm_s(IntCC::NotEqual, r, 0);
                let xr = self.b.ins().bxor(r, y);
                let neg = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, xr, 0);
                let adj = self.b.ins().band(rnz, neg);
                let adj = self.b.ins().uextend(types::I64, adj);
                if op == BinOp::FloorDiv {
                    self.b.ins().isub(q, adj)
                } else {
                    let ya = self.b.ins().imul(y, adj);
                    self.b.ins().iadd(r, ya)
                }
            }
        };
        if self.f.u.speculate {
            // Overflow, another type, x // 0: the interpreter goes on.
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            self.f.vals.insert(dst, Virt { ptr: None, in_slot: false, num: Unboxed::I64(v), ty: Some(self.l.long_type), konst: false, checked: false, global: false });
            return self.next(pc);
        }
        let join = self.b.create_block();
        let r = self.b.append_block_param(join, types::I64);
        self.b.ins().jump(join, &[BlockArg::Value(v)]);
        self.slow_path(slow, start, pc, |s| {
            let p = s.slot_load(dst);
            let other = s.b.create_block();
            let v = s.as_i64(p, other);
            s.b.ins().jump(join, &[BlockArg::Value(v)]);
            s.b.switch_to_block(other);
            s.bail_to(pc + 1);
        });
        self.b.switch_to_block(join);
        self.f.vals.insert(dst, Virt { ptr: None, in_slot: false, num: Unboxed::I64(r), ty: Some(self.l.long_type), konst: false, checked: false, global: false });
        self.next(pc);
    }

    /// `dst: Bool = a op b`: inline for two ints or two floats, else
    /// generic.
    fn compare_bool(&mut self, pc: usize, dst: Reg, op: CmpOp, a: Reg, b: Reg) {
        // An object compared with an int is speculatively an int.
        let int = |s: &Self, r: Reg| matches!(s.f.vals.get(&r).map(|v| v.num), Some(Unboxed::I64(_)));
        if int(self, a) && !int(self, b) {
            self.speculate_ints(pc, &[b]);
        } else if int(self, b) && !int(self, a) {
            self.speculate_ints(pc, &[a]);
        }
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let x = self.operand(a, pc, slow);
        let y = self.operand(b, pc, slow);
        let join = self.b.create_block();
        let res = self.b.append_block_param(join, types::I8);
        let icmp = |s: &mut Self, x: Value, y: Value| {
            let c = s.b.ins().icmp(int_cc(op), x, y);
            s.b.ins().jump(join, &[BlockArg::Value(c)]);
        };
        let fcmp = |s: &mut Self, x: Value, y: Value| {
            let c = s.b.ins().fcmp(float_cc(op), x, y);
            s.b.ins().jump(join, &[BlockArg::Value(c)]);
        };
        match (x, y) {
            (Operand::I64(x), Operand::I64(y)) => icmp(self, x, y),
            (Operand::F64(x), Operand::F64(y)) => fcmp(self, x, y),
            (Operand::I64(x), Operand::Obj(p)) => {
                let y = self.as_i64(p, slow);
                icmp(self, x, y)
            }
            (Operand::Obj(p), Operand::I64(y)) => {
                let x = self.as_i64(p, slow);
                icmp(self, x, y)
            }
            (Operand::F64(x), Operand::Obj(p)) => {
                let t = self.ob_type(p);
                self.guard_eq(t, self.l.float_type as i64, slow);
                let y = self.b.ins().load(types::F64, MF, p, self.l.fval);
                fcmp(self, x, y)
            }
            (Operand::Obj(p), Operand::F64(y)) => {
                let t = self.ob_type(p);
                self.guard_eq(t, self.l.float_type as i64, slow);
                let x = self.b.ins().load(types::F64, MF, p, self.l.fval);
                fcmp(self, x, y)
            }
            (Operand::Obj(p), Operand::Obj(q)) => {
                let (tp, tq) = (self.ob_type(p), self.ob_type(q));
                let same = self.b.ins().icmp(IntCC::Equal, tp, tq);
                self.guard(same, slow);
                let (ints, notint, floats) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                let c = self.b.ins().icmp_imm_s(IntCC::Equal, tp, self.l.long_type as i64);
                self.b.ins().brif(c, ints, &[], notint, &[]);
                self.b.switch_to_block(ints);
                let x = self.compact_value(p, slow);
                let y = self.compact_value(q, slow);
                icmp(self, x, y);
                self.b.switch_to_block(notint);
                let c = self.b.ins().icmp_imm_s(IntCC::Equal, tp, self.l.float_type as i64);
                self.b.ins().brif(c, floats, &[], slow, &[]);
                self.b.switch_to_block(floats);
                let x = self.b.ins().load(types::F64, MF, p, self.l.fval);
                let y = self.b.ins().load(types::F64, MF, q, self.l.fval);
                fcmp(self, x, y);
            }
            _ => {
                self.b.ins().jump(slow, &[]);
            }
        }
        self.slow_path(slow, start, pc, |s| {
            let v = s.slot_load(dst);
            let v = s.b.ins().ireduce(types::I8, v);
            s.b.ins().jump(join, &[BlockArg::Value(v)]);
        });
        self.b.switch_to_block(join);
        let r = self.b.ins().uextend(types::I64, res);
        self.set_var(dst, r);
        self.next(pc);
    }

    /// `ForIter` over a list's iterator: the next item, inline; the end and
    /// other iterators take the generic path.
    fn for_iter_list(&mut self, pc: usize, dst: Reg, iter: Reg, target: u32) {
        let li = self.l.list_iter.unwrap();
        let start = self.f.vals.clone();
        let (slow, end) = (self.b.create_block(), self.b.create_block());
        let it = self.obj(iter, pc, slow);
        let t = self.ob_type(it);
        self.guard_eq(t, li.tp as i64, slow);
        let seq = self.b.ins().load(self.ptr, MF, it, li.seq);
        let (more, exhausted) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(seq, more, &[], end, &[]);
        self.b.switch_to_block(more);
        let idx = self.b.ins().load(types::I64, MF, it, li.index);
        let size = self.b.ins().load(types::I64, MF, seq, li.size);
        // (An index of -1 is past any size.)
        let has = self.b.ins().icmp(IntCC::UnsignedLessThan, idx, size);
        let next = self.b.create_block();
        self.b.ins().brif(has, next, &[], exhausted, &[]);
        // The end, as the iterator marks it: no index, or no list.
        self.b.set_cold_block(exhausted);
        self.b.switch_to_block(exhausted);
        if li.end_index {
            let m = self.b.ins().iconst(types::I64, -1);
            self.b.ins().store(MF, m, it, li.index);
        } else {
            let z = self.b.ins().iconst(self.ptr, 0);
            self.b.ins().store(MF, z, it, li.seq);
            self.decref(seq);
        }
        self.b.ins().jump(end, &[]);
        self.b.switch_to_block(end);
        self.goto(pc, target);
        self.f.vals = start.clone();
        self.b.switch_to_block(next);
        let items = self.b.ins().load(self.ptr, MF, seq, li.items);
        let at = self.b.ins().ishl_imm_u(idx, 3);
        let at = self.b.ins().iadd(items, at);
        let item = self.b.ins().load(self.ptr, MF, at, 0);
        let n = self.b.ins().iadd_imm_s(idx, 1);
        self.b.ins().store(MF, n, it, li.index);
        // Storing releases `dst`'s old reference.
        self.secure_borrowed(pc + 1, pc, &[iter]);
        self.incref(item);
        self.put(dst, item);
        self.f.vals.insert(dst, Virt::ptr(item, true));
        if self.f.u.speculate {
            // Other iterators: the interpreter. The item continues here,
            // with what is known.
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            self.f.done = false;
            return;
        }
        self.goto(pc, pc as u32 + 1);
        self.f.vals = start;
        self.b.set_cold_block(slow);
        self.b.switch_to_block(slow);
        let was = std::mem::replace(&mut self.slow, true);
        self.generic(pc);
        self.slow = was;
        self.f.done = true;
    }

    /// `yield src`: the frame (the generator's) keeps what the code after
    /// the `yield` reads, and the value goes to the runtime (`YIELD`); the
    /// generator resumes at the next instruction (a label). Unbound: `step`
    /// raises.
    fn yield_value(&mut self, pc: usize, src: Reg) {
        let start = self.f.vals.clone();
        // An unboxed int, to a caller that takes one (`CTX_WANT_INT`): as
        // it is, boxed only if the generator itself needs it after.
        if let Some(Unboxed::I64(x)) = self.f.vals.get(&src).map(|v| v.num) {
            let (int, obj) = (self.b.create_block(), self.b.create_block());
            let want = self.b.ins().load(self.ptr, MF, self.ctx, CTX_WANT_INT);
            self.b.ins().brif(want, int, &[], obj, &[]);
            self.b.switch_to_block(int);
            self.flush(pc + 1);
            self.b.ins().store(MF, x, self.ctx, CTX_RET);
            let pcv = self.pc_val(pc);
            self.b.ins().jump(self.yielded_int, &[BlockArg::Value(pcv)]);
            self.f.vals = start.clone();
            self.b.switch_to_block(obj);
        }
        let slow = self.b.create_block();
        let p = self.obj(src, pc, slow);
        self.incref(p);
        self.flush(pc);
        self.b.ins().store(MF, p, self.ctx, CTX_RET);
        let pcv = self.pc_val(pc);
        self.b.ins().jump(self.yielded, &[BlockArg::Value(pcv)]);
        self.f.vals = start;
        self.b.set_cold_block(slow);
        self.b.switch_to_block(slow);
        let was = std::mem::replace(&mut self.slow, true);
        // (Resumed, it continues at the next instruction from here.)
        self.step(pc);
        self.slow = was;
    }

    /// A self tail call's test: the function object that runs this code
    /// (`f`) loops, another one is called.
    fn jump_if_not_self(&mut self, pc: usize, func: Reg, depth: Reg, target: u32, f: usize) {
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let p = self.obj(func, pc, slow);
        let (yes, no) = (self.b.create_block(), self.b.create_block());
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, p, f as i64);
        self.b.ins().brif(c, yes, &[], no, &[]);
        let after = self.f.vals.clone();
        self.b.switch_to_block(no);
        self.goto(pc, target);
        self.f.vals = after;
        self.b.switch_to_block(yes);
        // Counted, as CPython's frames would be.
        let var = self.f.vars[depth as usize].unwrap();
        let d = self.b.use_var(var);
        let d = self.b.ins().iadd_imm_s(d, 1);
        let st = self.call(self.h.tail_depth, &[d], Some(types::I32)).unwrap();
        self.b.def_var(var, d);
        self.check_status(st, pc);
        self.goto(pc, pc as u32 + 1);
        // Unbound: `step` raises.
        self.f.vals = start;
        self.b.set_cold_block(slow);
        self.b.switch_to_block(slow);
        let was = std::mem::replace(&mut self.slow, true);
        self.step(pc);
        self.slow = was;
        self.f.done = true;
    }

    /// `ForIter` over another iterator (a generator, ...): the helper takes
    /// the next item (or the end) without the registers in their slots,
    /// but where a handler would see them; the loop's unboxed values stay
    /// in registers.
    fn for_iter_call(&mut self, pc: usize, dst: Reg, iter: Reg, target: u32) {
        let covered = self.f.ir.handlers.iter().any(|h| h.ranges.iter().any(|(s, e)| (*s as usize) <= pc && pc < *e as usize));
        if covered {
            self.flush(pc);
        } else {
            // The iterator runs Python code: what borrowed pointers point
            // into may go.
            self.secure_borrowed(pc + 1, pc, &[iter]);
        }
        if self.f.vals.get(&iter).is_some_and(|v| !v.in_slot) {
            self.materialize(iter, pc);
        }
        self.clobber();
        // Over a generator, an item the loop keeps unboxed may come as a
        // machine int.
        let ints = matches!(self.f.u.specs[pc], Spec::Iter(ITER_GEN) | Spec::IterGen { .. }) && self.f.carry[pc + 1].iter().any(|&(r, k)| r == dst && k == Kind::I);
        let out = ints.then(|| self.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3)));
        if let Spec::IterGen { hot, tp } = self.f.u.specs[pc] {
            let helper = self.b.create_block();
            self.resume_gen(pc, dst, iter, target, hot, tp, out, helper);
            self.b.switch_to_block(helper);
        }
        let (d, it, p) = (self.i32c(dst), self.i32c(iter), self.pc_val(pc));
        let (ctx, regs) = (self.ctx, self.f.regs);
        let st = match out {
            Some(out) => {
                let op = self.b.ins().stack_addr(self.ptr, out, 0);
                self.call(self.h.for_iter_int, &[ctx, regs, d, it, p, op], Some(types::I32)).unwrap()
            }
            None => self.call(self.h.for_iter, &[ctx, regs, d, it, p], Some(types::I32)).unwrap(),
        };
        self.for_iter_result(pc, dst, target, st, out)
    }

    /// After `ForIter`'s helper (status `st`): an int in `out` (`NEXT_INT`),
    /// or `for_iter_status`.
    fn for_iter_result(&mut self, pc: usize, dst: Reg, target: u32, st: Value, out: Option<StackSlot>) {
        if let Some(out) = out {
            let (int, rest) = (self.b.create_block(), self.b.create_block());
            let is_int = self.b.ins().icmp_imm_u(IntCC::Equal, st, NEXT_INT as i64);
            self.b.ins().brif(is_int, int, &[], rest, &[]);
            self.b.switch_to_block(int);
            let x = self.b.ins().stack_load(self.ptr, types::I64, out, 0);
            self.next_int(pc, dst, x);
            self.b.switch_to_block(rest);
        }
        self.for_iter_status(pc, target, st)
    }

    /// The loop goes on with `dst` the int `x`.
    fn next_int(&mut self, pc: usize, dst: Reg, x: Value) {
        let saved = self.f.vals.clone();
        self.wrote(pc);
        self.f.vals.insert(dst, Virt { ptr: None, in_slot: false, num: Unboxed::I64(x), ty: Some(self.l.long_type), konst: false, checked: false, global: false });
        self.goto(pc, pc as u32 + 1);
        self.f.vals = saved;
    }

    /// `ForIter` over a Lizzard generator (its state at `hot`): if it is
    /// plainly suspended (no `yield from`, no handled exception, its code
    /// compiled, the sent value unused), compiled code takes `running` and
    /// enters its code itself, with a context of its own; a `yield` comes
    /// back here. Anything else continues at `helper` (`for_iter`).
    #[allow(clippy::too_many_arguments)]
    fn resume_gen(&mut self, pc: usize, dst: Reg, iter: Reg, target: u32, hot: i32, tp: usize, out: Option<StackSlot>, helper: Block) {
        let saved = self.f.vals.clone();
        let it = self.slot_load(iter);
        self.guard(it, helper);
        let t = self.ob_type(it);
        self.guard_eq(t, tp as i64, helper);
        let h = self.b.ins().iadd_imm_s(it, hot as i64);
        let plain = self.b.ins().load(types::I8, MF, h, GEN_PLAIN);
        self.guard(plain, helper);
        let status = self.b.ins().load(types::I8, MF, h, GEN_STATUS);
        self.guard_eq(status, GEN_SUSPENDED as i64, helper);
        let sent = self.b.ins().load(types::I32, MF, h, GEN_DST);
        self.guard_eq(sent, NO_SEND as i32 as i64, helper);
        let slots = self.b.ins().load(self.ptr, MF, h, GEN_SLOTS);
        self.guard(slots, helper);
        let cell = self.b.ins().load(self.ptr, MF, h, GEN_ENTRY);
        let entry = self.b.ins().load(self.ptr, MF, cell, 0);
        self.guard(entry, helper);
        // Taking `running`: with plain stores when no other thread can
        // reach the generator (the GIL; or, free-threaded, its only
        // reference is this thread's), as CPython's specialized loops do;
        // else atomically.
        let (zero8, one8) = (self.b.ins().iconst(types::I8, 0), self.b.ins().iconst(types::I8, 1));
        let owned = self.b.create_block();
        let alone = self.b.append_block_param(owned, types::I8);
        let unique = match self.l.rc {
            Rc::Gil => Some(self.b.ins().iconst(types::I8, 1)),
            Rc::Ft { tid, local, shared } => {
                let t = self.b.ins().load(self.ptr, MF, it, tid);
                let c1 = self.b.ins().icmp(IntCC::Equal, t, self.tid);
                let l = self.b.ins().load(types::I32, MF, it, local);
                let c2 = self.b.ins().icmp_imm_s(IntCC::Equal, l, 1);
                let sh = self.b.ins().load(self.ptr, MF, it, shared);
                let c3 = self.b.ins().icmp_imm_s(IntCC::Equal, sh, 0);
                let c = self.b.ins().band(c1, c2);
                Some(self.b.ins().band(c, c3))
            }
            Rc::Calls => None,
        };
        let atomic = self.b.create_block();
        match unique {
            Some(u) => {
                let plain = self.b.create_block();
                self.b.ins().brif(u, plain, &[], atomic, &[]);
                self.b.switch_to_block(plain);
                let r = self.b.ins().load(types::I8, MF, h, GEN_RUNNING);
                self.guard_eq(r, 0, helper);
                self.b.ins().store(MF, one8, h, GEN_RUNNING);
                self.b.ins().jump(owned, &[BlockArg::Value(one8)]);
            }
            None => {
                self.b.ins().jump(atomic, &[]);
            }
        }
        self.b.switch_to_block(atomic);
        let old = self.b.ins().atomic_cas(MemFlagsData::trusted(), h, zero8, one8);
        self.guard_eq(old, 0, helper);
        self.b.ins().jump(owned, &[BlockArg::Value(zero8)]);
        self.b.switch_to_block(owned);
        // Running: its context, entered after its `yield`.
        let ctx = self.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, CTX_SIZE as u32 + 8, 3));
        let z = self.b.ins().iconst(self.ptr, 0);
        self.b.ins().stack_store(self.ptr, self.tid, ctx, CTX_TID);
        self.b.ins().stack_store(self.ptr, z, ctx, CTX_RET);
        self.b.ins().stack_store(self.ptr, slots, ctx, CTX_SLOTS);
        let tgt = self.b.ins().load(self.ptr, MF, h, GEN_TARGET);
        self.b.ins().stack_store(self.ptr, tgt, ctx, CTX_TARGET);
        self.b.ins().stack_store(self.ptr, z, ctx, CTX_EXIT);
        let want = self.b.ins().iconst(self.ptr, out.is_some() as i64);
        self.b.ins().stack_store(self.ptr, want, ctx, CTX_WANT_INT);
        let at = self.b.ins().load(types::I32, MF, h, GEN_PC);
        let at = self.b.ins().iadd_imm_s(at, 1);
        self.b.ins().stack_store(self.ptr, at, ctx, CTX_SIZE);
        let cp = self.b.ins().stack_addr(self.ptr, ctx, 0);
        let pcp = self.b.ins().stack_addr(self.ptr, ctx, CTX_SIZE);
        let z32 = self.b.ins().iconst(types::I32, 0);
        let mut sig = Signature::new(self.call_conv);
        sig.params.extend([AbiParam::new(self.ptr), AbiParam::new(self.ptr), AbiParam::new(self.ptr), AbiParam::new(types::I32)]);
        sig.returns.push(AbiParam::new(types::I32));
        let sig = self.b.import_signature(sig);
        let call = self.b.ins().call_indirect(sig, entry, &[cp, slots, pcp, z32]);
        let st = self.b.inst_results(call)[0];
        let np = self.b.ins().stack_load(self.ptr, types::I32, ctx, CTX_SIZE);
        let (value, rest, int, other) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
        let c = self.b.ins().icmp_imm_u(IntCC::Equal, st, YIELD as i64);
        self.b.ins().brif(c, value, &[], rest, &[]);
        self.b.switch_to_block(rest);
        let c = self.b.ins().icmp_imm_u(IntCC::Equal, st, YIELD_INT as i64);
        self.b.ins().brif(c, int, &[], other, &[]);
        // Suspended at its next `yield`: where, and where a value sent there
        // goes; then it is not running.
        let suspended = |s: &mut Self| {
            s.b.ins().store(MF, np, h, GEN_PC);
            let table = s.b.ins().load(s.ptr, MF, h, GEN_YIELD_DST);
            let i = s.b.ins().uextend(types::I64, np);
            let i = s.b.ins().ishl_imm_u(i, 2);
            let a = s.b.ins().iadd(table, i);
            let d = s.b.ins().load(types::I32, MF, a, 0);
            s.b.ins().store(MF, d, h, GEN_DST);
            let zero8 = s.b.ins().iconst(types::I8, 0);
            let (plain, atomic, done) = (s.b.create_block(), s.b.create_block(), s.b.create_block());
            s.b.ins().brif(alone, plain, &[], atomic, &[]);
            s.b.switch_to_block(plain);
            s.b.ins().store(MF, zero8, h, GEN_RUNNING);
            s.b.ins().jump(done, &[]);
            s.b.switch_to_block(atomic);
            s.b.ins().atomic_store(MemFlagsData::trusted(), zero8, h);
            s.b.ins().jump(done, &[]);
            s.b.switch_to_block(done);
        };
        self.b.switch_to_block(value);
        suspended(self);
        let v = self.b.ins().stack_load(self.ptr, self.ptr, ctx, CTX_RET);
        self.put(dst, v);
        self.wrote(pc);
        self.f.vals.insert(dst, Virt::ptr(v, true));
        self.goto(pc, pc as u32 + 1);
        self.f.vals = saved.clone();
        self.b.switch_to_block(int);
        suspended(self);
        let x = self.b.ins().stack_load(self.ptr, types::I64, ctx, CTX_RET);
        self.next_int(pc, dst, x);
        // Returned, raised, bailed out: the runtime finishes it.
        self.b.set_cold_block(other);
        self.b.switch_to_block(other);
        let op = match out {
            Some(out) => self.b.ins().stack_addr(self.ptr, out, 0),
            None => self.b.ins().iconst(self.ptr, 0),
        };
        let d = self.i32c(dst);
        let regs = self.f.regs;
        let st = self.call(self.h.gen_resumed, &[it, cp, st, np, regs, d, op], Some(types::I32)).unwrap();
        self.for_iter_result(pc, dst, target, st, out);
        self.f.vals = saved;
        self.f.done = false;
    }

    /// After `ForIter`'s helper: its status `st`.
    fn for_iter_status(&mut self, pc: usize, target: u32, st: Value) {
        self.wrote(pc);
        // NEXT continues, TAKEN jumps, ERROR raises (an unbound iterator
        // too).
        let (next, other, taken, raised) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.b.ins().brif(st, other, &[], next, &[]);
        self.b.switch_to_block(other);
        let is_taken = self.b.ins().icmp_imm_u(IntCC::Equal, st, TAKEN as i64);
        self.b.ins().brif(is_taken, taken, &[], raised, &[]);
        self.b.set_cold_block(raised);
        self.b.switch_to_block(raised);
        let saved = self.f.vals.clone();
        self.raise(pc);
        self.f.vals = saved.clone();
        self.b.switch_to_block(taken);
        self.goto(pc, target);
        self.f.vals = saved;
        self.b.switch_to_block(next);
        self.goto(pc, pc as u32 + 1);
        self.f.done = true;
    }

    /// `dst: Bool = a is b` (or `is not`).
    fn compare_is(&mut self, pc: usize, dst: Reg, op: CmpOp, a: Reg, b: Reg) {
        let has = |s: &Self, r: Reg| s.f.vals.get(&r).is_none_or(|v| v.ptr.is_some());
        if !has(self, a) || !has(self, b) {
            return self.step(pc);
        }
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let x = self.obj(a, pc, slow);
        let y = self.obj(b, pc, slow);
        let cc = if op == CmpOp::Is { IntCC::Equal } else { IntCC::NotEqual };
        let c = self.b.ins().icmp(cc, x, y);
        let c = self.b.ins().uextend(types::I64, c);
        self.set_var(dst, c);
        let join = self.b.create_block();
        self.b.ins().jump(join, &[]);
        // Unbound: `step` raises.
        self.slow_path(slow, start, pc, |s| {
            s.b.ins().jump(join, &[]);
        });
        self.b.switch_to_block(join);
        self.next(pc);
    }

    fn truthy(&mut self, pc: usize, dst: Reg, src: Reg) {
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let join = self.b.create_block();
        let res = self.b.append_block_param(join, types::I64);
        let p = self.obj(src, pc, slow);
        let (not_true, not_false, is_none) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let one = self.b.ins().iconst(types::I64, 1);
        let zero = self.b.ins().iconst(types::I64, 0);
        let t = self.b.ins().icmp_imm_s(IntCC::Equal, p, self.l.true_ as i64);
        self.b.ins().brif(t, join, &[BlockArg::Value(one)], not_true, &[]);
        self.b.switch_to_block(not_true);
        let f = self.b.ins().icmp_imm_s(IntCC::Equal, p, self.l.false_ as i64);
        self.b.ins().brif(f, join, &[BlockArg::Value(zero)], not_false, &[]);
        self.b.switch_to_block(not_false);
        let n = self.b.ins().icmp_imm_s(IntCC::Equal, p, self.l.none as i64);
        self.b.ins().brif(n, is_none, &[], slow, &[]);
        self.b.switch_to_block(is_none);
        self.b.ins().jump(join, &[BlockArg::Value(zero)]);
        self.slow_path(slow, start, pc, |s| {
            let v = s.slot_load(dst);
            s.b.ins().jump(join, &[BlockArg::Value(v)]);
        });
        self.b.switch_to_block(join);
        self.set_var(dst, res);
        self.next(pc);
    }

    /// `return src`: a new reference for the runtime.
    fn ret(&mut self, pc: usize, src: Reg) {
        // An inlined function returning an int: unboxed.
        if let (Some(inl), Some(Unboxed::I64(x))) = (&self.f.inline, self.f.vals.get(&src).map(|v| v.num)) {
            if inl.ret_int {
                let exit = inl.exit;
                let z = self.b.ins().iconst(self.ptr, 0);
                self.b.ins().jump(exit, &[BlockArg::Value(z), BlockArg::Value(x)]);
                self.f.done = true;
                return;
            }
        }
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let v = self.f.vals.get(&src).copied();
        let p = match v {
            Some(Virt { ptr: None, in_slot: false, num: Unboxed::F64(x), .. }) => {
                let p = self.call(self.h.float_new, &[x], Some(self.ptr)).unwrap();
                self.guard(p, slow);
                p
            }
            Some(Virt { ptr: None, in_slot: false, num: Unboxed::I64(x), .. }) => {
                let p = self.new_int(x);
                self.guard(p, slow);
                p
            }
            Some(Virt { ptr: Some(p), in_slot: true, .. }) if !self.is_param(src) => {
                // The slot's reference moves out.
                let z = self.b.ins().iconst(types::I64, 0);
                self.slot_store(z, src);
                p
            }
            _ => {
                let p = self.obj(src, pc, slow);
                self.incref(p);
                p
            }
        };
        let c = self.const_of(p).filter(|_| self.immortal(p));
        let z = self.b.ins().iconst(types::I64, 0);
        match &mut self.f.inline {
            Some(inl) => {
                inl.ret_const = Some(match inl.ret_const {
                    None => c,
                    Some(prev) if prev == c => c,
                    Some(_) => None,
                });
                let exit = inl.exit;
                self.b.ins().jump(exit, &[BlockArg::Value(p), BlockArg::Value(z)]);
            }
            None => {
                let pcv = self.pc_val(pc);
                self.b.ins().jump(self.ret_block, &[BlockArg::Value(p), BlockArg::Value(pcv)]);
            }
        }
        self.f.vals = start;
        self.b.switch_to_block(slow);
        self.step(pc);
        self.f.done = true;
    }

    /// A call of a Lizzard function compiled code enters directly, with the
    /// arguments borrowed from the caller's registers.
    fn call_direct(&mut self, pc: usize, dst: Reg, func: Option<Reg>, obj: Option<Reg>, args: &[Reg]) {
        let u: &'a Unit<'a> = self.f.u;
        let Spec::Call(spec) = &u.specs[pc] else { unreachable!() };
        let covered = self.f.ir.handlers.iter().any(|h| h.ranges.iter().any(|(s, e)| (*s as usize) <= pc && pc < *e as usize));
        // The arguments must stay alive during the call; the function
        // need not (the guard proves it is the one the call's description
        // keeps).
        let mut keep: Vec<Reg> = obj.into_iter().chain(args.iter().copied()).collect();
        let spare = func.filter(|&r| !self.live_at(pc + 1, r));
        keep.extend(func.filter(|_| spare.is_none()));
        if covered {
            self.flush(pc);
        } else {
            self.secure_borrowed_but(pc + 1, pc, &keep, spare);
        }
        let mut all: Vec<Reg> = obj.into_iter().collect();
        all.extend(args);
        // Compiled in here: numbers the code has unboxed stay so (boxed
        // only if the callee's code needs their objects).
        let inline = spec.inline.as_deref().filter(|c| !spec.native && (self.f.depth == 0 || (self.f.depth == 1 && c.ir.code.len() <= INLINE_MAX_NESTED)) && inlinable(c));
        let skip = matches!(spec.guard, CallGuard::Ctor { .. }) as usize;
        let lazy: Vec<bool> = all
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                let q = spec.checks.get(i + skip).copied().unwrap_or(Quick::Slow);
                let v = self.f.vals.get(&r).copied();
                inline.is_some() && v.is_some_and(|v| v.ptr.is_none() && !matches!(v.num, Unboxed::None)) && self.known_pass(v, q)
            })
            .collect();
        // The arguments must stay alive during the call: in their slots or
        // kept by the code.
        for (i, &r) in all.iter().enumerate() {
            if let Some(v) = self.f.vals.get(&r).copied() {
                if !v.in_slot && !v.konst && !lazy[i] {
                    self.materialize(r, pc);
                }
            }
        }
        let start = self.f.vals.clone();
        let slow = self.b.create_block();
        let mut ctor = None;
        let mut receiver = None;
        match (&spec.guard, func, obj) {
            (CallGuard::Func(f), Some(fr), _) => {
                let p = self.obj(fr, pc, slow);
                self.guard_eq(p, *f as i64, slow);
            }
            (CallGuard::Ctor { tp, version, alloc, .. }, Some(fr), _) => {
                let p = self.obj(fr, pc, slow);
                self.guard_eq(p, *tp as i64, slow);
                let v = self.b.ins().load(types::I32, MF, p, self.l.tp_version);
                let c = self.b.ins().icmp_imm_s(IntCC::Equal, v, *version as i32 as i64);
                self.guard(c, slow);
                ctor = Some((*tp, *alloc));
            }
            (CallGuard::Method { tp, version, keys, nentries, valid }, _, Some(o)) => {
                let p = self.obj(o, pc, slow);
                if !self.verified(o, *tp) {
                    self.guard_type_of(o, p, *tp, *version, slow);
                }
                // The callee may rely on the receiver's type.
                receiver = Some(*tp);
                if *keys != 0 {
                    let v = self.b.ins().load(types::I8, MF, p, *valid);
                    self.guard(v, slow);
                    let k = self.b.ins().iconst(self.ptr, *keys as i64);
                    let n = self.b.ins().load(types::I64, MF, k, 0);
                    self.guard_eq(n, *nentries, slow);
                }
            }
            _ => unreachable!("a call spec for another instruction"),
        }
        let mut ptrs = Vec::new();
        let mut lazy_ptrs = Vec::new();
        // A constructor's `__init__` has the instance first.
        for (i, &r) in all.iter().enumerate() {
            if lazy[i] {
                lazy_ptrs.push(None);
                continue;
            }
            let p = self.obj(r, pc, slow);
            lazy_ptrs.push(Some(p));
            let q = spec.checks.get(i + skip).copied().unwrap_or(Quick::Slow);
            if !self.known_pass(self.f.vals.get(&r).copied(), q) {
                self.quick(p, q, slow);
            }
            ptrs.push(p);
        }
        // An instance about to be released, to construct in instead.
        let reuse = match &spec.guard {
            CallGuard::Ctor { tp, reuse: Some(r), .. } if self.f.u.speculate && spec.inline.as_deref().is_some_and(inlinable) => self.donor(pc, *tp, &all).map(|d| (d, r)),
            _ => None,
        };
        let inst = ctor.map(|(tp, alloc)| {
            let done = self.b.create_block();
            let o = self.b.append_block_param(done, self.ptr);
            let fresh = self.b.create_block();
            if let Some((d, r)) = reuse {
                let p = self.f.vals[&d].ptr.unwrap();
                self.reusable(p, r, fresh);
                self.incref(p);
                self.b.ins().jump(done, &[BlockArg::Value(p)]);
            } else {
                self.b.ins().jump(fresh, &[]);
            }
            self.b.switch_to_block(fresh);
            let tp = self.b.ins().iconst(self.ptr, tp as i64);
            let zero = self.b.ins().iconst(types::I64, 0);
            let n = self.call(alloc, &[tp, zero], Some(self.ptr)).unwrap();
            let failed = self.b.create_block();
            self.b.ins().brif(n, done, &[BlockArg::Value(n)], failed, &[]);
            self.b.switch_to_block(failed);
            self.raise(pc);
            self.b.switch_to_block(done);
            o
        });
        if let Some(o) = inst {
            ptrs.insert(0, o);
            lazy_ptrs.insert(0, Some(o));
        }
        let r = match spec.inline.as_deref() {
            _ if spec.native => {
                let frame = self.frame.expect("a frame for direct calls");
                for (i, p) in ptrs.iter().enumerate() {
                    self.b.ins().stack_store(self.ptr, *p, frame, i as i32 * 8);
                }
                let fp = self.b.ins().stack_addr(self.ptr, frame, 0);
                let callable = self.b.ins().iconst(self.ptr, spec.desc as i64);
                let n = self.b.ins().iconst(types::I64, ptrs.len() as i64);
                let none = self.b.ins().iconst(self.ptr, 0);
                self.call(self.h.vectorcall, &[callable, fp, n, none], Some(self.ptr)).unwrap()
            }
            // Not within an inlined function: one level.
            _ if inline.is_some() => {
                let callee = inline.unwrap();
                // Its first parameter's type is verified: the receiver's,
                // or the class of the instance just allocated.
                let first = receiver.or(ctor.map(|(tp, _)| tp));
                let mut known: Vec<Option<Virt>> = all.iter().map(|r| self.f.vals.get(r).copied()).collect();
                // An argument whose register dies here, which owns it.
                let mut donors: Vec<bool> = all.iter().map(|&r| self.f.inline.is_none() && !self.live_at(pc + 1, r) && all.iter().filter(|&&x| x == r).count() == 1 && !self.is_param(r) && self.f.vals.get(&r).is_some_and(|v| v.in_slot)).collect();
                if ctor.is_some() {
                    known.insert(0, None);
                    donors.insert(0, false);
                }
                let (r, int) = self.inline_call(pc, &lazy_ptrs, &known, &donors, callee, spec.desc, first);
                if let Some(x) = int {
                    return self.int_result(pc, dst, r, x, slow, start);
                }
                r
            }
            _ => {
                let frame = self.frame.expect("a frame for direct calls");
                for (i, p) in ptrs.iter().enumerate() {
                    self.b.ins().stack_store(self.ptr, *p, frame, i as i32 * 8);
                }
                let z = self.b.ins().iconst(types::I64, 0);
                for i in ptrs.len()..spec.nregs as usize {
                    self.b.ins().stack_store(self.ptr, z, frame, i as i32 * 8);
                }
                let fp = self.b.ins().stack_addr(self.ptr, frame, 0);
                let desc = self.b.ins().iconst(self.ptr, spec.desc as i64);
                if spec.entry != 0 {
                    self.native_call(fp, desc, spec.entry, spec.target)
                } else {
                    self.call(self.h.enter, &[self.ctx, fp, desc], Some(self.ptr)).unwrap()
                }
            }
        };
        let (ok, failed) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(r, ok, &[], failed, &[]);
        self.b.switch_to_block(failed);
        if let Some(o) = inst {
            self.decref(o);
        }
        // Materialized above if a handler covers it.
        self.raise(pc);
        self.b.switch_to_block(ok);
        let r = match inst {
            Some(o) => {
                // `__init__` returns None (immortal), else TypeError.
                let (none, other) = (self.b.create_block(), self.b.create_block());
                let c = self.b.ins().icmp_imm_s(IntCC::Equal, r, self.l.none as i64);
                self.b.ins().brif(c, none, &[], other, &[]);
                self.b.switch_to_block(other);
                self.call(self.h.init_returned, &[r], None);
                self.decref(o);
                self.raise(pc);
                self.b.switch_to_block(none);
                o
            }
            None => r,
        };
        // The callee ran Python code.
        self.clobber();
        if self.f.u.speculate {
            // A failed guard bails out: nothing joins, and an immortal
            // constant result (None) needs no slot.
            if self.immortal(r) {
                self.f.vals.insert(dst, Virt { ptr: Some(r), in_slot: false, num: Unboxed::None, ty: None, konst: true, checked: false, global: false });
            } else {
                self.put(dst, r);
                self.f.vals.insert(dst, Virt::ptr(r, true));
            }
            let cont = self.b.create_block();
            self.b.ins().jump(cont, &[]);
            self.bail_path(slow, start, pc);
            self.b.switch_to_block(cont);
            return self.next(pc);
        }
        self.put(dst, r);
        let join = self.b.create_block();
        self.b.ins().jump(join, &[]);
        self.slow_path(slow, start, pc, |s| {
            s.b.ins().jump(join, &[]);
        });
        self.b.switch_to_block(join);
        self.next(pc);
    }

    /// Compile `callee` here, its parameters `args` (borrowed), its frame
    /// in a stack slot: its result (a new reference), at the end of its
    /// code. An exception leaving it raises at the call (`pc`).
    #[allow(clippy::too_many_arguments)]
    fn inline_call(&mut self, pc: usize, args: &[Option<Value>], known: &[Option<Virt>], donors: &[bool], callee: &'a Unit<'a>, desc: usize, first: Option<usize>) -> (Value, Option<Value>) {
        let ir = callee.ir;
        let nregs = ir.reprs.len();
        // The frame: initialized where it is used (`init` below,
        // `prepare_frame`).
        let frame = self.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, nregs.max(1) as u32 * 8, 3));
        let z = self.b.ins().iconst(types::I64, 0);
        let regs = self.b.ins().stack_addr(self.ptr, frame, 0);
        let vars: Vec<Option<Variable>> = ir
            .reprs
            .iter()
            .map(|r| {
                matches!(r, Repr::I64 | Repr::Bool).then(|| {
                    let v = self.b.declare_var(types::I64);
                    self.b.def_var(v, z);
                    v
                })
            })
            .collect();
        let is_label = label_pcs(ir, |pc| callee.speculate && over_lists(callee, self.l, pc));
        let live = liveness(ir);
        // What the caller knows of the arguments: kinds for the analyses.
        let pk: Vec<Kind> = known
            .iter()
            .map(|k| match k.map(|v| v.num) {
                Some(Unboxed::I64(_)) => Kind::I,
                Some(Unboxed::F64(_)) => Kind::F,
                _ => Kind::Obj,
            })
            .collect();
        let carry = carried(callee, self.l.numbers, &is_label, &live, &pk);
        // Every return returns an int compiled code made: the result goes
        // back unboxed.
        let at = kinds(callee, &pk);
        let rets: Vec<usize> = ir.code.iter().enumerate().filter(|(_, i)| matches!(i, Instr::Return { .. })).map(|(pc, _)| pc).collect();
        let ret_int = self.f.u.speculate
            && self.l.numbers
            && rets.iter().any(|&pc| at[pc].iter().any(|k| *k != Kind::Top))
            && rets.iter().all(|&pc| match &ir.code[pc] {
                Instr::Return { src } => matches!(at[pc][*src as usize], Kind::I | Kind::Top),
                _ => false,
            });
        let labels = label_blocks(&mut self.b, &is_label, &carry);
        let exit = self.b.create_block();
        let result = self.b.append_block_param(exit, self.ptr);
        let result_int = self.b.append_block_param(exit, types::I64);
        let fail = self.b.create_block();
        let fail_pc = self.b.append_block_param(fail, types::I32);
        let raised = self.b.create_block();
        // The parameters' slots hold the arguments, and what the caller
        // knows about them holds (the first one's type may be verified).
        let seed: Vec<(Reg, Virt)> = args
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let k = known.get(i).copied().flatten();
                // (A lazy parameter: its value, and no object yet.)
                let mut v = Virt { ptr: p, in_slot: p.is_some(), num: Unboxed::None, ty: None, konst: false, checked: false, global: false };
                if let Some(k) = k {
                    v = Virt { num: k.num, ty: k.ty, konst: k.konst, checked: k.checked, ..v };
                }
                if let Some(tp) = first.filter(|_| i == 0) {
                    v.ty = Some(tp);
                    v.checked = true;
                }
                (i as Reg, v)
            })
            .collect();
        let inlined = Inlined { frame, desc, call_pc: pc, exit, fail, raised, donors: donors.to_vec(), args: args.to_vec(), ret_const: None, ret_int };
        let callee_fn = Fn { u: callee, ir, labels, catch_blocks: Vec::new(), regs, vars, vals: HashMap::new(), live, carry, nparams: ir.params.len() as u32, done: false, inline: Some(inlined), seed, written: Default::default(), cold_written: Default::default(), depth: self.f.depth + 1, cold: false };
        let caller = std::mem::replace(&mut self.f, callee_fn);
        let first = self.f.labels[0].expect("an entry");
        let init = self.b.create_block();
        self.b.ins().jump(init, &[]);
        self.body();
        if !self.f.done {
            self.b.ins().trap(TrapCode::unwrap_user(5));
        }
        // The slots the code may write (and the exit releases) start null,
        // as do the lazy parameters'.
        self.b.switch_to_block(init);
        for r in self.f.written.union(&self.f.cold_written).copied().collect::<Vec<_>>() {
            if r as usize >= args.len() {
                self.b.ins().store(MF, z, regs, r as i32 * 8);
            }
        }
        let lazy: Vec<Reg> = (0..args.len() as Reg).filter(|&r| args[r as usize].is_none()).collect();
        for &r in &lazy {
            self.b.ins().store(MF, z, regs, r as i32 * 8);
        }
        self.b.ins().jump(first, &[]);
        let ret_const = self.f.inline.as_ref().and_then(|i| i.ret_const).flatten();
        let done = std::mem::replace(&mut self.f, caller);
        // An exception: release the frame, raise at the call.
        self.b.switch_to_block(fail);
        let d = self.b.ins().iconst(self.ptr, desc as i64);
        self.call(self.h.unwind, &[regs, d, fail_pc], None);
        for &r in &lazy {
            let v = self.b.ins().load(types::I64, MF, regs, r as i32 * 8);
            self.xdecref(v);
        }
        self.b.ins().jump(raised, &[]);
        self.b.switch_to_block(raised);
        self.raise(pc);
        // The result: release what the frame holds (but the borrowed
        // parameters).
        self.b.switch_to_block(exit);
        // (A lazy parameter's slot holds the box made for it, if any.)
        let owned = |r: Reg| r >= ir.params.len() as u32 || lazy.contains(&r);
        for &r in done.written.iter().filter(|&&r| owned(r)) {
            let v = self.b.ins().load(types::I64, MF, regs, r as i32 * 8);
            self.xdecref(v);
        }
        // Written only on slow paths: when one ran.
        let cold: Vec<Reg> = done.cold_written.iter().copied().filter(|&r| owned(r) && !done.written.contains(&r)).collect();
        self.release_if_dirty(&cold, regs, false);
        // Always the same immortal constant (None): known here.
        match ret_const {
            Some(c) => (self.b.ins().iconst(self.ptr, c as i64), None),
            None => (result, ret_int.then_some(result_int)),
        }
    }


    /// Call a compiled function directly on `frame` (its arguments in
    /// place, borrowed): its result (a new reference), or null with an
    /// exception set. `enter` while it has no code.
    fn native_call(&mut self, fp: Value, desc: Value, entry: usize, target: usize) -> Value {
        let done = self.b.create_block();
        let r = self.b.append_block_param(done, self.ptr);
        let cell = self.b.ins().iconst(self.ptr, entry as i64);
        let code = self.b.ins().load(self.ptr, MF, cell, 0);
        let (native, helper) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(code, native, &[], helper, &[]);
        self.b.switch_to_block(helper);
        let v = self.call(self.h.enter, &[self.ctx, fp, desc], Some(self.ptr)).unwrap();
        self.b.ins().jump(done, &[BlockArg::Value(v)]);
        self.b.switch_to_block(native);
        let zero = self.b.ins().iconst(self.ptr, 0);
        let msg = self.b.ins().iconst(self.ptr, self.h.recursion_msg as i64);
        let deep = self.call(self.h.enter_recursive, &[msg], Some(types::I32)).unwrap();
        let go = self.b.create_block();
        self.b.ins().brif(deep, done, &[BlockArg::Value(zero)], go, &[]);
        self.b.switch_to_block(go);
        // The callee's context: thread, return value, frame, target, exit.
        let ctx = self.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, CTX_SIZE as u32 + 8, 3));
        self.b.ins().stack_store(self.ptr, self.tid, ctx, CTX_TID);
        self.b.ins().stack_store(self.ptr, zero, ctx, CTX_RET);
        self.b.ins().stack_store(self.ptr, fp, ctx, CTX_SLOTS);
        let t = self.b.ins().iconst(self.ptr, target as i64);
        self.b.ins().stack_store(self.ptr, t, ctx, CTX_TARGET);
        self.b.ins().stack_store(self.ptr, zero, ctx, CTX_EXIT);
        self.b.ins().stack_store(self.ptr, zero, ctx, CTX_WANT_INT);
        // `*pc`, after the context.
        let z32 = self.b.ins().iconst(types::I32, 0);
        self.b.ins().stack_store(self.ptr, z32, ctx, CTX_SIZE);
        let cp = self.b.ins().stack_addr(self.ptr, ctx, 0);
        let pcp = self.b.ins().stack_addr(self.ptr, ctx, CTX_SIZE);
        let mut sig = Signature::new(self.call_conv);
        sig.params.extend([AbiParam::new(self.ptr), AbiParam::new(self.ptr), AbiParam::new(self.ptr), AbiParam::new(types::I32)]);
        sig.returns.push(AbiParam::new(types::I32));
        let sig = self.b.import_signature(sig);
        let call = self.b.ins().call_indirect(sig, code, &[cp, fp, pcp, z32]);
        let st = self.b.inst_results(call)[0];
        self.call(self.h.leave_recursive, &[], None);
        let (exited, other, returned, stepped, bailed, raised) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, st, EXIT as i64);
        self.b.ins().brif(c, exited, &[], other, &[]);
        self.b.switch_to_block(exited);
        let v = self.b.ins().stack_load(self.ptr, self.ptr, ctx, CTX_RET);
        self.b.ins().brif(v, returned, &[], stepped, &[]);
        self.b.switch_to_block(returned);
        self.b.ins().jump(done, &[BlockArg::Value(v)]);
        self.b.switch_to_block(stepped);
        let v = self.call(self.h.take_exit, &[cp, desc], Some(self.ptr)).unwrap();
        self.b.ins().jump(done, &[BlockArg::Value(v)]);
        self.b.switch_to_block(other);
        let c = self.b.ins().icmp_imm_s(IntCC::Equal, st, BAIL as i64);
        self.b.ins().brif(c, bailed, &[], raised, &[]);
        self.b.switch_to_block(bailed);
        let pcv = self.b.ins().stack_load(self.ptr, types::I32, ctx, CTX_SIZE);
        let v = self.call(self.h.resume, &[self.ctx, fp, desc, pcv], Some(self.ptr)).unwrap();
        self.b.ins().jump(done, &[BlockArg::Value(v)]);
        self.b.switch_to_block(raised);
        let pcv = self.b.ins().stack_load(self.ptr, types::I32, ctx, CTX_SIZE);
        self.call(self.h.unwind, &[fp, desc, pcv], None);
        self.b.ins().jump(done, &[BlockArg::Value(zero)]);
        self.b.switch_to_block(done);
        r
    }



    /// The end of an inlined call returning an int: `x`, unless the
    /// interpreter finished the call (`r`, an object, not null): then its
    /// value if it fits in an `i64`; else the caller continues in the
    /// interpreter after the call, with the object.
    fn int_result(&mut self, pc: usize, dst: Reg, r: Value, x: Value, slow: Block, start: HashMap<Reg, Virt>) {
        let (boxed, done, big) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let v = self.b.append_block_param(done, types::I64);
        self.b.ins().brif(r, boxed, &[], done, &[BlockArg::Value(x)]);
        self.b.switch_to_block(boxed);
        let y = self.as_i64(r, big);
        self.decref(r);
        self.b.ins().jump(done, &[BlockArg::Value(y)]);
        self.b.set_cold_block(big);
        self.b.switch_to_block(big);
        let saved = self.f.vals.clone();
        self.clobber();
        let was = std::mem::replace(&mut self.slow, true);
        self.put(dst, r);
        self.f.vals.remove(&dst);
        self.bail_to(pc + 1);
        self.slow = was;
        self.f.vals = saved;
        self.b.switch_to_block(done);
        self.clobber();
        self.f.vals.insert(dst, Virt { ptr: None, in_slot: false, num: Unboxed::I64(v), ty: Some(self.l.long_type), konst: false, checked: false, global: false });
        let cont = self.b.create_block();
        self.b.ins().jump(cont, &[]);
        self.bail_path(slow, start, pc);
        self.b.switch_to_block(cont);
        self.next(pc);
    }

    /// A register holding an instance of `tp` that may be constructed in
    /// at call `pc` (`Reuse`): verified, not an argument, dead after the
    /// call, and the only owner of its reference (its slot, or for an
    /// inlined function's parameter, the caller's dying register's).
    fn donor(&self, pc: usize, tp: usize, args: &[Reg]) -> Option<Reg> {
        let mut regs: Vec<Reg> = self.f.vals.keys().copied().collect();
        regs.sort_unstable();
        regs.into_iter().find(|&r| {
            let v = self.f.vals[&r];
            let owner = match &self.f.inline {
                Some(inl) if r < self.f.nparams => inl.donors.get(r as usize).copied().unwrap_or(false),
                _ => v.in_slot && !self.is_param(r),
            };
            v.ty == Some(tp) && v.checked && v.ptr.is_some() && owner && !args.contains(&r) && !self.live_at(pc + 1, r)
        })
    }

    /// Continue when `p` can be constructed in (`Reuse`), else at `no`:
    /// one reference, by this thread; no weak references; its attributes
    /// those `__init__` assigns, in its order.
    fn reusable(&mut self, p: Value, r: &Reuse, no: Block) {
        let yes = self.b.create_block();
        self.unique(p, yes, no);
        self.b.switch_to_block(yes);
        if let Some(w) = r.weakref {
            let l = self.b.ins().load(self.ptr, MF, p, w);
            self.guard_eq(l, 0, no);
        }
        let Some(values) = r.values else { return };
        let size = self.b.ins().load(types::I8, MF, p, values + 1);
        self.guard_eq(size, r.order.len() as i64, no);
        let cap = self.b.ins().load(types::I8, MF, p, values);
        let cap = self.b.ins().uextend(types::I64, cap);
        let at = self.b.ins().ishl_imm_u(cap, 3);
        let at = self.b.ins().iadd(at, p);
        for (i, &ix) in r.order.iter().enumerate() {
            let b = self.b.ins().load(types::I8, MF, at, values + 8 + i as i32);
            self.guard_eq(b, ix as i64, no);
        }
    }

    /// Continue at `yes` when only the current thread references `p`, once.
    fn unique(&mut self, p: Value, yes: Block, no: Block) {
        match self.l.rc {
            Rc::Calls => {
                self.b.ins().jump(no, &[]);
            }
            Rc::Gil => {
                let c = self.b.ins().load(types::I32, MF, p, 0);
                let one = self.b.ins().icmp_imm_s(IntCC::Equal, c, 1);
                self.b.ins().brif(one, yes, &[], no, &[]);
            }
            Rc::Ft { tid, local, shared } => {
                let owner = self.b.ins().load(self.ptr, MF, p, tid);
                let own = self.b.ins().icmp(IntCC::Equal, owner, self.tid);
                let c = self.b.ins().load(types::I32, MF, p, local);
                let one = self.b.ins().icmp_imm_s(IntCC::Equal, c, 1);
                let s = self.b.ins().load(types::I64, MF, p, shared);
                let none = self.b.ins().icmp_imm_s(IntCC::Equal, s, 0);
                let a = self.b.ins().band(own, one);
                let a = self.b.ins().band(a, none);
                self.b.ins().brif(a, yes, &[], no, &[]);
            }
        }
    }

}
