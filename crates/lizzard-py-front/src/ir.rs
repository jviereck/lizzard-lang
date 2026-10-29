//! The register-based IR the runtime interprets: one `FuncIr` per checked
//! function. Registers have a fixed representation (`Repr`); `Obj` registers
//! hold owned Python objects, the others unboxed values.

pub type Reg = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repr {
    Obj,
    I64,
    F32,
    Bool,
}

#[derive(Clone, Debug)]
pub enum Const {
    None,
    Bool(bool),
    Int(i64),
    /// An int literal outside the i64 range, in decimal.
    BigInt(String),
    Float(f64),
    Str(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    MatMul,
    BitOr,
    BitAnd,
    BitXor,
    LShift,
    RShift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Is,
    IsNot,
    In,
    NotIn,
}

/// A runtime type check, derived from a static type.
#[derive(Clone, Debug, PartialEq)]
pub enum RtCheck {
    Any,
    Int,
    /// `float` also accepts `int`, as in the typing spec.
    Float,
    Bool,
    Str,
    None,
    Tuple,
    /// `isinstance` against the class with this qualified name (skipped if
    /// the name does not exist at runtime).
    Instance(String),
    /// A `numpy.ndarray`, optionally with a known scalar type (`dtype.type`)
    /// and number of dimensions.
    NdArray { scalar: Option<String>, ndim: Option<u32> },
    /// Any of the checks.
    Union(Vec<RtCheck>),
    /// `callable(obj)`
    Callable,
}

#[derive(Clone, Debug)]
pub struct CheckIr {
    pub check: RtCheck,
    /// What produced the value (`argument 'b'`, `np.empty_like(...)`).
    pub what: String,
    /// The declared type, for messages.
    pub expected: String,
    /// Where the type was declared (`.../numpy/__init__.pyi:57`), if not in
    /// the module itself.
    pub origin: Option<String>,
}

#[derive(Clone, Debug)]
pub enum Instr {
    Const { dst: Reg, idx: u32 },
    /// Module globals, then builtins.
    LoadGlobal { dst: Reg, name: u32 },
    StoreGlobal { name: u32, val: Reg },
    /// A new empty cell, for a variable an inner function captures.
    MakeCell { dst: Reg },
    /// The contents of a cell (UnboundLocalError if empty).
    LoadCell { dst: Reg, cell: Reg },
    StoreCell { cell: Reg, val: Reg },
    /// The contents of closure cell `idx` (NameError if empty).
    LoadFree { dst: Reg, idx: u32 },
    StoreFree { idx: u32, val: Reg },
    /// Closure cell `idx` itself, to pass on to an inner function.
    LoadFreeCell { dst: Reg, idx: u32 },
    /// A new function object for the module function with key `names[key]`,
    /// closing over the cells in `closure`.
    MakeFunction { dst: Reg, key: u32, closure: Vec<Reg>, defaults: Vec<Reg> },
    Move { dst: Reg, src: Reg },
    /// `dst = src`, and `src` becomes unbound: the reference moves, with no
    /// reference counting (assigning a temporary to a variable).
    Take { dst: Reg, src: Reg },
    LoadAttr { dst: Reg, obj: Reg, name: u32 },
    StoreAttr { obj: Reg, name: u32, val: Reg },
    DelAttr { obj: Reg, name: u32 },
    /// `func(*args[..n - kwnames.len()], **{kwnames: args[rest]})`
    Call { dst: Reg, func: Reg, args: Vec<Reg>, kwnames: Vec<u32>, direct: Option<Direct> },
    /// `obj.name(*args[..n - kwnames.len()], **{kwnames: args[rest]})`,
    /// without creating a bound method (CPython's `PyObject_VectorcallMethod`).
    /// The method is looked up after the arguments are evaluated, so the
    /// front end uses it only when they are side-effect free.
    CallMethod { dst: Reg, obj: Reg, name: u32, args: Vec<Reg>, kwnames: Vec<u32>, direct: Option<Direct> },
    /// `func(*args, **kwargs)` with a tuple and a dict (calls with `*`/`**`).
    CallEx { dst: Reg, func: Reg, args: Reg, kwargs: Option<Reg> },
    /// `list.extend(iter)`
    ListExtend { list: Reg, iter: Reg },
    /// `dst = tuple(src)`
    ListToTuple { dst: Reg, src: Reg },
    /// Merge mapping `src` into the keyword dict `dict` for `func`; a
    /// repeated key is a TypeError, as in a call.
    DictMerge { dict: Reg, src: Reg, func: Reg },
    /// `a op b`, or `a op= b` (`__iadd__`, ...) when `inplace`.
    BinOp { dst: Reg, op: BinOp, a: Reg, b: Reg, inplace: bool },
    Neg { dst: Reg, src: Reg },
    /// `~src`
    Invert { dst: Reg, src: Reg },
    /// `+src`
    Pos { dst: Reg, src: Reg },
    Compare { dst: Reg, op: CmpOp, a: Reg, b: Reg },
    /// `dst: Bool = bool(a op b)`, for a branch condition.
    CompareBool { dst: Reg, op: CmpOp, a: Reg, b: Reg },
    GetItem { dst: Reg, obj: Reg, key: Reg },
    SetItem { obj: Reg, key: Reg, val: Reg },
    DelItem { obj: Reg, key: Reg },
    BuildList { dst: Reg, items: Vec<Reg> },
    BuildTuple { dst: Reg, items: Vec<Reg> },
    BuildDict { dst: Reg, keys: Vec<Reg>, vals: Vec<Reg> },
    BuildSet { dst: Reg, items: Vec<Reg> },
    /// `set.add(val)` for comprehensions.
    SetAdd { set: Reg, val: Reg },
    BuildSlice { dst: Reg, lower: Reg, upper: Reg, step: Reg },
    /// `list.append(val)` for comprehensions.
    ListAppend { list: Reg, val: Reg },
    /// `dst = iter(obj)`
    GetIter { dst: Reg, obj: Reg },
    /// `dst = next(iter)`, or jump to `target` when the iterator is
    /// exhausted.
    ForIter { dst: Reg, iter: Reg, target: u32 },
    /// `dsts[0], dsts[1], ... = src`, with CPython's errors.
    Unpack { src: Reg, dsts: Vec<Reg> },
    /// `format(conv(val), spec)` for an f-string field; `conv` is `0`, `b's'`,
    /// `b'r'` or `b'a'`.
    Format { dst: Reg, val: Reg, conv: u8, spec: Option<Reg> },
    /// The concatenation of string registers.
    BuildString { dst: Reg, parts: Vec<Reg> },
    /// `dst: Bool = bool(src)`
    Truthy { dst: Reg, src: Reg },
    Not { dst: Reg, src: Reg },
    Jump { target: u32 },
    JumpIfFalse { cond: Reg, target: u32 },
    JumpIfTrue { cond: Reg, target: u32 },
    /// `raise exc` (a class is instantiated), or `raise exc from cause`.
    Raise { exc: Reg, cause: Option<Reg> },
    /// Suspend the generator, producing `src`; it resumes at the next
    /// instruction with the value sent in `dst` (or with an exception
    /// raised here, by `throw`).
    Yield { dst: Reg, src: Reg },
    /// `yield from iter`: the generator object delegates `send`, `throw` and
    /// `close` to `iter` until it finishes; then `dst` is its return value
    /// and execution continues at the next instruction.
    YieldFrom { dst: Reg, iter: Reg },
    /// Raise a caught exception again, as it is.
    Reraise { exc: Reg },
    /// `dst: Bool = isinstance(exc, cls)` for `except cls:`
    ExcMatch { dst: Reg, exc: Reg, cls: Reg },
    /// The exception being handled (`sys.exception()`), or None.
    GetHandled { dst: Reg },
    /// Set the exception being handled (None clears it).
    SetHandled { src: Reg },
    /// Make a register unbound again (`except E as e` ends with `del e`).
    Unbind { reg: Reg },
    /// `dst = type(src)`
    TypeOf { dst: Reg, src: Reg },
    /// `type(obj).name`, bound to `obj` (how `with` finds `__enter__`).
    LoadSpecial { dst: Reg, obj: Reg, name: u32 },
    Return { src: Reg },
    I64Const { dst: Reg, value: i64 },
    I64Lt { dst: Reg, a: Reg, b: Reg },
    I64Add { dst: Reg, a: Reg, b: Reg },
    /// `dst: I64 = len(obj)`
    Len { dst: Reg, obj: Reg },
    /// Unboxed value to object (`I64` to int, `F32` to `numpy.float32`,
    /// `Bool` to bool).
    Box { dst: Reg, src: Reg },
    /// Object to the representation of `dst` (`I64` or `F32`).
    Unbox { dst: Reg, src: Reg },
    /// Element `idx` of a 1-D float32 array, read directly from its buffer
    /// when the array allows it, else through `__getitem__`.
    ArrLoadF32 { dst: Reg, arr: Reg, idx: Reg },
    ArrStoreF32 { arr: Reg, idx: Reg, val: Reg },
    F32Bin { dst: Reg, op: BinOp, a: Reg, b: Reg },
    /// Raise `TypeError` unless `reg` passes `checks[check]`.
    Check { reg: Reg, check: u32 },
}

/// What the checker resolved a call to: a function of the same module with
/// plain positional parameters (`key` as in `FuncIr::key`). When the
/// function called at runtime is that one, the checks of the parameters in
/// `skip` (bit i: parameter i, the receiver first for a method) can be
/// skipped: the arguments' static types already satisfy them.
#[derive(Clone, Debug)]
pub struct Direct {
    pub key: String,
    pub skip: u64,
}

/// An exception raised by an instruction in one of `ranges` (`start..end`)
/// is stored in `exc` and execution continues at `target`. Handlers nest,
/// and an inner one comes after its outer ones: the last that covers the
/// instruction wins.
#[derive(Clone, Debug)]
pub struct HandlerIr {
    pub ranges: Vec<(u32, u32)>,
    pub target: u32,
    pub exc: Reg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    PosOnly,
    PosOrKw,
    /// `*args`: its register holds a tuple, and `check` applies to items.
    VarPos,
    KwOnly,
    /// `**kwargs`: its register holds a dict, and `check` applies to values.
    VarKw,
}

#[derive(Clone, Debug)]
pub struct ParamIr {
    pub name: String,
    pub kind: ParamKind,
    /// Index into `checks`.
    pub check: u32,
    /// The default value, an index into `consts`.
    pub default: Option<u32>,
    /// The default is a value the function object holds (evaluated where an
    /// inner function or lambda is created), in parameter order.
    pub dynamic_default: bool,
}

#[derive(Clone, Debug)]
pub struct FuncIr {
    /// The qualified name: `f`, or `C.m` for a method.
    pub name: String,
    /// How the module finds it: the name, or `C.x.setter` for a property
    /// setter.
    pub key: String,
    pub file: String,
    pub line: u32,
    /// Parameters occupy registers `0..params.len()` (all `Obj`).
    pub params: Vec<ParamIr>,
    pub reprs: Vec<Repr>,
    pub code: Vec<Instr>,
    /// Source line of each instruction.
    pub lines: Vec<u32>,
    pub consts: Vec<Const>,
    pub names: Vec<String>,
    pub checks: Vec<CheckIr>,
    pub handlers: Vec<HandlerIr>,
    /// The variables its closure provides, in cell order.
    pub free: Vec<String>,
    /// A generator function: calling it creates a generator object.
    pub generator: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ModuleIr {
    pub functions: Vec<FuncIr>,
}

impl Instr {
    /// The registers it reads.
    pub fn reads(&self) -> Vec<Reg> {
        use Instr::*;
        let mut v: Vec<Reg> = Vec::new();
        match self {
            Const { .. } | LoadGlobal { .. } | MakeCell { .. } | LoadFree { .. } | LoadFreeCell { .. } | Jump { .. } | GetHandled { .. } | Unbind { .. } | I64Const { .. } => {}
            StoreGlobal { val, .. } | StoreFree { val, .. } => v.push(*val),
            LoadCell { cell, .. } => v.push(*cell),
            StoreCell { cell, val } => v.extend([*cell, *val]),
            MakeFunction { closure, defaults, .. } => v.extend(closure.iter().chain(defaults)),
            Move { src, .. } | Take { src, .. } | Neg { src, .. } | Invert { src, .. } | Pos { src, .. } | ListToTuple { src, .. } | Truthy { src, .. } | Not { src, .. } | TypeOf { src, .. } | Box { src, .. } | Unbox { src, .. } | SetHandled { src } | Return { src } | Unpack { src, .. } => v.push(*src),
            LoadAttr { obj, .. } | DelAttr { obj, .. } | GetIter { obj, .. } | LoadSpecial { obj, .. } | Len { obj, .. } => v.push(*obj),
            StoreAttr { obj, val, .. } => v.extend([*obj, *val]),
            Call { func, args, .. } => {
                v.push(*func);
                v.extend(args);
            }
            CallMethod { obj, args, .. } => {
                v.push(*obj);
                v.extend(args);
            }
            CallEx { func, args, kwargs, .. } => {
                v.extend([*func, *args]);
                v.extend(kwargs);
            }
            ListExtend { list, iter } => v.extend([*list, *iter]),
            DictMerge { dict, src, func } => v.extend([*dict, *src, *func]),
            BinOp { a, b, .. } | Compare { a, b, .. } | CompareBool { a, b, .. } | I64Lt { a, b, .. } | I64Add { a, b, .. } | F32Bin { a, b, .. } => v.extend([*a, *b]),
            GetItem { obj, key, .. } | DelItem { obj, key } => v.extend([*obj, *key]),
            SetItem { obj, key, val } => v.extend([*obj, *key, *val]),
            BuildList { items, .. } | BuildTuple { items, .. } | BuildSet { items, .. } => v.extend(items),
            BuildDict { keys, vals, .. } => v.extend(keys.iter().chain(vals)),
            SetAdd { set, val } => v.extend([*set, *val]),
            BuildSlice { lower, upper, step, .. } => v.extend([*lower, *upper, *step]),
            ListAppend { list, val } => v.extend([*list, *val]),
            ForIter { iter, .. } | YieldFrom { iter, .. } => v.push(*iter),
            Format { val, spec, .. } => {
                v.push(*val);
                v.extend(spec);
            }
            BuildString { parts, .. } => v.extend(parts),
            JumpIfFalse { cond, .. } | JumpIfTrue { cond, .. } => v.push(*cond),
            Raise { exc, cause } => {
                v.push(*exc);
                v.extend(cause);
            }
            Yield { src, .. } => v.push(*src),
            Reraise { exc } => v.push(*exc),
            ExcMatch { exc, cls, .. } => v.extend([*exc, *cls]),
            ArrLoadF32 { arr, idx, .. } => v.extend([*arr, *idx]),
            ArrStoreF32 { arr, idx, val } => v.extend([*arr, *idx, *val]),
            Check { reg, .. } => v.push(*reg),
        }
        v
    }

    /// The registers it writes (`Unbind` and `Take` clear theirs: a write).
    pub fn writes(&self) -> Vec<Reg> {
        use Instr::*;
        match self {
            Const { dst, .. } | LoadGlobal { dst, .. } | MakeCell { dst } | LoadCell { dst, .. } | LoadFree { dst, .. } | LoadFreeCell { dst, .. } | MakeFunction { dst, .. } | Move { dst, .. } | LoadAttr { dst, .. } | Call { dst, .. } | CallMethod { dst, .. } | CallEx { dst, .. } | ListToTuple { dst, .. } | BinOp { dst, .. } | Neg { dst, .. } | Invert { dst, .. } | Pos { dst, .. } | Compare { dst, .. } | CompareBool { dst, .. } | GetItem { dst, .. } | BuildList { dst, .. } | BuildTuple { dst, .. } | BuildDict { dst, .. } | BuildSet { dst, .. } | BuildSlice { dst, .. } | GetIter { dst, .. } | ForIter { dst, .. } | Format { dst, .. } | BuildString { dst, .. } | Truthy { dst, .. } | Not { dst, .. } | Yield { dst, .. } | YieldFrom { dst, .. } | ExcMatch { dst, .. } | GetHandled { dst } | TypeOf { dst, .. } | LoadSpecial { dst, .. } | I64Const { dst, .. } | I64Lt { dst, .. } | I64Add { dst, .. } | Len { dst, .. } | Box { dst, .. } | Unbox { dst, .. } | ArrLoadF32 { dst, .. } | F32Bin { dst, .. } => vec![*dst],
            Take { dst, src } => vec![*dst, *src],
            Unbind { reg } => vec![*reg],
            Unpack { dsts, .. } => dsts.clone(),
            _ => Vec::new(),
        }
    }

    /// Where control can go next, besides an exception: the next
    /// instruction (`pc + 1`) and jump targets.
    pub fn successors(&self, pc: u32) -> Vec<u32> {
        use Instr::*;
        match self {
            Jump { target } => vec![*target],
            JumpIfFalse { target, .. } | JumpIfTrue { target, .. } | ForIter { target, .. } => vec![pc + 1, *target],
            Return { .. } | Raise { .. } | Reraise { .. } => Vec::new(),
            _ => vec![pc + 1],
        }
    }
}

impl FuncIr {
    /// Turn a `Move` of an object whose source is not read again into a
    /// `Take`: the reference moves instead of an incref and a decref.
    pub fn moves_to_takes(&mut self) {
        let n = self.code.len();
        let nregs = self.reprs.len();
        let words = nregs.div_ceil(64);
        // Exception edges: an instruction a handler covers may continue at it.
        let handler_of = |pc: u32| -> Vec<u32> { self.handlers.iter().filter(|h| h.ranges.iter().any(|(s, e)| *s <= pc && pc < *e)).map(|h| h.target).collect() };
        let succ: Vec<Vec<usize>> = (0..n).map(|pc| self.code[pc].successors(pc as u32).into_iter().map(|t| t as usize).filter(|&t| t < n).collect()).collect();
        let exc_succ: Vec<Vec<usize>> = (0..n).map(|pc| handler_of(pc as u32).into_iter().map(|t| t as usize).filter(|&t| t < n).collect()).collect();
        let reads: Vec<Vec<Reg>> = self.code.iter().map(|i| i.reads()).collect();
        let writes: Vec<Vec<Reg>> = self.code.iter().map(|i| i.writes()).collect();
        // Live registers before each instruction, to a fixed point.
        let mut live_in = vec![vec![0u64; words]; n];
        let mut changed = true;
        while changed {
            changed = false;
            for pc in (0..n).rev() {
                let mut out = vec![0u64; words];
                for &s in &succ[pc] {
                    for w in 0..words {
                        out[w] |= live_in[s][w];
                    }
                }
                for r in &writes[pc] {
                    out[*r as usize / 64] &= !(1 << (*r % 64));
                }
                for r in &reads[pc] {
                    out[*r as usize / 64] |= 1 << (*r % 64);
                }
                // If it raises, its writes have not happened: what a handler
                // reads is live before it.
                for &h in &exc_succ[pc] {
                    for w in 0..words {
                        out[w] |= live_in[h][w];
                    }
                }
                if out != live_in[pc] {
                    live_in[pc] = out;
                    changed = true;
                }
            }
        }
        for pc in 0..n {
            if let Instr::Move { dst, src } = self.code[pc] {
                if dst == src || self.reprs[src as usize] != Repr::Obj {
                    continue;
                }
                let live_after = succ[pc].iter().chain(&exc_succ[pc]).any(|&s| live_in[s][src as usize / 64] & (1 << (src % 64)) != 0);
                if !live_after {
                    self.code[pc] = Instr::Take { dst, src };
                }
            }
        }
    }
}
