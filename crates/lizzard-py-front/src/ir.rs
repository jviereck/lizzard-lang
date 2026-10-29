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
    LoadAttr { dst: Reg, obj: Reg, name: u32 },
    StoreAttr { obj: Reg, name: u32, val: Reg },
    /// `func(*args[..n - kwnames.len()], **{kwnames: args[rest]})`
    Call { dst: Reg, func: Reg, args: Vec<Reg>, kwnames: Vec<u32> },
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
