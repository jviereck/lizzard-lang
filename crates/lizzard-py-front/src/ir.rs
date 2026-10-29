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
    Move { dst: Reg, src: Reg },
    LoadAttr { dst: Reg, obj: Reg, name: u32 },
    /// `func(*args[..n - kwnames.len()], **{kwnames: args[rest]})`
    Call { dst: Reg, func: Reg, args: Vec<Reg>, kwnames: Vec<u32> },
    BinOp { dst: Reg, op: BinOp, a: Reg, b: Reg },
    Compare { dst: Reg, op: CmpOp, a: Reg, b: Reg },
    GetItem { dst: Reg, obj: Reg, key: Reg },
    SetItem { obj: Reg, key: Reg, val: Reg },
    /// `dst: Bool = bool(src)`
    Truthy { dst: Reg, src: Reg },
    Not { dst: Reg, src: Reg },
    Jump { target: u32 },
    JumpIfFalse { cond: Reg, target: u32 },
    JumpIfTrue { cond: Reg, target: u32 },
    Raise { exc: Reg },
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

#[derive(Clone, Debug)]
pub struct ParamIr {
    pub name: String,
    /// Index into `checks`.
    pub check: u32,
}

#[derive(Clone, Debug)]
pub struct FuncIr {
    pub name: String,
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
}

#[derive(Clone, Debug, Default)]
pub struct ModuleIr {
    pub functions: Vec<FuncIr>,
}
