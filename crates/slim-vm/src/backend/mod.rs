//! The `Backend` abstraction: opcode semantics are written once, generic over
//! a backend. The interpreter backend executes them directly; a future JIT
//! backend will run the same functions to emit machine code.
//!
//! Rules for semantics (see docs/DESIGN.md):
//! - operands are plain Rust values, only runtime data is `B::V`;
//! - guards come before side effects (ops must be restartable);
//! - complex ops call runtime helpers (`helper2`, `frame_op`).

pub mod interp;

use crate::code::Reg;
use crate::interp::{Frame, Thread};
use crate::value::Value;

/// Why an op leaves straight-line execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// An exception is pending in the thread.
    Raise,
    /// The frame returned; the value is in the backend.
    Return,
    /// A generator suspended.
    Yield,
}

pub type Flow<T> = Result<T, Exit>;

/// Runtime helper with value operands; returns `Value::UNINIT` if it raised.
pub type Helper1 = extern "C" fn(&mut Thread, Value) -> Value;
pub type Helper2 = extern "C" fn(&mut Thread, Value, Value) -> Value;

/// Runtime helper with access to the frame; returns 0 on success, 1 if it
/// raised, 2 if the frame returned (value in `Frame::ret`), 3 if it yielded.
pub type FrameHelper = extern "C" fn(&mut Thread, &mut Frame, u32, u32, u32, u32) -> u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    Lt,
    Le,
    Gt,
    Ge,
}

pub trait Backend {
    /// A runtime value (a `Value` word).
    type V: Copy;
    /// A runtime boolean.
    type B: Copy;

    fn reg(&mut self, r: Reg) -> Self::V;
    fn set_reg(&mut self, r: Reg, v: Self::V);
    fn imm(&mut self, bits: u64) -> Self::V;
    fn code_const(&mut self, idx: u32) -> Self::V;

    fn is_small_int(&mut self, v: Self::V) -> Self::B;
    fn eq_word(&mut self, a: Self::V, b: Self::V) -> Self::B;
    fn and(&mut self, a: Self::B, b: Self::B) -> Self::B;
    fn or(&mut self, a: Self::B, b: Self::B) -> Self::B;
    fn not(&mut self, a: Self::B) -> Self::B;
    /// Signed 64-bit add/sub/mul returning (result, overflowed).
    fn add_ovf(&mut self, a: Self::V, b: Self::V) -> (Self::V, Self::B);
    fn sub_ovf(&mut self, a: Self::V, b: Self::V) -> (Self::V, Self::B);
    fn mul_ovf(&mut self, a: Self::V, b: Self::V) -> (Self::V, Self::B);
    fn add_word(&mut self, a: Self::V, k: i64) -> Self::V;
    fn sar(&mut self, a: Self::V, n: u8) -> Self::V;
    fn cmp(&mut self, c: Cond, a: Self::V, b: Self::V) -> Self::B;
    fn bool_value(&mut self, c: Self::B) -> Self::V;

    fn if_v(
        &mut self,
        c: Self::B,
        t: impl FnOnce(&mut Self) -> Flow<Self::V>,
        e: impl FnOnce(&mut Self) -> Flow<Self::V>,
    ) -> Flow<Self::V>;
    fn if_unit(
        &mut self,
        c: Self::B,
        t: impl FnOnce(&mut Self) -> Flow<()>,
        e: impl FnOnce(&mut Self) -> Flow<()>,
    ) -> Flow<()>;

    fn jump(&mut self, target: u32) -> Flow<()>;
    /// Jumps to `target` if `c`, otherwise falls through.
    fn branch(&mut self, c: Self::B, target: u32) -> Flow<()>;
    fn ret(&mut self, v: Self::V) -> Flow<()>;

    fn helper1(&mut self, h: Helper1, a: Self::V) -> Flow<Self::V>;
    fn helper2(&mut self, h: Helper2, a: Self::V, b: Self::V) -> Flow<Self::V>;
    fn frame_op(&mut self, h: FrameHelper, a: u32, b: u32, c: u32, d: u32) -> Flow<()>;

    fn incref(&mut self, v: Self::V);
    fn decref(&mut self, v: Self::V);
}
