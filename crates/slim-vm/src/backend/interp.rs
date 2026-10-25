//! Interpreter backend: semantics run directly on values.

use super::{Backend, Cond, Exit, Flow, FrameHelper, Helper1, Helper2};
use crate::code::Reg;
use crate::interp::{Frame, Thread};
use crate::value::Value;

pub struct Interp<'a> {
    pub th: &'a mut Thread,
    pub frame: &'a mut Frame,
}

impl Backend for Interp<'_> {
    type V = Value;
    type B = bool;

    #[inline(always)]
    fn reg(&mut self, r: Reg) -> Value {
        // SAFETY: the compiler only emits registers below `nregs`.
        unsafe { *self.frame.regs.add(r as usize) }
    }

    #[inline(always)]
    fn set_reg(&mut self, r: Reg, v: Value) {
        unsafe { *self.frame.regs.add(r as usize) = v }
    }

    #[inline(always)]
    fn imm(&mut self, bits: u64) -> Value {
        Value(bits)
    }

    #[inline(always)]
    fn code_const(&mut self, idx: u32) -> Value {
        self.frame.code().consts[idx as usize]
    }

    #[inline(always)]
    fn is_small_int(&mut self, v: Value) -> bool {
        v.is_small_int()
    }

    #[inline(always)]
    fn eq_word(&mut self, a: Value, b: Value) -> bool {
        a == b
    }

    #[inline(always)]
    fn and(&mut self, a: bool, b: bool) -> bool {
        a & b
    }

    #[inline(always)]
    fn or(&mut self, a: bool, b: bool) -> bool {
        a | b
    }

    #[inline(always)]
    fn not(&mut self, a: bool) -> bool {
        !a
    }

    #[inline(always)]
    fn add_ovf(&mut self, a: Value, b: Value) -> (Value, bool) {
        let (r, o) = (a.0 as i64).overflowing_add(b.0 as i64);
        (Value(r as u64), o)
    }

    #[inline(always)]
    fn sub_ovf(&mut self, a: Value, b: Value) -> (Value, bool) {
        let (r, o) = (a.0 as i64).overflowing_sub(b.0 as i64);
        (Value(r as u64), o)
    }

    #[inline(always)]
    fn mul_ovf(&mut self, a: Value, b: Value) -> (Value, bool) {
        let (r, o) = (a.0 as i64).overflowing_mul(b.0 as i64);
        (Value(r as u64), o)
    }

    #[inline(always)]
    fn add_word(&mut self, a: Value, k: i64) -> Value {
        Value((a.0 as i64).wrapping_add(k) as u64)
    }

    #[inline(always)]
    fn sar(&mut self, a: Value, n: u8) -> Value {
        Value(((a.0 as i64) >> n) as u64)
    }

    #[inline(always)]
    fn cmp(&mut self, c: Cond, a: Value, b: Value) -> bool {
        let (a, b) = (a.0 as i64, b.0 as i64);
        match c {
            Cond::Lt => a < b,
            Cond::Le => a <= b,
            Cond::Gt => a > b,
            Cond::Ge => a >= b,
        }
    }

    #[inline(always)]
    fn bool_value(&mut self, c: bool) -> Value {
        Value::bool(c)
    }

    #[inline(always)]
    fn if_v(
        &mut self,
        c: bool,
        t: impl FnOnce(&mut Self) -> Flow<Value>,
        e: impl FnOnce(&mut Self) -> Flow<Value>,
    ) -> Flow<Value> {
        if c {
            t(self)
        } else {
            e(self)
        }
    }

    #[inline(always)]
    fn if_unit(
        &mut self,
        c: bool,
        t: impl FnOnce(&mut Self) -> Flow<()>,
        e: impl FnOnce(&mut Self) -> Flow<()>,
    ) -> Flow<()> {
        if c {
            t(self)
        } else {
            e(self)
        }
    }

    #[inline(always)]
    fn jump(&mut self, target: u32) -> Flow<()> {
        self.frame.pc = target;
        Ok(())
    }

    #[inline(always)]
    fn branch(&mut self, c: bool, target: u32) -> Flow<()> {
        if c {
            self.frame.pc = target;
        }
        Ok(())
    }

    #[inline(always)]
    fn ret(&mut self, v: Value) -> Flow<()> {
        self.frame.ret = v;
        Err(Exit::Return)
    }

    #[inline(always)]
    fn helper1(&mut self, h: Helper1, a: Value) -> Flow<Value> {
        let r = h(self.th, a);
        if r == Value::UNINIT {
            Err(Exit::Raise)
        } else {
            Ok(r)
        }
    }

    #[inline(always)]
    fn helper2(&mut self, h: Helper2, a: Value, b: Value) -> Flow<Value> {
        let r = h(self.th, a, b);
        if r == Value::UNINIT {
            Err(Exit::Raise)
        } else {
            Ok(r)
        }
    }

    #[inline(always)]
    fn frame_op(&mut self, h: FrameHelper, a: u32, b: u32, c: u32, d: u32) -> Flow<()> {
        match h(self.th, self.frame, a, b, c, d) {
            0 => Ok(()),
            1 => Err(Exit::Raise),
            2 => Err(Exit::Return),
            _ => Err(Exit::Yield),
        }
    }

    #[inline(always)]
    fn incref(&mut self, v: Value) {
        v.incref()
    }

    #[inline(always)]
    fn decref(&mut self, v: Value) {
        v.decref()
    }
}
