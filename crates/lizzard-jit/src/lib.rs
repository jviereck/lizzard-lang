//! Lizzard's Cranelift JIT: compiles a function's IR (`FuncIr`) to native
//! code that runs on the same frame as the interpreter.
//!
//! The frame is an array of 8-byte register slots (see `lizzard-py`'s
//! `frame.rs`). Compiled code does the control flow itself (jumps,
//! branches, exception handler dispatch, the entries a generator resumes
//! at), and the unboxed `i64`/`f32`/bool instructions inline. Every other
//! instruction calls the runtime's `step` helper, which executes that one
//! instruction with the interpreter's semantics, so both engines behave
//! the same and an instruction can move inline without changing the rest.
//!
//! A compiled function is
//! `extern "C" fn(ctx, regs: *mut u64, pc: *mut u32, pending: u32) -> u32`.
//! It starts at `*pc`: 0, or where a generator resumes (after a `yield`, or
//! at the `yield` itself with an exception pending, as `throw` raises it).
//! It returns a status and leaves `*pc` at the instruction that returned,
//! yielded or raised, like the interpreter's `exec`.
//!
//! This crate knows nothing about Python: the runtime passes the helpers'
//! addresses, and `ctx` is opaque here.

use std::ffi::c_void;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, AbiParam, Block, InstBuilder, MemFlagsData, SigRef, Signature, TrapCode, Value};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module};
use lizzard_py_front::ir::{BinOp, CmpOp, FuncIr, Instr, Repr};

/// The helper executed the instruction; continue with the next one.
pub const NEXT: u32 = 0;
/// The helper took the instruction's jump (`ForIter` at the end).
pub const TAKEN: u32 = 1;
/// The function returned or yielded (the runtime has the value).
pub const EXIT: u32 = 2;
/// An exception is set (in CPython's thread state).
pub const ERROR: u32 = 3;

/// The runtime functions compiled code calls, by address.
#[derive(Clone, Copy)]
pub struct Helpers {
    /// `step(ctx, regs, pc) -> status`: execute instruction `pc`
    /// (`NEXT`, `TAKEN`, `EXIT` or `ERROR`).
    pub step: usize,
    /// `catch(ctx, regs, reg)`: move the exception being raised into
    /// register `reg`, for a handler.
    pub catch: usize,
    /// `safepoint(ctx) -> status`: run pending signal handlers and let a
    /// stop-the-world pause happen (`NEXT` or `ERROR`). Called every
    /// `FUEL` loop iterations.
    pub safepoint: usize,
    /// The hot object instructions, with their operands as immediates
    /// (`NEXT` or `ERROR`, except `konst`):
    /// `load_attr(regs, dst, obj, name, ic)`
    pub load_attr: usize,
    /// `store_attr(regs, obj, val, name, ic)`
    pub store_attr: usize,
    /// `konst(regs, dst, obj)`: store a new reference to `obj`.
    pub konst: usize,
    /// `mov(regs, dst, src)` for objects.
    pub mov: usize,
    /// `box_i64(regs, dst, src)`
    pub box_i64: usize,
    /// `truthy(regs, dst, src)`
    pub truthy: usize,
    /// `binop(regs, dst, a, b, op)`: `op` is the `BinOp`'s index, plus 16
    /// in place.
    pub binop: usize,
    /// `compare(regs, dst, a, b, op)` with CPython's `Py_LT`... codes.
    pub compare: usize,
    /// `compare_bool(regs, dst, a, b, op)`: the same, as a `Bool`.
    pub compare_bool: usize,
    /// `take(regs, dst, src)`: move an object reference.
    pub take: usize,
    /// `ret(ctx, regs, src)`: hand `src` to the runtime as the return value
    /// (`NEXT`, then the code exits) or `ERROR`.
    pub ret: usize,
    /// `call(ctx, regs, pc)` for `Call` and `CallMethod`.
    pub call: usize,
    /// `for_iter(ctx, regs, dst, iter, pc)`: `NEXT`, `TAKEN` at the end, or
    /// `ERROR`.
    pub for_iter: usize,
    /// `check(ctx, regs, reg, check, pc)`
    pub check: usize,
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

/// What follows a helper call's status.
#[derive(Clone, Copy)]
enum After {
    /// No status: continue.
    Nothing,
    /// `NEXT` continues, anything else raises.
    Next,
    /// `NEXT` leaves the function (it returned).
    Exit,
    /// `NEXT` continues, `TAKEN` jumps to the target, else it raises.
    Branch(u32),
}

/// The objects compiled code refers to by address: the function's
/// constants and interned names. They must outlive the code.
pub struct Objects<'a> {
    pub consts: &'a [usize],
    pub names: &'a [usize],
    /// Each instruction's inline cache.
    pub ics: &'a [usize],
}

/// Loop iterations between safepoints.
const FUEL: i64 = 1024;

pub type Entry = unsafe extern "C" fn(ctx: *mut c_void, regs: *mut u64, pc: *mut u32, pending: u32) -> u32;

/// Compile `ir`. The code lives as long as the process.
pub fn compile(ir: &FuncIr, helpers: &Helpers, objects: &Objects) -> Result<Entry, String> {
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
        let mut sigs = Sigs::default();
        Lower::new(b, ir, helpers, objects, ptr, &mut sigs, &module).lower();
    }
    if std::env::var_os("LIZZARD_JIT_DUMP").is_some() {
        eprintln!("; {}\n{}", ir.name, ctx.func.display());
    }
    module.define_function(id, &mut ctx).map_err(|e| format!("{e:?}"))?;
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let code = module.get_finalized_function(id);
    // The module owns the code's memory; it stays mapped for the process.
    std::mem::forget(module);
    // SAFETY: the function was compiled with `Entry`'s signature.
    Ok(unsafe { std::mem::transmute::<*const u8, Entry>(code) })
}

#[derive(Default)]
struct Sigs {
    step: Option<SigRef>,
    catch: Option<SigRef>,
    safepoint: Option<SigRef>,
    /// Helpers taking the slots and N 32-bit operands (plus a pointer
    /// operand for some), returning a status.
    by_shape: std::collections::HashMap<(bool, usize, usize, bool), SigRef>,
}

struct Lower<'a> {
    b: FunctionBuilder<'a>,
    ir: &'a FuncIr,
    h: &'a Helpers,
    objects: &'a Objects<'a>,
    call_conv: cranelift_codegen::isa::CallConv,
    ptr: types::Type,
    sigs: &'a mut Sigs,
    /// One block per instruction.
    blocks: Vec<Block>,
    /// Per handler: stores the exception and jumps to the handler.
    catch_blocks: Vec<Block>,
    /// Leaves with `*pc` = its parameter and status `EXIT`.
    exit: Block,
    /// Leaves with `*pc` = its parameter and status `ERROR`.
    fail: Block,
    ctx: Value,
    regs: Value,
    pcp: Value,
    fuel: Variable,
    frontend: cranelift_codegen::isa::TargetFrontendConfig,
}

impl<'a> Lower<'a> {
    fn new(mut b: FunctionBuilder<'a>, ir: &'a FuncIr, h: &'a Helpers, objects: &'a Objects<'a>, ptr: types::Type, sigs: &'a mut Sigs, module: &JITModule) -> Lower<'a> {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let ps = b.block_params(entry).to_vec();
        let mk = |b: &mut FunctionBuilder, params: &[types::Type], ret: bool| {
            let mut s = Signature::new(module.isa().default_call_conv());
            s.params.extend(params.iter().map(|t| AbiParam::new(*t)));
            if ret {
                s.returns.push(AbiParam::new(types::I32));
            }
            b.import_signature(s)
        };
        sigs.step = Some(mk(&mut b, &[ptr, ptr, types::I32], true));
        sigs.catch = Some(mk(&mut b, &[ptr, ptr, types::I32], false));
        sigs.safepoint = Some(mk(&mut b, &[ptr], true));
        let blocks = (0..ir.code.len()).map(|_| b.create_block()).collect();
        let catch_blocks = (0..ir.handlers.len()).map(|_| b.create_block()).collect();
        let exit = b.create_block();
        b.append_block_param(exit, types::I32);
        let fail = b.create_block();
        b.append_block_param(fail, types::I32);
        let fuel = b.declare_var(types::I64);
        let frontend = module.target_config();
        let call_conv = module.isa().default_call_conv();
        let mut l = Lower { b, ir, h, objects, call_conv, ptr, sigs, blocks, catch_blocks, exit, fail, ctx: ps[0], regs: ps[1], pcp: ps[2], fuel, frontend };
        l.entry(ps[3]);
        l
    }

    /// Dispatch on `*pc`: 0, after a `yield`, or (with an exception
    /// pending) at the `yield`.
    fn entry(&mut self, pending: Value) {
        let f = FUEL;
        let fuel0 = self.b.ins().iconst(types::I64, f);
        self.b.def_var(self.fuel, fuel0);
        let pc = self.b.ins().load(types::I32, MemFlagsData::trusted(), self.pcp, 0);
        let resume = self.b.create_block();
        let throw = self.b.create_block();
        let bad = self.b.create_block();
        self.b.ins().brif(pending, throw, &[], resume, &[]);

        let yields: Vec<usize> = self.ir.code.iter().enumerate().filter(|(_, i)| matches!(i, Instr::Yield { .. } | Instr::YieldFrom { .. })).map(|(p, _)| p).collect();
        self.b.switch_to_block(resume);
        let mut sw = Switch::new();
        if !self.blocks.is_empty() {
            sw.set_entry(0, self.blocks[0]);
        }
        for &p in &yields {
            if p + 1 < self.blocks.len() {
                sw.set_entry(p as u128 + 1, self.blocks[p + 1]);
            }
        }
        sw.emit(&mut self.b, pc, bad);

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
        for i in 0..self.ir.code.len() {
            let blk = self.blocks[i];
            self.b.switch_to_block(blk);
            self.instr(i);
        }
        for (k, h) in self.ir.handlers.iter().enumerate() {
            let blk = self.catch_blocks[k];
            self.b.switch_to_block(blk);
            let exc = self.b.ins().iconst(types::I32, h.exc as i64);
            let f = self.b.ins().iconst(self.ptr, self.h.catch as i64);
            let sig = self.sigs.catch.unwrap();
            self.b.ins().call_indirect(sig, f, &[self.ctx, self.regs, exc]);
            self.b.ins().jump(self.blocks[h.target as usize], &[]);
        }
        for (blk, status) in [(self.exit, EXIT), (self.fail, ERROR)] {
            self.b.switch_to_block(blk);
            let pc = self.b.block_params(blk)[0];
            self.b.ins().store(MemFlagsData::trusted(), pc, self.pcp, 0);
            let s = self.b.ins().iconst(types::I32, status as i64);
            self.b.ins().return_(&[s]);
        }
        self.b.seal_all_blocks();
        let config = self.frontend;
        self.b.finalize(config);
    }

    /// An exception was raised by instruction `p`: its innermost handler,
    /// else leave the function.
    fn raise(&mut self, p: usize) {
        let at = p as u32;
        match self.ir.handlers.iter().rposition(|h| h.ranges.iter().any(|(s, e)| *s <= at && at < *e)) {
            Some(k) => {
                self.b.ins().jump(self.catch_blocks[k], &[]);
            }
            None => {
                let pc = self.b.ins().iconst(types::I32, p as i64);
                self.b.ins().jump(self.fail, &[pc.into()]);
            }
        }
    }

    fn slot(&self, r: u32) -> i32 {
        (r as i32) * 8
    }

    fn load(&mut self, ty: types::Type, r: u32) -> Value {
        let off = self.slot(r);
        self.b.ins().load(ty, MemFlagsData::trusted(), self.regs, off)
    }

    fn store(&mut self, v: Value, r: u32) {
        let off = self.slot(r);
        self.b.ins().store(MemFlagsData::trusted(), v, self.regs, off);
    }

    fn repr(&self, r: u32) -> Repr {
        self.ir.reprs[r as usize]
    }

    /// Continue at `target` from instruction `from`; a backward jump
    /// spends fuel and runs a safepoint when it is gone.
    fn goto(&mut self, from: usize, target: u32) {
        let t = target as usize;
        if t > from {
            self.b.ins().jump(self.blocks[t], &[]);
            return;
        }
        let fuel = self.b.use_var(self.fuel);
        let fuel = self.b.ins().iadd_imm_s(fuel, -1);
        self.b.def_var(self.fuel, fuel);
        let check = self.b.create_block();
        self.b.ins().brif(fuel, self.blocks[t], &[], check, &[]);
        self.b.switch_to_block(check);
        let f = self.b.ins().iconst(self.ptr, self.h.safepoint as i64);
        let sig = self.sigs.safepoint.unwrap();
        let call = self.b.ins().call_indirect(sig, f, &[self.ctx]);
        let status = self.b.inst_results(call)[0];
        let refuel = self.b.ins().iconst(types::I64, FUEL);
        self.b.def_var(self.fuel, refuel);
        let failed = self.b.create_block();
        self.b.ins().brif(status, failed, &[], self.blocks[t], &[]);
        self.b.switch_to_block(failed);
        self.raise(from);
    }

    /// A branch from instruction `from`: to `target` when `cond` is true.
    fn branch(&mut self, from: usize, cond: Value, target: u32, otherwise: u32) {
        let (yes, no) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(cond, yes, &[], no, &[]);
        self.b.switch_to_block(yes);
        self.goto(from, target);
        self.b.switch_to_block(no);
        self.goto(from, otherwise);
    }

    fn next(&mut self, i: usize) {
        if i + 1 < self.blocks.len() {
            self.goto(i, i as u32 + 1);
        } else {
            self.b.ins().trap(TrapCode::unwrap_user(2));
        }
    }

    fn instr(&mut self, i: usize) {
        let ir = self.ir;
        match &ir.code[i] {
            Instr::Jump { target } => self.goto(i, *target),
            Instr::JumpIfFalse { cond, target } => {
                let c = self.load(types::I64, *cond);
                self.branch(i, c, i as u32 + 1, *target);
            }
            Instr::JumpIfTrue { cond, target } => {
                let c = self.load(types::I64, *cond);
                self.branch(i, c, *target, i as u32 + 1);
            }
            Instr::I64Const { dst, value } => {
                let v = self.b.ins().iconst(types::I64, *value);
                self.store(v, *dst);
                self.next(i);
            }
            Instr::I64Lt { dst, a, b } => {
                let (x, y) = (self.load(types::I64, *a), self.load(types::I64, *b));
                let c = self.b.ins().icmp(IntCC::SignedLessThan, x, y);
                let c = self.b.ins().uextend(types::I64, c);
                self.store(c, *dst);
                self.next(i);
            }
            Instr::I64Add { dst, a, b } => {
                let (x, y) = (self.load(types::I64, *a), self.load(types::I64, *b));
                let v = self.b.ins().iadd(x, y);
                self.store(v, *dst);
                self.next(i);
            }
            Instr::Not { dst, src } => {
                let v = self.load(types::I64, *src);
                let v = self.b.ins().bxor_imm_u(v, 1);
                self.store(v, *dst);
                self.next(i);
            }
            Instr::F32Bin { dst, op, a, b } if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) => {
                let (x, y) = (self.load(types::F32, *a), self.load(types::F32, *b));
                let v = match op {
                    BinOp::Add => self.b.ins().fadd(x, y),
                    BinOp::Sub => self.b.ins().fsub(x, y),
                    BinOp::Mul => self.b.ins().fmul(x, y),
                    _ => self.b.ins().fdiv(x, y),
                };
                self.store(v, *dst);
                self.next(i);
            }
            Instr::Move { dst, src } if self.repr(*src) != Repr::Obj && self.repr(*dst) == self.repr(*src) => {
                let v = self.load(types::I64, *src);
                self.store(v, *dst);
                self.next(i);
            }
            Instr::Unbind { reg } if self.repr(*reg) != Repr::Obj => self.next(i),
            Instr::Const { dst, idx } => {
                let obj = self.objects.consts[*idx as usize];
                self.helper(i, self.h.konst, false, &[*dst], &[obj], After::Nothing);
            }
            Instr::Move { dst, src } if self.repr(*src) == Repr::Obj && self.repr(*dst) == Repr::Obj => self.helper(i, self.h.mov, false, &[*dst, *src], &[], After::Next),
            Instr::LoadAttr { dst, obj, name } => {
                let ptrs = [self.objects.names[*name as usize], self.objects.ics[i]];
                self.helper(i, self.h.load_attr, false, &[*dst, *obj], &ptrs, After::Next);
            }
            Instr::StoreAttr { obj, name, val } => {
                let ptrs = [self.objects.names[*name as usize], self.objects.ics[i]];
                self.helper(i, self.h.store_attr, false, &[*obj, *val], &ptrs, After::Next);
            }
            Instr::Return { src } => self.helper(i, self.h.ret, true, &[*src], &[], After::Exit),
            Instr::Call { .. } | Instr::CallMethod { .. } => self.helper(i, self.h.call, true, &[i as u32], &[], After::Next),
            Instr::ForIter { dst, iter, target } => self.helper(i, self.h.for_iter, true, &[*dst, *iter, i as u32], &[], After::Branch(*target)),
            Instr::Check { reg, check } => self.helper(i, self.h.check, true, &[*reg, *check, i as u32], &[], After::Next),
            Instr::Box { dst, src } if self.repr(*src) == Repr::I64 => self.helper(i, self.h.box_i64, false, &[*dst, *src], &[], After::Next),
            Instr::Truthy { dst, src } => self.helper(i, self.h.truthy, false, &[*dst, *src], &[], After::Next),
            Instr::BinOp { dst, op, a, b, inplace } => {
                let code = *op as u32 + if *inplace { 16 } else { 0 };
                self.helper(i, self.h.binop, false, &[*dst, *a, *b, code], &[], After::Next);
            }
            Instr::Compare { dst, op, a, b } if rich_code(*op).is_some() => {
                self.helper(i, self.h.compare, false, &[*dst, *a, *b, rich_code(*op).unwrap()], &[], After::Next);
            }
            Instr::CompareBool { dst, op, a, b } if rich_code(*op).is_some() => {
                self.helper(i, self.h.compare_bool, false, &[*dst, *a, *b, rich_code(*op).unwrap()], &[], After::Next);
            }
            Instr::Take { dst, src } if self.repr(*src) == Repr::Obj => self.helper(i, self.h.take, false, &[*dst, *src], &[], After::Next),
            Instr::Take { dst, src } => {
                let v = self.load(types::I64, *src);
                self.store(v, *dst);
                self.next(i);
            }
            instr => self.call_step(i, instr),
        }
    }

    /// Call a raw helper `f(regs, ops..., ptrs...)` for instruction `i`,
    /// then continue (or raise, when it returns a status).
    fn helper(&mut self, i: usize, f: usize, ctx: bool, ops: &[u32], ptrs: &[usize], after: After) {
        let status = !matches!(after, After::Nothing);
        let key = (ctx, ops.len(), ptrs.len(), status);
        let sig = match self.sigs.by_shape.get(&key) {
            Some(s) => *s,
            None => {
                let mut s = Signature::new(self.call_conv);
                if ctx {
                    s.params.push(AbiParam::new(self.ptr));
                }
                s.params.push(AbiParam::new(self.ptr));
                s.params.extend(ops.iter().map(|_| AbiParam::new(types::I32)));
                s.params.extend(ptrs.iter().map(|_| AbiParam::new(self.ptr)));
                if status {
                    s.returns.push(AbiParam::new(types::I32));
                }
                let r = self.b.import_signature(s);
                self.sigs.by_shape.insert(key, r);
                r
            }
        };
        let mut args: Vec<Value> = Vec::new();
        if ctx {
            args.push(self.ctx);
        }
        args.push(self.regs);
        for o in ops {
            args.push(self.b.ins().iconst(types::I32, *o as i64));
        }
        for p in ptrs {
            args.push(self.b.ins().iconst(self.ptr, *p as i64));
        }
        let fp = self.b.ins().iconst(self.ptr, f as i64);
        let call = self.b.ins().call_indirect(sig, fp, &args);
        if !status {
            self.next(i);
            return;
        }
        let st = self.b.inst_results(call)[0];
        let (other, ok) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(st, other, &[], ok, &[]);
        self.b.switch_to_block(ok);
        match after {
            After::Exit => {
                let pc = self.b.ins().iconst(types::I32, i as i64);
                self.b.ins().jump(self.exit, &[pc.into()]);
            }
            _ => self.next(i),
        }
        self.b.switch_to_block(other);
        if let After::Branch(target) = after {
            let (taken, raised) = (self.b.create_block(), self.b.create_block());
            let is_taken = self.b.ins().icmp_imm_s(IntCC::Equal, st, TAKEN as i64);
            self.b.ins().brif(is_taken, taken, &[], raised, &[]);
            self.b.switch_to_block(taken);
            self.goto(i, target);
            self.b.switch_to_block(raised);
        }
        self.raise(i);
    }

    /// Execute instruction `i` with the `step` helper and continue as it
    /// says.
    fn call_step(&mut self, i: usize, instr: &Instr) {
        let pc = self.b.ins().iconst(types::I32, i as i64);
        let f = self.b.ins().iconst(self.ptr, self.h.step as i64);
        let sig = self.sigs.step.unwrap();
        let call = self.b.ins().call_indirect(sig, f, &[self.ctx, self.regs, pc]);
        let status = self.b.inst_results(call)[0];
        let other = self.b.create_block();
        let next = self.b.create_block();
        self.b.ins().brif(status, other, &[], next, &[]);
        self.b.switch_to_block(next);
        self.next(i);

        self.b.switch_to_block(other);
        let raised = self.b.create_block();
        match instr {
            Instr::ForIter { target, .. } => {
                let taken = self.b.create_block();
                let is_taken = self.b.ins().icmp_imm_u(IntCC::Equal, status, TAKEN as i64);
                self.b.ins().brif(is_taken, taken, &[], raised, &[]);
                self.b.switch_to_block(taken);
                self.goto(i, *target);
            }
            Instr::Return { .. } | Instr::Yield { .. } | Instr::YieldFrom { .. } => {
                let exited = self.b.create_block();
                let is_exit = self.b.ins().icmp_imm_u(IntCC::Equal, status, EXIT as i64);
                self.b.ins().brif(is_exit, exited, &[], raised, &[]);
                self.b.switch_to_block(exited);
                let pc = self.b.ins().iconst(types::I32, i as i64);
                self.b.ins().jump(self.exit, &[pc.into()]);
            }
            _ => {
                self.b.ins().jump(raised, &[]);
            }
        }
        self.b.switch_to_block(raised);
        self.raise(i);
    }
}
