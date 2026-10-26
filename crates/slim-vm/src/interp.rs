//! VM state, frames and the dispatch loop.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::backend::interp::Interp;
use crate::backend::Exit;
use crate::code::{Code, Reg};
use crate::display;
use crate::object::{Body, FreeCheck, Obj};
use crate::ops::dispatch;
use crate::rc;
use crate::value::Value;

/// Marker: an exception is pending in the thread.
#[derive(Debug, Clone, Copy)]
pub struct Raised;

pub type VResult<T> = Result<T, Raised>;

pub struct ExcInfo {
    /// The exception instance (owned).
    pub value: Value,
    /// (function name, line), innermost first.
    pub tb: Vec<(String, u32)>,
    /// `raise X from Y` (owned) or UNINIT.
    pub cause: Value,
}

/// Builtin exception classes and traits.
pub struct Builtins {
    pub exception: Value,
    pub classes: Vec<(&'static str, Value)>,
}

impl Builtins {
    pub fn class(&self, name: &str) -> Value {
        self.classes
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("no builtin exception {name}"))
    }

    pub fn has_class(&self, name: &str) -> bool {
        self.classes.iter().any(|(n, _)| *n == name)
    }
}

pub struct Vm {
    pub globals: Box<[AtomicU64]>,
    pub global_names: Vec<String>,
    pub builtins: Builtins,
    pub next_class_id: AtomicU32,
    pub recursion_limit: u32,
}

impl Vm {
    pub fn global(&self, idx: u32) -> Value {
        Value(self.globals[idx as usize].load(Ordering::Acquire))
    }
}

pub struct Thread {
    pub vm: Arc<Vm>,
    pub exc: Option<ExcInfo>,
    pub depth: u32,
    pub rc_auto: bool,
    pub rc_threshold: usize,
    /// Traceback of the exception most recently caught (for bare `raise`).
    pub handled_tb: Vec<(String, u32)>,
}

#[repr(C)]
pub struct Frame {
    pub regs: *mut Value,
    pub code: *const Code,
    pub captures: *const Value,
    /// Next instruction.
    pub pc: u32,
    /// Instruction being executed.
    pub cur_pc: u32,
    pub ret: Value,
    /// Set by `yield`: the register receiving the sent value.
    pub yield_dst: u32,
    pub yielded: bool,
}

impl Frame {
    #[inline(always)]
    pub fn code(&self) -> &'static Code {
        // SAFETY: the code object outlives every frame executing it.
        unsafe { &*self.code }
    }

    #[inline(always)]
    pub fn reg(&self, r: Reg) -> Value {
        unsafe { *self.regs.add(r as usize) }
    }

    #[inline(always)]
    pub fn set(&mut self, r: Reg, v: Value) {
        unsafe { *self.regs.add(r as usize) = v }
    }

    /// Drops every owned register (frame exit or unwinding).
    pub fn drop_owned(&mut self) {
        let code = self.code();
        for (i, owned) in code.owned.iter().enumerate() {
            if *owned {
                let v = self.reg(i as Reg);
                v.decref();
                self.set(i as Reg, Value::UNINIT);
            }
        }
    }
}

impl Thread {
    pub fn new(vm: Arc<Vm>) -> Thread {
        Thread { vm, exc: None, depth: 0, rc_auto: true, rc_threshold: 10_000, handled_tb: Vec::new() }
    }

    /// Raises a builtin exception with a message.
    pub fn raise(&mut self, class: &str, msg: impl Into<String>) -> Raised {
        let cls = self.vm.builtins.class(class);
        let e = crate::helpers::new_exception(cls, Value::str(msg.into()));
        self.set_exception(e, Value::UNINIT);
        Raised
    }

    pub fn set_exception(&mut self, value: Value, cause: Value) {
        if let Some(old) = self.exc.take() {
            old.value.decref();
            old.cause.decref();
        }
        self.exc = Some(ExcInfo { value, tb: Vec::new(), cause });
    }

    /// Formats the pending exception as a Python-style traceback.
    pub fn format_exception(&self, filename: &str) -> String {
        let Some(e) = &self.exc else { return String::new() };
        let mut out = String::new();
        if e.cause != Value::UNINIT {
            out.push_str(&format!("{}: {}\n\n", e.cause.type_name(), display::exception_message(e.cause)));
            out.push_str("The above exception was the direct cause of the following exception:\n\n");
        }
        out.push_str("Traceback (most recent call last):\n");
        for (name, line) in e.tb.iter().rev() {
            out.push_str(&format!("  File \"{filename}\", line {line}, in {name}\n"));
        }
        let msg = display::exception_message(e.value);
        if msg.is_empty() {
            out.push_str(&e.value.type_name());
        } else {
            out.push_str(&format!("{}: {}", e.value.type_name(), msg));
        }
        out
    }

    /// Runs deferred decrements whose grace period passed and frees up to
    /// `limit` pending objects (running `__drop__` destructors). Returns the
    /// number of objects freed.
    pub fn rc_run(&mut self, limit: Option<usize>) -> usize {
        let h = rc::begin_run();
        for v in rc::take_deferred(h) {
            v.decref();
        }
        rc::free_retired(h);
        let mut freed = 0;
        while limit.is_none_or(|l| freed < l) {
            let Some(p) = rc::pop_ready(h) else { break };
            // SAFETY: the object was queued when its count reached zero and
            // its grace period passed, so no borrow taken before that can be
            // live; `free_check` catches borrows that revived it since.
            if unsafe { self.free_queued(p) } {
                freed += 1;
            }
        }
        freed
    }

    /// Frees a queued zero-count object (running its `__drop__`). Returns
    /// false if it was revived or resurrected instead.
    ///
    /// # Safety
    /// No borrow of the object taken before it reached zero may be live.
    unsafe fn free_queued(&mut self, p: *mut Obj) -> bool {
        let obj = unsafe { &mut *p };
        match obj.free_check() {
            FreeCheck::Free => {}
            FreeCheck::Requeue => {
                rc::requeue(p);
                return false;
            }
            FreeCheck::Skip => return false,
        }
        if let Body::Instance(inst) = &obj.body {
            if let Some(&m) = inst.class().methods.get("__drop__") {
                let me = Value(p as u64);
                // Hold a temporary reference while the destructor runs.
                obj.hold_for_drop();
                let saved = self.exc.take();
                match crate::helpers::call_simple(self, m, Some(me), &[]) {
                    Ok(v) => v.decref(),
                    Err(Raised) => {
                        let file = match m.body() {
                            Some(Body::Function(f)) => f.code.filename.to_string(),
                            _ => "<drop>".into(),
                        };
                        let msg = self.format_exception(&file);
                        eprintln!("Exception ignored in __drop__ of {}:\n{msg}", me.type_name());
                        if let Some(e) = self.exc.take() {
                            e.value.decref();
                            e.cause.decref();
                        }
                    }
                }
                self.exc = saved;
                if !obj.release_after_drop() {
                    // Resurrected: the destructor stored a new reference.
                    return false;
                }
            }
        }
        rc::set_cascade(true);
        obj.release_children();
        rc::set_cascade(false);
        #[cfg(debug_assertions)]
        let leak = Obj::poison(p);
        #[cfg(not(debug_assertions))]
        let leak = false;
        if !leak {
            // SAFETY: allocated by `Obj::alloc` as a Box.
            drop(unsafe { Box::from_raw(p) });
        }
        rc::note_free();
        true
    }

    /// Frees this thread's local objects before it exits: only their own
    /// thread may destroy them. No borrow is live any more and no other
    /// thread can reach a local object, so no grace period is needed.
    pub fn rc_free_local(&mut self) {
        loop {
            let (decs, objs) = rc::take_local();
            if decs.is_empty() && objs.is_empty() {
                break;
            }
            for v in decs {
                v.decref();
            }
            for p in objs {
                // SAFETY: see above; cascade entries were only reachable
                // through objects freed here.
                unsafe { self.free_queued(p) };
            }
        }
    }

    pub fn maybe_auto_rc(&mut self) {
        if self.rc_auto && rc::pending_count() > self.rc_threshold && rc::has_ready_work() {
            self.rc_run(Some(self.rc_threshold));
        }
    }
}

/// Executes a frame until it returns, yields or raises.
pub fn execute(th: &mut Thread, frame: &mut Frame) -> VResult<Value> {
    execute_inner(th, frame, false)
}

/// Like `execute`; with `raising`, first dispatches the pending exception as
/// if raised by the instruction at `frame.cur_pc` (generator `throw`).
pub fn execute_inner(th: &mut Thread, frame: &mut Frame, mut raising: bool) -> VResult<Value> {
    loop {
        let code = frame.code();
        let (pc, r) = if raising {
            raising = false;
            (frame.cur_pc, Err(Exit::Raise))
        } else {
            let pc = frame.pc;
            let ins = code.instrs[pc as usize];
            frame.cur_pc = pc;
            frame.pc = pc + 1;
            (pc, dispatch(&mut Interp { th: &mut *th, frame: &mut *frame }, ins))
        };
        match r {
            Ok(()) => {}
            Err(Exit::Return) | Err(Exit::Yield) => return Ok(frame.ret),
            Err(Exit::Raise) => {
                if let Some(h) = code.exc_table.iter().find(|e| e.start <= pc && pc < e.end) {
                    frame.pc = h.handler;
                    continue;
                }
                if let Some(e) = &mut th.exc {
                    e.tb.push((code.name.clone(), code.line_at(pc)));
                }
                frame.drop_owned();
                return Err(Raised);
            }
        }
    }
}

/// Runs a code object with prepared registers (parameters already in place).
pub fn run_code(th: &mut Thread, code: &Arc<Code>, regs: &mut [Value], captures: *const Value) -> VResult<Value> {
    if th.depth >= th.vm.recursion_limit {
        return Err(th.raise("RecursionError", "maximum recursion depth exceeded"));
    }
    th.depth += 1;
    rc::safepoint();
    let mut frame = Frame {
        regs: regs.as_mut_ptr(),
        code: Arc::as_ptr(code),
        captures,
        pc: 0,
        cur_pc: 0,
        ret: Value::UNIT,
        yield_dst: u32::MAX,
        yielded: false,
    };
    let r = execute(th, &mut frame);
    th.depth -= 1;
    if r.is_ok() {
        th.maybe_auto_rc();
    }
    r
}

