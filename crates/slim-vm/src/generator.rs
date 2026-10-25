//! Generators: a suspended frame owned by a heap object.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::code::Reg;
use crate::interp::{execute_inner, Frame, Raised, Thread, VResult};
use crate::object::{Body, Obj};
use crate::value::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Created,
    Suspended,
    Done,
}

pub struct GenState {
    pub regs: Vec<Value>,
    /// Resume point (instruction after the `yield`).
    pub pc: u32,
    pub yield_dst: Option<Reg>,
    pub status: Status,
    /// Value of the generator's `return` (owned), for `yield from`.
    pub retval: Value,
}

pub struct Generator {
    /// The generator function (owned reference: keeps code and captures alive).
    pub func: Value,
    pub state: Mutex<GenState>,
    pub running: AtomicBool,
}

impl Generator {
    pub fn name(&self) -> String {
        match self.func.body() {
            Some(Body::Function(f)) => f.code.name.clone(),
            _ => "?".into(),
        }
    }

    pub(crate) fn release(&mut self) {
        let st = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if st.status != Status::Done {
            if let Some(Body::Function(f)) = self.func.body() {
                for (i, owned) in f.code.owned.iter().enumerate() {
                    if *owned {
                        st.regs[i].decref();
                    }
                }
            }
        }
        st.retval.decref();
        self.func.decref();
    }
}

pub fn create(_th: &mut Thread, f: Value, regs: Vec<Value>) -> VResult<Value> {
    f.incref();
    let g = Generator {
        func: f,
        state: Mutex::new(GenState { regs, pc: 0, yield_dst: None, status: Status::Created, retval: Value::UNIT }),
        running: AtomicBool::new(false),
    };
    Ok(Obj::alloc(Body::Generator(g)))
}

pub fn as_generator(v: Value) -> Option<&'static Generator> {
    match v.body() {
        Some(Body::Generator(g)) => Some(g),
        _ => None,
    }
}

/// Resumes a generator with an owned `sent` value, or by throwing `throw`
/// (owned) at the suspension point. Returns the next yielded value (owned),
/// or `None` when the generator finished.
pub fn resume(th: &mut Thread, gv: Value, sent: Value, throw: Option<Value>) -> VResult<Option<Value>> {
    let g = as_generator(gv).expect("not a generator");
    if g.running.swap(true, Ordering::Acquire) {
        sent.decref();
        if let Some(t) = throw {
            t.decref();
        }
        return Err(th.raise("ValueError", "generator already executing"));
    }
    let r = resume_inner(th, g, sent, throw);
    g.running.store(false, Ordering::Release);
    r
}

fn resume_inner(th: &mut Thread, g: &Generator, sent: Value, throw: Option<Value>) -> VResult<Option<Value>> {
    let Some(Body::Function(func)) = g.func.body() else { unreachable!() };
    let (mut regs, pc, status, yield_dst) = {
        let mut st = g.state.lock().unwrap();
        (std::mem::take(&mut st.regs), st.pc, st.status, st.yield_dst)
    };
    let put_back = |regs: Vec<Value>, status: Status| {
        let mut st = g.state.lock().unwrap();
        st.regs = regs;
        st.status = status;
    };
    match status {
        Status::Done => {
            put_back(regs, Status::Done);
            sent.decref();
            if let Some(t) = throw {
                th.set_exception(t, Value::UNINIT);
                return Err(Raised);
            }
            return Ok(None);
        }
        Status::Created => {
            if let Some(t) = throw {
                // Never started: drop the frame and raise immediately.
                for (i, owned) in func.code.owned.iter().enumerate() {
                    if *owned {
                        regs[i].decref();
                    }
                }
                put_back(Vec::new(), Status::Done);
                sent.decref();
                th.set_exception(t, Value::UNINIT);
                return Err(Raised);
            }
            if sent != Value::UNIT && sent != Value::NONE {
                put_back(regs, Status::Created);
                sent.decref();
                return Err(th.raise("TypeError", "can't send a value to a just-started generator; send None first"));
            }
        }
        Status::Suspended => {
            if let Some(dst) = yield_dst {
                regs[dst as usize].decref();
                regs[dst as usize] = sent;
            } else {
                sent.decref();
            }
        }
    }
    if th.depth >= th.vm.recursion_limit {
        put_back(regs, status);
        return Err(th.raise("RecursionError", "maximum recursion depth exceeded"));
    }
    th.depth += 1;
    let mut frame = Frame {
        regs: regs.as_mut_ptr(),
        code: std::sync::Arc::as_ptr(&func.code),
        captures: func.captures.as_ptr(),
        pc,
        cur_pc: pc.saturating_sub(1),
        ret: Value::UNIT,
        yield_dst: u32::MAX,
        yielded: false,
    };
    let raising = match throw {
        Some(t) => {
            th.set_exception(t, Value::UNINIT);
            true
        }
        None => false,
    };
    let r = execute_inner(th, &mut frame, raising);
    th.depth -= 1;
    match r {
        Ok(v) if frame.yielded => {
            let mut st = g.state.lock().unwrap();
            st.regs = regs;
            st.pc = frame.pc;
            st.yield_dst = Some(frame.yield_dst as Reg);
            st.status = Status::Suspended;
            Ok(Some(v))
        }
        Ok(v) => {
            let mut st = g.state.lock().unwrap();
            st.regs = Vec::new();
            st.status = Status::Done;
            st.retval.decref();
            st.retval = v;
            Ok(None)
        }
        Err(e) => {
            put_back(Vec::new(), Status::Done);
            Err(e)
        }
    }
}

/// `yield` inside the generator frame: moves the value out and suspends.
pub fn yield_op(_th: &mut Thread, frame: &mut Frame, dst: Reg, src: Reg) -> u32 {
    let v = frame.reg(src);
    frame.set(src, Value::MOVED);
    frame.ret = v;
    frame.yield_dst = dst as u32;
    frame.yielded = true;
    3
}

/// Takes the return value of a finished generator (owned).
pub fn take_retval(gv: Value) -> Value {
    match as_generator(gv) {
        Some(g) => std::mem::replace(&mut g.state.lock().unwrap().retval, Value::UNIT),
        None => Value::UNIT,
    }
}
