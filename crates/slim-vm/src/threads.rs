//! Threads, locks and channels (milestone 6).

use crate::interp::{Thread, VResult};
use crate::value::Value;

pub fn spawn(th: &mut Thread, args: &[Value]) -> VResult<Value> {
    for a in &args[1..] {
        a.decref();
    }
    Err(th.raise("NotImplementedError", "threads are not implemented yet"))
}

pub fn new_lock() -> Value {
    Value::UNIT
}

pub fn new_channel() -> Value {
    Value::UNIT
}
