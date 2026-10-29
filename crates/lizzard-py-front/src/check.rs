//! Type checking of a lizzard module and lowering of its functions to IR.
//!
//! One pass does both: every expression is typed and lowered together.
//! Top-level statements (run by CPython) are checked the same way into a
//! throwaway function.

use std::collections::HashMap;

use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_text_size::Ranged;

use crate::db::*;
use crate::ir::*;

#[derive(Clone)]
struct Val {
    reg: Reg,
    ty: Ty,
    repr: Repr,
}

#[derive(Clone)]
struct Local {
    reg: Reg,
    ty: Ty,
    repr: Repr,
}

/// A failed call match: a message, or a mismatching argument.
enum Mismatch {
    Arity(String),
    Arg { name: String, expected: Ty, got: Ty },
}

struct Lower<'a> {
    db: &'a mut Db,
    m: ModId,
    file: FileId,
    toplevel: bool,
    fname: String,
    code: Vec<Instr>,
    lines: Vec<u32>,
    reprs: Vec<Repr>,
    locals: HashMap<String, Local>,
    consts: Vec<Const>,
    names: Vec<String>,
    checks: Vec<CheckIr>,
    ret: Ty,
    line: u32,
}

const BUILTIN_FNS: &[&str] = &["print", "len", "range"];

pub fn check_module(db: &mut Db, m: ModId, body: &[Stmt]) -> Vec<FuncIr> {
    let file = db.modules[m].file.expect("lizzard module without file");
    let mut funcs = Vec::new();
    let mut top = Lower::new(db, m, file, true, "<module>".into(), Ty::Any);
    for s in body {
        match s {
            Stmt::FunctionDef(d) => {
                if let Some(f) = top.function(d) {
                    funcs.push(f);
                }
            }
            s => top.stmt(s),
        }
    }
    funcs
}

impl<'a> Lower<'a> {
    fn new(db: &'a mut Db, m: ModId, file: FileId, toplevel: bool, fname: String, ret: Ty) -> Lower<'a> {
        Lower {
            db,
            m,
            file,
            toplevel,
            fname,
            code: Vec::new(),
            lines: Vec::new(),
            reprs: Vec::new(),
            locals: HashMap::new(),
            consts: Vec::new(),
            names: Vec::new(),
            checks: Vec::new(),
            ret,
            line: 0,
        }
    }

    // ---- helpers ----

    fn err(&mut self, off: u32, msg: impl Into<String>) {
        self.db.diags.push((self.file, off, msg.into()));
    }

    fn at(&mut self, node: &impl Ranged) -> u32 {
        let off: u32 = node.range().start().into();
        self.line = self.db.files[self.file].line_col(off).0;
        off
    }

    fn src(&self, node: &impl Ranged) -> String {
        let r = node.range();
        let s = &self.db.files[self.file].text[usize::from(r.start())..usize::from(r.end())];
        if s.len() > 40 {
            format!("{}...", &s[..s.char_indices().nth(37).map_or(s.len(), |(i, _)| i)])
        } else {
            s.to_string()
        }
    }

    fn reg(&mut self, repr: Repr) -> Reg {
        self.reprs.push(repr);
        (self.reprs.len() - 1) as Reg
    }

    fn emit(&mut self, i: Instr) -> usize {
        self.code.push(i);
        self.lines.push(self.line);
        self.code.len() - 1
    }

    fn here(&self) -> u32 {
        self.code.len() as u32
    }

    fn patch(&mut self, at: usize, to: u32) {
        match &mut self.code[at] {
            Instr::Jump { target } | Instr::JumpIfFalse { target, .. } | Instr::JumpIfTrue { target, .. } => *target = to,
            _ => unreachable!("patching a non-jump"),
        }
    }

    fn name(&mut self, s: &str) -> u32 {
        if let Some(i) = self.names.iter().position(|n| n == s) {
            return i as u32;
        }
        self.names.push(s.to_string());
        (self.names.len() - 1) as u32
    }

    fn konst(&mut self, c: Const, ty: Ty) -> Val {
        self.consts.push(c);
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::Const { dst, idx: (self.consts.len() - 1) as u32 });
        Val { reg: dst, ty, repr: Repr::Obj }
    }

    fn any(&mut self) -> Val {
        self.konst(Const::None, Ty::Any)
    }

    fn show(&self, t: &Ty) -> String {
        self.db.display(t)
    }

    fn assignable(&mut self, src: &Ty, dst: &Ty) -> bool {
        self.db.assign(src, dst, &mut Subst::new(), &mut false)
    }

    fn is_f32(&mut self, t: &Ty) -> bool {
        self.db.runtime_class(t).as_deref() == Some("numpy.float32")
    }

    /// `ndarray[tuple[int], dtype[float32]]`: element access is inline.
    fn vec_f32(&mut self, t: &Ty) -> bool {
        let Ty::Inst(c, args) = t else { return false };
        if self.db.qualname(*c) != "numpy.ndarray" || args.len() != 2 || !matches!(&args[0], Ty::Tuple(v) if v.as_slice() == [Ty::Int]) {
            return false;
        }
        match &args[1] {
            Ty::Inst(d, dargs) if self.db.qualname(*d) == "numpy.dtype" && dargs.len() == 1 => self.is_f32(&dargs[0]),
            _ => false,
        }
    }

    fn rtcheck(&mut self, t: &Ty) -> RtCheck {
        match t {
            Ty::Int => RtCheck::Int,
            Ty::Float => RtCheck::Float,
            Ty::Bool => RtCheck::Bool,
            Ty::Str => RtCheck::Str,
            Ty::None => RtCheck::None,
            Ty::Tuple(_) | Ty::TupleVar(_) => RtCheck::Tuple,
            Ty::Exc(n) => RtCheck::Instance(format!("builtins.{n}")),
            Ty::Inst(c, args) if self.db.qualname(*c) == "numpy.ndarray" => {
                let scalar = match args.get(1) {
                    Some(Ty::Inst(d, dargs)) if self.db.qualname(*d) == "numpy.dtype" => dargs.first().cloned().and_then(|s| self.db.runtime_class(&s)),
                    _ => None,
                };
                let ndim = match args.first() {
                    Some(Ty::Tuple(v)) => Some(v.len() as u32),
                    _ => None,
                };
                RtCheck::NdArray { scalar, ndim }
            }
            Ty::Inst(c, _) => RtCheck::Instance(self.db.runtime_class(t).unwrap_or_else(|| self.db.qualname(*c).to_string())),
            _ => RtCheck::Any,
        }
    }

    /// Check an object produced by Python code against its static type.
    fn check(&mut self, v: &Val, what: String, origin: Option<String>) {
        if v.repr != Repr::Obj {
            return;
        }
        let check = self.rtcheck(&v.ty);
        if check == RtCheck::Any {
            return;
        }
        let expected = self.show(&v.ty);
        self.checks.push(CheckIr { check, what, expected, origin });
        let idx = (self.checks.len() - 1) as u32;
        self.emit(Instr::Check { reg: v.reg, check: idx });
    }

    fn origin(&self, file: FileId, off: u32) -> Option<String> {
        (file != self.file).then(|| self.db.decl_origin(file, off))
    }

    fn convert(&mut self, v: &Val, to: Repr) -> Reg {
        if v.repr == to {
            return v.reg;
        }
        let dst = self.reg(to);
        match (v.repr, to) {
            (_, Repr::Obj) => self.emit(Instr::Box { dst, src: v.reg }),
            (Repr::Obj, Repr::Bool) => self.emit(Instr::Truthy { dst, src: v.reg }),
            (Repr::Obj, _) => self.emit(Instr::Unbox { dst, src: v.reg }),
            (_, _) => {
                let o = self.convert(v, Repr::Obj);
                let ov = Val { reg: o, ty: v.ty.clone(), repr: Repr::Obj };
                return self.convert(&ov, to);
            }
        };
        dst
    }

    fn obj(&mut self, v: &Val) -> Reg {
        self.convert(v, Repr::Obj)
    }

    // ---- functions ----

    fn function(&mut self, d: &ast::StmtFunctionDef) -> Option<FuncIr> {
        let off = self.at(d);
        if d.is_async || !d.decorator_list.is_empty() || d.type_params.is_some() {
            self.err(off, "async, decorated and generic functions are not supported by lizzard yet");
            return None;
        }
        let Some(Sym::Func(f)) = self.db.lookup_in(self.m, d.name.as_str()) else {
            return None;
        };
        let sigs = self.db.sigs(f);
        if sigs.len() != 1 || self.db.funcs[f].name != d.name.as_str() {
            self.err(off, format!("`{}` is defined more than once", d.name.as_str()));
            return None;
        }
        let sig = sigs[0].clone();
        let line = self.db.files[self.file].line_col(off).0;
        let path = self.db.files[self.file].path.clone();
        let mut l = Lower::new(self.db, self.m, self.file, false, d.name.as_str().to_string(), sig.ret.clone());
        l.at(d);
        let mut params = Vec::new();
        for p in &sig.params {
            if !matches!(p.kind, PKind::PosOnly | PKind::PosOrKw) || p.default {
                l.err(off, format!("parameter `{}`: only plain positional parameters are supported by lizzard yet", p.name));
            }
            let reg = l.reg(Repr::Obj);
            l.locals.insert(p.name.clone(), Local { reg, ty: p.ty.clone(), repr: Repr::Obj });
            let expected = l.show(&p.ty);
            let check = l.rtcheck(&p.ty);
            l.checks.push(CheckIr { check, what: format!("argument '{}'", p.name), expected, origin: None });
            params.push(ParamIr { name: p.name.clone(), check: (l.checks.len() - 1) as u32 });
        }
        for s in &d.body {
            l.stmt(s);
        }
        if !always_exits(&d.body) {
            if matches!(l.ret, Ty::None | Ty::Any) {
                let none = l.konst(Const::None, Ty::None);
                l.emit(Instr::Return { src: none.reg });
            } else {
                let off = d.name.range().start().into();
                let want = l.show(&l.ret);
                l.err(off, format!("`{}` may end without returning a value (declared to return {want})", d.name.as_str()));
            }
        }
        Some(FuncIr {
            name: l.fname,
            file: path,
            line,
            params,
            reprs: l.reprs,
            code: l.code,
            lines: l.lines,
            consts: l.consts,
            names: l.names,
            checks: l.checks,
        })
    }

    // ---- statements ----

    fn stmts(&mut self, body: &[Stmt]) {
        for s in body {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        let off = self.at(s);
        match s {
            Stmt::Expr(e) => {
                self.expr(&e.value);
            }
            Stmt::Assign(a) => {
                let [target] = a.targets.as_slice() else {
                    return self.err(off, "multiple assignment targets are not supported by lizzard yet");
                };
                match target {
                    Expr::Name(n) => {
                        let v = self.expr(&a.value);
                        self.assign_local(n.id.as_str(), v, None, off);
                    }
                    Expr::Subscript(sub) => {
                        let v = self.expr(&a.value);
                        self.store_item(sub, v);
                    }
                    _ => self.err(off, "this assignment target is not supported by lizzard yet"),
                }
            }
            Stmt::AnnAssign(a) => {
                let Expr::Name(n) = &*a.target else {
                    return self.err(off, "this assignment target is not supported by lizzard yet");
                };
                let ty = self.db.eval_type(&a.annotation, &Scope { m: self.m, tparams: &[], self_ty: None });
                match &a.value {
                    Some(v) => {
                        let v = self.expr(v);
                        self.assign_local(n.id.as_str(), v, Some(ty), off);
                    }
                    None => self.err(off, "a declaration without a value is not supported by lizzard yet"),
                }
            }
            Stmt::If(i) => self.if_stmt(&i.test, &i.body, &i.elif_else_clauses),
            Stmt::For(f) => self.for_stmt(f),
            Stmt::Raise(r) => {
                if r.cause.is_some() {
                    self.err(off, "`raise ... from` is not supported by lizzard yet");
                }
                let Some(exc) = &r.exc else {
                    return self.err(off, "a bare `raise` is not supported by lizzard yet");
                };
                let v = self.expr(exc);
                let ok = matches!(&v.ty, Ty::Any | Ty::Exc(_) | Ty::Inst(..)) || matches!(&v.ty, Ty::Type(t) if matches!(**t, Ty::Exc(_)));
                if !ok {
                    let t = self.show(&v.ty);
                    self.err(off, format!("exceptions must derive from BaseException, got {t}"));
                }
                let r = self.obj(&v);
                self.emit(Instr::Raise { exc: r });
            }
            Stmt::Return(r) if !self.toplevel => {
                let v = match &r.value {
                    Some(e) => self.expr(e),
                    None => self.konst(Const::None, Ty::None),
                };
                let want = self.ret.clone();
                let v = self.coerce(v, &want, off, |l, got, want| format!("`{}` returns {want}, but this returns {got}", l.fname));
                let r = self.obj(&v);
                self.emit(Instr::Return { src: r });
            }
            Stmt::Pass(_) => {}
            Stmt::Import(_) | Stmt::ImportFrom(_) if self.toplevel => self.import(s),
            Stmt::TypeAlias(t) if self.toplevel => {
                if let Expr::Name(n) = &*t.name {
                    self.db.lookup_in(self.m, n.id.as_str());
                }
            }
            Stmt::ClassDef(_) => self.err(off, "classes are not supported by lizzard yet"),
            Stmt::FunctionDef(_) => self.err(off, "nested functions are not supported by lizzard yet"),
            _ => self.err(off, "this statement is not supported by lizzard yet"),
        }
    }

    fn import(&mut self, s: &Stmt) {
        let off = self.at(s);
        match s {
            Stmt::Import(i) => {
                for a in &i.names {
                    if self.db.load_module(a.name.as_str()).is_none() {
                        self.err(off, format!("cannot find module `{}`", a.name.as_str()));
                    }
                }
            }
            Stmt::ImportFrom(i) => {
                let Some(module) = i.module.as_ref().filter(|_| i.level == 0) else {
                    return self.err(off, "relative imports are not supported by lizzard yet");
                };
                match self.db.load_module(module.as_str()) {
                    None => self.err(off, format!("cannot find module `{}`", module.as_str())),
                    Some(mm) => {
                        for a in &i.names {
                            if self.db.module_attr(mm, a.name.as_str()).is_none() {
                                self.err(off, format!("module `{}` has no attribute `{}`", module.as_str(), a.name.as_str()));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// `v` as a value of type `want`: an error if it cannot be, a runtime
    /// check if it is only known as `Any`.
    fn coerce(&mut self, v: Val, want: &Ty, off: u32, msg: impl FnOnce(&Self, String, String) -> String) -> Val {
        if !self.assignable(&v.ty, want) {
            let (got, want) = (self.show(&v.ty), self.show(want));
            let m = msg(self, got, want);
            self.err(off, m);
            return v;
        }
        if matches!(v.ty, Ty::Any) && !matches!(want, Ty::Any) {
            let r = self.obj(&v);
            let nv = Val { reg: r, ty: want.clone(), repr: Repr::Obj };
            let what = "value".to_string();
            self.check(&nv, what, None);
            return nv;
        }
        v
    }

    fn assign_local(&mut self, name: &str, v: Val, ann: Option<Ty>, off: u32) {
        if let Some(l) = self.locals.get(name).cloned() {
            if let Some(a) = &ann {
                if *a != l.ty {
                    let (old, new) = (self.show(&l.ty), self.show(a));
                    return self.err(off, format!("`{name}` was declared as {old}, cannot redeclare it as {new}"));
                }
            }
            let v = self.coerce(v, &l.ty, off, |_, got, want| format!("cannot assign {got} to `{name}` of type {want}"));
            let r = self.convert(&v, l.repr);
            self.emit(Instr::Move { dst: l.reg, src: r });
            return;
        }
        let v = match &ann {
            Some(t) => {
                let t = t.clone();
                let v = self.coerce(v, &t, off, |_, got, want| format!("cannot assign {got} to `{name}` of type {want}"));
                Val { ty: t, ..v }
            }
            None => v,
        };
        let reg = self.reg(v.repr);
        self.emit(Instr::Move { dst: reg, src: v.reg });
        self.locals.insert(name.to_string(), Local { reg, ty: v.ty, repr: v.repr });
    }

    fn if_stmt(&mut self, test: &Expr, body: &[Stmt], rest: &[ast::ElifElseClause]) {
        let c = self.expr(test);
        let b = self.convert(&c, Repr::Bool);
        let skip = self.emit(Instr::JumpIfFalse { cond: b, target: 0 });
        self.stmts(body);
        if rest.is_empty() {
            let h = self.here();
            self.patch(skip, h);
            return;
        }
        let end = self.emit(Instr::Jump { target: 0 });
        let h = self.here();
        self.patch(skip, h);
        match &rest[0].test {
            Some(t) => self.if_stmt(t, &rest[0].body, &rest[1..]),
            None => self.stmts(&rest[0].body),
        }
        let h = self.here();
        self.patch(end, h);
    }

    fn for_stmt(&mut self, f: &ast::StmtFor) {
        let off = self.at(f);
        let Expr::Name(target) = &*f.target else {
            return self.err(off, "only `for name in range(...)` is supported by lizzard yet");
        };
        let range_args = match &*f.iter {
            Expr::Call(c) if is_builtin_call(c, "range") && c.arguments.keywords.is_empty() && !self.locals.contains_key("range") => &c.arguments.args,
            _ => return self.err(off, "only `for name in range(...)` is supported by lizzard yet"),
        };
        if f.is_async || !f.orelse.is_empty() || range_args.is_empty() || range_args.len() > 2 {
            return self.err(off, "only `for name in range(stop)` and `range(start, stop)` are supported by lizzard yet");
        }
        let mut bounds = Vec::new();
        for a in range_args.iter() {
            let v = self.expr(a);
            let v = self.coerce(v, &Ty::Int, a.range().start().into(), |_, got, _| format!("range() needs int arguments, got {got}"));
            bounds.push(self.convert(&v, Repr::I64));
        }
        let ctr = self.reg(Repr::I64);
        match bounds.as_slice() {
            [_] => {
                self.emit(Instr::I64Const { dst: ctr, value: 0 });
            }
            [start, _] => {
                self.emit(Instr::Move { dst: ctr, src: *start });
            }
            _ => unreachable!(),
        }
        let stop = *bounds.last().unwrap();
        let one = self.reg(Repr::I64);
        self.emit(Instr::I64Const { dst: one, value: 1 });
        let top = self.here();
        let c = self.reg(Repr::Bool);
        self.emit(Instr::I64Lt { dst: c, a: ctr, b: stop });
        let exit = self.emit(Instr::JumpIfFalse { cond: c, target: 0 });
        self.assign_local(target.id.as_str(), Val { reg: ctr, ty: Ty::Int, repr: Repr::I64 }, None, off);
        self.stmts(&f.body);
        self.emit(Instr::I64Add { dst: ctr, a: ctr, b: one });
        self.emit(Instr::Jump { target: top });
        let h = self.here();
        self.patch(exit, h);
    }

    // ---- expressions ----

    fn expr(&mut self, e: &Expr) -> Val {
        let off = self.at(e);
        match e {
            Expr::NumberLiteral(n) => match &n.value {
                ast::Number::Int(i) => match i.as_i64() {
                    Some(v) => self.konst(Const::Int(v), Ty::Int),
                    None => self.konst(Const::BigInt(i.to_string()), Ty::Int),
                },
                ast::Number::Float(f) => self.konst(Const::Float(*f), Ty::Float),
                ast::Number::Complex { .. } => {
                    self.err(off, "complex numbers are not supported by lizzard yet");
                    self.any()
                }
            },
            Expr::StringLiteral(s) => self.konst(Const::Str(s.value.to_str().to_string()), Ty::Str),
            Expr::BooleanLiteral(b) => self.konst(Const::Bool(b.value), Ty::Bool),
            Expr::NoneLiteral(_) => self.konst(Const::None, Ty::None),
            Expr::Name(n) => self.load_name(n.id.as_str(), off),
            Expr::Attribute(a) => self.attr(a, e),
            Expr::Call(c) => self.call(c),
            Expr::Subscript(s) => self.load_item(s, e),
            Expr::BinOp(b) => self.binop(b, e),
            Expr::Compare(c) => self.compare(c, e),
            Expr::BoolOp(b) => self.boolop(b),
            Expr::UnaryOp(u) => match u.op {
                ast::UnaryOp::Not => {
                    let v = self.expr(&u.operand);
                    let b = self.convert(&v, Repr::Bool);
                    let dst = self.reg(Repr::Bool);
                    self.emit(Instr::Not { dst, src: b });
                    Val { reg: dst, ty: Ty::Bool, repr: Repr::Bool }
                }
                ast::UnaryOp::USub => match &*u.operand {
                    Expr::NumberLiteral(n) => match &n.value {
                        ast::Number::Int(i) if i.as_i64().is_some() => self.konst(Const::Int(-i.as_i64().unwrap()), Ty::Int),
                        ast::Number::Float(f) => self.konst(Const::Float(-f), Ty::Float),
                        _ => {
                            self.err(off, "this negation is not supported by lizzard yet");
                            self.any()
                        }
                    },
                    _ => {
                        self.err(off, "unary minus is only supported on literals by lizzard yet");
                        self.any()
                    }
                },
                _ => {
                    self.err(off, "this operator is not supported by lizzard yet");
                    self.any()
                }
            },
            _ => {
                self.err(off, "this expression is not supported by lizzard yet");
                self.any()
            }
        }
    }

    fn sym_value_ty(&mut self, s: Sym) -> Ty {
        match s {
            Sym::Module(mm) => Ty::Module(mm),
            Sym::Func(f) => Ty::Func(FuncRef::Def(f)),
            Sym::Class(c) => {
                let n = self.db.class_params(c).len();
                Ty::Type(Box::new(Ty::Inst(c, vec![Ty::Any; n])))
            }
            Sym::Alias(a) => Ty::Type(Box::new(self.db.aliases[a].target.clone())),
            Sym::Builtin(t) => Ty::Type(Box::new(t)),
            Sym::Value(t) => t,
            _ => Ty::Any,
        }
    }

    fn load_name(&mut self, name: &str, off: u32) -> Val {
        if let Some(l) = self.locals.get(name) {
            return Val { reg: l.reg, ty: l.ty.clone(), repr: l.repr };
        }
        let ty = match self.db.lookup_in(self.m, name) {
            Some(Sym::UserVar) if self.toplevel => {
                self.err(off, format!("`{name}` is used before it is assigned"));
                Ty::Any
            }
            Some(Sym::UserVar) => {
                self.err(off, format!("lizzard functions cannot use the module variable `{name}` yet"));
                Ty::Any
            }
            Some(s) => self.sym_value_ty(s),
            None => match name {
                n if BUILTIN_FNS.contains(&n) => Ty::Func(FuncRef::Builtin(BUILTIN_FNS.iter().find(|b| **b == n).unwrap())),
                n if EXCEPTIONS.contains(&n) => Ty::Type(Box::new(Ty::Exc(n.to_string()))),
                "int" => Ty::Type(Box::new(Ty::Int)),
                "float" => Ty::Type(Box::new(Ty::Float)),
                "str" => Ty::Type(Box::new(Ty::Str)),
                "bool" => Ty::Type(Box::new(Ty::Bool)),
                "__name__" => Ty::Str,
                _ => {
                    self.err(off, format!("name `{name}` is not defined"));
                    Ty::Any
                }
            },
        };
        let dst = self.reg(Repr::Obj);
        let n = self.name(name);
        self.emit(Instr::LoadGlobal { dst, name: n });
        Val { reg: dst, ty, repr: Repr::Obj }
    }

    fn attr(&mut self, a: &ast::ExprAttribute, e: &Expr) -> Val {
        let off: u32 = a.attr.range().start().into();
        let obj = self.expr(&a.value);
        let name = a.attr.as_str();
        let mut origin = None;
        let ty = match &obj.ty {
            Ty::Module(mm) => match self.db.module_attr(*mm, name) {
                Some(s) => self.sym_value_ty(s),
                None => {
                    let mn = self.db.modules[*mm].name.clone();
                    self.err(off, format!("module `{mn}` has no attribute `{name}`"));
                    Ty::Any
                }
            },
            Ty::Inst(c, _) => match self.db.find_member(*c, name) {
                Some((_, Member::Property(f))) => match self.resolve(f, Some(obj.ty.clone()), &[], &[]) {
                    Ok((t, file, doff)) => {
                        origin = self.origin(file, doff);
                        t
                    }
                    Err(_) => Ty::Any,
                },
                Some((_, Member::Method(f))) => Ty::Func(FuncRef::Bound(f, Box::new(obj.ty.clone()))),
                Some((owner, Member::Attr(t))) => {
                    let ps = self.db.class_params(owner);
                    let dst = Ty::Inst(owner, ps.iter().map(|p| Ty::Var(*p)).collect());
                    let mut s = Subst::new();
                    self.db.assign(&obj.ty, &dst, &mut s, &mut false);
                    erase_vars(&subst(&t, &s))
                }
                None => {
                    let t = self.show(&obj.ty);
                    self.err(off, format!("{t} has no attribute `{name}`"));
                    Ty::Any
                }
            },
            Ty::Any => Ty::Any,
            t => {
                let t = self.show(t);
                self.err(off, format!("attribute `{name}` of {t} is not supported by lizzard yet"));
                Ty::Any
            }
        };
        let o = self.obj(&obj);
        let dst = self.reg(Repr::Obj);
        let n = self.name(name);
        self.emit(Instr::LoadAttr { dst, obj: o, name: n });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        if !matches!(v.ty, Ty::Func(_) | Ty::Type(_) | Ty::Module(_)) {
            let what = self.src(e);
            self.check(&v, what, origin);
        }
        v
    }

    fn call(&mut self, c: &ast::ExprCall) -> Val {
        let off: u32 = c.range().start().into();
        if is_builtin_call(c, "len") && !self.locals.contains_key("len") && self.db.lookup_in(self.m, "len").is_none() {
            let [arg] = &c.arguments.args[..] else {
                self.err(off, "len() takes exactly one argument");
                return self.any();
            };
            let v = self.expr(arg);
            let ok = match &v.ty {
                Ty::Any | Ty::Str | Ty::Tuple(_) | Ty::TupleVar(_) => true,
                Ty::Inst(cl, _) => self.db.find_member(*cl, "__len__").is_some(),
                _ => false,
            };
            if !ok {
                let t = self.show(&v.ty);
                self.err(off, format!("{t} has no len()"));
            }
            let o = self.obj(&v);
            let dst = self.reg(Repr::I64);
            self.emit(Instr::Len { dst, obj: o });
            return Val { reg: dst, ty: Ty::Int, repr: Repr::I64 };
        }
        let fv = self.expr(&c.func);
        let mut pos = Vec::new();
        for a in c.arguments.args.iter() {
            if matches!(a, Expr::Starred(_)) {
                self.err(off, "`*args` in calls is not supported by lizzard yet");
            }
            pos.push(self.expr(a));
        }
        let mut kws = Vec::new();
        for k in c.arguments.keywords.iter() {
            let Some(name) = &k.arg else {
                self.err(off, "`**kwargs` in calls is not supported by lizzard yet");
                continue;
            };
            let v = self.expr(&k.value);
            kws.push((name.as_str().to_string(), v));
        }
        let pos_tys: Vec<Ty> = pos.iter().map(|v| v.ty.clone()).collect();
        let kw_tys: Vec<(String, Ty)> = kws.iter().map(|(n, v)| (n.clone(), v.ty.clone())).collect();
        let mut origin = None;
        let mut builtin_ctor = false;
        let ret = match fv.ty.clone() {
            Ty::Func(FuncRef::Builtin("print")) => Ty::None,
            Ty::Func(FuncRef::Builtin(n)) => {
                self.err(off, format!("{n}() is not supported here by lizzard yet"));
                Ty::Any
            }
            Ty::Type(t) if matches!(*t, Ty::Exc(_) | Ty::Int | Ty::Float | Ty::Str | Ty::Bool) => {
                builtin_ctor = true;
                *t
            }
            Ty::Func(FuncRef::Def(f)) => self.call_result(f, None, &pos_tys, &kw_tys, off, &mut origin),
            Ty::Func(FuncRef::Bound(f, recv)) => self.call_result(f, Some(*recv), &pos_tys, &kw_tys, off, &mut origin),
            Ty::Type(_) | Ty::Any => Ty::Any,
            t => {
                let t = self.show(&t);
                self.err(off, format!("{t} is not callable"));
                Ty::Any
            }
        };
        let func = self.obj(&fv);
        let mut args: Vec<Reg> = pos.iter().map(|v| self.obj(v)).collect();
        let mut kwnames = Vec::new();
        for (n, v) in &kws {
            args.push(self.obj(v));
            kwnames.push(self.name(n));
        }
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::Call { dst, func, args, kwnames });
        let v = Val { reg: dst, ty: ret, repr: Repr::Obj };
        if !builtin_ctor {
            let what = format!("{}(...)", self.src(&*c.func));
            self.check(&v, what, origin);
        }
        self.unbox_f32(v)
    }

    /// A value statically of type float32 is kept unboxed.
    fn unbox_f32(&mut self, v: Val) -> Val {
        if v.repr == Repr::Obj && self.is_f32(&v.ty) {
            let reg = self.convert(&v, Repr::F32);
            return Val { reg, ty: v.ty, repr: Repr::F32 };
        }
        v
    }

    fn call_result(&mut self, f: FuncId, recv: Option<Ty>, pos: &[Ty], kws: &[(String, Ty)], off: u32, origin: &mut Option<String>) -> Ty {
        match self.resolve(f, recv, pos, kws) {
            Ok((t, file, doff)) => {
                *origin = self.origin(file, doff);
                t
            }
            Err(msg) => {
                self.err(off, msg);
                Ty::Any
            }
        }
    }

    /// Resolve a call of `f` (overloads in order, the typing spec's rule for
    /// matches that depend on `Any`). Returns the result type and where the
    /// chosen signature is declared.
    fn resolve(&mut self, f: FuncId, recv: Option<Ty>, pos: &[Ty], kws: &[(String, Ty)]) -> Result<(Ty, FileId, u32), String> {
        let sigs = self.db.sigs(f);
        let fname = self.db.funcs[f].name.clone();
        let mut matches: Vec<(Ty, bool, FileId, u32)> = Vec::new();
        let mut first_err = None;
        for sig in sigs.iter() {
            match self.match_sig(sig, recv.as_ref(), pos, kws) {
                Ok((ret, any)) => matches.push((ret, any, sig.file, sig.off)),
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        if let Some((ret, any, file, off)) = matches.first().cloned() {
            if !any || matches.iter().all(|m| m.0 == ret) {
                return Ok((ret, file, off));
            }
            return Ok((Ty::Any, file, off));
        }
        let shown = |db: &Db, t: &Ty| db.display(t);
        if sigs.len() == 1 {
            return Err(match first_err {
                Some(Mismatch::Arg { name, expected, got }) => {
                    format!("argument `{name}` of {fname}(): expected {}, got {}", shown(self.db, &expected), shown(self.db, &got))
                }
                Some(Mismatch::Arity(m)) => format!("{fname}(): {m}"),
                None => format!("{fname}() cannot be called"),
            });
        }
        let mut tys: Vec<String> = pos.iter().map(|t| shown(self.db, t)).collect();
        tys.extend(kws.iter().map(|(n, t)| format!("{n}={}", shown(self.db, t))));
        Err(format!("no overload of {fname}() accepts ({})", tys.join(", ")))
    }

    fn match_sig(&mut self, sig: &Sig, recv: Option<&Ty>, pos: &[Ty], kws: &[(String, Ty)]) -> Result<(Ty, bool), Mismatch> {
        let mut pairs: Vec<(String, Ty, Ty)> = Vec::new();
        let mut params = sig.params.iter();
        if let Some(r) = recv {
            let Some(p) = params.next() else {
                return Err(Mismatch::Arity("missing `self`".into()));
            };
            pairs.push((p.name.clone(), p.ty.clone(), r.clone()));
        }
        let mut i = 0;
        let mut used = vec![false; kws.len()];
        let kw = |name: &str, used: &mut Vec<bool>| -> Option<Ty> {
            let k = kws.iter().position(|(n, _)| n == name)?;
            used[k] = true;
            Some(kws[k].1.clone())
        };
        for p in params {
            match p.kind {
                PKind::PosOnly | PKind::PosOrKw => {
                    if i < pos.len() {
                        pairs.push((p.name.clone(), p.ty.clone(), pos[i].clone()));
                        i += 1;
                    } else if let Some(t) = matches!(p.kind, PKind::PosOrKw).then(|| kw(&p.name, &mut used)).flatten() {
                        pairs.push((p.name.clone(), p.ty.clone(), t));
                    } else if !p.default {
                        return Err(Mismatch::Arity(format!("missing argument `{}`", p.name)));
                    }
                }
                PKind::VarPos => {
                    while i < pos.len() {
                        pairs.push((p.name.clone(), p.ty.clone(), pos[i].clone()));
                        i += 1;
                    }
                }
                PKind::KwOnly => match kw(&p.name, &mut used) {
                    Some(t) => pairs.push((p.name.clone(), p.ty.clone(), t)),
                    None if p.default => {}
                    None => return Err(Mismatch::Arity(format!("missing keyword argument `{}`", p.name))),
                },
                PKind::VarKw => {
                    for (k, (n, t)) in kws.iter().enumerate() {
                        if !used[k] {
                            used[k] = true;
                            pairs.push((n.clone(), p.ty.clone(), t.clone()));
                        }
                    }
                }
            }
        }
        if i < pos.len() {
            return Err(Mismatch::Arity(format!("takes {i} positional arguments but {} were given", pos.len())));
        }
        if let Some(k) = used.iter().position(|u| !u) {
            return Err(Mismatch::Arity(format!("unexpected keyword argument `{}`", kws[k].0)));
        }
        let mut s = Subst::new();
        let mut any = false;
        for (name, want, got) in pairs {
            if !self.db.assign(&got, &want, &mut s, &mut any) {
                return Err(Mismatch::Arg { name, expected: erase_vars(&subst(&want, &s)), got });
            }
        }
        Ok((erase_vars(&subst(&sig.ret, &s)), any))
    }

    /// The method `name` of `recv`, resolved for the given arguments.
    fn dunder(&mut self, recv: &Ty, name: &str, args: &[Ty]) -> Option<Result<(Ty, FileId, u32), String>> {
        let Ty::Inst(c, _) = recv else { return None };
        match self.db.find_member(*c, name)? {
            (_, Member::Method(f)) => Some(self.resolve(f, Some(recv.clone()), args, &[])),
            _ => None,
        }
    }

    fn load_item(&mut self, s: &ast::ExprSubscript, e: &Expr) -> Val {
        let off: u32 = s.range().start().into();
        let obj = self.expr(&s.value);
        let key = self.expr(&s.slice);
        if self.vec_f32(&obj.ty) && matches!(key.ty, Ty::Int | Ty::Bool) {
            let Ty::Inst(_, args) = &obj.ty else { unreachable!() };
            let Ty::Inst(_, dargs) = &args[1] else { unreachable!() };
            let elem = dargs[0].clone();
            let arr = self.obj(&obj);
            let idx = self.convert(&key, Repr::I64);
            let dst = self.reg(Repr::F32);
            self.emit(Instr::ArrLoadF32 { dst, arr, idx });
            return Val { reg: dst, ty: elem, repr: Repr::F32 };
        }
        let mut origin = None;
        let ty = match &obj.ty {
            Ty::Any => Ty::Any,
            t => match self.dunder(t, "__getitem__", std::slice::from_ref(&key.ty)) {
                Some(Ok((t, file, doff))) => {
                    origin = self.origin(file, doff);
                    t
                }
                Some(Err(m)) => {
                    self.err(off, m);
                    Ty::Any
                }
                None => {
                    let t = self.show(t);
                    self.err(off, format!("{t} is not subscriptable"));
                    Ty::Any
                }
            },
        };
        let (o, k) = (self.obj(&obj), self.obj(&key));
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::GetItem { dst, obj: o, key: k });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        let what = self.src(e);
        self.check(&v, what, origin);
        self.unbox_f32(v)
    }

    fn store_item(&mut self, s: &ast::ExprSubscript, val: Val) {
        let off: u32 = s.range().start().into();
        let obj = self.expr(&s.value);
        let key = self.expr(&s.slice);
        if self.vec_f32(&obj.ty) && matches!(key.ty, Ty::Int | Ty::Bool) && (self.is_f32(&val.ty) || val.ty == Ty::Float) {
            let arr = self.obj(&obj);
            let idx = self.convert(&key, Repr::I64);
            let v = if val.repr == Repr::F32 { val.reg } else { self.convert(&val, Repr::F32) };
            self.emit(Instr::ArrStoreF32 { arr, idx, val: v });
            return;
        }
        match &obj.ty {
            Ty::Any => {}
            t => match self.dunder(t, "__setitem__", &[key.ty.clone(), val.ty.clone()]) {
                Some(Ok(_)) => {}
                Some(Err(m)) => self.err(off, m),
                None => {
                    let t = self.show(t);
                    self.err(off, format!("{t} does not support item assignment"));
                }
            },
        }
        let (o, k, v) = (self.obj(&obj), self.obj(&key), self.obj(&val));
        self.emit(Instr::SetItem { obj: o, key: k, val: v });
    }

    fn binop(&mut self, b: &ast::ExprBinOp, e: &Expr) -> Val {
        let off: u32 = b.range().start().into();
        let (op, dunder, rdunder) = match b.op {
            ast::Operator::Add => (BinOp::Add, "__add__", "__radd__"),
            ast::Operator::Sub => (BinOp::Sub, "__sub__", "__rsub__"),
            ast::Operator::Mult => (BinOp::Mul, "__mul__", "__rmul__"),
            ast::Operator::Div => (BinOp::Div, "__truediv__", "__rtruediv__"),
            _ => {
                self.err(off, format!("operator `{}` is not supported by lizzard yet", b.op.as_str()));
                return self.any();
            }
        };
        let l = self.expr(&b.left);
        let r = self.expr(&b.right);
        let mut origin = None;
        let ty = match (&l.ty, &r.ty) {
            (Ty::Any, _) | (_, Ty::Any) => Ty::Any,
            (Ty::Int | Ty::Bool, Ty::Int | Ty::Bool) if op != BinOp::Div => Ty::Int,
            (Ty::Int | Ty::Bool | Ty::Float, Ty::Int | Ty::Bool | Ty::Float) => Ty::Float,
            (Ty::Str, Ty::Str) if op == BinOp::Add => Ty::Str,
            (lt, rt) => {
                let (lt, rt) = (lt.clone(), rt.clone());
                match self.dunder(&lt, dunder, std::slice::from_ref(&rt)) {
                    Some(Ok((t, file, doff))) => {
                        origin = self.origin(file, doff);
                        t
                    }
                    _ => match self.dunder(&rt, rdunder, std::slice::from_ref(&lt)) {
                        Some(Ok((t, file, doff))) => {
                            origin = self.origin(file, doff);
                            t
                        }
                        _ => {
                            let (ls, rs) = (self.show(&lt), self.show(&rt));
                            self.err(off, format!("unsupported operand types for {}: {ls} and {rs}", b.op.as_str()));
                            Ty::Any
                        }
                    },
                }
            }
        };
        // float32 arithmetic on two float32 values. (A Python float operand
        // could be an `np.float64` at runtime, which NumPy promotes, so mixed
        // operations go through NumPy.)
        if self.is_f32(&l.ty) && self.is_f32(&r.ty) && self.is_f32(&ty) {
            let (a, bb) = (self.convert(&l, Repr::F32), self.convert(&r, Repr::F32));
            let dst = self.reg(Repr::F32);
            self.emit(Instr::F32Bin { dst, op, a, b: bb });
            return Val { reg: dst, ty, repr: Repr::F32 };
        }
        let (a, bb) = (self.obj(&l), self.obj(&r));
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::BinOp { dst, op, a, b: bb });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        let what = self.src(e);
        self.check(&v, what, origin);
        self.unbox_f32(v)
    }

    fn compare(&mut self, c: &ast::ExprCompare, e: &Expr) -> Val {
        let off: u32 = c.range().start().into();
        if c.ops.len() != 1 {
            self.err(off, "chained comparisons are not supported by lizzard yet");
            return self.any();
        }
        let (op, dunder) = match c.ops[0] {
            ast::CmpOp::Eq => (CmpOp::Eq, "__eq__"),
            ast::CmpOp::NotEq => (CmpOp::Ne, "__ne__"),
            ast::CmpOp::Lt => (CmpOp::Lt, "__lt__"),
            ast::CmpOp::LtE => (CmpOp::Le, "__le__"),
            ast::CmpOp::Gt => (CmpOp::Gt, "__gt__"),
            ast::CmpOp::GtE => (CmpOp::Ge, "__ge__"),
            ast::CmpOp::Is => (CmpOp::Is, ""),
            ast::CmpOp::IsNot => (CmpOp::IsNot, ""),
            _ => {
                self.err(off, "`in` is not supported by lizzard yet");
                return self.any();
            }
        };
        let l = self.expr(&c.operands[0]);
        let r = self.expr(&c.operands[1]);
        let ty = match (&l.ty, &r.ty) {
            _ if dunder.is_empty() => Ty::Bool,
            (Ty::Any, _) | (_, Ty::Any) => Ty::Any,
            (lt @ Ty::Inst(..), rt) => {
                let (lt, rt) = (lt.clone(), rt.clone());
                match self.dunder(&lt, dunder, &[rt]) {
                    Some(Ok((t, _, _))) => t,
                    Some(Err(_)) => Ty::Any,
                    None => Ty::Bool,
                }
            }
            _ => Ty::Bool,
        };
        let (a, b) = (self.obj(&l), self.obj(&r));
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::Compare { dst, op, a, b });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        let what = self.src(e);
        self.check(&v, what, None);
        v
    }

    fn boolop(&mut self, b: &ast::ExprBoolOp) -> Val {
        let dst = self.reg(Repr::Obj);
        let mut ty: Option<Ty> = None;
        let mut exits = Vec::new();
        for (i, e) in b.values.iter().enumerate() {
            let v = self.expr(e);
            ty = match ty {
                None => Some(v.ty.clone()),
                Some(t) if t == v.ty => Some(t),
                Some(_) => Some(Ty::Any),
            };
            let o = self.obj(&v);
            self.emit(Instr::Move { dst, src: o });
            if i + 1 < b.values.len() {
                let t = self.reg(Repr::Bool);
                self.emit(Instr::Truthy { dst: t, src: dst });
                let j = match b.op {
                    ast::BoolOp::Or => Instr::JumpIfTrue { cond: t, target: 0 },
                    ast::BoolOp::And => Instr::JumpIfFalse { cond: t, target: 0 },
                };
                exits.push(self.emit(j));
            }
        }
        let h = self.here();
        for j in exits {
            self.patch(j, h);
        }
        Val { reg: dst, ty: ty.unwrap_or(Ty::Any), repr: Repr::Obj }
    }
}

fn is_builtin_call(c: &ast::ExprCall, name: &str) -> bool {
    matches!(&*c.func, Expr::Name(n) if n.id.as_str() == name)
}

fn always_exits(body: &[Stmt]) -> bool {
    match body.last() {
        Some(Stmt::Return(_) | Stmt::Raise(_)) => true,
        Some(Stmt::If(i)) => {
            always_exits(&i.body)
                && i.elif_else_clauses.last().is_some_and(|c| c.test.is_none())
                && i.elif_else_clauses.iter().all(|c| always_exits(&c.body))
        }
        _ => false,
    }
}
