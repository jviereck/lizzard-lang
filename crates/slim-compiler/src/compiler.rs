//! AST → register bytecode, with static ownership tracking.
//!
//! Every local has a flow-sensitive ownership state (live, moved, ...) and a
//! static type. Moves of values with a known move type are checked at compile
//! time; when the type is unknown the compiler emits dynamically checked
//! instructions (`MoveDyn`, `OwnDyn`, `CheckLive`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use slim_syntax::ast::*;
use slim_syntax::{Error, Span};
use slim_vm::code::{
    Arg, ArgList, ClassSpec, Code, ExcEntry, FieldSpec, ParamInfo, Reg, RetKind, TraitSpec, TypeDesc,
};
use slim_vm::object::{Body, Obj};
use slim_vm::ops::{BinKind, Instr, StoreMode, UnKind};
use slim_vm::Value;

use crate::scan;
use crate::ty::{elem_ty, resolve, Ty, TypeName};

type R<T> = Result<T, Error>;

fn own_err(msg: impl Into<String>, span: Span) -> Error {
    Error::new("OwnershipError", msg, span)
}

fn type_err(msg: impl Into<String>, span: Span) -> Error {
    Error::new("TypeError", msg, span)
}

fn name_err(msg: impl Into<String>, span: Span) -> Error {
    Error::new("NameError", msg, span)
}

fn syn_err(msg: impl Into<String>, span: Span) -> Error {
    Error::syntax(msg, span)
}

// ---------------------------------------------------------------- module info

#[derive(Clone, Debug)]
pub struct ParamSig {
    pub name: String,
    pub keep: bool,
    pub ty: Ty,
}

#[derive(Clone, Debug)]
pub struct Sig {
    pub params: Vec<ParamSig>,
    pub ret: Ty,
}

pub struct ClassInfo {
    pub fields: Vec<(String, Ty, TypeDesc)>,
    pub methods: HashMap<String, Sig>,
}

pub struct ModInfo {
    pub globals: Vec<String>,
    pub gindex: HashMap<String, u32>,
    pub module_vars: HashSet<String>,
    pub funcs: HashMap<String, Sig>,
    pub classes: HashMap<String, ClassInfo>,
    pub traits: HashMap<String, HashMap<String, Sig>>,
    pub builtin_classes: HashSet<String>,
    pub builtin_traits: HashSet<String>,
    pub filename: Arc<str>,
}

impl ModInfo {
    fn type_lookup(&self, n: &str) -> Option<(u32, TypeName)> {
        let idx = *self.gindex.get(n)?;
        if self.classes.contains_key(n) || self.builtin_classes.contains(n) {
            Some((idx, TypeName::Class))
        } else if self.traits.contains_key(n) || self.builtin_traits.contains(n) {
            Some((idx, TypeName::Trait))
        } else {
            None
        }
    }

    fn resolve_type(&self, t: &TypeExpr) -> R<(Ty, TypeDesc)> {
        resolve(t, &|n| self.type_lookup(n)).map_err(|m| type_err(m, t.span()))
    }

    fn sig_of(&self, f: &FuncDef, skip_self: bool) -> R<Sig> {
        let mut params = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            if skip_self && i == 0 {
                continue;
            }
            let ty = match &p.ty {
                Some(t) => self.resolve_type(t)?.0,
                None => Ty::Unknown,
            };
            params.push(ParamSig { name: p.name.clone(), keep: p.keep, ty });
        }
        let ret = match &f.ret {
            Some(t) => self.resolve_type(t)?.0,
            None => Ty::Unknown,
        };
        Ok(Sig { params, ret })
    }
}

/// Compiles a parsed module. Returns the module code and the global names.
pub fn compile_module(module: &Module, filename: &str) -> R<(Arc<Code>, Vec<String>)> {
    let mut globals: Vec<String> = slim_vm::builtins::names().into_iter().map(String::from).collect();
    let builtin_names: HashSet<String> = globals.iter().cloned().collect();
    let mut module_vars = Vec::new();
    scan::assigned_names(&module.body, &mut module_vars);
    let mut decls = HashSet::new();
    for s in &module.body.stmts {
        match &s.kind {
            StmtKind::Def(f) => {
                if let Some(b) = &f.body {
                    scan::global_decls(b, &mut decls);
                }
            }
            StmtKind::Class(c) => {
                for m in &c.methods {
                    if let Some(b) = &m.body {
                        scan::global_decls(b, &mut decls);
                    }
                }
            }
            _ => {}
        }
    }
    for n in module_vars.iter().chain(decls.iter()) {
        if !globals.contains(n) {
            globals.push(n.clone());
        }
    }
    let gindex: HashMap<String, u32> = globals.iter().enumerate().map(|(i, n)| (n.clone(), i as u32)).collect();
    let builtin_classes: HashSet<String> = slim_vm::builtins::names()
        .into_iter()
        .filter(|n| n.ends_with("Error") || *n == "StopIteration" || *n == "GeneratorExit")
        .filter(|n| *n != "LookupError" && *n != "ArithmeticError")
        .map(String::from)
        .collect();
    let builtin_traits: HashSet<String> =
        ["Exception", "LookupError", "ArithmeticError", "Iterator"].iter().map(|s| s.to_string()).collect();
    let mut info = ModInfo {
        globals: globals.clone(),
        gindex,
        module_vars: module_vars.iter().cloned().collect(),
        funcs: HashMap::new(),
        classes: HashMap::new(),
        traits: HashMap::new(),
        builtin_classes,
        builtin_traits,
        filename: filename.into(),
    };
    // Register class and trait names first so annotations can refer to them.
    for s in &module.body.stmts {
        match &s.kind {
            StmtKind::Class(c) => {
                if builtin_names.contains(&c.name) {
                    return Err(syn_err(format!("cannot redefine builtin `{}`", c.name), c.span));
                }
                info.classes.insert(c.name.clone(), ClassInfo { fields: Vec::new(), methods: HashMap::new() });
            }
            StmtKind::Trait(t) => {
                info.traits.insert(t.name.clone(), HashMap::new());
            }
            _ => {}
        }
    }
    for s in &module.body.stmts {
        match &s.kind {
            StmtKind::Def(f) => {
                let sig = info.sig_of(f, false)?;
                info.funcs.insert(f.name.clone(), sig);
            }
            StmtKind::Trait(t) => {
                let mut ms = HashMap::new();
                for m in &t.methods {
                    check_method_typing(m)?;
                    ms.insert(m.name.clone(), info.sig_of(m, true)?);
                }
                info.traits.insert(t.name.clone(), ms);
            }
            StmtKind::Class(c) => {
                let mut fields = Vec::new();
                for f in &c.fields {
                    if fields.iter().any(|(n, _, _): &(String, Ty, TypeDesc)| *n == f.name) {
                        return Err(syn_err(format!("duplicate field `{}`", f.name), f.span));
                    }
                    let (ty, d) = info.resolve_type(&f.ty)?;
                    fields.push((f.name.clone(), ty, d));
                }
                let mut methods = HashMap::new();
                for m in &c.methods {
                    check_method_typing(m)?;
                    if fields.iter().any(|(n, _, _)| *n == m.name) {
                        return Err(syn_err(format!("`{}` is both a field and a method", m.name), m.span));
                    }
                    methods.insert(m.name.clone(), info.sig_of(m, true)?);
                }
                info.classes.insert(c.name.clone(), ClassInfo { fields, methods });
            }
            _ => {}
        }
    }
    // Trait default methods become available on implementing classes.
    for s in &module.body.stmts {
        if let StmtKind::Class(c) = &s.kind {
            for (tname, tspan) in &c.traits {
                if let Some(tm) = info.traits.get(tname) {
                    let tm = tm.clone();
                    let ci = info.classes.get_mut(&c.name).unwrap();
                    for (n, sig) in tm {
                        ci.methods.entry(n).or_insert(sig);
                    }
                } else if !info.builtin_traits.contains(tname) {
                    if info.classes.contains_key(tname) || info.builtin_classes.contains(tname) {
                        return Err(type_err(
                            format!("`{tname}` is a class, not a trait; classes cannot inherit (use a field and delegate by hand)"),
                            *tspan,
                        ));
                    }
                    return Err(name_err(format!("unknown trait `{tname}`"), *tspan));
                }
            }
        }
    }
    for s in &module.body.stmts {
        if let StmtKind::Class(c) = &s.kind {
            for (tname, _) in &c.traits {
                let Some(tdef) = module.body.stmts.iter().find_map(|s| match &s.kind {
                    StmtKind::Trait(t) if t.name == *tname => Some(t),
                    _ => None,
                }) else {
                    continue;
                };
                for m in &tdef.methods {
                    if m.body.is_none() && !c.methods.iter().any(|cm| cm.name == m.name) {
                        return Err(type_err(
                            format!("class `{}` implements `{tname}` but does not define required method `{}`", c.name, m.name),
                            c.span,
                        ));
                    }
                }
            }
        }
    }
    let mut ctx = FnCtx::new(&info, "<module>".into(), true);
    ctx.state = Some(HashMap::new());
    ctx.block(&module.body)?;
    let r = ctx.tmp_owned();
    ctx.emit(Instr::LoadImm { dst: r, bits: Value::UNIT.0 as i32 });
    ctx.emit(Instr::Return { src: r });
    let code = ctx.finish(Vec::new(), RetKind::Any, false, false);
    Ok((Arc::new(code), globals))
}

fn check_method_typing(m: &FuncDef) -> R<()> {
    match m.params.first() {
        Some(p) if p.name == "self" => {
            if p.ty.is_some() {
                return Err(syn_err("`self` must not be annotated", p.span));
            }
        }
        _ => return Err(syn_err(format!("method `{}` must take `self` as its first parameter", m.name), m.span)),
    }
    for p in &m.params[1..] {
        if p.ty.is_none() {
            return Err(syn_err(
                format!("parameter `{}` of method `{}` needs a type annotation (all method parameters must be typed)", p.name, m.name),
                p.span,
            ));
        }
    }
    if m.ret.is_none() {
        return Err(syn_err(
            format!("method `{}` needs a return type (use `-> ()` if it returns nothing)", m.name),
            m.span,
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- ownership state

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Uninit,
    Live,
    Moved,
    MaybeMoved,
    /// Possibly moved by an operation whose ownership effect is only known at
    /// runtime; uses are checked dynamically.
    DynMoved,
    MaybeUninit,
}

#[derive(Clone, Debug, PartialEq)]
struct Var {
    st: St,
    ty: Ty,
}

type State = HashMap<String, Var>;

fn join_st(a: St, b: St) -> St {
    use St::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Live, Moved) | (Moved, Live) | (MaybeMoved, _) | (_, MaybeMoved) => MaybeMoved,
        (DynMoved, Moved) | (Moved, DynMoved) => MaybeMoved,
        (DynMoved, _) | (_, DynMoved) => DynMoved,
        (Uninit, Moved) | (Moved, Uninit) => Moved,
        _ => MaybeUninit,
    }
}

fn join_states(a: &State, b: &State) -> State {
    let mut out = State::new();
    for k in a.keys().chain(b.keys()) {
        if out.contains_key(k) {
            continue;
        }
        if k.starts_with('@') {
            // Narrowed field paths survive a join only if narrowed on both sides.
            if let (Some(x), Some(y)) = (a.get(k), b.get(k)) {
                if x == y {
                    out.insert(k.clone(), x.clone());
                }
            }
            continue;
        }
        let x = a.get(k).cloned().unwrap_or(Var { st: St::Uninit, ty: Ty::Unknown });
        let y = b.get(k).cloned().unwrap_or(Var { st: St::Uninit, ty: Ty::Unknown });
        let ty = if x.st == St::Uninit {
            y.ty.clone()
        } else if y.st == St::Uninit {
            x.ty.clone()
        } else {
            x.ty.join(&y.ty)
        };
        out.insert(k.clone(), Var { st: join_st(x.st, y.st), ty });
    }
    out
}

fn join_opt(a: Option<State>, b: Option<State>) -> Option<State> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => Some(join_states(&a, &b)),
    }
}

// ---------------------------------------------------------------- function context

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LocalKind {
    Owned,
    /// A non-`keep` parameter: borrowed for the whole call.
    Borrowed,
}

struct Local {
    reg: Reg,
    kind: LocalKind,
    /// Declared annotation: (static type, runtime descriptor).
    decl: Option<(Ty, TypeDesc)>,
}

#[derive(Clone, Debug, PartialEq)]
enum Own {
    /// A temporary that owns its value.
    Owned,
    /// The register of a named local (or a module variable), borrowed.
    Local(String),
    /// A borrow of a parameter, field, element, global or capture.
    Borrowed,
}

#[derive(Clone, Debug)]
struct Res {
    reg: Reg,
    own: Own,
    ty: Ty,
    /// Local register this borrow derives from (`p` for `p.a.b`).
    root: Option<Reg>,
    /// Human description for error messages.
    desc: String,
}

struct LoopCtx {
    head: u32,
    breaks: Vec<usize>,
    continues: Vec<usize>,
    break_state: Option<State>,
    cont_state: Option<State>,
    finally_depth: usize,
}

struct FnCtx<'m> {
    m: &'m ModInfo,
    name: String,
    instrs: Vec<Instr>,
    lines: Vec<u32>,
    consts: Vec<Value>,
    names: Vec<String>,
    arglists: Vec<ArgList>,
    types: Vec<TypeDesc>,
    codes: Vec<Arc<Code>>,
    classes: Vec<ClassSpec>,
    traits: Vec<TraitSpec>,
    exc_table: Vec<ExcEntry>,
    owned: Vec<bool>,
    reg_names: Vec<Option<String>>,

    locals: HashMap<String, Local>,
    captures: Vec<(String, Ty)>,
    state: Option<State>,
    is_module: bool,
    global_decls: HashSet<String>,
    owned_pool: Vec<Reg>,
    owned_used: usize,
    borrow_pool: Vec<Reg>,
    borrow_used: usize,
    loops: Vec<LoopCtx>,
    finally_stack: Vec<Block>,
    ret: RetKind,
    ret_ty: Ty,
    ret_desc: u32,
    is_generator: bool,
    line: u32,
    heap_borrows: usize,
    stmt_drops: Vec<Reg>,
    moved_log: Vec<String>,
    class_name: Option<String>,
    handler_exc: Vec<Reg>,
    /// Module-level annotations: name -> (static type, runtime descriptor).
    module_decl_table: HashMap<String, (Ty, TypeDesc)>,
    /// Static types of call argument registers (for `Some(x)` typing).
    reg_ty: HashMap<Reg, Ty>,
}

impl<'m> FnCtx<'m> {
    fn new(m: &'m ModInfo, name: String, is_module: bool) -> FnCtx<'m> {
        FnCtx {
            m,
            name,
            instrs: Vec::new(),
            lines: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            arglists: Vec::new(),
            types: Vec::new(),
            codes: Vec::new(),
            classes: Vec::new(),
            traits: Vec::new(),
            exc_table: Vec::new(),
            owned: Vec::new(),
            reg_names: Vec::new(),
            locals: HashMap::new(),
            captures: Vec::new(),
            state: None,
            is_module,
            global_decls: HashSet::new(),
            owned_pool: Vec::new(),
            owned_used: 0,
            borrow_pool: Vec::new(),
            borrow_used: 0,
            loops: Vec::new(),
            finally_stack: Vec::new(),
            ret: RetKind::Any,
            ret_ty: Ty::Unknown,
            ret_desc: 0,
            is_generator: false,
            line: 1,
            heap_borrows: 0,
            stmt_drops: Vec::new(),
            moved_log: Vec::new(),
            class_name: None,
            handler_exc: Vec::new(),
            module_decl_table: HashMap::new(),
            reg_ty: HashMap::new(),
        }
    }

    fn finish(self, params: Vec<ParamInfo>, ret: RetKind, is_generator: bool, is_method: bool) -> Code {
        Code {
            name: self.name,
            filename: self.m.filename.clone(),
            params,
            ret,
            nregs: self.owned.len() as u16,
            owned: self.owned,
            reg_names: self.reg_names,
            instrs: self.instrs,
            lines: self.lines,
            consts: self.consts,
            names: self.names,
            arglists: self.arglists,
            types: self.types,
            codes: self.codes,
            classes: self.classes,
            traits: self.traits,
            exc_table: self.exc_table,
            ncaptures: self.captures.len() as u16,
            is_generator,
            is_method,
        }
    }

    // ------------------------------------------------------------ emission

    fn emit(&mut self, i: Instr) -> usize {
        self.instrs.push(i);
        self.lines.push(self.line);
        self.instrs.len() - 1
    }

    fn here(&self) -> u32 {
        self.instrs.len() as u32
    }

    fn patch(&mut self, at: usize, target: u32) {
        match &mut self.instrs[at] {
            Instr::Jump { target: t }
            | Instr::JumpIfFalse { target: t, .. }
            | Instr::JumpIfTrue { target: t, .. }
            | Instr::MatchException { target: t, .. } => *t = target,
            Instr::ForIter { end, .. } => *end = target,
            other => panic!("cannot patch {other:?}"),
        }
    }

    fn jump(&mut self) -> usize {
        self.emit(Instr::Jump { target: u32::MAX })
    }

    fn konst(&mut self, v: Value) -> u32 {
        if let Some(i) = self.consts.iter().position(|c| *c == v) {
            return i as u32;
        }
        self.consts.push(v);
        (self.consts.len() - 1) as u32
    }

    fn str_const(&mut self, s: &str) -> u32 {
        if let Some(i) = self.consts.iter().position(|c| c.as_str() == Some(s)) {
            return i as u32;
        }
        let v = Obj::alloc_immortal(Body::Str(s.into()));
        self.consts.push(v);
        (self.consts.len() - 1) as u32
    }

    fn name_idx(&mut self, s: &str) -> u32 {
        if let Some(i) = self.names.iter().position(|n| n == s) {
            return i as u32;
        }
        self.names.push(s.to_string());
        (self.names.len() - 1) as u32
    }

    fn type_idx(&mut self, t: TypeDesc) -> u32 {
        if let Some(i) = self.types.iter().position(|x| *x == t) {
            return i as u32;
        }
        self.types.push(t);
        (self.types.len() - 1) as u32
    }

    fn arglist(&mut self, al: ArgList) -> u32 {
        self.arglists.push(al);
        (self.arglists.len() - 1) as u32
    }

    fn new_reg(&mut self, owned: bool, name: Option<String>) -> Reg {
        self.owned.push(owned);
        self.reg_names.push(name);
        (self.owned.len() - 1) as Reg
    }

    fn tmp_owned(&mut self) -> Reg {
        if self.owned_used < self.owned_pool.len() {
            self.owned_used += 1;
            return self.owned_pool[self.owned_used - 1];
        }
        let r = self.new_reg(true, None);
        self.owned_pool.push(r);
        self.owned_used += 1;
        r
    }

    fn tmp_borrow(&mut self) -> Reg {
        if self.borrow_used < self.borrow_pool.len() {
            self.borrow_used += 1;
            return self.borrow_pool[self.borrow_used - 1];
        }
        let r = self.new_reg(false, None);
        self.borrow_pool.push(r);
        self.borrow_used += 1;
        r
    }

    fn load_imm(&mut self, dst: Reg, v: Value) {
        self.emit(Instr::LoadImm { dst, bits: v.0 as i64 as i32 });
    }

    // ------------------------------------------------------------ state helpers

    fn st(&self, name: &str) -> Option<&Var> {
        self.state.as_ref().and_then(|s| s.get(name))
    }

    fn set_var(&mut self, name: &str, st: St, ty: Ty) {
        if let Some(s) = &mut self.state {
            s.insert(name.to_string(), Var { st, ty });
        }
    }

    fn set_st(&mut self, name: &str, st: St) {
        if let Some(s) = &mut self.state {
            if let Some(v) = s.get_mut(name) {
                v.st = st;
            } else {
                s.insert(name.to_string(), Var { st, ty: Ty::Unknown });
            }
        }
        if st == St::Moved || st == St::DynMoved {
            self.moved_log.push(name.to_string());
        }
    }

    fn is_global_name(&self, name: &str) -> bool {
        if self.is_module {
            return self.m.gindex.contains_key(name);
        }
        !self.locals.contains_key(name)
            && !self.captures.iter().any(|(n, _)| n == name)
            && self.m.gindex.contains_key(name)
    }

    /// Checks that a local (or module variable) may be used here.
    fn check_use(&mut self, name: &str, reg: Option<Reg>, span: Span) -> R<()> {
        let Some(v) = self.st(name).cloned() else {
            if self.is_module && self.m.module_vars.contains(name) {
                return Err(name_err(format!("`{name}` is used before it is assigned"), span));
            }
            return Ok(());
        };
        match v.st {
            St::Live => Ok(()),
            St::Moved => Err(own_err(format!("`{name}` was moved and can no longer be used; use `{name}++` before moving it to keep a reference"), span)),
            St::MaybeMoved => Err(own_err(format!("`{name}` may have been moved on some path and can no longer be used"), span)),
            St::Uninit => Err(name_err(format!("`{name}` is used before it is assigned"), span)),
            St::DynMoved | St::MaybeUninit => {
                if let Some(r) = reg {
                    self.emit(Instr::CheckLive { reg: r });
                }
                Ok(())
            }
        }
    }

    // ------------------------------------------------------------ blocks and statements

    fn block(&mut self, b: &Block) -> R<()> {
        for s in &b.stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    /// Compiles a block whose last line is its value, moved into `dst`.
    fn block_value(&mut self, b: &Block, dst: Reg) -> R<Ty> {
        let n = b.stmts.len();
        if n == 0 {
            self.load_imm(dst, Value::UNIT);
            return Ok(Ty::Unit);
        }
        for s in &b.stmts[..n - 1] {
            self.stmt(s)?;
        }
        let last = &b.stmts[n - 1];
        match &last.kind {
            StmtKind::Expr(e) => {
                self.line = last.span.line;
                let mark = self.mark();
                let r = self.value_into(e, dst)?;
                self.end_stmt(mark);
                Ok(r)
            }
            _ => {
                self.stmt(last)?;
                if self.state.is_some() {
                    self.load_imm(dst, Value::UNIT);
                }
                Ok(Ty::Unit)
            }
        }
    }

    fn mark(&mut self) -> (usize, usize) {
        self.heap_borrows = 0;
        (self.owned_used, self.borrow_used)
    }

    fn end_stmt(&mut self, mark: (usize, usize)) {
        for r in std::mem::take(&mut self.stmt_drops) {
            self.emit(Instr::Drop { reg: r });
        }
        self.owned_used = mark.0;
        self.borrow_used = mark.1;
        self.heap_borrows = 0;
    }

    fn stmt(&mut self, s: &Stmt) -> R<()> {
        self.line = s.span.line;
        if self.state.is_none() {
            // Unreachable code after return/break/raise: still compile it so
            // errors are reported, with a fresh state.
            self.state = Some(State::new());
        }
        let mark = self.mark();
        let saved_drops = std::mem::take(&mut self.stmt_drops);
        let r = self.stmt_inner(s);
        self.end_stmt(mark);
        self.stmt_drops = saved_drops;
        r
    }

    fn stmt_inner(&mut self, s: &Stmt) -> R<()> {
        match &s.kind {
            StmtKind::Expr(e) => {
                match &e.kind {
                    ExprKind::If { .. } | ExprKind::Try(_) => {
                        let dst = self.tmp_owned();
                        self.value_into(e, dst)?;
                        if self.state.is_some() {
                            self.emit(Instr::Drop { reg: dst });
                        }
                    }
                    ExprKind::Ellipsis => {}
                    _ => {
                        let r = self.expr(e)?;
                        self.discard(&r);
                    }
                }
                Ok(())
            }
            StmtKind::Pass => Ok(()),
            StmtKind::Assign { target, value } => self.assign(target, value, s.span, false),
            StmtKind::AnnAssign { target, ann, value } => {
                let ExprKind::Name(n) = &target.kind else {
                    return Err(syn_err("only simple names can be annotated", target.span));
                };
                let (ty, d) = self.m.resolve_type(ann)?;
                if let Some(l) = self.locals.get_mut(n) {
                    l.decl = Some((ty.clone(), d.clone()));
                } else if self.is_module {
                    self.module_decl_table.insert(n.clone(), (ty.clone(), d.clone()));
                }
                match value {
                    Some(v) => self.assign(target, v, s.span, false),
                    None => Ok(()),
                }
            }
            StmtKind::AugAssign { target, op, value } => self.aug_assign(target, *op, value, s.span),
            StmtKind::Move { targets, value } => self.move_chain(targets, value, s.span),
            StmtKind::Return(v) => self.return_stmt(v.as_ref(), s.span),
            StmtKind::While { cond, body } => self.while_stmt(cond, body),
            StmtKind::For { target, iter, body } => self.for_stmt(target, iter, body, s.span),
            StmtKind::Break => self.break_continue(true, s.span),
            StmtKind::Continue => self.break_continue(false, s.span),
            StmtKind::Def(f) => self.def_stmt(f),
            StmtKind::Class(c) => self.class_stmt(c),
            StmtKind::Trait(t) => self.trait_stmt(t),
            StmtKind::Raise { exc, cause } => self.raise_stmt(exc.as_ref(), cause.as_ref(), s.span),
            StmtKind::Global(names) => {
                if self.is_module {
                    return Err(syn_err("`global` is only meaningful inside a function", s.span));
                }
                for n in names {
                    if self.locals.contains_key(n) && !self.global_decls.contains(n) {
                        return Err(syn_err(format!("`{n}` is assigned before its `global` declaration"), s.span));
                    }
                }
                Ok(())
            }
            StmtKind::Del(targets) => {
                for t in targets {
                    self.del_target(t)?;
                }
                Ok(())
            }
            StmtKind::With { ctx, name, body } => self.with_stmt(ctx, name.as_deref(), body, s.span),
            StmtKind::Assert { cond, msg } => {
                // `assert c, m` == `if not c: raise AssertionError(m)`
                let fails = self.cond_jump(cond, true)?;
                let raise = self.jump();
                let h = self.here();
                for at in fails {
                    self.patch(at, h);
                }
                let _ = raise;
                let skip = self.jump();
                let raise_at = self.here();
                self.patch(raise, raise_at);
                let before = self.state.clone();
                let args: Vec<Expr> = msg.iter().cloned().collect();
                let call = Expr {
                    kind: ExprKind::Call(Box::new(Expr { kind: ExprKind::Name("AssertionError".into()), span: s.span }), args),
                    span: s.span,
                };
                self.raise_stmt(Some(&call), None, s.span)?;
                self.state = before;
                let h = self.here();
                self.patch(skip, h);
                Ok(())
            }
        }
    }

    /// `with ctx as name: body` ==
    /// `t = ctx++; name = t.__enter__(); try: body finally: t.__exit__()`
    fn with_stmt(&mut self, ctx: &Expr, name: Option<&str>, body: &Block, span: Span) -> R<()> {
        let hidden = format!("<with {}>", self.instrs.len());
        let reg = self.new_reg(true, Some(hidden.clone()));
        self.locals.insert(hidden.clone(), Local { reg, kind: LocalKind::Owned, decl: None });
        let r = self.expr(ctx)?;
        let ty = r.ty.clone();
        let o = if r.own == Own::Owned {
            r
        } else {
            let dst = self.tmp_owned();
            self.emit(Instr::Dup { dst, src: r.reg });
            Res { reg: dst, own: Own::Owned, ty: ty.clone(), root: None, desc: "value".into() }
        };
        self.emit(Instr::Mov { dst: reg, src: o.reg });
        self.set_var(&hidden, St::Live, ty);
        let hname = || Box::new(Expr { kind: ExprKind::Name(hidden.clone()), span });
        let method = |m: &str| Expr {
            kind: ExprKind::Call(Box::new(Expr { kind: ExprKind::Attr(hname(), m.into()), span }), Vec::new()),
            span,
        };
        let enter = method("__enter__");
        match name {
            Some(n) => {
                let target = Expr { kind: ExprKind::Name(n.into()), span };
                self.assign(&target, &enter, span, false)?;
            }
            None => {
                let r = self.expr(&enter)?;
                self.discard(&r);
            }
        }
        let fin = Block { stmts: vec![Stmt { kind: StmtKind::Expr(method("__exit__")), span }] };
        let t = TryExpr { body: body.clone(), handlers: Vec::new(), else_: None, finally: Some(fin) };
        let dst = self.tmp_owned();
        self.try_expr(&t, Some(dst))?;
        if self.state.is_some() {
            self.emit(Instr::Drop { reg: dst });
            self.emit(Instr::Drop { reg });
        }
        self.set_st(&hidden, St::Uninit);
        Ok(())
    }

    // ------------------------------------------------------------ expressions

    fn discard(&mut self, r: &Res) {
        if r.own == Own::Owned {
            self.emit(Instr::Drop { reg: r.reg });
        }
    }

    /// Evaluates `e` and moves its (owned) value into `dst`.
    fn value_into(&mut self, e: &Expr, dst: Reg) -> R<Ty> {
        match &e.kind {
            ExprKind::If { branches, else_ } => self.if_expr(branches, else_.as_ref(), Some(dst)),
            ExprKind::Try(t) => self.try_expr(t, Some(dst)),
            _ => {
                let r = self.expr_owned(e, "value")?;
                self.emit(Instr::Mov { dst, src: r.reg });
                Ok(r.ty)
            }
        }
    }

    /// Evaluates `e` into an owned temporary. Named locals are moved; borrows
    /// of copy types are copied; borrows of move types are an error.
    fn expr_owned(&mut self, e: &Expr, what: &str) -> R<Res> {
        if let ExprKind::If { branches, else_ } = &e.kind {
            let dst = self.tmp_owned();
            let ty = self.if_expr(branches, else_.as_ref(), Some(dst))?;
            return Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() });
        }
        let r = self.expr(e)?;
        self.to_owned(r, e.span, what)
    }

    fn to_owned(&mut self, r: Res, span: Span, what: &str) -> R<Res> {
        match &r.own {
            Own::Owned => Ok(r),
            Own::Local(name) => {
                let name = name.clone();
                self.move_local(&name, r, span)
            }
            Own::Borrowed => {
                let dst = self.tmp_owned();
                match r.ty.is_copy() {
                    Some(true) => {
                        self.emit(Instr::Dup { dst, src: r.reg });
                    }
                    Some(false) => {
                        return Err(own_err(
                            format!(
                                "cannot move {} (a borrowed `{}`) into {what}; use `{}++` to create a new reference{}",
                                r.desc,
                                r.ty.name(),
                                short(&r.desc),
                                if r.desc.starts_with("field") { ", or swap it out with `x <- obj.field <- new`" } else { "" }
                            ),
                            span,
                        ))
                    }
                    None => {
                        let w = self.name_idx(&r.desc);
                        self.emit(Instr::OwnDyn { dst, src: r.reg, what: w });
                    }
                }
                Ok(Res { reg: dst, own: Own::Owned, ty: r.ty, root: None, desc: "value".into() })
            }
        }
    }

    /// Moves a named local into a fresh owned temporary.
    fn move_local(&mut self, name: &str, r: Res, span: Span) -> R<Res> {
        let dst = self.tmp_owned();
        if self.is_module || self.is_global_name(name) {
            // A module variable: move it out of its global slot.
            let idx = self.m.gindex[name];
            match r.ty.is_copy() {
                Some(true) => {
                    self.emit(Instr::Dup { dst, src: r.reg });
                }
                Some(false) => {
                    self.emit(Instr::TakeGlobal { dst, idx });
                    self.set_st(name, St::Moved);
                }
                None => {
                    self.emit(Instr::TakeGlobal { dst, idx });
                    self.set_st(name, St::DynMoved);
                }
            }
            return Ok(Res { reg: dst, own: Own::Owned, ty: r.ty, root: None, desc: "value".into() });
        }
        let local = &self.locals[name];
        if local.kind == LocalKind::Borrowed {
            let r2 = Res { own: Own::Borrowed, desc: format!("parameter `{name}`"), ..r };
            return self.to_owned(r2, span, "this place");
        }
        let src = local.reg;
        match r.ty.is_copy() {
            Some(true) => {
                self.emit(Instr::Dup { dst, src });
            }
            Some(false) => {
                self.emit(Instr::Mov { dst, src });
                self.set_st(name, St::Moved);
            }
            None => {
                self.emit(Instr::MoveDyn { dst, src });
                self.set_st(name, St::DynMoved);
            }
        }
        Ok(Res { reg: dst, own: Own::Owned, ty: r.ty, root: None, desc: "value".into() })
    }

    fn name_expr(&mut self, n: &str, span: Span) -> R<Res> {
        if let Some(l) = self.locals.get(n) {
            let (reg, kind) = (l.reg, l.kind);
            let decl_ty = l.decl.as_ref().map(|d| d.0.clone());
            self.check_use(n, Some(reg), span)?;
            let ty = self.st(n).map(|v| v.ty.clone()).or(decl_ty).unwrap_or(Ty::Unknown);
            let own = if kind == LocalKind::Borrowed { Own::Borrowed } else { Own::Local(n.to_string()) };
            let desc = if kind == LocalKind::Borrowed { format!("parameter `{n}`") } else { format!("`{n}`") };
            return Ok(Res { reg, own, ty, root: Some(reg), desc });
        }
        if let Some(i) = self.captures.iter().position(|(c, _)| c == n) {
            let ty = self.captures[i].1.clone();
            let dst = self.tmp_borrow();
            self.emit(Instr::LoadCapture { dst, idx: i as u32 });
            return Ok(Res { reg: dst, own: Own::Borrowed, ty, root: None, desc: format!("captured `{n}`") });
        }
        if let Some(&idx) = self.m.gindex.get(n) {
            let dst = self.tmp_borrow();
            self.emit(Instr::LoadGlobal { dst, idx });
            let ty = self.global_ty(n);
            if self.is_module && self.m.module_vars.contains(n) {
                self.check_use(n, None, span)?;
                let ty = self.st(n).map(|v| v.ty.clone()).unwrap_or(ty);
                return Ok(Res { reg: dst, own: Own::Local(n.to_string()), ty, root: None, desc: format!("`{n}`") });
            }
            // Builtins, functions, classes and traits are immortal: borrowing
            // them never needs an epoch pin.
            let immortal = !self.m.module_vars.contains(n)
                || self.m.funcs.contains_key(n)
                || self.m.classes.contains_key(n)
                || self.m.traits.contains_key(n);
            if !immortal {
                self.heap_borrows += 1;
            }
            return Ok(Res { reg: dst, own: Own::Borrowed, ty, root: None, desc: format!("global `{n}`") });
        }
        Err(name_err(format!("name `{n}` is not defined"), span))
    }

    fn global_ty(&self, n: &str) -> Ty {
        if self.m.classes.contains_key(n) || self.m.builtin_classes.contains(n) {
            return Ty::Class(n.to_string());
        }
        if self.m.funcs.contains_key(n) {
            return Ty::StaticFn;
        }
        if let Some((t, _)) = self.module_decl_table.get(n) {
            return t.clone();
        }
        Ty::Unknown
    }

    fn expr(&mut self, e: &Expr) -> R<Res> {
        let owned = |reg, ty| Res { reg, own: Own::Owned, ty, root: None, desc: "value".into() };
        match &e.kind {
            ExprKind::Name(n) => self.name_expr(n, e.span),
            ExprKind::Int(v) => {
                let dst = self.tmp_owned();
                let i = i64::try_from(*v).map_err(|_| type_err("integer literal does not fit in a 64-bit int; use bigint(\"...\")", e.span))?;
                self.load_int(dst, i);
                Ok(owned(dst, Ty::Int))
            }
            ExprKind::Float(f) => {
                let dst = self.tmp_owned();
                let c = self.konst(Obj::alloc_immortal(Body::Float(*f)));
                self.emit(Instr::LoadConst { dst, idx: c });
                Ok(owned(dst, Ty::Float))
            }
            ExprKind::Str(s) => {
                let dst = self.tmp_owned();
                let c = self.str_const(s);
                self.emit(Instr::LoadConst { dst, idx: c });
                Ok(owned(dst, Ty::Str))
            }
            ExprKind::Bool(b) => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::bool(*b));
                Ok(owned(dst, Ty::Bool))
            }
            ExprKind::NoneLit => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::NONE);
                Ok(owned(dst, Ty::NoneT))
            }
            ExprKind::Unit => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::UNIT);
                Ok(owned(dst, Ty::Unit))
            }
            ExprKind::Ellipsis => Err(syn_err("`...` is only allowed as the body of a trait method", e.span)),
            ExprKind::Tuple(items) | ExprKind::List(items) => {
                let mut args = Vec::new();
                for it in items {
                    let r = self.expr_owned(it, "a collection")?;
                    args.push(Arg::new(r.reg, true, false, None));
                }
                let al = self.arglist(ArgList { args, pins: false });
                let dst = self.tmp_owned();
                if matches!(e.kind, ExprKind::Tuple(_)) {
                    self.emit(Instr::BuildTuple { dst, args: al });
                    Ok(owned(dst, Ty::Tuple))
                } else {
                    self.emit(Instr::BuildList { dst, args: al });
                    Ok(owned(dst, Ty::List))
                }
            }
            ExprKind::Dict(items) => {
                let mut args = Vec::new();
                for (k, v) in items {
                    let kr = self.expr_owned(k, "a dict key")?;
                    args.push(Arg::new(kr.reg, true, false, None));
                    let vr = self.expr_owned(v, "a dict")?;
                    args.push(Arg::new(vr.reg, true, false, None));
                }
                let al = self.arglist(ArgList { args, pins: false });
                let dst = self.tmp_owned();
                self.emit(Instr::BuildDict { dst, args: al });
                Ok(owned(dst, Ty::Dict))
            }
            ExprKind::Attr(obj, name) => self.attr_expr(obj, name, e.span),
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                if let Ty::Array(e) = o.ty {
                    // Array elements are scalars: read an owned copy.
                    let dst = self.tmp_owned();
                    self.emit(Instr::LoadElem { dst, obj: o.reg, idx: i.reg });
                    self.discard(&i);
                    self.discard(&o);
                    return Ok(owned(dst, e.map(elem_ty).unwrap_or(Ty::Unknown)));
                }
                let dst = self.tmp_borrow();
                self.emit(Instr::LoadIndex { dst, obj: o.reg, idx: i.reg });
                self.discard(&i);
                if o.own == Own::Owned {
                    self.stmt_drops.push(o.reg);
                }
                self.heap_borrows += 1;
                let ty = match &o.ty {
                    Ty::Str => Ty::Str,
                    _ => Ty::Unknown,
                };
                Ok(Res { reg: dst, own: Own::Borrowed, ty, root: o.root, desc: "an element".into() })
            }
            ExprKind::Call(f, args) => self.call_expr(f, args, e.span),
            ExprKind::Dup(inner) => {
                let r = self.expr(inner)?;
                if r.own == Own::Owned {
                    return Ok(r);
                }
                if let Own::Local(n) = &r.own {
                    let n = n.clone();
                    self.check_use(&n, Some(r.reg), e.span)?;
                }
                let dst = self.tmp_owned();
                self.emit(Instr::Dup { dst, src: r.reg });
                Ok(owned(dst, r.ty))
            }
            ExprKind::Unary(op, x) => {
                let r = self.expr(x)?;
                let dst = self.tmp_owned();
                let ty = match op {
                    UnOp::Not => {
                        self.emit(Instr::Not { dst, src: r.reg });
                        Ty::Bool
                    }
                    UnOp::Neg | UnOp::Pos | UnOp::Invert => {
                        let k = match op {
                            UnOp::Neg => UnKind::Neg,
                            UnOp::Pos => UnKind::Pos,
                            _ => UnKind::Invert,
                        };
                        self.emit(Instr::Unary { dst, op: k, src: r.reg });
                        match r.ty {
                            Ty::Int | Ty::Float | Ty::BigInt => r.ty.clone(),
                            _ => Ty::Unknown,
                        }
                    }
                };
                self.discard(&r);
                Ok(owned(dst, ty))
            }
            ExprKind::Binary(op, a, b) => {
                let x = self.expr(a)?;
                let y = self.expr(b)?;
                let dst = self.tmp_owned();
                let ins = match op {
                    BinOp::Add => Instr::Add { dst, a: x.reg, b: y.reg },
                    BinOp::Sub => Instr::Sub { dst, a: x.reg, b: y.reg },
                    BinOp::Mul => Instr::Mul { dst, a: x.reg, b: y.reg },
                    _ => {
                        let k = match op {
                            BinOp::Div => BinKind::Div,
                            BinOp::FloorDiv => BinKind::FloorDiv,
                            BinOp::Mod => BinKind::Mod,
                            BinOp::Pow => BinKind::Pow,
                            BinOp::BitAnd => BinKind::BitAnd,
                            BinOp::BitOr => BinKind::BitOr,
                            BinOp::BitXor => BinKind::BitXor,
                            BinOp::Shl => BinKind::Shl,
                            _ => BinKind::Shr,
                        };
                        Instr::Binary { dst, op: k, a: x.reg, b: y.reg }
                    }
                };
                self.emit(ins);
                let ty = binop_ty(*op, &x.ty, &y.ty);
                self.discard(&y);
                self.discard(&x);
                Ok(owned(dst, ty))
            }
            ExprKind::Compare(..) | ExprKind::And(..) | ExprKind::Or(..) => {
                let dst = self.tmp_owned();
                let f = self.cond_jump(e, false)?;
                self.load_imm(dst, Value::TRUE);
                let j = self.jump();
                for at in f {
                    let h = self.here();
                    self.patch(at, h);
                }
                self.load_imm(dst, Value::FALSE);
                let h = self.here();
                self.patch(j, h);
                Ok(owned(dst, Ty::Bool))
            }
            ExprKind::Ternary { cond, then, else_ } => {
                let dst = self.tmp_owned();
                let f = self.cond_jump(cond, false)?;
                let before = self.state.clone();
                let t1 = self.value_into(then, dst)?;
                let after_then = self.state.take();
                let j = self.jump();
                let h = self.here();
                for at in f {
                    self.patch(at, h);
                }
                self.state = before;
                let t2 = self.value_into(else_, dst)?;
                let h = self.here();
                self.patch(j, h);
                self.state = join_opt(after_then, self.state.take());
                Ok(owned(dst, t1.join(&t2)))
            }
            ExprKind::If { branches, else_ } => {
                let dst = self.tmp_owned();
                let ty = self.if_expr(branches, else_.as_ref(), Some(dst))?;
                Ok(owned(dst, ty))
            }
            ExprKind::Try(t) => {
                let dst = self.tmp_owned();
                let ty = self.try_expr(t, Some(dst))?;
                Ok(owned(dst, ty))
            }
            ExprKind::Lambda(f) => {
                let dst = self.tmp_owned();
                self.make_closure(f, dst)?;
                Ok(owned(dst, Ty::Func))
            }
            ExprKind::Yield(v) => {
                if !self.is_generator {
                    return Err(syn_err("`yield` outside of a function", e.span));
                }
                if self.heap_borrows > 0 {
                    return Err(own_err("`yield` cannot appear in an expression that borrows a field, element or global", e.span));
                }
                let src = match v {
                    Some(v) => self.expr_owned(v, "`yield`")?.reg,
                    None => {
                        let r = self.tmp_owned();
                        self.load_imm(r, Value::UNIT);
                        r
                    }
                };
                let dst = self.tmp_owned();
                self.emit(Instr::Yield { dst, src });
                Ok(owned(dst, Ty::Unknown))
            }
            ExprKind::YieldFrom(sub) => {
                if !self.is_generator {
                    return Err(syn_err("`yield from` outside of a function", e.span));
                }
                let s = self.expr(sub)?;
                let it = self.new_reg(true, Some("<yield from>".into()));
                self.emit(Instr::GetIter { dst: it, src: s.reg });
                self.discard(&s);
                let head = self.here();
                let item = self.tmp_owned();
                let fi = self.emit(Instr::ForIter { dst: item, iter: it, end: u32::MAX });
                let sent = self.tmp_owned();
                self.emit(Instr::Yield { dst: sent, src: item });
                self.emit(Instr::Drop { reg: sent });
                self.emit(Instr::Loop { target: head });
                let h = self.here();
                self.patch(fi, h);
                self.emit(Instr::Drop { reg: it });
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::UNIT);
                Ok(owned(dst, Ty::Unit))
            }
        }
    }

    fn load_int(&mut self, dst: Reg, i: i64) {
        if (-(1i64 << 29)..(1i64 << 29)).contains(&i) {
            self.load_imm(dst, Value::int_unchecked(i));
        } else {
            let v = match Value::small_int(i) {
                Some(v) => v,
                None => Obj::alloc_immortal(Body::Int(i)),
            };
            let c = self.konst(v);
            self.emit(Instr::LoadConst { dst, idx: c });
        }
    }

    fn attr_expr(&mut self, obj: &Expr, name: &str, span: Span) -> R<Res> {
        let o = self.expr(obj)?;
        self.option_check(&o, &format!("accessing `.{name}`"), span)?;
        let mut ty = Ty::Unknown;
        if let Ty::Instance(c) = &o.ty {
            if let Some(ci) = self.m.classes.get(c) {
                if let Some((_, fty, _)) = ci.fields.iter().find(|(n, _, _)| n == name) {
                    ty = fty.clone();
                } else if ci.methods.contains_key(name) {
                    ty = Ty::Func;
                } else {
                    return Err(Error::new("AttributeError", format!("class `{c}` has no field or method `{name}`"), span));
                }
            }
        }
        if ty == Ty::Unknown && name == "msg" {
            if let Ty::Instance(c) = &o.ty {
                if self.m.builtin_classes.contains(c) {
                    ty = Ty::Str;
                }
            }
        }
        if let ExprKind::Name(base) = &obj.kind {
            if let Some(v) = self.st(&format!("@{base}.{name}")) {
                ty = v.ty.clone();
            }
        }
        let dst = self.tmp_borrow();
        let n = self.name_idx(name);
        self.emit(Instr::LoadField { dst, obj: o.reg, name: n });
        if o.own == Own::Owned {
            self.stmt_drops.push(o.reg);
        }
        self.heap_borrows += 1;
        Ok(Res { reg: dst, own: Own::Borrowed, ty, root: o.root, desc: format!("field `{name}`") })
    }

    // ------------------------------------------------------------ conditions

    /// Emits a conditional jump; returns jump sites taken when `e` is
    /// `!when_true`... i.e. with `when_true = false`, the returned jumps are
    /// taken when the condition is false.
    fn cond_jump(&mut self, e: &Expr, jump_if: bool) -> R<Vec<usize>> {
        match &e.kind {
            ExprKind::Unary(UnOp::Not, x) => self.cond_jump(x, !jump_if),
            ExprKind::And(a, b) if !jump_if => {
                let mut j = self.cond_jump(a, false)?;
                j.extend(self.cond_jump(b, false)?);
                Ok(j)
            }
            ExprKind::Or(a, b) if jump_if => {
                let mut j = self.cond_jump(a, true)?;
                j.extend(self.cond_jump(b, true)?);
                Ok(j)
            }
            ExprKind::And(a, b) => {
                // jump if (a and b): skip when a is false.
                let skip = self.cond_jump(a, false)?;
                let j = self.cond_jump(b, true)?;
                let h = self.here();
                for at in skip {
                    self.patch(at, h);
                }
                Ok(j)
            }
            ExprKind::Or(a, b) => {
                // jump if !(a or b): skip when a is true.
                let skip = self.cond_jump(a, true)?;
                let j = self.cond_jump(b, false)?;
                let h = self.here();
                for at in skip {
                    self.patch(at, h);
                }
                Ok(j)
            }
            ExprKind::Compare(_, rest) if rest.len() > 1 => {
                let t = self.compare_chain(e)?;
                let j = if jump_if {
                    self.emit(Instr::JumpIfTrue { cond: t, target: u32::MAX })
                } else {
                    self.emit(Instr::JumpIfFalse { cond: t, target: u32::MAX })
                };
                Ok(vec![j])
            }
            ExprKind::Compare(first, rest) => {
                let (op, rhs) = &rest[0];
                let narrowing = self.narrowing_of(first, *op, rhs);
                let l = self.expr(first)?;
                let r = self.expr(rhs)?;
                let t = self.tmp_owned();
                self.emit_cmp(*op, t, l.reg, r.reg);
                self.discard(&r);
                self.discard(&l);
                let j = if jump_if {
                    self.emit(Instr::JumpIfTrue { cond: t, target: u32::MAX })
                } else {
                    self.emit(Instr::JumpIfFalse { cond: t, target: u32::MAX })
                };
                // `x is not None` narrows x on the fall-through path.
                if let Some((name, is_not_none)) = narrowing {
                    if is_not_none != jump_if {
                        self.narrow(&name);
                    }
                }
                Ok(vec![j])
            }
            _ => {
                let r = self.expr(e)?;
                let cond = if r.own == Own::Owned && r.ty != Ty::Bool {
                    // Reduce to a bool first so the value is dropped on both paths.
                    let t = self.tmp_owned();
                    let t2 = self.tmp_owned();
                    self.emit(Instr::Not { dst: t, src: r.reg });
                    self.emit(Instr::Drop { reg: r.reg });
                    self.emit(Instr::Not { dst: t2, src: t });
                    t2
                } else {
                    r.reg
                };
                let j = if jump_if {
                    self.emit(Instr::JumpIfTrue { cond, target: u32::MAX })
                } else {
                    self.emit(Instr::JumpIfFalse { cond, target: u32::MAX })
                };
                Ok(vec![j])
            }
        }
    }

    /// `a < b < c` as a bool value (operands evaluated lazily, each dropped).
    fn compare_chain(&mut self, e: &Expr) -> R<Reg> {
        let ExprKind::Compare(first, rest) = &e.kind else { unreachable!() };
        let dst = self.tmp_owned();
        let mut left = self.expr(first)?;
        let mut stubs: Vec<(usize, Option<Reg>)> = Vec::new();
        for (i, (op, rhs)) in rest.iter().enumerate() {
            let right = self.expr(rhs)?;
            self.emit_cmp(*op, dst, left.reg, right.reg);
            self.discard(&left);
            if i + 1 < rest.len() {
                let j = self.emit(Instr::JumpIfFalse { cond: dst, target: u32::MAX });
                stubs.push((j, (right.own == Own::Owned).then_some(right.reg)));
            }
            left = right;
        }
        self.discard(&left);
        let done = self.jump();
        let mut to_end = vec![done];
        for (j, owned) in stubs {
            let h = self.here();
            self.patch(j, h);
            if let Some(r) = owned {
                self.emit(Instr::Drop { reg: r });
            }
            to_end.push(self.jump());
        }
        let h = self.here();
        for j in to_end {
            self.patch(j, h);
        }
        Ok(dst)
    }

    fn emit_cmp(&mut self, op: CmpOp, dst: Reg, a: Reg, b: Reg) {
        let i = match op {
            CmpOp::Eq => Instr::Eq { dst, a, b },
            CmpOp::Ne => Instr::Ne { dst, a, b },
            CmpOp::Lt => Instr::Lt { dst, a, b },
            CmpOp::Le => Instr::Le { dst, a, b },
            CmpOp::Gt => Instr::Gt { dst, a, b },
            CmpOp::Ge => Instr::Ge { dst, a, b },
            CmpOp::Is => Instr::Is { dst, a, b },
            CmpOp::IsNot => Instr::IsNot { dst, a, b },
            CmpOp::In => Instr::In { dst, a, b },
            CmpOp::NotIn => Instr::NotIn { dst, a, b },
        };
        self.emit(i);
    }

    /// For `x is None` / `x is not None` on a local: (name, is_not_none).
    fn narrowing_of(&self, l: &Expr, op: CmpOp, r: &Expr) -> Option<(String, bool)> {
        if !matches!(r.kind, ExprKind::NoneLit) {
            return None;
        }
        let n = match &l.kind {
            ExprKind::Name(n) => n.clone(),
            ExprKind::Attr(o, f) => match &o.kind {
                ExprKind::Name(base) => format!("@{base}.{f}"),
                _ => return None,
            },
            _ => return None,
        };
        match op {
            CmpOp::IsNot => Some((n.clone(), true)),
            CmpOp::Is => Some((n.clone(), false)),
            _ => None,
        }
    }

    /// Records that a local is known not to be `None` on the current path.
    fn narrow(&mut self, name: &str) {
        if let Some(path) = name.strip_prefix('@') {
            // A field path: remember the narrowed type of `base.field`.
            let (base, field) = path.split_once('.').unwrap();
            let base_ty = self.local_ty(base);
            if let Ok(Ty::Option(inner)) = self.field_ty(&base_ty, field, Span::default()) {
                self.set_var(name, St::Live, *inner);
            }
            return;
        }
        if let Some(s) = &mut self.state {
            if let Some(v) = s.get_mut(name) {
                if let Ty::Option(inner) = &v.ty {
                    v.ty = (**inner).clone();
                }
            }
        }
    }

    fn local_ty(&self, n: &str) -> Ty {
        if let Some(v) = self.st(n) {
            return v.ty.clone();
        }
        self.locals.get(n).and_then(|l| l.decl.as_ref().map(|d| d.0.clone())).unwrap_or(Ty::Unknown)
    }

    /// Forgets narrowing of `base.field` paths after stores.
    fn unnarrow(&mut self, base: &str, field: Option<&str>) {
        if let Some(s) = &mut self.state {
            let prefix = match field {
                Some(f) => format!("@{base}.{f}"),
                None => format!("@{base}."),
            };
            s.retain(|k, _| !k.starts_with(&prefix));
        }
    }

    fn option_check(&self, r: &Res, what: &str, span: Span) -> R<()> {
        if let Ty::Option(inner) = &r.ty {
            return Err(type_err(
                format!(
                    "{} is `Option<{}>` and may be `None`; narrow it first (`if x is not None:`) or use `.some_or(default)` before {what}",
                    r.desc,
                    inner.name()
                ),
                span,
            ));
        }
        Ok(())
    }

    // ------------------------------------------------------------ if / while / for

    fn if_expr(&mut self, branches: &[(Expr, Block)], else_: Option<&Block>, dst: Option<Reg>) -> R<Ty> {
        let mut end_jumps = Vec::new();
        let mut out_state: Option<State> = None;
        let mut ty: Option<Ty> = None;
        for (cond, body) in branches {
            let before = self.state.clone();
            let f = self.cond_jump(cond, false)?;
            let false_state = {
                // The false path does not see narrowing applied for the true path.
                let mut s = self.state.clone();
                if let (Some(s), Some(b)) = (&mut s, &before) {
                    for (k, v) in b {
                        if let Some(cur) = s.get_mut(k) {
                            if cur.st == v.st {
                                cur.ty = v.ty.clone();
                            }
                        }
                    }
                }
                s
            };
            let t = match dst {
                Some(d) => self.block_value(body, d)?,
                None => {
                    self.block(body)?;
                    Ty::Unit
                }
            };
            if self.state.is_some() {
                ty = Some(ty.map_or(t.clone(), |x| x.join(&t)));
            }
            out_state = join_opt(out_state, self.state.take());
            end_jumps.push(self.jump());
            let h = self.here();
            for at in f {
                self.patch(at, h);
            }
            self.state = false_state;
            // `if x is None: ... else:` narrows x in the else branch.
            if let ExprKind::Compare(l, rest) = &cond.kind {
                if rest.len() == 1 {
                    if let Some((n, false)) = self.narrowing_of(l, rest[0].0, &rest[0].1) {
                        self.narrow(&n);
                    }
                }
            }
        }
        match else_ {
            Some(b) => {
                let t = match dst {
                    Some(d) => self.block_value(b, d)?,
                    None => {
                        self.block(b)?;
                        Ty::Unit
                    }
                };
                if self.state.is_some() {
                    ty = Some(ty.map_or(t.clone(), |x| x.join(&t)));
                }
            }
            None => {
                if let Some(d) = dst {
                    self.load_imm(d, Value::UNIT);
                }
                ty = Some(ty.map_or(Ty::Unit, |x| x.join(&Ty::Unit)));
            }
        }
        out_state = join_opt(out_state, self.state.take());
        let h = self.here();
        for at in end_jumps {
            self.patch(at, h);
        }
        self.state = out_state;
        Ok(ty.unwrap_or(Ty::Unknown))
    }

    /// A local the loop body assigns may hold the previous iteration's value
    /// at the head, so its reassignment must drop (a no-op on `UNINIT`/`MOVED`).
    fn widen_loop_head(&mut self, body: &Block, target: Option<&Expr>) {
        let mut names = Vec::new();
        scan::assigned_names(body, &mut names);
        if let Some(t) = target {
            let mut wrapped = Block::default();
            wrapped.stmts.push(Stmt {
                kind: StmtKind::Assign { target: t.clone(), value: t.clone() },
                span: t.span,
            });
            scan::assigned_names(&wrapped, &mut names);
        }
        for n in names {
            if !self.locals.contains_key(&n) {
                continue;
            }
            match self.st(&n).map(|v| v.st) {
                None | Some(St::Uninit) => {
                    let ty = self.st(&n).map(|v| v.ty.clone()).unwrap_or(Ty::Unknown);
                    self.set_var(&n, St::MaybeUninit, ty);
                }
                Some(St::Moved) => self.set_st(&n, St::MaybeMoved),
                _ => {}
            }
        }
    }

    fn while_stmt(&mut self, cond: &Expr, body: &Block) -> R<()> {
        self.widen_loop_head(body, None);
        let entry = self.state.clone();
        let head = self.here();
        let mark = self.mark();
        let exits = self.cond_jump(cond, false)?;
        self.end_stmt(mark);
        let after_cond = self.state.clone();
        self.loops.push(LoopCtx {
            head,
            breaks: Vec::new(),
            continues: Vec::new(),
            break_state: None,
            cont_state: None,
            finally_depth: self.finally_stack.len(),
        });
        self.block(body)?;
        let lp = self.loops.pop().unwrap();
        let end_body = join_opt(self.state.take(), lp.cont_state.clone());
        self.check_loop_moves(&entry, &end_body, body)?;
        let back = self.here();
        self.emit(Instr::Loop { target: head });
        for at in lp.continues {
            self.patch(at, back);
        }
        let end = self.here();
        for at in exits.into_iter().chain(lp.breaks) {
            self.patch(at, end);
        }
        let mut st = join_opt(after_cond, end_body);
        st = join_opt(st, lp.break_state);
        self.state = st;
        Ok(())
    }

    /// A value moved inside a loop body cannot be used in the next iteration.
    fn check_loop_moves(&mut self, entry: &Option<State>, end: &Option<State>, body: &Block) -> R<()> {
        let (Some(entry), Some(end)) = (entry, end) else { return Ok(()) };
        let mut used = HashSet::new();
        scan::used_names(body, &mut used);
        for (name, v) in end {
            let was_live = entry.get(name).is_some_and(|e| e.st == St::Live);
            if was_live && matches!(v.st, St::Moved | St::MaybeMoved) && used.contains(name) {
                return Err(own_err(
                    format!("`{name}` is moved inside the loop and would be used again in the next iteration; use `{name}++` or reassign it before the loop repeats"),
                    body.stmts.first().map(|s| s.span).unwrap_or_default(),
                ));
            }
        }
        Ok(())
    }

    fn for_stmt(&mut self, target: &Expr, iter: &Expr, body: &Block, span: Span) -> R<()> {
        // `for i in range(...)` with literal-sign steps compiles to a counter loop.
        if let ExprKind::Call(f, args) = &iter.kind {
            if let ExprKind::Name(fname) = &f.kind {
                if fname == "range" && !self.locals.contains_key("range") && !self.m.module_vars.contains("range") && (1..=3).contains(&args.len()) {
                    let step = match args.get(2) {
                        None => Some(1),
                        Some(Expr { kind: ExprKind::Int(v), .. }) => Some(*v as i64),
                        Some(Expr { kind: ExprKind::Unary(UnOp::Neg, inner), .. }) => match inner.kind {
                            ExprKind::Int(v) => Some(-(v as i64)),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(step) = step {
                        if step != 0 {
                            return self.for_range(target, args, step, body, span);
                        }
                    }
                }
            }
        }
        let it_reg = self.new_reg(true, Some("<iter>".into()));
        {
            let mark = self.mark();
            let src = self.expr(iter)?;
            self.emit(Instr::GetIter { dst: it_reg, src: src.reg });
            self.discard(&src);
            self.end_stmt(mark);
        }
        self.widen_loop_head(body, Some(target));
        let entry = self.state.clone();
        let head = self.here();
        let item = self.tmp_owned();
        let fi = self.emit(Instr::ForIter { dst: item, iter: it_reg, end: u32::MAX });
        self.assign_owned(target, Res { reg: item, own: Own::Owned, ty: Ty::Unknown, root: None, desc: "value".into() }, span)?;
        self.owned_used -= 1;
        let after_head = self.state.clone();
        self.loops.push(LoopCtx {
            head,
            breaks: Vec::new(),
            continues: Vec::new(),
            break_state: None,
            cont_state: None,
            finally_depth: self.finally_stack.len(),
        });
        self.block(body)?;
        let lp = self.loops.pop().unwrap();
        let end_body = join_opt(self.state.take(), lp.cont_state.clone());
        self.check_loop_moves(&entry, &end_body, body)?;
        let back = self.here();
        self.emit(Instr::Loop { target: head });
        for at in lp.continues {
            self.patch(at, back);
        }
        let end = self.here();
        self.patch(fi, end);
        for at in lp.breaks {
            self.patch(at, end);
        }
        self.emit(Instr::Drop { reg: it_reg });
        let st = join_opt(join_opt(entry, after_head), join_opt(end_body, lp.break_state));
        self.state = st;
        Ok(())
    }

    fn for_range(&mut self, target: &Expr, args: &[Expr], step: i64, body: &Block, span: Span) -> R<()> {
        let ExprKind::Name(var) = &target.kind else {
            return Err(syn_err("`for ... in range(...)` needs a simple loop variable", target.span));
        };
        let counter = self.new_reg(true, Some("<range i>".into()));
        let end = self.new_reg(true, Some("<range end>".into()));
        let step_reg = self.new_reg(true, Some("<range step>".into()));
        {
            let mark = self.mark();
            let (start, stop) = if args.len() == 1 { (None, &args[0]) } else { (Some(&args[0]), &args[1]) };
            match start {
                Some(s) => {
                    let r = self.expr_owned(s, "range")?;
                    self.emit(Instr::Mov { dst: counter, src: r.reg });
                }
                None => self.load_int(counter, 0),
            }
            let r = self.expr_owned(stop, "range")?;
            self.emit(Instr::Mov { dst: end, src: r.reg });
            self.load_int(step_reg, step);
            // Non-int bounds are rejected by the type check.
            let ti = self.type_idx(TypeDesc::Int);
            let w = self.name_idx("range() argument");
            self.emit(Instr::CheckType { reg: counter, ty: ti, what: w });
            self.emit(Instr::CheckType { reg: end, ty: ti, what: w });
            self.end_stmt(mark);
        }
        self.widen_loop_head(body, None);
        let entry = self.state.clone();
        let head = self.here();
        let t = self.tmp_owned();
        if step > 0 {
            self.emit(Instr::Lt { dst: t, a: counter, b: end });
        } else {
            self.emit(Instr::Gt { dst: t, a: counter, b: end });
        }
        let exit = self.emit(Instr::JumpIfFalse { cond: t, target: u32::MAX });
        let item = self.tmp_owned();
        self.emit(Instr::Dup { dst: item, src: counter });
        self.assign_owned(target, Res { reg: item, own: Own::Owned, ty: Ty::Int, root: None, desc: "value".into() }, span)?;
        self.owned_used -= 2;
        let _ = var;
        let after_head = self.state.clone();
        self.loops.push(LoopCtx {
            head: u32::MAX,
            breaks: Vec::new(),
            continues: Vec::new(),
            break_state: None,
            cont_state: None,
            finally_depth: self.finally_stack.len(),
        });
        self.block(body)?;
        let lp = self.loops.pop().unwrap();
        let end_body = join_opt(self.state.take(), lp.cont_state.clone());
        self.check_loop_moves(&entry, &end_body, body)?;
        let incr = self.here();
        for at in lp.continues {
            self.patch(at, incr);
        }
        let t2 = self.tmp_owned();
        self.emit(Instr::Add { dst: t2, a: counter, b: step_reg });
        self.emit(Instr::Mov { dst: counter, src: t2 });
        self.owned_used -= 1;
        self.emit(Instr::Loop { target: head });
        let out = self.here();
        self.patch(exit, out);
        for at in lp.breaks {
            self.patch(at, out);
        }
        let st = join_opt(join_opt(entry, after_head), join_opt(end_body, lp.break_state));
        self.state = st;
        Ok(())
    }

    fn break_continue(&mut self, is_break: bool, span: Span) -> R<()> {
        let Some(lp) = self.loops.last() else {
            return Err(syn_err(if is_break { "`break` outside loop" } else { "`continue` outside loop" }, span));
        };
        let depth = lp.finally_depth;
        let finallys: Vec<Block> = self.finally_stack[depth..].iter().rev().cloned().collect();
        for f in &finallys {
            self.block(f)?;
        }
        let j = self.jump();
        let st = self.state.take();
        let lp = self.loops.last_mut().unwrap();
        if is_break {
            lp.breaks.push(j);
            lp.break_state = join_opt(lp.break_state.take(), st);
        } else {
            if lp.head != u32::MAX {
                // `while`/`for` loops patch continues to their head.
            }
            lp.continues.push(j);
            lp.cont_state = join_opt(lp.cont_state.take(), st);
        }
        Ok(())
    }

    // ------------------------------------------------------------ assignment

    fn assign(&mut self, target: &Expr, value: &Expr, span: Span, _is_move: bool) -> R<()> {
        match &target.kind {
            ExprKind::Name(_) | ExprKind::Tuple(_) => {
                let v = match &value.kind {
                    ExprKind::If { .. } | ExprKind::Try(_) => {
                        let dst = self.tmp_owned();
                        let ty = self.value_into(value, dst)?;
                        Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() }
                    }
                    _ => {
                        let r = self.expr(value)?;
                        self.to_owned(r, value.span, "a variable")?
                    }
                };
                self.assign_owned(target, v, span)
            }
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                let fty = self.field_ty(&o.ty, field, target.span)?;
                if fty.is_move() {
                    return Err(own_err(
                        format!("field `{field}` holds a move type (`{}`); use `<-` to move a value in", fty.name()),
                        span,
                    ));
                }
                let v = self.expr(value)?;
                if v.ty.is_move() {
                    return Err(own_err(
                        format!("`=` can only store copy types into a field; use `<-` to move this `{}` in", v.ty.name()),
                        span,
                    ));
                }
                let v = self.to_owned(v, value.span, "a field")?;
                let n = self.name_idx(field);
                self.emit(Instr::StoreField { obj: o.reg, name: n, src: v.reg, mode: StoreMode::Copy });
                if let ExprKind::Name(base) = &obj.kind {
                    self.unnarrow(base, Some(field));
                }
                self.discard(&o);
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                let v = self.expr(value)?;
                if v.ty.is_move() {
                    return Err(own_err(
                        format!("`=` can only store copy types into an element; use `<-` to move this `{}` in", v.ty.name()),
                        span,
                    ));
                }
                if let Ty::Array(Some(e)) = o.ty {
                    let ok = match (elem_ty(e), &v.ty) {
                        (_, Ty::Unknown) => true,
                        (Ty::Float, Ty::Int | Ty::Float) => true,
                        (t, vt) => t == *vt,
                    };
                    if !ok {
                        return Err(type_err(format!("cannot store a `{}` in an `array<{}>`", v.ty.name(), e.name()), value.span));
                    }
                }
                let v = self.to_owned(v, value.span, "an element")?;
                self.emit(Instr::StoreIndex { obj: o.reg, idx: i.reg, src: v.reg, mode: StoreMode::Copy });
                self.discard(&i);
                self.discard(&o);
                Ok(())
            }
            _ => Err(syn_err("invalid assignment target", target.span)),
        }
    }

    fn field_ty(&self, obj_ty: &Ty, field: &str, span: Span) -> R<Ty> {
        if let Ty::Instance(c) = obj_ty {
            if let Some(ci) = self.m.classes.get(c) {
                return match ci.fields.iter().find(|(n, _, _)| n == field) {
                    Some((_, t, _)) => Ok(t.clone()),
                    None => Err(Error::new("AttributeError", format!("class `{c}` has no field `{field}`"), span)),
                };
            }
        }
        Ok(Ty::Unknown)
    }

    /// Stores an owned value into a name or tuple target.
    fn assign_owned(&mut self, target: &Expr, v: Res, span: Span) -> R<()> {
        match &target.kind {
            ExprKind::Name(n) => self.bind(n, v, target.span),
            ExprKind::Tuple(items) => {
                let mut dsts = Vec::new();
                for _ in items {
                    let r = self.tmp_owned();
                    dsts.push(Arg::new(r, true, false, None));
                }
                let al = self.arglist(ArgList { args: dsts.clone(), pins: false });
                self.emit(Instr::Unpack { src: v.reg, dsts: al });
                for (item, d) in items.iter().zip(dsts) {
                    let r = Res { reg: d.reg, own: Own::Owned, ty: Ty::Unknown, root: None, desc: "value".into() };
                    self.assign_owned(item, r, span)?;
                }
                Ok(())
            }
            ExprKind::Attr(..) | ExprKind::Index(..) => {
                Err(syn_err("only names can be bound here", target.span))
            }
            _ => Err(syn_err("invalid assignment target", target.span)),
        }
    }

    /// Binds an owned temporary to a local or module variable.
    fn bind(&mut self, n: &str, v: Res, span: Span) -> R<()> {
        if self.captures.iter().any(|(c, _)| c == n) {
            return Err(syn_err(format!("cannot assign to captured variable `{n}`"), span));
        }
        let decl = if let Some(l) = self.locals.get(n) {
            if l.kind == LocalKind::Borrowed {
                return Err(own_err(format!("cannot assign to borrowed parameter `{n}` (declare it `+ keep` to own it)"), span));
            }
            l.decl.clone()
        } else {
            self.module_decl_table.get(n).cloned()
        };
        if let Some((dty, _)) = &decl {
            if !ty_compatible(dty, &v.ty) {
                return Err(type_err(format!("`{n}` is declared `{}` but assigned a `{}`", dty.name(), v.ty.name()), span));
            }
        }
        let ty = match &decl {
            Some((dty, _)) => dty.clone(),
            None => v.ty.clone(),
        };
        if self.is_module || (!self.locals.contains_key(n) && self.global_decls.contains(n)) {
            let idx = self.m.gindex[n];
            if let Some((_, d)) = &decl {
                let ti = self.type_idx(d.clone());
                let w = self.name_idx(&format!("variable `{n}`"));
                self.emit(Instr::CheckType { reg: v.reg, ty: ti, what: w });
            }
            self.emit(Instr::StoreGlobal { idx, src: v.reg });
            if self.is_module {
                self.set_var(n, St::Live, ty);
            }
            return Ok(());
        }
        let reg = self.locals[n].reg;
        if let Some((_, d)) = &decl {
            let ti = self.type_idx(d.clone());
            let w = self.name_idx(&format!("variable `{n}`"));
            self.emit(Instr::CheckType { reg: v.reg, ty: ti, what: w });
        }
        let prev = self.st(n).map(|x| x.st).unwrap_or(St::Uninit);
        if !matches!(prev, St::Uninit | St::Moved) {
            self.emit(Instr::Drop { reg });
        }
        self.emit(Instr::Mov { dst: reg, src: v.reg });
        self.set_var(n, St::Live, ty);
        self.unnarrow(n, None);
        Ok(())
    }

    fn aug_assign(&mut self, target: &Expr, op: BinOp, value: &Expr, span: Span) -> R<()> {
        let bin = Expr { kind: ExprKind::Binary(op, Box::new(target.clone()), Box::new(value.clone())), span };
        match &target.kind {
            ExprKind::Name(_) => {
                let r = self.expr(&bin)?;
                self.assign_owned(target, r, span)
            }
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                let r = self.expr(&bin)?;
                let n = self.name_idx(field);
                self.emit(Instr::StoreField { obj: o.reg, name: n, src: r.reg, mode: StoreMode::Move });
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                let r = self.expr(&bin)?;
                self.emit(Instr::StoreIndex { obj: o.reg, idx: i.reg, src: r.reg, mode: StoreMode::Move });
                self.discard(&i);
                Ok(())
            }
            _ => Err(syn_err("invalid augmented assignment target", target.span)),
        }
    }

    /// `t1 <- t2 <- ... <- value`
    fn move_chain(&mut self, targets: &[Expr], value: &Expr, span: Span) -> R<()> {
        let mut cur = match &value.kind {
            ExprKind::If { .. } | ExprKind::Try(_) => {
                let dst = self.tmp_owned();
                let ty = self.value_into(value, dst)?;
                Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() }
            }
            _ => self.expr_owned(value, "this place")?,
        };
        // Swap through the inner targets from right to left.
        for t in targets[1..].iter().rev() {
            cur = self.swap_into(t, cur, span)?;
        }
        let t = &targets[0];
        match &t.kind {
            ExprKind::Name(_) => self.assign_owned(t, cur, span),
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                self.field_ty(&o.ty, field, t.span)?;
                let n = self.name_idx(field);
                self.emit(Instr::StoreField { obj: o.reg, name: n, src: cur.reg, mode: StoreMode::Move });
                if let ExprKind::Name(base) = &obj.kind {
                    self.unnarrow(base, Some(field));
                }
                self.discard(&o);
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                self.emit(Instr::StoreIndex { obj: o.reg, idx: i.reg, src: cur.reg, mode: StoreMode::Move });
                self.discard(&i);
                self.discard(&o);
                Ok(())
            }
            _ => Err(syn_err("invalid move target", t.span)),
        }
    }

    /// Stores owned `v` into `t`, returning the previous value (owned).
    fn swap_into(&mut self, t: &Expr, v: Res, span: Span) -> R<Res> {
        let dst = self.tmp_owned();
        match &t.kind {
            ExprKind::Attr(obj, field) => {
                let o = self.expr(obj)?;
                let fty = self.field_ty(&o.ty, field, t.span)?;
                let n = self.name_idx(field);
                self.emit(Instr::SwapField { dst, obj: o.reg, name: n, src: v.reg });
                if let ExprKind::Name(base) = &obj.kind {
                    self.unnarrow(base, Some(field));
                }
                self.discard(&o);
                Ok(Res { reg: dst, own: Own::Owned, ty: fty, root: None, desc: "value".into() })
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                self.emit(Instr::SwapIndex { dst, obj: o.reg, idx: i.reg, src: v.reg });
                self.discard(&i);
                self.discard(&o);
                Ok(Res { reg: dst, own: Own::Owned, ty: Ty::Unknown, root: None, desc: "value".into() })
            }
            ExprKind::Name(n) => {
                // `x <- y <- v`: x gets y's old value, y gets v.
                let old = self.name_expr(n, t.span)?;
                let old = self.to_owned(old, t.span, "a variable")?;
                self.emit(Instr::Mov { dst, src: old.reg });
                self.bind(n, v, span)?;
                Ok(Res { reg: dst, own: Own::Owned, ty: old.ty, root: None, desc: "value".into() })
            }
            _ => Err(syn_err("invalid swap target", t.span)),
        }
    }

    fn del_target(&mut self, t: &Expr) -> R<()> {
        match &t.kind {
            ExprKind::Name(n) => {
                let Some(l) = self.locals.get(n) else {
                    return Err(syn_err("`del` of a global is not supported", t.span));
                };
                if l.kind == LocalKind::Borrowed {
                    return Err(own_err(format!("cannot `del` borrowed parameter `{n}`"), t.span));
                }
                let reg = l.reg;
                self.emit(Instr::Drop { reg });
                self.set_st(n, St::Uninit);
                Ok(())
            }
            ExprKind::Index(obj, idx) => {
                let o = self.expr(obj)?;
                let i = self.expr(idx)?;
                let al = self.arglist(ArgList { args: vec![Arg::new(i.reg, false, false, None)], pins: false });
                let n = self.name_idx("pop");
                let dst = self.tmp_owned();
                self.emit(Instr::CallMethod { dst, recv: o.reg, name: n, args: al });
                self.emit(Instr::Drop { reg: dst });
                self.discard(&i);
                self.discard(&o);
                Ok(())
            }
            _ => Err(syn_err("invalid `del` target", t.span)),
        }
    }

    // ------------------------------------------------------------ calls

    /// Static signature of a call target, if known.
    fn static_sig(&self, f: &Expr) -> Option<(Sig, Ty)> {
        match &f.kind {
            ExprKind::Name(n) if self.is_global_name(n) => {
                if let Some(s) = self.m.funcs.get(n) {
                    if !self.m.module_vars.contains(n) || self.m.funcs.contains_key(n) {
                        return Some((s.clone(), s.ret.clone()));
                    }
                }
                if let Some(ci) = self.m.classes.get(n) {
                    let sig = ci.methods.get("__init__").cloned().unwrap_or(Sig { params: Vec::new(), ret: Ty::Unit });
                    return Some((sig, Ty::Instance(n.clone())));
                }
                if self.m.builtin_classes.contains(n) {
                    return Some((
                        Sig { params: vec![ParamSig { name: "msg".into(), keep: false, ty: Ty::Unknown }], ret: Ty::Unknown },
                        Ty::Instance(n.clone()),
                    ));
                }
                None
            }
            _ => None,
        }
    }

    fn builtin_call_info(&self, f: &Expr) -> Option<(u64, String)> {
        if let ExprKind::Name(n) = &f.kind {
            if self.is_global_name(n) && !self.m.module_vars.contains(n) {
                return slim_vm::builtins::keep_mask(n).map(|m| (m, n.clone()));
            }
        }
        None
    }

    fn call_expr(&mut self, f: &Expr, args: &[Expr], span: Span) -> R<Res> {
        if let ExprKind::Attr(recv, mname) = &f.kind {
            return self.method_call(recv, mname, args, span);
        }
        let sig = self.static_sig(f);
        let builtin = self.builtin_call_info(f);
        let callee = self.expr(f)?;
        if let Some((s, _)) = &sig {
            if s.params.len() != args.len() && !matches!(&f.kind, ExprKind::Name(n) if self.m.builtin_classes.contains(n)) {
                return Err(type_err(
                    format!("`{}` takes {} argument{} but {} {} given", callee_name(f), s.params.len(), if s.params.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "was" } else { "were" }),
                    span,
                ));
            }
        }
        let keep_of = |i: usize| -> Option<bool> {
            if let Some((s, _)) = &sig {
                return s.params.get(i).map(|p| p.keep);
            }
            if let Some((mask, _)) = &builtin {
                return Some(mask & (1 << i) != 0);
            }
            None
        };
        let al = self.call_args(args, &keep_of, None, span)?;
        let dst = self.tmp_owned();
        let a = self.arglist(al.clone());
        self.emit(Instr::Call { dst, func: callee.reg, args: a });
        self.after_call(&al);
        self.discard(&callee);
        let mut ty = sig.map(|s| s.1).unwrap_or(Ty::Unknown);
        if let Some((_, n)) = &builtin {
            ty = match n.as_str() {
                "str" | "repr" | "chr" => Ty::Str,
                "int" | "len" | "ord" | "hash" | "rc_run" | "rc_pending" | "rc_live" => Ty::Int,
                "float" => Ty::Float,
                "bool" | "isinstance" => Ty::Bool,
                "bigint" => Ty::BigInt,
                "list" | "sorted" => Ty::List,
                "tuple" => Ty::Tuple,
                "dict" => Ty::Dict,
                "array" => Ty::Array(match args.first().map(|a| &a.kind) {
                    Some(ExprKind::Name(n)) if !self.locals.contains_key(n) && !self.m.module_vars.contains(n) => {
                        slim_vm::array::Elem::from_name(n)
                    }
                    _ => None,
                }),
                "print" | "rc_auto" => Ty::Unit,
                "Some" => match al.args.first().and_then(|a| self.reg_ty.get(&a.reg)).cloned() {
                    Some(t) => Ty::Option(Box::new(t)),
                    None => Ty::Unknown,
                },
                _ => Ty::Unknown,
            };
        }
        self.reg_ty.clear();
        Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() })
    }

    /// Evaluates call arguments. `keep_of(i)` is the static keep flag of
    /// parameter i, if known. `recv_root` is the receiver's root local.
    fn call_args(&mut self, args: &[Expr], keep_of: &dyn Fn(usize) -> Option<bool>, recv_root: Option<Reg>, span: Span) -> R<ArgList> {
        let mut out = Vec::new();
        let mut moved_locals: Vec<(String, Reg, Span)> = Vec::new();
        let mut borrowed_roots: Vec<Reg> = recv_root.into_iter().collect();
        for (i, a) in args.iter().enumerate() {
            let keep = keep_of(i);
            let r = self.expr(a)?;
            self.reg_ty.insert(r.reg, r.ty.clone());
            match &r.own {
                Own::Owned => out.push(Arg::new(r.reg, true, false, None)),
                Own::Local(n) => {
                    let n = n.clone();
                    let module_var = self.is_module || self.is_global_name(&n);
                    match keep {
                        Some(true) => {
                            // Static move into the callee.
                            let o = self.move_local(&n, r.clone(), a.span)?;
                            if !module_var {
                                moved_locals.push((n.clone(), self.locals[&n].reg, a.span));
                            }
                            out.push(Arg::new(o.reg, true, false, None));
                        }
                        Some(false) => {
                            if !module_var {
                                borrowed_roots.push(r.reg);
                            }
                            out.push(Arg::new(r.reg, false, !module_var, if module_var { None } else { Some(r.reg) }));
                        }
                        None => {
                            if module_var {
                                // Unknown callee: pass a borrow of the module
                                // variable; a `keep` parameter moves it out of
                                // its slot at runtime.
                                let mut arg = Arg::new(r.reg, false, false, None);
                                arg.global = self.m.gindex[&n];
                                out.push(arg);
                                if r.ty.is_copy() != Some(true) {
                                    self.set_st(&n, St::DynMoved);
                                }
                            } else {
                                out.push(Arg::new(r.reg, true, true, None));
                                if r.ty.is_copy() != Some(true) {
                                    self.set_st(&n, St::DynMoved);
                                }
                            }
                        }
                    }
                }
                Own::Borrowed => {
                    if keep == Some(true) {
                        let o = self.to_owned(r, a.span, "a `keep` parameter")?;
                        out.push(Arg::new(o.reg, true, false, None));
                    } else {
                        if let Some(root) = r.root {
                            borrowed_roots.push(root);
                        }
                        out.push(Arg::new(r.reg, false, false, r.root));
                    }
                }
            }
        }
        for (n, reg, s) in &moved_locals {
            if borrowed_roots.contains(reg) {
                return Err(own_err(format!("`{n}` is moved into this call while the same call also borrows it; pass `{n}++` instead"), *s));
            }
        }
        let _ = span;
        Ok(ArgList { args: out, pins: self.heap_borrows > 0 })
    }

    fn after_call(&mut self, al: &ArgList) {
        for a in &al.args {
            if a.owned && !a.local {
                self.emit(Instr::Drop { reg: a.reg });
            }
        }
    }

    fn method_call(&mut self, recv: &Expr, mname: &str, args: &[Expr], span: Span) -> R<Res> {
        let r = self.expr(recv)?;
        if mname != "some_or" {
            self.option_check(&r, &format!("calling `.{mname}()`"), span)?;
        }
        let mut sig: Option<Sig> = None;
        match &r.ty {
            Ty::Instance(c) => {
                if let Some(ci) = self.m.classes.get(c) {
                    match ci.methods.get(mname) {
                        Some(s) => sig = Some(s.clone()),
                        None => {
                            if !ci.fields.iter().any(|(n, _, _)| n == mname) {
                                return Err(Error::new("AttributeError", format!("class `{c}` has no method `{mname}`"), span));
                            }
                        }
                    }
                }
            }
            Ty::Trait(t) => {
                if let Some(tm) = self.m.traits.get(t) {
                    match tm.get(mname) {
                        Some(s) => sig = Some(s.clone()),
                        None => return Err(Error::new("AttributeError", format!("trait `{t}` has no method `{mname}`"), span)),
                    }
                }
            }
            _ => {}
        }
        if let Some(s) = &sig {
            if s.params.len() != args.len() {
                return Err(type_err(
                    format!("`{mname}()` takes {} argument{} but {} {} given", s.params.len(), if s.params.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "was" } else { "were" }),
                    span,
                ));
            }
        }
        let builtin_mask = match &r.ty {
            Ty::Str | Ty::List | Ty::Dict | Ty::Tuple | Ty::Generator | Ty::Array(_) => slim_vm::builtins::method_keep_mask(mname),
            _ => None,
        };
        let any_mask = if mname == "some_or" { Some(1u64 << 1) } else { None };
        let keep_of = |i: usize| -> Option<bool> {
            if let Some(s) = &sig {
                return s.params.get(i).map(|p| p.keep);
            }
            if let Some(m) = builtin_mask.or(any_mask) {
                return Some(m & (1 << (i + 1)) != 0);
            }
            None
        };
        let al = self.call_args(args, &keep_of, r.root, span)?;
        let dst = self.tmp_owned();
        let a = self.arglist(al.clone());
        let n = self.name_idx(mname);
        self.emit(Instr::CallMethod { dst, recv: r.reg, name: n, args: a });
        self.after_call(&al);
        if r.own == Own::Owned {
            self.emit(Instr::Drop { reg: r.reg });
        }
        let ty = match (&sig, mname, &r.ty) {
            (Some(s), _, _) => s.ret.clone(),
            (None, "upper" | "lower" | "strip" | "lstrip" | "rstrip" | "replace" | "join", Ty::Str) => Ty::Str,
            (None, "split", Ty::Str) | (None, "keys" | "values" | "items", Ty::Dict) => Ty::List,
            (None, "some_or", Ty::Option(inner)) => (**inner).clone(),
            _ => Ty::Unknown,
        };
        self.reg_ty.clear();
        Ok(Res { reg: dst, own: Own::Owned, ty, root: None, desc: "value".into() })
    }

    // ------------------------------------------------------------ return / raise

    fn return_stmt(&mut self, v: Option<&Expr>, span: Span) -> R<()> {
        if self.is_module {
            return Err(syn_err("`return` outside function", span));
        }
        let r = match v {
            Some(e) if self.is_generator => {
                let r = self.expr_owned(e, "the return value")?;
                r.reg
            }
            Some(e) => {
                let dst = self.tmp_owned();
                self.value_into(e, dst)?;
                dst
            }
            None => {
                let dst = self.tmp_owned();
                self.load_imm(dst, Value::UNIT);
                dst
            }
        };
        self.emit_return(r);
        Ok(())
    }

    fn emit_return(&mut self, r: Reg) {
        let finallys: Vec<Block> = self.finally_stack.iter().rev().cloned().collect();
        let saved = std::mem::take(&mut self.finally_stack);
        for f in &finallys {
            let _ = self.block(f);
        }
        self.finally_stack = saved;
        if !self.is_generator {
            match self.ret.clone() {
                RetKind::Typed(d) => {
                    let ti = self.type_idx(d);
                    self.emit(Instr::CheckType { reg: r, ty: ti, what: self.ret_desc });
                }
                RetKind::Unit => {
                    let ti = self.type_idx(TypeDesc::Unit);
                    self.emit(Instr::CheckType { reg: r, ty: ti, what: self.ret_desc });
                }
                RetKind::Any => {}
            }
        }
        self.emit(Instr::Return { src: r });
        self.state = None;
    }

    fn raise_stmt(&mut self, exc: Option<&Expr>, cause: Option<&Expr>, span: Span) -> R<()> {
        match exc {
            None => {
                let Some(&r) = self.handler_exc.last() else {
                    return Err(syn_err("bare `raise` is only allowed inside an `except` block", span));
                };
                self.emit(Instr::Reraise { exc: r });
            }
            Some(e) => {
                let r = self.expr_owned(e, "`raise`")?;
                let c = match cause {
                    Some(c) => self.expr_owned(c, "`raise ... from`")?.reg,
                    None => Reg::MAX,
                };
                self.emit(Instr::Raise { exc: r.reg, cause: c });
            }
        }
        self.state = None;
        Ok(())
    }

    // ------------------------------------------------------------ try

    fn try_expr(&mut self, t: &TryExpr, dst: Option<Reg>) -> R<Ty> {
        let dst = match dst {
            Some(d) => d,
            None => self.tmp_owned(),
        };
        let entry = self.state.clone();
        let log_start = self.moved_log.len();
        if let Some(f) = &t.finally {
            self.finally_stack.push(f.clone());
        }
        let start = self.here();
        let mut ty = self.block_value(&t.body, dst)?;
        let body_end = self.here();
        let mut end_jumps = Vec::new();
        // `else` runs only when the body did not raise (outside the handler range).
        let mut normal_state = self.state.take();
        let else_jump = self.jump();
        let handler_pc = self.here();
        // Handlers see every variable moved in the body as possibly moved.
        let mut hstate = join_opt(entry.clone(), normal_state.clone());
        if let Some(s) = &mut hstate {
            for n in &self.moved_log[log_start..] {
                if let Some(v) = s.get_mut(n) {
                    if v.st == St::Live {
                        v.st = St::DynMoved;
                    }
                }
            }
        }
        let mut handlers_state: Option<State> = None;
        let fin_start;
        if !t.handlers.is_empty() {
            self.exc_table.push(ExcEntry { start, end: body_end, handler: handler_pc });
            self.state = hstate.clone();
            let e = self.new_reg(true, Some("<exception>".into()));
            self.emit(Instr::TakeException { dst: e });
            let mut matches = Vec::new();
            let mut catch_all = None;
            for (i, h) in t.handlers.iter().enumerate() {
                if h.types.is_empty() {
                    if i != t.handlers.len() - 1 {
                        return Err(syn_err("a bare `except:` must be the last handler", h.span));
                    }
                    catch_all = Some(i);
                    continue;
                }
                for ty in &h.types {
                    let (_, d) = self.m.resolve_type(ty)?;
                    let ti = self.type_idx(d);
                    matches.push((i, self.emit(Instr::MatchException { exc: e, ty: ti, target: u32::MAX })));
                }
            }
            let mut targets = vec![u32::MAX; t.handlers.len()];
            match catch_all {
                Some(i) => {
                    let j = self.jump();
                    matches.push((i, j));
                }
                None => {
                    self.emit(Instr::Reraise { exc: e });
                }
            }
            for (i, h) in t.handlers.iter().enumerate() {
                self.state = hstate.clone();
                targets[i] = self.here();
                let exc_reg = match &h.name {
                    Some(n) => {
                        let ety = match h.types.as_slice() {
                            [TypeExpr::Named(n, _, _)] if self.m.classes.contains_key(n) || self.m.builtin_classes.contains(n) => Ty::Instance(n.clone()),
                            _ => Ty::Unknown,
                        };
                        let t = self.tmp_owned();
                        self.emit(Instr::Dup { dst: t, src: e });
                        self.bind(n, Res { reg: t, own: Own::Owned, ty: ety, root: None, desc: "value".into() }, h.span)?;
                        self.owned_used -= 1;
                        e
                    }
                    None => e,
                };
                self.handler_exc.push(exc_reg);
                let hty = self.block_value(&h.body, dst);
                self.handler_exc.pop();
                let hty = hty?;
                if self.state.is_some() {
                    ty = ty.join(&hty);
                    if let Some(n) = &h.name {
                        if let Some(l) = self.locals.get(n) {
                            let reg = l.reg;
                            self.emit(Instr::Drop { reg });
                            self.set_st(n, St::Uninit);
                        }
                    }
                    self.emit(Instr::Drop { reg: e });
                }
                handlers_state = join_opt(handlers_state, self.state.take());
                end_jumps.push(self.jump());
            }
            for (i, at) in matches {
                self.patch(at, targets[i]);
            }
        }
        fin_start = self.here();
        let h = self.here();
        self.patch(else_jump, h);
        self.state = normal_state.take();
        if let Some(b) = &t.else_ {
            ty = self.block_value(b, dst)?;
        }
        let after_else = self.state.take();
        let finally_block = if t.finally.is_some() { self.finally_stack.pop() } else { None };
        let normal_end = self.here();
        let h = self.here();
        for at in end_jumps.drain(..) {
            self.patch(at, h);
        }
        self.state = join_opt(after_else, handlers_state);
        if let Some(f) = &finally_block {
            // Normal path: run the finally body inline.
            self.block(f)?;
            let skip = self.jump();
            // Exceptional path: any exception from the body, handlers or else.
            let fin_handler = self.here();
            self.exc_table.push(ExcEntry { start, end: normal_end, handler: fin_handler });
            let saved = self.state.clone();
            let mut st = join_opt(entry, saved.clone());
            if let Some(s) = &mut st {
                for n in &self.moved_log[log_start..] {
                    if let Some(v) = s.get_mut(n) {
                        if v.st == St::Live {
                            v.st = St::DynMoved;
                        }
                    }
                }
            }
            self.state = st;
            let e2 = self.new_reg(true, Some("<exception>".into()));
            self.emit(Instr::TakeException { dst: e2 });
            self.block(f)?;
            self.emit(Instr::Reraise { exc: e2 });
            let h = self.here();
            self.patch(skip, h);
            self.state = saved;
        }
        let _ = fin_start;
        Ok(ty)
    }

    // ------------------------------------------------------------ definitions

    fn def_stmt(&mut self, f: &FuncDef) -> R<()> {
        let dst = self.tmp_owned();
        if self.is_module {
            let code = compile_function(self.m, f, None, &[])?;
            self.codes.push(Arc::new(code));
            let ci = (self.codes.len() - 1) as u32;
            let al = self.arglist(ArgList::default());
            self.emit(Instr::MakeFunction { dst, code: ci, captures: al });
            let idx = self.m.gindex[&f.name];
            self.emit(Instr::StoreGlobal { idx, src: dst });
            self.set_var(&f.name, St::Live, Ty::StaticFn);
            return Ok(());
        }
        self.make_closure(f, dst)?;
        let r = Res { reg: dst, own: Own::Owned, ty: Ty::Func, root: None, desc: "value".into() };
        self.bind(&f.name, r, f.span)
    }

    /// Compiles a nested function or lambda; captured locals move into it.
    fn make_closure(&mut self, f: &FuncDef, dst: Reg) -> R<()> {
        let body = f.body.as_ref().unwrap();
        let mut used = HashSet::new();
        scan::used_names(body, &mut used);
        let mut own_names: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
        scan::assigned_names(body, &mut own_names);
        let mut caps: Vec<(String, Ty)> = Vec::new();
        let mut cap_args = Vec::new();
        let mut names: Vec<&String> = used.iter().collect();
        names.sort();
        for n in names {
            if own_names.contains(n) {
                continue;
            }
            if let Some(l) = self.locals.get(n) {
                let (reg, kind) = (l.reg, l.kind);
                self.check_use(n, Some(reg), f.span)?;
                let ty = self.st(n).map(|v| v.ty.clone()).unwrap_or(Ty::Unknown);
                let r = Res {
                    reg,
                    own: if kind == LocalKind::Borrowed { Own::Borrowed } else { Own::Local(n.clone()) },
                    ty: ty.clone(),
                    root: Some(reg),
                    desc: format!("`{n}`"),
                };
                let o = self.to_owned(r, f.span, "a closure")?;
                cap_args.push(Arg::new(o.reg, true, false, None));
                caps.push((n.clone(), ty));
            } else if let Some((_, ty)) = self.captures.iter().find(|(c, _)| c == n).cloned() {
                let r = self.name_expr(n, f.span)?;
                let o = self.to_owned(r, f.span, "a closure")?;
                cap_args.push(Arg::new(o.reg, true, false, None));
                caps.push((n.clone(), ty));
            }
        }
        let code = compile_function(self.m, f, None, &caps)?;
        self.codes.push(Arc::new(code));
        let ci = (self.codes.len() - 1) as u32;
        let al = self.arglist(ArgList { args: cap_args, pins: false });
        self.emit(Instr::MakeFunction { dst, code: ci, captures: al });
        Ok(())
    }

    fn class_stmt(&mut self, c: &ClassDef) -> R<()> {
        if !self.is_module {
            return Err(syn_err("classes must be defined at module level", c.span));
        }
        let info = &self.m.classes[&c.name];
        let mut methods = Vec::new();
        for m in &c.methods {
            if m.name == "__init__" {
                check_init_assigns(c, m)?;
            }
            let code = compile_function(self.m, m, Some(&c.name), &[])?;
            self.codes.push(Arc::new(code));
            methods.push((m.name.clone(), (self.codes.len() - 1) as u32));
        }
        if !c.methods.iter().any(|m| m.name == "__init__") && !c.fields.is_empty() {
            return Err(syn_err(format!("class `{}` has fields and needs an `__init__` that assigns them", c.name), c.span));
        }
        let fields = info.fields.iter().map(|(n, _, d)| FieldSpec { name: n.clone(), ty: d.clone() }).collect();
        let traits = c.traits.iter().map(|(t, _)| (self.m.gindex[t], t.clone())).collect();
        self.classes.push(ClassSpec { name: c.name.clone(), fields, methods, traits });
        let dst = self.tmp_owned();
        self.emit(Instr::MakeClass { dst, spec: (self.classes.len() - 1) as u32 });
        let idx = self.m.gindex[&c.name];
        self.emit(Instr::StoreGlobal { idx, src: dst });
        self.set_var(&c.name, St::Live, Ty::Class(c.name.clone()));
        Ok(())
    }

    fn trait_stmt(&mut self, t: &TraitDef) -> R<()> {
        if !self.is_module {
            return Err(syn_err("traits must be defined at module level", t.span));
        }
        let mut methods = Vec::new();
        for m in &t.methods {
            let default = match &m.body {
                Some(_) => {
                    let code = compile_function(self.m, m, None, &[])?;
                    self.codes.push(Arc::new(code));
                    Some((self.codes.len() - 1) as u32)
                }
                None => None,
            };
            methods.push((m.name.clone(), default));
        }
        self.traits.push(TraitSpec { name: t.name.clone(), methods });
        let dst = self.tmp_owned();
        self.emit(Instr::MakeTrait { dst, spec: (self.traits.len() - 1) as u32 });
        let idx = self.m.gindex[&t.name];
        self.emit(Instr::StoreGlobal { idx, src: dst });
        self.set_var(&t.name, St::Live, Ty::Unknown);
        Ok(())
    }
}

/// Whether a value of static type `v` may be stored in a slot declared `d`.
fn ty_compatible(d: &Ty, v: &Ty) -> bool {
    match (d, v) {
        (Ty::Unknown, _) | (_, Ty::Unknown) => true,
        (Ty::Option(inner), Ty::NoneT) => { let _ = inner; true }
        (Ty::Option(a), Ty::Option(b)) => ty_compatible(a, b),
        (Ty::Option(a), b) => ty_compatible(a, b),
        (_, Ty::NoneT) => false,
        (Ty::Float, Ty::Int) => true,
        (Ty::BigInt, Ty::Int) => true,
        (Ty::Trait(_), Ty::Instance(_)) | (Ty::Trait(_), Ty::Trait(_)) => true,
        (Ty::Iter, Ty::Generator | Ty::Iter) => true,
        (Ty::Func, Ty::StaticFn) | (Ty::StaticFn, Ty::Func) => true,
        (a, b) => a == b,
    }
}

fn short(desc: &str) -> String {
    // "field `x`" -> "obj.x"-style hint; "parameter `p`" -> "p"
    match desc.split('`').nth(1) {
        Some(n) if desc.starts_with("field") => format!("obj.{n}"),
        Some(n) => n.to_string(),
        None => "value".into(),
    }
}

fn callee_name(f: &Expr) -> String {
    match &f.kind {
        ExprKind::Name(n) => format!("{n}()"),
        _ => "function".into(),
    }
}

fn binop_ty(op: BinOp, a: &Ty, b: &Ty) -> Ty {
    use Ty::*;
    match (op, a, b) {
        (BinOp::Div, Int | Float, Int | Float) => Float,
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow, Int, Int) => Int,
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow, Float, Int | Float)
        | (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow, Int, Float) => Float,
        (BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr, Int, Int) => Int,
        (_, BigInt, Int | BigInt) | (_, Int, BigInt) => BigInt,
        (BinOp::Add, Str, Str) | (BinOp::Mul, Str, Int) => Str,
        (BinOp::Add, List, List) | (BinOp::Mul, List, Int) => List,
        (BinOp::Add, Tuple, Tuple) => Tuple,
        _ => Unknown,
    }
}

/// Fields a block definitely assigns through `self.f = ...` / `self.f <- ...`.
fn definitely_assigned(b: &Block, out: &mut HashSet<String>) {
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Assign { target, value } | StmtKind::AnnAssign { target, value: Some(value), .. } => {
                field_target(target, out);
                if let ExprKind::If { branches, else_ } = &value.kind {
                    if_assigns(branches, else_.as_ref(), out);
                }
            }
            StmtKind::Move { targets, .. } => targets.iter().for_each(|t| field_target(t, out)),
            StmtKind::Expr(Expr { kind: ExprKind::If { branches, else_ }, .. }) => {
                if_assigns(branches, else_.as_ref(), out)
            }
            _ => {}
        }
    }
}

fn if_assigns(branches: &[(Expr, Block)], else_: Option<&Block>, out: &mut HashSet<String>) {
    let Some(e) = else_ else { return };
    let mut common: Option<HashSet<String>> = None;
    for b in branches.iter().map(|(_, b)| b).chain(std::iter::once(e)) {
        let mut s = HashSet::new();
        definitely_assigned(b, &mut s);
        common = Some(match common {
            None => s,
            Some(c) => c.intersection(&s).cloned().collect(),
        });
    }
    out.extend(common.unwrap_or_default());
}

fn field_target(t: &Expr, out: &mut HashSet<String>) {
    if let ExprKind::Attr(o, f) = &t.kind {
        if matches!(&o.kind, ExprKind::Name(n) if n == "self") {
            out.insert(f.clone());
        }
    }
}

fn check_init_assigns(c: &ClassDef, init: &FuncDef) -> R<()> {
    let mut assigned = HashSet::new();
    definitely_assigned(init.body.as_ref().unwrap(), &mut assigned);
    for f in &c.fields {
        if !assigned.contains(&f.name) {
            return Err(Error::new(
                "TypeError",
                format!("`{}.__init__` must assign field `{}` on every path", c.name, f.name),
                init.span,
            ));
        }
    }
    Ok(())
}

/// Compiles a function, method (`class_name`) or closure (`captures`).
pub fn compile_function(m: &ModInfo, f: &FuncDef, class_name: Option<&str>, captures: &[(String, Ty)]) -> R<Code> {
    let name = match class_name {
        Some(c) => format!("{c}.{}", f.name),
        None => f.name.clone(),
    };
    let mut ctx = FnCtx::new(m, name, false);
    ctx.class_name = class_name.map(String::from);
    ctx.captures = captures.to_vec();
    let body = f.body.as_ref().unwrap();
    ctx.is_generator = scan::has_yield(body);
    scan::global_decls(body, &mut ctx.global_decls);
    let mut state = State::new();
    let mut params = Vec::new();
    let mut checks = Vec::new();
    for (i, p) in f.params.iter().enumerate() {
        let is_self = i == 0 && p.name == "self" && (class_name.is_some() || ctx_is_trait_method(f));
        let (ty, desc) = match (&p.ty, is_self) {
            (_, true) => (class_name.map(|c| Ty::Instance(c.to_string())).unwrap_or(Ty::Unknown), None),
            (Some(t), _) => {
                let (ty, d) = m.resolve_type(t)?;
                (ty, Some(d))
            }
            (None, _) => (Ty::Unknown, None),
        };
        if ctx.is_generator && !p.keep && ty.is_copy() != Some(true) && !is_self {
            return Err(own_err(
                format!("generator `{}` outlives its call: move-type parameter `{}` must be declared `+ keep`", f.name, p.name),
                p.span,
            ));
        }
        if f.is_lambda && false {
            unreachable!();
        }
        let kind = if p.keep { LocalKind::Owned } else { LocalKind::Borrowed };
        let reg = ctx.new_reg(p.keep, Some(p.name.clone()));
        ctx.locals.insert(p.name.clone(), Local { reg, kind, decl: desc.clone().map(|d| (ty.clone(), d)) });
        state.insert(p.name.clone(), Var { st: St::Live, ty: ty.clone() });
        if let Some(d) = &desc {
            checks.push((reg, d.clone(), p.name.clone()));
        }
        params.push(ParamInfo { name: p.name.clone(), keep: p.keep, ty: desc });
    }
    let mut local_names = Vec::new();
    scan::assigned_names(body, &mut local_names);
    for n in local_names {
        if ctx.locals.contains_key(&n) || ctx.global_decls.contains(&n) {
            continue;
        }
        let reg = ctx.new_reg(true, Some(n.clone()));
        ctx.locals.insert(n, Local { reg, kind: LocalKind::Owned, decl: None });
    }
    ctx.state = Some(state);
    let fname = if f.is_lambda { "lambda".to_string() } else { format!("`{}()`", ctx.name) };
    for (reg, d, pname) in checks {
        let ti = ctx.type_idx(d);
        let w = ctx.name_idx(&format!("argument `{pname}` of {fname}"));
        ctx.emit(Instr::CheckType { reg, ty: ti, what: w });
    }
    let (ret, ret_ty) = match &f.ret {
        Some(TypeExpr::Unit(_)) => (RetKind::Unit, Ty::Unit),
        Some(t) => {
            let (ty, d) = m.resolve_type(t)?;
            (RetKind::Typed(d), ty)
        }
        None => (RetKind::Any, Ty::Unknown),
    };
    ctx.ret = if ctx.is_generator { RetKind::Any } else { ret.clone() };
    ctx.ret_ty = ret_ty;
    ctx.ret_desc = ctx.name_idx(&format!("return value of {fname}"));
    ctx.line = f.span.line;
    let dst = ctx.tmp_owned();
    if ctx.is_generator {
        ctx.block(body)?;
        if ctx.state.is_some() {
            ctx.load_imm(dst, Value::UNIT);
            ctx.emit(Instr::Return { src: dst });
        }
    } else if ret == RetKind::Unit {
        // `-> ()`: the last line's value is discarded.
        ctx.block(body)?;
        if ctx.state.is_some() {
            ctx.load_imm(dst, Value::UNIT);
            ctx.emit(Instr::Return { src: dst });
        }
    } else {
        ctx.block_value(body, dst)?;
        if ctx.state.is_some() {
            ctx.emit_return(dst);
        }
    }
    let is_method = class_name.is_some();
    Ok(ctx.finish(params, ret, ctx_is_gen(body), is_method))
}

fn ctx_is_gen(b: &Block) -> bool {
    scan::has_yield(b)
}

fn ctx_is_trait_method(f: &FuncDef) -> bool {
    f.params.first().is_some_and(|p| p.name == "self" && p.ty.is_none()) && !f.is_lambda
}
