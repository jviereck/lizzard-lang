//! Type checking of a Lizzard module and lowering of its functions to IR.
//!
//! One pass does both: every expression is typed and lowered together.
//! Top-level statements (run by CPython) are checked the same way into a
//! throwaway function. Methods are lowered like functions, under their
//! qualified names (`C.m`).

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

/// Jumps to patch at the end of a loop.
struct Loop {
    breaks: Vec<usize>,
    continues: Vec<usize>,
    /// `cleanups.len()` outside the loop: `break` and `continue` run the
    /// cleanups above it.
    depth: usize,
}

/// Code that must run when control leaves a `try`/`with` early (`return`,
/// `break`, `continue`).
#[derive(Clone)]
enum Cleanup {
    /// A `finally` body.
    Finally(Vec<Stmt>),
    /// Leaving an `except` block: restore the previously handled exception.
    Restore(Reg),
    /// Leaving a `with` block: call `__exit__(None, None, None)`.
    WithExit(Reg),
}

/// A protected range being emitted: its handler, where it (or its current
/// segment) starts, and `cleanups.len()` when it was opened.
struct Region {
    handler: usize,
    start: u32,
    depth: usize,
}

/// What a comprehension produces at its innermost level.
enum CompBody<'e> {
    List(&'e Expr),
    Dict(&'e Expr, &'e Expr),
}

/// The class whose method is being lowered.
#[derive(Clone)]
struct MethodCtx {
    class: ClassId,
    /// The receiver parameter (`self` / `cls`), if the method has one.
    receiver: Option<String>,
    is_init: bool,
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
    loops: Vec<Loop>,
    method: Option<MethodCtx>,
    /// Narrowed types of locals (`x`) and their attributes (`x.attr`), where
    /// they differ from the declared ones.
    narrow: HashMap<String, Ty>,
    /// Names declared `global` in this function.
    global_names: Vec<String>,
    /// Names the function assigns anywhere: they are local throughout.
    bound: Vec<String>,
    handlers: Vec<HandlerIr>,
    regions: Vec<Region>,
    cleanups: Vec<Cleanup>,
    /// The exceptions being handled, innermost last (for a bare `raise`).
    handled: Vec<Reg>,
}

const BUILTIN_FNS: &[&str] = &["print", "len", "type"];

pub fn check_module(db: &mut Db, m: ModId, body: &[Stmt]) -> Vec<FuncIr> {
    let file = db.modules[m].file.expect("Lizzard module without file");
    let mut funcs = Vec::new();
    let mut top = Lower::new(db, m, file, true, "<module>".into(), Ty::Any);
    // The top level first: it gives the module's globals their types.
    for s in body {
        if !matches!(s, Stmt::FunctionDef(_) | Stmt::ClassDef(_)) {
            top.stmt(s);
        }
    }
    let globals: Vec<(String, Ty)> = top.locals.iter().map(|(n, l)| (n.clone(), l.ty.clone())).collect();
    for (n, t) in globals {
        top.db.globals.insert((m, n), t);
    }
    for s in body {
        match s {
            Stmt::FunctionDef(d) => {
                if let Some(f) = top.function(d, None) {
                    funcs.push(f);
                }
            }
            Stmt::ClassDef(c) => funcs.extend(top.class_def(c)),
            _ => {}
        }
    }
    funcs
}

fn is_dunder(name: &str) -> bool {
    name.len() > 4 && name.starts_with("__") && name.ends_with("__")
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
            loops: Vec::new(),
            method: None,
            narrow: HashMap::new(),
            global_names: Vec::new(),
            bound: Vec::new(),
            handlers: Vec::new(),
            regions: Vec::new(),
            cleanups: Vec::new(),
            handled: Vec::new(),
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
            Instr::Jump { target } | Instr::JumpIfFalse { target, .. } | Instr::JumpIfTrue { target, .. } | Instr::ForIter { target, .. } => *target = to,
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

    fn const_idx(&mut self, c: Const) -> u32 {
        self.consts.push(c);
        (self.consts.len() - 1) as u32
    }

    fn konst(&mut self, c: Const, ty: Ty) -> Val {
        let idx = self.const_idx(c);
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::Const { dst, idx });
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

    fn builtin_inst(&mut self, class: &str, args: Vec<Ty>) -> Ty {
        match self.db.builtin_class(class) {
            Some(c) => Ty::Inst(c, args),
            None => Ty::Any,
        }
    }

    /// The type arguments of `t` if it is the builtin class `class`.
    fn builtin_args(&mut self, t: &Ty, class: &str) -> Option<Vec<Ty>> {
        let Ty::Inst(c, args) = t else { return None };
        (Some(*c) == self.db.builtin_class(class)).then(|| args.clone())
    }

    /// The class type whose members a value of type `t` has.
    fn lookup_ty(&mut self, t: &Ty) -> Ty {
        match t {
            Ty::Var(_) => self.db.upper(t),
            Ty::Str | Ty::Tuple(_) | Ty::TupleVar(_) => self.db.nominal(t).unwrap_or(Ty::Any),
            t => t.clone(),
        }
    }

    /// The least type both are assignable to: one of them, a common base
    /// class, or `object`.
    fn join(&mut self, a: &Ty, b: &Ty) -> Ty {
        if a == b {
            return a.clone();
        }
        if matches!(a, Ty::Any) || matches!(b, Ty::Any) {
            return Ty::Any;
        }
        if self.assignable(a, b) {
            return b.clone();
        }
        if self.assignable(b, a) {
            return a.clone();
        }
        if let Ty::Inst(c, _) = a {
            for k in self.db.mro(*c) {
                if self.db.class_params(k).is_empty() {
                    let cand = Ty::Inst(k, Vec::new());
                    if self.assignable(b, &cand) {
                        return cand;
                    }
                }
            }
        }
        Ty::Object
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
            // A rigid variable (`Self`, a class's type parameter) is checked
            // against its bound.
            Ty::Union(v) => {
                let cs: Vec<RtCheck> = v.iter().map(|m| self.rtcheck(m)).collect();
                if cs.contains(&RtCheck::Any) { RtCheck::Any } else { RtCheck::Union(cs) }
            }
            Ty::Var(v) if self.db.rigid.contains(v) => match self.db.tvs[*v].bound.clone() {
                Some(b) => self.rtcheck(&b),
                None => RtCheck::Any,
            },
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
        (file != self.file && self.db.files[file].path != BUILTINS_PATH).then(|| self.db.decl_origin(file, off))
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

    fn load_global(&mut self, name: &str, ty: Ty) -> Val {
        let dst = self.reg(Repr::Obj);
        let n = self.name(name);
        self.emit(Instr::LoadGlobal { dst, name: n });
        Val { reg: dst, ty, repr: Repr::Obj }
    }

    // ---- classes ----

    fn class_def(&mut self, cd: &ast::StmtClassDef) -> Vec<FuncIr> {
        let off = self.at(cd);
        let name = cd.name.as_str();
        let Some(Sym::Class(c)) = self.db.lookup_in(self.m, name) else {
            return Vec::new();
        };
        if self.db.user_class_def(c).is_none_or(|d| d.range() != cd.range()) {
            self.err(off, format!("`{name}` is defined more than once"));
            return Vec::new();
        }
        if !cd.decorator_list.is_empty() {
            self.err(off, "class decorators are not supported by Lizzard yet");
        }
        if let Some(args) = &cd.arguments {
            if !args.keywords.is_empty() {
                self.err(off, "class keyword arguments (`metaclass=...`) are not supported by Lizzard yet");
            }
            for b in args.args.iter() {
                let target = match b {
                    Expr::Subscript(s) => &*s.value,
                    e => e,
                };
                if matches!(self.db.sym_of_expr(self.m, target), Some(Sym::Special(_))) {
                    continue;
                }
                let tps = self.db.class_tparams(c);
                let t = self.db.eval_type(b, &Scope { m: self.m, tparams: &tps, self_ty: None });
                if !matches!(t, Ty::Inst(..) | Ty::Exc(_) | Ty::Object | Ty::Any) {
                    let ts = self.show(&t);
                    self.err(b.range().start().into(), format!("subclassing {ts} is not supported by Lizzard yet"));
                }
            }
        }
        self.db.mro(c);
        let mut funcs = Vec::new();
        let mut init = None;
        for s in &cd.body {
            let soff = self.at(s);
            match s {
                Stmt::FunctionDef(d) => {
                    if d.name.as_str() == "__init__" {
                        init = Some(d);
                    }
                    if let Some(f) = self.function(d, Some(c)) {
                        funcs.push(f);
                    }
                    self.check_override(c, d.name.as_str(), soff);
                }
                Stmt::AnnAssign(a) => {
                    let Expr::Name(n) = &*a.target else {
                        self.err(soff, "this class attribute declaration is not supported by Lizzard yet");
                        continue;
                    };
                    if let Some(v) = &a.value {
                        if let Some(Member::Attr(t)) = self.db.class_member(c, n.id.as_str()) {
                            let v = self.expr_hint(v, Some(&t));
                            let an = n.id.to_string();
                            self.coerce(v, &t, soff, |_, got, want| format!("cannot assign {got} to class attribute `{an}` of type {want}"));
                        }
                    }
                    self.check_override(c, n.id.as_str(), soff);
                }
                Stmt::Assign(a) => {
                    let [Expr::Name(n)] = a.targets.as_slice() else {
                        self.err(soff, "this class attribute assignment is not supported by Lizzard yet");
                        continue;
                    };
                    match self.db.class_member(c, n.id.as_str()) {
                        Some(Member::Attr(Ty::Any)) | None => {
                            self.err(soff, format!("class attribute `{}` needs a type annotation", n.id.as_str()));
                        }
                        _ => {
                            self.expr(&a.value);
                        }
                    }
                    self.check_override(c, n.id.as_str(), soff);
                }
                Stmt::Pass(_) => {}
                Stmt::Expr(e) if matches!(&*e.value, Expr::StringLiteral(_) | Expr::EllipsisLiteral(_)) => {}
                Stmt::ClassDef(_) => self.err(soff, "nested classes are not supported by Lizzard yet"),
                _ => self.err(soff, "this statement is not supported in a class body by Lizzard yet"),
            }
        }
        // Attributes declared in `__init__` (`self.x: T = ...`) override too.
        if let Some(init) = init {
            let mut names = Vec::new();
            init_attr_names(&init.body, &mut names);
            for (n, aoff) in names {
                self.check_override(c, &n, aoff);
            }
        }
        funcs
    }

    /// Check that `name` declared by class `c` is compatible with the member
    /// it overrides, if any.
    fn check_override(&mut self, c: ClassId, name: &str, off: u32) {
        if name == "__init__" {
            return;
        }
        let (Some(own), Some((owner, base))) = (self.db.class_member(c, name), self.db.find_member_after(c, Some(c), name)) else {
            return;
        };
        let cname = self.db.classes[c].name.clone();
        let bname = format!("{}.{name}", self.db.classes[owner].name);
        // The base's type parameters as `c` instantiates them, and its
        // `Self` as `c`'s.
        let cps = self.db.class_params(c);
        let ops = self.db.class_params(owner);
        let mut s = Subst::new();
        self.db.assign(&Ty::Inst(c, cps.iter().map(|p| Ty::Var(*p)).collect()), &Ty::Inst(owner, ops.iter().map(|p| Ty::Var(*p)).collect()), &mut s, &mut false);
        let (cself, oself) = (self.db.self_tv(c), self.db.self_tv(owner));
        s.insert(oself, Ty::Var(cself));
        let saved = std::mem::replace(&mut self.db.rigid, cps.iter().copied().chain([cself]).collect());
        let msg = match (own, base) {
            (Member::Attr(t), Member::Attr(bt)) => {
                let bt = deep_subst(&bt, &s);
                (!(self.assignable(&t, &bt) && self.assignable(&bt, &t))).then(|| format!("`{cname}.{name}` is declared as {}, but `{bname}` is {}", self.show(&t), self.show(&bt)))
            }
            (Member::Method(f), Member::Method(bf)) | (Member::Property(f), Member::Property(bf)) => self.sig_override(f, bf, &s, &format!("{cname}.{name}"), &bname),
            (own, _) => {
                let kind = |m: &Member| match m {
                    Member::Attr(_) => "an attribute",
                    Member::Method(_) => "a method",
                    Member::Property(_) => "a property",
                };
                Some(format!("`{cname}.{name}` is {}, but `{bname}` is {}", kind(&own), kind(&self.db.find_member_after(c, Some(c), name).unwrap().1)))
            }
        };
        self.db.rigid = saved;
        if let Some(m) = msg {
            self.err(off, m);
        }
    }

    fn sig_override(&mut self, f: FuncId, bf: FuncId, s: &Subst, own: &str, base: &str) -> Option<String> {
        let (sigs, bsigs) = (self.db.sigs(f), self.db.sigs(bf));
        let ([sig], [bsig]) = (sigs.as_slice(), bsigs.as_slice()) else { return None };
        let (kind, bkind) = (self.db.method_kind(f), self.db.method_kind(bf));
        if kind != bkind {
            return Some(format!("`{own}` and `{base}` are different kinds of methods (instance, class or static)"));
        }
        let skip = usize::from(kind != MethodKind::Static);
        let ps: Vec<Param> = sig.params.iter().skip(skip).cloned().collect();
        let bps: Vec<Param> = bsig.params.iter().skip(skip).cloned().collect();
        let required = ps.iter().filter(|p| !p.default).count();
        if ps.len() < bps.len() || required > bps.len() {
            return Some(format!("`{own}` takes {} arguments, but `{base}` takes {}: an override must accept the same arguments", ps.len(), bps.len()));
        }
        for (p, bp) in ps.iter().zip(&bps) {
            let bt = deep_subst(&bp.ty, s);
            if !self.assignable(&bt, &p.ty) {
                return Some(format!("parameter `{}` of `{own}` is {}, but `{base}` accepts {}: an override must accept it", p.name, self.show(&p.ty), self.show(&bt)));
            }
        }
        let bret = deep_subst(&bsig.ret, s);
        if !self.assignable(&sig.ret, &bret) {
            return Some(format!("`{own}` returns {}, but `{base}` returns {}: an override must return a compatible type", self.show(&sig.ret), self.show(&bret)));
        }
        None
    }

    // ---- functions ----

    fn function(&mut self, d: &ast::StmtFunctionDef, owner: Option<ClassId>) -> Option<FuncIr> {
        let off = self.at(d);
        let name = d.name.as_str();
        if d.is_async || d.type_params.is_some() {
            self.err(off, "async and generic functions are not supported by Lizzard yet");
            return None;
        }
        for dec in &d.decorator_list {
            let ok = owner.is_some() && matches!(&dec.expression, Expr::Name(n) if matches!(n.id.as_str(), "staticmethod" | "classmethod" | "property"));
            if !ok {
                let what = if owner.is_some() { "decorators other than @staticmethod, @classmethod and @property" } else { "decorated functions" };
                self.err(dec.range().start().into(), format!("{what} are not supported by Lizzard yet"));
                return None;
            }
        }
        let f = match owner {
            None => match self.db.lookup_in(self.m, name) {
                Some(Sym::Func(f)) => f,
                _ => return None,
            },
            Some(c) => match self.db.class_member(c, name) {
                Some(Member::Method(f) | Member::Property(f)) => f,
                _ => return None,
            },
        };
        let sigs = self.db.sigs(f);
        if sigs.len() != 1 || self.db.def_count(f) != 1 {
            self.err(off, format!("`{name}` is defined more than once"));
            return None;
        }
        let sig = sigs[0].clone();
        let kind = self.db.method_kind(f);
        let qual = match owner {
            Some(c) => format!("{}.{name}", self.db.classes[c].name),
            None => name.to_string(),
        };
        if owner.is_some() && name == "__init__" && sig.ret != Ty::None {
            self.err(d.name.range().start().into(), "`__init__` must return None");
        }
        let line = self.db.files[self.file].line_col(off).0;
        let path = self.db.files[self.file].path.clone();
        let saved_rigid = self.db.rigid.clone();
        if let Some(c) = owner {
            let mut rigid = self.db.class_params(c);
            rigid.push(self.db.self_tv(c));
            self.db.rigid = rigid;
        }
        let mut l = Lower::new(self.db, self.m, self.file, false, qual, sig.ret.clone());
        l.at(d);
        let mut keys = Vec::new();
        assigned_keys(&d.body, &mut keys);
        l.bound = keys.into_iter().filter(|k| !k.contains('.')).collect();
        let has_receiver = owner.is_some() && kind != MethodKind::Static;
        l.method = owner.map(|c| MethodCtx { class: c, receiver: if has_receiver { sig.params.first().map(|p| p.name.clone()) } else { None }, is_init: name == "__init__" });
        // Literal defaults, in order of the parameters that have them.
        let raw_defaults: Vec<Option<&Expr>> = d.parameters.posonlyargs.iter().chain(d.parameters.args.iter()).map(|p| p.default.as_deref()).collect();
        let mut params = Vec::new();
        for (i, p) in sig.params.iter().enumerate() {
            if !matches!(p.kind, PKind::PosOnly | PKind::PosOrKw) {
                l.err(off, format!("parameter `{}`: only positional parameters are supported by Lizzard yet", p.name));
            }
            let reg = l.reg(Repr::Obj);
            l.locals.insert(p.name.clone(), Local { reg, ty: p.ty.clone(), repr: Repr::Obj });
            let expected = l.show(&p.ty);
            // The receiver is bound by Python itself.
            let check = if i == 0 && has_receiver { RtCheck::Any } else { l.rtcheck(&p.ty) };
            l.checks.push(CheckIr { check, what: format!("argument '{}'", p.name), expected, origin: None });
            let default = match raw_defaults.get(i).copied().flatten() {
                Some(e) => l.default_value(e, &p.ty, &p.name),
                None => None,
            };
            params.push(ParamIr { name: p.name.clone(), check: (l.checks.len() - 1) as u32, default });
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
                l.err(off, format!("`{name}` may end without returning a value (declared to return {want})"));
            }
        } else {
            // Unreachable when the analysis is right; never run off the end.
            let none = l.konst(Const::None, Ty::None);
            l.emit(Instr::Return { src: none.reg });
        }
        let ir = FuncIr {
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
            handlers: l.handlers,
        };
        self.db.rigid = saved_rigid;
        Some(ir)
    }

    /// A parameter's default value: a literal of the parameter's type.
    fn default_value(&mut self, e: &Expr, ty: &Ty, pname: &str) -> Option<u32> {
        let off: u32 = e.range().start().into();
        let (c, t) = match e {
            Expr::NumberLiteral(n) => match &n.value {
                ast::Number::Int(i) => (i.as_i64().map_or_else(|| Const::BigInt(i.to_string()), Const::Int), Ty::Int),
                ast::Number::Float(f) => (Const::Float(*f), Ty::Float),
                ast::Number::Complex { .. } => (Const::None, Ty::Any),
            },
            Expr::UnaryOp(u) if u.op == ast::UnaryOp::USub => match &*u.operand {
                Expr::NumberLiteral(n) => match &n.value {
                    ast::Number::Int(i) if i.as_i64().is_some() => (Const::Int(-i.as_i64().unwrap()), Ty::Int),
                    ast::Number::Float(f) => (Const::Float(-f), Ty::Float),
                    _ => (Const::None, Ty::Any),
                },
                _ => (Const::None, Ty::Any),
            },
            Expr::StringLiteral(s) => (Const::Str(s.value.to_str().to_string()), Ty::Str),
            Expr::BooleanLiteral(b) => (Const::Bool(b.value), Ty::Bool),
            Expr::NoneLiteral(_) => (Const::None, Ty::None),
            _ => {
                self.err(off, format!("parameter `{pname}`: only literal default values are supported by Lizzard yet"));
                return None;
            }
        };
        if t == Ty::Any {
            self.err(off, format!("parameter `{pname}`: only literal default values are supported by Lizzard yet"));
            return None;
        }
        if !self.assignable(&t, ty) {
            let (got, want) = (self.show(&t), self.show(ty));
            self.err(off, format!("the default of `{pname}` is {got}, but the parameter is {want}"));
        }
        Some(self.const_idx(c))
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
                    return self.err(off, "multiple assignment targets are not supported by Lizzard yet");
                };
                let hint = self.target_hint(target);
                let v = self.expr_hint(&a.value, hint.as_ref());
                self.assign_target(target, v, off);
            }
            Stmt::AnnAssign(a) => self.ann_assign(a, off),
            Stmt::Global(g) if !self.toplevel => {
                for n in &g.names {
                    let n = n.as_str();
                    if self.locals.contains_key(n) {
                        self.err(off, format!("`{n}` is assigned before its `global` declaration"));
                    } else if !self.db.globals.contains_key(&(self.m, n.to_string())) {
                        self.err(off, format!("`{n}` must be assigned at the top level of the module to be used as a global"));
                    }
                    self.global_names.push(n.to_string());
                }
            }
            Stmt::Global(_) => {}
            Stmt::AugAssign(a) => self.aug_assign(a, off),
            Stmt::If(i) => self.if_stmt(&i.test, &i.body, &i.elif_else_clauses),
            Stmt::For(f) => self.for_stmt(f),
            Stmt::While(w) => self.while_stmt(w),
            Stmt::Break(_) | Stmt::Continue(_) => {
                if self.loops.is_empty() {
                    let what = if matches!(s, Stmt::Break(_)) { "break" } else { "continue" };
                    return self.err(off, format!("`{what}` outside a loop"));
                }
                let depth = self.loops.last().unwrap().depth;
                self.emit_cleanups(depth);
                let j = self.emit(Instr::Jump { target: 0 });
                let l = self.loops.last_mut().unwrap();
                if matches!(s, Stmt::Break(_)) {
                    l.breaks.push(j);
                } else {
                    l.continues.push(j);
                }
            }
            Stmt::Raise(r) => {
                let Some(exc) = &r.exc else {
                    match self.handled.last() {
                        Some(&exc) => {
                            self.emit(Instr::Reraise { exc });
                        }
                        None => self.err(off, "a bare `raise` is only allowed in an `except` block"),
                    }
                    return;
                };
                let v = self.expr(exc);
                if !self.is_exception(&v.ty) {
                    let t = self.show(&v.ty);
                    self.err(off, format!("exceptions must derive from BaseException, got {t}"));
                }
                let cause = r.cause.as_ref().map(|c| {
                    let cv = self.expr(c);
                    if cv.ty != Ty::None && !self.is_exception(&cv.ty) {
                        let t = self.show(&cv.ty);
                        self.err(off, format!("an exception cause must derive from BaseException or be None, got {t}"));
                    }
                    self.obj(&cv)
                });
                let r = self.obj(&v);
                self.emit(Instr::Raise { exc: r, cause });
            }
            Stmt::Try(t) => self.try_stmt(t, off),
            Stmt::With(w) => self.with_stmt(w, off),
            Stmt::Assert(a) => {
                let (yes, _) = self.cond_narrow(&a.test);
                let c = self.expr(&a.test);
                let b = self.convert(&c, Repr::Bool);
                let ok = self.emit(Instr::JumpIfTrue { cond: b, target: 0 });
                let func = self.load_global("AssertionError", Ty::Any);
                let args = match &a.msg {
                    Some(m) => {
                        let v = self.expr(m);
                        vec![self.obj(&v)]
                    }
                    None => Vec::new(),
                };
                let exc = self.reg(Repr::Obj);
                self.emit(Instr::Call { dst: exc, func: func.reg, args, kwnames: Vec::new() });
                self.emit(Instr::Raise { exc, cause: None });
                let h = self.here();
                self.patch(ok, h);
                self.apply_narrow(yes);
            }
            Stmt::Delete(d) => {
                for t in &d.targets {
                    match t {
                        Expr::Subscript(sub) => {
                            let obj = self.expr(&sub.value);
                            let key = self.expr(&sub.slice);
                            match obj.ty.clone() {
                                Ty::Any => {}
                                t => match self.dunder(&t, "__delitem__", std::slice::from_ref(&key.ty)) {
                                    Some(Ok(_)) => {}
                                    Some(Err(m)) => self.err(off, m),
                                    None => {
                                        let t = self.show(&t);
                                        self.err(off, format!("{t} does not support item deletion"));
                                    }
                                },
                            }
                            let (o, k) = (self.obj(&obj), self.obj(&key));
                            self.emit(Instr::DelItem { obj: o, key: k });
                        }
                        _ => self.err(off, "only `del x[key]` is supported by Lizzard yet"),
                    }
                }
            }
            Stmt::Return(r) if !self.toplevel => {
                let want = self.ret.clone();
                let v = match &r.value {
                    Some(e) => self.expr_hint(e, Some(&want)),
                    None => self.konst(Const::None, Ty::None),
                };
                let v = self.coerce(v, &want, off, |l, got, want| format!("`{}` returns {want}, but this returns {got}", l.fname));
                let r = self.obj(&v);
                self.emit_cleanups(0);
                self.emit(Instr::Return { src: r });
            }
            Stmt::Pass(_) => {}
            Stmt::Import(_) | Stmt::ImportFrom(_) if self.toplevel => self.import(s),
            Stmt::TypeAlias(t) if self.toplevel => {
                if let Expr::Name(n) = &*t.name {
                    self.db.lookup_in(self.m, n.id.as_str());
                }
            }
            Stmt::ClassDef(_) => self.err(off, "classes inside functions or blocks are not supported by Lizzard yet"),
            Stmt::FunctionDef(_) => self.err(off, "nested functions are not supported by Lizzard yet"),
            _ => self.err(off, "this statement is not supported by Lizzard yet"),
        }
    }

    // ---- exceptions ----

    fn new_handler(&mut self) -> usize {
        let exc = self.reg(Repr::Obj);
        self.handlers.push(HandlerIr { ranges: Vec::new(), target: 0, exc });
        self.handlers.len() - 1
    }

    fn open_region(&mut self, handler: usize) {
        let start = self.here();
        self.regions.push(Region { handler, start, depth: self.cleanups.len() });
    }

    fn close_region(&mut self) {
        let r = self.regions.pop().unwrap();
        let end = self.here();
        if end > r.start {
            self.handlers[r.handler].ranges.push((r.start, end));
        }
    }

    /// Emit the cleanups above `depth`, innermost first, for leaving their
    /// blocks early. Each runs outside the ranges its own block protects.
    fn emit_cleanups(&mut self, depth: usize) {
        let saved = self.cleanups.clone();
        for i in (depth..saved.len()).rev() {
            let pause = self.here();
            for r in self.regions.iter().filter(|r| r.depth > i) {
                if pause > r.start {
                    self.handlers[r.handler].ranges.push((r.start, pause));
                }
            }
            self.cleanups.truncate(i);
            match &saved[i] {
                Cleanup::Finally(body) => self.stmts(body),
                Cleanup::Restore(prev) => {
                    self.emit(Instr::SetHandled { src: *prev });
                }
                Cleanup::WithExit(exit) => self.call_exit_none(*exit),
            }
            let resume = self.here();
            for r in self.regions.iter_mut().filter(|r| r.depth > i) {
                r.start = resume;
            }
        }
        self.cleanups = saved;
    }

    fn call_exit_none(&mut self, exit: Reg) {
        let none = self.konst(Const::None, Ty::None);
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::Call { dst, func: exit, args: vec![none.reg; 3], kwnames: Vec::new() });
    }

    /// The type an `except` clause catches.
    fn caught_type(&mut self, e: &Expr) -> Ty {
        let off: u32 = e.range().start().into();
        match self.class_types(e) {
            Some(classes) => {
                for c in &classes {
                    if !self.is_exception(c) {
                        let t = self.show(c);
                        self.err(off, format!("cannot catch {t}: it does not derive from BaseException"));
                    }
                }
                union(classes)
            }
            None => Ty::Any,
        }
    }

    fn try_stmt(&mut self, t: &ast::StmtTry, off: u32) {
        if t.is_star {
            return self.err(off, "`except*` is not supported by Lizzard yet");
        }
        let mut keys = Vec::new();
        assigned_keys(std::slice::from_ref(&Stmt::Try(t.clone())), &mut keys);
        for k in &keys {
            self.invalidate(k);
        }
        let fin = (!t.finalbody.is_empty()).then(|| {
            self.cleanups.push(Cleanup::Finally(t.finalbody.to_vec()));
            let h = self.new_handler();
            self.open_region(h);
            h
        });
        let th = (!t.handlers.is_empty()).then(|| {
            let h = self.new_handler();
            self.open_region(h);
            h
        });
        self.stmts(&t.body);
        if th.is_some() {
            self.close_region();
        }
        self.stmts(&t.orelse);
        let mut to_end = vec![self.emit(Instr::Jump { target: 0 })];
        if let Some(th) = th {
            self.handlers[th].target = self.here();
            let exc = self.handlers[th].exc;
            let prev = self.reg(Repr::Obj);
            self.emit(Instr::GetHandled { dst: prev });
            self.emit(Instr::SetHandled { src: exc });
            self.cleanups.push(Cleanup::Restore(prev));
            let ch = self.new_handler();
            self.open_region(ch);
            for h in &t.handlers {
                let ast::ExceptHandler::ExceptHandler(h) = h;
                self.at(h);
                let mut skip = None;
                let ty = match &h.type_ {
                    Some(te) => {
                        let ty = self.caught_type(te);
                        let cls = self.expr(te);
                        let m = self.reg(Repr::Bool);
                        self.emit(Instr::ExcMatch { dst: m, exc, cls: cls.reg });
                        skip = Some(self.emit(Instr::JumpIfFalse { cond: m, target: 0 }));
                        ty
                    }
                    None => Ty::Exc("BaseException".into()),
                };
                let before = self.narrow.clone();
                // `as e` binds a name of the clause's own type; Python deletes
                // it after the clause.
                let outer = h.name.as_ref().map(|n| (n.to_string(), self.locals.remove(n.as_str())));
                if let Some((name, _)) = &outer {
                    self.invalidate(name);
                    let reg = self.reg(Repr::Obj);
                    self.emit(Instr::Move { dst: reg, src: exc });
                    self.locals.insert(name.clone(), Local { reg, ty, repr: Repr::Obj });
                }
                self.handled.push(exc);
                self.stmts(&h.body);
                self.handled.pop();
                if let Some((name, old)) = outer {
                    let reg = self.locals[&name].reg;
                    self.emit(Instr::Unbind { reg });
                    match old {
                        Some(l) => {
                            self.emit(Instr::Unbind { reg: l.reg });
                            self.locals.insert(name, l);
                        }
                        None => {
                            self.locals.remove(&name);
                        }
                    }
                }
                self.narrow = before;
                self.emit(Instr::SetHandled { src: prev });
                to_end.push(self.emit(Instr::Jump { target: 0 }));
                if let Some(j) = skip {
                    let h = self.here();
                    self.patch(j, h);
                }
            }
            self.close_region();
            self.cleanups.pop();
            // No clause matched.
            self.emit(Instr::SetHandled { src: prev });
            self.emit(Instr::Reraise { exc });
            // An exception in a clause: restore, and let it propagate.
            self.handlers[ch].target = self.here();
            self.emit(Instr::SetHandled { src: prev });
            let cexc = self.handlers[ch].exc;
            self.emit(Instr::Reraise { exc: cexc });
        }
        let end = self.here();
        for j in to_end {
            self.patch(j, end);
        }
        if let Some(fh) = fin {
            self.close_region();
            self.cleanups.pop();
            self.stmts(&t.finalbody);
            let skip = self.emit(Instr::Jump { target: 0 });
            // The exceptional path: the exception is being handled while the
            // `finally` body runs (an exception there chains it).
            self.handlers[fh].target = self.here();
            let fexc = self.handlers[fh].exc;
            let prev = self.reg(Repr::Obj);
            self.emit(Instr::GetHandled { dst: prev });
            self.emit(Instr::SetHandled { src: fexc });
            self.cleanups.push(Cleanup::Restore(prev));
            let rh = self.new_handler();
            self.open_region(rh);
            self.stmts(&t.finalbody);
            self.close_region();
            self.cleanups.pop();
            self.emit(Instr::SetHandled { src: prev });
            self.emit(Instr::Reraise { exc: fexc });
            self.handlers[rh].target = self.here();
            self.emit(Instr::SetHandled { src: prev });
            let rexc = self.handlers[rh].exc;
            self.emit(Instr::Reraise { exc: rexc });
            let h = self.here();
            self.patch(skip, h);
        }
        for k in &keys {
            self.invalidate(k);
        }
    }

    fn with_stmt(&mut self, w: &ast::StmtWith, off: u32) {
        if w.is_async {
            return self.err(off, "`async with` is not supported by Lizzard yet");
        }
        self.with_items(&w.items, &w.body, off);
    }

    fn with_items(&mut self, items: &[ast::WithItem], body: &[Stmt], off: u32) {
        let Some((item, rest)) = items.split_first() else {
            return self.stmts(body);
        };
        let mgr = self.expr(&item.context_expr);
        let mut entered = Ty::Any;
        if !matches!(mgr.ty, Ty::Any) {
            match self.dunder(&mgr.ty, "__enter__", &[]) {
                Some(Ok((t, _, _))) => entered = t,
                Some(Err(m)) => self.err(off, m),
                None => {
                    let t = self.show(&mgr.ty);
                    self.err(off, format!("{t} does not support the context manager protocol (`__enter__`/`__exit__`)"));
                }
            }
            let any = Ty::Any;
            if let Some(Err(m)) = self.dunder(&mgr.ty, "__exit__", &[any.clone(), any.clone(), any]) {
                self.err(off, m);
            }
        }
        let m = self.obj(&mgr);
        let exit = self.reg(Repr::Obj);
        let n = self.name("__exit__");
        self.emit(Instr::LoadSpecial { dst: exit, obj: m, name: n });
        let enter = self.reg(Repr::Obj);
        let n = self.name("__enter__");
        self.emit(Instr::LoadSpecial { dst: enter, obj: m, name: n });
        let val = self.reg(Repr::Obj);
        self.emit(Instr::Call { dst: val, func: enter, args: Vec::new(), kwnames: Vec::new() });
        let v = Val { reg: val, ty: entered, repr: Repr::Obj };
        self.check(&v, format!("{}.__enter__()", self.src(&item.context_expr)), None);
        if let Some(t) = &item.optional_vars {
            self.assign_target(t, v, off);
        }
        self.cleanups.push(Cleanup::WithExit(exit));
        let wh = self.new_handler();
        self.open_region(wh);
        self.with_items(rest, body, off);
        self.close_region();
        self.cleanups.pop();
        self.call_exit_none(exit);
        let skip = self.emit(Instr::Jump { target: 0 });
        // An exception in the body: `__exit__(type, exc, tb)` may suppress it.
        self.handlers[wh].target = self.here();
        let exc = self.handlers[wh].exc;
        let prev = self.reg(Repr::Obj);
        self.emit(Instr::GetHandled { dst: prev });
        self.emit(Instr::SetHandled { src: exc });
        let rh = self.new_handler();
        self.open_region(rh);
        let ty = self.reg(Repr::Obj);
        self.emit(Instr::TypeOf { dst: ty, src: exc });
        let tb = self.reg(Repr::Obj);
        let n = self.name("__traceback__");
        self.emit(Instr::LoadAttr { dst: tb, obj: exc, name: n });
        let r = self.reg(Repr::Obj);
        self.emit(Instr::Call { dst: r, func: exit, args: vec![ty, exc, tb], kwnames: Vec::new() });
        let b = self.reg(Repr::Bool);
        self.emit(Instr::Truthy { dst: b, src: r });
        self.close_region();
        self.emit(Instr::SetHandled { src: prev });
        let suppressed = self.emit(Instr::JumpIfTrue { cond: b, target: 0 });
        self.emit(Instr::Reraise { exc });
        self.handlers[rh].target = self.here();
        self.emit(Instr::SetHandled { src: prev });
        let rexc = self.handlers[rh].exc;
        self.emit(Instr::Reraise { exc: rexc });
        let end = self.here();
        self.patch(skip, end);
        self.patch(suppressed, end);
    }

    fn is_exception(&mut self, t: &Ty) -> bool {
        match t {
            Ty::Any | Ty::Exc(_) => true,
            Ty::Type(t) => self.is_exception(t),
            Ty::Inst(c, _) => self.db.exc_base(*c).is_some(),
            Ty::Var(_) => {
                let u = self.db.upper(t);
                !matches!(u, Ty::Var(_)) && self.is_exception(&u)
            }
            _ => false,
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
                    return self.err(off, "relative imports are not supported by Lizzard yet");
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

    /// The type an assignment target expects, for typing the value (`x =
    /// []` where `x: list[int]`), without evaluating the target.
    fn target_hint(&mut self, target: &Expr) -> Option<Ty> {
        match target {
            Expr::Name(n) => self.locals.get(n.id.as_str()).map(|l| l.ty.clone()),
            Expr::Attribute(a) => {
                let obj_ty = self.peek_type(&a.value)?;
                let Ty::Inst(c, _) = self.lookup_ty(&obj_ty) else { return None };
                match self.db.find_member(c, a.attr.as_str()) {
                    Some((owner, Member::Attr(t))) => Some(self.member_type(&obj_ty, owner, &t)),
                    _ => None,
                }
            }
            Expr::Subscript(s) => {
                let t = self.peek_type(&s.value)?;
                if let Some(a) = self.builtin_args(&t, "list") {
                    return a.into_iter().next();
                }
                self.builtin_args(&t, "dict").and_then(|a| a.get(1).cloned())
            }
            _ => None,
        }
    }

    /// The (narrowed) type of a local or `x.attr` without evaluating it.
    fn peek_type(&mut self, e: &Expr) -> Option<Ty> {
        match e {
            Expr::Name(n) => self.narrow.get(n.id.as_str()).cloned().or_else(|| self.locals.get(n.id.as_str()).map(|l| l.ty.clone())),
            Expr::Attribute(a) => {
                if let Some(t) = attr_path(a).and_then(|p| self.narrow.get(&p)) {
                    return Some(t.clone());
                }
                let obj_ty = self.peek_type(&a.value)?;
                if matches!(obj_ty, Ty::Union(_)) {
                    return self.attr_type(&obj_ty, a.attr.as_str(), &mut None).ok();
                }
                let Ty::Inst(c, _) = self.lookup_ty(&obj_ty) else { return None };
                match self.db.find_member(c, a.attr.as_str()) {
                    Some((owner, Member::Attr(t))) => Some(self.member_type(&obj_ty, owner, &t)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn assign_target(&mut self, target: &Expr, v: Val, off: u32) {
        match target {
            Expr::Name(n) => self.assign_local(n.id.as_str(), v, None, off),
            Expr::Subscript(sub) => {
                let obj = self.expr(&sub.value);
                let key = self.expr(&sub.slice);
                self.store_item(&obj, &key, v, off);
            }
            Expr::Attribute(a) => {
                let obj = self.expr(&a.value);
                self.store_attr(&obj, a, v);
            }
            Expr::Tuple(ast::ExprTuple { elts, .. }) | Expr::List(ast::ExprList { elts, .. }) => {
                if elts.iter().any(|e| matches!(e, Expr::Starred(_))) {
                    return self.err(off, "starred assignment targets are not supported by Lizzard yet");
                }
                let tys: Vec<Ty> = match &v.ty {
                    Ty::Tuple(ts) if ts.len() == elts.len() => ts.clone(),
                    Ty::Tuple(ts) => {
                        self.err(off, format!("cannot unpack a tuple of {} items into {} targets", ts.len(), elts.len()));
                        vec![Ty::Any; elts.len()]
                    }
                    t => {
                        let t = t.clone();
                        let e = self.iter_elem(&t, off);
                        vec![e; elts.len()]
                    }
                };
                let src = self.obj(&v);
                let dsts: Vec<Reg> = elts.iter().map(|_| self.reg(Repr::Obj)).collect();
                self.emit(Instr::Unpack { src, dsts: dsts.clone() });
                for ((e, reg), ty) in elts.iter().zip(dsts).zip(tys) {
                    let item = Val { reg, ty, repr: Repr::Obj };
                    let what = format!("item of {}", self.src(target));
                    self.check(&item, what, None);
                    self.assign_target(e, item, off);
                }
            }
            _ => self.err(off, "this assignment target is not supported by Lizzard yet"),
        }
    }

    fn ann_assign(&mut self, a: &ast::StmtAnnAssign, off: u32) {
        let mut tps = Vec::new();
        let mut self_ty = None;
        if let Some(m) = &self.method {
            tps = self.db.class_tparams(m.class);
            self_ty = Some(Ty::Var(self.db.self_tv(m.class)));
        }
        let ty = self.db.eval_type(&a.annotation, &Scope { m: self.m, tparams: &tps, self_ty: self_ty.as_ref() });
        let Some(value) = &a.value else {
            // `x: T` declares a local (or global) that is assigned later.
            let Expr::Name(n) = &*a.target else {
                return self.err(off, "a declaration without a value is only supported for names");
            };
            let name = n.id.as_str();
            if self.locals.contains_key(name) || self.global_names.iter().any(|g| g == name) {
                return self.err(off, format!("`{name}` is already declared"));
            }
            let reg = self.reg(Repr::Obj);
            self.locals.insert(name.to_string(), Local { reg, ty, repr: Repr::Obj });
            return;
        };
        match &*a.target {
            Expr::Name(n) => {
                let v = self.expr_hint(value, Some(&ty));
                self.assign_local(n.id.as_str(), v, Some(ty), off);
            }
            // `self.x: T = ...` in `__init__` declares the attribute.
            Expr::Attribute(at) => {
                let on_self = matches!((&*at.value, &self.method), (Expr::Name(n), Some(m)) if m.is_init && m.receiver.as_deref() == Some(n.id.as_str()));
                if !on_self {
                    return self.err(off, "attributes are declared in the class body or as `self.x: T = ...` in `__init__`");
                }
                let c = self.method.as_ref().unwrap().class;
                if let Some(Member::Attr(declared)) = self.db.class_member(c, at.attr.as_str()) {
                    if !(self.assignable(&ty, &declared) && self.assignable(&declared, &ty)) {
                        let (d, t) = (self.show(&declared), self.show(&ty));
                        return self.err(off, format!("`{}` is declared as {d}, cannot redeclare it as {t}", at.attr.as_str()));
                    }
                }
                let v = self.expr_hint(value, Some(&ty));
                let obj = self.expr(&at.value);
                self.store_attr(&obj, at, v);
            }
            _ => self.err(off, "this assignment target is not supported by Lizzard yet"),
        }
    }

    fn aug_assign(&mut self, a: &ast::StmtAugAssign, off: u32) {
        let what = self.src(a);
        match &*a.target {
            Expr::Name(n) => {
                let cur = self.load_name(n.id.as_str(), off);
                let r = self.expr(&a.value);
                let v = self.binop_vals(a.op, cur, r, what, off, true);
                self.assign_local(n.id.as_str(), v, None, off);
            }
            Expr::Attribute(at) => {
                let obj = self.expr(&at.value);
                let cur = self.attr_of(&obj, at, what.clone());
                let r = self.expr(&a.value);
                let v = self.binop_vals(a.op, cur, r, what, off, true);
                self.store_attr(&obj, at, v);
            }
            Expr::Subscript(sub) => {
                let obj = self.expr(&sub.value);
                let key = self.expr(&sub.slice);
                let cur = self.item_of(&obj, &key, sub, what.clone());
                let r = self.expr(&a.value);
                let v = self.binop_vals(a.op, cur, r, what, off, true);
                self.store_item(&obj, &key, v, off);
            }
            _ => self.err(off, "this assignment target is not supported by Lizzard yet"),
        }
    }

    fn assign_local(&mut self, name: &str, v: Val, ann: Option<Ty>, off: u32) {
        if self.global_names.iter().any(|g| g == name) {
            let Some(t) = self.db.globals.get(&(self.m, name.to_string())).cloned() else { return };
            if ann.is_some() {
                return self.err(off, format!("the global `{name}` cannot be annotated in a function"));
            }
            let v = self.coerce(v, &t, off, |_, got, want| format!("cannot assign {got} to the global `{name}` of type {want}"));
            let r = self.obj(&v);
            let n = self.name(name);
            self.emit(Instr::StoreGlobal { name: n, val: r });
            return;
        }
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
            self.invalidate(name);
            self.narrow_to(name.to_string(), &l.ty, &v.ty);
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
        self.invalidate(name);
        if let Some(a) = &ann {
            self.narrow_to(name.to_string(), a, &v.ty);
        }
        self.locals.insert(name.to_string(), Local { reg, ty: v.ty, repr: v.repr });
    }

    /// After assigning a value of type `got` to `key` declared `declared`:
    /// a union is narrowed to what was assigned.
    fn narrow_to(&mut self, key: String, declared: &Ty, got: &Ty) {
        if matches!(declared, Ty::Union(_)) && !matches!(got, Ty::Any | Ty::Union(_)) && got != declared {
            self.narrow.insert(key, got.clone());
        }
    }

    /// Forget narrowing of `key` (and of `key.attr` for a local).
    fn invalidate(&mut self, key: &str) {
        let prefix = format!("{key}.");
        self.narrow.retain(|k, _| k != key && !k.starts_with(&prefix));
    }

    /// The narrowing key of `x` or `x.attr` for a local `x`.
    fn narrow_key(&self, e: &Expr) -> Option<String> {
        match e {
            Expr::Name(n) if self.locals.contains_key(n.id.as_str()) => Some(n.id.to_string()),
            Expr::Attribute(a) => attr_path(a).filter(|p| self.locals.contains_key(p.split('.').next().unwrap())),
            _ => None,
        }
    }

    /// The current (narrowed) type of a narrowing key.
    fn key_type(&mut self, key: &str, e: &Expr) -> Option<Ty> {
        if let Some(t) = self.narrow.get(key) {
            return Some(t.clone());
        }
        self.peek_type(e)
    }

    /// What a condition tells about locals and their attributes: the types
    /// where it is true, and where it is false.
    fn cond_narrow(&mut self, test: &Expr) -> (Vec<(String, Ty)>, Vec<(String, Ty)>) {
        let none = (Vec::new(), Vec::new());
        match test {
            Expr::UnaryOp(u) if u.op == ast::UnaryOp::Not => {
                let (t, f) = self.cond_narrow(&u.operand);
                (f, t)
            }
            Expr::BoolOp(b) => {
                // Each operand is evaluated knowing the earlier ones held
                // (`and`) or failed (`or`).
                let before = self.narrow.clone();
                let mut facts = Vec::new();
                for v in &b.values {
                    let (t, f) = self.cond_narrow(v);
                    let part = if b.op == ast::BoolOp::And { t } else { f };
                    self.apply_narrow(part.clone());
                    facts.extend(part);
                }
                self.narrow = before;
                match b.op {
                    ast::BoolOp::And => (facts, Vec::new()),
                    ast::BoolOp::Or => (Vec::new(), facts),
                }
            }
            Expr::Compare(c) if c.ops.len() == 1 && matches!(c.ops[0], ast::CmpOp::Is | ast::CmpOp::IsNot) => {
                let subject = match (&c.operands[0], &c.operands[1]) {
                    (e, Expr::NoneLiteral(_)) | (Expr::NoneLiteral(_), e) => e,
                    _ => return none,
                };
                let Some(key) = self.narrow_key(subject) else { return none };
                let Some(cur) = self.key_type(&key, subject) else { return none };
                let is_none = if members(&cur).contains(&Ty::None) || matches!(cur, Ty::Any | Ty::Object) { Ty::None } else { cur.clone() };
                let not_none = without_none(&cur);
                let (yes, no) = (vec![(key.clone(), is_none)], vec![(key, not_none)]);
                if c.ops[0] == ast::CmpOp::Is { (yes, no) } else { (no, yes) }
            }
            Expr::Call(call) if is_builtin_call(call, "isinstance") && call.arguments.args.len() == 2 && !self.locals.contains_key("isinstance") => {
                let subject = &call.arguments.args[0];
                let Some(key) = self.narrow_key(subject) else { return none };
                let Some(cur) = self.key_type(&key, subject) else { return none };
                let Some(classes) = self.class_types(&call.arguments.args[1]) else { return none };
                let mut yes = Vec::new();
                let mut no = Vec::new();
                for m in members(&cur) {
                    let mut matched = false;
                    for c in &classes {
                        if self.assignable(&m, c) {
                            yes.push(m.clone());
                            matched = true;
                            break;
                        }
                        // A downcast: the class is a subtype of the member.
                        if matches!(m, Ty::Any | Ty::Object) || self.assignable(c, &m) {
                            yes.push(c.clone());
                        }
                    }
                    if !matched {
                        no.push(m.clone());
                    }
                }
                let yes = if yes.is_empty() { union(classes) } else { union(yes) };
                let no = if no.is_empty() { cur.clone() } else { union(no) };
                (vec![(key.clone(), yes)], if matches!(cur, Ty::Union(_)) { vec![(key, no)] } else { Vec::new() })
            }
            e => {
                // Truthiness: a true value is not None.
                let Some(key) = self.narrow_key(e) else { return none };
                let Some(cur) = self.key_type(&key, e) else { return none };
                if members(&cur).contains(&Ty::None) && matches!(cur, Ty::Union(_)) {
                    (vec![(key, without_none(&cur))], Vec::new())
                } else {
                    none
                }
            }
        }
    }

    /// The classes an `isinstance` second argument names.
    fn class_types(&mut self, e: &Expr) -> Option<Vec<Ty>> {
        let sym = match e {
            Expr::Tuple(t) => {
                let mut out = Vec::new();
                for x in &t.elts {
                    out.extend(self.class_types(x)?);
                }
                return Some(out);
            }
            Expr::Name(n) => self.db.lookup_name(self.m, n.id.as_str()),
            e @ Expr::Attribute(_) => self.db.sym_of_expr(self.m, e)?,
            _ => return None,
        };
        match sym {
            Sym::Class(k) => {
                let n = self.db.class_params(k).len();
                Some(vec![Ty::Inst(k, vec![Ty::Any; n])])
            }
            Sym::Builtin(Ty::TupleVar(_)) => Some(vec![Ty::TupleVar(Box::new(Ty::Any))]),
            Sym::Builtin(t @ (Ty::Int | Ty::Float | Ty::Str | Ty::Bool | Ty::Exc(_) | Ty::None)) => Some(vec![t]),
            _ => None,
        }
    }

    fn apply_narrow(&mut self, facts: Vec<(String, Ty)>) {
        for (k, t) in facts {
            self.narrow.insert(k, t);
        }
    }

    /// Forget narrowing of everything a loop body assigns: at the top of the
    /// loop it may hold a value from the previous iteration.
    fn invalidate_assigned(&mut self, body: &[Stmt]) {
        let mut keys = Vec::new();
        assigned_keys(body, &mut keys);
        for k in keys {
            self.invalidate(&k);
        }
    }

    fn if_stmt(&mut self, test: &Expr, body: &[Stmt], rest: &[ast::ElifElseClause]) {
        let (yes, no) = self.cond_narrow(test);
        let c = self.expr(test);
        let b = self.convert(&c, Repr::Bool);
        let skip = self.emit(Instr::JumpIfFalse { cond: b, target: 0 });
        let before = self.narrow.clone();
        self.apply_narrow(yes);
        self.stmts(body);
        let after_body = std::mem::replace(&mut self.narrow, before);
        self.apply_narrow(no);
        let body_exits = always_exits(body);
        if rest.is_empty() {
            let h = self.here();
            self.patch(skip, h);
            if !body_exits {
                self.narrow = merge_narrow(&after_body, &self.narrow);
            }
            return;
        }
        let end = self.emit(Instr::Jump { target: 0 });
        let h = self.here();
        self.patch(skip, h);
        match &rest[0].test {
            Some(t) => self.if_stmt(t, &rest[0].body, &rest[1..]),
            None => self.stmts(&rest[0].body),
        }
        let rest_exits = rest.last().is_some_and(|c| c.test.is_none()) && rest.iter().all(|c| always_exits(&c.body));
        match (body_exits, rest_exits) {
            (true, _) => {}
            (false, true) => self.narrow = after_body,
            (false, false) => self.narrow = merge_narrow(&after_body, &self.narrow),
        }
        let h = self.here();
        self.patch(end, h);
    }

    /// Lower a loop body and patch its `break`s to the loop's end and its
    /// `continue`s to `cont`.
    fn loop_body(&mut self, body: &[Stmt]) -> Loop {
        self.loops.push(Loop { breaks: Vec::new(), continues: Vec::new(), depth: self.cleanups.len() });
        self.stmts(body);
        self.loops.pop().unwrap()
    }

    fn end_loop(&mut self, l: Loop, cont: u32) {
        let end = self.here();
        for j in l.breaks {
            self.patch(j, end);
        }
        for j in l.continues {
            self.patch(j, cont);
        }
    }

    fn while_stmt(&mut self, w: &ast::StmtWhile) {
        let off = self.at(w);
        if !w.orelse.is_empty() {
            return self.err(off, "`while ... else` is not supported by Lizzard yet");
        }
        self.invalidate_assigned(&w.body);
        let at_top = self.narrow.clone();
        let (yes, no) = self.cond_narrow(&w.test);
        let top = self.here();
        let c = self.expr(&w.test);
        let b = self.convert(&c, Repr::Bool);
        let exit = self.emit(Instr::JumpIfFalse { cond: b, target: 0 });
        self.apply_narrow(yes);
        let l = self.loop_body(&w.body);
        self.emit(Instr::Jump { target: top });
        let h = self.here();
        self.patch(exit, h);
        let broke = !l.breaks.is_empty();
        self.end_loop(l, top);
        self.narrow = at_top;
        if !broke {
            self.apply_narrow(no);
        }
    }

    fn for_stmt(&mut self, f: &ast::StmtFor) {
        let off = self.at(f);
        if f.is_async || !f.orelse.is_empty() {
            return self.err(off, "`async for` and `for ... else` are not supported by Lizzard yet");
        }
        let range_args = match &*f.iter {
            Expr::Call(c)
                if is_builtin_call(c, "range")
                    && c.arguments.keywords.is_empty()
                    && (1..=2).contains(&c.arguments.args.len())
                    && !self.locals.contains_key("range")
                    && self.db.lookup_in(self.m, "range").is_none()
                    && matches!(&*f.target, Expr::Name(_)) =>
            {
                Some(&c.arguments.args)
            }
            _ => None,
        };
        let Some(range_args) = range_args else {
            return self.for_iter(f, off);
        };
        let Expr::Name(target) = &*f.target else { unreachable!() };
        self.invalidate_assigned(&f.body);
        let at_top = self.narrow.clone();
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
        let l = self.loop_body(&f.body);
        let incr = self.here();
        self.emit(Instr::I64Add { dst: ctr, a: ctr, b: one });
        self.emit(Instr::Jump { target: top });
        let h = self.here();
        self.patch(exit, h);
        self.end_loop(l, incr);
        self.narrow = at_top;
    }

    /// `for target in iterable:` over any iterable.
    fn for_iter(&mut self, f: &ast::StmtFor, off: u32) {
        self.invalidate_assigned(&f.body);
        let it = self.expr(&f.iter);
        let at_top = self.narrow.clone();
        let elem = self.iter_elem(&it.ty, f.iter.range().start().into());
        let o = self.obj(&it);
        let iter = self.reg(Repr::Obj);
        self.emit(Instr::GetIter { dst: iter, obj: o });
        let top = self.here();
        let x = self.reg(Repr::Obj);
        let exit = self.emit(Instr::ForIter { dst: x, iter, target: 0 });
        let item = Val { reg: x, ty: elem, repr: Repr::Obj };
        let what = format!("item of {}", self.src(&*f.iter));
        self.check(&item, what, None);
        self.assign_target(&f.target, item, off);
        let l = self.loop_body(&f.body);
        self.emit(Instr::Jump { target: top });
        let h = self.here();
        self.patch(exit, h);
        self.end_loop(l, top);
        self.narrow = at_top;
    }

    /// The type of the items iterating over a `t` produces.
    fn iter_elem(&mut self, t: &Ty, off: u32) -> Ty {
        if matches!(t, Ty::Any) {
            return Ty::Any;
        }
        let it = match self.dunder(t, "__iter__", &[]) {
            Some(Ok((r, _, _))) => r,
            Some(Err(m)) => {
                self.err(off, m);
                return Ty::Any;
            }
            None => {
                let ts = self.show(t);
                self.err(off, format!("{ts} is not iterable"));
                return Ty::Any;
            }
        };
        match self.dunder(&it, "__next__", &[]) {
            Some(Ok((e, _, _))) => e,
            _ if matches!(it, Ty::Any) => Ty::Any,
            _ => {
                let ts = self.show(t);
                self.err(off, format!("{ts} is not iterable"));
                Ty::Any
            }
        }
    }

    // ---- expressions ----

    fn expr(&mut self, e: &Expr) -> Val {
        self.expr_hint(e, None)
    }

    /// `e`, with the type expected where it is used: list, dict and tuple
    /// literals take their item types from it.
    fn expr_hint(&mut self, e: &Expr, hint: Option<&Ty>) -> Val {
        let off = self.at(e);
        match e {
            Expr::NumberLiteral(n) => match &n.value {
                ast::Number::Int(i) => match i.as_i64() {
                    Some(v) => self.konst(Const::Int(v), Ty::Int),
                    None => self.konst(Const::BigInt(i.to_string()), Ty::Int),
                },
                ast::Number::Float(f) => self.konst(Const::Float(*f), Ty::Float),
                ast::Number::Complex { .. } => {
                    self.err(off, "complex numbers are not supported by Lizzard yet");
                    self.any()
                }
            },
            Expr::StringLiteral(s) => self.konst(Const::Str(s.value.to_str().to_string()), Ty::Str),
            Expr::BooleanLiteral(b) => self.konst(Const::Bool(b.value), Ty::Bool),
            Expr::NoneLiteral(_) => self.konst(Const::None, Ty::None),
            Expr::FString(f) => self.fstring(f),
            Expr::Name(n) => self.load_name(n.id.as_str(), off),
            Expr::Attribute(a) => {
                if let Some(v) = self.super_attr(a) {
                    return v;
                }
                let obj = self.expr(&a.value);
                let what = self.src(e);
                self.attr_of(&obj, a, what)
            }
            Expr::Call(c) => self.call(c),
            Expr::Subscript(s) if matches!(self.db.sym_of_expr(self.m, &s.value), Some(Sym::Class(_))) => {
                // `C[int]`: the class with explicit type arguments.
                let tps = match &self.method {
                    Some(m) => self.db.class_tparams(m.class),
                    None => Vec::new(),
                };
                let t = self.db.eval_type(e, &Scope { m: self.m, tparams: &tps, self_ty: None });
                let obj = self.expr(&s.value);
                let key = self.expr(&s.slice);
                let (o, k) = (self.obj(&obj), self.obj(&key));
                let dst = self.reg(Repr::Obj);
                self.emit(Instr::GetItem { dst, obj: o, key: k });
                Val { reg: dst, ty: Ty::Type(Box::new(t)), repr: Repr::Obj }
            }
            Expr::Subscript(s) => {
                let obj = self.expr(&s.value);
                let key = self.expr(&s.slice);
                let what = self.src(e);
                self.item_of(&obj, &key, s, what)
            }
            Expr::Slice(s) => {
                let mut regs = Vec::new();
                for part in [&s.lower, &s.upper, &s.step] {
                    let v = match part {
                        Some(p) => {
                            let v = self.expr(p);
                            if !matches!(v.ty, Ty::Int | Ty::Bool | Ty::None | Ty::Any) {
                                let t = self.show(&v.ty);
                                self.err(p.range().start().into(), format!("slice indices must be int or None, got {t}"));
                            }
                            v
                        }
                        None => self.konst(Const::None, Ty::None),
                    };
                    regs.push(self.obj(&v));
                }
                let dst = self.reg(Repr::Obj);
                self.emit(Instr::BuildSlice { dst, lower: regs[0], upper: regs[1], step: regs[2] });
                let ty = self.builtin_inst("slice", Vec::new());
                Val { reg: dst, ty, repr: Repr::Obj }
            }
            Expr::BinOp(b) => {
                let l = self.expr(&b.left);
                let r = self.expr(&b.right);
                let what = self.src(e);
                self.binop_vals(b.op, l, r, what, off, false)
            }
            Expr::Compare(c) => self.compare(c, e),
            Expr::BoolOp(b) => self.boolop(b),
            Expr::If(i) => {
                let (yes, no) = self.cond_narrow(&i.test);
                let c = self.expr(&i.test);
                let b = self.convert(&c, Repr::Bool);
                let skip = self.emit(Instr::JumpIfFalse { cond: b, target: 0 });
                let dst = self.reg(Repr::Obj);
                let before = self.narrow.clone();
                self.apply_narrow(yes);
                let x = self.expr_hint(&i.body, hint);
                self.narrow = before.clone();
                self.apply_narrow(no);
                let xo = self.obj(&x);
                self.emit(Instr::Move { dst, src: xo });
                let end = self.emit(Instr::Jump { target: 0 });
                let h = self.here();
                self.patch(skip, h);
                let y = self.expr_hint(&i.orelse, hint);
                self.narrow = before;
                let yo = self.obj(&y);
                self.emit(Instr::Move { dst, src: yo });
                let h = self.here();
                self.patch(end, h);
                let ty = union(vec![x.ty.clone(), y.ty.clone()]);
                Val { reg: dst, ty, repr: Repr::Obj }
            }
            Expr::List(l) => self.list_lit(&l.elts, hint, off),
            Expr::Tuple(t) => self.tuple_lit(&t.elts, hint, off),
            Expr::Dict(d) => self.dict_lit(d, hint, off),
            Expr::ListComp(lc) => self.comprehension(&lc.generators, CompBody::List(&lc.elt), hint, off),
            Expr::Generator(_) => {
                self.err(off, "generator expressions are not supported by Lizzard yet; use a list comprehension");
                self.any()
            }
            Expr::DictComp(dc) => match &dc.key {
                Some(k) => self.comprehension(&dc.generators, CompBody::Dict(k, &dc.value), hint, off),
                None => {
                    self.err(off, "this comprehension is not supported by Lizzard yet");
                    self.any()
                }
            },
            Expr::UnaryOp(u) => match u.op {
                ast::UnaryOp::Not => {
                    let v = self.expr(&u.operand);
                    let b = self.convert(&v, Repr::Bool);
                    let dst = self.reg(Repr::Bool);
                    self.emit(Instr::Not { dst, src: b });
                    Val { reg: dst, ty: Ty::Bool, repr: Repr::Bool }
                }
                ast::UnaryOp::USub => match &*u.operand {
                    Expr::NumberLiteral(n) if matches!(&n.value, ast::Number::Int(i) if i.as_i64().is_some()) || matches!(n.value, ast::Number::Float(_)) => match &n.value {
                        ast::Number::Int(i) => self.konst(Const::Int(-i.as_i64().unwrap()), Ty::Int),
                        ast::Number::Float(f) => self.konst(Const::Float(-f), Ty::Float),
                        _ => unreachable!(),
                    },
                    operand => {
                        let v = self.expr(operand);
                        let ty = match &v.ty {
                            Ty::Any => Ty::Any,
                            Ty::Int | Ty::Bool => Ty::Int,
                            Ty::Float => Ty::Float,
                            t => {
                                let t = t.clone();
                                match self.dunder(&t, "__neg__", &[]) {
                                    Some(Ok((r, _, _))) => r,
                                    _ => {
                                        let ts = self.show(&t);
                                        self.err(off, format!("bad operand type for unary -: {ts}"));
                                        Ty::Any
                                    }
                                }
                            }
                        };
                        let src = self.obj(&v);
                        let dst = self.reg(Repr::Obj);
                        self.emit(Instr::Neg { dst, src });
                        let r = Val { reg: dst, ty, repr: Repr::Obj };
                        let what = self.src(e);
                        self.check(&r, what, None);
                        self.unbox_f32(r)
                    }
                },
                _ => {
                    self.err(off, "this operator is not supported by Lizzard yet");
                    self.any()
                }
            },
            _ => {
                self.err(off, "this expression is not supported by Lizzard yet");
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
            let ty = self.narrow.get(name).cloned().unwrap_or_else(|| l.ty.clone());
            return Val { reg: l.reg, ty, repr: l.repr };
        }
        let ty = match self.db.lookup_in(self.m, name) {
            Some(Sym::UserVar) if self.toplevel => {
                self.err(off, format!("`{name}` is used before it is assigned"));
                Ty::Any
            }
            _ if self.bound.iter().any(|b| b == name) && !self.global_names.iter().any(|g| g == name) => {
                self.err(off, format!("local variable `{name}` is used before it is assigned"));
                Ty::Any
            }
            Some(Sym::UserVar) => {
                // A module global: anything may rebind it, so it is checked.
                let t = self.db.globals.get(&(self.m, name.to_string())).cloned().unwrap_or(Ty::Any);
                let v = self.load_global(name, t);
                self.check(&v, format!("global `{name}`"), None);
                return self.unbox_f32(v);
            }
            Some(s) => self.sym_value_ty(s),
            None => match name {
                n if BUILTIN_FNS.contains(&n) => Ty::Func(FuncRef::Builtin(BUILTIN_FNS.iter().find(|b| **b == n).unwrap())),
                n if is_exception_name(n) => Ty::Type(Box::new(Ty::Exc(n.to_string()))),
                "int" => Ty::Type(Box::new(Ty::Int)),
                "float" => Ty::Type(Box::new(Ty::Float)),
                "str" => Ty::Type(Box::new(Ty::Str)),
                "bool" => Ty::Type(Box::new(Ty::Bool)),
                "object" => Ty::Type(Box::new(Ty::Object)),
                "tuple" => Ty::Type(Box::new(Ty::TupleVar(Box::new(Ty::Any)))),
                "super" => Ty::Any,
                "__name__" => Ty::Str,
                n if self.db.is_declared_builtin(n) => {
                    let s = self.db.builtin(n);
                    self.sym_value_ty(s)
                }
                _ => {
                    self.err(off, format!("name `{name}` is not defined"));
                    Ty::Any
                }
            },
        };
        self.load_global(name, ty)
    }

    /// The type of attribute `t` declared by class `owner`, as seen on a
    /// value of type `obj_ty` (with `owner`'s type parameters solved).
    fn member_type(&mut self, obj_ty: &Ty, owner: ClassId, t: &Ty) -> Ty {
        let ps = self.db.class_params(owner);
        let fresh: Subst = ps.iter().map(|p| (*p, Ty::Var(self.db.new_tv("_", Variance::Inv)))).collect();
        let dst = Ty::Inst(owner, ps.iter().map(|p| fresh[p].clone()).collect());
        let mut s = Subst::new();
        self.db.assign(obj_ty, &dst, &mut s, &mut false);
        self.db.erase(&deep_subst(&subst(t, &fresh), &s))
    }

    /// The type of a member found on an instance (`on_class` false) or on
    /// the class object itself.
    fn member_value_ty(&mut self, obj_ty: &Ty, owner: ClassId, m: Member, on_class: bool, origin: &mut Option<String>) -> Ty {
        match m {
            Member::Property(_) if on_class => Ty::Any,
            Member::Property(f) => match self.resolve(f, Some(obj_ty.clone()), &[], &[], None) {
                Ok((t, file, doff)) => {
                    *origin = self.origin(file, doff);
                    t
                }
                Err(_) => Ty::Any,
            },
            Member::Method(f) => match (self.db.method_kind(f), on_class) {
                (MethodKind::Static, _) | (MethodKind::Normal, true) => Ty::Func(FuncRef::Def(f)),
                (MethodKind::Normal, false) => Ty::Func(FuncRef::Bound(f, Box::new(obj_ty.clone()))),
                (MethodKind::Class, true) => Ty::Func(FuncRef::Bound(f, Box::new(Ty::Type(Box::new(obj_ty.clone()))))),
                (MethodKind::Class, false) => Ty::Func(FuncRef::Bound(f, Box::new(Ty::Type(Box::new(obj_ty.clone()))))),
            },
            Member::Attr(t) => self.member_type(obj_ty, owner, &t),
        }
    }

    /// `super().name` in a method.
    fn super_attr(&mut self, a: &ast::ExprAttribute) -> Option<Val> {
        let Expr::Call(c) = &*a.value else { return None };
        if !is_builtin_call(c, "super") || self.locals.contains_key("super") {
            return None;
        }
        let off: u32 = a.range().start().into();
        let ctx = self.method.clone();
        let (Some(ctx), true) = (ctx, c.arguments.args.is_empty() && c.arguments.keywords.is_empty()) else {
            self.err(off, "only `super().name` without arguments in a method is supported by Lizzard yet");
            return Some(self.any());
        };
        let Some(recv) = ctx.receiver.clone() else {
            self.err(off, "`super()` needs a `self` or `cls` parameter");
            return Some(self.any());
        };
        let class = ctx.class;
        let this = self.load_name(&recv, off);
        let cname = self.db.classes[class].name.clone();
        let cls = self.load_global(&cname, Ty::Any);
        let sup = self.load_global("super", Ty::Any);
        let sobj = self.reg(Repr::Obj);
        self.emit(Instr::Call { dst: sobj, func: sup.reg, args: vec![cls.reg, this.reg], kwnames: Vec::new() });
        let name = a.attr.as_str();
        let (inst_ty, on_class) = match &this.ty {
            Ty::Type(t) => ((**t).clone(), true),
            t => (t.clone(), false),
        };
        let mut origin = None;
        let ty = match self.db.find_member_after(class, Some(class), name) {
            Some((owner, m)) => self.member_value_ty(&inst_ty, owner, m, on_class, &mut origin),
            None if is_dunder(name) || self.db.has_any_base(class) => Ty::Any,
            None => {
                self.err(a.attr.range().start().into(), format!("no base class of `{cname}` has an attribute `{name}`"));
                Ty::Any
            }
        };
        let dst = self.reg(Repr::Obj);
        let n = self.name(name);
        self.emit(Instr::LoadAttr { dst, obj: sobj, name: n });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        if !matches!(v.ty, Ty::Func(_) | Ty::Type(_) | Ty::Module(_)) {
            let what = format!("super().{name}");
            self.check(&v, what, origin);
        }
        Some(v)
    }

    /// The type of attribute `name` of a value of type `obj_ty`, or the
    /// error to report.
    fn attr_type(&mut self, obj_ty: &Ty, name: &str, origin: &mut Option<String>) -> Result<Ty, String> {
        if let Ty::Union(ms) = obj_ty {
            if ms.contains(&Ty::None) {
                let t = self.show(obj_ty);
                return Err(format!("cannot read attribute `{name}` of {t}: the value may be None (check `is not None` first)"));
            }
            let mut tys = Vec::new();
            for m in ms {
                tys.push(self.attr_type(m, name, origin)?);
            }
            return Ok(union(tys));
        }
        let lt = self.lookup_ty(obj_ty);
        match (obj_ty, &lt) {
            (Ty::Module(mm), _) => match self.db.module_attr(*mm, name) {
                Some(s) => Ok(self.sym_value_ty(s)),
                None => Err(format!("module `{}` has no attribute `{name}`", self.db.modules[*mm].name)),
            },
            (_, Ty::Inst(c, _)) => match self.db.find_member(*c, name) {
                Some((owner, m)) => Ok(self.member_value_ty(&obj_ty.clone(), owner, m, false, origin)),
                // `object`'s own attributes (`__class__`, `__dict__`, ...).
                None if self.db.has_any_base(*c) || is_dunder(name) => Ok(Ty::Any),
                None => Err(format!("{} has no attribute `{name}`", self.show(&lt))),
            },
            (Ty::Type(t), _) if matches!(self.lookup_ty(t), Ty::Inst(..)) => {
                let Ty::Inst(c, _) = self.lookup_ty(t) else { unreachable!() };
                let inst = (**t).clone();
                match self.db.find_member(c, name) {
                    Some((owner, m)) => Ok(self.member_value_ty(&inst, owner, m, true, origin)),
                    // `type`'s own attributes (`__name__`, `__mro__`, ...).
                    None if self.db.has_any_base(c) || is_dunder(name) => Ok(Ty::Any),
                    None => Err(format!("{} has no attribute `{name}`", self.show(obj_ty))),
                }
            }
            // Builtin exceptions' attributes are not declared yet (`errno`,
            // `filename`, ...): only `args` is typed.
            (Ty::Exc(_), _) => Ok(if name == "args" { Ty::TupleVar(Box::new(Ty::Any)) } else { Ty::Any }),
            (Ty::Any, _) => Ok(Ty::Any),
            // `type`'s own attributes of any class (`__name__`, ...).
            (Ty::Type(_), _) if is_dunder(name) => Ok(Ty::Any),
            (Ty::None, _) => Err(format!("cannot read attribute `{name}` of None")),
            (t, _) => Err(format!("attribute `{name}` of {} is not supported by Lizzard yet", self.show(t))),
        }
    }

    /// `obj.name`; `path` is `x.name` when `obj` is the local `x`, for
    /// narrowing.
    fn attr_of(&mut self, obj: &Val, a: &ast::ExprAttribute, what: String) -> Val {
        let off: u32 = a.attr.range().start().into();
        let name = a.attr.as_str();
        let mut origin = None;
        let mut ty = match self.attr_type(&obj.ty, name, &mut origin) {
            Ok(t) => t,
            Err(m) => {
                self.err(off, m);
                Ty::Any
            }
        };
        if let Some(t) = attr_path(a).and_then(|p| self.narrow.get(&p)) {
            ty = t.clone();
        }
        let o = self.obj(obj);
        let dst = self.reg(Repr::Obj);
        let n = self.name(name);
        self.emit(Instr::LoadAttr { dst, obj: o, name: n });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        if !matches!(v.ty, Ty::Func(_) | Ty::Type(_) | Ty::Module(_)) {
            self.check(&v, what, origin);
        }
        self.unbox_f32(v)
    }

    fn store_attr(&mut self, obj: &Val, a: &ast::ExprAttribute, val: Val) {
        let off: u32 = a.attr.range().start().into();
        let name = a.attr.as_str();
        let mut lt = self.lookup_ty(&obj.ty);
        let mut inst_ty = obj.ty.clone();
        if let Ty::Type(t) = &obj.ty {
            // `C.attr = v`: a class attribute.
            if let Ty::Inst(c, args) = self.lookup_ty(t) {
                if !matches!(self.db.find_member(c, name), Some((_, Member::Attr(_)))) {
                    let ts = self.show(&obj.ty);
                    self.err(off, format!("{ts} has no class attribute `{name}`"));
                }
                lt = Ty::Inst(c, args);
                inst_ty = (**t).clone();
            }
        }
        let val = match &lt {
            Ty::Any => val,
            Ty::Inst(c, _) => match self.db.find_member(*c, name) {
                Some((owner, Member::Attr(t))) => {
                    let want = self.member_type(&inst_ty, owner, &t);
                    self.coerce(val, &want, off, |_, got, want| format!("cannot assign {got} to attribute `{name}` of type {want}"))
                }
                Some((_, Member::Method(_))) => {
                    self.err(off, format!("cannot assign to method `{name}`"));
                    val
                }
                Some((_, Member::Property(_))) => {
                    self.err(off, format!("cannot assign to property `{name}`"));
                    val
                }
                None if self.db.has_any_base(*c) => val,
                None => {
                    let t = self.show(&lt);
                    self.err(
                        off,
                        format!("{t} has no attribute `{name}`; declare it in the class body (`{name}: T`) or in `__init__` (`self.{name}: T = ...`)"),
                    );
                    val
                }
            },
            _ => {
                let t = self.show(&obj.ty);
                self.err(off, format!("cannot assign attribute `{name}` of {t}"));
                val
            }
        };
        let (o, v) = (self.obj(obj), self.obj(&val));
        let n = self.name(name);
        self.emit(Instr::StoreAttr { obj: o, name: n, val: v });
        if let Some(p) = attr_path(a) {
            self.invalidate(&p);
            if let Ok(declared) = self.attr_type(&obj.ty, name, &mut None) {
                self.narrow_to(p, &declared, &val.ty);
            }
        }
    }

    /// Argument types expected by the callee, for typing literal arguments.
    /// Where all overloads agree on a parameter's type (with the receiver's
    /// type arguments substituted and no type variables left).
    fn arg_hints(&mut self, callee: &Ty) -> (Vec<Option<Ty>>, HashMap<String, Ty>) {
        let none = (Vec::new(), HashMap::new());
        let (f, recv) = match callee {
            Ty::Func(FuncRef::Def(f)) => (*f, None),
            Ty::Func(FuncRef::Bound(f, r)) => (*f, Some((**r).clone())),
            Ty::Type(t) => match self.lookup_ty(t) {
                Ty::Inst(k, _) if self.db.is_user_class(k) || self.db.is_builtin_class(k) => match self.db.find_member(k, "__init__") {
                    Some((_, Member::Method(f))) => (f, Some((**t).clone())),
                    _ => return none,
                },
                _ => return none,
            },
            _ => return none,
        };
        // The owner's type parameters as the receiver instantiates them.
        let mut s = Subst::new();
        if let (Some(r), Some(owner)) = (&recv, self.db.funcs[f].owner) {
            let ps = self.db.class_params(owner);
            let lt = match r {
                Ty::Type(t) => self.lookup_ty(t),
                r => self.lookup_ty(r),
            };
            self.db.assign(&lt, &Ty::Inst(owner, ps.iter().map(|p| Ty::Var(*p)).collect()), &mut s, &mut false);
        }
        let skip = usize::from(recv.is_some());
        let sigs = self.db.sigs(f);
        let usable = |t: &Ty| {
            let mut v = Vec::new();
            collect_vars(t, &mut v);
            v.is_empty() && !has_any(t)
        };
        let mut pos: Vec<Option<Ty>> = Vec::new();
        let mut kw: HashMap<String, Ty> = HashMap::new();
        let mut conflict: Vec<String> = Vec::new();
        for (n, sig) in sigs.iter().enumerate() {
            for (i, p) in sig.params.iter().skip(skip).enumerate() {
                let t = deep_subst(&p.ty, &s);
                let t = usable(&t).then_some(t);
                if matches!(p.kind, PKind::PosOnly | PKind::PosOrKw) {
                    if pos.len() <= i {
                        pos.resize(i + 1, None);
                        pos[i] = t.clone();
                    } else if n > 0 && pos[i] != t {
                        pos[i] = None;
                    }
                }
                if matches!(p.kind, PKind::PosOrKw | PKind::KwOnly) {
                    match (kw.get(&p.name), t) {
                        (Some(old), Some(t)) if *old != t => conflict.push(p.name.clone()),
                        (None, Some(t)) => {
                            kw.insert(p.name.clone(), t);
                        }
                        _ => {}
                    }
                }
            }
        }
        for c in conflict {
            kw.remove(&c);
        }
        (pos, kw)
    }

    fn call(&mut self, c: &ast::ExprCall) -> Val {
        let off: u32 = c.range().start().into();
        if is_builtin_call(c, "len") && !self.locals.contains_key("len") && self.db.lookup_in(self.m, "len").is_none() {
            let [arg] = &c.arguments.args[..] else {
                self.err(off, "len() takes exactly one argument");
                return self.any();
            };
            let v = self.expr(arg);
            let lt = self.lookup_ty(&v.ty);
            let ok = match &lt {
                Ty::Any => true,
                Ty::Inst(cl, _) => self.db.find_member(*cl, "__len__").is_some() || self.db.has_any_base(*cl),
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
        let (pos_hints, kw_hints) = self.arg_hints(&fv.ty);
        let mut pos = Vec::new();
        for (i, a) in c.arguments.args.iter().enumerate() {
            if matches!(a, Expr::Starred(_)) {
                self.err(off, "`*args` in calls is not supported by Lizzard yet");
            }
            let hint = pos_hints.get(i).cloned().flatten();
            pos.push(self.expr_hint(a, hint.as_ref()));
        }
        let mut kws = Vec::new();
        for k in c.arguments.keywords.iter() {
            let Some(name) = &k.arg else {
                self.err(off, "`**kwargs` in calls is not supported by Lizzard yet");
                continue;
            };
            let hint = kw_hints.get(name.as_str()).cloned();
            let v = self.expr_hint(&k.value, hint.as_ref());
            kws.push((name.as_str().to_string(), v));
        }
        let pos_tys: Vec<Ty> = pos.iter().map(|v| v.ty.clone()).collect();
        let kw_tys: Vec<(String, Ty)> = kws.iter().map(|(n, v)| (n.clone(), v.ty.clone())).collect();
        let mut origin = None;
        let mut builtin_ctor = false;
        let ret = self.call_type(fv.ty.clone(), &pos_tys, &kw_tys, off, &mut origin, &mut builtin_ctor);
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

    /// The result of calling a value of type `fty`.
    fn call_type(&mut self, fty: Ty, pos: &[Ty], kws: &[(String, Ty)], off: u32, origin: &mut Option<String>, builtin_ctor: &mut bool) -> Ty {
        match fty {
            Ty::Func(FuncRef::Builtin("print")) => Ty::None,
            Ty::Func(FuncRef::Builtin("type")) if pos.len() == 1 && kws.is_empty() => match &pos[0] {
                Ty::Any | Ty::Union(_) => Ty::Any,
                t => Ty::Type(Box::new(self.lookup_ty(t))),
            },
            Ty::Func(FuncRef::Builtin(n)) => {
                self.err(off, format!("{n}() is not supported here by Lizzard yet"));
                Ty::Any
            }
            Ty::Type(t) if matches!(*t, Ty::Exc(_) | Ty::Int | Ty::Float | Ty::Str | Ty::Bool) => {
                *builtin_ctor = true;
                *t
            }
            Ty::Type(t) if matches!(self.lookup_ty(&t), Ty::Inst(k, _) if self.db.is_user_class(k) || self.db.is_builtin_class(k)) => {
                self.construct(&t, &pos, &kws, off)
            }
            Ty::Func(FuncRef::Def(f)) => self.call_result(f, None, &pos, &kws, off, origin),
            Ty::Func(FuncRef::Bound(f, recv)) => self.call_result(f, Some(*recv), &pos, &kws, off, origin),
            // A union of callables (`shape.area` for `shape: Circle | Rect`).
            Ty::Union(ms) if !ms.contains(&Ty::None) => {
                let mut tys = Vec::new();
                for m in ms {
                    tys.push(self.call_type(m, pos, kws, off, origin, builtin_ctor));
                }
                union(tys)
            }
            Ty::Type(_) | Ty::Any => Ty::Any,
            t => {
                let t = self.show(&t);
                self.err(off, format!("{t} is not callable"));
                Ty::Any
            }
        }
    }

    /// The instance a call of class `t` creates, checking the arguments
    /// against `__init__`.
    fn construct(&mut self, t: &Ty, pos: &[Ty], kws: &[(String, Ty)], off: u32) -> Ty {
        let Ty::Inst(c, _) = self.lookup_ty(t) else { return Ty::Any };
        let cname = self.db.classes[c].name.clone();
        let ps = self.db.class_params(c);
        let recv = match t {
            Ty::Var(_) => t.clone(),
            // Explicit type arguments (`Stack[int]()`).
            Ty::Inst(_, args) if !args.is_empty() && !args.iter().any(has_any) => t.clone(),
            _ => Ty::Inst(c, ps.iter().map(|_| Ty::Var(self.db.new_tv("_", Variance::Inv))).collect()),
        };
        match self.db.find_member(c, "__init__") {
            Some((_, Member::Method(f))) => match self.resolve(f, Some(recv.clone()), pos, kws, Some(&recv)) {
                Ok((ty, _, _)) => ty,
                Err(msg) => {
                    self.err(off, msg.replacen("__init__()", &format!("{cname}()"), 1));
                    self.db.erase(&recv)
                }
            },
            _ => {
                if self.db.exc_base(c).is_none() && !self.db.has_any_base(c) && (!pos.is_empty() || !kws.is_empty()) {
                    self.err(off, format!("{cname}() takes no arguments"));
                }
                self.db.erase(&recv)
            }
        }
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
        match self.resolve(f, recv, pos, kws, None) {
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
    /// matches that depend on `Any`). Returns the result type (or, for a
    /// constructor, `ret` with the solved type arguments) and where the
    /// chosen signature is declared.
    fn resolve(&mut self, f: FuncId, recv: Option<Ty>, pos: &[Ty], kws: &[(String, Ty)], ret: Option<&Ty>) -> Result<(Ty, FileId, u32), String> {
        let sigs = self.db.sigs(f);
        let fname = self.db.funcs[f].name.clone();
        let mut matches: Vec<(Ty, bool, FileId, u32)> = Vec::new();
        let mut first_err = None;
        for sig in sigs.iter() {
            match self.match_sig(sig, recv.as_ref(), pos, kws, ret) {
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

    fn match_sig(&mut self, sig: &Sig, recv: Option<&Ty>, pos: &[Ty], kws: &[(String, Ty)], ret: Option<&Ty>) -> Result<(Ty, bool), Mismatch> {
        // The callee's own type variables are solved per call, even when
        // they are the rigid variables of the function being checked.
        let fresh = {
            let mut tys: Vec<&Ty> = sig.params.iter().map(|p| &p.ty).collect();
            tys.push(&sig.ret);
            self.db.fresh_rigid(&tys)
        };
        let ptys: Vec<Ty> = sig.params.iter().map(|p| subst(&p.ty, &fresh)).collect();
        let mut pairs: Vec<(String, Ty, Ty)> = Vec::new();
        let mut params = sig.params.iter().zip(ptys);
        if let Some(r) = recv {
            let Some((p, pt)) = params.next() else {
                return Err(Mismatch::Arity("missing `self`".into()));
            };
            pairs.push((p.name.clone(), pt, r.clone()));
        }
        let mut i = 0;
        let mut used = vec![false; kws.len()];
        let kw = |name: &str, used: &mut Vec<bool>| -> Option<Ty> {
            let k = kws.iter().position(|(n, _)| n == name)?;
            used[k] = true;
            Some(kws[k].1.clone())
        };
        for (p, pt) in params {
            match p.kind {
                PKind::PosOnly | PKind::PosOrKw => {
                    if i < pos.len() {
                        pairs.push((p.name.clone(), pt, pos[i].clone()));
                        i += 1;
                    } else if let Some(t) = matches!(p.kind, PKind::PosOrKw).then(|| kw(&p.name, &mut used)).flatten() {
                        pairs.push((p.name.clone(), pt, t));
                    } else if !p.default {
                        return Err(Mismatch::Arity(format!("missing argument `{}`", p.name)));
                    }
                }
                PKind::VarPos => {
                    while i < pos.len() {
                        pairs.push((p.name.clone(), pt.clone(), pos[i].clone()));
                        i += 1;
                    }
                }
                PKind::KwOnly => match kw(&p.name, &mut used) {
                    Some(t) => pairs.push((p.name.clone(), pt, t)),
                    None if p.default => {}
                    None => return Err(Mismatch::Arity(format!("missing keyword argument `{}`", p.name))),
                },
                PKind::VarKw => {
                    for (k, (n, t)) in kws.iter().enumerate() {
                        if !used[k] {
                            used[k] = true;
                            pairs.push((n.clone(), pt.clone(), t.clone()));
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
                let expected = self.db.erase(&deep_subst(&want, &s));
                return Err(Mismatch::Arg { name, expected, got });
            }
        }
        let sret = subst(&sig.ret, &fresh);
        Ok((self.db.erase(&deep_subst(ret.unwrap_or(&sret), &s)), any))
    }

    /// The method `name` of `recv`, resolved for the given arguments.
    fn dunder(&mut self, recv: &Ty, name: &str, args: &[Ty]) -> Option<Result<(Ty, FileId, u32), String>> {
        if let Ty::Union(ms) = recv {
            // Every member must support it; the result is the union.
            let mut tys = Vec::new();
            let mut at = None;
            for m in ms {
                match self.dunder(m, name, args)? {
                    Ok((t, f, o)) => {
                        tys.push(t);
                        at.get_or_insert((f, o));
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
            let (f, o) = at?;
            return Some(Ok((union(tys), f, o)));
        }
        let Ty::Inst(c, _) = self.lookup_ty(recv) else { return None };
        match self.db.find_member(c, name) {
            Some((_, Member::Method(f))) => Some(self.resolve(f, Some(recv.clone()), args, &[], None)),
            None if self.db.has_any_base(c) => Some(Ok((Ty::Any, self.file, 0))),
            _ => None,
        }
    }

    fn item_of(&mut self, obj: &Val, key: &Val, s: &ast::ExprSubscript, what: String) -> Val {
        let off: u32 = s.range().start().into();
        if self.vec_f32(&obj.ty) && matches!(key.ty, Ty::Int | Ty::Bool) {
            let Ty::Inst(_, args) = &obj.ty else { unreachable!() };
            let Ty::Inst(_, dargs) = &args[1] else { unreachable!() };
            let elem = dargs[0].clone();
            let arr = self.obj(obj);
            let idx = self.convert(key, Repr::I64);
            let dst = self.reg(Repr::F32);
            self.emit(Instr::ArrLoadF32 { dst, arr, idx });
            return Val { reg: dst, ty: elem, repr: Repr::F32 };
        }
        let mut origin = None;
        let ty = match &obj.ty {
            Ty::Any => Ty::Any,
            // A literal index into a tuple of known length.
            Ty::Tuple(v) if int_literal(&s.slice).is_some() => {
                let i = int_literal(&s.slice).unwrap();
                let n = v.len() as i64;
                match usize::try_from(if i < 0 { i + n } else { i }).ok().and_then(|i| v.get(i)) {
                    Some(t) => t.clone(),
                    None => {
                        let t = self.show(&obj.ty);
                        self.err(off, format!("index {i} is out of range for {t}"));
                        Ty::Any
                    }
                }
            }
            t => {
                let t = t.clone();
                match self.dunder(&t, "__getitem__", std::slice::from_ref(&key.ty)) {
                    Some(Ok((t, file, doff))) => {
                        origin = self.origin(file, doff);
                        t
                    }
                    Some(Err(m)) => {
                        self.err(off, m);
                        Ty::Any
                    }
                    None => {
                        let t = self.show(&t);
                        self.err(off, format!("{t} is not subscriptable"));
                        Ty::Any
                    }
                }
            }
        };
        let (o, k) = (self.obj(obj), self.obj(key));
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::GetItem { dst, obj: o, key: k });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        self.check(&v, what, origin);
        self.unbox_f32(v)
    }

    fn store_item(&mut self, obj: &Val, key: &Val, val: Val, off: u32) {
        if self.vec_f32(&obj.ty) && matches!(key.ty, Ty::Int | Ty::Bool) && (self.is_f32(&val.ty) || val.ty == Ty::Float) {
            let arr = self.obj(obj);
            let idx = self.convert(key, Repr::I64);
            let v = if val.repr == Repr::F32 { val.reg } else { self.convert(&val, Repr::F32) };
            self.emit(Instr::ArrStoreF32 { arr, idx, val: v });
            return;
        }
        match obj.ty.clone() {
            Ty::Any => {}
            t => match self.dunder(&t, "__setitem__", &[key.ty.clone(), val.ty.clone()]) {
                Some(Ok(_)) => {}
                Some(Err(m)) => self.err(off, m),
                None => {
                    let t = self.show(&t);
                    self.err(off, format!("{t} does not support item assignment"));
                }
            },
        }
        let (o, k, v) = (self.obj(obj), self.obj(key), self.obj(&val));
        self.emit(Instr::SetItem { obj: o, key: k, val: v });
    }

    fn binop_vals(&mut self, op: ast::Operator, l: Val, r: Val, what: String, off: u32, inplace: bool) -> Val {
        let (bop, dunder, rdunder) = match op {
            ast::Operator::Add => (BinOp::Add, "__add__", "__radd__"),
            ast::Operator::Sub => (BinOp::Sub, "__sub__", "__rsub__"),
            ast::Operator::Mult => (BinOp::Mul, "__mul__", "__rmul__"),
            ast::Operator::Div => (BinOp::Div, "__truediv__", "__rtruediv__"),
            ast::Operator::FloorDiv => (BinOp::FloorDiv, "__floordiv__", "__rfloordiv__"),
            ast::Operator::Mod => (BinOp::Mod, "__mod__", "__rmod__"),
            _ => {
                self.err(off, format!("operator `{}` is not supported by Lizzard yet", op.as_str()));
                return self.any();
            }
        };
        let (lt, rt) = (l.ty.clone(), r.ty.clone());
        let (ty, origin) = match self.binop_type(bop, dunder, rdunder, &lt, &rt) {
            Some(x) => x,
            None => {
                let (ls, rs) = (self.show(&lt), self.show(&rt));
                self.err(off, format!("unsupported operand types for {}: {ls} and {rs}", op.as_str()));
                (Ty::Any, None)
            }
        };
        // float32 arithmetic on two float32 values. (A Python float operand
        // could be an `np.float64` at runtime, which NumPy promotes, so mixed
        // operations go through NumPy.)
        if matches!(bop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) && self.is_f32(&l.ty) && self.is_f32(&r.ty) && self.is_f32(&ty) {
            let (a, bb) = (self.convert(&l, Repr::F32), self.convert(&r, Repr::F32));
            let dst = self.reg(Repr::F32);
            self.emit(Instr::F32Bin { dst, op: bop, a, b: bb });
            return Val { reg: dst, ty, repr: Repr::F32 };
        }
        let (a, bb) = (self.obj(&l), self.obj(&r));
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::BinOp { dst, op: bop, a, b: bb, inplace });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        self.check(&v, what, origin);
        self.unbox_f32(v)
    }

    /// The result type of `lt op rt`, with where the operator's method is
    /// declared; `None` if the operands do not support it.
    fn binop_type(&mut self, bop: BinOp, dunder: &str, rdunder: &str, lt: &Ty, rt: &Ty) -> Option<(Ty, Option<String>)> {
        Some(match (lt, rt) {
            (Ty::Any, _) | (_, Ty::Any) => (Ty::Any, None),
            (Ty::Union(_), _) | (_, Ty::Union(_)) => {
                let mut tys = Vec::new();
                let mut origin = None;
                for l in members(lt) {
                    for r in members(rt) {
                        let (t, o) = self.binop_type(bop, dunder, rdunder, &l, &r)?;
                        tys.push(t);
                        origin = origin.or(o);
                    }
                }
                (union(tys), origin)
            }
            (Ty::Int | Ty::Bool, Ty::Int | Ty::Bool) if bop != BinOp::Div => (Ty::Int, None),
            (Ty::Int | Ty::Bool | Ty::Float, Ty::Int | Ty::Bool | Ty::Float) => (Ty::Float, None),
            (Ty::Str, Ty::Str) if bop == BinOp::Add => (Ty::Str, None),
            (Ty::Tuple(a), Ty::Tuple(b)) if bop == BinOp::Add => (Ty::Tuple(a.iter().chain(b).cloned().collect()), None),
            (lt, rt) => match self.dunder(lt, dunder, std::slice::from_ref(rt)) {
                Some(Ok((t, file, doff))) => (t, self.origin(file, doff)),
                _ => match self.dunder(rt, rdunder, std::slice::from_ref(lt)) {
                    Some(Ok((t, file, doff))) => (t, self.origin(file, doff)),
                    _ => return None,
                },
            },
        })
    }

    fn compare(&mut self, c: &ast::ExprCompare, e: &Expr) -> Val {
        let off: u32 = c.range().start().into();
        if c.ops.len() != 1 {
            self.err(off, "chained comparisons are not supported by Lizzard yet");
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
            ast::CmpOp::In => (CmpOp::In, "__contains__"),
            ast::CmpOp::NotIn => (CmpOp::NotIn, "__contains__"),
        };
        let l = self.expr(&c.operands[0]);
        let r = self.expr(&c.operands[1]);
        let ty = match (&l.ty, &r.ty) {
            _ if dunder.is_empty() => Ty::Bool,
            // `x in container`: `container.__contains__(x)`.
            (lt, rt) if matches!(op, CmpOp::In | CmpOp::NotIn) => {
                let (lt, rt) = (lt.clone(), rt.clone());
                match &rt {
                    Ty::Any => {}
                    _ => match self.dunder(&rt, "__contains__", std::slice::from_ref(&lt)) {
                        Some(Ok(_)) => {}
                        Some(Err(m)) => self.err(off, m),
                        None => {
                            let ts = self.show(&rt);
                            self.err(off, format!("{ts} does not support `in`"));
                        }
                    },
                }
                Ty::Bool
            }
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
        let mut tys = Vec::new();
        let mut exits = Vec::new();
        let before = self.narrow.clone();
        for (i, e) in b.values.iter().enumerate() {
            let last = i + 1 == b.values.len();
            let (yes, no) = if last { (Vec::new(), Vec::new()) } else { self.cond_narrow(e) };
            let v = self.expr(e);
            // `a or b` is `a` only when `a` is true, so not None.
            tys.push(if b.op == ast::BoolOp::Or && !last { without_none(&v.ty) } else { v.ty.clone() });
            let o = self.obj(&v);
            self.emit(Instr::Move { dst, src: o });
            if !last {
                let t = self.reg(Repr::Bool);
                self.emit(Instr::Truthy { dst: t, src: dst });
                let j = match b.op {
                    ast::BoolOp::Or => Instr::JumpIfTrue { cond: t, target: 0 },
                    ast::BoolOp::And => Instr::JumpIfFalse { cond: t, target: 0 },
                };
                exits.push(self.emit(j));
                // Later operands run only when the earlier ones were true
                // (`and`) or false (`or`).
                self.apply_narrow(if b.op == ast::BoolOp::And { yes } else { no });
            }
        }
        self.narrow = before;
        let h = self.here();
        for j in exits {
            self.patch(j, h);
        }
        Val { reg: dst, ty: union(tys), repr: Repr::Obj }
    }

    // ---- containers ----

    /// The items of a list or dict display, typed by `want` if given, else
    /// by joining their types.
    fn items(&mut self, elts: &[&Expr], want: Option<&Ty>, what: &str) -> (Vec<Reg>, Option<Ty>) {
        let mut vals = Vec::new();
        for e in elts {
            let off: u32 = e.range().start().into();
            if matches!(e, Expr::Starred(_)) {
                self.err(off, "`*` in displays is not supported by Lizzard yet");
            }
            let v = self.expr_hint(e, want);
            let v = match want {
                Some(t) => self.coerce(v, t, off, |_, got, want| format!("{what} must be {want}, got {got}")),
                None => v,
            };
            vals.push(v);
        }
        let ty = match want {
            Some(t) => Some(t.clone()),
            None => {
                let mut ty: Option<Ty> = None;
                for v in &vals {
                    ty = Some(match ty {
                        None => v.ty.clone(),
                        Some(t) => self.join(&t, &v.ty),
                    });
                }
                ty
            }
        };
        (vals.iter().map(|v| self.obj(v)).collect(), ty)
    }

    fn list_lit(&mut self, elts: &[Expr], hint: Option<&Ty>, off: u32) -> Val {
        let want = hint.and_then(|h| self.builtin_args(h, "list")).and_then(|a| a.into_iter().next());
        let refs: Vec<&Expr> = elts.iter().collect();
        let (items, elem) = self.items(&refs, want.as_ref(), "a list item");
        let elem = elem.unwrap_or_else(|| {
            self.err(off, "cannot infer the type of an empty list; annotate it (`x: list[int] = []`)");
            Ty::Any
        });
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::BuildList { dst, items });
        let ty = self.builtin_inst("list", vec![elem]);
        Val { reg: dst, ty, repr: Repr::Obj }
    }

    fn dict_lit(&mut self, d: &ast::ExprDict, hint: Option<&Ty>, off: u32) -> Val {
        let want = hint.and_then(|h| self.builtin_args(h, "dict"));
        let (wk, wv) = match &want {
            Some(a) if a.len() == 2 => (Some(a[0].clone()), Some(a[1].clone())),
            _ => (None, None),
        };
        let mut keys = Vec::new();
        let mut vals = Vec::new();
        let (mut kt, mut vt): (Option<Ty>, Option<Ty>) = (None, None);
        let mut unpacked = false;
        for item in d.items.iter() {
            let Some(k) = &item.key else {
                self.err(off, "`**` in dict displays is not supported by Lizzard yet");
                unpacked = true;
                continue;
            };
            let (kr, kty) = self.items(&[k], wk.as_ref(), "a dict key");
            let (vr, vty) = self.items(&[&item.value], wv.as_ref(), "a dict value");
            keys.extend(kr);
            vals.extend(vr);
            for (acc, t) in [(&mut kt, kty), (&mut vt, vty)] {
                if let Some(t) = t {
                    *acc = Some(match acc.take() {
                        None => t,
                        Some(a) => self.join(&a, &t),
                    });
                }
            }
        }
        let (kt, vt) = match (wk, wv, kt, vt) {
            (Some(k), Some(v), _, _) => (k, v),
            (_, _, Some(k), Some(v)) => (k, v),
            _ if unpacked => (Ty::Any, Ty::Any),
            _ => {
                self.err(off, "cannot infer the type of an empty dict; annotate it (`x: dict[str, int] = {}`)");
                (Ty::Any, Ty::Any)
            }
        };
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::BuildDict { dst, keys, vals });
        let ty = self.builtin_inst("dict", vec![kt, vt]);
        Val { reg: dst, ty, repr: Repr::Obj }
    }

    fn tuple_lit(&mut self, elts: &[Expr], hint: Option<&Ty>, off: u32) -> Val {
        let mut items = Vec::new();
        let mut tys = Vec::new();
        for (i, e) in elts.iter().enumerate() {
            if matches!(e, Expr::Starred(_)) {
                self.err(off, "`*` in tuples is not supported by Lizzard yet");
            }
            let h = match hint {
                Some(Ty::Tuple(v)) if v.len() == elts.len() => Some(v[i].clone()),
                Some(Ty::TupleVar(t)) => Some((**t).clone()),
                _ => None,
            };
            let v = self.expr_hint(e, h.as_ref());
            tys.push(v.ty.clone());
            items.push(self.obj(&v));
        }
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::BuildTuple { dst, items });
        Val { reg: dst, ty: Ty::Tuple(tys), repr: Repr::Obj }
    }

    /// A list or dict comprehension: an inline loop that appends to the
    /// result. The loop variables are local to it, as in Python 3.
    fn comprehension(&mut self, gens: &[ast::Comprehension], body: CompBody, hint: Option<&Ty>, off: u32) -> Val {
        let mut bound = Vec::new();
        for g in gens {
            target_names(&g.target, &mut bound);
        }
        let saved: Vec<(String, Option<Local>)> = bound.iter().map(|n| (n.clone(), self.locals.remove(n))).collect();
        let dst = self.reg(Repr::Obj);
        match body {
            CompBody::List(_) => self.emit(Instr::BuildList { dst, items: Vec::new() }),
            CompBody::Dict(..) => self.emit(Instr::BuildDict { dst, keys: Vec::new(), vals: Vec::new() }),
        };
        let mut elem = Vec::new();
        self.comp_level(gens, &body, hint, dst, &mut elem, off);
        for (n, l) in saved {
            match l {
                Some(l) => self.locals.insert(n, l),
                None => self.locals.remove(&n),
            };
        }
        let ty = match (body, elem.as_slice()) {
            (CompBody::List(_), [t]) => self.builtin_inst("list", vec![t.clone()]),
            (CompBody::Dict(..), [k, v]) => self.builtin_inst("dict", vec![k.clone(), v.clone()]),
            _ => Ty::Any,
        };
        Val { reg: dst, ty, repr: Repr::Obj }
    }

    fn comp_level(&mut self, gens: &[ast::Comprehension], body: &CompBody, hint: Option<&Ty>, dst: Reg, elem: &mut Vec<Ty>, off: u32) {
        let Some((g, rest)) = gens.split_first() else {
            match body {
                CompBody::List(e) => {
                    let want = hint.and_then(|h| self.builtin_args(h, "list")).and_then(|a| a.into_iter().next());
                    let (regs, t) = self.items(&[e], want.as_ref(), "a list item");
                    self.emit(Instr::ListAppend { list: dst, val: regs[0] });
                    elem.extend(t);
                }
                CompBody::Dict(k, v) => {
                    let want = hint.and_then(|h| self.builtin_args(h, "dict")).filter(|a| a.len() == 2);
                    let (kr, kt) = self.items(&[k], want.as_ref().map(|a| &a[0]), "a dict key");
                    let (vr, vt) = self.items(&[v], want.as_ref().map(|a| &a[1]), "a dict value");
                    self.emit(Instr::SetItem { obj: dst, key: kr[0], val: vr[0] });
                    elem.extend(kt);
                    elem.extend(vt);
                }
            }
            return;
        };
        if g.is_async {
            return self.err(off, "async comprehensions are not supported by Lizzard yet");
        }
        let it = self.expr(&g.iter);
        let ety = self.iter_elem(&it.ty, g.iter.range().start().into());
        let o = self.obj(&it);
        let iter = self.reg(Repr::Obj);
        self.emit(Instr::GetIter { dst: iter, obj: o });
        let top = self.here();
        let x = self.reg(Repr::Obj);
        let exit = self.emit(Instr::ForIter { dst: x, iter, target: 0 });
        let item = Val { reg: x, ty: ety, repr: Repr::Obj };
        let what = format!("item of {}", self.src(&g.iter));
        self.check(&item, what, None);
        self.assign_target(&g.target, item, off);
        let before = self.narrow.clone();
        for cond in &g.ifs {
            let (yes, _) = self.cond_narrow(cond);
            let c = self.expr(cond);
            let b = self.convert(&c, Repr::Bool);
            self.emit(Instr::JumpIfFalse { cond: b, target: top });
            self.apply_narrow(yes);
        }
        self.comp_level(rest, body, hint, dst, elem, off);
        self.narrow = before;
        self.emit(Instr::Jump { target: top });
        let h = self.here();
        self.patch(exit, h);
    }

    // ---- f-strings ----

    fn fstring(&mut self, f: &ast::ExprFString) -> Val {
        let mut parts = Vec::new();
        for part in f.value.iter() {
            match part {
                ast::FStringPartRef::Literal(l) => {
                    let v = self.konst(Const::Str(l.value.to_string()), Ty::Str);
                    parts.push(v.reg);
                }
                ast::FStringPartRef::FString(fs) => self.fstring_elements(&fs.elements, &mut parts),
            }
        }
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::BuildString { dst, parts });
        Val { reg: dst, ty: Ty::Str, repr: Repr::Obj }
    }

    fn fstring_elements(&mut self, elements: &ast::InterpolatedStringElements, parts: &mut Vec<Reg>) {
        for el in elements.iter() {
            match el {
                ast::InterpolatedStringElement::Literal(l) => {
                    let v = self.konst(Const::Str(l.value.to_string()), Ty::Str);
                    parts.push(v.reg);
                }
                ast::InterpolatedStringElement::Interpolation(i) => {
                    if i.debug_text.is_some() {
                        let off = i.range().start().into();
                        self.err(off, "`=` in f-strings is not supported by Lizzard yet");
                    }
                    let v = self.expr(&i.expression);
                    let val = self.obj(&v);
                    let spec = i.format_spec.as_ref().map(|fs| {
                        let mut sp = Vec::new();
                        self.fstring_elements(&fs.elements, &mut sp);
                        let dst = self.reg(Repr::Obj);
                        self.emit(Instr::BuildString { dst, parts: sp });
                        dst
                    });
                    let conv = i.conversion.to_byte().unwrap_or(0);
                    let dst = self.reg(Repr::Obj);
                    self.emit(Instr::Format { dst, val, conv, spec });
                    parts.push(dst);
                }
            }
        }
    }
}

/// `x.attr` for an attribute of a name.
fn attr_path(a: &ast::ExprAttribute) -> Option<String> {
    match &*a.value {
        Expr::Name(n) => Some(format!("{}.{}", n.id.as_str(), a.attr.as_str())),
        _ => None,
    }
}

fn without_none(t: &Ty) -> Ty {
    match t {
        Ty::Union(v) => union(v.iter().filter(|m| **m != Ty::None).cloned().collect()),
        t => t.clone(),
    }
}

/// Narrowing after two paths join: what both narrowed, as the union.
fn merge_narrow(a: &HashMap<String, Ty>, b: &HashMap<String, Ty>) -> HashMap<String, Ty> {
    a.iter().filter_map(|(k, t)| b.get(k).map(|u| (k.clone(), union(vec![t.clone(), u.clone()])))).collect()
}

/// The names and `x.attr` paths a block assigns.
fn assigned_keys(body: &[Stmt], out: &mut Vec<String>) {
    fn target(e: &Expr, out: &mut Vec<String>) {
        match e {
            Expr::Name(n) => out.push(n.id.to_string()),
            Expr::Attribute(a) => out.extend(attr_path(a)),
            Expr::Tuple(t) => t.elts.iter().for_each(|x| target(x, out)),
            Expr::List(l) => l.elts.iter().for_each(|x| target(x, out)),
            _ => {}
        }
    }
    for s in body {
        match s {
            Stmt::Assign(a) => a.targets.iter().for_each(|t| target(t, out)),
            Stmt::AnnAssign(a) => target(&a.target, out),
            Stmt::AugAssign(a) => target(&a.target, out),
            Stmt::For(f) => {
                target(&f.target, out);
                assigned_keys(&f.body, out);
            }
            Stmt::While(w) => assigned_keys(&w.body, out),
            Stmt::If(i) => {
                assigned_keys(&i.body, out);
                i.elif_else_clauses.iter().for_each(|c| assigned_keys(&c.body, out));
            }
            Stmt::Try(t) => {
                assigned_keys(&t.body, out);
                for h in &t.handlers {
                    let ast::ExceptHandler::ExceptHandler(h) = h;
                    out.extend(h.name.as_ref().map(|n| n.to_string()));
                    assigned_keys(&h.body, out);
                }
                assigned_keys(&t.orelse, out);
                assigned_keys(&t.finalbody, out);
            }
            Stmt::With(w) => {
                w.items.iter().filter_map(|i| i.optional_vars.as_deref()).for_each(|t| target(t, out));
                assigned_keys(&w.body, out);
            }
            _ => {}
        }
    }
}

fn is_builtin_call(c: &ast::ExprCall, name: &str) -> bool {
    matches!(&*c.func, Expr::Name(n) if n.id.as_str() == name)
}

/// An int literal, possibly negated.
fn int_literal(e: &Expr) -> Option<i64> {
    match e {
        Expr::NumberLiteral(n) => match &n.value {
            ast::Number::Int(i) => i.as_i64(),
            _ => None,
        },
        Expr::UnaryOp(u) if u.op == ast::UnaryOp::USub => int_literal(&u.operand).map(|i| -i),
        _ => None,
    }
}

fn target_names(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Name(n) => out.push(n.id.to_string()),
        Expr::Tuple(t) => t.elts.iter().for_each(|x| target_names(x, out)),
        Expr::List(l) => l.elts.iter().for_each(|x| target_names(x, out)),
        _ => {}
    }
}

/// Attributes declared as `self.x: T = ...` in an `__init__` body.
fn init_attr_names(body: &[Stmt], out: &mut Vec<(String, u32)>) {
    for s in body {
        match s {
            Stmt::AnnAssign(a) => {
                if let Expr::Attribute(at) = &*a.target {
                    out.push((at.attr.to_string(), a.range().start().into()));
                }
            }
            Stmt::If(i) => {
                init_attr_names(&i.body, out);
                for c in &i.elif_else_clauses {
                    init_attr_names(&c.body, out);
                }
            }
            Stmt::For(f) => init_attr_names(&f.body, out),
            Stmt::While(w) => init_attr_names(&w.body, out),
            _ => {}
        }
    }
}

fn has_break(body: &[Stmt]) -> bool {
    body.iter().any(|s| match s {
        Stmt::Break(_) => true,
        Stmt::If(i) => has_break(&i.body) || i.elif_else_clauses.iter().any(|c| has_break(&c.body)),
        _ => false,
    })
}

fn always_exits(body: &[Stmt]) -> bool {
    match body.last() {
        Some(Stmt::Return(_) | Stmt::Raise(_)) => true,
        Some(Stmt::If(i)) => {
            always_exits(&i.body)
                && i.elif_else_clauses.last().is_some_and(|c| c.test.is_none())
                && i.elif_else_clauses.iter().all(|c| always_exits(&c.body))
        }
        Some(Stmt::While(w)) => matches!(&*w.test, Expr::BooleanLiteral(b) if b.value) && !has_break(&w.body),
        Some(Stmt::Try(t)) => {
            always_exits(&t.finalbody)
                || ((always_exits(&t.body) || always_exits(&t.orelse))
                    && t.handlers.iter().all(|h| {
                        let ast::ExceptHandler::ExceptHandler(h) = h;
                        always_exits(&h.body)
                    }))
        }
        // `__exit__` may suppress an exception, so only a `return` exits.
        Some(Stmt::With(w)) => matches!(w.body.last(), Some(Stmt::Return(_))) || matches!(w.body.last(), Some(s) if always_exits(std::slice::from_ref(s)) && !matches!(s, Stmt::Raise(_))),
        _ => false,
    }
}
