//! Type checking of a Lizzard module and lowering of its functions to IR.
//!
//! One pass does both: every expression is typed and lowered together.
//! Top-level statements (run by CPython) are checked the same way into a
//! throwaway function. Methods are lowered like functions, under their
//! qualified names (`C.m`).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use ruff_python_ast::visitor::{walk_except_handler, walk_expr, walk_stmt, Visitor};
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
    /// `reg` holds a cell (the variable is captured by an inner function).
    cell: bool,
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
    Set(&'e Expr),
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
    /// Cells of the variables inner functions capture.
    cell_regs: HashMap<String, Reg>,
    /// The variables this function's closure provides, in cell order.
    free: Vec<(String, Ty)>,
    nonlocal_names: Vec<String>,
    /// Inner functions and lambdas lowered inside this function.
    nested: Vec<FuncIr>,
    /// In a generator function: what it yields and what `send` gives.
    /// (`ret` is what its `return` gives.)
    gen: Option<(Ty, Ty)>,
    /// Nesting depth of comprehensions being lowered (no `yield` there).
    in_comprehension: u32,
    /// Type parameters in scope for annotations in the body (the
    /// function's own, its class's, and the enclosing functions').
    tparams: Vec<(String, TvId)>,
}

/// What a function's body is: statements, or a lambda's expression.
#[derive(Clone, Copy)]
enum FnBody<'b> {
    Stmts(&'b [Stmt]),
    Lambda(&'b Expr),
}

/// Everything `build_function` needs to lower one function.
struct FnSpec<'b> {
    qual: String,
    key: String,
    /// The name in messages (`f`, `<lambda>`).
    name: String,
    name_off: u32,
    line: u32,
    sig: Sig,
    params: Option<&'b ast::Parameters>,
    body: FnBody<'b>,
    method: Option<MethodCtx>,
    has_receiver: bool,
    free: Vec<(String, Ty)>,
    rigid: Vec<TvId>,
    /// The expected result of a lambda.
    ret_hint: Option<Ty>,
    /// Defaults are evaluated where the function is created (inner
    /// functions and lambdas), not literals in the IR.
    dynamic_defaults: bool,
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
    funcs.append(&mut top.nested);
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
            loops: Vec::new(),
            method: None,
            narrow: HashMap::new(),
            global_names: Vec::new(),
            bound: Vec::new(),
            handlers: Vec::new(),
            regions: Vec::new(),
            cleanups: Vec::new(),
            handled: Vec::new(),
            cell_regs: HashMap::new(),
            free: Vec::new(),
            nonlocal_names: Vec::new(),
            nested: Vec::new(),
            gen: None,
            in_comprehension: 0,
            tparams: Vec::new(),
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

    /// The type arguments a builtin container `class` needs to be assignable
    /// to the expected type `hint` (`Iterable[float]` gives `list[float]`'s
    /// `[float]`), if they follow from it.
    fn container_hint(&mut self, hint: Option<&Ty>, class: &str) -> Option<Vec<Ty>> {
        let c = self.db.builtin_class(class)?;
        let ps = self.db.class_params(c);
        let me = Ty::Inst(c, ps.iter().map(|p| Ty::Var(*p)).collect());
        for h in members(hint?) {
            let Ty::Inst(hc, _) = &h else { continue };
            // `class` seen as the expected class, in terms of its own type
            // parameters; matching the hint into it solves them.
            let Some(view) = self.upcast(&me, *hc) else { continue };
            let mut s = Subst::new();
            if self.db.assign(&h, &view, &mut s, &mut false) {
                let args: Vec<Ty> = ps.iter().map(|p| s.get(p).cloned().unwrap_or(Ty::Object)).collect();
                if !args.iter().any(mentions_any) {
                    return Some(args);
                }
            }
        }
        None
    }

    /// `t` (a class instance) as an instance of its base class `target`.
    fn upcast(&mut self, t: &Ty, target: ClassId) -> Option<Ty> {
        self.db.base_view(t, target)
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
            Ty::Lit(_) | Ty::Union(_) if has_lit(t) => self.lookup_ty(&decay(t)),
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
        // Literals join to their union (`"r" if c else "w"` where a literal
        // type is expected), or with anything else to their base type.
        if has_lit(a) && has_lit(b) {
            return union(vec![a.clone(), b.clone()]);
        }
        if has_lit(a) || has_lit(b) {
            return self.join(&decay(a), &decay(b));
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
            // The value itself is not checked: a literal type is a promise
            // about Lizzard code, which the checker keeps.
            Ty::Lit(l) => self.rtcheck(&l.base()),
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
            // `isinstance` does not work with a Protocol (unless it is
            // runtime-checkable): the static check is the check.
            Ty::Inst(c, _) if self.db.is_protocol(*c) => RtCheck::Any,
            Ty::Inst(c, _) => RtCheck::Instance(self.db.runtime_class(t).unwrap_or_else(|| self.db.qualname(*c).to_string())),
            // A rigid variable (`Self`, a class's type parameter) is checked
            // against its bound.
            Ty::Callable(..) => RtCheck::Callable,
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
        if self.db.dataclass_opts(c).is_some() {
            if cd.decorator_list.len() > 1 {
                self.err(off, "class decorators other than @dataclass are not supported by Lizzard yet");
            }
            self.db.dataclass_fields(c);
        } else if !cd.decorator_list.is_empty() {
            self.err(off, "class decorators other than @dataclass are not supported by Lizzard yet");
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
                // A protocol's methods only declare an interface: they stay
                // Python functions.
                Stmt::FunctionDef(_) if self.db.is_protocol(c) => {}
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
        if d.is_async {
            self.err(off, "async functions are not supported by Lizzard yet");
            return None;
        }
        let accessor = ACCESSORS.into_iter().find(|k| is_accessor(d, name, k));
        for dec in &d.decorator_list {
            let ok = owner.is_some()
                && (matches!(&dec.expression, Expr::Name(n) if matches!(n.id.as_str(), "staticmethod" | "classmethod" | "property")) || accessor.is_some());
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
                Some(Member::Property(f)) if accessor.is_some() => match self.db.property_accessor(f, accessor.unwrap()) {
                    Some(af) => af,
                    None => return None,
                },
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
        // Its type variables (its own, its class's, `Self`) are rigid in
        // the body.
        let mut rigid = Vec::new();
        if let Some(c) = owner {
            rigid = self.db.class_params(c);
            rigid.push(self.db.self_tv(c));
        }
        for p in &sig.params {
            collect_vars(&p.ty, &mut rigid);
        }
        collect_vars(&sig.ret, &mut rigid);
        let has_receiver = owner.is_some() && kind != MethodKind::Static;
        let method = owner.map(|c| MethodCtx { class: c, receiver: if has_receiver { sig.params.first().map(|p| p.name.clone()) } else { None }, is_init: name == "__init__" });
        let key = match accessor {
            Some(kind) => format!("{qual}.{kind}"),
            None => qual.clone(),
        };
        let spec = FnSpec {
            qual,
            key,
            name: name.to_string(),
            name_off: d.name.range().start().into(),
            line: self.db.files[self.file].line_col(off).0,
            sig,
            params: Some(&d.parameters),
            body: FnBody::Stmts(&d.body),
            method,
            has_receiver,
            free: Vec::new(),
            rigid,
            ret_hint: None,
            dynamic_defaults: false,
        };
        Some(self.build_function(spec).0)
    }

    /// Lower one function (module-level, method, inner `def` or lambda).
    /// Returns its IR and, for a lambda, its body's type. Functions nested
    /// in it are added to `self.nested`.
    fn build_function(&mut self, spec: FnSpec) -> (FuncIr, Ty) {
        let path = self.db.files[self.file].path.clone();
        let saved_rigid = std::mem::replace(&mut self.db.rigid, spec.rigid);
        let scope = NameScope::of(spec.params, spec.body);
        // A generator function: `yield`s type against `Y`, `send` gives `S`,
        // and `return` gives `R` of its `Generator[Y, S, R]` result.
        let generator = matches!(spec.body, FnBody::Stmts(b) if contains_yield(b));
        let mut ret = spec.sig.ret.clone();
        let mut gen = None;
        if generator {
            match self.generator_types(&spec.sig.ret) {
                Some((y, s, r)) => {
                    gen = Some((y, s));
                    ret = r;
                }
                None => {
                    let t = self.show(&spec.sig.ret);
                    self.err(spec.name_off, format!("a generator function returns Iterator[T], Iterable[T] or Generator[Y, S, R], not {t}"));
                    gen = Some((Ty::Any, Ty::Any));
                    ret = Ty::Any;
                }
            }
        }
        let mut tparams = self.tparams.clone();
        if let Some(m) = &spec.method {
            tparams.extend(self.db.class_tparams(m.class));
        }
        let mut vars = Vec::new();
        for p in &spec.sig.params {
            collect_vars(&p.ty, &mut vars);
        }
        collect_vars(&spec.sig.ret, &mut vars);
        for v in vars {
            let name = self.db.tvs[v].name.clone();
            if !tparams.iter().any(|(n, _)| *n == name) {
                tparams.push((name, v));
            }
        }
        let mut l = Lower::new(self.db, self.m, self.file, false, spec.qual, ret);
        l.gen = gen;
        l.tparams = tparams;
        l.line = spec.line;
        l.bound = scope.local_names().into_iter().collect();
        l.method = spec.method;
        l.free = spec.free.clone();
        let params = match spec.params {
            Some(ps) => l.params(&spec.sig, ps, spec.has_receiver, spec.dynamic_defaults),
            None => l.params(&spec.sig, &ast::Parameters::default(), spec.has_receiver, spec.dynamic_defaults),
        };
        // Variables inner functions capture live in cells.
        let mut cells: Vec<String> = scope.cells().into_iter().collect();
        cells.sort();
        for c in cells {
            let cell = l.reg(Repr::Obj);
            l.emit(Instr::MakeCell { dst: cell });
            if let Some(p) = l.locals.get(&c).cloned() {
                l.emit(Instr::StoreCell { cell, val: p.reg });
                l.locals.insert(c.clone(), Local { reg: cell, ty: p.ty, repr: Repr::Obj, cell: true });
            }
            l.cell_regs.insert(c, cell);
        }
        let mut body_ty = Ty::None;
        match spec.body {
            FnBody::Stmts(body) => {
                for s in body {
                    l.stmt(s);
                }
                if !always_exits(body) {
                    if matches!(l.ret, Ty::None | Ty::Any) {
                        let none = l.konst(Const::None, Ty::None);
                        l.emit(Instr::Return { src: none.reg });
                    } else {
                        let want = l.show(&l.ret);
                        let n = spec.name.clone();
                        l.err(spec.name_off, format!("`{n}` may end without returning a value (declared to return {want})"));
                    }
                } else {
                    // Unreachable when the analysis is right; never run off the end.
                    let none = l.konst(Const::None, Ty::None);
                    l.emit(Instr::Return { src: none.reg });
                }
            }
            FnBody::Lambda(e) => {
                let v = l.expr_hint(e, spec.ret_hint.as_ref());
                body_ty = v.ty.clone();
                let r = l.obj(&v);
                l.emit(Instr::Return { src: r });
            }
        }
        let ir = FuncIr {
            name: l.fname,
            file: path,
            line: spec.line,
            params,
            reprs: l.reprs,
            code: l.code,
            lines: l.lines,
            consts: l.consts,
            names: l.names,
            checks: l.checks,
            handlers: l.handlers,
            key: spec.key,
            free: spec.free.into_iter().map(|(n, _)| n).collect(),
            generator,
        };
        let nested = l.nested;
        self.nested.extend(nested);
        self.db.rigid = saved_rigid;
        (ir, body_ty)
    }

    /// What a generator function annotated as returning `t` yields, accepts
    /// from `send` and returns.
    fn generator_types(&mut self, t: &Ty) -> Option<(Ty, Ty, Ty)> {
        if matches!(t, Ty::Any) {
            return Some((Ty::Any, Ty::Any, Ty::Any));
        }
        let Ty::Inst(c, args) = t else { return None };
        let name = self.db.classes[*c].name.clone();
        if !self.db.is_builtin_class(*c) {
            return None;
        }
        match (name.as_str(), args.as_slice()) {
            ("Generator", [y, s, r]) => Some((y.clone(), s.clone(), r.clone())),
            ("Iterator" | "Iterable", [y]) => Some((y.clone(), Ty::None, Ty::None)),
            _ => None,
        }
    }

    /// The cells to pass to an inner function whose closure needs `names`
    /// (some of which may be globals), with their types.
    fn closure_for(&mut self, names: &[String]) -> (Vec<(String, Ty)>, Vec<Reg>) {
        let mut free = Vec::new();
        let mut regs = Vec::new();
        for n in names {
            if let Some(&cell) = self.cell_regs.get(n) {
                // Not assigned yet where the inner function is created: Any.
                let ty = self.locals.get(n).map_or(Ty::Any, |l| l.ty.clone());
                free.push((n.clone(), ty));
                regs.push(cell);
            } else if let Some(i) = self.free.iter().position(|(f, _)| f == n) {
                let ty = self.free[i].1.clone();
                let dst = self.reg(Repr::Obj);
                self.emit(Instr::LoadFreeCell { dst, idx: i as u32 });
                free.push((n.clone(), ty));
                regs.push(dst);
            }
        }
        (free, regs)
    }

    /// The yield and send types where a `yield` is, or an error.
    fn yield_context(&mut self, off: u32) -> Option<(Ty, Ty)> {
        if self.in_comprehension > 0 {
            self.err(off, "`yield` inside a comprehension is not allowed");
            return None;
        }
        match self.gen.clone() {
            Some(g) => Some(g),
            None => {
                let msg = if self.toplevel { "`yield` outside a function" } else { "`yield` in a lambda is not supported by Lizzard" };
                self.err(off, msg);
                None
            }
        }
    }

    /// Evaluate an inner function's defaults where it is defined, in
    /// parameter order, checked against the parameters' types.
    fn eval_defaults(&mut self, sig: &Sig, ps: &ast::Parameters) -> Vec<Reg> {
        let defaults: HashMap<&str, &Expr> = ps
            .posonlyargs
            .iter()
            .chain(ps.args.iter())
            .chain(ps.kwonlyargs.iter())
            .filter_map(|p| p.default.as_deref().map(|d| (p.parameter.name.as_str(), d)))
            .collect();
        let mut regs = Vec::new();
        for p in &sig.params {
            if let Some(e) = defaults.get(p.name.as_str()) {
                let v = self.expr_hint(e, Some(&p.ty));
                let pn = p.name.clone();
                let v = self.coerce(v, &p.ty, e.range().start().into(), |_, got, want| format!("the default of `{pn}` is {got}, but the parameter is {want}"));
                regs.push(self.obj(&v));
            }
        }
        regs
    }

    /// The qualified name of a function defined in this one.
    fn inner_qual(&self, name: &str) -> String {
        if self.toplevel { name.to_string() } else { format!("{}.<locals>.{name}", self.fname) }
    }

    /// `def` inside a function (or in a block at the top level, where it is
    /// only checked: CPython runs it).
    fn nested_def(&mut self, d: &ast::StmtFunctionDef, off: u32) {
        let name = d.name.as_str();
        if d.is_async {
            return self.err(off, "async functions are not supported by Lizzard yet");
        }
        if let Some(dec) = d.decorator_list.first() {
            return self.err(dec.range().start().into(), "decorated inner functions are not supported by Lizzard yet");
        }
        let qual = self.inner_qual(name);
        let key = format!("{qual}@{off}");
        let f = self.db.local_func(self.m, &key, name, Rc::new(d.clone()), self.file);
        let sig = self.db.sigs(f)[0].clone();
        let mut rigid = self.db.rigid.clone();
        for p in &sig.params {
            collect_vars(&p.ty, &mut rigid);
        }
        collect_vars(&sig.ret, &mut rigid);
        let fty = Ty::Func(FuncRef::Def(f));
        // A recursive inner function captures its own name: its cell has
        // its type from the start.
        if let Some(&cell) = self.cell_regs.get(name) {
            self.locals.entry(name.to_string()).or_insert(Local { reg: cell, ty: fty.clone(), repr: Repr::Obj, cell: true });
        }
        let mut names: Vec<String> = NameScope::of(Some(&d.parameters), FnBody::Stmts(&d.body)).free().into_iter().collect();
        names.sort();
        let defaults = self.eval_defaults(&sig, &d.parameters);
        let (free, regs) = self.closure_for(&names);
        let spec = FnSpec {
            qual,
            key: key.clone(),
            name: name.to_string(),
            name_off: d.name.range().start().into(),
            line: self.db.files[self.file].line_col(off).0,
            sig,
            params: Some(&d.parameters),
            body: FnBody::Stmts(&d.body),
            method: None,
            has_receiver: false,
            free,
            rigid,
            ret_hint: None,
            dynamic_defaults: true,
        };
        let (ir, _) = self.build_function(spec);
        if !self.toplevel {
            self.nested.push(ir);
        }
        let dst = self.reg(Repr::Obj);
        let k = self.name(&key);
        self.emit(Instr::MakeFunction { dst, key: k, closure: regs, defaults });
        self.assign_local(name, Val { reg: dst, ty: fty, repr: Repr::Obj }, None, off);
    }

    /// `lambda params: body`, its parameter types taken from the expected
    /// `Callable` type.
    fn lambda(&mut self, l: &ast::ExprLambda, hint: Option<&Ty>, off: u32) -> Val {
        let mut names = Vec::new();
        let mut defaults = Vec::new();
        if let Some(ps) = l.parameters.as_deref() {
            if ps.vararg.is_some() || ps.kwarg.is_some() || !ps.kwonlyargs.is_empty() {
                self.err(off, "lambdas with `*`, `**` or keyword-only parameters are not supported by Lizzard yet");
            }
            for p in ps.posonlyargs.iter().chain(ps.args.iter()) {
                names.push(p.parameter.name.to_string());
                defaults.push(p.default.is_some());
            }
        }
        // The expected `Callable` gives the parameters without defaults (or
        // all of them); the others take their default's type.
        let required = defaults.iter().filter(|d| !**d).count();
        // A callback protocol stands for its `__call__`.
        let cands: Vec<Ty> = hint.into_iter().flat_map(members).collect();
        let cands: Vec<Ty> = cands
            .into_iter()
            .map(|m| match &m {
                Ty::Inst(c, _) if self.db.is_protocol(*c) => self.db.callable_of(&m).unwrap_or(m),
                _ => m,
            })
            .collect();
        let expected = cands.into_iter().find_map(|m| match m {
            Ty::Callable(Some(ps), r) if ps.len() == required || ps.len() == names.len() => Some((ps, *r)),
            Ty::Callable(None, r) if required == 0 => Some((Vec::new(), *r)),
            _ => None,
        });
        let concrete = |db: &Db, t: &Ty| {
            let mut v = Vec::new();
            collect_vars(t, &mut v);
            v.iter().all(|x| db.rigid.contains(x))
        };
        let (mut ptys, ret_hint): (Vec<Option<Ty>>, Option<Ty>) = match expected {
            Some((ps, r)) => {
                let ps = ps.iter().map(|p| Some(if concrete(self.db, p) { p.clone() } else { Ty::Any })).collect();
                (ps, concrete(self.db, &r).then_some(r))
            }
            None => (Vec::new(), None),
        };
        ptys.resize(names.len(), None);
        let mut default_regs = Vec::new();
        if let Some(ps) = l.parameters.as_deref() {
            for (i, p) in ps.posonlyargs.iter().chain(ps.args.iter()).enumerate() {
                if let Some(dflt) = &p.default {
                    let v = self.expr_hint(dflt, ptys[i].as_ref());
                    let v = match ptys[i].clone() {
                        Some(t) => self.coerce(v, &t, dflt.range().start().into(), |_, got, want| format!("the default of `{}` is {got}, but the parameter is {want}", p.parameter.name)),
                        None => {
                            ptys[i] = Some(v.ty.clone());
                            v
                        }
                    };
                    default_regs.push(self.obj(&v));
                }
            }
        }
        if ptys.iter().any(Option::is_none) {
            self.err(off, "cannot infer the parameter types of this lambda: use it where a `Callable[[...], ...]` is expected, or use a `def`");
        }
        let ptys: Vec<Ty> = ptys.into_iter().map(|t| t.unwrap_or(Ty::Any)).collect();
        let params: Vec<Param> = names.iter().zip(&ptys).zip(&defaults).map(|((n, t), d)| Param { name: n.clone(), ty: t.clone(), kind: PKind::PosOrKw, default: *d }).collect();
        let sig = Sig { params, ret: Ty::Any, file: self.file, off };
        let qual = self.inner_qual("<lambda>");
        let key = format!("{qual}@{off}");
        let mut fv: Vec<String> = NameScope::of(l.parameters.as_deref(), FnBody::Lambda(&l.body)).free().into_iter().collect();
        fv.sort();
        let (free, regs) = self.closure_for(&fv);
        let spec = FnSpec {
            qual,
            key: key.clone(),
            name: "<lambda>".into(),
            name_off: off,
            line: self.line,
            sig,
            params: l.parameters.as_deref(),
            body: FnBody::Lambda(&l.body),
            method: None,
            has_receiver: false,
            free,
            rigid: self.db.rigid.clone(),
            ret_hint,
            dynamic_defaults: true,
        };
        let (ir, body_ty) = self.build_function(spec);
        if !self.toplevel {
            self.nested.push(ir);
        }
        let dst = self.reg(Repr::Obj);
        let k = self.name(&key);
        self.emit(Instr::MakeFunction { dst, key: k, closure: regs, defaults: default_regs });
        // Callers see the parameters without defaults.
        let shown: Vec<Ty> = ptys.iter().zip(&defaults).filter(|(_, d)| !**d).map(|(t, _)| t.clone()).collect();
        Val { reg: dst, ty: Ty::Callable(Some(shown), Box::new(body_ty)), repr: Repr::Obj }
    }

    /// Bind a signature's parameters to registers `0..n` (in `sig` order).
    fn params(&mut self, sig: &Sig, ps: &ast::Parameters, has_receiver: bool, dynamic_defaults: bool) -> Vec<ParamIr> {
        let defaults: HashMap<&str, &Expr> = ps
            .posonlyargs
            .iter()
            .chain(ps.args.iter())
            .chain(ps.kwonlyargs.iter())
            .filter_map(|p| p.default.as_deref().map(|d| (p.parameter.name.as_str(), d)))
            .collect();
        let mut params = Vec::new();
        for (i, p) in sig.params.iter().enumerate() {
            let (kind, local_ty) = match p.kind {
                PKind::PosOnly => (ParamKind::PosOnly, p.ty.clone()),
                PKind::PosOrKw => (ParamKind::PosOrKw, p.ty.clone()),
                PKind::KwOnly => (ParamKind::KwOnly, p.ty.clone()),
                PKind::VarPos => (ParamKind::VarPos, Ty::TupleVar(Box::new(p.ty.clone()))),
                PKind::VarKw => (ParamKind::VarKw, self.builtin_inst("dict", vec![Ty::Str, p.ty.clone()])),
            };
            let reg = self.reg(Repr::Obj);
            self.locals.insert(p.name.clone(), Local { reg, ty: local_ty, repr: Repr::Obj, cell: false });
            let expected = self.show(&p.ty);
            // The receiver is bound by Python itself.
            let check = if i == 0 && has_receiver { RtCheck::Any } else { self.rtcheck(&p.ty) };
            self.checks.push(CheckIr { check, what: format!("argument '{}'", p.name), expected, origin: None });
            let dynamic_default = dynamic_defaults && defaults.contains_key(p.name.as_str());
            let default = match defaults.get(p.name.as_str()) {
                Some(e) if !dynamic_defaults => self.default_value(e, &p.ty, &p.name),
                _ => None,
            };
            params.push(ParamIr { name: p.name.clone(), kind, check: (self.checks.len() - 1) as u32, default, dynamic_default });
        }
        params
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
        let t = match literal_value(e) {
            Some(l) if has_lit(ty) => Ty::Lit(l),
            _ => t,
        };
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
                        // `del x` for a local: it is unbound afterwards.
                        Expr::Name(n) if self.locals.get(n.id.as_str()).is_some_and(|l| !l.cell) => {
                            let reg = self.locals[n.id.as_str()].reg;
                            let check = self.reg(self.reprs[reg as usize]);
                            self.emit(Instr::Move { dst: check, src: reg });
                            self.emit(Instr::Unbind { reg });
                            self.emit(Instr::Unbind { reg: check });
                            self.invalidate(n.id.as_str());
                        }
                        Expr::Attribute(at) => {
                            let obj = self.expr(&at.value);
                            self.del_attr(&obj, at);
                        }
                        _ => self.err(off, "only `del x[key]`, `del x.attr` and `del local` are supported by Lizzard yet"),
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
            Stmt::FunctionDef(d) => self.nested_def(d, off),
            Stmt::Nonlocal(n) if !self.toplevel => {
                for name in &n.names {
                    if self.free.iter().any(|(f, _)| f == name.as_str()) {
                        self.nonlocal_names.push(name.to_string());
                    } else {
                        self.err(off, format!("no binding for nonlocal `{}` found", name.as_str()));
                    }
                }
            }
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
                    self.locals.insert(name.clone(), Local { reg, ty, repr: Repr::Obj, cell: false });
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
        let tps = self.tparams.clone();
        let self_ty = self.method.as_ref().map(|m| Ty::Var(self.db.self_tv(m.class)));
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
            self.locals.insert(name.to_string(), Local { reg, ty, repr: Repr::Obj, cell: false });
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
                let v = self.binop_vals(a.op, cur, r, int_literal(&a.value), what, off, true);
                self.assign_local(n.id.as_str(), v, None, off);
            }
            Expr::Attribute(at) => {
                let obj = self.expr(&at.value);
                let cur = self.attr_of(&obj, at, what.clone());
                let r = self.expr(&a.value);
                let v = self.binop_vals(a.op, cur, r, int_literal(&a.value), what, off, true);
                self.store_attr(&obj, at, v);
            }
            Expr::Subscript(sub) => {
                let obj = self.expr(&sub.value);
                let key = self.expr(&sub.slice);
                let cur = self.item_of(&obj, &key, sub, what.clone());
                let r = self.expr(&a.value);
                let v = self.binop_vals(a.op, cur, r, int_literal(&a.value), what, off, true);
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
        if let Some(i) = self.nonlocal_names.iter().any(|n| n == name).then(|| self.free.iter().position(|(f, _)| f == name)).flatten() {
            let t = self.free[i].1.clone();
            let v = self.coerce(v, &t, off, |_, got, want| format!("cannot assign {got} to `{name}` of type {want}"));
            let r = self.obj(&v);
            self.emit(Instr::StoreFree { idx: i as u32, val: r });
            return;
        }
        if let Some(&cell) = self.cell_regs.get(name) {
            let declared = self.locals.get(name).map(|l| l.ty.clone());
            let v = match declared.as_ref().or(ann.as_ref()) {
                Some(t) => {
                    let t = t.clone();
                    let v = self.coerce(v, &t, off, |_, got, want| format!("cannot assign {got} to `{name}` of type {want}"));
                    Val { ty: t, ..v }
                }
                None => v,
            };
            let r = self.obj(&v);
            self.emit(Instr::StoreCell { cell, val: r });
            if declared.is_none() {
                self.locals.insert(name.to_string(), Local { reg: cell, ty: v.ty, repr: Repr::Obj, cell: true });
            }
            self.invalidate(name);
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
            if r == v.reg {
                self.release_temp(&v);
            }
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
            // An unannotated variable takes a literal's base type, as in mypy
            // (`n = one()` for `def one() -> Literal[1]` is an int).
            None if matches!(v.ty, Ty::Lit(_)) => Val { ty: decay(&v.ty), ..v },
            None => v,
        };
        let reg = self.reg(v.repr);
        self.emit(Instr::Move { dst: reg, src: v.reg });
        self.release_temp(&v);
        self.invalidate(name);
        if let Some(a) = &ann {
            self.narrow_to(name.to_string(), a, &v.ty);
        }
        self.locals.insert(name.to_string(), Local { reg, ty: v.ty, repr: v.repr, cell: false });
    }

    /// A temporary register's object, now held by a local, is released, so
    /// the local holds the only reference (`del x` frees it at once).
    fn release_temp(&mut self, v: &Val) {
        if v.repr == Repr::Obj && !self.locals.values().any(|l| l.reg == v.reg) && !self.handlers.iter().any(|h| h.exc == v.reg) {
            self.emit(Instr::Unbind { reg: v.reg });
        }
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
            Expr::Name(n) if self.locals.get(n.id.as_str()).is_some_and(|l| !l.cell) => Some(n.id.to_string()),
            Expr::Attribute(a) => attr_path(a).filter(|p| self.locals.get(p.split('.').next().unwrap()).is_some_and(|l| !l.cell)),
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
            // `x == "r"` for x of a literal type: x is `Literal["r"]`, and
            // otherwise one of the other members.
            Expr::Compare(c) if c.ops.len() == 1 && matches!(c.ops[0], ast::CmpOp::Eq | ast::CmpOp::NotEq) => {
                let (subject, lit) = match (literal_value(&c.operands[0]), literal_value(&c.operands[1])) {
                    (None, Some(l)) => (&c.operands[0], l),
                    (Some(l), None) => (&c.operands[1], l),
                    _ => return none,
                };
                let Some(key) = self.narrow_key(subject) else { return none };
                let Some(cur) = self.key_type(&key, subject) else { return none };
                let ms = members(&cur);
                if !ms.contains(&Ty::Lit(lit.clone())) {
                    return none;
                }
                let yes = vec![(key.clone(), Ty::Lit(lit.clone()))];
                // Only literal members can be ruled out: a `str` member may
                // still equal "r".
                let rest: Vec<Ty> = ms.iter().filter(|m| **m != Ty::Lit(lit.clone())).cloned().collect();
                let no = if !rest.is_empty() { vec![(key, union(rest))] } else { Vec::new() };
                if c.ops[0] == ast::CmpOp::Eq { (yes, no) } else { (no, yes) }
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
        // Where a `Literal` type is expected, a literal is its literal type.
        if let (Some(h), Some(l)) = (hint, literal_value(e)) {
            if has_lit(h) {
                let v = self.expr_hint(e, None);
                return Val { ty: Ty::Lit(l), ..v };
            }
        }
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
                let tps = self.tparams.clone();
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
                self.binop_vals(b.op, l, r, int_literal(&b.right), what, off, false)
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
            Expr::Set(st) => {
                let want = self.container_hint(hint, "set").and_then(|a| a.into_iter().next());
                let refs: Vec<&Expr> = st.elts.iter().collect();
                let (items, elem) = self.items(&refs, want.as_ref(), "a set item");
                let dst = self.reg(Repr::Obj);
                self.emit(Instr::BuildSet { dst, items });
                let ty = self.builtin_inst("set", vec![elem.unwrap_or(Ty::Any)]);
                Val { reg: dst, ty, repr: Repr::Obj }
            }
            Expr::SetComp(sc) => self.comprehension(&sc.generators, CompBody::Set(&sc.elt), hint, off),
            Expr::Tuple(t) => self.tuple_lit(&t.elts, hint, off),
            Expr::Dict(d) => self.dict_lit(d, hint, off),
            Expr::ListComp(lc) => self.comprehension(&lc.generators, CompBody::List(&lc.elt), hint, off),
            Expr::Lambda(l) => self.lambda(l, hint, off),
            Expr::Yield(y) => {
                let Some((yt, st)) = self.yield_context(off) else { return self.any() };
                let v = match &y.value {
                    Some(e) => self.expr_hint(e, Some(&yt)),
                    None => self.konst(Const::None, Ty::None),
                };
                let fname = self.fname.clone();
                let v = self.coerce(v, &yt, off, |_, got, want| format!("`{fname}` yields {want}, but this yields {got}"));
                let src = self.obj(&v);
                let dst = self.reg(Repr::Obj);
                self.emit(Instr::Yield { dst, src });
                // `next()` sends None whatever the send type is.
                let checked = Val { reg: dst, ty: union(vec![st.clone(), Ty::None]), repr: Repr::Obj };
                self.check(&checked, format!("the value sent to `{fname}`"), None);
                Val { reg: dst, ty: st, repr: Repr::Obj }
            }
            Expr::YieldFrom(y) => {
                let Some((yt, _)) = self.yield_context(off) else { return self.any() };
                let v = self.expr(&y.value);
                let elem = self.iter_elem(&v.ty, y.value.range().start().into());
                if !self.assignable(&elem, &yt) {
                    let (a, b, f) = (self.show(&elem), self.show(&yt), self.fname.clone());
                    self.err(off, format!("`yield from` yields {a}, but `{f}` yields {b}"));
                }
                // The result is the sub-generator's return value (None for
                // other iterators).
                let ty = match self.lookup_ty(&v.ty) {
                    Ty::Any => Ty::Any,
                    t => match self.db.builtin_class("Generator").and_then(|g| self.upcast(&t, g)) {
                        Some(Ty::Inst(_, args)) if args.len() == 3 => args[2].clone(),
                        _ => Ty::None,
                    },
                };
                let o = self.obj(&v);
                let iter = self.reg(Repr::Obj);
                self.emit(Instr::GetIter { dst: iter, obj: o });
                let dst = self.reg(Repr::Obj);
                self.emit(Instr::YieldFrom { dst, iter });
                let r = Val { reg: dst, ty, repr: Repr::Obj };
                let what = self.src(e);
                self.check(&r, what, None);
                r
            }
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
                        let ty = match &decay(&v.ty) {
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
                ast::UnaryOp::Invert | ast::UnaryOp::UAdd => {
                    let invert = u.op == ast::UnaryOp::Invert;
                    let v = self.expr(&u.operand);
                    let ty = match &decay(&v.ty) {
                        Ty::Any => Ty::Any,
                        Ty::Int | Ty::Bool => Ty::Int,
                        Ty::Float if !invert => Ty::Float,
                        t => {
                            let t = t.clone();
                            match self.dunder(&t, if invert { "__invert__" } else { "__pos__" }, &[]) {
                                Some(Ok((r, _, _))) => r,
                                _ => {
                                    let ts = self.show(&t);
                                    self.err(off, format!("bad operand type for unary {}: {ts}", if invert { "~" } else { "+" }));
                                    Ty::Any
                                }
                            }
                        }
                    };
                    let src = self.obj(&v);
                    let dst = self.reg(Repr::Obj);
                    self.emit(if invert { Instr::Invert { dst, src } } else { Instr::Pos { dst, src } });
                    let r = Val { reg: dst, ty, repr: Repr::Obj };
                    let what = self.src(e);
                    self.check(&r, what, None);
                    self.unbox_f32(r)
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
        if let Some(l) = self.locals.get(name).cloned() {
            if l.cell {
                let dst = self.reg(Repr::Obj);
                self.emit(Instr::LoadCell { dst, cell: l.reg });
                return Val { reg: dst, ty: l.ty, repr: Repr::Obj };
            }
            let ty = self.narrow.get(name).cloned().unwrap_or_else(|| l.ty.clone());
            return Val { reg: l.reg, ty, repr: l.repr };
        }
        if let Some(i) = self.free.iter().position(|(f, _)| f == name) {
            let ty = self.free[i].1.clone();
            let dst = self.reg(Repr::Obj);
            self.emit(Instr::LoadFree { dst, idx: i as u32 });
            return Val { reg: dst, ty, repr: Repr::Obj };
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
            Member::Property(f) => match self.resolve(self.db.property_getter(f), Some(obj_ty.clone()), &[], &[], None) {
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
            None if OBJECT_ATTRS.contains(&name) => Ty::Any,
            None => {
                let hint = self.untyped_hint(class);
                self.err(a.attr.range().start().into(), format!("no base class of `{cname}` has an attribute `{name}`{hint}"));
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
                None if OBJECT_ATTRS.contains(&name) => Ok(Ty::Any),
                // A builtin exception base's attributes (`args`, `__cause__`, ...).
                None => match self.db.exc_base(*c).and_then(|e| exc_attr(&e, name)) {
                    Some(t) => Ok(t),
                    None => Err(format!("{} has no attribute `{name}`{}", self.show(&lt), self.untyped_hint(*c))),
                },
            },
            (Ty::Type(t), _) if matches!(self.lookup_ty(t), Ty::Inst(..)) => {
                let Ty::Inst(c, _) = self.lookup_ty(t) else { unreachable!() };
                let inst = (**t).clone();
                match self.db.find_member(c, name) {
                    Some((owner, m)) => Ok(self.member_value_ty(&inst, owner, m, true, origin)),
                    // `type`'s own attributes (`__name__`, `__mro__`, ...).
                    None if OBJECT_ATTRS.contains(&name) || TYPE_ATTRS.contains(&name) => Ok(Ty::Any),
                    None => Err(format!("{} has no attribute `{name}`{}", self.show(obj_ty), self.untyped_hint(c))),
                }
            }
            (Ty::Exc(e), _) => match exc_attr(e, name) {
                Some(t) => Ok(t),
                None => Err(format!("{e} has no attribute `{name}`")),
            },
            (Ty::Any, _) => Ok(Ty::Any),
            // `type`'s own attributes of any class (`__name__`, ...).
            (Ty::Type(_), _) if OBJECT_ATTRS.contains(&name) || TYPE_ATTRS.contains(&name) => Ok(Ty::Any),
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

    /// For a message: why a class's members are unknown.
    fn untyped_hint(&mut self, c: ClassId) -> String {
        let bases = self.db.untyped_bases(c);
        if bases.is_empty() {
            return String::new();
        }
        let names: Vec<String> = bases.iter().map(|b| format!("`{b}`")).collect();
        format!(" (its base class {} has no type definitions)", names.join(", "))
    }

    /// `del obj.name`: a property needs a deleter.
    fn del_attr(&mut self, obj: &Val, a: &ast::ExprAttribute) {
        let off: u32 = a.attr.range().start().into();
        let name = a.attr.as_str();
        if let Ty::Inst(c, _) = self.lookup_ty(&obj.ty) {
            let cn = self.db.classes[c].name.clone();
            match self.db.find_member(c, name) {
                _ if self.db.is_frozen(c) => self.err(off, format!("cannot delete field `{name}` of frozen dataclass `{cn}`")),
                Some((_, Member::Property(f))) if self.db.property_accessor(f, "deleter").is_none() => {
                    self.err(off, format!("property `{name}` has no deleter"));
                }
                Some((_, Member::Method(_))) => self.err(off, format!("cannot delete method `{name}`")),
                None => {
                    let hint = self.untyped_hint(c);
                    self.err(off, format!("{cn} has no attribute `{name}`{hint}"));
                }
                _ => {}
            }
        } else if !matches!(obj.ty, Ty::Any) {
            let t = self.show(&obj.ty);
            self.err(off, format!("cannot delete attribute `{name}` of {t}"));
        }
        let o = self.obj(obj);
        let n = self.name(name);
        self.emit(Instr::DelAttr { obj: o, name: n });
        if let Some(p) = attr_path(a) {
            self.invalidate(&p);
        }
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
            Ty::Inst(c, _) if !matches!(obj.ty, Ty::Type(_)) && self.db.is_frozen(*c) => {
                let cn = self.db.classes[*c].name.clone();
                self.err(off, format!("cannot assign to field `{name}` of frozen dataclass `{cn}`"));
                val
            }
            Ty::Inst(c, _) => match self.db.find_member(*c, name) {
                Some((owner, Member::Attr(t))) => {
                    let want = self.member_type(&inst_ty, owner, &t);
                    self.coerce(val, &want, off, |_, got, want| format!("cannot assign {got} to attribute `{name}` of type {want}"))
                }
                Some((_, Member::Method(_))) => {
                    self.err(off, format!("cannot assign to method `{name}`"));
                    val
                }
                Some((_, Member::Property(f))) => {
                    match self.db.property_setter(f) {
                        Some(sf) => {
                            if let Err(m) = self.resolve(sf, Some(inst_ty.clone()), std::slice::from_ref(&val.ty), &[], None) {
                                self.err(off, m.replacen(&format!("{name}()"), &format!("the setter of `{name}`"), 1));
                            }
                        }
                        None => self.err(off, format!("property `{name}` has no setter")),
                    }
                    val
                }
                None => {
                    let t = self.show(&lt);
                    let hint = self.untyped_hint(*c);
                    self.err(
                        off,
                        format!("{t} has no attribute `{name}`{hint}; declare it in the class body (`{name}: T`) or in `__init__` (`self.{name}: T = ...`)"),
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
        // `reveal_type(x)` reports x's static type as a note, like mypy's, and
        // evaluates to x. Imported from `typing`, it is also called (and
        // prints x's runtime type); otherwise it is only a note.
        let reveal = is_builtin_call(c, "reveal_type") && !self.locals.contains_key("reveal_type") && c.arguments.args.len() == 1 && c.arguments.keywords.is_empty();
        if reveal && self.db.lookup_in(self.m, "reveal_type").is_none() {
            let v = self.expr(&c.arguments.args[0]);
            self.reveal(&v.ty, off);
            return v;
        }
        let fv = self.expr(&c.func);
        if c.arguments.args.iter().any(|a| matches!(a, Expr::Starred(_))) || c.arguments.keywords.iter().any(|k| k.arg.is_none()) {
            return self.star_call(c, fv, off);
        }
        let (pos_hints, kw_hints) = self.arg_hints(&fv.ty);
        // Lambdas are typed last, from what the other arguments tell about
        // the callee's type variables (creating them has no side effects).
        let is_lambda = |e: &Expr| matches!(e, Expr::Lambda(_));
        let mut pos: Vec<Option<Val>> = Vec::new();
        for (i, a) in c.arguments.args.iter().enumerate() {
            let hint = pos_hints.get(i).cloned().flatten();
            pos.push((!is_lambda(a)).then(|| self.expr_hint(a, hint.as_ref())));
        }
        let mut kws: Vec<(String, Option<Val>)> = Vec::new();
        for k in c.arguments.keywords.iter() {
            let name = k.arg.as_ref().unwrap();
            let hint = kw_hints.get(name.as_str()).cloned();
            kws.push((name.to_string(), (!is_lambda(&k.value)).then(|| self.expr_hint(&k.value, hint.as_ref()))));
        }
        if pos.iter().any(Option::is_none) || kws.iter().any(|(_, v)| v.is_none()) {
            let known_pos: Vec<Ty> = pos.iter().map(|v| v.as_ref().map_or(Ty::Any, |v| v.ty.clone())).collect();
            let known_kws: Vec<(String, Ty)> = kws.iter().map(|(n, v)| (n.clone(), v.as_ref().map_or(Ty::Any, |v| v.ty.clone()))).collect();
            let (lp, lk) = self.lambda_hints(&fv.ty, &known_pos, &known_kws);
            for (i, a) in c.arguments.args.iter().enumerate() {
                if pos[i].is_none() {
                    let hint = lp.get(i).cloned().flatten().or_else(|| pos_hints.get(i).cloned().flatten());
                    pos[i] = Some(self.expr_hint(a, hint.as_ref()));
                }
            }
            for (j, k) in c.arguments.keywords.iter().enumerate() {
                if kws[j].1.is_none() {
                    let name = kws[j].0.clone();
                    let hint = lk.get(&name).cloned().or_else(|| kw_hints.get(&name).cloned());
                    kws[j].1 = Some(self.expr_hint(&k.value, hint.as_ref()));
                }
            }
        }
        let pos: Vec<Val> = pos.into_iter().map(Option::unwrap).collect();
        let kws: Vec<(String, Val)> = kws.into_iter().map(|(n, v)| (n, v.unwrap())).collect();
        // A literal argument is its literal type, so that it matches a
        // `Literal` parameter (and picks a `Literal` overload).
        let lit_ty = |e: &Expr, v: &Val| literal_value(e).map_or_else(|| v.ty.clone(), Ty::Lit);
        let pos_tys: Vec<Ty> = pos.iter().zip(c.arguments.args.iter()).map(|(v, e)| lit_ty(e, v)).collect();
        let kw_tys: Vec<(String, Ty)> = kws.iter().zip(c.arguments.keywords.iter()).map(|((n, v), k)| (n.clone(), lit_ty(&k.value, v))).collect();
        let mut origin = None;
        let mut builtin_ctor = false;
        let mut ret = self.call_type(fv.ty.clone(), &pos_tys, &kw_tys, off, &mut origin, &mut builtin_ctor);
        if reveal {
            self.reveal(&pos[0].ty, off);
            ret = pos[0].ty.clone();
        }
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

    fn reveal(&mut self, t: &Ty, off: u32) {
        let (line, col) = self.db.files[self.file].line_col(off);
        let note = format!("{}:{line}:{col}: note: revealed type is `{}`", self.db.files[self.file].path, self.show(t));
        self.db.warnings.push(note);
    }

    /// The parameter types the arguments of a call bind to, with the
    /// callee's type variables solved from the known argument types (`Any`
    /// marks the ones not known yet): what a lambda argument should be.
    fn lambda_hints(&mut self, callee: &Ty, pos: &[Ty], kws: &[(String, Ty)]) -> (Vec<Option<Ty>>, HashMap<String, Ty>) {
        let (f, recv) = match callee {
            Ty::Func(FuncRef::Def(f)) => (*f, None),
            Ty::Func(FuncRef::Bound(f, r)) => (*f, Some((**r).clone())),
            Ty::Type(t) => match self.lookup_ty(t) {
                Ty::Inst(k, _) if self.db.is_user_class(k) || self.db.is_builtin_class(k) => match self.db.find_member(k, "__init__") {
                    Some((_, Member::Method(f))) => {
                        let ps = self.db.class_params(k);
                        let recv = Ty::Inst(k, ps.iter().map(|_| Ty::Var(self.db.new_tv("_", Variance::Inv))).collect());
                        (f, Some(recv))
                    }
                    _ => return (Vec::new(), HashMap::new()),
                },
                _ => return (Vec::new(), HashMap::new()),
            },
            _ => return (Vec::new(), HashMap::new()),
        };
        for sig in self.db.sigs(f).iter() {
            if let Some(h) = self.solve_args(sig, recv.as_ref(), pos, kws) {
                return h;
            }
        }
        (Vec::new(), HashMap::new())
    }

    fn solve_args(&mut self, sig: &Sig, recv: Option<&Ty>, pos: &[Ty], kws: &[(String, Ty)]) -> Option<(Vec<Option<Ty>>, HashMap<String, Ty>)> {
        let fresh = {
            let mut tys: Vec<&Ty> = sig.params.iter().map(|p| &p.ty).collect();
            tys.push(&sig.ret);
            self.db.fresh_rigid(&tys)
        };
        let params: Vec<(Param, Ty)> = sig.params.iter().map(|p| (p.clone(), subst(&p.ty, &fresh))).collect();
        let mut it = params.iter();
        let mut s = Subst::new();
        if let Some(r) = recv {
            let (_, t) = it.next()?;
            if !self.db.assign(r, t, &mut s, &mut false) {
                return None;
            }
        }
        let rest: Vec<&(Param, Ty)> = it.collect();
        let positional: Vec<&Ty> = rest.iter().filter(|(p, _)| matches!(p.kind, PKind::PosOnly | PKind::PosOrKw)).map(|(_, t)| t).collect();
        let var_pos = rest.iter().find(|(p, _)| matches!(p.kind, PKind::VarPos)).map(|(_, t)| t);
        let var_kw = rest.iter().find(|(p, _)| matches!(p.kind, PKind::VarKw)).map(|(_, t)| t);
        let mut pos_params = Vec::new();
        for i in 0..pos.len() {
            pos_params.push(positional.get(i).copied().or(var_pos)?.clone());
        }
        let mut kw_params = Vec::new();
        for (n, _) in kws {
            let t = rest.iter().find(|(p, _)| p.name == *n && matches!(p.kind, PKind::PosOrKw | PKind::KwOnly)).map(|(_, t)| t).or(var_kw)?;
            kw_params.push(t.clone());
        }
        for (a, p) in pos.iter().zip(&pos_params).chain(kws.iter().map(|(_, t)| t).zip(&kw_params)) {
            if *a != Ty::Any && !self.db.assign(a, p, &mut s, &mut false) {
                return None;
            }
        }
        let hints = pos_params.iter().map(|p| Some(deep_subst(p, &s))).collect();
        let kw_hints = kws.iter().zip(&kw_params).map(|((n, _), p)| (n.clone(), deep_subst(p, &s))).collect();
        Some((hints, kw_hints))
    }

    /// A call with `*iterable` or `**mapping` arguments. Fixed-length tuples
    /// spread into typed positional arguments; otherwise the arguments are
    /// only checked where the callee is entered.
    fn star_call(&mut self, c: &ast::ExprCall, fv: Val, off: u32) -> Val {
        let list = self.reg(Repr::Obj);
        self.emit(Instr::BuildList { dst: list, items: Vec::new() });
        let mut pos_tys = Vec::new();
        let mut exact = true;
        for a in c.arguments.args.iter() {
            match a {
                Expr::Starred(st) => {
                    let v = self.expr(&st.value);
                    match &v.ty {
                        Ty::Tuple(ts) => pos_tys.extend(ts.iter().cloned()),
                        t => {
                            let t = t.clone();
                            self.iter_elem(&t, st.range().start().into());
                            exact = false;
                        }
                    }
                    let r = self.obj(&v);
                    self.emit(Instr::ListExtend { list, iter: r });
                }
                a => {
                    let v = self.expr(a);
                    pos_tys.push(v.ty.clone());
                    let r = self.obj(&v);
                    self.emit(Instr::ListAppend { list, val: r });
                }
            }
        }
        let args = self.reg(Repr::Obj);
        self.emit(Instr::ListToTuple { dst: args, src: list });
        let kwargs = (!c.arguments.keywords.is_empty()).then(|| {
            let d = self.reg(Repr::Obj);
            self.emit(Instr::BuildDict { dst: d, keys: Vec::new(), vals: Vec::new() });
            d
        });
        let mut kw_tys = Vec::new();
        let func = self.obj(&fv);
        for k in c.arguments.keywords.iter() {
            let v = self.expr(&k.value);
            let d = kwargs.unwrap();
            match &k.arg {
                Some(name) => {
                    kw_tys.push((name.to_string(), v.ty.clone()));
                    let key = self.konst(Const::Str(name.to_string()), Ty::Str);
                    let one = self.reg(Repr::Obj);
                    let vr = self.obj(&v);
                    self.emit(Instr::BuildDict { dst: one, keys: vec![key.reg], vals: vec![vr] });
                    self.emit(Instr::DictMerge { dict: d, src: one, func });
                }
                None => {
                    exact = false;
                    if !matches!(v.ty, Ty::Any) && self.builtin_args(&v.ty, "dict").is_none_or(|a| !self.assignable(&a[0], &Ty::Str)) {
                        let t = self.show(&v.ty);
                        self.err(k.range().start().into(), format!("`**` needs a dict with str keys, got {t}"));
                    }
                    let vr = self.obj(&v);
                    self.emit(Instr::DictMerge { dict: d, src: vr, func });
                }
            }
        }
        let mut origin = None;
        let mut builtin_ctor = false;
        let ret = if exact {
            self.call_type(fv.ty.clone(), &pos_tys, &kw_tys, off, &mut origin, &mut builtin_ctor)
        } else {
            match &fv.ty {
                Ty::Func(FuncRef::Def(f) | FuncRef::Bound(f, _)) if self.db.sigs(*f).len() == 1 => {
                    let r = self.db.sigs(*f)[0].ret.clone();
                    self.db.erase(&r)
                }
                Ty::Type(t) => match self.lookup_ty(t) {
                    Ty::Inst(k, args) if self.db.is_user_class(k) => Ty::Inst(k, args.iter().map(|a| self.db.erase(a)).collect()),
                    _ => Ty::Any,
                },
                _ => Ty::Any,
            }
        };
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::CallEx { dst, func, args, kwargs });
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
            // An instance with `__call__`.
            t @ (Ty::Inst(..) | Ty::Var(_)) if matches!(self.lookup_ty(&t), Ty::Inst(c, _) if self.db.find_member(c, "__call__").is_some()) => {
                let Ty::Inst(c, _) = self.lookup_ty(&t) else { unreachable!() };
                match self.db.find_member(c, "__call__") {
                    Some((_, Member::Method(f))) => self.call_result(f, Some(t), pos, kws, off, origin),
                    _ => Ty::Any,
                }
            }
            Ty::Callable(ps, r) => {
                if let Some(ps) = ps {
                    if !kws.is_empty() {
                        self.err(off, "a `Callable` cannot be called with keyword arguments");
                    } else if ps.len() != pos.len() {
                        self.err(off, format!("expected {} arguments, got {}", ps.len(), pos.len()));
                    } else {
                        for (i, (p, a)) in ps.iter().zip(pos).enumerate() {
                            if !self.assignable(a, p) {
                                let a = if has_lit(p) { a.clone() } else { decay(a) };
                                let (a, p) = (self.show(&a), self.show(p));
                                self.err(off, format!("argument {}: expected {p}, got {a}", i + 1));
                            }
                        }
                    }
                }
                *r
            }
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
                let untyped = self.db.untyped_bases(c);
                if !untyped.is_empty() {
                    let hint = self.untyped_hint(c);
                    self.err(off, format!("cannot check the arguments of {cname}(){hint}"));
                } else if self.db.exc_base(c).is_none() && (!pos.is_empty() || !kws.is_empty()) {
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
        // Too many positional arguments is reported first, as CPython does.
        let skip = usize::from(recv.is_some());
        let npos = sig.params.iter().skip(skip).filter(|p| matches!(p.kind, PKind::PosOnly | PKind::PosOrKw)).count();
        if pos.len() > npos && !sig.params.iter().any(|p| matches!(p.kind, PKind::VarPos)) {
            return Err(Mismatch::Arity(format!("takes {npos} positional arguments but {} were given", pos.len())));
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
            // A literal argument (`f(1)`) is its literal type only for a
            // `Literal` parameter: elsewhere it solves type variables to its
            // base type (`max("a", "b")` is a str).
            let got = if matches!(got, Ty::Lit(_)) && !has_lit(&want) { decay(&got) } else { got };
            if !self.db.assign(&got, &want, &mut s, &mut any) {
                // Unsolved bounded variables read as their bounds.
                let mut vars = Vec::new();
                collect_vars(&want, &mut vars);
                let mut shown = s.clone();
                for v in vars {
                    if let (false, Some(b)) = (shown.contains_key(&v), self.db.tvs[v].bound.clone()) {
                        shown.insert(v, b);
                    }
                }
                let expected = self.db.erase(&deep_subst(&want, &shown));
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

    /// `l op r`; `r_lit` is the right operand if it is an int literal.
    fn binop_vals(&mut self, op: ast::Operator, l: Val, r: Val, r_lit: Option<i64>, what: String, off: u32, inplace: bool) -> Val {
        let (bop, dunder, rdunder) = match op {
            ast::Operator::Add => (BinOp::Add, "__add__", "__radd__"),
            ast::Operator::Sub => (BinOp::Sub, "__sub__", "__rsub__"),
            ast::Operator::Mult => (BinOp::Mul, "__mul__", "__rmul__"),
            ast::Operator::Div => (BinOp::Div, "__truediv__", "__rtruediv__"),
            ast::Operator::FloorDiv => (BinOp::FloorDiv, "__floordiv__", "__rfloordiv__"),
            ast::Operator::Mod => (BinOp::Mod, "__mod__", "__rmod__"),
            ast::Operator::Pow => (BinOp::Pow, "__pow__", "__rpow__"),
            ast::Operator::MatMult => (BinOp::MatMul, "__matmul__", "__rmatmul__"),
            ast::Operator::BitOr => (BinOp::BitOr, "__or__", "__ror__"),
            ast::Operator::BitAnd => (BinOp::BitAnd, "__and__", "__rand__"),
            ast::Operator::BitXor => (BinOp::BitXor, "__xor__", "__rxor__"),
            ast::Operator::LShift => (BinOp::LShift, "__lshift__", "__rlshift__"),
            ast::Operator::RShift => (BinOp::RShift, "__rshift__", "__rrshift__"),
        };
        // Arithmetic on literals is arithmetic on their base types.
        let (l, r) = (Val { ty: decay(&l.ty), ..l }, Val { ty: decay(&r.ty), ..r });
        let (lt, rt) = (l.ty.clone(), r.ty.clone());
        let pow = match (bop, &lt, &rt) {
            // `int ** int` is an int only for a non-negative exponent, and a
            // negative float base can give a complex number, as in typeshed.
            (BinOp::Pow, Ty::Int | Ty::Bool, Ty::Int | Ty::Bool) => Some(if r_lit.is_some_and(|n| n >= 0) { Ty::Int } else { Ty::Any }),
            (BinOp::Pow, Ty::Float, Ty::Int | Ty::Bool) => Some(Ty::Float),
            (BinOp::Pow, Ty::Int | Ty::Bool | Ty::Float, Ty::Float) => Some(Ty::Any),
            _ => None,
        };
        // `a += b` uses `a.__iadd__(b)` when it is defined.
        let in_place = if inplace && matches!(self.lookup_ty(&lt), Ty::Inst(..)) {
            match self.dunder(&lt, &format!("__i{}", &dunder[2..]), std::slice::from_ref(&rt)) {
                Some(Ok((t, file, doff))) => Some((t, self.origin(file, doff))),
                Some(Err(m)) => {
                    self.err(off, m);
                    Some((Ty::Any, None))
                }
                None => None,
            }
        } else {
            None
        };
        let (ty, origin) = match in_place.or_else(|| pow.map(|t| (t, None))).or_else(|| self.binop_type(bop, dunder, rdunder, &lt, &rt)) {
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
            (Ty::Bool, Ty::Bool) if matches!(bop, BinOp::BitOr | BinOp::BitAnd | BinOp::BitXor) => (Ty::Bool, None),
            (Ty::Int | Ty::Bool, Ty::Int | Ty::Bool) if bop != BinOp::Div => (Ty::Int, None),
            (Ty::Int | Ty::Bool | Ty::Float, Ty::Int | Ty::Bool | Ty::Float) if matches!(bop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::FloorDiv | BinOp::Mod) => {
                (Ty::Float, None)
            }
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

    /// A comparison; a chain `a < b <= c` is `a < b and b <= c` with `b`
    /// evaluated once, its value the first false result or the last one.
    fn compare(&mut self, c: &ast::ExprCompare, e: &Expr) -> Val {
        let mut l = self.expr(&c.operands[0]);
        if c.ops.len() == 1 {
            let r = self.expr(&c.operands[1]);
            let what = self.src(e);
            return self.compare_vals(c.ops[0], l, r, c.range().start().into(), what);
        }
        let dst = self.reg(Repr::Obj);
        let mut tys = Vec::new();
        let mut exits = Vec::new();
        for (i, op) in c.ops.iter().enumerate() {
            let rhs = &c.operands[i + 1];
            let r = self.expr(rhs);
            let off: u32 = c.operands[i].range().start().into();
            let what = format!("{} {} {}", self.src(&c.operands[i]), op.as_str(), self.src(rhs));
            let v = self.compare_vals(*op, l, r.clone(), off, what);
            tys.push(v.ty.clone());
            let o = self.obj(&v);
            self.emit(Instr::Move { dst, src: o });
            if i + 1 < c.ops.len() {
                let t = self.reg(Repr::Bool);
                self.emit(Instr::Truthy { dst: t, src: dst });
                exits.push(self.emit(Instr::JumpIfFalse { cond: t, target: 0 }));
            }
            l = r;
        }
        let h = self.here();
        for j in exits {
            self.patch(j, h);
        }
        Val { reg: dst, ty: union(tys), repr: Repr::Obj }
    }

    /// `l op r` for evaluated operands.
    fn compare_vals(&mut self, cop: ast::CmpOp, l: Val, r: Val, off: u32, what: String) -> Val {
        let (l, r) = (Val { ty: decay(&l.ty), ..l }, Val { ty: decay(&r.ty), ..r });
        let (op, dunder, reflected) = match cop {
            ast::CmpOp::Eq => (CmpOp::Eq, "__eq__", "__eq__"),
            ast::CmpOp::NotEq => (CmpOp::Ne, "__ne__", "__ne__"),
            ast::CmpOp::Lt => (CmpOp::Lt, "__lt__", "__gt__"),
            ast::CmpOp::LtE => (CmpOp::Le, "__le__", "__ge__"),
            ast::CmpOp::Gt => (CmpOp::Gt, "__gt__", "__lt__"),
            ast::CmpOp::GtE => (CmpOp::Ge, "__ge__", "__le__"),
            ast::CmpOp::Is => (CmpOp::Is, "", ""),
            ast::CmpOp::IsNot => (CmpOp::IsNot, "", ""),
            ast::CmpOp::In => (CmpOp::In, "__contains__", ""),
            ast::CmpOp::NotIn => (CmpOp::NotIn, "__contains__", ""),
        };
        let (lt, rt) = (l.ty.clone(), r.ty.clone());
        let ty = match (&lt, &rt) {
            _ if dunder.is_empty() => Ty::Bool,
            // `x in container`: `container.__contains__(x)`.
            _ if matches!(op, CmpOp::In | CmpOp::NotIn) => {
                if !matches!(rt, Ty::Any) {
                    match self.dunder(&rt, "__contains__", std::slice::from_ref(&lt)) {
                        Some(Ok(_)) => {}
                        Some(Err(m)) => self.err(off, m),
                        None => {
                            let ts = self.show(&rt);
                            self.err(off, format!("{ts} does not support `in`"));
                        }
                    }
                }
                Ty::Bool
            }
            (Ty::Any, _) | (_, Ty::Any) => Ty::Any,
            _ => self.compare_type(op, dunder, reflected, &lt, &rt, off, cop),
        };
        let (a, b) = (self.obj(&l), self.obj(&r));
        let dst = self.reg(Repr::Obj);
        self.emit(Instr::Compare { dst, op, a, b });
        let v = Val { reg: dst, ty, repr: Repr::Obj };
        self.check(&v, what, None);
        v
    }

    /// The type of a rich comparison: the left operand's method, else the
    /// right operand's reflected one (`a > b` is `b.__lt__(a)`), as Python
    /// tries them. Ordering a user class that defines neither is an error;
    /// `==` and `!=` fall back to identity.
    fn compare_type(&mut self, op: CmpOp, dunder: &str, reflected: &str, lt: &Ty, rt: &Ty, off: u32, cop: ast::CmpOp) -> Ty {
        let user = |l: &mut Self, t: &Ty| match l.lookup_ty(t) {
            Ty::Inst(c, _) => l.db.is_user_class(c),
            _ => false,
        };
        // Ordering builtin values of unrelated kinds raises TypeError.
        let kind = |t: &Ty| match t {
            Ty::Int | Ty::Float | Ty::Bool => Some(0),
            Ty::Str => Some(1),
            Ty::None => Some(2),
            _ => None,
        };
        let mut first_err = None;
        // Builtin values compare by the rules below, not their methods.
        if kind(lt).is_none() || kind(rt).is_none() {
            match self.dunder(lt, dunder, std::slice::from_ref(rt)) {
                Some(Ok((t, _, _))) => return t,
                Some(Err(m)) => first_err = Some(m),
                None => {}
            }
            match self.dunder(rt, reflected, std::slice::from_ref(lt)) {
                Some(Ok((t, _, _))) => return t,
                Some(Err(m)) => first_err = first_err.or(Some(m)),
                None => {}
            }
        }
        let mismatched = matches!((kind(lt), kind(rt)), (Some(a), Some(b)) if a != b || a == 2);
        if matches!(op, CmpOp::Eq | CmpOp::Ne) || !(user(self, lt) || user(self, rt) || mismatched) {
            return Ty::Bool;
        }
        match first_err {
            Some(m) => self.err(off, m),
            None => {
                let (a, b) = (self.show(lt), self.show(rt));
                self.err(off, format!("`{}` is not supported between {a} and {b}", cop.as_str()));
            }
        }
        Ty::Bool
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
        let want = self.container_hint(hint, "list").and_then(|a| a.into_iter().next());
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
        let want = self.container_hint(hint, "dict");
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
            CompBody::Set(_) => self.emit(Instr::BuildSet { dst, items: Vec::new() }),
            CompBody::Dict(..) => self.emit(Instr::BuildDict { dst, keys: Vec::new(), vals: Vec::new() }),
        };
        let mut elem = Vec::new();
        self.in_comprehension += 1;
        self.comp_level(gens, &body, hint, dst, &mut elem, off);
        self.in_comprehension -= 1;
        for (n, l) in saved {
            match l {
                Some(l) => self.locals.insert(n, l),
                None => self.locals.remove(&n),
            };
        }
        let ty = match (body, elem.as_slice()) {
            (CompBody::List(_), [t]) => self.builtin_inst("list", vec![t.clone()]),
            (CompBody::Set(_), [t]) => self.builtin_inst("set", vec![t.clone()]),
            (CompBody::Dict(..), [k, v]) => self.builtin_inst("dict", vec![k.clone(), v.clone()]),
            _ => Ty::Any,
        };
        Val { reg: dst, ty, repr: Repr::Obj }
    }

    fn comp_level(&mut self, gens: &[ast::Comprehension], body: &CompBody, hint: Option<&Ty>, dst: Reg, elem: &mut Vec<Ty>, off: u32) {
        let Some((g, rest)) = gens.split_first() else {
            match body {
                CompBody::List(e) => {
                    let want = self.container_hint(hint, "list").and_then(|a| a.into_iter().next());
                    let (regs, t) = self.items(&[e], want.as_ref(), "a list item");
                    self.emit(Instr::ListAppend { list: dst, val: regs[0] });
                    elem.extend(t);
                }
                CompBody::Set(e) => {
                    let want = self.container_hint(hint, "set").and_then(|a| a.into_iter().next());
                    let (regs, t) = self.items(&[e], want.as_ref(), "a set item");
                    self.emit(Instr::SetAdd { set: dst, val: regs[0] });
                    elem.extend(t);
                }
                CompBody::Dict(k, v) => {
                    let want = self.container_hint(hint, "dict").filter(|a| a.len() == 2);
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

/// The names a function's body uses and binds, for closures: which of its
/// variables inner functions capture (cells), and which names it needs from
/// enclosing functions (free).
#[derive(Default)]
struct NameScope {
    params: HashSet<String>,
    loads: HashSet<String>,
    bound: HashSet<String>,
    nested_free: HashSet<String>,
    globals: HashSet<String>,
    nonlocals: HashSet<String>,
}

impl NameScope {
    fn of(params: Option<&ast::Parameters>, body: FnBody) -> NameScope {
        let mut s = NameScope::default();
        if let Some(ps) = params {
            for p in ps.iter() {
                s.params.insert(p.name().to_string());
            }
        }
        match body {
            FnBody::Stmts(b) => b.iter().for_each(|st| s.visit_stmt(st)),
            FnBody::Lambda(e) => s.visit_expr(e),
        }
        s
    }

    /// Names local to the function (assigned in it), besides parameters.
    fn local_names(&self) -> HashSet<String> {
        self.bound.iter().filter(|n| !self.nonlocals.contains(*n) && !self.globals.contains(*n)).cloned().collect()
    }

    fn cells(&self) -> HashSet<String> {
        let mine: HashSet<String> = self.local_names().union(&self.params).cloned().collect();
        mine.intersection(&self.nested_free).cloned().collect()
    }

    fn free(&self) -> HashSet<String> {
        let mine: HashSet<String> = self.local_names().union(&self.params).cloned().collect();
        self.loads.iter().chain(&self.nested_free).chain(&self.nonlocals).filter(|n| !mine.contains(*n) && !self.globals.contains(*n)).cloned().collect()
    }

    fn nested(&mut self, params: Option<&ast::Parameters>, body: FnBody) {
        let inner = NameScope::of(params, body);
        self.nested_free.extend(inner.free());
    }
}

impl<'a> Visitor<'a> for NameScope {
    fn visit_stmt(&mut self, st: &'a Stmt) {
        match st {
            Stmt::FunctionDef(d) => {
                self.bound.insert(d.name.to_string());
                for dec in &d.decorator_list {
                    self.visit_expr(&dec.expression);
                }
                for p in d.parameters.iter_non_variadic_params() {
                    if let Some(dflt) = &p.default {
                        self.visit_expr(dflt);
                    }
                }
                self.nested(Some(&d.parameters), FnBody::Stmts(&d.body));
            }
            Stmt::ClassDef(c) => {
                self.bound.insert(c.name.to_string());
            }
            Stmt::Global(g) => self.globals.extend(g.names.iter().map(|n| n.to_string())),
            Stmt::Nonlocal(n) => self.nonlocals.extend(n.names.iter().map(|n| n.to_string())),
            Stmt::Import(i) => {
                for a in &i.names {
                    let n = a.asname.as_ref().map_or_else(|| a.name.split('.').next().unwrap_or("").to_string(), |x| x.to_string());
                    self.bound.insert(n);
                }
            }
            Stmt::ImportFrom(i) => {
                for a in &i.names {
                    self.bound.insert(a.asname.as_ref().unwrap_or(&a.name).to_string());
                }
            }
            _ => walk_stmt(self, st),
        }
    }

    fn visit_expr(&mut self, e: &'a Expr) {
        match e {
            Expr::Name(n) => {
                if n.ctx == ast::ExprContext::Load {
                    self.loads.insert(n.id.to_string());
                } else {
                    self.bound.insert(n.id.to_string());
                }
            }
            Expr::Lambda(l) => {
                if let Some(ps) = l.parameters.as_deref() {
                    for p in ps.iter_non_variadic_params() {
                        if let Some(dflt) = &p.default {
                            self.visit_expr(dflt);
                        }
                    }
                }
                self.nested(l.parameters.as_deref(), FnBody::Lambda(&l.body));
            }
            _ => walk_expr(self, e),
        }
    }

    /// Annotations are not evaluated by compiled code.
    fn visit_annotation(&mut self, _: &'a Expr) {}

    fn visit_except_handler(&mut self, h: &'a ast::ExceptHandler) {
        let ast::ExceptHandler::ExceptHandler(eh) = h;
        if let Some(n) = &eh.name {
            self.bound.insert(n.to_string());
        }
        walk_except_handler(self, h);
    }
}

/// The attributes of builtin exception `e` (the checker's own exception
/// classes have no definition file).
fn exc_attr(e: &str, name: &str) -> Option<Ty> {
    let any = Some(Ty::Any);
    match name {
        "args" => Some(Ty::TupleVar(Box::new(Ty::Any))),
        "__traceback__" | "__cause__" | "__context__" | "__notes__" | "with_traceback" | "add_note" => any,
        "__suppress_context__" => Some(Ty::Bool),
        n if OBJECT_ATTRS.contains(&n) => any,
        "errno" | "strerror" | "filename" | "filename2" if exc_sub(e, "OSError") => any,
        "value" if exc_sub(e, "StopIteration") => any,
        "code" if exc_sub(e, "SystemExit") => any,
        "name" | "path" if exc_sub(e, "ImportError") => any,
        "name" if exc_sub(e, "NameError") || exc_sub(e, "AttributeError") => Some(Ty::Str),
        "obj" if exc_sub(e, "AttributeError") => any,
        "encoding" | "object" | "start" | "end" | "reason" if exc_sub(e, "UnicodeError") => any,
        "msg" | "lineno" | "offset" | "text" | "filename" if exc_sub(e, "SyntaxError") => any,
        _ => None,
    }
}

/// Whether `t` contains `Any` (type variables are not `Any` here).
fn mentions_any(t: &Ty) -> bool {
    match t {
        Ty::Any => true,
        Ty::Tuple(v) | Ty::Union(v) | Ty::Inst(_, v) => v.iter().any(mentions_any),
        Ty::TupleVar(x) | Ty::Type(x) => mentions_any(x),
        _ => false,
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
            Stmt::FunctionDef(d) => out.push(d.name.to_string()),
            _ => {}
        }
    }
}

/// Whether a function body contains `yield` (not counting nested
/// functions and lambdas): it is a generator function.
fn contains_yield(body: &[Stmt]) -> bool {
    struct Finder(bool);
    impl<'a> Visitor<'a> for Finder {
        fn visit_stmt(&mut self, s: &'a Stmt) {
            if !matches!(s, Stmt::FunctionDef(_) | Stmt::ClassDef(_)) {
                walk_stmt(self, s);
            }
        }
        fn visit_expr(&mut self, e: &'a Expr) {
            match e {
                Expr::Yield(_) | Expr::YieldFrom(_) => self.0 = true,
                // Their own scopes (`yield` in a comprehension is an error).
                Expr::Lambda(_) | Expr::ListComp(_) | Expr::SetComp(_) | Expr::DictComp(_) | Expr::Generator(_) => {}
                e => walk_expr(self, e),
            }
        }
    }
    let mut f = Finder(false);
    body.iter().for_each(|s| f.visit_stmt(s));
    f.0
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
