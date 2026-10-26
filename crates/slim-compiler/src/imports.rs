//! `import m` / `from m import x`: native modules.
//!
//! Imports are resolved before compiling: each name an import binds is
//! rewritten to the qualified global it refers to (`dot` and `geom.dot`
//! become `geom.dot`), which the compiler then treats like any other
//! module-level function or class. A function's own parameters and locals
//! shadow imported names.

use std::collections::{HashMap, HashSet};

use slim_syntax::ast::*;
use slim_syntax::{Error, Span};
use slim_vm::native::ModuleSpec;

use crate::scan;

type R<T> = Result<T, Error>;

fn name_err(msg: impl Into<String>, span: Span) -> Error {
    Error::new("NameError", msg, span)
}

struct Rewriter<'a> {
    /// Module alias -> module spec.
    modules: HashMap<String, &'a ModuleSpec>,
    /// Imported name -> qualified global.
    names: HashMap<String, String>,
    /// Names bound by the enclosing functions.
    shadow: Vec<HashSet<String>>,
}

fn has_item(m: &ModuleSpec, item: &str) -> bool {
    m.functions.iter().any(|f| f.name == item) || m.classes.iter().any(|c| c.name == item) || m.traits.iter().any(|t| t.name == item)
}

/// Rewrites the imports of `module`. Returns the native modules it uses.
pub fn resolve<'a>(module: &mut Module, natives: &'a [ModuleSpec]) -> R<Vec<&'a ModuleSpec>> {
    let mut rw = Rewriter { modules: HashMap::new(), names: HashMap::new(), shadow: Vec::new() };
    let mut used: Vec<&ModuleSpec> = Vec::new();
    for s in &module.body.stmts {
        let find = |m: &str| natives.iter().find(|n| n.name == m).ok_or_else(|| name_err(format!("no module named `{m}`"), s.span));
        match &s.kind {
            StmtKind::Import { module: m, alias } => {
                let spec = find(m)?;
                rw.modules.insert(alias.clone().unwrap_or_else(|| m.clone()), spec);
                if !used.iter().any(|u| u.name == spec.name) {
                    used.push(spec);
                }
            }
            StmtKind::FromImport { module: m, names } => {
                let spec = find(m)?;
                for (n, alias) in names {
                    if !has_item(spec, n) {
                        return Err(name_err(format!("module `{m}` has no `{n}`"), s.span));
                    }
                    rw.names.insert(alias.clone().unwrap_or_else(|| n.clone()), format!("{m}.{n}"));
                }
                if !used.iter().any(|u| u.name == spec.name) {
                    used.push(spec);
                }
            }
            _ => {}
        }
    }
    if used.is_empty() {
        // Still reject imports in functions.
        rw.block(&mut module.body, true)?;
        return Ok(used);
    }
    let mut assigned = Vec::new();
    scan::assigned_names(&module.body, &mut assigned);
    for n in &assigned {
        if rw.names.contains_key(n) || rw.modules.contains_key(n) {
            return Err(Error::syntax(format!("`{n}` is imported and cannot be redefined"), Span::default()));
        }
    }
    rw.block(&mut module.body, true)?;
    Ok(used)
}

impl Rewriter<'_> {
    fn shadowed(&self, n: &str) -> bool {
        self.shadow.iter().any(|s| s.contains(n))
    }

    fn block(&mut self, b: &mut Block, top: bool) -> R<()> {
        for s in &mut b.stmts {
            self.stmt(s, top)?;
        }
        Ok(())
    }

    fn func(&mut self, f: &mut FuncDef) -> R<()> {
        for tp in &mut f.tparams {
            tp.bounds.iter_mut().for_each(|b| self.ty(b));
        }
        let mut bound: HashSet<String> = f.params.iter().map(|p| p.name.clone()).collect();
        if let Some(b) = &f.body {
            let mut names = Vec::new();
            scan::assigned_names(b, &mut names);
            bound.extend(names);
        }
        for p in &mut f.params {
            if let Some(t) = &mut p.ty {
                self.ty(t);
            }
        }
        if let Some(t) = &mut f.ret {
            self.ty(t);
        }
        self.shadow.push(bound);
        let r = match &mut f.body {
            Some(b) => self.block(b, false),
            None => Ok(()),
        };
        self.shadow.pop();
        r
    }

    fn stmt(&mut self, s: &mut Stmt, top: bool) -> R<()> {
        match &mut s.kind {
            StmtKind::Import { .. } | StmtKind::FromImport { .. } => {
                if !top {
                    return Err(Error::syntax("imports must be at module level", s.span));
                }
                Ok(())
            }
            StmtKind::Expr(e) => self.expr(e),
            StmtKind::Assign { target, value } | StmtKind::AugAssign { target, value, .. } => {
                self.expr(target)?;
                self.expr(value)
            }
            StmtKind::AnnAssign { target, ann, value } => {
                self.expr(target)?;
                self.ty(ann);
                match value {
                    Some(v) => self.expr(v),
                    None => Ok(()),
                }
            }
            StmtKind::Move { targets, value } => {
                for t in targets {
                    self.expr(t)?;
                }
                self.expr(value)
            }
            StmtKind::Def(f) => self.func(f),
            StmtKind::Class(c) => {
                for tp in &mut c.tparams {
                    tp.bounds.iter_mut().for_each(|b| self.ty(b));
                }
                for (n, args, _) in &mut c.traits {
                    if let Some(q) = self.names.get(n.as_str()) {
                        *n = q.clone();
                    }
                    args.iter_mut().for_each(|a| self.ty(a));
                }
                for f in &mut c.fields {
                    self.ty(&mut f.ty);
                }
                for m in &mut c.methods {
                    self.func(m)?;
                }
                Ok(())
            }
            StmtKind::Trait(t) => {
                for m in &mut t.methods {
                    self.func(m)?;
                }
                Ok(())
            }
            StmtKind::ImplFor { tr, targs, target, .. } => {
                if let Some(q) = self.names.get(tr.as_str()) {
                    *tr = q.clone();
                }
                targs.iter_mut().for_each(|a| self.ty(a));
                if let Some(q) = self.names.get(target.as_str()) {
                    *target = q.clone();
                } else if let Some((m, item)) = target.split_once('.') {
                    if let Some(spec) = self.modules.get(m) {
                        if !has_item(spec, item) {
                            return Err(Error::new("AttributeError", format!("module `{}` has no `{item}`", spec.name), s.span));
                        }
                        *target = format!("{}.{item}", spec.name);
                    }
                }
                Ok(())
            }
            StmtKind::Return(Some(e)) => self.expr(e),
            StmtKind::While { cond, body } => {
                self.expr(cond)?;
                self.block(body, false)
            }
            StmtKind::For { target, iter, body } => {
                self.expr(target)?;
                self.expr(iter)?;
                self.block(body, false)
            }
            StmtKind::Raise { exc, cause } => {
                if let Some(e) = exc {
                    self.expr(e)?;
                }
                match cause {
                    Some(c) => self.expr(c),
                    None => Ok(()),
                }
            }
            StmtKind::Del(ts) => ts.iter_mut().try_for_each(|t| self.expr(t)),
            StmtKind::With { ctx, body, .. } => {
                self.expr(ctx)?;
                self.block(body, false)
            }
            StmtKind::Assert { cond, msg } => {
                self.expr(cond)?;
                match msg {
                    Some(m) => self.expr(m),
                    None => Ok(()),
                }
            }
            StmtKind::Global(names) => {
                for n in names {
                    if self.names.contains_key(n) {
                        return Err(Error::syntax(format!("`{n}` is imported and cannot be declared `global`"), s.span));
                    }
                }
                Ok(())
            }
            StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue | StmtKind::Pass => Ok(()),
        }
    }

    fn ty(&mut self, t: &mut TypeExpr) {
        match t {
            TypeExpr::Unit(_) => {}
            TypeExpr::Send(inner) => self.ty(inner),
            TypeExpr::Named(n, args, _) => {
                if let Some(q) = self.names.get(n.as_str()) {
                    *n = q.clone();
                }
                args.iter_mut().for_each(|a| self.ty(a));
            }
        }
    }

    fn expr(&mut self, e: &mut Expr) -> R<()> {
        // `geom.dot`: an item of an imported module.
        if let ExprKind::Attr(base, item) = &e.kind {
            if let ExprKind::Name(m) = &base.kind {
                if !self.shadowed(m) {
                    if let Some(spec) = self.modules.get(m.as_str()) {
                        if !has_item(spec, item) {
                            return Err(Error::new("AttributeError", format!("module `{}` has no `{item}`", spec.name), e.span));
                        }
                        e.kind = ExprKind::Name(format!("{}.{item}", spec.name));
                        return Ok(());
                    }
                }
            }
        }
        match &mut e.kind {
            ExprKind::Name(n) => {
                if self.shadowed(n) {
                    return Ok(());
                }
                if let Some(q) = self.names.get(n.as_str()) {
                    *n = q.clone();
                } else if self.modules.contains_key(n.as_str()) {
                    return Err(Error::new("TypeError", format!("module `{n}` is not a value; use its items (`{n}.name`)"), e.span));
                }
                Ok(())
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_) | ExprKind::NoneLit | ExprKind::Unit | ExprKind::Ellipsis => {
                Ok(())
            }
            ExprKind::Tuple(xs) | ExprKind::List(xs) => xs.iter_mut().try_for_each(|x| self.expr(x)),
            ExprKind::Dict(kvs) => kvs.iter_mut().try_for_each(|(k, v)| {
                self.expr(k)?;
                self.expr(v)
            }),
            ExprKind::Attr(o, _) => self.expr(o),
            ExprKind::Index(o, i) => {
                self.expr(o)?;
                self.expr(i)
            }
            ExprKind::Call(f, args) => {
                self.expr(f)?;
                args.iter_mut().try_for_each(|a| self.expr(a))
            }
            ExprKind::TypeApp(f, ts) => {
                ts.iter_mut().for_each(|t| self.ty(t));
                self.expr(f)
            }
            ExprKind::Dup(x) | ExprKind::Unary(_, x) | ExprKind::YieldFrom(x) => self.expr(x),
            ExprKind::Binary(_, a, b) | ExprKind::And(a, b) | ExprKind::Or(a, b) => {
                self.expr(a)?;
                self.expr(b)
            }
            ExprKind::Compare(a, rest) => {
                self.expr(a)?;
                rest.iter_mut().try_for_each(|(_, x)| self.expr(x))
            }
            ExprKind::Ternary { cond, then, else_ } => {
                self.expr(cond)?;
                self.expr(then)?;
                self.expr(else_)
            }
            ExprKind::If { branches, else_ } => {
                for (c, b) in branches {
                    self.expr(c)?;
                    self.block(b, false)?;
                }
                match else_ {
                    Some(b) => self.block(b, false),
                    None => Ok(()),
                }
            }
            ExprKind::Try(t) => {
                self.block(&mut t.body, false)?;
                for h in &mut t.handlers {
                    h.types.iter_mut().for_each(|t| self.ty(t));
                    self.block(&mut h.body, false)?;
                }
                if let Some(b) = &mut t.else_ {
                    self.block(b, false)?;
                }
                match &mut t.finally {
                    Some(b) => self.block(b, false),
                    None => Ok(()),
                }
            }
            ExprKind::Lambda(f) => self.func(f),
            ExprKind::Yield(v) => match v {
                Some(v) => self.expr(v),
                None => Ok(()),
            },
        }
    }
}
