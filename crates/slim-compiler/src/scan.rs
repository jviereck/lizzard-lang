//! Syntactic pre-passes: assigned names, name uses and yields.

use std::collections::HashSet;

use slim_syntax::ast::*;

/// Names bound by assignments, `for` targets, `except ... as`, nested `def`s
/// and `del` in a function body (not looking into nested functions).
pub fn assigned_names(b: &Block, out: &mut Vec<String>) {
    for s in &b.stmts {
        stmt_assigned(s, out);
    }
}

fn push(out: &mut Vec<String>, n: &str) {
    if !out.iter().any(|x| x == n) {
        out.push(n.to_string());
    }
}

fn target_names(e: &Expr, out: &mut Vec<String>) {
    match &e.kind {
        ExprKind::Name(n) => push(out, n),
        ExprKind::Tuple(items) => items.iter().for_each(|i| target_names(i, out)),
        _ => {}
    }
}

fn stmt_assigned(s: &Stmt, out: &mut Vec<String>) {
    match &s.kind {
        StmtKind::Assign { target, value } => {
            target_names(target, out);
            expr_assigned(value, out);
        }
        StmtKind::AugAssign { target, value, .. } => {
            target_names(target, out);
            expr_assigned(value, out);
        }
        StmtKind::AnnAssign { target, value, .. } => {
            target_names(target, out);
            if let Some(v) = value {
                expr_assigned(v, out);
            }
        }
        StmtKind::Move { targets, value } => {
            targets.iter().for_each(|t| target_names(t, out));
            expr_assigned(value, out);
        }
        StmtKind::Def(f) => push(out, &f.name),
        StmtKind::Class(c) => push(out, &c.name),
        StmtKind::Trait(t) => push(out, &t.name),
        StmtKind::While { cond, body } => {
            expr_assigned(cond, out);
            assigned_names(body, out);
        }
        StmtKind::For { target, iter, body } => {
            target_names(target, out);
            expr_assigned(iter, out);
            assigned_names(body, out);
        }
        StmtKind::Expr(e) => expr_assigned(e, out),
        StmtKind::Return(Some(e)) => expr_assigned(e, out),
        StmtKind::With { name, body, .. } => {
            if let Some(n) = name {
                push(out, n);
            }
            assigned_names(body, out);
        }
        _ => {}
    }
}

/// Block-form `if`/`try` expressions contain statements.
fn expr_assigned(e: &Expr, out: &mut Vec<String>) {
    match &e.kind {
        ExprKind::If { branches, else_ } => {
            for (_, b) in branches {
                assigned_names(b, out);
            }
            if let Some(b) = else_ {
                assigned_names(b, out);
            }
        }
        ExprKind::Try(t) => {
            assigned_names(&t.body, out);
            for h in &t.handlers {
                if let Some(n) = &h.name {
                    push(out, n);
                }
                assigned_names(&h.body, out);
            }
            if let Some(b) = &t.else_ {
                assigned_names(b, out);
            }
            if let Some(b) = &t.finally {
                assigned_names(b, out);
            }
        }
        _ => {}
    }
}

/// `global` declarations in a function body.
pub fn global_decls(b: &Block, out: &mut HashSet<String>) {
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Global(names) => out.extend(names.iter().cloned()),
            StmtKind::While { body, .. } | StmtKind::For { body, .. } => global_decls(body, out),
            StmtKind::Expr(e) | StmtKind::Assign { value: e, .. } | StmtKind::Return(Some(e)) => {
                if let ExprKind::If { branches, else_ } = &e.kind {
                    for (_, b) in branches {
                        global_decls(b, out);
                    }
                    if let Some(b) = else_ {
                        global_decls(b, out);
                    }
                }
                if let ExprKind::Try(t) = &e.kind {
                    global_decls(&t.body, out);
                    for h in &t.handlers {
                        global_decls(&h.body, out);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Whether a function body yields (not counting nested functions).
pub fn has_yield(b: &Block) -> bool {
    let mut found = false;
    visit_block(b, &mut |e| {
        if matches!(e.kind, ExprKind::Yield(_) | ExprKind::YieldFrom(_)) {
            found = true;
        }
    });
    found
}

/// All names read in a function body, including inside nested lambdas.
pub fn used_names(b: &Block, out: &mut HashSet<String>) {
    visit_block_deep(b, &mut |e| {
        if let ExprKind::Name(n) = &e.kind {
            out.insert(n.clone());
        }
    });
}

fn visit_block(b: &Block, f: &mut dyn FnMut(&Expr)) {
    for s in &b.stmts {
        visit_stmt(s, f, false);
    }
}

fn visit_block_deep(b: &Block, f: &mut dyn FnMut(&Expr)) {
    for s in &b.stmts {
        visit_stmt(s, f, true);
    }
}

fn visit_stmt(s: &Stmt, f: &mut dyn FnMut(&Expr), deep: bool) {
    let ve = |e: &Expr, f: &mut dyn FnMut(&Expr)| visit_expr(e, f, deep);
    match &s.kind {
        StmtKind::Expr(e) => ve(e, f),
        StmtKind::Assign { target, value } => {
            ve(target, f);
            ve(value, f);
        }
        StmtKind::AugAssign { target, value, .. } => {
            ve(target, f);
            ve(value, f);
        }
        StmtKind::AnnAssign { target, value, .. } => {
            ve(target, f);
            if let Some(v) = value {
                ve(v, f);
            }
        }
        StmtKind::Move { targets, value } => {
            targets.iter().for_each(|t| ve(t, f));
            ve(value, f);
        }
        StmtKind::Return(Some(e)) => ve(e, f),
        StmtKind::While { cond, body } => {
            ve(cond, f);
            for s in &body.stmts {
                visit_stmt(s, f, deep);
            }
        }
        StmtKind::For { target, iter, body } => {
            ve(target, f);
            ve(iter, f);
            for s in &body.stmts {
                visit_stmt(s, f, deep);
            }
        }
        StmtKind::Raise { exc, cause } => {
            if let Some(e) = exc {
                ve(e, f);
            }
            if let Some(e) = cause {
                ve(e, f);
            }
        }
        StmtKind::Del(ts) => ts.iter().for_each(|t| ve(t, f)),
        StmtKind::With { ctx, body, .. } => {
            ve(ctx, f);
            for s in &body.stmts {
                visit_stmt(s, f, deep);
            }
        }
        StmtKind::Assert { cond, msg } => {
            ve(cond, f);
            if let Some(m) = msg {
                ve(m, f);
            }
        }
        StmtKind::Def(d) if deep => {
            if let Some(b) = &d.body {
                for s in &b.stmts {
                    visit_stmt(s, f, deep);
                }
            }
        }
        _ => {}
    }
}

fn visit_expr(e: &Expr, f: &mut dyn FnMut(&Expr), deep: bool) {
    f(e);
    let v = |x: &Expr, f: &mut dyn FnMut(&Expr)| visit_expr(x, f, deep);
    match &e.kind {
        ExprKind::Tuple(xs) | ExprKind::List(xs) => xs.iter().for_each(|x| v(x, f)),
        ExprKind::Dict(kv) => kv.iter().for_each(|(k, x)| {
            v(k, f);
            v(x, f)
        }),
        ExprKind::Attr(o, _) => v(o, f),
        ExprKind::Index(o, i) => {
            v(o, f);
            v(i, f)
        }
        ExprKind::Call(c, args) => {
            v(c, f);
            args.iter().for_each(|a| v(a, f))
        }
        ExprKind::Dup(x) | ExprKind::Unary(_, x) | ExprKind::YieldFrom(x) | ExprKind::TypeApp(x, _) => v(x, f),
        ExprKind::Yield(Some(x)) => v(x, f),
        ExprKind::Binary(_, a, b) | ExprKind::And(a, b) | ExprKind::Or(a, b) => {
            v(a, f);
            v(b, f)
        }
        ExprKind::Compare(a, rest) => {
            v(a, f);
            rest.iter().for_each(|(_, x)| v(x, f))
        }
        ExprKind::Ternary { cond, then, else_ } => {
            v(cond, f);
            v(then, f);
            v(else_, f)
        }
        ExprKind::If { branches, else_ } => {
            for (c, b) in branches {
                v(c, f);
                for s in &b.stmts {
                    visit_stmt(s, f, deep);
                }
            }
            if let Some(b) = else_ {
                for s in &b.stmts {
                    visit_stmt(s, f, deep);
                }
            }
        }
        ExprKind::Try(t) => {
            let blocks = std::iter::once(&t.body)
                .chain(t.handlers.iter().map(|h| &h.body))
                .chain(t.else_.iter())
                .chain(t.finally.iter());
            for b in blocks {
                for s in &b.stmts {
                    visit_stmt(s, f, deep);
                }
            }
        }
        ExprKind::Lambda(d) if deep => {
            if let Some(b) = &d.body {
                for s in &b.stmts {
                    visit_stmt(s, f, deep);
                }
            }
        }
        _ => {}
    }
}
