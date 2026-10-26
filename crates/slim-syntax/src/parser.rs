//! Recursive-descent parser.

use crate::ast::*;
use crate::token::{Span, Tok, Token};
use crate::Error;

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

type R<T> = Result<T, Error>;

pub fn parse(src: &str) -> R<Module> {
    let toks = crate::lexer::lex(src)?;
    let mut p = Parser { toks, pos: 0 };
    p.module()
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        let i = (self.pos + n).min(self.toks.len() - 1);
        &self.toks[i].tok
    }

    fn span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn advance(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok, what: &str) -> R<Span> {
        if self.at(t) {
            Ok(self.advance().span)
        } else {
            Err(self.unexpected(what))
        }
    }

    fn unexpected(&self, what: &str) -> Error {
        Error::syntax(format!("expected {what}, found {}", self.peek()), self.span())
    }

    fn name(&mut self, what: &str) -> R<(String, Span)> {
        match self.peek().clone() {
            Tok::Name(n) => {
                let s = self.advance().span;
                Ok((n, s))
            }
            _ => Err(self.unexpected(what)),
        }
    }

    fn module(&mut self) -> R<Module> {
        let mut stmts = Vec::new();
        while !self.at(&Tok::Eof) {
            if self.eat(&Tok::Newline) {
                continue;
            }
            stmts.push(self.stmt()?);
        }
        Ok(Module { body: Block { stmts } })
    }

    // ---------------------------------------------------------------- statements

    fn stmt(&mut self) -> R<Stmt> {
        let span = self.span();
        let kind = match self.peek() {
            Tok::Def => StmtKind::Def(Box::new(self.funcdef()?)),
            Tok::Class => StmtKind::Class(Box::new(self.classdef()?)),
            Tok::Trait => StmtKind::Trait(Box::new(self.traitdef()?)),
            Tok::While => {
                self.advance();
                let cond = self.expr()?;
                self.expect(&Tok::Colon, "`:`")?;
                let body = self.suite()?;
                StmtKind::While { cond, body }
            }
            Tok::For => {
                self.advance();
                let target = self.target_list()?;
                self.expect(&Tok::In, "`in`")?;
                let iter = self.expr_tuple()?;
                self.expect(&Tok::Colon, "`:`")?;
                let body = self.suite()?;
                StmtKind::For { target, iter, body }
            }
            Tok::With => {
                self.advance();
                let ctx = self.expr()?;
                let name = if self.eat(&Tok::As) { Some(self.name("name")?.0) } else { None };
                self.expect(&Tok::Colon, "`:`")?;
                let body = self.suite()?;
                StmtKind::With { ctx, name, body }
            }
            Tok::Indent => return Err(Error::syntax("unexpected indent", span)),
            _ => {
                let (stmt, ended_block) = self.simple_stmt()?;
                if !ended_block {
                    if self.at(&Tok::Semi) {
                        return Err(Error::syntax("`;` is not supported; put each statement on its own line", self.span()));
                    }
                    self.expect_line_end()?;
                }
                return Ok(stmt);
            }
        };
        Ok(Stmt { kind, span })
    }

    fn expect_line_end(&mut self) -> R<()> {
        match self.peek() {
            Tok::Newline => {
                self.advance();
                Ok(())
            }
            Tok::Eof | Tok::RBrace | Tok::Dedent => Ok(()),
            _ => Err(self.unexpected("end of line")),
        }
    }

    /// Parses a simple statement. Returns whether it ended with an indented
    /// block (in which case no NEWLINE follows).
    fn simple_stmt(&mut self) -> R<(Stmt, bool)> {
        let span = self.span();
        let mk = |kind| Stmt { kind, span };
        match self.peek() {
            Tok::Return => {
                self.advance();
                if matches!(self.peek(), Tok::Newline | Tok::Eof | Tok::Semi | Tok::RBrace | Tok::Dedent) {
                    return Ok((mk(StmtKind::Return(None)), false));
                }
                let (v, blk) = self.value()?;
                Ok((mk(StmtKind::Return(Some(v))), blk))
            }
            Tok::Pass => {
                self.advance();
                Ok((mk(StmtKind::Pass), false))
            }
            Tok::Break => {
                self.advance();
                Ok((mk(StmtKind::Break), false))
            }
            Tok::Continue => {
                self.advance();
                Ok((mk(StmtKind::Continue), false))
            }
            Tok::Raise => {
                self.advance();
                if matches!(self.peek(), Tok::Newline | Tok::Eof | Tok::RBrace | Tok::Dedent) {
                    return Ok((mk(StmtKind::Raise { exc: None, cause: None }), false));
                }
                let exc = self.expr()?;
                let cause = if self.eat(&Tok::From) { Some(self.expr()?) } else { None };
                Ok((mk(StmtKind::Raise { exc: Some(exc), cause }), false))
            }
            Tok::Assert => {
                self.advance();
                let cond = self.expr()?;
                let msg = if self.eat(&Tok::Comma) { Some(self.expr()?) } else { None };
                Ok((mk(StmtKind::Assert { cond, msg }), false))
            }
            Tok::Global => {
                self.advance();
                let mut names = vec![self.name("name")?.0];
                while self.eat(&Tok::Comma) {
                    names.push(self.name("name")?.0);
                }
                Ok((mk(StmtKind::Global(names)), false))
            }
            Tok::Del => {
                self.advance();
                let mut targets = vec![self.expr()?];
                while self.eat(&Tok::Comma) {
                    targets.push(self.expr()?);
                }
                Ok((mk(StmtKind::Del(targets)), false))
            }
            _ => self.expr_stmt(),
        }
    }

    fn expr_stmt(&mut self) -> R<(Stmt, bool)> {
        let span = self.span();
        let mk = |kind| Stmt { kind, span };
        let (first, blk) = self.value()?;
        if blk {
            return Ok((mk(StmtKind::Expr(first)), true));
        }
        let aug = |t: &Tok| -> Option<BinOp> {
            Some(match t {
                Tok::PlusEq => BinOp::Add,
                Tok::MinusEq => BinOp::Sub,
                Tok::StarEq => BinOp::Mul,
                Tok::SlashEq => BinOp::Div,
                Tok::DSlashEq => BinOp::FloorDiv,
                Tok::PercentEq => BinOp::Mod,
                _ => return None,
            })
        };
        match self.peek().clone() {
            Tok::Assign => {
                self.advance();
                check_target(&first)?;
                let (value, blk) = self.value()?;
                Ok((mk(StmtKind::Assign { target: first, value }), blk))
            }
            Tok::Colon => {
                self.advance();
                check_target(&first)?;
                let ann = self.type_expr()?;
                if self.eat(&Tok::Assign) {
                    let (value, blk) = self.value()?;
                    Ok((mk(StmtKind::AnnAssign { target: first, ann, value: Some(value) }), blk))
                } else {
                    Ok((mk(StmtKind::AnnAssign { target: first, ann, value: None }), false))
                }
            }
            Tok::LArrow => {
                check_target(&first)?;
                let mut targets = vec![first];
                loop {
                    self.advance();
                    let (e, blk) = self.value()?;
                    if !blk && self.at(&Tok::LArrow) {
                        check_target(&e)?;
                        targets.push(e);
                    } else {
                        return Ok((mk(StmtKind::Move { targets, value: e }), blk));
                    }
                }
            }
            t => {
                if let Some(op) = aug(&t) {
                    self.advance();
                    check_target(&first)?;
                    let (value, blk) = self.value()?;
                    return Ok((mk(StmtKind::AugAssign { target: first, op, value }), blk));
                }
                Ok((mk(StmtKind::Expr(first)), false))
            }
        }
    }

    /// A value position: allows block-form `if`/`try` and bare tuples.
    fn value(&mut self) -> R<(Expr, bool)> {
        match self.peek() {
            Tok::If => Ok((self.if_expr()?, true)),
            Tok::Try => Ok((self.try_expr()?, true)),
            Tok::Yield => Ok((self.yield_expr()?, false)),
            _ => Ok((self.expr_tuple()?, false)),
        }
    }

    fn if_expr(&mut self) -> R<Expr> {
        let span = self.expect(&Tok::If, "`if`")?;
        let mut branches = Vec::new();
        let cond = self.expr()?;
        self.expect(&Tok::Colon, "`:`")?;
        let body = self.suite()?;
        branches.push((cond, body));
        let mut else_ = None;
        loop {
            if self.eat(&Tok::Elif) {
                let cond = self.expr()?;
                self.expect(&Tok::Colon, "`:`")?;
                let body = self.suite()?;
                branches.push((cond, body));
            } else if self.eat(&Tok::Else) {
                self.expect(&Tok::Colon, "`:`")?;
                else_ = Some(self.suite()?);
                break;
            } else {
                break;
            }
        }
        Ok(Expr { kind: ExprKind::If { branches, else_ }, span })
    }

    fn try_expr(&mut self) -> R<Expr> {
        let span = self.expect(&Tok::Try, "`try`")?;
        self.expect(&Tok::Colon, "`:`")?;
        let body = self.suite()?;
        let mut handlers = Vec::new();
        while self.at(&Tok::Except) {
            let hspan = self.advance().span;
            let mut types = Vec::new();
            if !self.at(&Tok::Colon) {
                if self.at(&Tok::LParen) && !matches!(self.peek_at(1), Tok::RParen) {
                    self.advance();
                    types.push(self.type_expr()?);
                    while self.eat(&Tok::Comma) {
                        if self.at(&Tok::RParen) {
                            break;
                        }
                        types.push(self.type_expr()?);
                    }
                    self.expect(&Tok::RParen, "`)`")?;
                } else {
                    types.push(self.type_expr()?);
                }
            }
            let name = if self.eat(&Tok::As) { Some(self.name("name")?.0) } else { None };
            self.expect(&Tok::Colon, "`:`")?;
            let body = self.suite()?;
            handlers.push(Handler { types, name, body, span: hspan });
        }
        let else_ = if self.eat(&Tok::Else) {
            self.expect(&Tok::Colon, "`:`")?;
            Some(self.suite()?)
        } else {
            None
        };
        let finally = if self.eat(&Tok::Finally) {
            self.expect(&Tok::Colon, "`:`")?;
            Some(self.suite()?)
        } else {
            None
        };
        if handlers.is_empty() && finally.is_none() {
            return Err(Error::syntax("`try` needs at least one `except` or a `finally`", span));
        }
        Ok(Expr { kind: ExprKind::Try(Box::new(TryExpr { body, handlers, else_, finally })), span })
    }

    fn yield_expr(&mut self) -> R<Expr> {
        let span = self.expect(&Tok::Yield, "`yield`")?;
        if self.eat(&Tok::From) {
            let e = self.expr()?;
            return Ok(Expr { kind: ExprKind::YieldFrom(Box::new(e)), span });
        }
        if matches!(
            self.peek(),
            Tok::Newline | Tok::Eof | Tok::RParen | Tok::RBrace | Tok::Dedent | Tok::Semi
        ) {
            return Ok(Expr { kind: ExprKind::Yield(None), span });
        }
        let e = self.expr_tuple()?;
        Ok(Expr { kind: ExprKind::Yield(Some(Box::new(e))), span })
    }

    /// An indented block or a simple statement on the same line.
    fn suite(&mut self) -> R<Block> {
        if self.eat(&Tok::Newline) {
            if !self.at(&Tok::Indent) {
                return Err(self.unexpected("an indented block"));
            }
            self.advance();
            let mut stmts = Vec::new();
            while !self.at(&Tok::Dedent) && !self.at(&Tok::Eof) {
                if self.eat(&Tok::Newline) {
                    continue;
                }
                stmts.push(self.stmt()?);
            }
            self.eat(&Tok::Dedent);
            Ok(Block { stmts })
        } else {
            let (stmt, blk) = self.simple_stmt()?;
            if !blk {
                self.expect_line_end()?;
            }
            Ok(Block { stmts: vec![stmt] })
        }
    }

    fn params(&mut self, close: &Tok) -> R<Vec<Param>> {
        let mut params = Vec::new();
        while !self.at(close) {
            let (name, span) = self.name("parameter name")?;
            let mut ty = None;
            let mut keep = false;
            if self.eat(&Tok::Colon) {
                ty = Some(self.type_expr()?);
                if self.at(&Tok::Plus) && matches!(self.peek_at(1), Tok::Name(n) if n == "keep") {
                    self.advance();
                    self.advance();
                    keep = true;
                }
            }
            if params.iter().any(|p: &Param| p.name == name) {
                return Err(Error::syntax(format!("duplicate parameter `{name}`"), span));
            }
            params.push(Param { name, ty, keep, span });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(params)
    }

    /// `<T, U: Ord + Hash>` after a class, trait or function name.
    fn type_params(&mut self) -> R<Vec<TypeParam>> {
        let mut out: Vec<TypeParam> = Vec::new();
        if !self.eat(&Tok::Lt) {
            return Ok(out);
        }
        loop {
            let (name, span) = self.name("type parameter name")?;
            if out.iter().any(|p| p.name == name) {
                return Err(Error::syntax(format!("duplicate type parameter `{name}`"), span));
            }
            let mut bounds = Vec::new();
            if self.eat(&Tok::Colon) {
                bounds.push(self.type_expr()?);
                while self.eat(&Tok::Plus) {
                    bounds.push(self.type_expr()?);
                }
            }
            out.push(TypeParam { name, bounds, span });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.close_angle()?;
        Ok(out)
    }

    fn funcdef(&mut self) -> R<FuncDef> {
        let span = self.expect(&Tok::Def, "`def`")?;
        let (name, _) = self.name("function name")?;
        let tparams = self.type_params()?;
        self.expect(&Tok::LParen, "`(`")?;
        let params = self.params(&Tok::RParen)?;
        self.expect(&Tok::RParen, "`)`")?;
        let ret = if self.eat(&Tok::Arrow) { Some(self.type_expr()?) } else { None };
        self.expect(&Tok::Colon, "`:`")?;
        let body = self.suite()?;
        let body = if is_ellipsis_body(&body) { None } else { Some(body) };
        Ok(FuncDef { name, tparams, params, ret, body, is_lambda: false, span })
    }

    fn classdef(&mut self) -> R<ClassDef> {
        let span = self.expect(&Tok::Class, "`class`")?;
        let (name, _) = self.name("class name")?;
        let tparams = self.type_params()?;
        let mut traits = Vec::new();
        if self.eat(&Tok::LParen) {
            if !self.at(&Tok::RParen) {
                if !self.at(&Tok::Impl) {
                    return Err(Error::syntax(
                        "classes cannot inherit; list implemented traits as `class C(impl T1, T2)`",
                        self.span(),
                    ));
                }
                self.advance();
                loop {
                    self.eat(&Tok::Impl);
                    let (tname, tspan) = self.name("trait name")?;
                    let mut targs = Vec::new();
                    if self.eat(&Tok::Lt) {
                        loop {
                            targs.push(self.type_expr()?);
                            if !self.eat(&Tok::Comma) {
                                break;
                            }
                        }
                        self.close_angle()?;
                    }
                    traits.push((tname, targs, tspan));
                    if !self.eat(&Tok::Comma) || self.at(&Tok::RParen) {
                        break;
                    }
                }
            }
            self.expect(&Tok::RParen, "`)`")?;
        }
        self.expect(&Tok::Colon, "`:`")?;
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        self.expect(&Tok::Newline, "newline")?;
        if !self.eat(&Tok::Indent) {
            return Err(self.unexpected("an indented class body"));
        }
        while !self.at(&Tok::Dedent) && !self.at(&Tok::Eof) {
            match self.peek().clone() {
                Tok::Newline => {
                    self.advance();
                }
                Tok::Def => {
                    let f = self.funcdef()?;
                    if f.body.is_none() {
                        return Err(Error::syntax(
                            "class methods need a body (`...` bodies are only allowed in traits)",
                            f.span,
                        ));
                    }
                    methods.push(f);
                }
                Tok::Pass => {
                    self.advance();
                    self.expect_line_end()?;
                }
                Tok::Str(_) => {
                    self.advance();
                    self.expect_line_end()?;
                }
                Tok::Name(n) => {
                    let fspan = self.advance().span;
                    self.expect(&Tok::Colon, "`:` and a field type")?;
                    let ty = self.type_expr()?;
                    if self.at(&Tok::Assign) {
                        return Err(Error::syntax(
                            "fields cannot have default values; assign them in `__init__`",
                            self.span(),
                        ));
                    }
                    self.expect_line_end()?;
                    fields.push(FieldDecl { name: n, ty, span: fspan });
                }
                _ => return Err(self.unexpected("a field declaration or method")),
            }
        }
        self.eat(&Tok::Dedent);
        Ok(ClassDef { name, tparams, traits, fields, methods, span })
    }

    fn traitdef(&mut self) -> R<TraitDef> {
        let span = self.expect(&Tok::Trait, "`trait`")?;
        let (name, _) = self.name("trait name")?;
        let tparams = self.type_params()?;
        self.expect(&Tok::Colon, "`:`")?;
        let mut methods = Vec::new();
        self.expect(&Tok::Newline, "newline")?;
        if !self.eat(&Tok::Indent) {
            return Err(self.unexpected("an indented trait body"));
        }
        while !self.at(&Tok::Dedent) && !self.at(&Tok::Eof) {
            match self.peek() {
                Tok::Newline => {
                    self.advance();
                }
                Tok::Def => methods.push(self.funcdef()?),
                Tok::Pass | Tok::Str(_) => {
                    self.advance();
                    self.expect_line_end()?;
                }
                _ => return Err(self.unexpected("a method")),
            }
        }
        self.eat(&Tok::Dedent);
        Ok(TraitDef { name, tparams, methods, span })
    }

    fn target_list(&mut self) -> R<Expr> {
        let span = self.span();
        let first = self.bitor()?;
        if !self.at(&Tok::Comma) {
            check_target(&first)?;
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat(&Tok::Comma) {
            if self.at(&Tok::In) {
                break;
            }
            items.push(self.bitor()?);
        }
        let e = Expr { kind: ExprKind::Tuple(items), span };
        check_target(&e)?;
        Ok(e)
    }

    // ---------------------------------------------------------------- types

    pub fn type_expr(&mut self) -> R<TypeExpr> {
        let span = self.span();
        if self.at(&Tok::LParen) {
            self.advance();
            self.expect(&Tok::RParen, "`)` (the unit type is written `()`)")?;
            return Ok(TypeExpr::Unit(span));
        }
        let (name, span) = self.name("a type")?;
        let mut args = Vec::new();
        if self.eat(&Tok::Lt) {
            loop {
                args.push(self.type_expr()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.close_angle()?;
        }
        Ok(TypeExpr::Named(name, args, span))
    }

    /// Consumes one `>`, splitting `>>` and `>=` tokens when needed.
    fn close_angle(&mut self) -> R<()> {
        match self.peek() {
            Tok::Gt => {
                self.advance();
                Ok(())
            }
            Tok::Shr => {
                self.toks[self.pos].tok = Tok::Gt;
                Ok(())
            }
            Tok::Ge => {
                self.toks[self.pos].tok = Tok::Assign;
                Ok(())
            }
            _ => Err(self.unexpected("`>`")),
        }
    }

    // ---------------------------------------------------------------- expressions

    fn expr_tuple(&mut self) -> R<Expr> {
        let span = self.span();
        let first = self.expr()?;
        if !self.at(&Tok::Comma) {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat(&Tok::Comma) {
            if !self.starts_expr() {
                break;
            }
            items.push(self.expr()?);
        }
        Ok(Expr { kind: ExprKind::Tuple(items), span })
    }

    fn starts_expr(&self) -> bool {
        !matches!(
            self.peek(),
            Tok::Newline
                | Tok::Eof
                | Tok::RParen
                | Tok::RBracket
                | Tok::RBrace
                | Tok::Assign
                | Tok::Colon
                | Tok::LArrow
                | Tok::Dedent
                | Tok::Semi
        )
    }

    pub fn expr(&mut self) -> R<Expr> {
        if self.at(&Tok::Pipe) {
            return self.lambda();
        }
        self.ternary()
    }

    fn lambda(&mut self) -> R<Expr> {
        let span = self.expect(&Tok::Pipe, "`|`")?;
        let params = self.params(&Tok::Pipe)?;
        self.expect(&Tok::Pipe, "`|`")?;
        let body = if self.eat(&Tok::LBrace) {
            let mut stmts = Vec::new();
            while !self.at(&Tok::RBrace) {
                if self.at(&Tok::Eof) {
                    return Err(self.unexpected("`}`"));
                }
                if self.eat(&Tok::Newline) {
                    continue;
                }
                stmts.push(self.stmt()?);
            }
            self.expect(&Tok::RBrace, "`}`")?;
            Block { stmts }
        } else {
            let e = self.expr()?;
            let s = e.span;
            Block { stmts: vec![Stmt { kind: StmtKind::Expr(e), span: s }] }
        };
        let f = FuncDef {
            tparams: Vec::new(),
            name: "<lambda>".into(),
            params,
            ret: None,
            body: Some(body),
            is_lambda: true,
            span,
        };
        Ok(Expr { kind: ExprKind::Lambda(Box::new(f)), span })
    }

    fn ternary(&mut self) -> R<Expr> {
        let e = self.or_expr()?;
        if self.at(&Tok::If) {
            let span = self.advance().span;
            let cond = self.or_expr()?;
            self.expect(&Tok::Else, "`else` in conditional expression")?;
            let else_ = self.expr()?;
            return Ok(Expr {
                kind: ExprKind::Ternary { cond: Box::new(cond), then: Box::new(e), else_: Box::new(else_) },
                span,
            });
        }
        Ok(e)
    }

    fn or_expr(&mut self) -> R<Expr> {
        let mut e = self.and_expr()?;
        while self.at(&Tok::Or) {
            let span = self.advance().span;
            let r = self.and_expr()?;
            e = Expr { kind: ExprKind::Or(Box::new(e), Box::new(r)), span };
        }
        Ok(e)
    }

    fn and_expr(&mut self) -> R<Expr> {
        let mut e = self.not_expr()?;
        while self.at(&Tok::And) {
            let span = self.advance().span;
            let r = self.not_expr()?;
            e = Expr { kind: ExprKind::And(Box::new(e), Box::new(r)), span };
        }
        Ok(e)
    }

    fn not_expr(&mut self) -> R<Expr> {
        if self.at(&Tok::Not) {
            let span = self.advance().span;
            let e = self.not_expr()?;
            return Ok(Expr { kind: ExprKind::Unary(UnOp::Not, Box::new(e)), span });
        }
        self.comparison()
    }

    fn comparison(&mut self) -> R<Expr> {
        let first = self.bitor()?;
        let mut rest = Vec::new();
        let span = self.span();
        loop {
            let op = match self.peek() {
                Tok::Eq => CmpOp::Eq,
                Tok::Ne => CmpOp::Ne,
                Tok::Lt => CmpOp::Lt,
                Tok::Le => CmpOp::Le,
                Tok::Gt => CmpOp::Gt,
                Tok::Ge => CmpOp::Ge,
                Tok::In => CmpOp::In,
                Tok::Is => {
                    if matches!(self.peek_at(1), Tok::Not) {
                        self.advance();
                        CmpOp::IsNot
                    } else {
                        CmpOp::Is
                    }
                }
                Tok::Not if matches!(self.peek_at(1), Tok::In) => {
                    self.advance();
                    CmpOp::NotIn
                }
                _ => break,
            };
            self.advance();
            rest.push((op, self.bitor()?));
        }
        if rest.is_empty() {
            Ok(first)
        } else {
            Ok(Expr { kind: ExprKind::Compare(Box::new(first), rest), span })
        }
    }

    fn binary_level(
        &mut self,
        next: fn(&mut Self) -> R<Expr>,
        ops: &[(Tok, BinOp)],
    ) -> R<Expr> {
        let mut e = next(self)?;
        'outer: loop {
            for (t, op) in ops {
                if self.at(t) {
                    let span = self.advance().span;
                    let r = next(self)?;
                    e = Expr { kind: ExprKind::Binary(*op, Box::new(e), Box::new(r)), span };
                    continue 'outer;
                }
            }
            return Ok(e);
        }
    }

    fn bitor(&mut self) -> R<Expr> {
        self.binary_level(Self::bitxor, &[(Tok::Pipe, BinOp::BitOr)])
    }

    fn bitxor(&mut self) -> R<Expr> {
        self.binary_level(Self::bitand, &[(Tok::Caret, BinOp::BitXor)])
    }

    fn bitand(&mut self) -> R<Expr> {
        self.binary_level(Self::shift, &[(Tok::Amp, BinOp::BitAnd)])
    }

    fn shift(&mut self) -> R<Expr> {
        self.binary_level(Self::arith, &[(Tok::Shl, BinOp::Shl), (Tok::Shr, BinOp::Shr)])
    }

    fn arith(&mut self) -> R<Expr> {
        self.binary_level(Self::term, &[(Tok::Plus, BinOp::Add), (Tok::Minus, BinOp::Sub)])
    }

    fn term(&mut self) -> R<Expr> {
        self.binary_level(
            Self::factor,
            &[
                (Tok::Star, BinOp::Mul),
                (Tok::Slash, BinOp::Div),
                (Tok::DSlash, BinOp::FloorDiv),
                (Tok::Percent, BinOp::Mod),
            ],
        )
    }

    fn factor(&mut self) -> R<Expr> {
        let op = match self.peek() {
            Tok::Minus => UnOp::Neg,
            Tok::Plus => UnOp::Pos,
            Tok::Tilde => UnOp::Invert,
            _ => return self.power(),
        };
        let span = self.advance().span;
        let e = self.factor()?;
        Ok(Expr { kind: ExprKind::Unary(op, Box::new(e)), span })
    }

    fn power(&mut self) -> R<Expr> {
        let base = self.postfix()?;
        if self.at(&Tok::DStar) {
            let span = self.advance().span;
            let exp = self.factor()?;
            return Ok(Expr { kind: ExprKind::Binary(BinOp::Pow, Box::new(base), Box::new(exp)), span });
        }
        Ok(base)
    }

    fn postfix(&mut self) -> R<Expr> {
        let mut e = self.atom()?;
        loop {
            match self.peek() {
                Tok::LParen => {
                    let span = self.advance().span;
                    let mut args = Vec::new();
                    while !self.at(&Tok::RParen) {
                        args.push(self.expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)`")?;
                    e = Expr { kind: ExprKind::Call(Box::new(e), args), span };
                }
                Tok::Dot => {
                    let span = self.advance().span;
                    let (name, _) = self.name("attribute name")?;
                    e = Expr { kind: ExprKind::Attr(Box::new(e), name), span };
                }
                Tok::LBracket => {
                    let span = self.advance().span;
                    let idx = self.expr_tuple()?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    e = Expr { kind: ExprKind::Index(Box::new(e), Box::new(idx)), span };
                }
                Tok::PlusPlus => {
                    let span = self.advance().span;
                    e = Expr { kind: ExprKind::Dup(Box::new(e)), span };
                }
                Tok::ColonColon => {
                    let span = self.advance().span;
                    self.expect(&Tok::Lt, "`<` after `::` (type arguments: `f::<T>`)")?;
                    let mut args = Vec::new();
                    loop {
                        args.push(self.type_expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.close_angle()?;
                    e = Expr { kind: ExprKind::TypeApp(Box::new(e), args), span };
                }
                _ => return Ok(e),
            }
        }
    }

    fn atom(&mut self) -> R<Expr> {
        let span = self.span();
        let mk = |kind| Ok(Expr { kind, span });
        match self.peek().clone() {
            Tok::Name(n) => {
                self.advance();
                mk(ExprKind::Name(n))
            }
            Tok::Int(v) => {
                self.advance();
                mk(ExprKind::Int(v))
            }
            Tok::Float(v) => {
                self.advance();
                mk(ExprKind::Float(v))
            }
            Tok::Str(s) => {
                self.advance();
                mk(ExprKind::Str(s))
            }
            Tok::True => {
                self.advance();
                mk(ExprKind::Bool(true))
            }
            Tok::False => {
                self.advance();
                mk(ExprKind::Bool(false))
            }
            Tok::None => {
                self.advance();
                mk(ExprKind::NoneLit)
            }
            Tok::Ellipsis => {
                self.advance();
                mk(ExprKind::Ellipsis)
            }
            Tok::LParen => {
                self.advance();
                if self.eat(&Tok::RParen) {
                    return mk(ExprKind::Unit);
                }
                let e = if self.at(&Tok::Yield) { self.yield_expr()? } else { self.expr()? };
                if self.at(&Tok::Comma) {
                    let mut items = vec![e];
                    while self.eat(&Tok::Comma) {
                        if self.at(&Tok::RParen) {
                            break;
                        }
                        items.push(self.expr()?);
                    }
                    self.expect(&Tok::RParen, "`)`")?;
                    return mk(ExprKind::Tuple(items));
                }
                self.expect(&Tok::RParen, "`)`")?;
                Ok(e)
            }
            Tok::LBracket => {
                self.advance();
                let mut items = Vec::new();
                while !self.at(&Tok::RBracket) {
                    items.push(self.expr()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBracket, "`]`")?;
                mk(ExprKind::List(items))
            }
            Tok::LBrace => {
                self.advance();
                let mut items = Vec::new();
                while !self.at(&Tok::RBrace) {
                    let k = self.expr()?;
                    self.expect(&Tok::Colon, "`:` in dict literal")?;
                    let v = self.expr()?;
                    items.push((k, v));
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBrace, "`}`")?;
                mk(ExprKind::Dict(items))
            }
            Tok::Pipe => self.lambda(),
            Tok::If => Err(Error::syntax(
                "block `if` expressions are only allowed as a statement, an assignment value or a return value; use `a if c else b` inside expressions",
                span,
            )),
            _ => Err(self.unexpected("an expression")),
        }
    }
}

fn is_ellipsis_body(b: &Block) -> bool {
    matches!(b.stmts.as_slice(), [Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::Ellipsis, .. }), .. }])
}

fn check_target(e: &Expr) -> R<()> {
    match &e.kind {
        ExprKind::Name(_) | ExprKind::Attr(..) | ExprKind::Index(..) => Ok(()),
        ExprKind::Tuple(items) => items.iter().try_for_each(check_target),
        _ => Err(Error::syntax("invalid assignment target", e.span)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_samples() {
        let src = r#"
trait Log:
    def log(self, msg: str) -> (): ...

class Console(impl Log):
    def log(self, msg: str) -> ():
        print(msg)

class Service(impl Log):
    out: Console
    last: Option<str>
    count: int
    def __init__(self, out: Console + keep) -> ():
        self.out <- out
        self.last <- None
        self.count = 0
    def log(self, msg: str) -> ():
        self.out.log(msg)
        old <- self.last <- Some(msg++)
        if old is not None:
            print(old)
        self.count += 1

def classify(n: int) -> str:
    if n > 100:
        return "huge"
    size = if n > 10:
        "big"
    else:
        "small"
    size

double = |x: int| x * 2
report = |n: int| {
    if n > 10:
        print("big")
    n
}
d: dict<str, list<int>>= {}
"#;
        let m = parse(src).unwrap();
        assert_eq!(m.body.stmts.len(), 7);
    }
}
