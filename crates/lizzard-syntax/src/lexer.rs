//! Indentation-aware lexer.
//!
//! The lexer keeps a stack of contexts. A `Block` context tracks indentation
//! and emits NEWLINE / INDENT / DEDENT tokens; a `Bracket` context (inside
//! `()`, `[]` or a dict `{}`) ignores newlines like Python does. A lambda block
//! `|x| { ... }` pushes a fresh `Block` context whose base indentation is set
//! by the first non-blank line after the `{`.

use crate::token::{Span, Tok, Token};
use crate::Error;

enum Ctx {
    Block { indents: Vec<u32>, base_set: bool },
    Bracket(char),
}

struct Lexer {
    chars: Vec<char>,
    i: usize,
    line: u32,
    col: u32,
    toks: Vec<Token>,
    ctx: Vec<Ctx>,
    line_start: bool,
}

pub fn lex(src: &str) -> Result<Vec<Token>, Error> {
    let mut lx = Lexer {
        chars: src.chars().collect(),
        i: 0,
        line: 1,
        col: 0,
        toks: Vec::new(),
        ctx: vec![Ctx::Block { indents: vec![0], base_set: true }],
        line_start: true,
    };
    lx.run()?;
    Ok(lx.toks)
}

impl Lexer {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn peek_at(&self, n: usize) -> Option<char> {
        self.chars.get(self.i + n).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.i).copied()?;
        self.i += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 0;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn span(&self) -> Span {
        Span { line: self.line, col: self.col + 1 }
    }

    fn err(&self, msg: impl Into<String>) -> Error {
        Error::syntax(msg, self.span())
    }

    fn push(&mut self, tok: Tok, span: Span) {
        self.toks.push(Token { tok, span });
    }

    fn in_block(&self) -> bool {
        matches!(self.ctx.last(), Some(Ctx::Block { .. }))
    }

    /// True if the current logical line already has tokens, i.e. a NEWLINE
    /// must be emitted before the line ends.
    fn line_has_content(&self) -> bool {
        match self.toks.last() {
            None => false,
            Some(t) => !matches!(t.tok, Tok::Newline | Tok::Indent | Tok::Dedent)
                && !(t.tok == Tok::LBrace && self.in_block()),
        }
    }

    fn run(&mut self) -> Result<(), Error> {
        loop {
            if self.line_start && self.in_block() {
                let indent = self.measure_indent()?;
                match self.peek() {
                    None => break,
                    Some('\n') => {
                        self.bump();
                        continue;
                    }
                    Some('\r') => {
                        self.bump();
                        continue;
                    }
                    Some('#') => {
                        self.skip_comment();
                        continue;
                    }
                    Some('}') => {
                        self.line_start = false;
                    }
                    Some(_) => {
                        self.process_indent(indent)?;
                        self.line_start = false;
                    }
                }
            }
            let Some(c) = self.peek() else { break };
            let span = self.span();
            match c {
                ' ' | '\t' | '\r' => {
                    self.bump();
                }
                '#' => self.skip_comment(),
                '\\' if self.peek_at(1) == Some('\n') => {
                    self.bump();
                    self.bump();
                }
                '\n' => {
                    self.bump();
                    if self.in_block() {
                        if self.line_has_content() {
                            self.push(Tok::Newline, span);
                        }
                        self.line_start = true;
                    }
                }
                '0'..='9' => self.number()?,
                '.' if matches!(self.peek_at(1), Some('0'..='9')) => self.number()?,
                '"' | '\'' => self.string()?,
                c if c.is_alphabetic() || c == '_' => self.name(),
                _ => self.punct()?,
            }
        }
        // End of input.
        match self.ctx.last() {
            Some(Ctx::Bracket(c)) => {
                return Err(self.err(format!("unclosed `{c}` at end of file")));
            }
            _ if self.ctx.len() > 1 => {
                return Err(self.err("unclosed lambda block `{` at end of file"));
            }
            _ => {}
        }
        let span = self.span();
        if self.line_has_content() {
            self.push(Tok::Newline, span);
        }
        if let Some(Ctx::Block { indents, .. }) = self.ctx.last_mut() {
            let n = indents.len() - 1;
            indents.truncate(1);
            for _ in 0..n {
                self.toks.push(Token { tok: Tok::Dedent, span });
            }
        }
        self.push(Tok::Eof, span);
        Ok(())
    }

    fn skip_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.bump();
        }
    }

    fn measure_indent(&mut self) -> Result<u32, Error> {
        let mut col = 0u32;
        loop {
            match self.peek() {
                Some(' ') => {
                    col += 1;
                    self.bump();
                }
                Some('\t') => {
                    col = (col / 8 + 1) * 8;
                    self.bump();
                }
                _ => return Ok(col),
            }
        }
    }

    fn process_indent(&mut self, col: u32) -> Result<(), Error> {
        let span = self.span();
        let Some(Ctx::Block { indents, base_set }) = self.ctx.last_mut() else {
            return Ok(());
        };
        if !*base_set {
            *indents = vec![col];
            *base_set = true;
            return Ok(());
        }
        let top = *indents.last().unwrap();
        if col > top {
            indents.push(col);
            self.toks.push(Token { tok: Tok::Indent, span });
        } else if col < top {
            let mut dedents = 0;
            while indents.len() > 1 && col < *indents.last().unwrap() {
                indents.pop();
                dedents += 1;
            }
            if col != *indents.last().unwrap() {
                if indents.len() == 1 && col < indents[0] {
                    return Err(Error::syntax(
                        "line is indented less than the start of its lambda block",
                        span,
                    ));
                }
                return Err(Error::syntax("unindent does not match any outer indentation level", span));
            }
            for _ in 0..dedents {
                self.toks.push(Token { tok: Tok::Dedent, span });
            }
        }
        Ok(())
    }

    fn name(&mut self) {
        let span = self.span();
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        let tok = Tok::keyword(&s).unwrap_or(Tok::Name(s));
        self.push(tok, span);
    }

    fn number(&mut self) -> Result<(), Error> {
        let span = self.span();
        let mut s = String::new();
        if self.peek() == Some('0') && matches!(self.peek_at(1), Some('x' | 'X' | 'b' | 'B' | 'o' | 'O')) {
            self.bump();
            let radix = match self.bump().unwrap().to_ascii_lowercase() {
                'x' => 16,
                'b' => 2,
                _ => 8,
            };
            while let Some(c) = self.peek() {
                if c.is_digit(radix) {
                    s.push(c);
                } else if c != '_' {
                    break;
                }
                self.bump();
            }
            let v = u64::from_str_radix(&s, radix)
                .map_err(|_| Error::syntax("integer literal out of range", span))?;
            self.push(Tok::Int(v), span);
            return Ok(());
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
            } else if c == '_' {
            } else if c == '.' && !is_float && matches!(self.peek_at(1), Some('0'..='9')) {
                is_float = true;
                s.push(c);
            } else if (c == 'e' || c == 'E')
                && (matches!(self.peek_at(1), Some('0'..='9'))
                    || (matches!(self.peek_at(1), Some('+' | '-'))
                        && matches!(self.peek_at(2), Some('0'..='9'))))
            {
                is_float = true;
                s.push(c);
                self.bump();
                s.push(self.peek().unwrap());
            } else {
                break;
            }
            self.bump();
        }
        if is_float {
            let v: f64 = s.parse().map_err(|_| Error::syntax("invalid float literal", span))?;
            self.push(Tok::Float(v), span);
        } else {
            let v: u64 = s.parse().map_err(|_| {
                Error::syntax("integer literal out of range (use bigint(\"...\") for big numbers)", span)
            })?;
            self.push(Tok::Int(v), span);
        }
        Ok(())
    }

    fn string(&mut self) -> Result<(), Error> {
        let span = self.span();
        let q = self.bump().unwrap();
        let triple = self.peek() == Some(q) && self.peek_at(1) == Some(q);
        if triple {
            self.bump();
            self.bump();
        }
        let mut s = String::new();
        loop {
            let Some(c) = self.bump() else {
                return Err(Error::syntax("unterminated string literal", span));
            };
            if c == q {
                if !triple {
                    break;
                }
                if self.peek() == Some(q) && self.peek_at(1) == Some(q) {
                    self.bump();
                    self.bump();
                    break;
                }
                s.push(c);
                continue;
            }
            if c == '\n' && !triple {
                return Err(Error::syntax("unterminated string literal", span));
            }
            if c == '\\' {
                let Some(e) = self.bump() else {
                    return Err(Error::syntax("unterminated string literal", span));
                };
                match e {
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    '0' => s.push('\0'),
                    '\\' => s.push('\\'),
                    '\'' => s.push('\''),
                    '"' => s.push('"'),
                    '\n' => {}
                    'x' => {
                        let h: String = [self.bump(), self.bump()].iter().flatten().collect();
                        let v = u32::from_str_radix(&h, 16)
                            .map_err(|_| Error::syntax("invalid \\x escape", span))?;
                        s.push(char::from_u32(v).unwrap());
                    }
                    other => {
                        s.push('\\');
                        s.push(other);
                    }
                }
                continue;
            }
            s.push(c);
        }
        // Adjacent string literals are concatenated.
        if let Some(Token { tok: Tok::Str(prev), .. }) = self.toks.last_mut() {
            prev.push_str(&s);
            return Ok(());
        }
        self.push(Tok::Str(s), span);
        Ok(())
    }

    fn punct(&mut self) -> Result<(), Error> {
        let span = self.span();
        let c = self.peek().unwrap();
        let c1 = self.peek_at(1);
        let c2 = self.peek_at(2);
        let (tok, len) = match (c, c1, c2) {
            ('/', Some('/'), Some('=')) => (Tok::DSlashEq, 3),
            ('.', Some('.'), Some('.')) => (Tok::Ellipsis, 3),
            ('-', Some('>'), _) => (Tok::Arrow, 2),
            ('<', Some('-'), _) => (Tok::LArrow, 2),
            ('+', Some('+'), _) => (Tok::PlusPlus, 2),
            ('/', Some('/'), _) => (Tok::DSlash, 2),
            ('*', Some('*'), _) => (Tok::DStar, 2),
            ('=', Some('='), _) => (Tok::Eq, 2),
            ('!', Some('='), _) => (Tok::Ne, 2),
            ('<', Some('='), _) => (Tok::Le, 2),
            ('>', Some('='), _) => (Tok::Ge, 2),
            ('<', Some('<'), _) => (Tok::Shl, 2),
            ('>', Some('>'), _) => (Tok::Shr, 2),
            ('+', Some('='), _) => (Tok::PlusEq, 2),
            ('-', Some('='), _) => (Tok::MinusEq, 2),
            ('*', Some('='), _) => (Tok::StarEq, 2),
            ('/', Some('='), _) => (Tok::SlashEq, 2),
            ('%', Some('='), _) => (Tok::PercentEq, 2),
            ('(', ..) => (Tok::LParen, 1),
            (')', ..) => (Tok::RParen, 1),
            ('[', ..) => (Tok::LBracket, 1),
            (']', ..) => (Tok::RBracket, 1),
            ('{', ..) => (Tok::LBrace, 1),
            ('}', ..) => (Tok::RBrace, 1),
            (',', ..) => (Tok::Comma, 1),
            (':', Some(':'), _) => (Tok::ColonColon, 2),
            (':', ..) => (Tok::Colon, 1),
            ('.', ..) => (Tok::Dot, 1),
            (';', ..) => (Tok::Semi, 1),
            ('=', ..) => (Tok::Assign, 1),
            ('+', ..) => (Tok::Plus, 1),
            ('-', ..) => (Tok::Minus, 1),
            ('*', ..) => (Tok::Star, 1),
            ('/', ..) => (Tok::Slash, 1),
            ('%', ..) => (Tok::Percent, 1),
            ('<', ..) => (Tok::Lt, 1),
            ('>', ..) => (Tok::Gt, 1),
            ('|', ..) => (Tok::Pipe, 1),
            ('&', ..) => (Tok::Amp, 1),
            ('^', ..) => (Tok::Caret, 1),
            ('~', ..) => (Tok::Tilde, 1),
            _ => return Err(self.err(format!("unexpected character `{c}`"))),
        };
        for _ in 0..len {
            self.bump();
        }
        match tok {
            Tok::LParen => self.ctx.push(Ctx::Bracket('(')),
            Tok::LBracket => self.ctx.push(Ctx::Bracket('[')),
            Tok::LBrace => {
                let lambda = matches!(self.toks.last(), Some(Token { tok: Tok::Pipe, .. }));
                self.push(Tok::LBrace, span);
                if lambda {
                    self.ctx.push(Ctx::Block { indents: Vec::new(), base_set: false });
                } else {
                    self.ctx.push(Ctx::Bracket('{'));
                }
                return Ok(());
            }
            Tok::RParen | Tok::RBracket | Tok::RBrace => {
                let open = match tok {
                    Tok::RParen => '(',
                    Tok::RBracket => '[',
                    _ => '{',
                };
                match self.ctx.last() {
                    Some(Ctx::Bracket(o)) if *o == open => {
                        self.ctx.pop();
                    }
                    Some(Ctx::Block { .. }) if open == '{' && self.ctx.len() > 1 => {
                        // End of a lambda block: close the last line and all
                        // nested indentation levels.
                        if self.line_has_content() {
                            self.push(Tok::Newline, span);
                        }
                        if let Some(Ctx::Block { indents, .. }) = self.ctx.pop() {
                            for _ in 1..indents.len() {
                                self.toks.push(Token { tok: Tok::Dedent, span });
                            }
                        }
                    }
                    _ => return Err(Error::syntax(format!("unmatched {tok}"), span)),
                }
            }
            _ => {}
        }
        self.push(tok, span);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn indentation() {
        let t = toks("if x:\n    y\nz\n");
        assert_eq!(
            t,
            vec![
                Tok::If,
                Tok::Name("x".into()),
                Tok::Colon,
                Tok::Newline,
                Tok::Indent,
                Tok::Name("y".into()),
                Tok::Newline,
                Tok::Dedent,
                Tok::Name("z".into()),
                Tok::Newline,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn lambda_block() {
        let t = toks("f = |n| {\n        if n:\n            a\n        n\n}\n");
        assert_eq!(
            t,
            vec![
                Tok::Name("f".into()),
                Tok::Assign,
                Tok::Pipe,
                Tok::Name("n".into()),
                Tok::Pipe,
                Tok::LBrace,
                Tok::If,
                Tok::Name("n".into()),
                Tok::Colon,
                Tok::Newline,
                Tok::Indent,
                Tok::Name("a".into()),
                Tok::Newline,
                Tok::Dedent,
                Tok::Name("n".into()),
                Tok::Newline,
                Tok::RBrace,
                Tok::Newline,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn lambda_inside_call() {
        let t = toks("g(|x| {\n    x\n})\n");
        assert_eq!(
            t,
            vec![
                Tok::Name("g".into()),
                Tok::LParen,
                Tok::Pipe,
                Tok::Name("x".into()),
                Tok::Pipe,
                Tok::LBrace,
                Tok::Name("x".into()),
                Tok::Newline,
                Tok::RBrace,
                Tok::RParen,
                Tok::Newline,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn ownership_ops() {
        let t = toks("a <- b++\n");
        assert_eq!(
            t,
            vec![
                Tok::Name("a".into()),
                Tok::LArrow,
                Tok::Name("b".into()),
                Tok::PlusPlus,
                Tok::Newline,
                Tok::Eof
            ]
        );
    }
}
