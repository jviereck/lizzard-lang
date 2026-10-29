pub mod ast;
pub mod lexer;
pub mod parser;
pub mod token;

pub use parser::parse;
pub use token::Span;

/// A compile-time diagnostic (syntax, ownership or type error).
#[derive(Clone, Debug)]
pub struct Error {
    pub kind: &'static str,
    pub msg: String,
    pub span: Span,
}

impl Error {
    pub fn new(kind: &'static str, msg: impl Into<String>, span: Span) -> Error {
        Error { kind, msg: msg.into(), span }
    }

    pub fn syntax(msg: impl Into<String>, span: Span) -> Error {
        Error::new("SyntaxError", msg, span)
    }

    /// Renders the error with the offending source line (just the file for
    /// an error of the whole program, at line 0).
    pub fn render(&self, file: &str, src: &str) -> String {
        if self.span.line == 0 {
            return format!("  File \"{file}\"\n{}: {}", self.kind, self.msg);
        }
        let line = src.lines().nth(self.span.line.saturating_sub(1) as usize).unwrap_or("");
        let caret = " ".repeat(self.span.col.saturating_sub(1) as usize);
        format!(
            "  File \"{file}\", line {}\n    {line}\n    {caret}^\n{}: {}",
            self.span.line, self.kind, self.msg
        )
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}: {}", self.span.line, self.span.col, self.kind, self.msg)
    }
}
