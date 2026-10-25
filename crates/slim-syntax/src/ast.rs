use crate::token::Span;

#[derive(Clone, Debug)]
pub struct Module {
    pub body: Block,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Expr(Expr),
    /// `target = value`
    Assign { target: Expr, value: Expr },
    /// `target += value`
    AugAssign { target: Expr, op: BinOp, value: Expr },
    /// `target: ann [= value]`
    AnnAssign { target: Expr, ann: TypeExpr, value: Option<Expr> },
    /// `t1 <- t2 <- ... <- value`: every target receives the old value of the
    /// next one, the last target receives `value`.
    Move { targets: Vec<Expr>, value: Expr },
    Def(Box<FuncDef>),
    Class(Box<ClassDef>),
    Trait(Box<TraitDef>),
    Return(Option<Expr>),
    While { cond: Expr, body: Block },
    For { target: Expr, iter: Expr, body: Block },
    Break,
    Continue,
    Pass,
    Raise { exc: Option<Expr>, cause: Option<Expr> },
    Global(Vec<String>),
    Del(Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Name(String),
    Int(u64),
    Float(f64),
    Str(String),
    Bool(bool),
    NoneLit,
    Unit,
    Ellipsis,
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    /// `expr++`: a new counted reference.
    Dup(Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Chained comparison `a < b <= c`.
    Compare(Box<Expr>, Vec<(CmpOp, Expr)>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    /// `a if cond else b`
    Ternary { cond: Box<Expr>, then: Box<Expr>, else_: Box<Expr> },
    /// Block-form `if` expression.
    If { branches: Vec<(Expr, Block)>, else_: Option<Block> },
    Try(Box<TryExpr>),
    Lambda(Box<FuncDef>),
    Yield(Option<Box<Expr>>),
    YieldFrom(Box<Expr>),
}

#[derive(Clone, Debug)]
pub struct TryExpr {
    pub body: Block,
    pub handlers: Vec<Handler>,
    pub else_: Option<Block>,
    pub finally: Option<Block>,
}

#[derive(Clone, Debug)]
pub struct Handler {
    /// `None` for a bare `except:`; a tuple expression for `except (A, B)`.
    pub types: Vec<TypeExpr>,
    pub name: Option<String>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Pos,
    Not,
    Invert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::FloorDiv => "//",
            BinOp::Mod => "%",
            BinOp::Pow => "**",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Is,
    IsNot,
    In,
    NotIn,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub keep: bool,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FuncDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<TypeExpr>,
    /// `None` for a required trait method (`...` body).
    pub body: Option<Block>,
    pub is_lambda: bool,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FieldDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ClassDef {
    pub name: String,
    pub traits: Vec<(String, Span)>,
    pub fields: Vec<FieldDecl>,
    pub methods: Vec<FuncDef>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TraitDef {
    pub name: String,
    pub methods: Vec<FuncDef>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeExpr {
    /// `()`
    Unit(Span),
    /// `Name` or `Name<args>`
    Named(String, Vec<TypeExpr>, Span),
}

impl TypeExpr {
    pub fn span(&self) -> Span {
        match self {
            TypeExpr::Unit(s) | TypeExpr::Named(_, _, s) => *s,
        }
    }
}

impl std::fmt::Display for TypeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeExpr::Unit(_) => f.write_str("()"),
            TypeExpr::Named(n, args, _) => {
                f.write_str(n)?;
                if !args.is_empty() {
                    f.write_str("<")?;
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{a}")?;
                    }
                    f.write_str(">")?;
                }
                Ok(())
            }
        }
    }
}
