//! Abstract syntax tree for the Augur distribution-native language.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// `let name ~ Dist(...)` — a random variable with a prior distribution.
    Prior {
        name: String,
        dist: Expr,
        span: Span,
    },
    /// `let name = expr` — a deterministic (uncertainty-carrying) binding.
    Let {
        name: String,
        value: Expr,
        span: Span,
    },
    /// `observe Dist(...) = value` — likelihood / conditioning statement.
    Observe { dist: Expr, value: Expr, span: Span },
    /// `if cond { ... } else { ... }` — deterministic control flow.
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Paren(Box<Expr>),
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Bin {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Comparison expression (evaluates to 1.0 for true, 0.0 for false) used in
    /// conditional control flow.
    Cmp {
        op: CmpOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
}
