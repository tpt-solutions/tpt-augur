//! Abstract syntax tree for the Augur distribution-native language.

use serde::{Deserialize, Serialize};

/// Byte-range in the source text (`start..end`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

impl Span {
    /// Construct a span from byte offsets.
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
}

/// A parsed Augur model: an ordered sequence of statements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    /// Top-level statements in source order.
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

/// An expression in the Augur language.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Numeric literal.
    Num(f64),
    /// Variable reference.
    Var(String),
    /// Unary negation.
    Neg(Box<Expr>),
    /// Parenthesised sub-expression.
    Paren(Box<Expr>),
    /// Function or distribution constructor call.
    Call {
        name: String,
        args: Vec<Expr>,
    },
    /// Binary arithmetic expression.
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

/// Binary arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
}

/// Comparison operator used in `if` conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    /// Greater than (`>`).
    Gt,
    /// Less than (`<`).
    Lt,
    /// Greater than or equal (`>=`).
    Ge,
    /// Less than or equal (`<=`).
    Le,
    /// Equal (`==`).
    Eq,
    /// Not equal (`!=`).
    Ne,
}
