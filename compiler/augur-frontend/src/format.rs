//! Minimal source formatter / pretty-printer for Augur models.
//!
//! This is a pragmatic formatter that re-emits a canonical layout from the
//! parsed AST. When `tptb-format` (from the `tpt-gpu` toolchain) is wired in,
//! this can be replaced by delegating to it; the contract — `format(program)
//! -> String` — stays the same.

use crate::ast::{BinOp, CmpOp, Expr, Program, Stmt};

fn fmt_expr(e: &Expr) -> String {
    match e {
        Expr::Num(v) => {
            if v.fract() == 0.0 && v.abs() < 1e15 {
                format!("{:.1}", v)
            } else {
                format!("{v}")
            }
        }
        Expr::Var(n) => n.clone(),
        Expr::Neg(inner) => format!("-{}", fmt_expr(inner)),
        Expr::Paren(inner) => format!("({})", fmt_expr(inner)),
        Expr::Call { name, args } => {
            let a: Vec<String> = args.iter().map(fmt_expr).collect();
            format!("{name}({})", a.join(", "))
        }
        Expr::Bin { op, lhs, rhs } => {
            let s = format!("{} {} {}", fmt_expr(lhs), bin_op_str(*op), fmt_expr(rhs));
            s
        }
        Expr::Cmp { op, lhs, rhs } => {
            format!("{} {} {}", fmt_expr(lhs), cmp_op_str(*op), fmt_expr(rhs))
        }
    }
}

fn bin_op_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
    }
}

fn cmp_op_str(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Gt => ">",
        CmpOp::Lt => "<",
        CmpOp::Ge => ">=",
        CmpOp::Le => "<=",
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
    }
}

fn fmt_stmt(s: &Stmt, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match s {
        Stmt::Prior { name, dist, .. } => format!("{pad}let {name} ~ {}", fmt_expr(dist)),
        Stmt::Let { name, value, .. } => format!("{pad}let {name} = {}", fmt_expr(value)),
        Stmt::Observe { dist, value, .. } => {
            format!("{pad}observe {} = {}", fmt_expr(dist), fmt_expr(value))
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
            ..
        } => {
            let mut out = format!("{pad}if {} {{\n", fmt_expr(cond));
            for b in then_body {
                out += &fmt_stmt(b, indent + 1);
                out += "\n";
            }
            out += &format!("{pad}}}");
            if !else_body.is_empty() {
                out += " else {\n";
                for b in else_body {
                    out += &fmt_stmt(b, indent + 1);
                    out += "\n";
                }
                out += &format!("{pad}}}");
            }
            out
        }
    }
}

/// Render a parsed program to canonical source text.
pub fn format_program(program: &Program) -> String {
    let mut out = String::new();
    for (i, s) in program.statements.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out += &fmt_stmt(s, 0);
    }
    out
}
