//! Augur compiler frontend: lexing, parsing, and AST.
//!
//! Provides an error-tolerant parser so that partially-invalid programs still
//! produce a usable parse tree (see [`parse`]). This is the foundation that the
//! type-checker, formatter, and LSP build on.

#![warn(missing_docs)]

pub mod ast;
pub mod diagnostics;
pub mod format;
pub mod lexer;
pub mod parser;

pub use ast::{BinOp, CmpOp, Expr, Program, Span, Stmt};
pub use diagnostics::{Diagnostic, Severity};
pub use format::format_program;
pub use parser::{parse, ParseResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prior_and_observe() {
        let src = "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5";
        let r = parse(src);
        assert!(!r.has_errors(), "diagnostics: {:?}", r.diagnostics);
        assert_eq!(r.program.statements.len(), 2);
    }

    #[test]
    fn parses_deterministic_let_and_if() {
        let src =
            "let mu ~ Normal(0,1)\nlet shifted = mu + 2\nif mu > 0 { observe Normal(mu,1) = 1 }";
        let r = parse(src);
        assert!(!r.has_errors(), "diagnostics: {:?}", r.diagnostics);
        assert_eq!(r.program.statements.len(), 3);
    }

    #[test]
    fn error_tolerant_on_garbage() {
        let src = "let mu ~ Normal(0,1)\nthis is not valid @@@\nlet p ~ Beta(1,1)";
        let r = parse(src);
        // Still recovers the two valid priors.
        assert_eq!(r.program.statements.len(), 2);
        assert!(r.has_errors());
    }

    #[test]
    fn unary_minus_and_parens() {
        let src = "let x = -(a * (b + c)) / 2";
        let r = parse(src);
        assert!(!r.has_errors(), "diagnostics: {:?}", r.diagnostics);
    }
}
