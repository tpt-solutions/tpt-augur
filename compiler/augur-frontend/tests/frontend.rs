//! Integration tests for the Augur frontend: lexer, parser, formatter,
//! diagnostics, and AST.

use augur_frontend::{
    ast::{BinOp, CmpOp, Expr, Program, Span, Stmt},
    diagnostics::{Diagnostic, Severity},
    format_program, parse, ParseResult,
};
use augur_frontend::lexer::{lex, Tok};

fn no_errors(r: &ParseResult) -> bool {
    !r.has_errors()
}

#[test]
fn lex_emits_expected_tokens() {
    let (tokens, diags) = lex("let mu ~ Normal(0, 1)");
    assert!(diags.is_empty(), "unexpected lex diagnostics: {diags:?}");
    let kinds: Vec<_> = tokens.iter().map(|t| t.tok.clone()).collect();
    assert!(kinds.iter().any(|t| matches!(t, Tok::Let)));
    assert!(kinds.iter().any(|t| matches!(t, Tok::Tilde)));
    assert!(kinds.iter().any(|t| matches!(t, Tok::Num(v) if *v == 1.0)));
}

#[test]
fn lex_numbers_and_floats() {
    let (tokens, _) = lex("12 1.5 0.25");
    let nums: Vec<f64> = tokens
        .iter()
        .filter_map(|t| match t.tok {
            Tok::Num(v) => Some(v),
            _ => None,
        })
        .collect();
    assert_eq!(nums, vec![12.0, 1.5, 0.25]);
}

#[test]
fn lex_reports_invalid_characters() {
    let (_, diags) = lex("let x = 1 @ 2");
    assert!(diags.iter().any(|d| d.severity == Severity::Error));
}

#[test]
fn lex_reports_lone_bang_with_hint() {
    let (_, diags) = lex("!");
    assert!(diags
        .iter()
        .any(|d| d.message.contains("!=")));
}

#[test]
fn lex_skips_comments() {
    let (tokens, diags) = lex("# a comment\nlet x = 1");
    assert!(diags.is_empty());
    let has_let = tokens.iter().any(|t| matches!(t.tok, Tok::Let));
    assert!(has_let);
}

#[test]
fn parse_prior_and_observe() {
    let r = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    assert!(no_errors(&r), "diagnostics: {:?}", r.diagnostics);
    assert_eq!(r.program.statements.len(), 2);
}

#[test]
fn parse_let_and_if_else() {
    let r = parse(
        "let mu ~ Normal(0,1)\nlet shifted = mu + 2\nif mu > 0 { observe Normal(mu,1) = 1 } else { observe Normal(mu,1) = -1 }",
    );
    assert!(no_errors(&r), "diagnostics: {:?}", r.diagnostics);
    assert_eq!(r.program.statements.len(), 3);
    match &r.program.statements[2] {
        Stmt::If {
            then_body, else_body, ..
        } => {
            assert_eq!(then_body.len(), 1);
            assert_eq!(else_body.len(), 1);
        }
        other => panic!("expected If, got {other:?}"),
    }
}

#[test]
fn parse_operator_precedence() {
    let r = parse("let x = 1 + 2 * 3");
    assert!(no_errors(&r));
    if let Stmt::Let { value, .. } = &r.program.statements[0] {
        if let Expr::Bin { op: BinOp::Add, lhs, rhs } = value {
            assert!(matches!(&**lhs, Expr::Num(1.0)));
            if let Expr::Bin { op: BinOp::Mul, lhs: a, rhs: b } = &**rhs {
                assert!(matches!(&**a, Expr::Num(2.0)));
                assert!(matches!(&**b, Expr::Num(3.0)));
            } else {
                panic!("expected mul, got {rhs:?}");
            }
        } else {
            panic!("expected add, got {value:?}");
        }
    } else {
        panic!("expected let");
    }
}

#[test]
fn parse_unary_minus_and_parens() {
    let r = parse("let x = -(a * (b + c)) / 2");
    assert!(no_errors(&r), "diagnostics: {:?}", r.diagnostics);
}

#[test]
fn parse_comparison_is_cmp_expr() {
    let r = parse("if a > b {}");
    assert!(no_errors(&r));
    match &r.program.statements[0] {
        Stmt::If { cond, .. } => {
            assert!(matches!(cond, Expr::Cmp { op: CmpOp::Gt, .. }));
        }
        other => panic!("expected If, got {other:?}"),
    }
}

#[test]
fn error_tolerant_recovers_valid_statements() {
    let r = parse("let mu ~ Normal(0,1)\nthis is not valid @@@\nlet p ~ Beta(1,1)");
    assert_eq!(r.program.statements.len(), 2);
    assert!(r.has_errors());
}

#[test]
fn error_tolerant_recovers_after_block() {
    let r = parse(
        "if mu > 0 { let a = 1 @@@ bad } else { let b = 2 }\nlet tail ~ Normal(0,1)",
    );
    // The trailing prior must still be recovered.
    assert!(r
        .program
        .statements
        .iter()
        .any(|s| matches!(s, Stmt::Prior { .. })));
}

#[test]
fn format_is_idempotent() {
    let src = "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5\nlet s = mu + 1";
    let r = parse(src);
    assert!(no_errors(&r));
    let f1 = format_program(&r.program);
    let r2 = parse(&f1);
    assert!(no_errors(&r2), "diagnostics: {:?}", r2.diagnostics);
    let f2 = format_program(&r2.program);
    assert_eq!(f1, f2);
}

#[test]
fn format_renders_each_statement_kind() {
    let src = "let mu ~ Normal(0, 1)\nlet k = mu * 2\nobserve Normal(mu, 1) = 0.5\nif mu > 0 { let a = 1 } else { let b = 2 }";
    let r = parse(src);
    assert!(no_errors(&r));
    let f = format_program(&r.program);
    assert!(f.contains("let mu ~ Normal(0.0, 1.0)"));
    assert!(f.contains("let k = mu * 2.0"));
    assert!(f.contains("observe Normal(mu, 1.0) = 0.5"));
    assert!(f.contains("if mu > 0.0 {"));
    assert!(f.contains("} else {"));
}

#[test]
fn format_preserves_negation() {
    let r = parse("let x = -2");
    assert!(no_errors(&r));
    let f = format_program(&r.program);
    assert!(f.contains("let x = -2.0"));
}

#[test]
fn diagnostic_constructors_and_severity() {
    let e = Diagnostic::parse_error("boom", Span::new(0, 1));
    assert!(e.is_error());
    assert_eq!(e.severity, Severity::Error);
    let w = Diagnostic::warning("careful", Span::new(0, 1));
    assert!(!w.is_error());
    assert_eq!(w.severity, Severity::Warning);
    let le = Diagnostic::lex_error("lex boom", Span::new(0, 1));
    assert!(le.is_error());
}

#[test]
fn span_new_constructs() {
    let s = Span::new(3, 7);
    assert_eq!(s.start, 3);
    assert_eq!(s.end, 7);
}

#[test]
fn warnings_accessor_is_empty_for_frontend_only() {
    // Degenerate-parameter *warnings* are raised by the IR layer (augur_ir),
    // not the frontend parser, which only ever produces errors. The accessor
    // must therefore be empty on a raw parse result.
    let r = parse("let s ~ Normal(0, -1)");
    assert!(r.warnings().is_empty());
    // And a clean program has no warnings either.
    let r2 = parse("let mu ~ Normal(0, 1)");
    assert!(r2.warnings().is_empty());
}

#[test]
fn program_is_clone_and_debug() {
    let r = parse("let mu ~ Normal(0, 1)");
    let p: Program = r.program.clone();
    let _ = format!("{:?}", p);
    assert_eq!(p.statements.len(), 1);
}
