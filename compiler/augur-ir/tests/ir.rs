//! Integration tests for the Augur typed IR: lowering, evaluation, distribution
//! instantiation, the log-joint, and static analysis.

use augur_frontend::parse;
use augur_ir::lower::known_dist_arity;
use augur_ir::{
    eval, instantiate_dist, log_joint, lower, Diagnostic, Env, LowerResult, Model, Severity,
};

fn lower_ok(src: &str) -> Model {
    let r = parse(src);
    assert!(!r.has_errors(), "parse errors: {:?}", r.diagnostics);
    let lr = lower(&r.program);
    assert!(
        !lr.diagnostics.iter().any(Diagnostic::is_error),
        "type errors: {:?}",
        lr.diagnostics
    );
    lr.model
}

fn lower_with_diags(src: &str) -> LowerResult {
    let r = parse(src);
    assert!(!r.has_errors(), "parse errors: {:?}", r.diagnostics);
    lower(&r.program)
}

#[test]
fn lower_normal_normal_conjugate_shape() {
    let m = lower_ok("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    assert_eq!(m.prior_order, vec!["mu".to_string()]);
}

#[test]
fn lower_beta_binomial_conjugate_shape() {
    let m = lower_ok("let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7");
    assert_eq!(m.prior_order, vec!["p".to_string()]);
}

#[test]
fn lower_multiple_priors_preserve_order() {
    let m = lower_ok("let a ~ Normal(0,1)\nlet b ~ Normal(0,1)\nlet c ~ Beta(1,1)");
    assert_eq!(m.prior_order, vec!["a", "b", "c"]);
}

#[test]
fn undeclared_variable_is_error() {
    let lr = lower_with_diags("let x = y + 1");
    assert!(lr.diagnostics.iter().any(Diagnostic::is_error));
}

#[test]
fn duplicate_definition_is_error() {
    let lr = lower_with_diags("let mu ~ Normal(0,1)\nlet mu ~ Normal(0,1)");
    assert!(lr.diagnostics.iter().any(Diagnostic::is_error));
}

#[test]
fn unknown_distribution_is_error() {
    let lr = lower_with_diags("let x ~ Banana(1, 2)");
    assert!(lr.diagnostics.iter().any(Diagnostic::is_error));
}

#[test]
fn wrong_arity_is_error() {
    let lr = lower_with_diags("let x ~ Normal(0, 1, 2)");
    assert!(lr.diagnostics.iter().any(Diagnostic::is_error));
}

#[test]
fn distribution_used_as_value_is_error() {
    let lr = lower_with_diags("let x = Normal(0, 1)");
    assert!(lr.diagnostics.iter().any(Diagnostic::is_error));
}

#[test]
fn degenerate_literal_is_warning() {
    let lr = lower_with_diags("let s ~ Normal(0, -1)");
    assert!(lr
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Warning));
}

#[test]
fn degenerate_uniform_warning() {
    let lr = lower_with_diags("let u ~ Uniform(3, 1)");
    assert!(lr
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Warning));
}

#[test]
fn degenerate_bernoulli_warning() {
    let lr = lower_with_diags("let b ~ Bernoulli(1.5)");
    assert!(lr
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Warning));
}

#[test]
fn known_dist_arity_table() {
    assert_eq!(known_dist_arity("Normal"), Some(2));
    assert_eq!(known_dist_arity("Beta"), Some(2));
    assert_eq!(known_dist_arity("Gamma"), Some(2));
    assert_eq!(known_dist_arity("Uniform"), Some(2));
    assert_eq!(known_dist_arity("Binomial"), Some(2));
    assert_eq!(known_dist_arity("HalfNormal"), Some(1));
    assert_eq!(known_dist_arity("Exponential"), Some(1));
    assert_eq!(known_dist_arity("Poisson"), Some(1));
    assert_eq!(known_dist_arity("Bernoulli"), Some(1));
    assert_eq!(known_dist_arity("NotADist"), None);
}

#[test]
fn eval_arithmetic_and_precedence() {
    use augur_frontend::ast::Stmt;
    let r = parse("let x = 1 + 2 * 3 - 4 / 2");
    assert!(!r.has_errors());
    let value = match &r.program.statements[0] {
        Stmt::Let { value, .. } => {
            let env = Env::new();
            eval(value, &env)
        }
        _ => panic!("expected let"),
    };
    // 1 + (2*3) - (4/2) = 1 + 6 - 2 = 5
    assert!((value - 5.0).abs() < 1e-12, "value={value}");
}

#[test]
fn eval_undeclared_returns_nan() {
    let r = parse("let x = y");
    let value = match &r.program.statements[0] {
        augur_frontend::ast::Stmt::Let { value, .. } => eval(value, &Env::new()),
        _ => panic!("expected let"),
    };
    assert!(value.is_nan());
}

#[test]
fn eval_comparison_is_one_or_zero() {
    let r = parse("let t = 3 > 2\nlet f = 1 > 2");
    let v_t = match &r.program.statements[0] {
        augur_frontend::ast::Stmt::Let { value, .. } => eval(value, &Env::new()),
        _ => panic!(),
    };
    let v_f = match &r.program.statements[1] {
        augur_frontend::ast::Stmt::Let { value, .. } => eval(value, &Env::new()),
        _ => panic!(),
    };
    assert_eq!(v_t, 1.0);
    assert_eq!(v_f, 0.0);
}

#[test]
fn instantiate_dist_known_families() {
    use augur_frontend::ast::{Expr, Stmt};
    use augur_std::Dist;

    let r = parse("let a ~ Normal(0, 1)\nlet b ~ Beta(2, 3)\nlet c ~ Gamma(1, 2)\nlet d ~ Uniform(0, 1)\nlet e ~ Exponential(2)\nlet f ~ HalfNormal(1)\nlet g ~ Binomial(10, 0.3)\nlet h ~ Poisson(4)\nlet i ~ Bernoulli(0.5)");
    let env = Env::new();
    let items: Vec<&Stmt> = r.program.statements.iter().collect();
    let prior_exprs: Vec<&Expr> = items
        .iter()
        .filter_map(|s| match s {
            Stmt::Prior { dist, .. } => Some(dist),
            _ => None,
        })
        .collect();
    let expected = [
        Dist::Normal {
            mu: 0.0,
            sigma: 1.0,
        },
        Dist::Beta { a: 2.0, b: 3.0 },
        Dist::Gamma {
            shape: 1.0,
            rate: 2.0,
        },
        Dist::Uniform { lo: 0.0, hi: 1.0 },
        Dist::Exponential { rate: 2.0 },
        Dist::HalfNormal { sigma: 1.0 },
        Dist::Binomial { n: 10.0, p: 0.3 },
        Dist::Poisson { rate: 4.0 },
        Dist::Bernoulli { p: 0.5 },
    ];
    for (expr, exp) in prior_exprs.iter().zip(expected.iter()) {
        let got = instantiate_dist(expr, &env).expect("should instantiate");
        assert_eq!(got, *exp);
    }
}

#[test]
fn instantiate_unknown_dist_is_none() {
    let mut env = Env::new();
    env.insert("q".to_string(), 1.0);
    let unknown = augur_frontend::ast::Expr::Call {
        name: "Nope".to_string(),
        args: vec![],
    };
    assert!(instantiate_dist(&unknown, &env).is_none());
}

#[test]
fn log_joint_normal_normal_matches_hand_computation() {
    let m = lower_ok("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    let mut env = Env::new();
    let lp = log_joint(&m, &[0.25], &mut env);
    let prior = augur_std::Dist::Normal {
        mu: 0.0,
        sigma: 1.0,
    }
    .logp(0.25);
    let like = augur_std::Dist::Normal {
        mu: 0.25,
        sigma: 1.0,
    }
    .logp(0.5);
    assert!((lp - (prior + like)).abs() < 1e-12, "lp={lp}");
}

#[test]
fn log_joint_beta_binomial_matches_hand_computation() {
    let m = lower_ok("let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7");
    let mut env = Env::new();
    let lp = log_joint(&m, &[0.7], &mut env);
    let prior = augur_std::Dist::Beta { a: 1.0, b: 1.0 }.logp(0.7);
    let like = augur_std::Dist::Binomial { n: 10.0, p: 0.7 }.logp(7.0);
    assert!((lp - (prior + like)).abs() < 1e-12, "lp={lp}");
}

#[test]
fn log_joint_maximized_at_posterior_mode() {
    let m = lower_ok("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    let mut env = Env::new();
    let lp_mode = log_joint(&m, &[0.25], &mut env);
    let lp_off = log_joint(&m, &[0.0], &mut env);
    // The true posterior mode is at 0.25; any other point has lower log-joint.
    assert!(lp_mode > lp_off, "mode={lp_mode} off={lp_off}");
}

#[test]
fn log_joint_sum_equals_prior_plus_likelihood_when_deterministic() {
    let m = lower_ok("let mu ~ Normal(0, 1)\nlet sigma = 1\nobserve Normal(mu, sigma) = 0.5");
    let mut env = Env::new();
    let lp = log_joint(&m, &[0.5], &mut env);
    let prior = augur_std::Dist::Normal {
        mu: 0.0,
        sigma: 1.0,
    }
    .logp(0.5);
    let like = augur_std::Dist::Normal {
        mu: 0.5,
        sigma: 1.0,
    }
    .logp(0.5);
    assert!((lp - (prior + like)).abs() < 1e-12, "lp={lp}");
}

#[test]
fn if_gates_observe() {
    let src = "let mu ~ Normal(0,1)\nif mu > 0 { observe Normal(mu,1) = 1 } else { observe Normal(mu,1) = -1 }";
    let m = lower_ok(src);
    let mut env = Env::new();
    let lp_pos = log_joint(&m, &[0.6], &mut env);
    let lp_neg = log_joint(&m, &[-0.6], &mut env);
    let like_pos = augur_std::Dist::Normal {
        mu: 0.6,
        sigma: 1.0,
    }
    .logp(1.0);
    let like_neg = augur_std::Dist::Normal {
        mu: -0.6,
        sigma: 1.0,
    }
    .logp(-1.0);
    let prior_pos = augur_std::Dist::Normal {
        mu: 0.0,
        sigma: 1.0,
    }
    .logp(0.6);
    let prior_neg = augur_std::Dist::Normal {
        mu: 0.0,
        sigma: 1.0,
    }
    .logp(-0.6);
    assert!((lp_pos - (like_pos + prior_pos)).abs() < 1e-12);
    assert!((lp_neg - (like_neg + prior_neg)).abs() < 1e-12);
}
