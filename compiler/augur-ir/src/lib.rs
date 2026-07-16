//! Augur typed IR: lowering the frontend AST into a runnable [`Model`], with
//! type-checking, static analysis of degenerate parameters, and the
//! uncertainty-propagation evaluation used by the inference engines.

#![warn(missing_docs)]

pub mod lower;

pub use augur_frontend::Expr;
pub use lower::{
    eval, instantiate_dist, log_joint, lower, Diagnostic, Env, LowerResult, Model, ModelItem,
    Severity,
};

#[cfg(test)]
mod tests {
    use super::*;
    use augur_frontend::parse;

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

    #[test]
    fn normal_normal_conjugate_shape() {
        let m = lower_ok("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        assert_eq!(m.prior_order, vec!["mu".to_string()]);
    }

    #[test]
    fn beta_binomial_conjugate_shape() {
        let m = lower_ok("let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7");
        assert_eq!(m.prior_order, vec!["p".to_string()]);
    }

    #[test]
    fn undeclared_variable_is_error() {
        let r = parse("let x = y + 1");
        let lr = lower(&r.program);
        assert!(lr.diagnostics.iter().any(Diagnostic::is_error));
    }

    #[test]
    fn degenerate_literal_warned() {
        let r = parse("let s ~ Normal(0, -1)");
        let lr = lower(&r.program);
        assert!(lr.diagnostics.iter().any(|d| !d.is_error()));
    }

    #[test]
    fn log_joint_sums_prior_and_likelihood() {
        let m = lower_ok("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        let mut env = Env::new();
        // At mu = 0.5: prior logp = N(0.5|0,1), likelihood = N(0.5|0.5,1)
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
        // Positive branch observes +1, negative branch observes -1.
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
        assert!(
            (lp_pos
                - like_pos
                - augur_std::Dist::Normal {
                    mu: 0.0,
                    sigma: 1.0
                }
                .logp(0.6))
            .abs()
                < 1e-12
        );
        assert!(
            (lp_neg
                - like_neg
                - augur_std::Dist::Normal {
                    mu: 0.0,
                    sigma: 1.0
                }
                .logp(-0.6))
            .abs()
                < 1e-12
        );
    }
}
