//! Integration tests for the Augur inference runtime: every engine against
//! known closed-form posteriors, engine selection, transforms, trace
//! summaries, and the deterministic fallback.

use augur_frontend::parse;
use augur_ir::lower;
use augur_runtime::{
    common,
    engine::{Engine, InferOptions},
    hmc, mh, pf, select_engine, summarize, vi, InferenceResult,
};

fn model(src: &str) -> augur_ir::Model {
    let r = parse(src);
    assert!(!r.has_errors(), "parse: {:?}", r.diagnostics);
    let lr = lower(&r.program);
    assert!(
        !lr.diagnostics.iter().any(|d| d.is_error()),
        "type: {:?}",
        lr.diagnostics
    );
    lr.model
}

fn infer(src: &str, mut opts: InferOptions) -> InferenceResult {
    opts.num_chains = 2;
    opts.num_warmup = 300;
    opts.num_samples = 1200;
    run(&model(src), &opts)
}

fn run(m: &augur_ir::Model, opts: &InferOptions) -> InferenceResult {
    let engine = opts.engine.unwrap_or_else(|| select_engine(m));
    let trace = match engine {
        Engine::MetropolisHastings => mh::run_all(m, opts),
        Engine::Hmc => hmc::run_all(m, opts),
        Engine::Variational => vi::run_all(m, opts),
        Engine::ParticleFilter => pf::run_all(m, opts),
    };
    summarize(&trace, &m.prior_order, engine)
}

#[test]
fn normal_normal_conjugate_hmc() {
    let res = infer(
        "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5",
        InferOptions {
            engine: Some(Engine::Hmc),
            ..Default::default()
        },
    );
    let mean = res.mean_of("mu").unwrap();
    assert!((mean - 0.25).abs() < 0.05, "mean={mean}");
    let sd = res.summaries[0].sd;
    assert!((sd - 0.5f64.sqrt()).abs() < 0.05, "sd={sd}");
}

#[test]
fn normal_normal_conjugate_mh() {
    let res = infer(
        "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5",
        InferOptions {
            engine: Some(Engine::MetropolisHastings),
            ..Default::default()
        },
    );
    let mean = res.mean_of("mu").unwrap();
    assert!((mean - 0.25).abs() < 0.06, "mean={mean}");
}

#[test]
fn normal_normal_conjugate_vi() {
    let res = infer(
        "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5",
        InferOptions {
            engine: Some(Engine::Variational),
            ..Default::default()
        },
    );
    let mean = res.mean_of("mu").unwrap();
    assert!((mean - 0.25).abs() < 0.05, "mean={mean}");
}

#[test]
fn beta_binomial_conjugate_pf() {
    let res = infer(
        "let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7",
        InferOptions {
            engine: Some(Engine::ParticleFilter),
            ..Default::default()
        },
    );
    let mean = res.mean_of("p").unwrap();
    assert!((mean - 2.0 / 3.0).abs() < 0.05, "mean={mean}");
}

#[test]
fn beta_binomial_conjugate_hmc() {
    let res = infer(
        "let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7",
        InferOptions {
            engine: Some(Engine::Hmc),
            ..Default::default()
        },
    );
    let mean = res.mean_of("p").unwrap();
    assert!((mean - 2.0 / 3.0).abs() < 0.05, "mean={mean}");
}

#[test]
fn beta_binomial_conjugate_mh() {
    let res = infer(
        "let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7",
        InferOptions {
            engine: Some(Engine::MetropolisHastings),
            ..Default::default()
        },
    );
    let mean = res.mean_of("p").unwrap();
    assert!((mean - 2.0 / 3.0).abs() < 0.06, "mean={mean}");
}

#[test]
fn beta_binomial_vi_sanity() {
    let res = infer(
        "let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7",
        InferOptions {
            engine: Some(Engine::Variational),
            ..Default::default()
        },
    );
    let mean = res.mean_of("p").unwrap();
    assert!((mean - 2.0 / 3.0).abs() < 0.1, "mean={mean}");
    assert!(
        mean > 0.5,
        "VI should move toward the likelihood (mean={mean})"
    );
}

#[test]
fn gamma_observation_conjugate_hmc() {
    // posterior of lambda | Exp(lambda) obs 2, prior Gamma(2,2):
    // prior contributor (2-1)ln λ - 2λ, likelihood ln λ - 2λ
    // => Gamma(shape=3, rate=4), mean 3/4 = 0.75
    let res = infer(
        "let lambda ~ Gamma(2, 2)\nobserve Exponential(lambda) = 2",
        InferOptions {
            engine: Some(Engine::Hmc),
            ..Default::default()
        },
    );
    let mean = res.mean_of("lambda").unwrap();
    assert!((mean - 0.75).abs() < 0.1, "mean={mean}");
}

#[test]
fn auto_engine_selects_hmc_for_continuous() {
    let m = model("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    assert_eq!(select_engine(&m), Engine::Hmc);
}

#[test]
fn auto_engine_selects_pf_for_discrete() {
    let m = model("let k ~ Binomial(10, 0.5)\nobserve Normal(k, 1) = 6");
    assert_eq!(select_engine(&m), Engine::ParticleFilter);
}

#[test]
fn auto_engine_selects_vi_for_high_dimensional() {
    let priors: String = (0..16)
        .map(|i| format!("let x{i} ~ Normal(0, 1)"))
        .collect::<Vec<_>>()
        .join("\n");
    let m = model(&priors);
    assert_eq!(select_engine(&m), Engine::Variational);
}

#[test]
fn engine_from_str_and_as_str() {
    use std::str::FromStr;
    assert_eq!(Engine::from_str("hmc").unwrap(), Engine::Hmc);
    assert_eq!(Engine::from_str("mh").unwrap(), Engine::MetropolisHastings);
    assert_eq!(
        Engine::from_str("metropolis-hastings").unwrap(),
        Engine::MetropolisHastings
    );
    assert_eq!(Engine::from_str("vi").unwrap(), Engine::Variational);
    assert_eq!(Engine::from_str("pf").unwrap(), Engine::ParticleFilter);
    assert!(Engine::from_str("nonsense").is_err());
    assert_eq!(Engine::Hmc.as_str(), "hmc");
    assert_eq!(Engine::Variational.as_str(), "vi");
    assert_eq!(Engine::MetropolisHastings.as_str(), "mh");
    assert_eq!(Engine::ParticleFilter.as_str(), "pf");
}

#[test]
fn infer_options_defaults() {
    let o = InferOptions::default();
    assert!(o.engine.is_none());
    assert_eq!(o.num_chains, 4);
    assert_eq!(o.num_warmup, 1000);
    assert_eq!(o.num_samples, 2000);
    assert_eq!(o.seed, 0xC0FFEE);
}

#[test]
fn all_engines_produce_requested_chain_counts() {
    let m = model("let mu ~ Normal(0, 1)");
    let opts = InferOptions {
        num_chains: 3,
        num_warmup: 50,
        num_samples: 40,
        ..Default::default()
    };
    for engine in [
        Engine::Hmc,
        Engine::MetropolisHastings,
        Engine::Variational,
        Engine::ParticleFilter,
    ] {
        let trace = match engine {
            Engine::Hmc => hmc::run_all(&m, &opts),
            Engine::MetropolisHastings => mh::run_all(&m, &opts),
            Engine::Variational => vi::run_all(&m, &opts),
            Engine::ParticleFilter => pf::run_all(&m, &opts),
        };
        assert!(!trace.chains.is_empty(), "{engine:?} produced no chains");
        let total: usize = trace.chains.iter().map(|c| c.len()).sum();
        match engine {
            Engine::ParticleFilter => {
                // PF spreads `num_particles` samples across the requested chains.
                assert_eq!(total, opts.num_particles, "{engine:?} total mismatch");
                assert_eq!(
                    trace.chains.len(),
                    opts.num_chains,
                    "{engine:?} chain count"
                );
            }
            Engine::Variational => {
                // VI packs every draw into a single chain.
                assert_eq!(trace.chains.len(), 1);
                assert_eq!(total, opts.num_samples * opts.num_chains);
            }
            _ => {
                assert_eq!(
                    trace.chains.len(),
                    opts.num_chains,
                    "{engine:?} chain count"
                );
                for c in &trace.chains {
                    assert_eq!(c.len(), opts.num_samples, "{engine:?} chain length");
                }
                assert_eq!(total, opts.num_chains * opts.num_samples);
            }
        }
    }
}

#[test]
fn transforms_roundtrip() {
    use augur_frontend::ast::Expr;
    use augur_runtime::transforms::{transform_for, Transform};

    // Identity
    let (theta, logjac, dlj) = Transform::Identity.forward(2.0);
    assert_eq!(theta, 2.0);
    assert_eq!(logjac, 0.0);
    assert_eq!(dlj, 0.0);
    assert_eq!(Transform::Identity.inverse(2.0), 2.0);
    assert_eq!(Transform::Identity.jacobian(2.0), 1.0);

    // Log
    let (theta, logjac, _) = Transform::Log.forward(0.0);
    assert!((theta - 1.0).abs() < 1e-12);
    assert!((logjac - 0.0).abs() < 1e-12);
    assert_eq!(Transform::Log.inverse(1.0), 0.0);
    assert!((Transform::Log.jacobian(0.0) - 1.0).abs() < 1e-12);

    // Logit
    let t = Transform::Logit { lo: 0.0, hi: 1.0 };
    let (theta, _, _) = t.forward(0.0);
    assert!((theta - 0.5).abs() < 1e-12);
    let back = t.inverse(theta);
    assert!((back - 0.0).abs() < 1e-9);

    // transform_for picks correct families
    let halfnormal = Expr::Call {
        name: "HalfNormal".into(),
        args: vec![],
    };
    assert!(matches!(transform_for(&halfnormal), Transform::Log));
    let beta = Expr::Call {
        name: "Beta".into(),
        args: vec![],
    };
    assert!(matches!(transform_for(&beta), Transform::Logit { .. }));
    let normal = Expr::Call {
        name: "Normal".into(),
        args: vec![],
    };
    assert!(matches!(transform_for(&normal), Transform::Identity));
}

#[test]
fn grad_log_joint_matches_analytic() {
    let m = model("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    // logp(mu) = -0.5 mu^2 - 0.5 (mu - 0.5)^2 + const
    // d/dmu = -mu - (mu - 0.5) = -2 mu + 0.5
    let g0 = common::grad_log_joint(&m, &[0.0], 1e-4);
    assert!((g0[0] - 0.5).abs() < 0.05, "g0={}", g0[0]);
    let g_mode = common::grad_log_joint(&m, &[0.25], 1e-4);
    assert!(g_mode[0].abs() < 0.05, "g_mode={}", g_mode[0]);
}

#[test]
fn deterministic_fallback_evaluates_without_sampling() {
    let m = model("let a = 2\nlet b = a * 3\nif a > 0 { let c = b + 1 }");
    let env = common::eval_deterministic(&m);
    assert_eq!(env.get("a").copied(), Some(2.0));
    assert_eq!(env.get("b").copied(), Some(6.0));
    assert_eq!(env.get("c").copied(), Some(7.0));
}

#[test]
fn initial_point_uses_typical_points() {
    let m = model("let p ~ Beta(1, 1)\nlet mu ~ Normal(0, 1)");
    let pt = common::initial_point(&m, 0, 0xC0FFEE);
    assert_eq!(pt.len(), 2);
    assert!(
        pt[0] > 0.0 && pt[0] < 1.0,
        "beta init in support: {}",
        pt[0]
    );
}

#[test]
fn trace_summary_quantile_ordering_and_rhat() {
    let res = infer(
        "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5",
        InferOptions {
            engine: Some(Engine::Hmc),
            ..Default::default()
        },
    );
    let s = &res.summaries[0];
    assert!(s.q2_5 <= s.q50, "q2_5={} q50={}", s.q2_5, s.q50);
    assert!(s.q50 <= s.q97_5, "q50={} q97_5={}", s.q50, s.q97_5);
    assert!(s.rhat.is_finite());
    assert!(
        s.rhat < 1.2,
        "rhat should be near 1 for a healthy chain: {}",
        s.rhat
    );
    assert!(s.ess > 0.0);
    assert_eq!(res.num_chains, 2);
    assert_eq!(res.num_samples, 1200);
    assert_eq!(res.param_names, vec!["mu".to_string()]);
}
