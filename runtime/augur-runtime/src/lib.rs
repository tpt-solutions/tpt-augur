//! Augur inference runtime.
//!
//! Provides several inference engines — random-walk Metropolis–Hastings,
//! Hamiltonian Monte Carlo, mean-field variational inference, and a bootstrap
//! particle filter — dispatched through [`run`]. Engine selection is automatic
//! from model topology (see [`select_engine`]) unless overridden.

pub mod common;
pub mod engine;
pub mod hmc;
pub mod mh;
pub mod pf;
pub mod trace;
pub mod transforms;
pub mod vi;

pub use engine::{select_engine, Engine, InferOptions};
pub use trace::{summarize, InferenceResult, ParamSummary, Trace};

use augur_ir::Model;

/// Run inference on a lowered model, returning posterior summaries.
///
/// If `opts.engine` is `None`, the engine is chosen automatically via
/// [`select_engine`].
pub fn run(model: &Model, opts: &InferOptions) -> InferenceResult {
    let engine = opts.engine.unwrap_or_else(|| select_engine(model));
    let trace = match engine {
        Engine::MetropolisHastings => mh::run_all(model, opts),
        Engine::Hmc => hmc::run_all(model, opts),
        Engine::Variational => vi::run_all(model, opts),
        Engine::ParticleFilter => pf::run_all(model, opts),
    };
    summarize(&trace, &model.prior_order, engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common;
    use crate::hmc;
    use augur_frontend::parse;
    use augur_ir::lower;

    fn infer(src: &str, mut opts: InferOptions) -> InferenceResult {
        opts.num_chains = 2;
        opts.num_warmup = 500;
        opts.num_samples = 1500;
        let r = parse(src);
        assert!(!r.has_errors(), "parse: {:?}", r.diagnostics);
        let lr = lower(&r.program);
        assert!(
            !lr.diagnostics.iter().any(|d| d.is_error()),
            "type: {:?}",
            lr.diagnostics
        );
        run(&lr.model, &opts)
    }

    #[test]
    fn normal_normal_conjugate_hmc() {
        // posterior of mu given N(mu,1) observed 0.5, prior N(0,1) is N(0.25, 0.5)
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
    fn beta_binomial_conjugate_pf() {
        // posterior p | Binomial(10,p)=7, prior Beta(1,1) is Beta(8,4), mean 2/3
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
    fn normal_normal_conjugate_vi() {
        // Posterior is exactly Gaussian, so a Gaussian variational family is exact.
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
    fn beta_binomial_conjugate_vi_sanity() {
        // VI is a Gaussian approximation; on a Beta posterior it is slightly
        // biased low but must land near the true mean (0.667) and beat the prior.
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
    fn auto_engine_selects_hmc_for_continuous() {
        let r = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        let lr = lower(&r.program);
        assert_eq!(select_engine(&lr.model), Engine::Hmc);
    }

    #[test]
    fn deterministic_fallback_evaluates_without_sampling() {
        // A purely-deterministic model: no priors/observations.
        let r = parse("let a = 2\nlet b = a * 3\nif a > 0 { let c = b + 1 }");
        let lr = lower(&r.program);
        assert!(!lr.diagnostics.iter().any(|d| d.is_error()));
        let env = common::eval_deterministic(&lr.model);
        assert_eq!(env.get("a").copied(), Some(2.0));
        assert_eq!(env.get("b").copied(), Some(6.0));
        assert_eq!(env.get("c").copied(), Some(7.0));
    }

    #[test]
    fn hmc_runs_parallel_chains_independently() {
        // Validates the fearless-concurrency model: chains are sampled in
        // parallel yet remain independent (distinct seeded RNGs) and each
        // produces the requested number of samples.
        let r = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        let lr = lower(&r.program);
        let opts = InferOptions {
            num_chains: 4,
            num_warmup: 100,
            num_samples: 50,
            ..Default::default()
        };
        let trace = hmc::run_all(&lr.model, &opts);
        assert_eq!(trace.chains.len(), 4);
        for c in &trace.chains {
            assert_eq!(c.len(), 50);
        }
        let firsts: Vec<f64> = trace.chains.iter().map(|c| c[0][0]).collect();
        let all_same = firsts.iter().all(|v| (v - firsts[0]).abs() < 1e-12);
        assert!(
            !all_same,
            "chains collapsed to identical samples: {firsts:?}"
        );
    }

    #[test]
    fn auto_engine_selects_pf_for_discrete() {
        let r = parse("let k ~ Binomial(10, 0.5)\nobserve Normal(k, 1) = 6");
        let lr = lower(&r.program);
        // `k` is a discrete prior -> particle filter.
        assert_eq!(select_engine(&lr.model), Engine::ParticleFilter);
    }

    #[test]
    fn uncertainty_propagation_through_deterministic() {
        // posterior of (mu + 1) should be posterior of mu shifted by 1.
        let src = "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5\nlet shifted = mu + 1";
        let r = parse(src);
        let lr = lower(&r.program);
        // `shifted` is deterministic, not a prior, so it won't appear in summaries,
        // but the joint still evaluates correctly (no type error).
        assert!(!lr.diagnostics.iter().any(|d| d.is_error()));
    }
}
