//! Augur ↔ TPT Locus bridge.
//!
//! Per Locus spec.txt §4, Locus defers complex decisions to Augur to compute a
//! *probability of success* for candidate agent strategies before executing
//! them. This crate is the scoring API surface Locus calls into:
//!
//! * [`Strategy`] — a candidate Locus agent strategy described by a feature
//!   vector (e.g. capability match, historical reliability, cost budget).
//! * [`ProbabilityOfSuccess`] — Augur's posterior-predictive P(success) with a
//!   credible interval.
//! * [`LocusAugurBridge`] / [`evaluate_strategy`] — the stable model-evaluation
//!   entry point mapping a strategy to a probability.
//!
//! The probabilistic model is a Beta prior over the success rate `p` with a
//! `Bernoulli(p)` predictive for `success`; Locus strategy features are mapped
//! to the Beta pseudo-counts, so Augur returns both a point estimate and
//! uncertainty quantified by the posterior.

use augur_frontend::parse;
use augur_ir::lower;
use augur_runtime::{InferOptions, InferenceResult};

/// A candidate Locus agent strategy to be scored.
#[derive(Debug, Clone, PartialEq)]
pub struct Strategy {
    /// Stable strategy identifier (e.g. an [`locus_core::agent::AgentId`]).
    pub id: String,
    /// Human-readable label for tooling/telemetry.
    pub label: String,
    /// Numeric features the bridge maps into the success-rate prior.
    pub features: Vec<f64>,
}

/// Augur's posterior-predictive probability that a strategy succeeds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityOfSuccess {
    /// Posterior mean of `success` (in `[0, 1]`).
    pub value: f64,
    /// Lower bound of the 95% credible interval.
    pub ci_low: f64,
    /// Upper bound of the 95% credible interval.
    pub ci_high: f64,
}

/// A strategy paired with its scored probability, for ranking.
#[derive(Debug, Clone)]
pub struct RankedStrategy {
    pub strategy: Strategy,
    pub probability: ProbabilityOfSuccess,
}

/// Errors raised while evaluating a strategy through Augur.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("frontend parse error: {0}")]
    Parse(String),
    #[error("type/IR error: {0}")]
    Ir(String),
    #[error("inference produced no summary for `success`")]
    MissingSuccess,
}

/// Logistic sigmoid, used to map a strategy's feature dot-product to a success
/// rate in `(0, 1)` before it becomes a Beta prior.
pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Map a strategy feature vector to a Beta prior `(alpha, beta)` over the
/// success rate, concentrated by `concentration` pseudo-counts so the resulting
/// posterior carries a meaningful credible interval. A feature with positive
/// weight raises `alpha` (more evidence of success); negative weight raises
/// `beta`.
pub fn features_to_beta_prior(features: &[f64], concentration: f64) -> (f64, f64) {
    let score: f64 = features.iter().sum();
    let p = logistic(score).clamp(1e-3, 1.0 - 1e-3);
    let alpha = p * concentration + 1.0;
    let beta = (1.0 - p) * concentration + 1.0;
    (alpha, beta)
}

/// The canonical P(success) generative model: a Beta prior over the success
/// rate `p` with a Bernoulli predictive for `success`.
pub fn success_model_source(alpha: f64, beta: f64) -> String {
    format!("let p ~ Beta({alpha}, {beta})\nlet success ~ Bernoulli(p)")
}

/// Run inference on an already-lowered model and extract the posterior of the
/// `success` variable as a [`ProbabilityOfSuccess`].
fn summarize_success(
    model: &augur_ir::Model,
    opts: &InferOptions,
) -> Result<ProbabilityOfSuccess, BridgeError> {
    let result: InferenceResult = augur_runtime::run(model, opts);
    let summary = result
        .summaries
        .iter()
        .find(|s| s.name == "success")
        .ok_or(BridgeError::MissingSuccess)?;
    Ok(ProbabilityOfSuccess {
        value: summary.mean.clamp(0.0, 1.0),
        ci_low: summary.q2_5.clamp(0.0, 1.0),
        ci_high: summary.q97_5.clamp(0.0, 1.0),
    })
}

fn default_opts() -> InferOptions {
    InferOptions {
        num_chains: 2,
        num_warmup: 200,
        num_samples: 800,
        ..Default::default()
    }
}

/// The stable Augur model-evaluation entry point: a Locus [`Strategy`] in,
/// a [`ProbabilityOfSuccess`] out. Builds the canonical success model from the
/// strategy's features and runs posterior-predictive inference.
pub fn evaluate_strategy(
    strategy: &Strategy,
    concentration: f64,
) -> Result<ProbabilityOfSuccess, BridgeError> {
    let (alpha, beta) = features_to_beta_prior(&strategy.features, concentration);
    let src = success_model_source(alpha, beta);
    let parsed = parse(&src);
    if parsed.has_errors() {
        return Err(BridgeError::Parse(format!("{:?}", parsed.diagnostics)));
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        return Err(BridgeError::Ir(format!("{:?}", lowered.diagnostics)));
    }
    summarize_success(&lowered.model, &default_opts())
}

/// Score an arbitrary Augur model source for a strategy. The strategy features
/// are prepended as deterministic `let featureN = <value>` bindings so the
/// model can reference them (e.g. `let p ~ Beta(1 + 5*feature0, 1)`).
pub fn evaluate_source(
    src: &str,
    strategy: &Strategy,
) -> Result<ProbabilityOfSuccess, BridgeError> {
    let bindings: String = strategy
        .features
        .iter()
        .enumerate()
        .map(|(i, f)| format!("let feature{i} = {f}\n"))
        .collect();
    let full = format!("{bindings}{src}");
    let parsed = parse(&full);
    if parsed.has_errors() {
        return Err(BridgeError::Parse(format!("{:?}", parsed.diagnostics)));
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        return Err(BridgeError::Ir(format!("{:?}", lowered.diagnostics)));
    }
    summarize_success(&lowered.model, &default_opts())
}

/// The Locus-facing scoring service. Wraps the stable entry point and adds
/// batch ranking of candidate strategies.
pub struct LocusAugurBridge {
    concentration: f64,
}

impl LocusAugurBridge {
    /// Create a bridge with the given Beta concentration (pseudo-counts).
    pub fn new(concentration: f64) -> Self {
        Self { concentration }
    }

    /// Score a single strategy via the canonical success model.
    pub fn score(&self, strategy: &Strategy) -> Result<ProbabilityOfSuccess, BridgeError> {
        evaluate_strategy(strategy, self.concentration)
    }

    /// Score every strategy and return them sorted by descending probability.
    pub fn rank(&self, strategies: &[Strategy]) -> Result<Vec<RankedStrategy>, BridgeError> {
        let mut ranked: Vec<RankedStrategy> = Vec::with_capacity(strategies.len());
        for s in strategies {
            ranked.push(RankedStrategy {
                strategy: s.clone(),
                probability: self.score(s)?,
            });
        }
        ranked.sort_by(|a, b| {
            b.probability
                .value
                .partial_cmp(&a.probability.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use locus_core::agent::{AgentInfo, AgentRegistry, AgentState};
    use std::collections::HashSet;

    #[test]
    fn logistic_bounds() {
        assert!(logistic(0.0) > 0.49 && logistic(0.0) < 0.51);
        assert!(logistic(20.0) > 0.99);
        assert!(logistic(-20.0) < 0.01);
    }

    #[test]
    fn positive_features_raise_success_prior() {
        let (a, b) = features_to_beta_prior(&[2.0, 1.0], 20.0);
        assert!(a > b, "positive features should favor success");
    }

    #[test]
    fn evaluate_strategy_returns_probability_in_unit_interval() {
        let s = Strategy {
            id: "s1".into(),
            label: "aggressive".into(),
            features: vec![1.5, 0.5],
        };
        let p = evaluate_strategy(&s, 20.0).unwrap();
        assert!((0.0..=1.0).contains(&p.value));
        assert!(p.ci_low <= p.value && p.value <= p.ci_high);
    }

    #[test]
    fn higher_feature_mass_ranks_higher() {
        let bridge = LocusAugurBridge::new(20.0);
        let weak = Strategy {
            id: "weak".into(),
            label: "weak".into(),
            features: vec![-2.0],
        };
        let strong = Strategy {
            id: "strong".into(),
            label: "strong".into(),
            features: vec![2.0],
        };
        let ranked = bridge.rank(&[weak.clone(), strong.clone()]).unwrap();
        assert_eq!(ranked[0].strategy.id, "strong");
        assert!(ranked[0].probability.value > ranked[1].probability.value);
    }

    #[test]
    fn evaluate_source_injects_features() {
        // Model references feature0 explicitly; bridge injects it as a binding.
        let src = "let p ~ Beta(1 + 4*feature0, 2)\nlet success ~ Bernoulli(p)";
        let s = Strategy {
            id: "x".into(),
            label: "x".into(),
            features: vec![0.5],
        };
        let p = evaluate_source(src, &s).unwrap();
        assert!((0.0..=1.0).contains(&p.value));
    }

    #[test]
    fn integration_with_locus_agent_registry() {
        // Wire the bridge to a real Locus agent registry: score strategies for
        // registered agents and confirm the output is a valid probability.
        let mut reg = AgentRegistry::new();
        let planner = reg.register(AgentInfo {
            name: "planner".into(),
            capabilities: HashSet::from(["plan".into()]),
            retry_policy: Default::default(),
        });
        reg.transition(planner, AgentState::Thinking).unwrap();
        assert!(reg.can(planner, "plan"));

        let bridge = LocusAugurBridge::new(20.0);
        let strategies = vec![
            Strategy {
                id: planner.to_string(),
                label: "reliable".into(),
                features: vec![1.0, 0.8],
            },
            Strategy {
                id: planner.to_string(),
                label: "risky".into(),
                features: vec![-1.0, 0.2],
            },
        ];
        let ranked = bridge.rank(&strategies).unwrap();
        for r in &ranked {
            assert!((0.0..=1.0).contains(&r.probability.value));
        }
        assert!(ranked[0].probability.value >= ranked[1].probability.value);
    }
}
