//! Posterior traces and summary statistics.

use crate::engine::Engine;

/// Per-chain samples. `chains[c][s][p]` is the value of prior variable `p`
/// in sample `s` of chain `c`.
#[derive(Debug, Clone)]
pub struct Trace {
    pub chains: Vec<Vec<Vec<f64>>>,
}

impl Trace {
    /// Flatten all chains and samples into one stream per parameter.
    pub fn flatten(&self) -> Vec<Vec<f64>> {
        if self.chains.is_empty() {
            return Vec::new();
        }
        let n_params = self.chains[0].first().map(|s| s.len()).unwrap_or(0);
        let mut out = vec![Vec::new(); n_params];
        for chain in &self.chains {
            for sample in chain {
                for (p, v) in sample.iter().enumerate() {
                    out[p].push(*v);
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct ParamSummary {
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    pub q2_5: f64,
    pub q50: f64,
    pub q97_5: f64,
    /// Gelman–Rubin convergence diagnostic (1.0 == converged).
    pub rhat: f64,
    pub ess: f64,
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub engine: Engine,
    pub param_names: Vec<String>,
    pub summaries: Vec<ParamSummary>,
    pub num_samples: usize,
    pub num_chains: usize,
}

impl InferenceResult {
    pub fn mean_of(&self, name: &str) -> Option<f64> {
        self.summaries
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.mean)
    }
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = (sorted.len() as f64 - 1.0) * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn rhat_and_ess(chains: &[Vec<f64>]) -> (f64, f64) {
    let m = chains.len();
    let n = chains.first().map(|c| c.len()).unwrap_or(0);
    if m == 0 || n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let grand = mean(&means);
    let between = {
        let s: f64 = means.iter().map(|m| (m - grand).powi(2)).sum();
        n as f64 * s / (m as f64 - 1.0)
    };
    let within = {
        let mut s = 0.0;
        for c in chains {
            let cm = mean(c);
            s += c.iter().map(|x| (x - cm).powi(2)).sum::<f64>();
        }
        s / (m as f64 * (n as f64 - 1.0))
    };
    let var_plus = ((n as f64 - 1.0) / n as f64) * within + between / n as f64;
    let rhat = (var_plus / within.max(1e-12)).sqrt();
    // Crude effective-sample-size estimate from the ratio.
    let ess = (m as f64 * n as f64) / (rhat * rhat);
    (rhat, ess)
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0).max(1.0)
}

/// Build a summary from a trace and the prior variable names.
pub fn summarize(trace: &Trace, param_names: &[String], engine: Engine) -> InferenceResult {
    let flat = trace.flatten();
    let mut per_param_chains: Vec<Vec<Vec<f64>>> = vec![Vec::new(); param_names.len()];
    for chain in &trace.chains {
        for (p, col) in per_param_chains.iter_mut().enumerate() {
            col.push(chain.iter().map(|s| s[p]).collect());
        }
    }

    let summaries = param_names
        .iter()
        .enumerate()
        .map(|(p, name)| {
            let all = &flat[p];
            let mut sorted = all.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
            let (rhat, ess) = rhat_and_ess(&per_param_chains[p]);
            ParamSummary {
                name: name.clone(),
                mean: mean(all),
                sd: variance(all).sqrt(),
                q2_5: quantile_sorted(&sorted, 0.025),
                q50: quantile_sorted(&sorted, 0.5),
                q97_5: quantile_sorted(&sorted, 0.975),
                rhat,
                ess,
            }
        })
        .collect();

    let num_samples = trace.chains.first().map(|c| c.len()).unwrap_or(0);
    InferenceResult {
        engine,
        param_names: param_names.to_vec(),
        summaries,
        num_samples,
        num_chains: trace.chains.len(),
    }
}
