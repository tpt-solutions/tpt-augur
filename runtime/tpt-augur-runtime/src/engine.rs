//! Inference engine selection and shared configuration.
//!
//! [`select_engine`] inspects the model's probabilistic graph (prior families,
//! dimensionality, discrete vs. continuous latents) and picks a sensible default
//! engine, which a caller can override via [`InferOptions::engine`].

use tpt_augur_ir::Model;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    MetropolisHastings,
    Hmc,
    Variational,
    ParticleFilter,
}

impl Engine {
    pub fn as_str(&self) -> &'static str {
        match self {
            Engine::MetropolisHastings => "mh",
            Engine::Hmc => "hmc",
            Engine::Variational => "vi",
            Engine::ParticleFilter => "pf",
        }
    }
}

impl std::str::FromStr for Engine {
    type Err = String;

    fn from_str(s: &str) -> Result<Engine, Self::Err> {
        match s {
            "mh" | "metropolis" | "metropolis-hastings" => Ok(Engine::MetropolisHastings),
            "hmc" => Ok(Engine::Hmc),
            "vi" | "variational" => Ok(Engine::Variational),
            "pf" | "particle" | "particle-filter" => Ok(Engine::ParticleFilter),
            _ => Err(format!("unknown engine `{s}`")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferOptions {
    pub engine: Option<Engine>,
    pub num_chains: usize,
    pub num_warmup: usize,
    pub num_samples: usize,
    pub seed: u64,
    /// HMC / MH step size (proposal scale).
    pub step_size: f64,
    /// HMC trajectory length (number of leapfrog steps).
    pub hmc_steps: usize,
    /// Number of particles for the particle filter.
    pub num_particles: usize,
    /// VI optimisation iterations.
    pub vi_iters: usize,
}

impl Default for InferOptions {
    fn default() -> Self {
        InferOptions {
            engine: None,
            num_chains: 4,
            num_warmup: 1000,
            num_samples: 2000,
            seed: 0xC0FFEE,
            step_size: 0.25,
            hmc_steps: 15,
            num_particles: 2000,
            vi_iters: 3000,
        }
    }
}

/// Inspect the model topology and choose an engine automatically.
///
/// * Discrete priors (Binomial/Poisson/Bernoulli) => particle filter (SMC).
/// * High-dimensional continuous models => variational inference.
/// * Otherwise => HMC (Hamiltonian Monte Carlo).
pub fn select_engine(model: &Model) -> Engine {
    let mut has_discrete = false;
    for item in &model.items {
        collect_discrete(item, &mut has_discrete);
    }
    if has_discrete {
        return Engine::ParticleFilter;
    }
    if model.prior_order.len() > 15 {
        return Engine::Variational;
    }
    Engine::Hmc
}

fn collect_discrete(item: &tpt_augur_ir::ModelItem, out: &mut bool) {
    use tpt_augur_ir::ModelItem;
    match item {
        ModelItem::Prior {
            dist: tpt_augur_frontend::Expr::Call { name, .. },
            ..
        } => {
            if matches!(name.as_str(), "Binomial" | "Poisson" | "Bernoulli") {
                *out = true;
            }
        }
        ModelItem::If {
            then_items,
            else_items,
            ..
        } => {
            for i in then_items {
                collect_discrete(i, out);
            }
            for i in else_items {
                collect_discrete(i, out);
            }
        }
        _ => {}
    }
}
