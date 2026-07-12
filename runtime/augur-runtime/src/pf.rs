//! Bootstrap particle filter (sequential Monte Carlo) for posterior approximation.
//!
//! All prior variables form a single latent state. Observations are processed in
//! program order; each reweights the particle population by its likelihood, with
//! systematic resampling when the effective sample size collapses. This is the
//! automatically-selected engine for models with discrete latent variables and
//! doubles as a generic importance sampler for continuous ones.

use rand::Rng;

use augur_ir::{eval, instantiate_dist, Env, Model, ModelItem};
use augur_std::seeded_rng;

use crate::engine::InferOptions;
use crate::trace::Trace;

pub fn run_all(model: &Model, opts: &InferOptions) -> Trace {
    let n = opts.num_particles.max(1);
    let mut rng = seeded_rng(opts.seed ^ 0x165667B1);

    // Collect priors (in order) and observations.
    let mut prior_items = Vec::new();
    let mut observes = Vec::new();
    collect(model, &mut prior_items, &mut observes);

    // Initialise particles by sampling each prior in turn. Earlier priors are
    // visible to later ones (e.g. `success ~ Bernoulli(p)` sees `p`), so we
    // accumulate sampled values into the env as we go.
    let d = prior_items.len();
    let mut particles: Vec<Vec<f64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut env = Env::new();
        let mut vals = Vec::with_capacity(d);
        for (dist, name) in &prior_items {
            if let Some(d_) = instantiate_dist(dist, &env) {
                let v = d_.sample(&mut rng);
                // Apply support constraints for the prior family.
                let v = apply_support(&d_, v);
                env.insert(name.clone(), v);
                vals.push(v);
            } else {
                env.insert(name.clone(), 0.0);
                vals.push(0.0);
            }
        }
        particles.push(vals);
    }

    let mut log_w = vec![0.0f64; n];

    for (dist, value_expr) in &observes {
        let obs = eval(value_expr, &Env::new());
        for (p, particle) in particles.iter().enumerate() {
            let mut env = Env::new();
            for (i, (_, name)) in prior_items.iter().enumerate() {
                env.insert(name.clone(), particle[i]);
            }
            if let Some(d_) = instantiate_dist(dist, &env) {
                log_w[p] += d_.logp(obs);
            }
        }
        // Resample if effective sample size is low.
        let ess = effective_sample_size(&log_w);
        if ess < 0.5 * n as f64 {
            particles = systematic_resample(&particles, &log_w, &mut rng);
            log_w = vec![0.0; n];
        }
    }

    // If no resample happened at the last step, resample now so the output is
    // an i.i.d.-ish posterior draw.
    if observes.is_empty() || effective_sample_size(&log_w) < 0.5 * n as f64 {
        particles = systematic_resample(&particles, &log_w, &mut rng);
    }

    // Produce a trace; split particles across chains for summary compatibility.
    let chains = if opts.num_chains > 1 {
        particles
            .chunks((n as f64 / opts.num_chains as f64).ceil() as usize)
            .map(|c| c.to_vec())
            .collect()
    } else {
        vec![particles]
    };
    Trace { chains }
}

fn apply_support(d: &augur_std::Dist, v: f64) -> f64 {
    // `clamp`/`max` pass NaN through, so fall back to a support interior point
    // for any non-finite draw before constraining.
    let v = if v.is_finite() { v } else { 0.5 };
    match d {
        augur_std::Dist::Beta { .. } => v.clamp(1e-6, 1.0 - 1e-6),
        augur_std::Dist::HalfNormal { .. }
        | augur_std::Dist::Gamma { .. }
        | augur_std::Dist::Exponential { .. } => v.max(1e-6),
        augur_std::Dist::Uniform { lo, hi } => v.clamp(*lo, *hi),
        _ => v,
    }
}

fn effective_sample_size(log_w: &[f64]) -> f64 {
    let max = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return 1.0;
    }
    let sum_exp: f64 = log_w.iter().map(|w| (w - max).exp()).sum();
    let sum_sq: f64 = log_w.iter().map(|w| (w - max).exp().powi(2)).sum();
    if sum_sq == 0.0 {
        return 1.0;
    }
    sum_exp * sum_exp / sum_sq
}

fn systematic_resample<R: Rng + ?Sized>(
    particles: &[Vec<f64>],
    log_w: &[f64],
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let n = particles.len();
    let max = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_w.iter().map(|w| (w - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    let norm: Vec<f64> = weights.iter().map(|w| w / total).collect();

    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for w in &norm {
        acc += w;
        cdf.push(acc);
    }

    let step = 1.0 / n as f64;
    let u0: f64 = rng.gen::<f64>() * step;
    let mut out = Vec::with_capacity(n);
    let mut idx = 0;
    for i in 0..n {
        let u = u0 + i as f64 * step;
        while idx < n - 1 && u > cdf[idx] {
            idx += 1;
        }
        out.push(particles[idx].clone());
    }
    out
}

fn collect(
    model: &Model,
    priors: &mut Vec<(augur_ir::Expr, String)>,
    observes: &mut Vec<(augur_ir::Expr, augur_ir::Expr)>,
) {
    for item in &model.items {
        match item {
            ModelItem::Prior { name, dist, .. } => {
                priors.push((dist.clone(), name.clone()));
            }
            ModelItem::Observe { dist, value, .. } => {
                observes.push((dist.clone(), value.clone()));
            }
            ModelItem::If {
                then_items,
                else_items,
                ..
            } => {
                collect(
                    &Model {
                        items: then_items.clone(),
                        prior_order: model.prior_order.clone(),
                    },
                    priors,
                    observes,
                );
                collect(
                    &Model {
                        items: else_items.clone(),
                        prior_order: model.prior_order.clone(),
                    },
                    priors,
                    observes,
                );
            }
            _ => {}
        }
    }
}
