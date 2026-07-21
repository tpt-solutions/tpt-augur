//! Shared helpers for the inference engines.

use rand::Rng;
use tpt_augur_ir::{eval, instantiate_dist, log_joint, Env, Model, ModelItem};
use tpt_augur_std::seeded_rng;

/// Build an initial point for a chain by instantiating each prior at its
/// typical point (with light jitter so chains start dispersed but in-support).
pub fn initial_point(model: &Model, chain_idx: usize, base_seed: u64) -> Vec<f64> {
    let mut rng = seeded_rng(base_seed ^ (chain_idx as u64).wrapping_mul(0x9E3779B1));
    let mut env = Env::new();
    let mut values = Vec::with_capacity(model.prior_order.len());
    for item in &model.items {
        collect_prior(item, &mut env, &mut values, &mut rng);
    }
    values
}

fn collect_prior<R: Rng + ?Sized>(
    item: &ModelItem,
    env: &mut Env,
    values: &mut Vec<f64>,
    rng: &mut R,
) {
    match item {
        ModelItem::Prior { name, dist, .. } => {
            let v = if let Some(d) = instantiate_dist(dist, env) {
                let tp = d.typical_point();
                let scale = tp.abs().max(0.2);
                tp + scale * 0.1 * tpt_augur_std::std_normal(rng)
            } else {
                0.0
            };
            env.insert(name.clone(), v);
            values.push(v);
        }
        ModelItem::If {
            then_items,
            else_items,
            ..
        } => {
            for i in then_items {
                collect_prior(i, env, values, rng);
            }
            for i in else_items {
                collect_prior(i, env, values, rng);
            }
        }
        _ => {}
    }
}

/// Central finite-difference gradient of the log-joint w.r.t. the parameters.
pub fn grad_log_joint(model: &Model, values: &[f64], h: f64) -> Vec<f64> {
    let mut env = Env::new();
    let base = log_joint(model, values, &mut env);
    let mut g = Vec::with_capacity(values.len());
    for k in 0..values.len() {
        let mut vp = values.to_vec();
        let mut vm = values.to_vec();
        vp[k] += h;
        vm[k] -= h;
        let lp = {
            let mut e = Env::new();
            log_joint(model, &vp, &mut e)
        };
        let lm = {
            let mut e = Env::new();
            log_joint(model, &vm, &mut e)
        };
        g.push((lp - lm) / (2.0 * h));
        let _ = base;
    }
    g
}

/// Evaluate the deterministic portion of a model without any sampling.
///
/// Purely-deterministic models (only `let` bindings and `if` blocks, no
/// `Prior`/`Observe`) can be resolved to concrete values directly — this is
/// the deterministic-fallback path that avoids MCMC / importance-sampling
/// overhead entirely. Prior variables are skipped; if a `let` references a
/// prior name, it resolves to that prior's typical point so downstream
/// arithmetic still evaluates.
pub fn eval_deterministic(model: &Model) -> Env {
    let mut env = Env::new();
    eval_items(&model.items, &mut env);
    env
}

fn eval_items(items: &[ModelItem], env: &mut Env) {
    for item in items {
        match item {
            ModelItem::Let { name, value, .. } => {
                let v = eval(value, env);
                env.insert(name.clone(), v);
            }
            ModelItem::If {
                cond,
                then_items,
                else_items,
                ..
            } => {
                if eval(cond, env) > 0.0 {
                    eval_items(then_items, env);
                } else {
                    eval_items(else_items, env);
                }
            }
            _ => {}
        }
    }
}
