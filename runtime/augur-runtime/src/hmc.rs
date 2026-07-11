//! Hamiltonian Monte Carlo with leapfrog integration and finite-difference
//! gradients, operating in an *unconstrained* parameter space via support
//! transforms. Suitable for both continuous and bounded latent variables
//! (e.g. Beta ∈ (0,1), positive-only families).

use rand::Rng;
use std::thread;

use augur_ir::{log_joint, Env, Model, ModelItem};
use augur_std::seeded_rng;

use crate::common::initial_point;
use crate::engine::InferOptions;
use crate::trace::Trace;
use crate::transforms::{transform_for, Transform};

pub fn run(model: &Model, opts: &InferOptions, chain_idx: usize, out: &mut Vec<Vec<f64>>) {
    let d = model.prior_order.len();
    let mut rng = seeded_rng(opts.seed ^ (chain_idx as u64).wrapping_mul(0xC2B2AE35));
    let eps = opts.step_size;
    let l = opts.hmc_steps.max(1);

    // Build per-dimension transforms and the initial unconstrained point.
    let mut prior_exprs = Vec::new();
    collect_priors(model, &mut prior_exprs);
    let transforms: Vec<Transform> = prior_exprs.iter().map(transform_for).collect();
    let init = initial_point(model, chain_idx, opts.seed);
    let mut w: Vec<f64> = transforms
        .iter()
        .zip(&init)
        .map(|(t, x)| t.inverse(*x))
        .collect();

    // Unconstrained log-density: logp(theta) + log|dtheta/dw|.
    let unconstrained_lp = |w: &[f64]| -> (f64, Vec<f64>) {
        let mut theta = vec![0.0; d];
        for k in 0..d {
            let (th, _log_jac, _dlj) = transforms[k].forward(w[k]);
            theta[k] = th;
        }
        let mut env = Env::new();
        let lp = log_joint(model, &theta, &mut env);
        let log_jac_sum: f64 = (0..d).map(|k| transforms[k].forward(w[k]).1).sum();
        (lp + log_jac_sum, theta)
    };

    let mut lp_w = unconstrained_lp(&w).0;
    let total = opts.num_warmup + opts.num_samples;

    // Central finite-difference gradient of the unconstrained log-density.
    let grad = |w: &[f64]| -> Vec<f64> {
        let h = 1e-4;
        let mut g = vec![0.0; d];
        for k in 0..d {
            let mut wp = w.to_vec();
            let mut wm = w.to_vec();
            wp[k] += h;
            wm[k] -= h;
            let lp = unconstrained_lp(&wp).0;
            let lm = unconstrained_lp(&wm).0;
            g[k] = (lp - lm) / (2.0 * h);
        }
        g
    };

    for _it in 0..total {
        let p0: Vec<f64> = (0..d).map(|_| augur_std::std_normal(&mut rng)).collect();
        let mut w_prop = w.clone();
        let mut p = p0.clone();
        // Symmetric position leapfrog (q-half, p-full, q-half).
        for _ in 0..l {
            for k in 0..d {
                w_prop[k] += 0.5 * eps * p[k];
            }
            let g = grad(&w_prop);
            for k in 0..d {
                p[k] += eps * g[k];
            }
            for k in 0..d {
                w_prop[k] += 0.5 * eps * p[k];
            }
        }
        let lp_prop = unconstrained_lp(&w_prop).0;
        let kinetic0: f64 = p0.iter().map(|x| 0.5 * x * x).sum();
        let kinetic1: f64 = p.iter().map(|x| 0.5 * x * x).sum();
        let accept = (lp_prop - lp_w) + (kinetic0 - kinetic1);
        if accept >= 0.0 || rng.gen::<f64>().ln() < accept {
            w = w_prop;
            lp_w = lp_prop;
        }
        if _it >= opts.num_warmup {
            let theta: Vec<f64> = (0..d).map(|k| transforms[k].forward(w[k]).0).collect();
            out.push(theta);
        }
    }
}

pub fn run_all(model: &Model, opts: &InferOptions) -> Trace {
    // Fearless concurrency: each chain is an independent, deterministic-function
    // of its seed, so we sample them in parallel and collect the results. The
    // engines only touch thread-local RNG state and immutable model/option
    // data, so there is no shared mutable state to race on.
    let n = opts.num_chains;
    let mut handles = Vec::with_capacity(n);
    for c in 0..n {
        let model = model.clone();
        let opts = opts.clone();
        handles.push(thread::spawn(move || {
            let mut samples = Vec::with_capacity(opts.num_samples);
            run(&model, &opts, c, &mut samples);
            samples
        }));
    }
    let chains = handles
        .into_iter()
        .map(|h| h.join().expect("inference thread panicked"))
        .collect();
    Trace { chains }
}

fn collect_priors(model: &Model, out: &mut Vec<augur_frontend::Expr>) {
    for item in &model.items {
        match item {
            ModelItem::Prior { dist, .. } => out.push(dist.clone()),
            ModelItem::If {
                then_items,
                else_items,
                ..
            } => {
                collect_priors(
                    &Model {
                        items: then_items.clone(),
                        prior_order: model.prior_order.clone(),
                    },
                    out,
                );
                collect_priors(
                    &Model {
                        items: else_items.clone(),
                        prior_order: model.prior_order.clone(),
                    },
                    out,
                );
            }
            _ => {}
        }
    }
}
