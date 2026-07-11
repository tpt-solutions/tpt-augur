//! Mean-field Automatic Differentiation Variational Inference (ADVI).
//!
//! We fit a diagonal Gaussian `q` in an *unconstrained* space and map it back to
//! each parameter's support (identity for unbounded, log for positive, logit for
//! `(0,1)` and bounded intervals). This keeps posterior mass inside the support
//! so the ELBO is finite.
//!
//! The ELBO gradient uses the reparameterisation trick with an analytic
//! Jacobian: `theta = f(mu + sigma*z)`, so `d theta / d mu = f'(u)/sigma` and
//! `d theta / d sigma = f'(u)*z`. Only the likelihood gradient `d logp/d theta`
//! is obtained by a (noiseless) central finite difference; the rest is closed
//! form. Parameters are optimised with Adam.

use augur_ir::{Model, ModelItem};
use augur_std::seeded_rng;

use crate::common::{grad_log_joint, initial_point};
use crate::engine::InferOptions;
use crate::trace::Trace;
use crate::transforms::{transform_for, Transform};
// `Transform` and `transform_for` are provided by the shared `transforms` module.

struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    lr: f64,
}

impl Adam {
    fn new(dim: usize, lr: f64) -> Self {
        Adam {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            t: 0,
            lr,
        }
    }
    fn step(&mut self, grad: &[f64]) -> Vec<f64> {
        self.t += 1;
        let (b1, b2, eps) = (0.9, 0.999, 1e-8);
        let mut out = vec![0.0; grad.len()];
        for i in 0..grad.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            let mh = self.m[i] / (1.0 - b1.powi(self.t as i32));
            let vh = self.v[i] / (1.0 - b2.powi(self.t as i32));
            out[i] = self.lr * mh / (vh.sqrt() + eps);
        }
        out
    }
}

pub fn run_all(model: &Model, opts: &InferOptions) -> Trace {
    let d = model.prior_order.len();
    let mut rng = seeded_rng(opts.seed ^ 0x27D4EB2F);

    let mut prior_exprs = Vec::new();
    collect_priors(model, &mut prior_exprs);
    let transforms: Vec<Transform> = prior_exprs.iter().map(transform_for).collect();

    let init = initial_point(model, 0, opts.seed);
    let mut mu: Vec<f64> = transforms
        .iter()
        .zip(&init)
        .map(|(t, x)| t.inverse(*x))
        .collect();
    let mut r = vec![(0.3f64).ln(); d]; // sigma = 0.3 initial

    let mut adam_mu = Adam::new(d, 0.05);
    let mut adam_r = Adam::new(d, 0.05);

    let s = 8; // Monte-Carlo samples per gradient estimate
    let h = 1e-4; // finite-difference step for d logp / d theta

    for _ in 0..opts.vi_iters {
        let mut g_mu = vec![0.0; d];
        let mut g_r = vec![0.0; d];
        for _ in 0..s {
            let z: Vec<f64> = (0..d).map(|_| augur_std::std_normal(&mut rng)).collect();
            let mut theta = vec![0.0; d];
            let mut dlogjac_du = vec![0.0; d];
            for k in 0..d {
                let sigma = r[k].exp();
                let u = mu[k] + sigma * z[k];
                let (th, _log_jac, dlj) = transforms[k].forward(u);
                theta[k] = th;
                dlogjac_du[k] = dlj;
            }
            let g = grad_log_joint(model, &theta, h);
            for k in 0..d {
                let sigma = r[k].exp();
                let dtheta_du = {
                    // Recompute f'(u) from the stored transform.
                    let (_th, log_jac, _dlj) = transforms[k].forward(mu[k] + sigma * z[k]);
                    let _ = log_jac;
                    match &transforms[k] {
                        Transform::Identity => 1.0,
                        Transform::Log => theta[k], // exp(u)
                        Transform::Logit { lo, hi } => {
                            let s2 = (theta[k] - lo) / (hi - lo);
                            (hi - lo) * s2 * (1.0 - s2)
                        }
                    }
                };
                let dtheta_dmu = dtheta_du / sigma;
                let dtheta_dr = dtheta_du * sigma * z[k];
                // d(-logq)/dmu = 0; d(-logq)/dr = 1 - dlogjac_du * sigma * z
                let d_logq_dr = 1.0 - dlogjac_du[k] * sigma * z[k];
                g_mu[k] += g[k] * dtheta_dmu;
                g_r[k] += g[k] * dtheta_dr - d_logq_dr;
            }
        }
        for x in &mut g_mu {
            *x /= s as f64;
        }
        for x in &mut g_r {
            *x /= s as f64;
        }
        let d_mu = adam_mu.step(&g_mu);
        let d_r = adam_r.step(&g_r);
        for k in 0..d {
            mu[k] += d_mu[k];
            r[k] += d_r[k];
        }
    }

    let n = opts.num_samples * opts.num_chains.max(1);
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let z: Vec<f64> = (0..d).map(|_| augur_std::std_normal(&mut rng)).collect();
        let theta: Vec<f64> = (0..d)
            .map(|k| transforms[k].forward(mu[k] + r[k].exp() * z[k]).0)
            .collect();
        samples.push(theta);
    }
    Trace {
        chains: vec![samples],
    }
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
