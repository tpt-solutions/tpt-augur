//! Random-walk Metropolis–Hastings sampler.

use rand::Rng;
use std::thread;

use tpt_augur_ir::{log_joint, Env, Model};
use tpt_augur_std::seeded_rng;

use crate::common::initial_point;
use crate::engine::InferOptions;
use crate::trace::Trace;

pub fn run(model: &Model, opts: &InferOptions, chain_idx: usize, out: &mut Vec<Vec<f64>>) {
    let mut rng = seeded_rng(opts.seed ^ (chain_idx as u64).wrapping_mul(0x85EBCA6B));
    let mut values = initial_point(model, chain_idx, opts.seed);
    let mut env = Env::new();
    let mut lp = log_joint(model, &values, &mut env);

    let scale: Vec<f64> = values
        .iter()
        .map(|v| v.abs().max(0.5) * opts.step_size)
        .collect();
    let total = opts.num_warmup + opts.num_samples;

    for it in 0..total {
        let mut proposal = values.clone();
        for k in 0..proposal.len() {
            let z: f64 = tpt_augur_std::std_normal(&mut rng);
            proposal[k] += scale[k] * z;
        }
        let lp_prop = {
            let mut e = Env::new();
            log_joint(model, &proposal, &mut e)
        };
        let accept = lp_prop - lp;
        if accept >= 0.0 || rng.gen::<f64>().ln() < accept {
            values = proposal;
            lp = lp_prop;
        }
        if it >= opts.num_warmup {
            out.push(values.clone());
        }
    }
}

pub fn run_all(model: &Model, opts: &InferOptions) -> Trace {
    // Fearless concurrency: independent chains are sampled in parallel. Each
    // thread owns its own RNG and only reads the (cloned) model/options, so
    // there is no shared mutable state.
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
