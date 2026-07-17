# Benchmarks

This document records Augur's sampling throughput and compares it against
reference probabilistic-programming systems. Numbers below are measured on the
CPU inference path; GPU offload is tracked in [TODO.md](TODO.md) (Phase 4) and
is not yet enabled.

## How we measure

Throughput is reported as **effective posterior samples / second** across all
chains, after warmup, for a fixed model. The harness runs `augur run` with a
known seed and reports `num_samples * num_chains / wall_time`. Run it yourself:

```sh
cargo build --release --workspace
time cargo run --release -p tpt-augur-cli -- run examples/beta_binomial.augur -n 2000 -c 4
```

Reported timings are wall-clock on a single developer machine and are intended
as a rough order-of-magnitude, not a certified benchmark.

## Throughput (CPU, reference machine)

| Model | Engine | Samples/s (approx.) |
| --- | --- | --- |
| `beta_binomial` | `pf` | high (vectorised resampling) |
| `normal_mean` | `hmc` | moderate (leapfrog + finite-diff grad) |
| `bayesian_regression` | `hmc` | moderate |

Exact figures depend on dimensionality, `step_size`, and `hmc_steps`. The HMC
gradient is currently computed with a central finite difference; replacing it
with exact autodiff is a planned optimisation that should materially raise
throughput.

## Comparison vs. Pyro / Stan

A head-to-head comparison against Pyro (PyTorch) and CmdStan is **not yet
automated**. The intended methodology:

1. Port each Augur example to an equivalent Pyro/Stan model.
2. Match posterior summaries (means, 2.5%/97.5% quantiles) as the correctness
   gate.
3. Compare effective-samples/sec on identical hardware.

This comparison is blocked on the GPU/offload backend (Phase 4) to make the
comparison fair; CPU-vs-CPU numbers will be added once the harness exists.

## Notes

- Particle filter throughput scales with `num_particles`; larger populations are
  more accurate but slower.
- Variational inference trades per-step cost for many optimisation iterations
  (`vi_iters`); it is fast to converge for Gaussian posteriors.
