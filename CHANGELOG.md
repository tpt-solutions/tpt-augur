# Changelog

All notable changes to this project are documented in this file. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - Unreleased

### Added
- Workspace scaffolding: `tpt-augur-frontend`, `tpt-augur-ir`, `tpt-augur-runtime`,
  `tpt-augur-std`, `tpt-augur-pkg`, `tpt-augur-cli`.
- Error-tolerant recursive-descent parser and canonical formatter.
- Typed IR with lowering, type-checking, degenerate-parameter static analysis,
  and uncertainty propagation through standard math operations.
- Inference engines: Hamiltonian Monte Carlo, mean-field variational
  inference, bootstrap particle filter, and random-walk Metropolis–Hastings.
- Automatic engine selection from model topology, with explicit overrides.
- Standard library of concrete distributions with exact log-densities and
  samplers (Normal, HalfNormal, Beta, Gamma, Uniform, Exponential, Binomial,
  Poisson, Bernoulli).
- `augur` CLI with `run`, `check`, `fmt`, and `repl` subcommands.
- `Augur.toml` package manifest format and parser.
- Worked examples: Beta–Binomial, Normal–Normal, Bayesian regression, AR(1).
- CI (build + test + clippy + fmt + TODO-drift guard) and project docs.
