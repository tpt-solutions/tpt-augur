# TPT Augur

**A hardware-accelerated probabilistic programming language.**

Augur lets you write variables as *probability distributions* instead of
static values and runs massive parallel posterior sampling at native speed.
It compiles a small, distribution-native language down to a typed model and
executes it with several built-in Bayesian inference engines.

```augur
let mu ~ Normal(0, 1)
observe Normal(mu, 1) = 0.5
```

The posterior of `mu` is `Normal(0.25, 0.5)` — Augur recovers it automatically.

## Status

This repository is the compiler, runtime, standard-library, package manifest,
and CLI for Augur. It is self-contained and builds with a plain Rust
toolchain — no external `tpt-gpu` checkout is required to use the language or
run inference on CPU. Hardware offload, LSP tooling, and TPT-ecosystem
integrations (Keystone DB, Locus) are tracked in [TODO.md](TODO.md) and the
[design spec](spec.txt).

## Features

- **Distribution-native syntax** — `let x ~ Normal(0, 1)`; uncertainty
  propagates automatically through `+`, `-`, `*`, `/` and nested calls.
- **Built-in inference engines**, selected automatically from model topology
  or forced with an explicit override:
  - Hamiltonian Monte Carlo (`hmc`)
  - Mean-field variational inference (`vi`)
  - Bootstrap particle filter / SMC (`pf`)
  - Random-walk Metropolis–Hastings (`mh`)
- **Error-tolerant parser** — partially-broken programs still yield a usable
  parse tree with diagnostics, which powers the formatter and future editor
  tooling.
- **Type-checker & static analysis** — undeclared variables, arity
  mismatches, and degenerate distribution parameters are caught up front.
- **Mixed deterministic / probabilistic control flow** — standard `if/else`
  gates observations and priors safely alongside stochastic logic.

## Quickstart

```sh
# Build the workspace
cargo build --workspace

# Type-check a model
cargo run -p augur-cli -- check examples/beta_binomial.augur

# Run inference (engine auto-selected)
cargo run -p augur-cli -- run examples/beta_binomial.augur

# Pretty-print / canonicalise a model
cargo run -p augur-cli -- fmt examples/beta_binomial.augur

# Read a model from stdin and infer
cargo run -p augur-cli -- repl < examples/normal_mean.augur
```

## Language

| Construct | Meaning |
| --- | --- |
| `let name ~ Dist(a, b)` | A latent random variable with a prior. |
| `let name = expr` | A deterministic binding (carries the uncertainty of its inputs). |
| `observe Dist(...) = value` | A likelihood / conditioning statement. |
| `if cond { ... } else { ... }` | Deterministic control flow gating enclosed items. |

Supported distributions: `Normal`, `HalfNormal`, `Beta`, `Gamma`, `Uniform`,
`Exponential`, `Binomial`, `Poisson`, `Bernoulli`.

## CLI

| Command | Description |
| --- | --- |
| `augur run <file>` | Parse, type-check, and run inference. Flags: `-e/--engine`, `-n/--samples`, `-c/--chains`, `--warmup`, `--seed`. |
| `augur check <file>` | Type-check and report diagnostics without sampling. |
| `augur fmt <file>` | Emit canonical formatting. |
| `augur repl` | Read a model from stdin and infer with the auto-selected engine. |

## Workspace layout

| Crate | Role |
| --- | --- |
| `compiler/augur-frontend` | Lexer, AST, error-tolerant parser, formatter. |
| `compiler/augur-ir` | Typed IR: lowering, type-checking, uncertainty propagation. |
| `runtime/augur-runtime` | Inference engines (HMC, VI, PF, MH) and posterior summaries. |
| `stdlib/augur-std` | Concrete distributions with log-densities and samplers. |
| `pkg/augur-pkg` | Augur package manifest format (`Augur.toml`). |
| `tools/augur-cli` | The `augur` command-line interface. |

## Examples

See [`examples/`](examples):

- `beta_binomial.augur` — Beta–Binomial conjugacy (posterior mean ≈ 0.667).
- `normal_mean.augur` — Normal–Normal conjugacy.
- `bayesian_regression.augur` — Bayesian linear regression.
- `ar1_timeseries.augur` — AR(1) time-series filtering.

Run any of them with `cargo run -p augur-cli -- run examples/<file>`.

## Testing

```sh
cargo test --workspace
```

The runtime test suite validates each engine against known closed-form
posteriors (Normal–Normal, Beta–Binomial).

## License

Apache License 2.0 **with LLVM exceptions**. See [LICENSE](LICENSE).
