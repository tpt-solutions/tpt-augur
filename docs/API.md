# Augur API Reference

This is a curated overview of Augur's public API. The canonical,
machine-generated reference is produced with `cargo doc --workspace` (output
under `target/doc/`). Every item below is re-exported from a workspace
crate; import the crates directly.

## `augur-frontend` — lexing, parsing, formatting

| Item | Description |
| --- | --- |
| `parse(src: &str) -> ParseResult` | Error-tolerant parse; returns a `Program` plus `Diagnostic`s. |
| `format_program(program: &Program) -> String` | Canonical re-print of a parsed model (`augur fmt`). |
| `ast::{Program, Stmt, Expr, BinOp, CmpOp, Span}` | AST node types. `Expr::Call { name, args }` represents a distribution constructor. |
| `diagnostics::{Diagnostic, Severity}` | Parse/format diagnostics; `Diagnostic::is_error()`. |

`ParseResult { program, diagnostics }` — `has_errors()`, `warnings()`.

## `augur-ir` — typed IR, lowering, uncertainty

| Item | Description |
| --- | --- |
| `lower(program: &Program) -> LowerResult` | AST → `Model` with type errors / degenerate-parameter warnings. |
| `Model { items, prior_order }` | Lowered model; `prior_order` is the fixed sampling-vector shape. |
| `ModelItem::{Prior, Let, Observe, If}` | Items carry `Span`s for source mapping. |
| `eval(expr: &Expr, env: &Env) -> f64` | Evaluate a deterministic expression against bound values. |
| `instantiate_dist(expr: &Expr, env: &Env) -> Option<Dist>` | Build a concrete `Dist` from a distribution expression. |
| `log_joint(model: &Model, values: &[f64], env: &mut Env) -> f64` | Unnormalised log-posterior; the heart of uncertainty propagation. |
| `known_dist_arity(name) -> Option<usize>` | Expected argument count per known family. |

`Env = HashMap<String, f64>`. `LowerResult { model, diagnostics }`.

## `augur-runtime` — inference engines

| Item | Description |
| --- | --- |
| `run(model: &Model, opts: &InferOptions) -> InferenceResult` | Dispatch to the chosen (or auto-selected) engine, then summarise. |
| `select_engine(model: &Model) -> Engine` | Topology-based default: discrete latents → particle filter; high-dimensional → VI; else HMC. |
| `Engine::{MetropolisHastings, Hmc, Variational, ParticleFilter}` | `as_str()`, `FromStr` (`"mh" | "hmc" | "vi" | "pf"` …). |
| `InferOptions` | `engine: Option<Engine>`, `num_chains`, `num_warmup`, `num_samples`, `seed`, `step_size`, `hmc_steps`, `num_particles`, `vi_iters` (all `Default`). |
| `Trace { chains: Vec<Vec<Vec<f64>>> }` | `chains[c][s][p]` = value of prior `p` in sample `s` of chain `c`. |
| `InferenceResult { engine, param_names, summaries, num_samples, num_chains }` | `mean_of(name) -> Option<f64>`. |
| `ParamSummary { name, mean, sd, q2_5, q50, q97_5, rhat, ess }` | Posterior quantiles, Gelman–Rubin `rhat`, effective sample size. |
| `common::eval_deterministic(model: &Model) -> Env` | Deterministic-fallback: evaluate a model with no priors/observations, without sampling. |

Engines (each `run_all(model, opts) -> Trace`): `hmc`, `vi`, `pf`, `mh`.

## `augur-std` — distribution library

`Dist` enum: `Normal`, `HalfNormal`, `Beta`, `Gamma`, `Uniform`,
`Exponential`, `Binomial`, `Poisson`, `Bernoulli`.

| Method | Description |
| --- | --- |
| `logp(&self, x: f64) -> f64` | Exact log-density / log-mass. |
| `mean(&self) -> f64`, `variance(&self) -> f64` | Analytic moments where available. |
| `sample<R: Rng>(&self, rng: &mut R) -> f64` | Single draw. |
| `typical_point(&self) -> f64` | Initialisation point that stays in-support. |
| `seeded_rng(seed: u64) -> StdRng` | Reproducible RNG. |
| `std_normal<R: Rng>(&mut R) -> f64` | Standard-normal draw (Box–Muller). |
| `sample_gamma(...)`, `sample_poisson(...)` | Specialised samplers. |

## `augur-pkg` — package manifest

| Item | Description |
| --- | --- |
| `Manifest { modules, package, dependencies }` | Parsed `Augur.toml`. |
| `Dependency { version, git, path }` | One dependency entry. |
| `PackageMeta { name, version, authors, description }` | Package identity. |
| `Manifest::parse(src)`, `Manifest::load(path)` | Parse from string / file. |
| `Manifest::to_toml(&self) -> String` | Serialise back to TOML (round-trips). |
| `Manifest::to_cargo_deps(&self) -> String` | Render Augur deps as a Cargo `[dependencies]` table (the dependency half of the Cargo/FFI wrapper). |

## `augur-cli` — command line

`augur run|check|fmt|repl`. See `README.md` for flags and examples.
