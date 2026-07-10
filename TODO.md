# tpt-augur — Development Checklist

Tracks implementation progress against the roadmap in [spec.txt](spec.txt). Check items off
as they're completed; nothing is checked yet since no code exists in this repo beyond the
design doc.

Augur builds **on top of** the existing `tpt-gpu` toolchain (`../tpt-gpu/layer7_tptb`:
`tptb-core`, `tptb-lsp`, `tptb-cli`, `tptb-format`) rather than a from-scratch LLVM/MLIR
frontend — reuse those crates wherever practical and lower into `tpt-gpu`'s TPTIR
(`../tpt-gpu/layer3_tptc`) for hardware execution.

## Phase 0 — Repo & Toolchain Scaffolding

- [ ] Rust workspace layout: `compiler/` (frontend + MLIR passes), `runtime/` (sampling
      execution engine), `stdlib/` (distribution library), `pkg/` (`augur-pkg`) as workspace
      members
- [ ] Decide crate boundaries against reused `tpt-gpu` layer7 crates (what Augur owns vs.
      what it depends on from `tptb-core`/`tptb-lsp`)
- [ ] `Cargo.toml` workspace + `Cargo.lock`, pin `tpt-gpu` crates as path/git dependencies
- [ ] Basic CI (build + test on push)
- [ ] License, README, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md (match sibling repo
      conventions)
- [ ] CI check that this TODO.md's checked-off items correspond to code that actually exists,
      so the checklist can't silently drift from reality

## Phase 1 — Grammar & Parser (spec.txt §2 syntax)

- [ ] Tree-sitter grammar for distribution-native syntax (`let temp ~ Normal(0, 1)`)
- [ ] Error-tolerant parsing (partial/invalid programs still produce a usable parse tree)
- [ ] AST design covering probabilistic declarations, standard math ops, and deterministic
      control flow
- [ ] Evaluate reusing `tptb-core`'s existing parsing/formatting infrastructure instead of a
      fully separate parser crate
- [ ] `augur fmt` — code formatter (via `tptb-format` if reusable)

## Phase 2 — Type System & Uncertainty Propagation (spec.txt §2)

- [ ] Distribution types as first-class types in the type system
- [ ] Automatic uncertainty propagation through standard math operations (`+`, `-`, `*`, `/`,
      composed expressions)
- [ ] Deterministic/probabilistic interop: standard `if/else` mixed safely with probabilistic
      logic
- [ ] Type-checker with diagnostics (feeds LSP in Phase 6)
- [ ] Static analysis for malformed/degenerate distribution parameters

## Phase 3 — Inference Engine Selection & Algorithms (spec.txt §2)

- [ ] Model-topology analyzer that inspects a program's probabilistic graph to pick an
      inference strategy automatically
- [ ] **Hamiltonian Monte Carlo (HMC)** engine
- [ ] **Variational Inference (VI)** engine
- [ ] **Particle Filter** engine
- [ ] Engine selection heuristics/overrides (manual pragma to force a specific engine)
- [ ] Correctness test suite against known-closed-form posteriors

## Phase 4 — MLIR Lowering & tpt-gpu Backend Integration (spec.txt §3)

- [ ] Custom probabilistic MLIR dialect (distributions, sampling ops, inference-graph nodes)
- [ ] Probabilistic-specific MLIR optimization passes (pre-lowering)
- [ ] Lowering pass: Augur MLIR → TPTIR (`../tpt-gpu/layer3_tptc`)
- [ ] Integration with `layer7_tptb` toolchain (CLI build/run flow, not a parallel backend)
- [ ] Hardware-agnostic dispatch validated across NVIDIA, AMD, and Apple Silicon via existing
      tpt-gpu execution layers
- [ ] Benchmark parallel sampling throughput per hardware target

## Phase 5 — Runtime Engine (spec.txt §3 Runtime)

- [ ] Rust runtime for sampling execution (memory safety, zero-cost abstractions)
- [ ] Fearless-concurrency model for parallel MCMC chains
- [ ] Interop with tpt-gpu's runtime (`../tpt-gpu/layer4_tptr`)
- [ ] Deterministic-fallback execution path (non-probabilistic code runs without sampling
      overhead)
- [ ] Runtime error handling/diagnostics surfaced back to source locations

## Phase 6 — Developer Tooling / LSP (spec.txt §3 Developer Tooling)

- [ ] Extend/reuse `tptb-lsp` for Augur: distribution type-checking diagnostics
- [ ] Inference-graph visualization support (LSP custom request or companion tool)
- [ ] VS Code extension (syntax highlighting, LSP client wiring)
- [ ] Neovim LSP client configuration/docs
- [ ] `augur-cli` — build/run/test/repl commands (via or alongside `tptb-cli`)

## Phase 7 — Package Management (spec.txt §3 Package Management)

- [ ] `augur-pkg` wrapper over Cargo/Rust FFI
- [ ] Package manifest format for Augur-specific modules
- [ ] Publish/install flow for Augur module registry
- [ ] Versioning/dependency-resolution strategy documented

## Phase 8 — TPT Keystone DB Integration (spec.txt §4)

- [ ] Native query bindings from Augur models into Keystone DB
- [ ] Prior-distribution updates from real-time relational data (Keystone core engine)
- [ ] Prior-distribution updates from vector data (Keystone Prism/vector engine)
- [ ] Example model demonstrating live prior updates from a Keystone query
- [ ] Integration tests against a running Keystone instance

## Phase 9 — TPT Locus Integration (spec.txt §4) — ⚠ speculative, blocked on Locus

> `2tpt-locus` is currently only a `spec.txt` with no implementation. These tasks cannot start
> until Locus has real code to integrate against; keep them unchecked and revisit once Locus
> reaches at least a working prototype.

- [ ] Define the "probability of success" scoring API surface Locus will call into
- [ ] Expose a stable Augur model-evaluation entry point (strategy → probability output)
- [ ] Coordinate schema/interface with Locus's spec.txt §4 expectations once Locus has code
- [ ] End-to-end integration test once both projects have working builds

## Phase 10 — Documentation, Benchmarks & Release Readiness

- [ ] README.md (project overview, quickstart, examples)
- [ ] CHANGELOG.md
- [ ] BENCHMARKS.md (sampling throughput, comparison vs. Pyro/Stan, matching tpt-gpu's
      benchmark documentation convention)
- [ ] Worked examples (Bayesian regression, time-series filtering, etc.)
- [ ] API reference docs (stdlib distributions, inference engine options)
- [ ] v1.0.0 release checklist (mirrors `tpt-gpu/RELEASE_CHECKLIST.md`)
