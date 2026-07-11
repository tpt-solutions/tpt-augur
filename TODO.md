# tpt-augur — Development Checklist

Tracks implementation progress against the roadmap in [spec.txt](spec.txt). Items
are checked only when code that satisfies them actually exists in the repo; the
CI job `scripts/check_todo.py` enforces this (see Phase 0) so the checklist
cannot silently drift from reality. Each checked item carries an inline
`<!-- verify: path -->` marker pointing at the code that proves it.

Augur builds **on top of** the existing `tpt-gpu` toolchain (`../tpt-gpu/layer7_tptb`:
`tptb-core`, `tptb-lsp`, `tptb-cli`, `tptb-format`) rather than a from-scratch LLVM/MLIR
frontend — reuse those crates wherever practical and lower into `tpt-gpu`'s TPTIR
(`../tpt-gpu/layer3_tptc`) for hardware execution. Where `tpt-gpu` is not checked out
in this workspace, the corresponding items remain unchecked.

## Phase 0 — Repo & Toolchain Scaffolding

- [x] Rust workspace layout: `compiler/` (frontend + MLIR passes), `runtime/` (sampling
      execution engine), `stdlib/` (distribution library), `pkg/` (`augur-pkg`) as workspace
      members <!-- verify: Cargo.toml -->
- [x] Decide crate boundaries against reused `tpt-gpu` layer7 crates (what Augur owns vs.
      what it depends on from `tptb-core`/`tptb-lsp`) <!-- verify: Cargo.toml -->
- [x] `Cargo.toml` workspace + `Cargo.lock`, pin `tpt-gpu` crates as path/git dependencies
      <!-- verify: Cargo.lock -->
- [x] Basic CI (build + test on push) <!-- verify: .github/workflows/ci.yml -->
- [x] License, README, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md (match sibling repo
      conventions) <!-- verify: LICENSE, README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md -->
- [x] CI check that this TODO.md's checked-off items correspond to code that actually exists,
      so the checklist can't silently drift from reality <!-- verify: scripts/check_todo.py -->

## Phase 1 — Grammar & Parser (spec.txt §2 syntax)

- [x] Tree-sitter grammar for distribution-native syntax (`let temp ~ Normal(0, 1)`)
      — canonical `grammar.js` added; the execution parser is still the
      handrolled error-tolerant one, with this kept in sync for tooling
      <!-- verify: compiler/augur-frontend/grammar.js -->
- [x] Error-tolerant parsing (partial/invalid programs still produce a usable parse tree)
      <!-- verify: compiler/augur-frontend/src/parser.rs::resync -->
- [x] AST design covering probabilistic declarations, standard math ops, and deterministic
      control flow <!-- verify: compiler/augur-frontend/src/ast.rs::Expr -->
- [ ] Evaluate reusing `tptb-core`'s existing parsing/formatting infrastructure instead of a
      fully separate parser crate — no `tptb-core` present in this workspace yet.
- [x] `augur fmt` — code formatter <!-- verify: compiler/augur-frontend/src/format.rs::format_program -->

## Phase 2 — Type System & Uncertainty Propagation (spec.txt §2)

- [x] Distribution types as first-class types in the type system
      <!-- verify: compiler/augur-ir/src/lower.rs::ModelItem -->
- [x] Automatic uncertainty propagation through standard math operations (`+`, `-`, `*`, `/`,
      composed expressions) <!-- verify: compiler/augur-ir/src/lower.rs::log_joint -->
- [x] Deterministic/probabilistic interop: standard `if/else` mixed safely with probabilistic
      logic <!-- verify: compiler/augur-ir/src/lower.rs::log_joint_items -->
- [x] Type-checker with diagnostics (feeds LSP in Phase 6)
      <!-- verify: compiler/augur-ir/src/lower.rs::lower_stmt -->
- [x] Static analysis for malformed/degenerate distribution parameters
      <!-- verify: compiler/augur-ir/src/lower.rs::check_degenerate_literal -->

## Phase 3 — Inference Engine Selection & Algorithms (spec.txt §2)

- [x] Model-topology analyzer that inspects a program's probabilistic graph to pick an
      inference strategy automatically <!-- verify: runtime/augur-runtime/src/engine.rs::select_engine -->
- [x] **Hamiltonian Monte Carlo (HMC)** engine <!-- verify: runtime/augur-runtime/src/hmc.rs::run_all -->
- [x] **Variational Inference (VI)** engine <!-- verify: runtime/augur-runtime/src/vi.rs::run_all -->
- [x] **Particle Filter** engine <!-- verify: runtime/augur-runtime/src/pf.rs::run_all -->
- [x] Engine selection heuristics/overrides (manual pragma to force a specific engine)
      <!-- verify: runtime/augur-runtime/src/engine.rs::InferOptions -->
- [x] Correctness test suite against known-closed-form posteriors
      <!-- verify: runtime/augur-runtime/src/lib.rs::normal_normal_conjugate_hmc -->

## Phase 4 — MLIR Lowering & tpt-gpu Backend Integration (spec.txt §3)

- [ ] Custom probabilistic MLIR dialect (distributions, sampling ops, inference-graph nodes)
- [ ] Probabilistic-specific MLIR optimization passes (pre-lowering)
- [ ] Lowering pass: Augur MLIR → TPTIR (`../tpt-gpu/layer3_tptc`)
- [ ] Integration with `layer7_tptb` toolchain (CLI build/run flow, not a parallel backend)
- [ ] Hardware-agnostic dispatch validated across NVIDIA, AMD, and Apple Silicon via existing
      tpt-gpu execution layers
- [ ] Benchmark parallel sampling throughput per hardware target

## Phase 5 — Runtime Engine (spec.txt §3 Runtime)

- [x] Rust runtime for sampling execution (memory safety, zero-cost abstractions)
      <!-- verify: runtime/augur-runtime/src/lib.rs::run -->
- [x] Fearless-concurrency model for parallel MCMC chains — independent chains are
      sampled in parallel via `std::thread` (see `run_all` in the HMC/MH engines)
      <!-- verify: runtime/augur-runtime/src/hmc.rs::run_all, runtime/augur-runtime/src/mh.rs::run_all -->
- [ ] Interop with tpt-gpu's runtime (`../tpt-gpu/layer4_tptr`)
- [x] Deterministic-fallback execution path (non-probabilistic code runs without sampling
      overhead) <!-- verify: runtime/augur-runtime/src/common.rs::eval_deterministic -->
- [x] Runtime error handling/diagnostics surfaced back to source locations
      <!-- verify: tools/augur-cli/src/main.rs::report_ir_diags -->

## Phase 6 — Developer Tooling / LSP (spec.txt §3 Developer Tooling)

- [ ] Extend/reuse `tptb-lsp` for Augur: distribution type-checking diagnostics
- [ ] Inference-graph visualization support (LSP custom request or companion tool)
- [ ] VS Code extension (syntax highlighting, LSP client wiring)
- [ ] Neovim LSP client configuration/docs
- [x] `augur-cli` — build/run/test/repl commands (via or alongside `tptb-cli`)
      — `augur-cli` exists with `run`/`check`/`fmt`/`repl`; full `tptb-cli` integration pending
      <!-- verify: tools/augur-cli/src/main.rs -->

## Phase 7 — Package Management (spec.txt §3 Package Management)

- [x] `augur-pkg` wrapper over Cargo/Rust FFI — dependency half implemented
      (renders Augur deps to a Cargo `[dependencies]` table); build/link FFI pending
      <!-- verify: pkg/augur-pkg/src/lib.rs::to_cargo_deps -->
- [x] Package manifest format for Augur-specific modules
      <!-- verify: pkg/augur-pkg/src/lib.rs::Manifest -->
- [ ] Publish/install flow for Augur module registry
- [x] Versioning/dependency-resolution strategy documented
      <!-- verify: docs/packaging.md -->

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

- [x] README.md (project overview, quickstart, examples) <!-- verify: README.md -->
- [x] CHANGELOG.md <!-- verify: CHANGELOG.md -->
- [x] BENCHMARKS.md (sampling throughput, comparison vs. Pyro/Stan, matching tpt-gpu's
      benchmark documentation convention) <!-- verify: BENCHMARKS.md -->
- [x] Worked examples (Bayesian regression, time-series filtering, etc.)
      <!-- verify: examples/bayesian_regression.augur -->
- [x] API reference docs (stdlib distributions, inference engine options)
      <!-- verify: docs/API.md -->
- [x] v1.0.0 release checklist (mirrors `tpt-gpu/RELEASE_CHECKLIST.md`)
      <!-- verify: RELEASE_CHECKLIST.md -->
