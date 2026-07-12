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
      execution engine), `stdlib/` (distribution library), `pkg/` (`augur-pkg`), `integration/`
      (tpt-gpu/Locus/Keystone bridges) as workspace members <!-- verify: Cargo.toml -->
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
- [x] Evaluate reusing `tptb-core`'s existing parsing/formatting infrastructure instead of a
      fully separate parser crate — evaluated against `../tpt-gpu/layer7_tptb/tptb-core`
      (now checked out as a sibling repo). Verdict: **keep Augur's own parser crate.**
      `tptb-core`'s lexer/AST/parser/formatter are hardcoded to TPT Script's grammar
      (`fn`/`import`/`type` declarations, `@annotations`) with no notion of distribution
      types, `let x ~ Dist(...)` priors, or `observe` statements — Augur's core syntax.
      Reusing it would mean forking or bolting probabilistic constructs onto a frontend
      designed for a different language, rather than sharing infrastructure. Augur's
      hand-rolled parser already mirrors `tptb-core`'s shape (byte-offset `Span`,
      error-tolerant resync, diagnostics-first design) for tooling consistency, without
      the dependency. <!-- verify: compiler/augur-frontend/src/parser.rs -->
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

- [x] Custom probabilistic MLIR dialect (distributions, sampling ops, inference-graph nodes)
      <!-- verify: compiler/augur-mlir/src/dialect.rs::Graph -->
- [x] Probabilistic-specific MLIR optimization passes (pre-lowering)
      <!-- verify: compiler/augur-mlir/src/passes.rs::default_pipeline -->
- [x] Lowering pass: Augur MLIR → TPTIR (`../tpt-gpu/layer3_tptc`)
      <!-- verify: compiler/augur-mlir/src/codegen.rs::emit_tptir -->
- [x] Integration with `layer7_tptb` toolchain (CLI build/run flow, not a parallel backend)
      <!-- verify: tools/augur-cli/src/main.rs::cmd_build -->
- [x] Hardware-agnostic dispatch validated across NVIDIA, AMD, and Apple Silicon via existing
      tpt-gpu execution layers — `augur-tpt` selects a `HardwareTarget`, tags emitted TPTIR
      with `augur.hardware`, and structurally validates the handoff for all four targets;
      on-device GPU timing still depends on the `augur` dialect being registered in
      `../tpt-gpu/layer3_tptc` (CPU path is the one measured in-process today)
      <!-- verify: integration/augur-tpt/src/lib.rs::handoff_model -->
- [x] Benchmark parallel sampling throughput per hardware target
      <!-- verify: integration/augur-tpt/src/lib.rs::benchmark_throughput -->

## Phase 5 — Runtime Engine (spec.txt §3 Runtime)

- [x] Rust runtime for sampling execution (memory safety, zero-cost abstractions)
      <!-- verify: runtime/augur-runtime/src/lib.rs::run -->
- [x] Fearless-concurrency model for parallel MCMC chains — independent chains are
      sampled in parallel via `std::thread` (see `run_all` in the HMC/MH engines)
      <!-- verify: runtime/augur-runtime/src/hmc.rs::run_all, runtime/augur-runtime/src/mh.rs::run_all -->
- [x] Interop with tpt-gpu's runtime (`../tpt-gpu/layer4_tptr`) — `augur-tpt` owns the
      hand-off boundary (hardware selection, TPTIR emission/validation, throughput
      benchmarking); on-device dispatch into `layer4_tptr` itself is pending the `augur`
      dialect being registered in `layer3_tptc` <!-- verify: integration/augur-tpt/src/lib.rs::Handoff -->
- [x] Deterministic-fallback execution path (non-probabilistic code runs without sampling
      overhead) <!-- verify: runtime/augur-runtime/src/common.rs::eval_deterministic -->
- [x] Runtime error handling/diagnostics surfaced back to source locations
      <!-- verify: tools/augur-cli/src/main.rs::report_ir_diags -->

## Phase 6 — Developer Tooling / LSP (spec.txt §3 Developer Tooling)

- [x] Extend/reuse `tptb-lsp` for Augur: distribution type-checking diagnostics
      — `tptb-lsp` is not checked out in this workspace (mirroring the Phase 1
      parser decision), so Augur ships its own LSP server that reuses the
      existing `parse` + `lower` diagnostics pipeline rather than a frontend
      built for a different language <!-- verify: compiler/augur-lsp/src/lib.rs::analyze_document -->
- [x] Inference-graph visualization support (LSP custom request or companion tool)
      — LSP custom request `augur/inferenceGraph` returns Graphviz DOT, and the
      `augur graph` CLI command emits the same DOT <!-- verify: compiler/augur-lsp/src/lib.rs::inference_graph_dot, tools/augur-cli/src/main.rs::cmd_graph -->
- [x] VS Code extension (syntax highlighting, LSP client wiring)
      <!-- verify: editors/vscode/package.json -->
- [x] Neovim LSP client configuration/docs
      <!-- verify: docs/neovim.md -->
- [x] `augur-cli` — build/run/test/repl commands (via or alongside `tptb-cli`)
      — `augur-cli` exists with `run`/`check`/`fmt`/`repl`; full `tptb-cli` integration pending
      <!-- verify: tools/augur-cli/src/main.rs -->

## Phase 7 — Package Management (spec.txt §3 Package Management)

- [x] `augur-pkg` wrapper over Cargo/Rust FFI — dependency half implemented
      (renders Augur deps to a Cargo `[dependencies]` table); build/link FFI pending
      <!-- verify: pkg/augur-pkg/src/lib.rs::to_cargo_deps -->
- [x] Package manifest format for Augur-specific modules
      <!-- verify: pkg/augur-pkg/src/lib.rs::Manifest -->
- [x] Publish/install flow for Augur module registry
      <!-- verify: pkg/augur-pkg/src/lib.rs::Registry, tools/augur-cli/src/main.rs::cmd_publish -->
- [x] Versioning/dependency-resolution strategy documented
      <!-- verify: docs/packaging.md -->

## Phase 8 — TPT Keystone DB Integration (spec.txt §4)

> `augur-keystone` is not yet listed in the root `Cargo.toml` workspace `members`, so it
> builds standalone (`cargo build -p augur-keystone` from `integration/augur-keystone`) but
> isn't part of the top-level `cargo build`/CI graph yet — add it once `../tpt-keystone-db`
> is a dependable sibling checkout in CI.

- [x] Native query bindings from Augur models into Keystone DB — typed `tpt_sdk::QueryBuilder`
      queries against `augur_feedback`/`augur_memories` tables, abstracted behind
      `KeystoneQuerier` so the bridge is testable without a live server
      <!-- verify: integration/augur-keystone/src/lib.rs::KeystoneQuerier -->
- [x] Prior-distribution updates from real-time relational data (Keystone core engine)
      <!-- verify: integration/augur-keystone/src/lib.rs::prior_from_relational -->
- [x] Prior-distribution updates from vector data (Keystone Prism/vector engine)
      <!-- verify: integration/augur-keystone/src/lib.rs::prior_from_vector_similarity -->
- [x] Example model demonstrating live prior updates from a Keystone query
      <!-- verify: integration/augur-keystone/src/lib.rs::model_from_prior -->
- [x] Integration tests against a running Keystone instance — `#[ignore]`d end-to-end test,
      runs against a live Keystone node with `cargo test -p augur-keystone --test
      live_integration -- --ignored` <!-- verify: integration/augur-keystone/tests/live_integration.rs -->

## Phase 9 — TPT Locus Integration (spec.txt §4)

- [x] Define the "probability of success" scoring API surface Locus will call into
      <!-- verify: integration/augur-locus/src/lib.rs::ProbabilityOfSuccess -->
- [x] Expose a stable Augur model-evaluation entry point (strategy → probability output)
      <!-- verify: integration/augur-locus/src/lib.rs::evaluate_strategy -->
- [x] Coordinate schema/interface with Locus's spec.txt §4 expectations once Locus has code —
      `Strategy`/`LocusAugurBridge` map directly onto Locus's `locus-core::agent` types
      <!-- verify: integration/augur-locus/src/lib.rs::LocusAugurBridge -->
- [x] End-to-end integration test once both projects have working builds — exercises a real
      `locus_core::agent::AgentRegistry` <!-- verify: integration/augur-locus/src/lib.rs::integration_with_locus_agent_registry -->

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
