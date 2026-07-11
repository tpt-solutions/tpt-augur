# Contributing to TPT Augur

Thanks for your interest in contributing!

## Getting started

1. Fork and clone the repository.
2. Install a stable Rust toolchain (edition 2021).
3. Build and test:
   ```sh
   cargo build --workspace
   cargo test  --workspace
   ```

## Development workflow

- Keep `cargo fmt --all` clean; CI checks formatting.
- Run `cargo clippy --workspace --all-targets -- -D warnings` before opening a
  PR.
- Add or update tests for any behaviour change. Runtime changes should include
  a check against a known closed-form posterior where possible.
- Update [TODO.md](TODO.md) when you complete a checklist item, and add a
  `<!-- verify: path -->` marker so the drift-guard CI job can confirm the
  work exists.

## Commit messages

Use clear, imperative commit subjects (`add HMC jitter`, not `added HMC jitter`).

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting security issues

Please follow the process in [SECURITY.md](SECURITY.md). Do **not** open public
issues for vulnerabilities.
