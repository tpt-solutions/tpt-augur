# Release Checklist (v1.0.0)

Mirrors `tpt-gpu/RELEASE_CHECKLIST.md`. Work through every item before
tagging a release.

## Pre-release

- [ ] `cargo fmt --all -- --check` is clean.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean.
- [ ] `cargo test --workspace` is green (incl. the closed-form posterior checks).
- [ ] `python3 scripts/check_todo.py` passes (checked TODO items map to code).
- [ ] `cargo build --release --workspace` succeeds.
- [ ] CHANGELOG.md updated with the new version and notable changes.
- [ ] Version numbers bumped in `Cargo.toml` (`workspace.package.version`)
      and every member crate that publishes independently.
- [ ] README quickstart commands verified against a fresh checkout.
- [ ] Examples in `examples/` all `augur check` clean and execute.

## Docs

- [ ] README.md, BENCHMARKS.md, and CHANGELOG.md reviewed.
- [ ] `cargo doc --workspace --no-deps` builds without broken intra-doc links.
- [ ] Worked examples cover at least: conjugacy, regression, time-series.

## Packaging

- [ ] `tpt-augur-pkg` manifest format documented; `to_cargo_deps` output
      validated for the example manifests.
- [ ] CLI `--help` output for every subcommand reviewed.

## Publish

- [ ] Tag `vX.Y.Z` and push the tag.
- [ ] GitHub release notes summarise changes and link the CHANGELOG entry.
- [ ] Crates.io / internal registry publish (if applicable) succeeded.
- [ ] Post-release: bump to next `-dev` version.
