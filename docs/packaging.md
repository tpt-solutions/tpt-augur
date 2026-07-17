# Packaging & Dependency Resolution

How Augur packages are described, versioned, and resolved. This is
the documentation half of the [TODO.md](TODO.md) Phase 7 item; the
build/link half (invoking `cargo` / the Rust FFI) is future work.

## Manifest format (`Augur.toml`)

An Augur package is described by `Augur.toml`:

```toml
modules = ["bayes/regression.augur"]

[package]
name = "my-model"
version = "0.1.0"
authors = ["A. Student"]
description = "A Bayesian regression package"

[dependencies]
stdlib = { version = "0.1", path = "../stdlib/tpt-augur-std" }
gpu-kernel = { version = "1.0", git = "https://github.com/PhillipC05/tpt-gpu" }
```

- `modules` — entry-point model files the compiler can load.
- `[package]` — name + semver version (Cargo convention).
- `[dependencies]` — name → `{ version, git?, path? }`.

Parsed by `tpt_augur_pkg::Manifest` (`parse`, `load`, `to_toml`,
`to_cargo_deps`). Round-trips losslessly.

## Versioning

Augur follows semantic versioning (`MAJOR.MINOR.PATCH`), aligned
with the workspace `Cargo.toml` `version`. Bump the version
in `Cargo.toml` and `CHANGELOG.md` before each release
(see `RELEASE_CHECKLIST.md`).

## Dependency resolution strategy

Today, resolution is **delegated to Cargo**. Augur
dependencies are ordinary Rust crates (e.g. the standard library is
`tpt-augur-std`), so `Manifest::to_cargo_deps()` renders the
`[dependencies]` table a host crate declares to consume them:

```toml
[dependencies]
stdlib = { version = "0.1", path = "../stdlib/tpt-augur-std" }
gpu-kernel = { version = "1.0", git = "https://github.com/PhillipC05/tpt-gpu" }
```

Resolution (picking compatible versions, locking, publishing) is then
Cargo's job.

### Future work (not yet implemented)

- A dedicated **Augur module registry** with `augur pkg publish` /
  `augur pkg install` flows.
- A resolver that understands Augur-version constraints
  independently of the host Cargo graph.
- The FFI wrapper that actually invokes `cargo` to build/link a
  package's Rust dependencies.
