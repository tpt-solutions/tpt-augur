//! Augur package management.
//!
//! `augur-pkg` defines the Augur-specific module manifest format and a thin
//! wrapper over the underlying Cargo/Rust FFI. A package is described by an
//! `Augur.toml` manifest; this crate parses it and prepares a dependency
//! specification that the host Cargo workspace can consume.
//!
//! Registry publish/install flow and full version resolution are intentionally
//! out of scope for this slice (see TODO.md Phase 7); the manifest model and
//! parsing are the stable, tested substrate they build on.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PkgError {
    #[error("failed to read manifest: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse manifest: {0}")]
    Toml(#[from] toml::de::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dependency {
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    /// Modules exported by this package (entry points the compiler can load).
    /// Declared first so it serialises as a root-level array *before* the
    /// `[package]`/`[dependencies]` table headers (TOML attaches bare keys to
    /// the most recent table header, so order matters for round-tripping).
    #[serde(default)]
    pub modules: Vec<String>,
    pub package: PackageMeta,
    #[serde(default)]
    pub dependencies: std::collections::BTreeMap<String, Dependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackageMeta {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
}

impl Manifest {
    /// Parse a manifest from a TOML string.
    pub fn parse(src: &str) -> Result<Manifest, PkgError> {
        let m: Manifest = toml::from_str(src)?;
        Ok(m)
    }

    /// Load and parse an `Augur.toml` from disk.
    pub fn load(path: &Path) -> Result<Manifest, PkgError> {
        let s = std::fs::read_to_string(path)?;
        Manifest::parse(&s)
    }

    /// Render a minimal Cargo `[dependencies]` table from this manifest's
    /// Augur dependencies. This is the dependency half of the `augur-pkg`
    /// wrapper over Cargo: it turns `Augur.toml` dependencies into the
    /// `[dependencies]` entries a host crate would declare to consume them.
    /// (The companion half — invoking `cargo`/linking the Rust FFI — is
    /// future work; see TODO.md Phase 7.)
    pub fn to_cargo_deps(&self) -> String {
        let mut out = String::from("[dependencies]\n");
        for (name, dep) in &self.dependencies {
            let mut parts = vec![format!("version = {:?}", dep.version)];
            if let Some(git) = &dep.git {
                parts.push(format!("git = {:?}", git));
            }
            if let Some(path) = &dep.path {
                parts.push(format!("path = {:?}", path));
            }
            out.push_str(&format!("{} = {{ {} }}\n", name, parts.join(", ")));
        }
        out
    }

    /// Render the manifest back to TOML.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
modules = [
  "bayes/regression.augur",
  "bayes/forecast.augur",
]

[package]
name = "my-model"
version = "0.1.0"
authors = ["A. Student"]
description = "A Bayesian regression package"

[dependencies]
stdlib = { version = "0.1", path = "../stdlib/augur-std" }
gpu-kernel = { version = "1.0", git = "https://github.com/PhillipC05/tpt-gpu" }
"#;

    #[test]
    fn renders_cargo_dependencies() {
        let m = Manifest::parse(SAMPLE).unwrap();
        let deps = m.to_cargo_deps();
        assert!(deps.starts_with("[dependencies]\n"));
        assert!(deps.contains("stdlib = { version = \"0.1\", path = \"../stdlib/augur-std\" }"));
        assert!(deps.contains("git = \"https://github.com/PhillipC05/tpt-gpu\""));
    }

    #[test]
    fn parses_manifest() {
        let m = Manifest::parse(SAMPLE).unwrap();
        assert_eq!(m.package.name, "my-model");
        assert_eq!(m.package.version, "0.1.0");
        assert_eq!(m.dependencies.len(), 2);
        assert_eq!(
            m.dependencies["stdlib"].path.as_deref(),
            Some("../stdlib/augur-std")
        );
        assert_eq!(m.modules.len(), 2);
    }

    #[test]
    fn round_trips() {
        let m = Manifest::parse(SAMPLE).unwrap();
        let back = Manifest::parse(&m.to_toml()).unwrap();
        assert_eq!(m, back);
    }
}
