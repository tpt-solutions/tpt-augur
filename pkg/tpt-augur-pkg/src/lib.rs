//! Augur package management.
//!
//! `tpt-augur-pkg` defines the Augur-specific module manifest format and a thin
//! wrapper over the underlying Cargo/Rust FFI. A package is described by an
//! `Augur.toml` manifest; this crate parses it and prepares a dependency
//! specification that the host Cargo workspace can consume, and additionally
//! provides a local-registry publish/install flow (see TODO.md Phase 7) for
//! sharing Augur module packages without a remote registry server.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PkgError {
    #[error("failed to read manifest: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse manifest: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("failed to (de)serialize package: {0}")]
    Json(#[from] serde_json::Error),
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
    /// Augur dependencies. This is the dependency half of the `tpt-augur-pkg`
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

/// A single module source file bundled into a published package.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackageFile {
    /// Repository-relative path of the source (preserved on install).
    pub path: String,
    /// UTF-8 source contents.
    pub contents: String,
}

/// A package as stored in the registry: its manifest plus the module sources
/// it exports.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PublishedPackage {
    pub manifest: Manifest,
    pub sources: Vec<PackageFile>,
}

/// A filesystem-backed package registry used by the publish/install flow.
///
/// The default location is `$AUGUR_REGISTRY`, falling back to
/// `~/.augur/registry`. Packages are stored under `<root>/<name>/<version>/`
/// with the manifest as `Augur.toml` and a `.published.json` index describing
/// the exact sources installed by `install`. This is intentionally a local,
/// server-free registry so packages can be shared within a workspace or CI
/// cache without standing up a remote registry service.
pub struct Registry {
    root: PathBuf,
}

impl Registry {
    /// Open the registry rooted at the given directory (created on publish).
    pub fn new(root: PathBuf) -> Self {
        Registry { root }
    }

    /// Open the default registry: `$AUGUR_REGISTRY`, else `~/.augur/registry`.
    pub fn open_default() -> Result<Registry, PkgError> {
        let root = if let Ok(v) = std::env::var("AUGUR_REGISTRY") {
            PathBuf::from(v)
        } else {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".augur").join("registry")
        };
        Ok(Registry { root })
    }

    fn pkg_dir(&self, name: &str, version: &str) -> PathBuf {
        self.root.join(name).join(version)
    }

    /// Publish `pkg` to the registry, writing its manifest, module sources,
    /// and a `.published.json` index under `<root>/<name>/<version>/`.
    pub fn publish(&self, pkg: &PublishedPackage) -> Result<(), PkgError> {
        let dir = self.pkg_dir(&pkg.manifest.package.name, &pkg.manifest.package.version);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("Augur.toml"), pkg.manifest.to_toml())?;
        for f in &pkg.sources {
            let target = dir.join(&f.path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&target, &f.contents)?;
        }
        std::fs::write(dir.join(".published.json"), serde_json::to_string(pkg)?)?;
        Ok(())
    }

    /// Fetch a published package from the registry by exact name + version.
    pub fn get(&self, name: &str, version: &str) -> Result<PublishedPackage, PkgError> {
        let json = std::fs::read_to_string(self.pkg_dir(name, version).join(".published.json"))?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Resolve and materialize `name@version` into `dest`: writes the package's
    /// module sources (preserving their relative paths) and a resolved
    /// `Augur.toml` manifest. Returns the paths that were written.
    pub fn install(
        &self,
        name: &str,
        version: &str,
        dest: &Path,
    ) -> Result<Vec<PathBuf>, PkgError> {
        let pkg = self.get(name, version)?;
        std::fs::create_dir_all(dest)?;
        let mut written = Vec::new();
        for f in &pkg.sources {
            let target = dest.join(&f.path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&target, &f.contents)?;
            written.push(target);
        }
        let manifest_path = dest.join("Augur.toml");
        std::fs::write(&manifest_path, pkg.manifest.to_toml())?;
        written.push(manifest_path);
        Ok(written)
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
stdlib = { version = "0.1", path = "../stdlib/tpt-augur-std" }
gpu-kernel = { version = "1.0", git = "https://github.com/PhillipC05/tpt-gpu" }
"#;

    #[test]
    fn renders_cargo_dependencies() {
        let m = Manifest::parse(SAMPLE).unwrap();
        let deps = m.to_cargo_deps();
        assert!(deps.starts_with("[dependencies]\n"));
        assert!(deps.contains("stdlib = { version = \"0.1\", path = \"../stdlib/tpt-augur-std\" }"));
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
            Some("../stdlib/tpt-augur-std")
        );
        assert_eq!(m.modules.len(), 2);
    }

    #[test]
    fn round_trips() {
        let m = Manifest::parse(SAMPLE).unwrap();
        let back = Manifest::parse(&m.to_toml()).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn publish_then_install_round_trips_sources() {
        let m = Manifest::parse(SAMPLE).unwrap();
        let pkg = PublishedPackage {
            manifest: m,
            sources: vec![PackageFile {
                path: "bayes/regression.augur".to_string(),
                contents: "let mu ~ Normal(0, 1)".to_string(),
            }],
        };
        let dir = std::env::temp_dir().join("augur_pkg_test_registry");
        let _ = std::fs::remove_dir_all(&dir);
        let registry = Registry::new(dir.clone());

        registry.publish(&pkg).unwrap();
        let got = registry.get("my-model", "0.1.0").unwrap();
        assert_eq!(got, pkg);

        let dest = std::env::temp_dir().join("augur_pkg_test_install");
        let _ = std::fs::remove_dir_all(&dest);
        let written = registry.install("my-model", "0.1.0", &dest).unwrap();
        let source_out = dest.join("bayes/regression.augur");
        assert!(written.contains(&source_out));
        assert_eq!(
            std::fs::read_to_string(&source_out).unwrap(),
            "let mu ~ Normal(0, 1)"
        );
        assert!(dest.join("Augur.toml").exists());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dest);
    }
}
