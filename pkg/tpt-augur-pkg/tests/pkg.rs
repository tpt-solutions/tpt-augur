//! Integration tests for Augur package management.

use tpt_augur_pkg::{Manifest, PackageFile, PublishedPackage, Registry};
use std::path::PathBuf;

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
fn parses_manifest_fields() {
    let m = Manifest::parse(SAMPLE).unwrap();
    assert_eq!(m.package.name, "my-model");
    assert_eq!(m.package.version, "0.1.0");
    assert_eq!(m.package.authors, vec!["A. Student"]);
    assert_eq!(m.dependencies.len(), 2);
    assert_eq!(
        m.dependencies["stdlib"].path.as_deref(),
        Some("../stdlib/tpt-augur-std")
    );
    assert_eq!(m.modules.len(), 2);
}

#[test]
fn renders_cargo_dependencies() {
    let m = Manifest::parse(SAMPLE).unwrap();
    let deps = m.to_cargo_deps();
    assert!(deps.starts_with("[dependencies]\n"));
    assert!(deps.contains("stdlib = { version = \"0.1\", path = \"../stdlib/tpt-augur-std\" }"));
    assert!(deps.contains("git = \"https://github.com/PhillipC05/tpt-gpu\""));
}

#[test]
fn round_trips_through_toml() {
    let m = Manifest::parse(SAMPLE).unwrap();
    let back = Manifest::parse(&m.to_toml()).unwrap();
    assert_eq!(m, back);
}

#[test]
fn manifest_without_modules_parses() {
    let minimal = r#"
[package]
name = "bare"
version = "0.0.1"
"#;
    let m = Manifest::parse(minimal).unwrap();
    assert!(m.modules.is_empty());
    assert!(m.dependencies.is_empty());
    assert_eq!(m.package.name, "bare");
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
    let dir: PathBuf = std::env::temp_dir().join("augur_pkg_test_registry_2");
    let _ = std::fs::remove_dir_all(&dir);
    let registry = Registry::new(dir.clone());

    registry.publish(&pkg).unwrap();
    let got = registry.get("my-model", "0.1.0").unwrap();
    assert_eq!(got, pkg);

    let dest: PathBuf = std::env::temp_dir().join("augur_pkg_test_install_2");
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
