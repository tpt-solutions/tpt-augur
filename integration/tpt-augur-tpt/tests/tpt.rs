//! Integration tests for Augur ↔ tpt-gpu interop (hardware targets, TPTIR
//! handoff validation, throughput benchmarking).

use std::str::FromStr;

use tpt_augur_frontend::parse;
use tpt_augur_ir::lower;
use tpt_augur_tpt::{
    benchmark_throughput, handoff_model, select_hardware_target, validate_tptir, HardwareTarget,
    TptInteropError,
};

fn model(src: &str) -> tpt_augur_ir::Model {
    let p = parse(src);
    assert!(!p.has_errors(), "{:?}", p.diagnostics);
    let l = lower(&p.program);
    assert!(
        !l.diagnostics.iter().any(|d| d.is_error()),
        "{:?}",
        l.diagnostics
    );
    l.model
}

#[test]
fn hardware_target_round_trips() {
    for t in HardwareTarget::supported() {
        let parsed = HardwareTarget::from_str(t.as_str()).unwrap();
        assert_eq!(parsed, *t);
    }
    assert_eq!(
        HardwareTarget::from_str("cuda").unwrap(),
        HardwareTarget::Nvidia
    );
    assert_eq!(
        HardwareTarget::from_str("rocm").unwrap(),
        HardwareTarget::Amd
    );
    assert_eq!(
        HardwareTarget::from_str("metal").unwrap(),
        HardwareTarget::AppleSilicon
    );
    assert!(HardwareTarget::from_str("bogus").is_err());
}

#[test]
fn hardware_target_metadata() {
    assert_eq!(HardwareTarget::Cpu.as_str(), "cpu");
    assert_eq!(HardwareTarget::Nvidia.as_str(), "nvidia");
    assert_eq!(HardwareTarget::Cpu.tptc_target(), "text");
    assert_eq!(HardwareTarget::Nvidia.tptc_target(), "tptisa");
    assert!(HardwareTarget::Cpu.is_locally_measurable());
    assert!(!HardwareTarget::Nvidia.is_locally_measurable());
}

#[test]
fn select_defaults_to_cpu() {
    assert_eq!(select_hardware_target(None).unwrap(), HardwareTarget::Cpu);
    assert_eq!(
        select_hardware_target(Some("amd")).unwrap(),
        HardwareTarget::Amd
    );
    assert!(matches!(
        select_hardware_target(Some("nope")),
        Err(TptInteropError::UnknownHardware(_))
    ));
}

#[test]
fn handoff_emits_and_validates_tptir() {
    let m = model("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    let h = handoff_model(&m, "model", HardwareTarget::Cpu).unwrap();
    assert!(h.tptir.contains("augur.sample"));
    assert!(h.structurally_valid);
    assert_eq!(h.hardware, HardwareTarget::Cpu);
}

#[test]
fn accelerator_handoff_dispatches_without_error() {
    let m = model("let mu ~ Normal(0, 1)");
    for hw in [
        HardwareTarget::Nvidia,
        HardwareTarget::Amd,
        HardwareTarget::AppleSilicon,
    ] {
        let h = handoff_model(&m, "model", hw).unwrap();
        assert_eq!(h.hardware, hw);
        assert!(h.structurally_valid);
    }
}

#[test]
fn handoff_to_apple_silicon_tags_correctly() {
    let m = model("let p ~ Beta(1, 1)");
    let h = handoff_model(&m, "model", HardwareTarget::AppleSilicon).unwrap();
    assert!(h.tptir.contains("augur.hardware = \"apple_silicon\""));
}

#[test]
fn validate_tptir_accepts_well_formed_module() {
    let tptir = "module {\n  func.func @model() attributes {augur.hardware = \"cpu\"} {\n    ^entry:\n    \"augur.sample\"(%0) : (!augur.dist) -> f64\n    tptir.return\n  }\n}\n";
    assert!(validate_tptir(tptir).unwrap());
}

#[test]
fn validate_tptir_rejects_malformed_modules() {
    assert!(validate_tptir("not a module").is_err());
    assert!(validate_tptir("module { func.func @m() { ^entry: tptir.return } }").is_err());
}

#[test]
fn cpu_throughput_is_measured_and_positive() {
    let m = model("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    let r = benchmark_throughput(&m, HardwareTarget::Cpu, 200, 2).unwrap();
    assert!(r.measured);
    assert!(r.samples_per_sec > 0.0);
    assert_eq!(r.total_samples, 400);
}

#[test]
fn accelerator_throughput_reports_unmeasured() {
    let m = model("let mu ~ Normal(0, 1)");
    let r = benchmark_throughput(&m, HardwareTarget::Nvidia, 200, 2).unwrap();
    assert!(!r.measured);
    assert_eq!(r.total_samples, 400);
}
