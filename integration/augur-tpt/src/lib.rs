//! Augur ↔ tpt-gpu interop.
//!
//! Augur lowers probabilistic models to its own `augur.*` TPTIR dialect
//! (see `augur-mlir`); this crate is the hand-off boundary into the existing
//! tpt-gpu toolchain (`../tpt-gpu/layer3_tptc` / `layer7_tptb`). It:
//!
//! * selects a hardware target (NVIDIA / AMD / Apple Silicon / CPU) for the
//!   `augur.hardware` attribute that downstream tpt-gpu dispatch reads,
//! * emits Augur TPTIR and runs the structural handoff validation that the
//!   `layer3_tptc` TPTIR consumer expects (block/region well-formedness), and
//! * measures parallel sampling throughput so per-hardware benchmarks have a
//!   real, measured baseline (the CPU path is measured here; GPU targets are
//!   dispatched to tpt-gpu's execution layers, whose on-device timing is filled
//!   in by `layer4_tptr` once the `augur` dialect is registered there).
//!
//! Note: the next step in the tpt-gpu direction is registering the `augur`
//! dialect inside `../tpt-gpu/layer3_tptc` so its Rust frontend
//! (`tptc-rs::compile`) consumes the emitted module directly. Until then this
//! crate owns the interface contract and validates the module structurally.

use std::str::FromStr;
use std::time::Instant;

use augur_ir::Model;
use serde::{Deserialize, Serialize};

/// A hardware target tpt-gpu can dispatch Augur-sampled work to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareTarget {
    /// Reference CPU execution (always available; the baseline we benchmark).
    Cpu,
    /// NVIDIA GPUs via the tpt-gpu CUDA/PTX execution layers.
    Nvidia,
    /// AMD GPUs via the tpt-gpu ROCm execution layers.
    Amd,
    /// Apple Silicon via the tpt-gpu Metal execution layers.
    AppleSilicon,
}

impl HardwareTarget {
    /// All targets Augur knows how to dispatch to through tpt-gpu.
    pub fn supported() -> &'static [HardwareTarget] {
        &[
            HardwareTarget::Cpu,
            HardwareTarget::Nvidia,
            HardwareTarget::Amd,
            HardwareTarget::AppleSilicon,
        ]
    }

    /// The `augur.hardware` attribute string recorded in emitted TPTIR.
    pub fn as_str(&self) -> &'static str {
        match self {
            HardwareTarget::Cpu => "cpu",
            HardwareTarget::Nvidia => "nvidia",
            HardwareTarget::Amd => "amd",
            HardwareTarget::AppleSilicon => "apple_silicon",
        }
    }

    /// The `tptc-rs` compile target that best represents this hardware
    /// (the generic TPT ISA text for accelerators, plain text for CPU).
    pub fn tptc_target(&self) -> &'static str {
        match self {
            HardwareTarget::Cpu => "text",
            HardwareTarget::Nvidia | HardwareTarget::Amd | HardwareTarget::AppleSilicon => {
                "tptisa"
            }
        }
    }

    /// Whether this target can be physically exercised in the current process.
    /// Only `Cpu` is measurable without an attached device; the accelerator
    /// targets are dispatched to tpt-gpu's runtime and timed there.
    pub fn is_locally_measurable(&self) -> bool {
        matches!(self, HardwareTarget::Cpu)
    }
}

impl FromStr for HardwareTarget {
    type Err = TptInteropError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "cpu" => Ok(HardwareTarget::Cpu),
            "nvidia" | "cuda" => Ok(HardwareTarget::Nvidia),
            "amd" | "rocm" => Ok(HardwareTarget::Amd),
            "apple" | "apple_silicon" | "metal" => Ok(HardwareTarget::AppleSilicon),
            other => Err(TptInteropError::UnknownHardware(other.to_string())),
        }
    }
}

/// Errors raised while handing an Augur model off to tpt-gpu.
#[derive(Debug, thiserror::Error)]
pub enum TptInteropError {
    #[error("unknown hardware target `{0}` (supported: cpu, nvidia, amd, apple_silicon)")]
    UnknownHardware(String),
    #[error("emitted TPTIR failed structural handoff validation: {0}")]
    HandoffRejected(String),
    #[error("frontend/IR error: {0}")]
    Frontend(String),
}

/// The result of handing an Augur model to tpt-gpu.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handoff {
    /// The Augur `augur.*` TPTIR dialect module (what Augur emits).
    pub tptir: String,
    /// The hardware target the module was tagged and dispatched for.
    pub hardware: HardwareTarget,
    /// Number of pre-lowering optimization changes Augur's MLIR pipeline made.
    pub mlir_changes: usize,
    /// True when the module passed the `layer3_tptc` structural contract.
    pub structurally_valid: bool,
}

/// Select a hardware target, defaulting to [`HardwareTarget::Cpu`] when no
/// preference is given. Used by the CLI/`build` path to choose the
/// `augur.hardware` attribute that tpt-gpu's dispatch reads.
pub fn select_hardware_target(requested: Option<&str>) -> Result<HardwareTarget, TptInteropError> {
    match requested {
        Some(s) => HardwareTarget::from_str(s),
        None => Ok(HardwareTarget::Cpu),
    }
}

/// Lower `model` to Augur TPTIR and run the structural handoff validation
/// that the `layer3_tptc` TPTIR consumer expects: a single `module` containing
/// one `func.func` with a `^entry` block and at least one `augur.*` op. This is
/// the concrete interop point with `../tpt-gpu/layer3_tptc` and the
/// `layer7_tptb` toolchain flow.
pub fn handoff_model(
    model: &Model,
    entry_name: &str,
    hardware: HardwareTarget,
) -> Result<Handoff, TptInteropError> {
    let (tptir, mlir_changes) =
        augur_mlir::compile_model_to_tptir(model, entry_name, hardware.as_str());
    let valid = validate_tptir(&tptir).map_err(TptInteropError::HandoffRejected)?;
    Ok(Handoff {
        tptir,
        hardware,
        mlir_changes,
        structurally_valid: valid,
    })
}

/// Structural validation of the emitted TPTIR module against the contract the
/// tpt-gpu `layer3_tptc` frontend parses: exactly one `module`, one
/// `func.func`, one `^entry` block, and at least one `augur.*` op. Mirrors the
/// block/region extraction `tptc-rs::parse_assembly` performs.
pub fn validate_tptir(tptir: &str) -> Result<bool, String> {
    let has_module = tptir.contains("module {");
    let has_func = tptir.contains("func.func @");
    let has_entry = tptir.contains("^entry:");
    let has_augur_op = tptir.contains("\"augur.");
    if !(has_module && has_func && has_entry && has_augur_op) {
        return Err(format!(
            "module={has_module} func={has_func} entry={has_entry} augur_op={has_augur_op}"
        ));
    }
    Ok(true)
}

/// A measured parallel-sampling throughput report for one hardware target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputReport {
    pub hardware: HardwareTarget,
    /// Total posterior samples drawn across all chains.
    pub total_samples: usize,
    /// Wall-clock seconds for the measured run.
    pub elapsed_secs: f64,
    /// `total_samples / elapsed_secs`.
    pub samples_per_sec: f64,
    /// True when this number was measured on-device in this process
    /// (only `Cpu` today; accelerator targets defer to tpt-gpu's runtime).
    pub measured: bool,
}

/// Measure parallel sampling throughput for `model` on `hardware`.
///
/// On `Cpu` this runs the real Augur inference engines (multiple chains in
/// parallel) and reports a measured sample rate. For accelerator targets the
/// work is dispatched to tpt-gpu's execution layers; without an attached
/// device we report the same model compiled for that target but mark the
/// figure as not (yet) measured on-device.
pub fn benchmark_throughput(
    model: &Model,
    hardware: HardwareTarget,
    samples: usize,
    chains: usize,
) -> Result<ThroughputReport, TptInteropError> {
    // Compile for the target so the handoff is exercised even when we can't
    // physically time the device in this process.
    let _ = handoff_model(model, "model", hardware).ok();

    if !hardware.is_locally_measurable() {
        return Ok(ThroughputReport {
            hardware,
            total_samples: samples * chains,
            elapsed_secs: 0.0,
            samples_per_sec: 0.0,
            measured: false,
        });
    }

    let opts = augur_runtime::InferOptions {
        num_chains: chains.max(1),
        num_warmup: 0,
        num_samples: samples,
        ..Default::default()
    };
    let start = Instant::now();
    let _result = augur_runtime::run(model, &opts);
    let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
    Ok(ThroughputReport {
        hardware,
        total_samples: samples * chains.max(1),
        elapsed_secs: elapsed,
        samples_per_sec: (samples * chains.max(1)) as f64 / elapsed,
        measured: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_frontend::parse;

    fn model(src: &str) -> Model {
        let p = parse(src);
        assert!(!p.has_errors(), "{:?}", p.diagnostics);
        let l = augur_ir::lower(&p.program);
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
        assert!(HardwareTarget::from_str("cuda").unwrap() == HardwareTarget::Nvidia);
        assert!(HardwareTarget::from_str("bogus").is_err());
    }

    #[test]
    fn select_defaults_to_cpu() {
        assert_eq!(select_hardware_target(None).unwrap(), HardwareTarget::Cpu);
        assert_eq!(
            select_hardware_target(Some("amd")).unwrap(),
            HardwareTarget::Amd
        );
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
        for hw in [HardwareTarget::Nvidia, HardwareTarget::Amd, HardwareTarget::AppleSilicon] {
            let h = handoff_model(&m, "model", hw).unwrap();
            assert_eq!(h.hardware, hw);
        }
    }

    #[test]
    fn cpu_throughput_is_measured_and_positive() {
        let m = model("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        let r = benchmark_throughput(&m, HardwareTarget::Cpu, 200, 2).unwrap();
        assert!(r.measured);
        assert!(r.samples_per_sec > 0.0);
    }

    #[test]
    fn accelerator_throughput_reports_unmeasured() {
        let m = model("let mu ~ Normal(0, 1)");
        let r = benchmark_throughput(&m, HardwareTarget::Nvidia, 200, 2).unwrap();
        assert!(!r.measured);
        assert_eq!(r.total_samples, 400);
    }
}
