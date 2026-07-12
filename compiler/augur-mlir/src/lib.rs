//! Augur's probabilistic MLIR-compatible dialect: distributions, sampling,
//! and inference-graph control flow as an SSA op graph, with pre-lowering
//! optimization passes and a lowering pass to TPTIR text (spec.txt §3).
//!
//! Pipeline: `augur_ir::Model` -> [`build::build_graph`] -> [`dialect::Graph`]
//! -> [`passes::default_pipeline`] -> [`codegen::emit_tptir`] -> TPTIR text,
//! consumable by `../tpt-gpu/layer3_tptc`'s TPTIR toolchain.

pub mod build;
pub mod codegen;
pub mod dialect;
pub mod graphviz;
pub mod passes;

pub use build::build_graph;
pub use codegen::emit_tptir;
pub use dialect::{DistFamily, DistInstance, Graph, Op, ScalarOp, ValueId};
pub use graphviz::to_dot;
pub use passes::{default_pipeline, Pass, PassPipeline};

/// Convenience entry point: build the dialect graph from a type-checked
/// model, run the default optimization pipeline, and emit TPTIR text.
/// Returns the emitted text alongside the number of changes the pipeline
/// made (useful for CLI diagnostics).
pub fn compile_model_to_tptir(
    model: &augur_ir::Model,
    entry_name: &str,
    hardware: &str,
) -> (String, usize) {
    let mut graph = build_graph(model);
    let changes = default_pipeline().run(&mut graph);
    (emit_tptir(&graph, entry_name, hardware), changes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_frontend::parse;

    #[test]
    fn compile_model_to_tptir_end_to_end() {
        let parsed = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        assert!(!parsed.has_errors());
        let lowered = augur_ir::lower(&parsed.program);
        assert!(!lowered
            .diagnostics
            .iter()
            .any(augur_ir::Diagnostic::is_error));
        let (text, _changes) = compile_model_to_tptir(&lowered.model, "model", "cpu");
        assert!(text.contains("func.func @model"));
    }
}
