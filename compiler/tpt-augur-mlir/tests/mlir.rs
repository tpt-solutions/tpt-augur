//! Integration tests for the Augur MLIR-compatible probabilistic dialect:
//! graph construction, TPTIR codegen, optimization passes, and Graphviz output.

use tpt_augur_frontend::parse;
use tpt_augur_ir::lower;
use tpt_augur_mlir::{
    build_graph, compile_model_to_tptir, default_pipeline,
    dialect::{DistFamily, Graph, Op, ScalarOp},
    emit_tptir, to_dot,
};

fn graph(src: &str) -> Graph {
    let parsed = parse(src);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let lowered = lower(&parsed.program);
    assert!(
        !lowered.diagnostics.iter().any(|d| d.is_error()),
        "{:?}",
        lowered.diagnostics
    );
    build_graph(&lowered.model)
}

#[test]
fn build_emits_sample_and_observe_ops() {
    let g = graph("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    assert_eq!(g.prior_order, vec!["mu".to_string()]);
    assert!(g
        .ops
        .iter()
        .any(|op| matches!(op, Op::Sample { name, .. } if name == "mu")));
    assert!(g.ops.iter().any(|op| matches!(op, Op::Observe { .. })));
    assert!(g.ops.iter().any(|op| matches!(op, Op::Dist { .. })));
}

#[test]
fn build_lowers_if_to_cond() {
    let g = graph("let mu ~ Normal(0,1)\nif mu > 0 { observe Normal(mu,1) = 1 } else { observe Normal(mu,1) = -1 }");
    assert!(g.ops.iter().any(|op| matches!(op, Op::Cond { .. })));
}

#[test]
fn build_records_let_bindings() {
    let g = graph("let mu ~ Normal(0, 1)\nlet shifted = mu + 1");
    assert!(g
        .ops
        .iter()
        .any(|op| matches!(op, Op::Let { name, .. } if name == "shifted")));
    assert!(g.ops.iter().any(|op| matches!(
        op,
        Op::Scalar {
            op: ScalarOp::Add(..),
            ..
        }
    )));
}

#[test]
fn codegen_emits_module_and_ops() {
    let g = graph("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    let text = emit_tptir(&g, "model", "cpu");
    assert!(text.contains("module {"));
    assert!(text.contains("func.func @model"));
    assert!(text.contains("augur.prior_order = [\"mu\"]"));
    assert!(text.contains("\"augur.dist.normal\""));
    assert!(text.contains("\"augur.sample\""));
    assert!(text.contains("\"augur.observe\""));
    assert!(text.contains("tptir.return"));
}

#[test]
fn codegen_emits_cond_region() {
    let g = graph("let mu ~ Normal(0,1)\nif mu > 0 { observe Normal(mu,1) = 1 } else { observe Normal(mu,1) = -1 }");
    let text = emit_tptir(&g, "model", "cpu");
    assert!(text.contains("\"augur.cond\""));
    assert!(text.contains("\"augur.cmpf_gt\""));
}

#[test]
fn codegen_records_hardware_attribute() {
    let g = graph("let mu ~ Normal(0, 1)");
    let text = emit_tptir(&g, "model", "nvidia");
    assert!(text.contains("augur.hardware = \"nvidia\""));
}

#[test]
fn dot_contains_prior_and_observe() {
    let g = graph("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    let dot = to_dot(&g);
    assert!(dot.contains("digraph augur_inference_graph"));
    assert!(dot.contains("sample mu"));
    assert!(dot.contains("observe"));
    assert!(dot.contains("->"));
}

#[test]
fn dist_family_names_and_ops() {
    let cases = [
        ("Normal", DistFamily::Normal, "normal"),
        ("HalfNormal", DistFamily::HalfNormal, "half_normal"),
        ("Beta", DistFamily::Beta, "beta"),
        ("Gamma", DistFamily::Gamma, "gamma"),
        ("Uniform", DistFamily::Uniform, "uniform"),
        ("Exponential", DistFamily::Exponential, "exponential"),
        ("Binomial", DistFamily::Binomial, "binomial"),
        ("Poisson", DistFamily::Poisson, "poisson"),
        ("Bernoulli", DistFamily::Bernoulli, "bernoulli"),
    ];
    for (name, family, op) in cases {
        assert_eq!(DistFamily::from_name(name), Some(family));
        assert_eq!(family.op_name(), op);
    }
    assert_eq!(DistFamily::from_name("Nope"), None);
}

#[test]
fn pipeline_constant_folds_arithmetic() {
    let mut g = graph("let x = (1 + 2) * 3\nlet mu ~ Normal(x, 1)");
    let before = g.op_count();
    default_pipeline().run(&mut g);
    // The `(1+2)*3` scalar subtree should collapse to a single constant 9.
    assert!(g
        .ops
        .iter()
        .any(|op| matches!(op, Op::Constant { value, .. } if (*value - 9.0).abs() < 1e-9)));
    assert!(g.op_count() < before, "expected fewer ops after folding");
    // The sample op (a side-effecting op) must survive.
    assert!(g.ops.iter().any(|op| matches!(op, Op::Sample { .. })));
}

#[test]
fn pipeline_preserves_model_semantics() {
    let mut g = graph("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
    default_pipeline().run(&mut g);
    // Pipeline is a no-op structurally here but must keep the sample/observe.
    assert!(g.ops.iter().any(|op| matches!(op, Op::Sample { .. })));
    assert!(g.ops.iter().any(|op| matches!(op, Op::Observe { .. })));
}

#[test]
fn compile_to_tptir_end_to_end() {
    let m = lower(&parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5").program);
    let (text, _changes) = compile_model_to_tptir(&m.model, "model", "cpu");
    assert!(text.contains("func.func @model"));
    assert!(text.contains("\"augur.sample\""));
}
