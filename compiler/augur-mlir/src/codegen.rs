//! Lowers a probabilistic dialect [`Graph`] to TPTIR text assembly.
//!
//! The output is a single `module { func.func @entry ... }` in the generic
//! MLIR operation syntax TPTIR itself uses (see
//! `../tpt-gpu/layer3_tptc/spec/tptir_spec.md` §3.1, §7.1): `%result =
//! "namespace.op"(%operands) {attrs} : (operand_types) -> (result_types)`.
//! `augur.*` ops sit alongside `tptir.*` ops in the same module rather than
//! a separate, incompatible format, so the emitted text is what a TPTIR
//! consumer (`../tpt-gpu/layer3_tptc`) parses as an unregistered dialect
//! pending a native `augur` dialect registration there.
//!
//! Distribution values carry the opaque `!augur.dist` type; every other
//! value in this dialect is `f64` (including comparison results, matching
//! `augur_ir::eval`'s convention of representing booleans as `1.0`/`0.0`).

use crate::dialect::{Graph, Op, ScalarOp, ValueId};

/// Emit a full TPTIR text module for `graph`.
///
/// `entry_name` becomes the `func.func` symbol; `hardware` is recorded as
/// the `augur.hardware` attribute so downstream dispatch (see
/// `../tpt-gpu/layer3_tptc/rust/src/dispatch.rs`) can pick a target-specific
/// kernel variant without re-parsing the whole module.
pub fn emit_tptir(graph: &Graph, entry_name: &str, hardware: &str) -> String {
    let mut out = String::new();
    let priors = graph
        .prior_order
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!(
        "module {{\n  func.func @{entry_name}() attributes {{tptir.kernel, augur.prior_order = [{priors}], augur.hardware = \"{hardware}\"}} {{\n"
    ));
    out.push_str("    ^entry:\n");
    emit_ops(&graph.ops, &mut out, 6);
    out.push_str("      tptir.return\n");
    out.push_str("  }\n}\n");
    out
}

fn emit_ops(ops: &[Op], out: &mut String, indent: usize) {
    let pad = " ".repeat(indent);
    for op in ops {
        match op {
            Op::Constant { result, value } => {
                out.push_str(&format!(
                    "{pad}%{result} = \"augur.constant\"() {{value = {value} : f64}} : () -> f64\n"
                ));
            }
            Op::Dist { result, dist } => {
                let params = operand_list(&dist.params);
                let ptypes = type_list(dist.params.len(), "f64");
                out.push_str(&format!(
                    "{pad}%{result} = \"augur.dist.{}\"({params}) : ({ptypes}) -> !augur.dist\n",
                    dist.family.op_name()
                ));
            }
            Op::Sample { result, dist, name } => {
                out.push_str(&format!(
                    "{pad}%{result} = \"augur.sample\"(%{dist}) {{name = \"{name}\"}} : (!augur.dist) -> f64\n"
                ));
            }
            Op::Scalar { result, op: sop } => {
                let (mnemonic, operands) = scalar_parts(sop);
                let types = type_list(operands.len(), "f64");
                let operand_str = operand_list(&operands);
                out.push_str(&format!(
                    "{pad}%{result} = \"{mnemonic}\"({operand_str}) : ({types}) -> f64\n"
                ));
            }
            Op::Observe { dist, value } => {
                out.push_str(&format!(
                    "{pad}\"augur.observe\"(%{dist}, %{value}) : (!augur.dist, f64) -> ()\n"
                ));
            }
            Op::Let { name, value } => {
                out.push_str(&format!(
                    "{pad}\"augur.let\"(%{value}) {{name = \"{name}\"}} : (f64) -> ()\n"
                ));
            }
            Op::Cond {
                cond,
                then_ops,
                else_ops,
            } => {
                out.push_str(&format!("{pad}\"augur.cond\"(%{cond}) ({{\n"));
                emit_ops(then_ops, out, indent + 2);
                out.push_str(&format!("{pad}}}, {{\n"));
                emit_ops(else_ops, out, indent + 2);
                out.push_str(&format!("{pad}}}) : (f64) -> ()\n"));
            }
        }
    }
}

fn scalar_parts(sop: &ScalarOp) -> (&'static str, Vec<ValueId>) {
    match *sop {
        ScalarOp::Add(a, b) => ("augur.addf", vec![a, b]),
        ScalarOp::Sub(a, b) => ("augur.subf", vec![a, b]),
        ScalarOp::Mul(a, b) => ("augur.mulf", vec![a, b]),
        ScalarOp::Div(a, b) => ("augur.divf", vec![a, b]),
        ScalarOp::Neg(a) => ("augur.negf", vec![a]),
        ScalarOp::CmpGt(a, b) => ("augur.cmpf_gt", vec![a, b]),
        ScalarOp::CmpLt(a, b) => ("augur.cmpf_lt", vec![a, b]),
        ScalarOp::CmpGe(a, b) => ("augur.cmpf_ge", vec![a, b]),
        ScalarOp::CmpLe(a, b) => ("augur.cmpf_le", vec![a, b]),
        ScalarOp::CmpEq(a, b) => ("augur.cmpf_eq", vec![a, b]),
        ScalarOp::CmpNe(a, b) => ("augur.cmpf_ne", vec![a, b]),
    }
}

fn operand_list(ids: &[ValueId]) -> String {
    ids.iter().map(|id| format!("%{id}")).collect::<Vec<_>>().join(", ")
}

fn type_list(count: usize, ty: &str) -> String {
    std::iter::repeat(ty).take(count).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::build_graph;
    use crate::passes::default_pipeline;
    use augur_frontend::parse;

    fn compile(src: &str) -> String {
        let parsed = parse(src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let lowered = augur_ir::lower(&parsed.program);
        assert!(
            !lowered.diagnostics.iter().any(augur_ir::Diagnostic::is_error),
            "{:?}",
            lowered.diagnostics
        );
        let mut graph = build_graph(&lowered.model);
        default_pipeline().run(&mut graph);
        emit_tptir(&graph, "model", "cpu")
    }

    #[test]
    fn emits_module_with_sample_and_observe() {
        let text = compile("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        assert!(text.contains("module {"));
        assert!(text.contains("func.func @model"));
        assert!(text.contains("augur.prior_order = [\"mu\"]"));
        assert!(text.contains("\"augur.dist.normal\""));
        assert!(text.contains("\"augur.sample\""));
        assert!(text.contains("\"augur.observe\""));
        assert!(text.contains("tptir.return"));
    }

    #[test]
    fn emits_cond_region_for_if() {
        let src = "let mu ~ Normal(0,1)\nif mu > 0 { observe Normal(mu,1) = 1 } else { observe Normal(mu,1) = -1 }";
        let text = compile(src);
        assert!(text.contains("\"augur.cond\""));
        assert!(text.contains("\"augur.cmpf_gt\""));
    }

    #[test]
    fn records_hardware_attribute() {
        let text = compile("let mu ~ Normal(0, 1)");
        assert!(text.contains("augur.hardware = \"cpu\""));
    }
}
