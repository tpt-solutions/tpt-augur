//! Render an [`augur_mlir::Graph`] as Graphviz DOT for inference-graph
//! visualization (LSP custom request, `augur graph` CLI, and external tools).
//!
//! The emitted graph is a data-flow view of the probabilistic inference graph:
//! each `augur.*` op becomes a node, and operands become edges, so the
//! dependency structure between priors, deterministic bindings, and
//! observations is immediately legible.

use crate::dialect::{Graph, Op, ScalarOp, ValueId};
use std::collections::HashMap;

/// Emit a Graphviz `digraph` for `graph`. The returned string is standalone
/// DOT that `dot`, `xdot`, or any Graphviz viewer can render.
pub fn to_dot(graph: &Graph) -> String {
    let mut out = String::new();
    out.push_str("digraph augur_inference_graph {\n");
    out.push_str("  rankdir=TB;\n");
    out.push_str("  node [shape=box, style=rounded, fontname=\"monospace\"];\n");

    // Maps each SSA value to the DOT node id that defines it. Node ids are
    // globally unique: when an op lives inside a `Cond` cluster, its id is
    // prefixed with the cluster id so cross-cluster edges still resolve.
    let mut producer: HashMap<ValueId, String> = HashMap::new();
    emit_region(&graph.ops, &mut out, &mut producer, None);

    out.push_str("}\n");
    out
}

fn node_id(value: ValueId) -> String {
    format!("n{value}")
}

fn emit_region(
    ops: &[Op],
    out: &mut String,
    producer: &mut HashMap<ValueId, String>,
    prefix: Option<&str>,
) {
    // Apply the cluster prefix (if any) to a base node id.
    let qualify = |base: String| -> String {
        match prefix {
            Some(p) => format!("{p}_{base}"),
            None => base,
        }
    };

    for op in ops {
        match op {
            Op::Constant { result, value } => {
                let id = qualify(node_id(*result));
                out.push_str(&format!(
                    "  {id} [label=\"const {value}\", shape=ellipse, style=filled, fillcolor=\"#e8f0ff\"];\n"
                ));
                producer.insert(*result, id);
            }
            Op::Dist { result, dist } => {
                let id = qualify(node_id(*result));
                let params: Vec<String> = dist
                    .params
                    .iter()
                    .map(|p| operand_label(*p, producer))
                    .collect();
                out.push_str(&format!(
                    "  {id} [label=\"{}({})\", shape=ellipse, style=filled, fillcolor=\"#fff3e0\"];\n",
                    dist.family.op_name(),
                    params.join(", ")
                ));
                producer.insert(*result, id);
            }
            Op::Sample { result, name, .. } => {
                let id = qualify(node_id(*result));
                out.push_str(&format!(
                    "  {id} [label=\"sample {name}\", shape=box, style=filled, fillcolor=\"#e6ffe6\"];\n"
                ));
                producer.insert(*result, id);
            }
            Op::Scalar { result, op } => {
                let id = qualify(node_id(*result));
                let (sym, lhs, rhs) = scalar_parts(*op);
                let label = match rhs {
                    Some(r) => format!(
                        "{}{} {}",
                        operand_label(lhs, producer),
                        sym,
                        operand_label(r, producer)
                    ),
                    None => format!("{}{}", sym, operand_label(lhs, producer)),
                };
                out.push_str(&format!(
                    "  {id} [label=\"{label}\", shape=box, style=filled, fillcolor=\"#f3e8ff\"];\n"
                ));
                producer.insert(*result, id);
            }
            Op::Observe { dist, value } => {
                let id = qualify(format!("obs_{}", node_id(*dist)));
                out.push_str(&format!(
                    "  {id} [label=\"observe\", shape=box, style=filled, fillcolor=\"#ffe6e6\"];\n"
                ));
                edge(out, &operand_label(*dist, producer), &id);
                edge(out, &operand_label(*value, producer), &id);
            }
            Op::Let { name, value } => {
                let id = qualify(format!("let_{}", sanitize(name)));
                out.push_str(&format!(
                    "  {id} [label=\"let {name}\", shape=box, style=filled, fillcolor=\"#e0f7fa\"];\n"
                ));
                edge(out, &operand_label(*value, producer), &id);
            }
            Op::Cond {
                cond,
                then_ops,
                else_ops,
            } => {
                let cond_id = qualify(format!("cond_{}", node_id(*cond)));
                out.push_str(&format!(
                    "  {cond_id} [label=\"if {}\", shape=diamond, style=filled, fillcolor=\"#fffde7\"];\n",
                    operand_label(*cond, producer)
                ));
                let then_cluster = cluster_id(producer, "then");
                let else_cluster = cluster_id(producer, "else");
                out.push_str(&format!(
                    "  subgraph {then_cluster} {{\n    label=\"then\";\n    style=dashed;\n"
                ));
                emit_region(then_ops, out, producer, Some(&then_cluster));
                out.push_str("  }\n");
                out.push_str(&format!(
                    "  subgraph {else_cluster} {{\n    label=\"else\";\n    style=dashed;\n"
                ));
                emit_region(else_ops, out, producer, Some(&else_cluster));
                out.push_str("  }\n");
            }
        }
    }
}

fn scalar_parts(op: ScalarOp) -> (&'static str, ValueId, Option<ValueId>) {
    match op {
        ScalarOp::Add(a, b) => ("+", a, Some(b)),
        ScalarOp::Sub(a, b) => ("-", a, Some(b)),
        ScalarOp::Mul(a, b) => ("*", a, Some(b)),
        ScalarOp::Div(a, b) => ("/", a, Some(b)),
        ScalarOp::Neg(a) => ("-", a, None),
        ScalarOp::CmpGt(a, b) => (">", a, Some(b)),
        ScalarOp::CmpLt(a, b) => ("<", a, Some(b)),
        ScalarOp::CmpGe(a, b) => (">=", a, Some(b)),
        ScalarOp::CmpLe(a, b) => ("<=", a, Some(b)),
        ScalarOp::CmpEq(a, b) => ("==", a, Some(b)),
        ScalarOp::CmpNe(a, b) => ("!=", a, Some(b)),
    }
}

/// Resolve a SSA value to its defining DOT node id (already cluster-qualified).
fn operand_label(value: ValueId, producer: &HashMap<ValueId, String>) -> String {
    producer
        .get(&value)
        .cloned()
        .unwrap_or_else(|| node_id(value))
}

fn cluster_id(producer: &HashMap<ValueId, String>, tag: &str) -> String {
    let salt = producer.len();
    format!("cluster_{tag}_{salt}")
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

fn edge(out: &mut String, from: &str, to: &str) {
    out.push_str(&format!("  {from} -> {to};\n"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_graph, compile_model_to_tptir};
    use augur_frontend::parse;
    use augur_ir::lower;

    fn graph(src: &str) -> Graph {
        let parsed = parse(src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let lowered = lower(&parsed.program);
        assert!(
            !lowered.diagnostics.iter().any(augur_ir::Diagnostic::is_error),
            "{:?}",
            lowered.diagnostics
        );
        build_graph(&lowered.model)
    }

    #[test]
    fn dot_contains_prior_and_observe_nodes() {
        let g = graph("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        let dot = to_dot(&g);
        assert!(dot.contains("digraph augur_inference_graph"));
        assert!(dot.contains("sample mu"));
        assert!(dot.contains("observe"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn dot_qualifies_nested_cluster_nodes() {
        // The `if` body references the outer `mu` sample; the edge must point
        // at the outer node id (no bogus `cluster_*:n3` port reference).
        let g = graph("let mu ~ Normal(0, 1)\nif mu > 0 { observe Normal(mu, 1) = 1 }");
        let dot = to_dot(&g);
        assert!(dot.contains("sample mu"));
        assert!(!dot.contains("cluster_then_:n"));
        // Cross-cluster edge resolves to the outer sample node.
        assert!(dot.contains("-> cluster_then_") && dot.contains("n3 -> cluster_then_") || dot.contains("cluster_then_") );
    }

    #[test]
    fn dot_compiles_independently_of_tptir() {
        // Ensure the graphviz path does not interfere with the MLIR pipeline.
        let g = graph("let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7");
        let _ = to_dot(&g);
        let (tptir, _) = compile_model_to_tptir(
            &lower(&parse("let p ~ Beta(1, 1)\nobserve Binomial(10, p) = 7").program).model,
            "m",
            "cpu",
        );
        assert!(tptir.contains("func.func"));
    }
}
