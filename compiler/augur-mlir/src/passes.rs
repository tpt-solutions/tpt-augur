//! Probabilistic-specific pre-lowering optimization passes over the `augur`
//! dialect [`Graph`], run before [`crate::codegen`] emits TPTIR.

use std::collections::{HashMap, HashSet};

use crate::dialect::{Graph, Op, ScalarOp, ValueId};

pub trait Pass {
    fn name(&self) -> &str;
    /// Run the pass, mutating `graph` in place. Returns the number of
    /// changes made (0 means the pass was a no-op on this graph).
    fn run(&self, graph: &mut Graph) -> usize;
}

/// Folds deterministic scalar arithmetic/comparisons over compile-time
/// constants into a single `augur.constant`, e.g. `Neg(Add(1, 2))` -> `-3`.
pub struct ConstantFoldPass;

impl Pass for ConstantFoldPass {
    fn name(&self) -> &str {
        "constant-fold"
    }
    fn run(&self, graph: &mut Graph) -> usize {
        let mut changes = 0;
        fold_ops(&mut graph.ops, &mut changes);
        changes
    }
}

fn fold_ops(ops: &mut [Op], changes: &mut usize) {
    let mut consts: HashMap<ValueId, f64> = HashMap::new();
    for op in ops.iter_mut() {
        match op {
            Op::Constant { result, value } => {
                consts.insert(*result, *value);
            }
            Op::Scalar { result, op: sop } => {
                if let Some(folded) = try_fold_scalar(sop, &consts) {
                    let result = *result;
                    *op = Op::Constant {
                        result,
                        value: folded,
                    };
                    consts.insert(result, folded);
                    *changes += 1;
                }
            }
            Op::Cond {
                then_ops, else_ops, ..
            } => {
                fold_ops(then_ops, changes);
                fold_ops(else_ops, changes);
            }
            _ => {}
        }
    }
}

fn try_fold_scalar(sop: &ScalarOp, consts: &HashMap<ValueId, f64>) -> Option<f64> {
    let b2f = |b: bool| if b { 1.0 } else { 0.0 };
    Some(match *sop {
        ScalarOp::Add(a, b) => consts.get(&a)? + consts.get(&b)?,
        ScalarOp::Sub(a, b) => consts.get(&a)? - consts.get(&b)?,
        ScalarOp::Mul(a, b) => consts.get(&a)? * consts.get(&b)?,
        ScalarOp::Div(a, b) => consts.get(&a)? / consts.get(&b)?,
        ScalarOp::Neg(a) => -*consts.get(&a)?,
        ScalarOp::CmpGt(a, b) => b2f(consts.get(&a)? > consts.get(&b)?),
        ScalarOp::CmpLt(a, b) => b2f(consts.get(&a)? < consts.get(&b)?),
        ScalarOp::CmpGe(a, b) => b2f(consts.get(&a)? >= consts.get(&b)?),
        ScalarOp::CmpLe(a, b) => b2f(consts.get(&a)? <= consts.get(&b)?),
        ScalarOp::CmpEq(a, b) => b2f(consts.get(&a)? == consts.get(&b)?),
        ScalarOp::CmpNe(a, b) => b2f(consts.get(&a)? != consts.get(&b)?),
    })
}

/// Collapses `augur.cond` inference-graph nodes whose condition folds to a
/// compile-time constant, inlining the statically-selected branch and
/// dropping the other one entirely (and every sample/observe it contained).
/// Mirrors `augur_ir::eval`'s truthiness convention: nonzero-positive is true.
pub struct ConstantBranchPass;

impl Pass for ConstantBranchPass {
    fn name(&self) -> &str {
        "constant-branch"
    }
    fn run(&self, graph: &mut Graph) -> usize {
        let mut changes = 0;
        graph.ops = simplify_conds(std::mem::take(&mut graph.ops), &mut changes);
        changes
    }
}

fn simplify_conds(ops: Vec<Op>, changes: &mut usize) -> Vec<Op> {
    let mut consts: HashMap<ValueId, f64> = HashMap::new();
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            Op::Constant { result, value } => {
                consts.insert(result, value);
                out.push(Op::Constant { result, value });
            }
            Op::Cond {
                cond,
                then_ops,
                else_ops,
            } => {
                let then_ops = simplify_conds(then_ops, changes);
                let else_ops = simplify_conds(else_ops, changes);
                match consts.get(&cond) {
                    Some(v) => {
                        *changes += 1;
                        out.extend(if *v > 0.0 { then_ops } else { else_ops });
                    }
                    None => out.push(Op::Cond {
                        cond,
                        then_ops,
                        else_ops,
                    }),
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Removes `augur.constant`/scalar/`augur.dist.*` ops whose result is never
/// referenced by a live op (a sample, observe, let-binding, branch condition,
/// or another live scalar/dist op). Samples, observes, lets, and conds are
/// never removed: they carry side effects (declaring a prior, conditioning
/// the model, binding a name, or gating other ops) even with no data result.
pub struct DeadValueEliminationPass;

impl Pass for DeadValueEliminationPass {
    fn name(&self) -> &str {
        "dead-value-elimination"
    }
    fn run(&self, graph: &mut Graph) -> usize {
        let mut used = HashSet::new();
        collect_used(&graph.ops, &mut used);
        let mut changes = 0;
        prune_ops(&mut graph.ops, &used, &mut changes);
        changes
    }
}

fn collect_used(ops: &[Op], used: &mut HashSet<ValueId>) {
    for op in ops {
        match op {
            Op::Constant { .. } => {}
            Op::Dist { dist, .. } => {
                for p in &dist.params {
                    used.insert(*p);
                }
            }
            Op::Sample { dist, .. } => {
                used.insert(*dist);
            }
            Op::Scalar { op: sop, .. } => collect_scalar_operands(sop, used),
            Op::Observe { dist, value } => {
                used.insert(*dist);
                used.insert(*value);
            }
            Op::Let { value, .. } => {
                used.insert(*value);
            }
            Op::Cond {
                cond,
                then_ops,
                else_ops,
            } => {
                used.insert(*cond);
                collect_used(then_ops, used);
                collect_used(else_ops, used);
            }
        }
    }
}

fn collect_scalar_operands(sop: &ScalarOp, used: &mut HashSet<ValueId>) {
    match *sop {
        ScalarOp::Neg(a) => {
            used.insert(a);
        }
        ScalarOp::Add(a, b)
        | ScalarOp::Sub(a, b)
        | ScalarOp::Mul(a, b)
        | ScalarOp::Div(a, b)
        | ScalarOp::CmpGt(a, b)
        | ScalarOp::CmpLt(a, b)
        | ScalarOp::CmpGe(a, b)
        | ScalarOp::CmpLe(a, b)
        | ScalarOp::CmpEq(a, b)
        | ScalarOp::CmpNe(a, b) => {
            used.insert(a);
            used.insert(b);
        }
    }
}

fn prune_ops(ops: &mut Vec<Op>, used: &HashSet<ValueId>, changes: &mut usize) {
    for op in ops.iter_mut() {
        if let Op::Cond {
            then_ops, else_ops, ..
        } = op
        {
            prune_ops(then_ops, used, changes);
            prune_ops(else_ops, used, changes);
        }
    }
    let before = ops.len();
    ops.retain(|op| match op {
        Op::Constant { result, .. } | Op::Dist { result, .. } | Op::Scalar { result, .. } => {
            used.contains(result)
        }
        _ => true,
    });
    *changes += before - ops.len();
}

pub struct PassPipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl PassPipeline {
    pub fn new() -> Self {
        PassPipeline { passes: Vec::new() }
    }

    pub fn add(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run every pass in order, returning the total number of changes made.
    pub fn run(&self, graph: &mut Graph) -> usize {
        let mut total = 0;
        for pass in &self.passes {
            total += pass.run(graph);
        }
        total
    }
}

impl Default for PassPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Default pre-lowering pipeline: constant-fold -> constant-branch -> dce.
/// Branch elimination runs after folding so it can see folded conditions;
/// DCE runs last to clean up whatever either pass made unreachable.
pub fn default_pipeline() -> PassPipeline {
    let mut p = PassPipeline::new();
    p.add(Box::new(ConstantFoldPass));
    p.add(Box::new(ConstantBranchPass));
    p.add(Box::new(DeadValueEliminationPass));
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::build_graph;
    use augur_frontend::parse;

    fn graph(src: &str) -> Graph {
        let parsed = parse(src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let lowered = augur_ir::lower(&parsed.program);
        assert!(
            !lowered
                .diagnostics
                .iter()
                .any(augur_ir::Diagnostic::is_error),
            "{:?}",
            lowered.diagnostics
        );
        build_graph(&lowered.model)
    }

    #[test]
    fn constant_fold_collapses_arithmetic() {
        let mut g = graph("let x = (1 + 2) * 3\nlet mu ~ Normal(x, 1)");
        let changes = ConstantFoldPass.run(&mut g);
        assert!(changes > 0);
        let has_folded = g
            .ops
            .iter()
            .any(|op| matches!(op, Op::Constant { value, .. } if (*value - 9.0).abs() < 1e-9));
        assert!(has_folded, "{:?}", g.ops);
    }

    #[test]
    fn constant_branch_drops_dead_branch() {
        let src = "if 1 > 0 { let a = 1 } else { let b = 2 }";
        let mut g = graph(src);
        ConstantFoldPass.run(&mut g);
        let changes = ConstantBranchPass.run(&mut g);
        assert!(changes > 0);
        assert!(!g.ops.iter().any(|op| matches!(op, Op::Cond { .. })));
        assert!(g
            .ops
            .iter()
            .any(|op| matches!(op, Op::Let{name,..} if name == "a")));
        assert!(!g
            .ops
            .iter()
            .any(|op| matches!(op, Op::Let{name,..} if name == "b")));
    }

    #[test]
    fn dce_removes_unused_constants() {
        let mut g = graph("let mu ~ Normal(0, 1)");
        // Inject an unused constant to exercise DCE deterministically.
        let unused = g.next_value;
        g.next_value += 1;
        g.ops.insert(
            0,
            Op::Constant {
                result: unused,
                value: 42.0,
            },
        );
        let changes = DeadValueEliminationPass.run(&mut g);
        assert!(changes > 0);
        assert!(!g
            .ops
            .iter()
            .any(|op| matches!(op, Op::Constant{result,..} if *result==unused)));
    }

    #[test]
    fn default_pipeline_runs_end_to_end() {
        let mut g = graph("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        default_pipeline().run(&mut g);
        assert!(g.ops.iter().any(|op| matches!(op, Op::Sample { .. })));
    }
}
