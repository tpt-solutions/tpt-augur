//! Builds a probabilistic dialect [`Graph`] from an already type-checked
//! [`augur_ir::Model`]. Callers must lower via `augur_ir::lower` and confirm
//! there are no error diagnostics first — this stage assumes a well-formed
//! model (undeclared variables, unknown distributions, and arity mismatches
//! are all rejected earlier by `augur-ir`'s checker).

use std::collections::HashMap;

use augur_frontend::{BinOp, CmpOp, Expr};
use augur_ir::{Model, ModelItem};

use crate::dialect::{DistFamily, DistInstance, Graph, Op, ScalarOp, ValueId};

struct Builder {
    ops: Vec<Op>,
    next: ValueId,
    env: HashMap<String, ValueId>,
}

impl Builder {
    fn fresh(&mut self) -> ValueId {
        let v = self.next;
        self.next += 1;
        v
    }

    fn lower_expr(&mut self, expr: &Expr) -> ValueId {
        match expr {
            Expr::Num(v) => {
                let r = self.fresh();
                self.ops.push(Op::Constant { result: r, value: *v });
                r
            }
            Expr::Var(name) => *self
                .env
                .get(name)
                .unwrap_or_else(|| panic!("undeclared variable `{name}` reached MLIR build stage")),
            Expr::Neg(inner) => {
                let v = self.lower_expr(inner);
                let r = self.fresh();
                self.ops.push(Op::Scalar {
                    result: r,
                    op: ScalarOp::Neg(v),
                });
                r
            }
            Expr::Paren(inner) => self.lower_expr(inner),
            Expr::Bin { op, lhs, rhs } => {
                let a = self.lower_expr(lhs);
                let b = self.lower_expr(rhs);
                let r = self.fresh();
                let sop = match op {
                    BinOp::Add => ScalarOp::Add(a, b),
                    BinOp::Sub => ScalarOp::Sub(a, b),
                    BinOp::Mul => ScalarOp::Mul(a, b),
                    BinOp::Div => ScalarOp::Div(a, b),
                };
                self.ops.push(Op::Scalar { result: r, op: sop });
                r
            }
            Expr::Cmp { op, lhs, rhs } => {
                let a = self.lower_expr(lhs);
                let b = self.lower_expr(rhs);
                let r = self.fresh();
                let sop = match op {
                    CmpOp::Gt => ScalarOp::CmpGt(a, b),
                    CmpOp::Lt => ScalarOp::CmpLt(a, b),
                    CmpOp::Ge => ScalarOp::CmpGe(a, b),
                    CmpOp::Le => ScalarOp::CmpLe(a, b),
                    CmpOp::Eq => ScalarOp::CmpEq(a, b),
                    CmpOp::Ne => ScalarOp::CmpNe(a, b),
                };
                self.ops.push(Op::Scalar { result: r, op: sop });
                r
            }
            Expr::Call { name, .. } => {
                panic!("distribution constructor `{name}(..)` used as a scalar value")
            }
        }
    }

    fn lower_dist(&mut self, expr: &Expr) -> ValueId {
        match expr {
            Expr::Call { name, args, .. } => {
                let family = DistFamily::from_name(name)
                    .unwrap_or_else(|| panic!("unknown distribution `{name}` reached MLIR build stage"));
                let params = args.iter().map(|a| self.lower_expr(a)).collect();
                let r = self.fresh();
                self.ops.push(Op::Dist {
                    result: r,
                    dist: DistInstance { family, params },
                });
                r
            }
            _ => panic!("expected a distribution constructor (e.g. `Normal(0, 1)`)"),
        }
    }

    /// Lower a nested block (an `if`/`else` body) in its own ops buffer,
    /// sharing the variable environment with the enclosing scope.
    fn lower_sub_block(&mut self, items: &[ModelItem]) -> Vec<Op> {
        let outer = std::mem::take(&mut self.ops);
        self.lower_items(items);
        std::mem::replace(&mut self.ops, outer)
    }

    fn lower_items(&mut self, items: &[ModelItem]) {
        for item in items {
            match item {
                ModelItem::Prior { name, dist, .. } => {
                    let d = self.lower_dist(dist);
                    let r = self.fresh();
                    self.ops.push(Op::Sample {
                        result: r,
                        dist: d,
                        name: name.clone(),
                    });
                    self.env.insert(name.clone(), r);
                }
                ModelItem::Let { name, value, .. } => {
                    let v = self.lower_expr(value);
                    self.ops.push(Op::Let {
                        name: name.clone(),
                        value: v,
                    });
                    self.env.insert(name.clone(), v);
                }
                ModelItem::Observe { dist, value, .. } => {
                    let d = self.lower_dist(dist);
                    let v = self.lower_expr(value);
                    self.ops.push(Op::Observe { dist: d, value: v });
                }
                ModelItem::If {
                    cond,
                    then_items,
                    else_items,
                    ..
                } => {
                    let c = self.lower_expr(cond);
                    let then_ops = self.lower_sub_block(then_items);
                    let else_ops = self.lower_sub_block(else_items);
                    self.ops.push(Op::Cond {
                        cond: c,
                        then_ops,
                        else_ops,
                    });
                }
            }
        }
    }
}

/// Build a probabilistic dialect [`Graph`] from a type-checked [`Model`].
pub fn build_graph(model: &Model) -> Graph {
    let mut b = Builder {
        ops: Vec::new(),
        next: 0,
        env: HashMap::new(),
    };
    b.lower_items(&model.items);
    Graph {
        ops: b.ops,
        prior_order: model.prior_order.clone(),
        next_value: b.next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_frontend::parse;

    fn build(src: &str) -> Graph {
        let parsed = parse(src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let lowered = augur_ir::lower(&parsed.program);
        assert!(
            !lowered.diagnostics.iter().any(augur_ir::Diagnostic::is_error),
            "{:?}",
            lowered.diagnostics
        );
        build_graph(&lowered.model)
    }

    #[test]
    fn prior_and_observe_produce_sample_and_observe_ops() {
        let g = build("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
        assert_eq!(g.prior_order, vec!["mu".to_string()]);
        let has_sample = g.ops.iter().any(|op| matches!(op, Op::Sample { name, .. } if name == "mu"));
        let has_observe = g.ops.iter().any(|op| matches!(op, Op::Observe { .. }));
        assert!(has_sample && has_observe);
    }

    #[test]
    fn if_lowers_to_cond_op() {
        let src = "let mu ~ Normal(0,1)\nif mu > 0 { observe Normal(mu,1) = 1 } else { observe Normal(mu,1) = -1 }";
        let g = build(src);
        assert!(g.ops.iter().any(|op| matches!(op, Op::Cond { .. })));
    }
}
