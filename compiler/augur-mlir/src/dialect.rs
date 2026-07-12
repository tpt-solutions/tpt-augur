//! The `augur` probabilistic dialect: distributions, sampling, observation,
//! and inference-graph control-flow nodes, expressed as an SSA op graph.
//!
//! Ops are modeled after MLIR's generic operation form (see
//! `../tpt-gpu/layer3_tptc/spec/tptir_spec.md` §3.1, §8: dialects are a
//! namespace of ops following `%result = "namespace.op"(%operands) {attrs}
//! : (operand_types) -> (result_types)`), so [`crate::codegen`] can emit this
//! graph as `augur.*` ops alongside `tptir.*` in a single TPTIR module rather
//! than inventing an incompatible textual format.

use serde::{Deserialize, Serialize};

/// SSA value id, unique within a [`Graph`] (including nested `Cond` regions).
pub type ValueId = u32;

/// A distribution family recognized by the `augur.dist.*` ops. Mirrors
/// [`augur_ir::known_dist_arity`]'s name set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistFamily {
    Normal,
    HalfNormal,
    Beta,
    Gamma,
    Uniform,
    Exponential,
    Binomial,
    Poisson,
    Bernoulli,
}

impl DistFamily {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Normal" => Some(Self::Normal),
            "HalfNormal" => Some(Self::HalfNormal),
            "Beta" => Some(Self::Beta),
            "Gamma" => Some(Self::Gamma),
            "Uniform" => Some(Self::Uniform),
            "Exponential" => Some(Self::Exponential),
            "Binomial" => Some(Self::Binomial),
            "Poisson" => Some(Self::Poisson),
            "Bernoulli" => Some(Self::Bernoulli),
            _ => None,
        }
    }

    /// The `augur.dist.<op_name>` suffix.
    pub fn op_name(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::HalfNormal => "half_normal",
            Self::Beta => "beta",
            Self::Gamma => "gamma",
            Self::Uniform => "uniform",
            Self::Exponential => "exponential",
            Self::Binomial => "binomial",
            Self::Poisson => "poisson",
            Self::Bernoulli => "bernoulli",
        }
    }
}

/// A distribution instance: a family applied to SSA parameter values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistInstance {
    pub family: DistFamily,
    pub params: Vec<ValueId>,
}

/// Deterministic scalar operations over SSA values (mirrors
/// `augur_frontend::{BinOp, CmpOp}`, plus unary negate).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ScalarOp {
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    Neg(ValueId),
    CmpGt(ValueId, ValueId),
    CmpLt(ValueId, ValueId),
    CmpGe(ValueId, ValueId),
    CmpLe(ValueId, ValueId),
    CmpEq(ValueId, ValueId),
    CmpNe(ValueId, ValueId),
}

/// A single op in the probabilistic inference graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// `%result = augur.constant <value>`
    Constant { result: ValueId, value: f64 },
    /// `%result = augur.dist.<family>(%params...)` — instantiates a distribution.
    Dist { result: ValueId, dist: DistInstance },
    /// `%result = augur.sample(%dist) {name}` — declares a prior random variable.
    /// The variable participates in the model's `prior_order` sample vector.
    Sample {
        result: ValueId,
        dist: ValueId,
        name: String,
    },
    /// Deterministic scalar arithmetic/comparison.
    Scalar { result: ValueId, op: ScalarOp },
    /// `augur.observe(%dist, %value)` — conditions the model, accumulating
    /// log-density into the joint. No result.
    Observe { dist: ValueId, value: ValueId },
    /// `augur.let %name = %value` — deterministic named binding.
    Let { name: String, value: ValueId },
    /// `augur.cond(%cond) then { ... } else { ... }` — an inference-graph
    /// branch node: deterministic control flow that gates which nested ops
    /// (samples/observes/lets) execute at run time.
    Cond {
        cond: ValueId,
        then_ops: Vec<Op>,
        else_ops: Vec<Op>,
    },
}

/// The probabilistic inference graph for a whole model: an ordered region of
/// `augur.*` ops plus the fixed prior order (the sample-vector shape the
/// runtime's inference engines sample over).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub ops: Vec<Op>,
    pub prior_order: Vec<String>,
    pub next_value: ValueId,
}

impl Graph {
    /// Total op count, including ops nested inside `Cond` branches.
    pub fn op_count(&self) -> usize {
        fn count(ops: &[Op]) -> usize {
            ops.iter()
                .map(|op| match op {
                    Op::Cond {
                        then_ops, else_ops, ..
                    } => 1 + count(then_ops) + count(else_ops),
                    _ => 1,
                })
                .sum()
        }
        count(&self.ops)
    }
}
