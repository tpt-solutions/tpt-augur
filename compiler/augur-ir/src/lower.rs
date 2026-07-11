//! Typed intermediate representation bridging the frontend AST and the runtime.
//!
//! Lowering turns an [`augur_frontend::Program`] into a [`Model`] — an ordered
//! list of items (priors, deterministic bindings, observations, and conditional
//! blocks). The model owns the *uncertainty propagation* story: parameter and
//! deterministic expressions are evaluated in an environment of sampled values,
//! so uncertainty flows naturally through `+`, `-`, `*`, `/` and nested calls.

use std::collections::HashMap;

use augur_frontend::{BinOp, CmpOp, Expr, Program, Span, Stmt};
use augur_std::Dist;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelItem {
    /// A random variable with a prior distribution. Sampled during inference.
    Prior {
        name: String,
        dist: Expr,
        span: Span,
    },
    /// A deterministic binding whose value carries the uncertainty of its inputs.
    Let {
        name: String,
        value: Expr,
        span: Span,
    },
    /// A likelihood / conditioning statement.
    Observe { dist: Expr, value: Expr, span: Span },
    /// Deterministic control flow that gates the enclosed items at run time.
    If {
        cond: Expr,
        then_items: Vec<ModelItem>,
        else_items: Vec<ModelItem>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub items: Vec<ModelItem>,
    /// Names of every prior variable, in declaration order. This is the fixed
    /// shape of the vector sampled by the inference engines.
    pub prior_order: Vec<String>,
}

pub struct LowerResult {
    pub model: Model,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>, span: Span) -> Self {
        Diagnostic {
            severity: Severity::Error,
            message: message.into(),
            span,
        }
    }
    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            message: message.into(),
            span,
        }
    }
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// Environment mapping bound names to concrete `f64` values.
pub type Env = HashMap<String, f64>;

/// Lower an AST program into a typed [`Model`], reporting type errors and
/// warnings about degenerate parameters.
pub fn lower(program: &Program) -> LowerResult {
    let mut diagnostics = Vec::new();
    let mut prior_order = Vec::new();
    let mut bound: Vec<String> = Vec::new();
    let mut items = Vec::new();

    for stmt in &program.statements {
        lower_stmt(
            stmt,
            &mut items,
            &mut bound,
            &mut prior_order,
            &mut diagnostics,
        );
    }

    LowerResult {
        model: Model { items, prior_order },
        diagnostics,
    }
}

fn lower_stmt(
    stmt: &Stmt,
    out: &mut Vec<ModelItem>,
    bound: &mut Vec<String>,
    prior_order: &mut Vec<String>,
    diags: &mut Vec<Diagnostic>,
) {
    match stmt {
        Stmt::Prior { name, dist, span } => {
            check_dist_expr(dist, bound, diags);
            if bound.contains(name) {
                diags.push(Diagnostic::error(
                    format!("variable `{name}` is already defined"),
                    *span,
                ));
            }
            bound.push(name.clone());
            prior_order.push(name.clone());
            out.push(ModelItem::Prior {
                name: name.clone(),
                dist: dist.clone(),
                span: *span,
            });
        }
        Stmt::Let { name, value, span } => {
            check_value_expr(value, bound, diags);
            if bound.contains(name) {
                diags.push(Diagnostic::error(
                    format!("variable `{name}` is already defined"),
                    *span,
                ));
            }
            bound.push(name.clone());
            out.push(ModelItem::Let {
                name: name.clone(),
                value: value.clone(),
                span: *span,
            });
        }
        Stmt::Observe { dist, value, span } => {
            check_dist_expr(dist, bound, diags);
            check_value_expr(value, bound, diags);
            out.push(ModelItem::Observe {
                dist: dist.clone(),
                value: value.clone(),
                span: *span,
            });
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
            span,
        } => {
            check_value_expr(cond, bound, diags);
            let mut then_items = Vec::new();
            let mut else_items = Vec::new();
            for s in then_body {
                lower_stmt(s, &mut then_items, bound, prior_order, diags);
            }
            for s in else_body {
                lower_stmt(s, &mut else_items, bound, prior_order, diags);
            }
            out.push(ModelItem::If {
                cond: cond.clone(),
                then_items,
                else_items,
                span: *span,
            });
        }
    }
}

/// Verify a distribution expression references only declared names and has the
/// right shape for a known family. Degenerate parameter values are warnings.
fn check_dist_expr(expr: &Expr, bound: &[String], diags: &mut Vec<Diagnostic>) {
    match expr {
        Expr::Call { name, args, .. } => {
            let expected = known_dist_arity(name);
            match expected {
                Some(arity) if args.len() != arity => diags.push(Diagnostic::error(
                    format!("`{name}` expects {arity} argument(s), found {}", args.len()),
                    expr_span(expr),
                )),
                None => diags.push(Diagnostic::error(
                    format!("`{name}` is not a known distribution"),
                    expr_span(expr),
                )),
                _ => {}
            }
            for a in args {
                check_value_expr(a, bound, diags);
            }
            check_degenerate_literal(expr, diags);
        }
        _ => diags.push(Diagnostic::error(
            "expected a distribution constructor (e.g. `Normal(0, 1)`)",
            expr_span(expr),
        )),
    }
}

fn check_value_expr(expr: &Expr, bound: &[String], diags: &mut Vec<Diagnostic>) {
    match expr {
        Expr::Num(_) => {}
        Expr::Var(name) => {
            if !bound.contains(name) {
                diags.push(Diagnostic::error(
                    format!("use of undeclared variable `{name}`"),
                    expr_span(expr),
                ));
            }
        }
        Expr::Neg(e) | Expr::Paren(e) => check_value_expr(e, bound, diags),
        Expr::Bin { lhs, rhs, .. } => {
            check_value_expr(lhs, bound, diags);
            check_value_expr(rhs, bound, diags);
        }
        Expr::Cmp { lhs, rhs, .. } => {
            check_value_expr(lhs, bound, diags);
            check_value_expr(rhs, bound, diags);
        }
        Expr::Call { name, args, .. } => {
            diags.push(Diagnostic::error(
                format!("`{name}(..)` is a distribution, not a numeric value"),
                expr_span(expr),
            ));
            for a in args {
                check_value_expr(a, bound, diags);
            }
        }
    }
}

fn expr_span(expr: &Expr) -> Span {
    match expr {
        Expr::Call { .. } => Span::new(0, 0),
        _ => Span::new(0, 0),
    }
}

pub fn known_dist_arity(name: &str) -> Option<usize> {
    match name {
        "Normal" | "Beta" | "Gamma" | "Uniform" | "Binomial" => Some(2),
        "HalfNormal" | "Exponential" | "Poisson" | "Bernoulli" => Some(1),
        _ => None,
    }
}

/// Evaluate a (non-distribution) arithmetic expression in the environment.
pub fn eval(expr: &Expr, env: &Env) -> f64 {
    match expr {
        Expr::Num(v) => *v,
        Expr::Var(name) => env.get(name).copied().unwrap_or(f64::NAN),
        Expr::Neg(e) => -eval(e, env),
        Expr::Paren(e) => eval(e, env),
        Expr::Bin { op, lhs, rhs } => {
            let a = eval(lhs, env);
            let b = eval(rhs, env);
            match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
            }
        }
        Expr::Cmp { op, lhs, rhs } => {
            let a = eval(lhs, env);
            let b = eval(rhs, env);
            let t = match op {
                CmpOp::Gt => a > b,
                CmpOp::Lt => a < b,
                CmpOp::Ge => a >= b,
                CmpOp::Le => a <= b,
                CmpOp::Eq => a == b,
                CmpOp::Ne => a != b,
            };
            if t {
                1.0
            } else {
                0.0
            }
        }
        Expr::Call { .. } => f64::NAN,
    }
}

/// Instantiate a concrete [`Dist`] from a distribution expression.
pub fn instantiate_dist(expr: &Expr, env: &Env) -> Option<Dist> {
    match expr {
        Expr::Call { name, args, .. } => {
            let v: Vec<f64> = args.iter().map(|a| eval(a, env)).collect();
            match name.as_str() {
                "Normal" if v.len() == 2 => Some(Dist::Normal {
                    mu: v[0],
                    sigma: v[1],
                }),
                "HalfNormal" if v.len() == 1 => Some(Dist::HalfNormal { sigma: v[0] }),
                "Beta" if v.len() == 2 => Some(Dist::Beta { a: v[0], b: v[1] }),
                "Gamma" if v.len() == 2 => Some(Dist::Gamma {
                    shape: v[0],
                    rate: v[1],
                }),
                "Uniform" if v.len() == 2 => Some(Dist::Uniform { lo: v[0], hi: v[1] }),
                "Exponential" if v.len() == 1 => Some(Dist::Exponential { rate: v[0] }),
                "Binomial" if v.len() == 2 => Some(Dist::Binomial { n: v[0], p: v[1] }),
                "Poisson" if v.len() == 1 => Some(Dist::Poisson { rate: v[0] }),
                "Bernoulli" if v.len() == 1 => Some(Dist::Bernoulli { p: v[0] }),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Compute the unnormalised log-joint probability for a given assignment to the
/// prior variables. `values` must be aligned with [`Model::prior_order`].
///
/// This is the heart of uncertainty propagation: every parameter expression is
/// evaluated against the current sample, so deterministic transforms and
/// conditioning statements all see consistent, uncertainty-bearing values.
pub fn log_joint(model: &Model, values: &[f64], env: &mut Env) -> f64 {
    env.clear();
    for (name, v) in model.prior_order.iter().zip(values.iter()) {
        env.insert(name.clone(), *v);
    }
    let mut lp = 0.0;
    log_joint_items(&model.items, env, &mut lp);
    lp
}

fn log_joint_items(items: &[ModelItem], env: &mut Env, lp: &mut f64) {
    for item in items {
        match item {
            ModelItem::Prior { name, dist, .. } => {
                let v = env.get(name).copied().unwrap_or(f64::NAN);
                if let Some(d) = instantiate_dist(dist, env) {
                    *lp += d.logp(v);
                }
            }
            ModelItem::Let { name, value, .. } => {
                let v = eval(value, env);
                env.insert(name.clone(), v);
            }
            ModelItem::Observe { dist, value, .. } => {
                let obs = eval(value, env);
                if let Some(d) = instantiate_dist(dist, env) {
                    *lp += d.logp(obs);
                }
            }
            ModelItem::If {
                cond,
                then_items,
                else_items,
                ..
            } => {
                if eval(cond, env) > 0.0 {
                    log_joint_items(then_items, env, lp);
                } else {
                    log_joint_items(else_items, env, lp);
                }
            }
        }
    }
}

/// If every parameter of a distribution expression is a compile-time constant,
/// we can statically flag degenerate/invalid parameters (e.g. `Normal(0, -1)`,
/// `Beta(0, 1)`, `Uniform(3, 1)`). Non-literal parameters are left to runtime.
fn check_degenerate_literal(dist_expr: &Expr, diags: &mut Vec<Diagnostic>) {
    if let Expr::Call { name, args, .. } = dist_expr {
        let mut consts = Vec::new();
        for a in args {
            match const_value(a) {
                Some(v) => consts.push(v),
                None => return, // not all args are constants
            }
        }
        let v = consts;
        let bad = match name.as_str() {
            "Normal" | "HalfNormal" => v.last().copied().unwrap_or(0.0) <= 0.0,
            "Uniform" => v.len() == 2 && v[0] >= v[1],
            "Beta" | "Gamma" => v.iter().any(|x| *x <= 0.0),
            "Exponential" | "Poisson" => v.first().copied().unwrap_or(0.0) <= 0.0,
            "Binomial" => v.first().copied().unwrap_or(0.0) < 0.0,
            "Bernoulli" => {
                let p = v.first().copied().unwrap_or(0.0);
                !(0.0..=1.0).contains(&p)
            }
            _ => false,
        };
        if bad {
            diags.push(Diagnostic::warning(
                format!("degenerate parameters for `{name}` (non-positive or invalid range)"),
                expr_span(dist_expr),
            ));
        }
    }
}

/// Extract a compile-time constant value from a literal or negated literal.
fn const_value(e: &Expr) -> Option<f64> {
    match e {
        Expr::Num(x) => Some(*x),
        Expr::Neg(inner) => const_value(inner).map(|x| -x),
        Expr::Paren(inner) => const_value(inner),
        _ => None,
    }
}
