//! Parameter-space transforms used by the HMC and ADVI engines.
//!
//! Mapping an unconstrained variable `u` to the constrained support of a
//! distribution keeps inference inside the support, which is essential for
//! bounded families (Beta ∈ (0,1), positive-only HalfNormal/Gamma/Exponential)
//! and for stable Hamiltonian dynamics.

/// Per-dimension transform from unconstrained `u` to constrained `theta`.
#[derive(Clone)]
pub enum Transform {
    Identity,
    Log,
    Logit { lo: f64, hi: f64 },
}

impl Transform {
    /// `(theta, log|dtheta/du|, d(log|dtheta/du|)/du)`.
    pub fn forward(&self, u: f64) -> (f64, f64, f64) {
        let u = u.clamp(-30.0, 30.0);
        match self {
            Transform::Identity => (u, 0.0, 0.0),
            Transform::Log => {
                let theta = u.exp();
                (theta, u, 1.0) // dtheta/du = exp(u); d log(dtheta/du)/du = 1
            }
            Transform::Logit { lo, hi } => {
                let s = 1.0 / (1.0 + (-u).exp());
                let theta = lo + (hi - lo) * s;
                let log_jac = (hi - lo).ln() + s.ln() + (1.0 - s).ln();
                let dlogjac = 1.0 - 2.0 * s;
                (theta, log_jac, dlogjac)
            }
        }
    }

    /// Inverse map, constrained `theta` -> unconstrained `u`.
    pub fn inverse(&self, theta: f64) -> f64 {
        const E: f64 = 1e-6;
        match self {
            Transform::Identity => theta,
            Transform::Log => theta.max(E).ln(),
            Transform::Logit { lo, hi } => {
                let s = ((theta - lo) / (hi - lo)).clamp(E, 1.0 - E);
                (s / (1.0 - s)).ln()
            }
        }
    }

    /// Derivative `dtheta/du` at unconstrained point `u`.
    pub fn jacobian(&self, u: f64) -> f64 {
        let u = u.clamp(-30.0, 30.0);
        match self {
            Transform::Identity => 1.0,
            Transform::Log => u.exp(),
            Transform::Logit { lo, hi } => {
                let s = 1.0 / (1.0 + (-u).exp());
                (hi - lo) * s * (1.0 - s)
            }
        }
    }
}

/// Choose a transform for a prior distribution expression.
pub fn transform_for(dist_expr: &augur_frontend::Expr) -> Transform {
    if let augur_frontend::Expr::Call { name, args, .. } = dist_expr {
        match name.as_str() {
            "Beta" => Transform::Logit { lo: 0.0, hi: 1.0 },
            "Uniform" => {
                if args.len() == 2 {
                    let lo = const_val(&args[0]).unwrap_or(0.0);
                    let hi = const_val(&args[1]).unwrap_or(1.0);
                    if hi > lo {
                        return Transform::Logit { lo, hi };
                    }
                }
                Transform::Logit { lo: 0.0, hi: 1.0 }
            }
            "HalfNormal" | "Gamma" | "Exponential" => Transform::Log,
            _ => Transform::Identity,
        }
    } else {
        Transform::Identity
    }
}

fn const_val(e: &augur_frontend::Expr) -> Option<f64> {
    match e {
        augur_frontend::Expr::Num(x) => Some(*x),
        augur_frontend::Expr::Neg(inner) => const_val(inner).map(|x| -x),
        augur_frontend::Expr::Paren(inner) => const_val(inner),
        _ => None,
    }
}
