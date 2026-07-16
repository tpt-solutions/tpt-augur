//! Concrete probability distributions with log-density evaluation and sampling.
//!
//! Distributions are parameterised by plain `f64` values. The compiler frontend
//! is responsible for instantiating a [`Dist`] from a distribution expression by
//! evaluating its parameters in the current environment (this is how uncertainty
//! propagates: deterministic expressions become concrete numbers at sample time).

use rand::{Rng, SeedableRng};

/// Standard normal N(0,1) sample via Box–Muller.
pub fn std_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u: f64 = rng.gen_range(1e-300..1.0);
    let v: f64 = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
    (-2.0 * u.ln()).sqrt() * v.cos()
}

use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::special::{ln_beta, ln_choose, ln_gamma};

/// Probability distribution over a continuous or discrete sample space.
///
/// Each variant carries the named parameters of its family as plain `f64`
/// values. Use [`Dist::logp`], [`Dist::sample`], and [`Dist::mean`] to work
/// with a `Dist` at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Dist {
    /// Gaussian distribution N(μ, σ).
    Normal {
        mu: f64,
        sigma: f64,
    },
    /// Half-normal distribution with scale σ (support: x ≥ 0).
    HalfNormal {
        sigma: f64,
    },
    /// Beta distribution Beta(α, β) on [0, 1].
    Beta {
        a: f64,
        b: f64,
    },
    /// Gamma distribution Gamma(shape, rate).
    Gamma {
        shape: f64,
        rate: f64,
    },
    /// Continuous uniform distribution on [lo, hi].
    Uniform {
        lo: f64,
        hi: f64,
    },
    /// Exponential distribution Exp(λ) with rate λ.
    Exponential {
        rate: f64,
    },
    /// Binomial distribution Bin(n, p). `n` is the trial count, `p` the success probability.
    Binomial {
        n: f64,
        p: f64,
    },
    /// Poisson distribution Pois(λ) with rate λ.
    Poisson {
        rate: f64,
    },
    /// Bernoulli distribution Bernoulli(p) — binary (0/1) outcome.
    Bernoulli {
        p: f64,
    },
}

impl Dist {
    /// Log probability density (or mass) of `x` under this distribution.
    pub fn logp(&self, x: f64) -> f64 {
        const PI: f64 = std::f64::consts::PI;
        match self {
            Dist::Normal { mu, sigma } => {
                if *sigma <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * ((x - mu) / sigma).powi(2)
            }
            Dist::HalfNormal { sigma } => {
                if *sigma <= 0.0 || x < 0.0 {
                    return f64::NEG_INFINITY;
                }
                (2.0f64).ln() - sigma.ln() - 0.5 * (x / sigma).powi(2)
            }
            Dist::Beta { a, b } => {
                if x <= 0.0 || x >= 1.0 {
                    return f64::NEG_INFINITY;
                }
                (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta(*a, *b)
            }
            Dist::Gamma { shape, rate } => {
                if x < 0.0 || *shape <= 0.0 || *rate <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                (shape - 1.0) * x.ln() - rate * x - ln_gamma(*shape) + shape * rate.ln()
            }
            Dist::Uniform { lo, hi } => {
                if x < *lo || x > *hi {
                    return f64::NEG_INFINITY;
                }
                -(hi - lo).ln()
            }
            Dist::Exponential { rate } => {
                if x < 0.0 || *rate <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                rate.ln() - rate * x
            }
            Dist::Binomial { n, p } => {
                if *p < 0.0 || *p > 1.0 {
                    return f64::NEG_INFINITY;
                }
                ln_choose(*n, x) + x * p.ln() + (n - x) * (1.0 - p).ln()
            }
            Dist::Poisson { rate } => {
                if *rate <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                x * rate.ln() - rate - ln_gamma(x + 1.0)
            }
            Dist::Bernoulli { p } => {
                if *p < 0.0 || *p > 1.0 {
                    return f64::NEG_INFINITY;
                }
                if (x - 1.0).abs() < 1e-9 {
                    p.ln()
                } else if x.abs() < 1e-9 {
                    (1.0 - p).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Analytic mean where available.
    pub fn mean(&self) -> f64 {
        match self {
            Dist::Normal { mu, .. } => *mu,
            Dist::HalfNormal { sigma } => sigma * (2.0 / std::f64::consts::PI).sqrt(),
            Dist::Beta { a, b } => a / (a + b),
            Dist::Gamma { shape, rate } => shape / rate,
            Dist::Uniform { lo, hi } => (lo + hi) / 2.0,
            Dist::Exponential { rate } => 1.0 / rate,
            Dist::Binomial { n, p } => n * p,
            Dist::Poisson { rate } => *rate,
            Dist::Bernoulli { p } => *p,
        }
    }

    /// Analytic variance where available.
    pub fn variance(&self) -> f64 {
        match self {
            Dist::Normal { sigma, .. } => sigma * sigma,
            Dist::HalfNormal { sigma } => {
                let p = std::f64::consts::PI / 2.0;
                sigma * sigma * (1.0 - p)
            }
            Dist::Beta { a, b } => (a * b) / ((a + b).powi(2) * (a + b + 1.0)),
            Dist::Gamma { shape, rate } => shape / (rate * rate),
            Dist::Uniform { lo, hi } => (hi - lo).powi(2) / 12.0,
            Dist::Exponential { rate } => 1.0 / (rate * rate),
            Dist::Binomial { n, p } => n * p * (1.0 - p),
            Dist::Poisson { rate } => *rate,
            Dist::Bernoulli { p } => p * (1.0 - p),
        }
    }

    /// Draw a single sample using the supplied RNG.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            Dist::Normal { mu, sigma } => {
                let z = std_normal(rng);
                mu + sigma * z
            }
            Dist::HalfNormal { sigma } => {
                let z = std_normal(rng);
                sigma * z.abs()
            }
            Dist::Beta { a, b } => {
                let ga = sample_gamma(rng, *a, 1.0);
                let gb = sample_gamma(rng, *b, 1.0);
                ga / (ga + gb)
            }
            Dist::Gamma { shape, rate } => sample_gamma(rng, *shape, *rate),
            Dist::Uniform { lo, hi } => lo + (hi - lo) * rng.gen::<f64>(),
            Dist::Exponential { rate } => {
                let u: f64 = rng.gen_range(1e-300..1.0);
                -u.ln() / rate
            }
            Dist::Binomial { n, p } => {
                let trials = n.round() as u64;
                let p = p.clamp(1e-12, 1.0 - 1e-12);
                let mut k = 0;
                for _ in 0..trials {
                    if rng.gen_bool(p) {
                        k += 1;
                    }
                }
                k as f64
            }
            Dist::Poisson { rate } => sample_poisson(rng, *rate),
            Dist::Bernoulli { p } => {
                // `gen_bool` rejects p exactly 0/1 and NaN; a Beta-drawn p can
                // land on those via float rounding, so keep it strictly interior
                // (and fall back to 0.5 for non-finite values).
                let p = if p.is_finite() {
                    p.clamp(1e-12, 1.0 - 1e-12)
                } else {
                    0.5
                };
                if rng.gen_bool(p) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Default continuous initialisation point for MCMC (avoids `-inf` logp).
    pub fn typical_point(&self) -> f64 {
        match self {
            Dist::Beta { .. } => 0.5,
            Dist::HalfNormal { .. } | Dist::Gamma { .. } | Dist::Exponential { .. } => {
                self.mean().max(1e-2)
            }
            Dist::Uniform { lo, hi } => (lo + hi) / 2.0,
            Dist::Binomial { .. } | Dist::Poisson { .. } | Dist::Bernoulli { .. } => self.mean(),
            Dist::Normal { mu, .. } => *mu,
        }
    }
}

/// Marsaglia–Tsang gamma sampler. `rate` is the rate parameter (scale = 1/rate).
pub fn sample_gamma<R: Rng + ?Sized>(rng: &mut R, shape: f64, rate: f64) -> f64 {
    let shape = shape.max(1e-8);
    if shape < 1.0 {
        // Boost using Gamma(shape+1) * U^(1/shape)
        let u: f64 = rng.gen_range(1e-300..1.0);
        let g = sample_gamma(rng, shape + 1.0, 1.0);
        return g * u.powf(1.0 / shape) / rate;
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = loop {
            let z = std_normal(rng);
            let v = 1.0 + c * z;
            if v > 0.0 {
                break v * v * v;
            }
        };
        let u: f64 = rng.gen_range(1e-300..1.0);
        if u < 1.0 - 0.0331 * (z_diff(x)).powi(2) {
            return (d * x) / rate;
        }
        let z = (x - 1.0).sqrt();
        if u < 1.0 + 0.0331 * z.powi(4) && {
            let v = z_diff(x);
            (u - 1.0f64).abs() <= 0.5 * v * v + d * (1.0 - x + x.ln())
        } {
            return (d * x) / rate;
        }
    }
}

fn z_diff(x: f64) -> f64 {
    // Simplified acceptance approximation term used in the Marsaglia-Tsang method.
    3.0 * (x.cbrt() - 1.0)
}

/// Knuth's algorithm for small rates, transformed rejection for large rates.
pub fn sample_poisson<R: Rng + ?Sized>(rng: &mut R, rate: f64) -> f64 {
    if rate <= 0.0 {
        return 0.0;
    }
    if rate < 10.0 {
        let l = (-rate).exp();
        let mut k = 0;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.gen::<f64>();
            if p <= l {
                return (k - 1) as f64;
            }
        }
    } else {
        // Transformed rejection (Ptolomey-style approximation).
        let smu = rate.sqrt();
        let b = 0.931 + 2.53 * smu;
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u: f64 = rng.gen_range(1e-300..1.0);
            let v: f64 = rng.gen_range(-1.0..1.0);
            let us = 0.5 - u.abs();
            let k = (2.0 * a / us + b).floor() * v + rate;
            if us >= 0.07 && v >= vr - inv_alpha * smu * us {
                if k < 0.0 {
                    continue;
                }
                return k.floor();
            }
            if vr < us && us < 0.013 && v * v < us * us * vr * vr {
                if k < 0.0 {
                    continue;
                }
                return k.floor();
            }
        }
    }
}

/// Convenience helper to build a deterministic RNG for reproducible tests.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_logp_at_mean() {
        let d = Dist::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let lp = d.logp(0.0);
        assert!((lp + 0.5 * (2.0 * std::f64::consts::PI).ln()).abs() < 1e-12);
    }

    #[test]
    fn sample_stats_recover_parameters() {
        let mut rng = seeded_rng(42);
        let d = Dist::Normal {
            mu: 3.0,
            sigma: 2.0,
        };
        let n = 200_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let x = d.sample(&mut rng);
            sum += x;
            sumsq += x * x;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!((mean - 3.0).abs() < 0.05, "mean={mean}");
        assert!((var - 4.0).abs() < 0.1, "var={var}");
    }

    #[test]
    fn beta_sample_in_unit_interval() {
        let mut rng = seeded_rng(7);
        let d = Dist::Beta { a: 2.0, b: 5.0 };
        for _ in 0..1000 {
            let x = d.sample(&mut rng);
            assert!(x > 0.0 && x < 1.0);
        }
        assert!((d.mean() - 2.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn binomial_logp_sums_to_one_for_n_2() {
        let d = Dist::Binomial { n: 2.0, p: 0.3 };
        let total = d.logp(0.0).exp() + d.logp(1.0).exp() + d.logp(2.0).exp();
        assert!((total - 1.0).abs() < 1e-12, "total={total}");
    }
}
