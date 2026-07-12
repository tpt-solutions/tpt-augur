//! Integration tests for the Augur standard library distributions: log-density
//! correctness, analytic moments, and sampler fidelity.

use augur_std::{seeded_rng, std_normal, Dist};

#[test]
fn normal_logp_at_mean() {
    let d = Dist::Normal { mu: 0.0, sigma: 1.0 };
    let lp = d.logp(0.0);
    assert!((lp + 0.5 * (2.0 * std::f64::consts::PI).ln()).abs() < 1e-12);
}

#[test]
fn exp_logp_known_point() {
    // Exponential(lambda) pdf at x = 1/lambda (its mean) is lambda * e^-1.
    let lambda = 2.0;
    let d = Dist::Exponential { rate: lambda };
    let lp = d.logp(1.0 / lambda);
    assert!((lp - (lambda.ln() - 1.0)).abs() < 1e-12, "lp={lp}");
}

#[test]
fn beta_logp_at_half() {
    // Beta(2,2) at 0.5: pdf = 0.5 * 0.5 / B(2,2) = 0.25 / (1/(6)) = 1.5.
    let d = Dist::Beta { a: 2.0, b: 2.0 };
    let lp = d.logp(0.5);
    assert!((lp - 1.5f64.ln()).abs() < 1e-9, "lp={lp}");
}

#[test]
fn boundary_points_return_negative_infinity() {
    assert_eq!(Dist::Normal { mu: 0.0, sigma: -1.0 }.logp(0.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Beta { a: 2.0, b: 2.0 }.logp(0.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Beta { a: 2.0, b: 2.0 }.logp(1.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Uniform { lo: 0.0, hi: 1.0 }.logp(2.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Gamma { shape: 1.0, rate: 1.0 }.logp(-1.0), f64::NEG_INFINITY);
}

#[test]
fn means_match_analytic() {
    assert_eq!(Dist::Normal { mu: 3.0, sigma: 2.0 }.mean(), 3.0);
    assert!((Dist::HalfNormal { sigma: 2.0 }.mean() - 2.0 * (2.0 / std::f64::consts::PI).sqrt()).abs() < 1e-12);
    assert_eq!(Dist::Beta { a: 2.0, b: 4.0 }.mean(), 2.0 / 6.0);
    assert_eq!(Dist::Gamma { shape: 3.0, rate: 2.0 }.mean(), 1.5);
    assert_eq!(Dist::Uniform { lo: -1.0, hi: 3.0 }.mean(), 1.0);
    assert_eq!(Dist::Exponential { rate: 0.5 }.mean(), 2.0);
    assert_eq!(Dist::Binomial { n: 10.0, p: 0.3 }.mean(), 3.0);
    assert_eq!(Dist::Poisson { rate: 4.0 }.mean(), 4.0);
    assert_eq!(Dist::Bernoulli { p: 0.25 }.mean(), 0.25);
}

#[test]
fn variances_match_analytic() {
    assert!((Dist::Normal { mu: 0.0, sigma: 2.0 }.variance() - 4.0).abs() < 1e-12);
    assert!((Dist::Beta { a: 2.0, b: 4.0 }.variance() - (8.0 / 252.0)).abs() < 1e-12);
    assert!((Dist::Gamma { shape: 3.0, rate: 2.0 }.variance() - 0.75).abs() < 1e-12);
    assert!((Dist::Uniform { lo: 0.0, hi: 1.0 }.variance() - (1.0 / 12.0)).abs() < 1e-12);
    assert!((Dist::Exponential { rate: 0.5 }.variance() - 4.0).abs() < 1e-12);
    assert!((Dist::Binomial { n: 10.0, p: 0.3 }.variance() - 2.1).abs() < 1e-9);
    assert!((Dist::Poisson { rate: 4.0 }.variance() - 4.0).abs() < 1e-12);
    assert!((Dist::Bernoulli { p: 0.25 }.variance() - 0.1875).abs() < 1e-12);
}

#[test]
fn sample_stats_recover_parameters_normal() {
    let mut rng = seeded_rng(42);
    let d = Dist::Normal { mu: 3.0, sigma: 2.0 };
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
fn gamma_sample_is_positive() {
    let mut rng = seeded_rng(11);
    let d = Dist::Gamma { shape: 2.0, rate: 3.0 };
    for _ in 0..500 {
        let x = d.sample(&mut rng);
        assert!(x > 0.0);
    }
}

#[test]
fn exponential_sample_stats() {
    let mut rng = seeded_rng(19);
    let d = Dist::Exponential { rate: 1.0 };
    let n = 100_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += d.sample(&mut rng);
    }
    let mean = sum / n as f64;
    assert!((mean - 1.0).abs() < 0.05, "mean={mean}");
}

#[test]
fn binomial_logp_sums_to_one_for_n_2() {
    let d = Dist::Binomial { n: 2.0, p: 0.3 };
    let total = d.logp(0.0).exp() + d.logp(1.0).exp() + d.logp(2.0).exp();
    assert!((total - 1.0).abs() < 1e-12, "total={total}");
}

#[test]
fn poisson_mean_recovers_rate() {
    let mut rng = seeded_rng(23);
    let d = Dist::Poisson { rate: 5.0 };
    let n = 100_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += d.sample(&mut rng);
    }
    let mean = sum / n as f64;
    assert!((mean - 5.0).abs() < 0.1, "mean={mean}");
}

#[test]
fn bernoulli_mean_recovers_p() {
    let mut rng = seeded_rng(29);
    let d = Dist::Bernoulli { p: 0.4 };
    let n = 100_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += d.sample(&mut rng);
    }
    let mean = sum / n as f64;
    assert!((mean - 0.4).abs() < 0.02, "mean={mean}");
}

#[test]
fn typical_point_is_in_support() {
    assert!(Dist::Beta { a: 2.0, b: 2.0 }.typical_point() > 0.0);
    assert!(Dist::Beta { a: 2.0, b: 2.0 }.typical_point() < 1.0);
    assert!(Dist::HalfNormal { sigma: 1.0 }.typical_point() > 0.0);
    assert!(Dist::Gamma { shape: 1.0, rate: 1.0 }.typical_point() > 0.0);
    // Normal typical point equals the mean.
    assert_eq!(Dist::Normal { mu: -3.0, sigma: 1.0 }.typical_point(), -3.0);
}

#[test]
fn seeded_rng_is_reproducible() {
    let mut a = seeded_rng(123);
    let mut b = seeded_rng(123);
    for _ in 0..100 {
        assert_eq!(std_normal(&mut a), std_normal(&mut b));
    }
}

#[test]
fn half_normal_logp() {
    let d = Dist::HalfNormal { sigma: 1.0 };
    assert_eq!(d.logp(-1.0), f64::NEG_INFINITY);
    assert_eq!(Dist::HalfNormal { sigma: -1.0 }.logp(0.5), f64::NEG_INFINITY);
    let lp = d.logp(0.0);
    assert!((lp - 2.0f64.ln()).abs() < 1e-12, "lp={lp}");
}

#[test]
fn gamma_logp_invalid_params() {
    assert_eq!(Dist::Gamma { shape: -1.0, rate: 1.0 }.logp(1.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Gamma { shape: 1.0, rate: -1.0 }.logp(1.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Gamma { shape: 1.0, rate: 1.0 }.logp(-1.0), f64::NEG_INFINITY);
}

#[test]
fn uniform_logp_and_sample_in_range() {
    let d = Dist::Uniform { lo: -1.0, hi: 3.0 };
    let lp = d.logp(0.0);
    assert!((lp - (-(4.0f64).ln())).abs() < 1e-12, "lp={lp}");
    assert_eq!(d.logp(-2.0), f64::NEG_INFINITY);

    let mut rng = seeded_rng(5);
    for _ in 0..1000 {
        let x = d.sample(&mut rng);
        assert!(x >= -1.0 && x <= 3.0);
    }
}

#[test]
fn poisson_logp_known_point() {
    // Poisson(rate=3) at x=0 is e^-3.
    let d = Dist::Poisson { rate: 3.0 };
    let lp = d.logp(0.0);
    assert!((lp - (-3.0f64)).abs() < 1e-9, "lp={lp}");
    assert_eq!(Dist::Poisson { rate: -1.0 }.logp(0.0), f64::NEG_INFINITY);
}

#[test]
fn bernoulli_logp_both_outcomes_and_invalid_p() {
    let d = Dist::Bernoulli { p: 0.3 };
    assert!((d.logp(1.0) - 0.3f64.ln()).abs() < 1e-12);
    assert!((d.logp(0.0) - 0.7f64.ln()).abs() < 1e-12);
    assert_eq!(d.logp(2.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Bernoulli { p: 1.5 }.logp(1.0), f64::NEG_INFINITY);
}

#[test]
fn binomial_logp_invalid_p() {
    assert_eq!(Dist::Binomial { n: 5.0, p: -0.1 }.logp(1.0), f64::NEG_INFINITY);
    assert_eq!(Dist::Binomial { n: 5.0, p: 1.1 }.logp(1.0), f64::NEG_INFINITY);
}

#[test]
fn remaining_variances_match_analytic() {
    let hn = Dist::HalfNormal { sigma: 2.0 };
    let expected = 4.0 * (1.0 - std::f64::consts::PI / 2.0);
    assert!((hn.variance() - expected).abs() < 1e-12);
}

#[test]
fn binomial_sample_within_bounds() {
    let mut rng = seeded_rng(31);
    let d = Dist::Binomial { n: 10.0, p: 0.5 };
    for _ in 0..500 {
        let x = d.sample(&mut rng);
        assert!(x >= 0.0 && x <= 10.0);
    }
}

#[test]
fn remaining_typical_points() {
    assert_eq!(Dist::Uniform { lo: -2.0, hi: 4.0 }.typical_point(), 1.0);
    assert_eq!(
        Dist::Binomial { n: 10.0, p: 0.3 }.typical_point(),
        Dist::Binomial { n: 10.0, p: 0.3 }.mean()
    );
    assert_eq!(
        Dist::Poisson { rate: 4.0 }.typical_point(),
        Dist::Poisson { rate: 4.0 }.mean()
    );
    assert_eq!(
        Dist::Bernoulli { p: 0.4 }.typical_point(),
        Dist::Bernoulli { p: 0.4 }.mean()
    );
    assert!(Dist::Exponential { rate: 2.0 }.typical_point() > 0.0);
}

#[test]
fn gamma_sample_shape_below_one_uses_boost_branch() {
    let mut rng = seeded_rng(41);
    let d = Dist::Gamma { shape: 0.3, rate: 1.5 };
    for _ in 0..500 {
        let x = d.sample(&mut rng);
        assert!(x > 0.0);
    }
}

#[test]
fn poisson_sample_large_rate_uses_transformed_rejection() {
    let mut rng = seeded_rng(53);
    let d = Dist::Poisson { rate: 50.0 };
    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
        let x = d.sample(&mut rng);
        assert!(x >= 0.0);
        sum += x;
    }
    let mean = sum / n as f64;
    assert!((mean - 50.0).abs() < 2.0, "mean={mean}");
}

#[test]
fn std_normal_has_roughly_zero_mean() {
    let mut rng = seeded_rng(99);
    let n = 100_000;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += std_normal(&mut rng);
    }
    assert!((sum / n as f64).abs() < 0.02);
}
