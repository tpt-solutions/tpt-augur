//! Special mathematical functions used by the distribution implementations.

/// Natural log of the gamma function via the Lanczos approximation, with the
/// reflection formula applied for non-positive arguments.
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        // Gamma(z) = pi / (sin(pi z) * Gamma(1 - z))
        let denom = (std::f64::consts::PI * x).sin().abs() * (1.0 - x).ln_gamma_pos().exp();
        return (std::f64::consts::PI / denom).ln();
    }
    x.ln_gamma_pos()
}

trait LnGammaPos {
    fn ln_gamma_pos(self) -> f64;
}

impl LnGammaPos for f64 {
    fn ln_gamma_pos(self) -> f64 {
        // Lanczos coefficients for g = 7, n = 9.
        const G: f64 = 7.0;
        #[allow(clippy::excessive_precision)]
        const COEFS: [f64; 9] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let x = self - 1.0;
        let mut a = COEFS[0];
        let t = x + G + 0.5;
        for (i, c) in COEFS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let f = a;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + f.ln()
    }
}

/// Natural log of the beta function B(a, b).
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Log of the binomial coefficient C(n, k) for non-negative integer n, k.
pub fn ln_choose(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_known() {
        // Gamma(5) = 24, ln 24 ~ 3.178
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-9);
        assert!((ln_gamma(0.5) - (std::f64::consts::PI).sqrt().ln()).abs() < 1e-9);
    }

    #[test]
    fn ln_beta_matches_known() {
        // B(1,1) = 1 -> ln 0
        assert!(ln_beta(1.0, 1.0).abs() < 1e-12);
        // B(a,b) = (a-1)!(b-1)!/(a+b-1)! for integers
        let v = ln_beta(3.0, 4.0);
        let expected = 2.0f64.ln() + 6.0f64.ln() - 720.0f64.ln();
        assert!((v - expected).abs() < 1e-9);
    }
}
