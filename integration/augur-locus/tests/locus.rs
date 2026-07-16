//! Integration tests for the Augur ↔ TPT Locus bridge.

use augur_locus::{
    evaluate_source, evaluate_strategy, features_to_beta_prior, logistic, success_model_source,
    LocusAugurBridge, ProbabilityOfSuccess, Strategy,
};

#[test]
fn logistic_bounds() {
    assert!(logistic(0.0) > 0.49 && logistic(0.0) < 0.51);
    assert!(logistic(20.0) > 0.99);
    assert!(logistic(-20.0) < 0.01);
}

#[test]
fn positive_features_raise_success_prior() {
    let (a, b) = features_to_beta_prior(&[2.0, 1.0], 20.0);
    assert!(a > b, "positive features should favor success");
}

#[test]
fn success_model_source_format() {
    let src = success_model_source(3.0, 5.0);
    assert!(src.contains("let p ~ Beta(3, 5)"));
    assert!(src.contains("let success ~ Bernoulli(p)"));
}

#[test]
fn evaluate_strategy_returns_probability_in_unit_interval() {
    let s = Strategy {
        id: "s1".into(),
        label: "aggressive".into(),
        features: vec![1.5, 0.5],
    };
    let p = evaluate_strategy(&s, 20.0).unwrap();
    assert!((0.0..=1.0).contains(&p.value));
    assert!(p.ci_low <= p.value && p.value <= p.ci_high);
}

#[test]
fn probability_increases_with_feature_mass() {
    let weak = Strategy {
        id: "w".into(),
        label: "w".into(),
        features: vec![-2.0],
    };
    let strong = Strategy {
        id: "s".into(),
        label: "s".into(),
        features: vec![2.0],
    };
    let pw = evaluate_strategy(&weak, 20.0).unwrap();
    let ps = evaluate_strategy(&strong, 20.0).unwrap();
    assert!(ps.value > pw.value, "strong={:?} weak={:?}", ps, pw);
}

#[test]
fn bridge_ranks_strategies_by_probability() {
    let bridge = LocusAugurBridge::new(20.0);
    let weak = Strategy {
        id: "weak".into(),
        label: "weak".into(),
        features: vec![-2.0],
    };
    let strong = Strategy {
        id: "strong".into(),
        label: "strong".into(),
        features: vec![2.0],
    };
    let ranked = bridge.rank(&[weak, strong]).unwrap();
    assert_eq!(ranked[0].strategy.id, "strong");
    assert!(ranked[0].probability.value > ranked[1].probability.value);
}

#[test]
fn evaluate_source_injects_features() {
    let src = "let p ~ Beta(1 + 4*feature0, 2)\nlet success ~ Bernoulli(p)";
    let s = Strategy {
        id: "x".into(),
        label: "x".into(),
        features: vec![0.5],
    };
    let p = evaluate_source(src, &s).unwrap();
    assert!((0.0..=1.0).contains(&p.value));
}

#[test]
fn probability_of_success_is_copied_clamp() {
    let p = ProbabilityOfSuccess {
        value: 0.7,
        ci_low: 0.5,
        ci_high: 0.9,
    };
    let q = p;
    assert_eq!(p, q);
}
