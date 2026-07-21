//! Integration tests for the Augur ↔ TPT Keystone DB bridge.
//!
//! The query surface is exercised through a local in-process [`MockKeystone`]
//! implementing [`KeystoneQuerier`], so no live server is required.

use tpt_augur_keystone::{
    build_success_model, logistic, model_from_prior, prior_from_relational,
    prior_from_relational_query, prior_from_vector_query, prior_from_vector_similarity,
    tally_outcomes, KeystoneQuerier, RelationalPriorData,
};
use tpt_sdk::keystone::{KeystoneError, QueryResult, Row, Value};

struct MockKeystone {
    result: QueryResult,
}

impl KeystoneQuerier for MockKeystone {
    async fn query_params(
        &mut self,
        _sql: &str,
        _params: &[Value],
    ) -> Result<QueryResult, KeystoneError> {
        Ok(self.result.clone())
    }
}

fn feedback_result() -> QueryResult {
    QueryResult::new(
        vec!["model_id".into(), "outcome".into()],
        vec![
            Row::new(
                ["model_id", "outcome"],
                [Some(b"m".to_vec()), Some(b"success".to_vec())],
            ),
            Row::new(
                ["model_id", "outcome"],
                [Some(b"m".to_vec()), Some(b"success".to_vec())],
            ),
            Row::new(
                ["model_id", "outcome"],
                [Some(b"m".to_vec()), Some(b"failure".to_vec())],
            ),
        ],
        None,
    )
}

fn memory_result() -> QueryResult {
    QueryResult::new(
        vec!["strategy_id".into(), "similarity".into()],
        vec![
            Row::new(
                ["strategy_id", "similarity"],
                [Some(b"s".to_vec()), Some(b"0.8".to_vec())],
            ),
            Row::new(
                ["strategy_id", "similarity"],
                [Some(b"s".to_vec()), Some(b"0.4".to_vec())],
            ),
        ],
        None,
    )
}

#[test]
fn logistic_maps_scores() {
    assert!(logistic(0.0) > 0.49 && logistic(0.0) < 0.51);
    assert!(logistic(10.0) > 0.99);
    assert!(logistic(-10.0) < 0.01);
}

#[test]
fn prior_from_relational_adds_pseudocounts() {
    let (a, b) = prior_from_relational(
        &RelationalPriorData {
            successes: 2,
            failures: 1,
        },
        2.0,
    );
    assert_eq!((a, b), (4.0, 3.0));
}

#[test]
fn prior_from_vector_maps_similarity() {
    let (hi_a, hi_b) = prior_from_vector_similarity(0.9, 10.0);
    let (lo_a, lo_b) = prior_from_vector_similarity(-0.9, 10.0);
    assert!(hi_a > lo_a, "high similarity ⇒ higher success prior alpha");
    assert!(hi_b < lo_b, "high similarity ⇒ lower failure prior beta");
}

#[test]
fn prior_from_vector_is_symmetric_at_zero() {
    let (a, b) = prior_from_vector_similarity(0.0, 10.0);
    assert!((a - b).abs() < 1e-9, "symmetric at similarity 0");
}

#[test]
fn build_success_model_format() {
    let src = build_success_model(4.0, 3.0);
    assert!(src.contains("let p ~ Beta(4, 3)"));
    assert!(src.contains("let success ~ Bernoulli(p)"));
}

#[test]
fn model_from_prior_lowers() {
    let m = model_from_prior(4.0, 3.0).unwrap();
    assert_eq!(m.prior_order, vec!["p", "success"]);
}

#[test]
fn record_outcomes_tallies_text_outcomes() {
    let res = feedback_result();
    let data = tally_outcomes(&res);
    assert_eq!(data.successes, 2);
    assert_eq!(data.failures, 1);
}

#[tokio::test]
async fn relational_query_builds_prior() {
    let mut mock = MockKeystone {
        result: feedback_result(),
    };
    let (a, b) = prior_from_relational_query(&mut mock, "m", 2.0)
        .await
        .unwrap();
    assert_eq!((a, b), (4.0, 3.0));
}

#[tokio::test]
async fn vector_query_builds_prior() {
    let mut mock = MockKeystone {
        result: memory_result(),
    };
    let (a, b) = prior_from_vector_query(&mut mock, "s", 10.0).await.unwrap();
    assert!(a > b);
}

#[tokio::test]
async fn empty_result_is_an_error() {
    let mut mock = MockKeystone {
        result: QueryResult::new(vec!["model_id".into(), "outcome".into()], vec![], None),
    };
    let err = prior_from_relational_query(&mut mock, "m", 2.0).await;
    assert!(matches!(
        err,
        Err(tpt_augur_keystone::BridgeError::EmptyResult)
    ));
}
