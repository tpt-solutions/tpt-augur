//! Augur ↔ TPT Keystone DB integration (spec.txt §4).
//!
//! Augur models can be seeded by real data living in Keystone: historical
//! outcomes drive a Beta prior over a success rate, and vector-memory
//! similarity drives a prior over how promising a strategy is. This crate
//! provides:
//!
//! * **native query bindings** — typed [`tpt_sdk`] queries against Keystone
//!   tables that carry Augur prior data (relational `augur_feedback`, vector
//!   `augur_memories`),
//! * **prior updates from relational data** — [`prior_from_relational`],
//! * **prior updates from vector data** — [`prior_from_vector_similarity`],
//! * a **live-prior example model** built from a Keystone query result, and
//! * an integration test (`tests/live_integration.rs`, `#[ignore]`d so it only
//!   runs against a live Keystone instance).
//!
//! The query surface is abstracted behind [`KeystoneQuerier`] so the prior
//! math is unit-tested against a [`MockKeystone`] without a server, while
//! [`RealKeystone`] wraps `tpt_sdk`'s `KeystoneClient` for production.

use tpt_sdk::keystone::{KeystoneClient, KeystoneError, QueryResult, Value};
use tpt_sdk::query_builder::{Order, QueryBuilder, Table};

use tpt_augur_frontend::parse;
use tpt_augur_ir::lower;

/// Abstracts the Keystone query surface so the bridge is testable without a
/// running server and swappable for the real `KeystoneClient`.
#[allow(async_fn_in_trait)]
pub trait KeystoneQuerier {
    async fn query_params(
        &mut self,
        sql: &str,
        params: &[Value],
    ) -> Result<QueryResult, KeystoneError>;
}

/// A live Keystone connection backed by `tpt_sdk`'s `KeystoneClient`.
pub struct RealKeystone {
    client: KeystoneClient,
}

impl RealKeystone {
    /// Connect to a Keystone node's Postgres-wire listener (e.g. `127.0.0.1:5432`).
    pub async fn connect(addr: &str) -> Result<Self, KeystoneError> {
        Ok(Self {
            client: KeystoneClient::connect(addr).await?,
        })
    }
}

impl KeystoneQuerier for RealKeystone {
    async fn query_params(
        &mut self,
        sql: &str,
        params: &[Value],
    ) -> Result<QueryResult, KeystoneError> {
        self.client.query_params(sql, params).await
    }
}

/// Errors raised while building Augur priors from Keystone data.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("keystone query failed: {0}")]
    Keystone(#[from] KeystoneError),
    #[error("no rows returned for prior query")]
    EmptyResult,
    #[error("frontend/IR error building model: {0}")]
    Model(String),
}

/// Relational summary of past outcomes used to seed a Beta prior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationalPriorData {
    pub successes: u64,
    pub failures: u64,
}

/// Logistic sigmoid (maps a signed score into `(0, 1)`).
pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Map historical outcome counts to a Beta prior `(alpha, beta)` over the
/// success rate, adding `concentration` pseudo-counts of each kind so the
/// prior is never degenerate.
pub fn prior_from_relational(data: &RelationalPriorData, concentration: f64) -> (f64, f64) {
    let alpha = data.successes as f64 + concentration;
    let beta = data.failures as f64 + concentration;
    (alpha, beta)
}

/// Map a vector-memory similarity score in `[-1, 1]` to a Beta prior over how
/// promising a strategy is: high similarity ⇒ high success prior.
pub fn prior_from_vector_similarity(similarity: f64, concentration: f64) -> (f64, f64) {
    let p = logistic(similarity.clamp(-1.0, 1.0));
    (p * concentration + 1.0, (1.0 - p) * concentration + 1.0)
}

/// The canonical P(success) model seeded by a Beta prior.
pub fn build_success_model(alpha: f64, beta: f64) -> String {
    format!("let p ~ Beta({alpha}, {beta})\nlet success ~ Bernoulli(p)")
}

/// The Keystone table that records per-model outcome feedback.
pub struct FeedbackTable;
impl Table for FeedbackTable {
    const NAME: &'static str = "augur_feedback";
    const COLUMNS: &'static [&'static str] = &["model_id", "outcome"];
}

/// The Keystone table holding vector memories and their similarity scores.
pub struct MemoryTable;
impl Table for MemoryTable {
    const NAME: &'static str = "augur_memories";
    const COLUMNS: &'static [&'static str] = &["strategy_id", "similarity"];
}

/// Tally `success`/`failure` outcomes from a `FeedbackTable` query result.
pub fn tally_outcomes(result: &QueryResult) -> RelationalPriorData {
    let mut successes = 0u64;
    let mut failures = 0u64;
    for row in &result.rows {
        match row.get_value(
            row.column_names()
                .iter()
                .position(|c| c == "outcome")
                .unwrap_or(0),
        ) {
            Value::Text(s) => match s.as_str() {
                "success" => successes += 1,
                "failure" => failures += 1,
                _ => {}
            },
            Value::Int(1) => successes += 1,
            Value::Int(0) => failures += 1,
            _ => {}
        }
    }
    RelationalPriorData {
        successes,
        failures,
    }
}

/// Fetch and build a Beta prior from relational outcome feedback for `model_id`.
pub async fn prior_from_relational_query<Q: KeystoneQuerier>(
    q: &mut Q,
    model_id: &str,
    concentration: f64,
) -> Result<(f64, f64), BridgeError> {
    let (sql, params) = QueryBuilder::<FeedbackTable>::new()
        .filter_eq("model_id", Value::Text(model_id.to_string()))
        .build();
    let res = q.query_params(&sql, &params).await?;
    if res.rows.is_empty() {
        return Err(BridgeError::EmptyResult);
    }
    let data = tally_outcomes(&res);
    Ok(prior_from_relational(&data, concentration))
}

/// Fetch and build a Beta prior from vector-memory similarity for `strategy_id`.
pub async fn prior_from_vector_query<Q: KeystoneQuerier>(
    q: &mut Q,
    strategy_id: &str,
    concentration: f64,
) -> Result<(f64, f64), BridgeError> {
    let (sql, params) = QueryBuilder::<MemoryTable>::new()
        .filter_eq("strategy_id", Value::Text(strategy_id.to_string()))
        .order_by("similarity", Order::Desc)
        .limit(100)
        .build();
    let res = q.query_params(&sql, &params).await?;
    if res.rows.is_empty() {
        return Err(BridgeError::EmptyResult);
    }
    let sim_col = res
        .columns
        .iter()
        .position(|c| c == "similarity")
        .unwrap_or(0);
    let mut sum = 0.0;
    let mut n = 0u64;
    for row in &res.rows {
        match row.get_value(sim_col) {
            Value::Float(s) => {
                sum += s;
                n += 1;
            }
            Value::Int(i) => {
                sum += i as f64;
                n += 1;
            }
            _ => {}
        }
    }
    if n == 0 {
        return Err(BridgeError::EmptyResult);
    }
    Ok(prior_from_vector_similarity(sum / n as f64, concentration))
}

/// Build a runnable Augur model (already lowered) from a Keystone-derived
/// Beta prior. This is the "live prior update" surface: a Keystone query
/// produces `(alpha, beta)`, which becomes the model's prior over `p`.
pub fn model_from_prior(alpha: f64, beta: f64) -> Result<tpt_augur_ir::Model, BridgeError> {
    let src = build_success_model(alpha, beta);
    let parsed = parse(&src);
    if parsed.has_errors() {
        return Err(BridgeError::Model(format!("{:?}", parsed.diagnostics)));
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        return Err(BridgeError::Model(format!("{:?}", lowered.diagnostics)));
    }
    Ok(lowered.model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory querier returning a canned `QueryResult`.
    pub struct MockKeystone {
        pub result: QueryResult,
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
                    &["model_id", "outcome"],
                    &[Some(b"m".to_vec()), Some(b"success".to_vec())],
                ),
                Row::new(
                    &["model_id", "outcome"],
                    &[Some(b"m".to_vec()), Some(b"success".to_vec())],
                ),
                Row::new(
                    &["model_id", "outcome"],
                    &[Some(b"m".to_vec()), Some(b"failure".to_vec())],
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
                    &["strategy_id", "similarity"],
                    &[Some(b"s".to_vec()), Some(b"0.8".to_vec())],
                ),
                Row::new(
                    &["strategy_id", "similarity"],
                    &[Some(b"s".to_vec()), Some(b"0.4".to_vec())],
                ),
            ],
            None,
        )
    }

    use tpt_sdk::keystone::Row;

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

    #[tokio::test]
    async fn relational_query_builds_prior() {
        let mut mock = MockKeystone {
            result: feedback_result(),
        };
        let (a, b) = prior_from_relational_query(&mut mock, "m", 2.0)
            .await
            .unwrap();
        // 2 successes, 1 failure, +2 pseudocounts each ⇒ (4, 3)
        assert_eq!((a, b), (4.0, 3.0));
    }

    #[tokio::test]
    async fn vector_query_builds_prior() {
        let mut mock = MockKeystone {
            result: memory_result(),
        };
        let (a, b) = prior_from_vector_query(&mut mock, "s", 10.0).await.unwrap();
        // mean similarity 0.6 ⇒ above-even success prior
        assert!(a > b);
    }

    #[test]
    fn model_from_prior_lowers() {
        let model = model_from_prior(4.0, 3.0).unwrap();
        assert_eq!(model.prior_order, vec!["p", "success"]);
    }

    #[tokio::test]
    async fn empty_result_is_an_error() {
        let mut mock = MockKeystone {
            result: QueryResult::new(vec!["model_id".into(), "outcome".into()], vec![], None),
        };
        let err = prior_from_relational_query(&mut mock, "m", 2.0).await;
        assert!(matches!(err, Err(BridgeError::EmptyResult)));
    }
}
