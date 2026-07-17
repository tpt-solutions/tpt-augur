//! End-to-end integration test against a *running* TPT Keystone instance.
//!
//! Ignored by default so CI doesn't require a live database. Run with:
//! `cargo test -p tpt-augur-keystone --test live_integration -- --ignored`
//! against a Keystone node listening on `127.0.0.1:5432` with the
//! `augur_feedback(model_id, outcome)` and `augur_memories(strategy_id,
//! similarity)` tables populated.

use tpt_augur_keystone::{
    model_from_prior, prior_from_relational_query, prior_from_vector_query, RealKeystone,
};

#[tokio::test]
#[ignore = "requires a running TPT Keystone instance at 127.0.0.1:5432"]
async fn live_prior_update_end_to_end() {
    let mut real = RealKeystone::connect("127.0.0.1:5432")
        .await
        .expect("connect to keystone");

    // Relational branch: historical outcomes ⇒ Beta prior.
    let (alpha, beta) = prior_from_relational_query(&mut real, "demo_model", 2.0)
        .await
        .expect("relational prior query");
    let model = model_from_prior(alpha, beta).expect("lower model");
    assert_eq!(model.prior_order, vec!["p", "success"]);

    // Vector branch: memory similarity ⇒ Beta prior.
    let (v_alpha, v_beta) = prior_from_vector_query(&mut real, "demo_strategy", 10.0)
        .await
        .expect("vector prior query");
    assert!(v_alpha > 0.0 && v_beta > 0.0);
}
