//! Integration tests for the Augur LSP support library.

use augur_lsp::{analyze_document, hover_at, inference_graph_dot};

#[test]
fn clean_program_has_no_error_diagnostics() {
    let src = "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5";
    let diags = analyze_document(src);
    assert!(diags.iter().all(|d| d["severity"] == 2));
}

#[test]
fn undeclared_variable_is_reported() {
    let src = "let x = y + 1";
    let diags = analyze_document(src);
    assert!(!diags.is_empty());
    assert!(diags
        .iter()
        .any(|d| d["message"].as_str().unwrap_or("").contains("undeclared")));
}

#[test]
fn unknown_distribution_is_reported() {
    let src = "let x ~ Banana(1, 2)";
    let diags = analyze_document(src);
    assert!(diags
        .iter()
        .any(|d| d["message"].as_str().unwrap_or("").contains("not a known distribution")));
}

#[test]
fn degenerate_parameters_warn() {
    let src = "let s ~ Normal(0, -1)";
    let diags = analyze_document(src);
    assert!(diags
        .iter()
        .any(|d| d["severity"] == 2 && d["message"].as_str().unwrap_or("").contains("degenerate")));
}

#[test]
fn inference_graph_emits_digraph() {
    let src = "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5";
    let dot = inference_graph_dot(src).expect("graph");
    assert!(dot.contains("digraph augur_inference_graph"));
    assert!(dot.contains("sample mu"));
}

#[test]
fn inference_graph_none_for_broken_program() {
    assert!(inference_graph_dot("let x = y + 1").is_none());
}

#[test]
fn hover_on_distribution_returns_doc() {
    let src = "let mu ~ Normal(0, 1)";
    let doc = hover_at(src, 9);
    assert!(doc.is_some());
    assert!(doc.unwrap().contains("Gaussian"));
}

#[test]
fn hover_on_non_distribution_returns_none() {
    let src = "let mu ~ Normal(0, 1)";
    // Cursor sits on the variable name "mu" (offset 4), not a distribution.
    assert!(hover_at(src, 4).is_none());
}

#[test]
fn hover_at_out_of_bounds_is_none() {
    let src = "let mu ~ Normal(0, 1)";
    // Beyond the end of the source.
    assert!(hover_at(src, src.len() + 5).is_none());
}
