//! Language-server support for Augur: turn source text into LSP diagnostics,
//! inference-graph DOT, and hover docs — all derived from the existing
//! `augur_frontend` parser and `augur_ir::lower` type-checker (the same
//! diagnostics-first pipeline the CLI uses). `tptb-lsp` is not checked out in
//! this workspace (mirroring the Phase 1 parser decision), so Augur ships its
//! own small LSP server that reuses the diagnostics machinery rather than
//! depending on a frontend built for a different language.

use augur_frontend::{lexer, parse, Severity};
use augur_ir::lower;
use augur_mlir::{build_graph, to_dot};
use serde_json::{json, Value};

/// Analyze `src` and return a list of LSP-shaped `Diagnostic` objects
/// (distribution type-checking diagnostics + parse errors).
pub fn analyze_document(src: &str) -> Vec<Value> {
    let parsed = parse(src);
    let mut diags = Vec::new();
    for d in &parsed.diagnostics {
        diags.push(lsp_diag(
            src,
            &d.message,
            d.severity == Severity::Error,
            d.span.start,
            d.span.end,
        ));
    }
    // Only type-check when the parse is clean; otherwise the spanned IR
    // diagnostics would just be noise on top of the parse errors.
    if parsed.has_errors() {
        return diags;
    }
    let lowered = lower(&parsed.program);
    for d in &lowered.diagnostics {
        diags.push(lsp_diag(
            src,
            &d.message,
            d.is_error(),
            d.span.start,
            d.span.end,
        ));
    }
    diags
}

/// Build the probabilistic inference graph (DOT) for `src`, or `None` if the
/// program does not parse / type-check.
pub fn inference_graph_dot(src: &str) -> Option<String> {
    let parsed = parse(src);
    if parsed.has_errors() {
        return None;
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        return None;
    }
    let graph = build_graph(&lowered.model);
    Some(to_dot(&graph))
}

/// Hover documentation for the distribution constructor at `char_offset`, if
/// the cursor sits on one.
pub fn hover_at(src: &str, char_offset: usize) -> Option<String> {
    let (tokens, _) = lexer::lex(src);
    for tok in &tokens {
        if tok.span.start <= char_offset && char_offset < tok.span.end {
            if let augur_frontend::lexer::Tok::Ident(name) = &tok.tok {
                if let Some(doc) = distribution_doc(name) {
                    return Some(doc.to_string());
                }
            }
        }
    }
    None
}

fn distribution_doc(name: &str) -> Option<&'static str> {
    let doc = match name {
        "Normal" => {
            "**Normal(μ, σ)** — Gaussian prior.\n- μ: mean\n- σ: standard deviation (σ > 0)"
        }
        "HalfNormal" => "**HalfNormal(σ)** — half-Gaussian, support [0, ∞).\n- σ: scale (σ > 0)",
        "Beta" => "**Beta(α, β)** — support (0, 1), conjugate to Bernoulli/Binomial.\n- α, β > 0",
        "Gamma" => "**Gamma(α, β)** — support (0, ∞).\n- α: shape, β: rate (both > 0)",
        "Uniform" => "**Uniform(lo, hi)** — support [lo, hi].\n- requires lo < hi",
        "Exponential" => "**Exponential(λ)** — support [0, ∞).\n- λ: rate (λ > 0)",
        "Binomial" => "**Binomial(n, p)** — counts in n trials.\n- n ≥ 0, p ∈ [0, 1]",
        "Poisson" => "**Poisson(λ)** — count prior.\n- λ > 0",
        "Bernoulli" => "**Bernoulli(p)** — binary prior.\n- p ∈ [0, 1]",
        _ => return None,
    };
    Some(doc)
}

fn lsp_diag(src: &str, message: &str, is_error: bool, start: usize, end: usize) -> Value {
    json!({
        "range": range(src, start, end),
        "severity": if is_error { 1 } else { 2 },
        "source": "augur",
        "message": message,
    })
}

/// Convert a (start, end) char-offset span into an LSP `Range`. LSP positions
/// are 0-based lines with UTF-16 code-unit character offsets; Augur `Span`s
/// are 0-based char offsets, so we measure both line breaks and UTF-16 width.
fn range(src: &str, start: usize, end: usize) -> Value {
    let (sl, sc) = position_at(src, start);
    let (el, ec) = position_at(src, end.max(start));
    json!({
        "start": { "line": sl, "character": sc },
        "end": { "line": el, "character": ec },
    })
}

fn position_at(src: &str, offset: usize) -> (u32, u32) {
    let chars: Vec<char> = src.chars().collect();
    let mut line = 0u32;
    let mut line_start = 0usize;
    let target = offset.min(chars.len());
    for (i, ch) in chars.iter().enumerate().take(target) {
        if *ch == '\n' {
            line += 1;
            line_start = i + 1;
        }
    }
    let in_line: String = chars[line_start..target].iter().collect();
    let character: u32 = in_line.chars().map(|c| c.len_utf16() as u32).sum();
    (line, character)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_report_undeclared_variable() {
        let src = "let x = y + 1";
        let diags = analyze_document(src);
        assert!(!diags.is_empty());
        assert!(diags
            .iter()
            .any(|d| d["message"].as_str().unwrap_or("").contains("undeclared")));
    }

    #[test]
    fn clean_program_has_no_errors() {
        let src = "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5";
        let diags = analyze_document(src);
        assert!(diags.iter().all(|d| d["severity"] == 2));
    }

    #[test]
    fn degenerate_warning_surfaces() {
        let src = "let s ~ Normal(0, -1)";
        let diags = analyze_document(src);
        assert!(diags
            .iter()
            .any(|d| d["severity"] == 2
                && d["message"].as_str().unwrap_or("").contains("degenerate")));
    }

    #[test]
    fn inference_graph_emits_digraph() {
        let src = "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5";
        let dot = inference_graph_dot(src).expect("graph");
        assert!(dot.contains("digraph augur_inference_graph"));
        assert!(dot.contains("sample mu"));
    }

    #[test]
    fn hover_on_distribution_returns_doc() {
        let src = "let mu ~ Normal(0, 1)";
        // cursor on the "Normal" identifier (offset 9)
        let doc = hover_at(src, 9);
        assert!(doc.is_some());
        assert!(doc.unwrap().contains("Gaussian"));
    }

    #[test]
    fn position_conversion_handles_multibyte() {
        // "😀" (U+1F600) is a surrogate pair = 2 UTF-16 units; offset 2 lands
        // after the emoji and its following space, so the column is 3 units.
        let src = "😀 = 1";
        let (line, character) = position_at(src, 2);
        assert_eq!(line, 0);
        assert_eq!(character, 3);
    }
}
