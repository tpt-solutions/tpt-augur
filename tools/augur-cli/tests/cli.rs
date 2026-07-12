//! End-to-end integration tests that drive the compiled `augur` CLI binary.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_augur");

fn write_temp(name: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("augur_cli_{}_{}", std::process::id(), name));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    path
}

fn valid_model() -> &'static str {
    "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5"
}

#[test]
fn check_valid_model_succeeds() {
    let path = write_temp("valid.augur", valid_model());
    let out = Command::new(BIN).arg("check").arg(&path).output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("ok:"), "stdout was: {stdout}");
}

#[test]
fn check_invalid_model_fails() {
    let path = write_temp("invalid.augur", "let x = y + 1");
    let out = Command::new(BIN).arg("check").arg(&path).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("undeclared"), "stderr was: {stderr}");
}

#[test]
fn fmt_emits_canonical_source() {
    let path = write_temp("fmt.augur", "let mu ~ Normal(0,1)\nobserve Normal(mu,1)=0.5");
    let out = Command::new(BIN).arg("fmt").arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("let mu ~ Normal(0.0, 1.0)"), "stdout was: {stdout}");
}

#[test]
fn graph_emits_digraph() {
    let path = write_temp("graph.augur", valid_model());
    let out = Command::new(BIN).arg("graph").arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("digraph augur_inference_graph"), "stdout was: {stdout}");
}

#[test]
fn build_emits_tptir() {
    let path = write_temp("build.augur", valid_model());
    let out = Command::new(BIN).arg("build").arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("func.func @model"), "stdout was: {stdout}");
    assert!(stdout.contains("\"augur.sample\""), "stdout was: {stdout}");
}

#[test]
fn run_prints_posterior_summary() {
    let path = write_temp("run.augur", valid_model());
    let out = Command::new(BIN)
        .arg("run")
        .arg(&path)
        .args(["-n", "300", "-c", "2", "--warmup", "150"])
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("engine:"), "stdout was: {stdout}");
    assert!(stdout.contains("mu"), "stdout was: {stdout}");
}

#[test]
fn repl_reads_model_from_stdin() {
    let mut child = Command::new(BIN)
        .arg("repl")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(valid_model().as_bytes()).unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("mu"), "stdout was: {stdout}");
}
