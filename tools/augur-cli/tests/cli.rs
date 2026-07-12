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
fn check_missing_file_reports_error() {
    let path = std::env::temp_dir().join(format!("augur_cli_{}_missing.augur", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let out = Command::new(BIN).arg("check").arg(&path).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("error:"), "stderr was: {stderr}");
}

#[test]
fn run_with_invalid_model_fails() {
    let path = write_temp("run_invalid.augur", "let x = y + 1");
    let out = Command::new(BIN).arg("run").arg(&path).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("undeclared"), "stderr was: {stderr}");
}

#[test]
fn run_with_unknown_engine_fails() {
    let path = write_temp("run_bad_engine.augur", valid_model());
    let out = Command::new(BIN)
        .arg("run")
        .arg(&path)
        .args(["-e", "not-a-real-engine"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn run_with_explicit_engine_selects_it() {
    let path = write_temp("run_explicit_engine.augur", valid_model());
    let out = Command::new(BIN)
        .arg("run")
        .arg(&path)
        .args(["-e", "mh", "-n", "100", "-c", "1", "--warmup", "50"])
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("engine: mh (auto-selected: false)"), "stdout was: {stdout}");
}

#[test]
fn fmt_with_invalid_model_fails() {
    let path = write_temp("fmt_invalid.augur", "let x = ~~~ ###");
    let out = Command::new(BIN).arg("fmt").arg(&path).output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn build_with_invalid_model_fails() {
    let path = write_temp("build_invalid.augur", "let x = y + 1");
    let out = Command::new(BIN).arg("build").arg(&path).output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn build_writes_to_output_file() {
    let path = write_temp("build_out.augur", valid_model());
    let out_path = std::env::temp_dir().join(format!("augur_cli_{}_build_out.tptir", std::process::id()));
    let _ = std::fs::remove_file(&out_path);
    let out = Command::new(BIN)
        .arg("build")
        .arg(&path)
        .args(["-o"])
        .arg(&out_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("wrote TPTIR"), "stdout was: {stdout}");
    let written = std::fs::read_to_string(&out_path).unwrap();
    assert!(written.contains("func.func @model"));
    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn graph_with_invalid_model_fails() {
    let path = write_temp("graph_invalid.augur", "let x = y + 1");
    let out = Command::new(BIN).arg("graph").arg(&path).output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn graph_writes_to_output_file() {
    let path = write_temp("graph_out.augur", valid_model());
    let out_path = std::env::temp_dir().join(format!("augur_cli_{}_graph_out.dot", std::process::id()));
    let _ = std::fs::remove_file(&out_path);
    let out = Command::new(BIN)
        .arg("graph")
        .arg(&path)
        .args(["-o"])
        .arg(&out_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("wrote inference graph"), "stdout was: {stdout}");
    let written = std::fs::read_to_string(&out_path).unwrap();
    assert!(written.contains("digraph augur_inference_graph"));
    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn repl_with_invalid_model_fails() {
    let mut child = Command::new(BIN)
        .arg("repl")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(b"let x = y + 1").unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn publish_then_install_round_trip_via_cli() {
    let registry_dir = std::env::temp_dir().join(format!("augur_cli_{}_registry", std::process::id()));
    let _ = std::fs::remove_dir_all(&registry_dir);

    let manifest_path = write_temp(
        "Augur.toml",
        "modules = [\"model.augur\"]\n\n[package]\nname = \"cli-pkg\"\nversion = \"0.1.0\"\n",
    );
    let src_path = write_temp("model.augur", valid_model());

    let publish_out = Command::new(BIN)
        .env("AUGUR_REGISTRY", &registry_dir)
        .arg("publish")
        .arg(&manifest_path)
        .args(["--src"])
        .arg(&src_path)
        .output()
        .unwrap();
    assert!(
        publish_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&publish_out.stderr)
    );
    let publish_stdout = String::from_utf8_lossy(&publish_out.stdout);
    assert!(publish_stdout.contains("published cli-pkg@0.1.0"), "stdout was: {publish_stdout}");

    let dest_dir = std::env::temp_dir().join(format!("augur_cli_{}_install_dest", std::process::id()));
    let _ = std::fs::remove_dir_all(&dest_dir);
    let install_out = Command::new(BIN)
        .env("AUGUR_REGISTRY", &registry_dir)
        .arg("install")
        .arg("cli-pkg")
        .arg("0.1.0")
        .args(["-d"])
        .arg(&dest_dir)
        .output()
        .unwrap();
    assert!(
        install_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&install_out.stderr)
    );
    let install_stdout = String::from_utf8_lossy(&install_out.stdout);
    assert!(install_stdout.contains("installed"), "stdout was: {install_stdout}");

    let _ = std::fs::remove_dir_all(&registry_dir);
    let _ = std::fs::remove_dir_all(&dest_dir);
}

#[test]
fn install_missing_package_fails() {
    let registry_dir = std::env::temp_dir().join(format!("augur_cli_{}_registry_missing", std::process::id()));
    let _ = std::fs::remove_dir_all(&registry_dir);
    let dest_dir = std::env::temp_dir().join(format!("augur_cli_{}_install_dest_missing", std::process::id()));
    let _ = std::fs::remove_dir_all(&dest_dir);

    let out = Command::new(BIN)
        .env("AUGUR_REGISTRY", &registry_dir)
        .arg("install")
        .arg("does-not-exist")
        .arg("0.0.0")
        .args(["-d"])
        .arg(&dest_dir)
        .output()
        .unwrap();
    assert!(!out.status.success());

    let _ = std::fs::remove_dir_all(&registry_dir);
    let _ = std::fs::remove_dir_all(&dest_dir);
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
