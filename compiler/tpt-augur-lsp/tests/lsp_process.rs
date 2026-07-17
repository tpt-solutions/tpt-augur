//! End-to-end integration tests that drive the compiled `tpt-augur-lsp` binary
//! over its stdio JSON-RPC / LSP `Content-Length` framing protocol.

use std::io::{BufReader, Read, Write};
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_tpt-augur-lsp");

struct Client {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
}

impl Client {
    fn spawn() -> Self {
        let mut child = Command::new(BIN)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        Client {
            child,
            reader: BufReader::new(stdout),
        }
    }

    fn send(&mut self, msg: &Value) {
        let body = serde_json::to_string(msg).unwrap();
        let stdin = self.child.stdin.as_mut().unwrap();
        write!(stdin, "Content-Length: {}\r\n\r\n", body.len()).unwrap();
        stdin.write_all(body.as_bytes()).unwrap();
        stdin.flush().unwrap();
    }

    fn recv(&mut self) -> Value {
        use std::io::BufRead;
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            self.reader.read_line(&mut line).unwrap();
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
                content_length = rest.trim().parse().unwrap();
            }
        }
        let mut buf = vec![0u8; content_length];
        self.reader.read_exact(&mut buf).unwrap();
        serde_json::from_slice(&buf).unwrap()
    }
}

fn initialize(client: &mut Client) {
    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }));
    let resp = client.recv();
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"]["capabilities"]["hoverProvider"], true);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {}
    }));
}

#[test]
fn initialize_returns_capabilities() {
    let mut client = Client::spawn();
    initialize(&mut client);
    let _ = client.child.kill();
}

#[test]
fn did_open_publishes_diagnostics() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {
            "textDocument": {
                "uri": "file:///model.augur",
                "text": "let x = y + 1"
            }
        }
    }));
    let notif = client.recv();
    assert_eq!(notif["method"], "textDocument/publishDiagnostics");
    let diags = notif["params"]["diagnostics"].as_array().unwrap();
    assert!(!diags.is_empty());

    let _ = client.child.kill();
}

#[test]
fn hover_returns_markdown_for_known_distribution() {
    let mut client = Client::spawn();
    initialize(&mut client);

    let src = "let mu ~ Normal(0, 1)";
    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": { "textDocument": { "uri": "file:///hover.augur", "text": src } }
    }));
    let _diag = client.recv();

    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "textDocument/hover",
        "params": {
            "textDocument": { "uri": "file:///hover.augur" },
            "position": { "line": 0, "character": 9 }
        }
    }));
    let resp = client.recv();
    assert_eq!(resp["id"], 2);
    assert!(resp["result"]["contents"]["value"]
        .as_str()
        .unwrap_or("")
        .contains("Gaussian"));

    let _ = client.child.kill();
}

#[test]
fn hover_on_unopened_document_returns_null() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "textDocument/hover",
        "params": {
            "textDocument": { "uri": "file:///never-opened.augur" },
            "position": { "line": 0, "character": 0 }
        }
    }));
    let resp = client.recv();
    assert_eq!(resp["id"], 3);
    assert_eq!(resp["result"], Value::Null);

    let _ = client.child.kill();
}

#[test]
fn did_change_full_text_sync_updates_document() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": { "textDocument": { "uri": "file:///change.augur", "text": "let x = y + 1" } }
    }));
    let _diag = client.recv();

    // Full-text replacement (no `range`) with a now-valid program.
    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": { "uri": "file:///change.augur" },
            "contentChanges": [{ "text": "let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5" }]
        }
    }));
    let notif = client.recv();
    let diags = notif["params"]["diagnostics"].as_array().unwrap();
    assert!(diags.iter().all(|d| d["severity"] == 2), "diags: {diags:?}");

    let _ = client.child.kill();
}

#[test]
fn did_change_range_edit_updates_document() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": { "textDocument": { "uri": "file:///range.augur", "text": "let mu ~ Normal(0, 1)" } }
    }));
    let _diag = client.recv();

    // Replace "0, 1" (chars 16..20) with "0, 2".
    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": { "uri": "file:///range.augur" },
            "contentChanges": [{
                "range": {
                    "start": { "line": 0, "character": 16 },
                    "end": { "line": 0, "character": 20 }
                },
                "text": "0, 2"
            }]
        }
    }));
    let _diag2 = client.recv();

    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "augur/inferenceGraph",
        "params": { "textDocument": { "uri": "file:///range.augur" } }
    }));
    let resp = client.recv();
    assert!(resp["result"]["dot"]
        .as_str()
        .unwrap_or("")
        .contains("digraph augur_inference_graph"));

    let _ = client.child.kill();
}

#[test]
fn did_save_republishes_diagnostics() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": { "textDocument": { "uri": "file:///save.augur", "text": "let mu ~ Normal(0, 1)" } }
    }));
    let _diag = client.recv();

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didSave",
        "params": { "textDocument": { "uri": "file:///save.augur" } }
    }));
    let notif = client.recv();
    assert_eq!(notif["method"], "textDocument/publishDiagnostics");

    let _ = client.child.kill();
}

#[test]
fn inference_graph_for_unopened_document_is_null() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "augur/inferenceGraph",
        "params": { "textDocument": { "uri": "file:///nope.augur" } }
    }));
    let resp = client.recv();
    assert_eq!(resp["result"], Value::Null);

    let _ = client.child.kill();
}

#[test]
fn unknown_method_with_id_returns_method_not_found() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "textDocument/definition",
        "params": {}
    }));
    let resp = client.recv();
    assert_eq!(resp["id"], 6);
    assert_eq!(resp["error"]["code"], -32601);

    let _ = client.child.kill();
}

#[test]
fn unknown_notification_without_id_is_silently_ignored() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "$/cancelRequest",
        "params": {}
    }));

    // The server should not reply; a subsequent real request still gets a response,
    // proving the unknown notification was skipped rather than jamming the loop.
    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "initialize",
        "params": {}
    }));
    let resp = client.recv();
    assert_eq!(resp["id"], 7);

    let _ = client.child.kill();
}

#[test]
fn shutdown_then_exit_terminates_cleanly() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "shutdown",
        "params": Value::Null
    }));
    let resp = client.recv();
    assert_eq!(resp["id"], 8);
    assert_eq!(resp["result"], Value::Null);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "exit",
        "params": Value::Null
    }));

    let status = client.child.wait().unwrap();
    assert!(status.success(), "status: {status:?}");
}

#[test]
fn exit_without_shutdown_exits_nonzero() {
    let mut client = Client::spawn();
    initialize(&mut client);

    client.send(&json!({
        "jsonrpc": "2.0",
        "method": "exit",
        "params": Value::Null
    }));

    let status = client.child.wait().unwrap();
    assert!(!status.success(), "status: {status:?}");
}

#[test]
fn malformed_json_message_is_skipped_without_crashing() {
    let mut client = Client::spawn();

    // Send a syntactically invalid JSON body; the server should skip it and
    // keep serving subsequent well-formed requests.
    let body = "{not valid json";
    let stdin = client.child.stdin.as_mut().unwrap();
    write!(stdin, "Content-Length: {}\r\n\r\n", body.len()).unwrap();
    stdin.write_all(body.as_bytes()).unwrap();
    stdin.flush().unwrap();

    initialize(&mut client);

    let _ = client.child.kill();
}
