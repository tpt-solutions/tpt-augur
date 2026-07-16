//! Standalone LSP server for Augur, speaking JSON-RPC 2.0 over stdio with
//! LSP `Content-Length` framing. Depends only on the existing `augur-*`
//! crates and `serde_json` — no external LSP framework, so it builds (and runs)
//! without network access and stays in lockstep with the parser/type-checker.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

use augur_lsp::{analyze_document, hover_at, inference_graph_dot};

struct Server {
    documents: HashMap<String, String>,
    shutdown: bool,
}

fn main() {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    let mut server = Server {
        documents: HashMap::new(),
        shutdown: false,
    };

    while let Some(msg) = read_message(&mut reader) {
        let value: Value = match serde_json::from_str(&msg) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match handle(&mut server, value, &mut writer) {
            Ok(Some(response)) => {
                if write_message(&mut writer, &response).is_err() {
                    break;
                }
            }
            Ok(None) => {}
            Err(_) => break,
        }
    }
}

fn handle(server: &mut Server, msg: Value, out: &mut impl Write) -> io::Result<Option<Value>> {
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = msg.get("id").cloned();

    match method {
        "initialize" => {
            let result = json!({
                "capabilities": {
                    "textDocumentSync": 1,
                    "hoverProvider": true,
                    "definitionProvider": false,
                    "referencesProvider": false,
                    "documentSymbolProvider": false,
                },
                "serverInfo": { "name": "augur-lsp", "version": env!("CARGO_PKG_VERSION") }
            });
            Ok(Some(response(id, result)))
        }
        "initialized" => Ok(None),
        "shutdown" => {
            server.shutdown = true;
            Ok(Some(response(id, Value::Null)))
        }
        "exit" => {
            std::process::exit(if server.shutdown { 0 } else { 1 });
        }
        "textDocument/didOpen" => {
            let params = &msg["params"];
            let uri = params["textDocument"]["uri"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let text = params["textDocument"]["text"]
                .as_str()
                .unwrap_or("")
                .to_string();
            server.documents.insert(uri.clone(), text);
            publish_diagnostics(server, &uri, out)?;
            Ok(None)
        }
        "textDocument/didChange" => {
            let params = &msg["params"];
            let uri = params["textDocument"]["uri"]
                .as_str()
                .unwrap_or("")
                .to_string();
            if let Some(change) = params["contentChanges"].get(0) {
                let new_text = apply_change(server, &uri, change);
                server.documents.insert(uri.clone(), new_text);
            }
            publish_diagnostics(server, &uri, out)?;
            Ok(None)
        }
        "textDocument/didSave" => {
            let params = &msg["params"];
            let uri = params["textDocument"]["uri"]
                .as_str()
                .unwrap_or("")
                .to_string();
            publish_diagnostics(server, &uri, out)?;
            Ok(None)
        }
        "textDocument/hover" => {
            let params = &msg["params"];
            let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
            let line = params["position"]["line"].as_u64().unwrap_or(0) as u32;
            let character = params["position"]["character"].as_u64().unwrap_or(0) as u32;
            let result = server
                .documents
                .get(uri)
                .and_then(|src| {
                    let off = position_to_offset(src, line, character);
                    hover_at(src, off)
                })
                .map(|doc| {
                    json!({
                        "contents": { "kind": "markdown", "value": doc }
                    })
                });
            Ok(Some(response(id, result.unwrap_or(Value::Null))))
        }
        "augur/inferenceGraph" => {
            let params = &msg["params"];
            let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
            let result = server
                .documents
                .get(uri)
                .and_then(|src| inference_graph_dot(src))
                .map(|dot| json!({ "dot": dot }))
                .unwrap_or(Value::Null);
            Ok(Some(response(id, result)))
        }
        _ => {
            // Unknown notification/request: reply with method-not-found only if
            // it carries an id (i.e. it expects a response).
            if id.is_some() {
                let err = json!({
                    "code": -32601,
                    "message": format!("method not found: {method}"),
                });
                Ok(Some(response_error(id, err)))
            } else {
                Ok(None)
            }
        }
    }
}

fn publish_diagnostics(server: &Server, uri: &str, out: &mut impl Write) -> io::Result<()> {
    let diagnostics = match server.documents.get(uri) {
        Some(src) => analyze_document(src),
        None => Vec::new(),
    };
    let notification = json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": diagnostics,
        }
    });
    write_message(out, &notification)
}

fn apply_change(server: &Server, uri: &str, change: &Value) -> String {
    let current = server.documents.get(uri).cloned().unwrap_or_default();
    if let Some(text) = change.get("text").and_then(|t| t.as_str()) {
        if change.get("range").is_none() {
            return text.to_string();
        }
        // Range edit: replace the addressed span with `text`.
        if let (Some(start), Some(end)) = (
            change.get("range").and_then(|r| r.get("start")),
            change.get("range").and_then(|r| r.get("end")),
        ) {
            let so = position_to_offset(
                &current,
                start["line"].as_u64().unwrap_or(0) as u32,
                start["character"].as_u64().unwrap_or(0) as u32,
            );
            let eo = position_to_offset(
                &current,
                end["line"].as_u64().unwrap_or(0) as u32,
                end["character"].as_u64().unwrap_or(0) as u32,
            );
            let chars: Vec<char> = current.chars().collect();
            let mut next: String = chars[..so.min(chars.len())].iter().collect();
            next.push_str(text);
            next.extend(chars[eo.min(chars.len())..].iter());
            return next;
        }
    }
    current
}

fn response(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn response_error(id: Option<Value>, error: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": error })
}

/// Read a single LSP message (header block + JSON body) from `reader`.
fn read_message<R: BufRead>(reader: &mut R) -> Option<String> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 {
            return None; // EOF
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().ok();
        }
    }
    let len = content_length?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn write_message<W: Write>(writer: &mut W, msg: &Value) -> io::Result<()> {
    let body = serde_json::to_string(msg)?;
    let bytes = body.as_bytes();
    write!(writer, "Content-Length: {}\r\n\r\n", bytes.len())?;
    writer.write_all(bytes)?;
    writer.flush()?;
    Ok(())
}

/// Map an LSP (line, UTF-16 character) position back to a char offset in `src`.
fn position_to_offset(src: &str, line: u32, character: u32) -> usize {
    let chars: Vec<char> = src.chars().collect();
    let mut cur_line = 0u32;
    let mut unit = 0u32;
    for (idx, &c) in chars.iter().enumerate() {
        if cur_line == line && unit >= character {
            return idx;
        }
        if c == '\n' {
            if cur_line == line {
                return idx; // requested column past end of line
            }
            cur_line += 1;
            unit = 0;
        } else {
            unit += c.len_utf16() as u32;
        }
    }
    chars.len()
}
