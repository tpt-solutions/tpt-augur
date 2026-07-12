import * as path from "path";
import { workspace, ExtensionContext, window, commands } from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient;

export function activate(context: ExtensionContext) {
  // augur-lsp is built by the workspace (`cargo build -p augur-lsp`) and
  // resolves next to this extension's node_modules in a dev setup, or on PATH.
  const command =
    workspace.getConfiguration("augur").get<string>("lspPath") || "augur-lsp";

  const serverOptions: ServerOptions = {
    command,
    args: [],
    transport: TransportKind.stdio,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "augur" }],
    synchronize: {
      fileEvents: workspace.createFileSystemWatcher("**/*.augur"),
    },
  };

  client = new LanguageClient(
    "augur",
    "Augur Language Server",
    serverOptions,
    clientOptions
  );

  client.start().then(
    () => {
      // Custom request handled by augur-lsp: returns Graphviz DOT for the
      // inference graph of the active document.
      context.subscriptions.push(
        commands.registerCommand("augur.showInferenceGraph", async () => {
          const editor = window.activeTextEditor;
          if (!editor || editor.document.languageId !== "augur") {
            window.showErrorMessage("Open an .augur file first.");
            return;
          }
          const dot = await client.sendRequest<{ dot?: string }>(
            "augur/inferenceGraph",
            { textDocument: { uri: editor.document.uri.toString() } }
          );
          if (dot && dot.dot) {
            const doc = await workspace.openTextDocument({
              content: dot.dot,
              language: "dot",
            });
            await window.showTextDocument(doc, window.activeTextEditor?.viewColumn! + 1);
          } else {
            window.showErrorMessage(
              "Could not build an inference graph (parse or type errors?)."
            );
          }
        })
      );
    },
    (e) => window.showErrorMessage(`Failed to start Augur LSP: ${e}`)
  );
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
