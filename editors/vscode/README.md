# Augur VS Code Extension

Syntax highlighting and LSP features for the [Augur](https://github.com/tpt-solutions/tpt-augur)
probabilistic programming language.

## Features

- Syntax highlighting for `.augur` files (priors, observations, distributions,
  control flow).
- Live diagnostics from the Augur type-checker (distribution type errors,
  undeclared variables, degenerate-parameter warnings).
- Hover documentation for distribution constructors.
- **Augur: Show Inference Graph** command — renders the probabilistic
  inference graph as Graphviz DOT (requires a Graphviz viewer, e.g. the
  `graphviz` preview extension or `dot`).

## Building the language server

The extension talks to the `augur-lsp` binary, which is built from this repo:

```sh
cargo build -p augur-lsp
```

Make sure `augur-lsp` is on your `PATH`, or set the `augur.lspPath` setting to
its absolute path.

## Developing the extension

```sh
npm install
npm run compile      # tsc -> out/extension.js
# then press F5 in VS Code to launch the Extension Development Host
```
