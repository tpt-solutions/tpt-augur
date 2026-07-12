# Augur in Neovim

Neovim's built-in LSP client can talk to `augur-lsp` directly — no plugin
required beyond `nvim-lspconfig` (or a manual `vim.lsp.start` call). Diagnostics,
hover, and the inference-graph custom request all work.

## Prerequisites

Build the language server from this repo:

```sh
cargo build -p augur-lsp
# the binary lands at target/debug/augur-lsp
```

Make it available on your `PATH`, or reference it by absolute path below.

## Option A — `nvim-lspconfig` (recommended)

`augur-lsp` isn't bundled with `lspconfig`, but you can register it with
`lspconfig.configs`:

```lua
local lspconfig = require("lspconfig")
local configs = require("lspconfig.configs")

if not configs.augur then
  configs.augur = {
    default_config = {
      cmd = { "augur-lsp" },
      filetypes = { "augur" },
      root_dir = lspconfig.util.root_pattern(".git", "Augur.toml"),
      single_file_support = true,
    },
  }
end

lspconfig.augur.setup({
  on_attach = function(client, bufnr)
    vim.keymap.set("n", "K", vim.lsp.buf.hover, { buffer = bufnr })
  end,
})
```

Filetype detection for `.augur`:

```lua
vim.filetype.add({ extension = { augur = "augur" } })
```

## Option B — manual `vim.lsp.start`

```lua
vim.api.nvim_create_autocmd("FileType", {
  pattern = "augur",
  callback = function()
    vim.lsp.start({
      name = "augur",
      cmd = { "augur-lsp" },
      root_dir = vim.fs.dirname(vim.fs.find({ ".git", "Augur.toml" }, { upward = true })[1]),
    })
  end,
})
```

## Requesting the inference graph

The server answers a custom `augur/inferenceGraph` request. Call it from Lua:

```lua
local params = { textDocument = vim.lsp.util.make_text_document_params() }
local result = vim.lsp.buf_request_sync(0, "augur/inferenceGraph", params, 1000)
-- result is { <client_id> = { result = { dot = "digraph ..." } } }
```

Pipe the returned `dot` string to `dot -Tpng` (from Graphviz) to render it.
