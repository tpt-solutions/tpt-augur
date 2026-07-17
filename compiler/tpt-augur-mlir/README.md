# tpt-augur-mlir

Probabilistic MLIR-compatible dialect, optimization passes, and TPTIR
lowering for the [Augur](https://github.com/tpt-solutions/tpt-augur)
probabilistic programming language.

```rust
use tpt_augur_frontend::parse;
use tpt_augur_ir::lower;
use tpt_augur_mlir::compile_model_to_tptir;

let r = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
let lr = lower(&r.program);
let (tptir_text, _changes) = compile_model_to_tptir(&lr.model, "model", "cpu");
```

Part of the Augur workspace — see the
[main repository](https://github.com/tpt-solutions/tpt-augur) for the language
overview, examples, and other crates (`tpt-augur-frontend`, `tpt-augur-ir`,
`tpt-augur-runtime`, `tpt-augur-cli`).

## License

MIT OR Apache-2.0
