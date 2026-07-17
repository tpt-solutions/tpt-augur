# tpt-augur-ir

Typed intermediate representation for the
[Augur](https://github.com/tpt-solutions/tpt-augur) probabilistic programming
language: lowering from the AST, type-checking, degenerate-parameter static
analysis, and uncertainty propagation through standard math operations.

```rust
use tpt_augur_frontend::parse;
use tpt_augur_ir::lower;

let r = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
let lr = lower(&r.program);
assert_eq!(lr.model.prior_order, vec!["mu".to_string()]);
```

Part of the Augur workspace — see the
[main repository](https://github.com/tpt-solutions/tpt-augur) for the language
overview, examples, and other crates (`tpt-augur-frontend`, `tpt-augur-runtime`,
`tpt-augur-std`, `tpt-augur-cli`).

## License

MIT OR Apache-2.0
