# augur-frontend

Lexer, AST, and error-tolerant recursive-descent parser for the
[Augur](https://github.com/tpt-solutions/tpt-augur) probabilistic programming
language.

```rust
use augur_frontend::parse;

let result = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
assert!(!result.has_errors());
```

Part of the Augur workspace — see the
[main repository](https://github.com/tpt-solutions/tpt-augur) for the language
overview, examples, and other crates (`augur-ir`, `augur-runtime`,
`augur-std`, `augur-cli`).

## License

MIT OR Apache-2.0
