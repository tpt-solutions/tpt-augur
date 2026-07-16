# augur-runtime

Inference runtime for the [Augur](https://github.com/tpt-solutions/tpt-augur)
probabilistic programming language: random-walk Metropolis–Hastings,
Hamiltonian Monte Carlo, mean-field variational inference, and a bootstrap
particle filter, with automatic engine selection from model topology.

```rust
use augur_frontend::parse;
use augur_ir::lower;
use augur_runtime::{run, InferOptions};

let r = parse("let mu ~ Normal(0, 1)\nobserve Normal(mu, 1) = 0.5");
let lr = lower(&r.program);
let result = run(&lr.model, &InferOptions::default());
```

Part of the Augur workspace — see the
[main repository](https://github.com/tpt-solutions/tpt-augur) for the language
overview, examples, and other crates (`augur-frontend`, `augur-ir`,
`augur-std`, `augur-cli`).

## License

MIT OR Apache-2.0
