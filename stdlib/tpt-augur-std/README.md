# tpt-augur-std

Standard library of probability distributions for the
[Augur](https://github.com/tpt-solutions/tpt-augur) probabilistic programming
language: exact log-densities and samplers for Normal, HalfNormal, Beta,
Gamma, Uniform, Exponential, Binomial, Poisson, and Bernoulli.

```rust
use tpt_augur_std::{seeded_rng, Dist};

let normal = Dist::Normal { mu: 0.0, sigma: 1.0 };
let mut rng = seeded_rng(42);
let x = normal.sample(&mut rng);
let log_density = normal.logp(x);
```

Part of the Augur workspace — see the
[main repository](https://github.com/tpt-solutions/tpt-augur) for the language
overview, examples, and other crates (`tpt-augur-frontend`, `tpt-augur-ir`,
`tpt-augur-runtime`, `tpt-augur-cli`).

## License

MIT OR Apache-2.0
