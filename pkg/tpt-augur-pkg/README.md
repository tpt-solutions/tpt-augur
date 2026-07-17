# tpt-augur-pkg

`Augur.toml` package manifest format and registry wrapper for the
[Augur](https://github.com/tpt-solutions/tpt-augur) probabilistic programming
language.

```rust
use tpt_augur_pkg::Manifest;

let manifest = Manifest::parse(r#"
modules = ["src/model.augur"]

[package]
name = "my-model"
version = "0.1.0"
"#).unwrap();
```

Part of the Augur workspace — see the
[main repository](https://github.com/tpt-solutions/tpt-augur) for the language
overview and other crates (`tpt-augur-frontend`, `tpt-augur-ir`, `tpt-augur-cli`).

## License

MIT OR Apache-2.0
