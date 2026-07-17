# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| `0.1.x` | :white_check_mark: |

## Reporting a vulnerability

If you discover a security vulnerability in TPT Augur, please report it
privately rather than opening a public issue.

- Email the maintainers at **security@tpt.example** (replace with the real
  address before publishing).
- Include a description of the issue, steps to reproduce, and any potential
  impact.
- You will receive an acknowledgement within 5 business days, and we will keep
  you informed as the issue is triaged and resolved.

Please do not disclose the vulnerability publicly until a fix has been released.

## Scope notes

Augur is a probabilistic programming language and inference runtime. The
security-sensitive surfaces are:

- Model parsing / lowering (malformed input handling).
- The package manifest parser (`tpt-augur-pkg`), which reads `Augur.toml` files.
- Any future FFI / registry integration.

We treat panics-on-malformed-input and unsound posterior estimates from
untrusted model files as in-scope for this policy.
