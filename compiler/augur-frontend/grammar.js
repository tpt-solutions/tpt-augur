// Tree-sitter grammar for the Augur probabilistic programming language.
//
// Augur uses a hand-rolled, error-tolerant lexer/parser for execution
// (see compiler/augur-frontend/src/{lexer,parser}.rs); this grammar is the
// canonical, tooling-oriented description of the same surface syntax. It powers
// editor highlighting / structural tooling and is kept in sync with the parser.
//
// Build with: `tree-sitter generate && tree-sitter build`
// (https://tree-sitter.github.io/tree-sitter/creating-parsers)

module.exports = grammar({
  name: "augur",

  rules: {
    program: ($) =>
      seq(
        optional($._newline),
        repeat(seq($.statement, optional($._newline))),
        optional($._newline)
      ),

    statement: ($) =>
      choice($.let_prior, $.let_binding, $.observe, $.if_stmt),

    // `let name ~ Dist(...)` — a random variable with a prior.
    let_prior: ($) =>
      seq(
        "let",
        $.identifier,
        "~",
        $.distribution,
        optional($._newline)
      ),

    // `let name = expr` — a deterministic (uncertainty-carrying) binding.
    let_binding: ($) =>
      seq(
        "let",
        $.identifier,
        "=",
        $.expression,
        optional($._newline)
      ),

    // `observe Dist(...) = value` — likelihood / conditioning.
    observe: ($) =>
      seq(
        "observe",
        $.distribution,
        "=",
        $.expression,
        optional($._newline)
      ),

    // `if cond { ... } else { ... }` — deterministic control flow.
    if_stmt: ($) =>
      seq(
        "if",
        $.expression,
        $.block,
        optional(seq("else", $.block))
      ),

    block: ($) =>
      seq(
        "{",
        optional($._newline),
        repeat(seq($.statement, optional($._newline))),
        "}"
      ),

    distribution: ($) =>
      seq($.identifier, "(", commaSep($.expression), ")"),

    expression: ($) => $.comparison,

    comparison: ($) =>
      prec.left(
        1,
        seq(
          $.addition,
          repeat(
            seq(
              choice("==", "!=", ">", ">=", "<", "<="),
              $.addition
            )
          )
        )
      ),

    addition: ($) =>
      prec.left(
        seq(
          $.multiplication,
          repeat(seq(choice("+", "-"), $.multiplication))
        )
      ),

    multiplication: ($) =>
      prec.left(
        seq(
          $.unary,
          repeat(seq(choice("*", "/"), $.unary))
        )
      ),

    unary: ($) => choice(seq("-", $.unary), $.primary),

    primary: ($) =>
      choice(
        $.number,
        $.identifier,
        $.distribution,
        seq("(", $.expression, ")")
      ),

    identifier: ($) => /[a-zA-Z_][a-zA-Z0-9_]*/,
    number: ($) => /[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/,

    comment: ($) => token(seq("#", /[^\n]*/)),

    _newline: ($) =>
      token.immediate(choice("\n", /\s*\n\s*/, $.comment)),
  },

  extras: ($) => [/\s/, $.comment],
});

function commaSep(rule) {
  return seq(rule, repeat(seq(",", rule)));
}
