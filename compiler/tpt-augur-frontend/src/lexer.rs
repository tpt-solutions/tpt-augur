//! Lexer for the Augur language.

use crate::ast::Span;
use crate::diagnostics::Diagnostic;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Let,
    Observe,
    If,
    Else,
    Tilde,
    Equals,
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Num(f64),
    Ident(String),
    Newline,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

pub fn lex(src: &str) -> (Vec<Token>, Vec<Diagnostic>) {
    let mut tokens = Vec::new();
    let mut diags = Vec::new();
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0;
    let n = bytes.len();

    let push = |tok: Tok, start: usize, end: usize, tokens: &mut Vec<Token>| {
        tokens.push(Token {
            tok,
            span: Span::new(start, end),
        });
    };

    while i < n {
        let c = bytes[i];
        if c.is_whitespace() {
            if c == '\n' {
                push(Tok::Newline, i, i + 1, &mut tokens);
            }
            i += 1;
            continue;
        }
        if c == '#' {
            while i < n && bytes[i] != '\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        match c {
            '~' => {
                push(Tok::Tilde, start, i + 1, &mut tokens);
                i += 1;
            }
            '=' => {
                if i + 1 < n && bytes[i + 1] == '=' {
                    push(Tok::Eq, start, i + 2, &mut tokens);
                    i += 2;
                } else {
                    push(Tok::Equals, start, i + 1, &mut tokens);
                    i += 1;
                }
            }
            '!' => {
                if i + 1 < n && bytes[i + 1] == '=' {
                    push(Tok::Ne, start, i + 2, &mut tokens);
                    i += 2;
                } else {
                    diags.push(Diagnostic::lex_error(
                        "unexpected character `!` (did you mean `!=`?)",
                        Span::new(start, i + 1),
                    ));
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < n && bytes[i + 1] == '=' {
                    push(Tok::Ge, start, i + 2, &mut tokens);
                    i += 2;
                } else {
                    push(Tok::Gt, start, i + 1, &mut tokens);
                    i += 1;
                }
            }
            '<' => {
                if i + 1 < n && bytes[i + 1] == '=' {
                    push(Tok::Le, start, i + 2, &mut tokens);
                    i += 2;
                } else {
                    push(Tok::Lt, start, i + 1, &mut tokens);
                    i += 1;
                }
            }
            '+' => {
                push(Tok::Plus, start, i + 1, &mut tokens);
                i += 1;
            }
            '-' => {
                push(Tok::Minus, start, i + 1, &mut tokens);
                i += 1;
            }
            '*' => {
                push(Tok::Star, start, i + 1, &mut tokens);
                i += 1;
            }
            '/' => {
                push(Tok::Slash, start, i + 1, &mut tokens);
                i += 1;
            }
            '(' => {
                push(Tok::LParen, start, i + 1, &mut tokens);
                i += 1;
            }
            ')' => {
                push(Tok::RParen, start, i + 1, &mut tokens);
                i += 1;
            }
            '{' => {
                push(Tok::LBrace, start, i + 1, &mut tokens);
                i += 1;
            }
            '}' => {
                push(Tok::RBrace, start, i + 1, &mut tokens);
                i += 1;
            }
            ',' => {
                push(Tok::Comma, start, i + 1, &mut tokens);
                i += 1;
            }
            _ if c.is_ascii_digit() || (c == '.' && i + 1 < n && bytes[i + 1].is_ascii_digit()) => {
                let mut j = i + 1;
                let mut dots = if c == '.' { 1 } else { 0 };
                while j < n {
                    let d = bytes[j];
                    if d.is_ascii_digit() {
                        j += 1;
                    } else if d == '.' && dots == 0 {
                        dots += 1;
                        j += 1;
                    } else {
                        break;
                    }
                }
                let s: String = bytes[i..j].iter().collect();
                match s.parse::<f64>() {
                    Ok(v) => push(Tok::Num(v), start, j, &mut tokens),
                    Err(_) => diags.push(Diagnostic::lex_error(
                        format!("invalid number literal `{s}`"),
                        Span::new(start, j),
                    )),
                }
                i = j;
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let mut j = i + 1;
                while j < n && (bytes[j].is_ascii_alphanumeric() || bytes[j] == '_') {
                    j += 1;
                }
                let s: String = bytes[i..j].iter().collect();
                let tok = match s.as_str() {
                    "let" => Tok::Let,
                    "observe" => Tok::Observe,
                    "if" => Tok::If,
                    "else" => Tok::Else,
                    _ => Tok::Ident(s),
                };
                push(tok, start, j, &mut tokens);
                i = j;
            }
            other => {
                diags.push(Diagnostic::lex_error(
                    format!("unexpected character `{other}`"),
                    Span::new(start, i + 1),
                ));
                i += 1;
            }
        }
    }
    push(Tok::Eof, n, n, &mut tokens);
    (tokens, diags)
}
