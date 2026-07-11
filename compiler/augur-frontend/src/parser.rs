//! Recursive-descent, error-tolerant parser for the Augur language.
//!
//! The parser never panics on malformed input: it records [`Diagnostic`]s and
//! resynchronises to the next statement so a partially broken program still
//! yields a usable (partial) AST — this underpins editor tooling such as the
//! LSP and `augur fmt`.

use crate::ast::*;
use crate::diagnostics::{Diagnostic, Severity};
use crate::lexer::{Tok, Token};

pub struct ParseResult {
    pub program: Program,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn parse(src: &str) -> ParseResult {
    let (tokens, mut diagnostics) = crate::lexer::lex(src);
    let mut p = Parser {
        tokens: &tokens,
        pos: 0,
        diagnostics: Vec::new(),
    };
    let program = p.parse_program();
    diagnostics.append(&mut p.diagnostics);
    ParseResult {
        program,
        diagnostics,
    }
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn span(&self) -> Span {
        if self.pos < self.tokens.len() {
            self.tokens[self.pos].span
        } else {
            Span::new(0, 0)
        }
    }

    fn advance(&mut self) -> &Tok {
        let t = &self.tokens[self.pos].tok;
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn at(&self, tok: &Tok) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(tok)
    }

    fn eat(&mut self, tok: &Tok) -> bool {
        if self.at(tok) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, tok: &Tok) -> bool {
        if self.eat(tok) {
            true
        } else {
            self.diagnostics.push(Diagnostic::parse_error(
                format!("expected `{tok:?}`"),
                self.span(),
            ));
            false
        }
    }

    fn skip_newlines(&mut self) {
        while self.at(&Tok::Newline) {
            self.advance();
        }
    }

    /// Skip tokens until we reach a plausible statement boundary or block end.
    fn resync(&mut self) {
        let mut depth = 0;
        while self.pos < self.tokens.len() {
            match self.peek() {
                Tok::LBrace => {
                    depth += 1;
                    self.advance();
                }
                Tok::RBrace => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    self.advance();
                }
                Tok::Let | Tok::Observe | Tok::If if depth == 0 => return,
                Tok::Eof => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    fn parse_program(&mut self) -> Program {
        let mut statements = Vec::new();
        self.skip_newlines();
        while !self.at(&Tok::Eof) {
            match self.parse_stmt() {
                Some(s) => statements.push(s),
                None => self.resync(),
            }
            self.skip_newlines();
        }
        Program { statements }
    }

    fn parse_stmt(&mut self) -> Option<Stmt> {
        match self.peek() {
            Tok::Let => self.parse_let(),
            Tok::Observe => self.parse_observe(),
            Tok::If => self.parse_if(),
            other => {
                self.diagnostics.push(Diagnostic::parse_error(
                    format!("unexpected token `{other:?}` at statement position"),
                    self.span(),
                ));
                None
            }
        }
    }

    fn parse_let(&mut self) -> Option<Stmt> {
        let start = self.tokens[self.pos].span.start;
        self.advance(); // `let`
        let name = match self.peek() {
            Tok::Ident(n) => n.clone(),
            _ => {
                self.diagnostics.push(Diagnostic::parse_error(
                    "expected variable name after `let`",
                    self.span(),
                ));
                return None;
            }
        };
        self.advance();
        let stmt = if self.eat(&Tok::Tilde) {
            let dist = self.parse_expr();
            Stmt::Prior {
                name,
                dist,
                span: Span::new(start, self.tokens[self.pos].span.end),
            }
        } else if self.expect(&Tok::Equals) {
            let value = self.parse_expr();
            Stmt::Let {
                name,
                value,
                span: Span::new(start, self.tokens[self.pos].span.end),
            }
        } else {
            return None;
        };
        Some(stmt)
    }

    fn parse_observe(&mut self) -> Option<Stmt> {
        let start = self.tokens[self.pos].span.start;
        self.advance(); // `observe`
        let dist = self.parse_expr();
        if !self.expect(&Tok::Equals) {
            return None;
        }
        let value = self.parse_expr();
        Some(Stmt::Observe {
            dist,
            value,
            span: Span::new(start, self.tokens[self.pos].span.end),
        })
    }

    fn parse_if(&mut self) -> Option<Stmt> {
        let start = self.tokens[self.pos].span.start;
        self.advance(); // `if`
        let cond = self.parse_expr();
        if !self.expect(&Tok::LBrace) {
            return None;
        }
        let then_body = self.parse_block();
        let else_body = if self.at(&Tok::Else) {
            self.advance();
            if !self.expect(&Tok::LBrace) {
                Vec::new()
            } else {
                self.parse_block()
            }
        } else {
            Vec::new()
        };
        Some(Stmt::If {
            cond,
            then_body,
            else_body,
            span: Span::new(start, self.tokens[self.pos].span.end),
        })
    }

    fn parse_block(&mut self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        self.skip_newlines();
        while !self.at(&Tok::RBrace) && !self.at(&Tok::Eof) {
            match self.parse_stmt() {
                Some(s) => stmts.push(s),
                None => self.resync(),
            }
            self.skip_newlines();
        }
        self.eat(&Tok::RBrace);
        stmts
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Expr {
        let mut lhs = self.parse_add();
        loop {
            let op = match self.peek() {
                Tok::Gt => CmpOp::Gt,
                Tok::Lt => CmpOp::Lt,
                Tok::Ge => CmpOp::Ge,
                Tok::Le => CmpOp::Le,
                Tok::Eq => CmpOp::Eq,
                Tok::Ne => CmpOp::Ne,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_add();
            lhs = Expr::Cmp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        lhs
    }

    fn parse_add(&mut self) -> Expr {
        let mut lhs = self.parse_mul();
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_mul();
            lhs = Expr::Bin {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        lhs
    }

    fn parse_mul(&mut self) -> Expr {
        let mut lhs = self.parse_unary();
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary();
            lhs = Expr::Bin {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        lhs
    }

    fn parse_unary(&mut self) -> Expr {
        if self.at(&Tok::Minus) {
            self.advance();
            let e = self.parse_unary();
            return Expr::Neg(Box::new(e));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Expr {
        match self.peek() {
            Tok::Num(v) => {
                let v = *v;
                self.advance();
                Expr::Num(v)
            }
            Tok::Ident(name) => {
                let name = name.clone();
                self.advance();
                if self.at(&Tok::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    self.skip_newlines();
                    if !self.at(&Tok::RParen) {
                        args.push(self.parse_expr());
                        while self.eat(&Tok::Comma) {
                            self.skip_newlines();
                            args.push(self.parse_expr());
                        }
                    }
                    self.expect(&Tok::RParen);
                    Expr::Call { name, args }
                } else {
                    Expr::Var(name)
                }
            }
            Tok::LParen => {
                self.advance();
                let e = self.parse_expr();
                self.expect(&Tok::RParen);
                Expr::Paren(Box::new(e))
            }
            other => {
                self.diagnostics.push(Diagnostic::parse_error(
                    format!("unexpected token `{other:?}` in expression"),
                    self.span(),
                ));
                // Produce a dummy node so parsing can limp along.
                self.advance();
                Expr::Num(0.0)
            }
        }
    }
}

/// Convenience: count the distinct severity classes for reporting.
impl ParseResult {
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }

    pub fn warnings(&self) -> Vec<&Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .collect()
    }
}
