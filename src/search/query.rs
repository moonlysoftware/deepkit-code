//! Typed query routing before retrieval/provider work; v1 §9 step 1.
use super::{Error, fusion::Mode};
use crate::treat::fts::build_or_match;
use std::collections::BTreeSet;

/// One query text per leg, so dense query embedding runs once per active tier.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Parsed {
    pub(crate) lexical: Option<String>,
    pub(crate) dense: Option<String>,
    pub(crate) ident_terms: Vec<String>,
    explicit_syntax: bool,
}

impl Parsed {
    /// Builds the relaxed query without letting common grammar words dominate
    /// plain questions. Explicit syntax and identifier-only queries stay intact.
    pub(crate) fn relaxed_match(&self) -> String {
        let query = self.lexical.as_deref().unwrap_or_default();
        if self.explicit_syntax {
            return build_or_match(query);
        }
        let words = query
            .split(|ch: char| !(ch.is_alphanumeric() || matches!(ch, '_' | '$' | '.')))
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>();
        let natural = words.len() >= 2
            && words.iter().any(|word| {
                !self
                    .ident_terms
                    .iter()
                    .any(|identifier| identifier.as_str() == *word)
            });
        if !natural {
            return build_or_match(query);
        }
        let content = words
            .into_iter()
            .filter(|word| {
                self.ident_terms
                    .iter()
                    .any(|identifier| identifier.as_str() == *word)
                    || !grammatical(word)
            })
            .collect::<Vec<_>>()
            .join(" ");
        let relaxed = build_or_match(&content);
        if relaxed.is_empty() {
            build_or_match(query)
        } else {
            relaxed
        }
    }
}

fn grammatical(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "a" | "an"
            | "the"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "do"
            | "does"
            | "did"
            | "can"
            | "could"
            | "would"
            | "should"
            | "where"
            | "which"
            | "what"
            | "how"
    )
}
fn marker(text: &str) -> Option<(Mode, usize)> {
    if text.starts_with("lex:") {
        Some((Mode::Lexical, 4))
    } else if text.starts_with("vec:") {
        Some((Mode::Dense, 4))
    } else {
        None
    }
}
fn append(mode: Mode, text: &str, lexical: &mut Vec<String>, dense: &mut Vec<String>) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if mode != Mode::Dense {
        lexical.push(text.to_owned());
    }
    if mode != Mode::Lexical {
        dense.push(text.to_owned());
    }
}
fn quoted(text: &str) -> Result<(String, usize), Error> {
    let mut decoded = String::new();
    let mut escaped = false;
    for (index, ch) in text.char_indices().skip(1) {
        if escaped {
            if !matches!(ch, '"' | '\\') {
                decoded.push('\\');
            }
            decoded.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Ok((decoded, index + 1));
        } else {
            decoded.push(ch);
        }
    }
    Err(Error::InvalidQuery {
        reason: "unterminated typed phrase",
    })
}

/// Routes natural text by mode and explicit clauses by their prefix. An unquoted
/// typed clause ends at the next prefix outside quotes; a quoted clause ends at
/// its closing quote, returning subsequent untyped text to the requested mode.
/// Prefixes inside quotes or words stay literal. Empty typed clauses are errors.
pub(crate) fn parse(text: &str, mode: Mode) -> Result<Parsed, Error> {
    let mut markers = Vec::new();
    let mut in_quote = false;
    let mut escaped = false;
    let mut boundary = true;
    for (index, ch) in text.char_indices() {
        if !in_quote
            && boundary
            && let Some((mode, width)) = marker(&text[index..])
        {
            markers.push((index, mode, width));
        }
        if escaped {
            escaped = false;
        } else if in_quote && ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            in_quote = !in_quote;
        }
        boundary = ch.is_whitespace();
    }
    let mut lexical = Vec::new();
    let mut dense = Vec::new();
    let first = markers.first().map_or(text.len(), |m| m.0);
    append(mode, &text[..first], &mut lexical, &mut dense);
    for (position, &(start, route, width)) in markers.iter().enumerate() {
        let end = markers.get(position + 1).map_or(text.len(), |m| m.0);
        let body = text[start + width..end].trim();
        if body.is_empty() {
            return Err(Error::InvalidQuery {
                reason: "empty typed clause",
            });
        }
        if body.starts_with('"') {
            let (phrase, used) = quoted(body)?;
            if phrase.trim().is_empty() {
                return Err(Error::InvalidQuery {
                    reason: "empty typed phrase",
                });
            }
            if route == Mode::Lexical {
                // FTS tokenization ignores quotation punctuation. Preserve the
                // entire typed phrase rather than introducing nested delimiters.
                lexical.push(format!("\"{}\"", phrase.replace('"', " ")));
            } else {
                dense.push(phrase);
            }
            append(mode, &body[used..], &mut lexical, &mut dense);
        } else {
            append(route, body, &mut lexical, &mut dense);
        }
    }
    let lexical = (!lexical.is_empty()).then(|| lexical.join(" "));
    let dense = (!dense.is_empty()).then(|| dense.join(" "));
    // Exact-name boosts must not bypass an explicitly quoted phrase constraint.
    let mut unquoted = String::new();
    let mut in_quote = false;
    for ch in lexical.as_deref().unwrap_or_default().chars() {
        if ch == '"' {
            in_quote = !in_quote;
            unquoted.push(' ');
        } else if !in_quote {
            unquoted.push(ch);
        }
    }
    let mut seen = BTreeSet::new();
    let ident_terms = super::identifier_terms(&unquoted)
        .into_iter()
        .filter(|term| seen.insert((*term).to_owned()))
        .map(str::to_owned)
        .collect();
    Ok(Parsed {
        lexical,
        dense,
        ident_terms,
        explicit_syntax: !markers.is_empty() || text.contains('"'),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_questions_keep_meaning_and_original_dense_text() -> Result<(), Error> {
        let text = "How does a retry move from pending to failed without blocking, not waiting?";
        let parsed = parse(text, Mode::Hybrid)?;
        assert_eq!(
            parsed.relaxed_match(),
            build_or_match("retry move from pending to failed without blocking not waiting")
        );
        assert_eq!(parsed.lexical.as_deref(), Some(text));
        assert_eq!(parsed.dense.as_deref(), Some(text));

        let parsed = parse("Where is DoHTTP_readByID module.from", Mode::Hybrid)?;
        assert_eq!(
            parsed.relaxed_match(),
            build_or_match("DoHTTP_readByID module.from")
        );
        assert_eq!(parsed.ident_terms, vec!["DoHTTP_readByID", "module.from"]);
        Ok(())
    }

    #[test]
    fn relaxed_questions_preserve_identifiers_that_are_also_grammar_words() -> Result<(), Error> {
        let parsed = parse("Where can CAN run", Mode::Hybrid)?;
        assert_eq!(parsed.ident_terms, vec!["CAN"]);
        assert_eq!(parsed.relaxed_match(), build_or_match("CAN run"));
        Ok(())
    }

    #[test]
    fn relaxed_filter_preserves_explicit_syntax_names_and_empty_remainders() -> Result<(), Error> {
        for text in [
            "lex:Where is retry",
            "Where is retry vec:semantic request",
            "Where is \"retry policy\"",
            "lex:\"where is\" retry",
            "\"unterminated natural phrase",
            "is",
            "DoHTTP_readByID",
            "DoHTTP_readByID module.from",
            "Where is the",
            ". is",
            "!?",
        ] {
            let parsed = parse(text, Mode::Hybrid)?;
            assert_eq!(
                parsed.relaxed_match(),
                build_or_match(parsed.lexical.as_deref().unwrap_or_default()),
                "{text}"
            );
        }
        assert_eq!(
            parse("vec:where is retry", Mode::Hybrid)?.relaxed_match(),
            ""
        );
        Ok(())
    }

    #[test]
    fn typed_clauses_route_without_leaking_to_the_other_leg() -> Result<(), Error> {
        let parsed = parse(
            "natural lex:\"exact phrase\" more vec:vague payment idea",
            Mode::Hybrid,
        )?;
        assert_eq!(
            parsed.lexical.as_deref(),
            Some("natural \"exact phrase\" more")
        );
        assert_eq!(
            parsed.dense.as_deref(),
            Some("natural more vague payment idea")
        );
        assert_eq!(parse("lex:RetryPolicy", Mode::Hybrid)?.dense, None);
        assert_eq!(parse("vec:vague payment idea", Mode::Hybrid)?.lexical, None);
        assert_eq!(parse("ordinary text", Mode::Dense)?.lexical, None);
        assert_eq!(parse("ordinary text", Mode::Lexical)?.dense, None);
        assert_eq!(
            parse("lex:forced", Mode::Dense)?.lexical.as_deref(),
            Some("forced")
        );
        Ok(())
    }
    #[test]
    fn quotes_unicode_and_identifier_scope_are_preserved() -> Result<(), Error> {
        let parsed = parse(
            "lex:RetryPolicy vec:OtherName idea lex:snake_case",
            Mode::Hybrid,
        )?;
        assert_eq!(parsed.ident_terms, vec!["RetryPolicy", "snake_case"]);
        assert!(
            parse("lex:\"RetryPolicy absent\"", Mode::Hybrid)?
                .ident_terms
                .is_empty()
        );
        assert_eq!(
            parse("vec:\"café lex:literal\"", Mode::Hybrid)?
                .dense
                .as_deref(),
            Some("café lex:literal")
        );
        assert_eq!(
            parse("complex:word \"vec:literal\"", Mode::Lexical)?
                .lexical
                .as_deref(),
            Some("complex:word \"vec:literal\"")
        );
        assert_eq!(
            parse("vec:\"say \\\"hello\\\"\"", Mode::Hybrid)?
                .dense
                .as_deref(),
            Some("say \"hello\"")
        );
        assert_eq!(
            parse(r#"vec:"C:\src\thing""#, Mode::Hybrid)?
                .dense
                .as_deref(),
            Some(r"C:\src\thing")
        );
        Ok(())
    }
    #[test]
    fn malformed_typed_queries_fail_before_retrieval() {
        for text in [
            "lex:",
            "vec:   ",
            "lex:\"\"",
            "vec:\"unterminated",
            "lex: vec:valid",
        ] {
            assert!(parse(text, Mode::Hybrid).is_err(), "{text}");
        }
        assert_eq!(
            parse("  ", Mode::Hybrid).ok(),
            Some(Parsed {
                lexical: None,
                dense: None,
                ident_terms: vec![],
                explicit_syntax: false,
            })
        );
    }
}
