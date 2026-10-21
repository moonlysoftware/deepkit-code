//! Deterministic document mentions and test-path matching; v1.md §6.5.
use globset::{Glob, GlobSet, GlobSetBuilder};
use std::collections::BTreeSet;

/// One literal code-span identifier or path token, with a source-relative line.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Mention {
    pub(crate) value: String,
    pub(crate) line: usize,
    pub(crate) path: bool,
    pub(crate) code: bool,
}

fn token_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '$' | ':' | '.' | '/' | '-' | '\\')
}
fn tokens(text: &str, line: usize, code: bool, result: &mut BTreeSet<Mention>) {
    for (offset, segment) in text.split_inclusive('\n').enumerate() {
        for value in segment.split(|c| !token_char(c)) {
            let value = value.trim_matches([':', '\\']).trim_end_matches('.');
            if value.is_empty() {
                continue;
            }
            let path = value.contains('/')
                || value
                    .rsplit_once('.')
                    .is_some_and(|(a, b)| !a.is_empty() && !b.is_empty());
            if path
                || (code
                    && value
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic() || matches!(c, '_' | '$')))
            {
                result.insert(Mention {
                    value: value.to_owned(),
                    line: line + offset,
                    path,
                    code,
                });
            }
        }
    }
}

/// Extracts only explicit code-span identifiers; ordinary prose contributes paths
/// but never speculative identifier links. Matched backtick runs bound each span.
pub(crate) fn mentions(text: &str) -> Vec<Mention> {
    let mut result = BTreeSet::new();
    tokens(text, 0, false, &mut result);
    let bytes = text.as_bytes();
    let mut cursor = 0;
    let mut line = 0;
    while cursor < bytes.len() {
        if bytes[cursor] == b'\n' {
            line += 1;
            cursor += 1;
            continue;
        }
        if bytes[cursor] != b'`'
            || bytes[..cursor]
                .iter()
                .rev()
                .take_while(|&&b| b == b'\\')
                .count()
                % 2
                == 1
        {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < bytes.len() && bytes[cursor] == b'`' {
            cursor += 1;
        }
        let width = cursor - start;
        let content = cursor;
        let mut end = cursor;
        while end < bytes.len() {
            if bytes[end] != b'`' {
                end += 1;
                continue;
            }
            let run = end;
            while end < bytes.len() && bytes[end] == b'`' {
                end += 1;
            }
            if end - run == width {
                let span = &text[content..run];
                if width >= 3 && span.contains('\n') {
                    if let Some((_, body)) = span.split_once('\n') {
                        tokens(body, line + 1, true, &mut result);
                    }
                } else {
                    tokens(span, line, true, &mut result);
                }
                line += bytes[content..end].iter().filter(|&&b| b == b'\n').count();
                cursor = end;
                break;
            }
        }
        if end == bytes.len() && cursor == content {
            break;
        }
    }
    result.into_iter().collect()
}

/// Compiles configured test globs once for an indexing or navigation operation.
pub(crate) fn test_paths(patterns: &[String]) -> Result<GlobSet, globset::Error> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern)?);
    }
    builder.build()
}

/// Tries repository-relative and document-relative paths without filesystem I/O.
/// Parent components may normalize within the repository but never escape it.
pub(crate) fn paths(document: &str, token: &str) -> BTreeSet<String> {
    fn normalize(value: &str) -> Option<String> {
        if value.starts_with('/') || value.contains("://") {
            return None;
        }
        let mut pieces = Vec::new();
        for part in value.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    pieces.pop()?;
                }
                _ => pieces.push(part),
            }
        }
        (!pieces.is_empty()).then(|| pieces.join("/"))
    }
    let mut result = BTreeSet::new();
    if let Some(path) = normalize(token) {
        result.insert(path);
    }
    if let Some((parent, _)) = document.rsplit_once('/')
        && let Some(path) = normalize(&format!("{parent}/{token}"))
    {
        result.insert(path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn code_spans_paths_qualified_names_and_prose_are_distinct() {
        let values = mentions(
            "ordinary_name isn't a link. `should_retry()` and ``Retry::run``.\nSee src/retry.rs and `config.value`. ",
        );
        assert!(!values.iter().any(|m| m.value == "ordinary_name"));
        assert!(
            values
                .iter()
                .any(|m| m.value == "should_retry" && m.line == 0)
        );
        assert!(values.iter().any(|m| m.value == "Retry::run"));
        assert!(
            values
                .iter()
                .any(|m| m.value == "src/retry.rs" && m.line == 1 && m.path)
        );
        assert!(values.iter().any(|m| m.value == "config.value"));
        assert!(
            !mentions("\\`escaped_name\\` and `unclosed")
                .iter()
                .any(|m| !m.path)
        );
        assert_eq!(
            paths("docs/guide.md", "../src/retry.rs"),
            ["src/retry.rs".into()].into()
        );
        assert!(paths("README.md", "../../outside.rs").is_empty());
    }
}

#[cfg(test)]
mod fence_tests {
    use super::*;
    #[test]
    fn fences_skip_language_and_keep_relative_lines() {
        let values = mentions("# Examples\n```rust\nshould_retry();\n```\n");
        assert!(!values.iter().any(|m| m.value == "rust"));
        assert!(
            values
                .iter()
                .any(|m| m.value == "should_retry" && m.code && m.line == 2)
        );
        let values = mentions("``a`b`` and `café`.");
        assert!(values.iter().any(|m| m.value == "café" && m.code));
    }
}
