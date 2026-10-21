//! Embedding context and L1 unit identity; see v1.md §§3.2, 3.5, and 7.1.

use crate::{models::EmbedInput, store::UnitKind};

use super::ast::CHUNKER_VERSION;

/// The status text recorded when a file produces more units than its configured cap.
pub(crate) const MAX_UNITS_PER_FILE_STATUS: &str =
    "max_units_per_file cap hit; remainder not chunked and file marked truncated";

/// Structural unit data needed to create the stable embedding context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContextUnit<'a> {
    pub(crate) scope_path: Option<&'a str>,
    pub(crate) header: &'a str,
    pub(crate) language: Option<&'a str>,
    pub(crate) kind: &'a str,
    pub(crate) text: &'a str,
    pub(crate) is_document: bool,
}

/// Repository-local file data used in an embedding breadcrumb.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContextFile<'a> {
    pub(crate) repo_name: &'a str,
    pub(crate) path: &'a str,
}

/// A directory and its optional, later-phase module role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectoryContext<'a> {
    pub(crate) path: &'a str,
    pub(crate) role: Option<&'a str>,
}

/// Optional model-input context lines; L1 indexing uses the unchanged full context.
#[derive(Clone, Copy)]
pub(crate) struct Options {
    pub(crate) breadcrumb: bool,
    pub(crate) breadcrumb_roles: bool,
    pub(crate) kind_line: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            breadcrumb: true,
            breadcrumb_roles: true,
            kind_line: true,
        }
    }
}
impl From<&crate::config::Embedder> for Options {
    fn from(config: &crate::config::Embedder) -> Self {
        Self {
            breadcrumb: config.breadcrumb,
            breadcrumb_roles: config.breadcrumb_roles,
            kind_line: config.kind_line,
        }
    }
}

/// The four deterministic components handed to an embedder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EmbeddingInput {
    pub(crate) breadcrumb: String,
    pub(crate) header: String,
    pub(crate) kind_line: String,
    pub(crate) text: String,
    rendered_header: String,
    role_free_header: String,
}

impl EmbeddingInput {
    /// Renders the exact embedding document for reranking, with a Unicode scalar
    /// cap over the entire breadcrumb/header/kind/text payload. Truncation never
    /// separates a UTF-8 code point and adds no model-specific prompt or suffix.
    pub(crate) fn rerank_document(&self, max_chars: usize) -> String {
        self.rendered_header
            .chars()
            .chain(std::iter::once('\n'))
            .chain(self.text.chars())
            .take(max_chars)
            .collect()
    }

    /// Borrows the canonical four-part context as the sole embedder input shape.
    ///
    /// This is intentionally the only construction point for `EmbedInput`, so
    /// document embeddings and later reranker documents cannot drift from the
    /// breadcrumb, header, and kind-line contract in `v1.md` §7.1.
    pub(crate) fn as_embed_input(&self, kind: UnitKind) -> EmbedInput<'_> {
        EmbedInput::from_context(&self.rendered_header, &self.text, kind)
    }

    /// Retains path/header/kind changes while removing only generated roles, so
    /// scheduling can distinguish optional role refresh from required repairs.
    pub(crate) fn as_role_free_embed_input(&self, kind: UnitKind) -> EmbedInput<'_> {
        EmbedInput::from_context(&self.role_free_header, &self.text, kind)
    }

    /// Replaces a verified breadcrumb while retaining the original role-free
    /// input identity. Refuses an inconsistent component/header representation.
    pub(crate) fn with_breadcrumb(mut self, breadcrumb: String) -> Option<Self> {
        let breadcrumb = model_safe(&breadcrumb);
        let suffix = self.rendered_header.strip_prefix(&self.breadcrumb)?;
        self.rendered_header = format!("{breadcrumb}{suffix}");
        self.breadcrumb = breadcrumb;
        Some(self)
    }
}

// llama.cpp accepts Rust UTF-8 strings through a C boundary and cannot tokenize
// an embedded NUL. Keep the lexical unit unchanged, but give every embedding and
// reranking provider the same deterministic Unicode replacement at the canonical
// model-input construction point.
fn model_safe(value: &str) -> String {
    value.replace('\0', "\u{fffd}")
}

/// Builds the same four model-input components for an immutable history unit,
/// whose breadcrumb is already collated rather than derived from a file path.
pub(crate) fn history_input(
    breadcrumb: &str,
    header: &str,
    kind: &str,
    text: &str,
    options: Options,
) -> EmbeddingInput {
    let breadcrumb = if options.breadcrumb {
        model_safe(breadcrumb)
    } else {
        String::new()
    };
    let kind_line = if options.kind_line {
        model_safe(kind)
    } else {
        String::new()
    };
    let mut lines = Vec::new();
    if options.breadcrumb {
        lines.push(breadcrumb.as_str());
    }
    let header = model_safe(header);
    lines.push(header.as_str());
    if options.kind_line {
        lines.push(kind_line.as_str());
    }
    let rendered_header = lines.join("\n");
    EmbeddingInput {
        breadcrumb,
        header,
        kind_line,
        text: model_safe(text),
        role_free_header: rendered_header.clone(),
        rendered_header,
    }
}

/// The result of enforcing the per-file L1 unit cap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CappedUnits<T> {
    pub(crate) units: Vec<T>,
    pub(crate) truncated: bool,
    pub(crate) status_message: Option<&'static str>,
}

/// Builds the breadcrumb, structural header, kind line, and original unit text.
///
/// Directory roles are optional because they are not populated until phase 3.
pub(crate) fn build(
    unit: ContextUnit<'_>,
    file: ContextFile<'_>,
    dirs: &[DirectoryContext<'_>],
) -> EmbeddingInput {
    build_with_options(unit, file, dirs, Options::default())
}

/// Builds only the enabled model context lines without changing the unit's text
/// or structural header. Default options preserve the original bytes, including
/// an empty structural-header line. Disabled fields are absent from both rendered
/// input and its components, so context identity follows the actual payload.
pub(crate) fn build_with_options(
    unit: ContextUnit<'_>,
    file: ContextFile<'_>,
    dirs: &[DirectoryContext<'_>],
    options: Options,
) -> EmbeddingInput {
    let breadcrumb = if options.breadcrumb {
        model_safe(&breadcrumb(
            file,
            unit.scope_path.filter(|_| unit.is_document),
            dirs,
            options.breadcrumb_roles,
        ))
    } else {
        String::new()
    };
    let kind_line = if options.kind_line {
        match unit.language {
            Some(language) if !language.is_empty() => {
                model_safe(&format!("{language} {}", unit.kind))
            }
            _ => model_safe(unit.kind),
        }
    } else {
        String::new()
    };
    let header = model_safe(unit.header);
    let role_free_breadcrumb = (options.breadcrumb
        && options.breadcrumb_roles
        && dirs.iter().any(|dir| dir.role.is_some()))
    .then(|| {
        self::breadcrumb(
            file,
            unit.scope_path.filter(|_| unit.is_document),
            dirs,
            false,
        )
    });
    let mut lines = Vec::with_capacity(3);
    if options.breadcrumb {
        lines.push(breadcrumb.as_str());
    }
    lines.push(header.as_str());
    if options.kind_line {
        lines.push(kind_line.as_str());
    }
    let rendered_header = lines.join("\n");
    let role_free_header = if let Some(base) = role_free_breadcrumb.as_deref() {
        lines[0] = base;
        lines.join("\n")
    } else {
        rendered_header.clone()
    };
    EmbeddingInput {
        breadcrumb,
        header,
        kind_line,
        text: model_safe(unit.text),
        rendered_header,
        role_free_header,
    }
}

/// Renders the shared file/heading breadcrumb for model inputs and navigation.
/// Roles are supplied by the caller when available; headings apply only to docs.
pub(crate) fn breadcrumb(
    file: ContextFile<'_>,
    document_scope: Option<&str>,
    dirs: &[DirectoryContext<'_>],
    roles: bool,
) -> String {
    path_breadcrumb(file, document_scope, dirs, roles, false)
}

/// Builds a module breadcrumb whose final path segment is itself a directory.
pub(crate) fn directory_breadcrumb(
    file: ContextFile<'_>,
    dirs: &[DirectoryContext<'_>],
    roles: bool,
) -> String {
    path_breadcrumb(file, None, dirs, roles, true)
}

fn path_breadcrumb(
    file: ContextFile<'_>,
    document_scope: Option<&str>,
    dirs: &[DirectoryContext<'_>],
    roles: bool,
    directory: bool,
) -> String {
    let root_role = roles
        .then(|| {
            dirs.iter()
                .find(|dir| dir.path.is_empty())
                .and_then(|dir| dir.role)
        })
        .flatten();
    let mut breadcrumb = vec![match root_role {
        Some(role) => format!("{} ({role})", file.repo_name),
        None => file.repo_name.to_owned(),
    }];
    let mut current = String::new();
    let segments = file
        .path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    for (index, segment) in segments.iter().enumerate() {
        if !current.is_empty() {
            current.push('/');
        }
        current.push_str(segment);
        let is_file = !directory && index + 1 == segments.len();
        if is_file {
            breadcrumb.push((*segment).to_owned());
            continue;
        }
        let role = if roles {
            dirs.iter()
                .find(|directory| directory.path == current)
                .and_then(|directory| directory.role)
        } else {
            None
        };
        breadcrumb.push(match role {
            Some(role) => format!("{segment} ({role})"),
            None => (*segment).to_owned(),
        });
    }
    if let Some(scope_path) = document_scope {
        breadcrumb.extend(
            scope_path
                .split('›')
                .map(str::trim)
                .filter(|heading| !heading.is_empty())
                .map(str::to_owned),
        );
    }
    breadcrumb.join(" › ")
}

/// Returns the path-independent L1 identity required for deduplicated unit writes.
pub(crate) fn unit_hash(kind: &str, scope_path: Option<&str>, text: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for value in [kind, scope_path.unwrap_or_default(), text, CHUNKER_VERSION] {
        hasher.update(value.as_bytes());
        hasher.update(b"|");
    }
    *hasher.finalize().as_bytes()
}

/// Retains only the permitted leading units and marks a file when content remains.
pub(crate) fn cap_units<T>(mut units: Vec<T>, max_units_per_file: u64) -> CappedUnits<T> {
    let limit = match usize::try_from(max_units_per_file) {
        Ok(limit) => limit,
        Err(_) => usize::MAX,
    };
    let truncated = units.len() > limit;
    units.truncate(limit);
    CappedUnits {
        units,
        truncated,
        status_message: truncated.then_some(MAX_UNITS_PER_FILE_STATUS),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ContextFile, ContextUnit, DirectoryContext, MAX_UNITS_PER_FILE_STATUS, build, cap_units,
        unit_hash,
    };
    use crate::store::UnitKind;

    #[test]
    fn unit_hash_is_stable_when_a_file_is_renamed() {
        let before = unit_hash(
            "symbol",
            Some("impl RetryPolicy"),
            "fn should_retry(err: &ChargeError) -> RetryDecision",
        );
        let after = unit_hash(
            "symbol",
            Some("impl RetryPolicy"),
            "fn should_retry(err: &ChargeError) -> RetryDecision",
        );
        assert_eq!(before, after);
    }

    #[test]
    fn builder_matches_the_specification_example() {
        let input = build(
            ContextUnit {
                scope_path: Some("impl RetryPolicy"),
                header: "impl RetryPolicy › fn should_retry(err: &ChargeError, attempt: u32) -> RetryDecision",
                language: Some("rust"),
                kind: "function",
                text: "match err { /* ... */ }",
                is_document: false,
            },
            ContextFile {
                repo_name: "aurora",
                path: "packages/billing/src/retry.rs",
            },
            &[DirectoryContext {
                path: "packages/billing",
                role: Some("payment retries, invoicing"),
            }],
        );
        assert_eq!(
            input.breadcrumb,
            "aurora › packages › billing (payment retries, invoicing) › src › retry.rs"
        );
        assert_eq!(
            input.header,
            "impl RetryPolicy › fn should_retry(err: &ChargeError, attempt: u32) -> RetryDecision"
        );
        assert_eq!(input.kind_line, "rust function");
        assert_eq!(input.text, "match err { /* ... */ }");
        let embed_input = input.as_embed_input(UnitKind::Symbol);
        assert_eq!(
            embed_input.header,
            "aurora › packages › billing (payment retries, invoicing) › src › retry.rs\nimpl RetryPolicy › fn should_retry(err: &ChargeError, attempt: u32) -> RetryDecision\nrust function"
        );
    }

    #[test]
    fn rerank_cap_counts_unicode_across_all_canonical_components() {
        let context = build(
            ContextUnit {
                scope_path: Some("章 › 二"),
                header: "見出し",
                language: None,
                kind: "doc_section",
                text: "résumé 🦀",
                is_document: true,
            },
            ContextFile {
                repo_name: "仓库",
                path: "docs/a.md",
            },
            &[DirectoryContext {
                path: "docs",
                role: Some("指南"),
            }],
        );
        let input = context.as_embed_input(UnitKind::DocSection);
        let full = format!("{}\n{}", input.header, input.text);
        for cap in 0..=full.chars().count() + 2 {
            assert_eq!(
                context.rerank_document(cap),
                full.chars().take(cap).collect::<String>()
            );
        }
        assert_eq!(context.rerank_document(usize::MAX), full);
    }

    #[test]
    fn model_context_replaces_embedded_nul_without_changing_lexical_identity() {
        let raw = "alpha\0omega";
        let context = build(
            ContextUnit {
                scope_path: Some("section\0name"),
                header: "heading\0suffix",
                language: Some("text\0lang"),
                kind: "section\0kind",
                text: raw,
                is_document: true,
            },
            ContextFile {
                repo_name: "fixture",
                path: "hostile.txt",
            },
            &[],
        );
        let input = context.as_embed_input(UnitKind::DocSection);
        assert!(!input.header.contains('\0'));
        assert!(!input.text.contains('\0'));
        assert_eq!(input.text, "alpha\u{fffd}omega");
        assert_eq!(
            context
                .rerank_document(usize::MAX)
                .matches('\u{fffd}')
                .count(),
            5
        );
        assert_ne!(
            unit_hash("doc_section", Some("section"), raw),
            unit_hash("doc_section", Some("section"), input.text),
        );
    }

    #[test]
    fn configured_context_lines_preserve_defaults_and_omit_disabled_components() {
        let unit = ContextUnit {
            scope_path: None,
            header: "sig",
            language: Some("rust"),
            kind: "symbol",
            text: "body",
            is_document: false,
        };
        let file = ContextFile {
            repo_name: "repo",
            path: "src/a.rs",
        };
        let dirs = [DirectoryContext {
            path: "src",
            role: Some("role"),
        }];
        for (breadcrumb, roles, kind, expected) in [
            (
                true,
                true,
                true,
                "repo › src (role) › a.rs\nsig\nrust symbol\nbody",
            ),
            (
                true,
                false,
                true,
                "repo › src › a.rs\nsig\nrust symbol\nbody",
            ),
            (true, true, false, "repo › src (role) › a.rs\nsig\nbody"),
            (true, false, false, "repo › src › a.rs\nsig\nbody"),
            (false, true, true, "sig\nrust symbol\nbody"),
            (false, false, true, "sig\nrust symbol\nbody"),
            (false, true, false, "sig\nbody"),
            (false, false, false, "sig\nbody"),
        ] {
            let context = super::build_with_options(
                unit,
                file,
                &dirs,
                super::Options {
                    breadcrumb,
                    breadcrumb_roles: roles,
                    kind_line: kind,
                },
            );
            let input = context.as_embed_input(UnitKind::Symbol);
            assert_eq!(format!("{}\n{}", input.header, input.text), expected);
            assert_eq!(context.rerank_document(usize::MAX), expected);
            assert_eq!(context.breadcrumb.is_empty(), !breadcrumb);
            assert_eq!(context.kind_line.is_empty(), !kind);
            if breadcrumb && roles && kind {
                assert_eq!(context, build(unit, file, &dirs));
            }
        }
        let empty_header = super::build_with_options(
            ContextUnit { header: "", ..unit },
            file,
            &dirs,
            super::Options::default(),
        );
        assert_eq!(
            empty_header.rerank_document(usize::MAX),
            "repo › src (role) › a.rs\n\nrust symbol\nbody"
        );
    }

    #[test]
    fn unit_cap_marks_the_file_and_exposes_a_status_message() {
        let capped = cap_units(vec![1, 2, 3], 2);
        assert_eq!(capped.units, vec![1, 2]);
        assert!(capped.truncated);
        assert_eq!(capped.status_message, Some(MAX_UNITS_PER_FILE_STATUS));
    }
}
