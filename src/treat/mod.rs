//! Content routing, chunking, and enrichment; see `v1.md` §§5–6.
pub(crate) mod composite;
pub(crate) mod linker;
pub(crate) mod module;
pub(crate) mod notebook;
pub(crate) mod office;
pub(crate) mod pdf;
pub(crate) mod prompts;
pub(crate) mod symbol_history;

pub(crate) use ast::CHUNKER_VERSION;
pub(crate) use languages::VERSION as GRAMMAR_VERSION;
mod languages;
mod syntax;
mod token_count;

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use globset::{Glob, GlobSet, GlobSetBuilder};
use tree_sitter::{Node, Parser};

use crate::{
    config::Treatment,
    store::{FileKind, UnitKind},
};

// T0.13 connects extracted symbols and edges to the common unit write path.
// The graph is independently covered by permanent extraction/ranking tests now.
#[allow(
    dead_code,
    reason = "T0.13 wires the completed graph path to indexed unit writes"
)]
pub(crate) mod graph;

/// Shared embedding context and capped-unit bookkeeping; see v1.md §§3.2, 3.5, and 7.1.
#[allow(
    dead_code,
    reason = "T0.14 invokes the completed context builder from the index command"
)]
pub(crate) mod context;
pub(crate) mod pr;

/// A chunker selected by a treatment before a file enters the pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Chunker {
    Ast,
    Lines,
    Markdown,
    Pdf,
    Office,
    Notebook,
    Config,
    None,
}

/// Embedded source keeps its own lexical language inside a document container.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TextOrigin {
    File,
    EmbeddedCode { language: Option<String> },
}

/// A normalized L1 unit ready for lexical indexing and persistence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedUnit {
    pub(crate) origin: TextOrigin,
    pub(crate) kind: UnitKind,
    pub(crate) scope_path: String,
    pub(crate) header: String,
    pub(crate) text: String,
    pub(crate) line_start: usize,
    pub(crate) line_end: usize,
    pub(crate) has_code_fence: bool,
    pub(crate) idents: Vec<String>,
    pub(crate) truncated: bool,
}

/// Shared production allowance, including one overflow occurrence for cap reporting.
struct UnitBudget(std::cell::Cell<usize>);
impl UnitBudget {
    fn new(limit: usize) -> Self {
        Self(std::cell::Cell::new(limit))
    }
    fn exhausted(&self) -> bool {
        self.0.get() == 0
    }
    fn consume(&self) {
        self.0.set(self.0.get().saturating_sub(1));
    }
}

/// Counts L1 tokens with the shared deterministic tokenizer, including history units.
pub(crate) fn count_tokens(text: &str) -> usize {
    ast::token_count(text)
}

/// Applies one resolved chunker and returns its normalized L1 units.
///
/// This keeps the index command from duplicating the parser-specific metadata
/// that drives FTS columns, context, and the unit cap.
pub(crate) fn prepare_units(
    path: &str,
    language: Option<&str>,
    chunker: Chunker,
    source: &[u8],
    options: &crate::config::Chunk,
    max_units: u64,
) -> Result<Vec<PreparedUnit>, Error> {
    let budget =
        UnitBudget::new(usize::try_from(max_units.saturating_add(1)).unwrap_or(usize::MAX));
    match chunker {
        Chunker::Ast => ast::chunk_code_configured(
            options,
            &budget,
            language.and_then(|language| ast::AstLanguage::for_path(language, path)),
            source,
        )
        .map(|units| {
            units
                .into_iter()
                .map(|unit| PreparedUnit {
                    origin: TextOrigin::File,
                    kind: match unit.kind {
                        ast::AstUnitKind::Symbol => UnitKind::Symbol,
                        ast::AstUnitKind::Chunk => UnitKind::Chunk,
                    },
                    scope_path: unit.scope_path,
                    header: unit.header,
                    text: unit.text,
                    line_start: unit.line_start,
                    line_end: unit.line_end,
                    has_code_fence: false,
                    idents: Vec::new(),
                    truncated: false,
                })
                .collect()
        })
        .map_err(Error::Ast),
        Chunker::Lines => ast::chunk_code_configured(options, &budget, None, source)
            .map(|units| {
                units
                    .into_iter()
                    .map(|unit| PreparedUnit {
                        origin: TextOrigin::File,
                        kind: UnitKind::Chunk,
                        scope_path: unit.scope_path,
                        header: unit.header,
                        text: unit.text,
                        line_start: unit.line_start,
                        line_end: unit.line_end,
                        has_code_fence: false,
                        idents: Vec::new(),
                        truncated: false,
                    })
                    .collect()
            })
            .map_err(Error::Ast),
        Chunker::Markdown => {
            let document_units = if path.ends_with(".rst") || path.ends_with(".txt") {
                prose::chunk_plain_configured(&budget, source, options.markdown_range)
            } else {
                prose::chunk_markdown_configured(&budget, source, options.markdown_range)
            };
            Ok(document_units
                .into_iter()
                .map(|unit| PreparedUnit {
                    origin: TextOrigin::File,
                    kind: match unit.kind {
                        prose::TextUnitKind::Markdown | prose::TextUnitKind::Chunk => {
                            UnitKind::DocSection
                        }
                        prose::TextUnitKind::ConfigKey => UnitKind::ConfigKey,
                    },
                    scope_path: unit.scope_path,
                    header: unit.header,
                    text: unit.text,
                    line_start: unit.line_start,
                    line_end: unit.line_end,
                    has_code_fence: unit.has_code_fence,
                    idents: unit.idents,
                    truncated: unit.truncated,
                })
                .collect())
        }
        Chunker::Config => Ok(
            prose::chunk_config_bounded(&budget, config_format(path), source)
                .into_iter()
                .map(|unit| PreparedUnit {
                    origin: TextOrigin::File,
                    kind: UnitKind::ConfigKey,
                    scope_path: unit.scope_path,
                    header: unit.header,
                    text: unit.text,
                    line_start: unit.line_start,
                    line_end: unit.line_end,
                    has_code_fence: false,
                    idents: unit.idents,
                    truncated: unit.truncated,
                })
                .collect(),
        ),
        Chunker::Pdf | Chunker::Office | Chunker::Notebook | Chunker::None => Ok(Vec::new()),
    }
}

fn config_format(path: &str) -> prose::ConfigFormat {
    let name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if name.starts_with(".env") || name.ends_with(".env") {
        return prose::ConfigFormat::Env;
    }
    match Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("json") => prose::ConfigFormat::Json,
        Some("toml") => prose::ConfigFormat::Toml,
        Some("yaml" | "yml") => prose::ConfigFormat::Yaml,
        _ => prose::ConfigFormat::Env,
    }
}

// T0.13 wires these produced units into the store. Keeping the parser implementation
// together makes that integration mechanical while the phase-0 chunker tests remain
// executable now.
#[allow(
    dead_code,
    reason = "T0.13 is the first production unit-write path; this completed parser is exercised by its permanent goldens until then"
)]
mod ast {
    use super::*;

    struct AstOptions<'a> {
        config: &'a crate::config::Chunk,
        budget: &'a UnitBudget,
    }
    impl std::ops::Deref for AstOptions<'_> {
        type Target = crate::config::Chunk;
        fn deref(&self) -> &Self::Target {
            self.config
        }
    }

    /// Produces at most the shared allowance, preserving configured token bounds.
    pub(super) fn chunk_code_configured(
        config: &crate::config::Chunk,
        budget: &UnitBudget,
        language: Option<AstLanguage>,
        source: &[u8],
    ) -> Result<Vec<AstUnit>, AstError> {
        chunk_code_bounded(&AstOptions { config, budget }, language, source)
    }

    /// Stable chunker identity included in unit hashes from T0.13 onward.
    pub(crate) const CHUNKER_VERSION: &str = "1";

    const TARGET_TOKENS: usize = 500;
    pub(super) const MAX_TOKENS: usize = 1500;
    const MIN_TOKENS: usize = 40;

    pub(super) use super::languages::AstLanguage;
    use super::syntax::Syntax;

    /// One deterministic L1 code unit produced by the cAST chunker.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct AstUnit {
        pub(crate) kind: AstUnitKind,
        pub(crate) scope_path: String,
        pub(crate) header: String,
        pub(crate) text: String,
        pub(crate) line_start: usize,
        pub(crate) line_end: usize,
    }

    /// The unit kind selected by structural code chunking.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) enum AstUnitKind {
        Symbol,
        Chunk,
    }

    /// Chunks source with its known grammar or uses bounded line blocks for unknown languages.
    ///
    /// This is the sole code-chunker entry point so callers cannot accidentally skip the
    /// deterministic fallback when language detection has no static grammar.
    fn chunk_code_bounded(
        options: &AstOptions<'_>,
        language: Option<AstLanguage>,
        source: &[u8],
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        match language {
            Some(language) => chunk_ast_configured(options, language, source),
            None => chunk_lines(options, source),
        }
    }

    /// Chunks UTF-8 source with the cAST recipe for one statically linked grammar.
    fn chunk_ast_configured(
        options: &AstOptions<'_>,
        language: AstLanguage,
        source: &[u8],
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let source = match std::str::from_utf8(source) {
            Ok(source) => source,
            Err(_) => return chunk_lines(options, source),
        };
        if source.is_empty() {
            return Ok(Vec::new());
        }

        let mut parser = Parser::new();
        parser
            .set_language(&language.grammar())
            .map_err(|error| AstError::Language(error.to_string()))?;
        let tree = parser.parse(source, None).ok_or(AstError::ParseCancelled)?;
        let syntax = Syntax::new(language, tree.root_node(), source)?;
        let children = syntax.children(tree.root_node());
        let imports = children
            .iter()
            .copied()
            .filter(|node| is_import(language, node.kind()))
            .collect::<Vec<_>>();

        let mut units = import_unit(options, &imports, source)?;
        let mut pending_docs = Vec::new();
        let mut pending_other = Vec::new();
        for child in children {
            if options.budget.exhausted() {
                break;
            }
            if is_import(language, child.kind()) {
                continue;
            }
            if is_trivia(language, child.kind()) {
                continue;
            }
            if is_doc_comment(child.kind()) {
                pending_docs.push(child);
                continue;
            }
            if syntax.name(child).is_some() || is_definition(language, child.kind()) {
                units.extend(chunk_small_siblings(options, &pending_other, source)?);
                pending_other.clear();
                if options.budget.exhausted() {
                    break;
                }
                let start = pending_docs
                    .first()
                    .map_or(child.start_byte(), |comment| comment.start_byte());
                pending_docs.clear();
                units.extend(chunk_definition(
                    options, language, child, source, None, start, &syntax,
                )?);
                continue;
            }
            pending_other.append(&mut pending_docs);
            pending_other.push(child);
        }
        pending_other.append(&mut pending_docs);
        units.extend(chunk_small_siblings(options, &pending_other, source)?);

        if units.is_empty() {
            return chunk_lines(options, source.as_bytes());
        }
        units.sort_by_key(|unit| (unit.line_start, unit.line_end, unit.header.clone()));
        Ok(units)
    }

    /// Uses documented defaults for the permanent grammar fixtures.
    #[cfg(test)]
    pub(crate) fn chunk_ast(
        language: AstLanguage,
        source: &[u8],
    ) -> Result<Vec<AstUnit>, AstError> {
        chunk_code_configured(
            &crate::config::Chunk::default(),
            &UnitBudget::new(usize::MAX),
            Some(language),
            source,
        )
    }

    /// Uses documented defaults for line-fallback and grammar fixtures.
    #[cfg(test)]
    pub(crate) fn chunk_code(
        language: Option<AstLanguage>,
        source: &[u8],
    ) -> Result<Vec<AstUnit>, AstError> {
        chunk_code_configured(
            &crate::config::Chunk::default(),
            &UnitBudget::new(usize::MAX),
            language,
            source,
        )
    }

    fn is_definition(language: AstLanguage, kind: &str) -> bool {
        match language {
            AstLanguage::Rust => matches!(
                kind,
                "function_item"
                    | "struct_item"
                    | "enum_item"
                    | "union_item"
                    | "impl_item"
                    | "trait_item"
                    | "mod_item"
                    | "const_item"
                    | "static_item"
                    | "type_item"
            ),
            AstLanguage::Python => matches!(
                kind,
                "function_definition" | "class_definition" | "decorated_definition"
            ),
            AstLanguage::JavaScript => matches!(
                kind,
                "function_declaration"
                    | "generator_function_declaration"
                    | "class_declaration"
                    | "lexical_declaration"
                    | "variable_declaration"
                    | "export_statement"
            ),
            AstLanguage::TypeScript | AstLanguage::Tsx => matches!(
                kind,
                "function_declaration"
                    | "generator_function_declaration"
                    | "class_declaration"
                    | "abstract_class_declaration"
                    | "interface_declaration"
                    | "enum_declaration"
                    | "type_alias_declaration"
                    | "internal_module"
                    | "lexical_declaration"
                    | "variable_declaration"
                    | "export_statement"
                    | "ambient_declaration"
            ),
            AstLanguage::Go => matches!(
                kind,
                "function_declaration"
                    | "method_declaration"
                    | "type_declaration"
                    | "var_declaration"
                    | "const_declaration"
            ),
            AstLanguage::Java => matches!(
                kind,
                "class_declaration"
                    | "interface_declaration"
                    | "enum_declaration"
                    | "record_declaration"
                    | "annotation_type_declaration"
                    | "method_declaration"
                    | "constructor_declaration"
                    | "module_declaration"
            ),
            AstLanguage::Php => matches!(
                kind,
                "namespace_definition"
                    | "interface_declaration"
                    | "trait_declaration"
                    | "class_declaration"
                    | "enum_declaration"
                    | "function_definition"
                    | "method_declaration"
            ),
            _ => false,
        }
    }

    fn is_import(language: AstLanguage, kind: &str) -> bool {
        match language {
            AstLanguage::Rust => matches!(kind, "use_declaration" | "extern_crate_declaration"),
            AstLanguage::Python => matches!(kind, "import_statement" | "import_from_statement"),
            AstLanguage::JavaScript | AstLanguage::TypeScript | AstLanguage::Tsx => {
                kind == "import_statement"
            }
            AstLanguage::Go => kind == "import_declaration",
            AstLanguage::Java => matches!(kind, "import_declaration" | "package_declaration"),
            AstLanguage::Php => kind == "namespace_use_declaration",
            AstLanguage::C | AstLanguage::Cpp => kind == "preproc_include",
            AstLanguage::Kotlin => matches!(kind, "import" | "package_header"),
            AstLanguage::CSharp => kind == "using_directive",
            AstLanguage::Swift | AstLanguage::Scala => kind == "import_declaration",
            AstLanguage::Dart => {
                matches!(kind, "import_or_export" | "part_directive" | "library_name")
            }
            AstLanguage::Haskell => kind == "import",
            AstLanguage::Ocaml | AstLanguage::OcamlInterface => {
                matches!(kind, "open_module" | "include_module")
            }
            AstLanguage::Protobuf => matches!(kind, "import" | "package"),
            _ => false,
        }
    }

    pub(super) fn is_import_kind(language: AstLanguage, kind: &str) -> bool {
        is_import(language, kind)
    }

    fn is_doc_comment(kind: &str) -> bool {
        matches!(
            kind,
            "comment" | "line_comment" | "block_comment" | "doc_comment"
        )
    }

    fn is_trivia(language: AstLanguage, kind: &str) -> bool {
        matches!(language, AstLanguage::Php) && matches!(kind, "php_tag" | "php_end_tag")
    }

    fn is_container(kind: &str) -> bool {
        matches!(
            kind,
            "class_definition"
                | "class_declaration"
                | "abstract_class_declaration"
                | "interface_declaration"
                | "enum_declaration"
                | "struct_item"
                | "impl_item"
                | "trait_item"
                | "namespace_definition"
                | "module_declaration"
                | "internal_module"
        )
    }

    fn named_children(node: Node<'_>) -> Vec<Node<'_>> {
        let mut cursor = node.walk();
        node.named_children(&mut cursor).collect()
    }

    fn definition_name(node: Node<'_>, source: &str) -> Result<Option<String>, AstError> {
        find_named_field(node, "name", 3)
            .map(|name| node_text(name, source).map(str::to_owned))
            .transpose()
    }

    fn find_named_field<'tree>(
        node: Node<'tree>,
        field: &str,
        depth: usize,
    ) -> Option<Node<'tree>> {
        if let Some(child) = node.child_by_field_name(field) {
            return Some(child);
        }
        if depth == 0 {
            return None;
        }
        named_children(node)
            .into_iter()
            .find_map(|child| find_named_field(child, field, depth - 1))
    }

    // Tree-sitter end points are exclusive; a nonempty node ending at column
    // zero covers the preceding line, not an extra line beyond EOF.
    fn node_end_line(node: Node<'_>) -> usize {
        let end = node.end_position();
        if end.column == 0 && node.end_byte() > node.start_byte() {
            end.row.max(1)
        } else {
            end.row + 1
        }
    }

    fn node_text<'source>(node: Node<'_>, source: &'source str) -> Result<&'source str, AstError> {
        source
            .get(node.start_byte()..node.end_byte())
            .ok_or(AstError::InvalidRange)
    }

    fn signature(node: Node<'_>, source: &str) -> Result<String, AstError> {
        let text = node_text(node, source)?;
        Ok(text
            .lines()
            .map(str::trim)
            .find(|line| {
                !line.is_empty()
                    && !line.starts_with("//")
                    && !line.starts_with("/*")
                    && !line.starts_with('*')
                    && !line.starts_with('#')
            })
            .unwrap_or_default()
            .to_owned())
    }

    pub(super) fn token_count(text: &str) -> usize {
        super::token_count::count(text)
    }

    fn chunk_definition(
        options: &AstOptions<'_>,
        language: AstLanguage,
        node: Node<'_>,
        source: &str,
        parent_scope: Option<&str>,
        start_byte: usize,
        syntax: &Syntax,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let name = match syntax.name(node) {
            Some(name) => name.to_owned(),
            None => definition_name(node, source)?.unwrap_or_else(|| node.kind().to_owned()),
        };
        let scope_path = parent_scope.map_or(name.clone(), |scope| format!("{scope} › {name}"));
        let header_signature = signature(node, source)?;
        let text = source
            .get(start_byte..node.end_byte())
            .ok_or(AstError::InvalidRange)?;
        let line_start = source_line(source, start_byte);
        let line_end = node_end_line(node);
        if token_count(text) <= (options.max_tokens as usize) {
            return Ok(vec![unit(
                options,
                AstUnitKind::Symbol,
                &scope_path,
                &header_signature,
                text.to_owned(),
                line_start,
                line_end,
            )]);
        }

        let children = definition_children(language, node, syntax);
        if (is_container(node.kind()) || syntax.is_container(node)) && !children.is_empty() {
            let mut units = chunk_container_shell(
                options,
                node,
                &children,
                source,
                &scope_path,
                &header_signature,
                start_byte,
            )?;
            for child in children {
                if options.budget.exhausted() {
                    break;
                }
                units.extend(chunk_definition(
                    options,
                    language,
                    child.node,
                    source,
                    Some(&scope_path),
                    child.start_byte,
                    syntax,
                )?);
            }
            return Ok(units);
        }
        split_leaf(
            options,
            node,
            source,
            &scope_path,
            &header_signature,
            start_byte,
        )
    }

    #[derive(Clone, Copy)]
    struct DefinitionChild<'tree> {
        node: Node<'tree>,
        start_byte: usize,
    }

    fn definition_children<'tree>(
        language: AstLanguage,
        node: Node<'tree>,
        syntax: &Syntax,
    ) -> Vec<DefinitionChild<'tree>> {
        let body = node.child_by_field_name("body").unwrap_or(node);
        let mut children = Vec::new();
        let mut doc_start = None;
        for child in syntax.children(body) {
            if is_doc_comment(child.kind()) {
                doc_start.get_or_insert(child.start_byte());
            } else if syntax.name(child).is_some() || is_definition(language, child.kind()) {
                children.push(DefinitionChild {
                    node: child,
                    start_byte: doc_start.take().unwrap_or(child.start_byte()),
                });
            } else {
                doc_start = None;
            }
        }
        children
    }

    fn chunk_container_shell(
        options: &AstOptions<'_>,
        node: Node<'_>,
        children: &[DefinitionChild<'_>],
        source: &str,
        scope_path: &str,
        header_signature: &str,
        start_byte: usize,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let body = node.child_by_field_name("body").unwrap_or(node);
        let mut shell = String::new();
        let docs = source
            .get(start_byte..node.start_byte())
            .ok_or(AstError::InvalidRange)?
            .trim();
        if !docs.is_empty() {
            shell.push_str(docs);
            shell.push('\n');
        }
        shell.push_str(header_signature);
        let child_ranges = children
            .iter()
            .map(|child| (child.node.start_byte(), child.node.end_byte()))
            .collect::<Vec<_>>();
        for child in named_children(body) {
            if child_ranges
                .iter()
                .any(|(start, end)| child.start_byte() >= *start && child.end_byte() <= *end)
                || is_doc_comment(child.kind())
            {
                continue;
            }
            let text = node_text(child, source)?.trim();
            if !text.is_empty() {
                shell.push('\n');
                shell.push_str(text);
            }
        }
        shell.push_str("\nmethods:");
        for child in children {
            if options.budget.exhausted() {
                break;
            }
            shell.push_str("\n- ");
            shell.push_str(&signature(child.node, source)?);
        }

        let line_start = source_line(source, start_byte);
        let line_end = node_end_line(node);
        if token_count(&shell) <= (options.max_tokens as usize) {
            return Ok(vec![unit(
                options,
                AstUnitKind::Symbol,
                scope_path,
                header_signature,
                shell,
                line_start,
                line_end,
            )]);
        }
        let mut chunks =
            split_text_hard(options, &shell, scope_path, header_signature, line_start)?;
        if let Some(first) = chunks.first_mut() {
            first.kind = AstUnitKind::Symbol;
        }
        Ok(chunks)
    }

    fn split_leaf(
        options: &AstOptions<'_>,
        node: Node<'_>,
        source: &str,
        scope_path: &str,
        header_signature: &str,
        start_byte: usize,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let body = node.child_by_field_name("body");
        let Some(body) = body else {
            return split_text_hard(
                options,
                source
                    .get(start_byte..node.end_byte())
                    .ok_or(AstError::InvalidRange)?,
                scope_path,
                header_signature,
                source_line(source, start_byte),
            );
        };
        let statements = named_children(body);
        if statements.is_empty() {
            return split_text_hard(
                options,
                source
                    .get(start_byte..node.end_byte())
                    .ok_or(AstError::InvalidRange)?,
                scope_path,
                header_signature,
                source_line(source, start_byte),
            );
        }
        let mut pieces = Vec::new();
        let prelude = source
            .get(start_byte..statements[0].start_byte())
            .ok_or(AstError::InvalidRange)?
            .trim();
        if !prelude.is_empty() {
            pieces.push(TextPiece::new(
                prelude.to_owned(),
                source_line(source, start_byte),
                source_line(source, statements[0].start_byte()),
            ));
        }
        for statement in statements {
            pieces.push(TextPiece::new(
                node_text(statement, source)?.to_owned(),
                statement.start_position().row + 1,
                node_end_line(statement),
            ));
        }
        pack_pieces(options, pieces, scope_path, header_signature)
    }

    fn import_unit(
        options: &AstOptions<'_>,
        imports: &[Node<'_>],
        source: &str,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        if imports.is_empty() {
            return Ok(Vec::new());
        }
        let first = imports.first().ok_or(AstError::InvalidRange)?;
        let last = imports.last().ok_or(AstError::InvalidRange)?;
        let text = imports
            .iter()
            .map(|node| node_text(*node, source).map(str::to_owned))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");
        if token_count(&text) <= (options.max_tokens as usize) {
            return Ok(vec![unit(
                options,
                AstUnitKind::Chunk,
                "imports",
                "imports",
                text,
                first.start_position().row + 1,
                node_end_line(*last),
            )]);
        }
        split_text_hard(
            options,
            &text,
            "imports",
            "imports",
            first.start_position().row + 1,
        )
    }

    fn chunk_small_siblings(
        options: &AstOptions<'_>,
        nodes: &[Node<'_>],
        source: &str,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let pieces = nodes
            .iter()
            .map(|node| {
                Ok(TextPiece::new(
                    node_text(*node, source)?.to_owned(),
                    node.start_position().row + 1,
                    node_end_line(*node),
                ))
            })
            .collect::<Result<Vec<_>, AstError>>()?;
        pack_pieces(options, pieces, "", "chunk")
    }

    #[derive(Clone)]
    struct TextPiece {
        text: String,
        line_start: usize,
        line_end: usize,
    }

    impl TextPiece {
        fn new(text: String, line_start: usize, line_end: usize) -> Self {
            Self {
                text,
                line_start,
                line_end,
            }
        }
    }

    fn pack_pieces(
        options: &AstOptions<'_>,
        pieces: Vec<TextPiece>,
        scope_path: &str,
        header_signature: &str,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let mut units = Vec::new();
        let mut small = Vec::new();
        let mut small_tokens = 0;
        for piece in pieces {
            if options.budget.exhausted() {
                break;
            }
            let piece_tokens = token_count(&piece.text);
            if piece_tokens > (options.max_tokens as usize) {
                units.extend(flush_pieces(
                    options,
                    &mut small,
                    scope_path,
                    header_signature,
                ));
                small_tokens = 0;
                units.extend(split_text_hard(
                    options,
                    &piece.text,
                    scope_path,
                    header_signature,
                    piece.line_start,
                )?);
            } else if piece_tokens < (options.min_tokens as usize) {
                let candidate_tokens = small_tokens + piece_tokens;
                if !small.is_empty() && candidate_tokens > (options.target_tokens as usize) {
                    units.extend(flush_pieces(
                        options,
                        &mut small,
                        scope_path,
                        header_signature,
                    ));
                    small_tokens = 0;
                }
                small.push(piece);
                small_tokens += piece_tokens;
            } else {
                units.extend(flush_pieces(
                    options,
                    &mut small,
                    scope_path,
                    header_signature,
                ));
                small_tokens = 0;
                if options.budget.exhausted() {
                    break;
                }
                units.push(unit(
                    options,
                    AstUnitKind::Chunk,
                    scope_path,
                    header_signature,
                    piece.text,
                    piece.line_start,
                    piece.line_end,
                ));
            }
        }
        units.extend(flush_pieces(
            options,
            &mut small,
            scope_path,
            header_signature,
        ));
        Ok(units)
    }

    fn flush_pieces(
        options: &AstOptions<'_>,
        pieces: &mut Vec<TextPiece>,
        scope_path: &str,
        header_signature: &str,
    ) -> Vec<AstUnit> {
        if options.budget.exhausted() {
            return Vec::new();
        }
        if pieces.is_empty() {
            return Vec::new();
        }
        let first = match pieces.first() {
            Some(piece) => piece,
            None => return Vec::new(),
        };
        let last = match pieces.last() {
            Some(piece) => piece,
            None => return Vec::new(),
        };
        let text = pieces
            .iter()
            .map(|piece| piece.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let unit = unit(
            options,
            AstUnitKind::Chunk,
            scope_path,
            header_signature,
            text,
            first.line_start,
            last.line_end,
        );
        pieces.clear();
        vec![unit]
    }

    fn split_text_hard(
        options: &AstOptions<'_>,
        text: &str,
        scope_path: &str,
        header_signature: &str,
        first_line: usize,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let pieces = text
            .lines()
            .enumerate()
            .map(|(index, line)| {
                TextPiece::new(line.to_owned(), first_line + index, first_line + index)
            })
            .collect::<Vec<_>>();
        if pieces.is_empty() {
            return Ok(Vec::new());
        }
        let mut units = Vec::new();
        let mut current = Vec::new();
        let mut current_tokens = 0;
        for piece in pieces {
            if options.budget.exhausted() {
                break;
            }
            let piece_tokens = token_count(&piece.text);
            if piece_tokens > (options.max_tokens as usize) {
                units.extend(flush_pieces(
                    options,
                    &mut current,
                    scope_path,
                    header_signature,
                ));
                current_tokens = 0;
                units.extend(split_long_line(
                    options,
                    piece,
                    scope_path,
                    header_signature,
                )?);
                continue;
            }
            if !current.is_empty()
                && current_tokens + piece_tokens > (options.target_tokens as usize)
            {
                units.extend(flush_pieces(
                    options,
                    &mut current,
                    scope_path,
                    header_signature,
                ));
                current_tokens = 0;
            }
            current.push(piece);
            current_tokens += piece_tokens;
        }
        units.extend(flush_pieces(
            options,
            &mut current,
            scope_path,
            header_signature,
        ));
        Ok(units)
    }

    fn split_long_line(
        options: &AstOptions<'_>,
        piece: TextPiece,
        scope_path: &str,
        header_signature: &str,
    ) -> Result<Vec<AstUnit>, AstError> {
        let mut units = Vec::new();
        let mut remaining = piece.text.as_str();
        while !remaining.is_empty() && !options.budget.exhausted() {
            let cap = options.max_tokens as usize;
            let window = cap.saturating_mul(16).max(4);
            let ends = remaining
                .char_indices()
                .map(|(index, c)| index + c.len_utf8())
                .take_while(|end| *end <= window)
                .collect::<Vec<_>>();
            let mut low = 0usize;
            let mut high = ends.len() + 1;
            while high - low > 1 {
                let middle = low + (high - low) / 2;
                if token_count(&remaining[..ends[middle - 1]]) <= cap {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            if low == 0 {
                return Err(AstError::TokenCapTooSmall);
            }
            let end = ends[low - 1];
            units.push(unit(
                options,
                AstUnitKind::Chunk,
                scope_path,
                header_signature,
                remaining[..end].to_owned(),
                piece.line_start,
                piece.line_end,
            ));
            remaining = &remaining[end..];
        }
        Ok(units)
    }

    fn unit(
        options: &AstOptions<'_>,
        kind: AstUnitKind,
        scope_path: &str,
        header_signature: &str,
        text: String,
        line_start: usize,
        line_end: usize,
    ) -> AstUnit {
        options.budget.consume();
        AstUnit {
            kind,
            scope_path: scope_path.to_owned(),
            header: if scope_path.is_empty() {
                header_signature.to_owned()
            } else {
                format!("{scope_path} › {header_signature}")
            },
            text,
            line_start,
            line_end,
        }
    }

    fn source_line(source: &str, byte: usize) -> usize {
        source.get(..byte).map_or(1, |prefix| {
            prefix.bytes().filter(|byte| *byte == b'\n').count() + 1
        })
    }

    fn chunk_lines(options: &AstOptions<'_>, source: &[u8]) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let text = String::from_utf8_lossy(source);
        let mut paragraphs = Vec::new();
        let mut current = Vec::new();
        let mut start = 1;
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                if !current.is_empty() {
                    paragraphs.push(TextPiece::new(current.join("\n"), start, index));
                    current.clear();
                }
                start = index + 2;
            } else {
                current.push(line.to_owned());
            }
        }
        if !current.is_empty() {
            paragraphs.push(TextPiece::new(
                current.join("\n"),
                start,
                text.lines().count(),
            ));
        }
        pack_line_blocks(options, paragraphs)
    }

    fn pack_line_blocks(
        options: &AstOptions<'_>,
        pieces: Vec<TextPiece>,
    ) -> Result<Vec<AstUnit>, AstError> {
        if options.budget.exhausted() {
            return Ok(Vec::new());
        }
        let mut units = Vec::new();
        let mut current = Vec::new();
        let mut current_tokens = 0;
        for piece in pieces {
            if options.budget.exhausted() {
                break;
            }
            let piece_tokens = token_count(&piece.text);
            if piece_tokens > (options.max_tokens as usize) {
                units.extend(flush_pieces(options, &mut current, "", "lines"));
                current_tokens = 0;
                units.extend(split_text_hard(
                    options,
                    &piece.text,
                    "",
                    "lines",
                    piece.line_start,
                )?);
                continue;
            }
            if !current.is_empty()
                && current_tokens + piece_tokens > (options.target_tokens as usize)
            {
                units.extend(flush_pieces(options, &mut current, "", "lines"));
                current_tokens = 0;
            }
            current.push(piece);
            current_tokens += piece_tokens;
        }
        units.extend(flush_pieces(options, &mut current, "", "lines"));
        Ok(units)
    }

    /// Parse failures from the deterministic code chunker.
    #[derive(Debug, thiserror::Error)]
    pub(crate) enum AstError {
        #[error("chunk.max_tokens cannot hold a source Unicode character")]
        TokenCapTooSmall,
        #[error("tree-sitter rejected the static language: {0}")]
        Language(String),
        #[error("tree-sitter cancelled parsing")]
        ParseCancelled,
        #[error("the static tags query is invalid: {0}")]
        TagsQuery(String),
        #[error("tree-sitter reported an invalid source range")]
        InvalidRange,
    }
}

/// Uses the Markdown chunker's fence rule when replaying generated-page lexical columns.
pub(crate) fn contains_fenced_code(text: &str) -> bool {
    prose::contains_fenced_code(text)
}

// T0.13 gives these units persistent ids and writes them alongside code units.
// The implementation is kept local to `treat` until that common write path exists.
#[allow(
    dead_code,
    reason = "T0.13 is the first production unit-write path; permanent chunker fixtures exercise this implementation until then"
)]
mod prose {
    use super::UnitBudget;
    use std::collections::{BTreeMap, BTreeSet};

    use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

    const MARKDOWN_MIN_TOKENS: usize = 300;
    const MARKDOWN_MAX_TOKENS: usize = 800;
    const CONFIG_NEST_TOKENS: usize = 300;
    const MAX_CONFIG_BYTES: usize = 1024 * 1024;

    /// A deterministic non-code unit which T0.13 will turn into a database unit.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct TextUnit {
        pub(crate) kind: TextUnitKind,
        pub(crate) scope_path: String,
        pub(crate) header: String,
        pub(crate) text: String,
        pub(crate) line_start: usize,
        pub(crate) line_end: usize,
        pub(crate) has_code_fence: bool,
        pub(crate) idents: Vec<String>,
        pub(crate) truncated: bool,
    }

    /// Structural kinds supplied by document and configuration chunkers.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) enum TextUnitKind {
        Markdown,
        Chunk,
        ConfigKey,
    }

    /// Configuration syntaxes with a deterministic L1 key extractor.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) enum ConfigFormat {
        Json,
        Toml,
        Yaml,
        Env,
    }

    /// Splits Markdown by semantic boundaries while retaining the heading breadcrumb.
    pub(crate) fn chunk_markdown(source: &[u8]) -> Vec<TextUnit> {
        chunk_markdown_configured(&UnitBudget::new(usize::MAX), source, [300, 800])
    }

    /// Applies the effective Markdown token range to document chunks.
    pub(crate) fn chunk_markdown_configured(
        budget: &UnitBudget,
        source: &[u8],
        range: [u64; 2],
    ) -> Vec<TextUnit> {
        let text = String::from_utf8_lossy(source);
        let mut units = Vec::new();
        for section in markdown_sections(&text) {
            if budget.exhausted() {
                break;
            }
            let splitter = MarkdownSplitter::new(
                ChunkConfig::new(range[0] as usize..=range[1] as usize)
                    .with_sizer(tiktoken_rs::cl100k_base_singleton()),
            );
            let mut searched = 0;
            for chunk in splitter.chunks(section.text) {
                if budget.exhausted() {
                    break;
                }
                if chunk.trim().is_empty() {
                    continue;
                }
                let relative = section.text[searched..]
                    .find(chunk)
                    .map_or(searched, |offset| searched + offset);
                searched = relative.saturating_add(chunk.len());
                let start = section.start_byte.saturating_add(relative);
                let end = start.saturating_add(chunk.len());
                units.push(text_unit(
                    budget,
                    TextUnitInput {
                        kind: TextUnitKind::Markdown,
                        scope_path: &section.header,
                        header: &section.header,
                        text: chunk.to_owned(),
                        source: &text,
                        start,
                        end,
                        has_code_fence: contains_fenced_code(chunk),
                        idents: Vec::new(),
                        truncated: false,
                    },
                ));
            }
        }
        units
    }

    /// Splits plain reStructuredText or text into token-bounded prose chunks.
    pub(crate) fn chunk_plain(source: &[u8]) -> Vec<TextUnit> {
        chunk_plain_configured(&UnitBudget::new(usize::MAX), source, [300, 800])
    }

    /// Applies the effective Markdown token range to document chunks.
    pub(crate) fn chunk_plain_configured(
        budget: &UnitBudget,
        source: &[u8],
        range: [u64; 2],
    ) -> Vec<TextUnit> {
        let text = String::from_utf8_lossy(source);
        let splitter = TextSplitter::new(
            ChunkConfig::new(range[0] as usize..=range[1] as usize)
                .with_sizer(tiktoken_rs::cl100k_base_singleton()),
        );
        let mut searched = 0;
        splitter
            .chunks(&text)
            .filter(|chunk| !chunk.trim().is_empty())
            .take_while(|_| !budget.exhausted())
            .map(|chunk| {
                let relative = text[searched..]
                    .find(chunk)
                    .map_or(searched, |offset| searched + offset);
                searched = relative.saturating_add(chunk.len());
                text_unit(
                    budget,
                    TextUnitInput {
                        kind: TextUnitKind::Chunk,
                        scope_path: "text",
                        header: "text",
                        text: chunk.to_owned(),
                        source: &text,
                        start: relative,
                        end: relative.saturating_add(chunk.len()),
                        has_code_fence: false,
                        idents: Vec::new(),
                        truncated: false,
                    },
                )
            })
            .collect()
    }

    /// Uses the documented defaults for permanent configuration fixtures.
    #[cfg(test)]
    pub(crate) fn chunk_config(format: ConfigFormat, source: &[u8]) -> Vec<TextUnit> {
        chunk_config_bounded(&UnitBudget::new(usize::MAX), format, source)
    }

    /// Produces one configuration unit per top-level key, nesting oversized values.
    pub(super) fn chunk_config_bounded(
        budget: &UnitBudget,
        format: ConfigFormat,
        source: &[u8],
    ) -> Vec<TextUnit> {
        let (text, truncated) = cap_config_source(source);
        if truncated {
            return vec![text_unit(
                budget,
                TextUnitInput {
                    kind: TextUnitKind::ConfigKey,
                    scope_path: "config",
                    header: "config",
                    text,
                    source: "",
                    start: 0,
                    end: 0,
                    has_code_fence: false,
                    idents: Vec::new(),
                    truncated: true,
                },
            )];
        }
        let mut units = match format {
            ConfigFormat::Json => serde_json::from_str::<serde_json::Value>(&text).map_or_else(
                |_| fallback_config(budget, &text, false),
                |value| json_units(budget, &value, &text, "", false),
            ),
            ConfigFormat::Toml => toml::from_str::<toml::Value>(&text).map_or_else(
                |_| fallback_config(budget, &text, false),
                |value| toml_units(budget, &value, &text, "", false),
            ),
            ConfigFormat::Yaml => yaml_units(budget, &text, false),
            ConfigFormat::Env => env_units(budget, &text, false),
        };
        units.sort_by(|left, right| {
            left.line_start
                .cmp(&right.line_start)
                .then_with(|| left.scope_path.cmp(&right.scope_path))
        });
        units
    }

    struct MarkdownSection<'source> {
        start_byte: usize,
        header: String,
        text: &'source str,
    }

    fn markdown_sections(source: &str) -> Vec<MarkdownSection<'_>> {
        if source.is_empty() {
            return Vec::new();
        }
        let mut sections = Vec::new();
        let mut headings = Vec::new();
        let mut section_start = 0;
        let mut section_header = "document".to_owned();
        let mut offset = 0;
        let mut in_fence = false;
        for line in source.split_inclusive('\n') {
            let trimmed = line.trim_start();
            if is_fence_marker(trimmed) {
                in_fence = !in_fence;
            } else if !in_fence && let Some((level, heading)) = markdown_heading(trimmed) {
                if section_start < offset {
                    sections.push(MarkdownSection {
                        start_byte: section_start,
                        header: section_header.clone(),
                        text: &source[section_start..offset],
                    });
                }
                headings.resize(level, String::new());
                headings[level - 1] = heading;
                headings.truncate(level);
                section_header = headings
                    .iter()
                    .filter(|heading| !heading.is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" › ");
                section_start = offset;
            }
            offset = offset.saturating_add(line.len());
        }
        if section_start < source.len() {
            sections.push(MarkdownSection {
                start_byte: section_start,
                header: section_header,
                text: &source[section_start..],
            });
        }
        sections
    }

    fn markdown_heading(line: &str) -> Option<(usize, String)> {
        let count = line.bytes().take_while(|byte| *byte == b'#').count();
        if !(1..=6).contains(&count)
            || !line
                .as_bytes()
                .get(count)
                .is_some_and(u8::is_ascii_whitespace)
        {
            return None;
        }
        let heading = line[count..].trim().trim_end_matches('#').trim().to_owned();
        (!heading.is_empty()).then_some((count, heading))
    }

    fn is_fence_marker(line: &str) -> bool {
        line.starts_with("```") || line.starts_with("~~~")
    }

    pub(super) fn contains_fenced_code(text: &str) -> bool {
        text.lines().any(|line| is_fence_marker(line.trim_start()))
    }

    fn json_units(
        budget: &UnitBudget,
        value: &serde_json::Value,
        source: &str,
        prefix: &str,
        truncated: bool,
    ) -> Vec<TextUnit> {
        match value {
            serde_json::Value::Object(values) => values
                .iter()
                .take_while(|_| !budget.exhausted())
                .flat_map(|(key, value)| {
                    let path = join_path(prefix, key);
                    json_value_units(budget, value, source, &path, truncated)
                })
                .collect(),
            _ => fallback_config(budget, source, truncated),
        }
    }

    fn json_value_units(
        budget: &UnitBudget,
        value: &serde_json::Value,
        source: &str,
        path: &str,
        truncated: bool,
    ) -> Vec<TextUnit> {
        let rendered = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        let idents = json_value_idents(path, value);
        if token_count(&rendered) <= CONFIG_NEST_TOKENS {
            return vec![config_unit(
                budget, path, rendered, source, false, idents, truncated,
            )];
        }
        match value {
            serde_json::Value::Object(_) => json_units(budget, value, source, path, truncated),
            _ => split_config_text(budget, path, &rendered, source, idents, truncated),
        }
    }

    fn toml_units(
        budget: &UnitBudget,
        value: &toml::Value,
        source: &str,
        prefix: &str,
        truncated: bool,
    ) -> Vec<TextUnit> {
        match value {
            toml::Value::Table(values) => values
                .iter()
                .take_while(|_| !budget.exhausted())
                .flat_map(|(key, value)| {
                    let path = join_path(prefix, key);
                    toml_value_units(budget, value, source, &path, truncated)
                })
                .collect(),
            _ => fallback_config(budget, source, truncated),
        }
    }

    fn toml_value_units(
        budget: &UnitBudget,
        value: &toml::Value,
        source: &str,
        path: &str,
        truncated: bool,
    ) -> Vec<TextUnit> {
        let rendered = value.to_string();
        let idents = toml_value_idents(path, value);
        if token_count(&rendered) <= CONFIG_NEST_TOKENS {
            return vec![config_unit(
                budget, path, rendered, source, false, idents, truncated,
            )];
        }
        match value {
            toml::Value::Table(_) => toml_units(budget, value, source, path, truncated),
            _ => split_config_text(budget, path, &rendered, source, idents, truncated),
        }
    }

    #[derive(Clone)]
    struct YamlEntry {
        path: String,
        depth: usize,
        start: usize,
        line: usize,
    }

    fn yaml_units(budget: &UnitBudget, source: &str, truncated: bool) -> Vec<TextUnit> {
        let entries = yaml_entries(source);
        let roots = entries
            .iter()
            .filter(|entry| entry.depth == 1)
            .cloned()
            .collect::<Vec<_>>();
        if roots.is_empty() {
            return fallback_config(budget, source, truncated);
        }
        roots
            .iter()
            .enumerate()
            .take_while(|_| !budget.exhausted())
            .flat_map(|(index, root)| {
                let end = roots.get(index + 1).map_or(source.len(), |next| next.start);
                yaml_entry_units(budget, root, end, &entries, source, truncated)
            })
            .collect()
    }

    fn yaml_entry_units(
        budget: &UnitBudget,
        entry: &YamlEntry,
        end: usize,
        entries: &[YamlEntry],
        source: &str,
        truncated: bool,
    ) -> Vec<TextUnit> {
        let text = source
            .get(entry.start..end)
            .unwrap_or_default()
            .trim()
            .to_owned();
        let idents = entries
            .iter()
            .filter(|candidate| yaml_path_is_descendant(&candidate.path, &entry.path))
            .filter_map(|candidate| candidate.path.rsplit('.').next().map(str::to_owned))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if token_count(&text) <= CONFIG_NEST_TOKENS {
            return vec![text_unit(
                budget,
                TextUnitInput {
                    kind: TextUnitKind::ConfigKey,
                    scope_path: &entry.path,
                    header: &entry.path,
                    text,
                    source,
                    start: entry.start,
                    end,
                    has_code_fence: false,
                    idents,
                    truncated,
                },
            )];
        }
        let children = entries
            .iter()
            .filter(|candidate| candidate.depth == entry.depth + 1)
            .filter(|candidate| yaml_path_is_descendant(&candidate.path, &entry.path))
            .cloned()
            .collect::<Vec<_>>();
        if children.is_empty() {
            return split_config_text(budget, &entry.path, &text, source, idents, truncated);
        }
        children
            .iter()
            .enumerate()
            .take_while(|_| !budget.exhausted())
            .flat_map(|(index, child)| {
                let child_end = children.get(index + 1).map_or(end, |next| next.start);
                yaml_entry_units(budget, child, child_end, entries, source, truncated)
            })
            .collect()
    }

    fn yaml_entries(source: &str) -> Vec<YamlEntry> {
        let mut entries = Vec::new();
        let mut stack = Vec::<(usize, String)>::new();
        let mut list_indices = BTreeMap::<String, usize>::new();
        let mut offset = 0;
        for (line_index, line) in source.split_inclusive('\n').enumerate() {
            let without_newline = line.trim_end_matches(['\r', '\n']);
            let indent = without_newline
                .len()
                .saturating_sub(without_newline.trim_start().len());
            let trimmed = without_newline.trim_start();
            if let Some(item) = yaml_list_item(trimmed) {
                pop_yaml_stack(&mut stack, indent);
                let parent = yaml_stack_path(&stack);
                let index = list_indices.entry(parent).or_default();
                let segment = format!("[{index}]");
                *index = index.saturating_add(1);
                let path = yaml_join_path(&yaml_stack_path(&stack), &segment);
                entries.push(YamlEntry {
                    path,
                    depth: stack.len() + 1,
                    start: offset,
                    line: line_index + 1,
                });
                stack.push((indent, segment));
                if let Some(key) = yaml_key(item) {
                    let key_indent = indent.saturating_add(2);
                    pop_yaml_stack(&mut stack, key_indent);
                    let path = yaml_join_path(&yaml_stack_path(&stack), &key);
                    entries.push(YamlEntry {
                        path,
                        depth: stack.len() + 1,
                        start: offset,
                        line: line_index + 1,
                    });
                    stack.push((key_indent, key));
                }
            } else if let Some(key) = yaml_key(trimmed) {
                pop_yaml_stack(&mut stack, indent);
                let path = yaml_join_path(&yaml_stack_path(&stack), &key);
                entries.push(YamlEntry {
                    path,
                    depth: stack.len() + 1,
                    start: offset,
                    line: line_index + 1,
                });
                stack.push((indent, key));
            }
            offset = offset.saturating_add(line.len());
        }
        entries
    }

    fn pop_yaml_stack(stack: &mut Vec<(usize, String)>, indent: usize) {
        while stack
            .last()
            .is_some_and(|(parent_indent, _)| *parent_indent >= indent)
        {
            stack.pop();
        }
    }

    fn yaml_stack_path(stack: &[(usize, String)]) -> String {
        stack.iter().fold(String::new(), |path, (_, segment)| {
            yaml_join_path(&path, segment)
        })
    }

    fn yaml_join_path(parent: &str, segment: &str) -> String {
        if parent.is_empty() || segment.starts_with('[') {
            format!("{parent}{segment}")
        } else {
            format!("{parent}.{segment}")
        }
    }

    fn yaml_path_is_descendant(path: &str, parent: &str) -> bool {
        path == parent
            || path
                .strip_prefix(parent)
                .is_some_and(|suffix| suffix.starts_with('.') || suffix.starts_with('['))
    }

    fn yaml_list_item(line: &str) -> Option<&str> {
        let item = line.strip_prefix('-')?;
        if item.is_empty() || item.starts_with(char::is_whitespace) {
            Some(item.trim_start())
        } else {
            None
        }
    }

    fn yaml_key(line: &str) -> Option<String> {
        if line.is_empty() || line.starts_with('#') || line.starts_with('-') {
            return None;
        }
        let key = line.split_once(':')?.0.trim().trim_matches(['\'', '"']);
        (!key.is_empty() && !key.contains(char::is_whitespace)).then_some(key.to_owned())
    }

    fn env_units(budget: &UnitBudget, source: &str, truncated: bool) -> Vec<TextUnit> {
        let mut offset = 0;
        let mut units = Vec::new();
        for (line_index, line) in source.split_inclusive('\n').enumerate() {
            if budget.exhausted() {
                break;
            }
            let raw = line.trim_end_matches(['\r', '\n']);
            let assignment = raw
                .trim_start()
                .strip_prefix("export ")
                .unwrap_or(raw.trim_start());
            if let Some((key, _)) = assignment.split_once('=') {
                let key = key.trim();
                if !key.is_empty() && !key.contains(char::is_whitespace) {
                    units.extend(split_config_text_at(
                        budget,
                        key,
                        raw,
                        source,
                        (offset, line_index + 1),
                        vec![key.to_owned()],
                        truncated,
                    ));
                }
            }
            offset = offset.saturating_add(line.len());
        }
        if units.is_empty() {
            fallback_config(budget, source, truncated)
        } else {
            units
        }
    }

    fn split_config_text(
        budget: &UnitBudget,
        path: &str,
        text: &str,
        source: &str,
        idents: Vec<String>,
        truncated: bool,
    ) -> Vec<TextUnit> {
        let (start, line) = config_start(source, path);
        split_config_text_at(budget, path, text, source, (start, line), idents, truncated)
    }

    fn split_config_text_at(
        budget: &UnitBudget,
        path: &str,
        text: &str,
        source: &str,
        location: (usize, usize),
        idents: Vec<String>,
        truncated: bool,
    ) -> Vec<TextUnit> {
        let (start, line) = location;
        let splitter = TextSplitter::new(
            ChunkConfig::new(CONFIG_NEST_TOKENS).with_sizer(tiktoken_rs::cl100k_base_singleton()),
        );
        splitter
            .chunks(text)
            .filter(|chunk| !chunk.trim().is_empty())
            .take_while(|_| !budget.exhausted())
            .map(|chunk| {
                text_unit(
                    budget,
                    TextUnitInput {
                        kind: TextUnitKind::ConfigKey,
                        scope_path: path,
                        header: path,
                        text: chunk.to_owned(),
                        source,
                        start,
                        end: start.saturating_add(chunk.len()),
                        has_code_fence: false,
                        idents: idents.clone(),
                        truncated,
                    },
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|mut unit| {
                unit.line_start = line;
                unit.line_end = line.max(unit.line_end);
                unit
            })
            .collect()
    }

    fn config_unit(
        budget: &UnitBudget,
        path: &str,
        text: String,
        source: &str,
        has_code_fence: bool,
        idents: Vec<String>,
        truncated: bool,
    ) -> TextUnit {
        let (start, _) = config_start(source, path);
        text_unit(
            budget,
            TextUnitInput {
                kind: TextUnitKind::ConfigKey,
                scope_path: path,
                header: path,
                text,
                source,
                start,
                end: start,
                has_code_fence,
                idents,
                truncated,
            },
        )
    }

    fn config_start(source: &str, path: &str) -> (usize, usize) {
        let key = path.rsplit('.').next().unwrap_or(path);
        let mut offset = 0;
        for (index, line) in source.split_inclusive('\n').enumerate() {
            let trimmed = line.trim();
            let table = format!("[{path}]");
            if trimmed == table
                || trimmed.starts_with(&format!("{key} ="))
                || trimmed.starts_with(&format!("\"{key}\""))
                || trimmed.starts_with(&format!("{key}:"))
            {
                return (offset, index + 1);
            }
            offset = offset.saturating_add(line.len());
        }
        (0, 1)
    }

    fn json_idents(value: &serde_json::Value) -> Vec<String> {
        let mut idents = BTreeSet::new();
        collect_json_idents(value, &mut idents);
        idents.into_iter().collect()
    }

    fn json_value_idents(path: &str, value: &serde_json::Value) -> Vec<String> {
        let mut idents = json_idents(value).into_iter().collect::<BTreeSet<_>>();
        if !value.is_object() {
            idents.insert(path.rsplit('.').next().unwrap_or(path).to_owned());
        }
        idents.into_iter().collect()
    }

    fn collect_json_idents(value: &serde_json::Value, idents: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(values) => {
                for (key, child) in values {
                    idents.insert(key.to_owned());
                    collect_json_idents(child, idents);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    collect_json_idents(child, idents);
                }
            }
            _ => {}
        }
    }

    fn toml_idents(value: &toml::Value) -> Vec<String> {
        let mut idents = BTreeSet::new();
        collect_toml_idents(value, &mut idents);
        idents.into_iter().collect()
    }

    fn toml_value_idents(path: &str, value: &toml::Value) -> Vec<String> {
        let mut idents = toml_idents(value).into_iter().collect::<BTreeSet<_>>();
        if !value.is_table() {
            idents.insert(path.rsplit('.').next().unwrap_or(path).to_owned());
        }
        idents.into_iter().collect()
    }

    fn collect_toml_idents(value: &toml::Value, idents: &mut BTreeSet<String>) {
        match value {
            toml::Value::Table(values) => {
                for (key, child) in values {
                    idents.insert(key.to_owned());
                    collect_toml_idents(child, idents);
                }
            }
            toml::Value::Array(values) => {
                for child in values {
                    collect_toml_idents(child, idents);
                }
            }
            _ => {}
        }
    }

    fn fallback_config(budget: &UnitBudget, source: &str, truncated: bool) -> Vec<TextUnit> {
        if source.trim().is_empty() {
            Vec::new()
        } else {
            vec![text_unit(
                budget,
                TextUnitInput {
                    kind: TextUnitKind::ConfigKey,
                    scope_path: "config",
                    header: "config",
                    text: source.to_owned(),
                    source,
                    start: 0,
                    end: source.len(),
                    has_code_fence: false,
                    idents: Vec::new(),
                    truncated,
                },
            )]
        }
    }

    fn cap_config_source(source: &[u8]) -> (String, bool) {
        if source.len() <= MAX_CONFIG_BYTES {
            return (String::from_utf8_lossy(source).into_owned(), false);
        }
        (
            String::from_utf8_lossy(&source[..MAX_CONFIG_BYTES]).into_owned(),
            true,
        )
    }

    fn token_count(text: &str) -> usize {
        tiktoken_rs::cl100k_base_singleton()
            .encode_with_special_tokens(text)
            .len()
    }

    fn join_path(prefix: &str, key: &str) -> String {
        if prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{prefix}.{key}")
        }
    }

    struct TextUnitInput<'source> {
        kind: TextUnitKind,
        scope_path: &'source str,
        header: &'source str,
        text: String,
        source: &'source str,
        start: usize,
        end: usize,
        has_code_fence: bool,
        idents: Vec<String>,
        truncated: bool,
    }

    fn text_unit(budget: &UnitBudget, input: TextUnitInput<'_>) -> TextUnit {
        budget.consume();
        TextUnit {
            kind: input.kind,
            scope_path: input.scope_path.to_owned(),
            header: input.header.to_owned(),
            text: input.text,
            line_start: source_line(input.source, input.start),
            line_end: source_line(input.source, input.end.saturating_sub(1).max(input.start)),
            has_code_fence: input.has_code_fence,
            idents: input.idents,
            truncated: input.truncated,
        }
    }

    fn source_line(source: &str, byte: usize) -> usize {
        source
            .as_bytes()
            .get(..byte.min(source.len()))
            .map_or(1, |prefix| {
                prefix.iter().filter(|byte| **byte == b'\n').count() + 1
            })
    }
}

// T0.13 supplies the common indexed-unit write path. Keeping the FTS builder here
// makes its derived columns independently testable and usable for rebuilds now.
#[allow(
    dead_code,
    reason = "T0.13 wires indexed units to the store; T0.11 permanent goldens exercise this builder until then"
)]
pub(crate) mod fts {
    /// Lexical derivation generation; L1 fingerprints rebuild corrected columns.
    pub(crate) const VERSION: u32 = 2;

    use std::collections::BTreeSet;

    use crate::{config::Tokenize, store::FileKind};

    /// The four owned FTS5 columns derived from one indexed unit.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct LexicalColumns {
        pub(crate) header: String,
        pub(crate) idents: String,
        pub(crate) body: String,
        pub(crate) prose: String,
    }

    /// Input required to derive lexical columns without relying on FTS5 state.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct FtsInput<'source> {
        pub(crate) breadcrumb: &'source str,
        pub(crate) header: &'source str,
        pub(crate) text: &'source str,
        pub(crate) language: Option<&'source str>,
        pub(crate) kind: FileKind,
        pub(crate) extra_idents: &'source [&'source str],
        pub(crate) has_code_fence: bool,
    }

    /// Builds the four FTS5 columns specified by `v1.md` §6.3.
    pub(crate) fn build_columns(input: FtsInput<'_>, tokenize: &Tokenize) -> LexicalColumns {
        let split_subtokens = tokenize.split_subtokens
            && !input.language.is_some_and(|language| {
                tokenize
                    .split_subtokens_off_for
                    .iter()
                    .any(|off| off == language)
            });
        let mut identifiers = extract_identifiers(input.text)
            .into_iter()
            .collect::<BTreeSet<_>>();
        identifiers.extend(
            input
                .extra_idents
                .iter()
                .map(|identifier| (*identifier).to_owned()),
        );

        let (body, prose) = if input.kind == FileKind::Code {
            code_columns(input.text, input.language, split_subtokens)
        } else if input.has_code_fence {
            (
                fenced_code_body(input.text, input.language, split_subtokens),
                normalize_whitespace(input.text),
            )
        } else {
            (String::new(), normalize_whitespace(input.text))
        };

        LexicalColumns {
            header: flatten_header(input.breadcrumb, input.header),
            idents: identifiers.into_iter().collect::<Vec<_>>().join(" "),
            body,
            prose,
        }
    }

    /// Turns a free-text query into a quoted FTS5 MATCH expression.
    ///
    /// Identifier candidates retain their original spelling and also search their
    /// camel/snake-case sub-tokens, so code and natural-language spelling converge.
    pub(crate) fn build_match(query: &str) -> String {
        build_match_joined(query, " AND ")
    }

    /// Builds the relaxed lexical query without splitting quoted phrases or identifier groups.
    pub(crate) fn build_or_match(query: &str) -> String {
        build_match_joined(query, " OR ")
    }

    fn build_match_joined(query: &str, join: &str) -> String {
        let mut clauses = Vec::new();
        let mut remainder = query;
        while let Some(start) = remainder.find('"') {
            clauses.extend(identifier_clauses(&remainder[..start]));
            let quoted = &remainder[start + 1..];
            if let Some(end) = quoted.find('"') {
                let phrase = normalize_whitespace(&quoted[..end]);
                if !phrase.is_empty() {
                    clauses.push(fts_quote(&phrase));
                }
                remainder = &quoted[end + 1..];
            } else {
                clauses.extend(identifier_clauses(quoted));
                remainder = "";
            }
        }
        clauses.extend(identifier_clauses(remainder));
        clauses.join(join)
    }

    fn identifier_clauses(text: &str) -> Vec<String> {
        extract_identifiers(text)
            .into_iter()
            .map(|identifier| {
                let exact = fts_quote(&identifier);
                let subtokens = split_identifier(&identifier);
                if subtokens.len() <= 1 {
                    exact
                } else {
                    let expanded = subtokens
                        .iter()
                        .map(|subtoken| fts_quote(subtoken))
                        .collect::<Vec<_>>()
                        .join(" AND ");
                    format!("({exact} OR ({expanded}))")
                }
            })
            .collect()
    }

    fn fts_quote(term: &str) -> String {
        format!("\"{}\"", term.replace('"', "\"\""))
    }

    /// Renders the same searchable header during L1 publication and role refresh.
    pub(crate) fn flatten_header(breadcrumb: &str, header: &str) -> String {
        [breadcrumb, header]
            .into_iter()
            .flat_map(extract_identifiers)
            .flat_map(|identifier| split_identifier(&identifier))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn code_columns(text: &str, language: Option<&str>, split_subtokens: bool) -> (String, String) {
        let mut body = String::new();
        let mut prose = String::new();
        let mut offset = 0;
        while offset < text.len() {
            let rest = &text[offset..];
            if let Some((end, comment)) = line_comment(rest, language) {
                push_prose(&mut prose, comment);
                offset += end;
            } else if let Some((end, comment)) = block_comment(rest, language) {
                push_prose(&mut prose, comment);
                offset += end;
            } else if let Some((end, string)) = string_literal(rest, language) {
                push_prose(&mut prose, string);
                offset += end;
            } else if let Some((end, identifier)) = identifier_at(rest) {
                push_body_identifier(&mut body, identifier, split_subtokens);
                offset += end;
            } else if let Some(character) = rest.chars().next() {
                body.push(' ');
                offset += character.len_utf8();
            } else {
                break;
            }
        }
        (normalize_whitespace(&body), normalize_whitespace(&prose))
    }

    fn fenced_code_body(text: &str, language: Option<&str>, split_subtokens: bool) -> String {
        let mut in_fence = false;
        let mut code = String::new();
        for line in text.split_inclusive('\n') {
            if is_fence_marker(line.trim_start()) {
                in_fence = !in_fence;
            } else if in_fence {
                code.push_str(line);
            }
        }
        code_columns(&code, language, split_subtokens).0
    }

    fn line_comment<'source>(
        text: &'source str,
        language: Option<&str>,
    ) -> Option<(usize, &'source str)> {
        let marker = match language {
            Some("python" | "ruby" | "bash" | "elixir") if text.starts_with('#') => "#",
            Some("sql" | "lua" | "haskell") if text.starts_with("--") => "--",
            _ if text.starts_with("//") => "//",
            _ => return None,
        };
        let end = text.find('\n').unwrap_or(text.len());
        Some((end, text[marker.len()..end].trim()))
    }

    fn block_comment<'source>(
        text: &'source str,
        language: Option<&str>,
    ) -> Option<(usize, &'source str)> {
        if matches!(
            language,
            Some("python" | "ruby" | "bash" | "elixir" | "sql" | "lua")
        ) || !text.starts_with("/*")
        {
            return None;
        }
        // Only a present terminator consumes two trailing bytes. At EOF the
        // complete remaining text is prose, including short and UTF-8 comments.
        let (closing, comment_end) = text[2..]
            .find("*/")
            .map_or((text.len(), text.len()), |index| (index + 4, index + 2));
        Some((closing, text[2..comment_end].trim()))
    }

    fn string_literal<'source>(
        text: &'source str,
        language: Option<&str>,
    ) -> Option<(usize, &'source str)> {
        let quote = text.chars().next()?;
        if quote != '\'' && quote != '"' {
            return None;
        }
        let triple = matches!(language, Some("python"))
            && text.starts_with(&format!("{quote}{quote}{quote}"));
        let prefix = if triple {
            quote.len_utf8() * 3
        } else {
            quote.len_utf8()
        };
        let (closing, content_end) = if triple {
            let marker = format!("{quote}{quote}{quote}");
            text[prefix..].find(&marker).map_or_else(
                || (text.len(), text.len()),
                |index| {
                    let content_end = index.saturating_add(prefix);
                    (content_end.saturating_add(marker.len()), content_end)
                },
            )
        } else {
            quoted_end(text, prefix, quote).map_or_else(
                || (text.len(), text.len()),
                |closing| (closing, closing.saturating_sub(prefix)),
            )
        };
        Some((closing, text[prefix..content_end].trim()))
    }

    fn quoted_end(text: &str, start: usize, quote: char) -> Option<usize> {
        let mut escaped = false;
        for (offset, character) in text[start..].char_indices() {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == quote {
                return Some(start + offset + character.len_utf8());
            }
        }
        None
    }

    fn push_body_identifier(body: &mut String, identifier: &str, split_subtokens: bool) {
        let terms = if split_subtokens {
            split_identifier(identifier)
        } else {
            vec![identifier.to_ascii_lowercase()]
        };
        if !body.is_empty() {
            body.push(' ');
        }
        body.push_str(&terms.join(" "));
    }

    fn push_prose(prose: &mut String, content: &str) {
        if !content.is_empty() {
            if !prose.is_empty() {
                prose.push(' ');
            }
            prose.push_str(content);
        }
    }

    fn extract_identifiers(text: &str) -> Vec<String> {
        let mut identifiers = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let rest = &text[offset..];
            if let Some((end, identifier)) = identifier_at(rest) {
                identifiers.push(identifier.to_owned());
                offset += end;
            } else if let Some(character) = rest.chars().next() {
                offset += character.len_utf8();
            } else {
                break;
            }
        }
        identifiers
    }

    fn identifier_at(text: &str) -> Option<(usize, &str)> {
        let first = text.chars().next()?;
        if !is_identifier_character(first) {
            return None;
        }
        let end = text
            .char_indices()
            .find_map(|(index, character)| {
                (index > 0 && !is_identifier_character(character)).then_some(index)
            })
            .unwrap_or(text.len());
        Some((end, &text[..end]))
    }

    fn is_identifier_character(character: char) -> bool {
        character.is_alphanumeric() || character == '_' || character == '$'
    }

    fn split_identifier(identifier: &str) -> Vec<String> {
        let mut terms = Vec::new();
        for part in identifier.split(['_', '$']).filter(|part| !part.is_empty()) {
            let characters = part.chars().collect::<Vec<_>>();
            let mut start = 0;
            for index in 1..characters.len() {
                let previous = characters[index - 1];
                let current = characters[index];
                let next = characters.get(index + 1).copied();
                let boundary = (previous.is_lowercase() && current.is_uppercase())
                    || (previous.is_uppercase()
                        && current.is_uppercase()
                        && next.is_some_and(char::is_lowercase));
                if boundary {
                    terms.push(
                        characters[start..index]
                            .iter()
                            .collect::<String>()
                            .to_lowercase(),
                    );
                    start = index;
                }
            }
            if start < characters.len() {
                terms.push(
                    characters[start..]
                        .iter()
                        .collect::<String>()
                        .to_lowercase(),
                );
            }
        }
        terms
    }

    fn normalize_whitespace(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn is_fence_marker(line: &str) -> bool {
        line.starts_with("```") || line.starts_with("~~~")
    }
}

/// A file-level L3 job permitted by a treatment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Enrichment {
    DescribeSymbol,
    DescribeImage,
    VlmPage,
}

/// The effective treatment chosen for one repository-relative path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedTreatment {
    pub(crate) name: String,
    pub(crate) chunker: Chunker,
    pub(crate) max_level: i64,
    pub(crate) enrich: Vec<Enrichment>,
    pub(crate) never_send: bool,
}

struct CompiledTreatment {
    name: String,
    patterns: GlobSet,
    chunker: Option<Chunker>,
    max_level: i64,
    enrich: Vec<Enrichment>,
    never_send: bool,
}

/// First-match treatment router compiled from effective configuration entries.
pub(crate) struct Router {
    configured: Arc<[CompiledTreatment]>,
    vlm_configured: bool,
}

struct CachedRouter {
    treatments: Vec<Treatment>,
    configured: Arc<[CompiledTreatment]>,
}

// At most one additional compiled configuration is retained process-wide. This
// caches only a pure compilation result, never filesystem or policy authority.
static ROUTER_CACHE: Mutex<Option<CachedRouter>> = Mutex::new(None);

impl Router {
    /// Reuses compiled globs only for exactly equal treatment values. Callers still
    /// load and validate current configuration and source authority independently.
    pub(crate) fn new(treatments: &[Treatment], vlm_configured: bool) -> Result<Self, Error> {
        {
            let cached = ROUTER_CACHE
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(cached) = cached
                .as_ref()
                .filter(|entry| entry.treatments == treatments)
            {
                return Ok(Self {
                    configured: Arc::clone(&cached.configured),
                    vlm_configured,
                });
            }
        }
        // Compile outside the lock; independent configurations must not serialize
        // their parsing, and a rejected configuration must never reuse a valid one.
        let configured: Arc<[CompiledTreatment]> = treatments
            .iter()
            .map(CompiledTreatment::from_config)
            .collect::<Result<Vec<_>, _>>()?
            .into();
        *ROUTER_CACHE
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(CachedRouter {
            treatments: treatments.to_vec(),
            configured: Arc::clone(&configured),
        });
        Ok(Self {
            configured,
            vlm_configured,
        })
    }

    /// Resolves the first configured match or the v1 built-in default for this file kind.
    pub(crate) fn route(&self, path: &str, kind: FileKind) -> ResolvedTreatment {
        let path = Path::new(path);
        if let Some(treatment) = self
            .configured
            .iter()
            .find(|treatment| treatment.patterns.is_match(path))
        {
            return treatment.resolve(path, kind);
        }
        builtin_default(path, kind, self.vlm_configured)
    }
}

impl Default for Router {
    fn default() -> Self {
        match Self::new(&[], false) {
            Ok(router) => router,
            Err(_) => Self {
                configured: Arc::from([]),
                vlm_configured: false,
            },
        }
    }
}

impl CompiledTreatment {
    fn from_config(treatment: &Treatment) -> Result<Self, Error> {
        if treatment.max_level > 3 {
            return Err(Error::InvalidMaxLevel {
                name: treatment.name.clone(),
                value: treatment.max_level,
            });
        }
        let mut patterns = GlobSetBuilder::new();
        for pattern in &treatment.patterns {
            let glob = Glob::new(pattern).map_err(|source| Error::InvalidGlob {
                name: treatment.name.clone(),
                pattern: pattern.clone(),
                source,
            })?;
            patterns.add(glob);
        }
        let chunker = treatment
            .chunker
            .as_deref()
            .map(|value| parse_chunker(&treatment.name, value))
            .transpose()?;
        let enrich = treatment
            .enrich
            .iter()
            .map(|value| parse_enrichment(&treatment.name, value))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            name: treatment.name.clone(),
            patterns: patterns.build().map_err(|source| Error::BuildGlobSet {
                name: treatment.name.clone(),
                source,
            })?,
            chunker,
            max_level: i64::try_from(treatment.max_level).unwrap_or(i64::MAX),
            enrich,
            never_send: treatment.never_send,
        })
    }

    fn resolve(&self, path: &Path, kind: FileKind) -> ResolvedTreatment {
        let mut max_level = self.max_level;
        let enrich = if self.never_send {
            max_level = max_level.min(1);
            Vec::new()
        } else {
            self.enrich.clone()
        };
        ResolvedTreatment {
            name: self.name.clone(),
            chunker: self
                .chunker
                .unwrap_or_else(|| builtin_default(path, kind, false).chunker),
            max_level,
            enrich,
            never_send: self.never_send,
        }
    }
}

fn parse_chunker(name: &str, value: &str) -> Result<Chunker, Error> {
    match value {
        "ast" => Ok(Chunker::Ast),
        "lines" => Ok(Chunker::Lines),
        "markdown" => Ok(Chunker::Markdown),
        "pdf" => Ok(Chunker::Pdf),
        "office" => Ok(Chunker::Office),
        "config" => Ok(Chunker::Config),
        "none" => Ok(Chunker::None),
        _ => Err(Error::InvalidChunker {
            name: name.to_owned(),
            value: value.to_owned(),
        }),
    }
}

fn parse_enrichment(name: &str, value: &str) -> Result<Enrichment, Error> {
    match value {
        "describe_symbol" => Ok(Enrichment::DescribeSymbol),
        "describe_image" => Ok(Enrichment::DescribeImage),
        "vlm_page" => Ok(Enrichment::VlmPage),
        _ => Err(Error::InvalidEnrichment {
            name: name.to_owned(),
            value: value.to_owned(),
        }),
    }
}

fn builtin_default(path: &Path, kind: FileKind, vlm_configured: bool) -> ResolvedTreatment {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match kind {
        FileKind::Code => resolved("default-code", Chunker::Ast, 2, Vec::new()),
        FileKind::Config => resolved("default-config", Chunker::Config, 1, Vec::new()),
        FileKind::Image if vlm_configured => resolved(
            "default-image",
            Chunker::None,
            3,
            vec![Enrichment::DescribeImage],
        ),
        FileKind::Image => resolved("default-image", Chunker::None, 0, Vec::new()),
        FileKind::Doc if extension == "pdf" => resolved("default-pdf", Chunker::Pdf, 2, Vec::new()),
        FileKind::Doc if matches!(extension.as_str(), "docx" | "xlsx" | "pptx" | "odt") => {
            resolved("default-office", Chunker::Office, 2, Vec::new())
        }
        FileKind::Doc if extension == "ipynb" => {
            resolved("default-notebook", Chunker::Notebook, 2, Vec::new())
        }
        FileKind::Doc => resolved("default-markdown", Chunker::Markdown, 2, Vec::new()),
        FileKind::Data | FileKind::Other => resolved("default-l0", Chunker::None, 0, Vec::new()),
    }
}

fn resolved(
    name: &str,
    chunker: Chunker,
    max_level: i64,
    enrich: Vec<Enrichment>,
) -> ResolvedTreatment {
    ResolvedTreatment {
        name: name.to_owned(),
        chunker,
        max_level,
        enrich,
        never_send: false,
    }
}

/// Invalid treatment declarations that would otherwise make routing ambiguous.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Ast(#[from] ast::AstError),
    #[error("treatment {name} has invalid glob {pattern}: {source}")]
    InvalidGlob {
        name: String,
        pattern: String,
        #[source]
        source: globset::Error,
    },
    #[error("treatment {name} could not compile its glob set: {source}")]
    BuildGlobSet {
        name: String,
        #[source]
        source: globset::Error,
    },
    #[error("treatment {name} has unknown chunker {value}")]
    InvalidChunker { name: String, value: String },
    #[error("treatment {name} has max_level {value}, but levels range from 0 to 3")]
    InvalidMaxLevel { name: String, value: u64 },
    #[error("treatment {name} has unsupported file enrichment job {value}")]
    InvalidEnrichment { name: String, value: String },
}

#[cfg(test)]
mod tests {
    #[test]
    fn token_count_fast_path_preserves_specials_unicode_and_partial_markers() {
        let tokenizer = tiktoken_rs::cl100k_base_singleton();
        for text in [
            "",
            "pub fn example() { /* λ 中文 */ }\n",
            "a\r\nb\t c",
            "<|",
            "<|unknown|>",
            "ordinary prefix <|endoftext|> suffix",
        ] {
            assert_eq!(
                super::count_tokens(text),
                tokenizer.encode_with_special_tokens(text).len(),
                "{text:?}"
            );
        }
        for special in tokenizer.special_tokens() {
            let text = format!("before {special} after {special}");
            assert_eq!(
                super::count_tokens(&text),
                tokenizer.encode_with_special_tokens(&text).len()
            );
        }
    }
    use std::{fs, path::Path, process::Command};

    use super::ast::{AstError, AstLanguage, AstUnitKind, CHUNKER_VERSION, chunk_ast, chunk_code};
    use super::fts::{FtsInput, LexicalColumns, build_columns, build_match};
    use super::prose::{ConfigFormat, TextUnitKind, chunk_config, chunk_markdown, chunk_plain};
    use super::{Chunker, Enrichment, Router};
    use crate::{
        config::{Config, Treatment},
        store::FileKind,
    };
    use tree_sitter::Query;

    #[derive(Debug, Eq, PartialEq)]
    struct GoldenUnit {
        kind: AstUnitKind,
        scope_path: String,
        header: String,
        line_start: usize,
        line_end: usize,
    }

    #[derive(Debug, Eq, PartialEq)]
    struct TextGolden {
        kind: TextUnitKind,
        scope_path: String,
        header: String,
        has_code_fence: bool,
        idents: Vec<String>,
    }

    #[derive(Debug, Eq, PartialEq)]
    struct FtsGolden {
        columns: LexicalColumns,
    }

    fn parse_golden_units(text: &str) -> Result<Vec<GoldenUnit>, String> {
        text.trim()
            .split("\n- ")
            .map(|entry| {
                let mut kind = None;
                let mut scope_path = None;
                let mut header = None;
                let mut line_start = None;
                let mut line_end = None;
                for line in entry.trim_start_matches("- ").lines() {
                    let Some((key, value)) = line.trim().split_once(": ") else {
                        return Err(format!("invalid golden line: {line}"));
                    };
                    match key {
                        "kind" => {
                            kind = Some(match value {
                                "symbol" => AstUnitKind::Symbol,
                                "chunk" => AstUnitKind::Chunk,
                                _ => return Err(format!("unknown golden unit kind: {value}")),
                            });
                        }
                        "scope_path" => scope_path = Some(value.to_owned()),
                        "header" => header = Some(value.to_owned()),
                        "line_start" => {
                            line_start =
                                Some(value.parse().map_err(|error| {
                                    format!("invalid line_start {value}: {error}")
                                })?);
                        }
                        "line_end" => {
                            line_end =
                                Some(value.parse().map_err(|error| {
                                    format!("invalid line_end {value}: {error}")
                                })?);
                        }
                        _ => return Err(format!("unknown golden field: {key}")),
                    }
                }
                match (kind, scope_path, header, line_start, line_end) {
                    (
                        Some(kind),
                        Some(scope_path),
                        Some(header),
                        Some(line_start),
                        Some(line_end),
                    ) => Ok(GoldenUnit {
                        kind,
                        scope_path,
                        header,
                        line_start,
                        line_end,
                    }),
                    _ => Err("golden unit is missing a required field".to_owned()),
                }
            })
            .collect()
    }

    fn golden_fixture(language: &str) -> Result<(String, Vec<GoldenUnit>), std::io::Error> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("lang")
            .join(language);
        let input = fs::read_to_string(root.join("input")).or_else(|_| {
            let extension = match language {
                "rust" => "rs",
                "python" => "py",
                "javascript" => "js",
                "typescript" => "ts",
                "tsx" => "tsx",
                "go" => "go",
                "java" => "java",
                "php" => "php",
                _ => "",
            };
            fs::read_to_string(root.join(format!("input.{extension}")))
        })?;
        let expected = fs::read_to_string(root.join("expected.yaml"))?;
        parse_golden_units(&expected)
            .map_err(std::io::Error::other)
            .map(|units| (input, units))
    }

    fn parse_text_golden_units(text: &str) -> Result<Vec<TextGolden>, String> {
        text.trim()
            .split("\n- ")
            .map(|entry| {
                let mut kind = None;
                let mut scope_path = None;
                let mut header = None;
                let mut has_code_fence = false;
                let mut idents = Vec::new();
                for line in entry.trim_start_matches("- ").lines() {
                    let Some((key, value)) = line.trim().split_once(": ") else {
                        return Err(format!("invalid golden line: {line}"));
                    };
                    match key {
                        "kind" => {
                            kind = Some(match value {
                                "markdown" => TextUnitKind::Markdown,
                                "chunk" => TextUnitKind::Chunk,
                                "config_key" => TextUnitKind::ConfigKey,
                                _ => return Err(format!("unknown text unit kind: {value}")),
                            });
                        }
                        "scope_path" => scope_path = Some(value.to_owned()),
                        "header" => header = Some(value.to_owned()),
                        "has_code_fence" => {
                            has_code_fence = value.parse().map_err(|error| {
                                format!("invalid has_code_fence {value}: {error}")
                            })?;
                        }
                        "idents" => {
                            idents = value
                                .split(',')
                                .map(str::trim)
                                .filter(|identifier| !identifier.is_empty())
                                .map(ToOwned::to_owned)
                                .collect();
                        }
                        _ => return Err(format!("unknown text golden field: {key}")),
                    }
                }
                match (kind, scope_path, header) {
                    (Some(kind), Some(scope_path), Some(header)) => Ok(TextGolden {
                        kind,
                        scope_path,
                        header,
                        has_code_fence,
                        idents,
                    }),
                    _ => Err("text golden unit is missing a required field".to_owned()),
                }
            })
            .collect()
    }

    fn text_golden_fixture(
        input_relative: &str,
        expected_relative: &str,
    ) -> Result<(String, Vec<TextGolden>), std::io::Error> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let input = fs::read_to_string(root.join(input_relative))?;
        let expected = fs::read_to_string(root.join(expected_relative))?;
        parse_text_golden_units(&expected)
            .map_err(std::io::Error::other)
            .map(|units| (input, units))
    }

    fn fts_golden_fixture() -> Result<(String, FtsGolden), std::io::Error> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/fts");
        let input = fs::read_to_string(root.join("input.rs"))?;
        let expected = fs::read_to_string(root.join("expected.yaml"))?;
        let mut header = None;
        let mut idents = None;
        let mut body = None;
        let mut prose = None;
        for line in expected.lines() {
            let Some((key, value)) = line.split_once(": ") else {
                return Err(std::io::Error::other(format!(
                    "invalid FTS golden line: {line}"
                )));
            };
            match key {
                "header" => header = Some(value.to_owned()),
                "idents" => idents = Some(value.to_owned()),
                "body" => body = Some(value.to_owned()),
                "prose" => prose = Some(value.to_owned()),
                _ => {
                    return Err(std::io::Error::other(format!(
                        "unknown FTS golden field: {key}"
                    )));
                }
            }
        }
        match (header, idents, body, prose) {
            (Some(header), Some(idents), Some(body), Some(prose)) => Ok((
                input,
                FtsGolden {
                    columns: LexicalColumns {
                        header,
                        idents,
                        body,
                        prose,
                    },
                },
            )),
            _ => Err(std::io::Error::other(
                "FTS golden is missing a required column",
            )),
        }
    }

    fn treatment(
        name: &str,
        patterns: &[&str],
        chunker: Option<&str>,
        max_level: u64,
        enrich: &[&str],
        never_send: bool,
    ) -> Treatment {
        Treatment {
            name: name.to_owned(),
            patterns: patterns.iter().map(ToString::to_string).collect(),
            chunker: chunker.map(ToString::to_string),
            max_level,
            enrich: enrich.iter().map(ToString::to_string).collect(),
            never_send,
        }
    }

    #[test]
    fn generated_sources_stop_production_at_cap_plus_one() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        assert!(
            Command::new("python3")
                .arg(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("scripts/generate-chunk-cap-fixtures.py")
                )
                .arg(root.path())
                .status()?
                .success()
        );
        let settings = crate::config::Chunk {
            min_tokens: 2,
            target_tokens: 40,
            max_tokens: 80,
            markdown_range: [10, 30],
        };
        for (path, language, chunker) in [
            ("many.rs", Some("rust"), Chunker::Ast),
            ("long.rs", Some("rust"), Chunker::Ast),
            ("many.rs", None, Chunker::Lines),
            ("many.md", None, Chunker::Markdown),
            ("many.txt", None, Chunker::Markdown),
            ("many.json", None, Chunker::Config),
            ("many.toml", None, Chunker::Config),
            ("many.yaml", None, Chunker::Config),
            (".env", None, Chunker::Config),
        ] {
            let source = fs::read(root.path().join(path))?;
            for cap in [0, 1, 3] {
                let units = super::prepare_units(path, language, chunker, &source, &settings, cap)?;
                assert_eq!(units.len(), cap as usize + 1, "{path} cap {cap}");
                if matches!(chunker, Chunker::Ast | Chunker::Lines) {
                    assert!(
                        units.iter().all(|unit| tiktoken_rs::cl100k_base_singleton()
                            .encode_with_special_tokens(&unit.text)
                            .len()
                            <= 80),
                        "{path}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bounded_long_line_chunks_preserve_unicode_and_hard_token_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let text = "é🙂漢字 hello ".repeat(500);
        let options = crate::config::Chunk {
            min_tokens: 2,
            target_tokens: 40,
            max_tokens: 80,
            ..Default::default()
        };
        let units = super::prepare_units(
            "unknown",
            None,
            Chunker::Lines,
            text.as_bytes(),
            &options,
            1000,
        )?;
        assert_eq!(
            units.iter().map(|u| u.text.as_str()).collect::<String>(),
            text
        );
        assert!(units.iter().all(|u| {
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(&u.text)
                .len()
                <= 80
        }));
        Ok(())
    }

    #[test]
    fn first_configured_glob_wins() -> Result<(), super::Error> {
        let router = Router::new(
            &[
                treatment("first", &["src/**"], Some("lines"), 1, &[], false),
                treatment("second", &["**/*.rs"], Some("ast"), 3, &[], false),
            ],
            false,
        )?;
        let resolved = router.route("src/lib.rs", FileKind::Code);
        assert_eq!(resolved.name, "first");
        assert_eq!(resolved.chunker, Chunker::Lines);
        assert_eq!(resolved.max_level, 1);
        Ok(())
    }

    #[test]
    fn router_reuse_keeps_policy_changes_errors_and_existing_routers_independent() {
        let public = treatment(
            "public",
            &["src/**"],
            Some("ast"),
            3,
            &["describe_symbol"],
            false,
        );
        let first = Router::new(std::slice::from_ref(&public), false).unwrap();
        let mut protected = public.clone();
        protected.never_send = true;
        let protected_router = Router::new(std::slice::from_ref(&protected), false).unwrap();
        let private = protected_router.route("src/lib.rs", FileKind::Code);
        assert!(private.never_send);
        assert_eq!(private.max_level, 1);
        assert!(private.enrich.is_empty());
        assert!(!first.route("src/lib.rs", FileKind::Code).never_send);
        assert_eq!(first.route("src/lib.rs", FileKind::Code).max_level, 3);

        let mut invalid = public.clone();
        invalid.patterns = vec!["[".into()];
        assert!(Router::new(&[invalid], false).is_err());
        let mut lines = public;
        lines.name = "lines".into();
        lines.chunker = Some("lines".into());
        lines.max_level = 2;
        lines.enrich.clear();
        let changed = Router::new(&[lines, protected], false).unwrap();
        let resolved = changed.route("src/lib.rs", FileKind::Code);
        assert_eq!(resolved.name, "lines");
        assert_eq!(resolved.chunker, Chunker::Lines);
        assert_eq!(resolved.max_level, 2);
        assert!(resolved.enrich.is_empty());
        assert!(!resolved.never_send);
        assert_eq!(
            Router::new(&[], true)
                .unwrap()
                .route("image.png", FileKind::Image)
                .max_level,
            3
        );
        assert_eq!(
            Router::new(&[], false)
                .unwrap()
                .route("image.png", FileKind::Image)
                .max_level,
            0
        );
    }

    #[test]
    fn router_reuse_isolates_concurrent_configurations() {
        std::thread::scope(|scope| {
            for protected in [false, true] {
                scope.spawn(move || {
                    let configured = treatment(
                        "concurrent",
                        &["src/**"],
                        Some("ast"),
                        3,
                        &["describe_symbol"],
                        protected,
                    );
                    for _ in 0..100 {
                        let router = Router::new(std::slice::from_ref(&configured), false).unwrap();
                        let resolved = router.route("src/lib.rs", FileKind::Code);
                        assert_eq!(resolved.never_send, protected);
                        assert_eq!(resolved.max_level, if protected { 1 } else { 3 });
                        assert_eq!(resolved.enrich.is_empty(), protected);
                    }
                });
            }
        });
    }

    #[test]
    fn builtins_cover_every_file_kind_and_generated_is_first() -> Result<(), super::Error> {
        let router = Router::default();
        assert_eq!(
            router.route("src/lib.rs", FileKind::Code).chunker,
            Chunker::Ast
        );
        assert_eq!(
            router.route("README.md", FileKind::Doc).chunker,
            Chunker::Markdown
        );
        assert_eq!(
            router.route("manual.pdf", FileKind::Doc).chunker,
            Chunker::Pdf
        );
        assert_eq!(
            router.route("study.ipynb", FileKind::Doc).chunker,
            Chunker::Notebook
        );
        assert_eq!(router.route("data.toml", FileKind::Config).max_level, 1);
        assert_eq!(router.route("image.png", FileKind::Image).max_level, 0);
        assert_eq!(router.route("data.csv", FileKind::Data).max_level, 0);

        let configured = Config::default();
        let configured_router = Router::new(&configured.treatments, false)?;
        let generated = configured_router.route("src/vendor/library.rs", FileKind::Code);
        assert_eq!(generated.name, "generated");
        assert_eq!(generated.chunker, Chunker::None);
        assert_eq!(generated.max_level, 0);

        let with_vlm = Router::new(&[], true)?;
        let image = with_vlm.route("image.png", FileKind::Image);
        assert_eq!(image.max_level, 3);
        assert_eq!(image.enrich, vec![Enrichment::DescribeImage]);
        Ok(())
    }

    #[test]
    fn never_send_caps_the_treatment_at_l1_and_removes_enrichment() -> Result<(), super::Error> {
        let router = Router::new(
            &[treatment(
                "secret",
                &["**/*.key"],
                Some("ast"),
                3,
                &["describe_symbol"],
                true,
            )],
            false,
        )?;
        let resolved = router.route("config/private.key", FileKind::Code);
        assert_eq!(resolved.max_level, 1);
        assert!(resolved.enrich.is_empty());
        assert!(resolved.never_send);
        Ok(())
    }

    #[test]
    fn ast_chunker_is_deterministic_for_every_phase_zero_grammar() -> Result<(), AstError> {
        let cases = [
            (AstLanguage::Rust, "pub fn hello() {}\n"),
            (AstLanguage::Python, "def hello():\n    pass\n"),
            (AstLanguage::JavaScript, "function hello() {}\n"),
            (AstLanguage::TypeScript, "function hello(): void {}\n"),
            (AstLanguage::Tsx, "function Hello() { return <div />; }\n"),
            (AstLanguage::Go, "func Hello() {}\n"),
            (AstLanguage::Java, "class Hello {}\n"),
            (AstLanguage::Php, "<?php function hello() {}\n"),
        ];
        for (language, source) in cases {
            let first = chunk_ast(language, source.as_bytes())?;
            let second = chunk_ast(language, source.as_bytes())?;
            assert_eq!(first, second, "non-deterministic {language:?} chunks");
            assert_eq!(first.len(), 1, "missing {language:?} definition");
            assert_eq!(first[0].kind, AstUnitKind::Symbol);
            assert_eq!(first[0].scope_path.to_ascii_lowercase(), "hello");
            assert_eq!(first[0].line_start, 1);
        }
        assert_eq!(CHUNKER_VERSION, "1");
        Ok(())
    }

    #[test]
    fn prose_ranges_keep_multibyte_last_characters_on_their_source_line() {
        for source in ["first\nλ", "first\nλ\n", "first\r\nλ\r\n"] {
            for units in [
                chunk_plain(source.as_bytes()),
                chunk_markdown(source.as_bytes()),
            ] {
                assert_eq!(units.len(), 1);
                assert_eq!((units[0].line_start, units[0].line_end), (1, 2));
            }
        }
    }

    #[test]
    fn ast_template_ranges_include_last_covered_line_without_eof_overrun() -> Result<(), AstError> {
        for newline in ["\n", "\r\n"] {
            for terminated in [true, false] {
                let mut source = ["<x-input>", "    unicode λ", "</x-input>"].join(newline);
                if terminated {
                    source.push_str(newline);
                }
                let units = chunk_ast(AstLanguage::Php, source.as_bytes())?;
                assert_eq!(units.len(), 1);
                assert_eq!((units[0].line_start, units[0].line_end), (1, 3));
                assert_eq!(units[0].text, source);
            }
        }
        Ok(())
    }

    #[test]
    fn ast_chunker_falls_back_for_invalid_utf8_and_skips_empty_files() -> Result<(), AstError> {
        assert!(chunk_ast(AstLanguage::Rust, b"")?.is_empty());
        let fallback = chunk_ast(AstLanguage::Rust, b"first\n\nsecond\xff")?;
        assert_eq!(fallback.len(), 1);
        assert!(fallback.iter().all(|unit| unit.kind == AstUnitKind::Chunk));
        assert_eq!(
            chunk_code(None, b"first\n\nsecond")?.len(),
            1,
            "line fallback merges small blank-line-separated blocks toward its target"
        );
        Ok(())
    }

    #[test]
    fn every_static_grammar_exposes_a_valid_tags_query() -> Result<(), AstError> {
        let languages = [
            AstLanguage::Rust,
            AstLanguage::Python,
            AstLanguage::JavaScript,
            AstLanguage::TypeScript,
            AstLanguage::Tsx,
            AstLanguage::Go,
            AstLanguage::Java,
            AstLanguage::Php,
        ];
        for language in languages {
            Query::new(&language.grammar(), language.tags_query())
                .map_err(|error| AstError::TagsQuery(error.to_string()))?;
        }
        Ok(())
    }

    #[test]
    fn ast_chunker_matches_the_eight_language_goldens() -> Result<(), Box<dyn std::error::Error>> {
        let fixtures = [
            (AstLanguage::Rust, "rust"),
            (AstLanguage::Python, "python"),
            (AstLanguage::JavaScript, "javascript"),
            (AstLanguage::TypeScript, "typescript"),
            (AstLanguage::Tsx, "tsx"),
            (AstLanguage::Go, "go"),
            (AstLanguage::Java, "java"),
            (AstLanguage::Php, "php"),
        ];
        for (language, fixture) in fixtures {
            let (input, expected) = golden_fixture(fixture)?;
            let actual = chunk_ast(language, input.as_bytes())?
                .into_iter()
                .map(|unit| GoldenUnit {
                    kind: unit.kind,
                    scope_path: unit.scope_path,
                    header: unit.header,
                    line_start: unit.line_start,
                    line_end: unit.line_end,
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{fixture} golden changed");
        }
        Ok(())
    }

    #[test]
    fn ast_chunker_keeps_file_imports_and_preceding_docs_structural() -> Result<(), AstError> {
        let source =
            "use crate::one;\nuse crate::two;\n\n/// Greets the caller.\npub fn hello() {}\n";
        let units = chunk_ast(AstLanguage::Rust, source.as_bytes())?;
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].kind, AstUnitKind::Chunk);
        assert_eq!(units[0].scope_path, "imports");
        assert_eq!(units[0].header, "imports › imports");
        assert_eq!(units[0].text, "use crate::one;\nuse crate::two;");
        assert_eq!(units[1].kind, AstUnitKind::Symbol);
        assert_eq!(units[1].scope_path, "hello");
        assert!(units[1].text.starts_with("/// Greets the caller."));
        assert_eq!(units[1].line_start, 4);
        Ok(())
    }

    #[test]
    fn ast_chunker_splits_oversized_leaves_at_statement_boundaries() -> Result<(), AstError> {
        let source = format!("def count():\n{}", "    total += 1\n".repeat(900));
        let units = chunk_ast(AstLanguage::Python, source.as_bytes())?;
        assert!(units.len() > 1);
        assert!(units.iter().all(|unit| unit.scope_path == "count"));
        assert!(
            units
                .iter()
                .all(|unit| super::ast::token_count(&unit.text) <= super::ast::MAX_TOKENS)
        );
        assert!(units.iter().all(|unit| unit.kind == AstUnitKind::Chunk));
        Ok(())
    }

    #[test]
    fn ast_chunker_recurses_into_oversized_class_members() -> Result<(), AstError> {
        let members = (0..500)
            .map(|index| format!("  void method{index}() {{}}\n"))
            .collect::<String>();
        let source = format!("class Service {{\n{members}}}\n");
        let units = chunk_ast(AstLanguage::Java, source.as_bytes())?;
        assert!(units.iter().any(|unit| {
            unit.kind == AstUnitKind::Symbol
                && unit.scope_path == "Service"
                && unit.header == "Service › class Service {"
        }));
        assert!(units.iter().any(|unit| {
            unit.kind == AstUnitKind::Symbol
                && unit.scope_path == "Service › method0"
                && unit.header == "Service › method0 › void method0() {}"
        }));
        assert!(
            units
                .iter()
                .all(|unit| super::ast::token_count(&unit.text) <= super::ast::MAX_TOKENS)
        );
        Ok(())
    }

    #[test]
    fn hostile_ten_thousand_line_input_remains_deterministically_bounded() -> Result<(), AstError> {
        let source = "value += 1\n".repeat(10_000);
        let first = chunk_ast(AstLanguage::Python, source.as_bytes())?;
        let second = chunk_ast(AstLanguage::Python, source.as_bytes())?;
        assert_eq!(first, second);
        assert!(
            first
                .iter()
                .all(|unit| super::ast::token_count(&unit.text) <= super::ast::MAX_TOKENS)
        );
        Ok(())
    }

    #[test]
    fn markdown_chunker_matches_nested_heading_golden() -> Result<(), Box<dyn std::error::Error>> {
        let (input, expected) = text_golden_fixture("docs/README.md", "docs/README.expected.yaml")?;
        let actual = chunk_markdown(input.as_bytes())
            .into_iter()
            .map(|unit| TextGolden {
                kind: unit.kind,
                scope_path: unit.scope_path,
                header: unit.header,
                has_code_fence: unit.has_code_fence,
                idents: unit.idents,
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn plain_text_chunker_matches_rst_golden() -> Result<(), Box<dyn std::error::Error>> {
        let (input, expected) = text_golden_fixture("docs/guide.rst", "docs/guide.expected.yaml")?;
        let actual = chunk_plain(input.as_bytes())
            .into_iter()
            .map(|unit| TextGolden {
                kind: unit.kind,
                scope_path: unit.scope_path,
                header: unit.header,
                has_code_fence: unit.has_code_fence,
                idents: unit.idents,
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn config_chunker_matches_env_and_toml_goldens() -> Result<(), Box<dyn std::error::Error>> {
        for (format, input_path, expected_path) in [
            (ConfigFormat::Env, "config/.env", "config/env.expected.yaml"),
            (
                ConfigFormat::Toml,
                "config/Cargo.toml",
                "config/Cargo.expected.yaml",
            ),
        ] {
            let (input, expected) = text_golden_fixture(input_path, expected_path)?;
            let actual = chunk_config(format, input.as_bytes())
                .into_iter()
                .map(|unit| TextGolden {
                    kind: unit.kind,
                    scope_path: unit.scope_path,
                    header: unit.header,
                    has_code_fence: unit.has_code_fence,
                    idents: unit.idents,
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{input_path} golden changed");
        }
        Ok(())
    }

    #[test]
    fn config_chunker_handles_json_yaml_and_oversized_nested_values() {
        let json = format!(r#"{{"service":{{"notes":"{}"}}}}"#, "word ".repeat(400));
        let json_units = chunk_config(ConfigFormat::Json, json.as_bytes());
        assert!(json_units.len() > 1);
        assert!(
            json_units
                .iter()
                .all(|unit| unit.scope_path == "service.notes")
        );
        assert!(json_units[0].idents.contains(&"notes".to_owned()));

        let yaml = "server:\n  host: localhost\n  port: 3000\n";
        let yaml_units = chunk_config(ConfigFormat::Yaml, yaml.as_bytes());
        assert_eq!(yaml_units.len(), 1);
        assert_eq!(yaml_units[0].scope_path, "server");
        assert_eq!(
            yaml_units[0].idents,
            vec!["host".to_owned(), "port".to_owned(), "server".to_owned()]
        );
    }

    #[test]
    fn yaml_sequence_entries_use_stable_structural_indices() {
        let description = "word ".repeat(180);
        let source = format!(
            "body:\n  - type: textarea\n    description: {description}\n  - type: textarea\n    description: {description}\n"
        );
        let paths = chunk_config(ConfigFormat::Yaml, source.as_bytes())
            .into_iter()
            .map(|unit| unit.scope_path)
            .collect::<Vec<_>>();
        assert_eq!(paths, vec!["body[0]", "body[1]"]);
    }

    #[test]
    fn generated_five_megabyte_json_is_capped_and_marked_truncated()
    -> Result<(), Box<dyn std::error::Error>> {
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("scripts")
            .join("fixtures")
            .join("make-large-json.sh");
        let output = Command::new("sh")
            .arg(script)
            .arg((5 * 1024 * 1024).to_string())
            .output()?;
        if !output.status.success() {
            return Err(std::io::Error::other("large JSON fixture generator failed").into());
        }
        let units = chunk_config(ConfigFormat::Json, &output.stdout);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].text.len(), 1024 * 1024);
        assert!(units[0].truncated);
        Ok(())
    }

    #[test]
    fn fts_builder_matches_the_tokenization_golden() -> Result<(), Box<dyn std::error::Error>> {
        let (input, expected) = fts_golden_fixture()?;
        let config = Config::default();
        let actual = build_columns(
            FtsInput {
                breadcrumb: "deepkit-code › src/auth/retry-policy.rs",
                header: "RetryPolicy › getUserById",
                text: &input,
                language: Some("rust"),
                kind: FileKind::Code,
                extra_idents: &[],
                has_code_fence: false,
            },
            &config.tokenize,
        );
        assert_eq!(actual, expected.columns);
        Ok(())
    }

    #[test]
    fn fts_match_expands_identifier_forms_and_preserves_phrases() {
        assert_eq!(
            build_match("getUserById"),
            "(\"getUserById\" OR (\"get\" AND \"user\" AND \"by\" AND \"id\"))"
        );
        assert_eq!(
            build_match("get_user_by_id"),
            "(\"get_user_by_id\" OR (\"get\" AND \"user\" AND \"by\" AND \"id\"))"
        );
        assert_eq!(
            build_match("get user by id"),
            "\"get\" AND \"user\" AND \"by\" AND \"id\""
        );
        assert_eq!(build_match("\"retry policy\""), "\"retry policy\"");
    }

    #[test]
    fn fts_builder_preserves_unterminated_block_comments_and_utf8_boundaries() {
        let config = Config::default();
        for (text, prose, body) in [
            ("/*", "", ""),
            ("/*a", "a", ""),
            ("/*é", "é", ""),
            ("/*éx", "éx", ""),
            ("/*漢字", "漢字", ""),
            ("/* unfinished prose", "unfinished prose", ""),
            ("/**/", "", ""),
            ("/*é*/ let after = 1;", "é", "let after 1"),
        ] {
            let columns = build_columns(
                FtsInput {
                    breadcrumb: "repository › comment.rs",
                    header: "comment",
                    text,
                    language: Some("rust"),
                    kind: FileKind::Code,
                    extra_idents: &[],
                    has_code_fence: false,
                },
                &config.tokenize,
            );
            assert_eq!(columns.prose, prose, "{text:?}");
            assert_eq!(columns.body, body, "{text:?}");
        }
    }

    #[test]
    fn fts_builder_handles_unterminated_string_literals() {
        let config = Config::default();
        let unterminated = build_columns(
            FtsInput {
                breadcrumb: "repository › src › example.php",
                header: "example",
                text: "\"unterminated",
                language: Some("php"),
                kind: FileKind::Code,
                extra_idents: &[],
                has_code_fence: false,
            },
            &config.tokenize,
        );
        assert_eq!(unterminated.prose, "unterminated");

        let triple = build_columns(
            FtsInput {
                breadcrumb: "repository › src › example.py",
                header: "example",
                text: "\"\"\"",
                language: Some("python"),
                kind: FileKind::Code,
                extra_idents: &[],
                has_code_fence: false,
            },
            &config.tokenize,
        );
        assert!(triple.prose.is_empty());
    }

    #[test]
    fn fts_builder_feeds_markdown_code_fences_to_the_code_column() {
        let config = Config::default();
        let columns = build_columns(
            FtsInput {
                breadcrumb: "deepkit-code › docs › guide.md",
                header: "Guide",
                text: "# Guide\n\n```rust\nlet retryPolicy: usize = 3;\n```\n",
                language: Some("rust"),
                kind: FileKind::Doc,
                extra_idents: &[],
                has_code_fence: true,
            },
            &config.tokenize,
        );
        assert_eq!(columns.body, "let retry policy usize 3");
        assert!(
            columns
                .idents
                .split_whitespace()
                .any(|item| item == "retryPolicy")
        );
        assert!(columns.prose.contains("retryPolicy"));
    }
}

#[cfg(test)]
mod grammar_compatibility;

#[cfg(test)]
mod grammar_tests;
