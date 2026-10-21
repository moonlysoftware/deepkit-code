//! Definition captures on an already parsed cAST tree; v1.md §6.1.
use std::collections::BTreeMap;
use tree_sitter::{Node, QueryCursor, StreamingIterator};

use super::{ast::AstError, graph::tags_configuration, languages::AstLanguage};

struct Definition {
    name: String,
    container: bool,
}

/// Captured definitions select units without guessing from generic node names.
pub(super) struct Syntax {
    language: AstLanguage,
    definitions: BTreeMap<(usize, usize), Definition>,
}

impl Syntax {
    /// Reuses the immutable tags query against the chunker's existing parse tree.
    pub(super) fn new(
        language: AstLanguage,
        root: Node<'_>,
        source: &str,
    ) -> Result<Self, AstError> {
        let mut result = Self {
            language,
            definitions: BTreeMap::new(),
        };
        if !language.uses_tag_candidates() {
            return Ok(result);
        }
        let config =
            tags_configuration(language).map_err(|error| AstError::TagsQuery(error.to_string()))?;
        let names = config.query.capture_names();
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&config.query, root, source.as_bytes());
        while let Some(found) = matches.next() {
            let Some(name) = found
                .captures()
                .iter()
                .find(|capture| names[capture.index as usize] == "name")
            else {
                continue;
            };
            for capture in found.captures() {
                let Some(kind) = names[capture.index as usize].strip_prefix("definition.") else {
                    continue;
                };
                let name = source
                    .get(name.node.byte_range())
                    .ok_or(AstError::InvalidRange)?
                    .to_owned();
                let node = if matches!(language, AstLanguage::Ocaml)
                    && capture.node.kind() == "let_binding"
                {
                    capture
                        .node
                        .parent()
                        .filter(|parent| parent.kind() == "value_definition")
                        .unwrap_or(capture.node)
                } else {
                    capture.node
                };
                result
                    .definitions
                    .entry((node.start_byte(), node.end_byte()))
                    .or_insert(Definition {
                        name,
                        container: matches!(
                            kind,
                            "class"
                                | "struct"
                                | "enum"
                                | "interface"
                                | "module"
                                | "impl"
                                | "object"
                        ),
                    });
            }
        }
        if cursor.did_exceed_match_limit() {
            return Err(AstError::TagsQuery(
                "definition query exceeded its match limit".into(),
            ));
        }
        Ok(result)
    }

    /// The query's name capture, rather than a guessed name-field descendant.
    pub(super) fn name(&self, node: Node<'_>) -> Option<&str> {
        self.definitions
            .get(&(node.start_byte(), node.end_byte()))
            .map(|definition| definition.name.as_str())
    }

    /// Query-declared containers can split into children while retaining a shell.
    pub(super) fn is_container(&self, node: Node<'_>) -> bool {
        self.definitions
            .get(&(node.start_byte(), node.end_byte()))
            .is_some_and(|definition| definition.container)
    }

    /// Traverses grammar wrappers to expose captured definitions and imports.
    /// An explicit stack avoids recursive traversal through hostile wrapper depth.
    pub(super) fn children<'tree>(&self, node: Node<'tree>) -> Vec<Node<'tree>> {
        let mut cursor = node.walk();
        let children: Vec<_> = node.named_children(&mut cursor).collect();
        if !self.language.uses_tag_candidates() {
            return children;
        }
        let mut pending: Vec<_> = children.into_iter().rev().collect();
        let mut result = Vec::new();
        while let Some(child) = pending.pop() {
            let has_definition = self
                .definitions
                .range((child.start_byte(), 0)..(child.end_byte(), 0))
                .any(|(&(start, end), _)| start < child.end_byte() && end <= child.end_byte());
            let wrapper = matches!(self.language, AstLanguage::Haskell)
                && matches!(child.kind(), "declarations" | "imports");
            if self.name(child).is_some() || (!has_definition && !wrapper) {
                result.push(child);
            } else {
                let mut cursor = child.walk();
                let children: Vec<_> = child.named_children(&mut cursor).collect();
                pending.extend(children.into_iter().rev());
            }
        }
        result
    }
}
