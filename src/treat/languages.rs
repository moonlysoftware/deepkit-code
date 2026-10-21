//! Statically linked grammar registry and conservative dialect selection; v1.md §6.1.
use std::path::Path;
use tree_sitter::Language;

/// Changes when grammar/query behavior requires re-parsing unchanged files.
/// Kept separate from content hashing so unchanged partitions retain their payloads.
pub(crate) const VERSION: &str = "t2-10-2";

/// Grammars share this ordering with the immutable tags-query cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum AstLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    Php,
    Kotlin,
    C,
    Cpp,
    CSharp,
    Ruby,
    Swift,
    Scala,
    Bash,
    Html,
    Css,
    Sql,
    Dart,
    Elixir,
    Lua,
    Zig,
    Haskell,
    Ocaml,
    OcamlInterface,
    Protobuf,
    Json,
    Yaml,
    Toml,
    Markdown,
    MarkdownInline,
    Dockerfile,
}
impl AstLanguage {
    /// Complete registry, including separate TSX, OCaml-interface and Markdown-inline grammars.
    pub(super) const ALL: &[Self] = &[
        Self::Rust,
        Self::Python,
        Self::JavaScript,
        Self::TypeScript,
        Self::Tsx,
        Self::Go,
        Self::Java,
        Self::Php,
        Self::Kotlin,
        Self::C,
        Self::Cpp,
        Self::CSharp,
        Self::Ruby,
        Self::Swift,
        Self::Scala,
        Self::Bash,
        Self::Html,
        Self::Css,
        Self::Sql,
        Self::Dart,
        Self::Elixir,
        Self::Lua,
        Self::Zig,
        Self::Haskell,
        Self::Ocaml,
        Self::OcamlInterface,
        Self::Protobuf,
        Self::Json,
        Self::Yaml,
        Self::Toml,
        Self::Markdown,
        Self::MarkdownInline,
        Self::Dockerfile,
    ];
    /// Resolves canonical repository or notebook language identifiers.
    pub(super) fn from_language_id(language: &str) -> Option<Self> {
        match language {
            "rust" => Some(Self::Rust),
            "python" => Some(Self::Python),
            "javascript" => Some(Self::JavaScript),
            "typescript" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "php" => Some(Self::Php),
            "kotlin" => Some(Self::Kotlin),
            "c" => Some(Self::C),
            "cpp" => Some(Self::Cpp),
            "csharp" => Some(Self::CSharp),
            "ruby" => Some(Self::Ruby),
            "swift" => Some(Self::Swift),
            "scala" => Some(Self::Scala),
            "bash" => Some(Self::Bash),
            "html" => Some(Self::Html),
            "css" => Some(Self::Css),
            "sql" => Some(Self::Sql),
            "dart" => Some(Self::Dart),
            "elixir" => Some(Self::Elixir),
            "lua" => Some(Self::Lua),
            "zig" => Some(Self::Zig),
            "haskell" => Some(Self::Haskell),
            "ocaml" => Some(Self::Ocaml),
            "ocaml-interface" => Some(Self::OcamlInterface),
            "protobuf" => Some(Self::Protobuf),
            "json" => Some(Self::Json),
            "yaml" => Some(Self::Yaml),
            "toml" => Some(Self::Toml),
            "markdown" => Some(Self::Markdown),
            "markdown-inline" => Some(Self::MarkdownInline),
            "dockerfile" => Some(Self::Dockerfile),
            _ => None,
        }
    }
    /// Selects a matching dialect or retains bounded lines when none is linked.
    pub(super) fn for_path(language: &str, path: &str) -> Option<Self> {
        let extension = Path::new(path)
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if matches!(
            (language, extension.as_str()),
            ("css", "scss" | "sass") | ("haskell", "lhs")
        ) {
            return None;
        }
        if language == "ocaml" && extension == "mli" {
            return Some(Self::OcamlInterface);
        }
        Self::from_language_id(language)
    }
    /// Config and document grammars never create code graph rows or edges.
    pub(super) fn is_code(self) -> bool {
        !matches!(
            self,
            Self::Json | Self::Yaml | Self::Toml | Self::Markdown | Self::MarkdownInline
        )
    }
    /// Existing cAST behavior remains protected by its eight permanent goldens.
    pub(super) fn uses_tag_candidates(self) -> bool {
        !matches!(
            self,
            Self::Rust
                | Self::Python
                | Self::JavaScript
                | Self::TypeScript
                | Self::Tsx
                | Self::Go
                | Self::Java
                | Self::Php
        )
    }
    /// Returns the pinned statically linked parser for this registry entry.
    pub(super) fn grammar(self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::Php => tree_sitter_php::LANGUAGE_PHP.into(),
            Self::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Self::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Self::Swift => tree_sitter_swift::LANGUAGE.into(),
            Self::Scala => tree_sitter_scala::LANGUAGE.into(),
            Self::Bash => tree_sitter_bash::LANGUAGE.into(),
            Self::Html => tree_sitter_html::LANGUAGE.into(),
            Self::Css => tree_sitter_css::LANGUAGE.into(),
            Self::Sql => tree_sitter_sequel::LANGUAGE.into(),
            Self::Dart => tree_sitter_dart::LANGUAGE.into(),
            Self::Elixir => tree_sitter_elixir::LANGUAGE.into(),
            Self::Lua => tree_sitter_lua::LANGUAGE.into(),
            Self::Zig => tree_sitter_zig::LANGUAGE.into(),
            Self::Haskell => tree_sitter_haskell::LANGUAGE.into(),
            Self::Ocaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
            Self::OcamlInterface => tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into(),
            Self::Protobuf => tree_sitter_proto::LANGUAGE.into(),
            Self::Json => tree_sitter_json::LANGUAGE.into(),
            Self::Yaml => tree_sitter_yaml::LANGUAGE.into(),
            Self::Toml => tree_sitter_toml_ng::LANGUAGE.into(),
            Self::Markdown => tree_sitter_md::LANGUAGE.into(),
            Self::MarkdownInline => tree_sitter_md::INLINE_LANGUAGE.into(),
            Self::Dockerfile => tree_sitter_dockerfile_updated::language(),
        }
    }
    /// Returns the upstream or attributed local tags query paired with this grammar.
    pub(super) fn tags_query(self) -> &'static str {
        match self {
            Self::Rust => tree_sitter_rust::TAGS_QUERY,
            Self::Python => tree_sitter_python::TAGS_QUERY,
            Self::JavaScript => tree_sitter_javascript::TAGS_QUERY,
            Self::TypeScript => tree_sitter_typescript::TAGS_QUERY,
            Self::Tsx => tree_sitter_typescript::TAGS_QUERY,
            Self::Go => tree_sitter_go::TAGS_QUERY,
            Self::Java => tree_sitter_java::TAGS_QUERY,
            Self::Php => tree_sitter_php::TAGS_QUERY,
            Self::Kotlin => include_str!("grammar_queries/kotlin.scm"),
            Self::C => include_str!("grammar_queries/c.scm"),
            Self::Cpp => include_str!("grammar_queries/cpp.scm"),
            Self::CSharp => include_str!("grammar_queries/csharp.scm"),
            Self::Ruby => tree_sitter_ruby::TAGS_QUERY,
            Self::Swift => tree_sitter_swift::TAGS_QUERY,
            Self::Scala => include_str!("grammar_queries/scala.scm"),
            Self::Bash => include_str!("grammar_queries/bash.scm"),
            Self::Html => include_str!("grammar_queries/html.scm"),
            Self::Css => include_str!("grammar_queries/css.scm"),
            Self::Sql => include_str!("grammar_queries/sql.scm"),
            Self::Dart => tree_sitter_dart::TAGS_QUERY,
            Self::Elixir => tree_sitter_elixir::TAGS_QUERY,
            Self::Lua => tree_sitter_lua::TAGS_QUERY,
            Self::Zig => include_str!("grammar_queries/zig.scm"),
            Self::Haskell => include_str!("grammar_queries/haskell.scm"),
            Self::Ocaml => tree_sitter_ocaml::TAGS_QUERY,
            Self::OcamlInterface => include_str!("grammar_queries/ocaml-interface.scm"),
            Self::Protobuf => include_str!("grammar_queries/protobuf.scm"),
            Self::Json => include_str!("grammar_queries/json.scm"),
            Self::Yaml => include_str!("grammar_queries/yaml.scm"),
            Self::Toml => include_str!("grammar_queries/toml.scm"),
            Self::Markdown => include_str!("grammar_queries/markdown.scm"),
            Self::MarkdownInline => include_str!("grammar_queries/markdown-inline.scm"),
            Self::Dockerfile => include_str!("grammar_queries/dockerfile.scm"),
        }
    }
}
