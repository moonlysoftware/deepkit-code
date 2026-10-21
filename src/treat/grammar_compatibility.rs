//! Isolated T2.10 ABI/query preparation; not production grammar acceptance (v1 §6.1).
use tree_sitter::{Language, Parser};
#[test]
fn candidate_grammars_parse_and_exported_tags_compile() {
    let cases: Vec<(&str, Language, &str, Option<&str>)> = vec![
        (
            "kotlin",
            tree_sitter_kotlin_ng::LANGUAGE.into(),
            "fun shouldRetry(value: Int): Boolean = value > 0\n",
            Some(include_str!("grammar_queries/kotlin.scm")),
        ),
        (
            "c",
            tree_sitter_c::LANGUAGE.into(),
            "int should_retry(int value) { return value > 0; }\n",
            Some(include_str!("grammar_queries/c.scm")),
        ),
        (
            "cpp",
            tree_sitter_cpp::LANGUAGE.into(),
            "bool should_retry(int value) { return value > 0; }\n",
            Some(include_str!("grammar_queries/cpp.scm")),
        ),
        (
            "csharp",
            tree_sitter_c_sharp::LANGUAGE.into(),
            "class Retry { public bool ShouldRetry(int value) { return value > 0; } }\n",
            Some(include_str!("grammar_queries/csharp.scm")),
        ),
        (
            "ruby",
            tree_sitter_ruby::LANGUAGE.into(),
            "def should_retry(value)\n  value > 0\nend\n",
            Some(tree_sitter_ruby::TAGS_QUERY),
        ),
        (
            "swift",
            tree_sitter_swift::LANGUAGE.into(),
            "func shouldRetry(_ value: Int) -> Bool { return value > 0 }\n",
            Some(tree_sitter_swift::TAGS_QUERY),
        ),
        (
            "scala",
            tree_sitter_scala::LANGUAGE.into(),
            "object Retry { def shouldRetry(value: Int): Boolean = value > 0 }\n",
            Some(include_str!("grammar_queries/scala.scm")),
        ),
        (
            "bash",
            tree_sitter_bash::LANGUAGE.into(),
            "should_retry() { test \"$1\" -gt 0; }\n",
            Some(include_str!("grammar_queries/bash.scm")),
        ),
        (
            "html",
            tree_sitter_html::LANGUAGE.into(),
            "<section id=\"retry\"><p>Retry guide</p></section>\n",
            Some(include_str!("grammar_queries/html.scm")),
        ),
        (
            "css",
            tree_sitter_css::LANGUAGE.into(),
            ".retry { color: red; }\n",
            Some(include_str!("grammar_queries/css.scm")),
        ),
        (
            "sql",
            tree_sitter_sequel::LANGUAGE.into(),
            "CREATE TABLE retry (id INTEGER PRIMARY KEY);\n",
            Some(include_str!("grammar_queries/sql.scm")),
        ),
        (
            "dart",
            tree_sitter_dart::LANGUAGE.into(),
            "bool shouldRetry(int value) { return value > 0; }\n",
            Some(tree_sitter_dart::TAGS_QUERY),
        ),
        (
            "elixir",
            tree_sitter_elixir::LANGUAGE.into(),
            "defmodule Retry do\n  def should_retry(value), do: value > 0\nend\n",
            Some(tree_sitter_elixir::TAGS_QUERY),
        ),
        (
            "lua",
            tree_sitter_lua::LANGUAGE.into(),
            "function should_retry(value)\n  return value > 0\nend\n",
            Some(tree_sitter_lua::TAGS_QUERY),
        ),
        (
            "zig",
            tree_sitter_zig::LANGUAGE.into(),
            "pub fn shouldRetry(value: i32) bool { return value > 0; }\n",
            Some(include_str!("grammar_queries/zig.scm")),
        ),
        (
            "haskell",
            tree_sitter_haskell::LANGUAGE.into(),
            "shouldRetry value = value > 0\n",
            Some(include_str!("grammar_queries/haskell.scm")),
        ),
        (
            "ocaml",
            tree_sitter_ocaml::LANGUAGE_OCAML.into(),
            "let should_retry value = value > 0\n",
            Some(tree_sitter_ocaml::TAGS_QUERY),
        ),
        (
            "ocaml-interface",
            tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into(),
            "val should_retry : int -> bool\n",
            Some(include_str!("grammar_queries/ocaml-interface.scm")),
        ),
        (
            "protobuf",
            tree_sitter_proto::LANGUAGE.into(),
            "syntax = \"proto3\";\nmessage Retry { int32 count = 1; }\n",
            Some(include_str!("grammar_queries/protobuf.scm")),
        ),
        (
            "json",
            tree_sitter_json::LANGUAGE.into(),
            "{\"retry\": 3}\n",
            Some(include_str!("grammar_queries/json.scm")),
        ),
        (
            "yaml",
            tree_sitter_yaml::LANGUAGE.into(),
            "retry: 3\n",
            Some(include_str!("grammar_queries/yaml.scm")),
        ),
        (
            "toml",
            tree_sitter_toml_ng::LANGUAGE.into(),
            "retry = 3\n",
            Some(include_str!("grammar_queries/toml.scm")),
        ),
        (
            "markdown",
            tree_sitter_md::LANGUAGE.into(),
            "# Retry\n\nUse `should_retry`.\n",
            Some(include_str!("grammar_queries/markdown.scm")),
        ),
        (
            "markdown-inline",
            tree_sitter_md::INLINE_LANGUAGE.into(),
            "Use `should_retry` and [guide](guide.md).\n",
            Some(include_str!("grammar_queries/markdown-inline.scm")),
        ),
        (
            "dockerfile",
            tree_sitter_dockerfile_updated::language(),
            "FROM alpine:3.20 AS base\nARG RETRY=3\nRUN echo \"$RETRY\"\n",
            Some(include_str!("grammar_queries/dockerfile.scm")),
        ),
    ];
    let mut failures = Vec::new();
    let mut parse_failures = 0;
    for (name, grammar, source, tags) in cases {
        let mut parser = Parser::new();
        if let Err(error) = parser.set_language(&grammar) {
            failures.push(format!("{name}: ABI {error}"));
            continue;
        }
        let tree = parser.parse(source, None).unwrap();
        if tree.root_node().has_error() {
            parse_failures += 1;
            failures.push(format!("{name}: parse {}", tree.root_node().to_sexp()));
        }
        if let Some(query) = tags {
            match tree_sitter_tags::TagsConfiguration::new(grammar, query, "") {
                Err(error) => failures.push(format!("{name}: query {error}")),
                Ok(config) => {
                    let mut context = tree_sitter_tags::TagsContext::new();
                    let (tags, _) = context
                        .generate_tags(&config, source.as_bytes(), None)
                        .unwrap();
                    let names = tags
                        .map(|tag| {
                            let tag = tag.unwrap();
                            source[tag.name_range].to_owned()
                        })
                        .collect::<Vec<_>>();
                    let expected = match name {
                        "kotlin" | "swift" | "dart" | "zig" | "scala" | "haskell" => "shouldRetry",
                        "csharp" => "ShouldRetry",
                        "html" => "section",
                        "css" | "sql" | "json" | "yaml" | "toml" => "retry",
                        "protobuf" | "markdown" => "Retry",
                        "markdown-inline" => "`should_retry`",
                        "dockerfile" => "RETRY",
                        _ => "should_retry",
                    };
                    if !names.iter().any(|value| value == expected) {
                        failures.push(format!("{name}: missing {expected} in {names:?}"));
                    }
                    eprintln!("candidate={name} tag_names={names:?}");
                }
            }
        }
        eprintln!(
            "candidate={name} parse_error={}",
            tree.root_node().has_error()
        );
    }
    eprintln!("parse_failures={parse_failures}");
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn c_definition_tags_cover_body_hunks() {
    let source = "int should_retry(int value) {\n    return value > 0;\n}\n";
    let body_offset = source.find("return").unwrap();
    for (query, expected_body) in [
        (tree_sitter_c::TAGS_QUERY, false),
        (include_str!("grammar_queries/c.scm"), true),
    ] {
        let config =
            tree_sitter_tags::TagsConfiguration::new(tree_sitter_c::LANGUAGE.into(), query, "")
                .unwrap();
        let mut context = tree_sitter_tags::TagsContext::new();
        let (tags, parse_error) = context
            .generate_tags(&config, source.as_bytes(), None)
            .unwrap();
        assert!(!parse_error);
        let definitions = tags
            .map(Result::unwrap)
            .filter(|tag| tag.is_definition)
            .collect::<Vec<_>>();
        let definition = definitions
            .iter()
            .find(|tag| &source[tag.name_range.clone()] == "should_retry")
            .unwrap();
        assert_eq!(definition.range.contains(&body_offset), expected_body);
        if expected_body {
            assert_eq!(&source[definition.range.clone()], source.trim_end());
        }
    }
}

#[test]
fn cpp_definition_tags_cover_function_and_qualified_method_bodies() {
    for source in [
        "bool should_retry(int value) {\n    return value > 0;\n}\n",
        "bool Retry::should_retry(int value) {\n    return value > 0;\n}\n",
    ] {
        let body_offset = source.find("return").unwrap();
        for (query, expected_body) in [
            (tree_sitter_cpp::TAGS_QUERY, false),
            (include_str!("grammar_queries/cpp.scm"), true),
        ] {
            let config = tree_sitter_tags::TagsConfiguration::new(
                tree_sitter_cpp::LANGUAGE.into(),
                query,
                "",
            )
            .unwrap();
            let mut context = tree_sitter_tags::TagsContext::new();
            let (tags, parse_error) = context
                .generate_tags(&config, source.as_bytes(), None)
                .unwrap();
            assert!(!parse_error);
            let definitions = tags
                .map(Result::unwrap)
                .filter(|tag| tag.is_definition)
                .collect::<Vec<_>>();
            let definition = definitions
                .iter()
                .find(|tag| &source[tag.name_range.clone()] == "should_retry")
                .unwrap();
            assert_eq!(definition.range.contains(&body_offset), expected_body);
            if expected_body {
                assert_eq!(&source[definition.range.clone()], source.trim_end());
            }
        }
    }
}
