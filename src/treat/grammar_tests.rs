//! Production grammar routing, chunk and graph acceptance for v1.md §6.1.
use super::{Chunker, Router, graph, languages::AstLanguage, prepare_units};
use crate::{
    config::Chunk,
    store::{FileKind, UnitKind},
    sync::classify_path,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Fixture {
    language: String,
    file: String,
    expected_name: String,
}

#[test]
fn remaining_language_production_goldens() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/lang");
    let cases: Vec<Fixture> =
        serde_json::from_str(&fs::read_to_string(root.join("remaining.json"))?)?;
    assert_eq!(cases.len(), 25);
    let mut parse_failures = 0;
    let mut failures = Vec::new();
    for (id, extension) in [
        ("rust", "rs"),
        ("python", "py"),
        ("javascript", "js"),
        ("typescript", "ts"),
        ("tsx", "tsx"),
        ("go", "go"),
        ("java", "java"),
        ("php", "php"),
    ] {
        let source = fs::read_to_string(root.join(id).join(format!("input.{extension}")))?;
        let language = AstLanguage::from_language_id(id).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&language.grammar())?;
        parse_failures += usize::from(parser.parse(source, None).unwrap().root_node().has_error());
        graph::tags_configuration(language)?;
    }
    for case in cases {
        let source = fs::read_to_string(root.join(&case.language).join(&case.file))?;
        let language = AstLanguage::from_language_id(&case.language).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&language.grammar())?;
        let tree = parser.parse(&source, None).unwrap();
        parse_failures += usize::from(tree.root_node().has_error());
        let config = graph::tags_configuration(language)?;
        let mut context = tree_sitter_tags::TagsContext::new();
        let (tags, error) = context.generate_tags(config, source.as_bytes(), None)?;
        assert!(!error, "{}", case.language);
        let tags = tags.collect::<Result<Vec<_>, _>>()?;
        assert!(
            tags.iter()
                .any(|tag| source[tag.name_range.clone()] == case.expected_name),
            "{} expected name",
            case.language
        );
        let (kind, detected) = classify_path(&case.file);
        let route = Router::default().route(&case.file, kind);
        let units = prepare_units(
            &case.file,
            detected,
            route.chunker,
            source.as_bytes(),
            &Chunk::default(),
            100,
        )?;
        assert!(!units.is_empty(), "{} produced no units", case.language);
        assert_eq!(
            units,
            prepare_units(
                &case.file,
                detected,
                route.chunker,
                source.as_bytes(),
                &Chunk::default(),
                100
            )?
        );
        for unit in &units {
            assert!(
                unit.line_start >= 1
                    && unit.line_start <= unit.line_end
                    && unit.line_end <= source.lines().count(),
                "{} range {:?}",
                case.language,
                unit
            );
            assert!(!unit.text.is_empty());
        }
        let graph = graph::extract_file_graph(detected.unwrap(), &case.file, source.as_bytes())?;
        if language.is_code() {
            let graph = graph.as_ref().unwrap();
            assert!(!graph.parse_error, "{} graph parse", case.language);
            assert!(
                graph
                    .symbols
                    .iter()
                    .any(|symbol| symbol.name == case.expected_name),
                "{} graph name",
                case.language
            );
            assert!(
                units.iter().any(|unit| unit.kind == UnitKind::Symbol),
                "{} lacks AST symbols",
                case.language
            );
        } else {
            assert!(graph.is_none(), "{} leaked a code graph", case.language);
            assert!(units.iter().all(|unit| unit.kind != UnitKind::Symbol));
        }
        let actual = json!({
            "units": units.iter().map(|u| json!({"kind":format!("{:?}",u.kind),"scope":u.scope_path,"header":u.header,"text":u.text,"start":u.line_start,"end":u.line_end})).collect::<Vec<_>>(),
            "symbols": graph.as_ref().map(|g| g.symbols.iter().map(|s| json!({"name":s.name,"qualified":s.qualified,"kind":s.kind,"start":s.line_start,"end":s.line_end})).collect::<Vec<_>>()),
            "edges": graph.as_ref().map(|g| g.edges.iter().map(|e| json!({"owner":e.source_index,"name":e.dst_name,"kind":format!("{:?}",e.kind),"line":e.line})).collect::<Vec<_>>())
        });

        let expected_path = root.join(&case.language).join("production.json");
        let expected: Value = fs::read_to_string(expected_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null);
        if actual != expected {
            eprintln!("GRAMMAR_GOLDEN {} {}", case.language, actual);
            failures.push(case.language);
        }
    }
    eprintln!("production_parse_failures={parse_failures}");
    assert_eq!(parse_failures, 0);
    assert!(
        failures.is_empty(),
        "unreviewed or changed goldens: {failures:?}"
    );
    Ok(())
}

#[test]
fn lang_table_covers_every_detection_extension() -> Result<(), Box<dyn std::error::Error>> {
    let router = Router::default();
    for &(extension, kind, language) in crate::sync::LANGUAGE_EXTENSIONS {
        let path = format!("input.{extension}");
        assert_eq!(classify_path(&path), (kind, Some(language)));
        assert_eq!(classify_path(&path.to_uppercase()), (kind, Some(language)));
        let route = router.route(&path, kind);
        assert!(
            matches!(
                route.chunker,
                Chunker::Ast | Chunker::Lines | Chunker::Markdown | Chunker::Config
            ),
            "{extension}"
        );
        if kind == FileKind::Code && !matches!(extension, "scss" | "sass" | "lhs") {
            assert!(
                AstLanguage::for_path(language, &path).is_some(),
                "{extension} missing grammar"
            );
        }
        assert!(
            !prepare_units(
                &path,
                Some(language),
                route.chunker,
                b"value = 1\n",
                &Chunk::default(),
                100
            )?
            .is_empty(),
            "{extension} missing chunker"
        );
    }
    for path in ["Dockerfile", ".env", ".env.local"] {
        let (kind, language) = classify_path(path);
        let route = router.route(path, kind);
        assert!(matches!(route.chunker, Chunker::Ast | Chunker::Config));
        assert!(
            !prepare_units(
                path,
                language,
                route.chunker,
                b"ARG VALUE=1\n",
                &Chunk::default(),
                100
            )?
            .is_empty()
        );
    }
    assert_eq!(
        AstLanguage::for_path("ocaml", "input.mli"),
        Some(AstLanguage::OcamlInterface)
    );
    for (language, path) in [
        ("css", "input.scss"),
        ("css", "input.sass"),
        ("haskell", "input.lhs"),
    ] {
        assert!(AstLanguage::for_path(language, path).is_none());
        assert!(graph::extract_file_graph(language, path, b"opaque\n")?.is_none());
    }
    Ok(())
}

#[test]
fn pointer_returning_c_and_cpp_definitions_keep_body_ranges()
-> Result<(), Box<dyn std::error::Error>> {
    for (language, path, source) in [
        (
            "c",
            "input.c",
            "char *message(void) {\n  return \"héllo\";\n}\n",
        ),
        (
            "c",
            "input.c",
            "char **message(char **input) {\n  return input;\n}\n",
        ),
        (
            "cpp",
            "input.cpp",
            "char *message() {\n  return nullptr;\n}\n",
        ),
        (
            "cpp",
            "input.cpp",
            "int &message(int &input) {\n  return input;\n}\n",
        ),
    ] {
        let graph = graph::extract_file_graph(language, path, source.as_bytes())?.unwrap();
        assert!(!graph.parse_error);
        let symbol = graph.symbols.iter().find(|s| s.name == "message").unwrap();
        assert_eq!((symbol.line_start, symbol.line_end), (1, 3));
        let units = prepare_units(
            path,
            Some(language),
            Chunker::Ast,
            source.as_bytes(),
            &Chunk::default(),
            100,
        )?;
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].kind, UnitKind::Symbol);
        assert_eq!(units[0].text, source.trim_end());
    }
    Ok(())
}

#[test]
fn new_grammar_large_containers_and_leaves_obey_token_and_unit_caps()
-> Result<(), Box<dyn std::error::Error>> {
    let members = (0..30)
        .map(|i| format!("  public int value{i}() {{ return {i}; }}\n"))
        .collect::<String>();
    let source = format!("class Values {{\n{members}}}\n");
    let chunk = Chunk {
        target_tokens: 40,
        max_tokens: 80,
        min_tokens: 1,
        ..Chunk::default()
    };
    let all = prepare_units(
        "input.cs",
        Some("csharp"),
        Chunker::Ast,
        source.as_bytes(),
        &chunk,
        100,
    )?;
    assert!(all.len() > 20);
    assert!(all.iter().any(|u| u.scope_path.contains("value29")));
    assert!(all.iter().all(|u| super::count_tokens(&u.text) <= 80));
    let capped = prepare_units(
        "input.cs",
        Some("csharp"),
        Chunker::Ast,
        source.as_bytes(),
        &chunk,
        2,
    )?;
    assert_eq!(
        capped.len(),
        3,
        "one overflow occurrence reports admission refusal"
    );
    let body = (0..200)
        .map(|i| format!("value += {i};\n"))
        .collect::<String>();
    let source = format!("int total(int value) {{\n{body}return value;\n}}\n");
    let units = prepare_units(
        "input.c",
        Some("c"),
        Chunker::Ast,
        source.as_bytes(),
        &chunk,
        1000,
    )?;
    assert!(units.len() > 3);
    assert!(units.iter().all(|u| super::count_tokens(&u.text) <= 80));
    assert!(units.iter().any(|u| u.text.contains("return value")));
    let broken = graph::extract_file_graph("c", "input.c", b"int broken( {\n")?.unwrap();
    assert!(broken.parse_error, "malformed input must remain observable");
    Ok(())
}

#[test]
fn adjacent_definition_does_not_turn_the_preceding_statement_into_a_wrapper()
-> Result<(), Box<dyn std::error::Error>> {
    let source = include_bytes!("../../fixtures/lang/c/adjacent.c");
    for (language, path) in [("c", "input.c"), ("cpp", "input.cpp")] {
        let units = prepare_units(
            path,
            Some(language),
            Chunker::Ast,
            source,
            &Chunk::default(),
            100,
        )?;
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].kind, UnitKind::Chunk);
        assert_eq!(units[0].text, "int value;");
        assert_eq!(units[1].kind, UnitKind::Symbol);
        assert_eq!(units[1].text, "int read_value(void){return value;}");
    }
    Ok(())
}
