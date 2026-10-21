//! Notebook source cells, excluding outputs and attachments; v1.md §6.2.
use super::{Chunker, PreparedUnit};
use serde::Deserialize;

/// Prepared cells or a stable catalogue-only reason without source content.
pub(crate) struct Extracted {
    pub(crate) units: Vec<PreparedUnit>,
    pub(crate) reason: Option<&'static str>,
}
impl Extracted {
    fn refused(reason: &'static str) -> Self {
        Self {
            units: Vec::new(),
            reason: Some(reason),
        }
    }
}

#[derive(Deserialize)]
struct Notebook {
    nbformat: u64,
    #[serde(default)]
    metadata: Option<Metadata>,
    cells: Vec<Cell>,
}
#[derive(Default, Deserialize)]
struct Metadata {
    language_info: Option<Language>,
    kernelspec: Option<Kernel>,
}
#[derive(Deserialize)]
struct Language {
    name: Option<String>,
}
#[derive(Deserialize)]
struct Kernel {
    language: Option<String>,
}
#[derive(Deserialize)]
struct Cell {
    cell_type: String,
    source: Option<CellSource>,
    // Outputs, attachments and arbitrary metadata are ignored by the deserializer.
}
#[derive(Deserialize)]
#[serde(untagged)]
enum CellSource {
    Text(String),
    Lines(Vec<String>),
}
impl CellSource {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Lines(lines) => lines.concat(),
        }
    }
}

/// Parses byte-admitted nbformat 4 source and shares one unit allowance across cells.
/// Nothing executes: recognized kernel languages select existing AST grammars;
/// unknown kernels use bounded lines. Markdown uses the existing heading chunker.
pub(crate) fn extract<E>(
    source: &[u8],
    options: &crate::config::Chunk,
    max_units: u64,
    check: impl Fn() -> Result<(), E>,
) -> Result<Extracted, E> {
    check()?;
    let Ok(notebook) = serde_json::from_slice::<Notebook>(source) else {
        return Ok(Extracted::refused("malformed_notebook"));
    };
    if notebook.nbformat != 4 {
        return Ok(Extracted::refused("unsupported_notebook_version"));
    }
    let metadata = notebook.metadata.unwrap_or_default();
    let language = metadata
        .language_info
        .and_then(|info| info.name)
        .or_else(|| metadata.kernelspec.and_then(|kernel| kernel.language))
        .map(|name| name.to_ascii_lowercase());
    let language = match language.as_deref() {
        Some("python3" | "python2") => Some("python"),
        Some("c++") => Some("cpp"),
        Some("c#") => Some("csharp"),
        other => other,
    };
    let mut output = Extracted {
        units: Vec::new(),
        reason: None,
    };
    for (index, cell) in notebook.cells.into_iter().enumerate() {
        check()?;
        if output.units.len() as u64 > max_units {
            break;
        }
        let (path, chunker) = match cell.cell_type.as_str() {
            "code" => ("cell", Chunker::Ast),
            "markdown" => ("cell.md", Chunker::Markdown),
            _ => continue,
        };
        let Some(source) = cell.source else {
            return Ok(Extracted::refused("malformed_notebook"));
        };
        let source = source.into_text();
        let remaining = max_units.saturating_sub(output.units.len() as u64);
        let Ok(units) = super::prepare_units(
            path,
            language,
            chunker,
            source.as_bytes(),
            options,
            remaining,
        ) else {
            return Ok(Extracted::refused("malformed_notebook"));
        };
        for mut unit in units {
            let heading = if chunker == Chunker::Markdown && unit.header != "document" {
                unit.header.clone()
            } else {
                unit.text
                    .lines()
                    .find(|line| !line.trim().is_empty())
                    .unwrap_or_default()
                    .trim()
                    .to_owned()
            };
            unit.kind = crate::store::UnitKind::Chunk;
            if chunker == Chunker::Ast {
                unit.origin = super::TextOrigin::EmbeddedCode {
                    language: language.map(str::to_owned),
                };
            }
            let prefix = format!("cell {}", index + 1);
            unit.header = if heading.is_empty() {
                prefix.clone()
            } else {
                format!("{prefix} › {heading}")
            };
            unit.scope_path = if unit.scope_path.is_empty() {
                prefix
            } else {
                format!("{prefix} › {}", unit.scope_path)
            };
            output.units.push(unit);
        }
    }
    check()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn read(value: &serde_json::Value, cap: u64) -> Extracted {
        extract(
            &serde_json::to_vec(value).unwrap(),
            &Default::default(),
            cap,
            || Ok::<_, ()>(()),
        )
        .unwrap()
    }

    #[test]
    fn mixed_cells_match_golden_and_omit_outputs_attachments_and_raw_cells() {
        let source = std::fs::read("fixtures/docs/walkthrough.ipynb").unwrap();
        let output = extract(&source, &Default::default(), 100, || Ok::<_, ()>(())).unwrap();
        assert_eq!(output.reason, None);
        assert!(!output.units.is_empty());
        assert!(
            output
                .units
                .iter()
                .all(|unit| unit.kind == crate::store::UnitKind::Chunk)
        );
        let actual = serde_json::Value::Array(output.units.iter().map(|u| json!({
            "kind":u.kind.as_str(),"scope_path":u.scope_path,"header":u.header,"text":u.text,
            "line_start":u.line_start,"line_end":u.line_end,
        })).collect());
        let expected: serde_json::Value = serde_json::from_slice(
            &std::fs::read("fixtures/docs/walkthrough.ipynb.golden.json").unwrap(),
        )
        .unwrap();
        assert_eq!(actual, expected);
        for sentinel in ["OUTPUT_SENTINEL", "ATTACHMENT_SENTINEL", "RAW_SENTINEL"] {
            assert!(!actual.to_string().contains(sentinel));
        }
        assert!(
            output
                .units
                .iter()
                .any(|u| u.scope_path.contains("normalize_name"))
        );
        assert!(output.units.iter().any(|u| u.header.starts_with("cell 5")));
    }

    #[test]
    fn unknown_kernel_uses_lines_and_source_arrays_join_without_extra_newlines() {
        let doc = json!({"nbformat":4,"metadata":{"kernelspec":{"language":"unknown"}},"cells":[
            {"cell_type":"code","source":["first ","line\n","second line"]}
        ]});
        let output = read(&doc, 100);
        assert_eq!(output.reason, None);
        assert_eq!(output.units.len(), 1);
        assert_eq!(output.units[0].text, "first line\nsecond line");
        assert_eq!(output.units[0].header, "cell 1 › first line");
        let missing_metadata =
            json!({"nbformat":4,"cells":[{"cell_type":"code","source":"first line\nsecond line"}]});
        let mut expected = output.units;
        assert_eq!(
            expected[0].origin,
            super::super::TextOrigin::EmbeddedCode {
                language: Some("unknown".into())
            }
        );
        expected[0].origin = super::super::TextOrigin::EmbeddedCode { language: None };
        assert_eq!(read(&missing_metadata, 100).units, expected);
    }

    #[test]
    fn two_thousand_cells_share_one_cap_and_stop_work_after_the_overflow() {
        let source = serde_json::to_vec(&json!({"nbformat":4,"cells":(0..2000).map(|n|json!({
            "cell_type":"markdown","source":format!("# Section {n}\n\nSearch this notebook.")
        })).collect::<Vec<_>>()}))
        .unwrap();
        let checks = std::cell::Cell::new(0);
        let output = extract(&source, &Default::default(), 500, || {
            checks.set(checks.get() + 1);
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(
            output.units.len(),
            501,
            "500 plus one overflow for the shared cap publisher"
        );
        assert!(
            output
                .units
                .last()
                .unwrap()
                .header
                .starts_with("cell 501 ›")
        );
        assert!(checks.get() < 510, "must not chunk the other 1499 cells");
        let interrupted = extract(&source, &Default::default(), 500, || Err("cancelled"));
        assert_eq!(interrupted.err(), Some("cancelled"));
    }

    #[test]
    fn malformed_and_unsupported_notebooks_have_fixed_l0_reasons() {
        assert_eq!(
            extract(b"{broken", &Default::default(), 100, || Ok::<_, ()>(()))
                .unwrap()
                .reason,
            Some("malformed_notebook")
        );
        assert_eq!(
            read(&json!({"nbformat":3,"cells":[]}), 100).reason,
            Some("unsupported_notebook_version")
        );
        assert_eq!(
            read(&json!({"nbformat":4,"cells":[{"cell_type":"code"}]}), 100).reason,
            Some("malformed_notebook")
        );
        assert_eq!(read(&json!({"nbformat":4,"cells":[]}), 100).units.len(), 0);
    }
}
