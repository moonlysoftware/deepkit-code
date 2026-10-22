//! Office documents through the shared Markdown chunker; v1.md §6.2 and §3.5.

use std::{io::Cursor, path::Path};

use office_oxide::{Document, DocumentFormat, DocumentIR, OfficeError};

use super::{Chunker, PreparedUnit};

/// Normalized units or a fixed, non-sensitive reason for catalogue-only fallback.
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

/// Parses captured, byte-admitted OOXML and preserves sheet/slide identity.
/// One allowance spans the whole document, including an overflow unit for the
/// existing cap-retention logic. Cooperative checks fence each section's work.
pub(crate) fn extract<E>(
    path: &str,
    source: &[u8],
    options: &crate::config::Chunk,
    max_units: u64,
    check: impl Fn() -> Result<(), E>,
) -> Result<Extracted, E> {
    check()?;
    let extension = Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let format = match extension.as_str() {
        "docx" => DocumentFormat::Docx,
        "xlsx" => DocumentFormat::Xlsx,
        "pptx" => DocumentFormat::Pptx,
        _ => return Ok(Extracted::refused("unsupported_office")),
    };
    let document = match Document::from_reader(Cursor::new(source.to_vec()), format) {
        Ok(document) => document,
        Err(OfficeError::UnsupportedFormat(message)) if message.contains("password-protected") => {
            return Ok(Extracted::refused("encrypted_office"));
        }
        Err(_) => return Ok(Extracted::refused("malformed_office")),
    };
    check()?;
    let ir = document.to_ir();
    let mut output = Extracted {
        units: Vec::new(),
        reason: None,
    };
    for (index, section) in ir.sections.into_iter().enumerate() {
        check()?;
        if output.units.len() as u64 > max_units {
            break;
        }
        // Render sections separately so repeated slide titles and sheet names
        // never lose their source locator. Default rendering keeps image alt
        // text, without expanding embedded images into base64 model inputs.
        let section = DocumentIR {
            metadata: ir.metadata.clone(),
            sections: vec![section],
        };
        let markdown = section.to_markdown();
        let remaining = max_units.saturating_sub(output.units.len() as u64);
        let Ok(units) = super::prepare_units(
            "section.md",
            None,
            Chunker::Markdown,
            markdown.as_bytes(),
            options,
            remaining,
        ) else {
            return Ok(Extracted::refused("malformed_office"));
        };
        for mut unit in units {
            let prefix = match format {
                DocumentFormat::Xlsx => Some(format!("sheet {}", index + 1)),
                DocumentFormat::Pptx => Some(format!("slide {}", index + 1)),
                _ => None,
            };
            if let Some(prefix) = prefix {
                unit.scope_path = qualify(&prefix, &unit.scope_path);
                unit.header = qualify(&prefix, &unit.header);
            }
            output.units.push(unit);
        }
    }
    check()?;
    Ok(output)
}

fn qualify(prefix: &str, value: &str) -> String {
    if value.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix} › {value}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn read(path: &str, source: &[u8], cap: u64) -> Extracted {
        extract(path, source, &Default::default(), cap, || Ok::<_, ()>(())).unwrap()
    }

    #[test]
    fn formats_preserve_headings_sheet_slide_context_and_text() {
        for (name, phrase, prefix) in [
            ("handbook.docx", "interrupted update", "Repository Handbook"),
            ("inventory.xlsx", "manual.pdf", "sheet 1"),
            ("workflow.pptx", "original source", "slide 1"),
        ] {
            let bytes = fs::read(format!("fixtures/docs/{name}")).unwrap();
            assert!(bytes.len() < 1_000_000);
            let output = read(name, &bytes, 100);
            let actual = serde_json::Value::Array(output.units.iter().map(|u| serde_json::json!({
                "kind":u.kind.as_str(),"scope_path":u.scope_path,"header":u.header,"text":u.text,
                "line_start":u.line_start,"line_end":u.line_end,
            })).collect());
            let expected: serde_json::Value = serde_json::from_slice(
                &fs::read(format!("fixtures/docs/{name}.golden.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(actual, expected, "{name} golden");
            assert_eq!(output.reason, None);
            assert_eq!(output.units.len(), 2, "{name}: {:?}", output.units);

            assert!(output.units[0].header.starts_with(prefix));
            assert!(output.units.iter().any(|u| u.text.contains(phrase)));
            assert!(
                output
                    .units
                    .iter()
                    .all(|u| u.kind == crate::store::UnitKind::DocSection)
            );
        }
    }

    #[test]
    fn corrupt_and_encrypted_inputs_are_catalogue_only() {
        for extension in ["docx", "xlsx", "pptx"] {
            let output = read(
                &format!("broken.{extension}"),
                b"not an Office archive",
                100,
            );
            assert_eq!(output.reason, Some("malformed_office"));
            assert!(output.units.is_empty());
        }
        let output = read(
            "encrypted.docx",
            include_bytes!("../../fixtures/docs/encrypted.docx"),
            100,
        );
        assert_eq!(output.reason, Some("encrypted_office"));
        assert!(output.units.is_empty());
        assert_eq!(
            read("unknown.doc", b"legacy", 100).reason,
            Some("unsupported_office")
        );
    }

    #[test]
    fn one_unit_allowance_spans_sections_and_cancellation_propagates() {
        let bytes = include_bytes!("../../fixtures/docs/inventory.xlsx");
        let zero = read("inventory.xlsx", bytes, 0);
        assert_eq!(zero.units.len(), 1, "one overflow unit, not one per sheet");
        let one = read("inventory.xlsx", bytes, 1);
        assert_eq!(one.units.len(), 2);
        assert!(one.units[1].header.starts_with("sheet 2"));
        let cancelled = extract("inventory.xlsx", bytes, &Default::default(), 100, || {
            Err("cancelled")
        });
        assert_eq!(cancelled.err(), Some("cancelled"));
    }
}
