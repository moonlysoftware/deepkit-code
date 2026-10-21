//! Bounded digital PDF extraction and scanned-page routing; v1.md §6.2 and §3.5.

pub(crate) mod scan;

use pdf_oxide::{PdfDocument, converters::ConversionOptions};

use super::{Chunker, PreparedUnit};

/// Extraction output keeps catalogue state alongside the units it describes.
pub(crate) struct Extracted {
    pub(crate) units: Vec<PreparedUnit>,
    pub(crate) needs_vlm: bool,
    pub(crate) truncated: bool,
    pub(crate) reason: Option<&'static str>,
}

impl Extracted {
    fn refused(reason: &'static str) -> Self {
        Self {
            units: Vec::new(),
            needs_vlm: false,
            truncated: false,
            reason: Some(reason),
        }
    }
}

/// Extracts only admitted pages, then applies the specified 40-character/60% probe.
/// The caller bounds source bytes before allocation and checks its work fence at
/// every page. A capped document is classified using only its admitted prefix.
pub(crate) fn extract<E>(
    source: &[u8],
    max_pages: u64,
    options: &crate::config::Chunk,
    max_units: u64,
    check: impl Fn() -> Result<(), E>,
) -> Result<Extracted, E> {
    check()?;
    let Ok(document) = PdfDocument::from_bytes(source.to_vec()) else {
        return Ok(Extracted::refused("malformed_pdf"));
    };
    if document.is_encrypted() {
        return Ok(Extracted::refused("encrypted_pdf"));
    }
    let Ok(total) = document.page_count() else {
        return Ok(Extracted::refused("malformed_pdf"));
    };
    if total == 0 {
        return Ok(Extracted::refused("empty_pdf"));
    }
    let admitted = total.min(usize::try_from(max_pages).unwrap_or(usize::MAX));
    let mut text_pages = 0usize;
    for page in 0..admitted {
        check()?;
        let Ok(text) = document.extract_text(page) else {
            return Ok(Extracted::refused("malformed_pdf"));
        };
        text_pages += usize::from(text.trim().chars().count() >= 40);
    }
    let truncated = admitted < total;
    let digital = admitted > 0 && (text_pages as u128) * 5 >= (admitted as u128) * 3;
    let mut result = Extracted {
        units: Vec::new(),
        needs_vlm: admitted > 0 && !digital,
        truncated,
        reason: if truncated {
            Some("max_pdf_pages")
        } else if !digital {
            Some("needs_vlm")
        } else {
            None
        },
    };
    if !digital {
        return Ok(result);
    }
    let title = title(&document);
    // Images are handled separately by the VLM route; no image decoding, file
    // writes, OCR, links fetched from the document or base64 expansion here.
    let conversion = ConversionOptions {
        include_images: false,
        image_output_dir: None,
        ..Default::default()
    };
    for page in 0..admitted {
        check()?;
        if result.units.len() as u64 > max_units {
            break;
        }
        let Ok(markdown) = document.to_markdown(page, &conversion) else {
            return Ok(Extracted::refused("malformed_pdf"));
        };
        let remaining = max_units.saturating_sub(result.units.len() as u64);
        let Ok(units) = page_units(&markdown, page, title.as_deref(), options, remaining) else {
            return Ok(Extracted::refused("malformed_pdf"));
        };
        result.units.extend(units);
    }
    check()?;
    Ok(result)
}

/// Chunks either digital or transcribed Markdown from a zero-based PDF page.
/// Lines refer to extracted text; explicit page/title breadcrumbs locate the PDF.
/// The shared chunker retains its overflow marker so callers cannot call capped
/// output a complete page. Callers must reject truncated VLM publication.
pub(crate) fn page_units(
    markdown: &str,
    page: usize,
    title: Option<&str>,
    options: &crate::config::Chunk,
    maximum: u64,
) -> Result<Vec<PreparedUnit>, super::Error> {
    let mut units = super::prepare_units(
        "page.md",
        None,
        Chunker::Markdown,
        markdown.as_bytes(),
        options,
        maximum,
    )?;
    for unit in &mut units {
        let page_header = if unit.header.is_empty() {
            format!("page {}", (page as u128) + 1)
        } else {
            format!("page {} › {}", (page as u128) + 1, unit.header)
        };
        unit.scope_path = match title {
            Some(title) => format!("{title} › {page_header}"),
            None => page_header.clone(),
        };
        unit.header = page_header;
    }
    Ok(units)
}

fn title(document: &PdfDocument) -> Option<String> {
    let info = document.trailer().as_dict()?.get("Info")?;
    let info = document.resolve_object(info).ok()?;
    let title = document
        .resolve_object(info.as_dict()?.get("Title")?)
        .ok()?;
    let bytes = title.as_string()?;
    let title = if bytes.starts_with(&[0xfe, 0xff]) {
        String::from_utf16_lossy(
            &bytes[2..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_be_bytes([b[0], b[1]]))
                .collect::<Vec<_>>(),
        )
    } else if let Some(bytes) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        bytes
            .iter()
            .filter_map(|b| pdf_oxide::fonts::font_dict::pdfdoc_encoding_lookup(*b))
            .collect()
    };
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    (!title.is_empty()).then_some(title)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, process::Command};

    fn read(bytes: &[u8], pages: u64, units: u64) -> Extracted {
        extract(bytes, pages, &Default::default(), units, || Ok::<_, ()>(())).unwrap()
    }

    #[test]
    fn three_digital_documents_preserve_text_page_headers_and_metadata() {
        for (name, title, phrases) in [
            (
                "paper",
                Some("Small search paper"),
                vec![
                    "Search Evaluation",
                    "We compare lexical retrieval",
                    "Older answers remain useful.",
                ],
            ),
            (
                "invoice",
                None,
                vec!["Invoice 1042", "Example Workshop", "50.00"],
            ),
            (
                "manual",
                Some("Indexer Manual"),
                vec!["Installation", "Searching", "original context."],
            ),
        ] {
            let bytes = fs::read(format!("fixtures/docs/{name}.pdf")).unwrap();
            assert!(bytes.len() < 1_000_000);
            let output = read(&bytes, 500, 100);
            assert!(!output.needs_vlm && !output.truncated);
            assert_eq!(output.reason, None);
            assert!(!output.units.is_empty());
            let text = output
                .units
                .iter()
                .map(|u| u.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            for phrase in phrases {
                assert!(text.contains(phrase), "{name}: missing {phrase} in {text}");
            }
            for unit in &output.units {
                assert_eq!(unit.kind, crate::store::UnitKind::DocSection);
                assert!(unit.header.starts_with("page "));
                if let Some(title) = title {
                    assert!(unit.scope_path.starts_with(&format!("{title} › page ")));
                }
            }
            let actual = serde_json::Value::Array(output.units.iter().map(|u| serde_json::json!({
                "kind":u.kind.as_str(),"scope_path":u.scope_path,"header":u.header,"text":u.text,
                "line_start":u.line_start,"line_end":u.line_end,
            })).collect());
            let expected: serde_json::Value = serde_json::from_slice(
                &fs::read(format!("fixtures/docs/{name}.golden.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(actual, expected, "{name} golden");
        }
    }

    #[test]
    fn transcribed_markdown_keeps_page_title_tables_and_visible_overflow() {
        let markdown = "# Inventory\n\nVisible page transcription.\n\n| Item | Count |\n| --- | --- |\n| Blue widgets | 42 |\n\n```text\nLOT-A7\n```\n";
        let units = page_units(
            markdown,
            3,
            Some("Warehouse manual"),
            &Default::default(),
            20,
        )
        .unwrap();
        assert!(!units.is_empty());
        let bodies = units
            .iter()
            .map(|unit| unit.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(bodies.contains("| Blue widgets | 42 |"));
        assert!(bodies.contains("LOT-A7"));
        assert!(
            units
                .iter()
                .all(|unit| unit.kind == crate::store::UnitKind::DocSection)
        );
        assert!(
            units
                .iter()
                .all(|unit| unit.scope_path.starts_with("Warehouse manual › page 4"))
        );
        assert!(units.iter().all(|unit| unit.header.starts_with("page 4")));
        assert!(
            units
                .iter()
                .all(|unit| unit.line_start > 0 && unit.line_end >= unit.line_start)
        );
        // A zero allowance exposes the shared overflow occurrence; publication
        // must reject this page rather than silently treating it as complete.
        assert!(
            !page_units(markdown, 3, None, &Default::default(), 0)
                .unwrap()
                .is_empty()
        );
        assert!(
            page_units("", 3, None, &Default::default(), 20)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn scanned_corrupt_and_encrypted_documents_stay_at_l0() {
        let scanned = read(include_bytes!("../../fixtures/docs/scanned.pdf"), 500, 100);
        assert!(scanned.needs_vlm);
        assert!(scanned.units.is_empty());
        assert_eq!(scanned.reason, Some("needs_vlm"));
        for bytes in [b"not a PDF".as_slice(), b"%PDF-1.7\ntruncated"] {
            let corrupt = read(bytes, 500, 100);
            assert!(corrupt.units.is_empty());
            assert_eq!(corrupt.reason, Some("malformed_pdf"));
        }
        let encrypted = read(
            include_bytes!("../../fixtures/docs/encrypted.pdf"),
            500,
            100,
        );
        assert_eq!(encrypted.reason, Some("encrypted_pdf"));
        assert!(encrypted.units.is_empty());
    }

    #[test]
    fn generated_six_hundred_pages_stop_before_the_first_unadmitted_page() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            Command::new("python3")
                .args([
                    "scripts/corpus/make-pdf-fixtures.py",
                    "--pages",
                    "600",
                    "--output"
                ])
                .arg(root.path())
                .status()
                .unwrap()
                .success()
        );
        let bytes = fs::read(root.path().join("large-manual.pdf")).unwrap();
        let limited = read(&bytes, 500, 1000);
        assert!(limited.truncated);
        assert_eq!(limited.reason, Some("max_pdf_pages"));
        assert_eq!(limited.units.len(), 500);
        assert!(limited.units[499].header.starts_with("page 500"));
        let zero = read(&bytes, 0, 100);
        assert!(zero.units.is_empty() && zero.truncated && !zero.needs_vlm);
        let units = read(&bytes, 3, 1);
        assert_eq!(
            units.units.len(),
            2,
            "retain one overflow unit for cap reporting"
        );
        let cancelled = extract(&bytes, 500, &Default::default(), 100, || Err("cancelled"));
        assert_eq!(cancelled.err(), Some("cancelled"));
    }

    #[test]
    fn probe_observes_character_and_page_fraction_boundaries() {
        let root = tempfile::tempdir().unwrap();
        for (characters, text_pages, digital) in [(40, 3, true), (40, 2, false), (39, 3, false)] {
            assert!(
                Command::new("python3")
                    .args([
                        "scripts/corpus/make-pdf-fixtures.py",
                        "--pages",
                        "5",
                        "--probe-chars",
                        &characters.to_string(),
                        "--text-pages",
                        &text_pages.to_string(),
                        "--output"
                    ])
                    .arg(root.path())
                    .status()
                    .unwrap()
                    .success()
            );
            let bytes = fs::read(root.path().join("large-manual.pdf")).unwrap();
            let output = read(&bytes, 5, 100);
            assert_eq!(
                output.needs_vlm, !digital,
                "{characters} chars on {text_pages}/5 pages"
            );
        }
    }
}
