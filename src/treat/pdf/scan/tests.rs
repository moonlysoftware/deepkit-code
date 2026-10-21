//! Generated faithful/unsupported/hostile page acceptance for the bounded extractor.
use super::*;
use std::{fs, path::Path, sync::atomic::AtomicBool};

pub(super) fn limits() -> Limits {
    Limits {
        document_bytes: 1024 * 1024,
        image_bytes: 1024 * 1024,
        pages: 500,
    }
}
pub(super) fn fixtures(pages: u32) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let output = std::process::Command::new("python3")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/corpus/make-scan-fixtures.py"))
        .arg("--output")
        .arg(root.path())
        .arg("--pages")
        .arg(pages.to_string())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    root
}
pub(super) fn bytes(root: &Path, case: &str) -> Vec<u8> {
    fs::read(root.join(format!("scanned-{case}.pdf"))).unwrap()
}
fn error(source: &[u8], page: u32, limits: Limits, work: Work<'_>) -> Error {
    match extract(source, page, limits, work) {
        Err(error) => error,
        Ok(_) => panic!("unexpected admitted page"),
    }
}

#[test]
fn jpeg_fixture_is_reproducible_and_retains_original_compressed_image() {
    let root = fixtures(1);
    let source = bytes(root.path(), "jpeg");
    assert_eq!(
        source,
        include_bytes!("../../../../fixtures/docs/scanned-jpeg.pdf")
    );
    let page = extract(&source, 1, limits(), Default::default()).unwrap();
    assert_eq!(
        (page.number, page.total_pages, page.mime),
        (1, 1, "image/jpeg")
    );
    assert_eq!(page.title.as_deref(), Some("Generated scan"));
    assert!(
        source
            .windows(page.bytes.len())
            .any(|part| part == page.bytes)
    );
    assert_eq!(
        image::ImageReader::new(Cursor::new(&page.bytes))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap(),
        (32, 16)
    );
}

#[test]
fn uncompressed_and_flate_rgb_and_compressed_page_streams_are_supported() {
    let root = fixtures(1);
    let raw = extract(&bytes(root.path(), "raw"), 1, limits(), Default::default()).unwrap();
    let flate = extract(
        &bytes(root.path(), "flate-raw"),
        1,
        limits(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(raw.mime, "image/png");
    assert_eq!(raw.bytes, flate.bytes);
    let pixels = image::load_from_memory(&raw.bytes).unwrap().to_rgb8();
    assert_eq!((pixels.width(), pixels.height()), (32, 16));
    assert!(pixels.pixels().all(|p| p.0 == [200, 20, 20]));
    let compressed = extract(
        &bytes(root.path(), "flate-content"),
        1,
        limits(),
        Default::default(),
    )
    .unwrap();
    let jpeg = extract(&bytes(root.path(), "jpeg"), 1, limits(), Default::default()).unwrap();
    assert_eq!(compressed.bytes, jpeg.bytes);
    let legacy = extract(
        include_bytes!("../../../../fixtures/docs/scanned.pdf"),
        1,
        limits(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(legacy.mime, "image/png");
}

#[test]
fn extra_paint_and_unsupported_page_semantics_require_a_renderer() {
    let root = fixtures(1);
    for case in [
        "overlay",
        "multiple",
        "clipped",
        "rotated",
        "masked",
        "inverted",
        "form",
        "annotations",
        "garbage",
    ] {
        assert_eq!(
            error(&bytes(root.path(), case), 1, limits(), Default::default()),
            Error::NoRenderer,
            "{case}"
        );
    }
    assert_eq!(
        error(
            &bytes(root.path(), "wrong-dimensions"),
            1,
            limits(),
            Default::default()
        ),
        Error::Malformed
    );
}

#[test]
fn source_page_image_inflate_and_png_output_caps_are_enforced() {
    let root = fixtures(2);
    let source = bytes(root.path(), "jpeg");
    assert_eq!(
        extract(&source, 2, limits(), Default::default())
            .unwrap()
            .number,
        2
    );
    assert_eq!(
        error(
            b"not a pdf",
            2,
            Limits {
                pages: 1,
                ..limits()
            },
            Default::default()
        ),
        Error::PageLimit
    );
    assert_eq!(
        error(
            &source,
            1,
            Limits {
                document_bytes: source.len() as u64 - 1,
                ..limits()
            },
            Default::default()
        ),
        Error::DocumentBytes
    );
    assert_eq!(
        error(
            &source,
            1,
            Limits {
                image_bytes: 16,
                ..limits()
            },
            Default::default()
        ),
        Error::ImageBytes
    );
    assert_eq!(
        error(
            &bytes(root.path(), "content-bomb"),
            1,
            limits(),
            Default::default()
        ),
        Error::NoRenderer
    );
    assert_eq!(
        error(
            &bytes(root.path(), "decode-bomb"),
            1,
            limits(),
            Default::default()
        ),
        Error::Malformed
    );
    assert_eq!(
        error(
            include_bytes!("../../../../fixtures/docs/scanned.pdf"),
            1,
            Limits {
                image_bytes: 16,
                ..limits()
            },
            Default::default()
        ),
        Error::ImageBytes
    );
}

#[test]
fn strict_paint_grammar_cannot_recover_past_unknown_operators() {
    let bounds = (0.0, 0.0, 612.0, 792.0);
    assert_eq!(
        single_image(
            b"% comment\nq 512 0 0 692 50 50 cm /#53can Do Q",
            bounds,
            Default::default()
        )
        .unwrap(),
        "Scan"
    );
    for source in [
        b"q /Scan Do Q ignored".as_slice(),
        b"q /Scan Do Q 0 0 m",
        b"q /Scan Do Q /Other Do",
        b"q /Scan Do Q Q",
        b"q 1 0 1 1 0 0 cm /Scan Do Q",
        b"q -1 0 0 1 0 0 cm /Scan Do Q",
        b"q 1e2 0 0 1 0 0 cm /Scan Do Q",
        b"q /Scan Do",
    ] {
        assert_eq!(
            single_image(source, bounds, Default::default()),
            Err(Error::NoRenderer)
        );
    }
    let stopped = AtomicBool::new(true);
    assert_eq!(
        error(
            b"not a pdf",
            1,
            limits(),
            Work {
                cancelled: Some(&stopped),
                deadline: None
            }
        ),
        Error::Stopped
    );
}
