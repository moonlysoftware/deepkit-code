//! Optional bounded PDFium rasterization for scanned pages; v1 §6.2/§6.8.
use super::{Capped, Error, Limits, Page, check};
use crate::sync::l1::Work;
use image::ImageEncoder;
use pdfium_render::prelude::{PdfRenderConfig, Pdfium};
use std::sync::OnceLock;

// pdfium-render installs process-global bindings. Keep one initialized handle;
// trying to bind again would reject every page after the first successful one.
static RENDERER: OnceLock<Result<Pdfium, Error>> = OnceLock::new();

// The spec fixes 150 dpi JPEG. Admit the four-byte bitmap separately from the
// encoded-image cap: up to four times that cap, with a 64 MiB working ceiling.
// Oversized pages are refused, never silently rendered at a different DPI.
fn dimensions(width: f32, height: f32, cap: u64) -> Result<(i32, i32), Error> {
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return Err(Error::Malformed);
    }
    let width = f64::from(width);
    let height = f64::from(height);
    let scale = 150.0_f64 / 72.0;
    let width = (width * scale).round().max(1.0);
    let height = (height * scale).round().max(1.0);
    if width > f64::from(i32::MAX) || height > f64::from(i32::MAX) {
        return Err(Error::ImageBytes);
    }
    let (width, height) = (width as i32, height as i32);
    if (width as u64) * (height as u64) * 4 > cap.saturating_mul(4).min(64 * 1024 * 1024) {
        return Err(Error::ImageBytes);
    }
    Ok((width, height))
}

/// Renders admitted, unencrypted input. Native parsing/rendering and image
/// codecs are synchronous: cancellation is checked around them, not inside.
pub(super) fn page(
    source: &[u8],
    number: u32,
    limits: Limits,
    work: Work<'_>,
) -> Result<Page, Error> {
    check(work)?;
    if source.len() as u64 > limits.document_bytes {
        return Err(Error::DocumentBytes);
    }
    if number == 0 || u64::from(number) > limits.pages {
        return Err(Error::PageLimit);
    }
    if limits.image_bytes < 4 {
        return Err(Error::ImageBytes);
    }
    // Preserve the fast path's encryption refusal and metadata semantics.
    let metadata =
        pdf_oxide::PdfDocument::from_bytes(source.to_vec()).map_err(|_| Error::Malformed)?;
    if metadata.is_encrypted() {
        return Err(Error::NoRenderer);
    }
    let title = super::super::title(&metadata);
    drop(metadata);
    check(work)?;
    let renderer = RENDERER
        .get_or_init(|| {
            Pdfium::bind_to_system_library()
                .map(Pdfium::new)
                .map_err(|_| Error::NoRenderer)
        })
        .as_ref()
        .map_err(|error| *error)?;
    check(work)?;
    let document = renderer
        .load_pdf_from_byte_slice(source, None)
        .map_err(|_| Error::Malformed)?;
    let total_pages = usize::try_from(document.pages().len()).map_err(|_| Error::Malformed)?;
    let index = i32::try_from(number - 1).map_err(|_| Error::PageLimit)?;
    if number as usize > total_pages {
        return Err(Error::PageLimit);
    }
    check(work)?;
    let page = document.pages().get(index).map_err(|_| Error::Malformed)?;
    let (width, height) = dimensions(page.width().value, page.height().value, limits.image_bytes)?;
    check(work)?;
    let bitmap = page
        .render_with_config(
            &PdfRenderConfig::new()
                .set_fixed_size(width, height)
                .render_annotations(true)
                .render_form_data(true),
        )
        .map_err(|_| Error::Malformed)?;
    check(work)?;
    if bitmap.width() != width || bitmap.height() != height {
        return Err(Error::Malformed);
    }
    let pixels = bitmap.as_image().map_err(|_| Error::Malformed)?.to_rgb8();
    check(work)?;
    let mut output = Capped {
        bytes: Vec::new(),
        cap: limits.image_bytes,
        overflow: false,
    };
    let encoded = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85).write_image(
        pixels.as_raw(),
        width as u32,
        height as u32,
        image::ExtendedColorType::Rgb8,
    );
    if output.overflow {
        return Err(Error::ImageBytes);
    }
    encoded.map_err(|_| Error::Malformed)?;
    check(work)?;
    Ok(Page {
        bytes: output.bytes,
        mime: "image/jpeg",
        number,
        total_pages,
        title,
        rendering_policy: Some(
            "pdfium-render/0.9.4-api7881-150dpi-jpeg85-bitmap4x64MiB-annotations-forms-v1",
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_and_cancellation_do_not_require_a_native_library() {
        let limits = Limits {
            document_bytes: 8,
            image_bytes: 4,
            pages: 1,
        };
        let cancelled = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            page(
                b"bad",
                1,
                limits,
                Work {
                    cancelled: Some(&cancelled),
                    ..Work::default()
                }
            ),
            Err(Error::Stopped)
        ));
        assert!(matches!(
            page(
                b"bad",
                1,
                limits,
                Work {
                    deadline: Some(std::time::Instant::now()),
                    ..Work::default()
                }
            ),
            Err(Error::Stopped)
        ));
        assert!(matches!(
            page(b"too large", 1, limits, Work::default()),
            Err(Error::DocumentBytes)
        ));
        assert!(matches!(
            page(b"bad", 0, limits, Work::default()),
            Err(Error::PageLimit)
        ));
        assert!(matches!(
            page(b"bad", 2, limits, Work::default()),
            Err(Error::PageLimit)
        ));
        assert!(matches!(
            page(
                b"bad",
                1,
                Limits {
                    image_bytes: 0,
                    ..limits
                },
                Work::default()
            ),
            Err(Error::ImageBytes)
        ));
    }

    #[test]
    #[ignore = "requires the pinned PDFium native library on the loader path"]
    fn native_page_rendering_preserves_overlay_rotation_and_multiple_paints() {
        let root = super::super::tests::fixtures(2);
        let limits = Limits {
            document_bytes: 1024 * 1024,
            image_bytes: 5 * 1024 * 1024,
            pages: 2,
        };
        for case in [
            "overlay",
            "multiple",
            "clipped",
            "rotated",
            "masked",
            "annotations",
            "stencil",
            "square-annotation",
        ] {
            let source = super::super::tests::bytes(root.path(), case);
            assert!(matches!(
                super::super::extract(&source, 1, limits, Work::default()),
                Err(Error::NoRenderer)
            ));
            let output = super::super::prepare(&source, 1, limits, Work::default()).unwrap();
            assert_eq!(output.mime, "image/jpeg");
            assert_eq!((output.number, output.total_pages), (1, 2));
            assert_eq!(output.title.as_deref(), Some("Generated scan"));
            assert!(output.rendering_policy.is_some());
            let pixels = image::load_from_memory(&output.bytes).unwrap().to_rgb8();
            assert_eq!(
                pixels.dimensions(),
                match case {
                    "rotated" => (1650, 1275),
                    "clipped" => (833, 1250),
                    _ => (1275, 1650),
                }
            );
            if case == "overlay" {
                assert_eq!(pixels.get_pixel(10, 1640).0, [0, 0, 0]);
                assert_eq!(pixels.get_pixel(40, 1640).0, [255, 255, 255]);
            }
            if case == "multiple" {
                assert_ne!(pixels.get_pixel(10, 1640).0, [255, 255, 255]);
                assert_ne!(pixels.get_pixel(10, 1640).0, [0, 0, 0]);
            }
            if case == "stencil" {
                assert_eq!(pixels.get_pixel(600, 800).0, [0, 0, 0]);
                assert_eq!(pixels.get_pixel(10, 1640).0, [255, 255, 255]);
            }
            if case == "square-annotation" {
                let red = pixels.get_pixel(10, 1640).0;
                assert!(red[0] >= 243 && red[1] <= 12 && red[2] <= 12, "{red:?}");
            }
            assert!(output.bytes.len() as u64 <= limits.image_bytes);
            let repeated = super::super::prepare(&source, 1, limits, Work::default()).unwrap();
            assert_eq!(output.bytes, repeated.bytes);
        }
        let source = super::super::tests::bytes(root.path(), "overlay");
        // Preserve offsets: the replacement media box has the same byte width.
        let marker = b"612 792";
        let offset = source
            .windows(marker.len())
            .position(|part| part == marker)
            .unwrap();
        let mut tiny_source = source.clone();
        tiny_source[offset..offset + marker.len()].copy_from_slice(b"001 001");
        assert_eq!(dimensions(1.0, 1.0, 16), Ok((2, 2)));
        assert!(
            matches!(
                page(
                    &tiny_source,
                    1,
                    Limits {
                        image_bytes: 16,
                        ..limits
                    },
                    Work::default()
                ),
                Err(Error::ImageBytes)
            ),
            "JPEG output must obey its cap even when the bitmap fits"
        );
        assert!(matches!(
            page(
                &source,
                1,
                Limits {
                    image_bytes: 100,
                    ..limits
                },
                Work::default()
            ),
            Err(Error::ImageBytes)
        ));
    }

    #[test]
    #[ignore = "requires a process without PDFium on its library loader path"]
    fn missing_native_library_keeps_complex_pages_visibly_deferred() {
        assert!(Pdfium::bind_to_system_library().is_err());
        let root = super::super::tests::fixtures(1);
        let source = super::super::tests::bytes(root.path(), "overlay");
        assert!(matches!(
            super::super::prepare(&source, 1, super::super::tests::limits(), Work::default()),
            Err(Error::NoRenderer)
        ));
        let fast = super::super::tests::bytes(root.path(), "jpeg");
        assert_eq!(
            super::super::prepare(&fast, 1, super::super::tests::limits(), Work::default())
                .unwrap()
                .mime,
            "image/jpeg"
        );
    }

    #[test]
    fn bitmap_admission_precedes_allocation_and_preserves_page_aspect() {
        assert_eq!(dimensions(612.0, 792.0, 5 * 1024 * 1024), Ok((1275, 1650)));
        assert_eq!(dimensions(792.0, 612.0, 5 * 1024 * 1024), Ok((1650, 1275)));
        assert_eq!(dimensions(612.0, 792.0, 2_000_000), Err(Error::ImageBytes));
        assert_eq!(
            dimensions(f32::MAX, f32::MAX, u64::MAX),
            Err(Error::ImageBytes)
        );
        assert_eq!(
            dimensions(100_000.0, 100_000.0, u64::MAX),
            Err(Error::ImageBytes)
        );
        assert_eq!(dimensions(1.0, 1.0, 0), Err(Error::ImageBytes));
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(dimensions(invalid, 1.0, u64::MAX), Err(Error::Malformed));
            assert_eq!(dimensions(1.0, invalid, u64::MAX), Err(Error::Malformed));
        }
    }
}
