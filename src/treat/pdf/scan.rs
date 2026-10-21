//! Bounded single-image PDF page extraction for VLM jobs; v1 §6.2/§6.8.
use crate::sync::l1::Work;
use image::ImageEncoder;
use pdf_oxide::{PdfDocument, object::Object};
use std::{
    collections::HashMap,
    io::{self, Cursor, Read, Write},
};

const MAX_CONTENT_BYTES: u64 = 64 * 1024;

#[cfg(feature = "pdfium")]
mod render;

/// Prepares faithful page pixels, preferring lossless embedded-image extraction.
/// The optional renderer is consulted only for unsupported page paint semantics.
pub(crate) fn prepare(
    source: &[u8],
    page: u32,
    limits: Limits,
    work: Work<'_>,
) -> Result<Page, Error> {
    let result = extract(source, page, limits, work);
    #[cfg(feature = "pdfium")]
    if matches!(result, Err(Error::NoRenderer)) {
        return render::page(source, page, limits, work);
    }
    result
}

/// Source, page and wire-image admission supplied by the current configuration.
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) document_bytes: u64,
    pub(crate) image_bytes: u64,
    pub(crate) pages: u64,
}

/// One independently reproducible page input; bytes and title are never logged.
pub(crate) struct Page {
    pub(crate) bytes: Vec<u8>,
    pub(crate) mime: &'static str,
    pub(crate) number: u32,
    pub(crate) total_pages: usize,
    pub(crate) title: Option<String>,
    pub(crate) rendering_policy: Option<&'static str>,
}

/// Fixed diagnostics do not expose source bytes or PDF dictionary values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Error {
    #[error("page preparation stopped")]
    Stopped,
    #[error("PDF exceeds the admitted source byte cap")]
    DocumentBytes,
    #[error("page is outside the admitted page range")]
    PageLimit,
    #[error("malformed PDF page")]
    Malformed,
    #[error("page requires a renderer")]
    NoRenderer,
    #[error("page image exceeds max_image_bytes")]
    ImageBytes,
}

fn check(work: Work<'_>) -> Result<(), Error> {
    work.check().map_err(|_| Error::Stopped)
}

/// Counts admitted pages without decoding the first image. Unsupported paint on
/// an early page must not hide later independently extractable scanned pages.
pub(crate) fn page_count(source: &[u8], limits: Limits, work: Work<'_>) -> Result<u32, Error> {
    check(work)?;
    if source.len() as u64 > limits.document_bytes {
        return Err(Error::DocumentBytes);
    }
    if limits.pages == 0 {
        return Ok(0);
    }
    let document = PdfDocument::from_bytes(source.to_vec()).map_err(|_| Error::Malformed)?;
    if document.is_encrypted() {
        return Err(Error::NoRenderer);
    }
    let count = document.page_count().map_err(|_| Error::Malformed)?;
    check(work)?;
    Ok((count as u64).min(limits.pages).min(u64::from(u32::MAX)) as u32)
}
fn resolve(document: &PdfDocument, object: &Object) -> Result<Object, Error> {
    document
        .resolve_object(object)
        .map_err(|_| Error::Malformed)
}
fn absent(
    document: &PdfDocument,
    dict: &HashMap<String, Object>,
    key: &str,
) -> Result<bool, Error> {
    Ok(match dict.get(key) {
        None => true,
        Some(value) => matches!(resolve(document, value)?, Object::Null),
    })
}
fn filter(document: &PdfDocument, dict: &HashMap<String, Object>) -> Result<Option<String>, Error> {
    let Some(value) = dict.get("Filter") else {
        return Ok(None);
    };
    match resolve(document, value)? {
        Object::Null => Ok(None),
        Object::Name(name) => Ok(Some(name)),
        Object::Array(values) if values.len() == 1 => match resolve(document, &values[0])? {
            Object::Name(name) => Ok(Some(name)),
            _ => Err(Error::NoRenderer),
        },
        _ => Err(Error::NoRenderer),
    }
}
fn read_capped(
    mut reader: impl Read,
    cap: u64,
    overflow: Error,
    work: Work<'_>,
) -> Result<Vec<u8>, Error> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        check(work)?;
        let read = reader.read(&mut buffer).map_err(|_| Error::Malformed)?;
        if read == 0 {
            return Ok(output);
        }
        if (output.len() as u64).saturating_add(read as u64) > cap {
            return Err(overflow);
        }
        output.extend_from_slice(&buffer[..read]);
    }
}
fn decoded_stream(
    document: &PdfDocument,
    object: &Object,
    cap: u64,
    overflow: Error,
    work: Work<'_>,
) -> Result<Vec<u8>, Error> {
    check(work)?;
    let Object::Stream { dict, data } = resolve(document, object)? else {
        return Err(Error::Malformed);
    };
    if !absent(document, &dict, "DecodeParms")? || !absent(document, &dict, "F")? {
        return Err(Error::NoRenderer);
    }
    match filter(document, &dict)?.as_deref() {
        None => read_capped(data.as_ref(), cap, overflow, work),
        Some("FlateDecode") => read_capped(
            flate2::read::ZlibDecoder::new(data.as_ref()),
            cap,
            overflow,
            work,
        ),
        _ => Err(Error::NoRenderer),
    }
}

fn content(
    document: &PdfDocument,
    page: &HashMap<String, Object>,
    cap: u64,
    work: Work<'_>,
) -> Result<Vec<u8>, Error> {
    let object = resolve(document, page.get("Contents").ok_or(Error::Malformed)?)?;
    let streams = match object {
        Object::Array(streams) => streams,
        stream => vec![stream],
    };
    let mut output = Vec::new();
    for stream in streams {
        check(work)?;
        if !output.is_empty() {
            if output.len() as u64 >= cap {
                return Err(Error::NoRenderer);
            }
            output.push(b'\n');
        }
        let bytes = decoded_stream(
            document,
            &stream,
            cap.saturating_sub(output.len() as u64),
            Error::NoRenderer,
            work,
        )?;
        output.extend_from_slice(&bytes);
    }
    Ok(output)
}

// Positive axis-aligned transforms preserve the embedded image's orientation.
// Other transforms, clipping, forms, text, transparency and extra paint require
// a renderer; a single images-only handle would not establish this property.
fn whitespace(byte: u8) -> bool {
    matches!(byte, 0 | 9 | 10 | 12 | 13 | 32)
}
fn delimiter(byte: u8) -> bool {
    whitespace(byte) || b"()<>[]{}/%".contains(&byte)
}
struct Tokens<'a> {
    remaining: &'a [u8],
}
impl<'a> Tokens<'a> {
    fn next(&mut self) -> Result<Option<&'a [u8]>, Error> {
        loop {
            while self.remaining.first().is_some_and(|b| whitespace(*b)) {
                self.remaining = &self.remaining[1..];
            }
            if self.remaining.first() == Some(&b'%') {
                let end = self
                    .remaining
                    .iter()
                    .position(|b| matches!(b, b'\r' | b'\n'))
                    .unwrap_or(self.remaining.len());
                self.remaining = &self.remaining[end..];
            } else {
                break;
            }
        }
        let Some(first) = self.remaining.first() else {
            return Ok(None);
        };
        if *first != b'/' && delimiter(*first) {
            return Err(Error::NoRenderer);
        }
        let end = 1 + self.remaining[1..]
            .iter()
            .position(|b| delimiter(*b))
            .unwrap_or(self.remaining.len() - 1);
        let token = &self.remaining[..end];
        self.remaining = &self.remaining[end..];
        Ok(Some(token))
    }
    fn required(&mut self) -> Result<&'a [u8], Error> {
        self.next()?.ok_or(Error::NoRenderer)
    }
}
fn number(token: &[u8]) -> Result<f32, Error> {
    if !token
        .iter()
        .all(|b| b.is_ascii_digit() || b"+-.".contains(b))
    {
        return Err(Error::NoRenderer);
    }
    std::str::from_utf8(token)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|n| n.is_finite())
        .ok_or(Error::NoRenderer)
}
fn name(token: &[u8]) -> Result<String, Error> {
    let mut bytes = Vec::new();
    let mut at = 1;
    while at < token.len() {
        if token[at] == b'#' {
            let pair = token.get(at + 1..at + 3).ok_or(Error::NoRenderer)?;
            let hex = std::str::from_utf8(pair).map_err(|_| Error::NoRenderer)?;
            bytes.push(u8::from_str_radix(hex, 16).map_err(|_| Error::NoRenderer)?);
            at += 3;
        } else {
            bytes.push(token[at]);
            at += 1;
        }
    }
    if bytes.is_empty() {
        return Err(Error::NoRenderer);
    }
    String::from_utf8(bytes).map_err(|_| Error::NoRenderer)
}
fn single_image(
    content: &[u8],
    bounds: (f32, f32, f32, f32),
    work: Work<'_>,
) -> Result<String, Error> {
    let (x0, y0, x1, y1) = bounds;
    if ![x0, y0, x1, y1].iter().all(|n| n.is_finite()) || x1 <= x0 || y1 <= y0 {
        return Err(Error::NoRenderer);
    }
    let mut matrix = [1.0_f32, 1.0, 0.0, 0.0];
    let mut stack = Vec::new();
    let mut image = None;
    let mut tokens = Tokens { remaining: content };
    // The general PDF parser recovers from unknown/malformed operators. This
    // small strict grammar must consume everything, so recovery cannot hide paint.
    while let Some(token) = tokens.next()? {
        check(work)?;
        match token {
            b"q" if stack.len() < 64 => stack.push(matrix),
            b"Q" => matrix = stack.pop().ok_or(Error::NoRenderer)?,
            token if token.starts_with(b"/") && image.is_none() => {
                if tokens.required()? != b"Do" {
                    return Err(Error::NoRenderer);
                }
                if matrix[2] < x0
                    || matrix[3] < y0
                    || matrix[2] + matrix[0] > x1
                    || matrix[3] + matrix[1] > y1
                {
                    return Err(Error::NoRenderer);
                }
                image = Some(name(token)?);
            }
            token => {
                let values = [
                    number(token)?,
                    number(tokens.required()?)?,
                    number(tokens.required()?)?,
                    number(tokens.required()?)?,
                    number(tokens.required()?)?,
                    number(tokens.required()?)?,
                ];
                if tokens.required()? != b"cm" {
                    return Err(Error::NoRenderer);
                }
                let [a, b, c, d, e, f] = values;
                if a <= 0.0 || d <= 0.0 || b != 0.0 || c != 0.0 {
                    return Err(Error::NoRenderer);
                }
                matrix = [
                    matrix[0] * a,
                    matrix[1] * d,
                    matrix[2] + matrix[0] * e,
                    matrix[3] + matrix[1] * f,
                ];
                if !matrix.iter().all(|n| n.is_finite()) {
                    return Err(Error::NoRenderer);
                }
            }
        }
    }
    if !stack.is_empty() {
        return Err(Error::NoRenderer);
    }
    image.ok_or(Error::NoRenderer)
}

struct Capped {
    bytes: Vec<u8>,
    cap: u64,
    overflow: bool,
}
impl Write for Capped {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if (self.bytes.len() as u64).saturating_add(bytes.len() as u64) > self.cap {
            self.overflow = true;
            return Err(io::Error::other("page image byte cap"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn integer(
    document: &PdfDocument,
    dict: &HashMap<String, Object>,
    key: &str,
) -> Result<u32, Error> {
    resolve(document, dict.get(key).ok_or(Error::Malformed)?)?
        .as_integer()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)
        .ok_or(Error::Malformed)
}
fn image_bytes(
    document: &PdfDocument,
    object: &Object,
    cap: u64,
    work: Work<'_>,
) -> Result<(Vec<u8>, &'static str), Error> {
    check(work)?;
    let Object::Stream { dict, data } = resolve(document, object)? else {
        return Err(Error::NoRenderer);
    };
    if dict.get("Subtype").and_then(Object::as_name) != Some("Image") {
        return Err(Error::NoRenderer);
    }
    for key in ["Mask", "SMask", "Decode", "DecodeParms", "OC", "F"] {
        if !absent(document, &dict, key)? {
            return Err(Error::NoRenderer);
        }
    }
    if let Some(mask) = dict.get("ImageMask")
        && !matches!(
            resolve(document, mask)?,
            Object::Boolean(false) | Object::Null
        )
    {
        return Err(Error::NoRenderer);
    }
    let width = integer(document, &dict, "Width")?;
    let height = integer(document, &dict, "Height")?;
    if integer(document, &dict, "BitsPerComponent")? != 8 {
        return Err(Error::NoRenderer);
    }
    let color = resolve(document, dict.get("ColorSpace").ok_or(Error::NoRenderer)?)?;
    let (channels, color) = match color.as_name() {
        Some("DeviceRGB") => (3_u64, image::ExtendedColorType::Rgb8),
        Some("DeviceGray") => (1_u64, image::ExtendedColorType::L8),
        _ => return Err(Error::NoRenderer),
    };
    if data.is_empty() || data.len() as u64 > cap {
        return Err(Error::ImageBytes);
    }
    match filter(document, &dict)?.as_deref() {
        Some("DCTDecode") => {
            let reader = image::ImageReader::with_format(
                Cursor::new(data.as_ref()),
                image::ImageFormat::Jpeg,
            );
            if reader.into_dimensions().map_err(|_| Error::Malformed)? != (width, height) {
                return Err(Error::Malformed);
            }
            check(work)?;
            Ok((data.to_vec(), "image/jpeg"))
        }
        None | Some("FlateDecode") => {
            // This extraction subset also bounds raw pixels by the image budget.
            // Larger decoded images require a renderer rather than an unbounded
            // inflate/allocation; wire PNG bytes are checked independently below.
            let expected = u64::from(width)
                .checked_mul(u64::from(height))
                .and_then(|n| n.checked_mul(channels))
                .ok_or(Error::NoRenderer)?;
            if expected > cap {
                return Err(Error::NoRenderer);
            }
            let pixels = decoded_stream(document, object, expected, Error::Malformed, work)?;
            if pixels.len() as u64 != expected {
                return Err(Error::Malformed);
            }
            let mut output = Capped {
                bytes: Vec::new(),
                cap,
                overflow: false,
            };
            let encoded = image::codecs::png::PngEncoder::new(&mut output)
                .write_image(&pixels, width, height, color);
            if output.overflow {
                return Err(Error::ImageBytes);
            }
            encoded.map_err(|_| Error::Malformed)?;
            check(work)?;
            Ok((output.bytes, "image/png"))
        }
        _ => Err(Error::NoRenderer),
    }
}

/// Extracts only a verified single-image page. Input/page caps precede the PDF
/// copy; stream inflation and PNG output use bounded writers/readers. Native PDF
/// object parsing and individual codec calls are cooperative, not preemptible.
/// Unsupported paint/transform/image semantics return NoRenderer for the host's
/// visible L0 fallback. This function never writes files or fetches resources.
pub(crate) fn extract(
    source: &[u8],
    page: u32,
    limits: Limits,
    work: Work<'_>,
) -> Result<Page, Error> {
    check(work)?;
    if page == 0 || u64::from(page) > limits.pages {
        return Err(Error::PageLimit);
    }
    if source.len() as u64 > limits.document_bytes {
        return Err(Error::DocumentBytes);
    }
    let document = PdfDocument::from_bytes(source.to_vec()).map_err(|_| Error::Malformed)?;
    if document.is_encrypted() {
        return Err(Error::NoRenderer);
    }
    let total_pages = document.page_count().map_err(|_| Error::Malformed)?;
    let index = usize::try_from(page - 1).map_err(|_| Error::PageLimit)?;
    if index >= total_pages {
        return Err(Error::PageLimit);
    }
    check(work)?;
    let object = document.get_page(index).map_err(|_| Error::Malformed)?;
    let dict = object.as_dict().ok_or(Error::Malformed)?;
    for key in ["Group", "Trans"] {
        if !absent(&document, dict, key)? {
            return Err(Error::NoRenderer);
        }
    }
    if let Some(annotations) = dict.get("Annots") {
        match resolve(&document, annotations)? {
            Object::Null => {}
            Object::Array(items) if items.is_empty() => {}
            _ => return Err(Error::NoRenderer),
        }
    }
    if let Some(rotation) = dict.get("Rotate") {
        match resolve(&document, rotation)? {
            Object::Integer(value) if value % 360 == 0 => {}
            _ => return Err(Error::NoRenderer),
        }
    }
    let content = content(
        &document,
        dict,
        limits.document_bytes.min(MAX_CONTENT_BYTES),
        work,
    )?;
    check(work)?;
    let bounds = document
        .get_page_visible_box(index)
        .map_err(|_| Error::Malformed)?;
    let name = single_image(&content, bounds, work)?;
    let resources = resolve(&document, dict.get("Resources").ok_or(Error::NoRenderer)?)?;
    let resources = resources.as_dict().ok_or(Error::NoRenderer)?;
    let images = resolve(
        &document,
        resources.get("XObject").ok_or(Error::NoRenderer)?,
    )?;
    let images = images.as_dict().ok_or(Error::NoRenderer)?;
    let image = images.get(&name).ok_or(Error::Malformed)?;
    let (bytes, mime) = image_bytes(&document, image, limits.image_bytes, work)?;
    check(work)?;
    Ok(Page {
        bytes,
        mime,
        number: page,
        total_pages,
        title: super::title(&document),
        rendering_policy: None,
    })
}

#[cfg(test)]
mod tests;
