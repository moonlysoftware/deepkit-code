//! Local Markdown/HTML image references for verified VLM context; v1 §6.8.
use super::{Error, source};
use crate::sync::l1::Work;
use std::{collections::BTreeMap, path::Path};
use tree_sitter::{Language, Node, Parser, Tree};

/// A repository-relative image reference and its nearby source text.
/// The caller must separately prove both files and their current privacy policy.
pub(crate) struct Reference {
    pub(crate) path: String,
    pub(crate) context: String,
}
/// Bounded prompt context; truncation is explicit so callers can report it.
#[derive(Default)]
pub(crate) struct References {
    pub(crate) items: Vec<Reference>,
    pub(crate) truncated: bool,
    bytes: usize,
    wanted: Option<String>,
}
impl References {
    fn add(
        &mut self,
        document: &str,
        destination: &str,
        heading: &str,
        paragraph: &str,
        cap: usize,
    ) {
        let Some(path) = local_path(document, destination) else {
            return;
        };
        if self.wanted.as_ref().is_some_and(|wanted| wanted != &path) {
            return;
        }
        if self.bytes.saturating_add(path.len()).saturating_add(2) >= cap {
            self.truncated = true;
            return;
        }
        let remaining = cap - self.bytes - path.len() - 2;
        self.truncated |= heading.trim().len() > remaining;
        let heading = prefix(heading.trim(), remaining);
        let body = prefix(paragraph.trim(), remaining.saturating_sub(heading.len()));
        let context = format!("{heading}\n{body}");
        self.truncated |= body.len() < paragraph.trim().len();
        self.bytes += path.len() + context.len();
        self.items.push(Reference { path, context });
    }
}
fn prefix(text: &str, maximum: usize) -> &str {
    let mut end = text.len().min(maximum);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}
fn parse(language: Language, text: &str, work: Work<'_>) -> Result<Tree, Error> {
    source::check_work(work)?;
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|_| Error::Invalid("image reference grammar"))?;
    let tree = parser
        .parse(text, None)
        .ok_or(Error::Invalid("image reference parse"))?;
    source::check_work(work)?;
    Ok(tree)
}
fn walk(
    root: Node<'_>,
    work: Work<'_>,
    mut visit: impl FnMut(Node<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut cursor = root.walk();
    loop {
        source::check_work(work)?;
        visit(cursor.node())?;
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return Ok(());
            }
        }
    }
}
fn text<'a>(node: Node<'_>, source: &'a str) -> &'a str {
    source.get(node.byte_range()).unwrap_or_default()
}
fn child<'a>(node: Node<'_>, kind: &str, source: &'a str) -> Option<&'a str> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .find(|n| n.kind() == kind)
        .map(|n| text(n, source))
}
fn label(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Parses image syntax rather than treating examples, code fences or ordinary
/// hyperlinks as references. Output allocation is bounded by the supplied cap.
pub(crate) fn extract(
    document: &str,
    bytes: &[u8],
    cap: usize,
    work: Work<'_>,
) -> Result<References, Error> {
    extract_for(document, bytes, None, cap, work)
}

/// Selects one image before allocating context, so unrelated images cannot
/// exhaust its prompt allowance. None collects all references for ordering.
pub(crate) fn extract_for(
    document: &str,
    bytes: &[u8],
    wanted: Option<&str>,
    cap: usize,
    work: Work<'_>,
) -> Result<References, Error> {
    source::check_work(work)?;
    let mut result = References {
        wanted: wanted.map(str::to_owned),
        ..Default::default()
    };
    if cap == 0 {
        return Ok(result);
    }
    let Ok(source) = std::str::from_utf8(bytes) else {
        return Ok(result);
    };
    if matches!(
        Path::new(document)
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("html" | "htm")
    ) {
        html(document, source, "", None, cap, work, &mut result)?;
        return Ok(result);
    }
    let tree = parse(tree_sitter_md::LANGUAGE.into(), source, work)?;
    let mut definitions = BTreeMap::new();
    walk(tree.root_node(), work, |node| {
        if node.kind() == "link_reference_definition"
            && let (Some(name), Some(destination)) = (
                child(node, "link_label", source),
                child(node, "link_destination", source),
            )
        {
            definitions.entry(label(name)).or_insert(destination);
        }
        Ok(())
    })?;
    let mut heading = "";
    walk(tree.root_node(), work, |node| {
        if matches!(node.kind(), "atx_heading" | "setext_heading") {
            heading = text(node, source).trim();
        }
        if node.kind() == "html_block" {
            html(
                document,
                text(node, source),
                heading,
                None,
                cap,
                work,
                &mut result,
            )?;
        }
        if node.kind() != "inline" {
            return Ok(());
        }
        let paragraph = text(node, source);
        let inline = parse(tree_sitter_md::INLINE_LANGUAGE.into(), paragraph, work)?;
        walk(inline.root_node(), work, |part| {
            if part.kind() == "image" {
                let destination = child(part, "link_destination", paragraph).or_else(|| {
                    let name = child(part, "link_label", paragraph)
                        .filter(|s| !label(s).is_empty())
                        .or_else(|| child(part, "image_description", paragraph))?;
                    definitions.get(&label(name)).copied()
                });
                if let Some(destination) = destination {
                    result.add(
                        document,
                        &markdown_destination(destination),
                        heading,
                        paragraph,
                        cap,
                    );
                }
            } else if part.kind() == "html_tag" {
                html(
                    document,
                    text(part, paragraph),
                    heading,
                    Some(paragraph),
                    cap,
                    work,
                    &mut result,
                )?;
            }
            Ok(())
        })
    })?;
    Ok(result)
}
fn html(
    document: &str,
    source: &str,
    initial_heading: &str,
    context: Option<&str>,
    cap: usize,
    work: Work<'_>,
    result: &mut References,
) -> Result<(), Error> {
    let tree = parse(tree_sitter_html::LANGUAGE.into(), source, work)?;
    let mut heading = initial_heading;
    walk(tree.root_node(), work, |node| {
        if !matches!(node.kind(), "start_tag" | "self_closing_tag") {
            return Ok(());
        }
        let Some(tag) = child(node, "tag_name", source) else {
            return Ok(());
        };
        if matches!(
            tag.to_ascii_lowercase().as_str(),
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
        ) {
            heading = node
                .parent()
                .map(|n| text(n, source))
                .unwrap_or(initial_heading);
        }
        if !tag.eq_ignore_ascii_case("img") {
            return Ok(());
        }
        let mut cursor = node.walk();
        for attribute in node
            .named_children(&mut cursor)
            .filter(|n| n.kind() == "attribute")
        {
            if !child(attribute, "attribute_name", source)
                .is_some_and(|name| name.eq_ignore_ascii_case("src"))
            {
                continue;
            }
            let Some(destination) = child(attribute, "attribute_value", source).or_else(|| {
                child(attribute, "quoted_attribute_value", source)
                    .map(|s| s.trim_matches(['\'', '"']))
            }) else {
                continue;
            };
            let mut paragraph = text(node, source);
            let mut ancestor = node.parent();
            while let Some(parent) = ancestor {
                source::check_work(work)?;
                let mut cursor = parent.walk();
                let tag = parent
                    .named_children(&mut cursor)
                    .find(|n| n.kind() == "start_tag")
                    .and_then(|n| child(n, "tag_name", source));
                if tag.is_some_and(|t| {
                    t.eq_ignore_ascii_case("figure") || t.eq_ignore_ascii_case("p")
                }) {
                    paragraph = text(parent, source);
                    break;
                }
                ancestor = parent.parent();
            }
            result.add(
                document,
                destination,
                heading,
                context.unwrap_or(paragraph),
                cap,
            );
            break;
        }
        Ok(())
    })
}

fn markdown_destination(value: &str) -> String {
    let mut chars = value.chars().peekable();
    let mut result = String::with_capacity(value.len());
    while let Some(character) = chars.next() {
        if character == '\\' && chars.peek().is_some_and(char::is_ascii_punctuation) {
            if let Some(escaped) = chars.next() {
                result.push(escaped);
            }
        } else {
            result.push(character);
        }
    }
    result
}
// Decode the XML-compatible named references and numeric Unicode references
// used by repository image paths. Other names are retained literally.
fn entities(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut remaining = value;
    while let Some(start) = remaining.find('&') {
        result.push_str(&remaining[..start]);
        remaining = &remaining[start..];
        let decoded = prefix(remaining, 33).find(';').and_then(|end| {
            let name = &remaining[1..end];
            let character = match name {
                "amp" => Some('&'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "lt" => Some('<'),
                "gt" => Some('>'),
                _ => name
                    .strip_prefix("#x")
                    .or_else(|| name.strip_prefix("#X"))
                    .and_then(|n| u32::from_str_radix(n, 16).ok())
                    .or_else(|| name.strip_prefix('#').and_then(|n| n.parse::<u32>().ok()))
                    .and_then(char::from_u32),
            }?;
            Some((end, character))
        });
        if let Some((end, character)) = decoded {
            result.push(character);
            remaining = &remaining[end + 1..];
        } else {
            result.push('&');
            remaining = &remaining[1..];
        }
    }
    result.push_str(remaining);
    result
}

/// Normalizes local URL paths only; filesystem confinement is checked separately.
/// External URLs, data URLs, encoded traversal and ambiguous separators never fetch.
fn local_path(document: &str, destination: &str) -> Option<String> {
    let decoded_entities = entities(destination);
    let destination = decoded_entities
        .trim()
        .trim_start_matches('<')
        .trim_end_matches('>');
    let destination = destination.split(['?', '#']).next()?;
    if destination.is_empty() || destination.starts_with("//") {
        return None;
    }
    let mut bytes = Vec::with_capacity(destination.len());
    let mut chars = destination.bytes();
    while let Some(byte) = chars.next() {
        if byte == b'%' {
            let hi = char::from(chars.next()?).to_digit(16)?;
            let lo = char::from(chars.next()?).to_digit(16)?;
            bytes.push((hi * 16 + lo) as u8);
        } else {
            bytes.push(byte);
        }
    }
    let decoded = std::str::from_utf8(&bytes).ok()?;
    if decoded.contains([':', '\\']) || decoded.chars().any(char::is_control) {
        return None;
    }
    let mut components: Vec<_> = if decoded.starts_with('/') {
        Vec::new()
    } else {
        document
            .rsplit_once('/')
            .map(|(parent, _)| parent.split('/').collect())
            .unwrap_or_default()
    };
    for component in decoded.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop()?;
            }
            part => components.push(part),
        }
    }
    (!components.is_empty()).then(|| components.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn markdown_images_include_heading_paragraph_and_reference_definitions() {
        let source = "# Architecture\n\nThe path ![Inline](../images/a%20b.png) matters.\n\n![Reference][KEY]\n\n[key]: /images/ref.png\n\n```md\n![Example](fake.png)\n```\n\n`![code](fake2.png)`\n\n[ordinary](fake3.png)\n";
        let result = extract("docs/guide.md", source.as_bytes(), 16_384, Work::default()).unwrap();
        assert_eq!(
            result
                .items
                .iter()
                .map(|i| i.path.as_str())
                .collect::<Vec<_>>(),
            ["images/a b.png", "images/ref.png"]
        );
        assert!(result.items[0].context.contains("Architecture"));
        assert!(result.items[0].context.contains("The path"));
        assert!(!result.truncated);
    }
    #[test]
    fn html_images_use_figure_caption_and_ignore_comments_and_scripts() {
        let source = "<h2>Network</h2><figure><img src='../images/net.png' alt='connections'><figcaption>Request flow</figcaption></figure><!-- <img src='fake.png'> --><script>const s=\"<img src='fake2.png'>\"</script><p>Logo <IMG SRC=/logo.png></p>";
        let result = extract(
            "docs/guide.html",
            source.as_bytes(),
            16_384,
            Work::default(),
        )
        .unwrap();
        assert_eq!(
            result
                .items
                .iter()
                .map(|i| i.path.as_str())
                .collect::<Vec<_>>(),
            ["images/net.png", "logo.png"]
        );
        assert!(result.items[0].context.contains("Network"));
        assert!(result.items[0].context.contains("Request flow"));
        assert!(result.items[1].context.contains("Logo"));
    }
    #[test]
    fn inline_html_and_collapsed_reference_images_are_found() {
        let source = "# Setup\n\nUse <img src=\"pic.png\"> here.\n\n![diagram][] and ![diagram]\n\n[diagram]: image.png\n";
        let result = extract("README.md", source.as_bytes(), 16_384, Work::default()).unwrap();
        assert_eq!(
            result
                .items
                .iter()
                .map(|i| i.path.as_str())
                .collect::<Vec<_>>(),
            ["pic.png", "image.png", "image.png"]
        );
    }

    #[test]
    fn image_destinations_decode_markdown_escapes_and_html_character_references() {
        let md = extract(
            "README.md",
            br"![plot](plot\(a\).png)",
            1024,
            Work::default(),
        )
        .unwrap();
        assert_eq!(md.items[0].path, "plot(a).png");
        let html = extract(
            "index.html",
            br#"<img src="plot&amp;axes&#x2e;png">"#,
            1024,
            Work::default(),
        )
        .unwrap();
        assert_eq!(html.items[0].path, "plot&axes.png");
        assert!(local_path("README.md", "&#x2e;&#x2e;/escape.png").is_none());
    }
    #[test]
    fn selecting_one_image_reserves_its_context_and_reports_heading_truncation() {
        let source = format!(
            "{} ![other](other.png)\n\n## Target\n\n![wanted](wanted.png) details",
            "unrelated ".repeat(1000)
        );
        let result = extract_for(
            "README.md",
            source.as_bytes(),
            Some("wanted.png"),
            100,
            Work::default(),
        )
        .unwrap();
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].path, "wanted.png");
        assert!(result.items[0].context.contains("details"));
        assert!(!result.truncated);
        let result = extract(
            "README.md",
            b"# long heading with ![a](a.png)",
            12,
            Work::default(),
        )
        .unwrap();
        assert!(result.truncated);
    }
    #[test]
    fn local_urls_do_not_fetch_or_escape_and_context_stays_bounded() {
        for bad in [
            "https://host/image.png",
            "//host/image.png",
            "data:image/png;base64,a",
            "../../escape.png",
            "%2e%2e/%2e%2e/escape.png",
            "x%00.png",
            "x%5c.png",
        ] {
            assert!(local_path("docs/guide.md", bad).is_none(), "{bad}");
        }
        let result = extract(
            "README.md",
            "# ünicode\n\n![a](a.png) plus ![b](b.png)".as_bytes(),
            23,
            Work::default(),
        )
        .unwrap();
        assert!(result.truncated);
        assert!(
            result
                .items
                .iter()
                .map(|r| r.path.len() + r.context.len())
                .sum::<usize>()
                <= 23
        );
    }
}
