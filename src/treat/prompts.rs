//! Embedded and bounded override prompts for exactly five L3 jobs; v1 §6.8.
mod render;
use crate::queue::Kind;
pub(crate) use render::Error;
use std::{collections::BTreeMap, io::Read, path::Path};

const MAX_TEMPLATE_BYTES: u64 = 64 * 1024;

/// One validated template without a Debug implementation that could expose overrides.
pub(crate) struct Template {
    text: String,
    pub(crate) fingerprint: [u8; 32],
}

/// Returns the canonical override filename stem; embedding is not an L3 prompt.
pub(crate) fn name(kind: Kind) -> Result<&'static str, Error> {
    match kind {
        Kind::DescribeSymbol => Ok("describe_symbol"),
        Kind::SummarizePr => Ok("summarize_pr"),
        Kind::SummarizeModule => Ok("summarize_module"),
        Kind::DescribeImage => Ok("describe_image"),
        Kind::VlmPage => Ok("vlm_page"),
        Kind::Embed => Err(Error::Template),
    }
}

/// Fixed summary ceilings; pages use the separately bounded T3.6 page preparation.
pub(crate) fn output_tokens(kind: Kind) -> Result<Option<u32>, Error> {
    match kind {
        Kind::DescribeSymbol => Ok(Some(120)),
        Kind::SummarizePr => Ok(Some(300)),
        Kind::SummarizeModule | Kind::DescribeImage => Ok(Some(200)),
        Kind::VlmPage => Ok(None),
        Kind::Embed => Err(Error::Template),
    }
}

fn embedded(kind: Kind) -> Result<&'static str, Error> {
    match kind {
        Kind::DescribeSymbol => Ok(include_str!("prompts/describe_symbol.md")),
        Kind::SummarizePr => Ok(include_str!("prompts/summarize_pr.md")),
        Kind::SummarizeModule => Ok(include_str!("prompts/summarize_module.md")),
        Kind::DescribeImage => Ok(include_str!("prompts/describe_image.md")),
        Kind::VlmPage => Ok(include_str!("prompts/vlm_page.md")),
        Kind::Embed => Err(Error::Template),
    }
}

impl Template {
    /// Loads one override, falling back only when absent. Refuses unreadable,
    /// malformed or oversized overrides instead of silently changing the prompt.
    pub(crate) fn load(root: &Path, directory: &str, kind: Kind) -> Result<Self, Error> {
        let filename = format!("{}.md", name(kind)?);
        let directory = Path::new(directory);
        let opened = if directory.is_absolute() {
            crate::sync::source::open_regular(directory, Path::new(&filename))
        } else {
            crate::sync::source::open_regular(root, &directory.join(filename))
        };
        let text = match opened {
            Ok(file) => {
                let metadata = file.metadata().map_err(|_| Error::Read)?;
                if !metadata.is_file() || metadata.len() > MAX_TEMPLATE_BYTES {
                    return Err(Error::Capacity);
                }
                let mut bytes = Vec::new();
                file.take(MAX_TEMPLATE_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|_| Error::Read)?;
                if bytes.len() as u64 > MAX_TEMPLATE_BYTES {
                    return Err(Error::Capacity);
                }
                String::from_utf8(bytes).map_err(|_| Error::Template)?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                embedded(kind)?.to_owned()
            }
            Err(_) => return Err(Error::Read),
        };
        if text.trim().is_empty() {
            return Err(Error::Template);
        }
        let fingerprint = *blake3::hash(text.as_bytes()).as_bytes();
        Ok(Self { text, fingerprint })
    }
    /// Keeps inserted code and documents literal; output allocation has an explicit cap.
    pub(crate) fn render(
        &self,
        fields: &BTreeMap<&str, &str>,
        byte_cap: usize,
    ) -> Result<String, Error> {
        render::render(&self.text, fields, byte_cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    #[test]
    fn five_embedded_prompts_match_reviewed_rendering_goldens_and_output_ceilings() {
        #[derive(serde::Deserialize)]
        struct Golden {
            kind: String,
            fields: BTreeMap<String, String>,
            output_tokens: Option<u32>,
            expected: String,
        }
        let cases: Vec<Golden> =
            serde_json::from_str(include_str!("../../fixtures/enrichment/prompts.json")).unwrap();
        assert_eq!(cases.len(), 5);
        let root = tempfile::tempdir().unwrap();
        for case in cases {
            let kind = match case.kind.as_str() {
                "describe_symbol" => Kind::DescribeSymbol,
                "summarize_pr" => Kind::SummarizePr,
                "summarize_module" => Kind::SummarizeModule,
                "describe_image" => Kind::DescribeImage,
                "vlm_page" => Kind::VlmPage,
                _ => panic!("unexpected golden job"),
            };
            assert_eq!(name(kind).unwrap(), case.kind);
            assert_eq!(output_tokens(kind).unwrap(), case.output_tokens);
            let fields = case
                .fields
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let template = Template::load(root.path(), "prompts", kind).unwrap();
            assert_eq!(template.render(&fields, 4096).unwrap(), case.expected);
        }
    }

    #[cfg(unix)]
    #[test]
    fn prompt_override_refuses_symlinks_and_nonregular_files_without_blocking() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("prompts")).unwrap();
        std::fs::write(root.path().join("elsewhere"), "private-marker").unwrap();
        let path = root.path().join("prompts/describe_symbol.md");
        std::os::unix::fs::symlink(root.path().join("elsewhere"), &path).unwrap();
        assert!(matches!(
            Template::load(root.path(), "prompts", Kind::DescribeSymbol),
            Err(Error::Read)
        ));
        std::fs::remove_file(&path).unwrap();
        mkfifo(&path, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR).unwrap();
        assert!(matches!(
            Template::load(root.path(), "prompts", Kind::DescribeSymbol),
            Err(Error::Read)
        ));
    }
    #[test]
    fn override_bytes_change_identity_and_invalid_overrides_never_fall_back() {
        let root = tempfile::tempdir().unwrap();
        let original = Template::load(root.path(), "prompts", Kind::DescribeSymbol).unwrap();
        std::fs::create_dir(root.path().join("prompts")).unwrap();
        let path = root.path().join("prompts/describe_symbol.md");
        std::fs::write(&path, "Explain {{code}} briefly.").unwrap();
        let changed = Template::load(root.path(), "prompts", Kind::DescribeSymbol).unwrap();
        assert_ne!(original.fingerprint, changed.fingerprint);
        assert_eq!(
            changed
                .render(&BTreeMap::from([("code", "return {{code}};")]), 100)
                .unwrap(),
            "Explain return {{code}}; briefly."
        );
        std::fs::write(&path, [0xff]).unwrap();
        assert!(matches!(
            Template::load(root.path(), "prompts", Kind::DescribeSymbol),
            Err(Error::Template)
        ));
        std::fs::write(&path, vec![b'x'; MAX_TEMPLATE_BYTES as usize + 1]).unwrap();
        assert!(matches!(
            Template::load(root.path(), "prompts", Kind::DescribeSymbol),
            Err(Error::Capacity)
        ));
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            Template::load(root.path(), "prompts", Kind::DescribeSymbol)
                .unwrap()
                .fingerprint,
            original.fingerprint
        );
        assert!(Template::load(root.path(), "prompts", Kind::Embed).is_err());
    }
}
