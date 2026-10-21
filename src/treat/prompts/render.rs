//! Bounded, one-pass prompt interpolation; v1 §6.8.
use std::collections::BTreeMap;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("enrichment prompt override could not be read safely")]
    Read,
    #[error("invalid enrichment prompt template")]
    Template,
    #[error("enrichment prompt requires an unavailable input field")]
    MissingInput,
    #[error("enrichment prompt exceeds its preparation byte limit")]
    Capacity,
}

/// Renders placeholders once; inserted source text is data and is never rescanned.
/// Callers supply only the job's allowed fields and a bounded combined byte cap.
pub(crate) fn render(
    template: &str,
    fields: &BTreeMap<&str, &str>,
    byte_cap: usize,
) -> Result<String, Error> {
    fn append(out: &mut String, value: &str, cap: usize) -> Result<(), Error> {
        if value.len() > cap.saturating_sub(out.len()) {
            return Err(Error::Capacity);
        }
        out.push_str(value);
        Ok(())
    }
    let mut rest = template;
    let mut out = String::new();
    while let Some(start) = rest.find("{{") {
        append(&mut out, &rest[..start], byte_cap)?;
        rest = &rest[start + 2..];
        let end = rest.find("}}").ok_or(Error::Template)?;
        let field = &rest[..end];
        if field.is_empty() || !field.bytes().all(|b| b.is_ascii_lowercase() || b == b'_') {
            return Err(Error::Template);
        }
        let value = fields.get(field).ok_or(Error::MissingInput)?;
        append(&mut out, value, byte_cap)?;
        rest = &rest[end + 2..];
    }
    append(&mut out, rest, byte_cap)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_placeholders_are_literal_and_unknown_override_fields_refuse() {
        let fields = BTreeMap::from([
            ("code", "return \"{{signature}}\";"),
            ("signature", "example()"),
        ]);
        assert_eq!(
            render("{{signature}}\n{{code}}", &fields, 100).unwrap(),
            "example()\nreturn \"{{signature}}\";"
        );
        assert!(matches!(
            render("{{credential}}", &fields, 100),
            Err(Error::MissingInput)
        ));
        assert!(matches!(
            render("{{code", &fields, 100),
            Err(Error::Template)
        ));
        assert!(matches!(
            render("{{code}}", &fields, 3),
            Err(Error::Capacity)
        ));
    }
}
