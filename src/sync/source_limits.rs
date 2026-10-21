//! Shared document/image admission before extraction; v1.md §3.5.
use crate::{
    config,
    store::{FileKind, l1::CapDeferral},
};

/// Captures source-format limits once for bounded indexing and future extractors.
pub(crate) struct SourceLimits {
    doc_bytes: u64,
    image_bytes: u64,
    pdf_pages: u64,
}
impl SourceLimits {
    /// Parses the same validated limits used by CLI and MCP indexing.
    pub(crate) fn new(limits: &config::Limits) -> Result<Self, config::Error> {
        Ok(Self {
            doc_bytes: limits.max_doc_bytes()?.0,
            image_bytes: limits.max_image_bytes()?.0,
            pdf_pages: limits.max_pdf_pages,
        })
    }
    /// Refuses new extraction while retaining any previously indexed snapshot.
    pub(crate) fn refusal(&self, kind: FileKind, bytes: u64) -> Option<CapDeferral> {
        match kind {
            FileKind::Doc if bytes > self.doc_bytes => Some(CapDeferral::DocumentBytes),
            FileKind::Image if bytes > self.image_bytes => Some(CapDeferral::ImageBytes),
            _ => None,
        }
    }
    /// Admits a prefix of PDF pages; actual extraction/VLM is implemented in Phase 3.
    pub(crate) fn pdf_pages(&self, total: u64) -> PageAdmission {
        PageAdmission {
            admitted: total.min(self.pdf_pages),
            skipped: total.saturating_sub(self.pdf_pages),
        }
    }
}
/// A bounded extraction plan that makes page truncation explicit.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PageAdmission {
    pub(crate) admitted: u64,
    pub(crate) skipped: u64,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_byte_and_page_boundaries_degrade_without_overflow() -> Result<(), config::Error> {
        let limits = SourceLimits::new(&config::Limits {
            max_doc_bytes: "8 B".into(),
            max_image_bytes: "4 B".into(),
            max_pdf_pages: 2,
            ..Default::default()
        })?;
        assert!(limits.refusal(FileKind::Doc, 8).is_none());
        assert!(matches!(
            limits.refusal(FileKind::Doc, 9),
            Some(CapDeferral::DocumentBytes)
        ));
        assert!(limits.refusal(FileKind::Image, 4).is_none());
        assert!(matches!(
            limits.refusal(FileKind::Image, 5),
            Some(CapDeferral::ImageBytes)
        ));
        assert!(limits.refusal(FileKind::Code, u64::MAX).is_none());
        assert_eq!(
            limits.pdf_pages(1),
            PageAdmission {
                admitted: 1,
                skipped: 0
            }
        );
        assert_eq!(
            limits.pdf_pages(2),
            PageAdmission {
                admitted: 2,
                skipped: 0
            }
        );
        assert_eq!(
            limits.pdf_pages(u64::MAX),
            PageAdmission {
                admitted: 2,
                skipped: u64::MAX - 2
            }
        );
        let zero = SourceLimits::new(&config::Limits {
            max_pdf_pages: 0,
            ..Default::default()
        })?;
        assert_eq!(
            zero.pdf_pages(5),
            PageAdmission {
                admitted: 0,
                skipped: 5
            }
        );
        Ok(())
    }
}
