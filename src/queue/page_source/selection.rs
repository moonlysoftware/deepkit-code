//! Finite scanned-document page selection without image decoding; v1 §6.8/§8.
use super::*;

/// Scheduling metadata only; dispatch still rebuilds policy and page authority.
pub(crate) struct Candidate {
    pub(crate) file: i64,
    pub(crate) page: u32,
    pub(crate) rank: f64,
    pub(crate) source_hash: [u8; 32],
}

/// Selects admitted scanned files by existing rank and each file's page order.
/// Only one PDF byte buffer is retained at a time; image extraction is deferred.
pub(crate) fn select(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    config: &Config,
    work: Work<'_>,
) -> Result<Vec<Candidate>, Error> {
    if !config.llm.enabled || !config.enrich.scanned_pages || config.llm.vlm_model.trim().is_empty()
    {
        return Ok(Vec::new());
    }
    let limits = scan::Limits {
        document_bytes: config
            .limits
            .max_doc_bytes()
            .map_err(|_| Error::Invalid("page document cap"))?
            .0
            .min(
                config
                    .limits
                    .max_file_bytes()
                    .map_err(|_| Error::Invalid("page file cap"))?
                    .0,
            ),
        image_bytes: config
            .limits
            .max_image_bytes()
            .map_err(|_| Error::Invalid("page image cap"))?
            .0,
        pages: config.limits.max_pdf_pages,
    };
    if limits.pages == 0 {
        return Ok(Vec::new());
    }
    let mut query=tx.prepare("WITH ranks AS MATERIALIZED(SELECT id,percent_rank() OVER(ORDER BY rank) AS pct FROM files)
        SELECT f.id,r.pct FROM files f JOIN ranks r ON r.id=f.id WHERE f.kind='doc' AND f.needs_vlm=1
        AND f.max_level>=3 AND f.l1_pending=0 AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint=f.l1_desired
        AND f.l1_deferred_reason IN ('needs_vlm','max_pdf_pages','no_renderer') AND f.size>=0 AND f.size<=?1
        ORDER BY r.pct DESC,f.path")?;
    let mut rows = query.query([limits.document_bytes.min(i64::MAX as u64) as i64])?;
    let mut selected = Vec::new();
    while let Some(row) = rows.next()? {
        source::check_work(work)?;
        let file = row.get(0)?;
        let rank = row.get(1)?;
        let Some(input) = visual_source::resolve(
            tx,
            root,
            config_root,
            config,
            file,
            visual_source::Purpose::ScannedPage,
            work,
        )?
        else {
            continue;
        };
        let count = match scan::page_count(&input.bytes, limits, work) {
            Ok(count) => count,
            Err(scan::Error::Stopped) => return Err(Error::WorkerStopped),
            Err(_) => continue,
        };
        for page in 1..=count {
            source::check_work(work)?;
            selected.push(Candidate {
                file,
                page,
                rank,
                source_hash: input.content_hash,
            });
        }
    }
    Ok(selected)
}
