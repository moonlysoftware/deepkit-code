//! Eligible source-kind inventory for embedded workers; v1 §5/§8.3.
use super::{Error, Store, UnitKind};
impl Store {
    /// Selects only completed, uncapped L1 source kinds that can enter L2, so an
    /// unused docs tier does not load models or resolve credentials during index.
    pub(crate) fn embedding_catalogue_kinds(
        &self,
        history_enabled: bool,
        prs_enabled: bool,
        top_symbols: u64,
        modules_min_files: u64,
    ) -> Result<Vec<UnitKind>, Error> {
        self.read_snapshot(|tx| {
            let mut query = tx.prepare("WITH active_histories AS (SELECT s.history_unit FROM symbols s JOIN units p ON p.id=s.unit_id JOIN files f ON f.id=s.file_id WHERE p.file_id=f.id AND p.kind IN ('symbol','chunk') AND f.max_level>=1 AND f.l1_pending=0 AND f.l1_fingerprint=f.l1_desired AND f.l1_deferred_reason IS NULL ORDER BY s.pagerank DESC,s.id LIMIT ?3) SELECT DISTINCT u.kind FROM units u JOIN files f ON f.id=u.file_id
                WHERE u.kind IN ('chunk','symbol','doc_section','config_key') AND u.ladder_level>=1 AND f.max_level>=2
                AND f.l1_pending=0 AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint=f.l1_desired
                AND (f.l1_deferred_reason IS NULL OR f.l1_deferred_reason='max_pdf_pages'
                    OR (u.kind='doc_section' AND u.ladder_level=3 AND f.needs_vlm=1
                        AND f.l1_deferred_reason IN ('needs_vlm','no_renderer')))
                UNION SELECT u.kind FROM units u JOIN commits c ON c.unit_id=u.id WHERE ?1 AND u.kind='commit' AND u.file_id IS NULL AND u.ladder_level>=1 UNION SELECT u.kind FROM units u JOIN prs p ON p.unit_id=u.id WHERE ?2 AND u.kind='pr' AND u.file_id IS NULL AND u.ladder_level>=1 UNION SELECT u.kind FROM units u JOIN symbols s ON s.history_unit=u.id WHERE ?1 AND u.kind='history' AND u.file_id IS NULL AND u.ladder_level>=1 AND u.id IN (SELECT history_unit FROM active_histories) UNION SELECT u.kind FROM units u JOIN dirs d ON d.module_unit=u.id WHERE u.kind='module' AND u.file_id IS NULL AND u.ladder_level>=1 AND d.file_count>=?4 AND EXISTS(SELECT 1 FROM unit_sources us JOIN files f ON us.source_kind='file' AND f.id=us.source_id WHERE us.unit_id=u.id AND f.max_level>=2)
                UNION SELECT u.kind FROM units u JOIN units parent ON parent.id=u.parent_unit
                WHERE u.kind='summary' AND u.ladder_level>=3 AND (
                  EXISTS(SELECT 1 FROM symbols s JOIN files f ON f.id=s.file_id WHERE s.summary_unit=u.id AND s.unit_id=parent.id
                    AND parent.file_id=f.id AND parent.kind IN ('symbol','chunk') AND f.max_level>=3
                    AND f.l1_pending=0 AND f.l1_fingerprint=f.l1_desired AND f.l1_deferred_reason IS NULL
                    AND u.source_key='summary:symbol:'||s.id)
                  OR EXISTS(SELECT 1 FROM prs p WHERE ?2 AND p.summary_unit=u.id AND p.unit_id=parent.id
                    AND parent.kind='pr' AND parent.file_id IS NULL AND u.source_key='summary:pr:'||p.id)
                  OR EXISTS(SELECT 1 FROM dirs d WHERE d.summary_unit=u.id AND d.module_unit=parent.id
                    AND parent.kind='module' AND parent.file_id IS NULL AND d.file_count>=?4
                    AND u.source_key='summary:module:'||d.module_unit)
                  OR EXISTS(SELECT 1 FROM files f WHERE f.id=parent.file_id AND f.kind='image'
                    AND parent.kind='image' AND parent.ladder_level=0 AND parent.parent_unit IS NULL AND f.max_level>=3
                    AND f.l1_pending=0 AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint=f.l1_desired
                    AND f.l1_deferred_reason IS NULL AND u.source_key='summary:image:'||f.id)) ORDER BY kind")?;
            let rows = query.query_map(rusqlite::params![history_enabled, prs_enabled, top_symbols.min(i64::MAX as u64) as i64,modules_min_files.min(i64::MAX as u64) as i64], |row| {
                let kind: String = row.get(0)?;
                Ok(match kind.as_str() {
                    "chunk" => UnitKind::Chunk,
                    "symbol" => UnitKind::Symbol,
                    "doc_section" => UnitKind::DocSection,
                    "commit" => UnitKind::Commit,
                    "pr" => UnitKind::PullRequest,
                    "history" => UnitKind::History,
                    "module" => UnitKind::Module,
                    "summary" => UnitKind::Summary,
                    _ => UnitKind::ConfigKey,
                })
            })?;
            rows.collect::<Result<Vec<_>, _>>()
        })?.map_err(Error::from)
    }
}
