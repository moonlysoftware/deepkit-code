//! Bounded file-target enumeration and persisted rank percentiles; v1 §8.3.
use super::{Error, host::Target, priority::Recency};
use crate::store::{Store, UnitKind};
use rusqlite::Transaction;

/// A scheduling candidate, not permission to dispatch. Hosts select the destination
/// from kind and supply recency from the current index/history run.
#[derive(Debug)]
pub(crate) struct Candidate {
    pub(crate) unit: i64,
    pub(crate) file: Option<i64>,
    pub(crate) author_time: Option<i64>,
    pub(crate) kind: UnitKind,
    pub(crate) rank_pct: f64,
}
impl Candidate {
    /// Derives age from immutable commit time; current file candidates stay current-run work.
    pub(crate) fn recency(&self, now: i64) -> Recency {
        self.author_time
            .map_or(Recency::ChangedFile, |time| Recency::Commit {
                age_days: now.saturating_sub(time).max(0) as f64 / 86400.0,
            })
    }
    /// Attaches caller-owned run recency without changing persisted rank selection.
    pub(crate) fn target(&self, recency: Recency) -> Target {
        Target {
            unit: self.unit,
            rank_pct: self.rank_pct,
            recency,
        }
    }
}

/// A finite keyset scan. No SQLite transaction or source text is retained between
/// pages, so provider execution cannot pin its catalogue snapshot in the WAL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Cursor {
    generation: i64,
    upper: i64,
    last: i64,
}
fn generation(tx: &Transaction<'_>) -> Result<i64, Error> {
    let pending: bool = tx.query_row(
        "SELECT value!='0' FROM meta WHERE key='graph_pending'",
        [],
        |r| r.get(0),
    )?;
    if pending {
        return Err(Error::Invalid(
            "graph ranking must finish before catalogue scheduling",
        ));
    }
    Ok(tx.query_row(
        "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
        [],
        |r| r.get(0),
    )?)
}
impl Cursor {
    /// Captures the current L1 generation and finite unit-id ceiling. Run on a
    /// blocking worker; later pages fail if another catalogue scan supersedes it.
    pub(crate) fn start(store: &Store) -> Result<Self, Error> {
        store.read_snapshot(|tx| {
            Ok(Self {
                generation: generation(tx)?,
                upper: tx.query_row("SELECT coalesce(max(id),0) FROM units", [], |r| r.get(0))?,
                last: 0,
            })
        })?
    }
    /// Reads at most limit candidates, advancing only after successful validation.
    /// Percentiles use the persisted graph population, with tied ranks equal; a
    /// unit takes its maximum symbol percentile or its file percentile if none.
    /// Commit ranks are shared by all composites within each page, avoiding
    /// repeated symbol-population scans for every history member. SQL window
    /// working space is separate from the bounded returned row count.
    pub(crate) fn next(&mut self, store: &Store, limit: usize) -> Result<Vec<Candidate>, Error> {
        let limit = i64::try_from(limit)
            .ok()
            .filter(|n| *n > 0)
            .ok_or(Error::Invalid("catalogue page size"))?;
        store.read_snapshot(|tx| {
        if generation(tx)? != self.generation {
            return Err(Error::Invalid(
                "catalogue selection was superseded; restart selection",
            ));
        }
        if self.last == self.upper {
            return Ok(Vec::new());
        }
        let mut query = tx.prepare(
            "WITH file_ranks AS MATERIALIZED (SELECT id,percent_rank() OVER(ORDER BY rank) AS pct FROM files),
             symbol_ranks AS MATERIALIZED (SELECT id,unit_id,coalesce(qualified,name) AS name,percent_rank() OVER(ORDER BY pagerank) AS pct FROM symbols),
             unit_ranks AS MATERIALIZED (SELECT unit_id,max(pct) AS pct FROM symbol_ranks GROUP BY unit_id),
             commit_ranks AS MATERIALIZED (
               SELECT c.id,c.unit_id,c.pr_id,c.author_time,coalesce(
                 (SELECT max(s.pct) FROM symbol_ranks s JOIN json_each(c.symbols_touched) j ON j.value=s.name),
                 (SELECT max(r.pct) FROM file_ranks r JOIN files f ON f.id=r.id JOIN json_each(c.files_touched) j ON j.value=f.path),0.0) AS pct
               FROM commits c),
             candidates AS (
             SELECT u.id,f.id AS file_id,u.kind,coalesce(s.pct,r.pct) AS pct,NULL AS author_time
             FROM units u JOIN files f ON f.id=u.file_id JOIN file_ranks r ON r.id=f.id
             LEFT JOIN unit_ranks s ON s.unit_id=u.id
             WHERE u.id>?1 AND u.id<=?2 AND u.kind IN ('chunk','symbol','doc_section','config_key')
             AND u.ladder_level>=1 AND f.max_level>=2 AND f.l1_pending=0
             AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint=f.l1_desired
             AND (f.l1_deferred_reason IS NULL OR f.l1_deferred_reason='max_pdf_pages'
               OR (u.kind='doc_section' AND u.ladder_level=3 AND f.needs_vlm=1
                 AND f.l1_deferred_reason IN ('needs_vlm','no_renderer')))
             UNION ALL
             SELECT u.id,NULL,u.kind,c.pct,c.author_time
             FROM commit_ranks c JOIN units u ON u.id=c.unit_id WHERE u.id>?1 AND u.id<=?2
             AND u.kind='commit' AND u.file_id IS NULL AND u.ladder_level>=1
             UNION ALL
             SELECT u.id,NULL,u.kind,coalesce(
               (SELECT max(c.pct) FROM commit_ranks c WHERE c.pr_id=p.id),0.0),p.merged_at
             FROM prs p JOIN units u ON u.id=p.unit_id WHERE u.id>?1 AND u.id<=?2
             AND u.kind='pr' AND u.file_id IS NULL AND u.ladder_level>=1
             UNION ALL
             SELECT u.id,NULL,u.kind,coalesce(
               (SELECT max(c.pct) FROM unit_sources us JOIN commit_ranks c ON c.id=us.source_id
                WHERE us.unit_id=u.id AND us.source_kind='commit' AND us.role='content'),0.0),
               (SELECT max(c.author_time) FROM unit_sources us JOIN commits c ON c.id=us.source_id
                WHERE us.unit_id=u.id AND us.source_kind='commit' AND us.role='content')
             FROM units u JOIN symbols s ON s.history_unit=u.id WHERE u.id>?1 AND u.id<=?2 AND u.kind='history' AND u.file_id IS NULL AND u.ladder_level>=1
             UNION ALL
             SELECT u.id,NULL,u.kind,coalesce((SELECT max(r.pct) FROM unit_sources us JOIN file_ranks r ON r.id=us.source_id WHERE us.unit_id=u.id AND us.source_kind='file'),0.0),NULL
             FROM units u JOIN dirs d ON d.module_unit=u.id WHERE u.id>?1 AND u.id<=?2 AND u.kind='module' AND u.file_id IS NULL AND u.ladder_level>=1
             UNION ALL
             SELECT u.id,NULL,u.kind,max(
               coalesce((SELECT max(r.pct) FROM file_ranks r WHERE r.id=p.file_id OR EXISTS(
                 SELECT 1 FROM unit_sources us WHERE us.unit_id=p.id AND us.source_kind='file' AND us.source_id=r.id)),0.0),
               coalesce((SELECT max(c.pct) FROM commit_ranks c JOIN prs pr ON pr.id=c.pr_id WHERE pr.unit_id=p.id),0.0)),
               (SELECT pr.merged_at FROM prs pr WHERE pr.unit_id=p.id)
             FROM units u JOIN units p ON p.id=u.parent_unit WHERE u.id>?1 AND u.id<=?2
             AND u.kind='summary' AND u.ladder_level>=3 AND (
               EXISTS(SELECT 1 FROM symbols s WHERE s.summary_unit=u.id AND s.unit_id=p.id AND u.source_key='summary:symbol:'||s.id)
               OR EXISTS(SELECT 1 FROM prs pr WHERE pr.summary_unit=u.id AND pr.unit_id=p.id AND u.source_key='summary:pr:'||pr.id)
               OR EXISTS(SELECT 1 FROM dirs d WHERE d.summary_unit=u.id AND d.module_unit=p.id AND u.source_key='summary:module:'||d.module_unit)
               OR EXISTS(SELECT 1 FROM files f WHERE f.id=p.file_id AND f.kind='image' AND p.kind='image'
                 AND p.ladder_level=0 AND p.parent_unit IS NULL AND u.source_key='summary:image:'||f.id)))
             SELECT id,file_id,kind,pct,author_time FROM candidates ORDER BY id LIMIT ?3")?;
        let rows = query.query_map(rusqlite::params![self.last, self.upper, limit], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, f64>(3)?,
                r.get::<_, Option<i64>>(4)?,
            ))
        })?;
        let mut candidates = Vec::new();
        for row in rows {
            let (unit, file, kind, rank_pct, author_time) = row?;
            let kind = match kind.as_str() {
                "chunk" => UnitKind::Chunk,
                "symbol" => UnitKind::Symbol,
                "doc_section" => UnitKind::DocSection,
                "config_key" => UnitKind::ConfigKey,
                "commit" => UnitKind::Commit,
                "pr" => UnitKind::PullRequest,
                "history" => UnitKind::History,
                "module" => UnitKind::Module,
                "summary" => UnitKind::Summary,
                _ => return Err(Error::Invalid("catalogue unit kind")),
            };
            if !rank_pct.is_finite() || !(0.0..=1.0).contains(&rank_pct) {
                return Err(Error::Invalid("catalogue rank percentile"));
            }
            candidates.push(Candidate {
                unit,
                file,
                kind,
                rank_pct,
                author_time,
            });
        }
        self.last = candidates
            .last()
            .map_or(self.upper, |candidate| candidate.unit);
        Ok(candidates)
        })?
    }
    /// Reports whether the finite captured id range has been exhausted.
    pub(crate) fn finished(&self) -> bool {
        self.last == self.upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue;
    use std::error::Error as StdError;

    #[test]
    fn history_priority_uses_only_member_commit_ranks_and_latest_time()
    -> Result<(), Box<dyn StdError>> {
        let (_root, store, _) = queue::source::tests::fixture()?;
        store.writer()?.execute_batch("INSERT INTO symbols(id,unit_id,file_id,name,kind,line_start,line_end,pagerank) VALUES
            (1,1,1,'low','function',1,1,0), (2,1,1,'middle','function',1,1,1), (3,1,1,'unrelated','function',1,1,2);
            INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
            SELECT 10,randomblob(32),content_hash,'commit:first','commit','first',0,0 FROM units WHERE id=1;
            INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
            SELECT 11,randomblob(32),content_hash,'commit:second','commit','second',0,0 FROM units WHERE id=1;
            INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
            SELECT 12,randomblob(32),content_hash,'history:low','history','history',0,0 FROM units WHERE id=1;
            INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
            SELECT 13,randomblob(32),content_hash,'commit:unrelated','commit','unrelated',0,0 FROM units WHERE id=1;
            INSERT INTO commits(id,sha,author_time,subject,parents,files_touched,symbols_touched,unit_id) VALUES
            (1,randomblob(20),100,'first',0,'[\"a.rs\"]','[\"low\"]',10),
            (2,randomblob(20),200,'second',0,'[\"a.rs\"]','[\"middle\"]',11),
            (3,randomblob(20),999,'unrelated',0,'[\"a.rs\"]','[\"unrelated\"]',13);
            INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES (12,'commit',1,'content'),(12,'commit',2,'content');
            UPDATE symbols SET history_unit=12 WHERE id=1;
            UPDATE meta SET value='0' WHERE key='graph_pending';")?;
        let candidates = Cursor::start(&store)?.next(&store, 100)?;
        let history = candidates
            .iter()
            .find(|candidate| candidate.unit == 12)
            .ok_or("missing history")?;
        assert_eq!(history.rank_pct, 0.5);
        assert_eq!(history.author_time, Some(200));
        assert!(matches!(history.recency(86600),Recency::Commit {age_days} if age_days==1.0));
        Ok(())
    }

    #[test]
    fn finite_pages_preserve_rank_ties_symbol_precedence_and_maximum_id()
    -> Result<(), Box<dyn StdError>> {
        let (_root, store, _) = queue::source::tests::fixture()?;
        store.writer()?.execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_desired,l1_fingerprint,rank)
             SELECT 2,'b.rs',blake3,size,mtime_ns,kind,max_level,treatment,l1_desired,l1_fingerprint,0 FROM files WHERE id=1;
             INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_desired,l1_fingerprint,rank)
             SELECT 3,'c.rs',blake3,size,mtime_ns,kind,max_level,treatment,l1_desired,l1_fingerprint,2 FROM files WHERE id=1;
             INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
             SELECT 3,randomblob(32),content_hash,'b.rs',2,'chunk',header,0,0 FROM units WHERE id=1;
             INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
             SELECT 4,randomblob(32),content_hash,'b.rs',2,'chunk',header,0,0 FROM units WHERE id=1;
             INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
             SELECT 9223372036854775807,randomblob(32),content_hash,'c.rs',3,'chunk',header,0,0 FROM units WHERE id=1;
             INSERT INTO symbols(id,unit_id,file_id,name,kind,line_start,line_end,pagerank) VALUES(1,3,2,'low','function',1,1,0.25),(2,3,2,'high','function',1,1,1.0);
             UPDATE meta SET value='0' WHERE key='graph_pending'; UPDATE files SET l1_pending=0")?;
        let mut cursor = Cursor::start(&store)?;
        assert!(cursor.next(&store, 0).is_err());
        let first = cursor.next(&store, 1)?;
        assert_eq!(first[0].unit, 1);
        assert_eq!(first[0].rank_pct, 0.0);
        let second = cursor.next(&store, 1)?;
        assert_eq!(second[0].unit, 3);
        assert_eq!(second[0].file, Some(2));
        assert_eq!(second[0].kind, UnitKind::Chunk);
        assert_eq!(
            second[0].rank_pct, 1.0,
            "maximum symbol rank overrides tied file rank"
        );
        assert_eq!(second[0].target(Recency::ChangedFile).unit, 3);
        let tied = cursor.next(&store, 1)?;
        assert_eq!(tied[0].unit, 4);
        assert_eq!(tied[0].rank_pct, first[0].rank_pct);
        let last = cursor.next(&store, 1)?;
        assert_eq!(last[0].unit, i64::MAX);
        assert_eq!(last[0].rank_pct, 1.0);
        assert!(cursor.finished());
        assert!(cursor.next(&store, 1)?.is_empty());
        Ok(())
    }

    #[test]
    fn pages_recheck_completion_and_generation_without_retaining_read_locks()
    -> Result<(), Box<dyn StdError>> {
        let (_root, store, _) = queue::source::tests::fixture()?;
        assert!(Cursor::start(&store).is_err());
        store.set_meta("graph_pending", "0")?;
        for column in [
            "l1_pending=1",
            "l1_deferred_reason='storage_full'",
            "l1_desired=x'01'",
            "max_level=1",
        ] {
            store
                .writer()?
                .execute_batch(&format!("UPDATE files SET {column} WHERE id=1"))?;
            let mut cursor = Cursor::start(&store)?;
            assert!(cursor.next(&store, 8)?.is_empty());
            assert!(cursor.finished());
            store.writer()?.execute_batch("UPDATE files SET l1_pending=0,l1_deferred_reason=NULL,l1_desired=l1_fingerprint,max_level=3; UPDATE files SET l1_pending=0")?;
        }
        store
            .writer()?
            .execute_batch("UPDATE files SET l1_deferred_reason='max_pdf_pages' WHERE id=1")?;
        assert!(!Cursor::start(&store)?.next(&store, 8)?.is_empty());
        store
            .writer()?
            .execute_batch("UPDATE files SET l1_deferred_reason=NULL WHERE id=1")?;
        let mut cursor = Cursor::start(&store)?;
        store.set_meta("graph_pending", "1")?;
        assert!(cursor.next(&store, 1).is_err());
        store.set_meta("graph_pending", "0")?;
        store.set_meta("l1_scan_generation", "1")?;
        assert!(cursor.next(&store, 1).is_err());
        let mut fresh = Cursor::start(&store)?;
        store.writer()?.execute_batch("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at) SELECT 2,randomblob(32),content_hash,source_key,file_id,kind,header,0,0 FROM units WHERE id=1")?;
        assert_eq!(fresh.next(&store, 8)?.len(), 1);
        assert!(fresh.finished());
        assert_eq!(Cursor::start(&store)?.next(&store, 8)?.len(), 2);
        store.set_meta("after_page", "writer can proceed")?;
        Ok(())
    }

    #[test]
    fn pr_rank_is_the_maximum_of_each_member_including_file_fallback()
    -> Result<(), Box<dyn StdError>> {
        let (_root, store, _) = queue::source::tests::fixture()?;
        store.writer()?.execute_batch("UPDATE files SET rank=0;
          INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,l1_desired,l1_fingerprint,rank)
          SELECT 2,'high.rs',blake3,size,mtime_ns,kind,max_level,treatment,l1_desired,l1_fingerprint,2 FROM files WHERE id=1;
          INSERT INTO symbols(id,unit_id,file_id,name,kind,line_start,line_end,pagerank) VALUES (1,1,1,'low','function',1,1,0),(2,1,1,'high','function',1,1,1);
          INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
          SELECT 2,randomblob(32),content_hash,'commit:1',NULL,'commit',header,0,0 FROM units WHERE id=1;
          INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
          SELECT 3,randomblob(32),content_hash,'commit:2',NULL,'commit',header,0,0 FROM units WHERE id=1;
          INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
          SELECT 4,randomblob(32),content_hash,'pr:1',NULL,'pr',header,0,0 FROM units WHERE id=1;
          INSERT INTO prs(id,title,merge_sha,strategy,commit_count,merged_at,unit_id) VALUES(1,'PR',randomblob(20),'merge',2,100,4);
          INSERT INTO commits(sha,author,author_time,subject,body,parents,files_touched,symbols_touched,pr_id,unit_id)
          VALUES(randomblob(20),'author',99,'one','',1,'[\"a.rs\"]','[\"low\"]',1,2),
          (randomblob(20),'author',100,'two','',1,'[\"high.rs\"]','[]',1,3);
          UPDATE meta SET value='0' WHERE key='graph_pending'; UPDATE files SET l1_pending=0;")?;
        let candidates = Cursor::start(&store)?.next(&store, 20)?;
        let pr = candidates
            .iter()
            .find(|c| c.kind == UnitKind::PullRequest)
            .unwrap();
        assert_eq!(
            pr.rank_pct, 1.0,
            "a low-ranked symbol in one member must not hide another member's high file rank"
        );
        assert_eq!(pr.author_time, Some(100));
        Ok(())
    }

    #[test]
    fn file_backed_selection_releases_its_snapshot_before_writer_work()
    -> Result<(), Box<dyn StdError>> {
        let (root, store, _) = queue::source::tests::fixture()?;
        store.set_meta("graph_pending", "0")?;
        let path = root.path().join("catalogue.db");
        store
            .writer()?
            .execute("VACUUM INTO ?1", [path.to_string_lossy().as_ref()])?;
        let file = crate::store::open(&path, crate::store::StoreLimits::default())?;
        let mut cursor = Cursor::start(&file)?;
        assert_eq!(cursor.next(&file, 1)?.len(), 1);
        file.set_meta("after_selection", "committed")?;
        assert!(!crate::store::wal::checkpoint(&file)?.busy);
        assert_eq!(crate::store::wal::wal_bytes(&*file.writer()?)?, 0);
        Ok(())
    }
}
