//! Atomic bounded commit records and their L1 occurrences; v1.md §3.2, §4.3, §6.2.

use super::{Store, UnitInsert, admission, insert_unit_in_transaction};
use rusqlite::{OptionalExtension, params};

/// One decoded commit; all owned text is prepared before acquiring the writer.
pub(crate) struct Commit<'a> {
    pub(crate) sha: &'a [u8],
    pub(crate) author: &'a str,
    pub(crate) author_time: i64,
    pub(crate) subject: &'a str,
    pub(crate) body: &'a str,
    pub(crate) parents: i64,
    pub(crate) files_json: &'a str,
    pub(crate) symbols_json: &'a str,
    pub(crate) additions: Option<i64>,
    pub(crate) deletions: Option<i64>,
    pub(crate) unit: UnitInsert<'a>,
}

/// Admission distinguishes existing immutable history from capacity deferral.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Outcome {
    Inserted,
    Existing,
    Capacity,
}

/// A definition range known to describe the requested immutable Git blob.
#[derive(Clone)]
pub(crate) struct SymbolRange {
    pub(crate) qualified: String,
    pub(crate) start: i64,
    pub(crate) end: i64,
}

impl Store {
    /// Reconciles retained commit search eligibility and history lexical columns
    /// after treatment or tokenizer changes without losing immutable records.
    pub(crate) fn refresh_history_fts(
        &self,
        settings: &crate::config::Config,
        commit_ceiling: &dyn Fn(&str) -> Result<i64, super::Error>,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<(), super::Error> {
        use crate::treat::fts::{FtsInput, build_columns};
        let key = blake3::hash(
            serde_json::json!({
                "version": crate::treat::fts::VERSION,
                "composites": "pr-history-v3",
                "tokenize": settings.tokenize,
                "treatments": settings.treatments,
                "vlm_enabled": settings.llm.enabled,
            })
            .to_string()
            .as_bytes(),
        )
        .to_hex()
        .to_string();
        let previous: Option<String> = self.checked_read_snapshot(check, |tx| {
            tx.query_row("SELECT value FROM meta WHERE key='history_fts'", [], |r| {
                r.get(0)
            })
            .optional()
            .map_err(super::Error::from)
        })??;
        if previous.as_deref() == Some(&key) {
            return Ok(());
        }
        let mut after = 0_i64;
        loop {
            let rows = self.checked_read_snapshot(check, |tx| {
                tx.prepare("SELECT u.id,u.scope_path,u.header,p.text,c.files_touched FROM units u JOIN unit_contents p ON p.hash=u.content_hash LEFT JOIN commits c ON c.unit_id=u.id WHERE u.id IN (SELECT unit_id FROM commits UNION SELECT unit_id FROM prs UNION SELECT history_unit FROM symbols) AND u.id>?1 ORDER BY u.id LIMIT 64")?
                    .query_map([after], |r| Ok((r.get::<_,i64>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?, r.get::<_,String>(3)?, r.get::<_,Option<String>>(4)?)))?
                    .collect::<Result<Vec<_>,_>>().map_err(super::Error::from)
            })??;
            if rows.is_empty() {
                break;
            }
            let mut columns = Vec::with_capacity(rows.len());
            for (id, breadcrumb, header, text, files) in &rows {
                check()?;
                let ceiling = files.as_deref().map(commit_ceiling).transpose()?;
                let eligible = ceiling.is_none_or(|level| level >= 1);
                columns.push((
                    *id,
                    eligible.then(|| {
                        build_columns(
                            FtsInput {
                                breadcrumb,
                                header,
                                text,
                                language: None,
                                kind: super::FileKind::Doc,
                                extra_idents: &[],
                                has_code_fence: false,
                            },
                            &settings.tokenize,
                        )
                    }),
                    ceiling,
                ));
            }
            admission::transaction_checked(self, admission::Kind::L1Growth, Some(check), |tx| {
                let vector_tables = tx
                    .prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
                    .query_map([], |r| r.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .map(|slug| super::vectors::vector_table_name(&slug))
                    .collect::<Result<Vec<_>, _>>()?;
                for (id, column, ceiling) in &columns {
                    check()?;
                    tx.execute("DELETE FROM fts_units WHERE rowid=?1", [id])?;
                    if let Some(ceiling) = ceiling {
                        if *ceiling >= 1 {
                            tx.execute(
                                "UPDATE units SET ladder_level=CASE WHEN ladder_level=0 THEN 1 ELSE min(ladder_level,?2) END WHERE id=?1",
                                params![id, ceiling],
                            )?;
                        } else {
                            tx.execute("UPDATE units SET ladder_level=0 WHERE id=?1", [id])?;
                        }
                        if *ceiling < 2 {
                            for table in &vector_tables {
                                tx.execute(&format!("DELETE FROM {table} WHERE unit_id=?1"), [id])?;
                            }
                        }
                    }
                    if let Some(column) = column {
                        tx.execute("INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES (?1,?2,?3,?4,?5)", params![id,column.header,column.idents,column.body,column.prose])?;
                    }
                }
                check()?;
                tx.commit()?;
                Ok::<(), super::Error>(())
            })?;
            if let Some((id, ..)) = rows.last() {
                after = *id;
            }
        }
        admission::transaction_checked(self, admission::Kind::Control, Some(check), |tx| {
            check()?;
            tx.execute("INSERT INTO meta(key,value) VALUES ('history_fts',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [&key])?;
            check()?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Reads progress and its completed input key in one short, budgeted snapshot.
    pub(crate) fn history_checkpoint(
        &self,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<(Option<String>, Option<String>), super::Error> {
        self.checked_read_snapshot(check, |tx| {
            let get = |key: &str| {
                tx.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))
                    .optional()
            };
            Ok((get("history_completed")?, get("history_status")?))
        })?
    }

    /// Publishes progress and optional completion atomically under the caller's
    /// original work/HEAD/config fence, including writer admission waits.
    pub(crate) fn history_progress(
        &self,
        status: &str,
        completed: Option<&str>,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<(), super::Error> {
        admission::transaction_checked(self, admission::Kind::Control, Some(check), |tx| {
            check()?;
            tx.execute("INSERT INTO meta(key,value) VALUES ('history_status',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[status])?;
            if let Some(key) = completed {
                tx.execute("INSERT INTO meta(key,value) VALUES ('history_completed',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[key])?;
            }
            check()?;
            tx.commit()?;
            Ok(())
        })
    }
    /// Counts retained history without allocating its identities, for admission and status.
    pub(crate) fn commit_count(
        &self,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<i64, super::Error> {
        self.checked_read_snapshot(check, |tx| {
            tx.query_row("SELECT count(*) FROM commits", [], |row| row.get(0))
                .map_err(Into::into)
        })?
    }
    /// Reuses current definitions only when the verified Git blob and completed
    /// L1 fingerprint match; otherwise history must parse its own source tree.
    pub(crate) fn historical_symbols(
        &self,
        path: &str,
        blob: &[u8],
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<Option<Vec<SymbolRange>>, super::Error> {
        self.checked_read_snapshot(check, |tx| {
            let file: Option<i64> = tx.query_row("SELECT id FROM files WHERE path=?1 AND git_blob=?2 AND l1_pending=0 AND l1_fingerprint=l1_desired", params![path,blob], |row| row.get(0)).optional()?;
            let Some(file) = file else { return Ok(None); };
            let mut query = tx.prepare("SELECT coalesce(qualified,name),line_start,line_end FROM symbols WHERE file_id=?1 ORDER BY line_start,id")?;
            let rows = query.query_map([file], |row| Ok(SymbolRange { qualified: row.get(0)?, start: row.get(1)?, end: row.get(2)? }))?;
            rows.collect::<Result<Vec<_>,_>>().map(Some).map_err(super::Error::from)
        })?
    }
    /// Looks up one immutable identity without loading an unbounded history set.
    pub(crate) fn has_commit(
        &self,
        sha: &[u8],
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<bool, super::Error> {
        self.checked_read_snapshot(check, |tx| {
            tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM commits WHERE sha=?1)",
                [sha],
                |row| row.get(0),
            )
            .map_err(Into::into)
        })?
    }

    /// Publishes immutable commits, units and provenance in one bounded writer
    /// transaction. Every source fence is polled inside the transaction; lowering
    /// a cap never evicts history and returns a capacity outcome for the first
    /// commit that cannot be retained.
    pub(crate) fn insert_commits(
        &self,
        commits: &[Commit<'_>],
        max_commits: u64,
        check: &dyn Fn() -> Result<(), super::Error>,
    ) -> Result<Vec<Outcome>, super::Error> {
        if commits.is_empty() {
            return Ok(Vec::new());
        }
        admission::transaction_checked(self, admission::Kind::L1Growth, Some(check), |tx| {
            check()?;
            let mut retained: u64 = tx
                .query_row("SELECT count(*) FROM commits", [], |row| {
                    row.get::<_, i64>(0)
                })?
                .try_into()
                .map_err(|_| super::Error::L1InputsChanged)?;
            let mut outcomes = Vec::with_capacity(commits.len());
            for commit in commits {
                check()?;
                let existing: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM commits WHERE sha=?1)",
                    [commit.sha],
                    |row| row.get(0),
                )?;
                if existing {
                    outcomes.push(Outcome::Existing);
                    continue;
                }
                if retained >= max_commits {
                    outcomes.push(Outcome::Capacity);
                    break;
                }
                let unit = insert_unit_in_transaction(&tx, commit.unit, None)?;
                if commit.unit.ladder_level == 0 {
                    tx.execute("DELETE FROM fts_units WHERE rowid=?1", [unit])?;
                }
                tx.execute("INSERT INTO commits (sha,author,author_time,subject,body,parents,files_touched,symbols_touched,additions,deletions,unit_id) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)", params![commit.sha,commit.author,commit.author_time,commit.subject,commit.body,commit.parents,commit.files_json,commit.symbols_json,commit.additions,commit.deletions,unit])?;
                let id = tx.last_insert_rowid();
                tx.execute("INSERT INTO unit_sources (unit_id,source_kind,source_id,role) VALUES (?1,'commit',?2,'content')",params![unit,id])?;
                retained += 1;
                outcomes.push(Outcome::Inserted);
            }
            check()?;
            tx.commit()?;
            Ok(outcomes)
        })
    }
}
