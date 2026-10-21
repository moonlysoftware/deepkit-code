//! Fenced catalogue and completed L1 publication; see v1.md §4.2–4.3.

use rusqlite::{OptionalExtension, Transaction, params};

use super::{DirUpsert, Error, FileUpsert, Store, admission};

/// Coherent scan/completion identity for restarting an embedding catalogue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Identity {
    pub(crate) indexed_head: Option<String>,
    pub(crate) scan: i64,
    pub(crate) completed: i64,
}

/// Identity of an equal-content occurrence within one source.
pub(crate) type OccurrenceKey = (Vec<u8>, u32);

/// A cap that completed catalogue work while retaining a previous L1 snapshot.
pub(crate) enum CapDeferral {
    SourceBytes,
    DocumentBytes,
    PdfPages,
    ImageBytes,
    Units,
    Storage,
}

/// Atomic extraction state published with a file's lexical snapshot.
#[derive(Clone, Copy, Default)]
pub(crate) struct Completion {
    pub(crate) truncated: bool,
    pub(crate) needs_vlm: bool,
    pub(crate) reason: Option<&'static str>,
}
impl From<bool> for Completion {
    fn from(truncated: bool) -> Self {
        Self {
            truncated,
            ..Self::default()
        }
    }
}

/// An observable deferred source; its retained L1 fingerprint is not the desired one.
pub(crate) struct DeferredFile {
    pub(crate) path: String,
    pub(crate) reason: String,
    pub(crate) retained_snapshot: bool,
}

/// A scanned source and the fingerprint of its desired L1 output.
pub(crate) struct Observation<'a> {
    pub(crate) file: FileUpsert<'a>,
    pub(crate) fingerprint: &'a [u8],
}

/// A complete prepared catalogue change, fenced against the snapshot it used.
pub(crate) struct Catalogue<'a> {
    /// Last whole-repository walk: cap reached and unsupported path count.
    /// Scoped publications preserve the previous whole-repository observation.
    pub(crate) discovery: Option<(bool, usize)>,
    /// Repository-wide ceiling; existing rows survive a lowered limit.
    pub(crate) max_files: u64,
    pub(crate) generation: i64,
    pub(crate) files: &'a [Observation<'a>],
    pub(crate) renames: &'a [(&'a str, &'a str)],
    pub(crate) deleted: &'a [String],
    pub(crate) dirs: &'a [DirUpsert<'a>],
    pub(crate) complete: bool,
    pub(crate) full: bool,
}

/// The exact file/configuration generation allowed to publish prepared units.
#[derive(Clone, Copy)]
pub(crate) struct Fence<'a> {
    pub(crate) generation: i64,
    pub(crate) file_id: i64,
    pub(crate) path: &'a str,
    pub(crate) content_hash: &'a [u8],
    pub(crate) fingerprint: &'a [u8],
}

/// Rejects output from a process whose captured catalogue has been superseded.
pub(super) fn check_generation(
    transaction: &Transaction<'_>,
    generation: i64,
) -> Result<(), Error> {
    let matches: bool = transaction.query_row(
        "SELECT CAST(value AS INTEGER) = ?1 FROM meta WHERE key='l1_scan_generation'",
        [generation],
        |r| r.get(0),
    )?;
    if matches {
        Ok(())
    } else {
        Err(Error::L1ScanSuperseded { generation })
    }
}

/// Checks the source and configuration identity under the publication writer lock.
pub(super) fn check_file(transaction: &Transaction<'_>, fence: Fence<'_>) -> Result<(), Error> {
    check_generation(transaction, fence.generation)?;
    let matches = transaction
        .prepare("SELECT id FROM files WHERE id=?1 AND path=?2 AND blake3=?3 AND l1_desired=?4")?
        .exists(params![
            fence.file_id,
            fence.path,
            fence.content_hash,
            fence.fingerprint
        ])?;
    if matches {
        Ok(())
    } else {
        Err(Error::FileIndexSuperseded {
            file_id: fence.file_id,
        })
    }
}

/// Reads L1 identity inside a caller's snapshot shared with other publication counters.
pub(crate) fn identity_in(tx: &Transaction<'_>) -> Result<Identity, Error> {
    Ok(tx.query_row(
                "SELECT (SELECT value FROM meta WHERE key='indexed_head'),
                        CAST((SELECT value FROM meta WHERE key='l1_scan_generation') AS INTEGER),
                        CAST((SELECT value FROM meta WHERE key='l1_completed_generation') AS INTEGER)",
                [],
                |row| Ok(Identity { indexed_head: row.get(0)?, scan: row.get(1)?, completed: row.get(2)? }),
            )?)
}

impl Store {
    /// Reads HEAD and both L1 generations in one SQLite statement, so a worker
    /// never combines metadata from different concurrent index publications.
    pub(crate) fn l1_identity(&self) -> Result<Identity, Error> {
        self.read_snapshot(identity_in)?
    }

    /// Resolves a published path to its current file ID and unfinished L1 state in
    /// one checked snapshot. New and renamed paths must not reuse pre-publication
    /// IDs. A missing or replaced identity is refusal, never completed work; later
    /// preparation/publication must repeat the returned identity's fence.
    pub(crate) fn l1_file_work(
        &self,
        generation: i64,
        path: &str,
        content_hash: &[u8],
        fingerprint: &[u8],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(i64, bool), Error> {
        self.checked_read_snapshot(check, |transaction| {
            check_generation(transaction, generation)?;
            check()?;
            let work = transaction.query_row(
                "SELECT id, l1_pending=1 OR (l1_fingerprint IS NOT l1_desired AND l1_deferred_reason IS NULL)
                 FROM files WHERE path=?1 AND blake3=?2 AND l1_desired=?3",
                params![path, content_hash, fingerprint],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).optional()?.ok_or(Error::L1InputsChanged)?;
            check()?;
            Ok(work)
        })?
    }

    /// Reads the catalogue generation with cooperative admission and final validation,
    /// so query result fences do not wait beyond their caller's work budget.
    pub(crate) fn l1_generation_checked(
        &self,
        validate: &dyn Fn() -> Result<(), Error>,
    ) -> Result<i64, Error> {
        self.checked_read_snapshot(validate, |tx| -> Result<i64, Error> {
            Ok(tx.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
                [],
                |row| row.get(0),
            )?)
        })?
    }

    /// Captures the generation before filesystem scanning, so competing scans cannot interleave.
    pub(crate) fn l1_generation(&self) -> Result<i64, Error> {
        self.with_reader(|connection| {
            Ok(connection.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
                [],
                |r| r.get(0),
            )?)
        })
    }

    /// Publishes source observations and pending fingerprints atomically, retaining completed output.
    /// Rechecks caller work admission after acquiring the writer, between records and
    /// immediately before commit; refusal rolls back the entire catalogue generation.
    /// Optional cheap admission checks also bound writer waits without replaying work.
    pub(crate) fn publish_catalogue(
        &self,
        admission_check: Option<&dyn Fn() -> Result<(), Error>>,
        catalogue: Catalogue<'_>,
        check_work: impl Fn() -> Result<(), Error>,
    ) -> Result<i64, Error> {
        self.publish_catalogue_validated(admission_check, catalogue, check_work, || Ok(()))
    }

    /// Revalidates expensive source observations at both transaction boundaries,
    /// while cheap work checks still run between every catalogue record. This keeps
    /// branch verification linear in its prepared sources instead of per-row rescans.
    pub(crate) fn publish_catalogue_validated(
        &self,
        admission_check: Option<&dyn Fn() -> Result<(), Error>>,
        catalogue: Catalogue<'_>,
        check_work: impl Fn() -> Result<(), Error>,
        validate_inputs: impl Fn() -> Result<(), Error>,
    ) -> Result<i64, Error> {
        admission::transaction_checked(
            self,
            admission::Kind::AtomicCatalogue,
            admission_check,
            |transaction| {
                check_work()?;
                validate_inputs()?;
                check_generation(&transaction, catalogue.generation)?;
                for path in catalogue.deleted {
                    check_work()?;
                    super::delete_file_in_transaction(&transaction, path)?;
                }
                for (from, to) in catalogue.renames {
                    check_work()?;
                    let file_id: i64 = transaction.query_row(
                        "SELECT id FROM files WHERE path=?1",
                        [from],
                        |r| r.get(0),
                    )?;
                    transaction
                        .execute("UPDATE files SET path=?1 WHERE id=?2", params![to, file_id])?;
                    super::occurrences::rename_occurrences(&transaction, file_id, to)?;
                    super::replace_path_name(&transaction, file_id, to)?;
                }
                let mut file_count: u64 =
                    transaction.query_row("SELECT count(*) FROM files", [], |row| {
                        row.get::<_, i64>(0).map(|count| count as u64)
                    })?;
                for observation in catalogue.files {
                    check_work()?;
                    let f = observation.file;
                    let exists = transaction
                        .prepare_cached("SELECT 1 FROM files WHERE path=?1")?
                        .exists([f.path])?;
                    if !exists {
                        if file_count >= catalogue.max_files {
                            return Err(Error::FileCapacity {
                                limit: catalogue.max_files,
                            });
                        }
                        file_count += 1;
                    }
                    // No-op scans do not rewrite catalogue or name-index pages.
                    let unchanged: Option<i64> = transaction
                .query_row(
                    "SELECT id FROM files WHERE path=?1 AND blake3=?2 AND size=?3 AND mtime_ns=?4
                 AND lang IS ?5 AND kind=?6 AND git_blob IS ?7 AND max_level=?8 AND treatment=?9",
                    params![
                        f.path,
                        f.blake3,
                        f.size,
                        f.mtime_ns,
                        f.lang,
                        f.kind.as_str(),
                        f.git_blob,
                        f.max_level,
                        f.treatment
                    ],
                    |r| r.get(0),
                )
                .optional()?;
                    let file_id = match unchanged {
                        Some(id) => id,
                        None => super::upsert_file_in_transaction(&transaction, f)?,
                    };
                    transaction.execute(
                "UPDATE files SET l1_desired=?1, l1_pending=1, l1_deferred_reason=NULL
                 WHERE id=?2 AND (l1_desired IS NOT ?1 OR (l1_fingerprint IS NOT ?1 AND l1_deferred_reason IS NULL) OR l1_deferred_reason='storage_full' OR ?3)",
                params![observation.fingerprint, file_id, catalogue.full],
            )?;
                }
                if catalogue.complete {
                    let retained = catalogue
                        .dirs
                        .iter()
                        .map(|d| d.path)
                        .collect::<std::collections::BTreeSet<_>>();
                    let existing = transaction
                        .prepare("SELECT path FROM dirs")?
                        .query_map([], |r| r.get::<_, String>(0))?
                        .collect::<Result<Vec<_>, _>>()?;
                    for path in existing {
                        check_work()?;
                        if !retained.contains(path.as_str()) {
                            transaction.execute("DELETE FROM dirs WHERE path=?1", [path])?;
                        }
                    }
                    for d in catalogue.dirs {
                        check_work()?;
                        transaction.execute("INSERT INTO dirs(path,merkle,file_count) VALUES (?1,?2,?3)
                    ON CONFLICT(path) DO UPDATE SET merkle=excluded.merkle,file_count=excluded.file_count
                    WHERE merkle IS NOT excluded.merkle OR file_count != excluded.file_count", params![d.path,d.merkle,d.file_count])?;
                    }
                }
                let generation: i64 = transaction.query_row(
                    "UPDATE meta SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT)
             WHERE key='l1_scan_generation' RETURNING CAST(value AS INTEGER)",
                    [],
                    |r| r.get(0),
                )?;
                if let Some((reached, skipped)) = catalogue.discovery {
                    let value = serde_json::json!({
                        "generation": generation,
                        "max_files": catalogue.max_files,
                        "max_files_reached": reached,
                        "skipped_non_unicode_paths": skipped,
                        "scope": "last_whole_repository_scan",
                    })
                    .to_string();
                    transaction.execute(
                        "INSERT INTO meta(key,value) VALUES ('discovery_status',?1)
                         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        [value],
                    )?;
                }
                validate_inputs()?;
                check_work()?;
                if let Some(check) = admission_check {
                    check()?;
                }
                transaction.commit()?;
                Ok(generation)
            },
        )
    }

    /// Publishes HEAD only after every admitted source has a completed L1 outcome.
    /// The caller rechecks filesystem HEAD, configuration and work admission after
    /// acquiring the writer and immediately before commit. Optional cheap admission
    /// checks bound lock waits independently of those filesystem validations.
    pub(crate) fn complete_l1_scan(
        &self,
        admission_check: Option<&dyn Fn() -> Result<(), Error>>,
        generation: i64,
        head: &str,
        root: &str,
        timestamp: i64,
        validate_inputs: impl Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        admission::transaction_checked(
            self,
            admission::Kind::Control,
            admission_check,
            |transaction| {
                validate_inputs()?;
                check_generation(&transaction, generation)?;
                let pending = transaction.prepare("SELECT 1 FROM files WHERE l1_pending=1 OR l1_desired IS NULL OR (l1_fingerprint IS NOT l1_desired AND l1_deferred_reason IS NULL) LIMIT 1")?.exists([])?;
                if pending {
                    return Err(Error::L1Incomplete { generation });
                }
                transaction.execute("UPDATE meta SET value=?1 WHERE key='repo_root'", [root])?;
                transaction.execute(
                    "UPDATE meta SET value=?1 WHERE key='indexed_at'",
                    [timestamp.to_string()],
                )?;
                transaction.execute("UPDATE meta SET value=?1 WHERE key='indexed_head'", [head])?;
                transaction.execute(
                    "UPDATE meta SET value=?1 WHERE key='l1_completed_generation'",
                    [generation.to_string()],
                )?;
                validate_inputs()?;
                if let Some(check) = admission_check {
                    check()?;
                }
                transaction.commit()?;
                Ok(())
            },
        )
    }

    /// Returns existing occurrence keys so a cap cannot silently evict valid old content.
    #[cfg(test)]
    pub(crate) fn file_occurrence_keys(
        &self,
        file_id: i64,
    ) -> Result<std::collections::BTreeSet<OccurrenceKey>, Error> {
        self.with_reader(|connection| {
            Ok(connection
                .prepare("SELECT content_hash,duplicate_ordinal FROM units WHERE file_id=?1")?
                .query_map([file_id], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<_, _>>()?)
        })
    }

    /// Reads retained cap evidence under the caller's original work budget and
    /// catalogue identity. Checks between rows and at return refuse late cancellation;
    /// publication must still repeat the fence under its writer lock.
    pub(crate) fn file_occurrence_keys_checked(
        &self,
        fence: Fence<'_>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<std::collections::BTreeSet<OccurrenceKey>, Error> {
        self.checked_read_snapshot(check, |transaction| {
            check_file(transaction, fence)?;
            check()?;
            let mut statement = transaction
                .prepare("SELECT content_hash,duplicate_ordinal FROM units WHERE file_id=?1")?;
            let mut rows = statement.query([fence.file_id])?;
            let mut keys = std::collections::BTreeSet::new();
            loop {
                check()?;
                let Some(row) = rows.next()? else { break };
                keys.insert((row.get(0)?, row.get(1)?));
            }
            check()?;
            Ok(keys)
        })?
    }

    /// Completes a capped L0 outcome without deleting or relabeling the retained L1 snapshot.
    /// Source/unit caps wait for changed inputs; storage deferrals retry once on
    /// each explicit catalogue pass because capacity can change without edits.
    /// Input/work checks run before mutation and immediately before commit. Optional
    /// cheap admission checks bound lock waits without repeating source validation.
    pub(crate) fn defer_file_l1(
        &self,
        admission_check: Option<&dyn Fn() -> Result<(), Error>>,
        fence: Fence<'_>,
        reason: CapDeferral,
        validate_inputs: impl Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        admission::transaction_checked(
            self,
            admission::Kind::Control,
            admission_check,
            |transaction| {
                check_file(&transaction, fence)?;
                validate_inputs()?;
                let reason = match reason {
                    CapDeferral::SourceBytes => "max_file_bytes",
                    CapDeferral::DocumentBytes => "max_doc_bytes",
                    CapDeferral::PdfPages => "max_pdf_pages",
                    CapDeferral::ImageBytes => "max_image_bytes",
                    CapDeferral::Units => "max_units_per_file",
                    CapDeferral::Storage => "storage_full",
                };
                transaction.execute(
                    "UPDATE files SET l1_pending=0,l1_deferred_reason=?1,truncated=1 WHERE id=?2",
                    params![reason, fence.file_id],
                )?;
                validate_inputs()?;
                if let Some(check) = admission_check {
                    check()?;
                }
                transaction.commit()?;
                Ok(())
            },
        )
    }

    /// Includes fresh L0-only files as well as cap-deferred retained snapshots.
    /// Oversized sources take precedence over a generic truncated-unit flag.
    pub(crate) fn source_cap_counts(
        &self,
        max_file_bytes: u64,
    ) -> Result<std::collections::BTreeMap<String, i64>, Error> {
        let cap = i64::try_from(max_file_bytes).unwrap_or(i64::MAX);
        self.with_reader(|db| Ok(db.prepare("SELECT reason,count(*) FROM (
            SELECT coalesce(l1_deferred_reason,CASE WHEN size>?1 THEN 'max_file_bytes' WHEN truncated=1 THEN 'max_units_per_file' END) AS reason FROM files
            ) WHERE reason IS NOT NULL GROUP BY reason ORDER BY reason")?.query_map([cap],|r|Ok((r.get(0)?,r.get(1)?)))?.collect::<Result<_,_>>()?))
    }

    /// Lists capped sources and whether previous searchable occurrences remain.
    pub(crate) fn l1_deferred_files(&self) -> Result<Vec<DeferredFile>, Error> {
        self.with_reader(|connection| Ok(connection.prepare("SELECT path,l1_deferred_reason,EXISTS(SELECT 1 FROM units WHERE units.file_id=files.id) FROM files WHERE l1_deferred_reason IS NOT NULL ORDER BY path")?
            .query_map([], |r| Ok(DeferredFile { path:r.get(0)?,reason:r.get(1)?,retained_snapshot:r.get(2)? }))?.collect::<Result<_,_>>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{FileKind, StoreLimits, open};
    use std::{
        error::Error as StdError,
        sync::{Arc, Barrier},
        thread,
    };

    fn observation(path: &'static str, fingerprint: &'static [u8; 32]) -> Observation<'static> {
        Observation {
            file: FileUpsert {
                path,
                blake3: &[1; 32],
                size: 0,
                mtime_ns: 1,
                lang: None,
                kind: FileKind::Other,
                git_blob: None,
                max_level: 0,
                treatment: "default",
                indexed_at: Some(1),
                needs_vlm: false,
            },
            fingerprint,
        }
    }

    fn publish(store: &Store, generation: i64, files: &[Observation<'_>]) -> Result<i64, Error> {
        store.publish_catalogue(
            None,
            Catalogue {
                discovery: None,
                max_files: u64::MAX,
                generation,
                files,
                renames: &[],
                deleted: &[],
                dirs: &[],
                complete: true,
                full: false,
            },
            || Ok(()),
        )
    }

    #[test]
    fn published_file_work_checks_identity_completion_and_every_boundary() {
        use std::cell::Cell;
        let directory = tempfile::tempdir().unwrap();
        for path in [
            std::path::Path::new(":memory:"),
            &directory.path().join("work.db"),
        ] {
            let store = open(path, Default::default()).unwrap();
            let generation = publish(&store, 0, &[observation("source", &[2; 32])]).unwrap();
            let id = store.index_snapshot().unwrap().files[0].id;
            let read = |check: &dyn Fn() -> Result<(), Error>| {
                store.l1_file_work(generation, "source", &[1; 32], &[2; 32], check)
            };
            let calls = Cell::new(0);
            let before = store.writer().unwrap().total_changes();
            assert_eq!(
                read(&|| {
                    calls.set(calls.get() + 1);
                    Ok(())
                })
                .unwrap(),
                (id, true)
            );
            for stop in 1..=calls.get() {
                let current = Cell::new(0);
                assert!(matches!(
                    read(&|| {
                        current.set(current.get() + 1);
                        if current.get() >= stop {
                            Err(Error::L1InputsChanged)
                        } else {
                            Ok(())
                        }
                    }),
                    Err(Error::L1InputsChanged)
                ));
                assert_eq!(read(&|| Ok(())).unwrap(), (id, true));
            }
            for (generation, path, hash, fingerprint) in [
                (generation - 1, "source", [1; 32], [2; 32]),
                (generation, "missing", [1; 32], [2; 32]),
                (generation, "source", [3; 32], [2; 32]),
                (generation, "source", [1; 32], [3; 32]),
            ] {
                assert!(matches!(
                    store.l1_file_work(generation, path, &hash, &fingerprint, &|| Ok(())),
                    Err(Error::L1InputsChanged | Error::L1ScanSuperseded { .. })
                ));
            }
            assert_eq!(store.writer().unwrap().total_changes(), before);
            // A cleared pending flag alone is not proof of a completed output.
            for (sql, pending) in [
                ("UPDATE files SET l1_pending=0", true),
                ("UPDATE files SET l1_fingerprint=l1_desired", false),
                ("UPDATE files SET l1_pending=1", true),
                (
                    "UPDATE files SET l1_pending=0,l1_fingerprint=NULL,l1_deferred_reason='max_file_bytes'",
                    false,
                ),
                ("UPDATE files SET l1_pending=1", true),
            ] {
                store.writer().unwrap().execute_batch(sql).unwrap();
                let before = store.writer().unwrap().total_changes();
                assert_eq!(read(&|| Ok(())).unwrap(), (id, pending));
                assert_eq!(store.writer().unwrap().total_changes(), before);
            }
            store
                .writer()
                .unwrap()
                .execute_batch("UPDATE files SET path='renamed',id=id+10")
                .unwrap();
            assert!(matches!(read(&|| Ok(())), Err(Error::L1InputsChanged)));
            assert_eq!(
                store
                    .l1_file_work(generation, "renamed", &[1; 32], &[2; 32], &|| Ok(()))
                    .unwrap(),
                (id + 10, true)
            );
            store
                .writer()
                .unwrap()
                .execute_batch("UPDATE files SET l1_desired=NULL")
                .unwrap();
            assert!(
                store
                    .l1_file_work(generation, "renamed", &[1; 32], &[2; 32], &|| Ok(()))
                    .is_err()
            );
        }
    }

    #[test]
    fn published_file_work_bounds_reader_wait_and_retries_without_writes() {
        use std::{
            sync::mpsc,
            time::{Duration, Instant},
        };
        let directory = tempfile::tempdir().unwrap();
        for path in [
            std::path::Path::new(":memory:"),
            &directory.path().join("wait.db"),
        ] {
            let store = open(path, Default::default()).unwrap();
            let generation = publish(&store, 0, &[observation("source", &[2; 32])]).unwrap();
            let before = store.writer().unwrap().total_changes();
            let held: Vec<_> = if store.readers.is_empty() {
                vec![store.writer().unwrap()]
            } else {
                store.readers.iter().map(|r| r.lock().unwrap()).collect()
            };
            std::thread::scope(|scope| {
                let (send, receive) = mpsc::channel();
                let store = &store;
                scope.spawn(move || {
                    let deadline = Instant::now() + Duration::from_millis(20);
                    let result =
                        store.l1_file_work(generation, "source", &[1; 32], &[2; 32], &|| {
                            if Instant::now() >= deadline {
                                Err(Error::L1InputsChanged)
                            } else {
                                Ok(())
                            }
                        });
                    send.send(matches!(result, Err(Error::L1InputsChanged)))
                        .unwrap();
                });
                let result = receive.recv_timeout(Duration::from_secs(2));
                drop(held);
                assert!(result.unwrap());
            });
            assert!(
                store
                    .l1_file_work(generation, "source", &[1; 32], &[2; 32], &|| Ok(()))
                    .unwrap()
                    .1
            );
            assert_eq!(store.pending_l1_paths().unwrap(), ["source"]);
            assert_eq!(store.writer().unwrap().total_changes(), before);
            for mutex in std::iter::once(&store.writer).chain(store.readers.iter()) {
                let connection = mutex.lock().unwrap();
                assert!(connection.is_autocommit());
                assert_eq!(
                    connection
                        .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                        .unwrap(),
                    5000
                );
            }
        }
    }

    #[test]
    fn checked_occurrence_keys_fence_identity_and_every_read_boundary() {
        use std::cell::Cell;
        let directory = tempfile::tempdir().unwrap();
        for path in [
            std::path::Path::new(":memory:"),
            &directory.path().join("keys.db"),
        ] {
            let store = open(path, Default::default()).unwrap();
            let generation = publish(&store, 0, &[observation("source", &[2; 32])]).unwrap();
            let file_id = store.index_snapshot().unwrap().files[0].id;
            store.writer().unwrap().execute_batch(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES (zeroblob(32),'same',1);
                 INSERT INTO units(hash,content_hash,source_key,duplicate_ordinal,file_id,kind,header,created_at,updated_at)
                 SELECT randomblob(32),zeroblob(32),path,0,id,'chunk','',1,1 FROM files;
                 INSERT INTO units(hash,content_hash,source_key,duplicate_ordinal,file_id,kind,header,created_at,updated_at)
                 SELECT randomblob(32),zeroblob(32),path,1,id,'chunk','',1,1 FROM files;"
            ).unwrap();
            let fence = Fence {
                generation,
                file_id,
                path: "source",
                content_hash: &[1; 32],
                fingerprint: &[2; 32],
            };
            let before = store.file_occurrence_keys(file_id).unwrap();
            assert_eq!(before.len(), 2);
            let calls = Cell::new(0);
            assert_eq!(
                store
                    .file_occurrence_keys_checked(fence, &|| {
                        calls.set(calls.get() + 1);
                        Ok(())
                    })
                    .unwrap(),
                before
            );
            for stop in 1..=calls.get() {
                let current = Cell::new(0);
                assert!(matches!(
                    store.file_occurrence_keys_checked(fence, &|| {
                        current.set(current.get() + 1);
                        if current.get() >= stop {
                            Err(Error::L1InputsChanged)
                        } else {
                            Ok(())
                        }
                    }),
                    Err(Error::L1InputsChanged)
                ));
                assert_eq!(
                    store
                        .file_occurrence_keys_checked(fence, &|| Ok(()))
                        .unwrap(),
                    before
                );
            }
            for stale in [
                Fence {
                    generation: generation - 1,
                    ..fence
                },
                Fence {
                    file_id: file_id + 1,
                    ..fence
                },
                Fence {
                    path: "renamed",
                    ..fence
                },
                Fence {
                    content_hash: &[3; 32],
                    ..fence
                },
                Fence {
                    fingerprint: &[3; 32],
                    ..fence
                },
            ] {
                assert!(matches!(
                    store.file_occurrence_keys_checked(stale, &|| Ok(())),
                    Err(Error::L1ScanSuperseded { .. } | Error::FileIndexSuperseded { .. })
                ));
            }
            assert_eq!(store.file_occurrence_keys(file_id).unwrap(), before);
            assert_eq!(store.l1_generation().unwrap(), generation);
            assert_eq!(store.pending_l1_paths().unwrap(), ["source"]);
        }
    }

    #[test]
    fn catalogue_file_cap_counts_retained_rows_and_rolls_back_whole_batches() {
        let store = super::super::open(":memory:", Default::default()).unwrap();
        let generation = publish(&store, 0, &[observation("retained", &[1; 32])]).unwrap();
        let before = store.index_snapshot().unwrap();
        let run = |limit, files: &[Observation<'_>]| {
            store.publish_catalogue(
                None,
                Catalogue {
                    discovery: None,
                    max_files: limit,
                    generation: store.l1_generation().unwrap(),
                    files,
                    renames: &[],
                    deleted: &[],
                    dirs: &[],
                    complete: false,
                    full: false,
                },
                || Ok(()),
            )
        };
        // A zero/lowered limit permits existing paths, but never a new path.
        run(0, &[observation("retained", &[1; 32])]).unwrap();
        for limit in [0, 1, 2] {
            let saved_generation = store.l1_generation().unwrap();
            let result = run(
                limit,
                &[
                    observation("retained", &[2; 32]),
                    observation("first", &[1; 32]),
                    observation("second", &[1; 32]),
                ],
            );
            assert!(
                matches!(result, Err(Error::FileCapacity { limit: actual }) if actual == limit)
            );
            assert_eq!(store.index_snapshot().unwrap().files, before.files);
            assert_eq!(store.l1_generation().unwrap(), saved_generation);
        }
        assert_eq!(store.l1_generation().unwrap(), generation + 1);
        // Repeated observations count once; retry after raising the cap succeeds.
        run(
            2,
            &[
                observation("first", &[1; 32]),
                observation("first", &[1; 32]),
            ],
        )
        .unwrap();
        assert_eq!(store.index_snapshot().unwrap().files.len(), 2);
    }

    #[test]
    fn catalogue_file_cap_uses_writer_state_and_accounts_for_renames_and_deletions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("code.db");
        let store = super::super::open(&path, Default::default()).unwrap();
        let second = super::super::open(&path, Default::default()).unwrap();
        let generation = publish(&store, 0, &[observation("old", &[1; 32])]).unwrap();
        let old_id = store.index_snapshot().unwrap().files[0].id;
        // Another writer changes rows without changing the scan generation.
        second
            .upsert_file(observation("unseen", &[1; 32]).file)
            .unwrap();
        let before = store.index_snapshot().unwrap();
        let run = |limit, deleted: &[String], files: &[Observation<'_>]| {
            store.publish_catalogue(
                None,
                Catalogue {
                    discovery: None,
                    max_files: limit,
                    generation,
                    files,
                    renames: &[("old", "renamed")],
                    deleted,
                    dirs: &[],
                    complete: false,
                    full: false,
                },
                || Ok(()),
            )
        };
        assert!(matches!(
            run(2, &[], &[observation("new", &[1; 32])]),
            Err(Error::FileCapacity { limit: 2 })
        ));
        assert_eq!(store.index_snapshot().unwrap().files, before.files);
        assert_eq!(store.l1_generation().unwrap(), generation);
        // Authorized deletion frees one slot; rename retains its original row.
        run(
            2,
            &["unseen".into()],
            &[
                observation("renamed", &[2; 32]),
                observation("new", &[1; 32]),
            ],
        )
        .unwrap();
        let after = store.index_snapshot().unwrap();
        assert_eq!(after.files.len(), 2);
        assert_eq!(
            after
                .files
                .iter()
                .find(|file| file.path == "renamed")
                .unwrap()
                .id,
            old_id
        );
        assert_eq!(after.dirs, before.dirs);
        assert_eq!(store.get_meta("indexed_head").unwrap().as_deref(), Some(""));
    }

    fn complete(
        store: &Store,
        generation: i64,
        path: &str,
        fingerprint: &[u8],
    ) -> Result<(), Error> {
        let file_id =
            store
                .writer()?
                .query_row("SELECT id FROM files WHERE path=?1", [path], |r| r.get(0))?;
        store.replace_file_content(
            None,
            Fence {
                generation,
                file_id,
                path,
                content_hash: &[1; 32],
                fingerprint,
            },
            &[],
            false,
            || Ok(()),
        )
    }

    #[test]
    fn l1_checked_writes_stop_at_both_writer_locks_without_validation_or_mutation() {
        use std::{
            cell::Cell,
            sync::mpsc,
            time::{Duration, Instant},
        };
        for sqlite_lock in [false, true] {
            for operation in 0..4 {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("code.db");
                let store = open(&path, StoreLimits::default()).unwrap();
                let files = [observation("a.bin", &[2; 32])];
                let generation = publish(&store, 0, &files).unwrap();
                if operation == 3 {
                    complete(&store, generation, "a.bin", &[2; 32]).unwrap();
                }
                let file_id = store.index_snapshot().unwrap().files[0].id;
                let fence = Fence {
                    generation,
                    file_id,
                    path: "a.bin",
                    content_hash: &[1; 32],
                    fingerprint: &[2; 32],
                };
                let run = |check: Option<&dyn Fn() -> Result<(), Error>>,
                           validate: &dyn Fn() -> Result<(), Error>| {
                    match operation {
                        0 => store
                            .publish_catalogue(
                                check,
                                Catalogue {
                                    discovery: None,
                                    max_files: u64::MAX,
                                    generation,
                                    files: &files,
                                    renames: &[],
                                    deleted: &[],
                                    dirs: &[],
                                    complete: false,
                                    full: false,
                                },
                                validate,
                            )
                            .map(|_| ()),
                        1 => store
                            .publish_file_content(check, fence, &[], true, validate)
                            .map(|_| ()),
                        2 => store.defer_file_l1(check, fence, CapDeferral::Units, validate),
                        _ => store.complete_l1_scan(
                            check,
                            generation,
                            "checked-head",
                            "/fixture",
                            2,
                            validate,
                        ),
                    }
                };
                let other = open(&path, StoreLimits::default()).unwrap();
                thread::scope(|scope| {
                    let mut held = if sqlite_lock {
                        other.writer().unwrap()
                    } else {
                        store.writer().unwrap()
                    };
                    let transaction = sqlite_lock.then(|| {
                        held.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                            .unwrap()
                    });
                    let (send, receive) = mpsc::channel();
                    let run = &run;
                    scope.spawn(move || {
                        let deadline = Instant::now() + Duration::from_millis(40);
                        let check = || {
                            if Instant::now() >= deadline {
                                Err(Error::L1InputsChanged)
                            } else {
                                Ok(())
                            }
                        };
                        let validations = Cell::new(0);
                        let result = run(Some(&check), &|| {
                            validations.set(validations.get() + 1);
                            Ok(())
                        });
                        send.send((
                            matches!(result, Err(Error::L1InputsChanged)),
                            validations.get(),
                        ))
                        .unwrap();
                    });
                    let result = receive.recv_timeout(Duration::from_secs(2));
                    drop(transaction);
                    drop(held);
                    assert_eq!(result.unwrap(), (true, 0));
                });
                assert_eq!(store.l1_generation().unwrap(), generation);
                assert_eq!(
                    store.pending_l1_paths().unwrap().len(),
                    usize::from(operation != 3)
                );
                assert!(store.l1_deferred_files().unwrap().is_empty());
                assert_eq!(store.get_meta("indexed_head").unwrap(), Some(String::new()));
                assert_eq!(
                    store
                        .writer()
                        .unwrap()
                        .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                        .unwrap(),
                    5000
                );
                let validations = Cell::new(0);
                run(Some(&|| Ok(())), &|| {
                    validations.set(validations.get() + 1);
                    Ok(())
                })
                .unwrap();
                assert!(validations.get() >= 2);
                assert_eq!(
                    store
                        .writer()
                        .unwrap()
                        .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                        .unwrap(),
                    5000
                );
            }
        }
    }

    #[test]
    fn l1_admission_cancellation_after_validation_rolls_back_and_retries() {
        use std::cell::Cell;
        let dir = tempfile::tempdir().unwrap();
        let store = open(dir.path().join("code.db"), StoreLimits::default()).unwrap();
        let files = [observation("a.bin", &[2; 32])];
        let generation = publish(&store, 0, &files).unwrap();
        let fence = Fence {
            generation,
            file_id: store.index_snapshot().unwrap().files[0].id,
            path: "a.bin",
            content_hash: &[1; 32],
            fingerprint: &[2; 32],
        };
        for defer in [false, true] {
            let cancelled = Cell::new(false);
            let calls = Cell::new(0);
            let check = || {
                if cancelled.get() {
                    Err(Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            };
            let validate = || {
                calls.set(calls.get() + 1);
                if calls.get() == 2 {
                    cancelled.set(true);
                }
                Ok(())
            };
            let result = if defer {
                store.defer_file_l1(Some(&check), fence, CapDeferral::Units, validate)
            } else {
                store
                    .publish_file_content(Some(&check), fence, &[], true, validate)
                    .map(|_| ())
            };
            assert!(matches!(result, Err(Error::L1InputsChanged)));
            assert_eq!(calls.get(), 2);
            assert_eq!(store.pending_l1_paths().unwrap(), ["a.bin"]);
            assert!(store.l1_deferred_files().unwrap().is_empty());
        }
        store
            .publish_file_content(Some(&|| Ok(())), fence, &[], false, || Ok(()))
            .unwrap();
        assert!(store.pending_l1_paths().unwrap().is_empty());
    }

    #[test]
    fn catalogue_work_refusal_rolls_back_rows_and_generation() {
        use std::cell::Cell;

        let database = tempfile::NamedTempFile::new().unwrap();
        let store = open(database.path(), StoreLimits::default()).unwrap();
        let old = [observation("old.bin", &[2; 32])];
        let generation = publish(&store, 0, &old).unwrap();
        complete(&store, generation, "old.bin", &[2; 32]).unwrap();
        let files = [observation("new.bin", &[3; 32])];
        let deleted = ["old.bin".to_owned()];
        // Entry, deletion, insertion, and final precommit admission can each refuse.
        // A late refusal must restore both the deleted row and its completed state.
        for refuse_at in 1..=4 {
            let calls = Cell::new(0);
            let result = store.publish_catalogue(
                None,
                Catalogue {
                    discovery: None,
                    max_files: u64::MAX,
                    generation,
                    files: &files,
                    renames: &[],
                    deleted: &deleted,
                    dirs: &[],
                    complete: false,
                    full: false,
                },
                || {
                    calls.set(calls.get() + 1);
                    if calls.get() == refuse_at {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(matches!(result, Err(Error::L1InputsChanged)));
            assert_eq!(calls.get(), refuse_at);
            assert_eq!(store.l1_generation().unwrap(), generation);
            assert!(store.pending_l1_paths().unwrap().is_empty());
            let snapshot = store.index_snapshot().unwrap();
            assert_eq!(snapshot.files.len(), 1);
            assert_eq!(snapshot.files[0].path, "old.bin");
        }
        assert_eq!(publish(&store, generation, &old).unwrap(), generation + 1);
    }

    #[test]
    fn file_and_cap_work_refusal_preserves_pending_completion_and_head() {
        use std::cell::Cell;

        let database = tempfile::NamedTempFile::new().unwrap();
        let store = open(database.path(), StoreLimits::default()).unwrap();
        let files = [observation("a.bin", &[2; 32])];
        let generation = publish(&store, 0, &files).unwrap();
        let file_id = store.index_snapshot().unwrap().files[0].id;
        let fence = Fence {
            generation,
            file_id,
            path: "a.bin",
            content_hash: &[1; 32],
            fingerprint: &[2; 32],
        };
        for defer in [false, true] {
            for refuse_at in [1, 2] {
                let calls = Cell::new(0);
                let check = || {
                    calls.set(calls.get() + 1);
                    if calls.get() == refuse_at {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                };
                let result = if defer {
                    store.defer_file_l1(None, fence, CapDeferral::Units, check)
                } else {
                    store
                        .publish_file_content(None, fence, &[], true, check)
                        .map(|_| ())
                };
                assert!(matches!(result, Err(Error::L1InputsChanged)));
                assert_eq!(calls.get(), refuse_at);
                assert_eq!(store.pending_l1_paths().unwrap(), ["a.bin"]);
                assert!(store.l1_deferred_files().unwrap().is_empty());
                let flags: (bool, Option<Vec<u8>>) = store
                    .writer()
                    .unwrap()
                    .query_row(
                        "SELECT truncated,l1_fingerprint FROM files WHERE id=?1",
                        [file_id],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .unwrap();
                assert_eq!(flags, (false, None));
            }
        }
        complete(&store, generation, "a.bin", &[2; 32]).unwrap();
        let calls = Cell::new(0);
        assert!(matches!(
            store.complete_l1_scan(None, generation, "new-head", "/fixture", 2, || {
                calls.set(calls.get() + 1);
                if calls.get() == 2 {
                    Err(Error::L1InputsChanged)
                } else {
                    Ok(())
                }
            }),
            Err(Error::L1InputsChanged)
        ));
        assert_eq!(calls.get(), 2);
        assert_eq!(store.get_meta("indexed_head").unwrap(), Some(String::new()));
        store
            .complete_l1_scan(None, generation, "new-head", "/fixture", 2, || Ok(()))
            .unwrap();
        assert_eq!(
            store.get_meta("indexed_head").unwrap(),
            Some("new-head".into())
        );
    }

    #[test]
    fn catalogue_and_completion_reject_superseded_generations() -> Result<(), Box<dyn StdError>> {
        let database = tempfile::NamedTempFile::new()?;
        let store = open(database.path(), StoreLimits::default())?;
        let files = [
            observation("a.bin", &[2; 32]),
            observation("b.bin", &[3; 32]),
        ];
        let first = publish(&store, 0, &files)?;
        assert_eq!(first, 1);
        assert_eq!(store.get_meta("indexed_head")?, Some(String::new()));
        assert!(matches!(
            store.complete_l1_scan(None, first, "one", "/fixture", 1, || Ok(())),
            Err(Error::L1Incomplete { .. })
        ));
        complete(&store, first, "a.bin", &[2; 32])?;
        let second = publish(&store, first, &files)?;
        assert_eq!(store.pending_l1_paths()?, ["b.bin"]);
        assert!(matches!(
            complete(&store, first, "b.bin", &[3; 32]),
            Err(Error::L1ScanSuperseded { .. })
        ));
        assert!(matches!(
            publish(&store, first, &[observation("wrong.bin", &[9; 32])]),
            Err(Error::L1ScanSuperseded { .. })
        ));
        assert_eq!(store.counts()?.files, 2);
        complete(&store, second, "b.bin", &[3; 32])?;
        store.complete_l1_scan(None, second, "two", "/fixture", 2, || Ok(()))?;
        assert!(matches!(
            store.complete_l1_scan(None, first, "one", "/fixture", 1, || Ok(())),
            Err(Error::L1ScanSuperseded { .. })
        ));
        assert!(matches!(
            store.update_graph_ranks(first, &[], &[]),
            Err(Error::L1ScanSuperseded { .. })
        ));
        assert_eq!(store.get_meta("indexed_head")?, Some("two".to_owned()));
        assert_eq!(store.get_meta("indexed_at")?, Some("2".to_owned()));

        let next_files = [
            observation("a.bin", &[4; 32]),
            observation("b.bin", &[3; 32]),
        ];
        let third = publish(&store, second, &next_files)?;
        assert_eq!(store.pending_l1_paths()?, ["a.bin"]); // only changed effective configuration
        assert!(matches!(
            complete(&store, third, "a.bin", &[2; 32]),
            Err(Error::FileIndexSuperseded { .. })
        ));
        complete(&store, third, "a.bin", &[4; 32])?;
        assert!(matches!(
            store.complete_l1_scan(None, third, "three", "/fixture", 3, || Err(
                Error::L1InputsChanged
            )),
            Err(Error::L1InputsChanged)
        ));
        assert_eq!(store.get_meta("indexed_head")?, Some("two".to_owned()));
        store.complete_l1_scan(None, third, "three", "/fixture", 3, || Ok(()))?;
        assert_eq!(
            store.get_meta("l1_completed_generation")?,
            Some(third.to_string())
        );
        Ok(())
    }

    #[test]
    fn competing_catalogue_writers_cannot_interleave() -> Result<(), Box<dyn StdError>> {
        let database = tempfile::NamedTempFile::new()?;
        let store = Arc::new(open(database.path(), StoreLimits::default())?);
        let barrier = Arc::new(Barrier::new(3));
        let handles = ["a.bin", "b.bin"].map(|path| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                match publish(&store, 0, &[observation(path, &[2; 32])]) {
                    Ok(1) => true,
                    Err(Error::L1ScanSuperseded { generation: 0 }) => false,
                    result => panic!("unexpected publication result: {result:?}"),
                }
            })
        });
        barrier.wait();
        let successes = handles
            .into_iter()
            .map(|h| usize::from(h.join().expect("writer thread")))
            .sum::<usize>();
        assert_eq!(successes, 1);
        assert_eq!(store.counts()?.files, 1);
        assert_eq!(store.l1_generation()?, 1);
        Ok(())
    }
    #[test]
    fn wal_refusal_preserves_catalogue_and_control_fences() -> Result<(), Box<dyn StdError>> {
        for page_size in [4096, 65536] {
            for operation in ["catalogue", "completion", "deferral"] {
                let dir = tempfile::tempdir()?;
                let path = dir.path().join("code.db");
                super::super::register_vector_extension()?;
                {
                    let db = rusqlite::Connection::open(&path)?;
                    db.pragma_update(None, "page_size", page_size)?;
                    db.execute_batch("VACUUM")?;
                }
                let store = open(
                    &path,
                    StoreLimits {
                        wal_max_bytes: 8 * 1024 * 1024,
                        ..Default::default()
                    },
                )?;
                let generation = publish(
                    &store,
                    store.l1_generation()?,
                    &[observation("a.bin", &[1; 32])],
                )?;
                complete(&store, generation, "a.bin", &[1; 32])?;
                store.complete_l1_scan(None, generation, "old", "root", 1, || Ok(()))?;
                let generation = publish(&store, generation, &[observation("a.bin", &[2; 32])])?;
                if operation == "completion" {
                    complete(&store, generation, "a.bin", &[2; 32])?;
                }
                let file_id = store
                    .writer()?
                    .query_row("SELECT id FROM files", [], |r| r.get(0))?;
                let fence = Fence {
                    generation,
                    file_id,
                    path: "a.bin",
                    content_hash: &[1; 32],
                    fingerprint: &[2; 32],
                };
                let trigger = if operation == "completion" {
                    "CREATE TRIGGER refuse_l1 AFTER UPDATE ON meta WHEN NEW.key='indexed_head' BEGIN INSERT INTO payload VALUES(zeroblob(16777216)); END;"
                } else {
                    "CREATE TRIGGER refuse_l1 AFTER UPDATE ON files BEGIN INSERT INTO payload VALUES(zeroblob(16777216)); END;"
                };
                store
                    .writer()?
                    .execute_batch("CREATE TABLE payload(value BLOB)")?;
                store.writer()?.execute_batch(trigger)?;
                super::super::wal::checkpoint(&store)?;
                let result = match operation {
                    "catalogue" => {
                        publish(&store, generation, &[observation("a.bin", &[3; 32])]).map(|_| ())
                    }
                    "completion" => {
                        store.complete_l1_scan(None, generation, "new", "root", 2, || Ok(()))
                    }
                    _ => store.defer_file_l1(None, fence, CapDeferral::Units, || Ok(())),
                };
                assert!(matches!(result, Err(Error::WalCapacity { .. })));
                assert_eq!(store.l1_generation()?, generation);
                let db = store.writer()?;
                assert_eq!(
                    db.query_row("SELECT value FROM meta WHERE key='indexed_head'", [], |r| r
                        .get::<_, String>(0))?,
                    "old"
                );
                assert_eq!(
                    db.query_row("SELECT l1_desired FROM files", [], |r| r
                        .get::<_, Vec<u8>>(0))?,
                    vec![2; 32]
                );
                assert_eq!(
                    db.query_row("SELECT count(*) FROM payload", [], |r| r.get::<_, i64>(0))?,
                    0
                );
                assert_eq!(
                    db.query_row("SELECT l1_pending FROM files", [], |r| r.get::<_, bool>(0))?,
                    operation != "completion"
                );
                assert!(super::super::wal::wal_bytes(&db)? <= 8 * 1024 * 1024);
                db.execute_batch("DROP TRIGGER refuse_l1")?;
                drop(db);
                match operation {
                    "catalogue" => {
                        publish(&store, generation, &[observation("a.bin", &[3; 32])])?;
                    }
                    "completion" => {
                        store.complete_l1_scan(None, generation, "new", "root", 2, || Ok(()))?
                    }
                    _ => store.defer_file_l1(None, fence, CapDeferral::Units, || Ok(()))?,
                }
            }
        }
        Ok(())
    }
    #[test]
    fn atomic_catalogue_can_use_more_than_a_small_queue_reservation()
    -> Result<(), Box<dyn StdError>> {
        let dir = tempfile::tempdir()?;
        let store = open(dir.path().join("code.db"), StoreLimits::default())?;
        store.writer()?.execute_batch("CREATE TABLE payload(value BLOB); CREATE TRIGGER expand_catalogue AFTER INSERT ON files BEGIN INSERT INTO payload VALUES(zeroblob(41943040)); END;")?;
        super::super::wal::checkpoint(&store)?;
        publish(
            &store,
            store.l1_generation()?,
            &[observation("a.bin", &[1; 32])],
        )?;
        assert_eq!(
            store
                .writer()?
                .query_row("SELECT length(value) FROM payload", [], |r| r
                    .get::<_, i64>(0))?,
            40 * 1024 * 1024
        );
        assert!(
            super::super::wal::wal_bytes(&*store.writer()?)?
                <= StoreLimits::default().wal_max_bytes
        );
        Ok(())
    }
}
