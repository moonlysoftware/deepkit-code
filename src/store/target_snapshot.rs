//! Bounded catalogue reads for query-owned L1 repair; see `v1.md` §3.1 and §4.6.

use super::{Error, IndexSnapshot, IndexedDir, IndexedFile, Store};
use rusqlite::OptionalExtension;
use std::{sync::TryLockError, sync::atomic::Ordering, thread, time::Duration};

/// Catalogue authority without inventing an occurrence for an empty or L0 file.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CatalogueSource {
    pub(crate) file: IndexedFile,
    pub(crate) pending: bool,
    pub(crate) desired: Option<Vec<u8>>,
    pub(crate) completed: Option<Vec<u8>>,
    pub(crate) deferred: Option<String>,
    pub(crate) has_occurrences: bool,
}

impl Store {
    /// Reads indexed HEAD and pending L1 coverage in one caller-bounded snapshot.
    /// Request inspection avoids scanning the unrelated full-status tables.
    pub(crate) fn inspection_checked(
        &self,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(Option<String>, i64), Error> {
        self.checked_read_snapshot(check, |transaction| {
            let indexed_head = transaction
                .query_row(
                    "SELECT value FROM meta WHERE key='indexed_head'",
                    [],
                    |row| row.get(0),
                )
                .optional()?;
            check()?;
            let pending = transaction.query_row(
                "SELECT COUNT(*) FROM files WHERE l1_pending=1",
                [],
                |row| row.get(0),
            )?;
            Ok((indexed_head, pending))
        })?
    }

    /// Reads source identity, completion and occurrence presence coherently under
    /// the caller's work budget. A settled deferral is not completed current L1.
    pub(crate) fn catalogue_source_checked(
        &self,
        path: &str,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Option<CatalogueSource>, Error> {
        self.checked_read_snapshot(check, |transaction| {
            Ok(transaction
                .query_row(
                    "SELECT id,path,blake3,size,mtime_ns,lang,kind,git_blob,max_level,treatment,
                        l1_pending,l1_desired,l1_fingerprint,l1_deferred_reason,
                        EXISTS(SELECT 1 FROM units WHERE file_id=files.id)
                 FROM files WHERE path=?1",
                    [path],
                    |row| {
                        Ok(CatalogueSource {
                            file: IndexedFile {
                                id: row.get(0)?,
                                path: row.get(1)?,
                                blake3: row.get(2)?,
                                size: row.get(3)?,
                                mtime_ns: row.get(4)?,
                                lang: row.get(5)?,
                                kind: row.get(6)?,
                                git_blob: row.get(7)?,
                                max_level: row.get(8)?,
                                treatment: row.get(9)?,
                            },
                            pending: row.get(10)?,
                            desired: row.get(11)?,
                            completed: row.get(12)?,
                            deferred: row.get(13)?,
                            has_occurrences: row.get(14)?,
                        })
                    },
                )
                .optional()?)
        })?
    }

    /// Runs a short snapshot with cooperative local admission and no SQLite busy
    /// sleeping. The caller supplies its original deadline/cancellation check and
    /// must check inside long callbacks; individual SQLite calls are not preempted.
    /// Callbacks must not write or retain SQLite handles. Their owned result is
    /// discarded if the final check refuses it, and the busy policy is restored.
    pub(crate) fn checked_read_snapshot<T>(
        &self,
        check: &dyn Fn() -> Result<(), Error>,
        operation: impl FnOnce(&rusqlite::Transaction<'_>) -> T,
    ) -> Result<T, Error> {
        check()?;
        let (mutex, label) = if self.readers.is_empty() {
            (&self.writer, "writer")
        } else {
            let index = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.readers.len();
            (&self.readers[index], "reader")
        };
        let connection = loop {
            check()?;
            match mutex.try_lock() {
                Ok(connection) => break connection,
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(Error::LockPoisoned { connection: label });
                }
            }
        };
        check()?;
        let timeout: i64 = connection.query_row("PRAGMA busy_timeout", [], |r| r.get(0))?;
        connection.busy_timeout(Duration::ZERO)?;
        let result = (|| {
            check()?;
            let transaction = connection.unchecked_transaction()?;
            let value = operation(&transaction);
            check()?;
            Ok(value)
        })();
        // The read transaction has dropped before restoring the reusable connection.
        connection.pragma_update(None, "busy_timeout", timeout)?;
        result
    }

    /// Reads the complete retained catalogue after its generation in one snapshot.
    /// No current file cap truncates old rows: absence and rename decisions require
    /// all retained paths. Checks between rows bound cooperative work; individual
    /// SQLite calls and allocations are not preempted. This does not grant current
    /// filesystem or completed-HEAD authority.
    pub(crate) fn catalogue_snapshot(
        &self,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(i64, IndexSnapshot), Error> {
        self.checked_read_snapshot(check, |transaction| {
            let generation = transaction.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
                [],
                |r| r.get(0),
            )?;
            check()?;
            let mut snapshot = IndexSnapshot::default();
            let mut statement = transaction.prepare(
                "SELECT id, path, blake3, size, mtime_ns, lang, kind, git_blob, max_level, treatment FROM files ORDER BY path",
            )?;
            let mut rows = statement.query([])?;
            loop {
                check()?;
                let Some(row) = rows.next()? else { break };
                snapshot.files.push(IndexedFile {
                    id: row.get(0)?, path: row.get(1)?, blake3: row.get(2)?,
                    size: row.get(3)?, mtime_ns: row.get(4)?, lang: row.get(5)?,
                    kind: row.get(6)?, git_blob: row.get(7)?, max_level: row.get(8)?,
                    treatment: row.get(9)?,
                });
            }
            check()?;
            let mut statement = transaction.prepare(
                "SELECT path, merkle, file_count FROM dirs ORDER BY path",
            )?;
            let mut rows = statement.query([])?;
            loop {
                check()?;
                let Some(row) = rows.next()? else { break };
                snapshot.dirs.push(IndexedDir {
                    path: row.get(0)?, merkle: row.get(1)?, file_count: row.get(2)?,
                });
            }
            check()?;
            Ok((generation, snapshot))
        })?
    }

    /// Reads at most one existing file and its scan generation in one snapshot.
    /// Cooperatively bounds local admission and disables SQLite busy sleeping for
    /// this read only. SQLite contention refuses the attempt; callers may retry
    /// under their original budget. No directory or unrelated file is loaded.
    pub(crate) fn target_snapshot(
        &self,
        path: &str,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(i64, IndexSnapshot), Error> {
        self.checked_read_snapshot(check, |transaction| {
            let generation = transaction.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='l1_scan_generation'",
                [],
                |r| r.get(0),
            )?;
            check()?;
            let file = transaction.query_row(
                "SELECT id, path, blake3, size, mtime_ns, lang, kind, git_blob, max_level, treatment FROM files WHERE path=?1",
                [path],
                |row| Ok(IndexedFile {
                    id: row.get(0)?, path: row.get(1)?, blake3: row.get(2)?,
                    size: row.get(3)?, mtime_ns: row.get(4)?, lang: row.get(5)?,
                    kind: row.get(6)?, git_blob: row.get(7)?, max_level: row.get(8)?,
                    treatment: row.get(9)?,
                }),
            ).optional()?;
            check()?;
            Ok((
                generation,
                IndexSnapshot {
                    files: file.into_iter().collect(),
                    dirs: Vec::new(),
                },
            ))
        })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, path::Path, sync::mpsc, time::Instant};

    fn fixture(path: &Path) -> Store {
        let store = super::super::open(path, Default::default()).unwrap();
        store
            .writer()
            .unwrap()
            .execute_batch(
                "INSERT INTO files(path,blake3,size,mtime_ns,kind,max_level,treatment)
             VALUES ('target.rs',zeroblob(32),10,1,'code',1,'code'),
                    ('unrelated.rs',zeroblob(32),20,1,'code',1,'code');",
            )
            .unwrap();
        store
    }

    #[test]
    fn inspection_snapshot_checks_every_boundary_and_retains_coherent_coverage() {
        let dir = tempfile::tempdir().unwrap();
        for path in [Path::new(":memory:"), &dir.path().join("inspection.db")] {
            let store = fixture(path);
            assert_eq!(
                store.inspection_checked(&|| Ok(())).unwrap(),
                (Some(String::new()), 2)
            );
            store
                .writer()
                .unwrap()
                .execute("DELETE FROM meta WHERE key='indexed_head'", [])
                .unwrap();
            assert_eq!(store.inspection_checked(&|| Ok(())).unwrap(), (None, 2));
            store.set_meta("indexed_head", "old").unwrap();
            let expected = (Some("old".to_owned()), 2);
            let changes = store.writer().unwrap().total_changes();
            let count = Cell::new(0);
            assert_eq!(
                store
                    .inspection_checked(&|| {
                        count.set(count.get() + 1);
                        Ok(())
                    })
                    .unwrap(),
                expected
            );
            for stop in 1..=count.get() {
                let calls = Cell::new(0);
                assert!(
                    matches!(
                        store.inspection_checked(&|| {
                            calls.set(calls.get() + 1);
                            if calls.get() == stop {
                                Err(Error::L1InputsChanged)
                            } else {
                                Ok(())
                            }
                        }),
                        Err(Error::L1InputsChanged)
                    ),
                    "boundary {stop}"
                );
                assert_eq!(store.inspection_checked(&|| Ok(())).unwrap(), expected);
            }
            assert_eq!(store.writer().unwrap().total_changes(), changes);
            if !store.readers.is_empty() {
                let calls = Cell::new(0);
                let snapshot = store
                    .inspection_checked(&|| {
                        calls.set(calls.get() + 1);
                        // First SELECT established the snapshot; publish another generation.
                        if calls.get() == 5 {
                            store.writer().unwrap().execute_batch(
                            "BEGIN IMMEDIATE; UPDATE meta SET value='new' WHERE key='indexed_head';
                             UPDATE files SET l1_pending=0; COMMIT;"
                        ).unwrap();
                        }
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(snapshot, expected);
                assert_eq!(
                    store.inspection_checked(&|| Ok(())).unwrap(),
                    (Some("new".into()), 0)
                );
            }
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
    fn scoped_search_readers_release_contended_pool_and_restore_wait_policy() {
        let dir = tempfile::tempdir().unwrap();
        let store = fixture(&dir.path().join("search.db"));
        let before = store.index_snapshot().unwrap();
        let changes = store.writer().unwrap().total_changes();
        for operation in 0..6 {
            let held: Vec<_> = store.readers.iter().map(|r| r.lock().unwrap()).collect();
            assert!(!held.is_empty());
            thread::scope(|scope| {
                let (send, receive) = mpsc::channel();
                let store = &store;
                scope.spawn(move || {
                    let deadline = Instant::now() + Duration::from_millis(20);
                    let check = || {
                        if Instant::now() >= deadline {
                            Err(Error::L1InputsChanged)
                        } else {
                            Ok(())
                        }
                    };
                    let result = match operation {
                        0 => store
                            .exact_name_units_checked("missing", 10, None, &[], &check)
                            .map(|_| ()),
                        1 => store
                            .query_fts_scoped_checked("missing", 10, None, &[], &check)
                            .map(|_| ()),
                        2 => store.symbol_seeds_checked(&[1], 20, &check).map(|_| ()),
                        4 => store.units_by_ids_checked(&[1], &check).map(|_| ()),
                        5 => store.inspection_checked(&check).map(|_| ()),
                        _ => store
                            .graph_neighbours_checked(&[1], 100, None, &[], &check)
                            .map(|_| ()),
                    };
                    send.send(matches!(result, Err(Error::L1InputsChanged)))
                        .unwrap();
                });
                let result = receive.recv_timeout(Duration::from_secs(2));
                drop(held);
                assert!(result.unwrap());
            });
            assert!(
                store
                    .exact_name_units_checked("missing", 10, None, &[], &|| Ok(()))
                    .unwrap()
                    .is_empty()
            );
            assert!(
                store
                    .query_fts_scoped_checked("missing", 10, None, &[], &|| Ok(()))
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(
            store
                .symbol_seeds_checked(&[1], 20, &|| Ok(()))
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .graph_neighbours_checked(&[1], 100, None, &[], &|| Ok(()))
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .units_by_ids_checked(&[1], &|| Ok(()))
                .unwrap()
                .is_empty()
        );
        for mutex in &store.readers {
            let connection = mutex.lock().unwrap();
            assert!(connection.is_autocommit());
            assert_eq!(
                connection
                    .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                    .unwrap(),
                5000
            );
        }
        assert_eq!(store.index_snapshot().unwrap(), before);
        assert_eq!(store.writer().unwrap().total_changes(), changes);
    }

    #[test]
    fn catalogue_authority_keeps_completion_and_occurrences_coherent_and_cancellable() {
        let dir = tempfile::tempdir().unwrap();
        for path in [Path::new(":memory:"), &dir.path().join("catalogue.db")] {
            let store = fixture(path);
            let before = store
                .catalogue_source_checked("target.rs", &|| Ok(()))
                .unwrap()
                .unwrap();
            assert!(before.pending);
            assert!(before.completed.is_none());
            assert!(!before.has_occurrences);
            assert!(
                store
                    .catalogue_source_checked("missing", &|| Ok(()))
                    .unwrap()
                    .is_none()
            );
            let calls = Cell::new(0);
            store
                .catalogue_source_checked("target.rs", &|| {
                    calls.set(calls.get() + 1);
                    Ok(())
                })
                .unwrap();
            for boundary in 1..=calls.get() {
                let count = Cell::new(0);
                assert!(matches!(
                    store.catalogue_source_checked("target.rs", &|| {
                        count.set(count.get() + 1);
                        if count.get() == boundary {
                            Err(Error::L1InputsChanged)
                        } else {
                            Ok(())
                        }
                    }),
                    Err(Error::L1InputsChanged)
                ));
                assert_eq!(
                    store
                        .catalogue_source_checked("target.rs", &|| Ok(()))
                        .unwrap()
                        .unwrap(),
                    before
                );
            }
            let held: Vec<_> = if store.readers.is_empty() {
                vec![store.writer().unwrap()]
            } else {
                store.readers.iter().map(|r| r.lock().unwrap()).collect()
            };
            let started = Instant::now();
            assert!(matches!(
                store.catalogue_source_checked("target.rs", &|| {
                    if started.elapsed() >= Duration::from_millis(20) {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                }),
                Err(Error::L1InputsChanged)
            ));
            drop(held);
            assert!(started.elapsed() < Duration::from_secs(1));
            store.writer().unwrap().execute_batch(
                "UPDATE files SET l1_pending=0,l1_desired=zeroblob(32),l1_deferred_reason='storage_full' WHERE path='target.rs';"
            ).unwrap();
            let deferred = store
                .catalogue_source_checked("target.rs", &|| Ok(()))
                .unwrap()
                .unwrap();
            assert!(!deferred.pending);
            assert!(deferred.completed.is_none());
            assert_eq!(deferred.deferred.as_deref(), Some("storage_full"));
            assert_ne!(deferred.desired, deferred.completed);
            store.writer().unwrap().execute_batch(
                "UPDATE files SET l1_fingerprint=l1_desired,l1_deferred_reason=NULL;
                 INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'source',1);
                 INSERT INTO units(hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
                 SELECT zeroblob(32),zeroblob(32),path,id,'chunk','header',1,1 FROM files WHERE path='target.rs';"
            ).unwrap();
            let completed = store
                .catalogue_source_checked("target.rs", &|| Ok(()))
                .unwrap()
                .unwrap();
            assert_eq!(completed.desired, completed.completed);
            assert!(completed.has_occurrences);
            if !store.readers.is_empty() {
                let calls = Cell::new(0);
                let snapshot = store.catalogue_source_checked("target.rs", &|| {
                    calls.set(calls.get()+1);
                    if calls.get() == 5 {
                        store.writer().unwrap().execute_batch(
                            "UPDATE files SET l1_pending=1,blake3=randomblob(32); DELETE FROM units;"
                        ).unwrap();
                    }
                    Ok(())
                }).unwrap().unwrap();
                assert_eq!(snapshot, completed);
                let next = store
                    .catalogue_source_checked("target.rs", &|| Ok(()))
                    .unwrap()
                    .unwrap();
                assert!(next.pending);
                assert!(!next.has_occurrences);
                assert_ne!(next.file.blake3, snapshot.file.blake3);
            }
        }
    }

    #[test]
    fn open_reads_check_every_boundary_preserve_locators_and_retry_after_contention() {
        let dir = tempfile::tempdir().unwrap();
        for path in [Path::new(":memory:"), &dir.path().join("open.db")] {
            let store = fixture(path);
            store.writer().unwrap().execute_batch(
                "UPDATE files SET l1_pending=0,l1_fingerprint=zeroblob(32),l1_desired=zeroblob(32),l1_deferred_reason='cap';
                 INSERT INTO unit_contents(hash,text,tokens) VALUES (zeroblob(32),'source',1);
                 INSERT INTO units(hash,content_hash,source_key,file_id,kind,header,line_start,line_end,created_at,updated_at)
                 SELECT zeroblob(32),zeroblob(32),path,id,'chunk','header',3,7,1,1 FROM files WHERE path='target.rs';
                 INSERT INTO symbols(file_id,unit_id,name,kind,line_start,line_end)
                 SELECT file_id,id,'target','function',2,9 FROM units;"
            ).unwrap();
            let id = store
                .unit_by_path("target.rs", None, None)
                .unwrap()
                .unwrap()
                .id;
            let operations = |op, check: &dyn Fn() -> Result<(), Error>| -> Result<String, Error> {
                Ok(match op {
                    0 => format!("{:?}", store.unit_by_id_checked(id, check)?),
                    1 => format!(
                        "{:?}",
                        store.unit_by_public_id_checked("u_000000000000", check)?
                    ),
                    2 => format!(
                        "{:?}",
                        store.unit_by_path_checked("target.rs", Some(4), Some(5), check)?
                    ),
                    3 => format!("{:?}", store.tool_symbols_checked(check)?),
                    4 => format!("{:?}", store.open_source_checked("u_000000000000", check)?),
                    5 => format!(
                        "{:?}",
                        store.open_deferred_reason(Some("target.rs"), check)?
                    ),
                    6 => format!("{:?}", store.query_fts_checked("target", 200, check)?),
                    _ => format!("{:?}", store.map_files_checked(check)?),
                })
            };
            let changes = store.writer().unwrap().total_changes();
            for op in 0..8 {
                let count = Cell::new(0);
                let expected = operations(op, &|| {
                    count.set(count.get() + 1);
                    Ok(())
                })
                .unwrap();
                assert_ne!(expected, "None");
                for stop in 1..=count.get() {
                    let calls = Cell::new(0);
                    assert!(
                        matches!(
                            operations(op, &|| {
                                calls.set(calls.get() + 1);
                                if calls.get() == stop {
                                    Err(Error::L1InputsChanged)
                                } else {
                                    Ok(())
                                }
                            }),
                            Err(Error::L1InputsChanged)
                        ),
                        "op {op}, boundary {stop}"
                    );
                    assert_eq!(operations(op, &|| Ok(())).unwrap(), expected);
                }
                let held: Vec<_> = if store.readers.is_empty() {
                    vec![store.writer().unwrap()]
                } else {
                    store.readers.iter().map(|r| r.lock().unwrap()).collect()
                };
                thread::scope(|scope| {
                    let (send, receive) = mpsc::channel();
                    let operation = &operations;
                    scope.spawn(move || {
                        let deadline = Instant::now() + Duration::from_millis(10);
                        send.send(matches!(
                            operation(op, &|| {
                                if Instant::now() >= deadline {
                                    Err(Error::L1InputsChanged)
                                } else {
                                    Ok(())
                                }
                            }),
                            Err(Error::L1InputsChanged)
                        ))
                        .unwrap();
                    });
                    let result = receive.recv_timeout(Duration::from_secs(2));
                    drop(held);
                    assert!(result.unwrap());
                });
                assert_eq!(operations(op, &|| Ok(())).unwrap(), expected);
            }
            assert_eq!(store.writer().unwrap().total_changes(), changes);
            let unit = store.unit_by_id_checked(id, &|| Ok(())).unwrap().unwrap();
            assert_eq!((unit.line_start, unit.line_end), (Some(3), Some(7)));
            assert_eq!(unit.text, "source");
            assert!(
                store
                    .unit_by_path_checked("missing", None, None, &|| Ok(()))
                    .unwrap()
                    .is_none()
            );
            store
                .writer()
                .unwrap()
                .execute("UPDATE files SET l1_pending=1", [])
                .unwrap();
            assert!(
                store
                    .open_source_checked("u_000000000000", &|| Ok(()))
                    .unwrap()
                    .is_none()
            );
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
    fn catalogue_snapshot_keeps_all_rows_and_generation_coherent() {
        let dir = tempfile::tempdir().unwrap();
        let store = fixture(&dir.path().join("catalogue.db"));
        store.writer().unwrap().execute_batch(
            "INSERT INTO dirs(path,merkle,file_count) VALUES ('',zeroblob(32),2),('nested',zeroblob(32),1);"
        ).unwrap();
        let before = store.index_snapshot().unwrap();
        let generation = store.l1_generation().unwrap();
        let calls = Cell::new(0);
        let (actual, snapshot) = store
            .catalogue_snapshot(&|| {
                calls.set(calls.get() + 1);
                // The generation SELECT has established the read snapshot.
                if calls.get() == 5 {
                    store
                        .writer()
                        .unwrap()
                        .execute_batch(
                            "BEGIN IMMEDIATE;
                     UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='l1_scan_generation';
                     UPDATE files SET size=99;
                     DELETE FROM dirs WHERE path='nested'; COMMIT;",
                        )
                        .unwrap();
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(actual, generation);
        assert_eq!(snapshot, before);
        let (next, snapshot) = store.catalogue_snapshot(&|| Ok(())).unwrap();
        assert_eq!(next, generation + 1);
        assert_eq!(snapshot, store.index_snapshot().unwrap());
        assert!(snapshot.files.iter().all(|f| f.size == 99));
        assert_eq!(snapshot.dirs.len(), 1);
    }

    #[test]
    fn catalogue_snapshot_refuses_every_boundary_and_contended_admission_then_retries() {
        let dir = tempfile::tempdir().unwrap();
        for path in [Path::new(":memory:"), &dir.path().join("catalogue.db")] {
            let store = fixture(path);
            store
                .writer()
                .unwrap()
                .execute_batch(
                    "INSERT INTO dirs(path,merkle,file_count) VALUES ('',zeroblob(32),2);",
                )
                .unwrap();
            let before = store.index_snapshot().unwrap();
            let changes = store.writer().unwrap().total_changes();
            let count = Cell::new(0);
            store
                .catalogue_snapshot(&|| {
                    count.set(count.get() + 1);
                    Ok(())
                })
                .unwrap();
            for stop in 1..=count.get() {
                let calls = Cell::new(0);
                assert!(
                    matches!(
                        store.catalogue_snapshot(&|| {
                            calls.set(calls.get() + 1);
                            if calls.get() == stop {
                                Err(Error::L1InputsChanged)
                            } else {
                                Ok(())
                            }
                        }),
                        Err(Error::L1InputsChanged)
                    ),
                    "boundary {stop}"
                );
            }
            let held: Vec<_> = if store.readers.is_empty() {
                vec![store.writer().unwrap()]
            } else {
                store.readers.iter().map(|r| r.lock().unwrap()).collect()
            };
            thread::scope(|scope| {
                let (send, receive) = mpsc::channel();
                let store = &store;
                scope.spawn(move || {
                    let deadline = Instant::now() + Duration::from_millis(20);
                    let result = store.catalogue_snapshot(&|| {
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
            assert_eq!(store.catalogue_snapshot(&|| Ok(())).unwrap().1, before);
            assert_eq!(store.writer().unwrap().total_changes(), changes);
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
    fn checked_snapshot_refuses_before_callback_and_discards_late_results() {
        let dir = tempfile::tempdir().unwrap();
        for path in [Path::new(":memory:"), &dir.path().join("checked.db")] {
            let store = fixture(path);
            let called = Cell::new(false);
            assert!(matches!(
                store.checked_read_snapshot(&|| Err(Error::L1InputsChanged), |_| {
                    called.set(true);
                }),
                Err(Error::L1InputsChanged)
            ));
            assert!(!called.get());
            let cancelled = Cell::new(false);
            assert!(matches!(
                store.checked_read_snapshot(
                    &|| if cancelled.get() {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    },
                    |tx| {
                        assert_eq!(
                            tx.query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                                .unwrap(),
                            0
                        );
                        cancelled.set(true);
                        "must not escape"
                    }
                ),
                Err(Error::L1InputsChanged)
            ));
            // Exercise every reusable connection after late refusal.
            for _ in 0..store.readers.len().max(1) {
                assert_eq!(
                    store
                        .read_snapshot(
                            |tx| tx.query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                        )
                        .unwrap()
                        .unwrap(),
                    5000
                );
            }
            assert_eq!(
                store
                    .checked_read_snapshot(&|| Ok(()), |_| "retry")
                    .unwrap(),
                "retry"
            );
        }
    }

    #[test]
    fn checked_snapshot_preserves_callback_errors_and_restores_custom_busy_policy() {
        let dir = tempfile::tempdir().unwrap();
        for path in [Path::new(":memory:"), &dir.path().join("error.db")] {
            let store = fixture(path);
            for mutex in std::iter::once(&store.writer).chain(store.readers.iter()) {
                mutex
                    .lock()
                    .unwrap()
                    .busy_timeout(Duration::from_millis(137))
                    .unwrap();
            }
            for _ in 0..store.readers.len().max(1) {
                let result = store
                    .checked_read_snapshot(&|| Ok(()), |tx| {
                        tx.query_row("SELECT nonexistent_column FROM files", [], |r| {
                            r.get::<_, i64>(0)
                        })
                    })
                    .unwrap();
                assert!(result.is_err());
            }
            for mutex in std::iter::once(&store.writer).chain(store.readers.iter()) {
                let connection = mutex.lock().unwrap();
                assert!(connection.is_autocommit());
                assert_eq!(
                    connection
                        .query_row("PRAGMA busy_timeout", [], |r| r.get::<_, i64>(0))
                        .unwrap(),
                    137
                );
            }
            assert_eq!(
                store
                    .checked_read_snapshot(&|| Ok(()), |tx| {
                        tx.query_row("SELECT COUNT(*) FROM files", [], |r| r.get::<_, i64>(0))
                    })
                    .unwrap()
                    .unwrap(),
                2
            );
        }
    }

    #[test]
    fn targeted_snapshot_matches_catalogue_without_reading_unrelated_rows_or_dirs() {
        let store = fixture(Path::new(":memory:"));
        let before = store.index_snapshot().unwrap();
        let generation = store.l1_generation().unwrap();
        // An unrelated row cannot be decoded as IndexedFile; a full snapshot fails.
        store
            .writer()
            .unwrap()
            .execute_batch(
                "UPDATE files SET size='invalid' WHERE path='unrelated.rs'; DROP TABLE dirs;",
            )
            .unwrap();
        let (actual, snapshot) = store.target_snapshot("target.rs", &|| Ok(())).unwrap();
        assert_eq!(actual, generation);
        assert_eq!(snapshot.files, [before.files[0].clone()]);
        assert!(snapshot.dirs.is_empty());
        assert!(
            store
                .target_snapshot("missing", &|| Ok(()))
                .unwrap()
                .1
                .files
                .is_empty()
        );
        assert!(store.index_snapshot().is_err());
    }

    #[test]
    fn targeted_snapshot_bounds_held_connections_and_restores_timeout_after_refusal() {
        let dir = tempfile::tempdir().unwrap();
        for path in [Path::new(":memory:"), &dir.path().join("code.db")] {
            let store = fixture(path);
            let held: Vec<_> = if store.readers.is_empty() {
                vec![store.writer().unwrap()]
            } else {
                store.readers.iter().map(|r| r.lock().unwrap()).collect()
            };
            thread::scope(|scope| {
                let (send, receive) = mpsc::channel();
                let store = &store;
                scope.spawn(move || {
                    let deadline = Instant::now() + Duration::from_millis(20);
                    let result = store.target_snapshot("target.rs", &|| {
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
            // Refuse after the read snapshot exists, not merely at admission.
            let calls = Cell::new(0);
            assert!(matches!(
                store.target_snapshot("target.rs", &|| {
                    calls.set(calls.get() + 1);
                    if calls.get() == 5 {
                        Err(Error::L1InputsChanged)
                    } else {
                        Ok(())
                    }
                }),
                Err(Error::L1InputsChanged)
            ));
            assert_eq!(
                store
                    .target_snapshot("target.rs", &|| Ok(()))
                    .unwrap()
                    .1
                    .files
                    .len(),
                1
            );
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
    fn targeted_snapshot_keeps_generation_and_file_coherent_during_concurrent_commit() {
        let dir = tempfile::tempdir().unwrap();
        let store = fixture(&dir.path().join("code.db"));
        let generation = store.l1_generation().unwrap();
        let calls = Cell::new(0);
        let (actual, snapshot) = store.target_snapshot("target.rs", &|| {
            calls.set(calls.get() + 1);
            if calls.get() == 5 {
                store.writer().unwrap().execute_batch(
                    "BEGIN IMMEDIATE; UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='l1_scan_generation';
                     UPDATE files SET size=99 WHERE path='target.rs'; COMMIT;"
                ).unwrap();
            }
            Ok(())
        }).unwrap();
        assert_eq!(actual, generation);
        assert_eq!(snapshot.files[0].size, 10);
        let (next, snapshot) = store.target_snapshot("target.rs", &|| Ok(())).unwrap();
        assert_eq!(next, generation + 1);
        assert_eq!(snapshot.files[0].size, 99);
    }
}
