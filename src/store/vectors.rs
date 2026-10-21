//! Typed sqlite-vector tables and exact scans; see `v1.md` §3.3.

use half::f16;
use rusqlite::{
    Connection, OptionalExtension, Transaction, params, params_from_iter, types::Value,
};

use super::{Error, Store, UnitKind, admission};

// Pinned sqlite-vector's MAX_TABLES (sqlite-vector.c). Do not persist more tables
// than a fresh native connection can register when the database is reopened.
const MAX_VECTOR_TABLES: i64 = 128;

/// Canonical physical table identity shared by indexing and query routing. Keeps
/// existing names stable; the full model digest prevents readable-name collisions.
pub(crate) fn embedding_table_slug(profile: &str, model: &str, dims: u64) -> String {
    // Keep the readable name bounded; the full model digest distinguishes names
    // that differ only in punctuation, Unicode or a truncated suffix.
    let readable: String = model
        .chars()
        .take(48)
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    format!(
        "{profile}_{readable}_{}_{dims}",
        blake3::hash(model.as_bytes()).to_hex()
    )
}

/// Metadata needed to create or reactivate one model-specific vector table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VectorTableSpec<'a> {
    pub(crate) slug: &'a str,
    pub(crate) model: &'a str,
    pub(crate) profile: &'a str,
    pub(crate) dims: usize,
    pub(crate) created_at: i64,
    pub(crate) active: bool,
}

/// Persisted bookkeeping for one `vec_<slug>` table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VectorTableInfo {
    pub(crate) slug: String,
    pub(crate) model: String,
    pub(crate) profile: String,
    pub(crate) dims: usize,
    pub(crate) created_at: i64,
    pub(crate) active: bool,
}

/// One fenced FLOAT16 vector write.
///
/// `embedding_hash` is the unit content hash that produced the embedding. It
/// is checked against `units.content_hash` in the write transaction so stale queue work
/// cannot attach a vector to newer content; the model identity lives in the
/// target table's metadata rather than a duplicated per-row column.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VectorInsert<'a> {
    pub(crate) table_slug: &'a str,
    pub(crate) unit_id: i64,
    pub(crate) ctx_hash: &'a [u8],
    pub(crate) embedding_hash: &'a [u8],
    pub(crate) embedding: &'a [f32],
}

/// Existing representations of one occurrence that a fenced replacement retires.
/// Planning reads only; publication must recompute this inside its transaction.
pub(crate) struct Replacement {
    target_exists: bool,
    retired_rows: Vec<(String, i64)>,
    retired_bytes: u64,
    row_ceiling: u64,
}
impl Replacement {
    /// Distinguishes additional vector bytes from same-size/shrinking replacements.
    pub(crate) fn grows_bytes(&self, dims: u64) -> bool {
        !self.target_exists && dims.saturating_mul(2) > self.retired_bytes
    }
}

/// Checks the post-replacement row count without deleting old coverage. Only rows
/// for this occurrence or its owned PR/module parent are credit; unrelated units
/// never fund a profile transition. Callers verify current summary provenance.
pub(crate) fn plan_replacement(
    store: &Store,
    tx: &Transaction<'_>,
    slug: &str,
    unit: i64,
    max_l2_units: u64,
) -> Result<Replacement, Error> {
    let max_l2_units = max_l2_units.min(store.max_l2_units);
    let table = vector_table_name(slug)?;
    let target_exists = tx.query_row(
        &format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE unit_id=?1)"),
        [unit],
        |r| r.get::<_, bool>(0),
    )?;
    // Only a summary attached to its own PR/module can replace that parent's
    // raw representation. Symbol descriptions retain their code vectors.
    let parent: Option<i64> = tx.query_row(
        "SELECT p.id FROM units u JOIN units p ON p.id=u.parent_unit
         WHERE u.id=?1 AND u.kind='summary' AND p.file_id IS NULL AND (
           (p.kind='pr' AND EXISTS(SELECT 1 FROM prs r WHERE r.unit_id=p.id AND r.summary_unit=u.id AND u.source_key='summary:pr:'||r.id))
           OR (p.kind='module' AND EXISTS(SELECT 1 FROM dirs d WHERE d.module_unit=p.id AND d.summary_unit=u.id AND u.source_key='summary:module:'||p.id)))",
        [unit],|r|r.get(0),
    ).optional()?;
    let names = tx
        .prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let mut retired_rows = Vec::new();
    let mut retired_bytes = 0_u64;
    for name in names {
        let table = vector_table_name(&name)?;
        for old_unit in [(name != slug).then_some(unit), parent]
            .into_iter()
            .flatten()
        {
            let bytes = tx
                .query_row(
                    &format!("SELECT length(v) FROM {table} WHERE unit_id=?1"),
                    [old_unit],
                    |r| r.get::<_, i64>(0),
                )
                .optional()?;
            if let Some(bytes) = bytes {
                retired_bytes = retired_bytes.saturating_add(
                    u64::try_from(bytes).map_err(|_| Error::InvalidVectorCount { count: bytes })?,
                );
                retired_rows.push((table.clone(), old_unit));
            }
        }
    }
    let current = count_vectors_in_transaction(tx)?;
    // A lowered cap freezes growth without evicting retained representations.
    let row_ceiling = max_l2_units.max(current);
    let after = current
        .saturating_sub(retired_rows.len() as u64)
        .saturating_add(u64::from(!target_exists));
    if after > row_ceiling {
        return Err(Error::VectorCapReached {
            current,
            max_l2_units,
        });
    }
    Ok(Replacement {
        target_exists,
        retired_rows,
        retired_bytes,
        row_ceiling,
    })
}

/// Typed SQL scope applied to the exact scan before selecting its top-k.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VectorScanPredicate<'a> {
    /// Actual stored kinds permitted by the selected embedding route.
    pub(crate) kinds: &'a [UnitKind],
    /// User-facing kinds, inherited from a summary's canonical parent.
    pub(crate) scope_kinds: &'a [UnitKind],
    pub(crate) path_globs: Option<&'a [String]>,
    pub(crate) exclude_ids: &'a [i64],
    /// Explicit migration only: suppress an old row when its kind's destination
    /// already contains that occurrence, before selecting the bounded top-k.
    pub(crate) replaced_by: &'a [(UnitKind, String)],
}

/// Context identity only: stale vectors remain usable until fenced replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContextState {
    Missing,
    Current,
    Stale,
}

/// Compares an occurrence's retained context with the caller's freshly resolved
/// header identity. This does not validate source policy or destination identity.
pub(crate) fn context_state_in(
    tx: &Transaction<'_>,
    slug: &str,
    unit: i64,
    expected: &[u8; 32],
) -> Result<ContextState, Error> {
    let table = vector_table_name(slug)?;
    let actual = tx
        .query_row(
            &format!("SELECT ctx_hash FROM {table} WHERE unit_id=?1"),
            [unit],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .optional()?;
    Ok(match actual {
        None => ContextState::Missing,
        Some(hash) if hash == expected => ContextState::Current,
        Some(_) => ContextState::Stale,
    })
}

/// Checks exact job output identity and physical presence in the same snapshot.
/// A completed job or matching context alone cannot prove its output survived.
pub(crate) fn publication_matches_in(
    tx: &Transaction<'_>,
    slug: &str,
    unit: i64,
    desired: &[u8; 32],
) -> Result<bool, Error> {
    let table = vector_table_name(slug)?;
    Ok(tx.query_row(
        &format!("SELECT EXISTS(SELECT 1 FROM embedding_publications p JOIN {table} v ON v.unit_id=p.unit_id WHERE p.unit_id=?1 AND p.table_slug=?2 AND p.desired_hash=?3)"),
        params![unit, slug, desired], |r| r.get(0),
    )?)
}

/// One exact dense-search candidate with cosine distance (lower is better).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VectorHit {
    pub(crate) unit_id: i64,
    pub(crate) distance: f64,
}

/// Creates native metadata only behind admission; automatic extension loading is
/// read-only so current-schema readers and capped opens cannot grow the WAL.
pub(super) fn initialize_metadata(store: &Store) -> Result<(), Error> {
    let exists: bool = store.writer()?.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type='table' AND name='_sqliteai_vector')",
        [],
        |r| r.get(0),
    )?;
    if !exists {
        admission::transaction::<_, Error>(store, admission::Kind::Migration, |transaction| {
            transaction.execute_batch("SELECT dkc_vector_metadata_init()")?;
            transaction.commit()?;
            Ok(())
        })?;
    }
    Ok(())
}

/// Initializes every table recorded in `embedding_tables` for one SQLite connection.
///
/// sqlite-vector keeps its registrations per connection, so reopening a database
/// must repeat this for the writer and each reader before a scan can run.
pub(super) fn initialize_registered_tables(connection: &Connection) -> Result<(), Error> {
    prune_registrations(connection)?;
    let snapshot = connection.unchecked_transaction()?;
    let connection: &Connection = &snapshot;
    let tables = connection
        .prepare("SELECT slug, dims FROM embedding_tables ORDER BY slug")?
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    for (slug, stored_dims) in tables {
        let dims = dims_from_db(&slug, stored_dims)?;
        let table = vector_table_name(&slug)?;
        initialize_table(connection, &table, dims)?;
    }
    Ok(())
}

impl Store {
    /// Creates/reactivates a model table atomically within WAL growth admission.
    /// Native connection registration follows durable SQL commit, so allocation
    /// rollback cannot leave a registration for a table that does not exist.
    /// Native registration errors are reported after durable table creation.
    pub(crate) fn ensure_vector_table(&self, spec: VectorTableSpec<'_>) -> Result<(), Error> {
        validate_dims(spec.slug, spec.dims)?;
        let table = vector_table_name(spec.slug)?;
        let dims = i64::try_from(spec.dims).map_err(|_| Error::InvalidVectorDimensions {
            slug: spec.slug.to_owned(),
        })?;
        admission::transaction::<_, Error>(self, admission::Kind::Growth, |transaction| {
            let existing: Option<(String, String, i64)> = transaction
                .query_row(
                    "SELECT model, profile, dims FROM embedding_tables WHERE slug = ?1",
                    [spec.slug],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()?;
            if let Some((model, profile, existing_dims)) = existing {
                if model != spec.model || profile != spec.profile || existing_dims != dims {
                    return Err(Error::VectorTableMismatch {
                        slug: spec.slug.to_owned(),
                    });
                }
                transaction.execute(
                    "UPDATE embedding_tables SET active = ?1 WHERE slug = ?2",
                    params![spec.active, spec.slug],
                )?;
            } else {
                let tables: i64 =
                    transaction
                        .query_row("SELECT count(*) FROM embedding_tables", [], |r| r.get(0))?;
                if tables >= MAX_VECTOR_TABLES {
                    return Err(Error::VectorTableCapacity {
                        max_tables: MAX_VECTOR_TABLES,
                    });
                }
                transaction.execute_batch(&format!(
                    "CREATE TABLE {table} (
                        unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,
                        v BLOB NOT NULL,
                        ctx_hash BLOB NOT NULL
                    );
                    CREATE INDEX {} ON {table}(unit_id);",
                    vector_count_index_name(spec.slug)?
                ))?;
                transaction.execute(
                    "INSERT INTO embedding_tables (slug, model, profile, dims, created_at, active)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        spec.slug,
                        spec.model,
                        spec.profile,
                        dims,
                        spec.created_at,
                        spec.active
                    ],
                )?;
            }
            transaction.commit()?;
            Ok(())
        })?;
        // vector_init mutates a connection-local native cache, not transactional
        // SQL state. Never register a speculative table before commit succeeds.
        {
            let writer = self.writer()?;
            prune_registrations(&writer)?;
            initialize_table(&writer, &table, spec.dims)?;
        }
        for reader in &self.readers {
            let reader = reader.lock().map_err(|_| Error::LockPoisoned {
                connection: "reader",
            })?;
            prune_registrations(&reader)?;
            initialize_table(&reader, &table, spec.dims)?;
        }
        Ok(())
    }

    /// Returns model-table bookkeeping in stable slug order for status and profile transitions.
    pub(crate) fn vector_tables(&self) -> Result<Vec<VectorTableInfo>, Error> {
        self.with_reader(|connection| {
            let tables = connection
                .prepare(
                    "SELECT slug, model, profile, dims, created_at, active
                     FROM embedding_tables ORDER BY slug",
                )?
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, bool>(5)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            tables
                .into_iter()
                .map(|(slug, model, profile, dims, created_at, active)| {
                    Ok(VectorTableInfo {
                        dims: dims_from_db(&slug, dims)?,
                        slug,
                        model,
                        profile,
                        created_at,
                        active,
                    })
                })
                .collect()
        })
    }

    /// Writes one model vector with WAL headroom and current-unit identity fencing.
    pub(crate) fn insert_vector(&self, vector: VectorInsert<'_>) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Growth, |transaction| {
            self.insert_vector_in(&transaction, vector)?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Writes using the caller's transaction so queue output and completion commit together.
    ///
    /// The transaction must belong to this store. This method never locks or commits;
    /// the caller supplies the desired-generation/context/policy and lease fences.
    /// Content identity, dimensions, normalization and the physical vector cap remain
    /// checked here, including writes already made in the same transaction.
    pub(crate) fn insert_vector_in(
        &self,
        transaction: &Transaction<'_>,
        vector: VectorInsert<'_>,
    ) -> Result<(), Error> {
        self.insert_vector_with_cap(transaction, vector, self.max_l2_units)
    }

    fn insert_vector_with_cap(
        &self,
        transaction: &Transaction<'_>,
        vector: VectorInsert<'_>,
        row_ceiling: u64,
    ) -> Result<(), Error> {
        let table = vector_table_name(vector.table_slug)?;
        let stored_dims: Option<i64> = transaction
            .query_row(
                "SELECT dims FROM embedding_tables WHERE slug = ?1",
                [vector.table_slug],
                |row| row.get(0),
            )
            .optional()?;
        let stored_dims = stored_dims.ok_or_else(|| Error::VectorTableMissing {
            slug: vector.table_slug.to_owned(),
        })?;
        let dims = dims_from_db(vector.table_slug, stored_dims)?;
        if vector.embedding.len() != dims {
            return Err(Error::VectorDimensionMismatch {
                slug: vector.table_slug.to_owned(),
                expected: dims,
                actual: vector.embedding.len(),
            });
        }
        let current_hash: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT content_hash FROM units WHERE id = ?1",
                [vector.unit_id],
                |row| row.get(0),
            )
            .optional()?;
        let current_hash = current_hash.ok_or(Error::VectorUnitMissing {
            unit_id: vector.unit_id,
        })?;
        if current_hash != vector.embedding_hash {
            return Err(Error::VectorFenceMismatch {
                unit_id: vector.unit_id,
            });
        }
        let exists: bool = transaction.query_row(
            &format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE unit_id = ?1)"),
            [vector.unit_id],
            |row| row.get(0),
        )?;
        if !exists {
            let current = count_vectors_in_transaction(transaction)?;
            if current >= row_ceiling {
                return Err(Error::VectorCapReached {
                    current,
                    max_l2_units: self.max_l2_units,
                });
            }
        }
        let dims = i64::try_from(dims).map_err(|_| Error::InvalidVectorDimensions {
            slug: vector.table_slug.to_owned(),
        })?;
        let embedding = normalize_vector(vector.table_slug, vector.embedding)?;
        let blob = f16_blob(&embedding);
        // Standalone writes cannot attest a queue generation. Fenced workers
        // install their proof after replacement, inside this same transaction.
        transaction.execute(
            "DELETE FROM embedding_publications WHERE unit_id=?1",
            [vector.unit_id],
        )?;
        transaction.execute(
            &format!(
                "INSERT INTO {table} (unit_id, v, ctx_hash)
                 VALUES (?1, vector_as_f16(?2, ?3), ?4)
                 ON CONFLICT(unit_id) DO UPDATE SET v = excluded.v, ctx_hash = excluded.ctx_hash"
            ),
            params![vector.unit_id, blob, dims, vector.ctx_hash],
        )?;
        Ok(())
    }

    /// Replaces this occurrence's older model representations atomically under
    /// growth admission, preserving them if dimensions/fences/storage reject output.
    pub(crate) fn replace_vector(&self, vector: VectorInsert<'_>) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Growth, |tx| {
            self.replace_vector_in(&tx, vector)?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Retires older representations inside the caller's fenced publication.
    /// Callers must propagate every error and roll back the transaction; no commit
    /// is allowed after failure. Unrelated occurrences and table metadata remain.
    pub(crate) fn replace_vector_in(
        &self,
        tx: &Transaction<'_>,
        vector: VectorInsert<'_>,
    ) -> Result<(), Error> {
        let plan = plan_replacement(
            self,
            tx,
            vector.table_slug,
            vector.unit_id,
            self.max_l2_units,
        )?;
        for (table, unit) in plan.retired_rows {
            tx.execute(&format!("DELETE FROM {table} WHERE unit_id=?1"), [unit])?;
            // Every old table for a credited parent is retired in this plan.
            tx.execute(
                "DELETE FROM embedding_publications WHERE unit_id=?1",
                [unit],
            )?;
        }
        self.insert_vector_with_cap(tx, vector, plan.row_ceiling)
    }

    /// Removes one L2 vector within control WAL space, retaining its L1 data.
    pub(crate) fn delete_vector(&self, table_slug: &str, unit_id: i64) -> Result<(), Error> {
        let table = vector_table_name(table_slug)?;
        admission::transaction(self, admission::Kind::Control, |transaction| {
            transaction.execute(
                &format!("DELETE FROM {table} WHERE unit_id = ?1"),
                [unit_id],
            )?;
            transaction.execute(
                "DELETE FROM embedding_publications WHERE unit_id=?1 AND table_slug=?2",
                params![unit_id, table_slug],
            )?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Runs exact FLOAT16/COSINE retrieval, applying scope before the final top-k selection.
    pub(crate) fn scan_vectors(
        &self,
        table_slug: &str,
        query: &[f32],
        top_n: usize,
        predicate: VectorScanPredicate<'_>,
    ) -> Result<Vec<VectorHit>, Error> {
        self.scan_vectors_checked(table_slug, query, top_n, predicate, |_| Ok(()))
    }

    /// Prunes idle native registrations, then shares one snapshot across all
    /// selected scans. Callers must do network work before entering this callback.
    pub(crate) fn vector_snapshot<T>(
        &self,
        operation: impl FnOnce(&Transaction<'_>) -> T,
    ) -> Result<T, Error> {
        self.with_reader(|connection| {
            prune_registrations(connection)?;
            let snapshot = connection.unchecked_transaction()?;
            Ok(operation(&snapshot))
        })
    }

    /// Validates query/table identity inside the same snapshot as the exact scan.
    /// The callback also permits deadline/cancellation checks after reader waits;
    /// it must not acquire another store connection or perform network I/O.
    pub(crate) fn scan_vectors_checked<E>(
        &self,
        table_slug: &str,
        query: &[f32],
        top_n: usize,
        predicate: VectorScanPredicate<'_>,
        check: impl FnOnce(&Transaction<'_>) -> Result<(), E>,
    ) -> Result<Vec<VectorHit>, E>
    where
        E: From<Error>,
    {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        vector_table_name(table_slug)?;
        self.with_reader(|connection| {
            // Reclaim absent registrations before taking the read snapshot; native
            // cleanup deliberately refuses transactions and active vector cursors.
            prune_registrations(connection)?;
            let snapshot = connection.unchecked_transaction()?;
            Ok(check(&snapshot).and_then(|()| {
                scan_vectors_in(&snapshot, table_slug, query, top_n, predicate).map_err(E::from)
            }))
        })
        .map_err(E::from)?
    }

    /// Counts every vector row across all registered tables for L2 cap enforcement.
    pub(crate) fn count_vectors(&self) -> Result<u64, Error> {
        self.with_reader(count_vectors)
    }

    /// Drops a completed transition's table and bookkeeping atomically within
    /// control WAL space; allocation failure retains both for a subsequent pass.
    pub(crate) fn drop_vector_table(&self, slug: &str) -> Result<(), Error> {
        let table = vector_table_name(slug)?;
        admission::transaction::<_, Error>(self, admission::Kind::Control, |transaction| {
            transaction.execute_batch(&format!("DROP TABLE IF EXISTS {table};"))?;
            transaction.execute("DELETE FROM embedding_tables WHERE slug = ?1", [slug])?;
            transaction.execute("DELETE FROM _sqliteai_vector WHERE tblname=?1", [&table])?;
            transaction.commit()?;
            Ok(())
        })?;
        self.prune_vector_registrations()
    }

    /// Retires a caller-selected obsolete table only if it contains no coverage.
    /// The caller must fence current profile policy before using this primitive.
    /// SQL deletion commits before native cleanup; a cleanup error is retryable.
    pub(crate) fn retire_empty_vector_table(&self, slug: &str) -> Result<bool, Error> {
        self.retire_empty_vector_table_if(slug, |_| Ok::<_, Error>(()))
    }

    /// Rechecks caller policy under writer admission and again before committing
    /// retirement. Any failed fence rolls back the table and metadata together.
    pub(crate) fn retire_empty_vector_table_if<E>(
        &self,
        slug: &str,
        mut current: impl FnMut(&Transaction<'_>) -> Result<(), E>,
    ) -> Result<bool, E>
    where
        E: From<Error> + From<rusqlite::Error>,
    {
        let table = vector_table_name(slug)?;
        let retired = admission::transaction(self, admission::Kind::Control, |tx| {
            current(&tx)?;
            let exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1)",
                [slug],
                |r| r.get(0),
            )?;
            if exists {
                let occupied: bool =
                    tx.query_row(&format!("SELECT EXISTS(SELECT 1 FROM {table})"), [], |r| {
                        r.get(0)
                    })?;
                if occupied {
                    return Ok(false);
                }
                tx.execute_batch(&format!("DROP TABLE {table}"))?;
                tx.execute("DELETE FROM embedding_tables WHERE slug=?1", [slug])?;
                tx.execute("DELETE FROM _sqliteai_vector WHERE tblname=?1", [&table])?;
            }
            current(&tx)?;
            tx.commit()?;
            Ok::<_, E>(true)
        })?;
        if retired {
            self.prune_vector_registrations()?;
        }
        Ok(retired)
    }

    fn prune_vector_registrations(&self) -> Result<(), Error> {
        prune_registrations(&*self.writer()?)?;
        for reader in &self.readers {
            let reader = reader.lock().map_err(|_| Error::LockPoisoned {
                connection: "reader",
            })?;
            prune_registrations(&reader)?;
        }
        Ok(())
    }
}

/// Validates a model slug before using its table name in dynamically built SQL.
pub(crate) fn vector_table_name(slug: &str) -> Result<String, Error> {
    if slug.is_empty()
        || !slug
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(Error::InvalidVectorTableSlug {
            slug: slug.to_owned(),
        });
    }
    Ok(format!("vec_{slug}"))
}

/// Returns the compact covering index used for exact vector-row counts.
fn vector_count_index_name(slug: &str) -> Result<String, Error> {
    Ok(format!("{}_unit_id_count", vector_table_name(slug)?))
}

/// Adds exact-count indexes to vector tables created before schema version 23.
pub(super) fn migrate_count_indexes(tx: &Transaction<'_>) -> Result<(), Error> {
    let slugs = tx
        .prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    for slug in slugs {
        let table = vector_table_name(&slug)?;
        let index = vector_count_index_name(&slug)?;
        tx.execute_batch(&format!(
            "CREATE INDEX IF NOT EXISTS {index} ON {table}(unit_id)"
        ))?;
    }
    Ok(())
}

fn validate_dims(slug: &str, dims: usize) -> Result<(), Error> {
    if dims == 0 || dims > i32::MAX as usize {
        return Err(Error::InvalidVectorDimensions {
            slug: slug.to_owned(),
        });
    }
    Ok(())
}

fn dims_from_db(slug: &str, dims: i64) -> Result<usize, Error> {
    if dims <= 0 {
        return Err(Error::InvalidVectorDimensions {
            slug: slug.to_owned(),
        });
    }
    usize::try_from(dims).map_err(|_| Error::InvalidVectorDimensions {
        slug: slug.to_owned(),
    })
}

/// Scans one selected table inside the caller's read snapshot, so metadata,
/// native registration and rows cannot span different committed table lifetimes.
/// Callers must prune idle registrations before starting long-lived snapshots.
pub(crate) fn scan_vectors_in(
    snapshot: &Transaction<'_>,
    table_slug: &str,
    query: &[f32],
    top_n: usize,
    predicate: VectorScanPredicate<'_>,
) -> Result<Vec<VectorHit>, Error> {
    if top_n == 0 {
        return Ok(Vec::new());
    }
    let table = vector_table_name(table_slug)?;
    let connection: &Connection = snapshot;
    let stored_dims: Option<i64> = connection
        .query_row(
            "SELECT dims FROM embedding_tables WHERE slug = ?1",
            [table_slug],
            |row| row.get(0),
        )
        .optional()?;
    let stored_dims = stored_dims.ok_or_else(|| Error::VectorTableMissing {
        slug: table_slug.to_owned(),
    })?;
    let dims = dims_from_db(table_slug, stored_dims)?;
    if query.len() != dims {
        return Err(Error::VectorDimensionMismatch {
            slug: table_slug.to_owned(),
            expected: dims,
            actual: query.len(),
        });
    }
    // The table may have been created by another process since open.
    // Register only this selected destination against the same snapshot.
    initialize_table(connection, &table, dims)?;
    let requested =
        i64::try_from(top_n).map_err(|_| Error::VectorLimitTooLarge { limit: top_n })?;
    let query = normalize_vector(table_slug, query)?;
    // Unscoped search uses the extension's bounded native heap. Scoped
    // search streams the exact scan through SQL predicates before LIMIT;
    // a global prefetch cannot guarantee any eligible results.
    let scan = if predicate.kinds.is_empty()
        && predicate.scope_kinds.is_empty()
        && predicate.path_globs.is_none()
        && predicate.exclude_ids.is_empty()
        && predicate.replaced_by.is_empty()
    {
        "vector_full_scan(?1, 'v', ?2, ?3)"
    } else {
        "vector_full_scan(?1, 'v', ?2)"
    };
    let mut sql = format!(
        "SELECT scan.rowid, scan.distance
         FROM {scan} AS scan
         CROSS JOIN units AS u ON u.id = scan.rowid
         LEFT JOIN files AS f ON f.id = u.file_id
         WHERE 1 = 1"
    );
    let mut values = vec![
        Value::Text(table),
        Value::Blob(f16_blob(&query)),
        Value::Integer(requested),
    ];
    let mut next_parameter = 4;
    if !predicate.kinds.is_empty() {
        let placeholders = predicate
            .kinds
            .iter()
            .map(|kind| {
                let placeholder = format!("?{next_parameter}");
                next_parameter += 1;
                values.push(Value::Text(kind.as_str().to_owned()));
                placeholder
            })
            .collect::<Vec<_>>();
        sql.push_str(&format!(" AND u.kind IN ({})", placeholders.join(", ")));
    }
    let mut scope = Vec::new();
    if !predicate.scope_kinds.is_empty() {
        let placeholders = predicate
            .scope_kinds
            .iter()
            .map(|kind| {
                let placeholder = format!("?{next_parameter}");
                next_parameter += 1;
                values.push(Value::Text(kind.as_str().to_owned()));
                placeholder
            })
            .collect::<Vec<_>>();
        scope.push(format!("u.kind IN ({})", placeholders.join(",")));
    }
    if let Some(path_globs) = predicate.path_globs {
        scope.push(format!(
            "EXISTS(SELECT 1 FROM json_each(?{next_parameter}) g WHERE f.path GLOB g.value)"
        ));
        values.push(Value::Text(serde_json::json!(path_globs).to_string()));
        next_parameter += 1;
    }
    if !scope.is_empty() {
        sql.push_str(" AND ");
        sql.push_str(&super::projection::scope_sql(&scope.join(" AND ")));
    }
    for (kind, destination) in predicate.replaced_by {
        let destination_table = vector_table_name(destination)?;
        let exists: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1 AND active=1)",
            [destination],
            |r| r.get(0),
        )?;
        if exists {
            sql.push_str(&format!(" AND (u.kind != ?{next_parameter} OR NOT EXISTS(SELECT 1 FROM {destination_table} replacement WHERE replacement.unit_id=u.id))"));
            values.push(Value::Text(kind.as_str().to_owned()));
            next_parameter += 1;
        }
    }
    if !predicate.exclude_ids.is_empty() {
        let exclusions = super::projection::exclusion_sql(predicate.exclude_ids, next_parameter);
        sql = format!("{}{sql} AND {}", exclusions.prefix, exclusions.predicate);
        values.push(Value::Text(
            serde_json::json!(predicate.exclude_ids).to_string(),
        ));
    }
    sql.push_str(" ORDER BY scan.distance ASC, scan.rowid ASC LIMIT ?3");
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values), |row| {
        Ok(VectorHit {
            unit_id: row.get(0)?,
            distance: row.get(1)?,
        })
    })?;
    let hits = rows.collect::<Result<Vec<_>, _>>();
    #[cfg(test)]
    scan_observation::executed(table_slug, &sql, predicate);
    hits.map_err(Into::into)
}

// Call only outside SQL transactions. Native code rejects live vector cursors;
// Store connection locks also keep pool borrowers from racing reclamation.
fn prune_registrations(connection: &Connection) -> Result<(), Error> {
    connection.query_row("SELECT dkc_vector_prune()", [], |_| Ok(()))?;
    Ok(())
}

fn initialize_table(connection: &Connection, table: &str, dims: usize) -> Result<(), Error> {
    let options = format!("dimension={dims},type=FLOAT16,distance=COSINE");
    connection.query_row(
        "SELECT vector_init(?1, 'v', ?2)",
        params![table, options],
        |_| Ok(()),
    )?;
    Ok(())
}

fn normalize_vector(slug: &str, values: &[f32]) -> Result<Vec<f32>, Error> {
    let squared_norm = values.iter().try_fold(0.0_f64, |sum, value| {
        if !value.is_finite() {
            return None;
        }
        Some(sum + f64::from(*value) * f64::from(*value))
    });
    let Some(squared_norm) = squared_norm else {
        return Err(Error::InvalidVectorValues {
            slug: slug.to_owned(),
        });
    };
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(Error::InvalidVectorValues {
            slug: slug.to_owned(),
        });
    }
    Ok(values
        .iter()
        .map(|value| (f64::from(*value) / norm) as f32)
        .collect())
}

fn f16_blob(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
    for value in values {
        blob.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
    blob
}

fn count_vectors(connection: &Connection) -> Result<u64, Error> {
    let tables = connection
        .prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let mut count = 0_u64;
    for slug in tables {
        let table = vector_table_name(&slug)?;
        let rows: i64 =
            connection.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })?;
        let rows = u64::try_from(rows).map_err(|_| Error::InvalidVectorCount { count: rows })?;
        count = count.saturating_add(rows);
    }
    Ok(count)
}

/// Counts physical rows, including pending writes, for shared cap admission.
pub(crate) fn count_vectors_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
) -> Result<u64, Error> {
    let tables = transaction
        .prepare("SELECT slug FROM embedding_tables ORDER BY slug")?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let mut count = 0_u64;
    for slug in tables {
        let table = vector_table_name(&slug)?;
        let rows: i64 =
            transaction.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })?;
        let rows = u64::try_from(rows).map_err(|_| Error::InvalidVectorCount { count: rows })?;
        count = count.saturating_add(rows);
    }
    Ok(count)
}

#[cfg(test)]
mod admission_tests;

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, error::Error as StdError, time::Instant};

    use half::f16;
    use rusqlite::params;

    use super::{
        Error, Store, VectorInsert, VectorScanPredicate, VectorTableSpec, f16_blob,
        vector_table_name,
    };
    use crate::store::{StoreLimits, open};

    const DIMS: usize = 8;

    pub(super) fn table(slug: &str, dims: usize) -> VectorTableSpec<'_> {
        VectorTableSpec {
            slug,
            model: "test-model",
            profile: "test",
            dims,
            created_at: 1,
            active: true,
        }
    }

    pub(super) fn insert_unit(store: &Store, id: i64) -> Result<[u8; 32], Box<dyn StdError>> {
        let mut hash = [0_u8; 32];
        hash[..8].copy_from_slice(&(id as u64).to_le_bytes());
        store.writer()?.execute(
            "INSERT INTO unit_contents(hash, text, tokens) VALUES (?1, 'vector fixture', 1)",
            [hash.as_slice()],
        )?;
        store.writer()?.execute(
            "INSERT INTO units (id, hash, content_hash, source_key, kind, header, created_at, updated_at)
             VALUES (?1, ?2, ?2, 'vector:' || ?1, 'chunk', 'vector fixture', 1, 1)",
            params![id, hash.as_slice()],
        )?;
        Ok(hash)
    }

    fn normalized(state: &mut u64, dims: usize) -> Vec<f32> {
        let mut vector = (0..dims)
            .map(|_| {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*state >> 40) as f32 / ((1_u32 << 24) as f32)) * 2.0 - 1.0
            })
            .collect::<Vec<_>>();
        let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        for value in &mut vector {
            *value /= length;
        }
        vector
    }

    fn f16_cosine_distance(left: &[f32], right: &[f32]) -> f64 {
        // Independent scalar f64 oracle over decoded FLOAT16 values. Rounding
        // changes vector lengths, so both decoded norms belong in the formula.
        let left = left
            .iter()
            .map(|v| f64::from(f16::from_f32(*v).to_f32()))
            .collect::<Vec<_>>();
        let right = right
            .iter()
            .map(|v| f64::from(f16::from_f32(*v).to_f32()))
            .collect::<Vec<_>>();
        let dot = left.iter().zip(&right).map(|(a, b)| a * b).sum::<f64>();
        let norm_left = left.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_right = right.iter().map(|v| v * v).sum::<f64>().sqrt();
        1.0 - (dot / (norm_left * norm_right)).clamp(-1.0, 1.0)
    }

    #[test]
    fn vector_tables_roundtrip_filter_and_reopen() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.ensure_vector_table(table("test_4", 4))?;
        let first_hash = insert_unit(&store, 1)?;
        let second_hash = insert_unit(&store, 2)?;
        store.insert_vector(VectorInsert {
            table_slug: "test_4",
            unit_id: 1,
            ctx_hash: &[3; 32],
            embedding_hash: &first_hash,
            embedding: &[1.0, 0.0, 0.0, 0.0],
        })?;
        store.insert_vector(VectorInsert {
            table_slug: "test_4",
            unit_id: 2,
            ctx_hash: &[4; 32],
            embedding_hash: &second_hash,
            embedding: &[0.0, 1.0, 0.0, 0.0],
        })?;
        assert_eq!(store.count_vectors()?, 2);
        assert_eq!(store.vector_tables()?.len(), 1);
        assert_eq!(
            store
                .scan_vectors(
                    "test_4",
                    &[0.9, 0.1, 0.0, 0.0],
                    2,
                    VectorScanPredicate::default()
                )?
                .into_iter()
                .map(|hit| hit.unit_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        store.delete_vector("test_4", 2)?;
        assert_eq!(store.count_vectors()?, 1);
        drop(store);

        let reopened = open(file.path(), StoreLimits::default())?;
        assert_eq!(
            reopened.scan_vectors(
                "test_4",
                &[1.0, 0.0, 0.0, 0.0],
                1,
                VectorScanPredicate::default()
            )?[0]
                .unit_id,
            1
        );
        Ok(())
    }

    #[test]
    fn decoded_cosine_handles_self_near_ties_negative_and_extreme_inputs()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.ensure_vector_table(table("adversarial_2", 2))?;
        let inputs = [
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [f32::MAX, 1.0],
            [f32::MIN_POSITIVE, f32::MIN_POSITIVE],
        ];
        for (index, input) in inputs.iter().enumerate() {
            let id = (index + 1) as i64;
            let hash = insert_unit(&store, id)?;
            store.insert_vector(VectorInsert {
                table_slug: "adversarial_2",
                unit_id: id,
                ctx_hash: &[0; 32],
                embedding_hash: &hash,
                embedding: input,
            })?;
        }
        let decoded = store
            .writer()?
            .prepare("SELECT unit_id,v FROM vec_adversarial_2 ORDER BY unit_id")?
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        for query in inputs {
            let query = super::normalize_vector("adversarial_2", &query)?;
            let actual = store.scan_vectors(
                "adversarial_2",
                &query,
                inputs.len(),
                VectorScanPredicate::default(),
            )?;
            assert_eq!(actual.len(), inputs.len());
            for hit in actual {
                let blob = &decoded
                    .iter()
                    .find(|(id, _)| *id == hit.unit_id)
                    .ok_or("missing stored vector")?
                    .1;
                let values = blob
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                    .collect::<Vec<_>>();
                let expected = f16_cosine_distance(&query, &values);
                assert!(
                    (hit.distance - expected).abs() <= 1e-5,
                    "unit={} actual={} decoded_f64_cosine={expected}",
                    hit.unit_id,
                    hit.distance
                );
            }
        }
        // The audit's close-angle witness was incorrectly clamped to a tie by 1-dot.
        for (id, angle) in [(1, 0.113_f32), (2, 0.12296_f32)] {
            let mut hash = [0; 32];
            hash[..8].copy_from_slice(&(id as u64).to_le_bytes());
            store.insert_vector(VectorInsert {
                table_slug: "adversarial_2",
                unit_id: id,
                ctx_hash: &[1; 32],
                embedding_hash: &hash,
                embedding: &[angle.cos(), angle.sin()],
            })?;
        }
        for id in 3..=6 {
            store.delete_vector("adversarial_2", id)?;
        }
        assert_eq!(
            store.scan_vectors(
                "adversarial_2",
                &[0.12296_f32.cos(), 0.12296_f32.sin()],
                1,
                VectorScanPredicate::default()
            )?[0]
                .unit_id,
            2
        );
        Ok(())
    }

    #[test]
    fn scoped_top_k_includes_eligible_vectors_beyond_the_global_prefetch()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.ensure_vector_table(table("scope_2", 2))?;
        for id in 1..=303 {
            let hash = insert_unit(&store, id)?;
            let vector = if id <= 300 {
                [1.0, 0.0]
            } else {
                [(id - 301) as f32 * 0.1, 1.0]
            };
            store.insert_vector(VectorInsert {
                table_slug: "scope_2",
                unit_id: id,
                ctx_hash: &[0; 32],
                embedding_hash: &hash,
                embedding: &vector,
            })?;
        }
        store.writer()?.execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment) VALUES(1,'docs/target.md',zeroblob(32),0,0,'doc',2,'default');
            UPDATE units SET file_id=1,kind='doc_section' WHERE id>300;")?;
        for predicate in [
            VectorScanPredicate {
                kinds: &[crate::store::UnitKind::DocSection],
                scope_kinds: &[],
                path_globs: None,
                exclude_ids: &[],
                replaced_by: &[],
            },
            VectorScanPredicate {
                kinds: &[],
                scope_kinds: &[],
                path_globs: Some(&["docs/*".into()]),
                exclude_ids: &[],
                replaced_by: &[],
            },
        ] {
            let hits = store.scan_vectors("scope_2", &[1.0, 0.0], 3, predicate)?;
            assert_eq!(
                hits.into_iter().map(|h| h.unit_id).collect::<Vec<_>>(),
                [303, 302, 301]
            );
        }
        store.writer()?.execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment) VALUES(2,'other/target.md',zeroblob(32),0,0,'doc',2,'default'); UPDATE units SET file_id=2 WHERE id=301;")?;
        let globs = vec!["docs/*".into(), "other/*".into(), "docs/target.md".into()];
        let hits = store.scan_vectors(
            "scope_2",
            &[1.0, 0.0],
            3,
            VectorScanPredicate {
                path_globs: Some(&globs),
                exclude_ids: &[302],
                ..Default::default()
            },
        )?;
        assert_eq!(
            hits.iter().map(|h| h.unit_id).collect::<Vec<_>>(),
            [303, 301]
        );
        assert!(
            store
                .scan_vectors(
                    "scope_2",
                    &[1.0, 0.0],
                    3,
                    VectorScanPredicate {
                        path_globs: Some(&[]),
                        ..Default::default()
                    }
                )?
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn vector_queries_reject_zero_nonfinite_and_malformed_inputs() -> Result<(), Box<dyn StdError>>
    {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.ensure_vector_table(table("invalid_2", 2))?;
        for query in [[0.0, 0.0], [f32::NAN, 1.0], [f32::INFINITY, 1.0]] {
            assert!(matches!(
                store.scan_vectors("invalid_2", &query, 1, VectorScanPredicate::default()),
                Err(Error::InvalidVectorValues { .. })
            ));
        }
        assert!(matches!(
            store.scan_vectors("invalid_2", &[1.0], 1, VectorScanPredicate::default()),
            Err(Error::VectorDimensionMismatch { .. })
        ));
        assert!(
            store
                .writer()?
                .prepare("SELECT distance FROM vector_full_scan('vec_invalid_2','v',X'00',1)")?
                .query_row([], |r| r.get::<_, f64>(0))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn native_top_k_breaks_boundary_ties_by_unit_id() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.ensure_vector_table(table("ties_2", 2))?;
        for id in 1..=10 {
            let hash = insert_unit(&store, id)?;
            let values = if id > 8 { [0.0, 1.0] } else { [1.0, 0.0] };
            store.insert_vector(VectorInsert {
                table_slug: "ties_2",
                unit_id: id,
                ctx_hash: &[0; 32],
                embedding_hash: &hash,
                embedding: &values,
            })?;
        }
        assert_eq!(
            store
                .scan_vectors("ties_2", &[0.0, 1.0], 5, VectorScanPredicate::default())?
                .into_iter()
                .map(|h| h.unit_id)
                .collect::<Vec<_>>(),
            [9, 10, 1, 2, 3]
        );
        Ok(())
    }

    #[test]
    fn decoded_cosine_random_vectors_cover_simd_boundaries() -> Result<(), Box<dyn StdError>> {
        for dims in [1, 7, 8, 31, 32, 33, 127, 1024] {
            let file = tempfile::NamedTempFile::new()?;
            let store = open(file.path(), StoreLimits::default())?;
            store.ensure_vector_table(table("tails", dims))?;
            let mut state = 31;
            let vectors = (0..64)
                .map(|_| normalized(&mut state, dims))
                .collect::<Vec<_>>();
            for (index, vector) in vectors.iter().enumerate() {
                let hash = insert_unit(&store, (index + 1) as i64)?;
                store.insert_vector(VectorInsert {
                    table_slug: "tails",
                    unit_id: (index + 1) as i64,
                    ctx_hash: &[0; 32],
                    embedding_hash: &hash,
                    embedding: vector,
                })?;
            }
            let query = super::normalize_vector("tails", &vectors[0])?;
            let rows = store
                .writer()?
                .prepare("SELECT unit_id,v FROM vec_tails ORDER BY unit_id")?
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;
            for hit in store.scan_vectors("tails", &query, 64, VectorScanPredicate::default())? {
                let blob = &rows[(hit.unit_id - 1) as usize].1;
                let vector = blob
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                    .collect::<Vec<_>>();
                let expected = f16_cosine_distance(&query, &vector);
                assert!(
                    (expected - hit.distance).abs() <= 1e-5,
                    "dims={dims} unit={} actual={} oracle={expected}",
                    hit.unit_id,
                    hit.distance
                );
            }
        }
        Ok(())
    }

    #[test]
    fn vector_tables_are_separate_and_drop_their_bookkeeping() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.ensure_vector_table(table("code_4", 4))?;
        store.ensure_vector_table(VectorTableSpec {
            slug: "docs_6",
            model: "docs-model",
            profile: "remote",
            dims: 6,
            created_at: 2,
            active: false,
        })?;
        assert_eq!(store.vector_tables()?.len(), 2);
        store.drop_vector_table("docs_6")?;
        assert_eq!(store.vector_tables()?.len(), 1);
        assert!(matches!(
            store.scan_vectors("docs_6", &[0.0; 6], 1, VectorScanPredicate::default()),
            Err(Error::VectorTableMissing { .. })
        ));
        Ok(())
    }

    #[test]
    fn vector_writes_fence_stale_content_and_enforce_the_global_cap()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(
            file.path(),
            StoreLimits {
                max_l2_units: 1,
                ..StoreLimits::default()
            },
        )?;
        store.ensure_vector_table(table("cap_2", 2))?;
        let first_hash = insert_unit(&store, 1)?;
        let second_hash = insert_unit(&store, 2)?;
        store.insert_vector(VectorInsert {
            table_slug: "cap_2",
            unit_id: 1,
            ctx_hash: &[1; 32],
            embedding_hash: &first_hash,
            embedding: &[1.0, 0.0],
        })?;
        store.insert_vector(VectorInsert {
            table_slug: "cap_2",
            unit_id: 1,
            ctx_hash: &[2; 32],
            embedding_hash: &first_hash,
            embedding: &[0.0, 1.0],
        })?;
        assert!(matches!(
            store.insert_vector(VectorInsert {
                table_slug: "cap_2",
                unit_id: 2,
                ctx_hash: &[3; 32],
                embedding_hash: &second_hash,
                embedding: &[1.0, 0.0],
            }),
            Err(Error::VectorCapReached {
                current: 1,
                max_l2_units: 1
            })
        ));
        assert!(matches!(
            store.insert_vector(VectorInsert {
                table_slug: "cap_2",
                unit_id: 1,
                ctx_hash: &[4; 32],
                embedding_hash: &[9; 32],
                embedding: &[1.0, 0.0],
            }),
            Err(Error::VectorFenceMismatch { unit_id: 1 })
        ));
        assert!(matches!(
            store.insert_vector(VectorInsert {
                table_slug: "cap_2",
                unit_id: 1,
                ctx_hash: &[5; 32],
                embedding_hash: &first_hash,
                embedding: &[f32::NAN, 0.0],
            }),
            Err(Error::InvalidVectorValues { .. })
        ));
        Ok(())
    }

    #[test]
    fn exact_float16_top_k_matches_bruteforce_on_ten_thousand_vectors()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.ensure_vector_table(table("bruteforce_8", DIMS))?;
        let mut state = 7_u64;
        let vectors = (0..10_000)
            .map(|_| normalized(&mut state, DIMS))
            .collect::<Vec<_>>();
        let query = normalized(&mut state, DIMS);
        let table = vector_table_name("bruteforce_8")?;
        {
            let mut writer = store.writer()?;
            let transaction = writer.transaction()?;
            for (index, vector) in vectors.iter().enumerate() {
                let id = i64::try_from(index + 1)?;
                let mut hash = [0_u8; 32];
                hash[..8].copy_from_slice(&(index as u64).to_le_bytes());
                transaction.execute("INSERT INTO unit_contents(hash, text, tokens) VALUES (?1, 'vector fixture', 1)", [hash.as_slice()])?;
                transaction.execute(
                    "INSERT INTO units (id, hash, content_hash, source_key, kind, header, created_at, updated_at)
                     VALUES (?1, ?2, ?2, 'vector:' || ?1, 'chunk', 'vector fixture', 1, 1)",
                    params![id, hash.as_slice()],
                )?;
                transaction.execute(
                    &format!("INSERT INTO {table} (unit_id, v, ctx_hash) VALUES (?1, vector_as_f16(?2, ?3), ?4)"),
                    params![id, f16_blob(vector), DIMS as i64, &[0_u8; 32]],
                )?;
            }
            transaction.commit()?;
        }
        let mut expected = vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                (
                    i64::try_from(index + 1).unwrap_or_default(),
                    f16_cosine_distance(&query, vector),
                )
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));
        let actual =
            store.scan_vectors("bruteforce_8", &query, 10, VectorScanPredicate::default())?;
        assert_eq!(actual.len(), 10);
        let boundary = expected[9].1 + 1e-5;
        for hit in actual {
            assert!(hit.distance <= boundary);
            assert!(expected.iter().any(|(unit_id, distance)| {
                *unit_id == hit.unit_id && (distance - hit.distance).abs() < 1e-5
            }));
        }
        Ok(())
    }

    #[test]
    #[ignore = "bench-machine FLOAT16 exact-scan measurement; run with --ignored --nocapture"]
    fn records_exact_scan_latency_at_one_hundred_and_three_hundred_thousand_vectors()
    -> Result<(), Box<dyn StdError>> {
        for count in [100_000_usize, 300_000] {
            let file = tempfile::NamedTempFile::new()?;
            let store = open(file.path(), StoreLimits::default())?;
            store.ensure_vector_table(table("latency_1024", 1_024))?;
            let mut state = 11_u64;
            let table = vector_table_name("latency_1024")?;
            let query = normalized(&mut state, 1_024);
            let started = Instant::now();
            {
                let mut writer = store.writer()?;
                let transaction = writer.transaction()?;
                for index in 0..count {
                    let id = i64::try_from(index + 1)?;
                    let mut hash = [0_u8; 32];
                    hash[..8].copy_from_slice(&(index as u64).to_le_bytes());
                    transaction.execute("INSERT INTO unit_contents(hash, text, tokens) VALUES (?1, 'vector fixture', 1)", [hash.as_slice()])?;
                    transaction.execute(
                        "INSERT INTO units (id, hash, content_hash, source_key, kind, header, created_at, updated_at)
                         VALUES (?1, ?2, ?2, 'vector:' || ?1, 'chunk', 'vector fixture', 1, 1)",
                        params![id, hash.as_slice()],
                    )?;
                    let vector = normalized(&mut state, 1_024);
                    transaction.execute(
                        &format!("INSERT INTO {table} (unit_id, v, ctx_hash) VALUES (?1, vector_as_f16(?2, ?3), ?4)"),
                        params![id, f16_blob(&vector), 1_024_i64, &[0_u8; 32]],
                    )?;
                }
                transaction.commit()?;
            }
            let populate = started.elapsed();
            let mut p95s = Vec::new();
            for repetition in 0..3 {
                for _ in 0..5 {
                    store.scan_vectors(
                        "latency_1024",
                        &query,
                        20,
                        VectorScanPredicate::default(),
                    )?;
                }
                let mut timings = Vec::new();
                for _ in 0..50 {
                    let scan_started = Instant::now();
                    let hits = store.scan_vectors(
                        "latency_1024",
                        &query,
                        20,
                        VectorScanPredicate::default(),
                    )?;
                    assert_eq!(hits.len(), 20);
                    timings.push(scan_started.elapsed().as_secs_f64() * 1_000.0);
                }
                timings.sort_by(f64::total_cmp);
                let p95 = timings[47];
                p95s.push(p95);
                eprintln!(
                    "vector_scan count={count} dims=1024 repetition={repetition} warmups=5 samples=50 p50_ms={:.3} p95_ms={p95:.3}",
                    timings[24]
                );
            }
            p95s.sort_by(f64::total_cmp);
            eprintln!(
                "vector_scan count={count} dims=1024 populate_ms={} median_p95_ms={:.3} optimized={} schema={} loadavg={}",
                populate.as_millis(),
                p95s[1],
                !cfg!(debug_assertions),
                store.get_meta("schema_version")?.unwrap_or_default(),
                std::fs::read_to_string("/proc/loadavg")
                    .unwrap_or_default()
                    .trim()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod replacement_tests;

#[cfg(test)]
mod lifecycle_tests;

#[cfg(test)]
mod cross_process_tests;

/// Scoped test evidence emitted only after consuming the actual vector SQL query.
#[cfg(test)]
pub(crate) mod scan_observation {
    use super::VectorScanPredicate;
    use std::{
        cell::RefCell,
        future::Future,
        sync::{Arc, Mutex},
    };

    #[derive(Clone, Debug)]
    pub(crate) struct Event {
        pub(crate) table: String,
        pub(crate) sql: String,
        pub(crate) kinds: Vec<crate::store::UnitKind>,
        pub(crate) replaced_by: Vec<(crate::store::UnitKind, String)>,
    }
    pub(crate) type Log = Arc<Mutex<Vec<Event>>>;
    tokio::task_local! { static TASK: Log; }
    thread_local! { static THREAD: RefCell<Option<Log>> = const { RefCell::new(None) }; }

    /// Captures only this future's routed scans, including its blocking scan task.
    pub(crate) async fn capture<F: Future>(future: F) -> (F::Output, Vec<Event>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let output = TASK.scope(Arc::clone(&log), future).await;
        let events = log.lock().unwrap().clone();
        (output, events)
    }
    /// Explicit transfer prevents unrelated tasks on the same executor inheriting hooks.
    pub(crate) fn current() -> Option<Log> {
        TASK.try_with(Arc::clone).ok()
    }
    pub(crate) struct Scope(Option<Log>);
    /// Installs only for the synchronous scan closure, restoring nested scopes on drop.
    pub(crate) fn enter(log: Option<Log>) -> Scope {
        Scope(THREAD.with(|slot| slot.replace(log)))
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            THREAD.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }
    pub(super) fn executed(table: &str, sql: &str, predicate: VectorScanPredicate<'_>) {
        THREAD.with(|slot| {
            if let Some(log) = slot.borrow().as_ref() {
                log.lock().unwrap().push(Event {
                    table: table.into(),
                    sql: sql.into(),
                    kinds: predicate.kinds.to_vec(),
                    replaced_by: predicate.replaced_by.to_vec(),
                });
            }
        });
    }
}
