//! Shared collapse identity and early exclusion eligibility; v1 §9 steps 3–8.
use super::{Error, UnitKind};
use rusqlite::{OptionalExtension, Transaction};

/// Persisted source authority for a bounded collapse contributor. This records
/// L1 publication identity, independently of L2 progress and filesystem freshness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourcePublication {
    pub(crate) path: String,
    pub(crate) hash: Vec<u8>,
    pub(crate) desired: Option<Vec<u8>>,
    completed: Option<Vec<u8>>,
    pending: bool,
    deferred: Option<String>,
    pub(crate) language: Option<String>,
    pub(crate) kind: String,
}

impl SourcePublication {
    /// Requires a completed, coherent L1 publication before its hidden evidence
    /// can support a complete selection. This does not verify filesystem bytes.
    pub(crate) fn is_complete(&self) -> bool {
        !self.pending
            && self.deferred.is_none()
            && self.hash.len() == 32
            && self.desired.as_ref().is_some_and(|value| value.len() == 32)
            && self.desired == self.completed
    }
}

/// Reads one contributing file in the caller's checked selection snapshot.
/// No filesystem scan or repair is implied by this database evidence.
pub(crate) fn source_publication(
    tx: &Transaction<'_>,
    file_id: i64,
) -> Result<Option<SourcePublication>, Error> {
    Ok(tx
        .query_row(
            "SELECT blake3,l1_desired,l1_fingerprint,l1_pending,l1_deferred_reason,lang,kind,path
         FROM files WHERE id=?1",
            [file_id],
            |r| {
                Ok(SourcePublication {
                    hash: r.get(0)?,
                    desired: r.get(1)?,
                    completed: r.get(2)?,
                    pending: r.get(3)?,
                    deferred: r.get(4)?,
                    language: r.get(5)?,
                    kind: r.get(6)?,
                    path: r.get(7)?,
                })
            },
        )
        .optional()?)
}

// `source` always denotes a units row in the caller's SQL. All expressions here
// are fixed SQL, never repository/query text. Keep range validation in one place.
const VALID_CHUNK: &str =
    "source.kind='chunk' AND source.byte_start>=0 AND source.byte_end>=source.byte_start";
const CONTAINING_SYMBOL:&str="SELECT candidate.id FROM units AS candidate
    WHERE candidate.kind='symbol' AND candidate.file_id=source.file_id
      AND candidate.byte_start<=source.byte_start AND candidate.byte_end>=source.byte_end
      AND candidate.byte_start>=0 AND candidate.byte_end>=candidate.byte_start
    ORDER BY candidate.byte_end-candidate.byte_start,candidate.byte_start DESC,candidate.id LIMIT 1";

/// Resolves a chunk's canonical containing symbol in the caller's read snapshot.
/// Byte bounds and the smallest-range tie-break match early exclusion expansion.
pub(crate) fn chunk_parent(tx: &Transaction<'_>, id: i64) -> Result<Option<i64>, Error> {
    Ok(tx.query_row(&format!("SELECT ({CONTAINING_SYMBOL}) FROM units AS source WHERE source.id=?1 AND {VALID_CHUNK}"),[id],|r|r.get::<_,Option<i64>>(0)).optional()?.flatten())
}

/// SQL fragments for filtering units aliased as `u`; the supplied positive binding
/// index receives the JSON array of original IDs, never interpolated ID text.
pub(crate) struct Exclusions {
    pub(crate) prefix: String,
    pub(crate) predicate: String,
}

/// Applies a structural search scope to the occurrence that a summary will
/// expose. Raw summaries have no file and retain their own embedding kind for
/// routing; scope inheritance must happen before candidate limits, not after
/// retrieval. UNION terminates corrupt parent cycles. Freshness and privacy
/// still require the independent source proofs in the search/queue layers.
pub(crate) fn scope_sql(predicate: &str) -> String {
    let inherited = predicate
        .replace("u.", "scope_owner.")
        .replace("f.", "scope_file.");
    format!(
        "((u.kind!='summary' AND ({predicate})) OR
          (u.kind='summary' AND EXISTS(
            WITH RECURSIVE scope_ancestors(id,parent_unit,kind,file_id) AS (
              SELECT id,parent_unit,kind,file_id FROM units WHERE id=u.parent_unit
              UNION
              SELECT parent.id,parent.parent_unit,parent.kind,parent.file_id
              FROM units parent JOIN scope_ancestors child ON parent.id=child.parent_unit
              WHERE child.kind='summary'
            )
            SELECT 1 FROM scope_ancestors scope_owner
            LEFT JOIN files scope_file ON scope_file.id=scope_owner.file_id
            WHERE scope_owner.kind!='summary' AND ({inherited})
          )))"
    )
}

/// Determines whether a named user scope can need summary vectors at all.
/// This reads occurrence metadata only, so an unrelated summary destination
/// cannot force model preparation for a scope with no eligible summary owners.
pub(crate) fn summaries_in_scope(tx: &Transaction<'_>, kinds: &[UnitKind]) -> Result<bool, Error> {
    if kinds.is_empty() {
        return Ok(true);
    }
    let predicate = scope_sql("u.kind IN (SELECT value FROM json_each(?1))");
    let names = kinds.iter().map(|kind| kind.as_str()).collect::<Vec<_>>();
    Ok(tx.query_row(
        &format!("SELECT EXISTS(SELECT 1 FROM units u WHERE u.kind='summary' AND {predicate})"),
        [serde_json::json!(names).to_string()],
        |row| row.get(0),
    )?)
}
/// Expands excluded symbols to chunks whose actual collapse target is that symbol,
/// then recursively excludes summary descendants. UNION terminates even corrupt
/// summary cycles. The expansion executes in the retrieval statement's snapshot,
/// before candidate bounds. Empty exclusions retain the cheap empty-set path.
pub(crate) fn exclusion_sql(ids: &[i64], parameter: usize) -> Exclusions {
    if ids.is_empty() {
        return Exclusions {
            prefix: String::new(),
            predicate: format!("u.id NOT IN (SELECT value FROM json_each(?{parameter}))"),
        };
    }
    Exclusions {
        prefix: format!(
            "WITH RECURSIVE excluded_units(id) AS (
            SELECT value FROM json_each(?{parameter})
            UNION
            SELECT source.id FROM json_each(?{parameter}) AS requested
            JOIN units AS target ON target.id=requested.value AND target.kind='symbol'
            JOIN units AS source ON source.file_id=target.file_id
            WHERE {VALID_CHUNK}
              AND source.byte_start>=target.byte_start AND source.byte_end<=target.byte_end
              AND target.id=({CONTAINING_SYMBOL})
            UNION
            SELECT summary.id FROM units AS summary
            JOIN excluded_units AS parent ON summary.parent_unit=parent.id
            WHERE summary.kind='summary'
        ) "
        ),
        predicate: "u.id NOT IN (SELECT id FROM excluded_units)".into(),
    }
}

#[cfg(test)]
mod tests;
