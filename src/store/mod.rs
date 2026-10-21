//! SQLite persistence and migrations; see `v1.md` §3.
pub(crate) mod enrichment;
pub(crate) mod linker;
mod module_graph;
pub(crate) mod modules;
pub(crate) mod symbol_history;

pub(crate) mod admission;
pub(crate) mod capacity;
pub(crate) mod checked_writer;
pub(crate) mod diagnostics;
pub(crate) mod embedding_bindings;
mod embedding_catalogue;
pub(crate) mod embedding_freshness;
mod existing;
mod graph_reuse;
pub(crate) mod history;
pub(crate) mod l1;
pub(crate) mod l1_session;
#[cfg(test)]
mod main_cap_tests;
mod maintenance;
mod shrink;
pub(crate) use shrink::ShrinkReport;
#[cfg(test)]
pub(crate) mod migration_tests;
mod names;
mod occurrences;
pub(crate) mod path;
pub(crate) mod projection;
pub(crate) mod prs;
#[cfg(test)]
mod public_id_tests;
mod target_snapshot;
pub(crate) use target_snapshot::CatalogueSource;
pub(crate) mod transitions;
pub(crate) mod unit_contexts;
pub(crate) mod vectors;
pub(crate) mod wal;
pub(crate) use maintenance::Maintenance;
use occurrences::insert_unit_in_transaction;
pub(crate) use occurrences::{IndexedUnitInput, PublicationOutcome};
#[cfg(test)]
pub(crate) use occurrences::{PublicationPoint, observe_publication, publication_observed};

use std::{
    fs,
    path::Path,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

use rusqlite::{Connection, OptionalExtension, params};

const READ_POOL_SIZE: usize = 4;
// Pack multiple 2 KiB FLOAT16 vector rows per page without the write
// amplification of 64 KiB pages during per-file L1 replacement. Existing
// databases retain their layout; only genuinely fresh databases use this size.
const SQLITE_PAGE_SIZE: i64 = 16_384;
// File-backed mmap keeps an exact scan's vector pages out of the bounded SQLite
// page cache while limiting the virtual mapping below the default database cap.
const SQLITE_MMAP_SIZE: i64 = 1_024 * 1_024 * 1_024;
const LATEST_SCHEMA_VERSION: i64 = 23;
const META_DEFAULTS: [(&str, &str); 7] = [
    ("chunker_version", "1"),
    ("dims", "1024"),
    ("indexed_at", "0"),
    ("indexed_head", ""),
    ("profile", ""),
    ("repo_root", ""),
    ("schema_version", "23"),
];
const MIGRATIONS: [Migration; 23] = [
    Migration {
        version: 1,
        sql: include_str!("migrations/0001_initial.sql"),
    },
    Migration {
        version: 2,
        sql: include_str!("migrations/0002_file_rank.sql"),
    },
    Migration {
        version: 3,
        sql: include_str!("migrations/0003_file_truncated.sql"),
    },
    Migration {
        version: 4,
        sql: include_str!("migrations/0004_l1_pending.sql"),
    },
    Migration {
        version: 5,
        sql: include_str!("migrations/0005_unit_occurrences.sql"),
    },
    Migration {
        version: 6,
        sql: include_str!("migrations/0006_l1_publication.sql"),
    },
    Migration {
        version: 7,
        sql: include_str!("migrations/0007_graph_lookup_indexes.sql"),
    },
    Migration {
        version: 8,
        sql: include_str!("migrations/0008_graph_pending.sql"),
    },
    Migration {
        version: 9,
        sql: include_str!("migrations/0009_queue_generations.sql"),
    },
    Migration {
        version: 10,
        sql: include_str!("migrations/0010_dispatch_admission.sql"),
    },
    Migration {
        version: 11,
        sql: include_str!("migrations/0011_outbound_identity.sql"),
    },
    Migration {
        version: 12,
        sql: include_str!("migrations/0012_embedding_bindings.sql"),
    },
    Migration {
        version: 13,
        sql: include_str!("migrations/0013_embedding_transition.sql"),
    },
    Migration {
        version: 14,
        sql: include_str!("migrations/0014_embedding_publications.sql"),
    },
    Migration {
        version: 15,
        sql: include_str!("migrations/0015_l1_delete_indexes.sql"),
    },
    Migration {
        version: 16,
        sql: include_str!("migrations/0016_verified_git_blobs.sql"),
    },
    Migration {
        version: 17,
        sql: include_str!("migrations/0017_module_summary.sql"),
    },
    Migration {
        version: 18,
        sql: include_str!("migrations/0018_vlm_pages.sql"),
    },
    Migration {
        version: 19,
        sql: include_str!("migrations/0019_page_requirements.sql"),
    },
    Migration {
        version: 20,
        sql: include_str!("migrations/0020_role_free_embedding.sql"),
    },
    Migration {
        version: 21,
        sql: include_str!("migrations/0021_unit_contexts.sql"),
    },
    Migration {
        version: 22,
        sql: include_str!("migrations/0022_edge_destinations.sql"),
    },
    Migration {
        version: 23,
        sql: include_str!("migrations/0023_vector_count_indexes.sql"),
    },
];

struct Migration {
    version: i64,
    sql: &'static str,
}

/// Limits that affect SQLite connection configuration before full config loading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StoreLimits {
    /// Hard main-file cap; a lowered limit preserves the existing allocation.
    pub(crate) db_max_bytes: u64,
    /// Upper bound for SQLite's retained write-ahead log, in bytes.
    pub(crate) wal_max_bytes: u64,
    /// Maximum number of L2 vector rows across every registered vector table.
    pub(crate) max_l2_units: u64,
    /// Maximum retained L3 summary occurrences.
    pub(crate) max_l3_units: u64,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            db_max_bytes: 2 * 1024 * 1024 * 1024,
            wal_max_bytes: 64 * 1024 * 1024,
            max_l2_units: 300_000,
            max_l3_units: 20_000,
        }
    }
}

/// A repository file record written by the sync phase.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FileUpsert<'a> {
    pub(crate) path: &'a str,
    pub(crate) blake3: &'a [u8],
    pub(crate) size: i64,
    pub(crate) mtime_ns: i64,
    pub(crate) lang: Option<&'a str>,
    pub(crate) kind: FileKind,
    pub(crate) git_blob: Option<&'a [u8]>,
    pub(crate) max_level: i64,
    pub(crate) treatment: &'a str,
    pub(crate) indexed_at: Option<i64>,
    pub(crate) needs_vlm: bool,
}

/// A file identity and digest from the last completed filesystem synchronization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IndexedFile {
    pub(crate) id: i64,
    pub(crate) path: String,
    pub(crate) blake3: Vec<u8>,
    pub(crate) size: i64,
    pub(crate) mtime_ns: i64,
    pub(crate) lang: Option<String>,
    pub(crate) kind: String,
    pub(crate) git_blob: Option<Vec<u8>>,
    pub(crate) max_level: i64,
    pub(crate) treatment: String,
}

/// Completed source generation used to fence source-backed navigation reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OpenSource {
    pub(crate) language: Option<String>,
    pub(crate) kind: String,
    pub(crate) hash: Vec<u8>,
    pub(crate) fingerprint: Vec<u8>,
    pub(crate) occurrence: Vec<u8>,
    pub(crate) scope_path: Option<String>,
}

/// A directory Merkle record from the last completed filesystem synchronization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IndexedDir {
    pub(crate) path: String,
    pub(crate) merkle: Vec<u8>,
    pub(crate) file_count: i64,
}

/// The file and directory records needed to compute an incremental synchronization.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct IndexSnapshot {
    pub(crate) files: Vec<IndexedFile>,
    pub(crate) dirs: Vec<IndexedDir>,
}

/// A directory Merkle record to persist after a filesystem synchronization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirUpsert<'a> {
    pub(crate) path: &'a str,
    pub(crate) merkle: &'a [u8],
    pub(crate) file_count: i64,
}

/// Broad file kinds accepted by the `files` schema check constraint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FileKind {
    Code,
    Doc,
    Config,
    Image,
    Data,
    Other,
}

impl FileKind {
    /// Returns the stable schema value used in catalogue and input fingerprints.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Doc => "doc",
            Self::Config => "config",
            Self::Image => "image",
            Self::Data => "data",
            Self::Other => "other",
        }
    }

    /// Converts a schema-validated kind string back into its typed representation.
    pub(crate) fn from_db(value: &str) -> Option<Self> {
        match value {
            "code" => Some(Self::Code),
            "doc" => Some(Self::Doc),
            "config" => Some(Self::Config),
            "image" => Some(Self::Image),
            "data" => Some(Self::Data),
            "other" => Some(Self::Other),
            _ => None,
        }
    }
}

/// A unit record and its derived contentless FTS columns.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnitInsert<'a> {
    /// Path-independent content hash; the store derives the occurrence hash.
    pub(crate) hash: &'a [u8],
    /// Required logical owner for fileless units; files use their persisted path.
    pub(crate) source_key: Option<&'a str>,
    pub(crate) duplicate_ordinal: u32,
    pub(crate) file_id: Option<i64>,
    pub(crate) kind: UnitKind,
    pub(crate) byte_start: Option<i64>,
    pub(crate) byte_end: Option<i64>,
    pub(crate) line_start: Option<i64>,
    pub(crate) line_end: Option<i64>,
    pub(crate) scope_path: Option<&'a str>,
    pub(crate) header: &'a str,
    pub(crate) text: &'a str,
    pub(crate) tokens: i64,
    pub(crate) ladder_level: i64,
    pub(crate) parent_unit: Option<i64>,
    pub(crate) created_at: i64,
    pub(crate) updated_at: i64,
    pub(crate) fts: FtsColumns<'a>,
}

/// Derived contentless FTS columns for one unit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FtsColumns<'a> {
    pub(crate) header: &'a str,
    pub(crate) idents: &'a str,
    pub(crate) body: &'a str,
    pub(crate) prose: &'a str,
}

/// One derived FTS row used to rebuild the contentless lexical index.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FtsRebuildRow<'a> {
    pub(crate) unit_id: i64,
    pub(crate) columns: FtsColumns<'a>,
}

/// The referent category represented by a trigram name-index row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FtsNameKind {
    Symbol,
    Path,
}

impl FtsNameKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Symbol => "symbol",
            Self::Path => "path",
        }
    }
}

/// Unit kinds accepted by the `units` schema check constraint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UnitKind {
    Chunk,
    Symbol,
    DocSection,
    ConfigKey,
    Image,
    Commit,
    PullRequest,
    History,
    Module,
    Summary,
}

impl UnitKind {
    /// Every stored unit kind participates in global embedding-route transitions.
    pub(crate) const ALL: [Self; 10] = [
        Self::Chunk,
        Self::Symbol,
        Self::DocSection,
        Self::ConfigKey,
        Self::Image,
        Self::Commit,
        Self::PullRequest,
        Self::History,
        Self::Module,
        Self::Summary,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::Symbol => "symbol",
            Self::DocSection => "doc_section",
            Self::ConfigKey => "config_key",
            Self::Image => "image",
            Self::Commit => "commit",
            Self::PullRequest => "pr",
            Self::History => "history",
            Self::Module => "module",
            Self::Summary => "summary",
        }
    }
}

/// Named kind group or a union of literal SQLite path globs, applied before limits.
#[derive(Clone, Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub(crate) enum SearchScope {
    One(String),
    Globs(Vec<String>),
}
impl From<String> for SearchScope {
    fn from(value: String) -> Self {
        Self::One(value)
    }
}
impl From<&str> for SearchScope {
    fn from(value: &str) -> Self {
        Self::One(value.to_owned())
    }
}
impl rusqlite::types::ToSql for SearchScope {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        let value = match self {
            Self::One(value) => serde_json::json!({"one": value}),
            Self::Globs(globs) => serde_json::json!({"globs": globs}),
        };
        Ok(rusqlite::types::ToSqlOutput::Owned(
            value.to_string().into(),
        ))
    }
}

// A single JSON parameter preserves binding numbers shared by all three legs.
const SEARCH_SCOPE_SQL: &str = "(?3 IS NULL OR
    CASE WHEN json_type(?3, '$.globs') = 'array' THEN
        EXISTS(SELECT 1 FROM json_each(?3, '$.globs') g WHERE f.path GLOB g.value)
    ELSE CASE json_extract(?3, '$.one')
             WHEN 'source' THEN u.kind IN ('chunk','symbol','config_key','doc_section','image')
             WHEN 'code' THEN u.kind IN ('chunk','symbol','config_key')
             WHEN 'docs' THEN u.kind IN ('doc_section','image','module')
             WHEN 'images' THEN u.kind = 'image'
             WHEN 'commits' THEN u.kind = 'commit'
             WHEN 'prs' THEN u.kind IN ('pr','history')
             ELSE f.path GLOB json_extract(?3, '$.one') END END)";

/// One lexical result from the contentless FTS index.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FtsHit {
    pub(crate) unit_id: i64,
    pub(crate) score: f64,
}

/// One trigram name-index result for symbol and path autocomplete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FtsNameHit {
    pub(crate) name: String,
    pub(crate) kind: FtsNameKind,
    pub(crate) ref_id: i64,
}

/// A symbol row derived from one source file's tags query.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SymbolInsert<'a> {
    pub(crate) unit_id: Option<i64>,
    pub(crate) name: &'a str,
    pub(crate) qualified: Option<&'a str>,
    pub(crate) kind: &'a str,
    pub(crate) signature: Option<&'a str>,
    pub(crate) line_start: i64,
    pub(crate) line_end: i64,
}

/// An edge whose endpoints are symbols written with the same indexed unit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnitEdgeInsert<'a> {
    pub(crate) src_symbol_index: usize,
    pub(crate) dst_name: &'a str,
    pub(crate) dst_symbol: Option<i64>,
    pub(crate) dst_symbol_index: Option<usize>,
    pub(crate) kind: GraphEdgeKind,
    pub(crate) line: Option<i64>,
}

/// The rows created by one atomic L1 unit write.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IndexedUnit {
    pub(crate) unit_id: i64,
    pub(crate) symbols: Vec<GraphSymbol>,
}

/// A persisted symbol with the file path needed for graph resolution.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphSymbol {
    pub(crate) id: i64,
    pub(crate) file_id: i64,
    pub(crate) path: String,
    pub(crate) name: String,
    pub(crate) qualified: Option<String>,
    pub(crate) kind: String,
    pub(crate) signature: Option<String>,
    pub(crate) line_start: i64,
    pub(crate) line_end: i64,
    pub(crate) pagerank: f64,
}

/// The schema-supported category of an approximate code-graph edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphEdgeKind {
    Ref,
    Import,
    DocMention,
    TestOf,
}

impl GraphEdgeKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ref => "ref",
            Self::Import => "import",
            Self::DocMention => "doc_mention",
            Self::TestOf => "test_of",
        }
    }

    fn from_db(value: &str) -> Result<Self, Error> {
        match value {
            "ref" => Ok(Self::Ref),
            "import" => Ok(Self::Import),
            "doc_mention" => Ok(Self::DocMention),
            "test_of" => Ok(Self::TestOf),
            unexpected => Err(Error::InvalidGraphEdgeKind {
                value: unexpected.to_owned(),
            }),
        }
    }
}

/// An edge row with a source symbol from the same file.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphEdgeInsert<'a> {
    pub(crate) src_symbol: i64,
    pub(crate) dst_name: &'a str,
    pub(crate) dst_symbol: Option<i64>,
    pub(crate) kind: GraphEdgeKind,
    pub(crate) line: Option<i64>,
}

/// A persisted approximate graph edge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphEdge {
    pub(crate) src_symbol: i64,
    pub(crate) dst_name: String,
    pub(crate) dst_symbol: Option<i64>,
    pub(crate) kind: GraphEdgeKind,
    pub(crate) file_id: i64,
    pub(crate) line: Option<i64>,
}

/// The graph rows required for extraction resolution and PageRank.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct GraphSnapshot {
    pub(crate) symbols: Vec<GraphSymbol>,
    pub(crate) edges: Vec<GraphEdge>,
    pub(crate) files: Vec<GraphFileRank>,
}

/// A persisted file rank row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphFileRank {
    pub(crate) file_id: i64,
    pub(crate) rank: f64,
}

/// A graph-neighbour unit candidate ordered by its persisted PageRank.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphUnitHit {
    pub(crate) unit_id: i64,
    pub(crate) rank: f64,
}

/// A symbol record enriched with its owning unit for tool-facing read queries.
///
/// The CLI uses this read model to format stable public symbol identifiers while
/// keeping SQLite row ids confined to this module.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ToolSymbol {
    pub(crate) id: i64,
    pub(crate) unit_id: Option<i64>,
    pub(crate) file_id: i64,
    pub(crate) path: String,
    pub(crate) name: String,
    pub(crate) qualified: Option<String>,
    pub(crate) kind: String,
    pub(crate) signature: Option<String>,
    pub(crate) line_start: i64,
    pub(crate) line_end: i64,
    pub(crate) pagerank: f64,
}

/// A ranked repository file used to build the compact `map` response.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MapFile {
    pub(crate) id: i64,
    pub(crate) path: String,
    pub(crate) rank: f64,
}

/// One symbol rank value to persist after a PageRank rebuild.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphSymbolRank {
    pub(crate) symbol_id: i64,
    pub(crate) rank: f64,
}

/// The persisted unit-cap state returned for a file in status reporting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FileTruncation {
    pub(crate) truncated: bool,
}

/// One stored unit with its repository-relative path for command results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StoredUnit {
    pub(crate) id: i64,
    pub(crate) hash: Vec<u8>,
    pub(crate) path: Option<String>,
    pub(crate) kind: String,
    pub(crate) line_start: Option<i64>,
    pub(crate) line_end: Option<i64>,
    pub(crate) header: String,
    pub(crate) text: String,
    pub(crate) ladder_level: i64,
}

/// Counts surfaced by the phase-0 status command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StoreCounts {
    pub(crate) files: i64,
    pub(crate) units: i64,
    pub(crate) symbols: i64,
    pub(crate) edges: i64,
    pub(crate) truncated_files: i64,
    pub(crate) l1_pending_files: i64,
}

/// Measured SQLite database footprint for `status` and cap enforcement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Footprint {
    pub(crate) bytes: i64,
    pub(crate) page_count: i64,
    pub(crate) page_size: i64,
    pub(crate) freelist_count: i64,
    pub(crate) tables: Vec<TableFootprint>,
}

/// A `dbstat` size estimate for one schema object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TableFootprint {
    pub(crate) name: String,
    pub(crate) bytes: i64,
}

/// Failures from SQLite storage setup and typed persistence operations.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("L1 inputs changed before publication; index again")]
    L1InputsChanged,
    #[error("max_files cap ({limit}) prevents catalogue growth; add ignores or raise the limit")]
    FileCapacity { limit: u64 },
    #[error("summary publication generation is invalid or exhausted")]
    SummaryGeneration,
    #[error("L1 scan generation {generation} was superseded; index again")]
    L1ScanSuperseded { generation: i64 },
    #[error("L1 scan generation {generation} still has incomplete work")]
    L1Incomplete { generation: i64 },
    #[error("file {file_id} changed during L1 preparation; index again")]
    FileIndexSuperseded { file_id: i64 },
    #[error("duplicate occurrence {unit_id} in one file publication")]
    DuplicateOccurrence { unit_id: i64 },
    #[error("fileless units require a logical source key")]
    UnitSourceRequired,
    #[error("content hash is already associated with different text")]
    ContentHashConflict,
    #[error("retained commit has invalid file provenance")]
    InvalidHistoryFileProvenance,
    #[error("unit id {public_id} is ambiguous; open by path or a current search hit id")]
    AmbiguousUnitId { public_id: String },
    #[error("occurrence migration failed foreign-key validation")]
    MigrationForeignKeys,
    #[error("trigram name index contains invalid or duplicate source identities")]
    InvalidNameIndex,

    #[error("failed to prepare database directory: {source}")]
    Io {
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    DatabasePath(#[from] path::Error),
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error("SQLite could not enforce the main database growth limit")]
    DatabaseGrowthLimit,
    #[error(
        "WAL cap {wal_max_bytes} bytes cannot fit {reservation_bytes} reserved bytes after {allocated_bytes} existing bytes"
    )]
    WalCapacity {
        wal_max_bytes: u64,
        allocated_bytes: u64,
        reservation_bytes: u64,
    },
    #[error("WAL reservation requires a file-backed NORMAL synchronous WAL database")]
    WalReservationUnsupported,
    #[error(transparent)]
    VectorRegistration(#[from] sqlite_vector_sys::RegistrationError),
    #[error("the {connection} SQLite connection lock was poisoned")]
    LockPoisoned { connection: &'static str },
    #[error("database schema version {found} is newer than supported version {supported}")]
    UnsupportedSchema { found: i64, supported: i64 },
    #[error("database schema version is invalid: {value}")]
    InvalidSchemaVersion { value: String },
    #[error("migration order is invalid: expected version {expected}, found {found}")]
    MigrationOrder { expected: i64, found: i64 },
    #[error("FTS result limit {limit} exceeds SQLite's supported range")]
    FtsLimitTooLarge { limit: usize },
    #[error("graph file {file_id} does not exist")]
    GraphFileMissing { file_id: i64 },
    #[error("graph source symbol {src_symbol} does not belong to file {file_id}")]
    GraphSourceFileMismatch { src_symbol: i64, file_id: i64 },
    #[error("an indexed unit needs a file before it can write graph rows")]
    GraphUnitFileRequired,
    #[error("graph symbol index {index} is outside the {symbols} symbols written with this unit")]
    GraphSymbolIndexOutOfBounds { index: usize, symbols: usize },
    #[error("graph edge destination cannot name both an existing and local symbol")]
    GraphDestinationAmbiguous,
    #[error("stored graph edge kind is invalid: {value}")]
    InvalidGraphEdgeKind { value: String },
    #[error("WAL maximum {wal_max_bytes} cannot be represented by SQLite")]
    WalLimitTooLarge { wal_max_bytes: u64 },
    #[error("vector table slug {slug:?} must contain only ASCII letters, digits, or underscores")]
    InvalidVectorTableSlug { slug: String },
    #[error("vector table {slug:?} requires 1 through 2147483647 dimensions")]
    InvalidVectorDimensions { slug: String },
    #[error("vector table {slug:?} is not registered")]
    VectorTableMissing { slug: String },
    #[error(
        "vector table {slug:?} metadata does not match its requested model, profile, or dimensions"
    )]
    VectorTableMismatch { slug: String },
    #[error("embedding transition refused: {reason}")]
    EmbeddingTransition { reason: &'static str },
    #[error("embedding provider binding refused: {reason}")]
    EmbeddingBinding { reason: &'static str },
    #[error("native vector table capacity reached ({max_tables} tables)")]
    VectorTableCapacity { max_tables: i64 },
    #[error("vector table {slug:?} expects {expected} dimensions, received {actual}")]
    VectorDimensionMismatch {
        slug: String,
        expected: usize,
        actual: usize,
    },
    #[error(
        "L2 vector cap hit ({current} / {max_l2_units}); increase limits.max_l2_units to schedule more vectors"
    )]
    VectorCapReached { current: u64, max_l2_units: u64 },
    #[error("max_l3_units cap; L3 units {current} / {max_l3_units}")]
    SummaryCapReached { current: u64, max_l3_units: u64 },
    #[error("max_units_per_file cap; units {current} / {max_units_per_file}")]
    FileUnitCapReached {
        current: u64,
        max_units_per_file: u64,
    },
    #[error("database 85 percent soft cap; L3 paused ({allocated} / {cap})")]
    EnrichmentDatabasePaused { allocated: u64, cap: u64 },
    #[error("vector unit {unit_id} does not exist")]
    VectorUnitMissing { unit_id: i64 },
    #[error("vector write for unit {unit_id} was fenced because its embedding hash is stale")]
    VectorFenceMismatch { unit_id: i64 },
    #[error("vector scan limit {limit} exceeds SQLite's supported range")]
    VectorLimitTooLarge { limit: usize },
    #[error("vector for table {slug:?} must contain finite values with a non-zero L2 norm")]
    InvalidVectorValues { slug: String },
    #[error("vector row count {count} is invalid")]
    InvalidVectorCount { count: i64 },
}

/// A single-writer, four-reader SQLite store for one repository index.
pub(crate) struct Store {
    /// Opening-time authority retained only by checked existing-ledger opens.
    database_authority: Option<path::Prepared>,
    /// Main-file cap from the configuration used to open this store.
    pub(crate) db_max_bytes: u64,
    /// WAL admission cap from the effective configuration used to open this store.
    pub(crate) wal_max_bytes: u64,
    writer: Mutex<Connection>,
    readers: Vec<Mutex<Connection>>,
    next_reader: AtomicUsize,
    max_l2_units: u64,
    max_l3_units: u64,
}

/// Opens a configured index only inside the canonical repository's .deepkit directory.
/// Preflight rejects traversal, linked ancestry and unsafe database/sidecar files
/// before ordinary SQLite setup can create directories or migrate a database.
pub(crate) fn open_repository(
    root: &Path,
    database: &Path,
    limits: StoreLimits,
) -> Result<Store, Error> {
    let destination = path::prepare(root, database)?;
    destination.check_for_open()?;
    open(path::system_path(&root.join(database)), limits)
}

/// Opens or migrates a trusted SQLite path; repository callers use `open_repository`.
pub(crate) fn open(path: impl AsRef<Path>, limits: StoreLimits) -> Result<Store, Error> {
    let path = path.as_ref();
    let in_memory = path == Path::new(":memory:");
    if !in_memory
        && let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| Error::Io { source })?;
    }

    register_vector_extension()?;
    let vfs = sqlite_vector_sys::register_wal_vfs()?;
    let fresh = is_fresh_database(path, in_memory)?;
    let writer = Connection::open_with_flags_and_vfs(path, Default::default(), vfs)?;
    configure_connection(&writer, limits, fresh)?;
    let mut store = Store {
        database_authority: None,
        db_max_bytes: limits.db_max_bytes,
        wal_max_bytes: limits.wal_max_bytes,
        writer: Mutex::new(writer),
        readers: Vec::with_capacity(READ_POOL_SIZE),
        next_reader: AtomicUsize::new(0),
        max_l2_units: limits.max_l2_units,
        max_l3_units: limits.max_l3_units,
    };
    migrate(&store)?;
    vectors::initialize_metadata(&store)?;
    vectors::initialize_registered_tables(&*store.writer()?)?;

    if !in_memory {
        for _ in 0..READ_POOL_SIZE {
            let reader = Connection::open_with_flags_and_vfs(path, Default::default(), vfs)?;
            configure_connection(&reader, limits, false)?;
            vectors::initialize_registered_tables(&reader)?;
            store.readers.push(Mutex::new(reader));
        }
    }

    Ok(store)
}

/// Opens only a verified existing ledger for accounted diagnostic inference.
/// Refuses initialization/migration; the canonical root bounds all SQLite paths.
pub(crate) fn open_existing_for_usage(
    root: &Path,
    database: &Path,
    limits: StoreLimits,
) -> Result<Store, existing::Error> {
    existing::open_existing_for_usage(root, database, limits)
}

/// Registers sqlite-vector before this process opens any application database.
pub(crate) fn register_vector_extension() -> Result<(), sqlite_vector_sys::RegistrationError> {
    sqlite_vector_sys::register_auto_extension()
}

impl Store {
    /// Rechecks a checked ledger's original file before outbound admission.
    /// Generic and in-memory stores retain their existing opening contract.
    pub(crate) fn checked_database_is_current(&self) -> bool {
        self.database_authority
            .as_ref()
            .is_none_or(|guard| guard.check_existing_database().is_ok())
    }

    /// Lists incomplete file work independently of filesystem change detection.
    pub(crate) fn pending_l1_paths(&self) -> Result<Vec<String>, Error> {
        self.with_reader(|connection| {
            connection
                .prepare("SELECT path FROM files WHERE l1_pending = 1 ORDER BY path")?
                .query_map([], |row| row.get(0))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(Into::into)
        })
    }

    /// Marks a full rebuild durably before it can discard any derived content.
    pub(crate) fn mark_all_l1_pending(&self) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::L1Growth, |transaction| {
            transaction.execute("UPDATE files SET l1_pending = 1", [])?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Records completion only while the observed source hash still matches.
    ///
    /// Whole-file publication and configuration fencing follow in T1.R1b;
    /// this marker makes interrupted per-unit writes recoverable in T1.R1a.
    pub(crate) fn complete_file_l1(
        &self,
        file_id: i64,
        expected_hash: &[u8],
    ) -> Result<bool, Error> {
        admission::transaction(self, admission::Kind::Control, |transaction| {
            let changed = transaction.execute(
                "UPDATE files SET l1_pending = 0 WHERE id = ?1 AND blake3 = ?2",
                params![file_id, expected_hash],
            )?;
            transaction.commit()?;
            Ok(changed == 1)
        })
    }

    /// Upserts a file and returns its SQLite row id for related units.
    pub(crate) fn upsert_file(&self, file: FileUpsert<'_>) -> Result<i64, Error> {
        admission::transaction(self, admission::Kind::Catalogue, |transaction| {
            let file_id = upsert_file_in_transaction(&transaction, file)?;
            transaction.commit()?;
            Ok(file_id)
        })
    }

    /// Returns the persisted file and directory identities used for an incremental sync.
    pub(crate) fn index_snapshot(&self) -> Result<IndexSnapshot, Error> {
        self.with_reader(|connection| {
            let files = connection
                .prepare("SELECT id, path, blake3, size, mtime_ns, lang, kind, git_blob, max_level, treatment FROM files ORDER BY path")?
                .query_map([], |row| {
                    Ok(IndexedFile {
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
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            let dirs = connection
                .prepare("SELECT path, merkle, file_count FROM dirs ORDER BY path")?
                .query_map([], |row| {
                    Ok(IndexedDir {
                        path: row.get(0)?,
                        merkle: row.get(1)?,
                        file_count: row.get(2)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IndexSnapshot { files, dirs })
        })
    }

    /// Renames one persisted file without replacing its SQLite identity or related units.
    pub(crate) fn rename_file(&self, from: &str, to: &str) -> Result<bool, Error> {
        admission::transaction(self, admission::Kind::Catalogue, |transaction| {
            let file_id: Option<i64> = transaction
                .query_row("SELECT id FROM files WHERE path = ?1", [from], |row| {
                    row.get(0)
                })
                .optional()?;
            let Some(file_id) = file_id else {
                return Ok(false);
            };
            transaction.execute(
                "UPDATE files SET path = ?1 WHERE path = ?2",
                params![to, from],
            )?;
            occurrences::rename_occurrences(&transaction, file_id, to)?;
            replace_path_name(&transaction, file_id, to)?;
            transaction.commit()?;
            Ok(true)
        })
    }

    /// Deletes a file and its derived lexical rows in one transaction.
    pub(crate) fn delete_file(&self, path: &str) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Catalogue, |transaction| {
            delete_file_in_transaction(&transaction, path)?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Upserts one directory Merkle record after its children have been synchronized.
    pub(crate) fn upsert_dir(&self, directory: DirUpsert<'_>) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Catalogue, |transaction| {
            transaction.execute(
            "INSERT INTO dirs (path, merkle, file_count) VALUES (?1, ?2, ?3)
             ON CONFLICT(path) DO UPDATE SET merkle = excluded.merkle, file_count = excluded.file_count",
            params![directory.path, directory.merkle, directory.file_count],
        )?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Removes a stale directory Merkle record after an incremental synchronization.
    pub(crate) fn delete_dir(&self, path: &str) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Catalogue, |transaction| {
            transaction.execute("DELETE FROM dirs WHERE path = ?1", [path])?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Persists whether a file hit its configured L1 unit cap for later status reporting.
    pub(crate) fn set_file_truncated(&self, file_id: i64, truncated: bool) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Control, |transaction| {
            let changed = transaction.execute(
                "UPDATE files SET truncated = ?1 WHERE id = ?2",
                params![truncated, file_id],
            )?;
            if changed == 0 {
                return Err(Error::GraphFileMissing { file_id });
            }
            transaction.commit()?;
            Ok(())
        })
    }

    /// Returns the persisted unit-cap state for one file when it still exists.
    pub(crate) fn file_truncation(&self, file_id: i64) -> Result<Option<FileTruncation>, Error> {
        self.with_reader(|connection| {
            connection
                .query_row(
                    "SELECT truncated FROM files WHERE id = ?1",
                    [file_id],
                    |row| {
                        Ok(FileTruncation {
                            truncated: row.get(0)?,
                        })
                    },
                )
                .optional()
                .map_err(Into::into)
        })
    }

    /// Removes a file's L1-derived rows while retaining its identity and catalogue metadata.
    pub(crate) fn clear_file_content(&self, file_id: i64) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Catalogue, |transaction| {
            transaction.execute(
                "DELETE FROM unit_sources WHERE source_kind = 'file' AND source_id = ?1",
                [file_id],
            )?;
            transaction.execute(
                "DELETE FROM fts_units WHERE rowid IN (SELECT id FROM units WHERE file_id = ?1)",
                [file_id],
            )?;
            names::remove_symbols(&transaction, file_id)?;
            transaction.execute("DELETE FROM symbols WHERE file_id = ?1", [file_id])?;
            transaction.execute("DELETE FROM units WHERE file_id = ?1", [file_id])?;
            transaction.execute(
                "UPDATE files SET truncated = 0, l1_pending = 1 WHERE id = ?1",
                [file_id],
            )?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Inserts a unit and its contentless FTS row in one transaction.
    pub(crate) fn insert_unit(&self, unit: UnitInsert<'_>) -> Result<i64, Error> {
        admission::transaction(self, admission::Kind::L1Growth, |transaction| {
            let before = (unit.kind == UnitKind::Summary
                || (unit.kind == UnitKind::DocSection && unit.ladder_level == 3))
                .then(|| self.enrichment_status_in(&transaction))
                .transpose()?;
            let unit_id = insert_unit_in_transaction(&transaction, unit, None)?;
            if let Some(before) = before {
                before.check_growth(&self.enrichment_status_in(&transaction)?)?;
            }
            transaction.commit()?;
            Ok(unit_id)
        })
    }

    /// Writes one L1 unit, its lexical row, and its graph rows in one transaction.
    ///
    /// New occurrences begin at L1; unchanged occurrences retain their vectors.
    /// Shared payloads retain separate locations, lexical rows and provenance.
    /// Symbols are attached to that occurrence regardless of the supplied symbol
    /// unit id, so callers cannot leave the graph and its source unit out of sync.
    pub(crate) fn insert_indexed_unit(
        &self,
        unit: UnitInsert<'_>,
        symbols: &[SymbolInsert<'_>],
        edges: &[UnitEdgeInsert<'_>],
    ) -> Result<IndexedUnit, Error> {
        admission::transaction(self, admission::Kind::L1Growth, |transaction| {
            let result = write_indexed_unit(&transaction, unit, symbols, edges, None)?;
            transaction.commit()?;
            Ok(result)
        })
    }

    /// Deletes a unit and its explicit contentless FTS row in one transaction.
    pub(crate) fn delete_unit(&self, unit_id: i64) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Catalogue, |transaction| {
            transaction.execute("DELETE FROM fts_units WHERE rowid = ?1", [unit_id])?;
            transaction.execute("DELETE FROM units WHERE id = ?1", [unit_id])?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Searches contentless FTS5 and returns row ids ordered by its BM25 score.
    pub(crate) fn query_fts(&self, query: &str, limit: usize) -> Result<Vec<FtsHit>, Error> {
        self.query_fts_scoped(query, limit, None, &[])
    }

    /// Filters occurrences before the lexical limit so unrelated files cannot exhaust a scope.
    pub(crate) fn query_fts_scoped(
        &self,
        query: &str,
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
    ) -> Result<Vec<FtsHit>, Error> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        i64::try_from(limit).map_err(|_| Error::FtsLimitTooLarge { limit })?;
        self.with_reader(|connection| {
            read_query_fts(connection, query, limit, scope, exclude_ids, &|| Ok(()))
        })
    }

    /// Reads lexical reference candidates under caller-owned admission and row checks.
    pub(crate) fn query_fts_checked(
        &self,
        query: &str,
        limit: usize,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<FtsHit>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_query_fts(connection, query, limit, None, &[], check)
        })?
    }

    /// Filters before the lexical limit under caller-owned admission and row checks.
    pub(crate) fn query_fts_scoped_checked(
        &self,
        query: &str,
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<FtsHit>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_query_fts(connection, query, limit, scope, exclude_ids, check)
        })?
    }

    /// Resolves exact trigram names to scoped symbol occurrences before limiting candidates.
    pub(crate) fn exact_name_units(
        &self,
        name: &str,
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
    ) -> Result<Vec<i64>, Error> {
        if name.chars().count() < 3 {
            return Ok(Vec::new());
        }
        i64::try_from(limit).map_err(|_| Error::FtsLimitTooLarge { limit })?;
        self.with_reader(|connection| {
            read_exact_name_units(connection, name, limit, scope, exclude_ids, &|| Ok(()))
        })
    }

    /// Resolves scoped exact names with caller-owned admission and per-row checks.
    #[cfg(test)]
    pub(crate) fn exact_name_units_checked(
        &self,
        name: &str,
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<i64>, Error> {
        self.exact_name_selection_checked(name, limit, scope, exclude_ids, check)
            .map(|(ids, _)| ids)
    }

    /// Retains the exact symbol contributors in the same snapshot as candidate
    /// admission, so final search validation can detect unchanged-ID replacements.
    pub(crate) fn exact_name_selection_checked(
        &self,
        name: &str,
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(Vec<i64>, Vec<String>), Error> {
        self.checked_read_snapshot(check, |connection| {
            let ids = read_exact_name_units(connection, name, limit, scope, exclude_ids, check)?;
            if ids.is_empty() {
                return Ok((ids, Vec::new()));
            }
            check()?;
            let query = format!("name:\"{}\"", name.replace('"', "\"\""));
            let mut statement = connection.prepare(
                "SELECT DISTINCT s.id, json_array(s.id,s.file_id,s.unit_id,s.name,
                    s.qualified,s.kind,s.signature,s.line_start,s.line_end)
                 FROM fts_names JOIN symbols s
                   ON s.id=fts_names.ref_id AND fts_names.kind='symbol'
                 WHERE fts_names MATCH ?1 AND s.name=?2
                   AND s.unit_id IN (SELECT value FROM json_each(?3))
                 ORDER BY s.id",
            )?;
            let mut rows =
                statement.query(params![query, name, serde_json::json!(ids).to_string()])?;
            let mut evidence = Vec::new();
            loop {
                check()?;
                let Some(row) = rows.next()? else { break };
                evidence.push(row.get(1)?);
            }
            check()?;
            Ok((ids, evidence))
        })?
    }

    /// Looks up one unit by its internal row id for a lexical result or open request.
    pub(crate) fn unit_by_id(&self, unit_id: i64) -> Result<Option<StoredUnit>, Error> {
        self.with_reader(|connection| read_unit_by_id(connection, unit_id, &|| Ok(())))
    }

    /// Reads navigation data under caller-owned admission/cancellation checks.
    /// Checks bracket the snapshot; individual SQLite statements are cooperative.
    pub(crate) fn unit_by_id_checked(
        &self,
        unit_id: i64,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Option<StoredUnit>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_unit_by_id(connection, unit_id, check)
        })?
    }

    /// Reads navigation data under caller-owned admission/cancellation checks.
    /// Checks bracket the snapshot; individual SQLite statements are cooperative.
    pub(crate) fn open_source_checked(
        &self,
        public_id: &str,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Option<OpenSource>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_open_source(connection, public_id, check)
        })?
    }

    /// Reads only the selected source's cap reason with cooperative admission.
    pub(crate) fn open_deferred_reason(
        &self,
        path: Option<&str>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Option<String>, Error> {
        self.checked_read_snapshot(check, |tx| {
            Ok(tx
                .query_row(
                    "SELECT l1_deferred_reason FROM files WHERE path=?1",
                    [path],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?
                .flatten())
        })?
    }

    /// Looks up one unit by the public id prefix exposed by the command interface.
    pub(crate) fn unit_by_public_id(&self, public_id: &str) -> Result<Option<StoredUnit>, Error> {
        self.with_reader(|connection| read_unit_by_public_id(connection, public_id, &|| Ok(())))
    }

    /// Reads navigation data under caller-owned admission/cancellation checks.
    /// Checks bracket the snapshot; individual SQLite statements are cooperative.
    pub(crate) fn unit_by_public_id_checked(
        &self,
        public_id: &str,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Option<StoredUnit>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_unit_by_public_id(connection, public_id, check)
        })?
    }

    /// Finds the first unit intersecting an optional line range in one file.
    pub(crate) fn unit_by_path(
        &self,
        path: &str,
        line_start: Option<i64>,
        line_end: Option<i64>,
    ) -> Result<Option<StoredUnit>, Error> {
        self.with_reader(|connection| {
            read_unit_by_path(connection, path, line_start, line_end, &|| Ok(()))
        })
    }

    /// Reads navigation data under caller-owned admission/cancellation checks.
    /// Checks bracket the snapshot; individual SQLite statements are cooperative.
    pub(crate) fn unit_by_path_checked(
        &self,
        path: &str,
        line_start: Option<i64>,
        line_end: Option<i64>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Option<StoredUnit>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_unit_by_path(connection, path, line_start, line_end, check)
        })?
    }

    /// Returns all symbols with the metadata needed by `symbol`, `refs`, and `map`.
    ///
    /// Ordering is deterministic so callers can apply stable exact and trigram
    /// matching without exposing SQLite's incidental row order.
    pub(crate) fn tool_symbols(&self) -> Result<Vec<ToolSymbol>, Error> {
        self.with_reader(|connection| read_tool_symbols(connection, &|| Ok(())))
    }

    /// Reads navigation data under caller-owned admission/cancellation checks.
    /// Checks bracket the snapshot; individual SQLite statements are cooperative.
    pub(crate) fn tool_symbols_checked(
        &self,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<ToolSymbol>, Error> {
        self.checked_read_snapshot(check, |connection| read_tool_symbols(connection, check))?
    }

    /// Reads symbols from one indexed path for source expansion and range navigation.
    ///
    /// This preserves the full-catalogue ordering within that file while avoiding
    /// a repository-wide symbol read on both open-source verification passes.
    pub(crate) fn tool_symbols_for_path_checked(
        &self,
        path: &str,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<ToolSymbol>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_tool_symbols_for_path(connection, path, check)
        })?
    }

    /// Reads symbols with an exact stored name without materializing the full catalogue.
    pub(crate) fn tool_symbols_named_checked(
        &self,
        name: &str,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<ToolSymbol>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_tool_symbols_named(connection, name, check)
        })?
    }

    /// Reads a selected symbol set without materializing unrelated repository symbols.
    pub(crate) fn tool_symbols_by_ids_checked(
        &self,
        ids: &[i64],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<ToolSymbol>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_tool_symbols_by_ids(connection, ids, check)
        })?
    }

    /// Lists paths with source occurrences so localization benchmarks cannot count history answers.
    pub(crate) fn searchable_source_paths(
        &self,
    ) -> Result<std::collections::BTreeSet<String>, Error> {
        self.with_reader(|connection| {
            connection
                .prepare(
                    "SELECT DISTINCT f.path FROM files f JOIN units u ON u.file_id = f.id
                 WHERE u.kind IN ('chunk','symbol','config_key','doc_section','image')",
                )?
                .query_map([], |row| row.get(0))?
                .collect::<Result<std::collections::BTreeSet<_>, _>>()
                .map_err(Into::into)
        })
    }

    /// Reads every map file and rank with caller-owned admission and row checks.
    ///
    /// Files without symbols remain present at rank zero so maps preserve the
    /// complete indexed catalogue instead of silently becoming code-only.
    pub(crate) fn map_files_checked(
        &self,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<MapFile>, Error> {
        self.checked_read_snapshot(check, |connection| {
            let mut statement =
                connection.prepare("SELECT id, path, rank FROM files ORDER BY path")?;
            let mut rows = statement.query([])?;
            let mut files = Vec::new();
            while let Some(row) = rows.next()? {
                check()?;
                files.push(MapFile {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    rank: row.get(2)?,
                });
            }
            check()?;
            Ok(files)
        })?
    }

    /// Returns persistent counts for status without exposing a raw connection.
    pub(crate) fn counts(&self) -> Result<StoreCounts, Error> {
        self.with_reader(|connection| {
            Ok(StoreCounts {
                files: count_rows(connection, "files")?,
                units: count_rows(connection, "units")?,
                symbols: count_rows(connection, "symbols")?,
                edges: count_rows(connection, "edges")?,
                truncated_files: connection.query_row(
                    "SELECT COUNT(*) FROM files WHERE truncated = 1",
                    [],
                    |row| row.get(0),
                )?,
                l1_pending_files: connection.query_row(
                    "SELECT COUNT(*) FROM files WHERE l1_pending = 1",
                    [],
                    |row| row.get(0),
                )?,
            })
        })
    }

    /// Reads persisted job-state counts without inferring worker health from missing activity.
    pub(crate) fn job_state_counts(
        &self,
    ) -> Result<std::collections::BTreeMap<String, i64>, Error> {
        self.with_reader(|connection| {
            connection
                .prepare("SELECT state, COUNT(*) FROM jobs GROUP BY state ORDER BY state")?
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<std::collections::BTreeMap<_, _>, _>>()
                .map_err(Into::into)
        })
    }

    /// Inserts one symbol or path name into the trigram autocomplete index.
    pub(crate) fn insert_fts_name(
        &self,
        name: &str,
        kind: FtsNameKind,
        ref_id: i64,
    ) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Growth, |transaction| {
            names::insert(&transaction, name, kind, ref_id)?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Finds names and paths by case-insensitive trigram substring matching.
    pub(crate) fn query_fts_names(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<FtsNameHit>, Error> {
        self.with_reader(|connection| read_query_fts_names(connection, query, limit, &|| Ok(())))
    }

    /// Reads trigram candidates with caller-owned admission and per-row cancellation.
    pub(crate) fn query_fts_names_checked(
        &self,
        query: &str,
        limit: usize,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<FtsNameHit>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_query_fts_names(connection, query, limit, check)
        })?
    }

    /// Replaces all contentless FTS rows from deterministic, caller-built columns.
    pub(crate) fn rebuild_fts(&self, rows: &[FtsRebuildRow<'_>]) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::L1Growth, |transaction| {
            transaction.execute("DELETE FROM fts_units", [])?;
            for row in rows {
                transaction.execute(
                "INSERT INTO fts_units (rowid, header, idents, body, prose) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![row.unit_id, row.columns.header, row.columns.idents, row.columns.body, row.columns.prose],
            )?;
            }
            names::rebuild(&transaction)?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Replaces one file's symbols and synchronized trigram names atomically.
    pub(crate) fn replace_file_symbols(
        &self,
        file_id: i64,
        symbols: &[SymbolInsert<'_>],
    ) -> Result<Vec<GraphSymbol>, Error> {
        admission::transaction(self, admission::Kind::L1Growth, |transaction| {
            let path: Option<String> = transaction
                .query_row("SELECT path FROM files WHERE id = ?1", [file_id], |row| {
                    row.get(0)
                })
                .optional()?;
            let Some(path) = path else {
                return Err(Error::GraphFileMissing { file_id });
            };
            names::remove_symbols(&transaction, file_id)?;
            transaction.execute("DELETE FROM symbols WHERE file_id = ?1", [file_id])?;

            let mut inserted = Vec::with_capacity(symbols.len());
            for symbol in symbols {
                transaction.execute(
                    "INSERT INTO symbols
                 (file_id, unit_id, name, qualified, kind, signature, line_start, line_end)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        file_id,
                        symbol.unit_id,
                        symbol.name,
                        symbol.qualified,
                        symbol.kind,
                        symbol.signature,
                        symbol.line_start,
                        symbol.line_end
                    ],
                )?;
                let id = transaction.last_insert_rowid();
                names::insert(&transaction, symbol.name, FtsNameKind::Symbol, id)?;
                inserted.push(GraphSymbol {
                    id,
                    file_id,
                    path: path.clone(),
                    name: symbol.name.to_owned(),
                    qualified: symbol.qualified.map(str::to_owned),
                    kind: symbol.kind.to_owned(),
                    signature: symbol.signature.map(str::to_owned),
                    line_start: symbol.line_start,
                    line_end: symbol.line_end,
                    pagerank: 0.0,
                });
            }
            transaction.commit()?;
            Ok(inserted)
        })
    }

    /// Replaces one file's edges after ensuring every source belongs to that file.
    pub(crate) fn replace_file_edges(
        &self,
        file_id: i64,
        edges: &[GraphEdgeInsert<'_>],
    ) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::L1Growth, |transaction| {
            for edge in edges {
                let source_file: Option<i64> = transaction
                    .query_row(
                        "SELECT file_id FROM symbols WHERE id = ?1",
                        [edge.src_symbol],
                        |row| row.get(0),
                    )
                    .optional()?;
                if source_file != Some(file_id) {
                    return Err(Error::GraphSourceFileMismatch {
                        src_symbol: edge.src_symbol,
                        file_id,
                    });
                }
            }
            transaction.execute("DELETE FROM edges WHERE file_id = ?1", [file_id])?;
            for edge in edges {
                transaction.execute(
                    "INSERT INTO edges (src_symbol, dst_name, dst_symbol, kind, file_id, line)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        edge.src_symbol,
                        edge.dst_name,
                        edge.dst_symbol,
                        edge.kind.as_str(),
                        file_id,
                        edge.line
                    ],
                )?;
            }
            transaction.commit()?;
            Ok(())
        })
    }

    /// Returns every persisted graph row in stable order for resolution and ranking.
    pub(crate) fn graph_snapshot(&self) -> Result<GraphSnapshot, Error> {
        self.with_reader(|connection| read_graph_snapshot(connection, &|| Ok(())))
    }

    /// Reads a coherent graph with caller-owned admission and per-row cancellation.
    pub(crate) fn graph_snapshot_checked(
        &self,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<GraphSnapshot, Error> {
        self.checked_read_snapshot(check, |connection| read_graph_snapshot(connection, check))?
    }

    /// Reads only edges that can affect the selected symbols' navigation response.
    ///
    /// Name matches remain included alongside resolved identifiers because unresolved
    /// references contribute to the same approximate caller and documentation fields.
    pub(crate) fn graph_edges_for_symbols_checked(
        &self,
        ids: &[i64],
        names: &[String],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<GraphEdge>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_graph_edges_for_symbols(connection, ids, names, check)
        })?
    }

    /// Selects symbol hits in their existing lexical order before applying the
    /// graph seed limit, so leading documents cannot consume symbol capacity.
    pub(crate) fn symbol_seeds(&self, ordered: &[i64], limit: usize) -> Result<Vec<i64>, Error> {
        i64::try_from(limit).map_err(|_| Error::FtsLimitTooLarge { limit })?;
        self.with_reader(|connection| read_symbol_seeds(connection, ordered, limit, &|| Ok(())))
    }

    /// Selects graph seeds in lexical order under caller-owned admission and row checks.
    pub(crate) fn symbol_seeds_checked(
        &self,
        ordered: &[i64],
        limit: usize,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<i64>, Error> {
        self.checked_read_snapshot(check, |connection| {
            read_symbol_seeds(connection, ordered, limit, check)
        })?
    }

    /// Hydrates a bounded retrieval window on one reader, preserving input order
    /// and omitting occurrences deleted since the candidate lists were produced.
    pub(crate) fn units_by_ids(&self, ids: &[i64]) -> Result<Vec<StoredUnit>, Error> {
        self.read_snapshot(|tx| read_units_by_ids(tx, ids, &|| Ok(())))?
    }

    /// Hydrates occurrences in input order with caller-owned reader admission and
    /// checks between rows; deleted occurrences are omitted without restarting work.
    pub(crate) fn units_by_ids_checked(
        &self,
        ids: &[i64],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<StoredUnit>, Error> {
        self.checked_read_snapshot(check, |tx| read_units_by_ids(tx, ids, check))?
    }

    /// Finds ranked one-hop graph neighbours for lexical seed units.
    pub(crate) fn graph_neighbours(
        &self,
        seed_units: &[i64],
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
    ) -> Result<Vec<GraphUnitHit>, Error> {
        self.with_reader(|connection| {
            read_graph_neighbours(
                connection,
                seed_units,
                limit,
                scope,
                exclude_ids,
                &|| Ok(()),
                false,
            )
            .map(|(hits, _)| hits)
        })
    }

    /// Expands graph seeds with caller-owned reader admission and per-seed/row checks.
    #[cfg(test)]
    pub(crate) fn graph_neighbours_checked(
        &self,
        seed_units: &[i64],
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<Vec<GraphUnitHit>, Error> {
        self.graph_selection_checked(seed_units, limit, scope, exclude_ids, check)
            .map(|(hits, _)| hits)
    }

    /// Retains graph association and rank contributors before reduction and limits
    /// in the same checked snapshot, so unchanged candidate IDs cannot hide changes.
    pub(crate) fn graph_selection_checked(
        &self,
        seed_units: &[i64],
        limit: usize,
        scope: Option<&SearchScope>,
        exclude_ids: &[i64],
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(Vec<GraphUnitHit>, Vec<String>), Error> {
        self.checked_read_snapshot(check, |connection| {
            read_graph_neighbours(
                connection,
                seed_units,
                limit,
                scope,
                exclude_ids,
                check,
                true,
            )
        })?
    }

    /// Persists the latest symbol and file rank values in one writer transaction.
    pub(crate) fn update_graph_ranks(
        &self,
        generation: i64,
        symbols: &[GraphSymbolRank],
        files: &[GraphFileRank],
    ) -> Result<(), Error> {
        self.update_graph_ranks_checked(generation, symbols, files, None)
    }

    /// Publishes ranks under cooperative writer admission and mutation checks.
    /// Cancellation rolls back ranks and graph_pending together; None preserves
    /// ordinary writer wait policy for existing callers.
    pub(crate) fn update_graph_ranks_checked(
        &self,
        generation: i64,
        symbols: &[GraphSymbolRank],
        files: &[GraphFileRank],
        check: Option<&dyn Fn() -> Result<(), Error>>,
    ) -> Result<(), Error> {
        let check_work = || check.map_or(Ok(()), |check| check());
        admission::transaction_checked(self, admission::Kind::Catalogue, check, |transaction| {
            check_work()?;
            l1::check_generation(&transaction, generation)?;
            for symbol in symbols {
                check_work()?;
                transaction
                    .prepare_cached(
                        "UPDATE symbols SET pagerank = ?1 WHERE id = ?2 AND pagerank != ?1",
                    )?
                    .execute(params![symbol.rank, symbol.symbol_id])?;
            }
            for file in files {
                check_work()?;
                transaction
                    .prepare_cached("UPDATE files SET rank = ?1 WHERE id = ?2 AND rank != ?1")?
                    .execute(params![file.rank, file.file_id])?;
            }
            check_work()?;
            transaction.execute("UPDATE files SET rank=0 WHERE rank != 0 AND NOT EXISTS (SELECT 1 FROM symbols WHERE symbols.file_id=files.id)", [])?;
            check_work()?;
            transaction.execute("UPDATE meta SET value='0' WHERE key='graph_pending'", [])?;
            check_work()?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Gets one metadata value without treating absence as an error.
    pub(crate) fn get_meta(&self, key: &str) -> Result<Option<String>, Error> {
        self.with_reader(|connection| {
            connection
                .query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| {
                    row.get(0)
                })
                .optional()
                .map_err(Into::into)
        })
    }

    /// Sets one metadata value, creating the key when absent.
    pub(crate) fn set_meta(&self, key: &str, value: &str) -> Result<(), Error> {
        admission::transaction(self, admission::Kind::Control, |transaction| {
            transaction.execute(
                "INSERT INTO meta (key, value) VALUES (?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Reads the actual connection mmap setting so diagnostics distinguish policy from applied state.
    pub(crate) fn mmap_size(&self) -> Result<i64, Error> {
        self.with_reader(|connection| pragma_i64(connection, "mmap_size"))
    }

    /// Reads SQLite's applied cache setting; negative values denote KiB, not pages.
    pub(crate) fn cache_size(&self) -> Result<i64, Error> {
        self.with_reader(|connection| pragma_i64(connection, "cache_size"))
    }

    /// Measures database pages, free pages, and optional per-object `dbstat` sizes.
    pub(crate) fn footprint(&self) -> Result<Footprint, Error> {
        self.with_reader(|connection| {
            let page_count = pragma_i64(connection, "page_count")?;
            let page_size = pragma_i64(connection, "page_size")?;
            let freelist_count = pragma_i64(connection, "freelist_count")?;
            Ok(Footprint {
                bytes: page_count.saturating_mul(page_size),
                page_count,
                page_size,
                freelist_count,
                tables: dbstat_footprints(connection)?,
            })
        })
    }

    /// Compacts on explicit request only when over a quarter of pages are free;
    /// admission failure preserves the existing index for a later retry.
    pub(crate) fn compact(&self) -> Result<(), Error> {
        admission::compact(self)
    }

    /// Locks the single writer so queue transactions share store serialization.
    pub(crate) fn writer(&self) -> Result<MutexGuard<'_, Connection>, Error> {
        self.writer.lock().map_err(|_| Error::LockPoisoned {
            connection: "writer",
        })
    }

    /// Runs a short read-only snapshot on the reader pool, releasing it before
    /// returning owned results. Callbacks must not write or retain SQLite handles.
    pub(crate) fn read_snapshot<T>(
        &self,
        operation: impl FnOnce(&rusqlite::Transaction<'_>) -> T,
    ) -> Result<T, Error> {
        self.with_reader(|connection| {
            let transaction = connection.unchecked_transaction()?;
            Ok(operation(&transaction))
        })
    }

    fn with_reader<T>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if self.readers.is_empty() {
            let writer = self.writer()?;
            return operation(&writer);
        }
        let index = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        let reader = self.readers[index]
            .lock()
            .map_err(|_| Error::LockPoisoned {
                connection: "reader",
            })?;
        operation(&reader)
    }
}

/// Removes one deleted source and its derived names inside a catalogue transaction.
fn delete_file_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    path: &str,
) -> Result<(), Error> {
    let file_id: Option<i64> = transaction
        .query_row("SELECT id FROM files WHERE path = ?1", [path], |row| {
            row.get(0)
        })
        .optional()?;
    if let Some(file_id) = file_id {
        transaction.execute(
            "DELETE FROM unit_sources WHERE source_kind = 'file' AND source_id = ?1",
            [file_id],
        )?;
        // The units_remove_derived trigger owns lexical deletion during the
        // file cascade. Pre-deleting would make it delete each FTS row twice.
        names::remove_path(transaction, file_id)?;
        names::remove_symbols(transaction, file_id)?;
        transaction.execute("DELETE FROM files WHERE id = ?1", [file_id])?;
    }
    Ok(())
}

/// Updates a catalogue record inside either a single-file or complete-scan transaction.
fn upsert_file_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    file: FileUpsert<'_>,
) -> Result<i64, Error> {
    transaction.execute(
            "INSERT INTO files (path, blake3, size, mtime_ns, lang, kind, git_blob, max_level, treatment, indexed_at, needs_vlm) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11) \
             ON CONFLICT(path) DO UPDATE SET \
             blake3 = excluded.blake3, size = excluded.size, mtime_ns = excluded.mtime_ns, \
             lang = excluded.lang, kind = excluded.kind, git_blob = excluded.git_blob, \
             max_level = excluded.max_level, treatment = excluded.treatment, \
             indexed_at = excluded.indexed_at, needs_vlm = CASE WHEN files.blake3=excluded.blake3 THEN files.needs_vlm ELSE excluded.needs_vlm END",
            params![file.path, file.blake3, file.size, file.mtime_ns, file.lang,
                file.kind.as_str(), file.git_blob, file.max_level, file.treatment,
                file.indexed_at, file.needs_vlm],
        )?;
    let file_id =
        transaction.query_row("SELECT id FROM files WHERE path = ?1", [file.path], |row| {
            row.get(0)
        })?;
    replace_path_name(transaction, file_id, file.path)?;
    Ok(file_id)
}

/// Writes an occurrence and graph rows within the caller's publication transaction.
fn write_indexed_unit(
    transaction: &rusqlite::Transaction<'_>,
    unit: UnitInsert<'_>,
    symbols: &[SymbolInsert<'_>],
    edges: &[UnitEdgeInsert<'_>],
    existing: Option<&std::collections::BTreeSet<i64>>,
) -> Result<IndexedUnit, Error> {
    if edges
        .iter()
        .any(|edge| edge.dst_symbol.is_some() && edge.dst_symbol_index.is_some())
    {
        return Err(Error::GraphDestinationAmbiguous);
    }
    for edge in edges {
        if edge.src_symbol_index >= symbols.len() {
            return Err(Error::GraphSymbolIndexOutOfBounds {
                index: edge.src_symbol_index,
                symbols: symbols.len(),
            });
        }
        if let Some(index) = edge.dst_symbol_index
            && index >= symbols.len()
        {
            return Err(Error::GraphSymbolIndexOutOfBounds {
                index,
                symbols: symbols.len(),
            });
        }
    }

    let writes_graph = !symbols.is_empty() || !edges.is_empty();
    let file_id = match (writes_graph, unit.file_id) {
        (true, Some(file_id)) => file_id,
        (true, None) => return Err(Error::GraphUnitFileRequired),
        (false, file_id) => file_id.unwrap_or_default(),
    };
    let path = if writes_graph {
        let path: Option<String> = transaction
            .prepare_cached("SELECT path FROM files WHERE id = ?1")?
            .query_row([file_id], |row| row.get(0))
            .optional()?;
        path.ok_or(Error::GraphFileMissing { file_id })?
    } else {
        String::new()
    };
    let unit_id = insert_unit_in_transaction(
        transaction,
        UnitInsert {
            ladder_level: 1,
            ..unit
        },
        existing,
    )?;
    if let Some(source_file_id) = unit.file_id {
        insert_file_unit_source(transaction, unit_id, source_file_id)?;
    }
    if existing.is_some_and(|ids| ids.contains(&unit_id)) {
        if let Some(symbols) =
            graph_reuse::retain(transaction, file_id, unit_id, &path, symbols, edges)?
        {
            return Ok(IndexedUnit { unit_id, symbols });
        }
        graph_reuse::remove(transaction, file_id, Some(unit_id))?;
    }
    let mut inserted = Vec::with_capacity(symbols.len());
    for symbol in symbols {
        transaction
            .prepare_cached(
                "INSERT INTO symbols
                 (file_id, unit_id, name, qualified, kind, signature, line_start, line_end)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?
            .execute(params![
                file_id,
                unit_id,
                symbol.name,
                symbol.qualified,
                symbol.kind,
                symbol.signature,
                symbol.line_start,
                symbol.line_end
            ])?;
        let id = transaction.last_insert_rowid();
        names::insert(transaction, symbol.name, FtsNameKind::Symbol, id)?;
        inserted.push(GraphSymbol {
            id,
            file_id,
            path: path.clone(),
            name: symbol.name.to_owned(),
            qualified: symbol.qualified.map(str::to_owned),
            kind: symbol.kind.to_owned(),
            signature: symbol.signature.map(str::to_owned),
            line_start: symbol.line_start,
            line_end: symbol.line_end,
            pagerank: 0.0,
        });
    }
    for edge in edges {
        let dst_symbol = edge
            .dst_symbol_index
            .map(|index| inserted[index].id)
            .or(edge.dst_symbol);
        transaction
            .prepare_cached(
                "INSERT INTO edges (src_symbol, dst_name, dst_symbol, kind, file_id, line)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?
            .execute(params![
                inserted[edge.src_symbol_index].id,
                edge.dst_name,
                dst_symbol,
                edge.kind.as_str(),
                file_id,
                edge.line
            ])?;
    }
    Ok(IndexedUnit {
        unit_id,
        symbols: inserted,
    })
}

/// Loads an occurrence through an existing read snapshot so result projection
/// and parent metadata cannot observe different database generations.
pub(crate) fn unit_by_id_in(
    snapshot: &rusqlite::Transaction<'_>,
    unit_id: i64,
) -> Result<Option<StoredUnit>, Error> {
    load_unit(snapshot, "WHERE u.id = ?1", [unit_id])
}

/// Checks the exact retrieved occurrence while permitting ladder-only advancement.
/// Reuse the caller's snapshot so freshness never borrows identity from a newer row.
pub(crate) fn occurrence_matches_in(
    snapshot: &rusqlite::Transaction<'_>,
    expected: &StoredUnit,
) -> Result<bool, Error> {
    let Some(mut current) = unit_by_id_in(snapshot, expected.id)? else {
        return Ok(false);
    };
    current.ladder_level = expected.ladder_level;
    Ok(&current == expected)
}

fn load_unit<P: rusqlite::Params>(
    connection: &Connection,
    predicate: &str,
    parameters: P,
) -> Result<Option<StoredUnit>, Error> {
    let query = format!(
        "SELECT u.id, u.hash, f.path, u.kind, u.line_start, u.line_end, u.header, c.text, u.ladder_level
         FROM units AS u JOIN unit_contents AS c ON c.hash = u.content_hash LEFT JOIN files AS f ON f.id = u.file_id {predicate}
         ORDER BY u.id LIMIT 1"
    );
    connection
        .query_row(&query, parameters, stored_unit_from_row)
        .optional()
        .map_err(Into::into)
}

fn stored_unit_from_row(row: &rusqlite::Row<'_>) -> Result<StoredUnit, rusqlite::Error> {
    Ok(StoredUnit {
        id: row.get(0)?,
        hash: row.get(1)?,
        path: row.get(2)?,
        kind: row.get(3)?,
        line_start: row.get(4)?,
        line_end: row.get(5)?,
        header: row.get(6)?,
        text: row.get(7)?,
        ladder_level: row.get(8)?,
    })
}

fn count_rows(connection: &Connection, table: &str) -> Result<i64, rusqlite::Error> {
    let query = format!("SELECT COUNT(*) FROM {table}");
    connection.query_row(&query, [], |row| row.get(0))
}

/// Records the actual file provenance of one L1 occurrence.
fn insert_file_unit_source(
    transaction: &rusqlite::Transaction<'_>,
    unit_id: i64,
    file_id: i64,
) -> Result<(), rusqlite::Error> {
    transaction.execute(
        "INSERT OR IGNORE INTO unit_sources (unit_id, source_kind, source_id, role)
         VALUES (?1, 'file', ?2, 'content')",
        params![unit_id, file_id],
    )?;
    Ok(())
}

fn replace_path_name(
    transaction: &rusqlite::Transaction<'_>,
    file_id: i64,
    path: &str,
) -> Result<(), Error> {
    names::insert(transaction, path, FtsNameKind::Path, file_id)
}

fn is_fresh_database(path: &Path, in_memory: bool) -> Result<bool, Error> {
    if in_memory {
        return Ok(true);
    }
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.len() == 0),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(source) => Err(Error::Io { source }),
    }
}

fn configure_connection(
    connection: &Connection,
    limits: StoreLimits,
    fresh: bool,
) -> Result<(), Error> {
    // Keep the bounded L1/graph statement working set resident across files.
    // Cached statements are reset between uses; source and transaction checks
    // still run on every publication.
    connection.set_prepared_statement_cache_capacity(64);

    if fresh {
        connection.pragma_update(None, "page_size", SQLITE_PAGE_SIZE)?;
        capacity::limit_connection_growth(connection, limits.db_max_bytes)?;
        // Deferred native startup leaves a genuinely empty database: configure
        // auto-vacuum before creating tables, without rewriting it via VACUUM.
        connection.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
    }
    let _: String =
        connection.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
    connection.pragma_update(None, "synchronous", "NORMAL")?;
    connection.pragma_update(None, "foreign_keys", true)?;
    connection.pragma_update(None, "busy_timeout", 5_000_i64)?;
    connection.pragma_update(None, "cache_size", -65_536_i64)?;
    connection.pragma_update(None, "mmap_size", SQLITE_MMAP_SIZE)?;
    connection.pragma_update(None, "temp_store", "MEMORY")?;
    let wal_max_bytes =
        i64::try_from(limits.wal_max_bytes).map_err(|_| Error::WalLimitTooLarge {
            wal_max_bytes: limits.wal_max_bytes,
        })?;
    connection.pragma_update(None, "journal_size_limit", wal_max_bytes)?;
    let page_size: u32 = connection.query_row("PRAGMA main.page_size", [], |row| row.get(0))?;
    connection.pragma_update(
        None,
        "wal_autocheckpoint",
        admission::autocheckpoint_frames(limits.wal_max_bytes, page_size),
    )?;
    Ok(())
}

fn migrate(store: &Store) -> Result<(), Error> {
    validate_migration_order(&MIGRATIONS)?;
    let current = schema_version(&*store.writer()?)?;
    if current > LATEST_SCHEMA_VERSION {
        return Err(Error::UnsupportedSchema {
            found: current,
            supported: LATEST_SCHEMA_VERSION,
        });
    }
    for migration in MIGRATIONS
        .iter()
        .filter(|migration| migration.version > current)
    {
        let rebuild = migration.version == 5;
        if rebuild {
            store.writer()?.pragma_update(None, "foreign_keys", false)?;
        }
        let result = admission::transaction(store, admission::Kind::Migration, |transaction| {
            // Another process may have migrated while this opener waited for the
            // writer. Recheck under BEGIN IMMEDIATE before applying any DDL.
            let latest = schema_version(&transaction)?;
            if latest > LATEST_SCHEMA_VERSION {
                return Err(Error::UnsupportedSchema {
                    found: latest,
                    supported: LATEST_SCHEMA_VERSION,
                });
            }
            if latest >= migration.version {
                return Ok(());
            }
            transaction.execute_batch(migration.sql)?;
            if migration.version == 15 {
                names::migrate(&transaction)?;
            }
            if migration.version == 23 {
                vectors::migrate_count_indexes(&transaction)?;
            }
            if rebuild {
                occurrences::migrate_occurrences(&transaction)?;
                let invalid = transaction
                    .prepare("PRAGMA foreign_key_check")?
                    .exists([])?;
                if invalid {
                    return Err(Error::MigrationForeignKeys);
                }
            }
            transaction.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [migration.version.to_string()],
            )?;
            transaction.commit()?;
            Ok(())
        });
        if rebuild {
            store.writer()?.pragma_update(None, "foreign_keys", true)?;
        }
        if matches!(result, Err(Error::WalCapacity { .. })) {
            // A competing opener can commit this migration between our
            // checkpoint and reservation. Its WAL frames can block our
            // reservation, but the required schema may already be published.
            let latest = schema_version(&*store.writer()?)?;
            if latest > LATEST_SCHEMA_VERSION {
                return Err(Error::UnsupportedSchema {
                    found: latest,
                    supported: LATEST_SCHEMA_VERSION,
                });
            }
            if latest >= migration.version {
                continue;
            }
        }
        result?;
    }

    let missing_defaults = {
        let connection = store.writer()?;
        let mut statement = connection.prepare("SELECT 1 FROM meta WHERE key=?1")?;
        let mut missing = false;
        for (key, _) in META_DEFAULTS {
            missing |= !statement.exists([key])?;
        }
        missing
    };
    if missing_defaults {
        admission::transaction::<_, Error>(store, admission::Kind::Control, |transaction| {
            for (key, value) in META_DEFAULTS {
                transaction.execute(
                    "INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO NOTHING",
                    params![key, value],
                )?;
            }
            transaction.commit()?;
            Ok(())
        })?;
    }
    Ok(())
}

fn validate_migration_order(migrations: &[Migration]) -> Result<(), Error> {
    for (expected, migration) in (1_i64..).zip(migrations) {
        if migration.version != expected {
            return Err(Error::MigrationOrder {
                expected,
                found: migration.version,
            });
        }
    }
    Ok(())
}

fn schema_version(connection: &Connection) -> Result<i64, Error> {
    let has_meta: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta')",
        [],
        |row| row.get(0),
    )?;
    if !has_meta {
        return Ok(0);
    }
    let version = connection
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    match version {
        Some(value) => value
            .parse::<i64>()
            .map_err(|_| Error::InvalidSchemaVersion { value }),
        None => Ok(0),
    }
}

fn pragma_i64(connection: &Connection, pragma: &str) -> Result<i64, Error> {
    connection
        .pragma_query_value(None, pragma, |row| row.get(0))
        .map_err(Into::into)
}

fn dbstat_footprints(connection: &Connection) -> Result<Vec<TableFootprint>, Error> {
    let mut statement = match connection
        .prepare("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY name")
    {
        Ok(statement) => statement,
        Err(rusqlite::Error::SqliteFailure(_, Some(message)))
            if message.contains("no such table: dbstat") =>
        {
            return Ok(Vec::new());
        }
        Err(error) => return Err(Error::Sqlite(error)),
    };
    statement
        .query_map([], |row| {
            Ok(TableFootprint {
                name: row.get(0)?,
                bytes: row.get(1)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn read_unit_by_id(
    connection: &Connection,
    unit_id: i64,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Option<StoredUnit>, Error> {
    check()?;

    load_unit(connection, "WHERE u.id = ?1", [unit_id])
}

// Public IDs expose six hash bytes. Comparing binary prefix bounds uses the
// existing occurrence/content indexes instead of formatting every stored hash.
const UNIT_PUBLIC_ID_RANGE_SQL: &str = "SELECT id FROM units
    WHERE (hash >= ?1 AND hash < ?2)
       OR (content_hash >= ?1 AND content_hash < ?2) LIMIT 2";
const UNIT_PUBLIC_ID_LAST_RANGE_SQL: &str = "SELECT id FROM units
    WHERE hash >= ?1 OR content_hash >= ?1 LIMIT 2";
const OPEN_SOURCE_RANGE_SQL: &str = "SELECT f.blake3, f.l1_fingerprint, u.hash,
    u.scope_path, f.lang, f.kind FROM units u JOIN files f ON f.id = u.file_id
    WHERE u.hash >= ?1 AND u.hash < ?2 AND f.l1_pending = 0
      AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint = f.l1_desired";
// Preserve the selective hash-first join even without an exclusive upper bound.
const OPEN_SOURCE_LAST_RANGE_SQL: &str = "SELECT f.blake3, f.l1_fingerprint, u.hash,
    u.scope_path, f.lang, f.kind FROM units u CROSS JOIN files f ON f.id = u.file_id
    WHERE u.hash >= ?1 AND f.l1_pending = 0
      AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint = f.l1_desired";

fn public_id_bounds(public_id: &str) -> Option<([u8; 6], Option<[u8; 6]>)> {
    let prefix = public_id.strip_prefix("u_").unwrap_or(public_id).as_bytes();
    if prefix.len() != 12 {
        return None;
    }
    let nibble = |byte| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    let mut lower = [0; 6];
    for (index, pair) in prefix.as_chunks::<2>().0.iter().enumerate() {
        lower[index] = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    let mut upper = lower;
    for byte in upper.iter_mut().rev() {
        if *byte != u8::MAX {
            *byte += 1;
            return Some((lower, Some(upper)));
        }
        *byte = 0;
    }
    Some((lower, None))
}

fn read_open_source(
    connection: &Connection,
    public_id: &str,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Option<OpenSource>, Error> {
    check()?;
    let Some((lower, upper)) = public_id_bounds(public_id) else {
        return Ok(None);
    };
    let read = |row: &rusqlite::Row<'_>| {
        Ok(OpenSource {
            hash: row.get(0)?,
            fingerprint: row.get(1)?,
            occurrence: row.get(2)?,
            scope_path: row.get(3)?,
            language: row.get(4)?,
            kind: row.get(5)?,
        })
    };
    match upper {
        Some(upper) => connection.query_row(
            OPEN_SOURCE_RANGE_SQL,
            rusqlite::params![lower.as_slice(), upper.as_slice()],
            read,
        ),
        None => connection.query_row(OPEN_SOURCE_LAST_RANGE_SQL, [lower.as_slice()], read),
    }
    .optional()
    .map_err(Into::into)
}

fn read_unit_by_public_id(
    connection: &Connection,
    public_id: &str,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Option<StoredUnit>, Error> {
    check()?;
    let Some((lower, upper)) = public_id_bounds(public_id) else {
        return Ok(None);
    };
    let mut query = connection.prepare(match upper {
        Some(_) => UNIT_PUBLIC_ID_RANGE_SQL,
        None => UNIT_PUBLIC_ID_LAST_RANGE_SQL,
    })?;
    let read = |row: &rusqlite::Row<'_>| row.get::<_, i64>(0);
    let ids = match upper {
        Some(upper) => query
            .query_map(rusqlite::params![lower.as_slice(), upper.as_slice()], read)?
            .collect::<Result<Vec<_>, _>>()?,
        None => query
            .query_map([lower.as_slice()], read)?
            .collect::<Result<Vec<_>, _>>()?,
    };
    match ids.as_slice() {
        [] => Ok(None),
        [id] => load_unit(connection, "WHERE u.id = ?1", [id]),
        _ => Err(Error::AmbiguousUnitId {
            public_id: public_id.to_owned(),
        }),
    }
}

fn read_unit_by_path(
    connection: &Connection,
    path: &str,
    line_start: Option<i64>,
    line_end: Option<i64>,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Option<StoredUnit>, Error> {
    check()?;

    let mut statement = connection.prepare(
                "SELECT u.id, u.hash, f.path, u.kind, u.line_start, u.line_end, u.header, c.text, u.ladder_level
                 FROM units AS u JOIN unit_contents AS c ON c.hash = u.content_hash
                 JOIN files AS f ON f.id = u.file_id
                 WHERE f.path = ?1
                   AND (?2 IS NULL OR u.line_end >= ?2)
                   AND (?3 IS NULL OR u.line_start <= ?3)
                 ORDER BY u.line_start, u.id LIMIT 1",
            )?;
    statement
        .query_row(params![path, line_start, line_end], stored_unit_from_row)
        .optional()
        .map_err(Into::into)
}

fn read_exact_name_units(
    connection: &Connection,
    name: &str,
    limit: usize,
    scope: Option<&SearchScope>,
    exclude_ids: &[i64],
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<i64>, Error> {
    check()?;
    if name.chars().count() < 3 {
        return Ok(Vec::new());
    }
    let sqlite_limit = i64::try_from(limit).map_err(|_| Error::FtsLimitTooLarge { limit })?;
    let query = format!("name:\"{}\"", name.replace('"', "\"\""));
    let projection::Exclusions { prefix, predicate } = projection::exclusion_sql(exclude_ids, 5);
    let mut statement = connection.prepare(&format!(
        "{prefix}SELECT DISTINCT u.id FROM fts_names
                 JOIN symbols s ON s.id = fts_names.ref_id AND fts_names.kind = 'symbol'
                 JOIN units u ON u.id = s.unit_id LEFT JOIN files f ON f.id = u.file_id
                 WHERE fts_names MATCH ?1 AND {SEARCH_SCOPE_SQL} AND s.name = ?4
                   AND {predicate}
                 ORDER BY u.id LIMIT ?2"
    ))?;
    let mut rows = statement.query(params![
        query,
        sqlite_limit,
        scope,
        name,
        serde_json::json!(exclude_ids).to_string()
    ])?;
    let mut ids = Vec::new();
    loop {
        check()?;
        let Some(row) = rows.next()? else { break };
        ids.push(row.get(0)?);
    }
    check()?;
    Ok(ids)
}

fn read_symbol_seeds(
    connection: &Connection,
    ordered: &[i64],
    limit: usize,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<i64>, Error> {
    check()?;
    let limit = i64::try_from(limit).map_err(|_| Error::FtsLimitTooLarge { limit })?;
    let mut statement = connection.prepare("SELECT u.id FROM json_each(?1) AS input JOIN units u ON u.id=input.value WHERE u.kind='symbol' ORDER BY CAST(input.key AS INTEGER) LIMIT ?2")?;
    let mut rows = statement.query(params![serde_json::json!(ordered).to_string(), limit])?;
    let mut ids = Vec::new();
    loop {
        check()?;
        let Some(row) = rows.next()? else { break };
        ids.push(row.get(0)?);
    }
    check()?;
    Ok(ids)
}

fn read_graph_neighbours(
    connection: &Connection,
    seed_units: &[i64],
    limit: usize,
    scope: Option<&SearchScope>,
    exclude_ids: &[i64],
    check: &dyn Fn() -> Result<(), Error>,
    retain: bool,
) -> Result<(Vec<GraphUnitHit>, Vec<String>), Error> {
    check()?;
    let projection::Exclusions { prefix, predicate } = projection::exclusion_sql(exclude_ids, 3);
    let scope_sql = SEARCH_SCOPE_SQL.replace("?3", "?2");
    let mut ranks = std::collections::BTreeMap::<i64, f64>::new();
    let mut associations = Vec::<String>::new();
    // SQLite JSON real formatting can round distinct ranks to the same text.
    // The alternate-form-2 representation retains their distinguishing digits.
    let evidence = if retain {
        "json_array(source.id,source.file_id,source.unit_id,source.name,source.qualified,
            source.kind,source.signature,source.line_start,source.line_end,
            candidate.id,candidate.file_id,candidate.unit_id,candidate.name,candidate.qualified,
            candidate.kind,candidate.signature,candidate.line_start,candidate.line_end,
            printf('%!.26g',candidate.pagerank),edge.rowid,edge.src_symbol,edge.dst_name,
            edge.dst_symbol,edge.kind,edge.file_id,edge.line)"
    } else {
        "NULL"
    };
    let mut statement = connection.prepare(&format!(
        "{prefix}SELECT candidate.unit_id, candidate.pagerank, {evidence}
         FROM symbols AS source
         JOIN edges AS edge ON edge.src_symbol = source.id
         JOIN symbols AS candidate
           ON candidate.id = edge.dst_symbol
           OR (edge.dst_symbol IS NULL AND candidate.name = edge.dst_name)
         JOIN units u ON u.id = candidate.unit_id
         LEFT JOIN files f ON f.id = u.file_id
         WHERE source.unit_id = ?1 AND {scope_sql} AND {predicate}
         UNION ALL
         SELECT candidate.unit_id, candidate.pagerank, {evidence}
         FROM symbols AS source
         JOIN edges AS edge
           ON edge.dst_symbol = source.id
           OR (edge.dst_symbol IS NULL AND edge.dst_name = source.name)
         JOIN symbols AS candidate ON candidate.id = edge.src_symbol
         JOIN units u ON u.id = candidate.unit_id
         LEFT JOIN files f ON f.id = u.file_id
         WHERE source.unit_id = ?1 AND {scope_sql} AND {predicate}",
    ))?;
    let excluded = serde_json::json!(exclude_ids).to_string();
    for seed_unit in seed_units {
        check()?;
        let mut rows = statement.query(params![seed_unit, scope, &excluded])?;
        loop {
            check()?;
            let Some(row) = rows.next()? else { break };
            let hit = GraphUnitHit {
                unit_id: row.get(0)?,
                rank: row.get(1)?,
            };
            if retain {
                associations.push(row.get(2)?);
            }
            ranks
                .entry(hit.unit_id)
                .and_modify(|rank| *rank = rank.max(hit.rank))
                .or_insert(hit.rank);
        }
    }
    let mut hits = Vec::with_capacity(ranks.len());
    for (unit_id, rank) in ranks {
        check()?;
        hits.push(GraphUnitHit { unit_id, rank });
    }
    check()?;
    hits.sort_by(|left, right| {
        right
            .rank
            .total_cmp(&left.rank)
            .then_with(|| left.unit_id.cmp(&right.unit_id))
    });
    check()?;
    hits.truncate(limit);
    associations.sort();
    check()?;
    Ok((hits, associations))
}

fn read_query_fts(
    connection: &Connection,
    query: &str,
    limit: usize,
    scope: Option<&SearchScope>,
    exclude_ids: &[i64],
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<FtsHit>, Error> {
    check()?;
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let sqlite_limit = i64::try_from(limit).map_err(|_| Error::FtsLimitTooLarge { limit })?;
    let projection::Exclusions { prefix, predicate } = projection::exclusion_sql(exclude_ids, 4);
    let scope_sql = projection::scope_sql(SEARCH_SCOPE_SQL);
    let mut statement = connection.prepare(&format!(
        "{prefix}SELECT u.id, bm25(fts_units, 3.0, 4.0, 1.0, 1.0)
         FROM fts_units JOIN units u ON u.id = fts_units.rowid
         LEFT JOIN files f ON f.id = u.file_id
         WHERE fts_units MATCH ?1 AND (?3 IS NULL OR {scope_sql})
           AND {predicate}
         ORDER BY bm25(fts_units, 3.0, 4.0, 1.0, 1.0), u.id LIMIT ?2"
    ))?;
    let mut rows = statement.query(params![
        query,
        sqlite_limit,
        scope,
        serde_json::json!(exclude_ids).to_string()
    ])?;
    let mut hits = Vec::new();
    loop {
        check()?;
        let Some(row) = rows.next()? else { break };
        hits.push(FtsHit {
            unit_id: row.get(0)?,
            score: row.get(1)?,
        });
    }
    check()?;
    Ok(hits)
}

fn read_tool_symbols(
    connection: &Connection,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<ToolSymbol>, Error> {
    read_tool_symbols_query(
        connection,
        "SELECT s.id, s.unit_id, s.file_id, f.path, s.name, s.qualified, s.kind,
                s.signature, s.line_start, s.line_end, s.pagerank
         FROM symbols AS s
         JOIN files AS f ON f.id = s.file_id
         ORDER BY s.name, f.path, s.line_start, s.id",
        [],
        check,
    )
}

fn read_tool_symbols_named(
    connection: &Connection,
    name: &str,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<ToolSymbol>, Error> {
    read_tool_symbols_query(
        connection,
        "SELECT s.id, s.unit_id, s.file_id, f.path, s.name, s.qualified, s.kind,
                s.signature, s.line_start, s.line_end, s.pagerank
         FROM symbols AS s
         JOIN files AS f ON f.id = s.file_id
         WHERE s.name=?1
         ORDER BY s.name, f.path, s.line_start, s.id",
        [name],
        check,
    )
}

fn read_tool_symbols_for_path(
    connection: &Connection,
    path: &str,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<ToolSymbol>, Error> {
    read_tool_symbols_query(
        connection,
        "SELECT s.id, s.unit_id, s.file_id, f.path, s.name, s.qualified, s.kind,
                s.signature, s.line_start, s.line_end, s.pagerank
         FROM files AS f JOIN symbols AS s ON s.file_id = f.id
         WHERE f.path = ?1
         ORDER BY s.name, f.path, s.line_start, s.id",
        [path],
        check,
    )
}

fn read_tool_symbols_by_ids(
    connection: &Connection,
    ids: &[i64],
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<ToolSymbol>, Error> {
    if ids.is_empty() {
        check()?;
        return Ok(Vec::new());
    }
    let encoded = serde_json::json!(ids).to_string();
    read_tool_symbols_query(
        connection,
        "SELECT s.id, s.unit_id, s.file_id, f.path, s.name, s.qualified, s.kind,
                s.signature, s.line_start, s.line_end, s.pagerank
         FROM symbols AS s
         JOIN files AS f ON f.id = s.file_id
         WHERE s.id IN (SELECT value FROM json_each(?1))
         ORDER BY s.name, f.path, s.line_start, s.id",
        [encoded],
        check,
    )
}

fn read_tool_symbols_query<P: rusqlite::Params>(
    connection: &Connection,
    sql: &str,
    parameters: P,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<ToolSymbol>, Error> {
    check()?;
    connection
        .prepare(sql)?
        .query_map(parameters, |row| {
            Ok(ToolSymbol {
                id: row.get(0)?,
                unit_id: row.get(1)?,
                file_id: row.get(2)?,
                path: row.get(3)?,
                name: row.get(4)?,
                qualified: row.get(5)?,
                kind: row.get(6)?,
                signature: row.get(7)?,
                line_start: row.get(8)?,
                line_end: row.get(9)?,
                pagerank: row.get(10)?,
            })
        })?
        .map(|row| {
            check()?;
            Ok(row?)
        })
        .collect::<Result<Vec<_>, Error>>()
}

fn read_graph_edges_for_symbols(
    connection: &Connection,
    ids: &[i64],
    names: &[String],
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<GraphEdge>, Error> {
    check()?;
    if ids.is_empty() && names.is_empty() {
        return Ok(Vec::new());
    }
    let ids = serde_json::json!(ids).to_string();
    let names = serde_json::json!(names).to_string();
    let mut statement = connection.prepare(
        "SELECT src_symbol, dst_name, dst_symbol, kind, file_id, line
         FROM edges
         WHERE src_symbol IN (SELECT value FROM json_each(?1))
            OR dst_symbol IN (SELECT value FROM json_each(?1))
            OR dst_name IN (SELECT value FROM json_each(?2))
         ORDER BY file_id, src_symbol, line, dst_name",
    )?;
    let mut rows = statement.query(params![ids, names])?;
    let mut edges = Vec::new();
    loop {
        check()?;
        let Some(row) = rows.next()? else { break };
        edges.push(GraphEdge {
            src_symbol: row.get(0)?,
            dst_name: row.get(1)?,
            dst_symbol: row.get(2)?,
            kind: GraphEdgeKind::from_db(&row.get::<_, String>(3)?)?,
            file_id: row.get(4)?,
            line: row.get(5)?,
        });
    }
    check()?;
    Ok(edges)
}

fn read_query_fts_names(
    connection: &Connection,
    query: &str,
    limit: usize,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<FtsNameHit>, Error> {
    let sqlite_limit = i64::try_from(limit).map_err(|_| Error::FtsLimitTooLarge { limit })?;
    check()?;
    let mut statement = connection.prepare(
        "SELECT name, kind, ref_id FROM fts_names WHERE fts_names MATCH ?1 \
         ORDER BY rank LIMIT ?2",
    )?;
    statement
        .query_map(params![query, sqlite_limit], |row| {
            let kind = match row.get::<_, String>(1)?.as_str() {
                "symbol" => FtsNameKind::Symbol,
                "path" => FtsNameKind::Path,
                unexpected => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        1,
                        rusqlite::types::Type::Text,
                        format!("invalid fts_names kind {unexpected}").into(),
                    ));
                }
            };
            Ok(FtsNameHit {
                name: row.get(0)?,
                kind,
                ref_id: row.get(2)?,
            })
        })?
        .map(|row| {
            check()?;
            Ok(row?)
        })
        .collect::<Result<Vec<_>, Error>>()
}

fn read_graph_snapshot(
    connection: &Connection,
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<GraphSnapshot, Error> {
    check()?;
    let symbols = connection
        .prepare(
            "SELECT s.id, s.file_id, f.path, s.name, s.qualified, s.kind, s.signature,
                    s.line_start, s.line_end, s.pagerank
             FROM symbols AS s JOIN files AS f ON f.id = s.file_id
             ORDER BY s.id",
        )?
        .query_map([], |row| {
            Ok(GraphSymbol {
                id: row.get(0)?,
                file_id: row.get(1)?,
                path: row.get(2)?,
                name: row.get(3)?,
                qualified: row.get(4)?,
                kind: row.get(5)?,
                signature: row.get(6)?,
                line_start: row.get(7)?,
                line_end: row.get(8)?,
                pagerank: row.get(9)?,
            })
        })?
        .map(|row| {
            check()?;
            Ok(row?)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let raw_edges = connection
        .prepare(
            "SELECT src_symbol, dst_name, dst_symbol, kind, file_id, line
             FROM edges ORDER BY file_id, src_symbol, line, dst_name",
        )?
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Option<i64>>(5)?,
            ))
        })?
        .map(|row| {
            check()?;
            Ok(row?)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let edges = raw_edges
        .into_iter()
        .map(
            |(src_symbol, dst_name, dst_symbol, kind, file_id, line)| -> Result<_, Error> {
                check()?;
                Ok(GraphEdge {
                    src_symbol,
                    dst_name,
                    dst_symbol,
                    kind: GraphEdgeKind::from_db(&kind)?,
                    file_id,
                    line,
                })
            },
        )
        .collect::<Result<Vec<_>, Error>>()?;
    let files = connection
        .prepare("SELECT id, rank FROM files ORDER BY id")?
        .query_map([], |row| {
            Ok(GraphFileRank {
                file_id: row.get(0)?,
                rank: row.get(1)?,
            })
        })?
        .map(|row| {
            check()?;
            Ok(row?)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(GraphSnapshot {
        symbols,
        edges,
        files,
    })
}

// Both hydration paths use the same snapshot/order and missing-occurrence rules.
fn read_units_by_ids(
    tx: &rusqlite::Transaction<'_>,
    ids: &[i64],
    check: &dyn Fn() -> Result<(), Error>,
) -> Result<Vec<StoredUnit>, Error> {
    const HYDRATION_BATCH_SIZE: usize = 1024;
    check()?;
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    // json_each retains duplicate IDs. Bounded batches let cancellation run
    // even when many requested IDs have vanished and therefore return no rows.
    let mut statement = tx.prepare(
        "SELECT u.id, u.hash, f.path, u.kind, u.line_start, u.line_end,
                u.header, c.text, u.ladder_level, CAST(selected.key AS INTEGER)
         FROM json_each(?1) AS selected
         JOIN units AS u ON u.id = selected.value
         JOIN unit_contents AS c ON c.hash = u.content_hash
         LEFT JOIN files AS f ON f.id = u.file_id",
    )?;
    let mut keyed = Vec::new();
    for (batch, chunk) in ids.chunks(HYDRATION_BATCH_SIZE).enumerate() {
        check()?;
        let encoded = serde_json::json!(chunk).to_string();
        let mut rows = statement.query([encoded])?;
        loop {
            check()?;
            let Some(row) = rows.next()? else { break };
            keyed.push(((batch, row.get::<_, i64>(9)?), stored_unit_from_row(row)?));
        }
    }
    check()?;
    // SQLite may choose a different join order. Sort small row headers in Rust
    // rather than making SQLite copy full content text into a sort B-tree.
    keyed.sort_unstable_by_key(|(position, _)| *position);
    Ok(keyed.into_iter().map(|(_, unit)| unit).collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn batched_unit_hydration_keeps_input_order_and_missing_id_behavior()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut connection = rusqlite::Connection::open_in_memory()?;
        connection.execute_batch(
            "CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT NOT NULL);
             CREATE TABLE unit_contents(hash BLOB PRIMARY KEY, text TEXT NOT NULL);
             CREATE TABLE units(id INTEGER PRIMARY KEY, hash BLOB NOT NULL,
                 file_id INTEGER, kind TEXT NOT NULL, line_start INTEGER, line_end INTEGER,
                 header TEXT NOT NULL, content_hash BLOB NOT NULL, ladder_level INTEGER NOT NULL);
             INSERT INTO files(id,path) VALUES(1,'a.rs');
             INSERT INTO unit_contents(hash,text) VALUES(X'01','first'),(X'02','second');
             INSERT INTO units(id,hash,file_id,kind,line_start,line_end,header,content_hash,ladder_level)
                 VALUES(1,X'11',1,'chunk',1,2,'one',X'01',1),
                       (2,X'22',NULL,'history',NULL,NULL,'two',X'02',0);",
        )?;
        let transaction = connection.transaction()?;
        let units = super::read_units_by_ids(&transaction, &[2, 1, 99, 2], &|| Ok(()))?;
        assert_eq!(
            units.iter().map(|unit| unit.id).collect::<Vec<_>>(),
            [2, 1, 2]
        );
        assert_eq!(units[0].path, None);
        assert_eq!(units[1].path.as_deref(), Some("a.rs"));
        assert_eq!(units[1].text, "first");
        assert_eq!(units[2], units[0]);
        let mut across_batches = vec![99; 1023];
        across_batches.extend([2, 1, 2]);
        let units = super::read_units_by_ids(&transaction, &across_batches, &|| Ok(()))?;
        assert_eq!(
            units.iter().map(|unit| unit.id).collect::<Vec<_>>(),
            [2, 1, 2]
        );
        assert!(super::read_units_by_ids(&transaction, &[], &|| Ok(()))?.is_empty());
        Ok(())
    }

    #[test]
    fn schema_ten_preserves_daily_charges_and_conservatively_dates_old_dispatches() {
        let connection = rusqlite::Connection::open_in_memory().unwrap();
        for migration in super::MIGRATIONS.iter().take(9) {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection.execute_batch("INSERT INTO usage(day,provider,calls,tokens,estimated_tokens) VALUES('1970-01-01','p',1,7,7);
            INSERT INTO usage_reservations(id,provider,day,calls,tokens,state,created_at) VALUES
            ('a','p','1970-01-01',1,5,'reserved',0),('b','p','1970-01-01',1,7,'dispatched',0),
            ('c','p','1970-01-01',1,7,'reconciled',0),('d','p','1970-01-01',1,5,'released',0);").unwrap();
        connection.execute_batch(super::MIGRATIONS[9].sql).unwrap();
        let timed: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM usage_reservations WHERE dispatched_at>0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(timed, 2);
        let charges:(i64,i64)=connection.query_row("SELECT (SELECT SUM(tokens) FROM usage),(SELECT SUM(tokens) FROM usage_reservations WHERE state IN ('reserved','dispatched'))",[],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
        assert_eq!(charges, (7, 12));
    }
    use std::{collections::BTreeSet, error::Error as StdError};

    use rusqlite::Connection;

    use crate::{
        config::Config,
        treat::fts::{FtsInput, build_columns, build_match},
    };

    use super::{
        Error, FileKind, FileUpsert, FtsColumns, FtsNameKind, FtsRebuildRow, GraphEdgeInsert,
        GraphEdgeKind, GraphFileRank, GraphSymbolRank, MIGRATIONS, Migration, SQLITE_MMAP_SIZE,
        SQLITE_PAGE_SIZE, StoreLimits, SymbolInsert, UnitEdgeInsert, UnitInsert, UnitKind, open,
        register_vector_extension, validate_migration_order,
    };

    const TABLES: [&str; 18] = [
        "config_effective",
        "commits",
        "dirs",
        "edges",
        "embedding_tables",
        "files",
        "jobs",
        "meta",
        "prs",
        "symbols",
        "unit_sources",
        "unit_contents",
        "units",
        "usage",
        "provider_admission",
        "usage_reservations",
        "fts_units",
        "fts_names",
    ];
    const INDEXES: [&str; 14] = [
        "commits_time",
        "edges_dst_name",
        "edges_dst_symbol",
        "files_l1_pending",
        "files_rank",
        "files_truncated",
        "jobs_claim",
        "symbols_file",
        "symbols_name",
        "symbols_rank",
        "units_file",
        "units_content",
        "units_kind",
        "sqlite_autoindex_meta_1",
    ];

    fn assert_vector_smoke(connection: &Connection) -> Result<(), Box<dyn StdError>> {
        let fts5_enabled: bool = connection.query_row(
            "SELECT sqlite_compileoption_used('ENABLE_FTS5')",
            [],
            |row| row.get(0),
        )?;
        assert!(fts5_enabled, "the bundled SQLite build must enable FTS5");
        let version: String =
            connection.query_row("SELECT vector_version()", [], |row| row.get(0))?;
        assert_eq!(version, "1.1.0");
        connection.execute_batch(
            "SELECT dkc_vector_metadata_init();
             CREATE TABLE embeddings (v BLOB NOT NULL);
             SELECT vector_init('embeddings', 'v', 'dimension=4,type=FLOAT16,distance=COSINE');
             INSERT INTO embeddings (v) VALUES (vector_as_f16('[1.0, 0.0, 0.0, 0.0]'));
             INSERT INTO embeddings (v) VALUES (vector_as_f16('[0.0, 1.0, 0.0, 0.0]'));
             INSERT INTO embeddings (v) VALUES (vector_as_f16('[0.0, 0.0, 1.0, 0.0]'));",
        )?;
        let rows = connection
            .prepare(
                "SELECT rowid FROM vector_full_scan(
                    'embeddings', 'v', vector_as_f16('[0.9, 0.1, 0.0, 0.0]'), 3
                 )",
            )?
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(rows, vec![1, 2, 3]);
        Ok(())
    }

    #[test]
    fn sqlite_vector_smoke_in_memory() -> Result<(), Box<dyn StdError>> {
        register_vector_extension()?;
        let connection = Connection::open_in_memory()?;
        assert_vector_smoke(&connection)
    }

    #[test]
    fn sqlite_vector_smoke_file_backed() -> Result<(), Box<dyn StdError>> {
        register_vector_extension()?;
        let file = tempfile::NamedTempFile::new()?;
        let connection = Connection::open(file.path())?;
        assert_vector_smoke(&connection)
    }

    #[test]
    fn fresh_open_creates_schema_pragmas_and_integrity() -> Result<(), Box<dyn StdError>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("nested/code.db");
        let store = open(&path, StoreLimits::default())?;
        let writer = store.writer()?;
        let names = writer
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<BTreeSet<_>, _>>()?;
        for name in TABLES.into_iter().chain(INDEXES) {
            assert!(names.contains(name), "missing schema object {name}");
        }

        let meta = writer
            .prepare("SELECT key FROM meta ORDER BY key")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<BTreeSet<_>, _>>()?;
        assert_eq!(
            meta,
            BTreeSet::from([
                "chunker_version".to_owned(),
                "dims".to_owned(),
                "graph_pending".to_owned(),
                "indexed_at".to_owned(),
                "indexed_head".to_owned(),
                "l1_scan_generation".to_owned(),
                "l1_completed_generation".to_owned(),
                "profile".to_owned(),
                "repo_root".to_owned(),
                "schema_version".to_owned(),
            ])
        );
        assert_eq!(
            writer.pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))?,
            "wal"
        );
        assert_eq!(
            writer.pragma_query_value(None, "synchronous", |row| row.get::<_, i64>(0))?,
            1
        );
        assert_eq!(
            writer.pragma_query_value(None, "foreign_keys", |row| row.get::<_, i64>(0))?,
            1
        );
        assert_eq!(
            writer.pragma_query_value(None, "busy_timeout", |row| row.get::<_, i64>(0))?,
            5_000
        );
        assert_eq!(
            writer.pragma_query_value(None, "cache_size", |row| row.get::<_, i64>(0))?,
            -65_536
        );
        assert_eq!(
            writer.pragma_query_value(None, "mmap_size", |row| row.get::<_, i64>(0))?,
            SQLITE_MMAP_SIZE
        );
        assert_eq!(
            writer.pragma_query_value(None, "temp_store", |row| row.get::<_, i64>(0))?,
            2
        );
        assert_eq!(
            writer.pragma_query_value(None, "page_size", |row| row.get::<_, i64>(0))?,
            SQLITE_PAGE_SIZE
        );
        assert_eq!(
            writer.pragma_query_value(None, "auto_vacuum", |row| row.get::<_, i64>(0))?,
            2
        );
        assert_eq!(
            writer.query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))?,
            "ok"
        );
        Ok(())
    }

    #[test]
    fn reopen_is_idempotent_and_retains_metadata() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.set_meta("repo_root", "/repo")?;
        drop(store);

        let reopened = open(file.path(), StoreLimits::default())?;
        assert_eq!(
            reopened.get_meta("schema_version")?,
            Some(super::LATEST_SCHEMA_VERSION.to_string())
        );
        assert_eq!(reopened.get_meta("repo_root")?, Some("/repo".to_owned()));
        Ok(())
    }

    #[test]
    fn migration_order_and_future_versions_are_rejected() -> Result<(), Box<dyn StdError>> {
        let migrations = [
            Migration {
                version: 1,
                sql: "",
            },
            Migration {
                version: 3,
                sql: "",
            },
        ];
        assert!(matches!(
            validate_migration_order(&migrations),
            Err(Error::MigrationOrder {
                expected: 2,
                found: 3
            })
        ));

        let file = tempfile::NamedTempFile::new()?;
        let connection = Connection::open(file.path())?;
        connection
            .execute_batch("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")?;
        connection.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?1)",
            [(super::LATEST_SCHEMA_VERSION + 1).to_string()],
        )?;
        drop(connection);
        assert!(matches!(
            open(file.path(), StoreLimits::default()),
            Err(Error::UnsupportedSchema {
                found,
                supported
            }) if found == super::LATEST_SCHEMA_VERSION + 1 && supported == super::LATEST_SCHEMA_VERSION
        ));
        Ok(())
    }

    #[test]
    fn file_schema_migrations_upgrade_a_schema_v1_database() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let connection = Connection::open(file.path())?;
        connection.execute_batch(MIGRATIONS[0].sql)?;
        connection.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', '1')",
            [],
        )?;
        connection.execute(
            "INSERT INTO files (path, blake3, size, mtime_ns, kind, max_level, treatment)
             VALUES ('legacy.rs', ?1, 0, 0, 'code', 2, 'default')",
            [&[9_u8; 32][..]],
        )?;
        connection.execute_batch("INSERT INTO jobs(kind,target_kind,target_id,target_hash,provider,priority,tokens,state,attempts,leased_until,lease_owner,created_at,updated_at)
            VALUES ('embed','unit',1,X'01','embed:local',1,10,'leased',2,999,'old-owner',1,2),
                   ('embed','unit',1,X'02','embed:local',1,10,'queued',0,NULL,NULL,1,2),
                   ('embed','unit',1,X'03','embed:local',1,10,'done',1,NULL,NULL,1,2);
            INSERT INTO usage(day,provider,calls,tokens) VALUES ('2026-09-06','embed:local',3,30);")?;
        drop(connection);

        let store = open(file.path(), StoreLimits::default())?;
        let legacy: (String, i64, Option<String>, String) = store.writer()?.query_row(
            "SELECT state,attempts,lease_owner,last_error FROM jobs WHERE id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        assert_eq!(legacy.0, "dead");
        assert_eq!(legacy.1, 2);
        assert!(legacy.2.is_none());
        assert!(legacy.3.contains("lacks desired fingerprint"));
        let retained: (i64, i64) = store.writer()?.query_row(
            "SELECT COUNT(*),COUNT(DISTINCT desired_hash) FROM jobs",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(retained, (3, 3));
        let states: Vec<String> = store
            .writer()?
            .prepare("SELECT state FROM jobs ORDER BY id")?
            .query_map([], |row| row.get(0))?
            .collect::<Result<_, _>>()?;
        assert_eq!(states, ["dead", "dead", "done"]);
        let usage: (i64, i64, i64) = store.writer()?.query_row(
            "SELECT calls,tokens,estimated_tokens FROM usage",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        assert_eq!(usage, (3, 30, 0));
        assert_eq!(
            store.get_meta("schema_version")?,
            Some(super::LATEST_SCHEMA_VERSION.to_string())
        );
        assert_eq!(store.pending_l1_paths()?, ["legacy.rs"]);
        let file_id = store.index_snapshot()?.files[0].id;
        assert!(!store.complete_file_l1(file_id, &[8; 32])?);
        assert_eq!(store.pending_l1_paths()?, ["legacy.rs"]);
        assert!(store.complete_file_l1(file_id, &[9; 32])?);
        assert!(store.pending_l1_paths()?.is_empty());
        store.rename_file("legacy.rs", "renamed.rs")?;
        assert_eq!(store.pending_l1_paths()?, ["renamed.rs"]);
        assert!(store.complete_file_l1(file_id, &[9; 32])?);
        store.clear_file_content(file_id)?;
        assert_eq!(store.pending_l1_paths()?, ["renamed.rs"]);
        assert!(store.complete_file_l1(file_id, &[9; 32])?);
        store.mark_all_l1_pending()?;
        assert_eq!(store.pending_l1_paths()?, ["renamed.rs"]);
        let writer = store.writer()?;
        let rank_column: i64 = writer.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('files') WHERE name = 'rank'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(rank_column, 1);
        let truncated_column: i64 = writer.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('files') WHERE name = 'truncated'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(truncated_column, 1);
        Ok(())
    }

    #[test]
    fn job_state_counts_read_persisted_failures() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        assert!(store.job_state_counts()?.is_empty());
        store.writer()?.execute_batch("INSERT INTO jobs(kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,created_at,updated_at)
            VALUES ('embed','unit',1,X'00',X'01','fixture',1,1,'dead',0,0),
                   ('embed','unit',2,X'00',X'01','fixture',1,1,'queued',0,0),
                   ('embed','unit',3,X'00',X'01','fixture',1,1,'dead',0,0)")?;
        let counts = store.job_state_counts()?;
        assert_eq!(counts.get("dead"), Some(&2));
        assert_eq!(counts.get("queued"), Some(&1));
        assert_eq!(counts.len(), 2);
        Ok(())
    }

    #[test]
    fn unit_writes_keep_fts_in_sync_and_report_footprint() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let file_id = store.upsert_file(FileUpsert {
            path: "src/retry.rs",
            blake3: &[7; 32],
            size: 42,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let unit_id = store.insert_unit(UnitInsert {
            hash: &[9; 32],
            source_key: None,
            duplicate_ordinal: 0,
            file_id: Some(file_id),
            kind: UnitKind::Symbol,
            byte_start: Some(0),
            byte_end: Some(42),
            line_start: Some(1),
            line_end: Some(2),
            scope_path: None,
            header: "fn should_retry",
            text: "fn should_retry(err: &ChargeError) -> RetryDecision",
            tokens: 12,
            ladder_level: 1,
            parent_unit: None,
            created_at: 1,
            updated_at: 1,
            fts: FtsColumns {
                header: "should_retry",
                idents: "should retry ChargeError RetryDecision",
                body: "fn should_retry err ChargeError RetryDecision",
                prose: "",
            },
        })?;
        assert_eq!(store.query_fts("retry", 10)?.len(), 1);
        let footprint = store.footprint()?;
        assert!(footprint.bytes > 0);
        store.delete_unit(unit_id)?;
        assert!(store.query_fts("retry", 10)?.is_empty());
        Ok(())
    }

    #[test]
    fn source_scope_excludes_history_and_summaries_before_limit() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let file_id = store.upsert_file(FileUpsert {
            path: "src/retry.rs",
            blake3: &[7; 32],
            size: 42,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        for (index, kind) in [
            UnitKind::Commit,
            UnitKind::PullRequest,
            UnitKind::History,
            UnitKind::Summary,
            UnitKind::Module,
            UnitKind::Symbol,
        ]
        .into_iter()
        .enumerate()
        {
            store.insert_unit(UnitInsert {
                hash: &[index as u8; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(file_id),
                kind,
                byte_start: Some(0),
                byte_end: Some(42),
                line_start: Some(1),
                line_end: Some(2),
                scope_path: None,
                header: "fn should_retry",
                text: "fn should_retry(err: &ChargeError) -> RetryDecision",
                tokens: 12,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "should_retry",
                    idents: "should retry ChargeError RetryDecision",
                    body: "fn should_retry err ChargeError RetryDecision",
                    prose: "",
                },
            })?;
        }
        assert_eq!(store.query_fts("retry", 10)?.len(), 6);
        let hits = store.query_fts_scoped("retry", 1, Some(&"source".into()), &[])?;
        assert_eq!(hits.len(), 1);
        assert_eq!(
            store
                .unit_by_id(hits[0].unit_id)?
                .ok_or("missing unit")?
                .kind,
            "symbol"
        );
        assert_eq!(
            store.searchable_source_paths()?,
            BTreeSet::from(["src/retry.rs".to_owned()])
        );
        Ok(())
    }

    #[test]
    fn indexed_duplicates_share_content_but_keep_distinct_locations()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let first_file = store.upsert_file(FileUpsert {
            path: "docs/first.md",
            blake3: &[21; 32],
            size: 16,
            mtime_ns: 1,
            lang: None,
            kind: FileKind::Doc,
            git_blob: None,
            max_level: 1,
            treatment: "default-doc",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let second_file = store.upsert_file(FileUpsert {
            path: "docs/second.md",
            blake3: &[22; 32],
            size: 16,
            mtime_ns: 1,
            lang: None,
            kind: FileKind::Doc,
            git_blob: None,
            max_level: 1,
            treatment: "default-doc",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let first = store.insert_indexed_unit(
            UnitInsert {
                hash: &[23; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(first_file),
                kind: UnitKind::DocSection,
                byte_start: None,
                byte_end: None,
                line_start: Some(1),
                line_end: Some(1),
                scope_path: Some("document"),
                header: "document",
                text: "shared content",
                tokens: 2,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "document",
                    idents: "shared content",
                    body: "shared content",
                    prose: "shared content",
                },
            },
            &[],
            &[],
        )?;
        let second = store.insert_indexed_unit(
            UnitInsert {
                hash: &[23; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(second_file),
                kind: UnitKind::DocSection,
                byte_start: None,
                byte_end: None,
                line_start: Some(7),
                line_end: Some(7),
                scope_path: Some("document"),
                header: "document",
                text: "shared content",
                tokens: 2,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "document",
                    idents: "shared content",
                    body: "shared content",
                    prose: "shared content",
                },
            },
            &[],
            &[],
        )?;
        assert_ne!(first.unit_id, second.unit_id);
        assert_eq!(store.counts()?.units, 2);
        assert_eq!(store.query_fts("shared", 10)?.len(), 2);
        assert!(matches!(
            store.unit_by_public_id("u_171717171717"),
            Err(Error::AmbiguousUnitId { .. })
        ));

        store.clear_file_content(first_file)?;
        assert_eq!(store.counts()?.units, 1);
        assert_eq!(
            store.unit_by_id(second.unit_id)?.and_then(|unit| unit.path),
            Some("docs/second.md".to_owned())
        );

        store.delete_file("docs/second.md")?;
        assert_eq!(store.counts()?.units, 0);
        assert!(store.query_fts("shared", 10)?.is_empty());
        Ok(())
    }

    #[test]
    fn indexed_unit_writes_lexical_and_graph_rows_atomically() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let file_id = store.upsert_file(FileUpsert {
            path: "src/atomic.rs",
            blake3: &[13; 32],
            size: 88,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        store.set_file_truncated(file_id, true)?;
        assert_eq!(
            store.file_truncation(file_id)?.map(|state| state.truncated),
            Some(true)
        );

        let written = store.insert_indexed_unit(
            UnitInsert {
                hash: &[14; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(file_id),
                kind: UnitKind::Symbol,
                byte_start: Some(0),
                byte_end: Some(88),
                line_start: Some(1),
                line_end: Some(4),
                scope_path: Some("Atomic"),
                header: "impl Atomic › fn write",
                text: "fn atomic_write() { helper(); }",
                tokens: 9,
                ladder_level: 3,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "Atomic write",
                    idents: "atomic write helper",
                    body: "fn atomic write helper",
                    prose: "",
                },
            },
            &[
                SymbolInsert {
                    unit_id: Some(999),
                    name: "atomic_write",
                    qualified: Some("Atomic::atomic_write"),
                    kind: "function",
                    signature: Some("fn atomic_write()"),
                    line_start: 1,
                    line_end: 3,
                },
                SymbolInsert {
                    unit_id: None,
                    name: "helper",
                    qualified: Some("Atomic::helper"),
                    kind: "function",
                    signature: Some("fn helper()"),
                    line_start: 4,
                    line_end: 4,
                },
            ],
            &[UnitEdgeInsert {
                src_symbol_index: 0,
                dst_name: "helper",
                dst_symbol: None,
                dst_symbol_index: Some(1),
                kind: GraphEdgeKind::Ref,
                line: Some(2),
            }],
        )?;
        assert_eq!(store.query_fts("atomic", 10)?.len(), 1);
        let snapshot = store.graph_snapshot()?;
        assert_eq!(snapshot.symbols, written.symbols);
        assert_eq!(snapshot.edges.len(), 1);
        assert_eq!(snapshot.edges[0].src_symbol, written.symbols[0].id);
        assert_eq!(snapshot.edges[0].dst_symbol, Some(written.symbols[1].id));
        {
            let writer = store.writer()?;
            let ladder_level: i64 = writer.query_row(
                "SELECT ladder_level FROM units WHERE id = ?1",
                [written.unit_id],
                |row| row.get(0),
            )?;
            let source_unit: i64 = writer.query_row(
                "SELECT unit_id FROM symbols WHERE id = ?1",
                [written.symbols[0].id],
                |row| row.get(0),
            )?;
            assert_eq!(ladder_level, 1);
            assert_eq!(source_unit, written.unit_id);
        }

        store.delete_file("src/atomic.rs")?;
        assert!(store.query_fts("atomic", 10)?.is_empty());
        assert!(store.graph_snapshot()?.symbols.is_empty());
        assert!(store.graph_snapshot()?.edges.is_empty());
        assert!(
            !store
                .query_fts_names("atomic", 10)?
                .iter()
                .any(|hit| hit.kind == FtsNameKind::Symbol)
        );

        let rollback_file = store.upsert_file(FileUpsert {
            path: "src/rollback.rs",
            blake3: &[15; 32],
            size: 10,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let result = store.insert_indexed_unit(
            UnitInsert {
                hash: &[16; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(rollback_file),
                kind: UnitKind::Symbol,
                byte_start: Some(0),
                byte_end: Some(10),
                line_start: Some(1),
                line_end: Some(1),
                scope_path: None,
                header: "fn rollback",
                text: "fn rollback() {}",
                tokens: 4,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "rollback",
                    idents: "rollback",
                    body: "fn rollback",
                    prose: "",
                },
            },
            &[SymbolInsert {
                unit_id: None,
                name: "rollback",
                qualified: None,
                kind: "function",
                signature: None,
                line_start: 1,
                line_end: 1,
            }],
            &[UnitEdgeInsert {
                src_symbol_index: 0,
                dst_name: "missing",
                dst_symbol: Some(-1),
                dst_symbol_index: None,
                kind: GraphEdgeKind::Ref,
                line: Some(1),
            }],
        );
        assert!(matches!(result, Err(Error::Sqlite(_))));
        assert!(store.query_fts("rollback", 10)?.is_empty());
        assert!(store.graph_snapshot()?.symbols.is_empty());
        Ok(())
    }

    #[test]
    fn graph_neighbours_follow_unresolved_symbol_names() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let target_file = store.upsert_file(FileUpsert {
            path: "src/target.rs",
            blake3: &[31; 32],
            size: 30,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 1,
            treatment: "source",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let source_file = store.upsert_file(FileUpsert {
            path: "src/source.rs",
            blake3: &[32; 32],
            size: 30,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 1,
            treatment: "source",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let target = store.insert_indexed_unit(
            UnitInsert {
                hash: &[33; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(target_file),
                kind: UnitKind::Symbol,
                byte_start: Some(0),
                byte_end: Some(30),
                line_start: Some(1),
                line_end: Some(1),
                scope_path: None,
                header: "fn target",
                text: "fn target() {}",
                tokens: 4,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "target",
                    idents: "target",
                    body: "fn target",
                    prose: "",
                },
            },
            &[SymbolInsert {
                unit_id: None,
                name: "target",
                qualified: Some("target"),
                kind: "function",
                signature: Some("fn target()"),
                line_start: 1,
                line_end: 1,
            }],
            &[],
        )?;
        let source = store.insert_indexed_unit(
            UnitInsert {
                hash: &[34; 32],
                source_key: None,
                duplicate_ordinal: 0,
                file_id: Some(source_file),
                kind: UnitKind::Symbol,
                byte_start: Some(0),
                byte_end: Some(30),
                line_start: Some(1),
                line_end: Some(1),
                scope_path: None,
                header: "fn source",
                text: "fn source() { target(); }",
                tokens: 6,
                ladder_level: 1,
                parent_unit: None,
                created_at: 1,
                updated_at: 1,
                fts: FtsColumns {
                    header: "source",
                    idents: "source target",
                    body: "fn source target",
                    prose: "",
                },
            },
            &[SymbolInsert {
                unit_id: None,
                name: "source",
                qualified: Some("source"),
                kind: "function",
                signature: Some("fn source()"),
                line_start: 1,
                line_end: 1,
            }],
            &[UnitEdgeInsert {
                src_symbol_index: 0,
                dst_name: "target",
                dst_symbol: None,
                dst_symbol_index: None,
                kind: GraphEdgeKind::Ref,
                line: Some(1),
            }],
        )?;
        store.update_graph_ranks(
            store.l1_generation()?,
            &[
                GraphSymbolRank {
                    symbol_id: target.symbols[0].id,
                    rank: 0.8,
                },
                GraphSymbolRank {
                    symbol_id: source.symbols[0].id,
                    rank: 0.2,
                },
            ],
            &[
                GraphFileRank {
                    file_id: target_file,
                    rank: 0.8,
                },
                GraphFileRank {
                    file_id: source_file,
                    rank: 0.2,
                },
            ],
        )?;

        let forward = store.graph_neighbours(&[source.unit_id], 10, None, &[])?;
        assert!(forward.iter().any(|hit| hit.unit_id == target.unit_id));
        let reverse = store.graph_neighbours(&[target.unit_id], 10, None, &[])?;
        assert!(reverse.iter().any(|hit| hit.unit_id == source.unit_id));
        Ok(())
    }

    #[test]
    fn lexical_query_forms_names_and_rebuild_share_one_index_contract()
    -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let file_id = store.upsert_file(FileUpsert {
            path: "src/retry_policy.rs",
            blake3: &[8; 32],
            size: 52,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let config = Config::default();
        let text = "pub fn getUserById(user_id: u64) {}";
        let columns = build_columns(
            FtsInput {
                breadcrumb: "deepkit-code › src › retry_policy.rs",
                header: "getUserById",
                text,
                language: Some("rust"),
                kind: FileKind::Code,
                extra_idents: &[],
                has_code_fence: false,
            },
            &config.tokenize,
        );
        let fts = FtsColumns {
            header: &columns.header,
            idents: &columns.idents,
            body: &columns.body,
            prose: &columns.prose,
        };
        let unit_id = store.insert_unit(UnitInsert {
            hash: &[10; 32],
            source_key: None,
            duplicate_ordinal: 0,
            file_id: Some(file_id),
            kind: UnitKind::Symbol,
            byte_start: Some(0),
            byte_end: Some(i64::try_from(text.len())?),
            line_start: Some(1),
            line_end: Some(1),
            scope_path: Some("getUserById"),
            header: "getUserById",
            text,
            tokens: 9,
            ladder_level: 1,
            parent_unit: None,
            created_at: 1,
            updated_at: 1,
            fts,
        })?;
        for query in ["get_user_by_id", "getUserById", "get user by id"] {
            assert_eq!(
                store.query_fts(&build_match(query), 10)?.len(),
                1,
                "{query}"
            );
        }

        let before = store.query_fts(&build_match("getUserById"), 10)?;
        store.rebuild_fts(&[FtsRebuildRow {
            unit_id,
            columns: fts,
        }])?;
        assert_eq!(store.query_fts(&build_match("getUserById"), 10)?, before);

        store.insert_fts_name("RetryPolicy", FtsNameKind::Symbol, 99)?;
        let names = store.query_fts_names("RetryPol", 10)?;
        assert!(names.iter().any(|hit| {
            hit.name == "RetryPolicy" && hit.kind == FtsNameKind::Symbol && hit.ref_id == 99
        }));
        let paths = store.query_fts_names("retry", 10)?;
        assert!(paths.iter().any(|hit| {
            hit.name == "src/retry_policy.rs"
                && hit.kind == FtsNameKind::Path
                && hit.ref_id == file_id
        }));

        store.delete_unit(unit_id)?;
        assert!(store.query_fts(&build_match("getUserById"), 10)?.is_empty());

        store.delete_file("src/retry_policy.rs")?;
        assert!(
            !store
                .query_fts_names("retry", 10)?
                .iter()
                .any(|hit| hit.kind == FtsNameKind::Path)
        );
        Ok(())
    }

    #[test]
    fn graph_rows_names_and_ranks_are_persisted_transactionally() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        let target_file = store.upsert_file(FileUpsert {
            path: "src/auth/target.rs",
            blake3: &[11; 32],
            size: 20,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let caller_file = store.upsert_file(FileUpsert {
            path: "src/caller.rs",
            blake3: &[12; 32],
            size: 20,
            mtime_ns: 1,
            lang: Some("rust"),
            kind: FileKind::Code,
            git_blob: None,
            max_level: 2,
            treatment: "default",
            indexed_at: Some(1),
            needs_vlm: false,
        })?;
        let targets = store.replace_file_symbols(
            target_file,
            &[SymbolInsert {
                unit_id: None,
                name: "Target",
                qualified: Some("Target"),
                kind: "struct",
                signature: Some("pub struct Target;"),
                line_start: 1,
                line_end: 1,
            }],
        )?;
        let callers = store.replace_file_symbols(
            caller_file,
            &[SymbolInsert {
                unit_id: None,
                name: "caller",
                qualified: Some("caller"),
                kind: "function",
                signature: Some("fn caller()"),
                line_start: 1,
                line_end: 3,
            }],
        )?;
        assert!(
            store
                .query_fts_names("Target", 10)?
                .iter()
                .any(|hit| hit.name == "Target" && hit.kind == FtsNameKind::Symbol)
        );

        store.replace_file_edges(
            caller_file,
            &[GraphEdgeInsert {
                src_symbol: callers[0].id,
                dst_name: "Target",
                dst_symbol: Some(targets[0].id),
                kind: GraphEdgeKind::Ref,
                line: Some(2),
            }],
        )?;
        store.update_graph_ranks(
            store.l1_generation()?,
            &[
                GraphSymbolRank {
                    symbol_id: targets[0].id,
                    rank: 0.6,
                },
                GraphSymbolRank {
                    symbol_id: callers[0].id,
                    rank: 0.4,
                },
            ],
            &[
                GraphFileRank {
                    file_id: target_file,
                    rank: 0.6,
                },
                GraphFileRank {
                    file_id: caller_file,
                    rank: 0.4,
                },
            ],
        )?;
        let snapshot = store.graph_snapshot()?;
        assert_eq!(snapshot.symbols.len(), 2);
        assert_eq!(snapshot.edges.len(), 1);
        assert_eq!(snapshot.edges[0].kind, GraphEdgeKind::Ref);
        assert_eq!(snapshot.edges[0].dst_symbol, Some(targets[0].id));
        assert_eq!(
            store.tool_symbols_named_checked("Target", &|| Ok(()))?[0].id,
            targets[0].id
        );
        let catalogue = store.tool_symbols_checked(&|| Ok(()))?;
        for path in ["src/auth/target.rs", "src/caller.rs", "src/missing.rs"] {
            assert_eq!(
                store.tool_symbols_for_path_checked(path, &|| Ok(()))?,
                catalogue
                    .iter()
                    .filter(|symbol| symbol.path == path)
                    .cloned()
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            store.tool_symbols_by_ids_checked(&[callers[0].id], &|| Ok(()))?[0].id,
            callers[0].id
        );
        assert_eq!(
            store.graph_edges_for_symbols_checked(
                &[targets[0].id],
                &["Target".to_owned()],
                &|| Ok(())
            )?,
            snapshot.edges
        );
        assert!(
            store
                .graph_edges_for_symbols_checked(&[], &[], &|| Ok(()))?
                .is_empty()
        );
        assert_eq!(
            snapshot
                .files
                .iter()
                .find(|file| file.file_id == target_file)
                .map(|file| file.rank),
            Some(0.6)
        );

        store.replace_file_symbols(caller_file, &[])?;
        assert!(store.graph_snapshot()?.edges.is_empty());
        assert!(
            !store
                .query_fts_names("caller", 10)?
                .iter()
                .any(|hit| hit.kind == FtsNameKind::Symbol)
        );
        Ok(())
    }

    #[test]
    fn wal_reader_observes_a_consistent_snapshot_during_a_write() -> Result<(), Box<dyn StdError>> {
        let file = tempfile::NamedTempFile::new()?;
        let store = open(file.path(), StoreLimits::default())?;
        store.set_meta("indexed_head", "before")?;
        let writer = Connection::open(file.path())?;
        writer.execute_batch(
            "BEGIN IMMEDIATE; UPDATE meta SET value = 'during' WHERE key = 'indexed_head';",
        )?;
        assert_eq!(store.get_meta("indexed_head")?, Some("before".to_owned()));
        writer.execute_batch("ROLLBACK")?;
        Ok(())
    }
}

#[cfg(test)]
mod graph_write_tests;
