//! Current file and immutable-history embedding and reranker inputs; see `v1.md` §7.1, §8.2 and §11.

pub(crate) use super::module_source::resolve_freshness_batch as resolve_module_freshness_batch;
pub(crate) use super::module_source::{
    SnapshotCache as ModuleSnapshotCache,
    resolve_freshness_batch_with_cache as resolve_module_freshness_batch_with_cache,
};

use super::{
    Error,
    embedding::{Occurrence, Policy},
};
use crate::sync::l1::Work;
use crate::{
    config::{self, Config},
    store::{FileKind, UnitKind},
    sync,
    treat::{
        Router,
        context::{self, ContextFile, ContextUnit, EmbeddingInput},
    },
};
use rusqlite::{OptionalExtension, Transaction};
use std::{fs, path::Path};

/// Current canonical source text and policy, ready for destination-specific preparation.
/// No Debug implementation: source text is not diagnostic output.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct FileInput {
    pub(crate) occurrence: Occurrence,
    pub(crate) context: EmbeddingInput,
    pub(crate) policy: Policy,
}

struct Row {
    hash: Vec<u8>,
    file_id: i64,
    kind: String,
    scope: Option<String>,
    header: String,
    text: String,
    path: String,
    file_hash: Vec<u8>,
    size: i64,
    language: Option<String>,
    file_kind: String,
    l1_fingerprint: Vec<u8>,
}

/// Resolves a completed L1 text occurrence against current configuration and bytes.
/// Run on a blocking worker with the caller's dispatch/completion transaction.
/// `config_root` is the original CLI configuration directory, which can differ
/// from the configured index root. A config change returns None so the host must
/// rebuild preparation; errors never authorize dispatch. Immutable commits use the
/// history resolver; other composites and directory roles are handled separately.
pub(crate) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
) -> Result<Option<FileInput>, Error> {
    resolve_with_work(tx, root, config_root, expected, unit_id, Work::default())
}

/// Resolves canonical source inputs under the caller's original cooperative budget.
/// The caller owns transaction admission; cancellation never returns prepared text.
pub(crate) fn resolve_with_work(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    resolve_for(
        tx,
        root,
        config_root,
        expected,
        unit_id,
        Purpose::Embedding,
        work,
    )
}

/// Verifies L1 source authority for read-only L2 eligibility inspection.
/// Unlike dispatch preparation, this may return protected or L1-only input;
/// callers must inspect policy and must never use it to authorize outbound work.
pub(crate) fn resolve_freshness(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    resolve_for(
        tx,
        root,
        config_root,
        expected,
        unit_id,
        Purpose::Freshness,
        work,
    )
}

/// Proves source and privacy without constructing optional folder roles.
/// Callers must add current roles before comparing an existing vector context.
pub(super) fn resolve_freshness_without_roles(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    resolve_base_for(
        tx,
        root,
        config_root,
        expected,
        unit_id,
        Purpose::Freshness,
        work,
    )
}

/// Maps query/worker cooperative refusal without exposing source diagnostics.
pub(super) fn check_work(work: Work<'_>) -> Result<(), Error> {
    work.check().map_err(|_| Error::WorkerStopped)
}

/// Prepared reranker text with its untruncated source identity for dispatch checks.
/// Private content deliberately has no Debug representation.
pub(crate) struct RerankInput {
    pub(crate) file: FileInput,
    pub(crate) document: String,
}

/// Resolves canonical file context for reranking, including unprotected L1 hits.
/// A completed file or verified immutable commit and exact owned provenance are
/// mandatory. Unknown or supplemental provenance refuses the entire candidate;
/// callers must retain whole-set RRF order when any candidate is refused. Call again
/// at dispatch and compare the complete prepared context, not just its truncation.
pub(crate) fn resolve_rerank(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
) -> Result<Option<RerankInput>, Error> {
    resolve_rerank_with_work(tx, root, config_root, expected, unit_id, Work::default())
}

/// Resolves supplementary context with the caller's cooperative work fence.
pub(crate) fn resolve_rerank_with_work(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    work: Work<'_>,
) -> Result<Option<RerankInput>, Error> {
    resolve_rerank_for(
        tx,
        root,
        config_root,
        expected,
        unit_id,
        Purpose::Rerank,
        work,
    )
}

/// Verifies supplemental chat context without generated directory roles, so a
/// module's role cannot feed back through caller/doc inputs into its own summary.
pub(crate) fn resolve_enrichment_context(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    work: Work<'_>,
) -> Result<Option<RerankInput>, Error> {
    resolve_rerank_for(
        tx,
        root,
        config_root,
        expected,
        unit_id,
        Purpose::Context,
        work,
    )
}

fn resolve_rerank_for(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    purpose: Purpose,
    work: Work<'_>,
) -> Result<Option<RerankInput>, Error> {
    let Some(file) = resolve_for(tx, root, config_root, expected, unit_id, purpose, work)? else {
        return Ok(None);
    };
    let max_chars = usize::try_from(expected.rerank.max_doc_chars)
        .ok()
        .filter(|limit| *limit > 0)
        .ok_or(Error::Invalid("reranker document cap"))?;
    let document = file.context.rerank_document(max_chars);
    Ok(Some(RerankInput { file, document }))
}

/// Resolves an L3 text target with current source bytes and outbound policy.
/// Only symbol/chunk parents and PR/module composites are valid text targets;
/// histories and L0 image/page jobs cannot enter this path.
pub(crate) fn resolve_enrichment(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
) -> Result<Option<FileInput>, Error> {
    resolve_enrichment_with_work(tx, root, config_root, expected, unit_id, Work::default())
}

/// Resolves chat inputs with cooperative source and cancellation checks.
pub(crate) fn resolve_enrichment_with_work(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    if !expected.llm.enabled {
        return Ok(None);
    }
    let allowed: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM units WHERE id=?1 AND kind IN ('symbol','chunk','pr','module'))",
        [unit_id], |row| row.get(0),
    )?;
    if !allowed {
        return Ok(None);
    }
    resolve_for(
        tx,
        root,
        config_root,
        expected,
        unit_id,
        Purpose::Enrichment,
        work,
    )
}

#[derive(Clone, Copy)]
enum Purpose {
    Embedding,
    Enrichment,
    Rerank,
    Freshness,
    Context,
}

fn resolve_for(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    purpose: Purpose,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    let Some(input) = resolve_base_for(tx, root, config_root, expected, unit_id, purpose, work)?
    else {
        return Ok(None);
    };
    if matches!(purpose, Purpose::Enrichment | Purpose::Context) {
        return Ok(Some(input));
    }
    super::roles::apply(tx, root, config_root, expected, input, work).map(Some)
}

fn resolve_base_for(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    expected: &Config,
    unit_id: i64,
    purpose: Purpose,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    check_work(work)?;
    let generated_page: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM units WHERE id=?1 AND kind='doc_section' AND ladder_level=3)",
        [unit_id],
        |r| r.get(0),
    )?;
    if generated_page {
        return super::page_source::resolve_output(tx, root, config_root, expected, unit_id, work);
    }
    let minimum_level = match purpose {
        Purpose::Embedding => 2,
        Purpose::Enrichment => 3,
        Purpose::Rerank | Purpose::Freshness | Purpose::Context => 1,
    };
    let current = config::reload::current(config_root)
        .map_err(|_| Error::Invalid("embedding configuration reload"))?;
    check_work(work)?;
    if &current != expected {
        return Ok(None);
    }
    let configured_root = config_root.join(&current.index.root);
    let canonical_root =
        fs::canonicalize(root).map_err(|_| Error::Invalid("embedding repository root"))?;
    if fs::canonicalize(configured_root).ok().as_ref() != Some(&canonical_root) {
        return Ok(None);
    }
    check_work(work)?;
    let row = tx.query_row(
        "SELECT u.content_hash,u.kind,u.scope_path,u.header,c.text,f.path,f.blake3,f.size,f.lang,f.kind,f.l1_fingerprint,f.id
         FROM units u JOIN unit_contents c ON c.hash=u.content_hash JOIN files f ON f.id=u.file_id
         WHERE u.id=?1 AND u.kind IN ('chunk','symbol','doc_section','config_key')
         AND u.ladder_level>=1 AND f.max_level>=?2 AND f.l1_pending=0
         AND f.l1_fingerprint IS NOT NULL AND f.l1_fingerprint=f.l1_desired
         AND (f.l1_deferred_reason IS NULL OR f.l1_deferred_reason='max_pdf_pages')",
        rusqlite::params![unit_id, minimum_level], |r| Ok(Row { hash:r.get(0)?,kind:r.get(1)?,scope:r.get(2)?,header:r.get(3)?,text:r.get(4)?,
            path:r.get(5)?,file_hash:r.get(6)?,size:r.get(7)?,language:r.get(8)?,file_kind:r.get(9)?,l1_fingerprint:r.get(10)?,file_id:r.get(11)? }),
    ).optional()?;
    check_work(work)?;
    let Some(row) = row else {
        let summary: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM units WHERE id=?1 AND kind='summary')",
            [unit_id],
            |r| r.get(0),
        )?;
        if summary {
            return super::summary_source::resolve(
                tx,
                &canonical_root,
                config_root,
                &current,
                unit_id,
                work,
            );
        }
        let module: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM units WHERE id=?1 AND kind='module')",
            [unit_id],
            |r| r.get(0),
        )?;
        if module {
            return super::module_source::resolve(
                tx,
                &canonical_root,
                config_root,
                &current,
                unit_id,
                minimum_level,
                matches!(purpose, Purpose::Freshness),
                work,
            );
        }
        return super::history_source::resolve(
            tx,
            &canonical_root,
            config_root,
            &current,
            unit_id,
            minimum_level,
            matches!(purpose, Purpose::Freshness),
            work,
        );
    };
    if matches!(
        purpose,
        Purpose::Rerank | Purpose::Enrichment | Purpose::Context
    ) {
        let proven: bool = tx.query_row(
            "SELECT COUNT(*)=1 AND COALESCE(SUM(source_kind='file' AND source_id=?2 AND role='content'),0)=1
             FROM unit_sources WHERE unit_id=?1",
            rusqlite::params![unit_id, row.file_id],
            |r| r.get(0),
        )?;
        if !proven {
            return Ok(None);
        }
    }
    let file_kind =
        FileKind::from_db(&row.file_kind).ok_or(Error::Invalid("embedding file kind"))?;
    let router = Router::new(&current.treatments, current.llm.enabled)
        .map_err(|_| Error::Invalid("embedding treatment policy"))?;
    let treatment = router.route(&row.path, file_kind);
    if (treatment.never_send && !matches!(purpose, Purpose::Freshness))
        || treatment.max_level < minimum_level
        || (matches!(purpose, Purpose::Enrichment)
            && !treatment
                .enrich
                .contains(&crate::treat::Enrichment::DescribeSymbol))
    {
        return Ok(None);
    }
    let file_hash: [u8; 32] = row
        .file_hash
        .as_slice()
        .try_into()
        .map_err(|_| Error::Invalid("embedding file hash"))?;
    let expected_l1 = sync::l1::file_fingerprint(
        &current,
        &sync::l1::repo_name(&canonical_root, &current),
        sync::l1::FingerprintFile {
            path: &row.path,
            content_hash: &file_hash,
            language: row.language.as_deref(),
            kind: file_kind,
            capped: false,
        },
        &treatment,
    );
    if expected_l1.as_slice() != row.l1_fingerprint {
        return Ok(None);
    }
    let max_bytes = current
        .limits
        .max_file_bytes()
        .map_err(|_| Error::Invalid("embedding source cap"))?
        .0;
    if row.size < 0
        || row.size as u64 > max_bytes
        || !source_matches(&canonical_root, &row, max_bytes, work)?
    {
        return Ok(None);
    }
    // Re-read after filesystem work so a configuration edit during hashing cannot
    // leave the earlier policy snapshot authorizing the request.
    if config::reload::current(config_root)
        .map_err(|_| Error::Invalid("embedding configuration reload"))?
        != current
    {
        return Ok(None);
    }
    check_work(work)?;
    let kind = match row.kind.as_str() {
        "chunk" => UnitKind::Chunk,
        "symbol" => UnitKind::Symbol,
        "doc_section" => UnitKind::DocSection,
        "config_key" => UnitKind::ConfigKey,
        _ => return Ok(None),
    };
    let context = context::build_with_options(
        ContextUnit {
            scope_path: row.scope.as_deref(),
            header: &row.header,
            language: row.language.as_deref(),
            kind: kind.as_str(),
            text: &row.text,
            is_document: kind == UnitKind::DocSection,
        },
        ContextFile {
            repo_name: &sync::l1::repo_name(&canonical_root, &current),
            path: &row.path,
        },
        &[],
        context::Options::from(&current.embedder),
    );
    let content_hash = row
        .hash
        .try_into()
        .map_err(|_| Error::Invalid("embedding content hash"))?;
    // L1 policy/config and embedding settings are credential-free; exclude even
    // the API-key environment variable name because changing credentials alone
    // does not change a document vector. Never serialize the full Config (HTTP token).
    let mut embedding = serde_json::to_value(&current.embedder)
        .map_err(|_| Error::Invalid("embedding settings fingerprint"))?;
    if let Some(remote) = embedding.get_mut("remote").and_then(|v| v.as_object_mut()) {
        remote.remove("api_key_env");
    }
    // Routing is independently represented by Prepared's selected destination.
    // A docs edit must not invalidate a vector whose input/space is unchanged.
    // Keep null (the existing single-mode serialization), rather than removing
    // the key, to preserve every previously recorded single-mode generation.
    // Full effective Config checks above and at dispatch/completion stay intact.
    embedding["docs"] = serde_json::Value::Null;
    // This flag schedules replacements; it changes neither model input nor
    // source policy. Preserve the historical default-false generation.
    embedding["reembed_on_role_change"] = serde_json::Value::Bool(false);
    let mut fingerprint = blake3::Hasher::new_derive_key("dkc file embedding policy v1");
    fingerprint.update(&sync::l1::settings_fingerprint(&current));
    fingerprint.update(embedding.to_string().as_bytes());
    check_work(work)?;
    Ok(Some(FileInput {
        occurrence: Occurrence {
            id: unit_id,
            content_hash,
            kind,
        },
        context,
        policy: Policy {
            fingerprint: *fingerprint.finalize().as_bytes(),
            max_level: treatment.max_level,
            never_send: treatment.never_send,
        },
    }))
}

fn source_matches(root: &Path, row: &Row, max_bytes: u64, work: Work<'_>) -> Result<bool, Error> {
    let observed = sync::l1::observe_source(root, &row.path, max_bytes, work, |_| {});
    // Preserve cancellation as a typed refusal even if a concurrent source error
    // occurred; source errors alone simply remove authority to prepare this input.
    check_work(work)?;
    let Ok(observed) = observed else {
        return Ok(false);
    };
    Ok(!observed.content_capped
        && observed.size() == row.size as u64
        && observed.content_hash().as_slice() == row.file_hash)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::store::{self, Store, StoreLimits};
    use rusqlite::params;

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    pub(crate) fn fixture() -> Result<(tempfile::TempDir, Store, Config), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        fs::create_dir(root.path().join(".deepkit"))?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='fixture'\n",
        )?;
        fs::write(root.path().join("a.rs"), "fn example() {}")?;
        let config = config::load(root.path(), None)?.config;
        let store = store::open(":memory:", StoreLimits::default())?;
        let file_hash = *blake3::hash(b"fn example() {}").as_bytes();
        let unit_hash = context::unit_hash("chunk", None, "fn example() {}");
        let treatment =
            Router::new(&config.treatments, config.llm.enabled)?.route("a.rs", FileKind::Code);
        let fingerprint = sync::l1::file_fingerprint(
            &config,
            "fixture",
            sync::l1::FingerprintFile {
                path: "a.rs",
                content_hash: &file_hash,
                language: Some("rust"),
                kind: FileKind::Code,
                capped: false,
            },
            &treatment,
        );
        store.writer()?.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,lang,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint)
            VALUES(1,'a.rs',?1,15,0,'rust','code',3,?2,0,?3,?3)", params![file_hash.as_slice(),treatment.name,fingerprint.as_slice()])?;
        store.writer()?.execute(
            "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'fn example() {}',5)",
            [unit_hash.as_slice()],
        )?;
        store.writer()?.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,created_at,updated_at)
            VALUES(1,?1,?1,'a.rs',1,'chunk','fn example()',0,0)", [unit_hash.as_slice()])?;
        // Keep catalogue size tied to the fixture bytes.
        store.writer()?.execute(
            "UPDATE files SET size=?1",
            [b"fn example() {}".len() as i64],
        )?;
        Ok((root, store, config))
    }
    #[test]
    fn enrichment_target_requires_opt_in_exact_provenance_and_current_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, mut config) = fixture()?;
        config.llm.enabled = true;
        config.treatments = vec![crate::config::Treatment {
            name: "source".to_owned(),
            patterns: vec!["**".to_owned()],
            max_level: 3,
            enrich: vec!["describe_symbol".to_owned()],
            ..Default::default()
        }];
        fs::write(
            root.path().join(".deepkit/config.toml"),
            toml::to_string(&config)?,
        )?;
        let config = crate::config::load(root.path(), None)?.config;
        let hash = *blake3::hash(b"fn example() {}").as_bytes();
        let treatment = Router::new(&config.treatments, true)?.route("a.rs", FileKind::Code);
        let fingerprint = sync::l1::file_fingerprint(
            &config,
            "fixture",
            sync::l1::FingerprintFile {
                path: "a.rs",
                content_hash: &hash,
                language: Some("rust"),
                kind: FileKind::Code,
                capped: false,
            },
            &treatment,
        );
        let mut writer = store.writer()?;
        let tx = writer.transaction()?;
        tx.execute(
            "UPDATE files SET l1_desired=?1,l1_fingerprint=?1,l1_pending=0",
            [fingerprint.as_slice()],
        )?;
        let resolve = || resolve_enrichment(&tx, root.path(), root.path(), &config, 1);
        assert!(resolve()?.is_none(), "unproven provenance must refuse");
        tx.execute("INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(1,'file',1,'content')", [])?;
        assert!(resolve()?.is_some());
        fs::write(root.path().join("a.rs"), "fn changed() {}")?;
        assert!(resolve()?.is_none());
        fs::write(root.path().join("a.rs"), "fn example() {}")?;
        assert!(resolve()?.is_some());
        let mut protected = config.clone();
        protected.treatments[0].never_send = true;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            toml::to_string(&protected)?,
        )?;
        assert!(
            resolve()?.is_none(),
            "a post-preparation policy change must refuse"
        );
        Ok(())
    }

    fn read(store: &Store, root: &Path, config: &Config) -> Result<Option<FileInput>, Error> {
        let mut writer = store.writer()?;
        let tx = writer.transaction()?;
        resolve(&tx, root, root, config, 1)
    }

    #[test]
    fn cooperative_preparation_preserves_identity_and_refuses_stopped_work()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};
        let (root, store, config) = fixture()?;
        let cancelled = AtomicBool::new(false);
        let work = Work {
            deadline: Some(Instant::now() + Duration::from_secs(30)),
            cancelled: Some(&cancelled),
        };
        let mut writer = store.writer()?;
        let tx = writer.transaction()?;
        let changes = tx.total_changes();
        let prepare = |work| {
            super::super::destination::prepare_file_with_work(
                &tx,
                root.path(),
                root.path(),
                &config,
                1,
                work,
            )
        };
        let (ordinary, _) =
            super::super::destination::prepare_file(&tx, root.path(), root.path(), &config, 1)?
                .ok_or("ordinary preparation")?;
        let (checked, _) = prepare(work)?.ok_or("checked preparation")?;
        assert_eq!(
            ordinary.current().desired_hash,
            checked.current().desired_hash
        );
        assert_eq!(
            ordinary.current().target_hash,
            checked.current().target_hash
        );
        assert_eq!(ordinary.context_hash(), checked.context_hash());
        assert_eq!(ordinary.input().text, checked.input().text);
        cancelled.store(true, Ordering::Release);
        assert!(matches!(prepare(work), Err(Error::WorkerStopped)));
        cancelled.store(false, Ordering::Release);
        assert!(matches!(
            prepare(Work {
                deadline: Some(Instant::now()),
                cancelled: None,
            }),
            Err(Error::WorkerStopped)
        ));
        assert!(prepare(work)?.is_some());
        // A cancelled request must refuse before even loading malformed config.
        fs::write(root.path().join(".deepkit/config.toml"), "[broken")?;
        cancelled.store(true, Ordering::Release);
        assert!(matches!(prepare(work), Err(Error::WorkerStopped)));
        assert_eq!(changes, tx.total_changes());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn canonical_preparation_refuses_fifo_and_recovers_without_a_writer() {
        use rustix::fs::Mode;
        use std::{sync::mpsc, time::Duration};
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            let (root, store, config) = fixture().unwrap();
            let path = root.path().join("a.rs");
            fs::remove_file(&path).unwrap();
            mkfifo(&path, Mode::RUSR | Mode::WUSR).unwrap();
            assert!(read(&store, root.path(), &config).unwrap().is_none());
            fs::remove_file(&path).unwrap();
            fs::create_dir(&path).unwrap();
            assert!(read(&store, root.path(), &config).unwrap().is_none());
            fs::remove_dir(&path).unwrap();
            fs::write(&path, "fn example() {}").unwrap();
            assert!(read(&store, root.path(), &config).unwrap().is_some());
            send.send(()).unwrap();
        });
        receive.recv_timeout(Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn current_bytes_and_completed_l1_are_required() -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, config) = fixture()?;
        let input = read(&store, root.path(), &config)?.ok_or("missing current input")?;
        assert_eq!(
            input.context.as_embed_input(input.occurrence.kind).text,
            "fn example() {}"
        );
        assert_eq!(input.context.breadcrumb, "fixture › a.rs");
        assert!(!input.policy.never_send);
        for state in [
            "l1_pending=1",
            "l1_fingerprint=NULL",
            "l1_desired=zeroblob(32)",
            "l1_deferred_reason='capped'",
        ] {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            tx.execute_batch(&format!("UPDATE files SET {state}"))?;
            assert!(resolve(&tx, root.path(), root.path(), &config, 1)?.is_none());
        }
        // An admitted PDF prefix is complete for its configured page budget;
        // a retained snapshot from a lowered cap still fails the desired fence.
        {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            tx.execute_batch("UPDATE files SET l1_deferred_reason='max_pdf_pages'")?;
            assert!(resolve(&tx, root.path(), root.path(), &config, 1)?.is_some());
            tx.execute_batch("UPDATE files SET l1_desired=zeroblob(32)")?;
            assert!(resolve(&tx, root.path(), root.path(), &config, 1)?.is_none());
        }
        fs::write(root.path().join("a.rs"), "fn changed() {}")?;
        assert!(read(&store, root.path(), &config)?.is_none());
        fs::remove_file(root.path().join("a.rs"))?;
        assert!(read(&store, root.path(), &config)?.is_none());
        Ok(())
    }

    #[test]
    fn config_edits_and_old_l1_publication_cannot_authorize_new_inputs()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, config) = fixture()?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='new-name'\n",
        )?;
        assert!(read(&store, root.path(), &config)?.is_none());
        let current = config::load(root.path(), None)?.config;
        // Even a newly constructed worker must reject L1 published under old settings.
        assert!(read(&store, root.path(), &current)?.is_none());
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='fixture'\n[[treatments]]\nname='protected'\nmatch=['*.rs']\nmax_level=3\nnever_send=true\n",
        )?;
        let protected = config::load(root.path(), None)?.config;
        assert!(read(&store, root.path(), &protected)?.is_none());
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='fixture'\n[limits]\nmax_file_bytes='1 B'\n",
        )?;
        let capped = config::load(root.path(), None)?.config;
        assert!(read(&store, root.path(), &capped)?.is_none());
        Ok(())
    }

    #[test]
    fn file_preparation_connects_current_source_destination_and_job_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, config) = fixture()?;
        let prepare = |config: &Config| -> Result<super::super::embedding::Prepared, Box<dyn std::error::Error>> {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            let (job, destination) = super::super::destination::prepare_file(&tx, root.path(), root.path(), config, 1)?.ok_or("missing prepared input")?;
            assert_eq!(job.enqueue(1.0).provider, destination.identity.provider);
            Ok(job)
        };
        let first = prepare(&config)?;
        assert_eq!(first.input().text, "fn example() {}");
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='fixture'\n[embedder.remote]\nmodel='replacement-model'\n",
        )?;
        let current = config::load(root.path(), None)?.config;
        let next = prepare(&current)?;
        assert_ne!(first.current().desired_hash, next.current().desired_hash);
        assert_eq!(first.current().target_hash, next.current().target_hash);
        assert_eq!(first.enqueue(1.0).provider, next.enqueue(1.0).provider);
        assert!(prepare(&config).is_err());
        Ok(())
    }

    fn rerank_read(
        store: &Store,
        root: &Path,
        config: &Config,
    ) -> Result<Option<RerankInput>, Error> {
        let mut writer = store.writer()?;
        let tx = writer.transaction()?;
        resolve_rerank(&tx, root, root, config, 1)
    }

    fn provenance(store: &Store) -> Result<(), Error> {
        store.writer()?.execute(
            "INSERT INTO unit_sources(unit_id,source_kind,source_id,role) VALUES(1,'file',1,'content')", [],
        )?;
        Ok(())
    }

    #[test]
    fn rerank_context_matches_embedding_and_refuses_unknown_provenance()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, config) = fixture()?;
        assert!(rerank_read(&store, root.path(), &config)?.is_none());
        provenance(&store)?;
        let embedding = read(&store, root.path(), &config)?.ok_or("embedding input")?;
        let rerank = rerank_read(&store, root.path(), &config)?.ok_or("rerank input")?;
        let input = embedding.context.as_embed_input(embedding.occurrence.kind);
        assert_eq!(rerank.document, format!("{}\n{}", input.header, input.text));
        assert_eq!(rerank.file.context, embedding.context);
        for mutation in [
            "UPDATE unit_sources SET role='context'",
            "UPDATE unit_sources SET role=NULL",
            "UPDATE unit_sources SET source_id=2",
            "INSERT INTO unit_sources VALUES(1,'unit',2,'role')",
            "INSERT INTO unit_sources VALUES(1,'file',2,'caller')",
            "UPDATE units SET kind='summary'",
        ] {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            tx.execute_batch(mutation)?;
            assert!(
                resolve_rerank(&tx, root.path(), root.path(), &config, 1)?.is_none(),
                "{mutation}"
            );
        }
        Ok(())
    }

    #[test]
    fn rerank_allows_l1_without_granting_embedding_or_protected_access()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, _) = fixture()?;
        provenance(&store)?;
        for never_send in [false, true] {
            fs::write(
                root.path().join(".deepkit/config.toml"),
                format!(
                    "[index]\nrepo_name='fixture'\n[rerank]\nmax_doc_chars=7\n[[treatments]]\nname='l1'\nmatch=['*.rs']\nmax_level=1\nnever_send={never_send}\n"
                ),
            )?;
            let config = config::load(root.path(), None)?.config;
            let treatment =
                Router::new(&config.treatments, config.llm.enabled)?.route("a.rs", FileKind::Code);
            let fingerprint = sync::l1::file_fingerprint(
                &config,
                "fixture",
                sync::l1::FingerprintFile {
                    path: "a.rs",
                    content_hash: blake3::hash(b"fn example() {}").as_bytes(),
                    language: Some("rust"),
                    kind: FileKind::Code,
                    capped: false,
                },
                &treatment,
            );
            store.writer()?.execute(
                "UPDATE files SET max_level=1,treatment='l1',l1_desired=?1,l1_fingerprint=?1",
                [fingerprint.as_slice()],
            )?;
            store
                .writer()?
                .execute("UPDATE files SET l1_pending=0", [])?;
            assert!(read(&store, root.path(), &config)?.is_none());
            let rerank = rerank_read(&store, root.path(), &config)?;
            if never_send {
                assert!(rerank.is_none());
            } else {
                let rerank = rerank.ok_or("L1 rerank input")?;
                assert_eq!(rerank.document, "fixture");
                assert_eq!(rerank.file.context.text, "fn example() {}");
                assert_eq!(rerank.file.policy.max_level, 1);
                assert!(!rerank.file.policy.never_send);
            }
        }
        Ok(())
    }

    #[test]
    fn rerank_revalidation_checks_changes_beyond_the_sent_prefix()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, _) = fixture()?;
        provenance(&store)?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='fixture'\n[rerank]\nmax_doc_chars=7\n",
        )?;
        let config = config::load(root.path(), None)?.config;
        let prepared = rerank_read(&store, root.path(), &config)?.ok_or("prepared prefix")?;
        assert_eq!(prepared.document, "fixture");
        fs::write(root.path().join("a.rs"), "fn example() { }")?;
        assert!(rerank_read(&store, root.path(), &config)?.is_none());
        fs::write(root.path().join("a.rs"), "fn example() {}")?;
        for state in [
            "l1_pending=1",
            "l1_desired=zeroblob(32)",
            "l1_deferred_reason='capped'",
        ] {
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            tx.execute_batch(&format!("UPDATE files SET {state}"))?;
            assert!(resolve_rerank(&tx, root.path(), root.path(), &config, 1)?.is_none());
        }
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='changed'\n",
        )?;
        assert!(rerank_read(&store, root.path(), &config)?.is_none());
        let changed = config::load(root.path(), None)?.config;
        assert!(rerank_read(&store, root.path(), &changed)?.is_none());
        Ok(())
    }

    #[test]
    fn configured_context_changes_model_bytes_and_legacy_generation_without_reindexing_l1()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, original) = fixture()?;
        provenance(&store)?;
        let legacy_context = read(&store, root.path(), &original)?
            .ok_or("original context")?
            .context;
        let l1 = sync::l1::settings_fingerprint(&original);
        for (breadcrumb, kind_line) in [(true, true), (true, false), (false, true), (false, false)]
        {
            fs::write(
                root.path().join(".deepkit/config.toml"),
                format!(
                    "[index]\nrepo_name='fixture'\n[embedder]\nbreadcrumb={breadcrumb}\nkind_line={kind_line}\n"
                ),
            )?;
            let config = config::load(root.path(), None)?.config;
            assert_eq!(sync::l1::settings_fingerprint(&config), l1);
            let source = read(&store, root.path(), &config)?.ok_or("current context")?;
            let rerank = rerank_read(&store, root.path(), &config)?.ok_or("rerank context")?;
            assert_eq!(source.context, rerank.file.context);
            let destination = super::super::destination::resolve(&config, source.occurrence.kind)?;
            // Emulate the previous renderer under the SAME configured policy.
            let legacy = super::super::embedding::Prepared::new(
                source.occurrence,
                legacy_context.clone(),
                destination.identity.clone(),
                source.policy,
            )?
            .ok_or("legacy job")?;
            let fixed = super::super::embedding::Prepared::new(
                source.occurrence,
                source.context,
                destination.identity,
                source.policy,
            )?
            .ok_or("fixed job")?;
            let input = fixed.input();
            assert_eq!(rerank.document, format!("{}\n{}", input.header, input.text));
            assert_eq!(input.header.contains("fixture › a.rs"), breadcrumb);
            assert_eq!(input.header.contains("rust chunk"), kind_line);
            assert_eq!(input.text, "fn example() {}");
            assert_eq!(fixed.current().target_hash, legacy.current().target_hash);
            if breadcrumb && kind_line {
                assert_eq!(fixed.context_hash(), legacy.context_hash());
                assert_eq!(fixed.current().desired_hash, legacy.current().desired_hash);
            } else {
                assert_ne!(fixed.context_hash(), legacy.context_hash());
                assert_ne!(fixed.current().desired_hash, legacy.current().desired_hash);
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn source_paths_reject_symlinks_and_repository_escape() -> Result<(), Box<dyn std::error::Error>>
    {
        use std::os::unix::fs::symlink;
        let (root, store, config) = fixture()?;
        fs::rename(root.path().join("a.rs"), root.path().join("real.rs"))?;
        symlink("real.rs", root.path().join("a.rs"))?;
        assert!(read(&store, root.path(), &config)?.is_none());
        fs::create_dir(root.path().join("real"))?;
        fs::write(root.path().join("real/file.rs"), b"code")?;
        symlink("real", root.path().join("linked"))?;
        assert!(sync::source::open_regular(root.path(), Path::new("linked/file.rs")).is_err());
        assert!(sync::source::open_regular(root.path(), Path::new("../a.rs")).is_err());
        assert!(sync::source::open_regular(root.path(), Path::new("/etc/passwd")).is_err());
        Ok(())
    }
    fn route_prepared(
        store: &Store,
        root: &Path,
        config: &Config,
    ) -> Result<super::super::embedding::Prepared, Box<dyn std::error::Error>> {
        let mut writer = store.writer()?;
        let tx = writer.transaction()?;
        let (prepared, _) = super::super::destination::prepare_file(&tx, root, root, config, 1)?
            .ok_or("prepared source unavailable")?;
        Ok(prepared)
    }

    #[test]
    fn role_refresh_setting_preserves_embedding_identity() -> Result<(), Box<dyn std::error::Error>>
    {
        let (root, store, original) = fixture()?;
        let before = route_prepared(&store, root.path(), &original)?;
        fs::write(
            root.path().join(".deepkit/config.toml"),
            "[index]\nrepo_name='fixture'\n[embedder]\nreembed_on_role_change=true\n",
        )?;
        let changed = config::load(root.path(), None)?.config;
        assert!(changed.embedder.reembed_on_role_change);
        let after = route_prepared(&store, root.path(), &changed)?;
        assert_eq!(before.current().desired_hash, after.current().desired_hash);
        assert_eq!(before.role_free_hash(), after.role_free_hash());
        Ok(())
    }

    #[test]
    fn docs_route_edits_preserve_single_policy_hash_and_primary_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::store::vectors::{self, VectorInsert, VectorTableSpec};
        let (root, store, single) = fixture()?;
        let source = read(&store, root.path(), &single)?.ok_or("source")?;
        // Reproduce the exact pre-change single-mode digest, including docs:null.
        let mut legacy = serde_json::to_value(&single.embedder)?;
        assert_eq!(legacy.get("docs"), Some(&serde_json::Value::Null));
        legacy
            .get_mut("remote")
            .and_then(|v| v.as_object_mut())
            .ok_or("remote")?
            .remove("api_key_env");
        let mut hash = blake3::Hasher::new_derive_key("dkc file embedding policy v1");
        hash.update(&sync::l1::settings_fingerprint(&single));
        hash.update(legacy.to_string().as_bytes());
        assert_eq!(source.policy.fingerprint, *hash.finalize().as_bytes());
        let prepared = route_prepared(&store, root.path(), &single)?;
        let binding =
            super::super::destination::resolve(&single, UnitKind::Chunk)?.binding(&single);
        let slug = binding.destination_slug();
        store.ensure_vector_table(VectorTableSpec {
            slug: &slug,
            model: &binding.model,
            profile: &binding.profile,
            dims: binding.dims as usize,
            created_at: 0,
            active: true,
        })?;
        store.bind_embedding_table(&slug, &binding)?;
        store.insert_vector(VectorInsert {
            table_slug: &slug,
            unit_id: 1,
            ctx_hash: prepared.context_hash(),
            embedding_hash: &prepared.current().target_hash,
            embedding: &vec![1.0; binding.dims as usize],
        })?;
        store.writer()?.execute(
            "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(1,?1,?2)",
            params![slug, prepared.current().desired_hash.as_slice()],
        )?;
        let mut dual = single.clone();
        dual.embedder.docs = Some(crate::config::DocsEmbedder {
            profile: "remote".into(),
            kinds: vec!["doc_section".into()],
        });
        let changes = store.writer()?.total_changes();
        let mut previous = single.clone();
        for current in [dual, single.clone()] {
            fs::write(root.path().join(".deepkit/config.toml"), current.dump()?)?;
            // Equal reusable vector identity never authorizes stale captured config.
            assert!(read(&store, root.path(), &previous)?.is_none());
            let next = route_prepared(&store, root.path(), &current)?;
            assert_eq!(prepared.current().desired_hash, next.current().desired_hash);
            assert_eq!(prepared.current().target_hash, next.current().target_hash);
            assert_eq!(prepared.context_hash(), next.context_hash());
            let mut writer = store.writer()?;
            let tx = writer.transaction()?;
            assert!(vectors::publication_matches_in(
                &tx,
                &slug,
                1,
                &next.current().desired_hash
            )?);
            previous = current;
        }
        assert_eq!(store.writer()?.total_changes(), changes);
        Ok(())
    }

    #[test]
    fn docs_selected_kind_still_changes_destination_generation_in_both_directions()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, store, single) = fixture()?;
        let before = route_prepared(&store, root.path(), &single)?;
        let mut dual = single.clone();
        // Docs is a configured routing role; selecting the fixture's actual kind
        // exercises the same per-kind resolver used for doc_section occurrences.
        dual.embedder.docs = Some(crate::config::DocsEmbedder {
            profile: "remote".into(),
            kinds: vec!["chunk".into()],
        });
        fs::write(root.path().join(".deepkit/config.toml"), dual.dump()?)?;
        assert!(read(&store, root.path(), &single)?.is_none());
        let routed = route_prepared(&store, root.path(), &dual)?;
        assert_ne!(before.current().desired_hash, routed.current().desired_hash);
        assert_ne!(before.enqueue(1.0).provider, routed.enqueue(1.0).provider);
        assert_eq!(before.current().target_hash, routed.current().target_hash);
        assert_eq!(before.context_hash(), routed.context_hash());
        fs::write(root.path().join(".deepkit/config.toml"), single.dump()?)?;
        assert!(read(&store, root.path(), &dual)?.is_none());
        let restored = route_prepared(&store, root.path(), &single)?;
        assert_eq!(
            before.current().desired_hash,
            restored.current().desired_hash
        );
        assert_ne!(
            routed.current().desired_hash,
            restored.current().desired_hash
        );
        Ok(())
    }
}
