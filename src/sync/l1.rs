//! L1 input fingerprints and source validation; see v1.md §4.2–4.3.

mod process;
pub(crate) use process::{FileInput, FileOutcome, PreparedFile, Processor};

use std::{collections::BTreeMap, io::Read, path::Path};

use serde_json::json;

use super::{ChangeSet, Error, GitState, Scan, ScannedFile};
use crate::{
    config::Config,
    store::{
        DirUpsert, FileUpsert, IndexSnapshot, Store,
        l1::{Catalogue, Observation},
    },
    treat::{CHUNKER_VERSION, Router},
};

/// Cooperative admission shared by catalogue preparation, source reads and L1 writes.
/// Individual filesystem/parser/SQLite calls finish before the next check runs.
#[derive(Clone, Copy, Default)]
pub(crate) struct Work<'a> {
    pub(crate) deadline: Option<std::time::Instant>,
    pub(crate) cancelled: Option<&'a std::sync::atomic::AtomicBool>,
}

impl Work<'_> {
    /// Refuses further work once the caller cancels or its original deadline passes.
    pub(crate) fn check(self) -> Result<(), Error> {
        if self
            .cancelled
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
            || self
                .deadline
                .is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
            return Err(Error::WorkStopped);
        }
        Ok(())
    }

    /// Only bounded callers opt into polling; unrestricted indexing retains SQLite busy policy.
    fn admission_check(
        self,
        check: &dyn Fn() -> Result<(), crate::store::Error>,
    ) -> Option<&dyn Fn() -> Result<(), crate::store::Error>> {
        (self.deadline.is_some() || self.cancelled.is_some()).then_some(check)
    }

    fn check_store(self) -> Result<(), crate::store::Error> {
        self.check()
            .map_err(|_| crate::store::Error::L1InputsChanged)
    }
}

/// The published catalogue generation and prepared per-source output identities.
pub(crate) struct Publication {
    pub(crate) generation: i64,
    pub(crate) changes: ChangeSet,
    pub(crate) fingerprints: BTreeMap<String, [u8; 32]>,
}

/// Resolves the repository name used in contextual embedding and lexical inputs.
pub(crate) fn repo_name(root: &Path, settings: &Config) -> String {
    if settings.index.repo_name.is_empty() {
        root.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("repository")
            .to_owned()
    } else {
        settings.index.repo_name.clone()
    }
}

/// Hashes only L1-relevant settings for a pre-publication configuration recheck.
/// Credentials and provider configuration never enter this fingerprint or the database.
pub(crate) fn settings_fingerprint(settings: &Config) -> [u8; 32] {
    *blake3::hash(json!({
        "index": settings.index, "store_path": settings.store.path,
        "max_doc_bytes": settings.limits.max_doc_bytes,
        "max_image_bytes": settings.limits.max_image_bytes,
        "max_pdf_pages": settings.limits.max_pdf_pages,
        "max_files": settings.limits.max_files, "max_file_bytes": settings.limits.max_file_bytes,
        "max_units_per_file": settings.limits.max_units_per_file,
        "chunk": settings.chunk, "tokenize": settings.tokenize,
        "treatments": settings.treatments, "vlm_enabled": settings.llm.enabled,
        "tests": settings.tests,
    }).to_string().as_bytes()).as_bytes()
}

/// Source fields participating in the published L1 configuration/context identity.
pub(crate) struct FingerprintFile<'a> {
    pub(crate) path: &'a str,
    pub(crate) content_hash: &'a [u8; 32],
    pub(crate) language: Option<&'a str>,
    pub(crate) kind: crate::store::FileKind,
    pub(crate) capped: bool,
}

/// Recomputes the same L1 identity for indexing and later embedding source checks.
pub(crate) fn file_fingerprint(
    settings: &Config,
    name: &str,
    file: FingerprintFile<'_>,
    treatment: &crate::treat::ResolvedTreatment,
) -> [u8; 32] {
    let subtokens = settings.tokenize.split_subtokens
        && !file.language.as_ref().is_some_and(|language| {
            settings
                .tokenize
                .split_subtokens_off_for
                .iter()
                .any(|value| value == language)
        });
    let mut input = json!({
        // Range metadata changed, but unit text/partitioning and content identity
        // did not. Refresh positions without invalidating unchanged embeddings.
        "contract": "l1-v5-inclusive-symbol-lines", "chunker_version": CHUNKER_VERSION,
        "grammar_version": crate::treat::GRAMMAR_VERSION,
        "tokenizer": "cl100k_base/tiktoken-rs-0.12", "fts_version": crate::treat::fts::VERSION,
        "content": file.content_hash, "path": file.path, "repo": name,
        "language": file.language, "kind": file.kind.as_str(), "capped": file.capped,
        "treatment": treatment.name, "chunker": format!("{:?}", treatment.chunker),
        "max_level": treatment.max_level, "never_send": treatment.never_send,
        "chunk": settings.chunk, "split_subtokens": subtokens,
        "max_units_per_file": settings.limits.max_units_per_file,
        "tests": settings.tests,
    });
    if treatment.chunker == crate::treat::Chunker::Pdf {
        input["pdf_extractor"] = json!("pdf_oxide-0.3.78/dkc-pdf-v1");
    }
    if treatment.chunker == crate::treat::Chunker::Notebook {
        input["notebook_extractor"] = json!("nbformat4/dkc-notebook-v2-lexical-language");
    }
    if treatment.chunker == crate::treat::Chunker::Office {
        input["office_extractor"] = json!("office_oxide-0.1.11/dkc-office-v1");
    }
    match file.kind {
        crate::store::FileKind::Doc => {
            input["max_doc_bytes"] = json!(settings.limits.max_doc_bytes);
            input["max_pdf_pages"] = json!(settings.limits.max_pdf_pages);
        }
        crate::store::FileKind::Image => {
            input["max_image_bytes"] = json!(settings.limits.max_image_bytes)
        }
        _ => {}
    }
    *blake3::hash(input.to_string().as_bytes()).as_bytes()
}

/// Captured inputs to one fenced catalogue publication.
pub(crate) struct PublishInput<'a> {
    pub(crate) work: Work<'a>,
    pub(crate) scan: &'a Scan,
    pub(crate) git: Option<&'a GitState>,
    pub(crate) snapshot: &'a IndexSnapshot,
    pub(crate) generation: i64,
    pub(crate) root: &'a Path,
    pub(crate) settings: &'a Config,
    pub(crate) full: bool,
}

/// Publishes a complete catalogue generation with per-file effective configuration fingerprints.
pub(crate) fn publish(store: &Store, input: PublishInput<'_>) -> Result<Publication, Error> {
    let PublishInput {
        work,
        scan,
        git,
        snapshot,
        generation,
        root,
        settings,
        full,
    } = input;
    publish_files(
        store,
        FilesInput {
            work,
            files: &scan.files,
            git,
            snapshot,
            generation,
            root,
            settings,
            full,
            discovery: Some((
                scan.status.max_files_reached,
                scan.status.skipped_non_unicode_paths,
            )),
            complete: scan.is_complete(),
            dirs: Some(&scan.dirs),
        },
        || work.check_store(),
    )
}

/// Applies the shared catalogue diff only inside a fully observed watcher scope.
/// Rechecks ignores, sources, HEAD and caller settings inside writer admission;
/// neither unrelated directory hashes nor completed HEAD are changed.
pub(crate) fn publish_paths(
    store: &Store,
    input: PublishInput<'_>,
    paths: &std::collections::BTreeSet<String>,
    validate_settings: impl Fn() -> Result<(), crate::store::Error>,
) -> Result<Publication, Error> {
    let PublishInput {
        work,
        scan,
        git,
        snapshot,
        generation,
        root,
        settings,
        ..
    } = input;
    let observed_head = git.and_then(|git| git.head.clone());
    let limits =
        super::index::limits(settings).map_err(|_| crate::store::Error::L1InputsChanged)?;
    let validate = || {
        work.check_store()?;
        validate_settings()?;
        if super::paths::scan(root, paths, limits, snapshot, work)
            .map_err(|_| crate::store::Error::L1InputsChanged)?
            != *scan
            || super::head::read(root, work).map_err(|_| crate::store::Error::L1InputsChanged)?
                != observed_head
        {
            return Err(crate::store::Error::L1InputsChanged);
        }
        validate_settings()?;
        work.check_store()
    };
    validate()?;
    let scoped = IndexSnapshot {
        files: snapshot
            .files
            .iter()
            .filter(|file| super::discovery::affected(paths, &file.path))
            .cloned()
            .collect(),
        dirs: Vec::new(),
    };
    publish_files(
        store,
        FilesInput {
            work,
            files: &scan.files,
            git,
            snapshot: &scoped,
            generation,
            root,
            settings,
            full: false,
            discovery: None,
            complete: scan.is_complete() && scan.status.skipped_non_unicode_paths == 0,
            dirs: None,
        },
        validate,
    )
}

/// Branch observations have file absence authority only after complete discovery;
/// they carry no Merkle directory authority and never complete indexed HEAD.
pub(crate) struct BranchInput<'a> {
    pub(crate) git: Option<&'a GitState>,
    pub(crate) work: Work<'a>,
    pub(crate) candidates: &'a super::discovery::Candidates,
    pub(crate) files: &'a [ScannedFile],
    pub(crate) snapshot: &'a IndexSnapshot,
    pub(crate) generation: i64,
    pub(crate) root: &'a Path,
    pub(crate) settings: &'a Config,
}

/// Publishes prepared branch files with repeatable source/discovery/HEAD validation
/// under writer admission. The caller must supply a complete catalogue snapshot
/// captured after its generation, and recheck effective settings and resolved root
/// in the validator. Incomplete discovery never authorizes deletion or rename.
pub(crate) fn publish_branch(
    store: &Store,
    input: BranchInput<'_>,
    validate_scope_and_settings: impl Fn() -> Result<(), crate::store::Error>,
) -> Result<Publication, Error> {
    let BranchInput {
        git,
        work,
        candidates,
        files,
        snapshot,
        generation,
        root,
        settings,
    } = input;
    work.check()?;
    // Prepared observations must represent exactly the captured candidates, once
    // each, before completeness may authorize removal of old catalogue paths.
    if files.len() != candidates.paths.len()
        || files
            .iter()
            .zip(&candidates.paths)
            .any(|(file, path)| &file.path != path)
        || candidates.paths.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(crate::store::Error::L1InputsChanged.into());
    }
    let max_files = usize::try_from(settings.limits.max_files).unwrap_or(usize::MAX);
    let max_bytes = settings.limits.max_file_bytes()?.0;
    let validate = || {
        work.check_store()?;
        validate_scope_and_settings()?;
        if super::discovery::discover(root, max_files, work)
            .map_err(|_| crate::store::Error::L1InputsChanged)?
            != *candidates
        {
            return Err(crate::store::Error::L1InputsChanged);
        }
        for file in files {
            work.check_store()?;
            verify_source(root, file, max_bytes, work)
                .map_err(|_| crate::store::Error::L1InputsChanged)?;
        }
        validate_scope_and_settings()?;
        if super::head::read(root, work).map_err(|_| crate::store::Error::L1InputsChanged)?
            != candidates.head
        {
            return Err(crate::store::Error::L1InputsChanged);
        }
        work.check_store()
    };
    validate()?;
    publish_files(
        store,
        FilesInput {
            work,
            files,
            git,
            snapshot,
            generation,
            root,
            settings,
            full: false,
            discovery: Some((
                candidates.max_files_reached,
                candidates.skipped_non_unicode_paths,
            )),
            complete: !candidates.max_files_reached && candidates.skipped_non_unicode_paths == 0,
            dirs: None,
        },
        validate,
    )
}

/// Internal file publication keeps absence authority separate from directory hashes.
struct FilesInput<'a> {
    discovery: Option<(bool, usize)>,
    work: Work<'a>,
    files: &'a [ScannedFile],
    git: Option<&'a GitState>,
    snapshot: &'a IndexSnapshot,
    generation: i64,
    root: &'a Path,
    settings: &'a Config,
    full: bool,
    complete: bool,
    dirs: Option<&'a [super::ScannedDir]>,
}

fn publish_files(
    store: &Store,
    input: FilesInput<'_>,
    validate: impl Fn() -> Result<(), crate::store::Error>,
) -> Result<Publication, Error> {
    let FilesInput {
        discovery,
        work,
        files,
        git,
        snapshot,
        generation,
        root,
        settings,
        full,
        complete,
        dirs,
    } = input;
    work.check()?;
    let router = Router::new(&settings.treatments, settings.llm.enabled)?;
    let name = repo_name(root, settings);
    let changes = super::change_set(snapshot, files, complete);
    let treatments = files
        .iter()
        .map(|file| {
            work.check()?;
            Ok(router.route(&file.path, file.kind))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut fingerprints = BTreeMap::new();
    for (file, treatment) in files.iter().zip(&treatments) {
        work.check()?;
        fingerprints.insert(
            file.path.clone(),
            file_fingerprint(
                settings,
                &name,
                FingerprintFile {
                    path: &file.path,
                    content_hash: &file.blake3,
                    language: file.lang.as_deref(),
                    kind: file.kind,
                    capped: file.content_capped,
                },
                treatment,
            ),
        );
    }
    let observations = files
        .iter()
        .zip(&treatments)
        .map(|(file, treatment)| {
            work.check()?;
            Ok(Observation {
                file: FileUpsert {
                    path: &file.path,
                    blake3: &file.blake3,
                    size: i64::try_from(file.size).map_err(|_| Error::FileSizeTooLarge {
                        path: file.path.clone(),
                        size: file.size,
                    })?,
                    mtime_ns: file.mtime_ns,
                    lang: file.lang.as_deref(),
                    kind: file.kind,
                    git_blob: git.and_then(|git| file.verified_blob(git)),
                    max_level: treatment.max_level,
                    treatment: &treatment.name,
                    indexed_at: Some(super::unix_timestamp()),
                    needs_vlm: false,
                },
                fingerprint: &fingerprints[&file.path],
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let renames = changes
        .renamed
        .iter()
        .map(|r| (r.from.as_str(), r.to.as_str()))
        .collect::<Vec<_>>();
    let reconcile_dirs = complete && dirs.is_some();
    let dirs = dirs
        .unwrap_or_default()
        .iter()
        .map(|d| DirUpsert {
            path: &d.path,
            merkle: &d.merkle,
            file_count: d.file_count,
        })
        .collect::<Vec<_>>();
    work.check()?;
    let check = || work.check_store();
    let generation = store.publish_catalogue_validated(
        work.admission_check(&check),
        Catalogue {
            discovery,
            max_files: settings.limits.max_files,
            generation,
            files: &observations,
            renames: &renames,
            deleted: &changes.deleted,
            dirs: &dirs,
            complete: reconcile_dirs,
            full,
        },
        || work.check_store(),
        validate,
    )?;
    Ok(Publication {
        generation,
        changes,
        fingerprints,
    })
}

/// Inputs for refreshing one already catalogued path. Capture generation before
/// snapshot/observation; concurrent catalogue changes then refuse publication.
pub(crate) struct TargetInput<'a> {
    pub(crate) work: Work<'a>,
    pub(crate) file: &'a ScannedFile,
    pub(crate) snapshot: &'a IndexSnapshot,
    pub(crate) generation: i64,
    pub(crate) root: &'a Path,
    pub(crate) settings: &'a Config,
}

/// Publishes one existing source observation without interpreting absent paths as
/// deletions, rewriting Merkle directories or advancing indexed HEAD. The caller
/// must recheck effective settings and any additional caller scope in the repeatable
/// validator. Current discovery eligibility is independently rechecked here;
/// a prior catalogue entry alone does not establish scope.
pub(crate) fn publish_target(
    store: &Store,
    input: TargetInput<'_>,
    validate_scope_and_settings: impl Fn() -> Result<(), crate::store::Error>,
) -> Result<Publication, Error> {
    let TargetInput {
        work,
        file,
        snapshot,
        generation,
        root,
        settings,
    } = input;
    work.check()?;
    let Some(previous) = snapshot.files.iter().find(|entry| entry.path == file.path) else {
        // New-file admission needs the repository-wide max_files accounting.
        return Err(crate::store::Error::L1InputsChanged.into());
    };
    let max_bytes = settings.limits.max_file_bytes()?.0;
    let router = Router::new(&settings.treatments, settings.llm.enabled)?;
    let treatment = router.route(&file.path, file.kind);
    let fingerprint = file_fingerprint(
        settings,
        &repo_name(root, settings),
        FingerprintFile {
            path: &file.path,
            content_hash: &file.blake3,
            language: file.lang.as_deref(),
            kind: file.kind,
            capped: file.content_capped,
        },
        &treatment,
    );
    let observation = Observation {
        file: FileUpsert {
            path: &file.path,
            blake3: &file.blake3,
            size: i64::try_from(file.size).map_err(|_| Error::FileSizeTooLarge {
                path: file.path.clone(),
                size: file.size,
            })?,
            mtime_ns: file.mtime_ns,
            lang: file.lang.as_deref(),
            kind: file.kind,
            // A lazy filesystem observation proves no new Git blob identity.
            git_blob: None,
            max_level: treatment.max_level,
            treatment: &treatment.name,
            indexed_at: Some(super::unix_timestamp()),
            needs_vlm: false,
        },
        fingerprint: &fingerprint,
    };
    let check = || work.check_store();
    let validate = || {
        work.check_store()?;
        validate_scope_and_settings()?;
        work.check_store()?;
        if !super::target_eligible(root, &file.path, work)
            .map_err(|_| crate::store::Error::L1InputsChanged)?
        {
            return Err(crate::store::Error::L1InputsChanged);
        }
        verify_source(root, file, max_bytes, work).map_err(|_| crate::store::Error::L1InputsChanged)
    };
    validate()?;
    let generation = store.publish_catalogue(
        work.admission_check(&check),
        Catalogue {
            discovery: None,
            max_files: settings.limits.max_files,
            generation,
            files: &[observation],
            renames: &[],
            deleted: &[],
            dirs: &[],
            complete: false,
            full: false,
        },
        validate,
    )?;
    let mut changes = ChangeSet::default();
    if previous.blake3.as_slice() != file.blake3 {
        changes.modified.push(file.path.clone());
    }
    Ok(Publication {
        generation,
        changes,
        fingerprints: BTreeMap::from([(file.path.clone(), fingerprint)]),
    })
}

/// Reads bounded source bytes and verifies they are the content captured by the scan.
pub(crate) fn read_source(
    root: &Path,
    file: &ScannedFile,
    max_bytes: u64,
    work: Work<'_>,
) -> Result<Vec<u8>, Error> {
    work.check()?;
    if file.content_capped {
        return Err(Error::SourceChanged {
            path: file.path.clone(),
        });
    }
    let mut source = Vec::new();
    read_verified(root, file, max_bytes, work, |bytes| {
        source.extend_from_slice(bytes)
    })?;
    Ok(source)
}

/// Rechecks source identity without reopening a pathname after descriptor validation.
/// Capped L0 observations use only descriptor metadata and never read oversized bytes.
pub(crate) fn verify_source(
    root: &Path,
    file: &ScannedFile,
    max_bytes: u64,
    work: Work<'_>,
) -> Result<(), Error> {
    work.check()?;
    if let Some((expected, identity)) = file.source_stamp.get().map(|proof| **proof)
        && identity == file.observation_identity()
    {
        let path = root.join(&file.path);
        let input = super::source::open_regular(root, Path::new(&file.path)).map_err(|source| {
            Error::FileIo {
                path: path.clone(),
                source,
            }
        })?;
        work.check()?;
        let metadata = input
            .metadata()
            .map_err(|source| Error::FileIo { path, source })?;
        work.check()?;
        if super::source::stamp(&metadata) == Some(expected)
            && (metadata.len() > max_bytes) == file.content_capped
        {
            return Ok(());
        }
    }
    read_verified(root, file, max_bytes, work, |_| {})
}

fn read_verified(
    root: &Path,
    file: &ScannedFile,
    max_bytes: u64,
    work: Work<'_>,
    consume: impl FnMut(&[u8]),
) -> Result<(), Error> {
    let observed = observe_source(root, &file.path, max_bytes, work, consume)?;
    if observed.size != file.size
        || observed.content_capped != file.content_capped
        || observed.blake3 != file.blake3
        || observed.kind != file.kind
        || observed.lang != file.lang
    {
        return Err(Error::SourceChanged {
            path: file.path.clone(),
        });
    }
    // A digest reused from the database earns a stamp only after this session
    // has read and verified all bytes. Future checks can reuse that observation.
    if let Some(proof) = observed.source_stamp.get() {
        let _ = file.source_stamp.set(proof.clone());
    }
    Ok(())
}

/// Observes one relative regular source through a bounded, cancellable descriptor read.
/// Oversized files return a metadata-only L0 marker. This does not apply ignore rules,
/// publish a catalogue, infer deletions, or advance HEAD; callers must establish scope
/// and revalidate the observation before publishing. Consumed bytes are provisional
/// until this function succeeds and must not be published from the callback.
pub(crate) fn observe_source(
    root: &Path,
    relative: &str,
    max_bytes: u64,
    work: Work<'_>,
    mut consume: impl FnMut(&[u8]),
) -> Result<ScannedFile, Error> {
    work.check()?;
    let path = root.join(relative);
    let io_error = |source| Error::FileIo {
        path: path.clone(),
        source,
    };
    let changed = || Error::SourceChanged {
        path: relative.to_owned(),
    };
    let mut input = super::source::open_regular(root, Path::new(relative)).map_err(io_error)?;
    work.check()?;
    let metadata = input.metadata().map_err(io_error)?;
    let source_stamp = super::source::stamp(&metadata);
    work.check()?;
    let size = metadata.len();
    let mtime_ns = super::modified_nanos(&metadata, &path)?;
    let (mut kind, language) = super::classify_path(relative);
    let mut language = language.map(str::to_owned);
    if size > max_bytes {
        work.check()?;
        return Ok(ScannedFile {
            path: relative.to_owned(),
            blake3: super::capped_digest(relative, size, mtime_ns),
            size,
            mtime_ns,
            lang: language,
            kind,
            content_capped: true,
            hash_reused_from_git: false,
            source_stamp: std::sync::OnceLock::new(),
        }
        .with_source_stamp(source_stamp));
    }
    let mut hash = blake3::Hasher::new();
    let mut buffer = [0; super::STREAM_BUFFER_BYTES];
    let mut total = 0_u64;
    let mut binary = false;
    loop {
        work.check()?;
        // One extra byte detects a file growing beyond the admitted size/cap.
        let limit = max_bytes
            .saturating_sub(total)
            .saturating_add(1)
            .min(buffer.len() as u64) as usize;
        let count = input.read(&mut buffer[..limit]).map_err(io_error)?;
        work.check()?;
        if count == 0 {
            break;
        }
        let prefix = super::BINARY_PREFIX_BYTES
            .saturating_sub(total)
            .min(count as u64) as usize;
        binary |= buffer[..prefix].contains(&0);
        total = total.saturating_add(count as u64);
        if total > max_bytes {
            return Err(changed());
        }
        hash.update(&buffer[..count]);
        consume(&buffer[..count]);
    }
    work.check()?;
    if binary && !super::is_known_document_or_image(kind) {
        kind = crate::store::FileKind::Other;
        language = None;
    }
    let final_metadata = input.metadata().map_err(io_error)?;
    work.check()?;
    if total != size
        || final_metadata.len() != size
        || super::modified_nanos(&final_metadata, &path)? != mtime_ns
        || super::source::stamp(&final_metadata) != source_stamp
    {
        return Err(changed());
    }
    Ok(ScannedFile {
        path: relative.to_owned(),
        blake3: *hash.finalize().as_bytes(),
        size,
        mtime_ns,
        lang: language,
        kind,
        content_capped: false,
        hash_reused_from_git: false,
        source_stamp: std::sync::OnceLock::new(),
    }
    .with_source_stamp(source_stamp))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::symlink, sync::mpsc, time::Duration};

    #[cfg(unix)]
    use crate::sync::source::mkfifo;

    fn scanned(root: &Path, cap: u64) -> ScannedFile {
        match super::super::scan_file(root, &root.join("source.rs"), cap, None).unwrap() {
            super::super::ScanFile::File(file) => file,
            _ => panic!("fixture must be a source file"),
        }
    }

    fn target_fixture() -> (tempfile::TempDir, Store, Config) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("nested")).unwrap();
        fs::write(root.path().join("source.rs"), "pub fn before() {}\n").unwrap();
        fs::write(root.path().join("nested/other.rs"), "pub fn other() {}\n").unwrap();
        let settings = Config::default();
        let store =
            crate::store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        let scan = super::super::scan(root.path(), Default::default()).unwrap();
        let published = publish(
            &store,
            PublishInput {
                work: Work::default(),
                scan: &scan,
                git: None,
                snapshot: &store.index_snapshot().unwrap(),
                generation: store.l1_generation().unwrap(),
                root: root.path(),
                settings: &settings,
                full: false,
            },
        )
        .unwrap();
        let processor = Processor::new(root.path(), &settings).unwrap();
        for file in &store.index_snapshot().unwrap().files {
            processor
                .process_file(
                    &store,
                    FileInput {
                        work: Work::default(),
                        file: scan.file(&file.path).unwrap(),
                        fence: crate::store::l1::Fence {
                            generation: published.generation,
                            file_id: file.id,
                            path: &file.path,
                            content_hash: &file.blake3,
                            fingerprint: &published.fingerprints[&file.path],
                        },
                    },
                    || Ok(()),
                )
                .unwrap();
        }
        store.set_meta("indexed_head", "unchanged-head").unwrap();
        (root, store, settings)
    }

    fn branch_files(
        root: &Path,
        settings: &Config,
    ) -> (super::super::discovery::Candidates, Vec<ScannedFile>) {
        let candidates = super::super::discovery::discover(
            root,
            settings.limits.max_files as usize,
            Work::default(),
        )
        .unwrap();
        let files = candidates
            .paths
            .iter()
            .map(|path| {
                observe_source(
                    root,
                    path,
                    settings.limits.max_file_bytes().unwrap().0,
                    Work::default(),
                    |_| {},
                )
                .unwrap()
            })
            .collect();
        (candidates, files)
    }

    #[test]
    fn branch_catalogue_preserves_rename_vectors_dirs_and_uncompleted_head() {
        use crate::store::vectors::{VectorInsert, VectorScanPredicate, VectorTableSpec};
        let (root, store, mut settings) = target_fixture();
        settings.limits.max_files = 2;
        let generation = store.l1_generation().unwrap();
        let before = store.index_snapshot().unwrap();
        let old = before
            .files
            .iter()
            .find(|file| file.path == "source.rs")
            .unwrap();
        let unit = store
            .unit_by_path("source.rs", None, None)
            .unwrap()
            .unwrap();
        let keys = store.file_occurrence_keys(old.id).unwrap();
        store
            .ensure_vector_table(VectorTableSpec {
                slug: "branch",
                model: "fixture",
                profile: "local",
                dims: 2,
                created_at: 0,
                active: true,
            })
            .unwrap();
        store
            .insert_vector(VectorInsert {
                table_slug: "branch",
                unit_id: unit.id,
                ctx_hash: &[1; 32],
                embedding_hash: &keys.first().unwrap().0,
                embedding: &[1.0, 0.0],
            })
            .unwrap();
        fs::rename(
            root.path().join("source.rs"),
            root.path().join("renamed.rs"),
        )
        .unwrap();
        fs::remove_file(root.path().join("nested/other.rs")).unwrap();
        fs::write(root.path().join("added.rs"), "pub fn added() {}\n").unwrap();
        let (candidates, files) = branch_files(root.path(), &settings);
        let checks = std::cell::Cell::new(0);
        let publication = publish_branch(
            &store,
            BranchInput {
                git: None,
                work: Work::default(),
                candidates: &candidates,
                files: &files,
                snapshot: &before,
                generation,
                root: root.path(),
                settings: &settings,
            },
            || {
                checks.set(checks.get() + 1);
                Ok(())
            },
        )
        .unwrap();
        // Three validation passes, independent of catalogue row count.
        assert_eq!(checks.get(), 6);
        assert_eq!(publication.changes.added, ["added.rs"]);
        assert_eq!(publication.changes.deleted, ["nested/other.rs"]);
        assert_eq!(publication.changes.renamed.len(), 1);
        let after = store.index_snapshot().unwrap();
        assert_eq!(after.dirs, before.dirs);
        assert_eq!(after.files.len(), 2);
        assert_eq!(
            after
                .files
                .iter()
                .find(|f| f.path == "renamed.rs")
                .unwrap()
                .id,
            old.id
        );
        assert_eq!(store.file_occurrence_keys(old.id).unwrap(), keys);
        let renamed = store.unit_by_id(unit.id).unwrap().unwrap();
        assert_eq!(renamed.path.as_deref(), Some("renamed.rs"));
        assert_ne!(renamed.hash, unit.hash);
        assert_eq!(store.count_vectors().unwrap(), 1);
        assert_eq!(
            store
                .scan_vectors("branch", &[1.0, 0.0], 1, VectorScanPredicate::default())
                .unwrap()[0]
                .unit_id,
            unit.id
        );
        assert_eq!(
            store.pending_l1_paths().unwrap(),
            ["added.rs", "renamed.rs"]
        );
        assert_eq!(
            store.get_meta("indexed_head").unwrap().as_deref(),
            Some("unchanged-head")
        );
        assert_eq!(publication.generation, generation + 1);
    }

    #[test]
    fn branch_catalogue_incomplete_discovery_preserves_absent_rows_and_refuses_growth() {
        for non_unicode in [false, true] {
            let (root, store, mut settings) = target_fixture();
            let generation = store.l1_generation().unwrap();
            let before = store.index_snapshot().unwrap();
            fs::remove_file(root.path().join("nested/other.rs")).unwrap();
            if non_unicode {
                use std::os::unix::ffi::OsStringExt;
                let non_unicode = root.path().join(std::ffi::OsString::from_vec(vec![255]));
                match fs::write(&non_unicode, "skipped") {
                    Ok(()) => {}
                    // APFS rejects non-UTF-8 names with EILSEQ, so this
                    // filesystem cannot exercise the walker’s skip counter.
                    Err(error) if cfg!(target_os = "macos") && error.raw_os_error() == Some(92) => {
                        continue;
                    }
                    Err(error) => panic!("creating non-Unicode fixture: {error}"),
                }
            } else {
                settings.limits.max_files = 1;
                fs::write(root.path().join("z-new.rs"), "new").unwrap();
            }
            let (candidates, files) = branch_files(root.path(), &settings);
            let result = publish_branch(
                &store,
                BranchInput {
                    git: None,
                    work: Work::default(),
                    candidates: &candidates,
                    files: &files,
                    snapshot: &before,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || Ok(()),
            );
            if files
                .iter()
                .any(|file| !before.files.iter().any(|old| old.path == file.path))
                && !non_unicode
            {
                assert!(matches!(
                    result,
                    Err(Error::Store(crate::store::Error::FileCapacity { .. }))
                ));
                assert_eq!(store.l1_generation().unwrap(), generation);
            } else {
                let publication = result.unwrap();
                assert!(publication.changes.deleted.is_empty());
                assert!(publication.changes.renamed.is_empty());
            }
            let after = store.index_snapshot().unwrap();
            assert_eq!(after.dirs, before.dirs);
            assert!(
                after
                    .files
                    .iter()
                    .any(|file| file.path == "nested/other.rs")
            );
            assert_eq!(
                store.get_meta("indexed_head").unwrap().as_deref(),
                Some("unchanged-head")
            );
        }
        // Force the new-file capacity path independent of filesystem walk order.
        let (root, store, mut settings) = target_fixture();
        settings.limits.max_files = 1;
        let generation = store.l1_generation().unwrap();
        let before = store.index_snapshot().unwrap();
        fs::remove_file(root.path().join("source.rs")).unwrap();
        fs::remove_file(root.path().join("nested/other.rs")).unwrap();
        fs::write(root.path().join("a.rs"), "a").unwrap();
        fs::write(root.path().join("b.rs"), "b").unwrap();
        let (candidates, files) = branch_files(root.path(), &settings);
        assert!(candidates.max_files_reached);
        assert!(matches!(
            publish_branch(
                &store,
                BranchInput {
                    git: None,
                    work: Work::default(),
                    candidates: &candidates,
                    files: &files,
                    snapshot: &before,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || Ok(())
            ),
            Err(Error::Store(crate::store::Error::FileCapacity { limit: 1 }))
        ));
        assert_eq!(store.index_snapshot().unwrap().files, before.files);
        assert_eq!(store.l1_generation().unwrap(), generation);
    }

    #[test]
    fn branch_catalogue_late_source_scope_and_work_refusals_roll_back_and_retry() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for at in [1, 3, 5] {
            for mutation in ["source", "ignore", "new", "settings", "cancel"] {
                let (root, store, settings) = target_fixture();
                let generation = store.l1_generation().unwrap();
                let before = store.index_snapshot().unwrap();
                fs::rename(
                    root.path().join("source.rs"),
                    root.path().join("renamed.rs"),
                )
                .unwrap();
                fs::remove_file(root.path().join("nested/other.rs")).unwrap();
                let (candidates, files) = branch_files(root.path(), &settings);
                let cancelled = AtomicBool::new(false);
                let checks = std::cell::Cell::new(0);
                let result = publish_branch(
                    &store,
                    BranchInput {
                        git: None,
                        work: Work {
                            cancelled: Some(&cancelled),
                            deadline: None,
                        },
                        candidates: &candidates,
                        files: &files,
                        snapshot: &before,
                        generation,
                        root: root.path(),
                        settings: &settings,
                    },
                    || {
                        checks.set(checks.get() + 1);
                        if checks.get() == at {
                            match mutation {
                                "source" => {
                                    fs::write(root.path().join("renamed.rs"), "changed").unwrap()
                                }
                                "ignore" => {
                                    fs::write(root.path().join(".deepkitignore"), "renamed.rs\n")
                                        .unwrap()
                                }
                                "new" => fs::write(root.path().join("new.rs"), "new").unwrap(),
                                "settings" => return Err(crate::store::Error::L1InputsChanged),
                                "cancel" => cancelled.store(true, Ordering::Release),
                                _ => unreachable!(),
                            }
                        }
                        Ok(())
                    },
                );
                assert!(result.is_err(), "{mutation} at {at}");
                let after = store.index_snapshot().unwrap();
                assert_eq!(after.files, before.files, "{mutation} at {at}");
                assert_eq!(after.dirs, before.dirs);
                assert_eq!(store.l1_generation().unwrap(), generation);
                assert!(store.pending_l1_paths().unwrap().is_empty());
                assert_eq!(
                    store.get_meta("indexed_head").unwrap().as_deref(),
                    Some("unchanged-head")
                );
                if mutation == "ignore" {
                    fs::remove_file(root.path().join(".deepkitignore")).unwrap();
                }
                let (candidates, files) = branch_files(root.path(), &settings);
                assert!(
                    publish_branch(
                        &store,
                        BranchInput {
                            git: None,
                            work: Work::default(),
                            candidates: &candidates,
                            files: &files,
                            snapshot: &before,
                            generation,
                            root: root.path(),
                            settings: &settings,
                        },
                        || Ok(())
                    )
                    .is_ok()
                );
            }
        }
    }

    #[test]
    fn branch_catalogue_rejects_incomplete_observations_and_superseded_generation() {
        let (root, store, settings) = target_fixture();
        let generation = store.l1_generation().unwrap();
        let before = store.index_snapshot().unwrap();
        let (candidates, files) = branch_files(root.path(), &settings);
        for count in 0..files.len() {
            assert!(
                publish_branch(
                    &store,
                    BranchInput {
                        git: None,
                        work: Work::default(),
                        candidates: &candidates,
                        files: &files[..count],
                        snapshot: &before,
                        generation,
                        root: root.path(),
                        settings: &settings,
                    },
                    || Ok(())
                )
                .is_err()
            );
        }
        let mut reversed = files.clone();
        reversed.reverse();
        assert!(
            publish_branch(
                &store,
                BranchInput {
                    git: None,
                    work: Work::default(),
                    candidates: &candidates,
                    files: &reversed,
                    snapshot: &before,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || Ok(())
            )
            .is_err()
        );
        assert_eq!(store.l1_generation().unwrap(), generation);
        store
            .set_meta("l1_scan_generation", &(generation + 1).to_string())
            .unwrap();
        assert!(matches!(
            publish_branch(
                &store,
                BranchInput {
                    git: None,
                    work: Work::default(),
                    candidates: &candidates,
                    files: &files,
                    snapshot: &before,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || Ok(())
            ),
            Err(Error::Store(crate::store::Error::L1ScanSuperseded { .. }))
        ));
        assert_eq!(store.index_snapshot().unwrap().files, before.files);
    }

    #[test]
    fn branch_catalogue_head_changes_and_writer_deadlines_refuse_with_truncated_index() {
        let (root, store, settings) = target_fixture();
        let git = |args: &[&str]| {
            assert!(
                std::process::Command::new("git")
                    .current_dir(root.path())
                    .args(args)
                    .output()
                    .unwrap()
                    .status
                    .success()
            );
        };
        git(&["init", "--initial-branch=main"]);
        git(&["add", "source.rs", "nested/other.rs"]);
        git(&[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "initial",
        ]);
        let generation = store.l1_generation().unwrap();
        let before = store.index_snapshot().unwrap();
        let (candidates, files) = branch_files(root.path(), &settings);
        git(&[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "next",
        ]);
        fs::write(root.path().join(".git/index"), b"DIRC").unwrap();
        assert!(
            publish_branch(
                &store,
                BranchInput {
                    git: None,
                    work: Work::default(),
                    candidates: &candidates,
                    files: &files,
                    snapshot: &before,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || Ok(())
            )
            .is_err()
        );
        let (candidates, files) = branch_files(root.path(), &settings);
        let held = store.writer().unwrap();
        let started = std::time::Instant::now();
        assert!(
            publish_branch(
                &store,
                BranchInput {
                    git: None,
                    work: Work {
                        deadline: Some(started + Duration::from_millis(30)),
                        cancelled: None
                    },
                    candidates: &candidates,
                    files: &files,
                    snapshot: &before,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || Ok(())
            )
            .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(held);
        assert_eq!(store.index_snapshot().unwrap().files, before.files);
        assert_eq!(store.l1_generation().unwrap(), generation);
        assert!(
            publish_branch(
                &store,
                BranchInput {
                    git: None,
                    work: Work::default(),
                    candidates: &candidates,
                    files: &files,
                    snapshot: &before,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || Ok(())
            )
            .is_ok()
        );
        assert_eq!(fs::read(root.path().join(".git/index")).unwrap(), b"DIRC");
        assert_eq!(
            store.get_meta("indexed_head").unwrap().as_deref(),
            Some("unchanged-head")
        );
    }

    #[test]
    fn targeted_catalogue_preserves_unrelated_sources_dirs_and_head_then_repairs() {
        let (root, store, settings) = target_fixture();
        let generation = store.l1_generation().unwrap();
        let snapshot = store.index_snapshot().unwrap();
        let before_keys = store
            .file_occurrence_keys(
                snapshot
                    .files
                    .iter()
                    .find(|f| f.path == "source.rs")
                    .unwrap()
                    .id,
            )
            .unwrap();
        fs::write(root.path().join("source.rs"), "pub fn after() {}\n").unwrap();
        let file = observe_source(root.path(), "source.rs", 1024, Work::default(), |_| {}).unwrap();
        let published = publish_target(
            &store,
            TargetInput {
                work: Work::default(),
                file: &file,
                snapshot: &snapshot,
                generation,
                root: root.path(),
                settings: &settings,
            },
            || Ok(()),
        )
        .unwrap();
        assert_eq!(published.changes.modified, ["source.rs"]);
        assert!(published.changes.added.is_empty());
        assert!(published.changes.deleted.is_empty());
        assert!(published.changes.renamed.is_empty());
        assert_eq!(
            store.pending_l1_paths().unwrap(),
            vec!["source.rs".to_owned()]
        );
        let after = store.index_snapshot().unwrap();
        assert_eq!(after.dirs, snapshot.dirs);
        assert_eq!(
            after.files.iter().find(|f| f.path == "nested/other.rs"),
            snapshot.files.iter().find(|f| f.path == "nested/other.rs")
        );
        assert_eq!(
            store.get_meta("indexed_head").unwrap().as_deref(),
            Some("unchanged-head")
        );
        let indexed = after.files.iter().find(|f| f.path == file.path).unwrap();
        assert_eq!(store.file_occurrence_keys(indexed.id).unwrap(), before_keys);
        Processor::new(root.path(), &settings)
            .unwrap()
            .process_target(
                &store,
                FileInput {
                    work: Work::default(),
                    file: &file,
                    fence: crate::store::l1::Fence {
                        generation: published.generation,
                        file_id: indexed.id,
                        path: &file.path,
                        content_hash: &file.blake3,
                        fingerprint: &published.fingerprints[&file.path],
                    },
                },
                || Ok(()),
            )
            .unwrap();
        assert!(store.pending_l1_paths().unwrap().is_empty());
        assert_ne!(store.file_occurrence_keys(indexed.id).unwrap(), before_keys);
        assert_eq!(
            store.get_meta("indexed_head").unwrap().as_deref(),
            Some("unchanged-head")
        );
    }

    #[test]
    fn targeted_processing_rechecks_scope_before_and_during_all_publications() {
        for cap in ["none", "source", "units"] {
            for ignore_at in [0, 1, 2] {
                let (root, store, mut settings) = target_fixture();
                if cap == "source" {
                    settings.limits.max_file_bytes = "4 B".into();
                } else if cap == "units" {
                    settings.limits.max_units_per_file = 0;
                }
                let snapshot = store.index_snapshot().unwrap();
                fs::write(root.path().join("source.rs"), "pub fn after() {}\n").unwrap();
                let file = observe_source(
                    root.path(),
                    "source.rs",
                    settings.limits.max_file_bytes().unwrap().0,
                    Work::default(),
                    |_| {},
                )
                .unwrap();
                let published = publish_target(
                    &store,
                    TargetInput {
                        work: Work::default(),
                        file: &file,
                        snapshot: &snapshot,
                        generation: store.l1_generation().unwrap(),
                        root: root.path(),
                        settings: &settings,
                    },
                    || Ok(()),
                )
                .unwrap();
                let before = store.index_snapshot().unwrap();
                let record = before.files.iter().find(|f| f.path == file.path).unwrap();
                let keys = store.file_occurrence_keys(record.id).unwrap();
                let symbols = store.tool_symbols().unwrap();
                let input = || FileInput {
                    work: Work::default(),
                    file: &file,
                    fence: crate::store::l1::Fence {
                        generation: published.generation,
                        file_id: record.id,
                        path: &file.path,
                        content_hash: &file.blake3,
                        fingerprint: &published.fingerprints[&file.path],
                    },
                };
                let ignore = root.path().join(".deepkitignore");
                if ignore_at == 0 {
                    fs::write(&ignore, "source.rs\n").unwrap();
                }
                let validations = std::cell::Cell::new(0);
                let processor = Processor::new(root.path(), &settings).unwrap();
                let result = processor.process_target(&store, input(), || {
                    validations.set(validations.get() + 1);
                    if validations.get() == ignore_at {
                        fs::write(&ignore, "source.rs\n").unwrap();
                    }
                    Ok(())
                });
                assert!(
                    matches!(
                        result,
                        Err(Error::Store(crate::store::Error::L1InputsChanged))
                    ),
                    "{cap}, validation {ignore_at}: {result:?}"
                );
                assert_eq!(validations.get(), ignore_at);
                let after = store.index_snapshot().unwrap();
                assert_eq!(after.files, before.files);
                assert_eq!(after.dirs, before.dirs);
                assert_eq!(store.l1_generation().unwrap(), published.generation);
                assert_eq!(store.file_occurrence_keys(record.id).unwrap(), keys);
                assert_eq!(store.tool_symbols().unwrap(), symbols);
                assert!(store.pending_l1_paths().unwrap().contains(&file.path));
                assert_eq!(
                    store.get_meta("indexed_head").unwrap().as_deref(),
                    Some("unchanged-head")
                );
                fs::remove_file(ignore).unwrap();
                let outcome = processor
                    .process_target(&store, input(), || Ok(()))
                    .unwrap();
                assert!(!store.pending_l1_paths().unwrap().contains(&file.path));
                if cap == "none" {
                    assert!(outcome.units > 0);
                    assert_ne!(store.file_occurrence_keys(record.id).unwrap(), keys);
                    assert!(
                        store
                            .tool_symbols()
                            .unwrap()
                            .iter()
                            .any(|s| s.name == "after")
                    );
                } else {
                    assert!(outcome.truncated);
                    assert_eq!(store.file_occurrence_keys(record.id).unwrap(), keys);
                }
            }
        }
    }

    #[test]
    fn targeted_catalogue_refuses_unknown_stopped_and_superseded_inputs() {
        let (root, store, settings) = target_fixture();
        let generation = store.l1_generation().unwrap();
        let snapshot = store.index_snapshot().unwrap();
        fs::write(root.path().join("new.rs"), "pub fn new() {}\n").unwrap();
        let new = observe_source(root.path(), "new.rs", 1024, Work::default(), |_| {}).unwrap();
        let file = observe_source(root.path(), "source.rs", 1024, Work::default(), |_| {}).unwrap();
        for (file, generation, work) in [
            (&new, generation, Work::default()),
            (&file, generation - 1, Work::default()),
            (
                &file,
                generation,
                Work {
                    deadline: Some(std::time::Instant::now()),
                    cancelled: None,
                },
            ),
        ] {
            assert!(
                publish_target(
                    &store,
                    TargetInput {
                        work,
                        file,
                        snapshot: &snapshot,
                        generation,
                        root: root.path(),
                        settings: &settings,
                    },
                    || Ok(())
                )
                .is_err()
            );
            assert_eq!(store.index_snapshot().unwrap(), snapshot);
            assert!(store.pending_l1_paths().unwrap().is_empty());
        }
        assert_eq!(store.l1_generation().unwrap(), generation);
    }

    #[test]
    fn targeted_catalogue_rolls_back_late_scope_source_and_cancellation_changes() {
        use std::{
            cell::Cell,
            sync::atomic::{AtomicBool, Ordering},
        };
        let (root, store, settings) = target_fixture();
        let generation = store.l1_generation().unwrap();
        let snapshot = store.index_snapshot().unwrap();
        for mode in ["scope", "source", "cancel", "ignore"] {
            fs::write(root.path().join("source.rs"), "pub fn after() {}\n").unwrap();
            let file =
                observe_source(root.path(), "source.rs", 1024, Work::default(), |_| {}).unwrap();
            let cancelled = AtomicBool::new(false);
            let calls = Cell::new(0);
            let result = publish_target(
                &store,
                TargetInput {
                    work: Work {
                        deadline: None,
                        cancelled: Some(&cancelled),
                    },
                    file: &file,
                    snapshot: &snapshot,
                    generation,
                    root: root.path(),
                    settings: &settings,
                },
                || {
                    calls.set(calls.get() + 1);
                    // Precheck, admitted transaction, record, then final precommit validation.
                    if calls.get() == 4 {
                        match mode {
                            "scope" => return Err(crate::store::Error::L1InputsChanged),
                            "source" => {
                                fs::write(root.path().join("source.rs"), "pub fn raced() {}\n")
                                    .unwrap()
                            }
                            "cancel" => cancelled.store(true, Ordering::Release),
                            "ignore" => {
                                fs::write(root.path().join(".deepkitignore"), "source.rs\n")
                                    .unwrap()
                            }
                            _ => unreachable!(),
                        }
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "{mode}");
            assert_eq!(calls.get(), 4);
            assert_eq!(store.index_snapshot().unwrap(), snapshot);
            assert_eq!(store.l1_generation().unwrap(), generation);
            assert!(store.pending_l1_paths().unwrap().is_empty());
            assert_eq!(
                store.get_meta("indexed_head").unwrap().as_deref(),
                Some("unchanged-head")
            );
        }
        fs::remove_file(root.path().join(".deepkitignore")).unwrap();
        let file = observe_source(root.path(), "source.rs", 1024, Work::default(), |_| {}).unwrap();
        publish_target(
            &store,
            TargetInput {
                work: Work::default(),
                file: &file,
                snapshot: &snapshot,
                generation,
                root: root.path(),
                settings: &settings,
            },
            || Ok(()),
        )
        .unwrap();
        assert_eq!(store.l1_generation().unwrap(), generation + 1);
    }

    #[test]
    fn targeted_observation_matches_scan_without_creating_a_catalogue() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("unrelated.rs"), b"untouched").unwrap();
        for (path, bytes, cap) in [
            ("source.rs", b"pub fn source() {}\n".as_slice(), 64),
            ("source.rs", b"binary\0payload", 64),
            ("source.rs", b"", 0),
            ("source.rs", b"oversized", 4),
            ("source.pdf", b"binary\0document", 64),
        ] {
            fs::write(root.path().join(path), bytes).unwrap();
            let mut captured = Vec::new();
            let observed = observe_source(root.path(), path, cap, Work::default(), |chunk| {
                captured.extend_from_slice(chunk);
            })
            .unwrap();
            let super::super::ScanFile::File(expected) =
                super::super::scan_file(root.path(), &root.path().join(path), cap, None).unwrap()
            else {
                panic!("regular source required")
            };
            assert_eq!(observed, expected);
            assert_eq!(
                captured,
                if observed.content_capped {
                    &[][..]
                } else {
                    bytes
                }
            );
        }
        assert_eq!(
            fs::read(root.path().join("unrelated.rs")).unwrap(),
            b"untouched"
        );
        assert!(!root.path().join(".deepkit").exists());
    }

    #[test]
    fn targeted_observation_stops_between_buffers_and_refuses_growth() {
        use std::{
            io::Write,
            sync::atomic::{AtomicBool, Ordering},
        };
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        let bytes = vec![b'x'; 130_000];
        fs::write(&path, &bytes).unwrap();
        let cancelled = AtomicBool::new(false);
        let mut consumed = 0;
        let result = observe_source(
            root.path(),
            "source.rs",
            130_000,
            Work {
                deadline: None,
                cancelled: Some(&cancelled),
            },
            |chunk| {
                consumed += chunk.len();
                cancelled.store(true, Ordering::Release);
            },
        );
        assert!(matches!(result, Err(Error::WorkStopped)));
        assert_eq!(consumed, super::super::STREAM_BUFFER_BYTES);
        let mut append = fs::OpenOptions::new().append(true).open(&path).unwrap();
        let result = observe_source(root.path(), "source.rs", 130_000, Work::default(), |_| {
            append.write_all(b"growth").unwrap();
        });
        assert!(matches!(result, Err(Error::SourceChanged { .. })));
        fs::write(&path, &bytes).unwrap();
        // A metadata change during hashing must refuse even with identical bytes.
        let result = observe_source(root.path(), "source.rs", 130_000, Work::default(), |_| {
            fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(std::time::UNIX_EPOCH))
                .unwrap();
        });
        assert!(matches!(result, Err(Error::SourceChanged { .. })));
        fs::write(&path, &bytes).unwrap();
        let observed =
            observe_source(root.path(), "source.rs", 130_000, Work::default(), |_| {}).unwrap();
        assert_eq!(observed.blake3, *blake3::hash(&bytes).as_bytes());
        assert!(matches!(
            observe_source(
                root.path(),
                "missing.rs",
                64,
                Work {
                    deadline: Some(std::time::Instant::now()),
                    cancelled: None
                },
                |_| panic!("expired observation must not read")
            ),
            Err(Error::WorkStopped)
        ));
    }

    #[test]
    fn targeted_observation_rejects_missing_redirected_and_nonregular_sources() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("source.rs"), b"outside").unwrap();
        symlink(outside.path(), root.path().join("redirect")).unwrap();
        symlink(
            outside.path().join("source.rs"),
            root.path().join("link.rs"),
        )
        .unwrap();
        mkfifo(
            &root.path().join("fifo.rs"),
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        )
        .unwrap();
        for path in [
            "missing.rs",
            "redirect/source.rs",
            "link.rs",
            "fifo.rs",
            "../source.rs",
            "",
        ] {
            assert!(
                observe_source(root.path(), path, 64, Work::default(), |_| panic!(
                    "invalid source must not read"
                ))
                .is_err(),
                "{path}"
            );
        }
    }

    #[test]
    fn only_explicit_work_limits_enable_checked_writer_admission() {
        let check = || Ok(());
        assert!(Work::default().admission_check(&check).is_none());
        assert!(
            Work {
                deadline: Some(std::time::Instant::now()),
                cancelled: None
            }
            .admission_check(&check)
            .is_some()
        );
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        assert!(
            Work {
                deadline: None,
                cancelled: Some(&cancelled)
            }
            .admission_check(&check)
            .is_some()
        );
    }

    #[test]
    fn cooperative_reads_stop_between_buffers_and_before_open() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("source.rs"), vec![b'x'; 130_000]).unwrap();
        let file = scanned(root.path(), 130_000);
        let cancelled = AtomicBool::new(false);
        let work = Work {
            deadline: None,
            cancelled: Some(&cancelled),
        };
        let mut consumed = 0;
        let result = read_verified(root.path(), &file, 130_000, work, |bytes| {
            consumed += bytes.len();
            cancelled.store(true, Ordering::Release);
        });
        assert!(matches!(result, Err(Error::WorkStopped)));
        assert_eq!(consumed, super::super::STREAM_BUFFER_BYTES);
        // Cancellation must take precedence over opening an absent source.
        fs::remove_file(root.path().join("source.rs")).unwrap();
        assert!(matches!(
            read_source(root.path(), &file, 130_000, work),
            Err(Error::WorkStopped)
        ));
        let expired = Work {
            deadline: Some(std::time::Instant::now()),
            cancelled: None,
        };
        assert!(matches!(
            verify_source(root.path(), &file, 130_000, expired),
            Err(Error::WorkStopped)
        ));
        cancelled.store(false, Ordering::Release);
        fs::write(root.path().join("source.rs"), vec![b'x'; 130_000]).unwrap();
        assert_eq!(
            read_source(root.path(), &file, 130_000, work)
                .unwrap()
                .len(),
            130_000
        );
    }

    #[test]
    fn cancelled_catalogue_keeps_generation_and_pending_work_then_retries() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("source.rs"), "pub fn source() {}\n").unwrap();
        let scan = super::super::scan(root.path(), Default::default()).unwrap();
        let store =
            crate::store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        let snapshot = store.index_snapshot().unwrap();
        let generation = store.l1_generation().unwrap();
        let settings = Config::default();
        let input = |work| PublishInput {
            scan: &scan,
            git: None,
            snapshot: &snapshot,
            generation,
            root: root.path(),
            settings: &settings,
            full: false,
            work,
        };
        assert!(matches!(
            publish(
                &store,
                input(Work {
                    deadline: Some(std::time::Instant::now()),
                    cancelled: None
                })
            ),
            Err(Error::WorkStopped)
        ));
        assert_eq!(store.l1_generation().unwrap(), generation);
        assert!(store.index_snapshot().unwrap().files.is_empty());
        assert!(store.pending_l1_paths().unwrap().is_empty());
        publish(&store, input(Work::default())).unwrap();
        assert!(
            store
                .pending_l1_paths()
                .unwrap()
                .contains(&"source.rs".to_owned())
        );
    }

    #[test]
    fn source_validation_detects_same_size_edits_with_restored_mtime() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        let original = b"pub fn original() {}\n";
        let replacement = b"pub fn replaced() {}\n";
        assert_eq!(original.len(), replacement.len());
        fs::write(&path, original).unwrap();
        let observed =
            observe_source(root.path(), "source.rs", 1024, Work::default(), |_| {}).unwrap();
        let mut persisted = observed.clone();
        persisted.source_stamp = std::sync::OnceLock::new();
        let mtime = fs::metadata(&path).unwrap().modified().unwrap();
        let restore_time = || {
            fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(mtime))
                .unwrap()
        };
        fs::write(&path, replacement).unwrap();
        restore_time();
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), mtime);
        assert!(matches!(
            verify_source(root.path(), &observed, 1024, Work::default()),
            Err(Error::SourceChanged { .. })
        ));
        assert!(verify_source(root.path(), &persisted, 1024, Work::default()).is_err());
        assert!(persisted.source_stamp.get().is_none());
        // A new inode with identical bytes may still validate after rehashing.
        let replacement_path = root.path().join("replacement");
        fs::write(&replacement_path, original).unwrap();
        fs::rename(&replacement_path, &path).unwrap();
        restore_time();
        verify_source(root.path(), &observed, 1024, Work::default()).unwrap();
        verify_source(root.path(), &persisted, 1024, Work::default()).unwrap();
        let mut forged = observed.clone();
        forged.blake3 = [0; 32];
        assert!(verify_source(root.path(), &forged, 1024, Work::default()).is_err());
    }

    #[test]
    fn bounded_l1_reads_verify_content_size_and_binary_classification() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        for bytes in [b"pub fn source() {}\n".as_slice(), b"binary\0payload", b""] {
            fs::write(&path, bytes).unwrap();
            let file = scanned(root.path(), 64);
            assert_eq!(
                read_source(root.path(), &file, 64, Work::default()).unwrap(),
                bytes
            );
            verify_source(root.path(), &file, 64, Work::default()).unwrap();
            let mut wrong = file.clone();
            wrong.lang = Some("incorrect".into());
            assert!(verify_source(root.path(), &wrong, 64, Work::default()).is_err());
            fs::write(&path, b"changed source").unwrap();
            assert!(read_source(root.path(), &file, 64, Work::default()).is_err());
            assert!(verify_source(root.path(), &file, 64, Work::default()).is_err());
        }
        fs::write(&path, vec![b'x'; 130_000]).unwrap();
        let file = scanned(root.path(), 130_000);
        assert_eq!(
            read_source(root.path(), &file, 130_000, Work::default())
                .unwrap()
                .len(),
            130_000
        );
        assert!(read_source(root.path(), &file, 129_999, Work::default()).is_err());
        // Grow after the descriptor metadata check and first read, exercising
        // the extra-byte bound rather than merely the initial stat refusal.
        use std::io::Write;
        let mut appended = false;
        let mut consumed = 0;
        assert!(
            read_verified(root.path(), &file, 130_000, Work::default(), |bytes| {
                consumed += bytes.len();
                if !appended {
                    appended = true;
                    fs::OpenOptions::new()
                        .append(true)
                        .open(&path)
                        .unwrap()
                        .write_all(b"growth")
                        .unwrap();
                }
            })
            .is_err()
        );
        assert!(consumed <= 130_000);
        fs::write(&path, vec![b'x'; 130_000]).unwrap();
        // Metadata captured before an append must not authorize a prefix hash.
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        assert!(verify_source(root.path(), &file, 130_000, Work::default()).is_err());
    }

    #[test]
    fn capped_l1_verification_uses_metadata_and_cannot_supply_chunk_bytes() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        fs::write(&path, b"oversized source").unwrap();
        let file = scanned(root.path(), 4);
        assert!(file.content_capped);
        verify_source(root.path(), &file, 4, Work::default()).unwrap();
        assert!(read_source(root.path(), &file, 4, Work::default()).is_err());
        assert!(verify_source(root.path(), &file, 64, Work::default()).is_err());
        fs::write(&path, b"tiny").unwrap();
        assert!(verify_source(root.path(), &file, 4, Work::default()).is_err());
    }

    #[test]
    fn l1_reads_refuse_identical_external_symlinks_and_fifo_replacements() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        fs::write(&path, b"pub fn original() {}\n").unwrap();
        fs::copy(&path, outside.path().join("source.rs")).unwrap();
        let file = scanned(root.path(), 64);
        fs::remove_file(&path).unwrap();
        symlink(outside.path().join("source.rs"), &path).unwrap();
        assert!(read_source(root.path(), &file, 64, Work::default()).is_err());
        assert!(verify_source(root.path(), &file, 64, Work::default()).is_err());
        fs::remove_file(&path).unwrap();
        mkfifo(&path, rustix::fs::Mode::RUSR).unwrap();
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            send.send((
                read_source(root.path(), &file, 64, Work::default()).is_err(),
                verify_source(root.path(), &file, 64, Work::default()).is_err(),
            ))
            .unwrap();
        });
        assert_eq!(
            receive.recv_timeout(Duration::from_secs(2)).unwrap(),
            (true, true)
        );
    }
}
