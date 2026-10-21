//! Repository discovery and synchronization; see `v1.md` §4.
pub(crate) mod linker;
pub(crate) mod modules;
pub(crate) mod source_limits;
pub(crate) mod symbol_history;

pub(crate) mod discovery;
pub(crate) mod head;
pub(crate) mod history;
pub(crate) mod hooks;
pub(crate) mod index;
#[cfg(test)]
mod index_loader_tests;
pub(crate) mod l1;
pub(crate) mod paths;
pub(crate) mod prs;
pub(crate) mod setup;
pub(crate) mod source;
mod tree_diff;
pub(crate) mod watch;

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use ignore::{WalkBuilder, WalkState};

use crate::{
    store::{self, DirUpsert, FileKind, FileUpsert, IndexSnapshot, Store},
    treat::Router,
};

const DEFAULT_MAX_FILES: usize = 250_000;
const DEFAULT_MAX_FILE_BYTES: u64 = 1024 * 1024;
const BINARY_PREFIX_BYTES: u64 = 8 * 1024;
const STREAM_HASH_THRESHOLD: u64 = 128 * 1024;
const STREAM_BUFFER_BYTES: usize = 64 * 1024;
/// Shared catalogue-cap action shown by indexing and status.
pub(crate) const MAX_FILES_MESSAGE: &str =
    "max_files cap hit, add ignores or treatments, then run dkc index again";

/// Limits applied while discovering repository files before treatments run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SyncLimits {
    pub(crate) max_files: usize,
    pub(crate) max_file_bytes: u64,
    pub(crate) threads: usize,
}

impl Default for SyncLimits {
    fn default() -> Self {
        Self {
            max_files: DEFAULT_MAX_FILES,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            threads: 0,
        }
    }
}

/// The bounded result of walking one repository.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Scan {
    files: Vec<ScannedFile>,
    dirs: Vec<ScannedDir>,
    pub(crate) status: SyncStatus,
}

impl Scan {
    /// Returns whether the walker observed the entire repository rather than stopping at a cap.
    pub(crate) fn is_complete(&self) -> bool {
        !self.status.max_files_reached
    }

    /// Returns one discovered file by its repository-relative path for L1 indexing.
    pub(crate) fn file(&self, path: &str) -> Option<&ScannedFile> {
        self.files.iter().find(|file| file.path == path)
    }

    /// Iterates discovered files in stable repository-relative path order.
    pub(crate) fn files(&self) -> impl Iterator<Item = &ScannedFile> {
        self.files.iter()
    }
}

/// A filesystem file found by the discovery walker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ScannedFile {
    pub(crate) path: String,
    blake3: [u8; blake3::OUT_LEN],
    size: u64,
    mtime_ns: i64,
    pub(crate) lang: Option<String>,
    pub(crate) kind: FileKind,
    pub(crate) content_capped: bool,
    hash_reused_from_git: bool,
    source_stamp: OnceLock<Box<(source::Stamp, [u8; 32])>>,
}

impl ScannedFile {
    /// Associates a digest with an OID only when the verified descriptor matches
    /// Git's non-racy clean stat entry. Dirty worktree bytes never inherit an OID.
    fn verified_blob<'a>(&self, git: &'a GitState) -> Option<&'a [u8]> {
        let (stamp, identity) = self.source_stamp.get()?.as_ref();
        if *identity != self.observation_identity()
            || !stamp.matches_index(git.clean_stats.get(&self.path)?)
        {
            return None;
        }
        git.tracked_blobs.get(&self.path).map(Vec::as_slice)
    }

    /// Binds a descriptor stamp to all facts established by the byte observation.
    fn with_source_stamp(self, stamp: Option<source::Stamp>) -> Self {
        if let Some(stamp) = stamp {
            let _ = self
                .source_stamp
                .set(Box::new((stamp, self.observation_identity())));
        }
        self
    }

    fn observation_identity(&self) -> [u8; 32] {
        let mut hash = blake3::Hasher::new();
        for part in [
            self.path.as_bytes(),
            self.kind.as_str().as_bytes(),
            self.lang.as_deref().unwrap_or_default().as_bytes(),
        ] {
            hash.update(&(part.len() as u64).to_le_bytes());
            hash.update(part);
        }
        hash.update(&self.blake3);
        hash.update(&[u8::from(self.lang.is_some())]);
        hash.update(&self.size.to_le_bytes());
        hash.update(&self.mtime_ns.to_le_bytes());
        hash.update(&[u8::from(self.content_capped)]);
        *hash.finalize().as_bytes()
    }

    /// Returns descriptor-observed byte length for source/catalogue comparison.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// Returns the observed content identity for a later fenced L1 publication.
    /// Capped observations carry a metadata marker rather than a content digest.
    pub(crate) fn content_hash(&self) -> &[u8; blake3::OUT_LEN] {
        &self.blake3
    }
}

/// A calculated directory Merkle record.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ScannedDir {
    path: String,
    merkle: [u8; blake3::OUT_LEN],
    file_count: i64,
}

/// Bounded-walk information surfaced by `status` once the CLI is wired.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SyncStatus {
    pub(crate) max_files_reached: bool,
    pub(crate) files_over_max_bytes: usize,
    pub(crate) hashes_reused_from_git: usize,
    pub(crate) skipped_non_unicode_paths: usize,
    pub(crate) messages: Vec<String>,
}

/// One detected rename that retains the existing file row identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Rename {
    pub(crate) id: i64,
    pub(crate) from: String,
    pub(crate) to: String,
}

/// The path-level changes between an indexed snapshot and a repository scan.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ChangeSet {
    pub(crate) added: Vec<String>,
    pub(crate) modified: Vec<String>,
    pub(crate) deleted: Vec<String>,
    pub(crate) renamed: Vec<Rename>,
}

/// Git state paired with a filesystem scan to retain blob identities and detect branch changes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GitState {
    pub(crate) head: Option<GitHead>,
    tracked_blobs: BTreeMap<String, Vec<u8>>,
    clean_stats: BTreeMap<String, gix::index::entry::Stat>,
    changed_paths: Option<BTreeSet<String>>,
    pub(crate) needs_full_rescan: bool,
    pub(crate) skipped_non_unicode_paths: usize,
}

/// The resolved object and reference currently named by `HEAD`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GitHead {
    pub(crate) id: String,
    pub(crate) reference: Option<String>,
    pub(crate) detached: bool,
}

/// A previously indexed tracked file that can avoid another content read.
#[derive(Clone, Debug)]
pub(crate) struct ReusedFile {
    clean_stat: gix::index::entry::Stat,
    blake3: [u8; blake3::OUT_LEN],
    size: i64,
    mtime_ns: i64,
    lang: Option<String>,
    kind: FileKind,
}

/// Failures while walking, hashing, or applying a repository synchronization.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("L1 work cancelled or its deadline expired")]
    WorkStopped,
    #[error("HEAD changed during repository discovery; retry reconciliation")]
    HeadChanged,
    #[error("source {path} changed during L1 preparation; index again")]
    SourceChanged { path: String },
    #[error("failed to resolve repository root {path}: {source}")]
    RootIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("repository root is not a directory: {path}")]
    RootNotDirectory { path: PathBuf },
    #[error("failed to read {path}: {source}")]
    FileIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("walker failed at {path}: {message}")]
    Walk { path: PathBuf, message: String },
    #[error("parallel walker result lock was poisoned")]
    CollectionLockPoisoned,
    #[error("file size cannot be stored in SQLite for {path}: {size} bytes")]
    FileSizeTooLarge { path: String, size: u64 },
    #[error("indexed file disappeared before it could be renamed from {from} to {to}")]
    RenameMissing { from: String, to: String },
    #[error("failed to inspect Git repository {path}: {message}")]
    Git { path: PathBuf, message: String },
    #[error(transparent)]
    Config(#[from] crate::config::Error),
    #[error(transparent)]
    Treatment(#[from] crate::treat::Error),
    #[error(transparent)]
    Store(#[from] store::Error),
}

/// Discovers Git metadata for a repository without treating a non-Git directory as an error.
pub(crate) fn inspect_git(
    root: impl AsRef<Path>,
    indexed_head: Option<&str>,
) -> Result<Option<GitState>, Error> {
    inspect_git_checked(root.as_ref(), indexed_head, l1::Work::default())
}

/// Reads optional reuse evidence while charging Git work to the original budget.
pub(crate) fn inspect_git_checked(
    root: &Path,
    indexed_head: Option<&str>,
    work: l1::Work<'_>,
) -> Result<Option<GitState>, Error> {
    work.check()?;
    let Some(repo) = head::discover(root)? else {
        return Ok(None);
    };
    let resolved_head = head::resolve(&repo, root)?;
    let mut tracked_blobs = BTreeMap::new();
    let mut clean_stats = BTreeMap::new();
    let mut skipped_non_unicode_paths = 0;
    if let Some(index) = repo.try_index().map_err(|error| Error::Git {
        path: root.to_path_buf(),
        message: error.to_string(),
    })? {
        for entry in index.entries() {
            work.check()?;
            if entry.stage() != gix::index::entry::Stage::Unconflicted
                || !entry.mode.intersects(
                    gix::index::entry::Mode::FILE | gix::index::entry::Mode::FILE_EXECUTABLE,
                )
            {
                continue;
            }
            let path = entry.path(&index);
            let Ok(path) = std::str::from_utf8(path) else {
                skipped_non_unicode_paths += 1;
                continue;
            };
            let path = normalize_path(path);
            if !entry.stat.is_racy(
                index.timestamp(),
                gix::index::entry::stat::Options {
                    trust_ctime: true,
                    check_stat: true,
                    use_nsec: true,
                    use_stdev: true,
                },
            ) {
                clean_stats.insert(path.clone(), entry.stat);
            }
            tracked_blobs.insert(path, entry.id.as_slice().to_vec());
        }
    }
    let needs_full_rescan = head::changed(
        indexed_head,
        resolved_head.as_ref().map(|current| current.id.as_str()),
    );
    let changed_paths = tree_diff::between(
        &repo,
        indexed_head,
        resolved_head.as_ref().map(|head| head.id.as_str()),
        work,
    )?;
    work.check()?;
    Ok(Some(GitState {
        head: resolved_head,
        tracked_blobs,
        clean_stats,
        changed_paths,
        needs_full_rescan,
        skipped_non_unicode_paths,
    }))
}

/// Walks a repository in parallel while applying Git, global, and Deepkit ignore rules.
pub(crate) fn scan(root: impl AsRef<Path>, limits: SyncLimits) -> Result<Scan, Error> {
    scan_with_reuse(root, limits, None, l1::Work::default())
}

/// Walks a repository and reuses content hashes for unchanged Git-indexed files.
pub(crate) fn scan_with_git(
    root: impl AsRef<Path>,
    limits: SyncLimits,
    git: &GitState,
    snapshot: &IndexSnapshot,
) -> Result<Scan, Error> {
    let mut scan = scan_with_reuse(
        root,
        limits,
        Some(Arc::new(reusable_files(
            git,
            snapshot,
            l1::Work::default(),
        )?)),
        l1::Work::default(),
    )?;
    scan.status.skipped_non_unicode_paths += git.skipped_non_unicode_paths;
    Ok(scan)
}

/// Walks with cooperative cancellation while retaining unchanged tracked-file reuse.
/// Individual filesystem reads finish before the next admission check.
pub(crate) fn scan_checked(
    root: &Path,
    limits: SyncLimits,
    git: Option<&GitState>,
    snapshot: &IndexSnapshot,
    work: l1::Work<'_>,
) -> Result<Scan, Error> {
    work.check()?;
    let reuse = git
        .map(|git| reusable_files(git, snapshot, work).map(Arc::new))
        .transpose()?;
    let mut scan = scan_with_reuse(root, limits, reuse, work)?;
    if let Some(git) = git {
        scan.status.skipped_non_unicode_paths += git.skipped_non_unicode_paths;
    }
    work.check()?;
    Ok(scan)
}

fn discovery_entry(entry: &ignore::DirEntry) -> bool {
    ![
        ".git",
        ".deepkit",
        ".gitignore",
        ".deepkitignore",
        ".ignore",
    ]
    .iter()
    .any(|ignored| entry.file_name() == OsStr::new(ignored))
}

/// Builds the shared eligible-file discovery policy without parsing the Git index.
pub(crate) fn discovery_builder(root: &Path) -> WalkBuilder {
    let mut builder = WalkBuilder::new(root);
    builder
        .current_dir(root)
        .standard_filters(true)
        .hidden(false)
        .add_custom_ignore_filename(".deepkitignore")
        .require_git(false)
        .follow_links(false)
        .filter_entry(discovery_entry);
    builder
}

/// Rechecks discovery eligibility using current ignore files, descending only
/// through the target's ancestors. No source contents are read or catalogue rows
/// changed. Call again at publication; an earlier match is not an authorization.
/// Walker filesystem calls remain cooperative: work is checked between entries.
pub(crate) fn target_eligible(root: &Path, path: &str, work: l1::Work<'_>) -> Result<bool, Error> {
    work.check()?;
    let relative = Path::new(path);
    if path.is_empty()
        || relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Ok(false);
    }
    let root = fs::canonicalize(root).map_err(|source| Error::RootIo {
        path: root.to_path_buf(),
        source,
    })?;
    if !root.is_dir() {
        return Err(Error::RootNotDirectory { path: root });
    }
    let target = root.join(relative);
    let filter_target = target.clone();
    let mut builder = discovery_builder(&root);
    // filter_entry replaces the previous predicate, so retain the shared exclusions.
    builder.filter_entry(move |entry| {
        discovery_entry(entry) && filter_target.starts_with(entry.path())
    });
    for entry in builder.build() {
        work.check()?;
        let entry = entry.map_err(|error| Error::Walk {
            path: root.clone(),
            message: error.to_string(),
        })?;
        if let Some(error) = entry.error() {
            return Err(Error::Walk {
                path: entry.path().to_path_buf(),
                message: error.to_string(),
            });
        }
        if entry.path() == target {
            work.check()?;
            return Ok(
                !entry.path_is_symlink() && entry.file_type().is_some_and(|kind| kind.is_file())
            );
        }
    }
    work.check()?;
    Ok(false)
}

fn scan_with_reuse(
    root: impl AsRef<Path>,
    limits: SyncLimits,
    reusable_files: Option<Arc<BTreeMap<String, ReusedFile>>>,
    work: l1::Work<'_>,
) -> Result<Scan, Error> {
    work.check()?;
    let root_input = root.as_ref();
    let root = fs::canonicalize(root_input).map_err(|source| Error::RootIo {
        path: root_input.to_path_buf(),
        source,
    })?;
    if !root.is_dir() {
        return Err(Error::RootNotDirectory { path: root });
    }

    let collected = Arc::new(Mutex::new(Collected::default()));
    let regular_files = Arc::new(AtomicUsize::new(0));
    let max_files_reached = Arc::new(AtomicBool::new(false));
    let mut builder = discovery_builder(&root);
    builder.threads(limits.threads);
    // Do not set WalkBuilder::max_filesize here. Files above the cap are
    // catalogued at L0 by `scan_file`, per v1.md §3.5 and PCR-002.
    builder.build_parallel().run({
        let root = root.clone();
        let collected = Arc::clone(&collected);
        let regular_files = Arc::clone(&regular_files);
        let max_files_reached = Arc::clone(&max_files_reached);
        let reusable_files = reusable_files.as_ref().map(Arc::clone);
        move || {
            let root = root.clone();
            let collected = Arc::clone(&collected);
            let regular_files = Arc::clone(&regular_files);
            let max_files_reached = Arc::clone(&max_files_reached);
            let reusable_files = reusable_files.as_ref().map(Arc::clone);
            Box::new(move |entry| {
                if let Err(error) = work.check() {
                    return record_error(&collected, error);
                }
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        return record_error(
                            &collected,
                            Error::Walk {
                                path: root.clone(),
                                message: error.to_string(),
                            },
                        );
                    }
                };
                if let Some(error) = entry.error() {
                    return record_error(
                        &collected,
                        Error::Walk {
                            path: entry.path().to_path_buf(),
                            message: error.to_string(),
                        },
                    );
                }
                if entry.path_is_symlink()
                    || !entry
                        .file_type()
                        .is_some_and(|file_type| file_type.is_file())
                {
                    return WalkState::Continue;
                }
                if regular_files.fetch_add(1, Ordering::Relaxed) >= limits.max_files {
                    max_files_reached.store(true, Ordering::Relaxed);
                    return WalkState::Quit;
                }
                match scan_file(
                    &root,
                    entry.path(),
                    limits.max_file_bytes,
                    reusable_files.as_deref(),
                ) {
                    Ok(ScanFile::File(file)) => match collected.lock() {
                        Ok(mut collected) => {
                            collected.files.push(file);
                            WalkState::Continue
                        }
                        Err(_) => WalkState::Quit,
                    },
                    Ok(ScanFile::NonUnicodePath) => match collected.lock() {
                        Ok(mut collected) => {
                            collected.skipped_non_unicode_paths += 1;
                            WalkState::Continue
                        }
                        Err(_) => WalkState::Quit,
                    },
                    Err(error) => record_error(&collected, error),
                }
            })
        }
    });

    work.check()?;
    let mut collected = collected
        .lock()
        .map_err(|_| Error::CollectionLockPoisoned)?;
    if let Some(error) = collected.error.take() {
        return Err(error);
    }
    collected
        .files
        .sort_by(|left, right| left.path.cmp(&right.path));
    let dirs = calculate_dirs(&collected.files);
    let files_over_max_bytes = collected
        .files
        .iter()
        .filter(|file| file.content_capped)
        .count();
    let hashes_reused_from_git = collected
        .files
        .iter()
        .filter(|file| file.hash_reused_from_git)
        .count();
    let reached = max_files_reached.load(Ordering::Relaxed);
    let mut messages = Vec::new();
    if reached {
        messages.push(MAX_FILES_MESSAGE.to_owned());
    }
    Ok(Scan {
        files: collected.files.clone(),
        dirs,
        status: SyncStatus {
            max_files_reached: reached,
            files_over_max_bytes,
            hashes_reused_from_git,
            skipped_non_unicode_paths: collected.skipped_non_unicode_paths,
            messages,
        },
    })
}

/// Computes and persists the safe incremental changes from a completed scan.
pub(crate) fn apply(store: &Store, scan: &Scan) -> Result<ChangeSet, Error> {
    apply_with_router(store, scan, &Router::default())
}

/// Persists a scan using a treatment router to set each indexed file's ceiling.
pub(crate) fn apply_with_router(
    store: &Store,
    scan: &Scan,
    router: &Router,
) -> Result<ChangeSet, Error> {
    apply_with_git_state(store, scan, None, router)
}

/// Persists a Git-aware catalogue and tracked blob IDs without claiming L1 completion.
pub(crate) fn apply_with_git(
    store: &Store,
    scan: &Scan,
    git: &GitState,
) -> Result<ChangeSet, Error> {
    apply_with_git_and_router(store, scan, git, &Router::default())
}

/// Persists a Git-aware scan with the configured treatment ceilings and blob IDs.
pub(crate) fn apply_with_git_and_router(
    store: &Store,
    scan: &Scan,
    git: &GitState,
    router: &Router,
) -> Result<ChangeSet, Error> {
    apply_with_git_state(store, scan, Some(git), router)
}

fn apply_with_git_state(
    store: &Store,
    scan: &Scan,
    git: Option<&GitState>,
    router: &Router,
) -> Result<ChangeSet, Error> {
    let snapshot = store.index_snapshot()?;
    let changes = change_set(&snapshot, &scan.files, scan.is_complete());
    let files_by_path = scan
        .files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<BTreeMap<_, _>>();

    for rename in &changes.renamed {
        if !store.rename_file(&rename.from, &rename.to)? {
            return Err(Error::RenameMissing {
                from: rename.from.clone(),
                to: rename.to.clone(),
            });
        }
    }
    for path in &changes.deleted {
        store.delete_file(path)?;
    }

    let mut changed_paths = changes
        .added
        .iter()
        .chain(&changes.modified)
        .chain(changes.renamed.iter().map(|rename| &rename.to))
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if let Some(git) = git {
        let prior_by_path = snapshot
            .files
            .iter()
            .map(|file| (file.path.as_str(), file))
            .collect::<BTreeMap<_, _>>();
        for file in &scan.files {
            let prior_blob = prior_by_path
                .get(file.path.as_str())
                .and_then(|prior| prior.git_blob.as_deref());
            let current_blob = file.verified_blob(git);
            if prior_blob != current_blob {
                changed_paths.insert(file.path.as_str());
            }
        }
    }
    for path in changed_paths {
        if let Some(file) = files_by_path.get(path) {
            let git_blob = git.and_then(|git| file.verified_blob(git));
            upsert_scanned_file(store, file, git_blob, router)?;
        }
    }

    if scan.is_complete() {
        apply_dirs(store, &snapshot, &scan.dirs)?;
    }
    Ok(changes)
}

#[derive(Default)]
struct Collected {
    files: Vec<ScannedFile>,
    skipped_non_unicode_paths: usize,
    error: Option<Error>,
}

enum ScanFile {
    File(ScannedFile),
    NonUnicodePath,
}

fn record_error(collected: &Mutex<Collected>, error: Error) -> WalkState {
    match collected.lock() {
        Ok(mut collected) => {
            if collected.error.is_none() {
                collected.error = Some(error);
            }
            WalkState::Quit
        }
        Err(_) => WalkState::Quit,
    }
}

fn scan_file(
    root: &Path,
    path: &Path,
    max_file_bytes: u64,
    reusable_files: Option<&BTreeMap<String, ReusedFile>>,
) -> Result<ScanFile, Error> {
    let relative = match path.strip_prefix(root) {
        Ok(relative) => relative,
        Err(_) => {
            return Err(Error::Walk {
                path: path.to_path_buf(),
                message: "walker returned a path outside its root".to_owned(),
            });
        }
    };
    let Some(path) = relative.to_str().map(normalize_path) else {
        return Ok(ScanFile::NonUnicodePath);
    };
    let absolute = root.join(relative);
    let metadata = fs::metadata(&absolute).map_err(|source| Error::FileIo {
        path: absolute.clone(),
        source,
    })?;
    let size = metadata.len();
    let mtime_ns = modified_nanos(&metadata, &absolute)?;
    if let Some(file) = reused_observation(&path, &metadata, max_file_bytes, reusable_files) {
        return Ok(ScanFile::File(file));
    }
    let (mut kind, lang) = classify_path(&path);
    let mut lang = lang.map(str::to_owned);
    let content_capped = size > max_file_bytes;
    if !content_capped && !is_known_document_or_image(kind) && has_binary_prefix(&absolute)? {
        kind = FileKind::Other;
        lang = None;
    }
    let blake3 = if content_capped {
        capped_digest(&path, size, mtime_ns)
    } else {
        hash_file(&absolute, max_file_bytes, size)?
    };
    let source_stamp = source::stamp(&metadata);
    let final_metadata = fs::metadata(&absolute).map_err(|source| Error::FileIo {
        path: absolute.clone(),
        source,
    })?;
    if source_stamp != source::stamp(&final_metadata)
        || final_metadata.len() != size
        || modified_nanos(&final_metadata, &absolute)? != mtime_ns
    {
        return Err(Error::SourceChanged { path });
    }
    Ok(ScanFile::File(
        ScannedFile {
            path,
            blake3,
            size,
            mtime_ns,
            lang,
            kind,
            content_capped,
            hash_reused_from_git: false,
            source_stamp: OnceLock::new(),
        }
        .with_source_stamp(source_stamp),
    ))
}

/// Observes one discovered candidate, reusing only clean, unchanged Git evidence.
/// Ignore/discovery/HEAD authority remains with the caller and its final fences.
pub(crate) fn observe_with_reuse(
    root: &Path,
    path: &str,
    max_bytes: u64,
    reusable: &BTreeMap<String, ReusedFile>,
    work: l1::Work<'_>,
) -> Result<ScannedFile, Error> {
    work.check()?;
    if reusable.contains_key(path) {
        let input =
            source::open_regular(root, Path::new(path)).map_err(|source| Error::FileIo {
                path: root.join(path),
                source,
            })?;
        let metadata = input.metadata().map_err(|source| Error::FileIo {
            path: root.join(path),
            source,
        })?;
        work.check()?;
        if let Some(file) = reused_observation(path, &metadata, max_bytes, Some(reusable)) {
            return Ok(file);
        }
    }
    l1::observe_source(root, path, max_bytes, work, |_| {})
}

fn reused_observation(
    path: &str,
    metadata: &fs::Metadata,
    max_bytes: u64,
    reusable: Option<&BTreeMap<String, ReusedFile>>,
) -> Option<ScannedFile> {
    let reused = reusable?.get(path)?;
    let stamp = source::stamp(metadata)?;
    let size = metadata.len();
    let mtime_ns = modified_nanos(metadata, Path::new(path)).ok()?;
    let (kind, language) = classify_path(path);
    if size > max_bytes
        || i64::try_from(size).ok()? != reused.size
        || mtime_ns != reused.mtime_ns
        || !stamp.matches_index(&reused.clean_stat)
        || kind != reused.kind
        || language != reused.lang.as_deref()
    {
        return None;
    }
    Some(
        ScannedFile {
            path: path.to_owned(),
            blake3: reused.blake3,
            size,
            mtime_ns,
            lang: reused.lang.clone(),
            kind: reused.kind,
            content_capped: false,
            hash_reused_from_git: true,
            source_stamp: OnceLock::new(),
        }
        .with_source_stamp(Some(stamp)),
    )
}

/// Selects known same-OID payloads outside the changed tree paths. Actual stat
/// verification is deferred until the candidate is observed through the filesystem.
pub(crate) fn reusable_files(
    git: &GitState,
    snapshot: &IndexSnapshot,
    work: l1::Work<'_>,
) -> Result<BTreeMap<String, ReusedFile>, Error> {
    snapshot
        .files
        .iter()
        .map(|file| {
            work.check()?;
            Ok::<_, Error>((|| {
                if git.needs_full_rescan
                    && git
                        .changed_paths
                        .as_ref()
                        .is_none_or(|paths| paths.contains(&file.path))
                {
                    return None;
                }
                let current_blob = git.tracked_blobs.get(&file.path)?;
                if file.git_blob.as_deref() != Some(current_blob.as_slice()) {
                    return None;
                }
                let blake3 = <[u8; blake3::OUT_LEN]>::try_from(file.blake3.as_slice()).ok()?;
                // Capped L0 digests describe stat metadata, not content. A higher
                // cap must hash the bytes before promoting an unchanged tracked file.
                if blake3
                    == capped_digest(&file.path, u64::try_from(file.size).ok()?, file.mtime_ns)
                {
                    return None;
                }
                let kind = FileKind::from_db(&file.kind)?;
                Some((
                    file.path.clone(),
                    ReusedFile {
                        clean_stat: *git.clean_stats.get(&file.path)?,
                        blake3,
                        size: file.size,
                        mtime_ns: file.mtime_ns,
                        lang: file.lang.clone(),
                        kind,
                    },
                ))
            })())
        })
        .filter_map(Result::transpose)
        .collect()
}

fn normalize_path(path: &str) -> String {
    path.replace(std::path::MAIN_SEPARATOR, "/")
}

fn modified_nanos(metadata: &fs::Metadata, path: &Path) -> Result<i64, Error> {
    let modified = metadata.modified().map_err(|source| Error::FileIo {
        path: path.to_path_buf(),
        source,
    })?;
    let nanos = match modified.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos(),
        Err(_) => 0,
    };
    Ok(i64::try_from(nanos).unwrap_or(i64::MAX))
}

fn is_known_document_or_image(kind: FileKind) -> bool {
    matches!(kind, FileKind::Doc | FileKind::Image)
}

fn has_binary_prefix(path: &Path) -> Result<bool, Error> {
    let file = fs::File::open(path).map_err(|source| Error::FileIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut prefix = Vec::with_capacity(BINARY_PREFIX_BYTES as usize);
    file.take(BINARY_PREFIX_BYTES)
        .read_to_end(&mut prefix)
        .map_err(|source| Error::FileIo {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(prefix.contains(&0))
}

fn hash_file(path: &Path, max_file_bytes: u64, size: u64) -> Result<[u8; blake3::OUT_LEN], Error> {
    if size <= STREAM_HASH_THRESHOLD {
        return hash_small_file(path, max_file_bytes);
    }
    hash_streamed_file(path, max_file_bytes)
}

fn hash_small_file(path: &Path, max_file_bytes: u64) -> Result<[u8; blake3::OUT_LEN], Error> {
    let file = fs::File::open(path).map_err(|source| Error::FileIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.take(max_file_bytes)
        .read_to_end(&mut bytes)
        .map_err(|source| Error::FileIo {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

fn hash_streamed_file(path: &Path, max_file_bytes: u64) -> Result<[u8; blake3::OUT_LEN], Error> {
    let file = fs::File::open(path).map_err(|source| Error::FileIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file.take(max_file_bytes));
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0; STREAM_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer).map_err(|source| Error::FileIo {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(*hasher.finalize().as_bytes())
}

fn capped_digest(path: &str, size: u64, mtime_ns: i64) -> [u8; blake3::OUT_LEN] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"dkc:l0-content-capped\\0");
    hasher.update(path.as_bytes());
    hasher.update(&size.to_le_bytes());
    hasher.update(&mtime_ns.to_le_bytes());
    *hasher.finalize().as_bytes()
}

// One inventory drives both constant-time detection and exhaustive route acceptance.
macro_rules! language_extensions {
    ($($($extension:literal)|+ => ($kind:ident, $language:literal),)*) => {
        fn classify_language_extension(extension: &str) -> Option<(FileKind, Option<&'static str>)> {
            match extension {
                $($($extension)|+ => Some((FileKind::$kind, Some($language))),)*
                _ => None,
            }
        }
        #[cfg(test)]
        pub(crate) const LANGUAGE_EXTENSIONS: &[(&str, FileKind, &str)] = &[
            $($(($extension, FileKind::$kind, $language),)+)*
        ];
    };
}
language_extensions! {
    "rs" => (Code, "rust"),
    "py" | "pyw" => (Code, "python"),
    "js" | "mjs" | "cjs" | "jsx" => (Code, "javascript"),
    "ts" => (Code, "typescript"),
    "tsx" => (Code, "tsx"),
    "go" => (Code, "go"),
    "java" => (Code, "java"),
    "kt" | "kts" => (Code, "kotlin"),
    "c" | "h" => (Code, "c"),
    "cc" | "cp" | "cpp" | "cxx" | "c++" | "hh" | "hpp" | "hxx" => (Code, "cpp"),
    "cs" => (Code, "csharp"),
    "php" => (Code, "php"),
    "rb" => (Code, "ruby"),
    "swift" => (Code, "swift"),
    "scala" | "sc" => (Code, "scala"),
    "sh" | "bash" | "zsh" => (Code, "bash"),
    "html" | "htm" => (Code, "html"),
    "css" | "scss" | "sass" => (Code, "css"),
    "sql" => (Code, "sql"),
    "dart" => (Code, "dart"),
    "ex" | "exs" => (Code, "elixir"),
    "lua" => (Code, "lua"),
    "zig" => (Code, "zig"),
    "hs" | "lhs" => (Code, "haskell"),
    "ml" | "mli" => (Code, "ocaml"),
    "proto" => (Code, "protobuf"),
    "json" | "json5" => (Config, "json"),
    "yaml" | "yml" => (Config, "yaml"),
    "toml" => (Config, "toml"),
    "md" | "markdown" | "mdx" | "rst" | "txt" => (Doc, "markdown"),
}

/// Classifies a path without reading its bytes, including historical policy checks.
pub(crate) fn classify_path(path: &str) -> (FileKind, Option<&'static str>) {
    let file_name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    if file_name == "dockerfile" {
        return (FileKind::Code, Some("dockerfile"));
    }
    if file_name.starts_with(".env") {
        return (FileKind::Config, Some("env"));
    }
    let extension = Path::new(&file_name)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default();
    if let Some(classification) = classify_language_extension(extension) {
        return classification;
    }
    match extension {
        "ini" | "cfg" | "conf" | "properties" => (FileKind::Config, None),
        "pdf" | "doc" | "docx" | "odt" | "xls" | "xlsx" | "ppt" | "pptx" => (FileKind::Doc, None),
        "ipynb" => (FileKind::Doc, None),
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff" => (FileKind::Image, None),
        "csv" | "tsv" | "xml" => (FileKind::Data, None),
        _ => (FileKind::Other, None),
    }
}

fn calculate_dirs(files: &[ScannedFile]) -> Vec<ScannedDir> {
    let mut dirs = BTreeMap::<String, DirNode>::new();
    dirs.insert(String::new(), DirNode::default());
    for file in files {
        let (parent, name) = split_parent(&file.path);
        ensure_ancestors(&mut dirs, parent);
        let node = dirs.entry(parent.to_owned()).or_default();
        node.children.insert(name.to_owned(), file.blake3);
        node.direct_file_count += 1;
    }

    let paths = dirs.keys().cloned().collect::<Vec<_>>();
    let mut output = Vec::with_capacity(paths.len());
    for path in paths.into_iter().rev() {
        let node = dirs.get(&path).cloned().unwrap_or_default();
        let mut hasher = blake3::Hasher::new();
        for (name, digest) in &node.children {
            hasher.update(name.as_bytes());
            hasher.update(digest);
        }
        let merkle = *hasher.finalize().as_bytes();
        let file_count = node.direct_file_count;
        output.push(ScannedDir {
            path: path.clone(),
            merkle,
            file_count,
        });
        if !path.is_empty() {
            let (parent, name) = split_parent(&path);
            let parent = dirs.entry(parent.to_owned()).or_default();
            parent.children.insert(name.to_owned(), merkle);
            parent.direct_file_count += file_count;
        }
    }
    output.sort_by(|left, right| left.path.cmp(&right.path));
    output
}

#[derive(Clone, Default)]
struct DirNode {
    children: BTreeMap<String, [u8; blake3::OUT_LEN]>,
    direct_file_count: i64,
}

fn ensure_ancestors(dirs: &mut BTreeMap<String, DirNode>, path: &str) {
    let mut current = String::new();
    if !path.is_empty() {
        for part in path.split('/') {
            if !current.is_empty() {
                current.push('/');
            }
            current.push_str(part);
            dirs.entry(current.clone()).or_default();
        }
    }
}

fn split_parent(path: &str) -> (&str, &str) {
    split_parent_option(path).unwrap_or(("", path))
}

fn split_parent_option(path: &str) -> Option<(&str, &str)> {
    path.rsplit_once('/')
}

fn change_set(snapshot: &IndexSnapshot, files: &[ScannedFile], complete: bool) -> ChangeSet {
    let current = files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    let previous = snapshot
        .files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    let mut changes = ChangeSet::default();
    for (path, file) in &current {
        match previous.get(path) {
            None => changes.added.push((*path).to_owned()),
            Some(previous) if previous.blake3.as_slice() != file.blake3 => {
                changes.modified.push((*path).to_owned());
            }
            Some(_) => {}
        }
    }
    if !complete {
        return changes;
    }
    let mut deleted = Vec::new();
    for (path, previous) in &previous {
        if !current.contains_key(path) {
            deleted.push(((*path).to_owned(), *previous));
        }
    }

    let added_by_digest = changes
        .added
        .iter()
        .filter_map(|path| current.get(path.as_str()).map(|file| (file.blake3, path)))
        .fold(
            BTreeMap::<[u8; blake3::OUT_LEN], Vec<&String>>::new(),
            |mut grouped, (digest, path)| {
                grouped.entry(digest).or_default().push(path);
                grouped
            },
        );
    let deleted_by_digest = deleted.iter().fold(
        BTreeMap::<Vec<u8>, Vec<(&String, &store::IndexedFile)>>::new(),
        |mut grouped, (path, file)| {
            grouped
                .entry(file.blake3.clone())
                .or_default()
                .push((path, file));
            grouped
        },
    );
    let mut renamed_from = BTreeSet::new();
    let mut renamed_to = BTreeSet::new();
    for (digest, added) in added_by_digest {
        let Some(deleted) = deleted_by_digest.get(digest.as_slice()) else {
            continue;
        };
        if added.len() == 1 && deleted.len() == 1 {
            let (from, indexed) = deleted[0];
            let to = added[0];
            changes.renamed.push(Rename {
                id: indexed.id,
                from: from.clone(),
                to: to.clone(),
            });
            renamed_from.insert(from.clone());
            renamed_to.insert(to.clone());
        }
    }
    changes.added.retain(|path| !renamed_to.contains(path));
    changes.deleted = deleted
        .into_iter()
        .filter(|(path, _)| !renamed_from.contains(path))
        .map(|(path, _)| path)
        .collect();
    changes
}

fn upsert_scanned_file(
    store: &Store,
    file: &ScannedFile,
    git_blob: Option<&[u8]>,
    router: &Router,
) -> Result<(), Error> {
    let size = i64::try_from(file.size).map_err(|_| Error::FileSizeTooLarge {
        path: file.path.clone(),
        size: file.size,
    })?;
    let treatment = router.route(&file.path, file.kind);
    store.upsert_file(FileUpsert {
        path: &file.path,
        blake3: &file.blake3,
        size,
        mtime_ns: file.mtime_ns,
        lang: file.lang.as_deref(),
        kind: file.kind,
        git_blob,
        max_level: treatment.max_level,
        treatment: &treatment.name,
        indexed_at: Some(unix_timestamp()),
        needs_vlm: false,
    })?;
    Ok(())
}

fn apply_dirs(store: &Store, snapshot: &IndexSnapshot, dirs: &[ScannedDir]) -> Result<(), Error> {
    let previous = snapshot
        .dirs
        .iter()
        .map(|directory| (directory.path.as_str(), directory))
        .collect::<BTreeMap<_, _>>();
    let current = dirs
        .iter()
        .map(|directory| (directory.path.as_str(), directory))
        .collect::<BTreeMap<_, _>>();
    for (path, directory) in &current {
        let unchanged = previous.get(path).is_some_and(|previous| {
            previous.merkle.as_slice() == directory.merkle
                && previous.file_count == directory.file_count
        });
        if !unchanged {
            store.upsert_dir(DirUpsert {
                path,
                merkle: &directory.merkle,
                file_count: directory.file_count,
            })?;
        }
    }
    for path in previous.keys().filter(|path| !current.contains_key(**path)) {
        store.delete_dir(path)?;
    }
    Ok(())
}

fn unix_timestamp() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, l1, target_eligible};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::{collections::BTreeMap, error::Error as StdError, fs, path::Path, process::Command};

    use proptest::prelude::*;

    use crate::{
        config::Treatment,
        store::{self, FileKind},
        sync::{
            MAX_FILES_MESSAGE, SyncLimits, apply, apply_with_git, apply_with_router, change_set,
            classify_path, inspect_git, scan, scan_with_git,
        },
        treat::Router,
    };

    fn write(root: &Path, relative: &str, contents: &[u8]) -> Result<(), Box<dyn StdError>> {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, contents)?;
        Ok(())
    }

    fn file_map(scan: &super::Scan) -> BTreeMap<&str, &super::ScannedFile> {
        scan.files
            .iter()
            .map(|file| (file.path.as_str(), file))
            .collect()
    }

    fn refresh_git_stat(root: &Path) -> Result<(), Box<dyn StdError>> {
        // Checkout may leave a deliberately racy index timestamp. Construct a
        // non-racy fixture explicitly before asserting clean digest reuse.
        std::thread::sleep(std::time::Duration::from_millis(10));
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["update-index", "--refresh"])
            .output()?;
        if !output.status.success() {
            return Err(std::io::Error::other("Git stat refresh failed").into());
        }
        Ok(())
    }

    fn make_git_repo(root: &Path) -> Result<(), Box<dyn StdError>> {
        let script = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/fixtures/make-git-repo.sh"
        );
        let status = Command::new("bash").arg(script).arg(root).status()?;
        if status.success() {
            return Ok(());
        }
        Err(std::io::Error::other("Git fixture script failed").into())
    }

    #[test]
    fn classifies_every_planned_language_and_kind() {
        let cases = [
            ("lib.rs", FileKind::Code, Some("rust")),
            ("lib.py", FileKind::Code, Some("python")),
            ("lib.js", FileKind::Code, Some("javascript")),
            ("lib.ts", FileKind::Code, Some("typescript")),
            ("lib.tsx", FileKind::Code, Some("tsx")),
            ("lib.go", FileKind::Code, Some("go")),
            ("lib.java", FileKind::Code, Some("java")),
            ("lib.kt", FileKind::Code, Some("kotlin")),
            ("lib.c", FileKind::Code, Some("c")),
            ("lib.cpp", FileKind::Code, Some("cpp")),
            ("lib.cs", FileKind::Code, Some("csharp")),
            ("lib.php", FileKind::Code, Some("php")),
            ("lib.rb", FileKind::Code, Some("ruby")),
            ("lib.swift", FileKind::Code, Some("swift")),
            ("lib.scala", FileKind::Code, Some("scala")),
            ("lib.sh", FileKind::Code, Some("bash")),
            ("lib.html", FileKind::Code, Some("html")),
            ("lib.css", FileKind::Code, Some("css")),
            ("lib.sql", FileKind::Code, Some("sql")),
            ("lib.dart", FileKind::Code, Some("dart")),
            ("lib.ex", FileKind::Code, Some("elixir")),
            ("lib.lua", FileKind::Code, Some("lua")),
            ("lib.zig", FileKind::Code, Some("zig")),
            ("lib.hs", FileKind::Code, Some("haskell")),
            ("lib.ml", FileKind::Code, Some("ocaml")),
            ("schema.proto", FileKind::Code, Some("protobuf")),
            ("data.json", FileKind::Config, Some("json")),
            ("data.yaml", FileKind::Config, Some("yaml")),
            ("data.toml", FileKind::Config, Some("toml")),
            ("README.md", FileKind::Doc, Some("markdown")),
            ("Dockerfile", FileKind::Code, Some("dockerfile")),
            (".env.local", FileKind::Config, Some("env")),
            (".envrc", FileKind::Config, Some("env")),
            ("diagram.png", FileKind::Image, None),
        ];
        for (path, kind, lang) in cases {
            assert_eq!(
                classify_path(path),
                (kind, lang),
                "wrong classification for {path}"
            );
        }
    }

    #[test]
    fn honors_nested_gitignore_and_deepkitignore() -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        write(root.path(), ".git/info/exclude", b"exclude-ignored.rs\n")?;
        write(root.path(), ".gitignore", b"root-ignored.rs\n")?;
        write(root.path(), ".deepkitignore", b"deepkit-ignored.rs\n")?;
        write(root.path(), "root-ignored.rs", b"fn ignored() {}")?;
        write(root.path(), "deepkit-ignored.rs", b"fn ignored() {}")?;
        write(root.path(), "exclude-ignored.rs", b"fn ignored() {}")?;
        write(root.path(), "nested/.gitignore", b"nested-ignored.rs\n")?;
        write(root.path(), "nested/nested-ignored.rs", b"fn ignored() {}")?;
        write(root.path(), "nested/kept.rs", b"fn kept() {}")?;
        write(root.path(), ".env.local", b"MODE=test")?;

        let result = scan(root.path(), SyncLimits::default())?;
        assert_eq!(
            result
                .files
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            vec![".env.local", "nested/kept.rs"]
        );
        Ok(())
    }

    #[test]
    fn targeted_eligibility_matches_discovery_and_reloads_ignore_rules()
    -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        for path in [
            "kept.rs",
            ".hidden.rs",
            "nested/kept.rs",
            "nested/ignored.rs",
            "blocked/keep.rs",
            "info.rs",
            "custom.rs",
            ".deepkit/private.rs",
        ] {
            write(root.path(), path, b"fn source() {}")?;
        }
        write(root.path(), ".git/info/exclude", b"info.rs\n")?;
        write(root.path(), ".gitignore", b"blocked/\n!blocked/keep.rs\n")?;
        write(root.path(), "nested/.ignore", b"ignored.rs\n")?;
        write(root.path(), ".deepkitignore", b"custom.rs\n")?;
        let scan = scan(root.path(), SyncLimits::default())?;
        assert_eq!(
            scan.files()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            [".hidden.rs", "kept.rs", "nested/kept.rs"]
        );
        for path in [
            "kept.rs",
            ".hidden.rs",
            "nested/kept.rs",
            "nested/ignored.rs",
            "blocked/keep.rs",
            "info.rs",
            "custom.rs",
            ".deepkit/private.rs",
            ".gitignore",
            "missing.rs",
            "nested",
        ] {
            assert_eq!(
                target_eligible(root.path(), path, l1::Work::default())?,
                scan.file(path).is_some(),
                "{path}"
            );
        }
        // A fresh check must observe both exclusion and re-inclusion, without an index run.
        write(root.path(), "nested/.deepkitignore", b"kept.rs\n")?;
        assert!(!target_eligible(
            root.path(),
            "nested/kept.rs",
            l1::Work::default()
        )?);
        write(root.path(), "nested/.deepkitignore", b"*.rs\n!kept.rs\n")?;
        assert!(target_eligible(
            root.path(),
            "nested/kept.rs",
            l1::Work::default()
        )?);
        Ok(())
    }

    #[test]
    fn targeted_eligibility_refuses_ignore_errors_without_descending_unrelated_trees()
    -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        write(root.path(), "nested/source.rs", b"fn source() {}")?;
        write(root.path(), "unrelated/.gitignore", b"[z-a]\n")?;
        write(root.path(), "unrelated/source.rs", b"fn unrelated() {}")?;
        assert!(target_eligible(
            root.path(),
            "nested/source.rs",
            l1::Work::default()
        )?);
        write(root.path(), "nested/.gitignore", b"[z-a]\n")?;
        assert!(matches!(
            target_eligible(root.path(), "nested/source.rs", l1::Work::default()),
            Err(Error::Walk { .. })
        ));
        fs::remove_file(root.path().join("nested/.gitignore"))?;
        assert!(target_eligible(
            root.path(),
            "nested/source.rs",
            l1::Work::default()
        )?);
        Ok(())
    }

    #[test]
    fn targeted_eligibility_refuses_invalid_paths_and_stopped_work() -> Result<(), Box<dyn StdError>>
    {
        let root = tempfile::tempdir()?;
        write(root.path(), "source.rs", b"fn source() {}")?;
        for path in [
            "",
            "../source.rs",
            "/source.rs",
            "./source.rs",
            "nested/../source.rs",
        ] {
            assert!(
                !target_eligible(root.path(), path, l1::Work::default())?,
                "{path}"
            );
        }
        let cancelled = AtomicBool::new(true);
        for work in [
            l1::Work {
                deadline: Some(std::time::Instant::now()),
                cancelled: None,
            },
            l1::Work {
                deadline: None,
                cancelled: Some(&cancelled),
            },
        ] {
            assert!(matches!(
                target_eligible(root.path(), "source.rs", work),
                Err(Error::WorkStopped)
            ));
        }
        cancelled.store(false, Ordering::Release);
        assert!(target_eligible(
            root.path(),
            "source.rs",
            l1::Work {
                deadline: None,
                cancelled: Some(&cancelled)
            }
        )?);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn targeted_eligibility_does_not_follow_source_or_directory_symlinks()
    -> Result<(), Box<dyn StdError>> {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir()?;
        let external = tempfile::tempdir()?;
        write(external.path(), "source.rs", b"fn external() {}")?;
        symlink(
            external.path().join("source.rs"),
            root.path().join("source.rs"),
        )?;
        symlink(external.path(), root.path().join("nested"))?;
        for path in ["source.rs", "nested/source.rs"] {
            assert!(!target_eligible(root.path(), path, l1::Work::default())?);
        }
        Ok(())
    }

    #[test]
    fn caps_files_without_deleting_an_existing_snapshot() -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        for index in 0..4 {
            write(root.path(), &format!("src/{index}.rs"), b"ok")?;
        }
        write(root.path(), "large.rs", b"12345678")?;
        let database = root.path().join(".deepkit/code.db");
        let store = store::open(&database, store::StoreLimits::default())?;
        let baseline = scan(root.path(), SyncLimits::default())?;
        apply(&store, &baseline)?;
        let baseline_count = store.index_snapshot()?.files.len();

        let capped = scan(
            root.path(),
            SyncLimits {
                max_files: 2,
                max_file_bytes: 4,
                threads: 1,
            },
        )?;
        assert!(capped.status.max_files_reached);
        assert_eq!(capped.status.messages, vec![MAX_FILES_MESSAGE.to_owned()]);
        assert_eq!(capped.files.len(), 2);
        let changes = apply(&store, &capped)?;
        assert!(changes.deleted.is_empty());
        assert_eq!(store.index_snapshot()?.files.len(), baseline_count);

        let uncapped = scan(
            root.path(),
            SyncLimits {
                max_files: 20,
                max_file_bytes: 4,
                threads: 1,
            },
        )?;
        let large = file_map(&uncapped)
            .get("large.rs")
            .copied()
            .ok_or("large file missing")?;
        assert!(large.content_capped);
        assert_eq!(uncapped.status.files_over_max_bytes, 1);
        Ok(())
    }

    #[test]
    fn binary_detection_keeps_known_documents_and_images() -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        write(root.path(), "source.rs", b"fn\0not_source")?;
        write(root.path(), "manual.pdf", b"%PDF\0")?;
        write(root.path(), "image.png", b"PNG\0")?;
        let result = scan(root.path(), SyncLimits::default())?;
        let files = file_map(&result);
        assert_eq!(files["source.rs"].kind, FileKind::Other);
        assert_eq!(files["manual.pdf"].kind, FileKind::Doc);
        assert_eq!(files["image.png"].kind, FileKind::Image);
        Ok(())
    }

    #[test]
    fn streamed_hash_matches_blake3_past_the_threshold() -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        let bytes = (0..(super::STREAM_HASH_THRESHOLD + 1))
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        write(root.path(), "large.rs", &bytes)?;
        let result = scan(root.path(), SyncLimits::default())?;
        let file = file_map(&result)
            .get("large.rs")
            .copied()
            .ok_or("large file missing")?;
        assert_eq!(file.blake3, *blake3::hash(&bytes).as_bytes());
        Ok(())
    }

    #[test]
    fn rename_preserves_the_existing_file_id() -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        write(root.path(), "src/old.rs", b"fn unchanged() {}")?;
        let database = root.path().join(".deepkit/code.db");
        let store = store::open(&database, store::StoreLimits::default())?;
        let first = scan(root.path(), SyncLimits::default())?;
        apply(&store, &first)?;
        let old_id = store.index_snapshot()?.files[0].id;

        fs::rename(
            root.path().join("src/old.rs"),
            root.path().join("src/new.rs"),
        )?;
        let second = scan(root.path(), SyncLimits::default())?;
        let changes = apply(&store, &second)?;
        assert_eq!(changes.renamed.len(), 1);
        assert_eq!(changes.renamed[0].from, "src/old.rs");
        assert_eq!(changes.renamed[0].to, "src/new.rs");
        let snapshot = store.index_snapshot()?;
        assert_eq!(snapshot.files.len(), 1);
        assert_eq!(snapshot.files[0].path, "src/new.rs");
        assert_eq!(snapshot.files[0].id, old_id);
        Ok(())
    }

    #[test]
    fn apply_persists_the_router_treatment_and_ceiling() -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        write(root.path(), "private.rs", b"fn private() {}")?;
        let store = store::open(
            root.path().join(".deepkit/code.db"),
            store::StoreLimits::default(),
        )?;
        let router = Router::new(
            &[Treatment {
                name: "private".to_owned(),
                patterns: vec!["**/*.rs".to_owned()],
                chunker: Some("ast".to_owned()),
                max_level: 3,
                enrich: vec!["describe_symbol".to_owned()],
                never_send: true,
            }],
            false,
        )?;
        let result = scan(root.path(), SyncLimits::default())?;
        apply_with_router(&store, &result, &router)?;
        let file = store
            .index_snapshot()?
            .files
            .into_iter()
            .find(|file| file.path == "private.rs")
            .ok_or_else(|| std::io::Error::other("private file was not persisted"))?;
        assert_eq!(file.treatment, "private");
        assert_eq!(file.max_level, 1);
        Ok(())
    }

    #[test]
    fn dirty_git_bytes_with_restored_mtime_never_inherit_a_clean_oid()
    -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        make_git_repo(root.path())?;
        let path = root.path().join("src/lib.rs");
        let original = fs::read_to_string(&path)?;
        let modified = original.replace("main", "dirt");
        assert_ne!(original, modified);
        assert_eq!(original.len(), modified.len());
        let mtime = fs::metadata(&path)?.modified()?;
        let store = store::open(root.path().join(".deepkit/code.db"), Default::default())?;
        refresh_git_stat(root.path())?;
        let first_git = inspect_git(root.path(), None)?.ok_or("Git fixture missing")?;
        let first = scan(root.path(), Default::default())?;
        apply_with_git(&store, &first, &first_git)?;
        let head = first_git.head.as_ref().ok_or("HEAD missing")?.id.clone();
        let before = store.index_snapshot()?;
        assert!(
            before
                .files
                .iter()
                .find(|f| f.path == "src/lib.rs")
                .ok_or("file missing")?
                .git_blob
                .is_some()
        );
        fs::write(&path, &modified)?;
        fs::File::options()
            .write(true)
            .open(&path)?
            .set_times(fs::FileTimes::new().set_modified(mtime))?;
        let git = inspect_git(root.path(), Some(&head))?.ok_or("Git missing")?;
        let dirty = scan_with_git(root.path(), Default::default(), &git, &before)?;
        let file = dirty.file("src/lib.rs").ok_or("dirty file missing")?;
        assert!(!file.hash_reused_from_git);
        assert_eq!(file.blake3, *blake3::hash(modified.as_bytes()).as_bytes());
        apply_with_git(&store, &dirty, &git)?;
        let persisted = store.index_snapshot()?;
        assert!(
            persisted
                .files
                .iter()
                .find(|f| f.path == "src/lib.rs")
                .ok_or("file missing")?
                .git_blob
                .is_none()
        );
        let again = scan_with_git(root.path(), Default::default(), &git, &persisted)?;
        assert!(
            !again
                .file("src/lib.rs")
                .ok_or("file missing")?
                .hash_reused_from_git
        );
        Ok(())
    }

    #[test]
    fn git_scan_persists_blobs_reuses_hashes_and_flags_branch_switch()
    -> Result<(), Box<dyn StdError>> {
        let root = tempfile::tempdir()?;
        make_git_repo(root.path())?;
        refresh_git_stat(root.path())?;
        write(root.path(), "untracked.rs", b"pub fn untracked() {}")?;
        let database = root.path().join(".deepkit/code.db");
        let store = store::open(&database, store::StoreLimits::default())?;

        let indexed_head = store.get_meta("indexed_head")?;
        let initial_git = inspect_git(root.path(), indexed_head.as_deref())?
            .ok_or_else(|| std::io::Error::other("fixture should be a Git repository"))?;
        assert_eq!(
            initial_git
                .head
                .as_ref()
                .and_then(|head| head.reference.as_deref()),
            Some("refs/heads/main")
        );
        assert!(initial_git.needs_full_rescan);

        let first = scan_with_git(
            root.path(),
            SyncLimits::default(),
            &initial_git,
            &store.index_snapshot()?,
        )?;
        assert!(file_map(&first).contains_key("untracked.rs"));
        assert_eq!(first.status.hashes_reused_from_git, 0);
        apply_with_git(&store, &first, &initial_git)?;
        assert_eq!(store.get_meta("indexed_head")?, Some(String::new()));
        // Catalogue application cannot claim L1 completion. Stand in for a
        // completed L1 run here to test subsequent Git branch-change detection.
        store.set_meta(
            "indexed_head",
            initial_git
                .head
                .as_ref()
                .map_or("", |head| head.id.as_str()),
        )?;

        let snapshot = store.index_snapshot()?;
        for file in &snapshot.files {
            assert_eq!(
                file.git_blob.as_deref(),
                initial_git.tracked_blobs.get(&file.path).map(Vec::as_slice),
                "wrong Git blob for {}",
                file.path
            );
        }
        assert!(
            snapshot
                .files
                .iter()
                .any(|file| file.path == "untracked.rs" && file.git_blob.is_none())
        );

        let indexed_head = store.get_meta("indexed_head")?;
        let second_git = inspect_git(root.path(), indexed_head.as_deref())?
            .ok_or_else(|| std::io::Error::other("fixture should remain a Git repository"))?;
        let second = scan_with_git(root.path(), SyncLimits::default(), &second_git, &snapshot)?;
        assert_eq!(
            second.status.hashes_reused_from_git,
            second_git.tracked_blobs.len()
        );
        assert_eq!(
            apply_with_git(&store, &second, &second_git)?,
            super::ChangeSet::default()
        );

        // An equal index/source timestamp is explicitly racy even when every
        // cached stat field matches. It must not authorize digest reuse.
        let source_mtime = fs::metadata(root.path().join("src/lib.rs"))?.modified()?;
        fs::File::options()
            .write(true)
            .open(root.path().join(".git/index"))?
            .set_times(fs::FileTimes::new().set_modified(source_mtime))?;
        let racy = inspect_git(root.path(), indexed_head.as_deref())?.ok_or("Git missing")?;
        assert!(!racy.clean_stats.contains_key("src/lib.rs"));
        let fallback = scan_with_git(root.path(), SyncLimits::default(), &racy, &snapshot)?;
        assert!(
            !fallback
                .file("src/lib.rs")
                .ok_or("file missing")?
                .hash_reused_from_git
        );

        let status = Command::new("git")
            .arg("-C")
            .arg(root.path())
            .args(["checkout", "--quiet", "alternate"])
            .status()?;
        if !status.success() {
            return Err(std::io::Error::other("failed to switch Git fixture branch").into());
        }
        let indexed_head = store.get_meta("indexed_head")?;
        let switched_git = inspect_git(root.path(), indexed_head.as_deref())?
            .ok_or_else(|| std::io::Error::other("fixture should remain a Git repository"))?;
        assert_eq!(
            switched_git
                .head
                .as_ref()
                .and_then(|head| head.reference.as_deref()),
            Some("refs/heads/alternate")
        );
        assert!(switched_git.needs_full_rescan);
        Ok(())
    }

    proptest! {
        #[test]
        fn merkle_change_set_matches_a_full_rescan_after_random_edits(
            contents in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..128), 1..12),
            removed in prop::collection::vec(any::<bool>(), 1..12),
            modified in prop::collection::vec(any::<bool>(), 1..12),
        ) {
            let root = tempfile::tempdir().map_err(|error| TestCaseError::fail(error.to_string()))?;
            let database = root.path().join(".deepkit/code.db");
            let store = store::open(&database, store::StoreLimits::default())
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            for (index, content) in contents.iter().enumerate() {
                write(root.path(), &format!("tree-{}/src-{index}.rs", index % 3), content)
                    .map_err(|error| TestCaseError::fail(error.to_string()))?;
            }
            let first = scan(root.path(), SyncLimits::default())
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            apply(&store, &first).map_err(|error| TestCaseError::fail(error.to_string()))?;
            let prior = store.index_snapshot().map_err(|error| TestCaseError::fail(error.to_string()))?;

            let expected_added = Vec::<String>::new();
            let mut expected_modified = Vec::<String>::new();
            let mut expected_deleted = Vec::<String>::new();
            for (index, content) in contents.iter().enumerate() {
                let path = format!("tree-{}/src-{index}.rs", index % 3);
                if removed[index % removed.len()] {
                    fs::remove_file(root.path().join(&path))
                        .map_err(|error| TestCaseError::fail(error.to_string()))?;
                    expected_deleted.push(path);
                } else if modified[index % modified.len()] {
                    let mut changed = content.clone();
                    changed.push(index as u8);
                    write(root.path(), &path, &changed)
                        .map_err(|error| TestCaseError::fail(error.to_string()))?;
                    expected_modified.push(path);
                }
            }
            expected_modified.sort();
            expected_deleted.sort();
            let next = scan(root.path(), SyncLimits::default())
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            let changes = change_set(&prior, &next.files, true);
            prop_assert_eq!(changes.added, expected_added);
            prop_assert_eq!(changes.modified, expected_modified);
            prop_assert_eq!(changes.deleted, expected_deleted);
            prop_assert!(changes.renamed.is_empty());

            let root_merkle = next.dirs.iter().find(|directory| directory.path.is_empty())
                .ok_or_else(|| TestCaseError::fail("root Merkle missing"))?;
            prop_assert_eq!(root_merkle.file_count, next.files.len() as i64);
        }
    }
}
