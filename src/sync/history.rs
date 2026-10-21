//! Bounded Git history and historical hunk/symbol intersections; v1.md §4.3 and §6.2.
pub(crate) mod hunks;
use super::l1::Work;
use crate::{
    config::Config,
    store::{self, FtsColumns, Store, UnitInsert, UnitKind},
    treat::{
        self,
        fts::{FtsInput, LexicalColumns, build_columns},
    },
};
use std::{
    collections::BTreeSet,
    ops::{ControlFlow, Range},
    path::Path,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error(transparent)]
    Sync(#[from] super::Error),
    #[error(transparent)]
    Config(#[from] crate::config::Error),
    #[error(transparent)]
    Store(#[from] store::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("Git history could not {0}")]
    Git(&'static str),
    #[error("Git history reached the {0} input bound")]
    Bound(&'static str),
    #[error("failed to start Git history preparation worker: {0}")]
    PreparationThread(std::io::Error),
    #[error("Git history preparation worker panicked")]
    PreparationWorkerPanicked,
}
fn git<T, E>(value: Result<T, E>, operation: &'static str) -> Result<T, Error> {
    value.map_err(|_| Error::Git(operation))
}

/// Persistent progress is explicit even when a caller's deadline interrupts ingestion.
#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(crate) struct Report {
    pub(crate) state: String,
    pub(crate) head: Option<String>,
    pub(crate) visited: u64,
    pub(crate) inserted: u64,
    pub(crate) existing: u64,
    pub(crate) retained: i64,
    pub(crate) max_count_reached: bool,
    pub(crate) max_commits_reached: bool,
    pub(crate) incomplete_diffs: u64,
    pub(crate) input_caps: std::collections::BTreeMap<String, u64>,
}

/// Per-file counts remain unknown for binary or over-limit blobs, never fabricated as zero.
#[derive(Debug)]
pub(crate) struct FileDiff {
    pub(crate) path: String,
    pub(crate) additions: Option<i64>,
    pub(crate) deletions: Option<i64>,
}

/// Immutable historical source data, also usable by subsequent PR reconstruction.
pub(crate) struct Commit {
    pub(crate) id: gix::ObjectId,
    pub(crate) parents: Vec<gix::ObjectId>,
    pub(crate) author: String,
    pub(crate) time: i64,
    pub(crate) date: String,
    pub(crate) subject: String,
    pub(crate) body: String,
    pub(crate) files: Vec<FileDiff>,
    pub(crate) symbols: BTreeSet<String>,
}

const PREPARATION_BATCH: usize = 64;
const PUBLICATION_BATCH: usize = 64;

struct PendingCommit {
    sha: Vec<u8>,
    author: String,
    author_time: i64,
    subject: String,
    body: String,
    parents: i64,
    files_json: String,
    symbols_json: String,
    additions: Option<i64>,
    deletions: Option<i64>,
    hash: [u8; 32],
    source_key: String,
    breadcrumb: String,
    text: String,
    tokens: i64,
    columns: LexicalColumns,
    ladder_level: i64,
    now: i64,
}

impl PendingCommit {
    fn as_store_commit(&self) -> store::history::Commit<'_> {
        store::history::Commit {
            sha: &self.sha,
            author: &self.author,
            author_time: self.author_time,
            subject: &self.subject,
            body: &self.body,
            parents: self.parents,
            files_json: &self.files_json,
            symbols_json: &self.symbols_json,
            additions: self.additions,
            deletions: self.deletions,
            unit: UnitInsert {
                hash: &self.hash,
                source_key: Some(&self.source_key),
                duplicate_ordinal: 0,
                file_id: None,
                kind: UnitKind::Commit,
                byte_start: None,
                byte_end: None,
                line_start: None,
                line_end: None,
                scope_path: Some(&self.breadcrumb),
                header: &self.subject,
                text: &self.text,
                tokens: self.tokens,
                ladder_level: self.ladder_level,
                parent_unit: None,
                created_at: self.now,
                updated_at: self.now,
                fts: FtsColumns {
                    header: &self.columns.header,
                    idents: &self.columns.idents,
                    body: &self.columns.body,
                    prose: &self.columns.prose,
                },
            },
        }
    }
}

fn publish_pending(
    pending: &mut Vec<PendingCommit>,
    settings: &Config,
    store: &Store,
    report: &mut Report,
    check: &dyn Fn() -> Result<(), store::Error>,
) -> Result<bool, Error> {
    if pending.is_empty() {
        return Ok(false);
    }
    let commits = pending
        .iter()
        .map(PendingCommit::as_store_commit)
        .collect::<Vec<_>>();
    let outcomes = store.insert_commits(&commits, settings.limits.max_commits, check)?;
    let mut capacity = false;
    for outcome in outcomes {
        match outcome {
            store::history::Outcome::Inserted => {
                report.inserted += 1;
                report.retained += 1;
            }
            store::history::Outcome::Existing => report.existing += 1,
            store::history::Outcome::Capacity => {
                report.max_commits_reached = true;
                capacity = true;
                break;
            }
        }
    }
    pending.clear();
    Ok(capacity)
}

/// Bounded memoization of immutable blob definitions across adjacent commits.
/// Entries depend on blob and path (synthetic module names include the path).
#[derive(Default)]
pub(crate) struct SymbolCache {
    entries: std::collections::BTreeMap<
        (gix::ObjectId, String),
        std::sync::Arc<Vec<store::history::SymbolRange>>,
    >,
    order: std::collections::VecDeque<((gix::ObjectId, String), usize)>,
    bytes: usize,
}
impl SymbolCache {
    fn insert(
        &mut self,
        key: (gix::ObjectId, String),
        symbols: std::sync::Arc<Vec<store::history::SymbolRange>>,
    ) {
        let size = key.1.len() * 2
            + symbols.capacity() * std::mem::size_of::<store::history::SymbolRange>()
            + symbols
                .iter()
                .map(|s| s.qualified.capacity())
                .sum::<usize>();
        const MAX_BYTES: usize = 8 * 1024 * 1024;
        if size > MAX_BYTES {
            return;
        }
        while self.entries.len() >= 128 || self.bytes + size > MAX_BYTES {
            let Some((old, size)) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&old);
            self.bytes -= size;
        }
        self.bytes += size;
        self.order.push_back((key.clone(), size));
        self.entries.insert(key, symbols);
    }
}

fn preparation_width(settings: &Config, total: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let requested = if settings.limits.index_threads == 0 {
        available.saturating_sub(1).max(1)
    } else {
        usize::try_from(settings.limits.index_threads).unwrap_or(usize::MAX)
    };
    requested.clamp(1, 8).min(total.max(1))
}

/// Parses a bounded newest-first window in parallel while keeping contiguous
/// commits on each worker so immutable before/after blobs still reuse one cache.
/// Results retain walk order; only the caller publishes them.
fn prepare_commits(
    root: &Path,
    ids: &[gix::ObjectId],
    settings: &Config,
    store: &Store,
    work: Work<'_>,
) -> Result<Vec<Result<Commit, Error>>, Error> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let width = preparation_width(settings, ids.len());
    if width == 1 {
        let repo = super::head::discover(root)?.ok_or(Error::Git("open worker repository"))?;
        let mut cache = SymbolCache::default();
        return Ok(ids
            .iter()
            .map(|id| read_commit(&repo, *id, settings, store, work, &mut cache))
            .collect());
    }
    let chunk_size = ids.len().div_ceil(width);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(width);
        for chunk in ids.chunks(chunk_size) {
            let handle = std::thread::Builder::new()
                .name("dkc-history".into())
                .spawn_scoped(scope, move || {
                    let repo =
                        super::head::discover(root)?.ok_or(Error::Git("open worker repository"))?;
                    let mut cache = SymbolCache::default();
                    Ok::<_, Error>(
                        chunk
                            .iter()
                            .map(|id| read_commit(&repo, *id, settings, store, work, &mut cache))
                            .collect::<Vec<_>>(),
                    )
                })
                .map_err(Error::PreparationThread)?;
            handles.push(handle);
        }
        let mut prepared = Vec::with_capacity(ids.len());
        for handle in handles {
            prepared.extend(
                handle
                    .join()
                    .map_err(|_| Error::PreparationWorkerPanicked)??,
            );
        }
        Ok(prepared)
    })
}

fn history_ceiling<'a>(paths: impl Iterator<Item = &'a str>, router: &treat::Router) -> i64 {
    let mut seen = false;
    let mut maximum = 0;
    for path in paths {
        seen = true;
        let (kind, _) = super::classify_path(path);
        maximum = maximum.max(router.route(path, kind).max_level);
    }
    if seen { maximum } else { 3 }
}

fn pending_commit(
    commit: Commit,
    settings: &Config,
    router: &treat::Router,
) -> Result<(PendingCommit, u64), Error> {
    let ladder_level =
        i64::from(history_ceiling(commit.files.iter().map(|file| file.path.as_str()), router) >= 1);
    let text = format!(
        "{}\nfiles:\n{}\nsymbols:\n{}",
        commit.body,
        commit
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        commit
            .symbols
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    );
    let breadcrumb = format!("commit › {} › {}", commit.date, commit.author);
    let hash = treat::context::unit_hash("commit", None, &text);
    let source_key = format!("commit:{}", commit.id);
    let columns = build_columns(
        FtsInput {
            breadcrumb: &breadcrumb,
            header: &commit.subject,
            text: &text,
            language: None,
            kind: store::FileKind::Doc,
            extra_idents: &[],
            has_code_fence: false,
        },
        &settings.tokenize,
    );
    let files_json = serde_json::to_string(
        &commit
            .files
            .iter()
            .map(|file| &file.path)
            .collect::<Vec<_>>(),
    )?;
    let symbols_json = serde_json::to_string(&commit.symbols)?;
    let additions = commit.files.iter().map(|file| file.additions).sum();
    let deletions = commit.files.iter().map(|file| file.deletions).sum();
    let incomplete_diffs = commit
        .files
        .iter()
        .filter(|file| file.additions.is_none())
        .count() as u64;
    let now = super::unix_timestamp();
    let tokens = treat::count_tokens(&text) as i64;
    Ok((
        PendingCommit {
            sha: commit.id.as_slice().to_vec(),
            author: commit.author,
            author_time: commit.time,
            subject: commit.subject,
            body: commit.body,
            parents: commit.parents.len() as i64,
            files_json,
            symbols_json,
            additions,
            deletions,
            hash,
            source_key,
            breadcrumb,
            text,
            tokens,
            columns,
            ladder_level,
            now,
        },
        incomplete_diffs,
    ))
}

/// Visits a bounded newest-first window, including all parents; previously stored
/// identities are skipped before tree diff or blob parsing. Every publication is fenced.
pub(crate) fn ingest(
    root: &Path,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    validate: &dyn Fn() -> Result<(), store::Error>,
) -> Result<Report, Error> {
    work.check()?;
    let Some(repo) = super::head::discover(root)? else {
        return Ok(Report {
            state: "non_git".into(),
            ..Default::default()
        });
    };
    let Some(head) = super::head::resolve(&repo, root)? else {
        return Ok(Report {
            state: "unborn".into(),
            ..Default::default()
        });
    };
    let mut report = Report {
        state: "disabled".into(),
        head: Some(head.id.clone()),
        ..Default::default()
    };
    if !settings.commits.enabled {
        return Ok(report);
    }
    let key = format!(
        "{}:{}:{}:{}:{}",
        head.id,
        settings.commits.max_count,
        settings.limits.max_commits,
        settings.limits.max_file_bytes,
        settings.limits.max_files
    );
    let check = || {
        work.check().map_err(|_| store::Error::L1InputsChanged)?;
        validate()?;
        if super::head::read(root, work)
            .map_err(|_| store::Error::L1InputsChanged)?
            .map(|h| h.id)
            .as_deref()
            != Some(head.id.as_str())
        {
            return Err(store::Error::L1InputsChanged);
        }
        Ok(())
    };
    check()?;
    let router = treat::Router::new(&settings.treatments, settings.llm.enabled)
        .map_err(super::Error::from)?;
    let commit_ceiling = |files_json: &str| {
        let paths: Vec<String> = serde_json::from_str(files_json)
            .map_err(|_| store::Error::InvalidHistoryFileProvenance)?;
        Ok(history_ceiling(paths.iter().map(String::as_str), &router))
    };
    store.refresh_history_fts(settings, &commit_ceiling, &check)?;
    let (completed, previous) = store.history_checkpoint(&check)?;
    if completed.as_deref() == Some(key.as_str()) {
        if let Some(previous) = previous {
            report = serde_json::from_str(&previous)?;
        }
        report.state = if report.max_commits_reached || !report.input_caps.is_empty() {
            "capped"
        } else {
            "not_needed"
        }
        .into();
        report.visited = 0;
        report.inserted = 0;
        report.existing = 0;
        return Ok(report);
    }
    report.retained = store.commit_count(&check)?;
    report.max_commits_reached = report.retained as u64 >= settings.limits.max_commits;
    report.state = "pending".into();
    store.history_progress(&serde_json::to_string(&report)?, None, &check)?;
    let id = git(gix::ObjectId::from_hex(head.id.as_bytes()), "decode HEAD")?;
    let mut walk = walk(&repo, id, false)?;
    let mut pending = Vec::with_capacity(PUBLICATION_BATCH);
    let mut finished = false;
    while !finished {
        work.check()?;
        if report.visited >= settings.commits.max_count {
            report.max_count_reached = walk.next().is_some();
            break;
        }
        let remaining = settings
            .limits
            .max_commits
            .saturating_sub(report.retained as u64);
        if remaining == 0 {
            report.max_commits_reached = true;
            break;
        }
        // Establish one durable recovery point before speculative parallel
        // preparation. Every later window stays bounded to one publication batch.
        let batch_limit = if report.retained == 0 {
            1
        } else {
            PREPARATION_BATCH.min(usize::try_from(remaining).unwrap_or(usize::MAX))
        };
        let mut ids = Vec::with_capacity(batch_limit);
        while ids.len() < batch_limit {
            work.check()?;
            if report.visited >= settings.commits.max_count {
                report.max_count_reached = walk.next().is_some();
                finished = true;
                break;
            }
            let Some(item) = walk.next() else {
                finished = true;
                break;
            };
            let item = git(item, "walk commit")?;
            work.check()?;
            report.visited += 1;
            if store.has_commit(item.id.as_slice(), &check)? {
                report.existing += 1;
            } else {
                ids.push(item.id);
            }
        }
        if ids.is_empty() {
            continue;
        }
        for prepared in prepare_commits(root, &ids, settings, store, work)? {
            let commit = match prepared {
                Ok(commit) => commit,
                Err(Error::Bound(reason)) => {
                    *report.input_caps.entry(reason.into()).or_default() += 1;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let (commit, incomplete_diffs) = pending_commit(commit, settings, &router)?;
            report.incomplete_diffs += incomplete_diffs;
            pending.push(commit);
            if pending.len() >= PUBLICATION_BATCH
                && publish_pending(&mut pending, settings, store, &mut report, &check)?
            {
                report.max_commits_reached = true;
                finished = true;
                break;
            }
        }
        if !pending.is_empty()
            && publish_pending(&mut pending, settings, store, &mut report, &check)?
        {
            report.max_commits_reached = true;
            break;
        }
    }
    if !report.max_commits_reached
        && publish_pending(&mut pending, settings, store, &mut report, &check)?
    {
        report.max_commits_reached = true;
    }
    check()?;
    report.max_commits_reached |= report.retained as u64 >= settings.limits.max_commits;
    report.state = if report.max_commits_reached || !report.input_caps.is_empty() {
        "capped"
    } else {
        "completed"
    }
    .into();
    store.history_progress(&serde_json::to_string(&report)?, Some(&key), &check)?;
    Ok(report)
}

/// Streams newest-first ancestry; PR membership can restrict traversal to the
/// first parent while commit ingestion includes every parent.
pub(crate) fn walk(
    repo: &gix::Repository,
    id: gix::ObjectId,
    first_parent: bool,
) -> Result<gix::revision::Walk<'_>, Error> {
    let walk = repo
        .rev_walk([id])
        .sorting(gix::revision::walk::Sorting::ByCommitTime(
            gix::traverse::commit::simple::CommitTimeOrder::NewestFirst,
        ));
    git(
        if first_parent {
            walk.first_parent_only()
        } else {
            walk
        }
        .all(),
        "start history walk",
    )
}

/// Computes metadata and changes from immutable trees, never from the current checkout.
pub(crate) fn read_commit(
    repo: &gix::Repository,
    id: gix::ObjectId,
    settings: &Config,
    store: &Store,
    work: Work<'_>,
    symbols: &mut SymbolCache,
) -> Result<Commit, Error> {
    work.check()?;
    // A commit message is source input too; respect the configured source-read bound.
    if git(repo.find_header(id), "inspect commit size")?.size()
        > settings.limits.max_file_bytes()?.0
    {
        return Err(Error::Bound("max_file_bytes commit"));
    }
    let raw = git(repo.find_commit(id), "read commit")?;
    let decoded = git(raw.decode(), "decode commit")?;
    let author = git(decoded.author(), "decode author")?;
    let time = git(author.time(), "decode author date")?;
    let message =
        std::str::from_utf8(decoded.message).map_err(|_| Error::Git("decode UTF-8 message"))?;
    let (subject, body) = message.split_once('\n').unwrap_or((message, ""));
    let mut commit = Commit {
        id,
        parents: raw.parent_ids().map(|id| id.detach()).collect(),
        author: author.name.to_string(),
        time: time.seconds,
        date: git(
            time.format(gix::date::time::format::SHORT),
            "format author date",
        )?,
        subject: subject.trim_end_matches('\r').to_owned(),
        body: body.trim_matches(['\n', '\r']).to_owned(),
        files: Vec::new(),
        symbols: BTreeSet::new(),
    };
    let after = git(raw.tree(), "read commit tree")?;
    let before = commit
        .parents
        .first()
        .map(|id| {
            git(repo.find_commit(*id), "read first parent")
                .and_then(|p| git(p.tree(), "read parent tree"))
        })
        .transpose()?;
    let mut visitor = Changes {
        tracker: Default::default(),
        changes: Vec::new(),
        limit: settings.limits.max_files,
        work,
        refused: false,
    };
    let diff = gix::diff::tree(
        gix::objs::TreeRefIter::from_bytes(
            before.as_ref().map_or(&[], |t| t.data.as_slice()),
            id.kind(),
        ),
        gix::objs::TreeRefIter::from_bytes(&after.data, id.kind()),
        &mut Default::default(),
        &repo.objects,
        &mut visitor,
    );
    work.check()?;
    if visitor.refused {
        return Err(Error::Bound("max_files or path encoding"));
    }
    git(diff, "diff commit trees")?;
    for change in visitor.changes {
        work.check()?;
        if change.gitlink {
            // Submodule object IDs belong to another repository. Their absent
            // blobs are not a corrupt history and do not justify a filesystem read.
            commit.files.push(FileDiff {
                path: change.path,
                additions: None,
                deletions: None,
            });
            continue;
        }
        let before = blob(repo, change.before, settings.limits.max_file_bytes()?.0)?;
        let after = blob(repo, change.after, settings.limits.max_file_bytes()?.0)?;
        let (Some(before), Some(after)) = (before, after) else {
            commit.files.push(FileDiff {
                path: change.path,
                additions: None,
                deletions: None,
            });
            continue;
        };
        let (Ok(before_text), Ok(after_text)) =
            (std::str::from_utf8(&before), std::str::from_utf8(&after))
        else {
            commit.files.push(FileDiff {
                path: change.path,
                additions: None,
                deletions: None,
            });
            continue;
        };
        let input = gix::diff::blob::InternedInput::new(before_text, after_text);
        let diff = gix::diff::blob::diff_with_slider_heuristics(
            gix::diff::blob::Algorithm::Histogram,
            &input,
        );
        let (_, language) = super::classify_path(&change.path);
        if let Some(language) = language {
            let hunks = diff.hunks().collect::<Vec<_>>();
            touched_symbols(
                store,
                &change.path,
                change.after,
                &after,
                language,
                hunks.iter().map(|h| h.after.clone()),
                work,
                &mut commit.symbols,
                symbols,
            )?;
            // Removed definitions have no range in the new tree; retain their names
            // from the removed parent ranges rather than consulting today's symbols.
            touched_symbols(
                store,
                &change.path,
                change.before,
                &before,
                language,
                hunks.iter().map(|h| h.before.clone()),
                work,
                &mut commit.symbols,
                symbols,
            )?;
        }
        commit.files.push(FileDiff {
            path: change.path,
            additions: Some(i64::from(diff.count_additions())),
            deletions: Some(i64::from(diff.count_removals())),
        });
    }
    work.check()?;
    Ok(commit)
}

#[allow(
    clippy::too_many_arguments,
    reason = "historical blob identity, content, hunks and work fence are independent inputs"
)]
fn touched_symbols(
    store: &Store,
    path: &str,
    id: Option<gix::ObjectId>,
    bytes: &[u8],
    language: &str,
    ranges: impl Iterator<Item = Range<u32>>,
    work: Work<'_>,
    out: &mut BTreeSet<String>,
    cache: &mut SymbolCache,
) -> Result<(), Error> {
    let Some(id) = id else {
        return Ok(());
    };
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let key = (id, path.to_owned());
    let symbols = if let Some(symbols) = cache.entries.get(&key) {
        std::sync::Arc::clone(symbols)
    } else {
        let symbols = match store.historical_symbols(path, id.as_slice(), &check)? {
            Some(symbols) => symbols,
            None => git(
                treat::graph::extract_file_symbols(language, path, bytes),
                "parse historical symbols",
            )?
            .unwrap_or_default()
            .into_iter()
            .map(|s| store::history::SymbolRange {
                qualified: s.qualified,
                start: s.line_start as i64,
                end: s.line_end as i64,
            })
            .collect(),
        };
        let symbols = std::sync::Arc::new(symbols);
        cache.insert(key, std::sync::Arc::clone(&symbols));
        symbols
    };
    for range in ranges.filter(|r| !r.is_empty()) {
        for symbol in symbols.iter() {
            if symbol.start <= i64::from(range.end) && symbol.end > i64::from(range.start) {
                out.insert(symbol.qualified.clone());
            }
        }
        work.check()?;
    }
    Ok(())
}
fn blob(
    repo: &gix::Repository,
    id: Option<gix::ObjectId>,
    max: u64,
) -> Result<Option<Vec<u8>>, Error> {
    let Some(id) = id else {
        return Ok(Some(Vec::new()));
    };
    let header = git(repo.find_header(id), "inspect blob")?;
    if header.kind() != gix::objs::Kind::Blob || header.size() > max {
        return Ok(None);
    }
    let data = git(repo.find_blob(id), "read blob")?.data.clone();
    if data.contains(&0) {
        return Ok(None);
    }
    Ok(Some(data))
}
struct Change {
    path: String,
    gitlink: bool,
    before: Option<gix::ObjectId>,
    after: Option<gix::ObjectId>,
}
struct Changes<'a> {
    tracker: gix::diff::tree::Recorder,
    changes: Vec<Change>,
    limit: u64,
    work: Work<'a>,
    refused: bool,
}
impl gix::diff::tree::Visit for Changes<'_> {
    fn pop_front_tracked_path_and_set_current(&mut self) {
        self.tracker.pop_front_tracked_path_and_set_current();
    }
    fn push_back_tracked_path_component(&mut self, c: &gix::bstr::BStr) {
        self.tracker.push_back_tracked_path_component(c);
    }
    fn push_path_component(&mut self, c: &gix::bstr::BStr) {
        self.tracker.push_path_component(c);
    }
    fn pop_path_component(&mut self) {
        self.tracker.pop_path_component();
    }
    fn visit(&mut self, change: gix::diff::tree::visit::Change) -> gix::diff::tree::visit::Action {
        use gix::diff::tree::visit::Change as C;
        if self.work.check().is_err() {
            return ControlFlow::Break(());
        }
        let (mode, before, after, gitlink) = match change {
            C::Addition {
                entry_mode, oid, ..
            } => (entry_mode, None, Some(oid), entry_mode.is_commit()),
            C::Deletion {
                entry_mode, oid, ..
            } => (entry_mode, Some(oid), None, entry_mode.is_commit()),
            C::Modification {
                entry_mode,
                previous_oid,
                previous_entry_mode,
                oid,
                ..
            } => (
                entry_mode,
                Some(previous_oid),
                Some(oid),
                entry_mode.is_commit() || previous_entry_mode.is_commit(),
            ),
        };
        if mode.is_tree() {
            return ControlFlow::Continue(());
        }
        let Ok(path) = std::str::from_utf8(self.tracker.path()) else {
            self.refused = true;
            return ControlFlow::Break(());
        };
        if self.changes.len() as u64 >= self.limit {
            self.refused = true;
            return ControlFlow::Break(());
        }
        self.changes.push(Change {
            path: path.to_owned(),
            gitlink,
            before,
            after,
        });
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, fs, process::Command, sync::atomic::AtomicBool, time::Instant};

    fn command(root: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().into()
    }
    fn fixture() -> (tempfile::TempDir, Store, Config, Vec<String>) {
        let root = tempfile::tempdir().unwrap();
        command(root.path(), &["init", "-q", "-b", "main"]);
        command(root.path(), &["config", "user.name", "History"]);
        command(
            root.path(),
            &["config", "user.email", "history@example.invalid"],
        );
        let mut ids = Vec::new();
        for n in 0..3 {
            fs::write(
                root.path().join("code.rs"),
                format!("pub fn number() -> u8 {{ {n} }}\n"),
            )
            .unwrap();
            command(root.path(), &["add", "code.rs"]);
            command(
                root.path(),
                &["commit", "-q", "-m", &format!("revision {n}")],
            );
            ids.push(command(root.path(), &["rev-parse", "HEAD"]));
        }
        let store =
            store::open(root.path().join("test.db"), store::StoreLimits::default()).unwrap();
        (root, store, Config::default(), ids)
    }

    #[test]
    fn caller_stop_and_changed_head_preserve_only_completed_commits_for_retry() {
        for move_head in [false, true] {
            let (root, store, settings, ids) = fixture();
            let observer = rusqlite::Connection::open(root.path().join("test.db")).unwrap();
            let triggered = Cell::new(false);
            let check = || {
                let count: i64 = observer
                    .query_row("SELECT count(*) FROM commits", [], |r| r.get(0))
                    .unwrap();
                if count == 1 && !triggered.replace(count == 1) {
                    if move_head {
                        command(root.path(), &["checkout", "-q", &ids[0]]);
                    } else {
                        return Err(store::Error::L1InputsChanged);
                    }
                }
                Ok(())
            };
            assert!(ingest(root.path(), &settings, &store, Work::default(), &check).is_err());
            assert!(triggered.get());
            assert_eq!(store.commit_count(&|| Ok(())).unwrap(), 1);
            assert!(store.history_checkpoint(&|| Ok(())).unwrap().0.is_none());
            command(root.path(), &["checkout", "-q", "main"]);
            let done = ingest(root.path(), &settings, &store, Work::default(), &|| Ok(())).unwrap();
            assert_eq!((done.inserted, done.existing, done.retained), (2, 1, 3));
        }
    }

    #[test]
    fn parallel_windows_match_serial_history_order_and_payloads() {
        let (root, _fixture_store, mut settings, _) = fixture();
        for number in 3..71 {
            fs::write(
                root.path().join("code.rs"),
                format!("pub fn number() -> u8 {{ {number} }}\n"),
            )
            .unwrap();
            command(root.path(), &["add", "code.rs"]);
            command(
                root.path(),
                &["commit", "-q", "-m", &format!("revision {number}")],
            );
        }
        settings.commits.max_count = 70;
        settings.limits.max_commits = 70;
        settings.limits.index_threads = 1;
        let serial_path = root.path().join("serial.db");
        let serial = store::open(&serial_path, store::StoreLimits::default()).unwrap();
        let serial_report =
            ingest(root.path(), &settings, &serial, Work::default(), &|| Ok(())).unwrap();
        settings.limits.index_threads = 8;
        let parallel_path = root.path().join("parallel.db");
        let parallel = store::open(&parallel_path, store::StoreLimits::default()).unwrap();
        let parallel_report = ingest(root.path(), &settings, &parallel, Work::default(), &|| {
            Ok(())
        })
        .unwrap();
        assert_eq!(
            (
                serial_report.visited,
                serial_report.inserted,
                serial_report.retained,
                serial_report.max_count_reached,
                serial_report.max_commits_reached,
                serial_report.incomplete_diffs,
            ),
            (
                parallel_report.visited,
                parallel_report.inserted,
                parallel_report.retained,
                parallel_report.max_count_reached,
                parallel_report.max_commits_reached,
                parallel_report.incomplete_diffs,
            )
        );
        assert!(serial_report.max_count_reached);
        assert!(serial_report.max_commits_reached);
        let rows = |path: &Path| {
            let connection = rusqlite::Connection::open(path).unwrap();
            let mut statement = connection
                .prepare("SELECT hex(sha),files_touched,symbols_touched FROM commits ORDER BY id")
                .unwrap();
            statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(rows(&serial_path), rows(&parallel_path));
    }

    #[test]
    fn cancelled_and_expired_work_does_not_start_history() {
        let (root, store, settings, _) = fixture();
        let cancelled = AtomicBool::new(true);
        for work in [
            Work {
                cancelled: Some(&cancelled),
                deadline: None,
            },
            Work {
                cancelled: None,
                deadline: Some(Instant::now()),
            },
        ] {
            assert!(matches!(
                ingest(root.path(), &settings, &store, work, &|| Ok(())),
                Err(Error::Sync(super::super::Error::WorkStopped))
            ));
        }
        assert_eq!(store.commit_count(&|| Ok(())).unwrap(), 0);
        assert!(store.history_checkpoint(&|| Ok(())).unwrap().0.is_none());
    }

    #[test]
    fn first_parent_walk_excludes_side_branch_while_all_parents_includes_it() {
        let (root, _, _, ids) = fixture();
        command(root.path(), &["checkout", "-q", "-b", "side", &ids[0]]);
        fs::write(root.path().join("side.rs"), "pub fn side() {}\n").unwrap();
        command(root.path(), &["add", "side.rs"]);
        command(root.path(), &["commit", "-q", "-m", "side"]);
        let side = command(root.path(), &["rev-parse", "HEAD"]);
        command(root.path(), &["checkout", "-q", "main"]);
        command(root.path(), &["merge", "--no-ff", "-m", "merge", "side"]);
        let repo = gix::discover(root.path()).unwrap();
        let head = repo.head_id().unwrap().detach();
        let first: Vec<_> = walk(&repo, head, true)
            .unwrap()
            .map(|c| c.unwrap().id.to_string())
            .collect();
        let all: Vec<_> = walk(&repo, head, false)
            .unwrap()
            .map(|c| c.unwrap().id.to_string())
            .collect();
        assert_eq!(first.len(), 4);
        assert_eq!(all.len(), 5);
        assert!(!first.contains(&side));
        assert!(all.contains(&side));
        assert_eq!(first[0], head.to_string());
        assert_eq!(all[0], head.to_string());
    }
    #[test]
    fn history_dispatch_obeys_current_policy_and_keeps_protected_local_reads() {
        let (root, store, mut settings, _) = fixture();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        let path = root.path().join(".deepkit/config.toml");
        settings.embedder.profile = "local".into();
        fs::write(&path, settings.dump().unwrap()).unwrap();
        let settings = crate::config::load(root.path(), None).unwrap().config;
        ingest(root.path(), &settings, &store, Work::default(), &|| Ok(())).unwrap();
        let unit: i64 = store
            .writer()
            .unwrap()
            .query_row("SELECT unit_id FROM commits ORDER BY id LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        let embedding = |config: &Config| {
            store
                .read_snapshot(|tx| {
                    crate::queue::source::resolve(tx, root.path(), root.path(), config, unit)
                })
                .unwrap()
                .unwrap()
        };
        assert!(embedding(&settings).is_some());
        for protected in [false, true] {
            let mut changed = settings.clone();
            changed.treatments.insert(
                0,
                crate::config::Treatment {
                    name: "history-policy".into(),
                    patterns: vec!["*.rs".into()],
                    max_level: 1,
                    never_send: protected,
                    ..Default::default()
                },
            );
            fs::write(&path, changed.dump().unwrap()).unwrap();
            let changed = crate::config::load(root.path(), None).unwrap().config;
            assert!(
                embedding(&settings).is_none(),
                "old configuration is not dispatch authority"
            );
            assert!(
                embedding(&changed).is_none(),
                "L1/protected history cannot dispatch"
            );
            let local = store
                .read_snapshot(|tx| {
                    crate::queue::source::resolve_freshness(
                        tx,
                        root.path(),
                        root.path(),
                        &changed,
                        unit,
                        Work::default(),
                    )
                })
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(local.policy.never_send, protected);
            assert_eq!(local.policy.max_level, 1);
        }
    }

    #[test]
    fn tokenizer_refresh_preserves_history_occurrences_and_recovers_lexical_projection() {
        let (root, store, mut settings, _) = fixture();
        ingest(root.path(), &settings, &store, Work::default(), &|| Ok(())).unwrap();
        let snapshot = || {
            store
                .read_snapshot(|tx| {
                    tx.prepare(
                        "SELECT id,hex(hash),updated_at FROM units WHERE kind='commit' ORDER BY id",
                    )?
                    .query_map([], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, i64>(2)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()
                })
                .unwrap()
                .unwrap()
        };
        let before = snapshot();
        let old = store.get_meta("history_fts").unwrap();
        store
            .writer()
            .unwrap()
            .execute(
                "DELETE FROM fts_units WHERE rowid IN (SELECT unit_id FROM commits)",
                [],
            )
            .unwrap();
        settings.tokenize.split_subtokens = !settings.tokenize.split_subtokens;
        let report = ingest(root.path(), &settings, &store, Work::default(), &|| Ok(())).unwrap();
        assert_eq!(report.inserted, 0);
        assert_eq!(snapshot(), before);
        assert_ne!(store.get_meta("history_fts").unwrap(), old);
        assert_eq!(store.query_fts("revision", 10).unwrap().len(), 3);
    }
}
