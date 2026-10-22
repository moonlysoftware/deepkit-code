//! Query-owned targeted L1 repair orchestration; see `v1.md` §4.6 and §9.

use super::service::{Context, Error};
use crate::{config, store, sync};
use std::{fs, path::PathBuf};

/// Repairs one previously selected path under the original request work signal.
/// A true result requires reretrieval and source verification, not a fresh claim:
/// publication may defer at a cap. Targeted work never completes indexed HEAD.
pub(super) fn target(
    context: &Context,
    path: &str,
    work: sync::l1::Work<'_>,
) -> Result<bool, Error> {
    work.check()?;
    if context
        .background_l1
        .load(std::sync::atomic::Ordering::Acquire)
        || !context.config.search.lazy_resync
    {
        return Ok(false);
    }
    validate_configuration(context, work)?;
    if !sync::target_eligible(&context.root, path, work)? {
        return Ok(false);
    }
    // Read only this path and its generation from one coherent snapshot.
    let (generation, snapshot) = context.store.target_snapshot(path, &|| {
        work.check().map_err(|_| store::Error::L1InputsChanged)
    })?;
    work.check()?;
    if !snapshot.files.iter().any(|file| file.path == path) {
        // New files require repository-wide max_files admission.
        return Ok(false);
    }
    let max_bytes = context
        .config
        .limits
        .max_file_bytes()?
        .0
        .min(256 * 1024 - 1);
    let file = sync::l1::observe_source(&context.root, path, max_bytes, work, |_| {})?;
    if file.content_capped {
        return Ok(false);
    }
    let validate = || validate_configuration(context, work);
    let publication = sync::l1::publish_target(
        &context.store,
        sync::l1::TargetInput {
            work,
            file: &file,
            snapshot: &snapshot,
            generation,
            root: &context.root,
            settings: &context.config,
        },
        validate,
    )?;
    work.check()?;
    let fingerprint = publication
        .fingerprints
        .get(path)
        .ok_or(store::Error::L1InputsChanged)?;
    let (file_id, _) = context.store.l1_file_work(
        publication.generation,
        path,
        file.content_hash(),
        fingerprint,
        &|| work.check().map_err(|_| store::Error::L1InputsChanged),
    )?;
    sync::l1::Processor::new(&context.root, &context.config)?.process_target(
        &context.store,
        sync::l1::FileInput {
            work,
            file: &file,
            fence: store::l1::Fence {
                generation: publication.generation,
                file_id,
                path,
                content_hash: file.content_hash(),
                fingerprint,
            },
        },
        validate,
    )?;
    publish_links(context, work)?;
    Ok(true)
}

fn publish_links(context: &Context, work: sync::l1::Work<'_>) -> Result<(), Error> {
    let result = sync::linker::ingest(
        &context.root,
        &context.config,
        &context.store,
        work,
        &|| validate_configuration(context, work),
    );
    work.check()?;
    if result
        .map_err(|_| super::Error::OpenSourceUnavailable)?
        .deferred
    {
        return Err(super::Error::OpenSourceUnavailable.into());
    }
    Ok(())
}

/// Reconciles branch L1 under one request budget. Completion is distinct from
/// per-file progress: capped discovery cannot advance indexed HEAD.
pub(super) fn branch(
    context: &Context,
    work: sync::l1::Work<'_>,
) -> Result<Option<(BranchProgress, bool)>, Error> {
    let Some((prepared, publication)) = branch_catalogue(context, work)? else {
        return Ok(None);
    };
    let progress = process_branch(context, &prepared, &publication, work)?;
    let completed = complete_branch(context, &prepared, &publication, work, || Ok(()))?;
    Ok(Some((progress, completed)))
}

/// Revalidates every captured input before and inside the completion transaction.
/// Failure leaves already committed file work intact and HEAD unadvanced.
fn complete_branch(
    context: &Context,
    prepared: &PreparedBranch,
    publication: &sync::l1::Publication,
    work: sync::l1::Work<'_>,
    boundary: impl Fn() -> Result<(), store::Error>,
) -> Result<bool, Error> {
    work.check()?;
    if prepared.candidates.max_files_reached || prepared.candidates.skipped_non_unicode_paths != 0 {
        return Ok(false);
    }
    let max_files = usize::try_from(context.config.limits.max_files).unwrap_or(usize::MAX);
    let max_bytes = context.config.limits.max_file_bytes()?.0;
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let validate = || {
        check()?;
        boundary()?;
        validate_configuration(context, work)?;
        if prepared.files.len() != prepared.candidates.paths.len()
            || prepared
                .files
                .iter()
                .zip(&prepared.candidates.paths)
                .any(|(file, path)| &file.path != path)
            || sync::discovery::discover(&context.root, max_files, work)
                .map_err(|_| store::Error::L1InputsChanged)?
                != prepared.candidates
        {
            return Err(store::Error::L1InputsChanged);
        }
        for file in &prepared.files {
            check()?;
            sync::l1::verify_source(&context.root, file, max_bytes, work)
                .map_err(|_| store::Error::L1InputsChanged)?;
        }
        validate_configuration(context, work)?;
        if sync::head::read(&context.root, work).map_err(|_| store::Error::L1InputsChanged)?
            != prepared.candidates.head
        {
            return Err(store::Error::L1InputsChanged);
        }
        check()
    };
    validate()?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| Error::Clock)?;
    let timestamp = i64::try_from(timestamp.as_secs()).map_err(|_| Error::Clock)?;
    context.store.complete_l1_scan(
        Some(&check),
        publication.generation,
        prepared
            .candidates
            .head
            .as_ref()
            .map_or("", |head| head.id.as_str()),
        &context.root.to_string_lossy(),
        timestamp,
        validate,
    )?;
    Ok(true)
}

/// Publishes query-owned branch catalogue changes while retaining pending L1.
/// The returned observations and publication must feed fenced Processor outcomes
/// before any caller may complete HEAD. Disabled lazy repair performs no reads
/// beyond the original work check and never publishes.
pub(super) fn branch_catalogue(
    context: &Context,
    work: sync::l1::Work<'_>,
) -> Result<Option<(PreparedBranch, sync::l1::Publication)>, Error> {
    work.check()?;
    if !context.config.search.lazy_resync {
        return Ok(None);
    }
    let prepared = prepare_branch(context, work)?;
    let publication = sync::l1::publish_branch(
        &context.store,
        sync::l1::BranchInput {
            git: prepared.git.as_ref(),
            work,
            candidates: &prepared.candidates,
            files: &prepared.files,
            snapshot: &prepared.snapshot,
            generation: prepared.generation,
            root: &context.root,
            settings: &context.config,
        },
        || validate_configuration(context, work),
    )?;
    work.check()?;
    Ok(Some((prepared, publication)))
}

/// Per-pass progress, not freshness or completed-HEAD authority. Already settled
/// files may include prior cap deferrals; their stored evidence remains authoritative.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct BranchProgress {
    pub(super) processed_files: usize,
    pub(super) already_settled_files: usize,
    pub(super) truncated_files: usize,
    pub(super) content_capped_files: usize,
    pub(super) discovery_incomplete: bool,
}

/// Processes admitted branch sources using their post-publication identities.
/// A later refusal preserves earlier file commits and all remaining pending work.
/// This never completes HEAD: full discovery/source revalidation must follow.
pub(super) fn process_branch(
    context: &Context,
    prepared: &PreparedBranch,
    publication: &sync::l1::Publication,
    work: sync::l1::Work<'_>,
) -> Result<BranchProgress, Error> {
    process_branch_checked(context, prepared, publication, work, |_| Ok(()))
}

fn process_branch_checked(
    context: &Context,
    prepared: &PreparedBranch,
    publication: &sync::l1::Publication,
    work: sync::l1::Work<'_>,
    boundary: impl Fn(&str) -> Result<(), store::Error>,
) -> Result<BranchProgress, Error> {
    let check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let validate = || {
        validate_configuration(context, work)?;
        if sync::head::read(&context.root, work).map_err(|_| store::Error::L1InputsChanged)?
            != prepared.candidates.head
        {
            return Err(store::Error::L1InputsChanged);
        }
        check()
    };
    validate()?;
    if prepared.files.len() != prepared.candidates.paths.len()
        || prepared
            .files
            .iter()
            .zip(&prepared.candidates.paths)
            .any(|(file, path)| &file.path != path)
        || context.store.l1_generation_checked(&check)? != publication.generation
    {
        return Err(store::Error::L1InputsChanged.into());
    }
    let processor = sync::l1::Processor::new(&context.root, &context.config)?;
    let max_bytes = context.config.limits.max_file_bytes()?.0;
    let mut progress = BranchProgress {
        discovery_incomplete: prepared.candidates.max_files_reached
            || prepared.candidates.skipped_non_unicode_paths != 0,
        ..Default::default()
    };
    for file in &prepared.files {
        let validate_file = || {
            check()?;
            boundary(&file.path)?;
            validate()
        };
        validate_file()?;
        let fingerprint = publication
            .fingerprints
            .get(&file.path)
            .ok_or(store::Error::L1InputsChanged)?;
        let (file_id, pending) = context.store.l1_file_work(
            publication.generation,
            &file.path,
            file.content_hash(),
            fingerprint,
            &check,
        )?;
        progress.content_capped_files += usize::from(file.content_capped);
        if !pending {
            // Settled rows still need current source/scope checks. Do not parse
            // them again or clear an earlier non-destructive cap deferral.
            if !sync::target_eligible(&context.root, &file.path, work)? {
                return Err(store::Error::L1InputsChanged.into());
            }
            sync::l1::verify_source(&context.root, file, max_bytes, work)?;
            progress.already_settled_files += 1;
        } else {
            let outcome = processor.process_target(
                &context.store,
                sync::l1::FileInput {
                    work,
                    file,
                    fence: store::l1::Fence {
                        generation: publication.generation,
                        file_id,
                        path: &file.path,
                        content_hash: file.content_hash(),
                        fingerprint,
                    },
                },
                validate_file,
            )?;
            progress.processed_files += 1;
            progress.truncated_files += usize::from(outcome.truncated);
        }
        work.check()?;
    }
    validate()?;
    if context.store.l1_generation_checked(&check)? != publication.generation {
        return Err(store::Error::L1InputsChanged.into());
    }
    publish_links(context, work)?;
    work.check()?;
    Ok(progress)
}

/// Provisional branch inputs; publication must repeat all fences and account for
/// existing catalogue paths before admitting new files or interpreting deletions.
pub(super) struct PreparedBranch {
    git: Option<sync::GitState>,
    pub(super) generation: i64,
    pub(super) snapshot: store::IndexSnapshot,
    pub(super) candidates: sync::discovery::Candidates,
    pub(super) files: Vec<sync::ScannedFile>,
}

/// Prepares bounded branch observations under the original query budget without
/// retaining source bytes or writing the catalogue. Oversized sources keep their
/// metadata-only L0 markers. Success is not indexed-HEAD completion or a filesystem
/// snapshot; capped/skipped candidates remain explicitly incomplete.
pub(super) fn prepare_branch(
    context: &Context,
    work: sync::l1::Work<'_>,
) -> Result<PreparedBranch, Error> {
    prepare_branch_checked(context, work, || Ok(()))
}

fn prepare_branch_checked(
    context: &Context,
    work: sync::l1::Work<'_>,
    mut boundary: impl FnMut() -> Result<(), Error>,
) -> Result<PreparedBranch, Error> {
    let mut check = || {
        work.check()?;
        boundary()?;
        work.check()?;
        Ok::<_, Error>(())
    };
    check()?;
    validate_configuration(context, work)?;
    let generation_check = || work.check().map_err(|_| store::Error::L1InputsChanged);
    let (generation, snapshot) = context.store.catalogue_snapshot(&generation_check)?;
    let max_files = usize::try_from(context.config.limits.max_files).unwrap_or(usize::MAX);
    let max_bytes = context.config.limits.max_file_bytes()?.0;
    let candidates = sync::discovery::discover(&context.root, max_files, work)?;
    check()?;
    // The index is optional evidence: corrupt/truncated indexes must not break
    // HEAD-only query freshness. Cancellation still belongs to the original work.
    let indexed_head = context.store.inspection_checked(&generation_check)?.0;
    let git = sync::inspect_git_checked(&context.root, indexed_head.as_deref(), work)
        .ok()
        .flatten();
    check()?;
    let reusable = git
        .as_ref()
        .map(|git| sync::reusable_files(git, &snapshot, work))
        .transpose()?
        .unwrap_or_default();
    let mut files = Vec::new();
    for path in &candidates.paths {
        check()?;
        if !sync::target_eligible(&context.root, path, work)? {
            return Err(store::Error::L1InputsChanged.into());
        }
        files.push(sync::observe_with_reuse(
            &context.root,
            path,
            max_bytes,
            &reusable,
            work,
        )?);
        check()?;
    }
    // Repeat discovery under current ignore rules. A newly included path or a
    // disappearing candidate must not turn the first observation into deletion
    // authority. A capped subset is still only a subset even if both walks agree.
    if sync::discovery::discover(&context.root, max_files, work)? != candidates {
        return Err(store::Error::L1InputsChanged.into());
    }
    for file in &files {
        check()?;
        sync::l1::verify_source(&context.root, file, max_bytes, work)?;
    }
    check()?;
    validate_configuration(context, work)?;
    if sync::head::read(&context.root, work)? != candidates.head {
        return Err(sync::Error::HeadChanged.into());
    }
    if context.store.l1_generation_checked(&generation_check)? != generation {
        return Err(store::Error::L1InputsChanged.into());
    }
    work.check()?;
    Ok(PreparedBranch {
        git,
        generation,
        snapshot,
        candidates,
        files,
    })
}

/// Reload all effective settings, including query policy, limits and roots.
/// Configuration failures refuse caller authority without copying configuration text.
pub(super) fn validate_configuration(
    context: &Context,
    work: sync::l1::Work<'_>,
) -> Result<(), store::Error> {
    work.check().map_err(|_| store::Error::L1InputsChanged)?;
    let current =
        config::reload::current(&context.config_root).map_err(|_| store::Error::L1InputsChanged)?;
    work.check().map_err(|_| store::Error::L1InputsChanged)?;
    if current != context.config {
        return Err(store::Error::L1InputsChanged);
    }
    let configured = PathBuf::from(&current.index.root);
    let root = fs::canonicalize(if configured.is_absolute() {
        configured
    } else {
        context.config_root.join(configured)
    })
    .map_err(|_| store::Error::L1InputsChanged)?;
    work.check().map_err(|_| store::Error::L1InputsChanged)?;
    let expected_root = store::path::system_path(&context.root);
    if root != expected_root {
        return Err(store::Error::L1InputsChanged);
    }
    Ok(())
}

/// Keeps L1-only search valid while the selected embedding provider changes.
/// Every other effective setting and the canonical repository root must match.
/// The bool reports a provider transition, so an unavailable L2 annotation can
/// be distinguished from a source or policy refusal.
pub(super) fn validate_lexical_configuration(
    context: &Context,
    work: sync::l1::Work<'_>,
) -> Result<bool, store::Error> {
    work.check().map_err(|_| store::Error::L1InputsChanged)?;
    let current =
        config::reload::current(&context.config_root).map_err(|_| store::Error::L1InputsChanged)?;
    work.check().map_err(|_| store::Error::L1InputsChanged)?;
    let changed = current.embedder != context.config.embedder;
    let mut expected = context.config.clone();
    expected.embedder = current.embedder.clone();
    if expected != current {
        return Err(store::Error::L1InputsChanged);
    }
    let configured = PathBuf::from(&current.index.root);
    let root = fs::canonicalize(if configured.is_absolute() {
        configured
    } else {
        context.config_root.join(configured)
    })
    .map_err(|_| store::Error::L1InputsChanged)?;
    work.check().map_err(|_| store::Error::L1InputsChanged)?;
    if root != store::path::system_path(&context.root) {
        return Err(store::Error::L1InputsChanged);
    }
    Ok(changed)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{sync::atomic::AtomicBool, time::Instant};

    fn fixture() -> (tempfile::TempDir, Context) {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("source.rs"), "pub fn before() {}\n").unwrap();
        fs::write(root.path().join("other.rs"), "pub fn unrelated() {}\n").unwrap();
        let context = Context::load(root.path()).unwrap();
        let scan = sync::scan(root.path(), Default::default()).unwrap();
        sync::l1::publish(
            &context.store,
            sync::l1::PublishInput {
                work: Default::default(),
                scan: &scan,
                git: None,
                snapshot: &context.store.index_snapshot().unwrap(),
                generation: context.store.l1_generation().unwrap(),
                root: &context.root,
                settings: &context.config,
                full: false,
            },
        )
        .unwrap();
        context
            .store
            .set_meta("indexed_head", "retained-head")
            .unwrap();
        (root, context)
    }

    #[test]
    fn branch_completion_revalidates_inputs_at_all_three_boundaries() {
        use std::{cell::Cell, sync::atomic::Ordering};
        for stage in 1..=3 {
            for mutation in ["source", "include", "ignore", "config", "head", "cancel"] {
                let (root, context) = fixture();
                let (prepared, publication) = branch_catalogue(&context, Default::default())
                    .unwrap()
                    .unwrap();
                process_branch(&context, &prepared, &publication, Default::default()).unwrap();
                let before = context.store.index_snapshot().unwrap();
                let indexed_at = context.store.get_meta("indexed_at").unwrap();
                let completed = context.store.get_meta("l1_completed_generation").unwrap();
                let calls = Cell::new(0);
                let cancelled = AtomicBool::new(false);
                let result = complete_branch(
                    &context,
                    &prepared,
                    &publication,
                    sync::l1::Work {
                        deadline: None,
                        cancelled: Some(&cancelled),
                    },
                    || {
                        calls.set(calls.get() + 1);
                        if calls.get() == stage {
                            match mutation {
                                "source" => {
                                    fs::write(root.path().join("other.rs"), "pub fn changed() {}\n")
                                        .unwrap()
                                }
                                "include" => {
                                    fs::write(root.path().join("new.rs"), "pub fn added() {}\n")
                                        .unwrap()
                                }
                                "ignore" => {
                                    fs::write(root.path().join(".deepkitignore"), "other.rs\n")
                                        .unwrap()
                                }
                                "config" => fs::write(
                                    root.path().join(".deepkit/config.local.toml"),
                                    "[search]\nlazy_resync=false\n",
                                )
                                .unwrap(),
                                "head" => {
                                    for args in [
                                        vec!["init"],
                                        vec![
                                            "-c",
                                            "user.name=Fixture",
                                            "-c",
                                            "user.email=fixture@example.invalid",
                                            "commit",
                                            "--allow-empty",
                                            "--no-gpg-sign",
                                            "-m",
                                            "new head",
                                        ],
                                    ] {
                                        assert!(
                                            std::process::Command::new("git")
                                                .arg("-C")
                                                .arg(root.path())
                                                .args(args)
                                                .output()
                                                .unwrap()
                                                .status
                                                .success()
                                        );
                                    }
                                }
                                "cancel" => cancelled.store(true, Ordering::Release),
                                _ => unreachable!(),
                            }
                        }
                        Ok(())
                    },
                );
                assert!(result.is_err(), "{stage}: {mutation}");
                assert_eq!(calls.get(), stage);
                assert_eq!(
                    context.store.get_meta("indexed_head").unwrap().as_deref(),
                    Some("retained-head")
                );
                assert_eq!(context.store.get_meta("indexed_at").unwrap(), indexed_at);
                assert_eq!(
                    context.store.get_meta("l1_completed_generation").unwrap(),
                    completed
                );
                assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
                assert_eq!(context.store.index_snapshot().unwrap().dirs, before.dirs);
                assert!(context.store.pending_l1_paths().unwrap().is_empty());
                if mutation == "config" {
                    fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
                }
                let (_, completed) = branch(&context, Default::default()).unwrap().unwrap();
                assert!(completed);
                assert_eq!(
                    context.store.get_meta("indexed_head").unwrap().unwrap(),
                    sync::head::read(&context.root, Default::default())
                        .unwrap()
                        .map_or(String::new(), |head| head.id)
                );
            }
        }
    }

    #[test]
    fn branch_completion_refuses_pending_and_superseded_work() {
        let (_root, context) = fixture();
        let (prepared, publication) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        assert!(
            complete_branch(
                &context,
                &prepared,
                &publication,
                Default::default(),
                || Ok(())
            )
            .is_err()
        );
        process_branch(&context, &prepared, &publication, Default::default()).unwrap();
        let (_, newer) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        assert_ne!(publication.generation, newer.generation);
        assert!(
            complete_branch(
                &context,
                &prepared,
                &publication,
                Default::default(),
                || Ok(())
            )
            .is_err()
        );
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
    }

    #[test]
    fn branch_completion_writer_deadline_preserves_files_and_retries() {
        let (root, context) = fixture();
        let (prepared, publication) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        process_branch(&context, &prepared, &publication, Default::default()).unwrap();
        let before = context.store.index_snapshot().unwrap();
        let writer = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let start = Instant::now();
        assert!(
            complete_branch(
                &context,
                &prepared,
                &publication,
                sync::l1::Work {
                    deadline: Some(start + std::time::Duration::from_millis(30)),
                    cancelled: None,
                },
                || Ok(())
            )
            .is_err()
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
        assert!(
            complete_branch(
                &context,
                &prepared,
                &publication,
                Default::default(),
                || Ok(())
            )
            .unwrap()
        );
    }

    #[test]
    fn branch_completion_distinguishes_discovery_caps_l0_and_empty_trees() {
        let (root, context) = fixture();
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nmax_files=1\n",
        )
        .unwrap();
        let capped = Context::load(root.path()).unwrap();
        let (progress, completed) = branch(&capped, Default::default()).unwrap().unwrap();
        assert!(progress.discovery_incomplete);
        assert!(!completed);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nmax_file_bytes='4 B'\n",
        )
        .unwrap();
        let capped = Context::load(root.path()).unwrap();
        let (progress, completed) = branch(&capped, Default::default()).unwrap().unwrap();
        assert_eq!(progress.content_capped_files, 2);
        assert!(completed);
        assert!(context.store.pending_l1_paths().unwrap().is_empty());
        fs::remove_file(root.path().join("source.rs")).unwrap();
        fs::remove_file(root.path().join("other.rs")).unwrap();
        assert!(branch(&capped, Default::default()).unwrap().unwrap().1);
        assert!(context.store.index_snapshot().unwrap().files.is_empty());
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[search]\nlazy_resync=false\n",
        )
        .unwrap();
        let disabled = Context::load(root.path()).unwrap();
        let changes = disabled.store.writer().unwrap().total_changes();
        assert!(branch(&disabled, Default::default()).unwrap().is_none());
        assert_eq!(disabled.store.writer().unwrap().total_changes(), changes);
    }

    #[test]
    fn branch_catalogue_reconciles_new_deleted_and_renamed_paths_without_completion() {
        use store::vectors::{VectorInsert, VectorTableSpec};
        let (root, context) = fixture();
        target(&context, "source.rs", Default::default()).unwrap();
        let unit = context
            .store
            .unit_by_path("source.rs", None, None)
            .unwrap()
            .unwrap();
        let before = context.store.index_snapshot().unwrap();
        let old = before.files.iter().find(|f| f.path == "source.rs").unwrap();
        let keys = context.store.file_occurrence_keys(old.id).unwrap();
        context
            .store
            .ensure_vector_table(VectorTableSpec {
                slug: "branch",
                model: "fixture",
                profile: "local",
                dims: 2,
                created_at: 0,
                active: true,
            })
            .unwrap();
        context
            .store
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
        fs::remove_file(root.path().join("other.rs")).unwrap();
        fs::write(root.path().join("added.rs"), "pub fn added() {}\n").unwrap();
        let (prepared, publication) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(prepared.snapshot, before);
        assert_eq!(publication.generation, prepared.generation + 1);
        assert_eq!(publication.changes.added, ["added.rs"]);
        assert_eq!(publication.changes.deleted, ["other.rs"]);
        assert_eq!(publication.changes.renamed.len(), 1);
        let after = context.store.index_snapshot().unwrap();
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
        assert_eq!(context.store.file_occurrence_keys(old.id).unwrap(), keys);
        assert_eq!(context.store.count_vectors().unwrap(), 1);
        assert_eq!(
            context
                .store
                .unit_by_id(unit.id)
                .unwrap()
                .unwrap()
                .path
                .as_deref(),
            Some("renamed.rs")
        );
        assert_eq!(
            context.store.pending_l1_paths().unwrap(),
            ["added.rs", "renamed.rs"]
        );
        let progress =
            process_branch(&context, &prepared, &publication, Default::default()).unwrap();
        assert_eq!(progress.processed_files, 2);
        assert_eq!(progress.already_settled_files, 0);
        assert_eq!(progress.truncated_files, 0);
        assert!(!progress.discovery_incomplete);
        assert!(context.store.pending_l1_paths().unwrap().is_empty());
        assert_eq!(context.store.count_vectors().unwrap(), 1);
        assert_eq!(context.store.file_occurrence_keys(old.id).unwrap(), keys);
        assert!(
            context
                .store
                .unit_by_path("added.rs", None, None)
                .unwrap()
                .is_some()
        );
        assert!(
            context
                .store
                .unit_by_path("other.rs", None, None)
                .unwrap()
                .is_none()
        );
        assert_eq!(context.store.index_snapshot().unwrap().dirs, before.dirs);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
    }

    #[test]
    fn branch_processing_retains_partial_commits_on_cancellation_and_retries() {
        use std::{cell::Cell, sync::atomic::Ordering};
        let (_root, context) = fixture();
        let (prepared, publication) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        let cancelled = AtomicBool::new(false);
        let calls = Cell::new(0);
        let result = process_branch_checked(
            &context,
            &prepared,
            &publication,
            sync::l1::Work {
                deadline: None,
                cancelled: Some(&cancelled),
            },
            |path| {
                calls.set(calls.get() + 1);
                if path == "source.rs" {
                    cancelled.store(true, Ordering::Release);
                }
                Ok(())
            },
        );
        assert!(result.is_err());
        assert!(calls.get() > 1);
        assert_eq!(context.store.pending_l1_paths().unwrap(), ["source.rs"]);
        assert!(
            context
                .store
                .unit_by_path("other.rs", None, None)
                .unwrap()
                .is_some()
        );
        let progress =
            process_branch(&context, &prepared, &publication, Default::default()).unwrap();
        assert_eq!(progress.processed_files, 1);
        assert_eq!(progress.already_settled_files, 1);
        assert!(context.store.pending_l1_paths().unwrap().is_empty());
        let changes = context.store.writer().unwrap().total_changes();
        let progress =
            process_branch(&context, &prepared, &publication, Default::default()).unwrap();
        assert_eq!(progress.processed_files, 0);
        assert_eq!(progress.already_settled_files, 2);
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
    }

    #[test]
    fn branch_processing_refuses_late_inputs_and_rolls_back_the_current_file() {
        use std::cell::Cell;
        for mutation in ["source", "ignore", "config", "head", "cancel"] {
            let (root, context) = fixture();
            // Begin with old lexical content so rollback must preserve real rows.
            target(&context, "other.rs", Default::default()).unwrap();
            let old = context
                .store
                .unit_by_path("other.rs", None, None)
                .unwrap()
                .unwrap();
            fs::write(root.path().join("other.rs"), "pub fn replacement() {}\n").unwrap();
            let (prepared, publication) = branch_catalogue(&context, Default::default())
                .unwrap()
                .unwrap();
            let before = context.store.index_snapshot().unwrap();
            let calls = Cell::new(0);
            let cancelled = AtomicBool::new(false);
            let result = process_branch_checked(
                &context,
                &prepared,
                &publication,
                sync::l1::Work {
                    deadline: None,
                    cancelled: Some(&cancelled),
                },
                |_| {
                    calls.set(calls.get() + 1);
                    // Before processing, writer admission, then before commit.
                    if calls.get() == 3 {
                        match mutation {
                            "source" => fs::write(
                                root.path().join("other.rs"),
                                "pub fn changed_again() {}\n",
                            )
                            .unwrap(),
                            "ignore" => {
                                fs::write(root.path().join(".deepkitignore"), "other.rs\n").unwrap()
                            }
                            "config" => fs::write(
                                root.path().join(".deepkit/config.local.toml"),
                                "[search]\nlazy_resync=false\n",
                            )
                            .unwrap(),
                            "head" => {
                                for args in [
                                    vec!["init"],
                                    vec![
                                        "-c",
                                        "user.name=Fixture",
                                        "-c",
                                        "user.email=fixture@example.invalid",
                                        "commit",
                                        "--allow-empty",
                                        "--no-gpg-sign",
                                        "-m",
                                        "head fence",
                                    ],
                                ] {
                                    assert!(
                                        std::process::Command::new("git")
                                            .arg("-C")
                                            .arg(root.path())
                                            .args(args)
                                            .output()
                                            .unwrap()
                                            .status
                                            .success()
                                    );
                                }
                                fs::write(root.path().join(".git/index"), b"short").unwrap();
                            }
                            "cancel" => cancelled.store(true, std::sync::atomic::Ordering::Release),
                            _ => unreachable!(),
                        }
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "{mutation}");
            assert_eq!(calls.get(), 3, "{mutation}");
            assert_eq!(
                context.store.index_snapshot().unwrap().files,
                before.files,
                "{mutation}"
            );
            assert_eq!(
                context
                    .store
                    .unit_by_path("other.rs", None, None)
                    .unwrap()
                    .unwrap()
                    .id,
                old.id
            );
            assert_eq!(
                context.store.pending_l1_paths().unwrap(),
                ["other.rs", "source.rs"]
            );
            assert_eq!(
                context.store.get_meta("indexed_head").unwrap().as_deref(),
                Some("retained-head")
            );
            fs::write(root.path().join("other.rs"), "pub fn replacement() {}\n").unwrap();
            for path in [".deepkitignore", ".deepkit/config.local.toml"] {
                if root.path().join(path).exists() {
                    fs::remove_file(root.path().join(path)).unwrap();
                }
            }
            if mutation == "head" {
                fs::remove_dir_all(root.path().join(".git")).unwrap();
            }
            assert_eq!(
                process_branch(&context, &prepared, &publication, Default::default())
                    .unwrap()
                    .processed_files,
                2
            );
        }
    }

    #[test]
    fn branch_processing_bounds_writer_wait_and_refuses_superseded_publication() {
        let (root, context) = fixture();
        let (prepared, publication) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        let before = context.store.index_snapshot().unwrap();
        let writer = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let start = Instant::now();
        assert!(
            process_branch(
                &context,
                &prepared,
                &publication,
                sync::l1::Work {
                    deadline: Some(start + std::time::Duration::from_millis(30)),
                    cancelled: None,
                }
            )
            .is_err()
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
        assert_eq!(
            process_branch(&context, &prepared, &publication, Default::default())
                .unwrap()
                .processed_files,
            2
        );
        fs::write(root.path().join("other.rs"), "pub fn newer() {}\n").unwrap();
        let changes = context.store.writer().unwrap().total_changes();
        // Already-settled observations must still validate their current source.
        assert!(process_branch(&context, &prepared, &publication, Default::default()).is_err());
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        let (new_prepared, new_publication) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        let changes = context.store.writer().unwrap().total_changes();
        assert!(process_branch(&context, &prepared, &publication, Default::default()).is_err());
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        assert_eq!(
            process_branch(
                &context,
                &new_prepared,
                &new_publication,
                Default::default()
            )
            .unwrap()
            .processed_files,
            1
        );
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
    }

    #[test]
    fn branch_processing_accounts_for_incomplete_discovery_and_cap_deferrals() {
        let (root, context) = fixture();
        // The discovery cap applies before sorting filesystem entries. Seed the
        // admitted file so either traversal order exercises retained cap deferral.
        let candidates = sync::discovery::discover(root.path(), 1, Default::default()).unwrap();
        assert_eq!(candidates.paths.len(), 1);
        let retained_path = &candidates.paths[0];
        let pending_path = if retained_path == "other.rs" {
            "source.rs"
        } else {
            "other.rs"
        };
        target(&context, retained_path, Default::default()).unwrap();
        let old = context
            .store
            .unit_by_path(retained_path, None, None)
            .unwrap()
            .unwrap();
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nmax_files=1\nmax_file_bytes='4 B'\n",
        )
        .unwrap();
        let capped = Context::load(root.path()).unwrap();
        let (prepared, publication) = branch_catalogue(&capped, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(prepared.candidates.paths, candidates.paths);
        let progress =
            process_branch(&capped, &prepared, &publication, Default::default()).unwrap();
        assert_eq!(progress.processed_files, 1);
        assert_eq!(progress.truncated_files, 1);
        assert_eq!(progress.content_capped_files, 1);
        assert!(progress.discovery_incomplete);
        assert_eq!(
            context
                .store
                .unit_by_path(retained_path, None, None)
                .unwrap()
                .unwrap()
                .id,
            old.id
        );
        assert_eq!(context.store.pending_l1_paths().unwrap(), [pending_path]);
        let changes = capped.store.writer().unwrap().total_changes();
        let retry = process_branch(&capped, &prepared, &publication, Default::default()).unwrap();
        assert_eq!(retry.processed_files, 0);
        assert_eq!(retry.already_settled_files, 1);
        assert!(retry.discovery_incomplete);
        assert_eq!(capped.store.writer().unwrap().total_changes(), changes);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
    }

    #[test]
    fn branch_catalogue_refuses_disabled_config_capacity_and_writer_timeout_then_retries() {
        let (root, mut context) = fixture();
        let before = context.store.index_snapshot().unwrap();
        let generation = context.store.l1_generation().unwrap();
        context.config.search.lazy_resync = false;
        assert!(
            branch_catalogue(&context, Default::default())
                .unwrap()
                .is_none()
        );
        context.config.search.lazy_resync = true;
        let local = root.path().join(".deepkit/config.local.toml");
        fs::write(&local, "[limits]\nmax_files=1\n").unwrap();
        assert!(branch_catalogue(&context, Default::default()).is_err());
        // The sole discovered new path cannot displace either retained old row.
        fs::remove_file(root.path().join("source.rs")).unwrap();
        fs::remove_file(root.path().join("other.rs")).unwrap();
        fs::write(root.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        fs::write(root.path().join("b.rs"), "pub fn b() {}\n").unwrap();
        let capped = Context::load(root.path()).unwrap();
        assert!(branch_catalogue(&capped, Default::default()).is_err());
        assert_eq!(context.store.index_snapshot().unwrap(), before);
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        fs::remove_file(&local).unwrap();
        let writer = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let start = Instant::now();
        assert!(
            branch_catalogue(
                &context,
                sync::l1::Work {
                    deadline: Some(start + std::time::Duration::from_millis(30)),
                    cancelled: None,
                }
            )
            .is_err()
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(context.store.index_snapshot().unwrap(), before);
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        let (_, publication) = branch_catalogue(&context, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(publication.changes.added, ["a.rs", "b.rs"]);
        assert_eq!(publication.changes.deleted, ["other.rs", "source.rs"]);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
    }

    #[test]
    fn branch_preparation_retains_bounded_observations_and_never_writes() {
        let (root, context) = fixture();
        let generation = context.store.l1_generation().unwrap();
        let before = context.store.index_snapshot().unwrap();
        let changes = context.store.writer().unwrap().total_changes();
        let prepared = prepare_branch(&context, Default::default()).unwrap();
        assert_eq!(prepared.generation, generation);
        assert_eq!(prepared.snapshot, before);
        assert_eq!(prepared.candidates.head, None);
        assert_eq!(prepared.candidates.paths, ["other.rs", "source.rs"]);
        assert!(!prepared.candidates.max_files_reached);
        for file in &prepared.files {
            assert_eq!(
                file,
                &sync::l1::observe_source(
                    root.path(),
                    &file.path,
                    1024,
                    Default::default(),
                    |_| {}
                )
                .unwrap()
            );
        }
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nmax_files=1\nmax_file_bytes='4 B'\n",
        )
        .unwrap();
        let capped = Context::load(root.path()).unwrap();
        let prepared = prepare_branch(&capped, Default::default()).unwrap();
        assert_eq!(prepared.files.len(), 1);
        assert!(prepared.files[0].content_capped);
        assert!(prepared.candidates.max_files_reached);
        assert!(prepared.files[0].size() > 4);
        assert_eq!(prepared.generation, generation);
        assert_eq!(prepared.snapshot, before);
        assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
    }

    #[test]
    fn branch_preparation_refuses_late_source_policy_config_and_generation_changes() {
        for mutation in [
            "content",
            "delete",
            "ignore",
            "include",
            "config",
            "generation",
            "link",
        ] {
            let (root, context) = fixture();
            let before = context.store.index_snapshot().unwrap();
            let generation = context.store.l1_generation().unwrap();
            let changes = context.store.writer().unwrap().total_changes();
            let mut checks = 0;
            let result = prepare_branch_checked(&context, Default::default(), || {
                checks += 1;
                // First path has been observed; mutate before observing the next.
                if checks == 5 {
                    match mutation {
                        "content" => {
                            fs::write(root.path().join("other.rs"), "pub fn changed() {}\n")
                                .unwrap()
                        }
                        "delete" => fs::remove_file(root.path().join("other.rs")).unwrap(),
                        "ignore" => {
                            fs::write(root.path().join(".deepkitignore"), "other.rs\n").unwrap()
                        }
                        "include" => {
                            fs::write(root.path().join("new.rs"), "pub fn new() {}\n").unwrap()
                        }
                        "config" => fs::write(
                            root.path().join(".deepkit/config.local.toml"),
                            "[search]\nlazy_resync=false\n",
                        )
                        .unwrap(),
                        "generation" => context
                            .store
                            .set_meta("l1_scan_generation", &(generation + 1).to_string())
                            .unwrap(),
                        "link" => {
                            fs::remove_file(root.path().join("other.rs")).unwrap();
                            std::os::unix::fs::symlink("source.rs", root.path().join("other.rs"))
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                }
                Ok(())
            });
            assert!(result.is_err(), "{mutation}");
            assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
            assert_eq!(
                context.store.writer().unwrap().total_changes(),
                changes + u64::from(mutation == "generation")
            );
            assert_eq!(
                context.store.get_meta("indexed_head").unwrap().as_deref(),
                Some("retained-head")
            );
            if mutation == "link" {
                fs::remove_file(root.path().join("other.rs")).unwrap();
            }
            fs::write(root.path().join("other.rs"), "pub fn unrelated() {}\n").unwrap();
            for path in [".deepkitignore", ".deepkit/config.local.toml", "new.rs"] {
                if root.path().join(path).exists() {
                    fs::remove_file(root.path().join(path)).unwrap();
                }
            }
            assert!(
                prepare_branch(&context, Default::default()).is_ok(),
                "retry {mutation}"
            );
        }
    }

    #[test]
    fn branch_preparation_fences_head_without_loading_the_git_index() {
        use std::process::Command;
        let root = tempfile::tempdir().unwrap();
        assert!(
            Command::new("bash")
                .arg(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/scripts/fixtures/make-git-repo.sh"
                ))
                .arg(root.path())
                .output()
                .unwrap()
                .status
                .success()
        );
        let context = Context::load(root.path()).unwrap();
        fs::write(root.path().join(".git/index"), b"short").unwrap();
        let before = prepare_branch(&context, Default::default()).unwrap();
        assert!(before.candidates.head.is_some());
        let indexed_head = context.store.get_meta("indexed_head").unwrap();
        let changes = context.store.writer().unwrap().total_changes();
        let mut checks = 0;
        let result = prepare_branch_checked(&context, Default::default(), || {
            checks += 1;
            if checks == 5 {
                fs::write(root.path().join(".git/HEAD"), "ref: refs/heads/alternate\n").unwrap();
            }
            Ok(())
        });
        assert!(result.is_err());
        let alternate = prepare_branch(&context, Default::default()).unwrap();
        assert_ne!(alternate.candidates.head, before.candidates.head);
        assert_eq!(alternate.generation, before.generation);
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        assert_eq!(fs::read(root.path().join(".git/index")).unwrap(), b"short");
        assert!(context.store.index_snapshot().unwrap().files.is_empty());
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap(),
            indexed_head
        );
    }

    #[test]
    fn branch_preparation_preserves_original_work_at_every_boundary_and_reader_admission() {
        use std::sync::{Arc, atomic::Ordering};
        let (root, mut context) = fixture();
        let mut total = 0;
        prepare_branch_checked(&context, Default::default(), || {
            total += 1;
            Ok(())
        })
        .unwrap();
        let changes = context.store.writer().unwrap().total_changes();
        for stop in 1..=total {
            let cancelled = AtomicBool::new(false);
            let mut checks = 0;
            let result = prepare_branch_checked(
                &context,
                sync::l1::Work {
                    cancelled: Some(&cancelled),
                    deadline: None,
                },
                || {
                    checks += 1;
                    if checks == stop {
                        cancelled.store(true, Ordering::Release);
                    }
                    Ok(())
                },
            );
            assert!(
                matches!(result, Err(Error::Sync(sync::Error::WorkStopped))),
                "boundary {stop}"
            );
            assert_eq!(checks, stop);
        }
        assert_eq!(context.store.writer().unwrap().total_changes(), changes);
        assert!(
            prepare_branch(
                &context,
                sync::l1::Work {
                    deadline: Some(Instant::now()),
                    cancelled: None
                }
            )
            .is_err()
        );
        context.store = Arc::new(store::open(":memory:", Default::default()).unwrap());
        let held = context.store.writer().unwrap();
        let start = Instant::now();
        assert!(
            prepare_branch(
                &context,
                sync::l1::Work {
                    deadline: Some(start + std::time::Duration::from_millis(20)),
                    cancelled: None,
                }
            )
            .is_err()
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        drop(held);
        assert!(prepare_branch(&context, Default::default()).is_ok());
        assert!(root.path().join("source.rs").is_file());
    }

    fn request() -> super::super::pipeline::Request {
        super::super::pipeline::Request {
            query: "before".into(),
            mode: super::super::fusion::Mode::Lexical,
            scope: Some("source.rs".into()),
            k: 10,
            exclude_ids: vec![],
        }
    }

    fn branch_git(root: &std::path::Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

    fn commit_branch(root: &std::path::Path) -> String {
        branch_git(root, &["add", "--all", "*.rs"]);
        branch_git(
            root,
            &[
                "-c",
                "user.name=Branch Test",
                "-c",
                "user.email=branch@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "branch",
            ],
        );
        branch_git(root, &["rev-parse", "HEAD"])
    }

    #[tokio::test]
    async fn search_branch_checkout_discovers_new_matches_and_reloads_completed_identity() {
        let (root, context) = fixture();
        branch_git(root.path(), &["init", "--quiet"]);
        let first = commit_branch(root.path());
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        fs::rename(
            root.path().join("source.rs"),
            root.path().join("renamed.rs"),
        )
        .unwrap();
        fs::remove_file(root.path().join("other.rs")).unwrap();
        fs::write(
            root.path().join("added.rs"),
            "pub fn newly_discovered() {}\n",
        )
        .unwrap();
        let second = commit_branch(root.path());
        branch_git(root.path(), &["checkout", "--quiet", &first]);
        branch_git(root.path(), &["checkout", "--quiet", &second]);
        let mut query = request();
        query.query = "newly_discovered".into();
        query.scope = None;
        let result = context.search(query, false).await.unwrap();
        assert_eq!(
            result["index"]["branch_resync"]["state"], "completed",
            "{result}"
        );
        assert_eq!(result["index"]["indexed_head"], second);
        assert_eq!(result["index"]["partial"], false);
        assert_eq!(result["index"]["lexical"], "complete");
        assert_eq!(result["index"]["l1_pending_files"], 0);
        assert_eq!(result["hits"][0]["path"], "added.rs");
        assert_eq!(result["hits"][0]["fresh"]["l1"], "current");
        let mut query = request();
        query.scope = None;
        let renamed = context.search(query, false).await.unwrap();
        assert_eq!(renamed["index"]["branch_resync"]["state"], "not_needed");
        assert_eq!(renamed["hits"][0]["path"], "renamed.rs");
        let snapshot = context.store.index_snapshot().unwrap();
        assert!(
            !snapshot
                .files
                .iter()
                .any(|file| matches!(file.path.as_str(), "other.rs" | "source.rs"))
        );
    }

    #[tokio::test]
    async fn symbol_branch_checkout_discovers_additions_renames_and_deletions() {
        let (root, context) = fixture();
        branch_git(root.path(), &["init", "--quiet"]);
        let first = commit_branch(root.path());
        assert_eq!(
            context.symbol("before").await.unwrap()[0]["path"],
            "source.rs"
        );
        fs::rename(
            root.path().join("source.rs"),
            root.path().join("renamed.rs"),
        )
        .unwrap();
        fs::remove_file(root.path().join("other.rs")).unwrap();
        fs::write(
            root.path().join("added.rs"),
            "pub fn newly_discovered() {}\n",
        )
        .unwrap();
        let second = commit_branch(root.path());
        branch_git(root.path(), &["checkout", "--quiet", &first]);
        branch_git(root.path(), &["checkout", "--quiet", &second]);
        let result = context.symbol("newly_discovered").await.unwrap();
        assert_eq!(result[0]["path"], "added.rs");
        assert_eq!(result[0]["lines"], serde_json::json!([1, 1]));
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some(second.as_str())
        );
        assert_eq!(
            context.symbol("before").await.unwrap()[0]["path"],
            "renamed.rs"
        );
        assert_eq!(
            context.symbol("unrelated").await.unwrap(),
            serde_json::json!([])
        );
    }

    #[tokio::test]
    async fn symbol_branch_refuses_disabled_incomplete_policy_and_pending_empty_arrays() {
        let (root, context) = fixture();
        let override_path = root.path().join(".deepkit/config.local.toml");
        let generation = context.store.l1_generation().unwrap();
        fs::write(&override_path, "[search]\nlazy_resync=false\n").unwrap();
        let disabled = Context::load(root.path()).unwrap();
        for name in ["before", "absent"] {
            assert_eq!(
                disabled.symbol(name).await.unwrap_err().code(),
                "source_unavailable"
            );
        }
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        fs::write(&override_path, "[limits]\nmax_files=1\n").unwrap();
        let capped = Context::load(root.path()).unwrap();
        assert_eq!(
            capped.symbol("absent").await.unwrap_err().code(),
            "source_unavailable"
        );
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("retained-head")
        );
        fs::write(&override_path, "[limits]\nmax_files=3\n").unwrap();
        assert_eq!(
            capped.symbol("absent").await.unwrap_err().code(),
            "source_unavailable"
        );
        fs::remove_file(override_path).unwrap();
        assert_eq!(
            context.symbol("absent").await.unwrap(),
            serde_json::json!([])
        );
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=1 WHERE path='other.rs'", [])
            .unwrap();
        // Pending work outside the selected name can hide another match.
        for name in ["before", "absent"] {
            assert_eq!(
                context.symbol(name).await.unwrap_err().code(),
                "source_unavailable"
            );
        }
        target(&context, "other.rs", Default::default()).unwrap();
        assert_eq!(
            context.symbol("absent").await.unwrap(),
            serde_json::json!([])
        );
        assert_eq!(
            context.symbol("before").await.unwrap()[0]["path"],
            "source.rs"
        );
    }

    #[tokio::test]
    async fn search_branch_disabled_capped_and_refused_never_claim_complete_zero_hits() {
        let (root, context) = fixture();
        let mut query = request();
        query.query = "absent_token".into();
        let override_path = root.path().join(".deepkit/config.local.toml");
        fs::write(&override_path, "[search]\nlazy_resync=false\n").unwrap();
        let disabled = Context::load(root.path()).unwrap();
        let generation = context.store.l1_generation().unwrap();
        let result = disabled.search(query, false).await.unwrap();
        assert_eq!(result["index"]["branch_resync"]["state"], "disabled");
        assert_eq!(result["index"]["partial"], true);
        assert_eq!(result["index"]["lexical"], "incomplete");
        assert_eq!(result["hits"], serde_json::json!([]));
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        fs::write(&override_path, "[limits]\nmax_files=1\n").unwrap();
        let capped = Context::load(root.path()).unwrap();
        let result = capped.search(request(), false).await.unwrap();
        assert_eq!(
            result["index"]["branch_resync"]["state"], "incomplete",
            "{result}"
        );
        assert_eq!(
            result["index"]["branch_resync"]["discovery_incomplete"],
            true
        );
        assert_eq!(result["index"]["partial"], true);
        assert_eq!(result["index"]["lexical"], "incomplete");
        assert_eq!(result["index"]["indexed_head"], "retained-head");
        // A context captured before the config change cannot publish its policy.
        fs::write(&override_path, "[limits]\nmax_files=3\n").unwrap();
        let result = capped.search(request(), false).await.unwrap();
        assert_eq!(result["index"]["branch_resync"]["state"], "refused");
        assert_eq!(result["index"]["partial"], true);
        assert_eq!(result["index"]["lexical"], "incomplete");
        fs::remove_file(override_path).unwrap();
        let result = context.search(request(), false).await.unwrap();
        assert_eq!(
            result["index"]["branch_resync"]["state"], "completed",
            "{result}"
        );
        assert_eq!(result["index"]["l1_pending_files"], 0);
        assert_eq!(result["index"]["partial"], false);
        assert_eq!(result["hits"][0]["path"], "source.rs");
    }

    #[tokio::test]
    async fn public_repair_deduplicates_reretrieves_and_refreshes_pending_inspection() {
        let (root, context) = fixture();
        context.store.set_meta("indexed_head", "").unwrap();
        fs::write(
            root.path().join("source.rs"),
            "pub fn before_one() {}\npub fn before_two() {}\n",
        )
        .unwrap();
        target(&context, "source.rs", Default::default()).unwrap();
        let initial = context.search(request(), true).await.unwrap();
        assert_eq!(initial["hits"].as_array().unwrap().len(), 2);
        let generation = context.store.l1_generation().unwrap();
        fs::write(root.path().join("source.rs"),
            "\n\npub fn before_one() { let updated = 1; }\npub fn before_two() { let updated = 2; }\n").unwrap();
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=1 WHERE path='source.rs'", [])
            .unwrap();
        let result = context.search(request(), true).await.unwrap();
        assert_eq!(result["index"]["lazy_resync"]["attempted_files"], 1);
        assert_eq!(result["index"]["lazy_resync"]["published_files"], 1);
        assert_eq!(result["index"]["partial"], true);
        assert_eq!(result["index"]["lexical"], "incomplete");
        assert_eq!(result["index"]["l1_pending_files"], 1); // unrelated file
        assert_eq!(context.store.l1_generation().unwrap(), generation + 1);
        assert_eq!(result["index"]["indexed_head"], "");
        assert_eq!(result["hits"].as_array().unwrap().len(), 2);
        for hit in result["hits"].as_array().unwrap() {
            assert_eq!(hit["path"], "source.rs");
            assert!(hit["lines"][0].as_u64().unwrap() >= 3);
            assert_eq!(hit["fresh"]["state"], "fresh");
            assert!(
                !initial["hits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|old| old["id"] == hit["id"])
            );
        }
        // Replacement no longer matches: old IDs must not be returned or patched.
        fs::write(root.path().join("source.rs"), "pub fn unrelated() {}\n").unwrap();
        let result = context.search(request(), false).await.unwrap();
        assert_eq!(result["hits"], serde_json::json!([]));
        assert_eq!(result["index"]["partial"], true);
        assert_eq!(result["index"]["lexical"], "incomplete");
        assert_eq!(result["index"]["lazy_resync"]["published_files"], 1);
        // Repairing a selected file leaves unrelated pending work incomplete.
        // Once that work is completed, the no-match response is complete too.
        target(&context, "other.rs", Default::default()).unwrap();
        let complete = context.search(request(), false).await.unwrap();
        assert_eq!(complete["hits"], serde_json::json!([]));
        assert_eq!(complete["index"]["l1_pending_files"], 0);
        assert_eq!(complete["index"]["partial"], false);
        assert_eq!(complete["index"]["lexical"], "complete");
    }

    #[tokio::test]
    async fn public_repair_preserves_scope_exclusions_and_refused_partial_state() {
        let (root, context) = fixture();
        context.store.set_meta("indexed_head", "").unwrap();
        target(&context, "source.rs", Default::default()).unwrap();
        fs::write(root.path().join("other.rs"), "pub fn before() {}\n").unwrap();
        target(&context, "other.rs", Default::default()).unwrap();
        let id = crate::search::lexical(
            &context.store,
            "before",
            10,
            Some("source.rs"),
            &context.config.search,
        )
        .unwrap()[0]
            .unit
            .id;
        fs::write(root.path().join("source.rs"), "\n\npub fn before() {}\n").unwrap();
        let mut excluded = request();
        excluded.exclude_ids.push(id);
        assert_eq!(
            context.search(excluded, false).await.unwrap()["index"]["lazy_resync"]["attempted_files"],
            0
        );
        let generation = context.store.l1_generation().unwrap();
        fs::write(root.path().join(".deepkitignore"), "source.rs\n").unwrap();
        let refused = context.search(request(), false).await.unwrap();
        assert_eq!(refused["hits"], serde_json::json!([]));
        assert_eq!(refused["index"]["partial"], true);
        assert_eq!(refused["index"]["lazy_resync"]["refused_files"], 1);
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        fs::remove_file(root.path().join(".deepkitignore")).unwrap();
        let repaired = context.search(request(), false).await.unwrap();
        assert_eq!(repaired["hits"].as_array().unwrap().len(), 1);
        assert_eq!(repaired["hits"][0]["path"], "source.rs");
        assert_eq!(repaired["hits"][0]["lines"][0], 3);
        assert_eq!(repaired["index"]["partial"], false);
    }

    #[tokio::test]
    async fn public_repair_writer_deadline_retains_snapshot_and_retry_repairs() {
        let (root, context) = fixture();
        target(&context, "source.rs", Default::default()).unwrap();
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nquery_timeout_ms=100\n",
        )
        .unwrap();
        let context = Context::load(root.path()).unwrap();
        let generation = context.store.l1_generation().unwrap();
        let before = context.store.index_snapshot().unwrap();
        fs::write(root.path().join("source.rs"), "\n\npub fn before() {}\n").unwrap();
        let writer = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let start = Instant::now();
        let result = context.search(request(), false).await.unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(result["hits"], serde_json::json!([]));
        assert_eq!(result["index"]["partial"], true);
        assert_eq!(result["index"]["reason"], "query_timeout");
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
        // A longer subsequent request can finish; the timed-out work stays cancelled.
        fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
        let result = Context::load(root.path())
            .unwrap()
            .search(request(), false)
            .await
            .unwrap();
        assert_eq!(result["index"]["partial"], false);
        assert_eq!(result["hits"][0]["lines"][0], 3);
        assert_eq!(context.store.l1_generation().unwrap(), generation + 1);
    }

    #[tokio::test]
    async fn map_repair_refreshes_selected_signatures_focus_and_budget() {
        let (root, context) = fixture();
        // Start from a completed branch so this exercises selected-file repair.
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        let snapshot = context.store.index_snapshot().unwrap();
        let generation = context.store.l1_generation().unwrap();
        let focus = ["source.rs".to_owned()];
        fs::write(
            root.path().join("source.rs"),
            "\n\npub fn replacement(value: u32) -> u32 { value }\n",
        )
        .unwrap();
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=1 WHERE path='source.rs'", [])
            .unwrap();
        // Unselected files must not be repaired, including files outside the budget.
        assert_eq!(
            context.map(1, &focus).await.unwrap()["index"]["lazy_resync"]["attempted_files"],
            0
        );
        assert_eq!(
            context.map(2000, &["other.rs".into()]).await.unwrap()["index"]["lazy_resync"]["attempted_files"],
            0
        );
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        let repaired = context.map(2000, &focus).await.unwrap();
        assert_eq!(repaired["index"]["lazy_resync"]["attempted_files"], 1);
        assert_eq!(repaired["index"]["lazy_resync"]["published_files"], 1);
        assert_eq!(repaired["index"]["partial"], false);
        assert_eq!(repaired["nodes"].as_array().unwrap().len(), 1);
        assert_eq!(repaired["nodes"][0]["symbols"][0]["name"], "replacement");
        assert_eq!(
            repaired["nodes"][0]["symbols"][0]["lines"],
            serde_json::json!([3, 3])
        );
        let tree = repaired["tree"].as_str().unwrap();
        assert!(tree.contains("replacement"));
        assert!(!tree.contains("before"));
        assert!(
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(tree)
                .len()
                <= 2000
        );
        assert_eq!(repaired["index"]["indexed_head"], "");
        assert_eq!(repaired["index"]["l1_pending_files"], 0);
        let after = context.store.index_snapshot().unwrap();
        assert_eq!(
            after.files.iter().find(|f| f.path == "other.rs"),
            snapshot.files.iter().find(|f| f.path == "other.rs")
        );
        // Symbol focus must be rerun: its former file no longer matches after repair.
        fs::write(root.path().join("source.rs"), "pub fn final_name() {}\n").unwrap();
        let removed = context.map(2000, &["replacement".into()]).await.unwrap();
        assert_eq!(removed["index"]["lazy_resync"]["published_files"], 1);
        assert_eq!(removed["nodes"], serde_json::json!([]));
        assert_eq!(removed["tree"], "");
        assert_eq!(removed["index"]["partial"], false);
    }

    #[tokio::test]
    async fn map_repair_refusal_deadline_and_retry_preserve_snapshot() {
        let (root, context) = fixture();
        // Start from a completed branch so this exercises selected-file repair.
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        let snapshot = context.store.index_snapshot().unwrap();
        let generation = context.store.l1_generation().unwrap();
        let focus = ["source.rs".to_owned()];
        fs::write(root.path().join("source.rs"), "\n\npub fn before() {}\n").unwrap();
        fs::write(root.path().join(".deepkitignore"), "source.rs\n").unwrap();
        let refused = context.map(2000, &focus).await.unwrap();
        assert_eq!(refused["index"]["lazy_resync"]["refused_files"], 1);
        assert_eq!(refused["index"]["partial"], true);
        assert_eq!(refused["index"]["suppressed_files"], 1);
        assert_eq!(refused["nodes"], serde_json::json!([]));
        assert_eq!(refused["tree"], "");
        fs::remove_file(root.path().join(".deepkitignore")).unwrap();
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nquery_timeout_ms=100\n",
        )
        .unwrap();
        let bounded = Context::load(root.path()).unwrap();
        let writer = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let start = Instant::now();
        let timed = bounded.map(2000, &focus).await.unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(timed["index"]["reason"], "query_timeout");
        assert_eq!(timed["nodes"], serde_json::json!([]));
        assert_eq!(timed["tree"], "");
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(
            context.store.index_snapshot().unwrap().files,
            snapshot.files
        );
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
        let repaired = context.map(2000, &focus).await.unwrap();
        assert_eq!(repaired["index"]["partial"], false);
        assert_eq!(
            repaired["nodes"][0]["symbols"][0]["lines"],
            serde_json::json!([3, 3])
        );
        assert_eq!(context.store.l1_generation().unwrap(), generation + 1);
    }

    #[tokio::test]
    async fn renamed_symbol_waits_for_new_path_discovery_after_catalogue_removal() {
        let (root, context) = fixture();
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        assert_eq!(
            context.symbol("before").await.unwrap()[0]["path"],
            "source.rs"
        );
        fs::rename(root.path().join("source.rs"), root.path().join("moved.rs")).unwrap();
        let mapped = context.map(2000, &["source.rs".into()]).await.unwrap();
        assert_eq!(mapped["nodes"], serde_json::json!([]));
        assert_eq!(
            context.symbol("before").await.unwrap_err().code(),
            "source_unavailable"
        );
        // Model the old-path deletion arriving before new-path discovery. No
        // watcher runs in this fixture, so both observations are deterministic.
        context.store.delete_file("source.rs").unwrap();
        assert_eq!(
            context.symbol("before").await.unwrap(),
            serde_json::json!([])
        );
        assert!(
            !context
                .store
                .index_snapshot()
                .unwrap()
                .files
                .iter()
                .any(|f| f.path == "moved.rs")
        );
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        let discovered = context.symbol("before").await.unwrap();
        assert_eq!(discovered.as_array().unwrap().len(), 1);
        assert_eq!(discovered[0]["path"], "moved.rs");
        assert_eq!(discovered[0]["lines"], serde_json::json!([1, 1]));
    }

    #[tokio::test]
    async fn symbol_repair_refreshes_pending_ranges_docs_and_removed_names() {
        let (root, context) = fixture();
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        let initial = context.symbol("before").await.unwrap();
        let generation = context.store.l1_generation().unwrap();
        let other = context
            .store
            .index_snapshot()
            .unwrap()
            .files
            .into_iter()
            .find(|file| file.path == "other.rs")
            .unwrap();
        fs::write(
            root.path().join("source.rs"),
            "\n/// Updated documentation.\npub fn before() {}\n",
        )
        .unwrap();
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=1 WHERE path='source.rs'", [])
            .unwrap();
        let repaired = context
            .symbol(initial[0]["id"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(repaired[0]["id"], initial[0]["id"]);
        assert_eq!(repaired[0]["lines"], serde_json::json!([3, 3]));
        assert_eq!(repaired[0]["doc"], "Updated documentation.");
        assert_eq!(context.store.l1_generation().unwrap(), generation + 1);
        assert_eq!(
            context.store.get_meta("indexed_head").unwrap().as_deref(),
            Some("")
        );
        assert_eq!(
            context
                .store
                .index_snapshot()
                .unwrap()
                .files
                .into_iter()
                .find(|file| file.path == "other.rs")
                .unwrap(),
            other
        );
        fs::write(root.path().join("source.rs"), "pub fn replacement() {}\n").unwrap();
        assert_eq!(
            context.symbol("before").await.unwrap(),
            serde_json::json!([])
        );
        assert_eq!(
            context.symbol("replacement").await.unwrap()[0]["lines"],
            serde_json::json!([1, 1])
        );
    }

    #[tokio::test]
    async fn symbol_repair_refusal_deadline_and_retry_preserve_snapshot() {
        let (root, context) = fixture();
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        let snapshot = context.store.index_snapshot().unwrap();
        let generation = context.store.l1_generation().unwrap();
        fs::write(root.path().join("source.rs"), "\n\npub fn before() {}\n").unwrap();
        fs::write(root.path().join(".deepkitignore"), "source.rs\n").unwrap();
        assert_eq!(
            context.symbol("before").await.unwrap_err().code(),
            "source_unavailable"
        );
        fs::remove_file(root.path().join(".deepkitignore")).unwrap();
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nquery_timeout_ms=100\n",
        )
        .unwrap();
        let bounded = Context::load(root.path()).unwrap();
        let writer = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let start = Instant::now();
        assert_eq!(
            bounded.symbol("before").await.unwrap_err().code(),
            "timeout"
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(
            context.store.index_snapshot().unwrap().files,
            snapshot.files
        );
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
        let repaired = context.symbol("before").await.unwrap();
        assert_eq!(repaired[0]["lines"], serde_json::json!([3, 3]));
        assert_eq!(context.store.l1_generation().unwrap(), generation + 1);
    }

    #[tokio::test]
    async fn reference_repair_reretrieves_lines_counts_and_removed_relationships() {
        let (root, context) = fixture();
        let path = root.path().join("source.rs");
        let original = "pub fn caller() { before(); }\npub fn second() { before(); }\n";
        fs::write(&path, original).unwrap();
        // Start from a completed branch so this exercises selected-file repair.
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        let initial = context.refs("before", None, false).await.unwrap();
        assert_eq!(initial["groups"][0]["lines"], serde_json::json!([1, 2]));
        let generation = context.store.l1_generation().unwrap();
        let other = context
            .store
            .index_snapshot()
            .unwrap()
            .files
            .into_iter()
            .find(|file| file.path == "other.rs")
            .unwrap();
        fs::write(&path, format!("\n\n{original}")).unwrap();
        context
            .store
            .writer()
            .unwrap()
            .execute("UPDATE files SET l1_pending=1 WHERE path='source.rs'", [])
            .unwrap();
        let repaired = context.refs("before", Some(1), false).await.unwrap();
        assert_eq!(repaired["groups"][0]["lines"], serde_json::json!([3, 4]));
        assert_eq!(repaired["total"], 2);
        assert_eq!(repaired["returned_total"], 2);
        assert_eq!(repaired["index"]["partial"], false);
        assert_eq!(repaired["index"]["lazy_resync"]["published_files"], 1);
        assert_eq!(repaired["index"]["l1_pending_files"], 0);
        assert_eq!(repaired["index"]["indexed_head"], "");
        assert_eq!(context.store.l1_generation().unwrap(), generation + 1);
        assert_eq!(
            context
                .store
                .index_snapshot()
                .unwrap()
                .files
                .into_iter()
                .find(|file| file.path == "other.rs")
                .unwrap(),
            other
        );
        fs::write(&path, "pub fn caller() {}\n").unwrap();
        // A filtered-out group cannot select a file for repair.
        let filtered = context.refs("before", Some(1), true).await.unwrap();
        assert_eq!(filtered["groups"], serde_json::json!([]));
        assert_eq!(filtered["index"]["lazy_resync"]["attempted_files"], 0);
        let removed = context.refs("before", Some(1), false).await.unwrap();
        assert_eq!(removed["groups"], serde_json::json!([]));
        assert_eq!(removed["total"], 0);
        assert_eq!(removed["indexed_total"], 0);
        assert_eq!(removed["index"]["partial"], false);
        assert_eq!(removed["index"]["lazy_resync"]["published_files"], 1);
    }

    #[tokio::test]
    async fn reference_repair_refusal_and_writer_timeout_preserve_snapshot_and_retry() {
        let (root, context) = fixture();
        let path = root.path().join("source.rs");
        fs::write(&path, "pub fn caller() { before(); }\n").unwrap();
        // Start from a completed branch so this exercises selected-file repair.
        assert!(branch(&context, Default::default()).unwrap().unwrap().1);
        let generation = context.store.l1_generation().unwrap();
        let snapshot = context.store.index_snapshot().unwrap();
        fs::write(&path, "\n\npub fn caller() { before(); }\n").unwrap();
        fs::write(root.path().join(".deepkitignore"), "source.rs\n").unwrap();
        let refused = context.refs("before", None, false).await.unwrap();
        assert_eq!(refused["groups"], serde_json::json!([]));
        assert!(refused["total"].is_null());
        assert_eq!(refused["index"]["partial"], true);
        assert_eq!(refused["index"]["lazy_resync"]["refused_files"], 1);
        fs::remove_file(root.path().join(".deepkitignore")).unwrap();
        fs::write(
            root.path().join(".deepkit/config.local.toml"),
            "[limits]\nquery_timeout_ms=100\n",
        )
        .unwrap();
        let bounded = Context::load(root.path()).unwrap();
        let writer = rusqlite::Connection::open(root.path().join(".deepkit/code.db")).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let start = Instant::now();
        let timed = bounded.refs("before", None, false).await.unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(timed["index"]["reason"], "query_timeout");
        assert!(timed["total"].is_null());
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        assert_eq!(
            context.store.index_snapshot().unwrap().files,
            snapshot.files
        );
        fs::remove_file(root.path().join(".deepkit/config.local.toml")).unwrap();
        let repaired = context.refs("before", None, false).await.unwrap();
        assert_eq!(repaired["groups"][0]["lines"], serde_json::json!([3]));
        assert_eq!(repaired["index"]["partial"], false);
        assert_eq!(context.store.l1_generation().unwrap(), generation + 1);
    }

    #[test]
    fn targeted_orchestration_repairs_pending_and_changed_sources_without_completing_head() {
        let (root, context) = fixture();
        let before = context.store.index_snapshot().unwrap();
        for content in ["pub fn before() {}\n", "\n\npub fn after() {}\n"] {
            fs::write(root.path().join("source.rs"), content).unwrap();
            assert!(target(&context, "source.rs", Default::default()).unwrap());
            let snapshot = context.store.index_snapshot().unwrap();
            assert_eq!(snapshot.dirs, before.dirs);
            assert_eq!(
                snapshot.files.iter().find(|f| f.path == "other.rs"),
                before.files.iter().find(|f| f.path == "other.rs")
            );
            assert_eq!(context.store.pending_l1_paths().unwrap(), ["other.rs"]);
            let file = snapshot
                .files
                .iter()
                .find(|f| f.path == "source.rs")
                .unwrap();
            assert_eq!(file.blake3, blake3::hash(content.as_bytes()).as_bytes());
            assert!(
                !context
                    .store
                    .file_occurrence_keys(file.id)
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(
                context.store.get_meta("indexed_head").unwrap().as_deref(),
                Some("retained-head")
            );
        }
        let result = super::super::navigation::symbol(&context.store, "after").unwrap();
        assert!(!result.is_empty());
        assert!(
            super::super::navigation::symbol(&context.store, "before")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn targeted_orchestration_refuses_unknown_ignored_large_disabled_and_stopped_work() {
        let (root, mut context) = fixture();
        let before = context.store.index_snapshot().unwrap();
        let generation = context.store.l1_generation().unwrap();
        fs::write(root.path().join("new.rs"), "pub fn new() {}\n").unwrap();
        assert!(!target(&context, "new.rs", Default::default()).unwrap());
        fs::write(root.path().join(".deepkitignore"), "source.rs\n").unwrap();
        assert!(!target(&context, "source.rs", Default::default()).unwrap());
        fs::remove_file(root.path().join(".deepkitignore")).unwrap();
        fs::write(root.path().join("source.rs"), vec![b' '; 256 * 1024]).unwrap();
        assert!(!target(&context, "source.rs", Default::default()).unwrap());
        context.config.search.lazy_resync = false;
        assert!(!target(&context, "source.rs", Default::default()).unwrap());
        context.config.search.lazy_resync = true;
        let cancelled = AtomicBool::new(true);
        for work in [
            sync::l1::Work {
                deadline: Some(Instant::now()),
                cancelled: None,
            },
            sync::l1::Work {
                deadline: None,
                cancelled: Some(&cancelled),
            },
        ] {
            assert!(target(&context, "source.rs", work).is_err());
        }
        assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
        assert_eq!(context.store.l1_generation().unwrap(), generation);
        fs::write(root.path().join("source.rs"), "pub fn retry() {}\n").unwrap();
        assert!(target(&context, "source.rs", Default::default()).unwrap());
    }

    #[test]
    fn repair_configuration_reload_refuses_policy_limits_roots_and_invalid_config_then_retries() {
        let (root, context) = fixture();
        let local = root.path().join(".deepkit/config.local.toml");
        let before = context.store.index_snapshot().unwrap();
        let generation = context.store.l1_generation().unwrap();
        for text in [
            "[search]\nlazy_resync=false\n",
            "[limits]\nmax_units_per_file=1\n",
            "[limits]\nwal_max_bytes='1 MiB'\n",
            "[index]\nroot='elsewhere'\n",
            "[store]\npath='.deepkit/other.db'\n",
            "[tokenize]\nsplit_subtokens=false\n",
            "invalid = [",
        ] {
            fs::write(&local, text).unwrap();
            assert!(
                matches!(
                    validate_configuration(&context, Default::default()),
                    Err(store::Error::L1InputsChanged)
                ),
                "{text}"
            );
            assert!(target(&context, "source.rs", Default::default()).is_err());
            assert_eq!(context.store.index_snapshot().unwrap().files, before.files);
            assert_eq!(context.store.l1_generation().unwrap(), generation);
        }
        fs::remove_file(local).unwrap();
        validate_configuration(&context, Default::default()).unwrap();
        assert!(target(&context, "source.rs", Default::default()).unwrap());
    }

    #[test]
    fn late_configuration_changes_roll_back_each_targeted_publication_and_allow_retry() {
        use std::cell::Cell;
        for stage in ["catalogue", "completion"] {
            let (root, context) = fixture();
            assert!(target(&context, "source.rs", Default::default()).unwrap());
            fs::write(root.path().join("source.rs"), "pub fn replacement() {}\n").unwrap();
            let generation = context.store.l1_generation().unwrap();
            let snapshot = context.store.index_snapshot().unwrap();
            let previous = snapshot
                .files
                .iter()
                .find(|f| f.path == "source.rs")
                .unwrap();
            let keys = context.store.file_occurrence_keys(previous.id).unwrap();
            let file = sync::l1::observe_source(
                &context.root,
                "source.rs",
                1024,
                Default::default(),
                |_| {},
            )
            .unwrap();
            let calls = Cell::new(0);
            let local = root.path().join(".deepkit/config.local.toml");
            let change_at_commit = || {
                calls.set(calls.get() + 1);
                // Catalogue validates before admission as well as inside the
                // transaction; Processor validates twice inside publication.
                let final_call = if stage == "catalogue" { 3 } else { 2 };
                if calls.get() == final_call {
                    fs::write(&local, "[search]\nlazy_resync=false\n").unwrap();
                }
                validate_configuration(&context, Default::default())
            };
            let publication = sync::l1::publish_target(
                &context.store,
                sync::l1::TargetInput {
                    work: Default::default(),
                    file: &file,
                    snapshot: &snapshot,
                    generation,
                    root: &context.root,
                    settings: &context.config,
                },
                || {
                    if stage == "catalogue" {
                        change_at_commit()
                    } else {
                        validate_configuration(&context, Default::default())
                    }
                },
            );
            if stage == "catalogue" {
                assert!(publication.is_err());
                assert_eq!(calls.get(), 3);
                assert_eq!(
                    context.store.index_snapshot().unwrap().files,
                    snapshot.files
                );
                assert_eq!(context.store.l1_generation().unwrap(), generation);
            } else {
                let publication = publication.unwrap();
                let pending = context.store.index_snapshot().unwrap();
                let result = sync::l1::Processor::new(&context.root, &context.config)
                    .unwrap()
                    .process_target(
                        &context.store,
                        sync::l1::FileInput {
                            work: Default::default(),
                            file: &file,
                            fence: store::l1::Fence {
                                generation: publication.generation,
                                file_id: previous.id,
                                path: "source.rs",
                                content_hash: file.content_hash(),
                                fingerprint: &publication.fingerprints["source.rs"],
                            },
                        },
                        change_at_commit,
                    );
                assert!(result.is_err());
                assert_eq!(calls.get(), 2);
                assert_eq!(context.store.index_snapshot().unwrap().files, pending.files);
                assert!(
                    context
                        .store
                        .pending_l1_paths()
                        .unwrap()
                        .iter()
                        .any(|p| p == "source.rs")
                );
            }
            assert_eq!(
                context.store.file_occurrence_keys(previous.id).unwrap(),
                keys
            );
            assert_eq!(
                context.store.get_meta("indexed_head").unwrap().as_deref(),
                Some("retained-head")
            );
            fs::remove_file(local).unwrap();
            assert!(target(&context, "source.rs", Default::default()).unwrap());
            assert!(
                !super::super::navigation::symbol(&context.store, "replacement")
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
