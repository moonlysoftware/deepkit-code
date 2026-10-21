//! Cooperative HEAD observation for query freshness and reconciliation; v1.md §4.2, §4.6.

use super::{Error, GitHead, l1::Work};
use std::path::Path;

/// Compares completed and observed commit identity, including transitions to or
/// from unborn/non-Git state. Empty metadata represents no completed commit.
/// Equal commit IDs remain equal when a branch is renamed or HEAD is detached.
pub(crate) fn changed(indexed: Option<&str>, observed: Option<&str>) -> bool {
    indexed.filter(|id| !id.is_empty()) != observed.filter(|id| !id.is_empty())
}

/// Reads HEAD without loading the Git index or trees. Checks the original work
/// signal around Git calls; individual filesystem calls are not preempted.
/// This is an observation, not proof of L1 completion or a stable checkout.
pub(crate) fn read(root: &Path, work: Work<'_>) -> Result<Option<GitHead>, Error> {
    read_checked(root, || work.check())
}

fn read_checked(
    root: &Path,
    mut check: impl FnMut() -> Result<(), Error>,
) -> Result<Option<GitHead>, Error> {
    check()?;
    let repo = discover(root);
    check()?;
    let Some(repo) = repo? else {
        return Ok(None);
    };
    let head = resolve(&repo, root);
    check()?;
    head
}

/// Shares repository discovery semantics with full indexing, including non-Git roots.
pub(super) fn discover(root: &Path) -> Result<Option<gix::Repository>, Error> {
    match gix::discover(root) {
        Ok(repo) => Ok(Some(repo)),
        Err(gix::discover::Error::Discover(
            gix::discover::upwards::Error::NoGitRepository { .. }
            | gix::discover::upwards::Error::NoGitRepositoryWithinCeiling { .. }
            | gix::discover::upwards::Error::NoGitRepositoryWithinFs { .. },
        )) => Ok(None),
        Err(error) => Err(Error::Git {
            path: root.to_path_buf(),
            message: error.to_string(),
        }),
    }
}

/// Resolves attached, detached and unborn HEAD identically for indexing and queries.
pub(super) fn resolve(repo: &gix::Repository, root: &Path) -> Result<Option<GitHead>, Error> {
    let head = repo.head().map_err(|error| Error::Git {
        path: root.to_path_buf(),
        message: error.to_string(),
    })?;
    Ok(head.id().map(|id| GitHead {
        id: id.to_hex().to_string(),
        reference: head.referent_name().map(ToString::to_string),
        detached: head.is_detached(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        process::Command,
        sync::atomic::{AtomicBool, Ordering},
        time::Instant,
    };

    fn git(root: &Path, args: &[&str]) {
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(root)
                .args(args)
                .output()
                .unwrap()
                .status
                .success()
        );
    }

    fn fixture(root: &Path) {
        assert!(
            Command::new("bash")
                .arg(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/scripts/fixtures/make-git-repo.sh"
                ))
                .arg(root)
                .output()
                .unwrap()
                .status
                .success()
        );
    }

    #[test]
    fn reconciliation_detects_first_commit_and_unborn_checkout() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path());
        let original = read(root.path(), Work::default()).unwrap().unwrap().id;
        for indexed in [None, Some("")] {
            assert!(
                super::super::inspect_git(root.path(), indexed)
                    .unwrap()
                    .unwrap()
                    .needs_full_rescan
            );
        }
        git(root.path(), &["symbolic-ref", "HEAD", "refs/heads/unborn"]);
        let unborn = super::super::inspect_git(root.path(), Some(&original))
            .unwrap()
            .unwrap();
        assert!(unborn.head.is_none());
        assert!(unborn.needs_full_rescan);
        assert!(
            !super::super::inspect_git(root.path(), Some(""))
                .unwrap()
                .unwrap()
                .needs_full_rescan
        );
        git(root.path(), &["checkout", "--quiet", "--detach", &original]);
        assert!(
            !super::super::inspect_git(root.path(), Some(&original))
                .unwrap()
                .unwrap()
                .needs_full_rescan
        );
    }

    #[test]
    fn head_observation_tracks_checkout_detach_and_ignores_corrupt_index() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path());
        let main = read(root.path(), Work::default()).unwrap().unwrap();
        assert_eq!(main.reference.as_deref(), Some("refs/heads/main"));
        assert!(!main.detached);
        git(root.path(), &["checkout", "--quiet", "alternate"]);
        let alternate = read(root.path(), Work::default()).unwrap().unwrap();
        assert_ne!(alternate.id, main.id);
        git(root.path(), &["checkout", "--quiet", "--detach", &main.id]);
        let detached = read(root.path(), Work::default()).unwrap().unwrap();
        assert_eq!(detached.id, main.id);
        assert!(detached.detached);
        assert_eq!(detached.reference, None);
        fs::write(root.path().join(".git/index"), b"invalid index").unwrap();
        assert_eq!(read(root.path(), Work::default()).unwrap(), Some(detached));
        // The patched full-index loader now refuses the original13-byte input
        // without a panic; HEAD-only observation remains independent.
        assert!(super::super::inspect_git(root.path(), Some(&main.id)).is_err());
        assert!(!root.path().join(".deepkit").exists());
    }

    #[test]
    fn head_observation_preserves_absent_unborn_and_invalid_head_semantics() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(read(root.path(), Work::default()).unwrap(), None);
        git(root.path(), &["init", "--initial-branch=main"]);
        assert_eq!(read(root.path(), Work::default()).unwrap(), None);
        // A syntactically valid symbolic HEAD pointing at a malformed ref is an error.
        fs::write(
            root.path().join(".git/refs/heads/main"),
            b"invalid object id\n",
        )
        .unwrap();
        assert!(matches!(
            read(root.path(), Work::default()),
            Err(Error::Git { .. })
        ));
    }

    #[test]
    fn head_observation_stops_before_discovery_and_after_each_git_stage() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path());
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            read(
                root.path(),
                Work {
                    cancelled: Some(&cancelled),
                    deadline: None
                }
            ),
            Err(Error::WorkStopped)
        ));
        assert!(matches!(
            read(
                root.path(),
                Work {
                    cancelled: None,
                    deadline: Some(Instant::now())
                }
            ),
            Err(Error::WorkStopped)
        ));
        for stop_at in 2..=3 {
            cancelled.store(false, Ordering::Release);
            let mut checks = 0;
            let result = read_checked(root.path(), || {
                checks += 1;
                if checks == stop_at {
                    cancelled.store(true, Ordering::Release);
                }
                Work {
                    cancelled: Some(&cancelled),
                    deadline: None,
                }
                .check()
            });
            assert!(matches!(result, Err(Error::WorkStopped)));
            assert_eq!(checks, stop_at);
            cancelled.store(false, Ordering::Release);
            assert!(
                read(
                    root.path(),
                    Work {
                        cancelled: Some(&cancelled),
                        deadline: None
                    }
                )
                .unwrap()
                .is_some()
            );
        }
    }
}
