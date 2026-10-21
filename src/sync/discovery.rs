//! Query-owned discovery candidates for branch reconciliation; v1.md §4.1–4.6.

use super::{Error, GitHead, discovery_builder, head, l1::Work};
use std::{collections::BTreeSet, fs, path::Path};

/// A bounded observation of paths, not proof of source or indexed-HEAD completion.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Candidates {
    pub(crate) head: Option<GitHead>,
    pub(crate) paths: Vec<String>,
    pub(crate) max_files_reached: bool,
    pub(crate) skipped_non_unicode_paths: usize,
}

/// Finds regular-file candidates using current shared ignore policy without loading
/// the Git index or reading source contents. Retains at most `max_files` paths;
/// one extra regular file proves overflow. Cancellation discards the observation.
/// Checks bracket walker calls, including exhaustion; individual filesystem calls,
/// ignore parsing and filtered sibling enumeration remain cooperative boundaries.
/// Callers must revalidate policy/source/configuration/generation before publication.
pub(crate) fn discover(root: &Path, max_files: usize, work: Work<'_>) -> Result<Candidates, Error> {
    discover_checked(root, max_files, work, || work.check())
}

fn discover_checked(
    root: &Path,
    max_files: usize,
    work: Work<'_>,
    check: impl FnMut() -> Result<(), Error>,
) -> Result<Candidates, Error> {
    discover_filtered(root, None, max_files, work, check)
}

/// Discovers only affected paths and descendants, applying shared ignores before
/// any source read. Ancestors are traversed only to reach the requested scope.
pub(crate) fn discover_paths(
    root: &Path,
    paths: &BTreeSet<String>,
    max_files: usize,
    work: Work<'_>,
) -> Result<Candidates, Error> {
    if paths.iter().any(|path| {
        Path::new(path)
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    }) {
        return Err(Error::Walk {
            path: root.to_path_buf(),
            message: "watch paths must be repository relative".into(),
        });
    }
    discover_filtered(root, Some(paths), max_files, work, || work.check())
}

/// Tests path-component ancestry, so a directory event never includes a sibling
/// that merely shares its string prefix.
pub(crate) fn affected(paths: &BTreeSet<String>, path: &str) -> bool {
    paths
        .iter()
        .any(|parent| Path::new(path).starts_with(parent))
}

// Ordered paths avoid scanning every requested file for every walker entry.
// An entry either has a requested ancestor or prefixes the first target at/after it.
fn intersects_targets(targets: &BTreeSet<std::path::PathBuf>, entry: &Path) -> bool {
    entry.ancestors().any(|parent| targets.contains(parent))
        || targets
            .range(entry.to_path_buf()..)
            .next()
            .is_some_and(|target| target.starts_with(entry))
}

fn discover_filtered(
    root: &Path,
    paths: Option<&BTreeSet<String>>,
    max_files: usize,
    work: Work<'_>,
    mut check: impl FnMut() -> Result<(), Error>,
) -> Result<Candidates, Error> {
    check()?;
    let root = fs::canonicalize(root).map_err(|source| Error::RootIo {
        path: root.to_path_buf(),
        source,
    })?;
    check()?;
    if !root.is_dir() {
        return Err(Error::RootNotDirectory { path: root });
    }
    let observed_head = head::read(&root, work)?;
    check()?;
    let mut builder = discovery_builder(&root);
    if let Some(paths) = paths {
        let targets = paths
            .iter()
            .map(|path| root.join(path))
            .collect::<BTreeSet<_>>();
        builder.filter_entry(move |entry| {
            super::discovery_entry(entry) && intersects_targets(&targets, entry.path())
        });
    }
    let mut walker = builder.build();
    let mut result = Candidates {
        head: observed_head,
        paths: Vec::new(),
        max_files_reached: false,
        skipped_non_unicode_paths: 0,
    };
    let mut regular_files = 0;
    loop {
        check()?;
        let entry = walker.next();
        check()?;
        let Some(entry) = entry else { break };
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
        if entry.path_is_symlink() || !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        if regular_files == max_files {
            result.max_files_reached = true;
            break;
        }
        regular_files += 1;
        let relative = entry
            .path()
            .strip_prefix(&root)
            .map_err(|error| Error::Walk {
                path: entry.path().to_path_buf(),
                message: error.to_string(),
            })?;
        let Some(path) = relative.to_str() else {
            result.skipped_non_unicode_paths += 1;
            continue;
        };
        result.paths.push(super::normalize_path(path));
    }
    check()?;
    if head::read(&root, work)? != result.head {
        return Err(Error::HeadChanged);
    }
    check()?;
    result.paths.sort_unstable();
    check()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{process::Command, sync::atomic::AtomicBool, time::Instant};

    proptest::proptest! {
        #[test]
        fn ordered_scope_lookup_matches_component_ancestry(
            paths in proptest::collection::vec("([a-z]{1,4}/){0,5}[a-z]{1,4}", 0..100),
            entry in "([a-z]{1,4}/){0,5}[a-z]{1,4}",
        ) {
            let targets = paths.iter().map(|p| Path::new("/repo").join(p))
                .collect::<BTreeSet<_>>();
            for path in paths.iter().chain(std::iter::once(&entry)) {
                let path = Path::new("/repo").join(path);
                for candidate in path.ancestors() {
                    let expected = targets.iter().any(|target|
                        target.starts_with(candidate) || candidate.starts_with(target));
                    proptest::prop_assert_eq!(intersects_targets(&targets, candidate), expected);
                }
            }
        }
    }

    fn write(root: &Path, path: &str, bytes: &[u8]) {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn candidates_match_walk_policy_reload_ignores_and_ignore_corrupt_git_index() {
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
        for path in [
            "new.rs",
            ".hidden.rs",
            "nested/kept.rs",
            "nested/ignored.rs",
            "custom.rs",
            "excluded.rs",
            ".deepkit/private.rs",
        ] {
            write(root.path(), path, b"fn source() {}\n");
        }
        write(root.path(), ".deepkitignore", b"custom.rs\n");
        write(root.path(), "nested/.ignore", b"ignored.rs\n");
        write(root.path(), ".git/info/exclude", b"excluded.rs\n");
        write(root.path(), ".git/index", b"short");
        let before = discover(root.path(), 100, Work::default()).unwrap();
        let scan = super::super::scan(root.path(), super::super::SyncLimits::default()).unwrap();
        assert_eq!(
            before.paths,
            scan.files()
                .map(|file| file.path.clone())
                .collect::<Vec<_>>()
        );
        assert!(before.head.is_some());
        assert!(!before.max_files_reached);
        write(root.path(), "nested/.deepkitignore", b"kept.rs\n");
        let after = discover(root.path(), 100, Work::default()).unwrap();
        assert!(!after.paths.iter().any(|path| path == "nested/kept.rs"));
        assert!(after.paths.iter().any(|path| path == "new.rs"));
        assert_eq!(fs::read(root.path().join(".git/index")).unwrap(), b"short");
        assert_eq!(
            fs::read(root.path().join(".deepkit/private.rs")).unwrap(),
            b"fn source() {}\n"
        );
    }

    #[test]
    fn candidates_distinguish_exact_cap_overflow_empty_and_ignore_failure() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            !discover(root.path(), 0, Work::default())
                .unwrap()
                .max_files_reached
        );
        for path in ["one.rs", "two.rs", "three.rs"] {
            write(root.path(), path, b"x");
        }
        for cap in 0..=4 {
            let result = discover(root.path(), cap, Work::default()).unwrap();
            assert_eq!(result.paths.len(), cap.min(3));
            assert_eq!(result.max_files_reached, cap < 3);
            assert_eq!(result.head, None);
        }
        write(root.path(), ".gitignore", b"[z-a]\n");
        assert!(matches!(
            discover(root.path(), 10, Work::default()),
            Err(Error::Walk { .. })
        ));
        fs::remove_file(root.path().join(".gitignore")).unwrap();
        assert_eq!(
            discover(root.path(), 10, Work::default())
                .unwrap()
                .paths
                .len(),
            3
        );
        assert!(!root.path().join(".deepkit").exists());
    }

    #[test]
    fn candidates_follow_checkout_add_delete_rename_and_untracked_paths() {
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
        let git = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .arg("-C")
                    .arg(root.path())
                    .args(args)
                    .output()
                    .unwrap()
                    .status
                    .success()
            );
        };
        let main = discover(root.path(), 100, Work::default()).unwrap();
        git(&["checkout", "--quiet", "alternate"]);
        git(&["mv", "src/lib.rs", "src/renamed.rs"]);
        git(&["rm", "--quiet", "README.md"]);
        write(root.path(), "branch/new.rs", b"fn new() {}\n");
        git(&["add", "branch/new.rs"]);
        git(&["commit", "--quiet", "-m", "alternate paths"]);
        write(root.path(), "untracked.rs", b"fn untracked() {}\n");
        let alternate = discover(root.path(), 100, Work::default()).unwrap();
        assert_ne!(alternate.head, main.head);
        assert_eq!(
            alternate.paths,
            ["branch/new.rs", "src/renamed.rs", "untracked.rs"]
        );
        git(&["checkout", "--quiet", "main"]);
        let restored = discover(root.path(), 100, Work::default()).unwrap();
        assert_eq!(restored.head, main.head);
        let mut expected = main.paths;
        expected.push("untracked.rs".into());
        expected.sort_unstable();
        assert_eq!(restored.paths, expected);
        assert!(!root.path().join(".deepkit").exists());
    }

    #[test]
    fn candidates_refuse_stopped_work_at_every_boundary_and_changed_head() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "source.rs", b"x");
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
                discover(&root.path().join("missing"), 10, work),
                Err(Error::WorkStopped)
            ));
        }
        let mut total = 0;
        discover_checked(root.path(), 10, Work::default(), || {
            total += 1;
            Ok(())
        })
        .unwrap();
        for stop_at in 1..=total {
            let mut checks = 0;
            let result = discover_checked(root.path(), 10, Work::default(), || {
                checks += 1;
                if checks == stop_at {
                    Err(Error::WorkStopped)
                } else {
                    Ok(())
                }
            });
            assert!(
                matches!(result, Err(Error::WorkStopped)),
                "boundary {stop_at}"
            );
            assert_eq!(checks, stop_at);
        }
        // Repository creation during discovery changes absent HEAD into a commit.
        let mut checks = 0;
        let result = discover_checked(root.path(), 10, Work::default(), || {
            checks += 1;
            if checks == 4 {
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
            }
            Ok(())
        });
        assert!(matches!(result, Err(Error::HeadChanged)));
        assert!(
            discover(root.path(), 100, Work::default())
                .unwrap()
                .head
                .is_some()
        );
    }

    #[cfg(unix)]
    #[test]
    fn candidates_skip_links_special_files_and_report_non_unicode_paths() {
        use std::os::unix::{ffi::OsStringExt, fs::symlink};
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        write(outside.path(), "external.rs", b"x");
        symlink(outside.path(), root.path().join("linked-dir")).unwrap();
        symlink(
            outside.path().join("external.rs"),
            root.path().join("linked.rs"),
        )
        .unwrap();
        assert!(
            Command::new("mkfifo")
                .arg(root.path().join("pipe.rs"))
                .status()
                .unwrap()
                .success()
        );
        let non_unicode = root
            .path()
            .join(std::ffi::OsString::from_vec(b"bad-\xff.rs".to_vec()));
        let non_unicode_created = match fs::write(&non_unicode, b"x") {
            Ok(()) => true,
            // APFS cannot represent this fixture; still exercise links and FIFO
            // filtering, and require no invented non-Unicode skip count.
            Err(error) if cfg!(target_os = "macos") && error.raw_os_error() == Some(92) => false,
            Err(error) => panic!("creating non-Unicode fixture: {error}"),
        };
        let result = discover(root.path(), 1, Work::default()).unwrap();
        assert!(result.paths.is_empty());
        assert_eq!(
            result.skipped_non_unicode_paths,
            usize::from(non_unicode_created)
        );
        assert!(!result.max_files_reached);
        assert_eq!(
            discover(root.path(), 0, Work::default())
                .unwrap()
                .max_files_reached,
            non_unicode_created,
        );
    }
}
