//! Bounded Git tree candidates for branch freshness; see v1.md §4.3.
use std::{collections::BTreeSet, ops::ControlFlow};

use super::{Error, l1::Work};

/// Returns changed paths without reading blob contents or the working index.
/// Missing/unborn/pruned history and oversized diffs fall back to a full scan.
pub(super) fn between(
    repo: &gix::Repository,
    old: Option<&str>,
    new: Option<&str>,
    work: Work<'_>,
) -> Result<Option<BTreeSet<String>>, Error> {
    work.check()?;
    let (Some(old), Some(new)) = (old, new) else {
        return Ok(None);
    };
    if old == new {
        return Ok(Some(BTreeSet::new()));
    }
    let tree = |id: &str| {
        repo.find_object(gix::ObjectId::from_hex(id.as_bytes()).ok()?)
            .ok()?
            .try_into_commit()
            .ok()?
            .tree()
            .ok()
    };
    let trees = (tree(old), tree(new));
    work.check()?;
    let (Some(old), Some(new)) = trees else {
        return Ok(None);
    };
    let mut visitor = Candidates {
        tracker: Default::default(),
        paths: BTreeSet::new(),
        complete: true,
        work,
    };
    // The tree-only API needs no attribute/filter resource cache and never opens
    // the working index, unlike the higher-level rewrite-aware diff platform.
    let result = gix::diff::tree(
        gix::objs::TreeRefIter::from_bytes(&old.data, old.id.kind()),
        gix::objs::TreeRefIter::from_bytes(&new.data, new.id.kind()),
        &mut gix::diff::tree::State::default(),
        &repo.objects,
        &mut visitor,
    );
    work.check()?;
    Ok((result.is_ok() && visitor.complete).then_some(visitor.paths))
}

struct Candidates<'a> {
    tracker: gix::diff::tree::Recorder,
    paths: BTreeSet<String>,
    complete: bool,
    work: Work<'a>,
}

impl gix::diff::tree::Visit for Candidates<'_> {
    fn pop_front_tracked_path_and_set_current(&mut self) {
        self.tracker.pop_front_tracked_path_and_set_current();
    }
    fn push_back_tracked_path_component(&mut self, component: &gix::bstr::BStr) {
        self.tracker.push_back_tracked_path_component(component);
    }
    fn push_path_component(&mut self, component: &gix::bstr::BStr) {
        self.tracker.push_path_component(component);
    }
    fn pop_path_component(&mut self) {
        self.tracker.pop_path_component();
    }
    fn visit(&mut self, change: gix::diff::tree::visit::Change) -> gix::diff::tree::visit::Action {
        use gix::diff::tree::visit::Change;
        if self.work.check().is_err() {
            return ControlFlow::Break(());
        }
        let mode = match change {
            Change::Addition { entry_mode, .. }
            | Change::Deletion { entry_mode, .. }
            | Change::Modification { entry_mode, .. } => entry_mode,
        };
        if mode.is_tree() {
            return ControlFlow::Continue(());
        }
        let Ok(path) = std::str::from_utf8(self.tracker.path()) else {
            self.complete = false;
            return ControlFlow::Break(());
        };
        self.paths.insert(super::normalize_path(path));
        // Bound this optimization; fallback discovery still uses configured caps.
        if self.paths.len() > super::DEFAULT_MAX_FILES {
            self.complete = false;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, process::Command, time::Instant};

    #[test]
    fn distant_tree_candidates_ignore_index_corruption_and_bound_work() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            Command::new("bash")
                .arg(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/scripts/fixtures/make-git-repo.sh"
                ))
                .arg("--distant")
                .arg(root.path())
                .output()
                .unwrap()
                .status
                .success()
        );
        let revision = |name: &str| {
            String::from_utf8(
                Command::new("git")
                    .arg("-C")
                    .arg(root.path())
                    .args(["rev-parse", name])
                    .output()
                    .unwrap()
                    .stdout,
            )
            .unwrap()
            .trim()
            .to_owned()
        };
        let old = revision("main");
        let new = revision("alternate");
        let repo = gix::discover(root.path()).unwrap();
        fs::write(root.path().join(".git/index"), b"truncated").unwrap();
        let paths = between(&repo, Some(&old), Some(&new), Work::default())
            .unwrap()
            .unwrap();
        assert_eq!(paths.len(), 1000);
        assert!(paths.contains("src/lib.rs"));
        assert!(paths.contains("src/generated/probe_0998.rs"));
        assert!(!paths.contains("README.md"));
        assert_eq!(
            between(&repo, Some(&old), Some(&old), Work::default()).unwrap(),
            Some(BTreeSet::new())
        );
        assert!(
            between(&repo, Some("missing"), Some(&new), Work::default())
                .unwrap()
                .is_none()
        );
        assert!(
            between(&repo, None, Some(&new), Work::default())
                .unwrap()
                .is_none()
        );
        assert!(matches!(
            between(
                &repo,
                Some(&old),
                Some(&new),
                Work {
                    deadline: Some(Instant::now()),
                    cancelled: None
                }
            ),
            Err(Error::WorkStopped)
        ));
    }
}
