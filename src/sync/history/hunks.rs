//! Bounded immutable member hunks for PR enrichment; v1 §6.8.
use super::*;
use std::collections::BTreeMap;

/// Bounds metadata enumeration, blob reads and the complete selected hunk text.
pub(crate) struct Limits {
    pub(crate) file_bytes: u64,
    pub(crate) files: u64,
    pub(crate) tokens: usize,
}
/// Selected complete hunks; omitted data is explicit and never a fabricated patch.
pub(crate) struct Selected {
    pub(crate) text: String,
    pub(crate) truncated: bool,
}

/// Reads first-parent changes only for caller-authorized paths, in supplied rank
/// order. The caller verifies commit ownership and current outbound policy first.
pub(crate) fn read(
    repo: &gix::Repository,
    id: gix::ObjectId,
    paths: &[String],
    limits: Limits,
    work: Work<'_>,
) -> Result<Selected, Error> {
    work.check()?;
    if git(repo.find_header(id), "inspect hunk commit")?.size() > limits.file_bytes {
        return Err(Error::Bound("hunk commit bytes"));
    }
    let commit = git(repo.find_commit(id), "read hunk commit")?;
    let after = git(commit.tree(), "read hunk tree")?;
    let before = commit
        .parent_ids()
        .next()
        .map(|id| {
            git(repo.find_commit(id.detach()), "read hunk parent")
                .and_then(|p| git(p.tree(), "read hunk parent tree"))
        })
        .transpose()?;
    let mut visitor = Changes {
        tracker: Default::default(),
        changes: Vec::new(),
        limit: limits.files,
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
        return Err(Error::Bound("hunk paths"));
    }
    git(diff, "diff hunk trees")?;
    let ranks = paths
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    visitor
        .changes
        .retain(|c| ranks.contains_key(c.path.as_str()));
    visitor
        .changes
        .sort_by_key(|c| ranks.get(c.path.as_str()).copied());
    let mut result = Selected {
        text: String::new(),
        truncated: false,
    };
    for change in visitor.changes {
        work.check()?;
        if change.gitlink {
            result.truncated = true;
            continue;
        }
        let before = blob(repo, change.before, limits.file_bytes)?;
        let after = blob(repo, change.after, limits.file_bytes)?;
        let (Some(before), Some(after)) = (before, after) else {
            result.truncated = true;
            continue;
        };
        let (Ok(before), Ok(after)) = (std::str::from_utf8(&before), std::str::from_utf8(&after))
        else {
            result.truncated = true;
            continue;
        };
        let input = gix::diff::blob::InternedInput::new(before, after);
        let diff = gix::diff::blob::diff_with_slider_heuristics(
            gix::diff::blob::Algorithm::Histogram,
            &input,
        );
        let old = before.lines().collect::<Vec<_>>();
        let new = after.lines().collect::<Vec<_>>();
        for hunk in diff.hunks() {
            work.check()?;
            let mut text = format!(
                "{}\n@@ -{},{} +{},{} @@\n",
                change.path,
                hunk.before.start + u32::from(!hunk.before.is_empty()),
                hunk.before.end - hunk.before.start,
                hunk.after.start + u32::from(!hunk.after.is_empty()),
                hunk.after.end - hunk.after.start
            );
            for line in old
                .get(hunk.before.start as usize..hunk.before.end as usize)
                .ok_or(Error::Git("validate old hunk range"))?
            {
                text.push('-');
                text.push_str(line);
                text.push('\n');
            }
            for line in new
                .get(hunk.after.start as usize..hunk.after.end as usize)
                .ok_or(Error::Git("validate new hunk range"))?
            {
                text.push('+');
                text.push_str(line);
                text.push('\n');
            }
            let previous = result.text.len();
            result.text.push_str(&text);
            if treat::count_tokens(&result.text) > limits.tokens {
                result.text.truncate(previous);
                result.truncated = true;
                return Ok(result);
            }
        }
    }
    work.check()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, process::Command, sync::atomic::AtomicBool};
    fn git_command(root: &Path, args: &[&str]) {
        let output = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    #[test]
    fn immutable_hunks_obey_authorized_paths_order_exact_token_cap_and_cancellation() {
        let root = tempfile::tempdir().unwrap();
        git_command(root.path(), &["init", "-q"]);
        git_command(root.path(), &["config", "user.name", "Fixture"]);
        git_command(
            root.path(),
            &["config", "user.email", "fixture@example.invalid"],
        );
        for name in ["a.rs", "b.rs", "private.rs"] {
            fs::write(root.path().join(name), "old line\n").unwrap();
        }
        git_command(root.path(), &["add", "."]);
        git_command(root.path(), &["commit", "-qm", "initial"]);
        fs::write(root.path().join("a.rs"), "public alpha 日本語\n").unwrap();
        fs::write(root.path().join("b.rs"), "public beta\n").unwrap();
        fs::write(root.path().join("private.rs"), "PRIVATE_SENTINEL\n").unwrap();
        git_command(root.path(), &["add", "."]);
        git_command(root.path(), &["commit", "-qm", "change"]);
        let repo = gix::open(root.path()).unwrap();
        let id = repo.head_id().unwrap().detach();
        let paths = vec!["b.rs".to_owned(), "a.rs".to_owned()];
        let limits = |tokens| Limits {
            file_bytes: 4096,
            files: 100,
            tokens,
        };
        let full = read(&repo, id, &paths, limits(6000), Work::default()).unwrap();
        assert!(!full.truncated);
        assert!(full.text.starts_with("b.rs\n"));
        assert!(full.text.contains("+public alpha 日本語"));
        assert!(!full.text.contains("PRIVATE_SENTINEL"));
        let count = treat::count_tokens(&full.text);
        assert_eq!(
            read(&repo, id, &paths, limits(count), Work::default())
                .unwrap()
                .text,
            full.text
        );
        let capped = read(&repo, id, &paths, limits(count - 1), Work::default()).unwrap();
        assert!(capped.truncated);
        assert!(treat::count_tokens(&capped.text) < count);
        let zero = read(&repo, id, &paths, limits(0), Work::default()).unwrap();
        assert!(zero.text.is_empty());
        assert!(zero.truncated);
        let cancelled = AtomicBool::new(true);
        assert!(
            read(
                &repo,
                id,
                &paths,
                limits(6000),
                Work {
                    deadline: None,
                    cancelled: Some(&cancelled)
                }
            )
            .is_err()
        );
        // The immutable object result does not use changed checkout bytes.
        fs::write(root.path().join("b.rs"), "checkout changed\n").unwrap();
        assert_eq!(
            read(&repo, id, &paths, limits(6000), Work::default())
                .unwrap()
                .text,
            full.text
        );
    }
}
