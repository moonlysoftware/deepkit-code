//! Cooperative two-tip ancestry painting for exact PR membership; v1.md §4.4.
use crate::sync::{history::Error, l1::Work};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

struct Node {
    parents: Vec<gix::ObjectId>,
    time: i64,
    flags: u8,
}
fn node(
    repo: &gix::Repository,
    id: gix::ObjectId,
    maximum: u64,
    work: Work<'_>,
) -> Result<Node, Error> {
    work.check()?;
    let header = repo
        .find_header(id)
        .map_err(|_| Error::Git("read PR ancestor header"))?;
    if header.size() > maximum {
        return Err(Error::Bound("commit_metadata"));
    }
    let commit = repo
        .find_commit(id)
        .map_err(|_| Error::Git("read PR ancestor"))?;
    let time = commit
        .time()
        .map_err(|_| Error::Git("read PR ancestor time"))?
        .seconds;
    let parents = commit.parent_ids().map(|id| id.detach()).collect();
    work.check()?;
    Ok(Node {
        parents,
        time,
        flags: 0,
    })
}

/// Finds ancestors(head) minus ancestors(base), stopping after the member cap.
/// Paint both sides to their overlap before yielding, so skewed dates do not hide
/// exclusive ancestors. Poll each object/edge; gix's hidden-frontier pass cannot.
pub(super) fn members(
    repo: &gix::Repository,
    base: gix::ObjectId,
    head: gix::ObjectId,
    limit: usize,
    maximum: u64,
    work: Work<'_>,
) -> Result<Option<Vec<gix::ObjectId>>, Error> {
    let mut graph = BTreeMap::<gix::ObjectId, Node>::new();
    let mut queue = BinaryHeap::new();
    for (id, flag) in [(head, 1), (base, 2)] {
        let n = graph.entry(id).or_insert(node(repo, id, maximum, work)?);
        n.flags |= flag;
        queue.push((n.time, id));
    }
    while queue
        .iter()
        .any(|(_, id)| graph.get(id).is_some_and(|n| n.flags & 4 == 0))
    {
        work.check()?;
        let Some((_, id)) = queue.pop() else {
            break;
        };
        let Some(n) = graph.get(&id) else {
            return Err(Error::Git("retain PR ancestry frontier"));
        };
        let flags = if n.flags == 3 { 7 } else { n.flags };
        let parents = n.parents.clone();
        for parent in parents {
            work.check()?;
            if let std::collections::btree_map::Entry::Vacant(entry) = graph.entry(parent) {
                entry.insert(node(repo, parent, maximum, work)?);
            }
            let Some(n) = graph.get_mut(&parent) else {
                return Err(Error::Git("retain PR ancestor"));
            };
            if n.flags & flags != flags {
                n.flags |= flags;
                queue.push((n.time, parent));
            }
        }
    }
    let mut pending = vec![head];
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    while let Some(id) = pending.pop() {
        work.check()?;
        if !seen.insert(id) {
            continue;
        }
        if graph.get(&id).is_some_and(|n| n.flags & 3 == 3) {
            continue;
        }
        if result.len() >= limit {
            return Ok(None);
        }
        if let std::collections::btree_map::Entry::Vacant(entry) = graph.entry(id) {
            entry.insert(node(repo, id, maximum, work)?);
        }
        let Some(n) = graph.get(&id) else {
            return Err(Error::Git("retain PR member"));
        };
        pending.extend(n.parents.iter().copied());
        result.push(id);
    }
    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    #[test]
    fn skewed_dag_members_match_git_set_difference_and_cancel_before_work() {
        let root = tempfile::tempdir().unwrap();
        let git = |args: &[&str], date: &str| {
            let output = Command::new("git")
                .current_dir(root.path())
                .env("GIT_AUTHOR_DATE", date)
                .env("GIT_COMMITTER_DATE", date)
                .env("GIT_AUTHOR_NAME", "Ancestry")
                .env("GIT_AUTHOR_EMAIL", "ancestry@example.invalid")
                .env("GIT_COMMITTER_NAME", "Ancestry")
                .env("GIT_COMMITTER_EMAIL", "ancestry@example.invalid")
                .args(args)
                .stdin(Stdio::null())
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_owned()
        };
        git(&["init", "-q"], "2026-01-01T12:00:00Z");
        let tree = git(&["mktree"], "2026-01-01T12:00:00Z");
        let mut ids: Vec<String> = Vec::new();
        for n in 0..35 {
            let mut args = vec!["commit-tree", &tree, "-m", "node"];
            if n > 0 {
                args.extend(["-p", ids[(n - 1) / 2].as_str()]);
            }
            if n > 3 && n % 5 == 0 {
                args.extend(["-p", ids[n - 2].as_str()]);
            }
            let date = if n % 2 == 0 {
                "2030-01-01T12:00:00Z"
            } else {
                "2020-01-01T12:00:00Z"
            };
            ids.push(git(&args, date));
        }
        let repo = gix::discover(root.path()).unwrap();
        for (a, b) in [(34, 33), (25, 12), (15, 28), (21, 19), (2, 31)] {
            let expected = git(
                &["rev-list", &ids[b], &format!("^{}", ids[a])],
                "2026-01-01T12:00:00Z",
            )
            .lines()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
            let oid = |i: usize| gix::ObjectId::from_hex(ids[i].as_bytes()).unwrap();
            let actual = members(&repo, oid(a), oid(b), 500, 1_000_000, Work::default())
                .unwrap()
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .collect::<BTreeSet<_>>();
            assert_eq!(actual, expected, "{a} {b}");
            let stopped = std::sync::atomic::AtomicBool::new(true);
            assert!(matches!(
                members(
                    &repo,
                    oid(a),
                    oid(b),
                    500,
                    1_000_000,
                    Work {
                        deadline: None,
                        cancelled: Some(&stopped)
                    }
                ),
                Err(Error::Sync(crate::sync::Error::WorkStopped))
            ));
        }
    }
}
