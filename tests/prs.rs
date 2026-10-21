//! Public PR reconstruction, ownership, caps and recent contracts; v1.md §4.4/§10.1.
use rusqlite::Connection;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn fixture(styles: &[&str]) -> (tempfile::TempDir, Vec<Value>) {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("histories");
    let result = Command::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/fixtures/make-histories.sh"),
    )
    .arg(&output)
    .args(styles)
    .output()
    .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let values = serde_json::from_slice(&fs::read(output.join("expected.json")).unwrap()).unwrap();
    (temp, values)
}
fn root(value: &Value) -> PathBuf {
    PathBuf::from(value["root"].as_str().unwrap())
}
fn command(root: &Path, args: &[&str]) -> Output {
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
        .env("DKC_MODELS_DIR", root.join("absent-model-cache"))
        .args(args)
        .output()
        .unwrap()
}
fn success(root: &Path, args: &[&str]) -> Value {
    let output = command(root, args);
    assert!(
        output.status.success(),
        "{args:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn index(root: &Path) -> Value {
    success(root, &["index", "--level", "1", "--json"])
}
fn append(root: &Path, config: &str) {
    let p = root.join(".deepkit/config.toml");
    fs::write(&p, fs::read_to_string(&p).unwrap() + config).unwrap();
}
fn db(root: &Path) -> Connection {
    Connection::open(root.join(".deepkit/code.db")).unwrap()
}

#[test]
fn generated_styles_reconstruct_exact_members_and_preserve_rebase_and_large_fallback() {
    let (_temp, fixtures) = fixture(&[]);
    for expected in fixtures {
        let root = root(&expected);
        let report = index(&root);
        let db = db(&root);
        let rows=db.prepare("SELECT lower(hex(merge_sha)),number,strategy,id,commit_count,unit_id FROM prs ORDER BY id").unwrap().query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,Option<i64>>(1)?,r.get::<_,String>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?,r.get::<_,i64>(5)?))).unwrap().collect::<Result<Vec<_>,_>>().unwrap();
        let expected_prs = expected["prs"].as_array().unwrap();
        assert_eq!(
            rows.len(),
            expected_prs.len(),
            "{} {report}",
            expected["style"]
        );
        for (merge, number, strategy, id, count, unit) in rows {
            let wanted = &expected_prs[0];
            assert_eq!(merge, wanted["merge"]);
            assert_eq!(json!(number), wanted["number"]);
            assert_eq!(strategy, wanted["strategy"]);
            let mut members = db
                .prepare("SELECT lower(hex(sha)) FROM commits WHERE pr_id=?1 ORDER BY sha")
                .unwrap()
                .query_map([id], |r| r.get::<_, String>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            members.sort();
            let mut wanted_members = wanted["members"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            wanted_members.sort();
            assert_eq!(members, wanted_members);
            assert_eq!(count, members.len() as i64);
            assert_eq!(
                db.query_row(
                    "SELECT count(*) FROM unit_sources WHERE unit_id=?1 AND source_kind='commit'",
                    [unit],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                count
            );
            let recent = success(&root, &["recent", "billing", "--json"]);
            assert_eq!(recent.as_array().unwrap().len(), 1, "{recent}");
            assert_eq!(recent[0]["kind"], "pr");
            let opened = success(
                &root,
                &["open", recent[0]["id"].as_str().unwrap(), "--json"],
            );
            assert_eq!(opened["kind"], "pr");
            assert_eq!(opened["sources"].as_array().unwrap().len(), count as usize);
            assert!(opened["text"].as_str().unwrap().contains("billing.rs"));
            let commits = success(&root, &["recent", "billing", "--commits", "--json"]);
            assert_eq!(
                commits.as_array().unwrap().len(),
                count as usize + usize::from(strategy != "squash")
            );
            assert!(
                commits
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|v| v["kind"] == "commit")
            );
            let unfiltered = success(&root, &["recent", "--json"]);
            assert_eq!(unfiltered[0]["title"], "Uncovered follow-up");
            assert!(
                unfiltered
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v["kind"] == "pr")
            );
        }
        if expected["style"] == "over-cap" {
            assert_eq!(report["prs"]["deferred"]["max_pr_commits"], 1);
        }
        let before: i64 = db
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get(0))
            .unwrap();
        let repeat = index(&root);
        assert_eq!(repeat["prs"]["inserted"], 0);
        assert_eq!(repeat["prs"]["visited"], 0);
        assert_eq!(
            before,
            db.query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))
                .unwrap()
        );
    }
}

#[test]
fn pr_caps_retain_rows_and_raised_member_cap_retries_without_duplicate_ownership() {
    let (_temp, fixtures) = fixture(&["github-merge"]);
    let root = root(&fixtures[0]);
    append(&root, "[limits]\nmax_pr_commits=1\n");
    assert_eq!(index(&root)["prs"]["deferred"]["max_pr_commits"], 1);
    assert_eq!(
        db(&root)
            .query_row("SELECT count(*) FROM prs", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let p = root.join(".deepkit/config.toml");
    fs::write(
        &p,
        fs::read_to_string(&p)
            .unwrap()
            .replace("max_pr_commits=1", "max_pr_commits=500"),
    )
    .unwrap();
    assert_eq!(index(&root)["prs"]["inserted"], 1);
    let identities = db(&root)
        .query_row("SELECT id,unit_id FROM prs", [], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
        })
        .unwrap();
    append(&root, "max_prs=0\n");
    let limited = index(&root);
    assert_eq!(limited["prs"]["state"], "capped");
    assert_eq!(
        db(&root)
            .query_row("SELECT id,unit_id FROM prs", [], |r| Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?
            )))
            .unwrap(),
        identities
    );
}

#[test]
fn pr_publication_rolls_back_sources_and_member_assignments_and_resumes() {
    let (_temp, fixtures) = fixture(&["github-merge"]);
    let root = root(&fixtures[0]);
    append(&root, "[commits]\nprs=false\n");
    index(&root);
    db(&root).execute_batch("CREATE TRIGGER reject_pr_source BEFORE INSERT ON unit_sources WHEN EXISTS(SELECT 1 FROM units WHERE id=NEW.unit_id AND kind='pr') BEGIN SELECT RAISE(ABORT,'fixture PR abort'); END;").unwrap();
    let p = root.join(".deepkit/config.toml");
    fs::write(
        &p,
        fs::read_to_string(&p)
            .unwrap()
            .replace("prs=false", "prs=true"),
    )
    .unwrap();
    assert!(
        !command(&root, &["index", "--level", "1", "--json"])
            .status
            .success()
    );
    let connection = db(&root);
    for query in [
        "SELECT count(*) FROM prs",
        "SELECT count(*) FROM units WHERE kind='pr'",
        "SELECT count(*) FROM commits WHERE pr_id IS NOT NULL",
    ] {
        assert_eq!(
            connection
                .query_row(query, [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    connection
        .execute_batch("DROP TRIGGER reject_pr_source;")
        .unwrap();
    assert_eq!(index(&root)["prs"]["inserted"], 1);
}

#[test]
fn recent_filters_dates_deadlines_and_missing_provenance_refuse_stale_arrays() {
    let (_temp, fixtures) = fixture(&["github-merge"]);
    let root = root(&fixtures[0]);
    index(&root);
    let recent = success(
        &root,
        &[
            "recent",
            "--path",
            "billing.rs",
            "--since",
            "2026-01-05",
            "--json",
        ],
    );
    assert_eq!(recent.as_array().unwrap().len(), 1);
    let id = recent[0]["id"].as_str().unwrap();
    assert!(
        success(
            &root,
            &["recent", "billing", "--since", "2026-01-06", "--json"]
        )
        .as_array()
        .unwrap()
        .is_empty()
    );
    for args in [
        vec!["recent", "-k", "0", "--json"],
        vec!["recent", "--since", "bad", "--json"],
        vec!["recent", "--path", "../outside", "--json"],
    ] {
        let r = command(&root, &args);
        assert!(!r.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&r.stdout).unwrap()["error"]["code"],
            "invalid_arguments"
        );
    }
    db(&root)
        .execute_batch("DELETE FROM unit_sources WHERE unit_id=(SELECT unit_id FROM prs LIMIT 1);")
        .unwrap();
    for args in [
        vec!["recent", "billing", "--json"],
        vec!["open", id, "--json"],
    ] {
        let r = command(&root, &args);
        assert!(!r.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&r.stdout).unwrap()["error"]["code"],
            "source_unavailable"
        );
    }
    append(&root, "[limits]\nquery_timeout_ms=0\n");
    let r = command(&root, &["recent", "--json"]);
    assert_eq!(
        serde_json::from_slice::<Value>(&r.stdout).unwrap()["error"]["code"],
        "timeout"
    );
}

#[test]
fn custom_patterns_enable_house_styles_without_rewriting_existing_members() {
    let (_temp, fixtures) = fixture(&["plain"]);
    let root = root(&fixtures[0]);
    append(&root, "[commits]\npr_patterns=[]\n");
    assert_eq!(index(&root)["prs"]["inserted"], 0);
    let p = root.join(".deepkit/config.toml");
    fs::write(
        &p,
        fs::read_to_string(&p).unwrap().replace(
            "pr_patterns=[]",
            r#"pr_patterns=[{regex="^Merge branch '(?P<title>.+)' into .*",strategy="plain"}]"#,
        ),
    )
    .unwrap();
    assert_eq!(index(&root)["prs"]["inserted"], 1);
    assert_eq!(
        success(&root, &["recent", "billing", "--json"])[0]["title"],
        "topic"
    );
}

#[test]
fn newer_overlapping_marker_defers_without_stealing_retained_pr_ownership() {
    let (_temp, fixtures) = fixture(&["github-merge"]);
    let root = root(&fixtures[0]);
    index(&root);
    let git = |args: &[&str]| {
        let o = Command::new("git")
            .current_dir(&root)
            .args(args)
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8(o.stdout).unwrap().trim().to_owned()
    };
    let base = git(&["rev-list", "--max-parents=0", "HEAD"]);
    let head = git(&["rev-parse", "HEAD"]);
    let tree = git(&["rev-parse", "HEAD^{tree}"]);
    let outer = git(&[
        "commit-tree",
        &tree,
        "-p",
        &base,
        "-p",
        &head,
        "-m",
        "Merge pull request #999 from fixture/outer",
        "-m",
        "Outer marker",
    ]);
    git(&["reset", "--hard", &outer]);
    let report = index(&root);
    assert_eq!(report["prs"]["deferred"]["member_already_owned"], 1);
    assert_eq!(
        db(&root)
            .query_row("SELECT count(*) FROM prs", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        db(&root)
            .query_row(
                "SELECT count(*) FROM commits WHERE pr_id IS NOT NULL",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
}

#[tokio::test]
async fn pr_embeddings_use_collated_context_and_mixed_protected_sources_never_dispatch() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(|r: &wiremock::Request| {
            let input: Value = serde_json::from_slice(&r.body).unwrap();
            ResponseTemplate::new(200).set_body_json(
                json!({"embeddings":vec![vec![1.0,0.0];input["input"].as_array().unwrap().len()]}),
            )
        })
        .mount(&server)
        .await;
    for protected in [false, true] {
        let (_temp, fixtures) = fixture(&["mixed"]);
        let root = root(&fixtures[0]);
        let p = root.join(".deepkit/config.toml");
        fs::write(
            &p,
            fs::read_to_string(&p).unwrap().replace(
                "profile='local'",
                &format!(
                    "profile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'",
                    server.uri()
                ),
            ),
        )
        .unwrap();
        if protected {
            append(
                &root,
                "[[treatments]]\nname='private'\nmatch=['private.rs']\nmax_level=1\nnever_send=true\n",
            );
        }
        let index2 = || {
            tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                .current_dir(&root)
                .env("XDG_CONFIG_HOME", root.join("absent-user-config"))
                .env_remove("DKC_TEST_INACTIVE_HISTORY_KEY")
                .args(["index", "--level", "2", "--json"])
                .output()
        };
        let before = server.received_requests().await.unwrap().len();
        let output = index2().await.unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = server.received_requests().await.unwrap();
        let inputs = requests[before..]
            .iter()
            .flat_map(|r| {
                serde_json::from_slice::<Value>(&r.body).unwrap()["input"]
                    .as_array()
                    .unwrap()
                    .clone()
            })
            .map(|s| s.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        if protected {
            assert!(
                !inputs
                    .iter()
                    .any(|s| s.contains("history of billing\nhistory\n")),
                "mixed history must not dispatch: {inputs:?}"
            );
            assert!(
                !inputs
                    .iter()
                    .any(|s| s.contains("PR #101") || s.contains("private.rs")),
                "{inputs:?}"
            );
        } else {
            assert!(
                inputs
                    .iter()
                    .any(|s| s.contains("history of billing\nhistory\n")),
                "history must embed: {inputs:?}"
            );
            assert!(
                inputs.iter().any(|s| s
                    .starts_with("PR #101 › Billing changes\nmerged 2026-01-05 › 2 commits ›")
                    && s.contains("\npr\n")
                    && s.contains("files:\nbilling.rs\nprivate.rs")),
                "{inputs:?}"
            );
        }
        let db = db(&root);
        let vectors: i64 = db
            .query_row(
                "SELECT count(*) FROM prs p JOIN units u ON u.id=p.unit_id WHERE u.ladder_level=2",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vectors, i64::from(!protected));
        let recent = success(&root, &["recent", "billing", "--json"]);
        assert_eq!(recent[0]["kind"], "pr");
        let jobs: i64 = db
            .query_row("SELECT count(*) FROM jobs", [], |r| r.get(0))
            .unwrap();
        let calls = requests.len();
        assert!(index2().await.unwrap().status.success());
        assert_eq!(server.received_requests().await.unwrap().len(), calls);
        assert_eq!(
            db.query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            jobs
        );
        if !protected {
            let identity:(i64,Vec<u8>,i64)=db.query_row("SELECT u.id,u.content_hash,(SELECT count(*) FROM jobs WHERE target_id=u.id) FROM prs p JOIN units u ON u.id=p.unit_id",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap();
            fs::write(
                root.join("after.rs"),
                "pub fn uncovered() {}\npub fn unrelated() {}\n",
            )
            .unwrap();
            assert!(index2().await.unwrap().status.success());
            let after:(i64,Vec<u8>,i64)=db.query_row("SELECT u.id,u.content_hash,(SELECT count(*) FROM jobs WHERE target_id=u.id) FROM prs p JOIN units u ON u.id=p.unit_id",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap();
            assert_eq!(identity, after);
            assert!(
                !server.received_requests().await.unwrap()[calls..]
                    .iter()
                    .any(|r| String::from_utf8_lossy(&r.body).contains("PR #101"))
            );
            // Retained but rank-disabled histories must not activate a second
            // provider or require its deliberately absent credential.
            append(
                &root,
                &format!(
                    "[enrich]\ntop_symbols=0\n[embedder.docs]\nprofile='remote'\nkinds=['history']\n[embedder.remote]\nurl='{}'\napi_key_env='DKC_TEST_INACTIVE_HISTORY_KEY'\n",
                    server.uri()
                ),
            );
            let inactive = index2().await.unwrap();
            assert!(
                inactive.status.success(),
                "{}",
                String::from_utf8_lossy(&inactive.stderr)
            );
        }
    }
}

#[test]
fn legacy_composites_migrate_outside_discovery_window_and_refusal_retains_derived_rows() {
    let (_temp, fixtures) = fixture(&["github-merge"]);
    let root = root(&fixtures[0]);
    index(&root);
    let db = db(&root);
    let original = db.prepare("SELECT u.id,u.kind,u.content_hash,u.source_key,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.kind IN ('pr','history') ORDER BY u.id").unwrap()
        .query_map([], |r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,Vec<u8>>(2)?,r.get::<_,String>(3)?,r.get::<_,String>(4)?))).unwrap().collect::<Result<Vec<_>,_>>().unwrap();
    assert!(original.iter().any(|r| r.1 == "pr"));
    assert!(original.iter().any(|r| r.1 == "history"));
    let owners = db
        .prepare("SELECT id,pr_id FROM commits ORDER BY id")
        .unwrap()
        .query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Option<i64>>(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    db.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE vec_legacy_fixture(unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,v BLOB NOT NULL,ctx_hash BLOB NOT NULL); INSERT INTO embedding_tables(slug,model,profile,dims,created_at,active) VALUES ('legacy_fixture','fixture','local',2,1,0);").unwrap();
    for (id, kind, _, source, _) in &original {
        let shas = db.prepare("SELECT c.sha FROM unit_sources s JOIN commits c ON c.id=s.source_id WHERE s.unit_id=?1 AND s.source_kind='commit' ORDER BY c.sha").unwrap().query_map([id],|r|r.get::<_,Vec<u8>>(0)).unwrap().collect::<Result<Vec<_>,_>>().unwrap();
        let mut legacy = blake3::Hasher::new();
        legacy.update(if kind == "pr" {
            b"pr|pr-collator-v1"
        } else {
            b"history|symbol-history-v1"
        });
        for sha in shas {
            legacy.update(b"|");
            legacy.update(&sha);
        }
        let payload = *legacy.finalize().as_bytes();
        let mut occurrence = blake3::Hasher::new();
        for part in [payload.as_slice(), source.as_bytes(), &0u32.to_le_bytes()] {
            occurrence.update(&(part.len() as u64).to_le_bytes());
            occurrence.update(part);
        }
        db.execute("INSERT OR IGNORE INTO unit_contents(hash,text,tokens) SELECT ?2,c.text,c.tokens FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1",rusqlite::params![id,payload.as_slice()]).unwrap();
        db.execute(
            "UPDATE units SET hash=?2,content_hash=?3 WHERE id=?1",
            rusqlite::params![
                id,
                occurrence.finalize().as_bytes().as_slice(),
                payload.as_slice()
            ],
        )
        .unwrap();
        let child = blake3::hash(format!("legacy summary {id}").as_bytes());
        db.execute(
            "INSERT INTO unit_contents(hash,text,tokens) VALUES (?1,'Retained summary',2)",
            [child.as_bytes().as_slice()],
        )
        .unwrap();
        db.execute("INSERT INTO units(hash,content_hash,source_key,kind,header,parent_unit,ladder_level,created_at,updated_at) VALUES (?1,?1,?2,'summary','summary',?3,3,1,1)",rusqlite::params![child.as_bytes().as_slice(),format!("legacy-summary:{id}"),id]).unwrap();
        db.execute("INSERT INTO vec_legacy_fixture(unit_id,v,ctx_hash) VALUES (?1,X'003C0000',zeroblob(32))",[id]).unwrap();
    }
    append(&root, "\n[commits]\nmax_count=1\n[enrich]\ntop_symbols=0\n");
    db.execute_batch("UPDATE meta SET value='legacy-collator-completion' WHERE key='pr_completed'; CREATE TRIGGER refuse_composite_sources BEFORE INSERT ON unit_sources WHEN EXISTS(SELECT 1 FROM units WHERE id=NEW.unit_id AND kind IN ('pr','history')) BEGIN SELECT RAISE(ABORT,'fixture migration refusal'); END;").unwrap();
    let refused = command(&root, &["index", "--level", "1", "--json"]);
    assert!(!refused.status.success());
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("fixture migration refusal"),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    for (id, _, hash, _, text) in &original {
        let (actual,body):(Vec<u8>,String)=db.query_row("SELECT u.content_hash,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
        assert_ne!(&actual, hash);
        assert_eq!(&body, text);
    }
    assert_eq!(
        db.query_row("SELECT count(*) FROM vec_legacy_fixture", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        original.len() as i64
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM units WHERE source_key LIKE 'legacy-summary:%'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        original.len() as i64
    );
    db.execute_batch("DROP TRIGGER refuse_composite_sources")
        .unwrap();
    let migrated = index(&root);
    assert_eq!(migrated["prs"]["visited"], 1, "{migrated}");
    assert_eq!(migrated["histories"]["selected"], 0, "{migrated}");
    assert!(
        migrated["histories"]["migration_selected"]
            .as_u64()
            .unwrap()
            > 0,
        "{migrated}"
    );
    assert_eq!(migrated["prs"]["updated"], 1, "{migrated}");
    assert_eq!(migrated["prs"]["migration_visited"], 1, "{migrated}");
    for (_, kind, hash, source, text) in &original {
        let (actual,body):(Vec<u8>,String)=db.query_row("SELECT u.content_hash,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.kind=?1 AND u.source_key=?2",rusqlite::params![kind,source],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
        assert_eq!(&actual, hash);
        assert_eq!(&body, text);
    }
    let after = db
        .prepare("SELECT id,pr_id FROM commits ORDER BY id")
        .unwrap()
        .query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Option<i64>>(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(after, owners);
    assert_eq!(
        db.query_row("SELECT count(*) FROM vec_legacy_fixture", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM units WHERE source_key LIKE 'legacy-summary:%'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    let noop = index(&root);
    assert_eq!(noop["prs"]["updated"], 0, "{noop}");
    assert_eq!(noop["histories"]["inserted"], 0, "{noop}");
    let recent = success(&root, &["recent", "billing", "--json"]);
    assert!(
        recent.as_array().unwrap().iter().any(|r| r["kind"] == "pr"),
        "{recent}"
    );
}

#[test]
fn pr_ranking_cli_retains_cohorts_failures_and_existing_reports_without_model_work() {
    let (_temp, fixtures) = fixture(&["github-merge"]);
    let expected = &fixtures[0];
    let root = root(expected);
    index(&root);
    let directory = tempfile::tempdir().unwrap();
    let queries = directory.path().join("queries.json");
    let ranks = directory.path().join("ranks.json");
    let output = directory.path().join("comparison.json");
    let missing = "0000000000000000000000000000000000000000";
    let mut questions = json!({"pin":expected["head"],
        "original":[{"rank":1,"merge":expected["prs"][0]["merge"],"title":"Billing changes"}],
        "older_controls":[{"rank":201,"merge":missing,"title":"Billing changes"}]});
    let weights = json!({"pin":expected["head"],"prs":[
        {"rank":1,"merge":expected["prs"][0]["merge"],"candidate_score_multiplier":1.20},
        {"rank":201,"merge":missing,"candidate_score_multiplier":1.10}]});
    fs::write(&queries, serde_json::to_vec(&questions).unwrap()).unwrap();
    fs::write(&ranks, serde_json::to_vec(&weights).unwrap()).unwrap();
    let before = db(&root).query_row(
        "SELECT (SELECT count(*) FROM jobs),(SELECT count(*) FROM usage),(SELECT count(*) FROM usage_reservations)",
        [], |r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?))).unwrap();
    let run = |out: &Path| {
        command(
            &root,
            &[
                "bench",
                "pr-ranking",
                "--root",
                root.to_str().unwrap(),
                "--queries",
                queries.to_str().unwrap(),
                "--ranks",
                ranks.to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
                "--json",
            ],
        )
    };
    let result = run(&output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let saved = fs::read(&output).unwrap();
    let report: Value = serde_json::from_slice(&saved).unwrap();
    assert_eq!(report["state"], "diagnostic_complete");
    assert_eq!(report["recent_denominator"], 1);
    assert_eq!(report["older_denominator"], 1);
    assert_eq!(report["queries"].as_array().unwrap().len(), 2);
    assert_eq!(report["queries"][0]["expected_pr_stored"], true);
    assert_eq!(report["queries"][1]["expected_pr_stored"], false);
    for (i, cohort) in ["recent", "older"].iter().enumerate() {
        assert_eq!(report["queries"][i]["cohort"], *cohort);
        assert_eq!(report["queries"][i]["query"], "Billing changes");
        assert_eq!(report["queries"][i]["arms"].as_array().unwrap().len(), 3);
    }
    assert!(
        report["queries"][1]["arms"]
            .as_array()
            .unwrap()
            .iter()
            .all(|a| a["raw_hit_pr_at_3"] == false)
    );
    assert_eq!(report["service_response_budget_applied"], false);
    assert_eq!(report["current_public_authority_verified"], false);
    let overwrite = run(&output);
    assert!(!overwrite.status.success());
    assert!(String::from_utf8_lossy(&overwrite.stderr).contains("refusing to overwrite"));
    assert_eq!(fs::read(&output).unwrap(), saved);
    questions["older_controls"][0]["title"] = json!("vec:billing");
    fs::write(&queries, serde_json::to_vec(&questions).unwrap()).unwrap();
    let partial = directory.path().join("partial.json");
    assert!(!run(&partial).status.success());
    let report: Value = serde_json::from_slice(&fs::read(partial).unwrap()).unwrap();
    assert_eq!(report["state"], "completed_with_query_failures");
    assert_eq!(report["queries"].as_array().unwrap().len(), 2);
    assert_eq!(report["older_denominator"], 1);
    assert!(
        report["queries"][1]["error"]
            .as_str()
            .unwrap()
            .contains("dense clause")
    );
    let mut invalid = weights.clone();
    invalid["prs"][1]["candidate_score_multiplier"] = json!(2.0);
    fs::write(&ranks, serde_json::to_vec(&invalid).unwrap()).unwrap();
    let rejected = directory.path().join("rejected.json");
    assert!(!run(&rejected).status.success());
    assert!(!rejected.exists());
    let after = db(&root).query_row(
        "SELECT (SELECT count(*) FROM jobs),(SELECT count(*) FROM usage),(SELECT count(*) FROM usage_reservations)",
        [], |r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?))).unwrap();
    assert_eq!(before, after);
    assert!(!root.join("absent-model-cache").exists());
}
