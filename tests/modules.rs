//! Module subtree counts, canonical identity and incremental publication; v1.md §6.6–6.7.
use rusqlite::Connection;
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path, process::Command};

fn index(root: &Path) -> Value {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-config"))
        .args(["index", "--level", "1", "--no-queue", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"),"[index]\nrepo_name='modules'\n[embedder]\nprofile='local'\n[llm]\nenabled=false\n[commits]\nenabled=false\n[rerank]\nmode='off'\n").unwrap();
    for directory in ["alpha", "beta"] {
        fs::create_dir(root.path().join(directory)).unwrap();
        for n in 0..5 {
            fs::write(
                root.path().join(format!("{directory}/{directory}{n}.rs")),
                format!("pub fn item{n}() {{}}\n"),
            )
            .unwrap();
        }
    }
    fs::write(
        root.path().join("alpha/README.md"),
        "# Alpha\n\nPurpose of alpha.\n\n## Details\n\nSecond section.\n",
    )
    .unwrap();
    root
}
type Modules = BTreeMap<String, (i64, Vec<u8>, String, String)>;
fn modules(db: &Connection) -> Modules {
    db.prepare("SELECT d.path,u.id,u.content_hash,u.header,c.text FROM dirs d JOIN units u ON u.id=d.module_unit JOIN unit_contents c ON c.hash=u.content_hash ORDER BY d.path").unwrap()
        .query_map([],|r|Ok((r.get(0)?,(r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)))).unwrap().collect::<Result<_,_>>().unwrap()
}

#[test]
fn source_deletion_retires_unqualified_modules_but_threshold_changes_retain_them() {
    let root = fixture();
    let root = root.path();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let initial = modules(&db);
    let config = root.join(".deepkit/config.toml");
    let original = fs::read_to_string(&config).unwrap();
    fs::write(
        &config,
        format!("{original}\n[enrich]\nmodules_min_files=20\n"),
    )
    .unwrap();
    let limited = index(root);
    assert_eq!(limited["modules"]["removed"], 0, "{limited}");
    assert_eq!(modules(&db), initial);
    fs::write(config, original).unwrap();
    fs::remove_file(root.join("beta/beta0.rs")).unwrap();
    let removed = index(root);
    assert_eq!(removed["modules"]["removed"], 1, "{removed}");
    assert!(!modules(&db).contains_key("beta"));
    let old_exists: bool = db
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM units WHERE kind='module' AND source_key='module:beta')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(!old_exists);
    assert_eq!(modules(&db)["alpha"], initial["alpha"]);
    fs::remove_dir_all(root.join("alpha")).unwrap();
    index(root);
    assert!(modules(&db).is_empty());
    let count: i64 = db
        .query_row("SELECT count(*) FROM units WHERE kind='module'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn lazy_module_refresh_does_not_repair_a_large_contributor() {
    let root = fixture();
    let root = root.path();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let initial = modules(&db);
    let before: Vec<u8> = db
        .query_row(
            "SELECT blake3 FROM files WHERE path='beta/beta0.rs'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    fs::write(root.join("beta/beta0.rs"), vec![b' '; 1024 * 1024]).unwrap();
    let mapped = invoke(root, &["map", "--focus", "alpha", "--json"]);
    assert!(
        mapped.status.success(),
        "{}",
        String::from_utf8_lossy(&mapped.stderr)
    );
    let body: Value = serde_json::from_slice(&mapped.stdout).unwrap();
    assert_eq!(body["index"]["partial"], true, "{body}");
    let after: Vec<u8> = db
        .query_row(
            "SELECT blake3 FROM files WHERE path='beta/beta0.rs'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(before, after);
    assert_eq!(modules(&db), initial);
}

#[test]
fn module_context_change_preserves_occurrence_and_summary() {
    let root = fixture();
    let root = root.path();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let initial = modules(&db);
    let module = initial["alpha"].0;
    let hash = blake3::hash(b"retained summary across module context change");
    db.execute(
        "INSERT INTO unit_contents(hash,text,tokens) VALUES (?1,'Retained role',2)",
        [hash.as_bytes().as_slice()],
    )
    .unwrap();
    db.execute("INSERT INTO units(hash,content_hash,source_key,kind,header,parent_unit,ladder_level,created_at,updated_at) VALUES (?1,?1,'summary:alpha','summary','role',?2,3,1,1)",rusqlite::params![hash.as_bytes().as_slice(),module]).unwrap();
    let summary = db.last_insert_rowid();
    db.execute_batch("CREATE TRIGGER refuse_module_delete BEFORE DELETE ON units WHEN OLD.kind='module' BEGIN SELECT RAISE(ABORT,'context must retain module'); END;").unwrap();
    let config = root.join(".deepkit/config.toml");
    let changed = fs::read_to_string(&config)
        .unwrap()
        .replace("repo_name='modules'", "repo_name='renamed-repository'");
    fs::write(config, changed).unwrap();
    index(root);
    assert_eq!(modules(&db), initial);
    let retained: (i64, String) = db.query_row("SELECT child.id,parent.scope_path FROM units child JOIN units parent ON parent.id=child.parent_unit WHERE child.id=?1 AND parent.id=?2",rusqlite::params![summary,module],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
    assert_eq!(retained.0, summary);
    assert!(retained.1.contains("renamed-repository"), "{retained:?}");
}

#[test]
fn subtree_counts_readme_provenance_and_incremental_noop() {
    let root = fixture();
    let root = root.path();
    let first = index(root);
    assert_eq!(first["modules"]["inserted"], 3, "{first}");
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let initial = modules(&db);
    assert_eq!(
        initial.keys().map(String::as_str).collect::<Vec<_>>(),
        ["", "alpha", "beta"]
    );
    assert_eq!(initial[""].2, "module · 11 files · 10 symbols");
    assert_eq!(initial["alpha"].2, "module · 6 files · 5 symbols");
    assert_eq!(initial["beta"].2, "module · 5 files · 5 symbols");
    assert!(initial["alpha"].3.contains("Purpose of alpha."));
    assert!(!initial["alpha"].3.contains("Second section."));
    let sources:(i64,i64)=db.query_row("SELECT sum(source_kind='file'),sum(source_kind='unit') FROM unit_sources WHERE unit_id=?1",[initial["alpha"].0],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
    assert_eq!(sources, (6, 1));
    let noop = index(root);
    assert_eq!(noop["modules"]["inserted"], 0);
    assert_eq!(noop["modules"]["unchanged"], 3);
    assert_eq!(modules(&db), initial);
    fs::write(
        root.join("alpha/alpha0.rs"),
        "pub fn item0() { let changed = 1; }\n",
    )
    .unwrap();
    let update = index(root);
    assert_eq!(update["modules"]["inserted"], 2, "{update}");
    let changed = modules(&db);
    assert_eq!(changed["beta"], initial["beta"]);
    for path in ["", "alpha"] {
        assert_ne!(changed[path].1, initial[path].1);
    }
    fs::rename(root.join("alpha/alpha0.rs"), root.join("alpha/renamed.rs")).unwrap();
    index(root);
    let renamed = modules(&db);
    assert_ne!(renamed["alpha"].1, changed["alpha"].1);
    assert_eq!(renamed["beta"], changed["beta"]);
    assert!(renamed["alpha"].3.contains("renamed.rs"));
}

#[test]
fn identical_byte_multisets_with_different_filenames_have_distinct_payloads() {
    let root = fixture();
    let root = root.path();
    fs::remove_file(root.join("alpha/README.md")).unwrap();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let rows = modules(&db);
    assert_eq!(rows["alpha"].2, rows["beta"].2);
    assert_ne!(rows["alpha"].1, rows["beta"].1);
    assert_ne!(rows["alpha"].3, rows["beta"].3);
    let payloads:i64=db.query_row("SELECT count(*) FROM unit_contents WHERE hash IN (SELECT content_hash FROM units WHERE kind='module')",[],|r|r.get(0)).unwrap();
    assert_eq!(payloads, 3);
}

fn invoke(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
        .current_dir(root)
        .env("XDG_CONFIG_HOME", root.join("absent-config"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn public_module_open_lists_file_and_readme_sources_and_refuses_stale_text() {
    let root = fixture();
    let root = root.path();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let id:String=db.query_row("SELECT 'u_'||substr(lower(hex(u.hash)),1,12) FROM units u JOIN dirs d ON d.module_unit=u.id WHERE d.path='alpha'",[],|r|r.get(0)).unwrap();
    let open = invoke(root, &["open", &id, "--json"]);
    assert!(
        open.status.success(),
        "{} {}",
        String::from_utf8_lossy(&open.stdout),
        String::from_utf8_lossy(&open.stderr)
    );
    let body: Value = serde_json::from_slice(&open.stdout).unwrap();
    assert_eq!(body["kind"], "module");
    let sources = body["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 7);
    assert!(
        sources
            .iter()
            .any(|s| s["label"] == "alpha/alpha0.rs" && s["kind"] == "file")
    );
    assert!(sources.iter().any(|s| s["kind"] == "doc_section"));
    fs::write(root.join("alpha/alpha0.rs"), "pub fn changed() {}\n").unwrap();
    let stale = invoke(root, &["open", &id, "--json"]);
    assert!(
        !stale.status.success(),
        "{}",
        String::from_utf8_lossy(&stale.stdout)
    );
    index(root);
    let status = invoke(root, &["status", "--json"]);
    assert!(status.status.success());
    let body: Value = serde_json::from_slice(&status.stdout).unwrap();
    assert_eq!(body["modules"]["state"], "completed");
}

#[test]
fn map_headers_reject_unowned_summary_and_preserve_exact_token_budgets() {
    let root = fixture();
    let root = root.path();
    index(root);
    let mapped = invoke(root, &["map", "--budget", "2000", "--json"]);
    assert!(
        mapped.status.success(),
        "{} {}",
        String::from_utf8_lossy(&mapped.stdout),
        String::from_utf8_lossy(&mapped.stderr)
    );
    let body: Value = serde_json::from_slice(&mapped.stdout).unwrap();
    let tree = body["tree"].as_str().unwrap();
    assert!(
        tree.contains("./ — module · 11 files · 10 symbols"),
        "{body}"
    );
    assert!(
        tree.contains("alpha/ — module · 6 files · 5 symbols"),
        "{body}"
    );
    assert_eq!(body["index"]["partial"], false, "{body}");
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let module: i64 = db
        .query_row("SELECT module_unit FROM dirs WHERE path='alpha'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let hash = blake3::hash(b"module summary fixture");
    db.execute("INSERT INTO unit_contents(hash,text,tokens) VALUES (?1,'Alpha role\nSecond summary line',7)",[hash.as_bytes().as_slice()]).unwrap();
    db.execute("INSERT INTO units(hash,content_hash,source_key,kind,header,parent_unit,ladder_level,created_at,updated_at) VALUES (?1,?1,'summary:alpha','summary','summary',?2,3,1,1)",rusqlite::params![hash.as_bytes().as_slice(),module]).unwrap();
    let summary = db.last_insert_rowid();
    db.execute("INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES (?1,'summary','','Alpha role','Alpha role')",[summary]).unwrap();
    for budget in [1, 20, 50, 100, 2000] {
        let budget_arg = budget.to_string();
        let mapped = invoke(root, &["map", "--budget", &budget_arg, "--json"]);
        assert!(mapped.status.success());
        let body: Value = serde_json::from_slice(&mapped.stdout).unwrap();
        let tree = body["tree"].as_str().unwrap();
        assert!(
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(tree)
                .len()
                <= budget,
            "{body}"
        );
        assert!(!tree.contains("Second summary line"));
        if budget == 2000 {
            // An unowned child is not a verified module description.
            assert!(tree.contains("alpha/"), "{body}");
            assert!(!tree.contains("Alpha role"), "{body}");
        }
        assert_eq!(body["index"]["partial"], false, "{body}");
        for node in body["nodes"].as_array().unwrap() {
            assert!(tree.contains(node["path"].as_str().unwrap().rsplit('/').next().unwrap()));
        }
    }
}

#[test]
fn refused_module_publication_retains_old_units_sources_and_fts_then_retries() {
    let root = fixture();
    let root = root.path();
    index(root);
    let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
    let before = modules(&db);
    let source_count: i64 = db
        .query_row(
            "SELECT count(*) FROM unit_sources WHERE unit_id IN (SELECT module_unit FROM dirs)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    db.execute_batch("CREATE TRIGGER refuse_module_source BEFORE INSERT ON unit_sources WHEN (SELECT kind FROM units WHERE id=NEW.unit_id)='module' BEGIN SELECT RAISE(ABORT,'injected module source refusal'); END;").unwrap();
    fs::write(root.join("alpha/alpha0.rs"), "pub fn replacement() {}\n").unwrap();
    let failed = invoke(root, &["index", "--level", "1", "--no-queue", "--json"]);
    assert!(!failed.status.success());
    assert_eq!(modules(&db), before);
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM unit_sources WHERE unit_id IN (SELECT module_unit FROM dirs)",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        source_count
    );
    for (id, _, _, _) in before.values() {
        assert_eq!(
            db.query_row("SELECT count(*) FROM fts_units WHERE rowid=?1", [id], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
    db.execute_batch("DROP TRIGGER refuse_module_source;")
        .unwrap();
    let retried = index(root);
    assert_eq!(retried["modules"]["inserted"], 2);
    let after = modules(&db);
    assert_ne!(after["alpha"].1, before["alpha"].1);
    assert_eq!(after["beta"], before["beta"]);
}

#[tokio::test]
async fn module_embedding_fences_external_private_sources_and_context_jobs_preserve_identity() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/api/embed")).respond_with(|request:&wiremock::Request| {
        let body:Value=serde_json::from_slice(&request.body).unwrap();
        ResponseTemplate::new(200).set_body_json(serde_json::json!({"embeddings":vec![vec![1.0,0.0];body["input"].as_array().unwrap().len()]}))
    }).mount(&server).await;
    for protected in [false, true] {
        let fixture = fixture();
        let root = fixture.path();
        fs::write(root.join("alpha/alpha0.rs"), "pub fn alpha_zero() {}\n").unwrap();
        fs::write(
            root.join("private.rs"),
            "pub fn caller() { alpha_zero(); }\n",
        )
        .unwrap();
        let config = root.join(".deepkit/config.toml");
        let mut text = fs::read_to_string(&config).unwrap().replace(
            "profile='local'",
            &format!(
                "profile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'",
                server.uri()
            ),
        );
        if protected {
            text.push_str("\n[[treatments]]\nname='private'\nmatch=['private.rs']\nmax_level=1\nnever_send=true\n");
        }
        fs::write(&config, text).unwrap();
        let index2 = || {
            tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
                .current_dir(root)
                .env("XDG_CONFIG_HOME", root.join("absent-config"))
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
            .map(|v| v.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let module_inputs = inputs
            .iter()
            .filter(|s| s.contains("\nmodule\n"))
            .collect::<Vec<_>>();
        assert!(
            module_inputs.iter().any(|s| s.contains("module · 5 files")),
            "{inputs:?}"
        );
        for header in ["module · 6 files", "module · 12 files"] {
            assert_eq!(
                module_inputs.iter().any(|s| s.contains(header)),
                !protected,
                "{inputs:?}"
            );
        }
        if protected {
            assert!(
                !inputs.iter().any(|s| s.contains("private.rs")),
                "{inputs:?}"
            );
        }
        let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
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
            let initial = modules(&db);
            let module = initial["alpha"].0;
            let slug: String = db
                .query_row(
                    "SELECT slug FROM embedding_tables WHERE active=1",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(slug.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'));
            let sql = format!("SELECT ctx_hash FROM vec_{slug} WHERE unit_id=?1");
            let context_before: Vec<u8> = db.query_row(&sql, [module], |r| r.get(0)).unwrap();
            fs::write(
                &config,
                fs::read_to_string(&config)
                    .unwrap()
                    .replace("repo_name='modules'", "repo_name='new-context'"),
            )
            .unwrap();
            let updated = index2().await.unwrap();
            assert!(
                updated.status.success(),
                "{}",
                String::from_utf8_lossy(&updated.stderr)
            );
            assert_eq!(modules(&db), initial);
            let context_after: Vec<u8> = db.query_row(&sql, [module], |r| r.get(0)).unwrap();
            assert_ne!(context_before, context_after);
            let requests = server.received_requests().await.unwrap();
            assert!(
                requests[calls..]
                    .iter()
                    .any(|r| String::from_utf8_lossy(&r.body).contains("new-context"))
            );
            let calls = requests.len();
            let jobs: i64 = db
                .query_row("SELECT count(*) FROM jobs", [], |r| r.get(0))
                .unwrap();
            assert!(index2().await.unwrap().status.success());
            assert_eq!(server.received_requests().await.unwrap().len(), calls);
            assert_eq!(
                db.query_row("SELECT count(*) FROM jobs", [], |r| r.get::<_, i64>(0))
                    .unwrap(),
                jobs
            );
        }
    }
}

#[tokio::test]
async fn remote_module_reranking_refuses_mixed_provenance_and_never_injects_unverified_roles() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/rerank")).respond_with(|request:&wiremock::Request| {
        let body:Value=serde_json::from_slice(&request.body).unwrap();
        let documents=body["documents"].as_array().unwrap();
        ResponseTemplate::new(200).set_body_json(serde_json::json!({"results":documents.iter().enumerate().map(|(index,_)|serde_json::json!({"index":index,"relevance_score":0.9})).collect::<Vec<_>>()}))
    }).mount(&server).await;
    for protected in [false, true] {
        let fixture = fixture();
        let root = fixture.path();
        fs::write(root.join("alpha/alpha0.rs"), "pub fn alpha_zero() {}\n").unwrap();
        fs::write(
            root.join("private.rs"),
            "pub fn caller() { alpha_zero(); }\n",
        )
        .unwrap();
        let config = root.join(".deepkit/config.toml");
        let mut text=fs::read_to_string(&config).unwrap().replace("mode='off'",&format!("mode='on'\nprovider='remote'\n[rerank.remote]\nurl='{}'\nmodel='fixture'\napi_key_env='DKC_MODULE_RERANK_KEY'",server.uri()));
        text.push_str("\n[limits]\nquery_timeout_ms=10000\nrerank_timeout_ms=5000\n");
        if protected {
            text.push_str("\n[[treatments]]\nname='private'\nmatch=['private.rs']\nmax_level=1\nnever_send=true\n");
        }
        fs::write(config, text).unwrap();
        index(root);
        let db = Connection::open(root.join(".deepkit/code.db")).unwrap();
        let module = modules(&db)["alpha"].0;
        let hash = blake3::hash(b"unverified-role-material");
        db.execute(
            "INSERT INTO unit_contents(hash,text,tokens) VALUES (?1,'unverified-role-material',1)",
            [hash.as_bytes().as_slice()],
        )
        .unwrap();
        db.execute("INSERT INTO units(hash,content_hash,source_key,kind,header,parent_unit,ladder_level,created_at,updated_at) VALUES (?1,?1,'role:alpha','summary','unverified-role-material',?2,3,1,1)",rusqlite::params![hash.as_bytes().as_slice(),module]).unwrap();
        let before = server.received_requests().await.unwrap().len();
        let output = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("dkc"))
            .current_dir(root)
            .env("XDG_CONFIG_HOME", root.join("absent-config"))
            .env("DKC_MODULE_RERANK_KEY", "fixture-key")
            .args([
                "search", "module", "--mode", "lexical", "-k", "10", "--json",
            ])
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let body: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(
            body["hits"]
                .as_array()
                .unwrap()
                .iter()
                .any(|h| h["kind"] == "module"),
            "{body}"
        );
        assert_eq!(
            body["index"]["rerank"],
            if protected { "privacy" } else { "complete" },
            "{body}"
        );
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len() - before, usize::from(!protected));
        for request in &requests[before..] {
            let text = String::from_utf8_lossy(&request.body);
            assert!(text.contains("module"), "{text}");
            assert!(!text.contains("unverified-role-material"), "{text}");
        }
    }
}
