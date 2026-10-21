//! Source locations and vector retention through edits/upgrades; v1 §3.2–3.4.

use std::{collections::BTreeSet, error::Error, fs, path::Path, process::Command};

use assert_cmd::cargo::cargo_bin;
use rusqlite::{Connection, params};
use serde_json::Value;
use tempfile::TempDir;

const SHARED: &str = "pub fn repeat_me() -> i32 { 42 }\n";
type OccurrenceId = (String, String, i64);
type StoredVector = (i64, Vec<u8>, Vec<u8>);

fn dkc(root: &Path, args: &[&str]) -> Result<Value, Box<dyn Error>> {
    let result = Command::new(cargo_bin("dkc"))
        .current_dir(root)
        .args(args)
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(serde_json::from_slice(&result.stdout)?)
}

fn repo() -> Result<TempDir, Box<dyn Error>> {
    let root = TempDir::new()?;
    assert!(
        Command::new("git")
            .current_dir(root.path())
            .args(["init", "--quiet"])
            .status()?
            .success()
    );
    dkc(root.path(), &["init", "--json"])?;
    Ok(root)
}

fn db(root: &Path) -> Result<Connection, rusqlite::Error> {
    Connection::open(root.join(".deepkit/code.db"))
}

fn ids(root: &Path) -> Result<Vec<OccurrenceId>, Box<dyn Error>> {
    Ok(db(root)?
        .prepare(
            "SELECT f.path, lower(hex(u.hash)), u.line_start FROM units u
         JOIN files f ON f.id = u.file_id ORDER BY f.path, u.line_start",
        )?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<Vec<_>, _>>()?)
}

fn seed_vectors(root: &Path) -> Result<(), Box<dyn Error>> {
    let connection = db(root)?;
    connection.execute_batch(
        "CREATE TABLE vec_fixture(unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,
            v BLOB NOT NULL, ctx_hash BLOB NOT NULL);
         INSERT INTO embedding_tables VALUES ('fixture', 'fixture-model', 'local', 2, 0, 1);
         INSERT INTO vec_fixture SELECT u.id, X'003c0000', u.hash FROM units u
           JOIN files f ON f.id = u.file_id WHERE f.max_level >= 2;",
    )?;
    Ok(())
}

fn vectors(root: &Path) -> Result<Vec<StoredVector>, Box<dyn Error>> {
    Ok(db(root)?
        .prepare("SELECT unit_id, v, ctx_hash FROM vec_fixture ORDER BY unit_id")?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<Vec<_>, _>>()?)
}

#[test]
fn duplicate_locations_survive_edits_rename_rebuild_and_deletion() -> Result<(), Box<dyn Error>> {
    let root = repo()?;
    // The context header includes repo_name. A random temp basename can tokenize
    // to "b", which is also the old file-name token checked after the rename.
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[index]\nrepo_name='occurrencefixture'\n",
    )?;

    fs::write(root.path().join("a.rs"), SHARED)?;
    fs::write(
        root.path().join("b.rs"),
        format!("{}{SHARED}", "\n".repeat(8)),
    )?;
    fs::write(
        root.path().join("twice.rs"),
        format!("{SHARED}\n\n{SHARED}"),
    )?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(dkc(root.path(), &["status", "--json"])?["units"], 4);
    assert_eq!(
        db(root.path())?.query_row("SELECT count(*) FROM unit_contents", [], |r| r
            .get::<_, i64>(0))?,
        1
    );
    let original_ids = ids(root.path())?;
    assert_eq!(
        original_ids
            .iter()
            .map(|(_, hash, _)| hash)
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    for (path, hash, line) in &original_ids {
        let opened = dkc(
            root.path(),
            &["open", &format!("u_{}", &hash[..12]), "--json"],
        )?;
        assert_eq!(opened["path"], *path);
        assert_eq!(opened["lines"][0], *line);
    }
    assert_eq!(
        dkc(root.path(), &["open", "b.rs", "--json"])?["lines"][0],
        9
    );
    let scoped = dkc(
        root.path(),
        &[
            "search",
            "repeat_me",
            "--no-rerank",
            "--scope",
            "b.rs",
            "--json",
        ],
    )?;
    assert_eq!(scoped["hits"].as_array().ok_or("missing hits")?.len(), 1);
    assert_eq!(scoped["hits"][0]["lines"][0], 9);
    let symbols = dkc(root.path(), &["symbol", "repeat_me", "--json"])?;
    for symbol in symbols.as_array().ok_or("missing symbols")? {
        // v1 §3.4 gives equal-named symbols in one file the same public ID.
        // Refuse that ambiguity; path/range and occurrence IDs still locate both.
        if symbol["path"] == "twice.rs" {
            let ambiguous = Command::new(cargo_bin("dkc"))
                .current_dir(root.path())
                .args([
                    "open",
                    symbol["id"].as_str().ok_or("missing symbol id")?,
                    "--json",
                ])
                .output()?;
            assert!(!ambiguous.status.success());
            assert!(String::from_utf8_lossy(&ambiguous.stderr).contains("ambiguous"));
            let line = symbol["lines"][0].as_i64().ok_or("missing symbol line")?;
            assert_eq!(
                dkc(
                    root.path(),
                    &["open", &format!("twice.rs:{line}-{line}"), "--json"]
                )?["lines"][0],
                line
            );
            continue;
        }
        let opened = dkc(
            root.path(),
            &[
                "open",
                symbol["id"].as_str().ok_or("missing symbol id")?,
                "--json",
            ],
        )?;
        assert_eq!(opened["path"], symbol["path"]);
        assert_eq!(opened["lines"], symbol["lines"]);
    }
    seed_vectors(root.path())?;
    let retained = vectors(root.path())?;
    for flag in ["--full", "--rebuild-fts"] {
        dkc(root.path(), &["index", "--level", "1", flag, "--json"])?;
        assert_eq!(ids(root.path())?, original_ids);
        assert_eq!(vectors(root.path())?, retained);
    }
    // A neighboring unit changes; the shared occurrence and its vector remain.
    fs::write(
        root.path().join("a.rs"),
        format!("{SHARED}\npub fn neighbor() -> bool {{ true }}\n"),
    )?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(vectors(root.path())?, retained);
    fs::write(
        root.path().join("b.rs"),
        format!("{}{SHARED}", "\n".repeat(15)),
    )?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(vectors(root.path())?, retained);
    assert_eq!(
        dkc(root.path(), &["open", "b.rs", "--json"])?["lines"][0],
        16
    );
    fs::rename(root.path().join("b.rs"), root.path().join("moved.rs"))?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(vectors(root.path())?, retained); // old context fingerprint survives, for later re-embed
    let connection = db(root.path())?;
    let matches: i64 = connection.query_row(
        "SELECT count(*) FROM fts_units JOIN units u ON u.id=fts_units.rowid
         JOIN files f ON f.id=u.file_id WHERE fts_units MATCH 'header:moved' AND f.path='moved.rs'",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(matches, 1);
    let old_matches: i64 = connection.query_row(
        "SELECT count(*) FROM fts_units JOIN units u ON u.id=fts_units.rowid
         JOIN files f ON f.id=u.file_id WHERE fts_units MATCH 'header:b' AND f.path='moved.rs'",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(old_matches, 0);
    drop(connection);
    fs::remove_file(root.path().join("a.rs"))?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(vectors(root.path())?.len(), 3);
    assert_eq!(
        dkc(root.path(), &["open", "moved.rs", "--json"])?["lines"][0],
        16
    );
    // Remove the second equal occurrence in one file; the first keeps its row/vector.
    fs::write(root.path().join("twice.rs"), SHARED)?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(vectors(root.path())?.len(), 2);
    let before_fresh = ids(root.path())?;
    dkc(root.path(), &["clean", "--json"])?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(ids(root.path())?, before_fresh);
    // Delete the other source first in the freshly rebuilt database too.
    fs::remove_file(root.path().join("moved.rs"))?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(
        dkc(root.path(), &["open", "twice.rs", "--json"])?["lines"][0],
        1
    );
    fs::remove_file(root.path().join("twice.rs"))?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(
        db(root.path())?.query_row("SELECT count(*) FROM unit_contents", [], |r| r
            .get::<_, i64>(0))?,
        0
    );
    Ok(())
}

#[test]
fn equal_content_keeps_source_policy_and_per_file_caps() -> Result<(), Box<dyn Error>> {
    let root = repo()?;
    fs::write(root.path().join("public.rs"), SHARED)?;
    fs::write(root.path().join("protected.rs"), format!("\n\n{SHARED}"))?;
    fs::write(
        root.path().join("capped.rs"),
        format!("{SHARED}\n\n{SHARED}"),
    )?;
    fs::write(
        root.path().join(".deepkit/config.local.toml"),
        "[limits]\nmax_units_per_file = 1\n[[treatments]]\nname='protected'\nmatch=['protected.rs']\nmax_level=3\nnever_send=true\nchunker='ast'\n",
    )?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    let rows = db(root.path())?
        .prepare(
            "SELECT f.path, f.max_level, f.truncated, count(u.id) FROM files f
         JOIN units u ON u.file_id=f.id GROUP BY f.id ORDER BY f.path",
        )?
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(rows.len(), 3);
    assert_eq!((rows[0].2, rows[0].3), (1, 1));
    assert_eq!(rows[1].1, 1);
    assert_eq!(rows[2].1, 2);
    seed_vectors(root.path())?;
    assert_eq!(vectors(root.path())?.len(), 2);
    let protected_vectors: i64 = db(root.path())?.query_row(
        "SELECT count(*) FROM vec_fixture v JOIN units u ON u.id=v.unit_id
         JOIN files f ON f.id=u.file_id WHERE f.path='protected.rs'",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(protected_vectors, 0);
    let provenance: i64 = db(root.path())?.query_row(
        "SELECT count(*) FROM unit_sources s JOIN units u ON u.id=s.unit_id
         WHERE s.source_kind='file' AND s.source_id != u.file_id",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(provenance, 0);
    assert_eq!(
        dkc(
            root.path(),
            &[
                "search",
                "repeat_me",
                "--no-rerank",
                "--scope",
                "protected.rs",
                "--json"
            ]
        )?["hits"][0]["lines"][0],
        3
    );
    Ok(())
}

#[test]
fn upgrades_legacy_dedup_without_copying_vectors_or_locations() -> Result<(), Box<dyn Error>> {
    let root = repo()?;
    fs::write(root.path().join("a.rs"), SHARED)?;
    fs::write(
        root.path().join("b.rs"),
        format!("{}{SHARED}", "\n".repeat(8)),
    )?;
    fs::write(
        root.path().join("c.rs"),
        "pub fn known_context() -> i32 { 7 }\n",
    )?;
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    let expected_ids = ids(root.path())?;
    let (hash, header, scope, text): (Vec<u8>, String, String, String) = db(root.path())?.query_row(
        "SELECT u.content_hash,u.header,u.scope_path,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.source_key='a.rs'",
        [], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)),
    )?;
    let known: (Vec<u8>, String, String, String) = db(root.path())?.query_row(
        "SELECT u.content_hash,u.header,u.scope_path,c.text FROM units u JOIN unit_contents c ON c.hash=u.content_hash WHERE u.source_key='c.rs'",
        [], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)),
    )?;
    dkc(root.path(), &["clean", "--json"])?;
    let connection = db(root.path())?;
    for sql in [
        include_str!("../src/store/migrations/0001_initial.sql"),
        include_str!("../src/store/migrations/0002_file_rank.sql"),
        include_str!("../src/store/migrations/0003_file_truncated.sql"),
        include_str!("../src/store/migrations/0004_l1_pending.sql"),
    ] {
        connection.execute_batch(sql)?;
    }
    connection.execute("INSERT INTO meta VALUES ('schema_version','4')", [])?;
    for (id, path) in [(1, "a.rs"), (2, "b.rs"), (3, "c.rs")] {
        connection.execute("INSERT INTO files(id,path,blake3,size,mtime_ns,lang,kind,max_level,treatment,l1_pending)
            VALUES (?1,?2,?3,0,0,'rust','code',2,'default',0)", params![id,path,blake3::hash(&fs::read(root.path().join(path))?).as_bytes()])?;
    }
    connection.execute("INSERT INTO units(id,hash,file_id,kind,line_start,line_end,scope_path,header,text,tokens,created_at,updated_at)
        VALUES(42,?1,1,'symbol',1,1,?2,?3,?4,10,0,0)", params![hash,scope,header,text])?;
    connection.execute("INSERT INTO units(id,hash,file_id,kind,line_start,line_end,scope_path,header,text,tokens,created_at,updated_at)
        VALUES(43,?1,3,'symbol',1,1,?2,?3,?4,10,0,0)", params![known.0,known.2,known.1,known.3])?;
    connection.execute_batch("INSERT INTO unit_sources VALUES (42,'file',1,'content'),(42,'file',2,'content'),(43,'file',3,'content');
        INSERT INTO symbols(file_id,unit_id,name,qualified,kind,line_start,line_end) VALUES
          (1,42,'repeat_me','repeat_me','function',1,1),(2,42,'repeat_me','repeat_me','function',9,9);
        INSERT INTO fts_units(rowid,header,idents,body,prose) VALUES (42,'a rs','repeat_me','repeat me','');")?;
    drop(connection);
    seed_vectors(root.path())?;
    let before = vectors(root.path())?;
    let migrated = dkc(root.path(), &["status", "--json"])?;
    assert_eq!(migrated["l1_pending_files"], 3);
    assert_eq!(migrated["units"], 2); // other locations must come from reparsing
    let symbol_owners = db(root.path())?
        .prepare("SELECT file_id, unit_id FROM symbols ORDER BY file_id")?
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(symbol_owners, [(1, Some(42)), (2, None)]);
    let after = vectors(root.path())?;
    assert_eq!((after[0].0, &after[0].1), (42, &before[0].1));
    assert_eq!(after[0].2, vec![0; 32]); // legacy shared context was ambiguous
    assert_eq!(after[1], before[1]); // known context and vector bytes remain unchanged
    dkc(root.path(), &["index", "--level", "1", "--json"])?;
    assert_eq!(ids(root.path())?, expected_ids);
    assert_eq!(vectors(root.path())?, after); // no vector copied to b.rs
    assert_eq!(
        dkc(root.path(), &["open", "b.rs", "--json"])?["lines"][0],
        9
    );
    let integrity: String =
        db(root.path())?.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    assert_eq!(integrity, "ok");
    assert!(
        !db(root.path())?
            .prepare("PRAGMA foreign_key_check")?
            .exists([])?
    );
    Ok(())
}
