//! Last prepared context bookkeeping for status; v1 §7.1/§13.
use super::Error;
use crate::config::Config;
use rusqlite::{Transaction, params};

/// Binds header configuration without credentials or provider/model resolution.
pub(crate) fn recipe(config: &Config) -> [u8; 32] {
    let mut hash = blake3::Hasher::new_derive_key("dkc prepared context recipe v1");
    hash.update(&[
        u8::from(config.embedder.breadcrumb),
        u8::from(config.embedder.breadcrumb_roles),
        u8::from(config.embedder.kind_line),
    ]);
    hash.update(config.index.repo_name.as_bytes());
    *hash.finalize().as_bytes()
}

/// Saves caller-verified context in its publication transaction. This record
/// never authorizes retrieval or outbound work; unchanged records cause no write.
pub(crate) fn record(
    tx: &Transaction<'_>,
    unit: i64,
    content: &[u8; 32],
    context: &[u8; 32],
    config: &Config,
) -> Result<(), Error> {
    tx.execute("INSERT INTO unit_contexts(unit_id,content_hash,ctx_hash,recipe_hash) VALUES(?1,?2,?3,?4)
        ON CONFLICT(unit_id) DO UPDATE SET content_hash=excluded.content_hash,ctx_hash=excluded.ctx_hash,recipe_hash=excluded.recipe_hash
        WHERE content_hash!=excluded.content_hash OR ctx_hash!=excluded.ctx_hash OR recipe_hash!=excluded.recipe_hash",
        params![unit,content,context,recipe(config)])?;
    Ok(())
}

/// Counts compare persisted prepared context only, never live source freshness.
#[derive(Debug, serde::Serialize)]
pub(crate) struct Counts {
    source: &'static str,
    current: i64,
    stale: i64,
    unknown: i64,
}

/// Reads one validated physical vector table without resolving models or secrets.
/// Legacy, changed-content and changed-configuration records remain unknown.
pub(crate) fn counts(
    tx: &Transaction<'_>,
    slug: &str,
    config: Option<&Config>,
) -> Result<Counts, Error> {
    let name = super::vectors::vector_table_name(slug)?;
    let recipe = config.map(recipe);
    let (total, known, current): (i64,i64,i64) = tx.query_row(&format!(
        "SELECT count(*),coalesce(sum(h.unit_id IS NOT NULL),0),coalesce(sum(h.ctx_hash=v.ctx_hash),0)
         FROM {name} v JOIN units u ON u.id=v.unit_id LEFT JOIN unit_contexts h ON h.unit_id=v.unit_id
         AND h.content_hash=u.content_hash AND h.recipe_hash=?1"),
        [recipe.as_ref().map(|v|v.as_slice())], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
    Ok(Counts {
        source: "last_verified_prepared_context",
        current,
        stale: known - current,
        unknown: total - known,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, StoreLimits};

    #[test]
    fn file_move_invalidates_owned_summary_and_history_context_without_touching_other_units() {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        let config = Config::default();
        let mut db = store.writer().unwrap();
        db.execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment)
            VALUES(1,'before.rs',zeroblob(32),0,0,'code',3,'fixture');
            INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'fixture',1);
            INSERT INTO units(id,hash,content_hash,source_key,kind,header,file_id,parent_unit,created_at,updated_at) VALUES
            (1,x'01',zeroblob(32),'symbol','symbol','before',1,NULL,0,0),
            (2,x'02',zeroblob(32),'summary','summary','before',NULL,1,0,0),
            (3,x'03',zeroblob(32),'history','history','before',NULL,NULL,0,0),
            (4,x'04',zeroblob(32),'control','chunk','control',NULL,NULL,0,0);
            INSERT INTO symbols(file_id,unit_id,name,kind,line_start,line_end,summary_unit,history_unit)
            VALUES(1,1,'fixture','function',1,1,2,3);
            CREATE TABLE vec_move(unit_id INTEGER PRIMARY KEY REFERENCES units(id),ctx_hash BLOB);
            INSERT INTO vec_move SELECT id,zeroblob(32) FROM units;").unwrap();
        let tx = db.transaction().unwrap();
        for unit in 1..=4 {
            record(&tx, unit, &[0; 32], &[0; 32], &config).unwrap();
        }
        tx.execute("UPDATE files SET path='before.rs' WHERE id=1", [])
            .unwrap();
        assert_eq!(counts(&tx, "move", Some(&config)).unwrap().current, 4);
        tx.execute("UPDATE files SET path='after.rs' WHERE id=1", [])
            .unwrap();
        let observed = counts(&tx, "move", Some(&config)).unwrap();
        assert_eq!(
            (observed.current, observed.stale, observed.unknown),
            (1, 0, 3)
        );
        assert_eq!(
            tx.query_row("SELECT unit_id FROM unit_contexts", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            4
        );
        assert_eq!(
            tx.query_row("SELECT count(*) FROM vec_move", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            4
        );
    }

    #[test]
    fn per_table_counts_preserve_unknown_legacy_and_changed_recipe_or_content() {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        let config = Config::default();
        let mut db = store.writer().unwrap();
        db.execute_batch("CREATE TABLE vec_a(unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,ctx_hash BLOB);
            CREATE TABLE vec_b(unit_id INTEGER PRIMARY KEY REFERENCES units(id) ON DELETE CASCADE,ctx_hash BLOB);
            INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'fixture',1);
            INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES
            (1,x'01',zeroblob(32),'a','chunk','header',0,0),(2,x'02',zeroblob(32),'b','chunk','header',0,0),(3,x'03',zeroblob(32),'c','chunk','header',0,0);
            INSERT INTO vec_a VALUES(1,zeroblob(32)),(2,zeroblob(32)),(3,zeroblob(32));").unwrap();
        db.execute(
            "INSERT INTO vec_b VALUES(1,?1),(2,?1)",
            [[1u8; 32].as_slice()],
        )
        .unwrap();
        let tx = db.transaction().unwrap();
        assert_eq!(counts(&tx, "a", Some(&config)).unwrap().unknown, 3);
        record(&tx, 1, &[0; 32], &[0; 32], &config).unwrap();
        record(&tx, 2, &[0; 32], &[1; 32], &config).unwrap();
        let a = counts(&tx, "a", Some(&config)).unwrap();
        assert_eq!((a.current, a.stale, a.unknown), (1, 1, 1));
        let b = counts(&tx, "b", Some(&config)).unwrap();
        assert_eq!((b.current, b.stale, b.unknown), (1, 1, 0));
        let before = tx.total_changes();
        record(&tx, 2, &[0; 32], &[1; 32], &config).unwrap();
        assert_eq!(tx.total_changes(), before);
        assert_eq!(counts(&tx, "a", None).unwrap().unknown, 3);
        for field in ["kind_line", "breadcrumb", "breadcrumb_roles", "repo_name"] {
            let mut changed = config.clone();
            match field {
                "kind_line" => changed.embedder.kind_line = !changed.embedder.kind_line,
                "breadcrumb" => changed.embedder.breadcrumb = !changed.embedder.breadcrumb,
                "breadcrumb_roles" => {
                    changed.embedder.breadcrumb_roles = !changed.embedder.breadcrumb_roles
                }
                "repo_name" => changed.index.repo_name = "renamed repository".into(),
                _ => unreachable!(),
            }
            let count = counts(&tx, "a", Some(&changed)).unwrap();
            assert_eq!(
                (count.current, count.stale, count.unknown),
                (0, 0, 3),
                "{field}"
            );
        }
        tx.execute(
            "UPDATE unit_contexts SET content_hash=?1 WHERE unit_id=1",
            [[2u8; 32].as_slice()],
        )
        .unwrap();
        assert_eq!(counts(&tx, "a", Some(&config)).unwrap().unknown, 2);
        assert!(
            tx.execute("UPDATE unit_contexts SET ctx_hash=x'01'", [])
                .is_err()
        );
        tx.execute("DELETE FROM units WHERE id=2", []).unwrap();
        assert_eq!(
            tx.query_row("SELECT count(*) FROM unit_contexts", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        tx.execute("UPDATE units SET header='new header' WHERE id=1", [])
            .unwrap();
        assert_eq!(
            tx.query_row("SELECT count(*) FROM unit_contexts", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(counts(&tx, "a", Some(&config)).unwrap().unknown, 2);
    }
}
