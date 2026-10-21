//! Capacity eviction order, retained L1 and actual disk accounting; v1.md §3.5.
use super::*;
use crate::store::{self, StoreLimits, vectors::VectorTableSpec};
use std::error::Error as StdError;

#[test]
fn shrink_evicts_generated_page_sections_with_l3_before_l2_and_retains_anchor()
-> Result<(), Box<dyn StdError>> {
    let (_dir, mut store) = fixture(true)?;
    store.writer()?.execute_batch("UPDATE files SET kind='doc' WHERE id=1;
        INSERT INTO unit_contents(hash,text,tokens) VALUES(x'ffff','',0);
        INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,ladder_level,created_at,updated_at)
        VALUES(2000,x'ffff',x'ffff','low.rs',1,'image','page 1',0,0,0);
        UPDATE units SET kind='doc_section',file_id=1,parent_unit=2000,source_key='low.rs' WHERE id>=1001 AND id<=1004;
        INSERT INTO page_publications(file_id,page,parent_unit,desired_hash,sections,output_hash)
        VALUES(1,1,2000,zeroblob(32),4,zeroblob(32));
        CREATE TRIGGER record_page AFTER DELETE ON units WHEN OLD.kind='doc_section' AND OLD.ladder_level=3
        BEGIN INSERT INTO eviction_events(kind,unit_id) VALUES('page',OLD.id); END;")?;
    store.maintain()?;
    wal::checkpoint(&store)?;
    let allocated = sizes(&*store.writer()?)?.0;
    store.db_max_bytes = allocated;
    let report = store.shrink()?;
    assert_eq!(
        (report.summaries_removed, report.page_sections_removed),
        (0, 4)
    );
    assert!(report.vectors_removed > 0);
    assert!(store.unit_by_id(2000)?.is_some());
    let db = store.writer()?;
    assert_eq!(
        db.query_row("SELECT count(*) FROM units WHERE kind='chunk'", [], |r| r
            .get::<_, i64>(
            0
        ))?,
        512
    );
    let (last_page,first_vector):(i64,i64)=db.query_row("SELECT (SELECT max(id) FROM eviction_events WHERE kind='page'),(SELECT min(id) FROM eviction_events WHERE kind='vector')",[],|r|Ok((r.get(0)?,r.get(1)?)))?;
    assert!(last_page < first_vector);
    // The retained manifest no longer proves complete output after explicit removal.
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM units WHERE parent_unit=2000",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        0
    );
    Ok(())
}

fn fixture(vectors: bool) -> Result<(tempfile::TempDir, Store), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    let store = store::open(dir.path().join("code.db"), StoreLimits::default())?;
    for (slug, active) in [("current", true), ("retired", false)] {
        store.ensure_vector_table(VectorTableSpec {
            slug,
            model: slug,
            profile: "test",
            dims: 1024,
            created_at: 1,
            active,
        })?;
    }
    {
        let mut db = store.writer()?;
        let tx = db.transaction()?;
        tx.execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment,rank) VALUES(1,'low.rs',zeroblob(32),1,1,'code',3,'code',0.1),(2,'high.rs',zeroblob(32),1,1,'code',3,'code',0.9);
            CREATE TABLE eviction_events(id INTEGER PRIMARY KEY,kind TEXT,unit_id INTEGER);
            CREATE TRIGGER record_summary AFTER DELETE ON units WHEN OLD.kind='summary' BEGIN INSERT INTO eviction_events(kind,unit_id) VALUES('summary',OLD.id); END;
            CREATE TRIGGER record_current AFTER DELETE ON vec_current BEGIN INSERT INTO eviction_events(kind,unit_id) VALUES('vector',OLD.unit_id); END;
            CREATE TRIGGER record_retired AFTER DELETE ON vec_retired BEGIN INSERT INTO eviction_events(kind,unit_id) VALUES('vector',OLD.unit_id); END;")?;
        let vector = half::f16::from_f32(1.0 / 32.0)
            .to_bits()
            .to_le_bytes()
            .repeat(1024);
        for id in 1_i64..=512 {
            let mut hash = [0_u8; 32];
            hash[..8].copy_from_slice(&id.to_le_bytes());
            tx.execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,?2,3)",
                params![hash.as_slice(), format!("retained source {id}")],
            )?;
            tx.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,ladder_level,created_at,updated_at) VALUES(?1,?2,?2,'fixture:'||?1,?3,'chunk','retained',?4,1,1)",params![id,hash.as_slice(),if id<=256 {1}else{2},if vectors{2}else{1}])?;
            tx.execute(
                "INSERT INTO fts_units(rowid,header,body) VALUES(?1,'retained',?2)",
                params![id, format!("retained source {id}")],
            )?;
            if vectors {
                tx.execute(
                    "INSERT INTO vec_current(unit_id,v,ctx_hash) VALUES(?1,?2,zeroblob(32))",
                    params![id, &vector],
                )?;
                if id <= 16 {
                    tx.execute(
                        "INSERT INTO vec_retired(unit_id,v,ctx_hash) VALUES(?1,?2,zeroblob(32))",
                        params![id, &vector],
                    )?;
                }
            }
        }
        if vectors {
            for id in 1001_i64..=1004 {
                let mut hash = [0_u8; 32];
                hash[..8].copy_from_slice(&id.to_le_bytes());
                tx.execute(
                    "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,?2,16384)",
                    params![hash.as_slice(), "summary ".repeat(8192)],
                )?;
                tx.execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,ladder_level,created_at,updated_at) VALUES(?1,?2,?2,'summary:'||?1,'summary','summary',3,1,1)",params![id,hash.as_slice()])?;
                tx.execute(
                    "INSERT INTO fts_units(rowid,header,body) VALUES(?1,'summary','summary')",
                    [id],
                )?;
            }
        }
        tx.commit()?;
    }
    store.maintain()?;
    wal::checkpoint(&store)?;
    Ok((dir, store))
}

#[test]
fn shrink_removes_summaries_then_low_rank_vectors_and_reaches_physical_target()
-> Result<(), Box<dyn StdError>> {
    let (dir, mut store) = fixture(true)?;
    let before = std::fs::metadata(dir.path().join("code.db"))?.len();
    store.db_max_bytes = before;
    let report = store.shrink()?;
    assert_eq!(report.state, "target_reached", "{report:?}");
    assert_eq!(report.summaries_removed, 4);
    assert!(
        report.vectors_removed > 16 && report.vectors_removed < 528,
        "{report:?}"
    );
    assert!(report.file_bytes <= report.target_bytes);
    assert_eq!(
        report.file_bytes,
        std::fs::metadata(dir.path().join("code.db"))?.len()
    );
    let db = store.writer()?;
    assert_eq!(
        db.query_row("SELECT count(*) FROM units WHERE kind='chunk'", [], |r| r
            .get::<_, i64>(
            0
        ))?,
        512
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM fts_units WHERE fts_units MATCH 'retained'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        512
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM unit_contents", [], |r| r
            .get::<_, i64>(0))?,
        512
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM vec_retired", [], |r| r
            .get::<_, i64>(0))?,
        0
    );
    assert_eq!(
        db.query_row(
            "SELECT max(id) FROM eviction_events WHERE kind='summary'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        4
    );
    let evicted: Vec<i64> = db
        .prepare(
            "SELECT DISTINCT unit_id FROM eviction_events WHERE kind='vector' ORDER BY unit_id",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    assert_eq!(evicted, (1..=evicted.len() as i64).collect::<Vec<_>>());
    assert_eq!(db.query_row("SELECT count(*) FROM units WHERE ladder_level=2 AND id NOT IN (SELECT unit_id FROM vec_current)",[],|r|r.get::<_,i64>(0))?,0);
    assert_eq!(
        db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
        "ok"
    );
    drop(db);
    let again = store.shrink()?;
    assert_eq!((again.summaries_removed, again.vectors_removed), (0, 0));
    Ok(())
}

#[test]
fn lowered_cap_retains_l1_and_reports_unreachable_without_eviction() -> Result<(), Box<dyn StdError>>
{
    let (dir, store) = fixture(false)?;
    let before = std::fs::metadata(dir.path().join("code.db"))?.len();
    drop(store);
    let store = store::open(
        dir.path().join("code.db"),
        StoreLimits {
            db_max_bytes: before,
            ..Default::default()
        },
    )?;
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
        512
    );
    let report = store.shrink()?;
    assert_eq!(report.state, "cannot_reach_target", "{report:?}");
    assert_eq!((report.summaries_removed, report.vectors_removed), (0, 0));
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM units", [], |r| r.get::<_, i64>(0))?,
        512
    );
    Ok(())
}

#[test]
fn held_reader_keeps_physical_reclamation_honest_and_resume_preserves_remaining_vectors()
-> Result<(), Box<dyn StdError>> {
    let (dir, mut store) = fixture(true)?;
    let path = dir.path().join("code.db");
    store.db_max_bytes = std::fs::metadata(&path)?.len();
    let mut db = Connection::open(&path)?;
    let reader = db.transaction()?;
    assert_eq!(
        reader.query_row("SELECT count(*) FROM vec_current", [], |r| r
            .get::<_, i64>(0))?,
        512
    );
    let first = store.shrink()?;
    assert_eq!(first.state, "paused", "{first:?}");
    assert!(first.checkpoint_busy, "{first:?}");
    assert!(first.allocated_bytes <= first.target_bytes, "{first:?}");
    assert!(first.file_bytes > first.target_bytes, "{first:?}");
    let retained = store
        .writer()?
        .query_row("SELECT count(*) FROM vec_current", [], |r| {
            r.get::<_, i64>(0)
        })?;
    assert_eq!(
        reader.query_row("SELECT count(*) FROM vec_current", [], |r| r
            .get::<_, i64>(0))?,
        512
    );
    assert!(wal::wal_bytes(&*store.writer()?)? <= store.wal_max_bytes);
    reader.commit()?;
    let resumed = store.shrink()?;
    assert_eq!(resumed.state, "target_reached", "{resumed:?}");
    assert_eq!((resumed.summaries_removed, resumed.vectors_removed), (0, 0));
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM vec_current", [], |r| r
                .get::<_, i64>(0))?,
        retained
    );
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT count(*) FROM units WHERE kind='chunk'", [], |r| r
                .get::<_, i64>(
                0
            ))?,
        512
    );
    Ok(())
}
