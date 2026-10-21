//! Public hash-prefix range lookup regressions; v1 §§3.2, 10.1.
use super::*;

fn fixture() -> Store {
    let store = open(":memory:", StoreLimits::default()).unwrap();
    store.writer().unwrap().execute(
        "INSERT INTO files(id,path,blake3,size,mtime_ns,lang,kind,max_level,treatment,l1_pending,l1_desired,l1_fingerprint)
         VALUES(1,'fixture.rs',?1,1,0,'rust','code',3,'code',0,?2,?2)",
        params![[7_u8;32].as_slice(), [8_u8;32].as_slice()],
    ).unwrap();
    store
}

fn hash(prefix: [u8; 6], tail: u8) -> [u8; 32] {
    let mut value = [tail; 32];
    value[..6].copy_from_slice(&prefix);
    value
}

// Explicit hashes model prefix collisions without brute-force hashing. The
// payload/occurrence relation and unique full hashes remain schema-valid.
fn insert(store: &Store, id: i64, occurrence: &[u8], content: &[u8]) {
    let writer = store.writer().unwrap();
    writer
        .execute(
            "INSERT OR IGNORE INTO unit_contents(hash,text,tokens) VALUES(?1,'fixture',1)",
            [content],
        )
        .unwrap();
    writer.execute("INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,line_start,line_end,created_at,updated_at)
        VALUES(?1,?2,?3,'fixture.rs',1,'chunk','fixture',1,1,0,0)",params![id,occurrence,content]).unwrap();
}

#[test]
fn public_ids_accept_raw_and_prefixed_lowercase_at_min_carry_and_max_boundaries() {
    let store = fixture();
    for (id, prefix, text) in [
        (1, [0, 0, 0, 0, 0, 0], "000000000000"),
        (2, [0x12, 0x34, 0x56, 0x78, 0xff, 0xff], "12345678ffff"),
        (3, [0xff; 6], "ffffffffffff"),
    ] {
        insert(&store, id, &hash(prefix, 0xff), &hash([id as u8; 6], 0));
        for public in [text.to_owned(), format!("u_{text}")] {
            assert_eq!(store.unit_by_public_id(&public).unwrap().unwrap().id, id);
            let source = store
                .open_source_checked(&public, &|| Ok(()))
                .unwrap()
                .unwrap();
            assert_eq!(source.occurrence, hash(prefix, 0xff));
        }
    }
    // First value at the incremented upper bound must not leak into the range.
    insert(
        &store,
        4,
        &hash([0x12, 0x34, 0x56, 0x79, 0, 0], 0),
        &hash([4; 6], 0),
    );
    assert_eq!(
        store.unit_by_public_id("12345678ffff").unwrap().unwrap().id,
        2
    );
    // Content aliases retain their historical unit lookup behavior.
    assert_eq!(
        store.unit_by_public_id("020202020202").unwrap().unwrap().id,
        2
    );
    assert!(
        store
            .open_source_checked("020202020202", &|| Ok(()))
            .unwrap()
            .is_none()
    );
}

#[test]
fn malformed_public_ids_never_match_or_accept_case_folding() {
    let store = fixture();
    insert(&store, 1, &hash([0xab; 6], 0), &hash([0xcd; 6], 0));
    for public in [
        "",
        "u_",
        "ABABABABABAB",
        "u_ABABABABABAB",
        "U_abababababab",
        "abababababa",
        "ababababababa",
        "abababababa!",
        "abababababag",
        " abababababab",
        "abababababab ",
        "u_u_abababababab",
        "λabababababa",
        "ababab\0babab",
    ] {
        assert!(
            store.unit_by_public_id(public).unwrap().is_none(),
            "{public:?}"
        );
        assert!(
            store
                .open_source_checked(public, &|| Ok(()))
                .unwrap()
                .is_none(),
            "{public:?}"
        );
    }
    assert!(store.unit_by_public_id("abababababab").unwrap().is_some());
}

#[test]
fn matching_both_aliases_deduplicates_one_row_but_cross_row_matches_are_ambiguous() {
    let store = fixture();
    let prefix = [0x42; 6];
    insert(&store, 1, &hash(prefix, 1), &hash(prefix, 2));
    assert_eq!(
        store.unit_by_public_id("424242424242").unwrap().unwrap().id,
        1
    );
    // Second row matches through content while the first matches through both.
    insert(&store, 2, &hash([0x71; 6], 1), &hash(prefix, 3));
    for public in ["424242424242", "u_424242424242"] {
        assert!(
            matches!(store.unit_by_public_id(public),Err(Error::AmbiguousUnitId{public_id}) if public_id==public)
        );
    }
    let occurrences = fixture();
    insert(&occurrences, 1, &hash(prefix, 1), &hash([1; 6], 0));
    insert(&occurrences, 2, &hash(prefix, 2), &hash([2; 6], 0));
    assert!(matches!(
        occurrences.unit_by_public_id("424242424242"),
        Err(Error::AmbiguousUnitId { .. })
    ));
}

#[test]
fn open_source_range_lookup_preserves_completed_source_fences_and_work_checks() {
    let store = fixture();
    insert(&store, 1, &hash([0x42; 6], 0), &hash([3; 6], 0));
    let source = store
        .open_source_checked("u_424242424242", &|| Ok(()))
        .unwrap()
        .unwrap();
    assert_eq!(source.hash, vec![7; 32]);
    assert_eq!(source.fingerprint, vec![8; 32]);
    for update in [
        "l1_pending=1",
        "l1_fingerprint=NULL",
        "l1_desired=NULL",
        "l1_desired=zeroblob(32)",
    ] {
        store
            .writer()
            .unwrap()
            .execute_batch(&format!("UPDATE files SET {update} WHERE id=1"))
            .unwrap();
        assert!(
            store
                .open_source_checked("u_424242424242", &|| Ok(()))
                .unwrap()
                .is_none()
        );
        // An incomplete source does not erase the independently addressable unit.
        assert!(store.unit_by_public_id("u_424242424242").unwrap().is_some());
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE files SET l1_pending=0,l1_fingerprint=?1,l1_desired=?1 WHERE id=1",
                [[8_u8; 32].as_slice()],
            )
            .unwrap();
    }
    assert_eq!(
        store
            .open_source_checked("424242424242", &|| Ok(()))
            .unwrap(),
        Some(source)
    );
    assert!(matches!(
        store.open_source_checked("424242424242", &|| Err(Error::L1InputsChanged)),
        Err(Error::L1InputsChanged)
    ));
    assert!(matches!(
        store.unit_by_public_id_checked("424242424242", &|| Err(Error::L1InputsChanged)),
        Err(Error::L1InputsChanged)
    ));
}

#[test]
fn production_public_id_queries_use_existing_hash_indexes_without_units_scan() {
    let store = fixture();
    insert(&store, 1, &hash([0x42; 6], 0), &hash([3; 6], 0));
    let writer = store.writer().unwrap();
    for (sql, last, content_alias) in [
        (UNIT_PUBLIC_ID_RANGE_SQL, false, true),
        (UNIT_PUBLIC_ID_LAST_RANGE_SQL, true, true),
        (OPEN_SOURCE_RANGE_SQL, false, false),
        (OPEN_SOURCE_LAST_RANGE_SQL, true, false),
    ] {
        let mut statement = writer
            .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
            .unwrap();
        let lower = [0x42_u8; 6];
        let upper = [0x42, 0x42, 0x42, 0x42, 0x42, 0x43];
        let lower_bytes = lower.as_slice();
        let upper_bytes = upper.as_slice();
        let parameters: Vec<&dyn rusqlite::ToSql> = if last {
            vec![&lower_bytes]
        } else {
            vec![&lower_bytes, &upper_bytes]
        };
        let details = statement
            .query_map(parameters.as_slice(), |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let plan = details.join("\n");
        assert!(
            !details.iter().any(|detail| detail == "SCAN units"
                || detail.starts_with("SCAN units ")
                || detail == "SCAN u"
                || detail.starts_with("SCAN u ")),
            "{plan}"
        );
        assert!(
            plan.contains("sqlite_autoindex_units_1"),
            "occurrence index missing: {plan}"
        );
        if content_alias {
            assert!(
                plan.contains("units_content"),
                "content index missing: {plan}"
            );
        }
    }
}

#[test]
fn binary_prefix_bounds_carry_and_stop_at_all_ff() {
    assert_eq!(
        public_id_bounds("000000000000"),
        Some(([0; 6], Some([0, 0, 0, 0, 0, 1])))
    );
    assert_eq!(
        public_id_bounds("u_12345678ffff"),
        Some((
            [0x12, 0x34, 0x56, 0x78, 0xff, 0xff],
            Some([0x12, 0x34, 0x56, 0x79, 0, 0])
        ))
    );
    assert_eq!(public_id_bounds("ffffffffffff"), Some(([0xff; 6], None)));
    assert_eq!(public_id_bounds("FFFFFFFFFFFF"), None);
}
