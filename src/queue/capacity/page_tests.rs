//! Variable-width page reservations and durable capacity recovery; v1 §3.5/§8.
use super::*;
use crate::{
    queue::{self, Kind, Target},
    store::{self, StoreLimits},
};

fn page(store: &Store, id: i64, provider: &str, units: u32) -> i64 {
    let input = queue::Enqueue {
        kind: Kind::VlmPage,
        target: Target::File,
        provider,
        ..queue::tests::input(id, &[1; 32])
    };
    let job = queue::enqueue(store, &input, 0).unwrap();
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE jobs SET page=1,required_l3_units=?2 WHERE id=?1",
            params![job, units],
        )
        .unwrap();
    job
}

#[test]
fn page_replacement_credit_requires_exact_file_hash_and_ordinal() {
    let store = store::open(
        ":memory:",
        StoreLimits {
            max_l3_units: 2,
            ..Default::default()
        },
    )
    .unwrap();
    // Bind the job before filling coverage. Generic enqueue has no ordinal and
    // correctly cannot claim page-specific replacement credit at a full cap.
    let id = page(&store, 1, "chat:owned", 2);
    store.writer().unwrap().execute_batch("INSERT INTO files(id,path,blake3,size,mtime_ns,kind,max_level,treatment)
        VALUES(1,'scan.pdf',zeroblob(32),1,0,'doc',3,'fixture');
        INSERT INTO unit_contents(hash,text,tokens) VALUES(zeroblob(32),'',0),(x'01','first',1),(x'02','second',1);
        INSERT INTO units(id,hash,content_hash,source_key,file_id,kind,header,ladder_level,parent_unit,created_at,updated_at)
        VALUES(1,x'11',zeroblob(32),'scan.pdf',1,'image','page 1',0,NULL,0,0),
        (2,x'12',x'01','scan.pdf',1,'doc_section','first',3,1,0,0),
        (3,x'13',x'02','scan.pdf',1,'doc_section','second',3,1,0,0);").unwrap();
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE files SET blake3=?1 WHERE id=1",
            [[7_u8; 32].as_slice()],
        )
        .unwrap();
    store
        .writer()
        .unwrap()
        .execute(
            "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'',0)",
            [[7_u8; 32].as_slice()],
        )
        .unwrap();
    // The page owner and file must share the exact source bytes.
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE units SET content_hash=?1 WHERE id=1",
            [[7_u8; 32].as_slice()],
        )
        .unwrap();
    let mut input = queue::Enqueue {
        kind: Kind::VlmPage,
        target: Target::File,
        ..queue::tests::input(1, &[1; 32])
    };
    let growth = |input: &queue::Enqueue<'_>, ordinal, required| {
        input_growth(&store.writer().unwrap(), input, Some(ordinal), required).unwrap()
    };
    assert_eq!(growth(&input, 1, 3), 1);
    assert_eq!(growth(&input, 2, 3), 3);
    input.target_id = 2;
    assert_eq!(growth(&input, 1, 3), 3);
    input.target_id = 1;
    input.target_hash = &[8; 32];
    assert_eq!(growth(&input, 1, 3), 3);
    let owner = queue::Owner::new(&store).unwrap();
    assert_eq!(
        queue::claim(&store, "chat:owned", &owner, 0, 1, 100).unwrap()[0].id,
        id
    );
    assert_eq!(
        enrichment_claim_status(&store, &store.writer().unwrap(), 0)
            .unwrap()
            .growth_units,
        0
    );
}

#[test]
fn page_claims_reserve_sections_across_providers_and_fit_smaller_jobs() {
    let store = store::open(
        ":memory:",
        StoreLimits {
            max_l3_units: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let a = page(&store, 1, "chat:a", 3);
    let b = page(&store, 2, "chat:b", 2);
    let c = page(&store, 3, "chat:b", 1);
    let owner = queue::Owner::new(&store).unwrap();
    assert_eq!(
        queue::claim(&store, "chat:a", &owner, 0, 10, 100).unwrap()[0].id,
        a
    );
    let batch = queue::claim(&store, "chat:b", &owner, 0, 10, 100).unwrap();
    assert_eq!(batch.iter().map(|l| l.id).collect::<Vec<_>>(), [c]);
    let status = enrichment_claim_status(&store, &store.writer().unwrap(), 0).unwrap();
    assert_eq!(
        (
            status.active_leases,
            status.growth_leases,
            status.growth_units
        ),
        (2, 2, 4)
    );
    assert_eq!(status.available_growth(), 0);
    // Expired leases cannot publish or reserve room in another provider's claim.
    let later = queue::claim(&store, "chat:b", &owner, queue::LEASE_SECONDS + 1, 1, 100).unwrap();
    assert_eq!(later[0].id, b);
    let status =
        enrichment_claim_status(&store, &store.writer().unwrap(), queue::LEASE_SECONDS + 1)
            .unwrap();
    assert_eq!((status.active_leases, status.growth_units), (1, 2));
}

#[test]
fn observed_page_requirement_survives_restart_and_refuses_insufficient_resume() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("code.db");
    let limits = StoreLimits {
        max_l3_units: 2,
        ..Default::default()
    };
    let store = store::open(&path, limits).unwrap();
    let id = page(&store, 1, "chat:page", 1);
    let owner = queue::Owner::new(&store).unwrap();
    let lease = queue::claim(&store, "chat:page", &owner, 0, 1, 100)
        .unwrap()
        .remove(0);
    assert_eq!(
        queue::retry::schedule(
            &store,
            &lease,
            queue::retry::Outcome::PageCapacity {
                reason: Reason::Summaries,
                units: 3
            },
            || 1
        )
        .unwrap(),
        queue::retry::Scheduled::RetryAt(PAUSED)
    );
    assert_eq!(resume(&store, "chat:page", || 2).unwrap(), 0);
    assert!(
        queue::claim(&store, "chat:page", &owner, 2, 1, 100)
            .unwrap()
            .is_empty()
    );
    // A superseded worker cannot overwrite the persisted output requirement.
    assert_eq!(
        queue::retry::schedule(
            &store,
            &lease,
            queue::retry::Outcome::PageCapacity {
                reason: Reason::Summaries,
                units: 1
            },
            || 2
        )
        .unwrap(),
        queue::retry::Scheduled::LeaseLost
    );
    drop(store);
    let store = store::open(&path, limits).unwrap();
    let required: i64 = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT required_l3_units FROM jobs WHERE id=?1",
            [id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(required, 3);
    assert_eq!(resume(&store, "chat:page", || 3).unwrap(), 0);
    drop(store);
    let store = store::open(
        &path,
        StoreLimits {
            max_l3_units: 3,
            ..limits
        },
    )
    .unwrap();
    assert_eq!(resume(&store, "chat:page", || 4).unwrap(), 1);
    let next = queue::claim(
        &store,
        "chat:page",
        &queue::Owner::new(&store).unwrap(),
        4,
        1,
        100,
    )
    .unwrap();
    assert_eq!(next[0].id, id);
    assert!(next[0].generation > lease.generation);
}
