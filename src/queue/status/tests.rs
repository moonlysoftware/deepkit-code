//! Shared read-only queue evidence preserves status semantics and durable accounting.
use super::*;
use rusqlite::{Connection, OpenFlags, params};
use serde_json::Value;
use std::fs;

const NOW: i64 = 100;
const PRIVATE_PROVIDER: &str = "private-provider-identifier-fixture";
const PRIVATE_ERROR: &str = "private-provider-error-body-fixture";

fn seed(store: &Store) {
    let connection = store.writer().unwrap();
    for (id, state, attempts, not_before, leased_until, reason) in [
        (1, "queued", 0, 0, None, None),
        (2, "failed", 1, 110, None, Some(retry::RATE_REASON)),
        (3, "failed", 8, 0, None, Some(PRIVATE_ERROR)),
        (4, "leased", 1, 0, Some(99), None),
        (5, "dead", 8, 0, None, Some(PRIVATE_ERROR)),
        (6, "done", 1, 0, None, None),
        (
            7,
            "queued",
            0,
            capacity::PAUSED,
            None,
            Some(capacity::Reason::Wal.message()),
        ),
        (8, "queued", 0, 120, None, Some(PRIVATE_ERROR)),
        (9, "leased", 1, 0, Some(200), None),
    ] {
        connection.execute(
            "INSERT INTO jobs(id,kind,target_kind,target_id,target_hash,desired_hash,provider,priority,tokens,state,attempts,not_before,leased_until,lease_owner,lease_generation,last_error,created_at,updated_at)
             VALUES(?1,'embed','unit',?1,zeroblob(32),zeroblob(32),?2,1,3,?3,?4,?5,?6,'fixture-owner',1,?7,0,0)",
            params![id, PRIVATE_PROVIDER, state, attempts, not_before, leased_until, reason],
        ).unwrap();
    }
    connection.execute("INSERT INTO usage(day,provider,calls,tokens,estimated_tokens) VALUES('1970-01-01',?1,5,50,60)", [PRIVATE_PROVIDER]).unwrap();
    connection.execute("INSERT INTO usage(day,provider,calls,tokens,estimated_tokens) VALUES('1969-12-31',?1,99,999,999)", [PRIVATE_PROVIDER]).unwrap();
    for (id, state, calls, tokens) in [
        ("a", "reserved", 2, 20),
        ("b", "dispatched", 3, 30),
        ("c", "released", 7, 70),
        ("d", "reconciled", 11, 110),
    ] {
        connection.execute(
            "INSERT INTO usage_reservations(id,provider,day,calls,tokens,state,created_at,dispatched_at)
             VALUES(?1,?2,'1969-12-31',?3,?4,?5,0,0)",
            params![id, PRIVATE_PROVIDER, calls, tokens, state],
        ).unwrap();
    }
}

fn ro(path: &std::path::Path) -> Connection {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .unwrap()
}

fn observation_json(snapshot: Snapshot) -> Value {
    let mut value = serde_json::to_value(snapshot).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("wal");
    object.remove("database");
    object.remove("enrichment_claims");
    value
}

#[test]
fn read_only_observation_matches_status_with_and_without_job_listing() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("code.db");
    let store = crate::store::open(&path, Default::default()).unwrap();
    seed(&store);
    for limit in [None, Some(3)] {
        let expected = observation_json(snapshot_with_jobs(&store, NOW, limit).unwrap());
        let mut connection = ro(&path);
        let transaction = connection.transaction().unwrap();
        let observed = read_in(&transaction, NOW, limit).unwrap();
        assert_eq!(serde_json::to_value(observed).unwrap(), expected);
        // Extraction leaves the caller's transaction live and never commits it.
        assert!(!transaction.is_autocommit());
    }
}

#[test]
fn health_preserves_leases_delays_attempt_caps_and_uncertain_charges_without_writes() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("code.db");
    let store = crate::store::open(&path, Default::default()).unwrap();
    seed(&store);
    let before = observation_json(snapshot_with_jobs(&store, NOW, Some(1000)).unwrap());
    let main_bytes = fs::read(&path).unwrap();
    let wal_path = root.path().join("code.db-wal");
    let wal_bytes = fs::read(&wal_path).unwrap();
    let mut connection = ro(&path);
    let transaction = connection.transaction().unwrap();
    let health = read_in(&transaction, NOW, None).unwrap().health().unwrap();
    assert_eq!(health.jobs.queued, 3);
    assert_eq!(health.jobs.failed, 2);
    assert_eq!(health.jobs.dead, 1);
    assert_eq!(health.jobs.done, 1);
    assert_eq!(health.jobs.leased, 2);
    assert_eq!(health.jobs.ready, 1);
    assert_eq!(health.jobs.deferred, 3);
    assert_eq!(health.jobs.expired_leases, 1);
    assert_eq!(health.jobs.next_ready_at, Some(110));
    assert_eq!(
        (
            health.calls_today,
            health.tokens_today,
            health.estimated_tokens_today
        ),
        (5, 50, 60)
    );
    assert_eq!((health.reserved_calls, health.reserved_tokens), (2, 20));
    assert_eq!((health.uncertain_calls, health.uncertain_tokens), (3, 30));
    assert_eq!(
        health.deferrals[&HealthCategory::ProviderRate].earliest_at,
        Some(110)
    );
    assert_eq!(health.deferrals[&HealthCategory::Other].jobs, 1);
    assert_eq!(health.deferrals[&HealthCategory::CapWal].jobs, 1);
    assert_eq!(health.deferrals[&HealthCategory::CapWal].earliest_at, None);
    let serialized = serde_json::to_string(&health).unwrap();
    assert!(!serialized.contains(PRIVATE_PROVIDER));
    assert!(!serialized.contains(PRIVATE_ERROR));
    assert!(!serialized.contains("fixture-owner"));
    drop(transaction);
    drop(connection);
    assert_eq!(
        observation_json(snapshot_with_jobs(&store, NOW, Some(1000)).unwrap()),
        before
    );
    assert_eq!(fs::read(&path).unwrap(), main_bytes);
    assert_eq!(fs::read(&wal_path).unwrap(), wal_bytes);
}

#[test]
fn read_only_transaction_observes_committed_wal_without_recovery_or_implicit_commit() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("code.db");
    let store = crate::store::open(&path, Default::default()).unwrap();
    seed(&store);
    let mut connection = ro(&path);
    let transaction = connection.transaction().unwrap();
    assert_eq!(
        read_in(&transaction, NOW, None)
            .unwrap()
            .health()
            .unwrap()
            .jobs
            .ready,
        1
    );
    // The existing reader pins its WAL snapshot; a separate ordinary writer commits.
    store
        .writer()
        .unwrap()
        .execute("UPDATE jobs SET not_before=0 WHERE id=2", [])
        .unwrap();
    assert_eq!(
        read_in(&transaction, NOW, None)
            .unwrap()
            .health()
            .unwrap()
            .jobs
            .ready,
        1
    );
    drop(transaction);
    let transaction = connection.transaction().unwrap();
    assert_eq!(
        read_in(&transaction, NOW, None)
            .unwrap()
            .health()
            .unwrap()
            .jobs
            .ready,
        2
    );
}

#[test]
fn health_aggregates_providers_and_refuses_invalid_numeric_evidence() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("code.db");
    let store = crate::store::open(&path, Default::default()).unwrap();
    seed(&store);
    store.writer().unwrap().execute("INSERT INTO usage(day,provider,calls,tokens,estimated_tokens) VALUES('1970-01-01','second-private-provider',2,7,8)", []).unwrap();
    let mut connection = ro(&path);
    let transaction = connection.transaction().unwrap();
    let observed = read_in(&transaction, NOW, None).unwrap();
    let health = observed.health().unwrap();
    assert_eq!(
        (
            health.calls_today,
            health.tokens_today,
            health.estimated_tokens_today
        ),
        (7, 57, 68)
    );
    assert!(
        !serde_json::to_string(&health)
            .unwrap()
            .contains("second-private-provider")
    );
    drop(transaction);
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE usage SET calls=-1 WHERE provider='second-private-provider'",
            [],
        )
        .unwrap();
    let transaction = connection.transaction().unwrap();
    assert!(matches!(
        read_in(&transaction, NOW, None).unwrap().health(),
        Err(Error::Invalid("negative queue health counter"))
    ));
}

#[test]
fn health_rejects_mixed_sign_reservations_without_changing_status() {
    for state in ["reserved", "dispatched"] {
        for (calls, tokens) in [(-1, 1), (1, -1)] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("code.db");
            let store = crate::store::open(&path, Default::default()).unwrap();
            seed(&store);
            store
                .writer()
                .unwrap()
                .execute(
                    "INSERT INTO usage_reservations(id,provider,day,calls,tokens,state,created_at)
                 VALUES('negative-fixture',?1,'1969-12-31',?2,?3,?4,0)",
                    params![PRIVATE_PROVIDER, calls, tokens, state],
                )
                .unwrap();
            let expected = observation_json(snapshot_with_jobs(&store, NOW, None).unwrap());
            // The old status contract still reports the signed SQL sums, which
            // remain positive here despite an invalid individual reservation.
            let (calls_key, tokens_key, base_calls, base_tokens) = if state == "reserved" {
                ("reserved_calls", "reserved_tokens", 2, 20)
            } else {
                ("uncertain_calls", "uncertain_tokens", 3, 30)
            };
            let provider = &expected["providers"][PRIVATE_PROVIDER];
            assert_eq!(provider[calls_key], base_calls + calls);
            assert_eq!(provider[tokens_key], base_tokens + tokens);
            let mut connection = ro(&path);
            let transaction = connection.transaction().unwrap();
            let observed = read_in(&transaction, NOW, None).unwrap();
            assert_eq!(serde_json::to_value(&observed).unwrap(), expected);
            assert!(matches!(
                observed.health(),
                Err(Error::Invalid("negative queue health counter"))
            ));
            assert!(!transaction.is_autocommit());
            drop(transaction);
            assert_eq!(
                observation_json(snapshot_with_jobs(&store, NOW, None).unwrap()),
                expected
            );
        }
    }
}

#[test]
fn provider_codes_are_allowlisted_and_never_expose_persisted_text() {
    let root = tempfile::tempdir().unwrap();
    let store = crate::store::open(root.path().join("code.db"), Default::default()).unwrap();
    seed(&store);
    for (reason, known) in [
        ("chat provider HTTP 400; context_length_exceeded", true),
        (
            "chat provider HTTP 400; private-provider-error-body-fixture",
            false,
        ),
        (
            "chat provider HTTP 400; context_length_exceeded private-provider-error-body-fixture",
            false,
        ),
    ] {
        store
            .writer()
            .unwrap()
            .execute("UPDATE jobs SET last_error=?1 WHERE id=5", [reason])
            .unwrap();
        let snapshot =
            serde_json::to_value(snapshot_with_jobs(&store, NOW, Some(1000)).unwrap()).unwrap();
        let rendered = snapshot.to_string();
        assert!(!rendered.contains(PRIVATE_ERROR));
        assert_eq!(rendered.contains("context_length_exceeded"), known);
        assert_eq!(retry::chat_http_status(reason), known.then_some(400));
    }
}
