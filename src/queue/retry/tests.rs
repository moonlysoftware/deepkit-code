//! Retry, quota and crash-recovery regressions for v1 §8.2.
use super::*;
use crate::{
    queue::{
        self, Owner,
        usage::{self, Admission, Budget, Limits, Reservation},
    },
    store::{self, StoreLimits},
};

fn setup() -> (Store, Owner, Lease) {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0).unwrap();
    let owner = Owner::new(&store).unwrap();
    let lease = queue::claim(&store, "embed:local", &owner, 0, 1, 10)
        .unwrap()
        .remove(0);
    (store, owner, lease)
}
fn reserve(store: &Store, lease: &Lease, now: i64) -> Reservation {
    match usage::reserve(
        store,
        "embed:local",
        10,
        0,
        Budget {
            calls: None,
            tokens: None,
        },
        Some(lease),
        || now,
    )
    .unwrap()
    {
        Admission::Reserved(r) => r,
        other => panic!("unexpected {other:?}"),
    }
}
fn dispatch(store: &Store, lease: &Lease, now: i64) {
    let r = reserve(store, lease, now);
    assert_eq!(
        usage::mark_dispatched(
            store,
            &r,
            Some(lease),
            || now,
            |_, _| Ok(Limits {
                budget: Budget {
                    calls: None,
                    tokens: None
                },
                rpm: None,
                tpm: None
            })
        )
        .unwrap(),
        usage::Dispatch::Recorded
    );
}
fn job(store: &Store) -> (String, i64, i64) {
    store
        .writer()
        .unwrap()
        .query_row(
            "SELECT state,attempts,not_before FROM jobs WHERE id=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap()
}
fn allocations(store: &Store) -> (i64, i64) {
    store.writer().unwrap().query_row("SELECT COUNT(*) FILTER(WHERE state='reserved'),COUNT(*) FILTER(WHERE state='dispatched') FROM usage_reservations",[],|r|Ok((r.get(0)?,r.get(1)?))).unwrap()
}

#[test]
fn retry_backoff_honors_provider_lower_bound_and_stops_after_eight_dispatches() {
    let (store, owner, mut lease) = setup();
    let mut now = 0;
    for attempt in 1..=8 {
        dispatch(&store, &lease, now);
        let result = schedule(
            &store,
            &lease,
            Outcome::Transient {
                retry_after: Some(Duration::from_millis(90500)),
            },
            || now,
        )
        .unwrap();
        if attempt == 8 {
            assert_eq!(result, Scheduled::Dead);
            assert_eq!(job(&store).0, "dead");
        } else {
            assert_eq!(result, Scheduled::RetryAt(now + 91));
            assert!(
                queue::claim(&store, "embed:local", &owner, now + 90, 1, 10)
                    .unwrap()
                    .is_empty()
            );
            now += 91;
            lease = queue::claim(&store, "embed:local", &owner, now, 1, 10)
                .unwrap()
                .remove(0);
        }
    }
    assert_eq!(job(&store).1, 8);
    assert_eq!(allocations(&store), (0, 8));
    assert!(
        queue::claim(&store, "embed:local", &owner, now + 1000, 1, 10)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn jitter_is_bounded_and_retry_requires_a_dispatched_attempt() {
    let mut delays = std::collections::BTreeSet::new();
    for seed in 0..128 {
        for attempt in 1..8 {
            let delay = retry_delay(attempt, seed).unwrap();
            assert!((1..=60).contains(&delay));
            if attempt == 4 {
                delays.insert(delay);
            }
        }
    }
    assert!(delays.len() > 1);
    let (store, _, lease) = setup();
    assert!(
        schedule(
            &store,
            &lease,
            Outcome::Transient { retry_after: None },
            || 0
        )
        .is_err()
    );
    assert_eq!(job(&store), ("leased".into(), 0, 0));
}

#[test]
fn permanent_empty_and_incomplete_errors_are_terminal_and_never_persist_provider_bodies() {
    for outcome in [
        Outcome::Permanent,
        Outcome::EmptyContent,
        Outcome::IncompleteContent,
    ] {
        let (store, _, lease) = setup();
        dispatch(&store, &lease, 0);
        reserve(&store, &lease, 0);
        assert_eq!(
            schedule(&store, &lease, outcome, || 1).unwrap(),
            Scheduled::Dead
        );
        assert_eq!(allocations(&store), (0, 1));
        assert_eq!(job(&store).1, 1);
        assert_eq!(
            schedule(&store, &lease, Outcome::Rate { retry_at: 60 }, || 2).unwrap(),
            Scheduled::LeaseLost
        );
    }
}

#[test]
fn budget_and_rate_waits_release_leases_without_spending_attempts() {
    for (outcome, until) in [
        (Outcome::Budget { retry_at: 86400 }, 86400),
        (Outcome::Rate { retry_at: 60 }, 60),
    ] {
        let (store, owner, lease) = setup();
        reserve(&store, &lease, 0);
        assert_eq!(
            schedule(&store, &lease, outcome, || 1).unwrap(),
            Scheduled::RetryAt(until)
        );
        assert_eq!(job(&store), ("queued".into(), 0, until));
        assert_eq!(allocations(&store), (0, 0));
        assert!(!queue::renew(&store, &lease, 2).unwrap());
        assert!(
            queue::claim(&store, "embed:local", &owner, until - 1, 1, 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            queue::claim(&store, "embed:local", &owner, until, 1, 10)
                .unwrap()
                .len(),
            1
        );
    }
}

#[test]
fn expired_quota_deadlines_requeue_without_bypassing_fresh_dispatch_admission() {
    for rate in [false, true] {
        for elapsed in [0, 1] {
            let (store, owner, mut lease) = setup();
            let observed = if rate { 59 } else { 86399 };
            let deadline = if rate { 60 } else { 86400 };
            if rate {
                dispatch(&store, &lease, 0);
            } else {
                lease = queue::claim(&store, "embed:local", &owner, observed, 1, 10)
                    .unwrap()
                    .remove(0);
            }
            let held = reserve(&store, &lease, observed);
            let limits = || Limits {
                budget: Budget {
                    calls: if rate { None } else { Some(0) },
                    tokens: None,
                },
                rpm: if rate { Some(1) } else { None },
                tpm: None,
            };
            let admitted = usage::mark_dispatched(
                &store,
                &held,
                Some(&lease),
                || observed,
                |_, _| Ok(limits()),
            )
            .unwrap();
            let outcome = match admitted {
                usage::Dispatch::RateLimited { retry_at } if rate => {
                    assert_eq!(retry_at, deadline);
                    Outcome::Rate { retry_at }
                }
                usage::Dispatch::BudgetExhausted { retry_at } if !rate => {
                    assert_eq!(retry_at, deadline);
                    Outcome::Budget { retry_at }
                }
                other => panic!("unexpected admission {other:?}"),
            };
            let now = deadline + elapsed;
            assert_eq!(
                schedule(&store, &lease, outcome, || now).unwrap(),
                Scheduled::RetryAt(now)
            );
            assert_eq!(job(&store), ("queued".into(), i64::from(rate), now));
            assert_eq!(allocations(&store), (0, i64::from(rate)));
            let next = queue::claim(&store, "embed:local", &owner, now, 1, 10)
                .unwrap()
                .remove(0);
            let held = reserve(&store, &next, now);
            let admission =
                usage::mark_dispatched(&store, &held, Some(&next), || now, |_, _| Ok(limits()))
                    .unwrap();
            if rate {
                assert_eq!(admission, usage::Dispatch::Recorded);
                assert_eq!(job(&store).1, 2);
            } else {
                assert_eq!(
                    admission,
                    usage::Dispatch::BudgetExhausted { retry_at: 172800 }
                );
                assert_eq!(job(&store).1, 0);
            }
        }
    }
}

#[test]
fn recovery_releases_only_unsent_expired_work_and_fences_old_workers() {
    let (store, owner, old) = setup();
    dispatch(&store, &old, 0);
    reserve(&store, &old, 0);
    assert_eq!(
        recover(&store, "embed:local", || 119).unwrap(),
        Recovered::default()
    );
    let current = queue::claim(&store, "embed:local", &owner, 120, 1, 10)
        .unwrap()
        .remove(0);
    assert_eq!(allocations(&store), (0, 1));
    reserve(&store, &current, 120);
    assert_eq!(
        schedule(&store, &old, Outcome::Permanent, || 121).unwrap(),
        Scheduled::LeaseLost
    );
    assert_eq!(allocations(&store), (1, 1));
    assert_eq!(
        recover(&store, "embed:local", || 121).unwrap(),
        Recovered::default()
    );
    for _ in 0..7 {
        dispatch(&store, &current, 121);
    }
    assert_eq!(
        recover(&store, "embed:local", || 240).unwrap(),
        Recovered::default()
    );
    // The last successful admission refreshed the otherwise-expired lease.
    let recovered = recover(&store, "embed:local", || 241).unwrap();
    assert_eq!(
        recovered,
        Recovered {
            jobs: 1,
            reservations: 1
        }
    );
    assert_eq!(job(&store).0, "dead");
    assert_eq!(allocations(&store), (0, 8));
    assert!(
        queue::claim(&store, "embed:local", &owner, 241, 1, 10)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn failed_transition_rolls_back_reservation_release_and_preserves_ownership() {
    let (store, _, lease) = setup();
    reserve(&store, &lease, 0);
    store.writer().unwrap().execute_batch("CREATE TRIGGER fail_retry BEFORE UPDATE OF state ON jobs BEGIN SELECT RAISE(ABORT,'injected retry failure'); END;").unwrap();
    assert!(schedule(&store, &lease, Outcome::Budget { retry_at: 86400 }, || 1).is_err());
    assert_eq!(allocations(&store), (1, 0));
    assert_eq!(job(&store).0, "leased");
}

#[test]
fn midnight_recovery_releases_old_query_holds_but_keeps_uncertain_dispatch() {
    let (store, _, _) = setup();
    let query = || match usage::reserve(
        &store,
        "query",
        1,
        0,
        Budget {
            calls: None,
            tokens: None,
        },
        None,
        || 86399,
    )
    .unwrap()
    {
        Admission::Reserved(r) => r,
        other => panic!("unexpected {other:?}"),
    };
    let sent = query();
    query();
    usage::mark_dispatched(
        &store,
        &sent,
        None,
        || 86399,
        |_, _| {
            Ok(Limits {
                budget: Budget {
                    calls: None,
                    tokens: None,
                },
                rpm: None,
                tpm: None,
            })
        },
    )
    .unwrap();
    assert_eq!(
        recover(&store, "query", || 86399).unwrap(),
        Recovered::default()
    );
    assert_eq!(
        recover(&store, "query", || 86400).unwrap(),
        Recovered {
            jobs: 0,
            reservations: 1
        }
    );
    assert_eq!(allocations(&store), (0, 1));
    assert!(usage::reconcile(&store, &sent, Some(1)).unwrap());
}

#[test]
fn manual_retry_cannot_revive_old_lease_or_reset_attempts() {
    let (store, owner, old) = setup();
    dispatch(&store, &old, 0);
    assert_eq!(
        schedule(&store, &old, Outcome::Permanent, || 1).unwrap(),
        Scheduled::Dead
    );
    assert_eq!(request(&store, Some(old.id), 2).unwrap(), 1);
    let current = queue::claim(&store, "embed:local", &owner, 2, 1, 10)
        .unwrap()
        .remove(0);
    assert!(current.generation > old.generation);
    assert_eq!(request(&store, None, 3).unwrap(), 0);
    assert_eq!(
        schedule(&store, &old, Outcome::Permanent, || 3).unwrap(),
        Scheduled::LeaseLost
    );
    assert!(!queue::renew(&store, &old, 3).unwrap());
    assert!(queue::renew(&store, &current, 3).unwrap());
    assert_eq!(job(&store), ("leased".into(), 1, 1));
    assert_eq!(allocations(&store), (0, 1));
}

#[test]
fn manual_retry_is_atomic_and_rejects_invalid_inputs() {
    let (store, _, lease) = setup();
    dispatch(&store, &lease, 0);
    schedule(&store, &lease, Outcome::Permanent, || 1).unwrap();
    for (id, now) in [(Some(0), 1), (Some(-1), 1), (None, -1)] {
        assert!(request(&store, id, now).is_err());
    }
    store.writer().unwrap().execute_batch("CREATE TRIGGER fail_request AFTER UPDATE OF state ON jobs BEGIN SELECT RAISE(ABORT,'injected request failure'); END;").unwrap();
    assert!(request(&store, None, 2).is_err());
    assert_eq!(job(&store), ("dead".into(), 1, 1));
    assert_eq!(allocations(&store), (0, 1));
}

#[test]
fn recovery_reports_each_phase_progress_and_rechecks_new_ownership() {
    for fail_jobs in [true, false] {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        {
            let mut db = store.writer().unwrap();
            let tx = db.transaction().unwrap();
            for id in 1..=130 {
                queue::enqueue_in(&tx, &queue::tests::input(id, &[1; 32]), 0, None).unwrap();
                tx.execute(
                    "INSERT INTO usage_reservations
                    (id,provider,day,job_id,lease_generation,calls,tokens,state,created_at)
                    VALUES(?1,'embed:local','1970-01-01',?2,1,1,10,'reserved',0)",
                    params![format!("r-{id:04}"), id],
                )
                .unwrap();
            }
            tx.execute_batch("UPDATE jobs SET state='leased',lease_generation=1,lease_owner='old',leased_until=120,attempts=2;").unwrap();
            tx.execute_batch(if fail_jobs {
                "CREATE TRIGGER fail_batch BEFORE UPDATE ON jobs WHEN OLD.id=65 BEGIN SELECT RAISE(ABORT,'fixture'); END;"
            } else {
                "CREATE TRIGGER fail_batch BEFORE UPDATE ON usage_reservations WHEN OLD.id='r-0065' BEGIN SELECT RAISE(ABORT,'fixture'); END;"
            }).unwrap();
            tx.commit().unwrap();
        }
        let Error::RecoveryIncomplete {
            jobs,
            reservations,
            source,
        } = recover(&store, "embed:local", || 120).unwrap_err()
        else {
            panic!("missing committed recovery counts");
        };
        assert_eq!(
            (jobs, reservations),
            if fail_jobs { (64, 0) } else { (130, 64) }
        );
        assert!(matches!(*source, Error::Sql(_)));
        {
            let db = store.writer().unwrap();
            db.execute_batch("DROP TRIGGER fail_batch;
                UPDATE jobs SET state='leased',lease_generation=2,lease_owner='new',leased_until=300 WHERE id=130;
                INSERT INTO usage_reservations
                    (id,provider,day,job_id,lease_generation,calls,tokens,state,created_at)
                    VALUES('current','embed:local','1970-01-01',130,2,1,10,'reserved',121);").unwrap();
        }
        assert_eq!(
            recover(&store, "embed:local", || 121).unwrap(),
            Recovered {
                jobs: if fail_jobs { 65 } else { 0 },
                reservations: 130 - reservations
            }
        );
        let db = store.writer().unwrap();
        assert_eq!(
            db.query_row(
                "SELECT state FROM usage_reservations WHERE id='current'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
            "reserved"
        );
        assert_eq!(
            db.query_row("SELECT lease_owner FROM jobs WHERE id=130", [], |r| r
                .get::<_, String>(0))
                .unwrap(),
            "new"
        );
    }
}

#[test]
fn claim_rechecks_expiry_and_cancellation_after_recovery() {
    let (store, owner, old) = setup();
    dispatch(&store, &old, 0);
    reserve(&store, &old, 0);
    let mut samples = 0;
    let leases = queue::claim_with_clock(
        &store,
        "embed:local",
        &owner,
        || {
            samples += 1;
            if samples == 1 { 119 } else { 120 }
        },
        1,
        10,
    )
    .unwrap();
    assert_eq!(samples, 2);
    assert_eq!(leases.len(), 1);
    assert_eq!(leases[0].generation, old.generation + 1);
    assert_eq!(allocations(&store), (0, 1));
    let mut samples = 0;
    let error = queue::claim_with_clock(
        &store,
        "embed:local",
        &owner,
        || {
            samples += 1;
            if samples == 1 { 240 } else { -1 }
        },
        1,
        10,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Invalid("lease timestamp")));
    assert_eq!(job(&store).0, "failed");
    assert_eq!(allocations(&store), (0, 1));
}

#[test]
fn chat_status_diagnostics_reject_arbitrary_persisted_text() {
    assert_eq!(chat_http_status("chat provider HTTP 400"), Some(400));
    assert_eq!(chat_http_status("chat provider HTTP 413"), Some(413));
    for reason in [
        "permanent provider error",
        "chat provider HTTP 400 synthetic provider body",
        "chat provider HTTP 99",
        "chat provider HTTP 600",
        "chat provider HTTP https://example.invalid/private",
    ] {
        assert_eq!(chat_http_status(reason), None);
    }
    let outcome =
        Outcome::from_provider_attempt(&crate::models::openai_compat::chat::Error::Input.into());
    assert!(matches!(
        outcome,
        Outcome::PermanentChat(ChatFailure::Input)
    ));
}
