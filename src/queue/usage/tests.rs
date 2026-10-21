//! Daily-boundary, durable accounting and independent-process races for v1 §8.
use super::*;
use crate::store::{self, StoreLimits};

const BUDGET: Budget = Budget {
    calls: Some(2),
    tokens: Some(100),
};
fn held(admission: Admission) -> Reservation {
    match admission {
        Admission::Reserved(r) => r,
        other => panic!("expected reservation: {other:?}"),
    }
}
fn book(store: &Store, tokens: i64, now: i64) -> Admission {
    reserve(store, "llm:remote", tokens, 0, BUDGET, None, || now).unwrap()
}

#[test]
fn daily_reservations_include_maximum_output_and_release_only_unsent_work() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    let a = held(reserve(&store, "llm:remote", 30, 40, BUDGET, None, || 86399).unwrap());
    assert!(matches!(
        book(&store, 31, 86399),
        Admission::BudgetExhausted { retry_at: 86400 }
    ));
    assert!(release(&store, &a).unwrap());
    assert!(!release(&store, &a).unwrap());
    let b = held(book(&store, 100, 86399));
    assert_eq!(
        mark_dispatched(&store, &b, None, || 86399, unlimited).unwrap(),
        Dispatch::Recorded
    );
    assert!(!release(&store, &b).unwrap());
    assert_eq!(
        mark_dispatched(&store, &b, None, || 86399, unlimited).unwrap(),
        Dispatch::Unavailable
    );
    assert!(matches!(
        book(&store, 1, 86399),
        Admission::BudgetExhausted { .. }
    ));
    let next = held(book(&store, 100, 86400));
    assert_eq!(
        mark_dispatched(&store, &next, None, || 172800, unlimited).unwrap(),
        Dispatch::DayExpired
    );
    assert!(release(&store, &next).unwrap());
    assert!(reconcile(&store, &b, Some(20)).unwrap());
    assert!(!reconcile(&store, &b, Some(20)).unwrap());
    let usage: (String, i64, i64, i64) = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT day,calls,tokens,estimated_tokens FROM usage",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(usage, ("1970-01-01".into(), 1, 20, 0));
}

#[test]
fn estimates_overruns_and_call_caps_are_persisted_without_double_counting() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    for actual in [None, Some(80)] {
        let a = held(book(&store, 30, 0));
        assert!(!reconcile(&store, &a, actual).unwrap());
        assert_eq!(
            mark_dispatched(&store, &a, None, || 0, unlimited).unwrap(),
            Dispatch::Recorded
        );
        assert!(reconcile(&store, &a, actual).unwrap());
    }
    let totals: (i64, i64, i64) = store
        .writer()
        .unwrap()
        .query_row("SELECT calls,tokens,estimated_tokens FROM usage", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(totals, (2, 110, 30));
    assert!(matches!(
        book(&store, 0, 0),
        Admission::BudgetExhausted { .. }
    ));
    assert!(matches!(
        reserve(
            &store,
            "other",
            0,
            0,
            Budget {
                calls: Some(0),
                tokens: None
            },
            None,
            || 0
        )
        .unwrap(),
        Admission::BudgetExhausted { .. }
    ));
    held(
        reserve(
            &store,
            "other",
            i64::MAX,
            0,
            Budget {
                calls: None,
                tokens: None,
            },
            None,
            || 0,
        )
        .unwrap(),
    );
    assert!(
        reserve(
            &store,
            "other",
            1,
            0,
            Budget {
                calls: None,
                tokens: None
            },
            None,
            || 0
        )
        .is_err()
    );
    assert!(reserve(&store, "other", i64::MAX, 1, BUDGET, None, || 0).is_err());
}

#[test]
fn queued_reservations_cannot_dispatch_under_expired_or_replaced_ownership() {
    use crate::queue::{self, Owner};
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0).unwrap();
    let owner = Owner::new(&store).unwrap();
    let lease = queue::claim(&store, "embed:local", &owner, 0, 1, 10)
        .unwrap()
        .remove(0);
    let a = held(reserve(&store, "embed:local", 10, 0, BUDGET, Some(&lease), || 0).unwrap());
    assert!(matches!(
        reserve(&store, "embed:local", 10, 0, BUDGET, Some(&lease), || 120).unwrap(),
        Admission::LeaseLost
    ));
    assert_eq!(
        mark_dispatched(&store, &a, Some(&lease), || 120, unlimited).unwrap(),
        Dispatch::LeaseLost
    );
    let replacement = queue::claim(&store, "embed:local", &owner, 120, 1, 10)
        .unwrap()
        .remove(0);
    assert_eq!(
        mark_dispatched(&store, &a, Some(&replacement), || 121, unlimited).unwrap(),
        Dispatch::Unavailable
    );
    assert_eq!(
        mark_dispatched(&store, &a, None, || 121, unlimited).unwrap(),
        Dispatch::Unavailable
    );
    // Claim recovery already released the old generation's unsent allocation.
    assert!(!release(&store, &a).unwrap());
}

#[test]
fn queued_admission_refreshes_a_lease_after_long_authoritative_resolution() {
    use crate::queue::{self, Owner};
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0).unwrap();
    let lease = queue::claim(
        &store,
        "embed:local",
        &Owner::new(&store).unwrap(),
        0,
        1,
        10,
    )
    .unwrap()
    .remove(0);
    let reservation =
        held(reserve(&store, "embed:local", 10, 0, BUDGET, Some(&lease), || 0).unwrap());
    let mut clock = [1_i64, 120].into_iter();
    assert_eq!(
        mark_dispatched_checked(
            &store,
            &reservation,
            Some(&lease),
            || clock.next().unwrap(),
            |tx, provider| unlimited(tx, provider).map(Some),
        )
        .unwrap(),
        Dispatch::Recorded
    );
    let job: (i64, i64) = store
        .writer()
        .unwrap()
        .query_row("SELECT attempts,leased_until FROM jobs", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(job, (1, 240));
}

#[test]
fn failed_reconciliation_rolls_back_usage_and_keeps_uncertain_charge() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    let a = held(book(&store, 100, 0));
    mark_dispatched(&store, &a, None, || 0, unlimited).unwrap();
    store.writer().unwrap().execute_batch("CREATE TRIGGER fail_settlement BEFORE UPDATE OF state ON usage_reservations WHEN NEW.state='reconciled' BEGIN SELECT RAISE(ABORT,'injected settlement failure'); END;").unwrap();
    assert!(reconcile(&store, &a, Some(5)).is_err());
    let count: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM usage", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
    assert!(matches!(
        book(&store, 1, 0),
        Admission::BudgetExhausted { .. }
    ));
    store
        .writer()
        .unwrap()
        .execute_batch("DROP TRIGGER fail_settlement")
        .unwrap();
    assert!(reconcile(&store, &a, Some(5)).unwrap());
    held(book(&store, 95, 0));
}

#[test]
fn reservation_process_child() {
    let Ok(path) = std::env::var("DKC_USAGE_TEST_DB") else {
        return;
    };
    let result = std::env::var("DKC_USAGE_TEST_RESULT").unwrap();
    let gate = std::env::var("DKC_USAGE_TEST_GATE").unwrap();
    let store = store::open(path, StoreLimits::default()).unwrap();
    std::fs::write(format!("{result}.ready"), b"ready").unwrap();
    let start = std::time::Instant::now();
    while !std::path::Path::new(&gate).exists() {
        assert!(start.elapsed() < std::time::Duration::from_secs(10));
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let rate = std::env::var("DKC_USAGE_TEST_RATE").as_deref() == Ok("1");
    let admission = book(&store, if rate { 1 } else { 100 }, 0);
    let admitted = if let Admission::Reserved(r) = admission {
        let outcome = mark_dispatched(
            &store,
            &r,
            None,
            || 0,
            |tx, p| {
                let mut limits = unlimited(tx, p)?;
                if rate {
                    limits.rpm = Some(1);
                }
                Ok(limits)
            },
        )
        .unwrap();
        assert!(matches!(
            outcome,
            Dispatch::Recorded | Dispatch::RateLimited { retry_at: 60 }
        ));
        outcome == Dispatch::Recorded
    } else {
        false
    };
    std::fs::write(result, if admitted { b"1" } else { b"0" }).unwrap();
    // Exit with no provider response: durable dispatched accounting must survive.
}

#[test]
fn independent_processes_share_budget_and_restart_retains_uncertain_dispatch() {
    process_race(false);
}

#[test]
fn independent_processes_share_rolling_request_capacity() {
    process_race(true);
}

fn process_race(rate: bool) {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("code.db");
    drop(store::open(&db, StoreLimits::default()).unwrap());
    let gate = dir.path().join("go");
    let results = [dir.path().join("a"), dir.path().join("b")];
    let mut children = results
        .iter()
        .map(|result| {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "queue::usage::tests::reservation_process_child",
                    "--nocapture",
                ])
                .env("DKC_USAGE_TEST_DB", &db)
                .env("DKC_USAGE_TEST_RESULT", result)
                .env("DKC_USAGE_TEST_GATE", &gate)
                .env("DKC_USAGE_TEST_RATE", if rate { "1" } else { "0" })
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let start = std::time::Instant::now();
    while !results.iter().all(|p| p.with_extension("ready").exists()) {
        if start.elapsed() > std::time::Duration::from_secs(10) {
            for child in &mut children {
                let _ = child.kill();
                let _ = child.wait();
            }
            panic!("child admission barrier timed out");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    std::fs::write(&gate, b"go").unwrap();
    for child in &mut children {
        assert!(child.wait().unwrap().success());
    }
    assert_eq!(
        results
            .iter()
            .map(|p| std::fs::read_to_string(p).unwrap().parse::<u32>().unwrap())
            .sum::<u32>(),
        1
    );
    let reopened = store::open(&db, StoreLimits::default()).unwrap();
    assert!(matches!(
        book(&reopened, 1, 0),
        Admission::BudgetExhausted { .. }
    ));
    held(book(&reopened, 100, 86400));
}

fn unlimited(_: &Transaction<'_>, _: &str) -> Result<Limits, Error> {
    Ok(Limits {
        budget: Budget {
            calls: None,
            tokens: None,
        },
        rpm: None,
        tpm: None,
    })
}

fn rates(rpm: i64, tpm: i64) -> Limits {
    Limits {
        budget: Budget {
            calls: None,
            tokens: None,
        },
        rpm: Some(rpm),
        tpm: Some(tpm),
    }
}
fn query_reservation(store: &Store, tokens: i64, now: i64) -> Reservation {
    held(
        reserve(
            store,
            "llm:remote",
            tokens,
            0,
            Budget {
                calls: None,
                tokens: None,
            },
            None,
            || now,
        )
        .unwrap(),
    )
}

#[test]
fn rolling_rates_survive_settlement_and_do_not_reset_at_wall_minute() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    let a = query_reservation(&store, 70, 59);
    assert_eq!(
        mark_dispatched(&store, &a, None, || 59, |_, _| Ok(rates(2, 100))).unwrap(),
        Dispatch::Recorded
    );
    reconcile(&store, &a, Some(1)).unwrap();
    let b = query_reservation(&store, 31, 60);
    assert_eq!(
        mark_dispatched(&store, &b, None, || 60, |_, _| Ok(rates(2, 100))).unwrap(),
        Dispatch::RateLimited { retry_at: 119 }
    );
    assert_eq!(
        mark_dispatched(&store, &b, None, || 118, |_, _| Ok(rates(2, 100))).unwrap(),
        Dispatch::RateLimited { retry_at: 119 }
    );
    assert_eq!(
        mark_dispatched(&store, &b, None, || 119, |_, _| Ok(rates(2, 100))).unwrap(),
        Dispatch::Recorded
    );
    let c = query_reservation(&store, 0, 119);
    assert_eq!(
        mark_dispatched(&store, &c, None, || 119, |_, _| Ok(rates(1, 100))).unwrap(),
        Dispatch::RateLimited { retry_at: 179 }
    );
    assert_eq!(
        mark_dispatched(&store, &c, None, || 119, |_, _| Ok(rates(0, 100))).unwrap(),
        Dispatch::RequestTooLarge
    );
}

#[test]
fn dispatch_rechecks_current_budget_and_rolls_back_failed_attempt_accounting() {
    use crate::queue::{self, Owner};
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0).unwrap();
    let lease = queue::claim(
        &store,
        "embed:local",
        &Owner::new(&store).unwrap(),
        0,
        1,
        10,
    )
    .unwrap()
    .remove(0);
    let a = held(reserve(&store, "embed:local", 10, 0, BUDGET, Some(&lease), || 0).unwrap());
    let reduced = Limits {
        budget: Budget {
            calls: Some(0),
            tokens: None,
        },
        rpm: Some(1),
        tpm: Some(100),
    };
    assert_eq!(
        mark_dispatched(&store, &a, Some(&lease), || 0, |_, _| Ok(reduced)).unwrap(),
        Dispatch::BudgetExhausted { retry_at: 86400 }
    );
    store.writer().unwrap().execute_batch("CREATE TRIGGER reject_dispatch BEFORE UPDATE OF state ON usage_reservations WHEN NEW.state='dispatched' BEGIN SELECT RAISE(ABORT,'injected dispatch failure'); END;").unwrap();
    assert!(mark_dispatched(&store, &a, Some(&lease), || 0, |_, _| Ok(rates(1, 100))).is_err());
    let counts: (i64, i64) = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT attempts,(SELECT COUNT(*) FROM provider_admission) FROM jobs",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(counts, (0, 0));
    store
        .writer()
        .unwrap()
        .execute_batch("DROP TRIGGER reject_dispatch")
        .unwrap();
    for attempt in 0..9 {
        let r = if attempt == 0 {
            None
        } else {
            Some(held(
                reserve(
                    &store,
                    "embed:local",
                    0,
                    0,
                    Budget {
                        calls: None,
                        tokens: None,
                    },
                    Some(&lease),
                    || 0,
                )
                .unwrap(),
            ))
        };
        let outcome = mark_dispatched(
            &store,
            r.as_ref().unwrap_or(&a),
            Some(&lease),
            || 0,
            unlimited,
        )
        .unwrap();
        assert_eq!(
            outcome,
            if attempt < 8 {
                Dispatch::Recorded
            } else {
                Dispatch::AttemptsExhausted
            }
        );
    }
    let attempts: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT attempts FROM jobs", [], |r| r.get(0))
        .unwrap();
    assert_eq!(attempts, 8);
}

#[test]
fn limits_resolution_refreshes_live_lease_or_refuses_cross_midnight() {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    let a = query_reservation(&store, 1, 86399);
    let mut times = [86399, 86400].into_iter();
    assert_eq!(
        mark_dispatched(&store, &a, None, || times.next().unwrap(), unlimited).unwrap(),
        Dispatch::DayExpired
    );
    use crate::queue::{self, Owner};
    queue::enqueue(&store, &queue::tests::input(1, &[1; 32]), 0).unwrap();
    let lease = queue::claim(
        &store,
        "embed:local",
        &Owner::new(&store).unwrap(),
        0,
        1,
        10,
    )
    .unwrap()
    .remove(0);
    let b = held(reserve(&store, "embed:local", 1, 0, BUDGET, Some(&lease), || 0).unwrap());
    let mut times = [119, 120].into_iter();
    assert_eq!(
        mark_dispatched(
            &store,
            &b,
            Some(&lease),
            || times.next().unwrap(),
            unlimited
        )
        .unwrap(),
        Dispatch::Recorded
    );
    let job: (i64, i64) = store
        .writer()
        .unwrap()
        .query_row("SELECT attempts,leased_until FROM jobs", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(job, (1, 240));
}
