//! Generation and competing-lease regressions for v1 §8; dispatch tests follow separately.
use super::*;
use crate::store::{self, StoreLimits};
use proptest::prelude::*;

pub(super) fn input(id: i64, desired: &[u8; 32]) -> Enqueue<'_> {
    Enqueue {
        kind: Kind::Embed,
        target: Target::Unit,
        target_id: id,
        target_hash: &[7; 32],
        desired_hash: desired,
        provider: "embed:local",
        priority: 1.0,
        tokens: 10,
    }
}

#[test]
fn enqueue_preserves_terminal_work_and_distinguishes_desired_inputs()
-> Result<(), Box<dyn std::error::Error>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    let a = input(1, &[1; 32]);
    let id = enqueue(&store, &a, 0)?;
    assert_eq!(id, enqueue(&store, &a, 1)?);
    for state in ["done", "dead"] {
        store.writer()?.execute(
            "UPDATE jobs SET state=?2,attempts=8 WHERE id=?1",
            params![id, state],
        )?;
        assert_eq!(id, enqueue(&store, &a, 2)?);
        let persisted: String =
            store
                .writer()?
                .query_row("SELECT state FROM jobs WHERE id=?1", [id], |r| r.get(0))?;
        assert_eq!(persisted, state);
    }
    let next = enqueue(&store, &input(1, &[2; 32]), 3)?;
    assert_ne!(id, next); // Same source, new rendered context/model/policy fingerprint.
    let mut conflict = input(1, &[2; 32]);
    conflict.provider = "embed:remote";
    assert!(matches!(
        enqueue(&store, &conflict, 4),
        Err(Error::IdentityConflict)
    ));
    let owner = Owner::new(&store)?;
    let leases = claim(&store, "embed:local", &owner, 4, 10, 100)?;
    assert_eq!(leases.len(), 1);
    assert_eq!(leases[0].id, next);
    Ok(())
}

#[test]
fn claims_fit_tokens_preserve_priority_and_do_not_count_dispatches()
-> Result<(), Box<dyn std::error::Error>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    for (id, priority, tokens) in [(1, 10.0, 100), (2, 3.0, 7), (3, 2.0, 5), (4, 1.0, 3)] {
        let mut job = input(id, &[1; 32]);
        job.priority = priority;
        job.tokens = tokens;
        enqueue(&store, &job, 0)?;
    }
    let mut remote = input(5, &[1; 32]);
    remote.provider = "embed:remote";
    enqueue(&store, &remote, 0)?;
    let delayed = enqueue(&store, &input(6, &[1; 32]), 0)?;
    store.writer()?.execute(
        "UPDATE jobs SET state='failed',not_before=20 WHERE id=?1",
        [delayed],
    )?;
    let owner = Owner::new(&store)?;
    let batch = claim(&store, "embed:local", &owner, 0, 8, 10)?;
    assert_eq!(batch.iter().map(|l| l.id).collect::<Vec<_>>(), [2, 4]);
    assert_eq!(batch.iter().map(|l| l.tokens).sum::<i64>(), 10);
    let attempts: i64 = store
        .writer()?
        .query_row("SELECT SUM(attempts) FROM jobs", [], |r| r.get(0))?;
    assert_eq!(attempts, 0);
    assert_eq!(claim(&store, "embed:local", &owner, 0, 1, 10)?[0].id, 3);
    assert!(claim(&store, "embed:local", &owner, 0, 8, 10)?.is_empty());
    assert_eq!(
        claim(&store, "embed:local", &owner, 20, 1, 10)?[0].id,
        delayed
    );
    Ok(())
}

#[test]
fn independent_connections_claim_once_and_fence_expired_renewal()
-> Result<(), Box<dyn std::error::Error>> {
    let file = tempfile::NamedTempFile::new()?;
    let first = store::open(file.path(), StoreLimits::default())?;
    let second = store::open(file.path(), StoreLimits::default())?;
    enqueue(&first, &input(1, &[1; 32]), 0)?;
    let a = Owner::new(&first)?;
    let b = Owner::new(&second)?;
    assert_ne!(a.0, b.0);
    assert!(a.0.starts_with(&format!("{}:", std::process::id())));
    let barrier = std::sync::Barrier::new(2);
    let (left, right) = std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            barrier.wait();
            claim(&first, "embed:local", &a, 0, 1, 10)
        });
        let right = scope.spawn(|| {
            barrier.wait();
            claim(&second, "embed:local", &b, 0, 1, 10)
        });
        (left.join().unwrap(), right.join().unwrap())
    });
    let mut leases = left?;
    leases.extend(right?);
    assert_eq!(leases.len(), 1);
    let old = &leases[0];
    assert!(renew(&second, old, 119)?);
    assert!(claim(&first, "embed:local", &b, 120, 1, 10)?.is_empty());
    assert!(!renew(&first, old, 239)?); // Expired, even without replacement.
    let replacement = claim(&second, "embed:local", &b, 239, 1, 10)?;
    assert_eq!(replacement.len(), 1);
    assert_eq!(replacement[0].generation, old.generation + 1);
    assert!(!renew(&first, old, 240)?);
    assert!(renew(&first, &replacement[0], 240)?);
    Ok(())
}

#[test]
fn enrichment_kinds_include_l0_image_and_page_targets() -> Result<(), Box<dyn std::error::Error>> {
    let store = store::open(":memory:", StoreLimits::default())?;
    for kind in [
        Kind::DescribeSymbol,
        Kind::SummarizePr,
        Kind::SummarizeModule,
        Kind::DescribeImage,
        Kind::VlmPage,
    ] {
        let mut job = input(1, &[1; 32]);
        job.kind = kind;
        job.target = match kind {
            Kind::DescribeSymbol => Target::Symbol,
            Kind::SummarizePr => Target::Pr,
            _ => Target::File,
        };
        job.provider = if matches!(kind, Kind::VlmPage) {
            "vlm:remote"
        } else {
            "llm:remote"
        };
        enqueue(&store, &job, 0)?;
    }
    assert_eq!(
        claim(&store, "llm:remote", &Owner::new(&store)?, 0, 5, 50)?.len(),
        4
    );
    assert_eq!(
        claim(&store, "vlm:remote", &Owner::new(&store)?, 0, 5, 50)?.len(),
        1
    );
    // Admission and dependency checks belong to scheduling/dispatch, not target FKs.
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn generated_two_worker_lease_interleavings(ops in prop::collection::vec((0u8..3,0i64..150),1..60)) {
        let store=store::open(":memory:",StoreLimits::default()).unwrap();
        enqueue(&store,&input(1,&[1;32]),0).unwrap();
        let owners=[Owner::new(&store).unwrap(),Owner::new(&store).unwrap()];
        let mut history: Vec<Lease>=Vec::new();
        let mut current: Option<(i64,i64)>=None; // Independent generation/expiry model.
        let mut generation=0;
        let mut now=0;
        for (operation,advance) in ops {
            now+=advance;
            if operation<2 {
                let batch=claim(&store,"embed:local",&owners[operation as usize],now,1,10).unwrap();
                let available=current.is_none_or(|(_,until)|until<=now);
                prop_assert_eq!(batch.len(),usize::from(available));
                if available {generation+=1;prop_assert_eq!(batch[0].generation,generation);current=Some((generation,now+120));history.extend(batch);}
            } else if let Some(lease)=history.first() {
                let expected=current.is_some_and(|(g,until)|g==lease.generation && until>now);
                prop_assert_eq!(renew(&store,lease,now).unwrap(),expected);
                if expected {current=Some((lease.generation,now+120));}
            }
        }
    }
}

fn publication_store() -> Store {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    store
        .writer()
        .unwrap()
        .execute_batch(
            "CREATE TABLE current_inputs(target_hash BLOB,desired_hash BLOB);
         CREATE TABLE published(value INTEGER);",
        )
        .unwrap();
    store
        .writer()
        .unwrap()
        .execute(
            "INSERT INTO current_inputs VALUES(?1,?2)",
            params![[7u8; 32].as_slice(), [1u8; 32].as_slice()],
        )
        .unwrap();
    enqueue(&store, &input(1, &[1; 32]), 0).unwrap();
    store
}

fn resolve_current(tx: &Transaction<'_>, _: &Lease) -> Result<Option<CurrentInput>, Error> {
    Ok(tx
        .query_row(
            "SELECT target_hash,desired_hash FROM current_inputs",
            [],
            |r| {
                Ok(CurrentInput {
                    target_hash: r.get(0)?,
                    desired_hash: r.get(1)?,
                })
            },
        )
        .optional()?)
}

fn publish_one(tx: &Transaction<'_>) -> Result<(), Error> {
    tx.execute("INSERT INTO published VALUES(1)", [])?;
    Ok(())
}

fn publication_state(store: &Store) -> (i64, String) {
    store
        .writer()
        .unwrap()
        .query_row(
            "SELECT (SELECT COUNT(*) FROM published),state FROM jobs WHERE id=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
}

#[test]
fn completion_publishes_once_and_retires_only_owned_obsolete_work() {
    for mutation in [
        None,
        Some("UPDATE current_inputs SET target_hash=zeroblob(32)"),
        Some("UPDATE current_inputs SET desired_hash=zeroblob(32)"),
        Some("DELETE FROM current_inputs"),
    ] {
        let store = publication_store();
        let lease = claim(
            &store,
            "embed:local",
            &Owner::new(&store).unwrap(),
            0,
            1,
            10,
        )
        .unwrap()
        .remove(0);
        if let Some(sql) = mutation {
            store.writer().unwrap().execute(sql, []).unwrap();
        }
        let expected = if mutation.is_none() {
            Completion::Published
        } else {
            Completion::Discarded
        };
        assert_eq!(
            complete(&store, &lease, || 1, resolve_current, publish_one).unwrap(),
            expected
        );
        assert_eq!(
            publication_state(&store),
            (i64::from(mutation.is_none()), "done".into())
        );
        assert_eq!(
            complete(
                &store,
                &lease,
                || 2,
                |_, _| panic!("retired resolver"),
                |_| panic!("duplicate publication")
            )
            .unwrap(),
            Completion::LeaseLost
        );
    }
}

#[test]
fn completion_keeps_writer_owned_publication_that_outlives_its_lease() {
    let store = publication_store();
    let lease = claim(
        &store,
        "embed:local",
        &Owner::new(&store).unwrap(),
        0,
        1,
        10,
    )
    .unwrap()
    .remove(0);
    let mut calls = 0;
    assert_eq!(
        complete(
            &store,
            &lease,
            || {
                calls += 1;
                if calls == 1 { 1 } else { 120 }
            },
            resolve_current,
            publish_one,
        )
        .unwrap(),
        Completion::Published
    );
    assert_eq!(publication_state(&store), (1, "done".into()));
}

#[test]
fn completion_rolls_back_output_on_error_clock_regression_and_failed_done_transition() {
    for failure in [0, 2, 3] {
        let store = publication_store();
        let lease = claim(
            &store,
            "embed:local",
            &Owner::new(&store).unwrap(),
            0,
            1,
            10,
        )
        .unwrap()
        .remove(0);
        if failure == 2 {
            store.writer().unwrap().execute_batch("CREATE TRIGGER reject_done BEFORE UPDATE OF state ON jobs WHEN NEW.state='done' BEGIN SELECT RAISE(ABORT,'injected done failure'); END;").unwrap();
        }
        let mut calls = 0;
        let result = complete(
            &store,
            &lease,
            || {
                calls += 1;
                if calls == 1 {
                    1
                } else if failure == 3 {
                    0
                } else {
                    2
                }
            },
            resolve_current,
            |tx| {
                publish_one(tx)?;
                if failure == 0 {
                    Err(Error::Invalid("injected publication failure"))
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(publication_state(&store), (0, "leased".into()));
        if failure == 2 {
            store
                .writer()
                .unwrap()
                .execute_batch("DROP TRIGGER reject_done")
                .unwrap();
        }
        let replacement = claim(
            &store,
            "embed:local",
            &Owner::new(&store).unwrap(),
            120,
            1,
            10,
        )
        .unwrap()
        .remove(0);
        assert_eq!(
            complete(&store, &replacement, || 121, resolve_current, publish_one).unwrap(),
            Completion::Published
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn generated_completion_expiry_and_reclaim_fences(advances in prop::collection::vec(0i64..180,1..30), obsolete in any::<bool>()) {
        let store=publication_store();
        let owner=Owner::new(&store).unwrap();
        let mut leases=claim(&store,"embed:local",&owner,0,1,10).unwrap();
        let mut now=0;
        let mut expiry=120;
        for advance in advances {
            now+=advance;
            let next=claim(&store,"embed:local",&owner,now,1,10).unwrap();
            prop_assert_eq!(next.len(),usize::from(now>=expiry));
            if !next.is_empty() {expiry=now+120; leases.extend(next);}
        }
        if obsolete {store.writer().unwrap().execute("DELETE FROM current_inputs",[]).unwrap();}
        for lease in &leases[..leases.len()-1] {
            prop_assert_eq!(complete(&store,lease,||now,|_,_|panic!("stale resolution"), |_|panic!("stale write")).unwrap(),Completion::LeaseLost);
        }
        prop_assert_eq!(publication_state(&store),(0,"leased".into()));
        let current=leases.last().unwrap();
        prop_assert_eq!(complete(&store,current,||now,resolve_current,publish_one).unwrap(),if obsolete {Completion::Discarded} else {Completion::Published});
        prop_assert_eq!(publication_state(&store),(i64::from(!obsolete),"done".into()));
    }
}

#[test]
fn expired_completion_cannot_discard_before_reclaim_or_retire_new_desired_job() {
    let store = publication_store();
    let owner = Owner::new(&store).unwrap();
    let lease = claim(&store, "embed:local", &owner, 0, 1, 10)
        .unwrap()
        .remove(0);
    assert_eq!(
        complete(
            &store,
            &lease,
            || 120,
            |_, _| panic!("expired resolver"),
            |_| panic!("expired publication")
        )
        .unwrap(),
        Completion::LeaseLost
    );
    let next = enqueue(&store, &input(1, &[2; 32]), 120).unwrap();
    store
        .writer()
        .unwrap()
        .execute(
            "UPDATE current_inputs SET desired_hash=?1",
            [[2u8; 32].as_slice()],
        )
        .unwrap();
    let old = claim(&store, "embed:local", &owner, 120, 1, 10)
        .unwrap()
        .remove(0);
    assert_eq!(
        complete(
            &store,
            &old,
            || 121,
            resolve_current,
            |_| panic!("obsolete publication")
        )
        .unwrap(),
        Completion::Discarded
    );
    let new = claim(&store, "embed:local", &owner, 121, 1, 10)
        .unwrap()
        .remove(0);
    assert_eq!(new.id, next);
    assert_eq!(
        complete(&store, &new, || 122, resolve_current, publish_one).unwrap(),
        Completion::Published
    );
    assert_eq!(publication_state(&store), (1, "done".into()));
}
