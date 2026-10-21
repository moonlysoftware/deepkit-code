//! Real FLOAT16 publication shares the queue completion transaction; v1 §3.3 and §8.

use super::*;
use crate::store::{
    self, StoreLimits,
    vectors::{VectorInsert, VectorTableSpec},
};

fn fixture(cap: u64) -> Result<Store, Box<dyn std::error::Error>> {
    let store = store::open(
        ":memory:",
        StoreLimits {
            max_l2_units: cap,
            ..StoreLimits::default()
        },
    )?;
    store.ensure_vector_table(VectorTableSpec {
        slug: "queue_test",
        model: "fixture",
        profile: "test",
        dims: 2,
        created_at: 0,
        active: true,
    })?;
    store.writer()?.execute_batch(
        "INSERT INTO unit_contents(hash,text,tokens) VALUES (zeroblob(32),'fixture',1);
         INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at)
         VALUES (1,zeroblob(32),zeroblob(32),'one','chunk','fixture',0,0),
                (2,randomblob(32),zeroblob(32),'two','chunk','fixture',0,0);",
    )?;
    Ok(store)
}

fn lease(store: &Store, target: i64, desired: u8, now: i64) -> Result<Lease, Error> {
    let input = Enqueue {
        kind: Kind::Embed,
        target: Target::Unit,
        target_id: target,
        target_hash: &[0; 32],
        desired_hash: &[desired; 32],
        provider: "fixture",
        priority: 1.0,
        tokens: 1,
    };
    enqueue(store, &input, now)?;
    Ok(claim(store, "fixture", &Owner::new(store)?, now, 1, 1)?.remove(0))
}

fn current(desired: u8) -> Option<CurrentInput> {
    Some(CurrentInput {
        target_hash: [0; 32],
        desired_hash: [desired; 32],
    })
}

fn publish(store: &Store, tx: &Transaction<'_>, unit: i64, context: u8) -> Result<(), Error> {
    store.insert_vector_in(
        tx,
        VectorInsert {
            table_slug: "queue_test",
            unit_id: unit,
            ctx_hash: &[context; 32],
            embedding_hash: &[0; 32],
            embedding: &[0.6, 0.8],
        },
    )?;
    tx.execute("UPDATE units SET ladder_level=2 WHERE id=?1", [unit])?;
    Ok(())
}

type Persisted = (String, i64, Option<Vec<u8>>);

fn persisted(store: &Store, id: i64) -> Result<Persisted, Error> {
    Ok(store.writer()?.query_row(
        "SELECT j.state,u.ladder_level,v.ctx_hash FROM jobs j JOIN units u ON u.id=j.target_id
         LEFT JOIN vec_queue_test v ON v.unit_id=u.id WHERE j.id=?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?)
}

#[test]
fn real_vector_and_ladder_commit_with_job_and_replace_at_full_cap()
-> Result<(), Box<dyn std::error::Error>> {
    let store = fixture(1)?;
    let first = lease(&store, 1, 1, 0)?;
    assert_eq!(
        complete(
            &store,
            &first,
            || 1,
            |_, _| Ok(current(1)),
            |tx| publish(&store, tx, 1, 1)
        )?,
        Completion::Published
    );
    assert_eq!(
        persisted(&store, first.id)?,
        ("done".into(), 2, Some(vec![1; 32]))
    );
    let next = lease(&store, 1, 2, 2)?;
    // A stale owner cannot replace the real stored vector.
    assert_eq!(
        complete(
            &store,
            &first,
            || 3,
            |_, _| Ok(current(1)),
            |tx| publish(&store, tx, 1, 9)
        )?,
        Completion::LeaseLost
    );
    assert_eq!(
        complete(
            &store,
            &next,
            || 3,
            |_, _| Ok(current(2)),
            |tx| publish(&store, tx, 1, 2)
        )?,
        Completion::Published
    );
    assert_eq!(
        persisted(&store, next.id)?,
        ("done".into(), 2, Some(vec![2; 32]))
    );
    let other = lease(&store, 2, 1, 4)?;
    assert!(matches!(
        complete(
            &store,
            &other,
            || 5,
            |_, _| Ok(current(1)),
            |tx| publish(&store, tx, 2, 1)
        ),
        Err(Error::Store(store::Error::VectorCapReached { .. }))
    ));
    assert_eq!(persisted(&store, other.id)?, ("leased".into(), 1, None));
    assert_eq!(store.count_vectors()?, 1);
    Ok(())
}

#[test]
fn lease_expiry_and_sql_failure_rollback_real_vectors_and_ladder()
-> Result<(), Box<dyn std::error::Error>> {
    let store = fixture(2)?;
    let first = lease(&store, 1, 1, 0)?;
    let mut ticks = [1, 120].into_iter();
    assert_eq!(
        complete(
            &store,
            &first,
            || ticks.next().unwrap(),
            |_, _| Ok(current(1)),
            |tx| publish(&store, tx, 1, 1)
        )?,
        Completion::Published
    );
    assert_eq!(
        persisted(&store, first.id)?,
        ("done".into(), 2, Some(vec![1; 32]))
    );
    let second = lease(&store, 2, 1, 2)?;
    store.writer()?.execute_batch("CREATE TRIGGER reject_done BEFORE UPDATE OF state ON jobs WHEN NEW.state='done' BEGIN SELECT RAISE(ABORT,'injected completion failure'); END;")?;
    assert!(
        complete(
            &store,
            &second,
            || 3,
            |_, _| Ok(current(1)),
            |tx| publish(&store, tx, 2, 1)
        )
        .is_err()
    );
    assert_eq!(persisted(&store, second.id)?, ("leased".into(), 1, None));
    store.writer()?.execute_batch("DROP TRIGGER reject_done")?;
    // A superseded context retires the job without publishing a vector or ladder level.
    assert_eq!(
        complete(
            &store,
            &second,
            || 4,
            |_, _| Ok(current(2)),
            |tx| publish(&store, tx, 2, 1)
        )?,
        Completion::Discarded
    );
    assert_eq!(persisted(&store, second.id)?, ("done".into(), 1, None));
    Ok(())
}

#[test]
fn cap_counts_uncommitted_vectors_and_rolls_back_the_whole_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let store = fixture(1)?;
    let first = lease(&store, 1, 1, 0)?;
    let result = complete(
        &store,
        &first,
        || 1,
        |_, _| Ok(current(1)),
        |tx| {
            publish(&store, tx, 1, 1)?;
            publish(&store, tx, 2, 1)
        },
    );
    assert!(matches!(
        result,
        Err(Error::Store(store::Error::VectorCapReached {
            current: 1,
            max_l2_units: 1
        }))
    ));
    assert_eq!(store.count_vectors()?, 0);
    assert_eq!(persisted(&store, first.id)?, ("leased".into(), 1, None));
    Ok(())
}
