//! Occurrence-specific vector freshness in a single snapshot; v1 §§3.3, 4.6, 7.1.
use super::{Error, embedding_bindings, vectors};
use rusqlite::{OptionalExtension, Transaction, params};

/// Physical representation of the selected space, independent of L1 eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    Missing,
    Current,
    Stale,
}

/// Checks only the configured destination and occurrence, without model loading,
/// native vector registration, queue writes or scanning retained fallback tables.
/// Callers supply freshly resolved content/context and desired publication identity,
/// and must separately verify
/// source, policy, configuration and query budget. Missing is not proof that no
/// old-tier vector exists; stale data remains intact for transition retrieval.
pub(crate) fn read(
    tx: &Transaction<'_>,
    binding: &embedding_bindings::Binding,
    unit: i64,
    content: &[u8; 32],
    context: &[u8; 32],
    desired: &[u8; 32],
) -> Result<State, Error> {
    let slug = embedding_bindings::select(tx, binding)?;
    let metadata: Option<(String, String, i64, bool)> = tx
        .query_row(
            "SELECT profile,model,dims,active FROM embedding_tables WHERE slug=?1",
            [&slug],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()?;
    let Some((profile, model, dims, active)) = metadata else {
        return Ok(State::Missing);
    };
    if !active
        || profile != binding.profile
        || model != binding.model
        || dims != binding.dims as i64
        || embedding_bindings::read(tx, &slug)?.is_none_or(|actual| !actual.same_space(binding))
    {
        return Ok(State::Stale);
    }
    let table = vectors::vector_table_name(&slug)?;
    let matches: Option<bool> = tx.query_row(
        &format!("SELECT u.content_hash=?2 AND v.ctx_hash=?3 AND COALESCE(p.desired_hash=?5,0) FROM {table} v JOIN units u ON u.id=v.unit_id LEFT JOIN embedding_publications p ON p.unit_id=v.unit_id AND p.table_slug=?4 WHERE v.unit_id=?1"),
        params![unit, content, context, slug, desired], |r| r.get(0),
    ).optional()?;
    Ok(match matches {
        None => State::Missing,
        Some(true) => State::Current,
        Some(false) => State::Stale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, Store, StoreLimits, embedding_bindings::Binding};

    fn binding() -> Binding {
        Binding {
            profile: "ollama".into(),
            model: "model".into(),
            dims: 2,
            endpoint: "http://localhost:11434/api/embed".into(),
            artifact_hash: None,
            query_prefix: String::new(),
            api_key_env: None,
        }
    }
    fn prepare(store: &Store, binding: &Binding) -> String {
        let slug = binding.destination_slug();
        store
            .ensure_vector_table(vectors::VectorTableSpec {
                slug: &slug,
                model: &binding.model,
                profile: &binding.profile,
                dims: binding.dims as usize,
                created_at: 0,
                active: true,
            })
            .unwrap();
        store.bind_embedding_table(&slug, binding).unwrap();
        slug
    }
    fn state(
        store: &Store,
        binding: &Binding,
        unit: i64,
        content: &[u8; 32],
        context: &[u8; 32],
    ) -> State {
        store
            .read_snapshot(|tx| read(tx, binding, unit, content, context, &[4; 32]))
            .unwrap()
            .unwrap()
    }
    fn fixture() -> (Store, Binding, String) {
        let store = store::open(":memory:", StoreLimits::default()).unwrap();
        let binding = binding();
        let slug = prepare(&store, &binding);
        store
            .writer()
            .unwrap()
            .execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'same text',2)",
                [[1u8; 32].as_slice()],
            )
            .unwrap();
        for id in [1i64, 2] {
            store.writer().unwrap().execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(?1,?2,?3,?4,'chunk','',0,0)", params![id,[id as u8;32].as_slice(),[1u8;32].as_slice(),format!("source:{id}")]).unwrap();
        }
        store
            .insert_vector(vectors::VectorInsert {
                table_slug: &slug,
                unit_id: 1,
                ctx_hash: &[2; 32],
                embedding_hash: &[1; 32],
                embedding: &[1.0, 0.0],
            })
            .unwrap();
        store.writer().unwrap().execute("INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(1,?1,?2)", params![slug,[4u8;32].as_slice()]).unwrap();
        (store, binding, slug)
    }

    #[test]
    fn occurrence_freshness_requires_current_content_and_context_without_writes() {
        let (store, binding, slug) = fixture();
        let before = store.writer().unwrap().total_changes();
        assert_eq!(
            state(&store, &binding, 1, &[1; 32], &[2; 32]),
            State::Current
        );
        assert_eq!(
            state(&store, &binding, 2, &[1; 32], &[2; 32]),
            State::Missing
        );
        assert_eq!(state(&store, &binding, 1, &[1; 32], &[3; 32]), State::Stale);
        assert_eq!(state(&store, &binding, 1, &[3; 32], &[2; 32]), State::Stale);
        assert_eq!(store.writer().unwrap().total_changes(), before);
        assert_eq!(
            store
                .read_snapshot(|tx| read(tx, &binding, 1, &[1; 32], &[2; 32], &[5; 32]))
                .unwrap()
                .unwrap(),
            State::Stale
        );
        store
            .writer()
            .unwrap()
            .execute("DELETE FROM embedding_publications WHERE unit_id=1", [])
            .unwrap();
        assert_eq!(state(&store, &binding, 1, &[1; 32], &[2; 32]), State::Stale);
        store.writer().unwrap().execute("INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(1,?1,?2)", params![slug,[4u8;32].as_slice()]).unwrap();
        // Matching retained bytes cannot prove freshness after occurrence replacement.
        store
            .writer()
            .unwrap()
            .execute(
                "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'replacement',1)",
                [[3u8; 32].as_slice()],
            )
            .unwrap();
        store
            .writer()
            .unwrap()
            .execute(
                "UPDATE units SET content_hash=?1 WHERE id=1",
                [[3u8; 32].as_slice()],
            )
            .unwrap();
        assert_eq!(state(&store, &binding, 1, &[1; 32], &[2; 32]), State::Stale);
        store.delete_vector(&slug, 1).unwrap();
        assert_eq!(
            state(&store, &binding, 1, &[3; 32], &[2; 32]),
            State::Missing
        );
    }

    #[test]
    fn selected_space_requires_known_active_identity_and_never_reads_other_vectors() {
        let (store, binding, slug) = fixture();
        let mut other = binding.clone();
        other.endpoint = "http://localhost:11435/api/embed".into();
        assert_eq!(state(&store, &other, 1, &[1; 32], &[2; 32]), State::Missing);
        let other_slug = prepare(&store, &other);
        // A deliberately absent unrelated physical table is a tripwire for inventory scans.
        store
            .writer()
            .unwrap()
            .execute(
                &format!(
                    "DROP TABLE {}",
                    vectors::vector_table_name(&other_slug).unwrap()
                ),
                [],
            )
            .unwrap();
        assert_eq!(
            state(&store, &binding, 1, &[1; 32], &[2; 32]),
            State::Current
        );
        let mut query = binding.clone();
        query.query_prefix = "query changed".into();
        assert_eq!(state(&store, &query, 1, &[1; 32], &[2; 32]), State::Current);
        for column in ["active=0", "model='other'", "dims=3", "profile='remote'"] {
            store
                .writer()
                .unwrap()
                .execute(
                    &format!("UPDATE embedding_tables SET {column} WHERE slug=?1"),
                    [&slug],
                )
                .unwrap();
            assert_eq!(state(&store, &binding, 1, &[1; 32], &[2; 32]), State::Stale);
            store.writer().unwrap().execute("UPDATE embedding_tables SET active=1,model='model',dims=2,profile='ollama' WHERE slug=?1", [&slug]).unwrap();
        }
        store.writer().unwrap().execute("UPDATE embedding_bindings SET endpoint='http://localhost:11436/api/embed' WHERE slug=?1", [&slug]).unwrap();
        assert_eq!(state(&store, &binding, 1, &[1; 32], &[2; 32]), State::Stale);
        store
            .writer()
            .unwrap()
            .execute("DELETE FROM embedding_bindings WHERE slug=?1", [&slug])
            .unwrap();
        assert_eq!(state(&store, &binding, 1, &[1; 32], &[2; 32]), State::Stale);
    }
}
