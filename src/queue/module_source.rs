//! Current directory and transitive graph authority for modules; v1.md §6.6–6.7/§8.
mod batch;
use super::{Error, source::FileInput};
#[cfg(test)]
use crate::{config, store, sync};
use crate::{config::Config, sync::l1::Work};
pub(crate) use batch::SnapshotCache;
use rusqlite::Transaction;
use std::path::Path;

/// Verifies canonical text, provenance, discovery and every contributor before
/// allowing module output or outbound work. No authority survives this call.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit source authority inputs"
)]
pub(super) fn resolve(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    unit: i64,
    minimum: i64,
    permit_protected: bool,
    work: Work<'_>,
) -> Result<Option<FileInput>, Error> {
    Ok(batch::resolve(
        tx,
        root,
        config_root,
        current,
        &[unit],
        minimum,
        permit_protected,
        work,
    )?
    .remove(&unit))
}

/// Verifies a set of module labels in one read observation, including protected
/// sources. This is read-only freshness evidence, never outbound authorization.
/// A final response check must call again; nothing is cached between observations.
pub(crate) fn resolve_freshness_batch(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    units: &[i64],
    work: Work<'_>,
) -> Result<std::collections::BTreeMap<i64, FileInput>, Error> {
    batch::resolve(tx, root, config_root, current, units, 1, true, work)
}

/// Map-only variant retaining database snapshots across revision-fenced calls.
/// The caller must discard this cache after any local or external DB write.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit source authority and request cache inputs"
)]
pub(crate) fn resolve_freshness_batch_with_cache(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    units: &[i64],
    work: Work<'_>,
    cache: &mut Option<SnapshotCache>,
) -> Result<std::collections::BTreeMap<i64, FileInput>, Error> {
    batch::resolve_with_cache(
        tx,
        root,
        config_root,
        current,
        units,
        1,
        true,
        work,
        Some(cache),
    )
}

/// Batch counterpart of module enrichment preparation. It has the same L3 and
/// privacy admission as an individual enrichment parent observation.
pub(crate) fn resolve_enrichment_batch(
    tx: &Transaction<'_>,
    root: &Path,
    config_root: &Path,
    current: &Config,
    units: &[i64],
    work: Work<'_>,
) -> Result<std::collections::BTreeMap<i64, FileInput>, Error> {
    batch::resolve(tx, root, config_root, current, units, 3, false, work)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::UnitKind;
    use std::fs;

    fn fixture(protected: bool) -> (tempfile::TempDir, store::Store, Config, i64) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join(".deepkit")).unwrap();
        fs::create_dir(root.path().join("alpha")).unwrap();
        fs::write(root.path().join(".deepkit/config.toml"),format!("[index]\nrepo_name='modules'\n[commits]\nenabled=false\n[llm]\nenabled=false\n[[treatments]]\nname='private'\nmatch=['private.rs']\nmax_level=1\nnever_send={protected}\nchunker='ast'\n[[treatments]]\nname='public'\nmatch=['**/*.rs']\nmax_level=2\nchunker='ast'\n")).unwrap();
        for n in 0..5 {
            fs::write(
                root.path().join(format!("alpha/leaf{n}.rs")),
                format!("pub fn leaf{n}() {{}}\n"),
            )
            .unwrap();
        }
        fs::write(
            root.path().join("private.rs"),
            "pub fn invoke() { leaf0(); }\n",
        )
        .unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        let store = store::open(root.path().join(".deepkit/code.db"), Default::default()).unwrap();
        sync::index::run(
            sync::index::Input {
                config_root: root.path(),
                root: root.path(),
                settings: &config,
                store: &store,
                full: false,
                work: Work::default(),
            },
            |_| {},
        )
        .unwrap();
        let unit = store
            .read_snapshot(|tx| {
                tx.query_row("SELECT module_unit FROM dirs WHERE path='alpha'", [], |r| {
                    r.get(0)
                })
            })
            .unwrap()
            .unwrap();
        (root, store, config, unit)
    }

    fn read(
        store: &store::Store,
        root: &Path,
        config: &Config,
        unit: i64,
        minimum: i64,
        protected: bool,
    ) -> Option<FileInput> {
        store
            .read_snapshot(|tx| {
                resolve(
                    tx,
                    root,
                    root,
                    config,
                    unit,
                    minimum,
                    protected,
                    Work::default(),
                )
            })
            .unwrap()
            .unwrap()
    }

    #[test]
    fn freshness_batch_shares_sources_but_rechecks_ignores_bytes_and_configuration() {
        let (root, store, config, unit) = fixture(true);
        let parent: i64 = store
            .read_snapshot(|tx| {
                tx.query_row("SELECT module_unit FROM dirs WHERE path=''", [], |r| {
                    r.get(0)
                })
            })
            .unwrap()
            .unwrap();
        let read_batch = || {
            store
                .read_snapshot(|tx| {
                    resolve_freshness_batch(
                        tx,
                        root.path(),
                        root.path(),
                        &config,
                        &[unit, parent, unit, -1],
                        Work::default(),
                    )
                })
                .unwrap()
                .unwrap()
        };
        let batch = read_batch();
        assert_eq!(batch.len(), 2);
        for id in [unit, parent] {
            let one = read(&store, root.path(), &config, id, 1, true).unwrap();
            let input = &batch[&id];
            assert_eq!(input.occurrence.content_hash, one.occurrence.content_hash);
            assert_eq!(input.context, one.context);
            assert_eq!(input.policy.fingerprint, one.policy.fingerprint);
            assert!(input.policy.never_send);
        }
        fs::write(root.path().join(".deepkitignore"), "private.rs\n").unwrap();
        assert!(read_batch().is_empty());
        fs::remove_file(root.path().join(".deepkitignore")).unwrap();
        let saved = fs::read(root.path().join("private.rs")).unwrap();
        fs::write(root.path().join("private.rs"), "pub fn changed() {}\n").unwrap();
        assert!(read_batch().is_empty());
        fs::write(root.path().join("private.rs"), saved).unwrap();
        assert_eq!(read_batch().len(), 2);
        let config_path = root.path().join(".deepkit/config.toml");
        let saved = fs::read_to_string(&config_path).unwrap();
        fs::write(
            &config_path,
            saved.replace("repo_name='modules'", "repo_name='changed'"),
        )
        .unwrap();
        assert!(read_batch().is_empty());
    }

    #[test]
    fn cached_module_snapshots_still_recheck_live_source_bytes() {
        let (root, store, config, unit) = fixture(false);
        let mut cache = None;
        let mut read = || {
            store
                .read_snapshot(|tx| {
                    resolve_freshness_batch_with_cache(
                        tx,
                        root.path(),
                        root.path(),
                        &config,
                        &[unit],
                        Work::default(),
                        &mut cache,
                    )
                })
                .unwrap()
                .unwrap()
        };
        assert!(read().contains_key(&unit));
        let path = root.path().join("alpha/leaf0.rs");
        let original = fs::read(&path).unwrap();
        fs::write(&path, "pub fn changed() {}\n").unwrap();
        assert!(!read().contains_key(&unit));
        fs::write(&path, original).unwrap();
        assert!(read().contains_key(&unit));
    }

    #[test]
    fn module_ladder_max_does_not_override_protected_external_graph_sources() {
        for protected in [false, true] {
            let (root, store, config, unit) = fixture(protected);
            let input = read(&store, root.path(), &config, unit, 1, true).unwrap();
            assert_eq!(input.policy.max_level, 2);
            assert_eq!(input.policy.never_send, protected);
            assert!(
                input.context.text.contains("used by:\n: 1"),
                "{}",
                input.context.text
            );
            let count:i64=store.read_snapshot(|tx|tx.query_row("SELECT count(*) FROM unit_sources us JOIN files f ON us.source_kind='file' AND us.source_id=f.id WHERE us.unit_id=?1 AND f.path='private.rs'",[unit],|r|r.get(0))).unwrap().unwrap();
            assert_eq!(count, 1);
            assert_eq!(
                read(&store, root.path(), &config, unit, 2, false).is_none(),
                protected
            );
            assert_eq!(
                read(&store, root.path(), &config, unit, 1, false).is_none(),
                protected
            );
        }
    }

    #[test]
    fn module_refuses_changed_extra_ignored_and_external_sources_then_recovers() {
        let (root, store, config, unit) = fixture(false);
        let root = root.path();
        assert!(read(&store, root, &config, unit, 2, false).is_some());
        for (path, text) in [
            ("alpha/leaf0.rs", "pub fn changed() {}\n"),
            ("private.rs", "pub fn invoke() {}\n"),
        ] {
            let saved = fs::read(root.join(path)).unwrap();
            fs::write(root.join(path), text).unwrap();
            assert!(read(&store, root, &config, unit, 1, true).is_none());
            fs::write(root.join(path), saved).unwrap();
            assert!(read(&store, root, &config, unit, 2, false).is_some());
        }
        fs::write(root.join("alpha/extra.rs"), "pub fn extra() {}\n").unwrap();
        assert!(read(&store, root, &config, unit, 1, true).is_none());
        fs::remove_file(root.join("alpha/extra.rs")).unwrap();
        for ignored in ["alpha/leaf1.rs\n", "private.rs\n"] {
            fs::write(root.join(".deepkitignore"), ignored).unwrap();
            assert!(read(&store, root, &config, unit, 1, true).is_none());
            fs::remove_file(root.join(".deepkitignore")).unwrap();
            assert!(read(&store, root, &config, unit, 2, false).is_some());
        }
    }

    #[tokio::test]
    async fn external_policy_change_after_enqueue_refuses_prepared_module_dispatch() {
        use crate::{
            queue::{self, destination, http_embedding, pool::Factory as _, usage, worker},
            store::vectors::VectorTableSpec,
        };
        use std::{sync::Arc, time::Duration};
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings":[[1.0,0.0]]})),
            )
            .mount(&server)
            .await;
        let (root, store, _, unit) = fixture(false);
        let file = root.path().join(".deepkit/config.toml");
        let text = fs::read_to_string(&file).unwrap()
            + &format!(
                "\n[embedder]\nprofile='ollama'\ndims=2\n[embedder.ollama]\nurl='{}'\n",
                server.uri()
            );
        fs::write(&file, &text).unwrap();
        let config = config::load(root.path(), None).unwrap().config;
        let store = Arc::new(store);
        store
            .ensure_vector_table(VectorTableSpec {
                slug: "module_policy",
                model: &config.embedder.ollama.model,
                profile: "ollama",
                dims: 2,
                created_at: 0,
                active: true,
            })
            .unwrap();
        let prepared = store
            .read_snapshot(|tx| {
                destination::prepare_file(tx, root.path(), root.path(), &config, unit)
            })
            .unwrap()
            .unwrap()
            .unwrap()
            .0;
        let id = queue::enqueue(&store, &prepared.enqueue(1.0), 0).unwrap();
        let lease = queue::claim(
            &store,
            prepared.enqueue(1.0).provider,
            &queue::Owner::new(&store).unwrap(),
            0,
            1,
            10000,
        )
        .unwrap()
        .remove(0);
        assert_eq!(id, lease.id);
        let factory = http_embedding::Factory::new(
            Arc::clone(&store),
            http_embedding::Settings {
                root: root.path().into(),
                config_root: root.path().into(),
                config,
                kind: UnitKind::Module,
                table_slug: "module_policy".into(),
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                timeout: Duration::from_secs(2),
            },
        )
        .unwrap();
        let job = factory.prepare(&store, &lease).unwrap();
        fs::write(&file, text.replace("never_send=false", "never_send=true")).unwrap();
        let result = worker::execute(
            &store,
            &lease,
            &worker::Settings {
                provider: prepared.enqueue(1.0).provider,
                budget: usage::Budget {
                    calls: None,
                    tokens: None,
                },
                max_output_tokens: 0,
                timeout: Duration::from_secs(2),
                heartbeat: Duration::from_secs(30),
            },
            &job,
            || 1,
        )
        .await;
        assert_eq!(
            result.unwrap(),
            worker::Finished::Completed(queue::Completion::Discarded)
        );
        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(store.count_vectors().unwrap(), 0);
        let publications: i64 = store
            .writer()
            .unwrap()
            .query_row("SELECT count(*) FROM embedding_publications", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(publications, 0);
    }
}
