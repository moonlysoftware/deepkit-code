//! Local provider dispatch uses the same queue fences as HTTP; v1 §7.2/§8.2.
use super::*;
use crate::{
    models::registry::Registry,
    queue,
    store::vectors::{VectorScanPredicate, VectorTableSpec},
};
use std::{error::Error as StdError, fs};

fn settings(root: &std::path::Path, config: Config) -> Settings {
    Settings {
        root: root.into(),
        config_root: root.into(),
        config,
        kind: UnitKind::Chunk,
        table_slug: "local_fixture_256".into(),
        budget: usage::Budget {
            calls: None,
            tokens: None,
        },
        timeout: Duration::from_secs(30),
    }
}
fn local_config(root: &std::path::Path) -> Result<Config, Box<dyn StdError>> {
    fs::write(
        root.join(".deepkit/config.toml"),
        "[index]\nrepo_name='fixture'\n[embedder]\nprofile='local'\ndims=256\n[embedder.local]\ngpu_layers=0\nthreads=2\n[models]\nunload_after_idle='1s'\n",
    )?;
    Ok(config::load(root, None)?.config)
}

#[tokio::test]
async fn local_factory_rejects_unverified_weights_before_inference() -> Result<(), Box<dyn StdError>>
{
    let (root, store, _) = queue::source::tests::fixture()?;
    let config = local_config(root.path())?;
    let cache = tempfile::tempdir()?;
    let registry = Registry::embedded()?;
    let model = registry.get(&config.embedder.local.model)?;
    fs::write(cache.path().join(&model.file), b"not a model")?;
    let native = LocalProvider::load_registered(
        cache.path().into(),
        config.embedder.local.clone(),
        config.embedder.dims,
        Duration::from_secs(1),
    )
    .await;
    assert!(matches!(
        native,
        Err(crate::models::Error::Local(
            crate::models::llama::Error::Artifact
        ))
    ));

    let result = Factory::new_local(
        Arc::new(store),
        settings(root.path(), config),
        cache.path().into(),
    )
    .await;
    assert!(matches!(
        result,
        Err(Error::Invalid(
            "local embedding artifact or model initialization"
        ))
    ));
    Ok(())
}

#[tokio::test]
#[ignore = "requires DKC_MODELS_DIR with the pinned Qwen embedding artifact; CPU-only inference"]
async fn local_queue_publishes_fenced_vectors_and_refuses_changed_cache_after_idle()
-> Result<(), Box<dyn StdError>> {
    let (root, store, _) = queue::source::tests::fixture()?;
    let config = local_config(root.path())?;
    let registry = Registry::embedded()?;
    let model = registry.get(&config.embedder.local.model)?;
    let cache = tempfile::tempdir()?;
    let source = crate::models::registry::cache_root()?.join(&model.file);
    // A generated private copy permits replacement without touching the user's model.
    fs::copy(source, cache.path().join(&model.file))?;
    let store = Arc::new(store);
    store.ensure_vector_table(VectorTableSpec {
        slug: "local_fixture_256",
        model: &model.name,
        profile: "local",
        dims: 256,
        created_at: 0,
        active: true,
    })?;
    let factory = Factory::new_local(
        Arc::clone(&store),
        settings(root.path(), config.clone()),
        cache.path().into(),
    )
    .await?;
    let id = factory
        .schedule_file(1, 1.0, queue::priority::Recency::ChangedFile, || 0)?
        .ok_or("no job")?;
    let report = Arc::clone(&factory)
        .run_ready(
            RunLimits {
                max_jobs: 1,
                heartbeat: Duration::from_secs(1),
            },
            || 0,
        )
        .await?;
    assert_eq!(report.published, 1);
    assert_eq!(store.count_vectors()?, 1);
    let status = serde_json::to_value(queue::status::snapshot(&store, 0)?)?;
    assert_eq!(status["providers"]["embed:local"]["calls_today"], 1);
    assert!(
        status["providers"]["embed:local"]["estimated_tokens_today"]
            .as_i64()
            .unwrap_or(0)
            > 0
    );

    let vector: Vec<u8> = store.writer()?.query_row(
        "SELECT v FROM vec_local_fixture_256 WHERE unit_id=1",
        [],
        |r| r.get(0),
    )?;
    let query = match &factory.provider {
        Provider::Local(provider) => provider.embed_query("example").await?,
        _ => unreachable!(),
    };
    let hits = store.scan_vectors(
        "local_fixture_256",
        &query,
        1,
        VectorScanPredicate::default(),
    )?;
    assert_eq!(hits[0].unit_id, 1);
    assert_eq!(
        store
            .writer()?
            .query_row("SELECT state FROM jobs WHERE id=?1", [id], |r| r
                .get::<_, String>(0))?,
        "done"
    );
    assert_eq!(
        factory.schedule_file(1, 1.0, queue::priority::Recency::ChangedFile, || 1)?,
        Some(id)
    );
    assert_eq!(
        Arc::clone(&factory)
            .run_ready(
                RunLimits {
                    max_jobs: 1,
                    heartbeat: Duration::from_secs(1)
                },
                || 1
            )
            .await?
            .claimed,
        0
    );
    // Source changes are rejected before local dispatch, preserving the old vector.
    fs::write(root.path().join("a.rs"), "fn different() {}")?;
    assert!(
        factory
            .schedule_file(1, 1.0, queue::priority::Recency::ChangedFile, || 2)?
            .is_none()
    );
    fs::write(root.path().join("a.rs"), "fn example() {}")?;
    // Wait for the configured idle unload; replacement must fail verification on reload.
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if matches!(&factory.provider, Provider::Local(provider) if provider.resident_for_test() == Some(false)) { break; }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }).await?;
    fs::rename(
        cache.path().join(&model.file),
        cache.path().join("original.gguf"),
    )?;
    fs::write(cache.path().join(&model.file), b"changed cached weights")?;
    // Simulate persisted recovery work for the same desired generation.
    store
        .writer()?
        .execute("UPDATE jobs SET state='queued' WHERE id=?1", [id])?;
    let failed = Arc::clone(&factory)
        .run_ready(
            RunLimits {
                max_jobs: 1,
                heartbeat: Duration::from_secs(1),
            },
            || 2,
        )
        .await?;
    assert_eq!(failed.dead, 1);
    assert_eq!(failed.published, 0);
    assert_eq!(
        store.writer()?.query_row(
            "SELECT v FROM vec_local_fixture_256 WHERE unit_id=1",
            [],
            |r| r.get::<_, Vec<u8>>(0)
        )?,
        vector
    );
    Ok(())
}
