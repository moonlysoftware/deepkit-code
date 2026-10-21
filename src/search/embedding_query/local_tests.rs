//! Local query-to-vector retrieval and explicitly estimated usage; v1 §§7.2, 8.2, 9.
use super::*;
use crate::{
    search::dense,
    store::{
        self, StoreLimits,
        vectors::{VectorInsert, VectorTableSpec},
    },
};
use std::fs;

fn setup() -> (tempfile::TempDir, Arc<Store>, Settings) {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".deepkit")).unwrap();
    fs::write(root.path().join(".deepkit/config.toml"),"[index]\nrepo_name='local-query'\n[embedder]\nprofile='local'\ndims=32\n[embedder.local]\ngpu_layers=0\nthreads=2\nn_ctx=512\nbatch_tokens=512\n[limits]\nquery_timeout_ms=5000\n").unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let settings = Settings {
        config_root: root.path().into(),
        config,
        kind: UnitKind::Chunk,
        budget: usage::Budget {
            calls: Some(4),
            tokens: Some(1000),
        },
    };
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    (root, store, settings)
}

#[tokio::test]
async fn local_query_factory_refuses_unverified_cache_before_inference() {
    let (_root, store, settings) = setup();
    let cache = tempfile::tempdir().unwrap();
    fs::write(
        cache
            .path()
            .join(crate::models::llama::EMBEDDING_MODEL_FILE),
        b"not a model",
    )
    .unwrap();
    assert!(matches!(
        Factory::new_local(Arc::clone(&store), settings, cache.path().into()).await,
        Err(queue::Error::Invalid(
            "local query artifact or model initialization"
        ))
    ));
    let count: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT count(*) FROM usage_reservations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
#[ignore = "requires DKC_MODELS_DIR with the verified local embedding artifact; CPU only"]
async fn local_query_retrieves_vectors_and_records_estimates_without_outbound_identity() {
    let (root, store, settings) = setup();
    let cache = std::env::var_os("DKC_MODELS_DIR")
        .expect("DKC_MODELS_DIR must point to the verified model cache");
    let model = settings.config.embedder.local.model.clone();
    let factory = Factory::new_local(Arc::clone(&store), settings, PathBuf::from(cache))
        .await
        .unwrap();
    let query = "where are failed payments retried?";
    let vector = factory
        .embed(
            query.into(),
            Instant::now() + Duration::from_secs(5),
            || 100,
        )
        .await
        .unwrap();
    assert_eq!(vector.len(), 32);
    let Provider::Local(provider) = &factory.provider else {
        panic!("expected local provider");
    };
    let baseline = crate::models::Embedder::embed_query(provider, query)
        .await
        .unwrap();
    assert_eq!(baseline.len(), vector.len());
    assert!(
        baseline
            .iter()
            .zip(&vector)
            .all(|(a, b)| (a - b).abs() < 0.00001),
        "prepared query must preserve the existing instruction rendering"
    );

    let norm = vector.iter().map(|v| v * v).sum::<f32>();
    assert!((norm - 1.0).abs() < 0.001);
    store
        .ensure_vector_table(VectorTableSpec {
            slug: "local_selected",
            model: &model,
            profile: "local",
            dims: 32,
            created_at: 1,
            active: true,
        })
        .unwrap();
    let hash = [7_u8; 32];
    store
        .writer()
        .unwrap()
        .execute(
            "INSERT INTO unit_contents(hash,text,tokens) VALUES(?1,'local retrieval fixture',1)",
            [hash.as_slice()],
        )
        .unwrap();
    store.writer().unwrap().execute("INSERT INTO units(id,hash,content_hash,source_key,kind,header,created_at,updated_at) VALUES(1,?1,?1,'local-query:1','chunk','fixture',1,1)",[hash.as_slice()]).unwrap();
    store
        .insert_vector(VectorInsert {
            table_slug: "local_selected",
            unit_id: 1,
            ctx_hash: &hash,
            embedding_hash: &hash,
            embedding: &vector,
        })
        .unwrap();
    let hits = dense::retrieve(
        Arc::clone(&factory),
        "local_selected".into(),
        query.into(),
        dense::Scope::default(),
        Instant::now() + Duration::from_secs(5),
        || 100,
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].unit_id, 1);
    assert!(hits[0].distance.abs() < 0.001);
    let usage: (i64, i64, i64) = store
        .writer()
        .unwrap()
        .query_row(
            "SELECT calls,tokens,estimated_tokens FROM usage WHERE provider='embed:local'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(usage.0, 2);
    assert!(usage.1 > 0);
    assert_eq!(usage.1, usage.2);
    let endpoints:i64 = store.writer().unwrap().query_row("SELECT count(*) FROM usage_reservations WHERE endpoint_kind IS NOT NULL OR endpoint_url IS NOT NULL OR sent_units IS NOT NULL",[],|r|r.get(0)).unwrap();
    assert_eq!(endpoints, 0);
    let cfg_path = root.path().join(".deepkit/config.toml");
    let cfg = fs::read_to_string(&cfg_path).unwrap();
    fs::write(&cfg_path, cfg.replace("threads=2", "threads=3")).unwrap();
    assert!(matches!(
        factory
            .embed(
                query.into(),
                Instant::now() + Duration::from_secs(5),
                || 100
            )
            .await,
        Err(admission::Error::Refused(usage::Dispatch::Superseded))
    ));
    let calls: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT sum(calls) FROM usage", [], |r| r.get(0))
        .unwrap();
    assert_eq!(calls, 2);
    println!(
        "CPU local query: dims=32, retrieved=1, calls={}, estimated_tokens={}, outbound_endpoints=0",
        usage.0, usage.2
    );
}
