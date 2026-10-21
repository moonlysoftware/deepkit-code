//! First-use concurrency, deadline retention and no-download refusal; v1 §7.4.
use super::*;
use crate::{
    config::Config,
    queue::usage,
    store::{self, StoreLimits},
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
fn setup(profile: &str) -> (tempfile::TempDir, Arc<Store>, Settings) {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join(".deepkit")).unwrap();
    let mut config = Config::default();
    config.embedder.profile = profile.into();
    config.limits.models_cache_max_bytes = "0 B".into();
    config.rerank.remote.url = "http://127.0.0.1:9".into();
    std::fs::write(
        root.path().join(".deepkit/config.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    let settings = Settings {
        config_root: root.path().into(),
        config,
        limits: usage::Limits {
            budget: usage::Budget {
                calls: Some(10),
                tokens: Some(10000),
            },
            rpm: Some(10),
            tpm: Some(10000),
        },
    };
    (
        root,
        Arc::new(store::open(":memory:", StoreLimits::default()).unwrap()),
        settings,
    )
}
fn scripted(work: Option<Work>) -> Arc<Host> {
    let (_root, store, settings) = setup("remote");
    Host::with_work(work, store, settings.config)
}
#[tokio::test]
async fn one_attempt_survives_waiter_timeout_and_cancellation_and_reuses_factory() {
    let (_root, store, settings) = setup("remote");
    let factory = Factory::build(store, settings, |config, timeout| {
        Ok(models::rerank::remote::Remote::with_api_key(
            config,
            timeout,
            "fixture-key".into(),
        )?)
    })
    .unwrap();
    let expected = Arc::clone(&factory);
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let (release, wait) = tokio::sync::oneshot::channel();
    let host = scripted(Some(Box::pin(async move {
        counted.fetch_add(1, Ordering::SeqCst);
        wait.await.map_err(|_| Error::Task)?;
        Ok(factory)
    })));
    assert_eq!(host.peek().status().state, "idle");
    assert!(matches!(host.get(Instant::now()).await, Outcome::Idle));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(matches!(
        host.get(Instant::now() + Duration::from_millis(20)).await,
        Outcome::Pending
    ));
    let other = Arc::clone(&host);
    let waiter =
        tokio::spawn(async move { other.get(Instant::now() + Duration::from_secs(5)).await });
    tokio::task::yield_now().await;
    waiter.abort();
    let _ = waiter.await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    release.send(()).unwrap();
    for _ in 0..2 {
        let Outcome::Ready(actual) = host.get(Instant::now() + Duration::from_secs(2)).await else {
            panic!("not ready")
        };
        assert!(Arc::ptr_eq(&actual, &expected));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn failed_task_is_terminal_and_never_retried_by_later_waiters() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let host = scripted(Some(Box::pin(async move {
        counted.fetch_add(1, Ordering::SeqCst);
        Err(Error::Input)
    })));
    for _ in 0..3 {
        assert!(matches!(
            host.get(Instant::now() + Duration::from_secs(1)).await,
            Outcome::Failed(_)
        ));
        assert_eq!(host.peek().status().state, "failed");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let panic = scripted(Some(Box::pin(async {
        panic!("fixture preparation task failure")
    })));
    assert!(
        matches!(panic.get(Instant::now()+Duration::from_secs(1)).await,Outcome::Failed(error) if matches!(*error,Error::Task))
    );
}
#[tokio::test]
async fn off_stale_and_zero_cache_cap_do_not_download_or_publish_a_provider() {
    let (root, store, mut settings) = setup("local");
    let cache = root.path().join("cache");
    settings.config.rerank.mode = "off".into();
    let off = Host::new(Arc::clone(&store), settings.clone(), cache.clone());
    assert!(matches!(
        off.get(Instant::now() + Duration::from_secs(1)).await,
        Outcome::Off
    ));
    assert!(!cache.exists());
    settings.config.rerank.mode = "on".into(); // Persisted config still auto.
    let stale = Host::new(Arc::clone(&store), settings.clone(), cache.clone());
    assert!(
        matches!(stale.get(Instant::now()+Duration::from_secs(1)).await,Outcome::Failed(error) if matches!(*error,Error::Privacy))
    );
    assert!(!cache.exists());
    settings.config.rerank.mode = "auto".into();
    let capped = Host::new(Arc::clone(&store), settings.clone(), cache.clone());
    assert!(matches!(capped.peek(), Outcome::Idle));
    assert!(!cache.exists());
    let outcome = capped.get(Instant::now() + Duration::from_secs(3)).await;
    assert!(
        matches!(outcome,Outcome::Failed(error) if matches!(*error,Error::Download(models::download::Error::Cap {..})))
    );
    let saved = models::rerank::health::read(&store, &settings.config)
        .unwrap()
        .unwrap();
    assert_eq!(saved.state, models::rerank::health::State::Unavailable);
    assert!(!cache.join(saved.file.unwrap()).exists());
    assert_eq!(
        store
            .writer()
            .unwrap()
            .query_row("SELECT count(*) FROM usage_reservations", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn configuration_changed_during_load_never_publishes_the_completed_factory() {
    let (root, store, settings) = setup("remote");
    let (began, started) = tokio::sync::oneshot::channel();
    let (release, wait) = tokio::sync::oneshot::channel();
    let work = prepare_with(
        store,
        settings,
        root.path().join("unused-cache"),
        move |store, settings, _| async move {
            let factory = Factory::build(store, settings, |config, timeout| {
                Ok(models::rerank::remote::Remote::with_api_key(
                    config,
                    timeout,
                    "fixture-key".into(),
                )?)
            })?;
            began.send(()).unwrap();
            wait.await.map_err(|_| Error::Task)?;
            Ok(factory)
        },
    );
    let host = scripted(Some(Box::pin(work)));
    let cloned = Arc::clone(&host);
    let pending =
        tokio::spawn(async move { cloned.get(Instant::now() + Duration::from_secs(3)).await });
    started.await.unwrap();
    std::fs::write(
        root.path().join(".deepkit/config.toml"),
        "[rerank]\nmode='off'\n",
    )
    .unwrap();
    release.send(()).unwrap();
    assert!(
        matches!(pending.await.unwrap(),Outcome::Failed(error) if matches!(*error,Error::Privacy))
    );
}
#[test]
fn runtime_shutdown_cannot_leave_a_phantom_pending_preparation() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let host = scripted(Some(Box::pin(std::future::pending())));
    assert!(matches!(
        runtime.block_on(host.get(Instant::now() + Duration::from_millis(10))),
        Outcome::Pending
    ));
    drop(runtime);
    assert!(matches!(host.peek(),Outcome::Failed(error) if matches!(*error,Error::Task)));
}

#[tokio::test]
async fn adaptive_read_wait_obeys_deadline_without_starting_preparation() {
    let host = scripted(Some(Box::pin(async {
        panic!("selection started preparation")
    })));
    let store = Arc::clone(&host.store);
    let (locked, ready) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let holder = tokio::task::spawn_blocking(move || {
        let _guard = store.writer().unwrap();
        locked.send(()).unwrap();
        wait.recv().unwrap();
    });
    ready.await.unwrap();
    let result = host
        .selection_status(Instant::now() + Duration::from_millis(20))
        .await;
    release.send(()).unwrap();
    holder.await.unwrap();
    assert!(matches!(result, Err(Error::Deadline)));
    assert!(matches!(host.peek(), Outcome::Idle));
    assert_eq!(
        host.selection_status(Instant::now() + Duration::from_secs(1))
            .await
            .unwrap()
            .effective_candidates,
        50
    );
    assert!(matches!(host.peek(), Outcome::Idle));
}

#[tokio::test]
async fn forced_off_and_custom_bounds_do_not_depend_on_calibration_storage() {
    for (mode, candidates) in [("on", 50), ("off", 50), ("auto", 12)] {
        let (_root, store, mut settings) = setup("remote");
        settings.config.rerank.mode = mode.into();
        settings.config.rerank.candidates = candidates;
        store
            .writer()
            .unwrap()
            .execute_batch("DROP TABLE meta")
            .unwrap();
        let host = Host::with_work(None, store, settings.config);
        let status = host
            .selection_status(Instant::now() + Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(status.effective_candidates, candidates);
        assert!(!status.reduced);
    }
}
