//! Exact sampling boundaries, persistence, invalidation and write refusal.
use super::*;
use crate::store::StoreLimits;

fn fixture() -> (tempfile::TempDir, Arc<Control>) {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join(".deepkit")).unwrap();
    std::fs::write(
        root.path().join(".deepkit/config.toml"),
        "[rerank]\nmode='auto'\n",
    )
    .unwrap();
    let config = config::load(root.path(), None).unwrap().config;
    let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
    let control = Control::new(store, root.path().into(), config).unwrap();
    (root, control)
}
async fn settled(control: &Control) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while control.persistence.load(Ordering::SeqCst) == 1 {
        assert!(Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
#[tokio::test]
async fn only_first_full_call_samples_and_exact_threshold_stays_at_fifty() {
    let (_root, control) = fixture();
    assert!(control.start(20, Instant::now()).is_none());
    let sample = control.start(50, Instant::now()).unwrap();
    assert!(control.start(50, Instant::now()).is_none());
    drop(sample); // Early cancellation releases the slot.
    assert_eq!(control.phase.load(Ordering::SeqCst), 0);
    control.observe(THRESHOLD, true);
    assert_eq!(control.snapshot().effective_candidates, 50);
    assert!(control.start(50, Instant::now()).is_none());
    assert!(control.store.get_meta(KEY).unwrap().is_none());
    for (mode, count) in [("on", 50), ("off", 50), ("auto", 20), ("auto", 100)] {
        let mut config = control.config.clone();
        config.rerank.mode = mode.into();
        config.rerank.candidates = count;
        let other = Control::new(Arc::clone(&control.store), control.root.clone(), config).unwrap();
        assert!(other.start(50, Instant::now()).is_none());
    }
}
#[tokio::test]
async fn slow_incomplete_sample_persists_and_configuration_change_invalidates_it() {
    let (_root, control) = fixture();
    let sample = control
        .start(50, Instant::now() - Duration::from_millis(1600))
        .unwrap();
    drop(sample);
    assert_eq!(control.snapshot().effective_candidates, 20);
    assert!(control.snapshot().incomplete_sample);
    settled(&control).await;
    assert_eq!(control.snapshot().persistence, "saved");
    let restarted = Control::new(
        Arc::clone(&control.store),
        control.root.clone(),
        control.config.clone(),
    )
    .unwrap();
    assert_eq!(restarted.snapshot().effective_candidates, 20);
    assert_eq!(
        read_status(&control.store, &control.config)
            .unwrap()
            .effective_candidates,
        20
    );
    let mut other = control.config.clone();
    other.rerank.remote.model = "changed-model".into();
    assert_eq!(
        read_status(&control.store, &other)
            .unwrap()
            .effective_candidates,
        50
    );
    assert_eq!(
        control
            .store
            .writer()
            .unwrap()
            .query_row("SELECT count(*) FROM usage_reservations", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
#[tokio::test]
async fn failed_or_stale_persistence_keeps_live_reduction_visible_without_forging_a_saved_one() {
    for stale in [false, true] {
        let (root, control) = fixture();
        if stale {
            std::fs::write(
                root.path().join(".deepkit/config.toml"),
                "[rerank]\nmode='on'\n",
            )
            .unwrap();
        } else {
            control.store.writer().unwrap().execute_batch("CREATE TRIGGER deny_auto BEFORE INSERT ON meta WHEN NEW.key='rerank_auto_v1' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
        }
        control.observe(THRESHOLD + Duration::from_nanos(1), true);
        settled(&control).await;
        assert_eq!(control.snapshot().effective_candidates, 20);
        assert_eq!(control.snapshot().persistence, "deferred");
        assert!(control.store.get_meta(KEY).unwrap().is_none());
    }
    let (_root, control) = fixture();
    control.store.set_meta(KEY, "invalid calibration").unwrap();
    assert_eq!(
        read_status(&control.store, &control.config)
            .unwrap()
            .effective_candidates,
        50
    );
}
