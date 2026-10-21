//! Worker lifecycle tests use authoritative SQLite inputs and injected provider work.
use super::*;
use crate::{
    queue::{self, Owner},
    store::{self, StoreLimits},
};
use rusqlite::params;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

#[derive(Clone)]
struct TestWork {
    store: Arc<Store>,
    calls: Arc<AtomicUsize>,
    behavior: u8,
    clock: Arc<AtomicI64>,
}
#[async_trait]
impl Work for TestWork {
    type Output = i64;
    fn current(&self, tx: &Transaction<'_>, _: &Lease) -> Result<Option<CurrentInput>, Error> {
        let allowed: bool = tx.query_row("SELECT allowed FROM current_input", [], |r| r.get(0))?;
        Ok(allowed.then_some(CurrentInput {
            target_hash: [1; 32],
            desired_hash: [2; 32],
        }))
    }
    fn limits(&self, _: &Transaction<'_>, _: &str) -> Result<usage::Limits, Error> {
        Ok(usage::Limits {
            budget: usage::Budget {
                calls: None,
                tokens: None,
            },
            rpm: Some(if self.behavior == 5 { 0 } else { 100 }),
            tpm: Some(10000),
        })
    }
    async fn call(&self) -> Result<Response<i64>, AttemptError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.behavior {
            1 => {
                self.store
                    .writer()
                    .unwrap()
                    .execute("UPDATE current_input SET allowed=0", [])
                    .unwrap();
            }
            2 => return Err(AttemptError::Rejected { status: 401 }),
            3 => return std::future::pending().await,
            8..=11 => {
                if matches!(self.behavior, 9 | 11) {
                    self.store
                        .writer()
                        .unwrap()
                        .execute("UPDATE jobs SET lease_generation=lease_generation+1", [])
                        .unwrap();
                }
                let error = if self.behavior >= 10 {
                    crate::models::openai_compat::chat::Error::IncompleteContent { tokens: Some(6) }
                } else {
                    crate::models::openai_compat::chat::Error::ThinkingConsumedBudget {
                        tokens: Some(6),
                    }
                };
                return Err(error.into());
            }
            4 => {
                self.clock.store(119, Ordering::SeqCst);
                loop {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    let until: i64 = self
                        .store
                        .writer()
                        .unwrap()
                        .query_row("SELECT leased_until FROM jobs", [], |r| r.get(0))
                        .unwrap();
                    if until == 239 {
                        break;
                    }
                }
                self.clock.store(121, Ordering::SeqCst);
            }
            12 | 14 => self.clock.store(29, Ordering::SeqCst),
            13 => self.clock.store(120, Ordering::SeqCst),
            15 => {
                self.clock.store(29, Ordering::SeqCst);
                self.store
                    .writer()
                    .unwrap()
                    .execute("UPDATE jobs SET lease_generation=lease_generation+1", [])
                    .unwrap();
            }
            6 => {
                self.store
                    .writer()
                    .unwrap()
                    .execute("UPDATE jobs SET lease_generation=lease_generation+1", [])
                    .unwrap();
                return std::future::pending().await;
            }
            _ => {}
        }
        Ok(Response {
            output: 7,
            tokens: Some(5),
        })
    }
    fn publish(&self, tx: &Transaction<'_>, output: i64) -> Result<(), Error> {
        tx.execute("INSERT INTO output VALUES(?1)", [output])?;
        if self.behavior == 12 {
            self.clock.store(130, Ordering::SeqCst);
        } else if self.behavior == 14 {
            self.clock.store(149, Ordering::SeqCst);
        }
        if self.behavior == 7 {
            return Err(Error::Invalid("injected output failure"));
        }
        Ok(())
    }
}
fn setup() -> (Arc<Store>, Lease) {
    let store = store::open(":memory:", StoreLimits::default()).unwrap();
    store.writer().unwrap().execute_batch("CREATE TABLE current_input(allowed INTEGER); INSERT INTO current_input VALUES(1); CREATE TABLE output(value INTEGER);").unwrap();
    queue::enqueue(
        &store,
        &queue::Enqueue {
            kind: queue::Kind::Embed,
            target: queue::Target::Unit,
            target_id: 1,
            target_hash: &[1; 32],
            desired_hash: &[2; 32],
            provider: "embed:test",
            priority: 1.0,
            tokens: 10,
        },
        0,
    )
    .unwrap();
    let lease = queue::claim(&store, "embed:test", &Owner::new(&store).unwrap(), 0, 1, 10)
        .unwrap()
        .remove(0);
    (Arc::new(store), lease)
}
fn settings() -> Settings<'static> {
    Settings {
        provider: "embed:test",
        budget: usage::Budget {
            calls: None,
            tokens: None,
        },
        max_output_tokens: 0,
        timeout: Duration::from_secs(1),
        heartbeat: Duration::from_millis(10),
    }
}
fn snapshot(store: &Store) -> (String, i64, i64, i64) {
    store.writer().unwrap().query_row("SELECT state,attempts,(SELECT COUNT(*) FROM output),(SELECT COUNT(*) FROM usage_reservations WHERE state='dispatched') FROM jobs",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).unwrap()
}

#[tokio::test]
async fn chat_empty_or_incomplete_response_settles_actual_usage_without_retry_even_after_lease_loss()
 {
    for behavior in [8, 9, 10, 11] {
        let (store, lease) = setup();
        let work = TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior,
            clock: Arc::new(AtomicI64::new(0)),
        };
        let result = execute(&store, &lease, &settings(), &work, || 0)
            .await
            .unwrap();
        assert_eq!(
            result,
            Finished::Scheduled(if matches!(behavior, 8 | 10) {
                retry::Scheduled::Dead
            } else {
                retry::Scheduled::LeaseLost
            })
        );
        assert_eq!(work.calls.load(Ordering::SeqCst), 1);
        let usage: (i64, i64, i64) = store
            .writer()
            .unwrap()
            .query_row("SELECT calls,tokens,estimated_tokens FROM usage", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!(usage, (1, 6, 0));
        let snapshot = snapshot(&store);
        assert_eq!((snapshot.2, snapshot.3), (0, 0));
        if behavior == 10 {
            let reason: String = store
                .writer()
                .unwrap()
                .query_row("SELECT last_error FROM jobs", [], |row| row.get(0))
                .unwrap();
            assert_eq!(reason, "incomplete page transcription; diagnosis required");
            assert_eq!(snapshot.0, "dead");
            assert_eq!(snapshot.1, 1);
        }
    }
}

#[tokio::test]
async fn worker_publishes_once_and_rechecks_policy_after_response() {
    for behavior in [0, 1] {
        let (store, lease) = setup();
        let now = Arc::new(AtomicI64::new(0));
        let work = TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior,
            clock: Arc::clone(&now),
        };
        let result = execute(&store, &lease, &settings(), &work, {
            let now = Arc::clone(&now);
            move || now.load(Ordering::SeqCst)
        })
        .await
        .unwrap();
        assert_eq!(
            result,
            Finished::Completed(if behavior == 0 {
                Completion::Published
            } else {
                Completion::Discarded
            })
        );
        assert_eq!(
            snapshot(&store),
            ("done".into(), 1, i64::from(behavior == 0), 0)
        );
        let tokens: i64 = store
            .writer()
            .unwrap()
            .query_row("SELECT tokens FROM usage", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tokens, 5);
        assert_eq!(
            execute(&store, &lease, &settings(), &work, || 0)
                .await
                .unwrap(),
            Finished::LeaseLost
        );
        assert_eq!(work.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn worker_checks_policy_and_quota_before_polling_provider() {
    for behavior in [0, 5, 8] {
        let (store, lease) = setup();
        let now = Arc::new(AtomicI64::new(0));
        if behavior == 0 {
            store
                .writer()
                .unwrap()
                .execute("UPDATE current_input SET allowed=0", [])
                .unwrap();
        }
        let work = TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior,
            clock: Arc::clone(&now),
        };
        let mut cfg = settings();
        if behavior == 8 {
            cfg.budget.calls = Some(0);
        }
        let result = execute(&store, &lease, &cfg, &work, || 0).await.unwrap();
        assert_eq!(work.calls.load(Ordering::SeqCst), 0);
        assert_eq!(snapshot(&store).1, 0);
        match behavior {
            0 => assert_eq!(result, Finished::Completed(Completion::Discarded)),
            5 => assert_eq!(result, Finished::Scheduled(retry::Scheduled::Dead)),
            _ => assert_eq!(
                result,
                Finished::Scheduled(retry::Scheduled::RetryAt(86400))
            ),
        }
    }
}

#[tokio::test]
async fn worker_renews_active_call_and_fences_lost_ownership() {
    for behavior in [4, 6] {
        let (store, lease) = setup();
        let now = Arc::new(AtomicI64::new(0));
        let work = TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior,
            clock: Arc::clone(&now),
        };
        let result = execute(&store, &lease, &settings(), &work, {
            let now = Arc::clone(&now);
            move || now.load(Ordering::SeqCst)
        })
        .await
        .unwrap();
        assert_eq!(
            result,
            if behavior == 4 {
                Finished::Completed(Completion::Published)
            } else {
                Finished::LeaseLost
            }
        );
        assert_eq!(snapshot(&store).3, i64::from(behavior == 6));
    }
}

#[tokio::test]
async fn worker_deadline_and_cancellation_preserve_uncertain_charges() {
    for cancel in [false, true] {
        let (store, lease) = setup();
        let now = Arc::new(AtomicI64::new(0));
        let work = TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior: 3,
            clock: Arc::clone(&now),
        };
        let mut cfg = settings();
        if !cancel {
            cfg.timeout = Duration::from_millis(30);
        }
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            execute(&store, &lease, &cfg, &work, || 0),
        )
        .await;
        if cancel {
            assert!(result.is_err());
            assert_eq!(snapshot(&store).0, "leased");
        } else {
            assert!(matches!(
                result.unwrap().unwrap(),
                Finished::Scheduled(retry::Scheduled::RetryAt(_))
            ));
            assert_eq!(snapshot(&store).0, "failed");
        }
        assert_eq!(snapshot(&store).3, 1);
        assert_eq!(work.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn worker_terminal_errors_and_failed_publication_do_not_lose_accounting() {
    for behavior in [2, 7] {
        let (store, lease) = setup();
        let now = Arc::new(AtomicI64::new(0));
        let work = TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior,
            clock: Arc::clone(&now),
        };
        let result = execute(&store, &lease, &settings(), &work, || 0).await;
        if behavior == 2 {
            assert_eq!(result.unwrap(), Finished::Scheduled(retry::Scheduled::Dead));
        } else {
            assert!(result.is_err());
            assert_eq!(snapshot(&store).0, "leased");
        }
        assert_eq!(snapshot(&store).2, 0);
        let charged:i64=store.writer().unwrap().query_row("SELECT COALESCE((SELECT SUM(tokens) FROM usage),0)+COALESCE((SELECT SUM(tokens) FROM usage_reservations WHERE state='dispatched'),0)",params![],|r|r.get(0)).unwrap();
        assert_eq!(charged, if behavior == 2 { 10 } else { 5 });
    }
}

#[test]
fn contended_heartbeat_samples_time_after_sqlite_write_lock() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let first = store::open(file.path(), StoreLimits::default()).unwrap();
    let second = store::open(file.path(), StoreLimits::default()).unwrap();
    queue::enqueue(
        &first,
        &queue::Enqueue {
            kind: queue::Kind::Embed,
            target: queue::Target::Unit,
            target_id: 1,
            target_hash: &[1; 32],
            desired_hash: &[2; 32],
            provider: "embed:test",
            priority: 1.0,
            tokens: 10,
        },
        0,
    )
    .unwrap();
    let lease = queue::claim(&first, "embed:test", &Owner::new(&first).unwrap(), 0, 1, 10)
        .unwrap()
        .remove(0);
    let now = Arc::new(AtomicI64::new(119));
    let started = std::sync::Barrier::new(2);
    let mut guard = first.writer().unwrap();
    let tx = guard
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    std::thread::scope(|scope| {
        let renewal = scope.spawn(|| {
            started.wait();
            queue::renew_with_clock(&second, &lease, {
                let now = Arc::clone(&now);
                move || now.load(Ordering::SeqCst)
            })
            .unwrap()
        });
        started.wait();
        std::thread::sleep(Duration::from_millis(30));
        now.store(120, Ordering::SeqCst);
        tx.commit().unwrap();
        assert!(!renewal.join().unwrap());
    });
}

#[derive(Clone)]
struct HttpWork {
    base: TestWork,
    provider: Arc<crate::models::openai_compat::OpenAiEmbedder>,
}
#[async_trait]
impl Work for HttpWork {
    type Output = Vec<Vec<f32>>;
    fn current(&self, tx: &Transaction<'_>, lease: &Lease) -> Result<Option<CurrentInput>, Error> {
        self.base.current(tx, lease)
    }
    fn limits(&self, tx: &Transaction<'_>, provider: &str) -> Result<usage::Limits, Error> {
        self.base.limits(tx, provider)
    }
    async fn call(&self) -> Result<Response<Self::Output>, AttemptError> {
        let docs = [crate::models::EmbedInput::from_context(
            "context",
            "body",
            crate::store::UnitKind::Symbol,
        )];
        self.provider
            .embed_documents_once(&docs)
            .await
            .map(|output| Response {
                output,
                tokens: None,
            })
    }
    fn publish(&self, tx: &Transaction<'_>, output: Self::Output) -> Result<(), Error> {
        assert_eq!(output, vec![vec![0.6, 0.8, 0.0, 0.0]]);
        self.base.publish(tx, output.len() as i64)
    }
}

#[tokio::test]
async fn worker_drives_real_remote_adapter_and_settles_estimated_usage() {
    use crate::models::openai_compat::{OpenAiConfig, OpenAiEmbedder};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            include_str!("../../../fixtures/openai_compat/embeddings-success.json"),
            "application/json",
        ))
        .expect(1)
        .mount(&server)
        .await;
    let (store, lease) = setup();
    let now = Arc::new(AtomicI64::new(0));
    let work = HttpWork {
        base: TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior: 0,
            clock: Arc::clone(&now),
        },
        provider: Arc::new(
            OpenAiEmbedder::with_api_key(
                OpenAiConfig {
                    url: format!("{}/v1", server.uri()),
                    model: "worker-test".into(),
                    api_key_env: "UNUSED_TEST_KEY".into(),
                    encoding_format: "base64".into(),
                    query_prefix: String::new(),
                    max_batch_items: 1,
                    max_batch_tokens: 100,
                    dims: 4,
                    timeout: Duration::from_secs(1),
                },
                "synthetic-test-key".into(),
            )
            .unwrap(),
        ),
    };
    assert_eq!(
        execute(&store, &lease, &settings(), &work, || 0)
            .await
            .unwrap(),
        Finished::Completed(Completion::Published)
    );
    assert_eq!(snapshot(&store), ("done".into(), 1, 1, 0));
    let estimated: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT estimated_tokens FROM usage", [], |r| r.get(0))
        .unwrap();
    assert_eq!(estimated, 10);
    server.verify().await;
}

#[derive(Clone)]
struct PendingWork {
    base: TestWork,
    dropped: Arc<AtomicBool>,
}
#[async_trait]
impl Work for PendingWork {
    type Output = i64;
    fn current(&self, tx: &Transaction<'_>, lease: &Lease) -> Result<Option<CurrentInput>, Error> {
        self.base.current(tx, lease)
    }
    fn limits(&self, tx: &Transaction<'_>, provider: &str) -> Result<usage::Limits, Error> {
        self.base.limits(tx, provider)
    }
    async fn call(&self) -> Result<Response<i64>, AttemptError> {
        self.base.calls.fetch_add(1, Ordering::SeqCst);
        let _drop = CancelOnDrop(Arc::clone(&self.dropped));
        std::future::pending().await
    }
    fn publish(&self, tx: &Transaction<'_>, output: i64) -> Result<(), Error> {
        self.base.publish(tx, output)
    }
}

fn hold_writer(store: Arc<Store>) -> (std::sync::mpsc::Sender<()>, std::thread::JoinHandle<()>) {
    let (release, wait) = std::sync::mpsc::channel();
    let (entered, ready) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _guard = store.writer().unwrap();
        entered.send(()).unwrap();
        // Always release on test failure, so a regression cannot hang the suite.
        let _ = wait.recv_timeout(Duration::from_secs(2));
    });
    ready.recv_timeout(Duration::from_secs(1)).unwrap();
    (release, thread)
}

#[tokio::test(flavor = "current_thread")]
async fn blocked_heartbeat_does_not_block_deadline_or_other_runtime_timers() {
    let (store, lease) = setup();
    let work = PendingWork {
        base: TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior: 0,
            clock: Arc::new(AtomicI64::new(0)),
        },
        dropped: Arc::new(AtomicBool::new(false)),
    };
    let mut cfg = settings();
    cfg.timeout = Duration::from_millis(50);
    cfg.heartbeat = Duration::from_millis(10);
    let (result, ()) = tokio::join!(execute(&store, &lease, &cfg, &work, || 0), async {
        while work.base.calls.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let (release, thread) = hold_writer(Arc::clone(&store));
        let started = std::time::Instant::now();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let elapsed = started.elapsed();
        let dropped = work.dropped.load(Ordering::SeqCst);
        let _ = release.send(());
        thread.join().unwrap();
        assert!(
            elapsed < Duration::from_secs(1),
            "SQLite blocked the single-thread runtime: {elapsed:?}"
        );
        assert!(
            dropped,
            "deadline must drop the provider future before retry waits for SQL"
        );
    });
    assert!(matches!(
        result.unwrap(),
        Finished::Scheduled(retry::Scheduled::RetryAt(_))
    ));
    assert_eq!(snapshot(&store), ("failed".into(), 1, 0, 1));
}

#[tokio::test(flavor = "current_thread")]
async fn cancelling_blocked_admission_does_not_dispatch_after_lock_release() {
    let (store, lease) = setup();
    let work = TestWork {
        store: Arc::clone(&store),
        calls: Arc::new(AtomicUsize::new(0)),
        behavior: 0,
        clock: Arc::new(AtomicI64::new(0)),
    };
    let (release, thread) = hold_writer(Arc::clone(&store));
    let cfg = settings();
    let started = std::time::Instant::now();
    let result = tokio::time::timeout(
        Duration::from_millis(30),
        execute(&store, &lease, &cfg, &work, || 0),
    )
    .await;
    let elapsed = started.elapsed();
    let _ = release.send(());
    thread.join().unwrap();
    assert!(result.is_err());
    assert!(elapsed < Duration::from_secs(1));
    tokio::time::timeout(Duration::from_secs(1), async {
        while Arc::strong_count(&work.calls) > 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(work.calls.load(Ordering::SeqCst), 0);
    assert_eq!(snapshot(&store), ("leased".into(), 0, 0, 0));
    let reservations: i64 = store
        .writer()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM usage_reservations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(reservations, 0);
}

// This fixture tests the queue's L0 file admission boundary only. The real T3.6
// image/page context builder, privacy resolver, VLM transport and output writer
// remain separate acceptance work; none is supplied by this mock Work.
#[derive(Clone)]
struct L0FileWork {
    file_id: i64,
    hash: [u8; 32],
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Work for L0FileWork {
    type Output = i64;

    fn current(&self, tx: &Transaction<'_>, _: &Lease) -> Result<Option<CurrentInput>, Error> {
        let allowed: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM files WHERE id=?1 AND blake3=?2)
             AND (SELECT allowed FROM current_input)",
            params![self.file_id, self.hash.as_slice()],
            |row| row.get(0),
        )?;
        Ok(allowed.then_some(CurrentInput {
            target_hash: self.hash,
            desired_hash: [2; 32],
        }))
    }

    fn limits(&self, _: &Transaction<'_>, _: &str) -> Result<usage::Limits, Error> {
        Ok(usage::Limits {
            budget: usage::Budget {
                calls: None,
                tokens: None,
            },
            rpm: Some(100),
            tpm: Some(10000),
        })
    }

    async fn call(&self) -> Result<Response<i64>, AttemptError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Response {
            output: self.file_id,
            tokens: Some(5),
        })
    }

    fn publish(&self, tx: &Transaction<'_>, output: i64) -> Result<(), Error> {
        tx.execute("INSERT INTO output VALUES(?1)", [output])?;
        Ok(())
    }
}

fn l0_page_fixture() -> Vec<u8> {
    // A complete one-page blank PDF; no parser or renderer runs in this test.
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>",
    ];
    let mut pdf = String::from("%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.push_str(&format!("{} 0 obj\n{object}\nendobj\n", index + 1));
    }
    let xref = pdf.len();
    pdf.push_str("xref\n0 4\n0000000000 65535 f \n");
    for offset in offsets {
        pdf.push_str(&format!("{offset:010} 00000 n \n"));
    }
    pdf.push_str(&format!(
        "trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ));
    pdf.into_bytes()
}

async fn check_l0_file_admission(deny_privacy: bool, deny_budget: bool) {
    for kind in [queue::Kind::DescribeImage, queue::Kind::VlmPage] {
        let root = tempfile::tempdir().unwrap();
        let (path, file_kind, bytes) = if matches!(kind, queue::Kind::DescribeImage) {
            (
                "image.svg",
                store::FileKind::Image,
                br#"<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><rect width="1" height="1"/></svg>"#.to_vec(),
            )
        } else {
            ("page.pdf", store::FileKind::Doc, l0_page_fixture())
        };
        std::fs::write(root.path().join(path), &bytes).unwrap();
        let hash = *blake3::hash(&bytes).as_bytes();
        let store = Arc::new(store::open(":memory:", StoreLimits::default()).unwrap());
        let file_id = store
            .upsert_file(store::FileUpsert {
                path,
                blake3: &hash,
                size: bytes.len() as i64,
                mtime_ns: 0,
                lang: None,
                kind: file_kind,
                git_blob: None,
                // Permission for L3 is not a completed L1/L2 publication.
                max_level: 3,
                treatment: "l0_admission_fixture",
                indexed_at: None,
                needs_vlm: matches!(kind, queue::Kind::VlmPage),
            })
            .unwrap();
        store
            .writer()
            .unwrap()
            .execute_batch(
                "CREATE TABLE current_input(allowed INTEGER);
             INSERT INTO current_input VALUES(1);
             CREATE TABLE output(value INTEGER);",
            )
            .unwrap();
        if deny_privacy {
            store
                .writer()
                .unwrap()
                .execute("UPDATE current_input SET allowed=0", [])
                .unwrap();
        }
        queue::enqueue(
            &store,
            &queue::Enqueue {
                kind,
                target: queue::Target::File,
                target_id: file_id,
                target_hash: &hash,
                desired_hash: &[2; 32],
                provider: "vlm:test",
                priority: 1.0,
                tokens: 10,
            },
            0,
        )
        .unwrap();
        let lease = queue::claim(&store, "vlm:test", &Owner::new(&store).unwrap(), 0, 1, 10)
            .unwrap()
            .remove(0);
        let work = L0FileWork {
            file_id,
            hash,
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut cfg = settings();
        cfg.provider = "vlm:test";
        if deny_budget {
            cfg.budget.calls = Some(0);
        }
        let result = execute(&store, &lease, &cfg, &work, || 0).await.unwrap();
        let admitted = !deny_privacy && !deny_budget;
        assert_eq!(
            result,
            if deny_privacy {
                Finished::Completed(Completion::Discarded)
            } else if deny_budget {
                Finished::Scheduled(retry::Scheduled::RetryAt(86400))
            } else {
                Finished::Completed(Completion::Published)
            }
        );
        assert_eq!(work.calls.load(Ordering::SeqCst), usize::from(admitted));
        let (_, attempts, outputs, dispatched) = snapshot(&store);
        assert_eq!(
            (attempts, outputs, dispatched),
            (i64::from(admitted), i64::from(admitted), 0)
        );
        let writer = store.writer().unwrap();
        let ledger: (i64, i64, i64) = writer
            .query_row(
                "SELECT COALESCE((SELECT SUM(calls) FROM usage),0),
                    COALESCE((SELECT SUM(tokens) FROM usage),0),
                    (SELECT COUNT(*) FROM usage_reservations WHERE state='reconciled')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            ledger,
            (
                i64::from(admitted),
                5 * i64::from(admitted),
                i64::from(admitted)
            )
        );
        let l0: (i64, i64, i64, i64) = writer
            .query_row(
                "SELECT (SELECT COUNT(*) FROM files WHERE id=?1 AND indexed_at IS NULL AND ladder_level=0),
                    (SELECT COUNT(*) FROM units),
                    (SELECT COUNT(*) FROM embedding_publications),
                    (SELECT COUNT(*) FROM embedding_tables)",
                [file_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(l0, (1, 0, 0, 0));
        assert_eq!(std::fs::read(root.path().join(path)).unwrap(), bytes);
    }
}

#[tokio::test]
async fn worker_admits_l0_image_and_page_file_jobs_without_l1_or_l2() {
    check_l0_file_admission(false, false).await;
}

#[tokio::test]
async fn worker_denies_l0_image_and_page_before_call_when_daily_budget_is_zero() {
    check_l0_file_admission(false, true).await;
}

#[tokio::test]
async fn worker_denies_l0_image_and_page_before_call_when_mock_privacy_disallows() {
    check_l0_file_admission(true, false).await;
}

#[tokio::test]
async fn response_renews_live_lease_but_never_revives_expired_or_replaced_work() {
    for behavior in [12, 13, 14, 15] {
        let (store, lease) = setup();
        let now = Arc::new(AtomicI64::new(0));
        let work = TestWork {
            store: Arc::clone(&store),
            calls: Arc::new(AtomicUsize::new(0)),
            behavior,
            clock: Arc::clone(&now),
        };
        let mut settings = settings();
        settings.heartbeat = Duration::from_secs(30);
        let result = execute(&store, &lease, &settings, &work, {
            let now = Arc::clone(&now);
            move || now.load(Ordering::SeqCst)
        })
        .await
        .unwrap();
        assert_eq!(
            result,
            match behavior {
                12 | 14 => Finished::Completed(Completion::Published),
                _ => Finished::LeaseLost,
            }
        );
        assert_eq!(work.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            snapshot(&store),
            (
                if matches!(behavior, 12 | 14) {
                    "done"
                } else {
                    "leased"
                }
                .into(),
                1,
                i64::from(matches!(behavior, 12 | 14)),
                0,
            )
        );
        let usage: (i64, i64) = store
            .writer()
            .unwrap()
            .query_row("SELECT calls,tokens FROM usage", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(usage, (1, 5));
    }
}
