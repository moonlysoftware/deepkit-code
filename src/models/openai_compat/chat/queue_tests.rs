//! Real chat transport under durable queue quotas; v1 §8.2 and T3.4.
use super::*;
use crate::{
    models::attempt::AttemptError,
    queue::{self, usage, worker},
    store::{self, Store, StoreLimits},
};
use rusqlite::Transaction;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[derive(Clone)]
struct Job {
    client: Arc<Client>,
    prepared: Arc<Prepared>,
    config: crate::config::Llm,
    kind: &'static str,
    endpoint: String,
}
#[async_trait]
impl worker::Work for Job {
    type Output = ();
    fn current(
        &self,
        _: &Transaction<'_>,
        _: &queue::Lease,
    ) -> Result<Option<queue::CurrentInput>, queue::Error> {
        Ok(Some(queue::CurrentInput {
            target_hash: [1; 32],
            desired_hash: [2; 32],
        }))
    }
    fn limits(&self, _: &Transaction<'_>, _: &str) -> Result<usage::Limits, queue::Error> {
        super::super::limits(&self.config)
    }
    fn outbound(&self) -> Result<Option<usage::Endpoint>, queue::Error> {
        Ok(Some(usage::Endpoint {
            kind: self.kind,
            url: self.endpoint.clone(),
            units: 1,
        }))
    }
    fn max_output_tokens(&self) -> Option<i64> {
        Some(self.prepared.max_output_tokens)
    }
    async fn call(&self) -> Result<worker::Response<()>, AttemptError> {
        let reply = self.client.post_once(&self.prepared).await?;
        Ok(worker::Response {
            output: (),
            tokens: reply.tokens,
        })
    }
    fn publish(&self, _: &Transaction<'_>, _: ()) -> Result<(), queue::Error> {
        Ok(())
    }
}
fn enqueue(store: &Store, provider: &str, id: i64, tokens: i64) -> queue::Lease {
    queue::enqueue(
        store,
        &queue::Enqueue {
            kind: if id == 1 {
                queue::Kind::DescribeSymbol
            } else {
                queue::Kind::DescribeImage
            },
            target: queue::Target::Unit,
            target_id: id,
            target_hash: &[1; 32],
            desired_hash: &[2; 32],
            provider,
            priority: 1.0,
            tokens,
        },
        0,
    )
    .unwrap();
    queue::claim(
        store,
        provider,
        &queue::Owner::new(store).unwrap(),
        0,
        1,
        tokens,
    )
    .unwrap()
    .remove(0)
}

#[tokio::test]
async fn chat_text_and_image_jobs_share_daily_and_minute_caps_across_database_handles() {
    for cap in ["daily", "rpm", "tpm"] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"content":"A useful description."}}],
                "usage":{"prompt_tokens":10,"completion_tokens":30,"total_tokens":40}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("code.db");
        let store = Arc::new(store::open(&path, StoreLimits::default()).unwrap());
        let other = Arc::new(store::open(&path, StoreLimits::default()).unwrap());
        let mut config = crate::config::Llm {
            enabled: true,
            url: format!("{}/v1", server.uri()),
            model: "fixture-text".into(),
            vlm_model: "fixture-image".into(),
            ..Default::default()
        };
        if cap == "daily" {
            config.max_calls_per_day = 1;
        }
        if cap == "rpm" {
            config.rpm = 1;
        }
        // Preparation is independent of quota values; build again with final limits.
        let client =
            Client::with_key(&config, Duration::from_secs(3), 1024, "fixture".into()).unwrap();
        let text = client
            .prepare_text("Summary.", "Public source.", 120)
            .unwrap();
        let mut image = Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut image, image::ImageFormat::Png)
            .unwrap();
        let picture = client
            .prepare_image(image.get_ref(), "image/png", "Describe.", 200)
            .unwrap();
        if cap == "tpm" {
            config.tpm = (text.input_tokens + text.max_output_tokens)
                .max(picture.input_tokens + picture.max_output_tokens)
                as u64
                + 10;
        }
        let client = Arc::new(
            Client::with_key(&config, Duration::from_secs(3), 1024, "fixture".into()).unwrap(),
        );
        let endpoint = client.endpoint.to_string();
        let provider = super::super::provider_id(&endpoint);
        let text = client
            .prepare_text("Summary.", "Public source.", 120)
            .unwrap();
        let picture = client
            .prepare_image(image.get_ref(), "image/png", "Describe.", 200)
            .unwrap();
        for (index, (prepared, kind, db)) in [(text, "llm", &store), (picture, "vlm", &other)]
            .into_iter()
            .enumerate()
        {
            let lease = enqueue(db, &provider, index as i64 + 1, prepared.input_tokens);
            let settings = worker::Settings {
                provider: &provider,
                budget: super::super::limits(&config).unwrap().budget,
                // Direct execution must use the prepared bound, not this fallback.
                max_output_tokens: 777,
                timeout: Duration::from_secs(3),
                heartbeat: Duration::from_secs(1),
            };
            let job = Job {
                client: Arc::clone(&client),
                prepared: Arc::new(prepared),
                config: config.clone(),
                kind,
                endpoint: endpoint.clone(),
            };
            let result = worker::execute(db, &lease, &settings, &job, || 0)
                .await
                .unwrap();
            if index == 0 {
                assert_eq!(
                    result,
                    worker::Finished::Completed(queue::Completion::Published)
                );
            } else {
                assert!(matches!(
                    result,
                    worker::Finished::Scheduled(queue::retry::Scheduled::RetryAt(_))
                ));
                let (state, attempts, before, reason): (String, i64, i64, String) = db
                    .writer()
                    .unwrap()
                    .query_row(
                        "SELECT state,attempts,not_before,last_error FROM jobs WHERE id=?1",
                        [lease.id],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                    )
                    .unwrap();
                assert_eq!(state, "queued");
                assert_eq!(attempts, 0);
                assert!(before > 0);
                assert!(reason.contains(if cap == "daily" {
                    "daily budget"
                } else {
                    "rate limit"
                }));
            }
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[tokio::test]
#[ignore = "requires DKC_LIVE_GATEWAY=1 and privately resolved Moonly credentials"]
async fn live_gateway_chat_text_and_image_keep_nonempty_results_and_durable_usage()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("DKC_LIVE_GATEWAY").as_deref() != Ok("1") {
        return Err("set DKC_LIVE_GATEWAY=1 for the explicit chat diagnostic".into());
    }
    let directory = tempfile::Builder::new()
        .prefix("dkc-live-chat-")
        .tempdir()?
        .keep();
    let path = directory.join("code.db");
    println!("live chat persistent usage ledger: {}", path.display());
    let store = Arc::new(store::open(&path, StoreLimits::default())?);
    let config = crate::config::Llm {
        enabled: true,
        ..Default::default()
    };
    assert_eq!(config.model, "Qwen3.5-35B-A3B");
    assert_eq!(config.vlm_model, "Qwen3.5-35B-A3B");
    let client = Arc::new(Client::from_config(
        &config,
        Duration::from_secs(60),
        1024 * 1024,
    )?);
    let text = client.prepare_text(
        "Answer briefly in plain English.",
        "What is two plus two?",
        120,
    )?;
    let mut image = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(64, 64, image::Rgb([255, 0, 0])))
        .write_to(&mut image, image::ImageFormat::Png)?;
    let picture = client.prepare_image(
        image.get_ref(),
        "image/png",
        "Name the main color in this image.",
        200,
    )?;
    let endpoint = client.endpoint.to_string();
    let provider = super::super::provider_id(&endpoint);
    let clock = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_secs()).ok())
            .unwrap_or(-1)
    };
    for (index, (prepared, kind)) in [(text, "llm"), (picture, "vlm")].into_iter().enumerate() {
        let now = clock();
        queue::enqueue(
            &store,
            &queue::Enqueue {
                kind: if index == 0 {
                    queue::Kind::DescribeSymbol
                } else {
                    queue::Kind::DescribeImage
                },
                target: queue::Target::Unit,
                target_id: index as i64 + 1,
                target_hash: &[1; 32],
                desired_hash: &[2; 32],
                provider: &provider,
                priority: 1.0,
                tokens: prepared.input_tokens,
            },
            now,
        )?;
        let lease = queue::claim(
            &store,
            &provider,
            &queue::Owner::new(&store)?,
            now,
            1,
            prepared.input_tokens,
        )?
        .remove(0);
        let settings = worker::Settings {
            provider: &provider,
            budget: super::super::limits(&config)?.budget,
            max_output_tokens: prepared.max_output_tokens,
            timeout: Duration::from_secs(60),
            heartbeat: Duration::from_secs(10),
        };
        let job = Job {
            client: Arc::clone(&client),
            prepared: Arc::new(prepared),
            config: config.clone(),
            kind,
            endpoint: endpoint.clone(),
        };
        let started = std::time::Instant::now();
        let result = worker::execute(&store, &lease, &settings, &job, clock).await?;
        assert_eq!(
            result,
            worker::Finished::Completed(queue::Completion::Published),
            "live chat response must be nonempty; retain the ledger on failure"
        );
        println!(
            "live chat {kind}: nonempty=true latency_ms={}",
            started.elapsed().as_millis()
        );
    }
    let totals: (i64, i64, i64) = store.writer()?.query_row(
        "SELECT SUM(calls),SUM(tokens),SUM(estimated_tokens) FROM usage",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    assert_eq!(totals.0, 2);
    assert!(totals.1 > 0);
    assert_eq!(totals.2, 0, "live provider must report usage");
    let kinds: i64 = store.writer()?.query_row(
        "SELECT COUNT(DISTINCT endpoint_kind) FROM usage_reservations WHERE state='reconciled'",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(kinds, 2);
    println!(
        "live chat usage: calls={} tokens={} estimated_tokens={} cost=null",
        totals.0, totals.1, totals.2
    );
    Ok(())
}

#[tokio::test]
async fn mixed_chat_pool_reserves_each_prepared_output_ceiling() {
    struct Jobs(std::collections::BTreeMap<i64, Job>);
    impl queue::pool::Factory for Jobs {
        type Job = Job;
        fn prepare(&self, _: &Store, lease: &queue::Lease) -> Result<Job, queue::Error> {
            self.0
                .get(&lease.id)
                .cloned()
                .ok_or(queue::Error::Invalid("fixture job"))
        }
    }
    for both in [true, false] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"content":"A useful description."}}]
            })))
            .expect(if both { 2 } else { 1 })
            .mount(&server)
            .await;
        let root = tempfile::tempdir().unwrap();
        let store =
            Arc::new(store::open(root.path().join("code.db"), StoreLimits::default()).unwrap());
        let mut config = crate::config::Llm {
            enabled: true,
            url: format!("{}/v1", server.uri()),
            model: "fixture".into(),
            ..Default::default()
        };
        let client = Arc::new(
            Client::with_key(&config, Duration::from_secs(3), 1024, "fixture".into()).unwrap(),
        );
        let endpoint = client.endpoint.to_string();
        let provider = super::super::provider_id(&endpoint);
        let prepared = [120, 300]
            .into_iter()
            .map(|ceiling| {
                client
                    .prepare_text("Summary.", "Public source.", ceiling)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let expected = prepared
            .iter()
            .take(if both { 2 } else { 1 })
            .map(|p| p.input_tokens + p.max_output_tokens)
            .sum::<i64>();
        config.max_tokens_per_day = expected as u64;
        let mut jobs = std::collections::BTreeMap::new();
        let mut inputs = 0;
        for (i, prepared) in prepared.into_iter().enumerate() {
            let target = i as i64 + 1;
            inputs += prepared.input_tokens;
            queue::enqueue(
                &store,
                &queue::Enqueue {
                    kind: if i == 0 {
                        queue::Kind::DescribeSymbol
                    } else {
                        queue::Kind::SummarizePr
                    },
                    target: if i == 0 {
                        queue::Target::Symbol
                    } else {
                        queue::Target::Pr
                    },
                    target_id: target,
                    target_hash: &[1; 32],
                    desired_hash: &[2; 32],
                    provider: &provider,
                    priority: 2.0 - i as f64,
                    tokens: prepared.input_tokens,
                },
                0,
            )
            .unwrap();
            let id = store
                .read_snapshot(|tx| {
                    tx.query_row("SELECT id FROM jobs WHERE target_id=?1", [target], |r| {
                        r.get::<_, i64>(0)
                    })
                })
                .unwrap()
                .unwrap();
            jobs.insert(
                id,
                Job {
                    client: Arc::clone(&client),
                    prepared: Arc::new(prepared),
                    config: config.clone(),
                    kind: "llm",
                    endpoint: endpoint.clone(),
                },
            );
        }
        let settings = Arc::new(queue::pool::Settings {
            provider: provider.clone(),
            concurrency: 1,
            max_jobs: 2,
            max_claim_tokens: inputs,
            budget: super::super::limits(&config).unwrap().budget,
            max_output_tokens: 777,
            timeout: Duration::from_secs(3),
            heartbeat: Duration::from_secs(1),
        });
        let report =
            queue::pool::run_ready(Arc::clone(&store), settings, Arc::new(Jobs(jobs)), || 0)
                .await
                .unwrap();
        assert_eq!(report.claimed, 2);
        assert_eq!(report.published, if both { 2 } else { 1 });
        assert_eq!(report.deferred, if both { 0 } else { 1 });
        assert!(report.errors.is_empty());
        let actual = store
            .read_snapshot(|tx| {
                tx.query_row(
                    "SELECT tokens,estimated_tokens FROM usage WHERE provider=?1",
                    [&provider],
                    |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
                )
            })
            .unwrap()
            .unwrap();
        assert_eq!(actual, (expected, expected));
        let calls = server.received_requests().await.unwrap();
        let ceilings = calls
            .iter()
            .map(|r| {
                serde_json::from_slice::<Value>(&r.body).unwrap()["max_tokens"]
                    .as_i64()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(ceilings, if both { vec![120, 300] } else { vec![120] });
    }
}

#[tokio::test]
async fn permanent_chat_diagnostics_keep_status_without_payloads_or_automatic_retries() {
    let marker = "synthetic provider body marker";
    for (status, body, reason, code, http_status, terminal) in [
        (400, marker.to_owned(), "chat provider HTTP 400", "chat_http_error", Some(400), true),
        (400, json!({"error":{"code":"context_length_exceeded","message":marker}}).to_string(), "chat provider HTTP 400; context_length_exceeded", "chat_http_error", Some(400), true),
        (413, marker.to_owned(), "chat provider HTTP 413", "chat_http_error", Some(413), true),
        (401, marker.to_owned(), "chat provider unauthorized", "chat_unauthorized", None, true),
        (200, marker.to_owned(), "invalid chat response", "chat_invalid_response", None, true),
        (200, marker.repeat(5_000), "chat response body limit", "chat_response_body_limit", None, true),
        (429, marker.to_owned(), "transient provider error", "retry", None, false),
        (503, marker.to_owned(), "transient provider error", "retry", None, false),
        (200, json!({"choices":[{"message":{"content":""}}],"usage":{"completion_tokens":6,"total_tokens":9}}).to_string(), "empty provider content; diagnosis required", "empty_content", None, true),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .expect(1)
            .mount(&server)
            .await;
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(store::open(directory.path().join("code.db"), StoreLimits::default()).unwrap());
        let config = crate::config::Llm {
            enabled: true,
            url: format!("{}/v1", server.uri()),
            model: "fixture-diagnostics".into(),
            ..Default::default()
        };
        let client = Arc::new(Client::with_key(&config, Duration::from_secs(3), 1024, "synthetic diagnostic key".into()).unwrap());
        let endpoint = client.endpoint.to_string();
        let provider = super::super::provider_id(&endpoint);
        let prepared = client.prepare_text("Summary.", "Public fixture source.", 120).unwrap();
        let lease = enqueue(&store, &provider, 1, prepared.input_tokens);
        let settings = worker::Settings {
            provider: &provider,
            budget: super::super::limits(&config).unwrap().budget,
            max_output_tokens: 120,
            timeout: Duration::from_secs(3),
            heartbeat: Duration::from_secs(1),
        };
        let job = Job { client, prepared: Arc::new(prepared), config, kind: "llm", endpoint };
        let result = worker::execute(&store, &lease, &settings, &job, || 0).await.unwrap();
        if terminal {
            assert_eq!(result, worker::Finished::Scheduled(queue::retry::Scheduled::Dead));
        } else {
            assert!(matches!(result, worker::Finished::Scheduled(queue::retry::Scheduled::RetryAt(_))));
        }
        let observed: (String, i64, String) = store.writer().unwrap().query_row(
            "SELECT state,attempts,last_error FROM jobs WHERE id=?1", [lease.id],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
        ).unwrap();
        assert_eq!(observed, (if terminal { "dead" } else { "failed" }.into(), 1, reason.into()));
        let snapshot = serde_json::to_value(queue::status::snapshot_with_jobs(&store, 0, Some(10)).unwrap()).unwrap();
        assert_eq!(snapshot["entries"][0]["reason"], code);
        assert_eq!(snapshot["entries"][0].get("http_status"), http_status.map(|s|json!(s)).as_ref());
        assert_eq!(snapshot["entries"][0].get("provider_code"),
            reason.ends_with("; context_length_exceeded").then(||json!("context_length_exceeded")).as_ref());
        let encoded = snapshot.to_string();
        assert!(!encoded.contains(marker));
        assert!(!encoded.contains("synthetic diagnostic key"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}
