//! One-attempt Cohere-style remote reranking; v1 §7.2 and T1.8.
use super::{Error, ResponseReason};
use crate::{
    config,
    models::{self, Reranker, attempt::parse_retry_after, openai_compat::resolve_api_key},
};
use async_trait::async_trait;
use reqwest::{
    Client, Url,
    header::{AUTHORIZATION, HeaderValue, RETRY_AFTER},
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

type Result<T> = std::result::Result<T, Error>;

/// A reusable remote scorer. The client/authorization and private inputs have no
/// Debug representation; the hosting layer owns policy checks and retry admission.
pub(crate) struct Remote {
    client: Client,
    endpoint: Url,
    authorization: HeaderValue,
    model: String,
    id: String,
    max_docs: usize,
    max_doc_chars: usize,
}
fn endpoint(base: &str) -> Result<Url> {
    let base = Url::parse(base).map_err(|_| Error::Configuration)?;
    if !matches!(base.scheme(), "http" | "https")
        || base.host_str().is_none()
        || !base.username().is_empty()
        || base.password().is_some()
        || base.query().is_some()
        || base.fragment().is_some()
    {
        return Err(Error::Configuration);
    }
    Url::parse(&format!("{}/rerank", base.as_str().trim_end_matches('/')))
        .map_err(|_| Error::Configuration)
}
fn validate(config: &config::Rerank, timeout: Duration) -> Result<(usize, usize)> {
    let chars = usize::try_from(config.max_doc_chars).map_err(|_| Error::Configuration)?;
    let docs = usize::try_from(config.candidates).map_err(|_| Error::Configuration)?;
    if chars == 0 || docs == 0 || timeout.is_zero() || config.remote.model.trim().is_empty() {
        return Err(Error::Configuration);
    }
    Ok((chars, docs))
}
impl Remote {
    /// Resolves credentials using the same environment/0600-file policy as embeddings.
    pub(crate) fn from_config(config: &config::Rerank, timeout: Duration) -> models::Result<Self> {
        validate(config, timeout)?;
        let endpoint = endpoint(&config.remote.url)?;
        let key = resolve_api_key(
            &config.remote.api_key_env,
            &endpoint,
            config::credentials_path().as_deref(),
        )?;
        Ok(Self::with_api_key(config, timeout, key)?)
    }
    /// Accepts an injected credential for tests or an already resolved credential source.
    pub(crate) fn with_api_key(
        config: &config::Rerank,
        timeout: Duration,
        key: String,
    ) -> Result<Self> {
        let (max_doc_chars, max_docs) = validate(config, timeout)?;
        let endpoint = endpoint(&config.remote.url)?;
        if key.trim().is_empty() {
            return Err(Error::Configuration);
        }
        let mut authorization =
            HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| Error::Configuration)?;
        authorization.set_sensitive(true);
        let client = Client::builder()
            .use_rustls_tls()
            .user_agent(concat!("dkc/", env!("CARGO_PKG_VERSION")))
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .timeout(timeout)
            .build()
            .map_err(|_| Error::Client)?;
        Ok(Self {
            client,
            endpoint,
            authorization,
            model: config.remote.model.clone(),
            id: format!("remote/{}", config.remote.model),
            max_docs,
            max_doc_chars,
        })
    }
    /// Makes at most one request and returns scores in original candidate order.
    /// Empty sets make no request; oversized sets refuse instead of silently splitting.
    /// Unicode scalar boundaries enforce max_doc_chars without corrupting UTF-8.
    pub(crate) async fn score_once(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() || docs.len() > self.max_docs {
            return Err(Error::Input);
        }
        let documents: Vec<_> = docs
            .iter()
            .map(|doc| {
                let end = doc
                    .char_indices()
                    .nth(self.max_doc_chars)
                    .map_or(doc.len(), |(i, _)| i);
                &doc[..end]
            })
            .collect();
        let cap = response_cap(&documents)?;
        let request = Request {
            model: &self.model,
            query,
            documents: &documents,
            top_n: documents.len(),
            return_documents: false,
        };
        let mut response = self
            .client
            .post(self.endpoint.clone())
            .header(AUTHORIZATION, self.authorization.clone())
            .json(&request)
            .send()
            .await
            .map_err(transport)?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Http {
                status: status.as_u16(),
                retry_after: parse_retry_after(response.headers().get(RETRY_AFTER)),
            });
        }
        // Bound both declared and streamed response bytes, including metadata.
        if response.content_length().is_some_and(|n| n > cap as u64) {
            return Err(Error::Response(ResponseReason::BodyLimit));
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(transport)? {
            if chunk.len() > cap.saturating_sub(body.len()) {
                return Err(Error::Response(ResponseReason::BodyLimit));
            }
            body.extend_from_slice(&chunk);
        }
        decode(&body, docs.len())
    }
}
/// Bounds scores/metadata plus one compact UTF-8 JSON-string echo per document.
/// Some compatible servers echo documents despite return_documents=false. Use the
/// already truncated wire inputs, never original unbounded document lengths.
/// Alternate escaping or excessive metadata can still exceed this finite bound.
fn response_cap(documents: &[&str]) -> Result<usize> {
    let base = documents
        .len()
        .checked_mul(256)
        .and_then(|n| n.checked_add(16 * 1024))
        .ok_or(Error::Input)?;
    documents.iter().try_fold(base, |cap, document| {
        let encoded = serde_json::to_string(document).map_err(|_| Error::Input)?;
        cap.checked_add(encoded.len()).ok_or(Error::Input)
    })
}
fn transport(error: reqwest::Error) -> Error {
    if error.is_timeout() {
        Error::Timeout
    } else {
        Error::Transport
    }
}
#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [&'a str],
    top_n: usize,
    return_documents: bool,
}
#[derive(Deserialize)]
struct Response {
    results: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    index: usize,
    relevance_score: f64,
}
fn decode(body: &[u8], expected: usize) -> Result<Vec<f32>> {
    let response: Response =
        serde_json::from_slice(body).map_err(|_| Error::Response(ResponseReason::Schema))?;
    if response.results.len() != expected {
        return Err(Error::Response(ResponseReason::Count));
    }
    let mut scores = vec![None; expected];
    for row in response.results {
        if !row.relevance_score.is_finite() || !(0.0..=1.0).contains(&row.relevance_score) {
            return Err(Error::Response(ResponseReason::Score));
        }
        let score = scores
            .get_mut(row.index)
            .ok_or(Error::Response(ResponseReason::Index))?;
        if score.is_some() {
            return Err(Error::Response(ResponseReason::Index));
        }
        *score = Some(row.relevance_score as f32);
    }
    scores
        .into_iter()
        .map(|s| s.ok_or(Error::Response(ResponseReason::Index)))
        .collect()
}
#[async_trait]
impl Reranker for Remote {
    fn id(&self) -> &str {
        &self.id
    }
    fn max_doc_chars(&self) -> usize {
        self.max_doc_chars
    }
    async fn score(&self, query: &str, docs: &[&str]) -> models::Result<Vec<f32>> {
        Ok(self.score_once(query, docs).await?)
    }
}

#[cfg(test)]
mod tests;
