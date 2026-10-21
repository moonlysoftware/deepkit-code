//! Single-attempt text/image transport; v1 §§7.1–7.2 and 8.2.
use super::wire::{self, Error, Reply};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::{
    Client as Http, Url,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue, RETRY_AFTER},
};
use serde_json::{Value, json};
use std::{io::Cursor, sync::Arc, time::Duration};
use tokio::sync::Semaphore;

/// Private immutable wire bytes and the estimates reserved before dispatch.
pub(crate) struct Prepared {
    body: Vec<u8>,
    contract: [u8; 32],
    pub(crate) input_tokens: i64,
    pub(crate) max_output_tokens: i64,
    pub(crate) fingerprint: [u8; 32],
}
/// Text and image jobs share one endpoint, configuration and process capacity.
pub(crate) struct Client {
    http: Http,
    contract: [u8; 32],
    endpoint: Url,
    authorization: HeaderValue,
    model: String,
    vlm_model: String,
    extra: Value,
    image_bytes: u64,
    slots: Arc<Semaphore>,
}
impl Client {
    /// Normalized credential-free destination used by durable provider accounting.
    pub(crate) fn endpoint(&self) -> &str {
        self.endpoint.as_str()
    }

    /// Resolves secrets through the existing environment/0600-file policy.
    pub(crate) fn from_config(
        config: &crate::config::Llm,
        timeout: Duration,
        image_bytes: u64,
    ) -> crate::models::Result<Self> {
        let endpoint = endpoint(&config.url)?;
        let key = crate::models::openai_compat::resolve_api_key(
            &config.api_key_env,
            &endpoint,
            crate::config::credentials_path().as_deref(),
        )?;
        Self::with_key(config, timeout, image_bytes, key).map_err(Into::into)
    }
    /// Injects a resolved key for fixed fixtures without consulting user credentials.
    pub(crate) fn with_key(
        config: &crate::config::Llm,
        timeout: Duration,
        image_bytes: u64,
        key: String,
    ) -> Result<Self, Error> {
        let concurrency = usize::try_from(config.concurrency).map_err(|_| Error::Input)?;
        if !config.enabled
            || timeout.is_zero()
            || concurrency == 0
            || concurrency > Semaphore::MAX_PERMITS
            || key.trim().is_empty()
        {
            return Err(Error::Input);
        }
        let mut authorization =
            HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| Error::Input)?;
        authorization.set_sensitive(true);
        let http = Http::builder()
            .use_rustls_tls()
            .user_agent(concat!("dkc/", env!("CARGO_PKG_VERSION")))
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .timeout(timeout)
            .build()
            .map_err(|_| Error::Input)?;
        let contract = *blake3::hash(
            &serde_json::to_vec(&(config, timeout.as_millis(), image_bytes))
                .map_err(|_| Error::Input)?,
        )
        .as_bytes();
        Ok(Self {
            http,
            contract,
            endpoint: endpoint(&config.url)?,
            authorization,
            model: config.model.clone(),
            vlm_model: config.vlm_model.clone(),
            extra: serde_json::to_value(&config.extra_body).map_err(|_| Error::Input)?,
            image_bytes,
            slots: Arc::new(Semaphore::new(concurrency)),
        })
    }
    /// Renders text once so dispatch and accounting use the same immutable request.
    pub(crate) fn prepare_text(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<Prepared, Error> {
        let body = super::text_body(&self.model, &self.extra, system, user, max_tokens)?;
        prepared(body, None, max_tokens, self.contract)
    }
    /// Admits bytes and inspects dimensions before allocating an image data URI.
    /// Patch-based image tokens are explicit estimates, never reported actual usage.
    pub(crate) fn prepare_image(
        &self,
        image: &[u8],
        mime: &str,
        instruction: &str,
        max_tokens: u32,
    ) -> Result<Prepared, Error> {
        let (body, image_tokens) = image_body(
            &self.vlm_model,
            &self.extra,
            self.image_bytes,
            image,
            mime,
            instruction,
            max_tokens,
        )?;
        prepared(body, Some(image_tokens), max_tokens, self.contract)
    }
    /// Sends one already-admitted request; callers own durable quotas and retries.
    pub(crate) async fn post_once(&self, prepared: &Prepared) -> Result<Reply, Error> {
        if prepared.contract != self.contract {
            return Err(Error::Input);
        }
        let _slot = self.slots.acquire().await.map_err(|_| Error::Input)?;
        let response = self
            .http
            .post(self.endpoint.clone())
            .header(AUTHORIZATION, self.authorization.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(prepared.body.clone())
            .send()
            .await
            .map_err(|_| Error::Transport)?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(Error::Unauthorized);
        }
        if !status.is_success() {
            let retry_after =
                crate::models::attempt::parse_retry_after(response.headers().get(RETRY_AFTER));
            // The existing request deadline bounds body reads too. Failure to read
            // diagnostics must not change the original HTTP retry classification.
            let reason = crate::models::http_body::read(response, 16 * 1024)
                .await
                .ok()
                .and_then(|bytes| wire::http_reason(&bytes));
            return Err(Error::Http {
                status: status.as_u16(),
                retry_after,
                reason,
            });
        }
        let cap = usize::try_from(prepared.max_output_tokens)
            .ok()
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| n.checked_add(64 * 1024))
            .ok_or(Error::Input)?
            .min(8 * 1024 * 1024);
        let bytes = crate::models::http_body::read(response, cap)
            .await
            .map_err(|error| match error {
                crate::models::http_body::Error::TooLarge => Error::BodyLimit,
                crate::models::http_body::Error::Transport => Error::Transport,
            })?;
        wire::decode(&bytes)
    }
}
/// Shares admitted image bytes and wire identity with credential-free freshness checks.
pub(super) fn image_body(
    model: &str,
    extra: &Value,
    image_bytes: u64,
    image: &[u8],
    mime: &str,
    instruction: &str,
    max_tokens: u32,
) -> Result<(Value, i64), Error> {
    if image.is_empty() || image.len() as u64 > image_bytes || instruction.trim().is_empty() {
        return Err(Error::Input);
    }
    let reader = image::ImageReader::new(Cursor::new(image))
        .with_guessed_format()
        .map_err(|_| Error::Input)?;
    let expected = match mime {
        "image/png" => image::ImageFormat::Png,
        "image/jpeg" => image::ImageFormat::Jpeg,
        "image/webp" => image::ImageFormat::WebP,
        "image/gif" => image::ImageFormat::Gif,
        _ => return Err(Error::Input),
    };
    if reader.format() != Some(expected) {
        return Err(Error::Input);
    }
    let (width, height) = reader.into_dimensions().map_err(|_| Error::Input)?;
    if width == 0 || height == 0 {
        return Err(Error::Input);
    }
    let image_tokens = u64::from(width)
        .div_ceil(16)
        .checked_mul(u64::from(height).div_ceil(16))
        .and_then(|n| n.checked_add(85))
        .and_then(|n| i64::try_from(n).ok())
        .ok_or(Error::Input)?;
    let messages = json!([{"role":"user","content":[{"type":"text","text":instruction},{"type":"image_url","image_url":{"url":format!("data:{mime};base64,{}",STANDARD.encode(image))}}]}]);
    let body = wire::request(model, messages, max_tokens, extra)?;
    Ok((body, image_tokens))
}
pub(super) fn endpoint(base: &str) -> Result<Url, Error> {
    let value = crate::config::queue::embedding_endpoint(base, "chat/completions")
        .map_err(|_| Error::Input)?;
    Url::parse(&value).map_err(|_| Error::Input)
}
fn prepared(
    body: Value,
    image_tokens: Option<i64>,
    max_tokens: u32,
    contract: [u8; 32],
) -> Result<Prepared, Error> {
    let mut estimate = body.clone();
    if image_tokens.is_some() {
        estimate["messages"][0]["content"][1]["image_url"]["url"] = json!("<image>");
    }
    let estimate = serde_json::to_string(&estimate).map_err(|_| Error::Input)?;
    let tokens = i64::try_from(
        tiktoken_rs::cl100k_base_singleton()
            .encode_with_special_tokens(&estimate)
            .len(),
    )
    .map_err(|_| Error::Input)?;
    let input_tokens = tokens
        .checked_add(image_tokens.unwrap_or(0))
        .ok_or(Error::Input)?;
    let body = serde_json::to_vec(&body).map_err(|_| Error::Input)?;
    Ok(Prepared {
        contract,
        fingerprint: *blake3::hash(&body).as_bytes(),
        body,
        input_tokens,
        max_output_tokens: i64::from(max_tokens),
    })
}
#[async_trait]
impl crate::models::Llm for Client {
    async fn complete(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> crate::models::Result<crate::models::Completion> {
        let prepared = self.prepare_text(system, user, max_tokens)?;
        Ok(self.post_once(&prepared).await?.completion)
    }
}
#[async_trait]
impl crate::models::Vlm for Client {
    async fn describe(
        &self,
        image: &[u8],
        mime: &str,
        instruction: &str,
    ) -> crate::models::Result<String> {
        let prepared = self.prepare_image(image, mime, instruction, 200)?;
        Ok(self.post_once(&prepared).await?.completion.content)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "queue_tests.rs"]
mod queue_tests;
