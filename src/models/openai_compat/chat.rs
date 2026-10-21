//! Shared text/image completion clients and safe usage errors; v1 §§7–8.
mod transport;
mod wire;
pub(crate) use transport::{Client, Prepared};
pub(crate) use wire::{Error, HttpReason};

/// Normalizes a credential-free endpoint for stored generation identity checks.
pub(crate) fn endpoint_identity(base: &str) -> Result<String, Error> {
    Ok(transport::endpoint(base)?.to_string())
}

/// Builds the shared text wire shape for dispatch and credential-free proof checks.
pub(super) fn text_body(
    model: &str,
    extra: &serde_json::Value,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Result<serde_json::Value, Error> {
    if user.trim().is_empty() {
        return Err(Error::Input);
    }
    wire::request(
        model,
        serde_json::json!([{"role":"system","content":system},{"role":"user","content":user}]),
        max_tokens,
        extra,
    )
}

/// Verifies stored text generations without loading credentials or creating HTTP clients.
pub(crate) fn text_fingerprint(
    config: &crate::config::Llm,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Result<[u8; 32], Error> {
    let extra = serde_json::to_value(&config.extra_body).map_err(|_| Error::Input)?;
    let body = text_body(&config.model, &extra, system, user, max_tokens)?;
    Ok(*blake3::hash(&serde_json::to_vec(&body).map_err(|_| Error::Input)?).as_bytes())
}

/// Verifies image generations using the same format/cap checks and exact body as
/// dispatch, without resolving credentials or creating a client.
pub(crate) fn image_fingerprint(
    config: &crate::config::Llm,
    image_bytes: u64,
    image: &[u8],
    mime: &str,
    instruction: &str,
    max_tokens: u32,
) -> Result<[u8; 32], Error> {
    let extra = serde_json::to_value(&config.extra_body).map_err(|_| Error::Input)?;
    let (body, _) = transport::image_body(
        &config.vlm_model,
        &extra,
        image_bytes,
        image,
        mime,
        instruction,
        max_tokens,
    )?;
    Ok(*blake3::hash(&serde_json::to_vec(&body).map_err(|_| Error::Input)?).as_bytes())
}

/// Shares one configured chat quota across text/image models at a normalized endpoint.
pub(crate) fn provider_id(endpoint: &str) -> String {
    format!("chat:remote:{}", blake3::hash(endpoint.as_bytes()).to_hex())
}

/// Converts configured daily/minute caps without overflow or treating zero as unlimited.
pub(crate) fn limits(
    config: &crate::config::Llm,
) -> Result<crate::queue::usage::Limits, crate::queue::Error> {
    let count = |value| {
        i64::try_from(value)
            .map_err(|_| crate::queue::Error::Invalid("chat quota exceeds supported range"))
    };
    Ok(crate::queue::usage::Limits {
        budget: crate::queue::usage::Budget {
            calls: Some(count(config.max_calls_per_day)?),
            tokens: Some(count(config.max_tokens_per_day)?),
        },
        rpm: Some(count(config.rpm)?),
        tpm: Some(count(config.tpm)?),
    })
}
