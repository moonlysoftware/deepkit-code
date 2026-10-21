//! Shared HTTP embedding attempt errors and retry metadata; v1 §§7.2 and 8.2.
use std::time::Duration;

/// One provider attempt failure, retaining scheduling metadata without provider bodies,
/// request URLs, credentials or input text. The caller owns all subsequent retries.
#[derive(Debug, thiserror::Error)]
pub(crate) enum AttemptError {
    #[error(transparent)]
    Chat(#[from] crate::models::openai_compat::chat::Error),
    #[error(transparent)]
    Rerank(#[from] crate::models::rerank::Error),
    #[error(transparent)]
    LocalRerank(Box<crate::models::Error>),
    #[error("retryable embedding request failure (HTTP {status:?})")]
    Retryable {
        status: Option<u16>,
        retry_after: Option<Duration>,
    },
    #[error("embedding request rejected (HTTP {status})")]
    Rejected { status: u16 },
    #[error("provider rejected dimensions; next admitted attempt uses local truncation")]
    DimensionsRejected,
    #[error("embedding input does not fit one nonempty request")]
    InvalidInput,
    #[error("invalid embedding response; diagnosis required")]
    InvalidResponse,
}

impl AttemptError {
    /// Preserves billed usage from unusable responses without inferring missing counts.
    pub(crate) fn reported_tokens(&self) -> Option<i64> {
        match self {
            Self::Chat(error) => error.reported_tokens(),
            _ => None,
        }
    }
}

/// An immutable, provider-bound query input and its admission token estimate.
/// It intentionally has no Debug/Serialize implementation because queries are private.
pub(crate) struct PreparedQuery {
    text: String,
    tokens: i64,
    contract: [u8; 32],
}
impl PreparedQuery {
    /// Renders private query text once and refuses empty or oversized requests.
    pub(super) fn new(
        contract: [u8; 32],
        prefix: &str,
        query: &str,
        max_tokens: usize,
    ) -> Result<Self, AttemptError> {
        if query.trim().is_empty() {
            return Err(AttemptError::InvalidInput);
        }
        let text = format!("{prefix}{query}");
        let count = tiktoken_rs::cl100k_base_singleton()
            .encode_with_special_tokens(&text)
            .len();
        if count > max_tokens {
            return Err(AttemptError::InvalidInput);
        }
        let tokens = i64::try_from(count).map_err(|_| AttemptError::InvalidInput)?;
        Ok(Self {
            text,
            tokens,
            contract,
        })
    }
    /// Estimates the exact rendered query input for shared admission; providers'
    /// actual usage, when available, supersedes this estimate on reconciliation.
    pub(crate) fn tokens(&self) -> i64 {
        self.tokens
    }
    /// Borrows the admitted bytes only for the provider contract that prepared them.
    pub(super) fn input(&self, contract: [u8; 32]) -> Result<&str, AttemptError> {
        if self.contract != contract {
            return Err(AttemptError::InvalidInput);
        }
        Ok(&self.text)
    }
}
/// Separates query preparations across endpoint/model/instruction/dimension changes.
pub(super) fn query_contract(endpoint: &str, model: &str, prefix: &str, dims: usize) -> [u8; 32] {
    let mut hash = blake3::Hasher::new_derive_key("dkc query preparation v1");
    for part in [
        endpoint.as_bytes(),
        model.as_bytes(),
        prefix.as_bytes(),
        &(dims as u64).to_le_bytes(),
    ] {
        hash.update(&(part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    *hash.finalize().as_bytes()
}

/// One admitted embedding response with provider usage when reported. Missing
/// usage stays None so queue accounting labels its reserved estimate honestly.
#[derive(Debug)]
pub(crate) struct EmbeddingBatch {
    pub(crate) embeddings: Vec<Vec<f32>>,
    pub(crate) tokens: Option<i64>,
}

/// Converts reported counts into SQLite's range without accepting negative,
/// fractional or overflowing usage as a successful zero-cost response.
pub(crate) fn token_count(tokens: Option<u64>) -> Result<Option<i64>, AttemptError> {
    tokens
        .map(|tokens| i64::try_from(tokens).map_err(|_| AttemptError::InvalidResponse))
        .transpose()
}

/// Parses Retry-After seconds or HTTP dates without retaining provider header text.
pub(crate) fn parse_retry_after(value: Option<&reqwest::header::HeaderValue>) -> Option<Duration> {
    parse_retry_after_at(value, std::time::SystemTime::now())
}

fn parse_retry_after_at(
    value: Option<&reqwest::header::HeaderValue>,
    now: std::time::SystemTime,
) -> Option<Duration> {
    let value = value?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value)
        .ok()
        .map(|date| date.duration_since(now).unwrap_or(Duration::ZERO))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    #[test]
    fn retry_after_accepts_seconds_and_http_dates_without_early_retries() {
        use reqwest::header::HeaderValue;
        let epoch = std::time::SystemTime::UNIX_EPOCH;
        for (value, expected) in [
            ("120", Some(120)),
            ("Thu, 01 Jan 1970 00:02:00 GMT", Some(120)),
            ("invalid", None),
        ] {
            assert_eq!(
                super::parse_retry_after_at(Some(&HeaderValue::from_str(value).unwrap()), epoch),
                expected.map(Duration::from_secs)
            );
        }
        let past = HeaderValue::from_static("Thu, 01 Jan 1970 00:02:00 GMT");
        assert_eq!(
            super::parse_retry_after_at(Some(&past), epoch + Duration::from_secs(180)),
            Some(Duration::ZERO)
        );
    }
}
