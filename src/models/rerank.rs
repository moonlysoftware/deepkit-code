//! Reranker providers and score contracts; v1 §§7.1–7.2 and 9 step 7.
use std::time::Duration;

/// Safe reranking failures: no query, document, credentials, URL or response body.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("reranker smoke test failed; disabled model {file}")]
    Smoke { file: String },
    #[error("native reranker {operation} failed")]
    Native { operation: &'static str },
    #[error("invalid reranker configuration")]
    Configuration,
    #[error("invalid or oversized reranker input")]
    Input,
    #[error("reranker HTTP client initialization failed")]
    Client,
    #[error("reranker request timed out")]
    Timeout,
    #[error("reranker request transport failed")]
    Transport,
    #[error("reranker returned HTTP {status}")]
    Http {
        status: u16,
        retry_after: Option<Duration>,
    },
    #[error("invalid reranker scores or response; diagnosis required")]
    Response(ResponseReason),
}

/// Closed response-validation reasons; variants carry no provider-controlled data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResponseReason {
    BodyLimit,
    Schema,
    Count,
    Index,
    Score,
}
impl ResponseReason {
    /// Stable diagnostic code without raw response, parser, index or score values.
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::BodyLimit => "body_limit",
            Self::Schema => "schema",
            Self::Count => "count",
            Self::Index => "index",
            Self::Score => "score",
        }
    }
}

pub(crate) mod remote;

/// Shared three-pair relevance health gate; v1 §7.5.
pub(crate) mod smoke;

/// Last verified local health evidence for status; v1 §§7.5, 13.
pub(crate) mod health;
