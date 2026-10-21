//! Shared bounded chat wire preparation and decoding; v1 §§7.1, 7.2 and 8.2.
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;

/// A single failed chat attempt; never carries prompts, credentials or provider text.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("invalid chat configuration or request")]
    Input,
    #[error("chat provider rejected the API key")]
    Unauthorized,
    #[error("chat provider returned HTTP {status}")]
    Http {
        status: u16,
        retry_after: Option<Duration>,
        reason: Option<HttpReason>,
    },
    #[error("chat request transport failed")]
    Transport,
    #[error("chat response exceeded the bounded response size")]
    BodyLimit,
    #[error("invalid chat response; diagnosis required")]
    Response { tokens: Option<i64> },
    #[error(
        "chat returned empty content after consuming completion tokens; run dkc doctor and check llm.extra_body.chat_template_kwargs.enable_thinking=false"
    )]
    ThinkingConsumedBudget { tokens: Option<i64> },
    #[error("chat returned empty content; diagnosis required")]
    EmptyContent { tokens: Option<i64> },
    #[error(
        "page transcription is incomplete; inspect the output ceiling and provider finish reason"
    )]
    IncompleteContent { tokens: Option<i64> },
}
impl Error {
    /// Returns observed total usage even when the assistant content is unusable.
    pub(crate) fn reported_tokens(&self) -> Option<i64> {
        match self {
            Self::Response { tokens }
            | Self::ThinkingConsumedBudget { tokens }
            | Self::EmptyContent { tokens }
            | Self::IncompleteContent { tokens } => *tokens,
            _ => None,
        }
    }
}

/// Recognized provider codes only; arbitrary error messages never cross the wire edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HttpReason {
    ContextLengthExceeded,
}
impl HttpReason {
    /// Stable source-free spelling for durable diagnostics.
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::ContextLengthExceeded => "context_length_exceeded",
        }
    }
    /// Admits only an exact known code, never a provider's free-form message.
    pub(crate) fn from_code(code: &str) -> Option<Self> {
        match code {
            "context_length_exceeded" => Some(Self::ContextLengthExceeded),
            _ => None,
        }
    }
}

/// Reads structured error codes only; malformed or unrecognized bodies stay unknown.
pub(super) fn http_reason(bytes: &[u8]) -> Option<HttpReason> {
    let value: Value = serde_json::from_slice(bytes).ok()?;
    let error = value.get("error")?.as_object()?;
    ["code", "type"].into_iter().find_map(|key| {
        error
            .get(key)
            .and_then(Value::as_str)
            .and_then(HttpReason::from_code)
    })
}

/// Successful text and optional observed usage; absent usage remains an estimate.
pub(crate) struct Reply {
    pub(crate) completion: crate::models::Completion,
    pub(crate) tokens: Option<i64>,
    complete: bool,
}
impl Reply {
    /// Page publication requires an explicit completed response, while preserving
    /// observed usage when a nonempty transcription was cut short or is unproven.
    pub(crate) fn require_complete(&self) -> Result<(), Error> {
        if self.complete {
            Ok(())
        } else {
            Err(Error::IncompleteContent {
                tokens: self.tokens,
            })
        }
    }
}

/// Recursively merges option objects; scalar/array leaves replace their base value.
fn merge(base: &mut Value, extra: &Value) {
    match (base, extra) {
        (Value::Object(base), Value::Object(extra)) => {
            for (key, value) in extra {
                merge(base.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        (base, extra) => *base = extra.clone(),
    }
}

/// Builds one non-streaming completion request bound to its admitted identity.
/// Extra options cannot replace model, source messages or the reserved ceiling.
pub(crate) fn request(
    model: &str,
    messages: Value,
    max_tokens: u32,
    extra: &Value,
) -> Result<Value, Error> {
    let options = extra.as_object().ok_or(Error::Input)?;
    if model.trim().is_empty()
        || max_tokens == 0
        || messages.as_array().is_none_or(Vec::is_empty)
        || options.keys().any(|key| {
            matches!(
                key.as_str(),
                "model" | "messages" | "max_tokens" | "max_completion_tokens" | "stream" | "n"
            )
        })
    {
        return Err(Error::Input);
    }
    let mut body =
        json!({"model":model,"messages":messages,"max_tokens":max_tokens,"stream":false,"n":1});
    merge(&mut body, extra);
    Ok(body)
}

#[derive(Deserialize)]
struct Wire {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}
#[derive(Deserialize)]
struct Choice {
    message: Message,
    // Unknown/malformed reasons refuse page completion without losing valid usage.
    finish_reason: Option<Value>,
}
#[derive(Deserialize)]
struct Message {
    content: Option<String>,
}
#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
}
impl Usage {
    fn total(&self) -> Result<Option<i64>, Error> {
        let sum = self
            .prompt_tokens
            .zip(self.completion_tokens)
            .map(|(a, b)| a.checked_add(b).ok_or(Error::Response { tokens: None }))
            .transpose()?;
        if self
            .total_tokens
            .zip(sum)
            .is_some_and(|(total, sum)| total < sum)
        {
            return Err(Error::Response { tokens: None });
        }
        self.total_tokens
            .or(sum)
            .map(|total| i64::try_from(total).map_err(|_| Error::Response { tokens: None }))
            .transpose()
    }
}

/// Decodes a bounded response, retaining token accounting for empty assistant text.
pub(crate) fn decode(bytes: &[u8]) -> Result<Reply, Error> {
    let wire: Wire = serde_json::from_slice(bytes).map_err(|_| Error::Response { tokens: None })?;
    let tokens = wire.usage.as_ref().map(Usage::total).transpose()?.flatten();
    if wire.choices.len() != 1 {
        return Err(Error::Response { tokens });
    }
    let choice = wire
        .choices
        .into_iter()
        .next()
        .ok_or(Error::Response { tokens })?;
    let complete = choice.finish_reason.as_ref().and_then(Value::as_str) == Some("stop");
    let content = choice.message.content.unwrap_or_default();
    if content.trim().is_empty() {
        return Err(
            if wire
                .usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens)
                .is_some_and(|n| n > 0)
            {
                Error::ThinkingConsumedBudget { tokens }
            } else {
                Error::EmptyContent { tokens }
            },
        );
    }
    Ok(Reply {
        completion: crate::models::Completion { content },
        tokens,
        complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deep_merge_preserves_nested_defaults_and_changes_only_requested_leaves() {
        let mut base =
            json!({"chat_template_kwargs":{"enable_thinking":true,"retained":2},"temperature":0.5});
        merge(
            &mut base,
            &json!({"chat_template_kwargs":{"enable_thinking":false},"temperature":0.1}),
        );
        assert_eq!(
            base,
            json!({"chat_template_kwargs":{"enable_thinking":false,"retained":2},"temperature":0.1})
        );
    }
    #[test]
    fn empty_completion_retains_actual_usage_without_becoming_success() {
        let body=serde_json::to_vec(&json!({"choices":[{"message":{"content":""}}],"usage":{"prompt_tokens":10,"completion_tokens":120,"total_tokens":130}})).unwrap();
        let error = match decode(&body) {
            Err(error) => error,
            Ok(_) => panic!("empty content accepted"),
        };
        assert!(matches!(error, Error::ThinkingConsumedBudget { .. }));
        assert_eq!(error.reported_tokens(), Some(130));
        assert!(error.to_string().contains("enable_thinking=false"));
    }
    #[test]
    fn valid_reply_preserves_content_and_missing_usage_remains_unknown() {
        let reply =
            decode(br#"{"choices":[{"message":{"content":"A useful summary."}}]}"#).unwrap();
        assert_eq!(reply.completion.content, "A useful summary.");
        assert_eq!(reply.tokens, None);
        let error = decode(br#"{"choices":[{"message":{"content":null}}]}"#)
            .err()
            .unwrap();
        assert!(matches!(error, Error::EmptyContent { tokens: None }));
    }
    #[test]
    fn page_completion_requires_explicit_stop_and_preserves_usage_on_refusal() {
        for reason in [
            json!("stop"),
            json!("length"),
            json!("content_filter"),
            json!("tool_calls"),
            Value::Null,
            json!(7),
        ] {
            let bytes = serde_json::to_vec(&json!({
                "choices":[{"message":{"content":"# Partial or complete page"},"finish_reason":reason}],
                "usage":{"prompt_tokens":11,"completion_tokens":23,"total_tokens":34}
            })).unwrap();
            let reply = decode(&bytes).unwrap();
            assert_eq!(reply.completion.content, "# Partial or complete page");
            if reason == "stop" {
                reply.require_complete().unwrap();
            } else {
                let error = reply.require_complete().unwrap_err();
                assert!(matches!(
                    error,
                    Error::IncompleteContent { tokens: Some(34) }
                ));
                assert_eq!(error.reported_tokens(), Some(34));
                assert!(!error.to_string().contains("# Partial"));
            }
        }
        let missing =
            decode(br#"{"choices":[{"message":{"content":"Existing summary compatibility"}}]}"#)
                .unwrap();
        assert_eq!(missing.completion.content, "Existing summary compatibility");
        assert!(matches!(
            missing.require_complete(),
            Err(Error::IncompleteContent { tokens: None })
        ));
    }

    #[test]
    fn extra_body_cannot_change_admitted_identity_or_output_ceiling() {
        for key in [
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "n",
        ] {
            let mut extra = json!({});
            extra[key] = json!("override");
            assert!(
                request(
                    "fixture",
                    json!([{"role":"user","content":"Prompt."}]),
                    120,
                    &extra
                )
                .is_err()
            );
        }
        let body = request(
            "fixture",
            json!([{"role":"user","content":"Summarize."}]),
            120,
            &json!({"chat_template_kwargs":{"enable_thinking":false}}),
        )
        .unwrap();
        assert_eq!(body["max_tokens"], 120);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    }
}
