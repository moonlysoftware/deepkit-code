//! Shared bounded provider model discovery; v1 §§7.2, 10.2, 13.
use super::http_body;
use reqwest::{Client, Url};
use serde::Deserialize;
use std::{collections::BTreeSet, time::Duration};

/// Closed diagnostics never retain endpoint, credentials, model IDs or response text.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("Invalid model discovery endpoint")]
    Endpoint,
    #[error("Could not initialize model discovery")]
    Client,
    #[error("Model discovery could not reach the configured provider")]
    Transport,
    #[error("The provider rejected the API key (401); check the configured credential")]
    Unauthorized,
    #[error("Model discovery returned HTTP {0}")]
    Http(u16),
    #[error("Model discovery response exceeds 1 MiB")]
    TooLarge,
    #[error("Model discovery response is not a valid model list")]
    Malformed,
    #[error("A configured model is missing from provider discovery")]
    Missing,
}

#[derive(Deserialize)]
struct Listing {
    data: Vec<Model>,
}
#[derive(Deserialize)]
struct Model {
    id: String,
}

/// One reusable no-retry, no-redirect client. Dropping an async operation cancels
/// its request; the total request/body timeout remains fifteen seconds.
pub(crate) struct Discovery {
    client: Client,
}
impl Discovery {
    /// Builds bounded transport without accessing a provider or reading credentials.
    pub(crate) fn new() -> Result<Self, Error> {
        let client = Client::builder()
            .user_agent(concat!("dkc/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| Error::Client)?;
        Ok(Self { client })
    }

    /// Retrieves an OpenAI-compatible list from a pre-resolved /models endpoint.
    /// Optional credentials stay in the request, never errors. Listing IDs are
    /// provider data for membership checks and must not be printed as diagnostics.
    pub(crate) async fn list(
        &self,
        endpoint: &Url,
        key: Option<&str>,
    ) -> Result<BTreeSet<String>, Error> {
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(Error::Endpoint);
        }
        let mut request = self.client.get(endpoint.clone());
        if let Some(key) = key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|_| Error::Transport)?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(Error::Unauthorized);
        }
        if !response.status().is_success() {
            return Err(Error::Http(response.status().as_u16()));
        }
        let body = http_body::read(response, http_body::LISTING_BYTES)
            .await
            .map_err(|error| match error {
                http_body::Error::TooLarge => Error::TooLarge,
                http_body::Error::Transport => Error::Transport,
            })?;
        let listing: Listing = serde_json::from_slice(&body).map_err(|_| Error::Malformed)?;
        Ok(listing.data.into_iter().map(|model| model.id).collect())
    }

    /// Verifies all configured models with one listing request and no body/model
    /// identifiers in failures; does not probe inference or claim model health.
    pub(crate) async fn verify(
        &self,
        endpoint: &Url,
        key: Option<&str>,
        expected: &BTreeSet<String>,
    ) -> Result<(), Error> {
        let available = self.list(endpoint, key).await?;
        if !expected.is_subset(&available) {
            return Err(Error::Missing);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    #[tokio::test]
    async fn listing_uses_user_agent_and_auth_and_never_retries_or_redirects() {
        for status in [200, 401, 503, 302] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/models"))
                .and(header(
                    "user-agent",
                    concat!("dkc/", env!("CARGO_PKG_VERSION")),
                ))
                .and(header("authorization", "Bearer fixture"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .set_body_json(serde_json::json!({"data":[{"id":"embed"},{"id":"rank"}]}))
                        .insert_header("location", format!("{}/must-not-follow", server.uri())),
                )
                .expect(1)
                .mount(&server)
                .await;
            let endpoint = Url::parse(&format!("{}/v1/models", server.uri())).unwrap();
            let result = Discovery::new()
                .unwrap()
                .verify(
                    &endpoint,
                    Some("fixture"),
                    &BTreeSet::from(["embed".into(), "rank".into()]),
                )
                .await;
            match status {
                200 => assert!(result.is_ok()),
                401 => assert!(matches!(result, Err(Error::Unauthorized))),
                _ => assert!(matches!(result, Err(Error::Http(code)) if code == status)),
            }
            assert_eq!(server.received_requests().await.unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn missing_and_malformed_lists_have_safe_errors() {
        let sentinel = "synthetic-private-model-value";
        for (body, missing) in [
            ("{not-json synthetic-private-model-value", false),
            ("{\"data\":[]}", true),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(200).set_body_string(body))
                .expect(1)
                .mount(&server)
                .await;
            let error = Discovery::new()
                .unwrap()
                .verify(
                    &Url::parse(&server.uri()).unwrap(),
                    None,
                    &BTreeSet::from([sentinel.into()]),
                )
                .await
                .unwrap_err();
            assert!(matches!(
                (&error, missing),
                (Error::Missing, true) | (Error::Malformed, false)
            ));
            assert!(!format!("{error} {error:?}").contains(sentinel));
        }
    }

    #[tokio::test]
    async fn advertised_and_streamed_listing_overflow_are_rejected() {
        let cap = http_body::LISTING_BYTES;
        let oversized = vec![b'x'; cap + 1];
        let mut chunked = format!("{:x}\r\n", oversized.len()).into_bytes();
        chunked.extend_from_slice(&oversized);
        chunked.extend_from_slice(b"\r\n0\r\n\r\n");
        for (headers, body) in [
            (format!("Content-Length: {}\r\n", cap + 1), Vec::new()),
            ("Transfer-Encoding: chunked\r\n".into(), chunked),
        ] {
            let (url, task) = http_body::tests::serve_once(200, headers, body).await;
            let result = Discovery::new()
                .unwrap()
                .list(&Url::parse(&url).unwrap(), None)
                .await;
            assert!(matches!(result, Err(Error::TooLarge)));
            task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn credential_bearing_endpoint_is_rejected_before_dispatch() {
        let server = MockServer::start().await;
        let endpoint = Url::parse(&format!(
            "{}/models?token=synthetic-private-value",
            server.uri()
        ))
        .unwrap();
        let error = Discovery::new()
            .unwrap()
            .list(&endpoint, None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Endpoint));
        assert!(!format!("{error} {error:?}").contains("synthetic-private-value"));
        assert!(server.received_requests().await.unwrap().is_empty());
    }
}
