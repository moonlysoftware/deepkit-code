//! Bounded provider response buffering; v1 §§7.2 and 8.2.

/// Allows default 64-item base64 or 32-item JSON batches with native 4096-wide
/// vectors and ample framing headroom. Exceptional larger batches fail explicitly.
pub(super) const EMBEDDING_BYTES: usize = 16 * 1024 * 1024;
/// Installed model listings are metadata, never model artifacts.
pub(super) const LISTING_BYTES: usize = 1024 * 1024;
/// Enough provider error text to recognize dimension rejection without trusting bodies.
pub(super) const ERROR_BYTES: usize = 64 * 1024;

/// No provider body, URL, configuration or transport source is retained.
#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("Provider response exceeds the supported size limit")]
    TooLarge,
    #[error("Provider response transport failed")]
    Transport,
}

/// Refuses oversized advertised or streamed bodies, without allocating from headers.
pub(super) async fn read(mut response: reqwest::Response, cap: usize) -> Result<Vec<u8>, Error> {
    if response
        .content_length()
        .is_some_and(|length| length > cap as u64)
    {
        return Err(Error::TooLarge);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| Error::Transport)? {
        if chunk.len() > cap.saturating_sub(body.len()) {
            return Err(Error::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Raw framing exercises absent Content-Length/chunked bodies and oversized
    // declared lengths independently of wiremock's automatic length generation.
    pub(crate) async fn serve_once(
        status: u16,
        headers: String,
        body: Vec<u8>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let read = stream.read(&mut buffer).await.unwrap();
                assert!(read > 0, "fixture request ended before its headers");
                request.extend_from_slice(&buffer[..read]);
                assert!(
                    request.len() <= 64 * 1024,
                    "fixture request exceeded its bound"
                );
                if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&request[..end]).unwrap();
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let head = format!("HTTP/1.1 {status} Test\r\nConnection: close\r\n{headers}\r\n");
            // Oversize rejection may close the connection before its body is sent.
            let _ = stream.write_all(head.as_bytes()).await;
            let _ = stream.write_all(&body).await;
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn rejects_advertised_and_chunked_overflow_but_accepts_exact_limit() {
        for (headers, body, allowed) in [
            ("Content-Length: 33\r\n", Vec::new(), false),
            (
                "Transfer-Encoding: chunked\r\n",
                b"20\r\n01234567890123456789012345678901\r\n1\r\nx\r\n0\r\n\r\n".to_vec(),
                false,
            ),
            (
                "Content-Length: 32\r\n",
                b"01234567890123456789012345678901".to_vec(),
                true,
            ),
        ] {
            let (url, task) = serve_once(200, headers.into(), body).await;
            let response = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(2))
                .build()
                .unwrap()
                .get(url)
                .send()
                .await
                .unwrap();
            let result = read(response, 32).await;
            if allowed {
                assert_eq!(result.unwrap().len(), 32);
            } else {
                assert!(matches!(result, Err(Error::TooLarge)));
            }
            task.await.unwrap();
        }
    }
}
