//! Durable remote embedding dimension negotiation; v1 §§7.2 and 8.2.

use super::{Error, embedding::Destination};
use crate::{models::openai_compat::OpenAiEmbedder, store::Store};

/// Restores a prior endpoint/model/dimension rejection before the first request
/// and returns the credential-free metadata key used for later persistence.
pub(crate) fn restore(
    store: &Store,
    destination: &Destination,
    provider: &OpenAiEmbedder,
) -> Result<String, Error> {
    let key = key(destination);
    if store.get_meta(&key)?.as_deref() == Some("1") {
        provider.remember_dimensions_unsupported();
    }
    Ok(key)
}

/// Persists a confirmed rejection so a later CLI process omits `dimensions` on
/// its first request to the same endpoint, model and configured vector width.
pub(crate) fn remember(store: &Store, key: &str) -> Result<(), Error> {
    store.set_meta(key, "1")?;
    Ok(())
}

fn key(destination: &Destination) -> String {
    let mut identity = blake3::Hasher::new_derive_key("dkc remote dimensions capability v1");
    for field in [
        destination.provider.as_bytes(),
        destination.model.as_bytes(),
        &destination.dims.to_le_bytes(),
    ] {
        identity.update(&(field.len() as u64).to_le_bytes());
        identity.update(field);
    }
    format!(
        "remote_dimensions_unsupported:{}",
        identity.finalize().to_hex()
    )
}
