//! Credential-free vector provider provenance; v1 §7.3 profile transitions.
use super::{Error, Store, admission, vectors::vector_table_name};
use rusqlite::{OptionalExtension, Transaction, params};

/// Only provider identity and query rendering, never a resolved credential or a
/// whole configuration dump. Execution limits remain current host policy.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Binding {
    pub(crate) profile: String,
    pub(crate) model: String,
    pub(crate) dims: u64,
    pub(crate) endpoint: String,
    pub(crate) artifact_hash: Option<[u8; 32]>,
    pub(crate) query_prefix: String,
    pub(crate) api_key_env: Option<String>,
}
/// A populated legacy table cannot truthfully acquire inferred provider history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    Known,
    LegacyUnknown,
}
fn refused(reason: &'static str) -> Error {
    Error::EmbeddingBinding { reason }
}
impl Binding {
    fn validate(&self) -> Result<(), Error> {
        if self.model.is_empty() || self.dims == 0 || self.dims > i32::MAX as u64 {
            return Err(refused("invalid model or dimensions"));
        }
        match self.profile.as_str() {
            "local"
                if self.endpoint.is_empty()
                    && self.artifact_hash.is_some()
                    && self.api_key_env.is_none() =>
            {
                Ok(())
            }
            "remote" | "ollama" if self.artifact_hash.is_none() => {
                let url =
                    reqwest::Url::parse(&self.endpoint).map_err(|_| refused("invalid endpoint"))?;
                if !matches!(url.scheme(), "http" | "https")
                    || url.host_str().is_none()
                    || !url.username().is_empty()
                    || url.password().is_some()
                    || url.query().is_some()
                    || url.fragment().is_some()
                {
                    return Err(refused("endpoint must be credential-free HTTP"));
                }
                if self.profile == "remote" {
                    if !url.path().ends_with("/embeddings")
                        || self
                            .api_key_env
                            .as_ref()
                            .is_none_or(|key| key.is_empty() || key.contains(['=', '\0']))
                    {
                        return Err(refused("invalid remote binding"));
                    }
                } else if !url.path().ends_with("/api/embed") || self.api_key_env.is_some() {
                    return Err(refused("invalid Ollama binding"));
                }
                Ok(())
            }
            _ => Err(refused("invalid provider or artifact binding")),
        }
    }
    /// Deterministic destination for a complete vector space, including endpoint
    /// or pinned artifact. Query-only instructions/key references do not fork it.
    pub(crate) fn destination_slug(&self) -> String {
        let dims = self.dims.to_le_bytes();
        let mut hash = blake3::Hasher::new_derive_key("dkc embedding vector space v1");
        for part in [
            self.profile.as_bytes(),
            self.model.as_bytes(),
            dims.as_slice(),
            self.endpoint.as_bytes(),
            self.artifact_hash
                .as_ref()
                .map_or(&[][..], |h| h.as_slice()),
        ] {
            hash.update(&(part.len() as u64).to_le_bytes());
            hash.update(part);
        }
        format!(
            "{}_s_{}",
            super::vectors::embedding_table_slug(&self.profile, &self.model, self.dims),
            hash.finalize().to_hex()
        )
    }

    /// Document-space identity excludes query-only rendering and key references;
    /// those may change without replacing already valid document vectors.
    pub(crate) fn same_space(&self, other: &Self) -> bool {
        self.profile == other.profile
            && self.model == other.model
            && self.dims == other.dims
            && self.endpoint == other.endpoint
            && self.artifact_hash == other.artifact_hash
    }
}

/// Reads owned provider provenance inside a caller's snapshot. None means unknown,
/// including upgraded legacy rows; current configuration is never substituted.
pub(crate) fn read(tx: &Transaction<'_>, slug: &str) -> Result<Option<Binding>, Error> {
    let binding = tx.query_row(
        "SELECT profile,model,dims,endpoint,artifact_hash,query_prefix,api_key_env FROM embedding_bindings WHERE slug=?1",
        [slug], |r| Ok(Binding { profile: r.get(0)?, model: r.get(1)?, dims: u64::from(r.get::<_,u32>(2)?), endpoint: r.get(3)?, artifact_hash: r.get(4)?, query_prefix: r.get(5)?, api_key_env: r.get(6)? }),
    ).optional()?;
    if let Some(binding) = &binding {
        binding.validate()?;
    }
    Ok(binding)
}

fn inspect(tx: &Transaction<'_>, slug: &str, binding: &Binding) -> Result<(State, bool), Error> {
    let metadata: Option<(String, String, i64)> = tx
        .query_row(
            "SELECT profile,model,dims FROM embedding_tables WHERE slug=?1",
            [slug],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    if metadata
        != Some((
            binding.profile.clone(),
            binding.model.clone(),
            binding.dims as i64,
        ))
    {
        return Err(refused("table metadata mismatch"));
    }
    if let Some(prior) = read(tx, slug)? {
        if !prior.same_space(binding) {
            return Err(refused("existing provider identity differs"));
        }
        return Ok((State::Known, prior != *binding));
    }
    let table = vector_table_name(slug)?;
    let populated: bool =
        tx.query_row(&format!("SELECT EXISTS(SELECT 1 FROM {table})"), [], |r| {
            r.get(0)
        })?;
    Ok(if populated {
        (State::LegacyUnknown, false)
    } else {
        (State::Known, true)
    })
}
/// Resolves a destination inside an existing snapshot, allowing a whole routing
/// map to be selected atomically without nested reader acquisition.
pub(crate) fn select(tx: &Transaction<'_>, binding: &Binding) -> Result<String, Error> {
    binding.validate()?;
    let destination = binding.destination_slug();
    let exists: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1)",
        [&destination],
        |r| r.get(0),
    )?;
    if exists {
        return Ok(destination);
    }
    let canonical =
        super::vectors::embedding_table_slug(&binding.profile, &binding.model, binding.dims);
    if read(tx, &canonical)?.is_some_and(|prior| prior.same_space(binding)) {
        return Ok(canonical);
    }
    Ok(destination)
}
impl Store {
    /// Selects one deterministic destination using metadata only. Reuses an older
    /// canonical name only with a matching proven binding; unknown legacy data is
    /// preserved separately. A conflicting occupied hash name is returned for the
    /// normal metadata/binding fences to reject, never bypassed with another alias.
    pub(crate) fn embedding_destination(&self, binding: &Binding) -> Result<String, Error> {
        self.read_snapshot(|tx| select(tx, binding))?
    }

    /// Captures provider provenance only before the first vector, or refreshes
    /// query-only settings for an already proven identical space. Rechecks inside
    /// admitted writing, preserving prior records on conflict or capacity refusal.
    /// Exact no-ops and unknown populated legacy tables need no write reservation.
    pub(crate) fn bind_embedding_table(
        &self,
        slug: &str,
        binding: &Binding,
    ) -> Result<State, Error> {
        binding.validate()?;
        vector_table_name(slug)?;
        let (state, change) = self.read_snapshot(|tx| inspect(tx, slug, binding))??;
        if !change {
            return Ok(state);
        }
        admission::transaction(self, admission::Kind::Growth, |tx| {
            let (state, change) = inspect(&tx, slug, binding)?;
            if change {
                tx.execute("INSERT INTO embedding_bindings(slug,profile,model,dims,endpoint,artifact_hash,query_prefix,api_key_env) VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(slug) DO UPDATE SET query_prefix=excluded.query_prefix,api_key_env=excluded.api_key_env",
                    params![slug,binding.profile,binding.model,binding.dims as i64,binding.endpoint,binding.artifact_hash.as_ref().map(|hash|hash.as_slice()),binding.query_prefix,binding.api_key_env])?;
            }
            tx.commit()?;
            Ok(state)
        })
    }
}

#[cfg(test)]
mod tests;
