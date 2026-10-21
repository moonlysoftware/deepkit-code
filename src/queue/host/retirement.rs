//! Retire only obsolete empty representations under current policy; v1 §7.3/§16.
use super::*;
use crate::{
    config,
    store::embedding_bindings::{self, Binding},
};
use rusqlite::OptionalExtension;
use std::{fs, sync::atomic::AtomicBool};

#[derive(Clone)]
pub(super) struct Policy {
    pub(super) root: PathBuf,
    pub(super) config_root: PathBuf,
    pub(super) config: Config,
}
/// Cleanup progress is independent of embedding publication; prior successful
/// drops remain reported when a later table fails admission or a policy fence.
#[derive(Debug, Default)]
pub(crate) struct Report {
    pub(crate) retired: Vec<String>,
    pub(crate) error: Option<Error>,
}
impl Policy {
    fn check(&self, cancelled: &AtomicBool) -> Result<(), Error> {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Invalid("embedding retirement cancelled"));
        }
        let current = config::load(&self.config_root, None)
            .map_err(|_| Error::Invalid("embedding retirement configuration reload"))?
            .config;
        if current != self.config {
            return Err(Error::Invalid("embedding retirement configuration changed"));
        }
        let root = fs::canonicalize(&self.root)
            .map_err(|_| Error::Invalid("embedding retirement root"))?;
        if fs::canonicalize(self.config_root.join(&current.index.root))
            .ok()
            .as_ref()
            != Some(&root)
        {
            return Err(Error::Invalid("embedding retirement root changed"));
        }
        Ok(())
    }
    fn protected(&self) -> Result<Vec<Binding>, Error> {
        // Resolve metadata only: no credentials, clients, artifact I/O or native loads.
        // Protect both configured profiles even when this pass selects only code.
        let mut config = self.config.clone();
        let docs = config.embedder.docs.take();
        let mut profiles = vec![config.embedder.profile.clone()];
        if let Some(docs) = docs {
            profiles.push(docs.profile);
        }
        profiles
            .into_iter()
            .map(|profile| {
                config.embedder.profile = profile.clone();
                let resolved = destination::resolve(&config, UnitKind::Chunk)?;
                Ok(resolved.binding(&config))
            })
            .collect()
    }
    pub(super) fn run(&self, store: &Store, cancelled: &AtomicBool) -> Report {
        let mut report = Report::default();
        let result = (|| {
            self.check(cancelled)?;
            let protected = self.protected()?;
            // Metadata is capped by the native table bound, so this pass is finite.
            let tables = store.vector_tables()?;
            for table in tables {
                if store.read_snapshot(|tx| protected_in(tx, &table.slug, &protected))?? {
                    continue;
                }
                if store.retire_empty_vector_table_if(&table.slug, |tx| {
                    self.check(cancelled)?;
                    if protected_in(tx, &table.slug, &protected)? {
                        return Err(Error::Invalid(
                            "embedding retirement destination became current",
                        ));
                    }
                    Ok(())
                })? {
                    report.retired.push(table.slug);
                }
            }
            Ok::<_, Error>(())
        })();
        report.error = result.err();
        report
    }
}
fn protected_in(
    tx: &rusqlite::Transaction<'_>,
    slug: &str,
    protected: &[Binding],
) -> Result<bool, Error> {
    let identity: Option<(String, String, i64)> = tx
        .query_row(
            "SELECT profile,model,dims FROM embedding_tables WHERE slug=?1",
            [slug],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    let Some((profile, model, dims)) = identity else {
        return Ok(false);
    };
    if let Some(binding) = embedding_bindings::read(tx, slug)? {
        if binding.profile != profile
            || binding.model != model
            || i64::try_from(binding.dims).ok() != Some(dims)
        {
            return Err(Error::Invalid(
                "embedding retirement binding metadata mismatch",
            ));
        }
        return Ok(protected.iter().any(|current| current.same_space(&binding)));
    }
    // Preserve unknown legacy records that could still be configured. Cleanup
    // cannot infer an old endpoint from a shared profile/model name.
    Ok(protected.iter().any(|current| {
        current.profile == profile
            && current.model == model
            && i64::try_from(current.dims).ok() == Some(dims)
    }))
}
