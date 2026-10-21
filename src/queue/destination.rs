//! Configured embedding destinations and provider limits; v1 §7.2–7.4 and §8.2.

use super::{Error, embedding::Destination, usage};
use crate::{config::Config, models::registry::Registry, store::UnitKind};

/// Validated settings for one selected provider; contains no resolved credentials.
pub(crate) struct Resolved {
    pub(crate) identity: Destination,
    pub(crate) concurrency: usize,
    pub(crate) max_batch_items: usize,
    pub(crate) max_batch_tokens: i64,
    pub(crate) max_doc_tokens: i64,
    rpm: Option<i64>,
    tpm: Option<i64>,
}
impl Resolved {
    /// Bounds one admitted provider request by both the model batch contract and
    /// any configured per-minute token ceiling. A document larger than this
    /// still reaches dispatch alone so the queue records the permanent refusal.
    pub(crate) fn max_request_tokens(&self) -> i64 {
        self.tpm
            .map_or(self.max_batch_tokens, |tpm| self.max_batch_tokens.min(tpm))
    }

    /// Captures query rendering and credential references alongside the resolved
    /// document space; no environment lookup or resolved credential is involved.
    pub(crate) fn binding(&self, config: &Config) -> crate::store::embedding_bindings::Binding {
        let (profile, prefix, key) = if self.identity.provider == "embed:local" {
            ("local", &config.embedder.local.query_prefix, None)
        } else if self.identity.provider.starts_with("embed:remote:") {
            (
                "remote",
                &config.embedder.remote.query_prefix,
                Some(config.embedder.remote.api_key_env.clone()),
            )
        } else {
            ("ollama", &config.embedder.ollama.query_prefix, None)
        };
        crate::store::embedding_bindings::Binding {
            profile: profile.into(),
            model: self.identity.model.clone(),
            dims: self.identity.dims,
            endpoint: self.identity.endpoint.clone(),
            artifact_hash: self.identity.artifact_hash,
            query_prefix: prefix.clone(),
            api_key_env: key,
        }
    }

    /// Combines current configured rates with the host's explicit daily budget.
    /// Explicit benchmark bounds and configured host budgets share admission.
    pub(crate) fn limits(&self, budget: usage::Budget) -> Result<usage::Limits, Error> {
        if budget.calls.is_some_and(|n| n < 0) || budget.tokens.is_some_and(|n| n < 0) {
            return Err(Error::Invalid("negative daily embedding budget"));
        }
        Ok(usage::Limits {
            budget,
            rpm: self.rpm,
            tpm: self.tpm,
        })
    }
}
/// Resolves current file authority, destination and canonical job inputs together.
/// The caller must invoke this again at dispatch/completion and compare generations.
pub(crate) fn prepare_file(
    tx: &rusqlite::Transaction<'_>,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    unit_id: i64,
) -> Result<Option<(super::embedding::Prepared, Resolved)>, Error> {
    prepare_file_with_work(
        tx,
        root,
        config_root,
        config,
        unit_id,
        crate::sync::l1::Work::default(),
    )
}

/// Prepares the same canonical job identity with cooperative source and result checks.
/// Caller must acquire the transaction under the same Work before invoking this API.
pub(crate) fn prepare_file_with_work(
    tx: &rusqlite::Transaction<'_>,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    unit_id: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<(super::embedding::Prepared, Resolved)>, Error> {
    let Some(source) =
        super::source::resolve_with_work(tx, root, config_root, config, unit_id, work)?
    else {
        return Ok(None);
    };
    let destination = resolve(config, source.occurrence.kind)?;
    let prepared = super::embedding::Prepared::new(
        source.occurrence,
        source.context,
        destination.identity.clone(),
        source.policy,
    )?;
    super::source::check_work(work)?;
    Ok(prepared.map(|prepared| (prepared, destination)))
}

/// Verified L2 eligibility or selected-space evidence, separate from source refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileFreshness {
    Ineligible,
    Representation(crate::store::embedding_freshness::State),
}

struct FreshnessInput {
    embedding: Option<(super::embedding::Identity, Resolved)>,
}

/// Returns eligibility only after verifying completed L1, current bytes and policy.
/// Ineligible files never resolve a provider or consult a physical vector table.
fn prepare_freshness(
    tx: &rusqlite::Transaction<'_>,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    unit_id: i64,
    work: crate::sync::l1::Work<'_>,
    roles: Option<&mut super::roles::Batch>,
) -> Result<Option<FreshnessInput>, Error> {
    let Some(mut source) = super::source::resolve_freshness_without_roles(
        tx,
        root,
        config_root,
        config,
        unit_id,
        work,
    )?
    else {
        return Ok(None);
    };
    if source.policy.never_send || source.policy.max_level < 2 {
        return Ok(Some(FreshnessInput { embedding: None }));
    }
    let destination = resolve(config, source.occurrence.kind)?;
    // An absent representation cannot be context-stale. Preserve source and
    // policy proof above, but avoid rebuilding every ancestor summary merely
    // to report missing L2 when models/embeddings are unavailable.
    let slug = crate::store::embedding_bindings::select(tx, &destination.binding(config))?;
    let active: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedding_tables WHERE slug=?1 AND active=1)",
        [&slug],
        |r| r.get(0),
    )?;
    if active {
        let table = crate::store::vectors::vector_table_name(&slug)?;
        let present: bool = tx.query_row(
            &format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE unit_id=?1)"),
            [unit_id],
            |r| r.get(0),
        )?;
        if present {
            source = if let Some(roles) = roles {
                roles.apply(tx, root, config_root, config, source, work)?
            } else {
                super::roles::apply(tx, root, config_root, config, source, work)?
            };
        }
    }
    let prepared = super::embedding::Identity::new(
        source.occurrence,
        &source.context,
        &destination.identity,
        source.policy,
    )?;
    super::source::check_work(work)?;
    Ok(prepared.map(|prepared| FreshnessInput {
        embedding: Some((prepared, destination)),
    }))
}

/// Observes selected-space L2 evidence under the caller's original work budget.
/// None means canonical source authority was refused, not that no vector exists.
/// Ineligible is verified policy, while missing/stale representations are pending.
/// The caller must revalidate its result generation before annotating public hits;
/// this coherent database snapshot is not an atomic filesystem snapshot.
/// No providers, models, queue writes or fallback vector tables are consulted.
pub(crate) fn file_freshness(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    unit_id: i64,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<FileFreshness>, Error> {
    file_freshness_checked(store, root, config_root, config, unit_id, work, || {})
}

/// Binds canonical L2 evidence to the exact occurrence retained by retrieval.
/// Rechecks in a new snapshot to reject same-generation publication changes;
/// callers still own the final catalogue-generation and source checks.
pub(crate) fn occurrence_freshness(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    occurrence: &crate::store::StoredUnit,
    work: crate::sync::l1::Work<'_>,
) -> Result<Option<FileFreshness>, Error> {
    occurrence_freshness_checked(store, root, config_root, config, occurrence, work, || {})
}

/// Observes a bounded search result batch, whose caller separately fences occurrences.
/// Two independent snapshots bracket storage evidence with current source and
/// context proof. Shared ancestor roles are re-proved at each snapshot's end,
/// including paths with no role, and never reused across snapshots. The caller performs
/// final source, occurrence and catalogue-generation checks.
pub(crate) fn search_occurrences_freshness(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    occurrences: &[&crate::store::StoredUnit],
    work: crate::sync::l1::Work<'_>,
) -> Result<Vec<Option<FileFreshness>>, Error> {
    search_occurrences_freshness_checked(store, root, config_root, config, occurrences, work, || {})
}

#[cfg(test)]
fn search_occurrence_freshness_checked(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    occurrence: &crate::store::StoredUnit,
    work: crate::sync::l1::Work<'_>,
    between_snapshots: impl FnOnce(),
) -> Result<Option<FileFreshness>, Error> {
    Ok(search_occurrences_freshness_checked(
        store,
        root,
        config_root,
        config,
        &[occurrence],
        work,
        between_snapshots,
    )?
    .remove(0))
}

fn search_occurrences_freshness_checked(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    occurrences: &[&crate::store::StoredUnit],
    work: crate::sync::l1::Work<'_>,
    between_snapshots: impl FnOnce(),
) -> Result<Vec<Option<FileFreshness>>, Error> {
    super::source::check_work(work)?;
    if occurrences.is_empty() {
        return Ok(Vec::new());
    }
    #[derive(PartialEq, Eq)]
    struct Observation {
        state: FileFreshness,
        unit: i64,
        content: [u8; 32],
        desired: [u8; 32],
        context: [u8; 32],
    }
    impl Observation {
        fn same_pending_input(&self, other: &Self) -> bool {
            use crate::store::embedding_freshness::State;
            let pending = |state| {
                matches!(
                    state,
                    FileFreshness::Representation(State::Missing | State::Stale)
                )
            };
            // Admitting an empty/inactive table may change Missing to Stale.
            // Both publicly mean pending; neither authorizes a current vector.
            // Retain the independently verified input, not transient metadata.
            pending(self.state)
                && pending(other.state)
                && self.unit == other.unit
                && self.content == other.content
                && self.desired == other.desired
                && self.context == other.context
        }
    }
    let observe = || -> Result<Vec<Option<Observation>>, Error> {
        super::source::check_work(work)?;
        let result = store.checked_read_snapshot(
            &|| {
                work.check()
                    .map_err(|_| crate::store::Error::L1InputsChanged)
            },
            |tx| {
                let mut roles = super::roles::Batch::default();
                let mut observations = occurrences
                    .iter()
                    .map(|occurrence| {
                        // Pin occurrence/storage identity before source verification.
                        // Each observation owns a new SQLite snapshot; no role authority
                        // is cached across the intervening publication/source boundary.
                        if !FreshnessTarget::Occurrence(occurrence).matches(tx)? {
                            tracing::debug!(
                                reason = "occurrence_changed",
                                "search L2 authority refused"
                            );
                            return Ok(None);
                        }
                        let Some(unit) = represented_unit(tx, occurrence.id)? else {
                            tracing::debug!(
                                reason = "representation_unavailable",
                                "search L2 authority refused"
                            );
                            return Ok(None);
                        };
                        let Some(input) = prepare_freshness(
                            tx,
                            root,
                            config_root,
                            config,
                            unit,
                            work,
                            Some(&mut roles),
                        )?
                        else {
                            tracing::debug!(
                                reason = "source_or_identity_unavailable",
                                "search L2 authority refused"
                            );
                            return Ok(None);
                        };
                        let Some((identity, destination)) = input.embedding else {
                            return Ok(Some(Observation {
                                state: FileFreshness::Ineligible,
                                unit,
                                content: [0; 32],
                                desired: [0; 32],
                                context: [0; 32],
                            }));
                        };
                        let current = identity.current();
                        let state = crate::store::embedding_freshness::read(
                            tx,
                            &destination.binding(config),
                            unit,
                            &current.target_hash,
                            identity.context_hash(),
                            &current.desired_hash,
                        )?;
                        Ok::<_, Error>(Some(Observation {
                            state: FileFreshness::Representation(state),
                            unit,
                            content: current.target_hash,
                            desired: current.desired_hash,
                            context: *identity.context_hash(),
                        }))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                // The cache belongs only to this snapshot. Re-prove its full
                // positive and negative path set before retaining any result.
                if !roles.verify(tx, root, config_root, config, work)? {
                    observations.iter_mut().for_each(|item| *item = None);
                }
                Ok::<_, Error>(observations)
            },
        );
        super::source::check_work(work)?;
        result?
    };
    let before = observe()?;
    between_snapshots();
    // Re-prove sources after the first storage observation, even for missing or
    // stale vectors. Two observations replace nested before/after pairs (four
    // full role reconstructions for a current vector).
    let after = observe()?;
    Ok(before.into_iter().zip(after).map(|(before, after)| {
        let same_pending_input = before.as_ref().zip(after.as_ref())
            .is_some_and(|(before, after)| before.same_pending_input(after));
        if before == after || same_pending_input { after.map(|input| input.state) }
        else {
            tracing::debug!(before=?before.as_ref().map(|v| v.state), after=?after.as_ref().map(|v| v.state), reason="observation_changed", "search L2 authority refused");
            None
        }
    }).collect())
}

fn occurrence_freshness_checked(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    occurrence: &crate::store::StoredUnit,
    work: crate::sync::l1::Work<'_>,
    between_snapshots: impl FnOnce(),
) -> Result<Option<FileFreshness>, Error> {
    let observe = || {
        read_file_freshness(
            store,
            root,
            config_root,
            config,
            FreshnessTarget::Occurrence(occurrence),
            work,
            || {},
        )
    };
    let before = observe()?;
    if before.is_none() {
        return Ok(None);
    }
    between_snapshots();
    let after = observe()?;
    Ok(if before == after { after } else { None })
}

#[derive(Clone, Copy)]
enum FreshnessTarget<'a> {
    Id(i64),
    Occurrence(&'a crate::store::StoredUnit),
}

impl FreshnessTarget<'_> {
    fn id(self) -> i64 {
        match self {
            Self::Id(id) => id,
            Self::Occurrence(unit) => unit.id,
        }
    }

    fn matches(self, tx: &rusqlite::Transaction<'_>) -> Result<bool, Error> {
        let Self::Occurrence(expected) = self else {
            return Ok(true);
        };
        Ok(crate::store::occurrence_matches_in(tx, expected)?)
    }
}

fn file_freshness_checked(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    unit_id: i64,
    work: crate::sync::l1::Work<'_>,
    after_observation: impl FnOnce(),
) -> Result<Option<FileFreshness>, Error> {
    read_file_freshness(
        store,
        root,
        config_root,
        config,
        FreshnessTarget::Id(unit_id),
        work,
        after_observation,
    )
}

/// Image parents have no text vector. Their owned description is the searchable
/// representation; this mapping is only for freshness, never raw-image enqueue.
fn represented_unit(tx: &rusqlite::Transaction<'_>, unit: i64) -> Result<Option<i64>, Error> {
    use rusqlite::OptionalExtension;
    Ok(tx
        .query_row(
            "SELECT CASE WHEN p.kind='image' THEN (
        SELECT s.id FROM units s JOIN files f ON f.id=p.file_id
        WHERE s.parent_unit=p.id AND s.kind='summary' AND f.kind='image'
        AND p.ladder_level=0 AND p.parent_unit IS NULL
        AND s.source_key='summary:image:'||f.id
    ) ELSE p.id END FROM units p WHERE p.id=?1",
            [unit],
            |r| r.get::<_, Option<i64>>(0),
        )
        .optional()?
        .flatten())
}

fn read_file_freshness(
    store: &crate::store::Store,
    root: &std::path::Path,
    config_root: &std::path::Path,
    config: &Config,
    target: FreshnessTarget<'_>,
    work: crate::sync::l1::Work<'_>,
    after_observation: impl FnOnce(),
) -> Result<Option<FileFreshness>, Error> {
    let unit_id = target.id();
    super::source::check_work(work)?;
    let result = store.checked_read_snapshot(
        &|| {
            work.check()
                .map_err(|_| crate::store::Error::L1InputsChanged)
        },
        |tx| {
            if !target.matches(tx)? {
                return Ok(None);
            }
            let prepare = || {
                let Some(id) = represented_unit(tx, unit_id)? else {
                    return Ok(None);
                };
                Ok::<_, Error>(
                    prepare_freshness(tx, root, config_root, config, id, work, None)?
                        .map(|input| (id, input)),
                )
            };
            let Some((represented, before)) = prepare()? else {
                return Ok(None);
            };
            let Some((before, destination)) = before.embedding else {
                after_observation();
                return Ok(
                    matches!(prepare()?, Some((_, FreshnessInput { embedding: None })))
                        .then_some(FileFreshness::Ineligible),
                );
            };
            let identity = before.current();
            let state = crate::store::embedding_freshness::read(
                tx,
                &destination.binding(config),
                represented,
                &identity.target_hash,
                before.context_hash(),
                &identity.desired_hash,
            )?;
            after_observation();
            // A filesystem/config edit during storage observation invalidates the
            // evidence even when the SQLite snapshot still contains matching rows.
            let Some((
                after_id,
                FreshnessInput {
                    embedding: Some((after, _)),
                },
            )) = prepare()?
            else {
                return Ok(None);
            };
            let current = after.current();
            if after_id != represented
                || current.target_hash != identity.target_hash
                || current.desired_hash != identity.desired_hash
                || after.context_hash() != before.context_hash()
            {
                return Ok(None);
            }
            Ok(Some(FileFreshness::Representation(state)))
        },
    );
    // Preserve typed cancellation even when admission or the callback also failed.
    super::source::check_work(work)?;
    result?
}

fn positive(value: u64) -> Result<usize, Error> {
    if value == 0 {
        return Err(Error::Invalid("zero embedding execution limit"));
    }
    usize::try_from(value).map_err(|_| Error::Invalid("embedding execution limit overflow"))
}
fn tokens(value: u64) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::Invalid("embedding token or rate limit overflow"))
}
fn endpoint(base: &str, suffix: &str) -> Result<String, Error> {
    crate::config::queue::embedding_endpoint(base, suffix)
        .map_err(|_| Error::Invalid("embedding endpoint must use credential-free HTTP base URL"))
}

/// Selects only the configured tier for this kind, without loading models or keys.
/// Local model identity is pinned to registry bytes; this does not verify cache bytes.
/// An absent docs tier never inspects a second provider or vector table.
pub(crate) fn resolve(config: &Config, kind: UnitKind) -> Result<Resolved, Error> {
    let embedder = &config.embedder;
    let profile = embedder
        .docs
        .as_ref()
        .filter(|docs| docs.kinds.iter().any(|value| value == kind.as_str()))
        .map_or(embedder.profile.as_str(), |docs| docs.profile.as_str());
    positive(embedder.dims)?;
    let max_doc_tokens = tokens(embedder.max_doc_tokens)?;
    if max_doc_tokens == 0 {
        return Err(Error::Invalid("zero embedding document limit"));
    }
    let (model, endpoint, artifact_hash, concurrency, items, batch_tokens, rpm, tpm) = match profile
    {
        "remote" => {
            let remote = &embedder.remote;
            positive(remote.max_batch_tokens)?;
            (
                remote.model.clone(),
                endpoint(&remote.url, "embeddings")?,
                None,
                positive(remote.concurrency)?,
                positive(remote.max_batch_items)?,
                tokens(remote.max_batch_tokens)?,
                Some(tokens(remote.rpm)?),
                Some(tokens(remote.tpm)?),
            )
        }
        "ollama" => {
            let ollama = &embedder.ollama;
            (
                ollama.model.clone(),
                endpoint(&ollama.url, "api/embed")?,
                None,
                positive(ollama.concurrency)?,
                positive(ollama.max_batch_items)?,
                max_doc_tokens
                    .checked_mul(tokens(ollama.max_batch_items)?)
                    .ok_or(Error::Invalid("embedding batch token overflow"))?,
                None,
                None,
            )
        }
        "local" => {
            let local = &embedder.local;
            let registry =
                Registry::embedded().map_err(|_| Error::Invalid("embedding registry"))?;
            let model = registry
                .get(&local.model)
                .map_err(|_| Error::Invalid("unregistered embedding model"))?;
            if model.pooling != "last" || embedder.dims > model.dims as u64 {
                return Err(Error::Invalid("local model is not a compatible embedder"));
            }
            positive(local.batch_tokens)?;
            positive(local.n_ctx)?;
            let mut hash = [0; 32];
            for (byte, pair) in hash
                .iter_mut()
                .zip(model.sha256.as_bytes().as_chunks::<2>().0)
            {
                let text = std::str::from_utf8(pair)
                    .map_err(|_| Error::Invalid("embedding artifact hash"))?;
                *byte = u8::from_str_radix(text, 16)
                    .map_err(|_| Error::Invalid("embedding artifact hash"))?;
            }
            // One native model runner; llama's own token batch handles parallel inputs.
            (
                local.model.clone(),
                String::new(),
                Some(hash),
                1,
                positive(local.batch_tokens)?,
                tokens(local.batch_tokens.min(local.n_ctx))?,
                None,
                None,
            )
        }
        _ => return Err(Error::Invalid("unknown embedding profile")),
    };
    if model.is_empty() {
        return Err(Error::Invalid("empty embedding model"));
    }
    // Provider-wide rates/budgets belong to an endpoint, not to individual models
    // or tiers; both tiers using this endpoint therefore share admission.
    let provider = crate::config::queue::embedding_provider(profile, &endpoint);
    Ok(Resolved {
        identity: Destination {
            provider,
            endpoint,
            model,
            artifact_hash,
            dims: embedder.dims,
        },
        concurrency,
        max_batch_items: items,
        max_batch_tokens: batch_tokens,
        max_doc_tokens,
        rpm,
        tpm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DocsEmbedder;

    fn eligibility_fixture(
        max_level: i64,
        never_send: bool,
    ) -> Result<(tempfile::TempDir, crate::store::Store, Config), Box<dyn std::error::Error>> {
        use crate::{config, store::FileKind, sync, treat::Router};
        let (root, store, _) = super::super::source::tests::fixture()?;
        std::fs::write(
            root.path().join(".deepkit/config.toml"),
            format!(
                "[index]\nrepo_name='fixture'\n[[treatments]]\nname='eligibility'\nmatch=['*.rs']\nmax_level={max_level}\nnever_send={never_send}\n"
            ),
        )?;
        let config = config::load(root.path(), None)?.config;
        let treatment =
            Router::new(&config.treatments, config.llm.enabled)?.route("a.rs", FileKind::Code);
        let fingerprint = sync::l1::file_fingerprint(
            &config,
            "fixture",
            sync::l1::FingerprintFile {
                path: "a.rs",
                content_hash: blake3::hash(b"fn example() {}").as_bytes(),
                language: Some("rust"),
                kind: FileKind::Code,
                capped: false,
            },
            &treatment,
        );
        store.writer()?.execute(
            "UPDATE files SET max_level=?1,treatment='eligibility',l1_desired=?2,l1_fingerprint=?2,l1_pending=0",
            rusqlite::params![max_level, fingerprint.as_slice()],
        )?;
        // Catalogue policy changes deliberately trigger durable pending work.
        store
            .writer()?
            .execute("UPDATE files SET l1_pending=0", [])?;
        Ok((root, store, config))
    }

    #[test]
    fn occurrence_evidence_rejects_mismatched_retrieved_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        for level in [1, 3] {
            let (root, store, config) = eligibility_fixture(level, false)?;
            let original = store.unit_by_id(1)?.ok_or("unit")?;
            let read = |unit: &crate::store::StoredUnit| {
                occurrence_freshness(
                    &store,
                    root.path(),
                    root.path(),
                    &config,
                    unit,
                    Default::default(),
                )
            };
            assert!(read(&original)?.is_some());
            let changes = store.writer()?.total_changes();
            for field in 0..8 {
                let mut changed = original.clone();
                match field {
                    0 => changed.id += 1,
                    1 => changed.hash[0] ^= 1,
                    2 => changed.path = Some("other.rs".into()),
                    3 => changed.kind = "symbol".into(),
                    4 => changed.line_start = Some(99),
                    5 => changed.line_end = Some(100),
                    6 => changed.header.push_str(" obsolete"),
                    _ => changed.text.push_str(" obsolete"),
                }
                assert_eq!(read(&changed)?, None, "level={level}, field={field}");
            }
            let mut advanced = original.clone();
            advanced.ladder_level += 1;
            assert_eq!(read(&advanced)?, read(&original)?);
            assert_eq!(store.writer()?.total_changes(), changes);
        }
        Ok(())
    }

    #[test]
    fn occurrence_evidence_rechecks_same_generation_replacement_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::atomic::{AtomicBool, Ordering};
        for level in [1, 3] {
            let (root, store, config) = eligibility_fixture(level, false)?;
            let original = store.unit_by_id(1)?.ok_or("unit")?;
            let generation = store.l1_generation()?;
            let result = occurrence_freshness_checked(
                &store,
                root.path(),
                root.path(),
                &config,
                &original,
                Default::default(),
                || {
                    store
                        .writer()
                        .unwrap()
                        .execute("UPDATE units SET header='new header' WHERE id=1", [])
                        .unwrap();
                },
            )?;
            assert_eq!(result, None);
            assert_eq!(store.l1_generation()?, generation);
            let current = store.unit_by_id(1)?.ok_or("current unit")?;
            assert!(
                occurrence_freshness(
                    &store,
                    root.path(),
                    root.path(),
                    &config,
                    &current,
                    Default::default()
                )?
                .is_some()
            );
            let cancelled = AtomicBool::new(false);
            let work = crate::sync::l1::Work {
                deadline: None,
                cancelled: Some(&cancelled),
            };
            assert!(matches!(
                occurrence_freshness_checked(
                    &store,
                    root.path(),
                    root.path(),
                    &config,
                    &current,
                    work,
                    || cancelled.store(true, Ordering::Relaxed),
                ),
                Err(Error::WorkerStopped)
            ));
            cancelled.store(false, Ordering::Relaxed);
            assert!(
                occurrence_freshness(&store, root.path(), root.path(), &config, &current, work)?
                    .is_some()
            );
        }
        Ok(())
    }

    #[test]
    fn pending_search_evidence_survives_empty_table_admission_but_not_source_changes()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::store::{embedding_freshness::State, vectors};
        for change_source in [false, true] {
            let (root, store, config) = super::super::source::tests::fixture()?;
            let original = store.unit_by_id(1)?.ok_or("unit")?;
            let (_, destination) = store
                .read_snapshot(|tx| prepare_file(tx, root.path(), root.path(), &config, 1))??
                .ok_or("prepared input")?;
            let binding = destination.binding(&config);
            let slug = binding.destination_slug();
            assert_eq!(
                search_occurrences_freshness(
                    &store,
                    root.path(),
                    root.path(),
                    &config,
                    &[&original],
                    Default::default()
                )?,
                vec![Some(FileFreshness::Representation(State::Missing))]
            );
            let result = search_occurrence_freshness_checked(
                &store,
                root.path(),
                root.path(),
                &config,
                &original,
                Default::default(),
                || {
                    // Startup admits the physical table before its active binding.
                    // No vector or publication exists at either boundary.
                    store
                        .ensure_vector_table(vectors::VectorTableSpec {
                            slug: &slug,
                            model: &binding.model,
                            profile: &binding.profile,
                            dims: binding.dims as usize,
                            created_at: 0,
                            active: false,
                        })
                        .unwrap();
                    if change_source {
                        std::fs::write(root.path().join("a.rs"), "fn changed() {}\n").unwrap();
                    }
                },
            )?;
            assert_eq!(
                result,
                if change_source {
                    None
                } else {
                    Some(FileFreshness::Representation(State::Stale))
                }
            );
        }
        Ok(())
    }

    #[test]
    fn occurrence_evidence_rechecks_publication_without_catalogue_change()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::store::{embedding_freshness::State, vectors};
        let (root, store, config) = super::super::source::tests::fixture()?;
        let original = store.unit_by_id(1)?.ok_or("unit")?;
        let (prepared, destination) = store
            .read_snapshot(|tx| prepare_file(tx, root.path(), root.path(), &config, 1))??
            .ok_or("prepared input")?;
        let binding = destination.binding(&config);
        let slug = binding.destination_slug();
        store.ensure_vector_table(vectors::VectorTableSpec {
            slug: &slug,
            model: &binding.model,
            profile: &binding.profile,
            dims: binding.dims as usize,
            created_at: 0,
            active: true,
        })?;
        store.bind_embedding_table(&slug, &binding)?;
        store.insert_vector(vectors::VectorInsert {
            table_slug: &slug,
            unit_id: 1,
            ctx_hash: prepared.context_hash(),
            embedding_hash: &prepared.current().target_hash,
            embedding: &vec![1.0; binding.dims as usize],
        })?;
        store.writer()?.execute(
            "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(1,?1,?2)",
            rusqlite::params![slug, prepared.current().desired_hash.as_slice()],
        )?;
        let read = || {
            occurrence_freshness(
                &store,
                root.path(),
                root.path(),
                &config,
                &original,
                Default::default(),
            )
        };
        assert_eq!(read()?, Some(FileFreshness::Representation(State::Current)));
        let mut changed = original.clone();
        changed.header.push_str(" obsolete");
        assert_eq!(
            search_occurrences_freshness(
                &store,
                root.path(),
                root.path(),
                &config,
                &[&original, &changed, &original],
                Default::default(),
            )?,
            vec![
                Some(FileFreshness::Representation(State::Current)),
                None,
                Some(FileFreshness::Representation(State::Current))
            ]
        );
        let original_bytes = std::fs::read(root.path().join("a.rs"))?;
        assert_eq!(
            search_occurrence_freshness_checked(
                &store,
                root.path(),
                root.path(),
                &config,
                &original,
                Default::default(),
                || {
                    std::fs::write(root.path().join("a.rs"), "fn changed() {}\n").unwrap();
                },
            )?,
            None,
            "a source edit between observations must refuse current coverage",
        );
        std::fs::write(root.path().join("a.rs"), original_bytes)?;
        let generation = store.l1_generation()?;
        assert_eq!(
            search_occurrence_freshness_checked(
                &store,
                root.path(),
                root.path(),
                &config,
                &original,
                Default::default(),
                || {
                    store
                        .writer()
                        .unwrap()
                        .execute(
                            "UPDATE embedding_publications SET desired_hash=zeroblob(32)",
                            [],
                        )
                        .unwrap();
                },
            )?,
            None
        );
        assert_eq!(store.l1_generation()?, generation);
        assert_eq!(read()?, Some(FileFreshness::Representation(State::Stale)));
        // Pending evidence cannot claim a current publication; final search
        // source/occurrence/generation checks still apply to retained hits.
        let repeated = std::cell::Cell::new(false);
        assert_eq!(
            search_occurrence_freshness_checked(
                &store,
                root.path(),
                root.path(),
                &config,
                &original,
                Default::default(),
                || repeated.set(true),
            )?,
            Some(FileFreshness::Representation(State::Stale))
        );
        assert!(repeated.get());
        Ok(())
    }

    #[test]
    fn file_eligibility_distinguishes_verified_policy_from_refused_authority()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::store::embedding_freshness::State;
        for (level, protected) in [(1, false), (1, true), (3, true), (3, false)] {
            let (root, store, config) = eligibility_fixture(level, protected)?;
            let eligible = level >= 2 && !protected;
            let read = || {
                file_freshness(
                    &store,
                    root.path(),
                    root.path(),
                    &config,
                    1,
                    Default::default(),
                )
            };
            let changes = store.writer()?.total_changes();
            assert_eq!(
                read()?,
                Some(if eligible {
                    FileFreshness::Representation(State::Missing)
                } else {
                    FileFreshness::Ineligible
                })
            );
            assert_eq!(store.writer()?.total_changes(), changes);
            // Read-only inspection must not widen worker dispatch permissions.
            assert_eq!(
                store
                    .read_snapshot(|tx| prepare_file(tx, root.path(), root.path(), &config, 1))??
                    .is_some(),
                eligible
            );
            std::fs::write(root.path().join("a.rs"), "fn changed() {}")?;
            assert_eq!(read()?, None);
            std::fs::write(root.path().join("a.rs"), "fn example() {}")?;
            store
                .writer()?
                .execute("UPDATE files SET l1_pending=1", [])?;
            assert_eq!(read()?, None);
            store.writer()?.execute(
                "UPDATE files SET l1_pending=0,l1_fingerprint=zeroblob(32)",
                [],
            )?;
            assert_eq!(read()?, None);
        }
        Ok(())
    }

    #[test]
    fn ineligible_observation_rechecks_source_config_and_original_work()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::sync::l1::Work;
        use std::sync::atomic::{AtomicBool, Ordering};
        for mutation in ["source", "config", "cancel"] {
            let (root, store, config) = eligibility_fixture(3, true)?;
            let config_path = root.path().join(".deepkit/config.toml");
            let original = std::fs::read(&config_path)?;
            let cancelled = AtomicBool::new(false);
            let work = Work {
                deadline: None,
                cancelled: Some(&cancelled),
            };
            let changes = store.writer()?.total_changes();
            let result =
                file_freshness_checked(&store, root.path(), root.path(), &config, 1, work, || {
                    match mutation {
                        "source" => {
                            std::fs::write(root.path().join("a.rs"), "fn changed() {}").unwrap()
                        }
                        "config" => {
                            std::fs::write(&config_path, "[index]\nrepo_name='changed'\n").unwrap()
                        }
                        _ => cancelled.store(true, Ordering::Relaxed),
                    }
                });
            if mutation == "cancel" {
                assert!(matches!(result, Err(Error::WorkerStopped)));
            } else {
                assert_eq!(result?, None);
            }
            assert_eq!(store.writer()?.total_changes(), changes);
            cancelled.store(false, Ordering::Relaxed);
            std::fs::write(root.path().join("a.rs"), "fn example() {}")?;
            std::fs::write(&config_path, original)?;
            assert_eq!(
                file_freshness(&store, root.path(), root.path(), &config, 1, work)?,
                Some(FileFreshness::Ineligible)
            );
        }
        Ok(())
    }

    #[test]
    fn file_evidence_uses_canonical_publication_without_writes()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::store::{embedding_freshness::State, vectors};
        let (root, store, config) = super::super::source::tests::fixture()?;
        let read = || {
            file_freshness(
                &store,
                root.path(),
                root.path(),
                &config,
                1,
                Default::default(),
            )
        };
        assert_eq!(read()?, Some(FileFreshness::Representation(State::Missing)));
        let (prepared, destination) = store
            .read_snapshot(|tx| prepare_file(tx, root.path(), root.path(), &config, 1))??
            .ok_or("prepared input")?;
        let binding = destination.binding(&config);
        let slug = binding.destination_slug();
        store.ensure_vector_table(vectors::VectorTableSpec {
            slug: &slug,
            model: &binding.model,
            profile: &binding.profile,
            dims: binding.dims as usize,
            created_at: 0,
            active: true,
        })?;
        store.bind_embedding_table(&slug, &binding)?;
        store.insert_vector(vectors::VectorInsert {
            table_slug: &slug,
            unit_id: 1,
            ctx_hash: prepared.context_hash(),
            embedding_hash: &prepared.current().target_hash,
            embedding: &vec![1.0; binding.dims as usize],
        })?;
        // Physical coverage and ladder state alone cannot establish publication.
        assert_eq!(read()?, Some(FileFreshness::Representation(State::Stale)));
        store.writer()?.execute(
            "INSERT INTO embedding_publications(unit_id,table_slug,desired_hash) VALUES(1,?1,?2)",
            rusqlite::params![slug, prepared.current().desired_hash.as_slice()],
        )?;
        let changes = store.writer()?.total_changes();
        assert_eq!(read()?, Some(FileFreshness::Representation(State::Current)));
        assert_eq!(store.writer()?.total_changes(), changes);
        store.writer()?.execute(
            "UPDATE embedding_publications SET desired_hash=zeroblob(32)",
            [],
        )?;
        assert_eq!(read()?, Some(FileFreshness::Representation(State::Stale)));
        std::fs::write(root.path().join("a.rs"), "fn changed() {}")?;
        assert_eq!(read()?, None);
        std::fs::write(root.path().join("a.rs"), "fn example() {}")?;
        assert_eq!(read()?, Some(FileFreshness::Representation(State::Stale)));
        assert_eq!(
            file_freshness(
                &store,
                root.path(),
                root.path(),
                &config,
                999,
                Default::default()
            )?,
            None
        );
        Ok(())
    }

    #[test]
    fn file_evidence_rechecks_source_configuration_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::{store::embedding_freshness::State, sync::l1::Work};
        use std::sync::atomic::{AtomicBool, Ordering};
        for mutation in ["source", "config", "cancel"] {
            let (root, store, config) = super::super::source::tests::fixture()?;
            let cancelled = AtomicBool::new(false);
            let work = Work {
                deadline: None,
                cancelled: Some(&cancelled),
            };
            let changes = store.writer()?.total_changes();
            let result =
                file_freshness_checked(&store, root.path(), root.path(), &config, 1, work, || {
                    match mutation {
                        "source" => {
                            std::fs::write(root.path().join("a.rs"), "fn changed() {}").unwrap()
                        }
                        "config" => std::fs::write(
                            root.path().join(".deepkit/config.toml"),
                            "[index]\nrepo_name='changed'\n",
                        )
                        .unwrap(),
                        _ => cancelled.store(true, Ordering::Relaxed),
                    }
                });
            if mutation == "cancel" {
                assert!(matches!(result, Err(Error::WorkerStopped)));
            } else {
                assert_eq!(result?, None);
            }
            assert_eq!(store.writer()?.total_changes(), changes);
            cancelled.store(false, Ordering::Relaxed);
            std::fs::write(root.path().join("a.rs"), "fn example() {}")?;
            std::fs::write(
                root.path().join(".deepkit/config.toml"),
                "[index]\nrepo_name='fixture'\n",
            )?;
            assert_eq!(
                file_freshness(&store, root.path(), root.path(), &config, 1, work)?,
                Some(FileFreshness::Representation(State::Missing))
            );
        }
        Ok(())
    }

    #[test]
    fn file_evidence_bounds_admission_and_retries_after_deadline()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::{store::embedding_freshness::State, sync::l1::Work};
        use std::time::{Duration, Instant};
        let (root, store, config) = super::super::source::tests::fixture()?;
        let held = store.writer()?;
        let work = Work {
            deadline: Some(Instant::now() + Duration::from_millis(15)),
            cancelled: None,
        };
        let started = Instant::now();
        assert!(matches!(
            file_freshness(&store, root.path(), root.path(), &config, 1, work),
            Err(Error::WorkerStopped)
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(held);
        assert!(matches!(
            file_freshness(&store, root.path(), root.path(), &config, 1, work),
            Err(Error::WorkerStopped)
        ));
        assert_eq!(
            file_freshness(
                &store,
                root.path(),
                root.path(),
                &config,
                1,
                Default::default()
            )?,
            Some(FileFreshness::Representation(State::Missing))
        );
        Ok(())
    }

    #[test]
    fn single_mode_ignores_unused_profiles_and_docs_tier_routes_only_configured_kinds()
    -> Result<(), Error> {
        let mut config = Config::default();
        config.embedder.profile = "local".into();
        config.embedder.remote.url = "invalid unused URL".into();
        let local = resolve(&config, UnitKind::Chunk)?;
        assert!(local.identity.artifact_hash.is_some());
        assert_eq!(local.concurrency, 1);
        assert!(local.identity.endpoint.is_empty());
        assert!(
            resolve(&config, UnitKind::DocSection)?
                .identity
                .artifact_hash
                .is_some()
        );
        config.embedder.docs = Some(DocsEmbedder {
            profile: "ollama".into(),
            kinds: vec!["doc_section".into()],
        });
        assert!(
            resolve(&config, UnitKind::Chunk)?
                .identity
                .artifact_hash
                .is_some()
        );
        assert_eq!(
            resolve(&config, UnitKind::DocSection)?.identity.endpoint,
            "http://127.0.0.1:11434/api/embed"
        );
        config.embedder.local.model = "qwen3-reranker-0.6b-q8_0".into();
        assert!(resolve(&config, UnitKind::Chunk).is_err());
        Ok(())
    }

    #[test]
    fn rates_budgets_and_endpoint_identity_do_not_depend_on_credentials_or_model()
    -> Result<(), Error> {
        let mut config = Config::default();
        config.embedder.remote.url = "https://EXAMPLE.test/v1/".into();
        config.embedder.remote.rpm = 0;
        let first = resolve(&config, UnitKind::Chunk)?;
        let limits = first.limits(usage::Budget {
            calls: Some(0),
            tokens: Some(9),
        })?;
        assert_eq!(limits.rpm, Some(0));
        assert_eq!(limits.tpm, Some(2_000_000));
        assert_eq!(limits.budget.calls, Some(0));
        assert_eq!(limits.budget.tokens, Some(9));
        assert_eq!(first.max_batch_tokens, 32_000);
        assert_eq!(first.max_request_tokens(), 32_000);
        assert_eq!(first.max_batch_items, 64);
        assert_eq!(first.max_doc_tokens, 2048);
        assert_eq!(first.concurrency, 2);
        assert_eq!(
            first.identity.endpoint,
            "https://example.test/v1/embeddings"
        );
        config.embedder.remote.max_batch_tokens = 1_000;
        config.embedder.remote.tpm = 500;
        assert_eq!(resolve(&config, UnitKind::Chunk)?.max_request_tokens(), 500);
        config.embedder.remote.api_key_env = "A_DIFFERENT_VARIABLE_NAME".into();
        config.embedder.remote.model = "other-model".into();
        assert_eq!(
            first.identity.provider,
            resolve(&config, UnitKind::Chunk)?.identity.provider
        );
        config.embedder.remote.url = "https://other.test/v1".into();
        assert_ne!(
            first.identity.provider,
            resolve(&config, UnitKind::Chunk)?.identity.provider
        );
        assert!(
            first
                .limits(usage::Budget {
                    calls: Some(-1),
                    tokens: None
                })
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn invalid_endpoints_zero_execution_bounds_and_overflows_fail_closed() {
        for url in [
            "file:///tmp/model",
            "https://user:password@example.test/v1",
            "https://example.test/v1?key=value",
            "https://example.test/v1#fragment",
        ] {
            let mut config = Config::default();
            config.embedder.remote.url = url.into();
            assert!(resolve(&config, UnitKind::Chunk).is_err());
        }
        for field in 0..4 {
            let mut config = Config::default();
            match field {
                0 => config.embedder.remote.concurrency = 0,
                1 => config.embedder.remote.max_batch_items = 0,
                2 => config.embedder.remote.tpm = u64::MAX,
                _ => config.embedder.max_doc_tokens = 0,
            }
            assert!(resolve(&config, UnitKind::Chunk).is_err());
        }
    }
}
