//! Canonical embedding job identities; see `v1.md` §7.1 and §8.2.

use super::{CurrentInput, Enqueue, Error, Kind, Target};
use crate::{models::EmbedInput, store::UnitKind, treat::context::EmbeddingInput};

/// Credential-free destination identity, including pinned local artifact identity.
/// Scheduling limits and query-only instructions do not change document vectors.
#[derive(Clone)]
pub(crate) struct Destination {
    pub(crate) provider: String,
    pub(crate) endpoint: String,
    pub(crate) model: String,
    pub(crate) artifact_hash: Option<[u8; 32]>,
    pub(crate) dims: u64,
}

/// Authoritative effective policy/configuration digest supplied by the resolver.
/// The digest must include transitive context provenance, not just the target file.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Policy {
    pub(crate) fingerprint: [u8; 32],
    pub(crate) max_level: i64,
    pub(crate) never_send: bool,
}

/// One occurrence's content identity, distinct from identical content elsewhere.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Occurrence {
    pub(crate) id: i64,
    pub(crate) content_hash: [u8; 32],
    pub(crate) kind: UnitKind,
}

/// Owned canonical inputs for blocking preparation and async dispatch.
/// Intentionally has no Debug implementation: source text is not diagnostic data.
#[derive(Clone)]
pub(crate) struct Prepared {
    occurrence: Occurrence,
    context: EmbeddingInput,
    destination: Destination,
    desired_hash: [u8; 32],
    context_hash: [u8; 32],
    role_free_hash: [u8; 32],
    tokens: i64,
}

fn digest(domain: &str, parts: &[&[u8]]) -> [u8; 32] {
    let mut hash = blake3::Hasher::new_derive_key(domain);
    for part in parts {
        // A fixed-width length separates fields even when inputs contain delimiters.
        hash.update(&(part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    *hash.finalize().as_bytes()
}

/// Hashes the exact canonical header shared by publication and status bookkeeping.
pub(crate) fn context_hash(context: &EmbeddingInput, kind: UnitKind) -> [u8; 32] {
    digest(
        "dkc embedding context v1",
        &[context.as_embed_input(kind).header.as_bytes()],
    )
}

/// Canonical publication identity without scheduling token estimates.
/// Read-only freshness checks need these hashes, never an enqueueable job.
pub(crate) struct Identity {
    target_hash: [u8; 32],
    context_hash: [u8; 32],
    desired_hash: [u8; 32],
    role_free_hash: [u8; 32],
}
impl Identity {
    /// Hashes the exact canonical input and policy using the dispatch contract.
    /// Source authority must still be verified by the caller before and after use.
    pub(crate) fn new(
        occurrence: Occurrence,
        context: &EmbeddingInput,
        destination: &Destination,
        policy: Policy,
    ) -> Result<Option<Self>, Error> {
        if policy.never_send || policy.max_level < 2 {
            return Ok(None);
        }
        if occurrence.id <= 0
            || destination.provider.is_empty()
            || destination.model.is_empty()
            || destination.dims == 0
        {
            return Err(Error::Invalid("embedding identity"));
        }
        let input = context.as_embed_input(occurrence.kind);
        let context_hash = context_hash(context, occurrence.kind);
        let artifact = destination
            .artifact_hash
            .as_ref()
            .map(|hash| hash.as_slice())
            .unwrap_or_default();
        let generation = |input: EmbedInput<'_>| {
            digest(
                "dkc embedding desired generation v1",
                &[
                    &occurrence.id.to_le_bytes(),
                    &occurrence.content_hash,
                    occurrence.kind.as_str().as_bytes(),
                    input.header.as_bytes(),
                    input.text.as_bytes(),
                    destination.provider.as_bytes(),
                    destination.endpoint.as_bytes(),
                    destination.model.as_bytes(),
                    artifact,
                    &destination.dims.to_le_bytes(),
                    &policy.fingerprint,
                    &policy.max_level.to_le_bytes(),
                    &[u8::from(policy.never_send)],
                ],
            )
        };
        let desired_hash = generation(input);
        let role_free_hash = generation(context.as_role_free_embed_input(occurrence.kind));
        Ok(Some(Self {
            target_hash: occurrence.content_hash,
            context_hash,
            desired_hash,
            role_free_hash,
        }))
    }

    /// Returns the same context key stored beside a published vector.
    pub(crate) fn context_hash(&self) -> &[u8; 32] {
        &self.context_hash
    }

    /// Returns the content and desired keys needed for a freshness comparison.
    pub(crate) fn current(&self) -> CurrentInput {
        CurrentInput {
            target_hash: self.target_hash,
            desired_hash: self.desired_hash,
        }
    }
}

impl Prepared {
    /// Derives a full desired generation from the exact provider input and policy.
    /// Returns None for L0/L1-only or protected content before estimating tokens.
    /// This does not verify source freshness: reconstruct under the dispatch and
    /// completion resolvers rather than treating a prepared snapshot as authority.
    pub(crate) fn new(
        occurrence: Occurrence,
        context: EmbeddingInput,
        destination: Destination,
        policy: Policy,
    ) -> Result<Option<Self>, Error> {
        let Some(identity) = Identity::new(occurrence, &context, &destination, policy)? else {
            return Ok(None);
        };
        let input = context.as_embed_input(occurrence.kind);
        let rendered = format!("{}\n{}", input.header, input.text);
        let tokens = i64::try_from(
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(&rendered)
                .len(),
        )
        .map_err(|_| Error::Invalid("embedding token estimate"))?;
        Ok(Some(Self {
            occurrence,
            context,
            destination,
            desired_hash: identity.desired_hash,
            context_hash: identity.context_hash,
            role_free_hash: identity.role_free_hash,
            tokens,
        }))
    }

    /// Borrows the same canonical input whose bytes identify the desired generation.
    pub(crate) fn input(&self) -> EmbedInput<'_> {
        self.context.as_embed_input(self.occurrence.kind)
    }

    /// Returns context identity for the vector row's staleness check.
    pub(crate) fn context_hash(&self) -> &[u8; 32] {
        &self.context_hash
    }

    /// Identifies every embedding input except generated directory roles. A
    /// matching persisted key can defer role-only refresh without hiding edits.
    pub(crate) fn role_free_hash(&self) -> &[u8; 32] {
        &self.role_free_hash
    }

    /// Returns freshly resolved identity for admission/completion comparison.
    pub(crate) fn current(&self) -> CurrentInput {
        CurrentInput {
            target_hash: self.occurrence.content_hash,
            desired_hash: self.desired_hash,
        }
    }

    /// Builds ordinary scheduling metadata with the specified rank/recency weights.
    /// Capacity admission and source validation remain the scheduler's responsibility.
    pub(crate) fn enqueue_ranked(
        &self,
        rank_pct: f64,
        recency: super::priority::Recency,
    ) -> Result<Enqueue<'_>, Error> {
        Ok(self.enqueue(super::priority::calculate(Kind::Embed, rank_pct, recency)?))
    }

    /// Builds enqueue metadata without allowing scheduling priority to alter identity.
    /// Explicit scores also support the separate low-priority context-refresh path.
    pub(crate) fn enqueue(&self, priority: f64) -> Enqueue<'_> {
        Enqueue {
            kind: Kind::Embed,
            target: Target::Unit,
            target_id: self.occurrence.id,
            target_hash: &self.occurrence.content_hash,
            desired_hash: &self.desired_hash,
            provider: &self.destination.provider,
            priority,
            tokens: self.tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue,
        store::{self, StoreLimits},
        treat::context::{self, ContextFile, ContextUnit},
    };

    fn context(path: &str, text: &str) -> EmbeddingInput {
        context::build(
            ContextUnit {
                scope_path: None,
                header: "fn example()",
                language: Some("rust"),
                kind: "chunk",
                text,
                is_document: false,
            },
            ContextFile {
                repo_name: "fixture",
                path,
            },
            &[],
        )
    }
    fn destination() -> Destination {
        Destination {
            provider: "embed:remote".into(),
            endpoint: "https://example.test/v1".into(),
            model: "fixture".into(),
            artifact_hash: None,
            dims: 1024,
        }
    }
    fn policy() -> Policy {
        Policy {
            fingerprint: [3; 32],
            max_level: 2,
            never_send: false,
        }
    }
    fn occurrence() -> Occurrence {
        Occurrence {
            id: 1,
            content_hash: [1; 32],
            kind: UnitKind::Chunk,
        }
    }
    fn prepared() -> Prepared {
        Prepared::new(
            occurrence(),
            context("a.rs", "code"),
            destination(),
            policy(),
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn role_free_identity_defers_only_roles_and_preserves_all_other_inputs() {
        let input = |path: &str, role: Option<&str>, text: &str| {
            context::build(
                ContextUnit {
                    scope_path: None,
                    header: "fn example()",
                    language: Some("rust"),
                    kind: "chunk",
                    text,
                    is_document: false,
                },
                ContextFile {
                    repo_name: "fixture",
                    path,
                },
                &[context::DirectoryContext {
                    path: "billing",
                    role,
                }],
            )
        };
        let prepare = |path: &str, role, text: &str, destination, policy| {
            Prepared::new(occurrence(), input(path, role, text), destination, policy)
                .unwrap()
                .unwrap()
        };
        let base = prepare("billing/a.rs", None, "code", destination(), policy());
        let role = prepare(
            "billing/a.rs",
            Some("payment retries"),
            "code",
            destination(),
            policy(),
        );
        assert_eq!(*base.role_free_hash(), base.current().desired_hash);
        assert_eq!(base.role_free_hash(), role.role_free_hash());
        assert_ne!(base.current().desired_hash, role.current().desired_hash);
        assert_ne!(base.context_hash(), role.context_hash());
        let changed_role = prepare(
            "billing/a.rs",
            Some("invoice processing"),
            "code",
            destination(),
            policy(),
        );
        assert_eq!(role.role_free_hash(), changed_role.role_free_hash());
        assert_ne!(
            role.current().desired_hash,
            changed_role.current().desired_hash
        );
        for changed in [
            prepare(
                "billing/b.rs",
                Some("payment retries"),
                "code",
                destination(),
                policy(),
            ),
            prepare(
                "billing/a.rs",
                Some("payment retries"),
                "changed code",
                destination(),
                policy(),
            ),
            prepare(
                "billing/a.rs",
                Some("payment retries"),
                "code",
                Destination {
                    model: "changed".into(),
                    ..destination()
                },
                policy(),
            ),
            prepare(
                "billing/a.rs",
                Some("payment retries"),
                "code",
                destination(),
                Policy {
                    fingerprint: [9; 32],
                    ..policy()
                },
            ),
        ] {
            assert_ne!(role.role_free_hash(), changed.role_free_hash());
        }
    }

    #[test]
    fn read_only_identity_preserves_the_published_history_hash_contract() {
        let occurrence = Occurrence {
            id: 7,
            content_hash: [1; 32],
            kind: UnitKind::History,
        };
        let context = context::history_input(
            "commit › 2026-09-10 › fixture",
            "sample",
            "history",
            "body",
            context::Options::default(),
        );
        let identity = Identity::new(occurrence, &context, &destination(), policy())
            .unwrap()
            .unwrap();
        // Fixed from the pre-refactor 67e47ff contract, independently of Identity.
        assert_eq!(
            blake3::Hash::from_bytes(*identity.context_hash())
                .to_hex()
                .as_str(),
            "1705114154344c45278944e61fada2696edde914015ef759d66dda6cf6bc44ca"
        );
        assert_eq!(
            blake3::Hash::from_bytes(identity.current().desired_hash)
                .to_hex()
                .as_str(),
            "5b14ec9faf2a31c00d188d0e4396acb292089c710aaaecbe599e020c988830a8"
        );
        assert_eq!(identity.current().target_hash, [1; 32]);
        let prepared = Prepared::new(occurrence, context, destination(), policy())
            .unwrap()
            .unwrap();
        assert_eq!(
            prepared.current().desired_hash,
            identity.current().desired_hash
        );
        assert!(prepared.enqueue(1.0).tokens > 0);
    }

    #[test]
    fn canonical_generation_reuses_done_jobs_but_context_model_and_policy_changes_enqueue()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store::open(":memory:", StoreLimits::default())?;
        let base = prepared();
        let id = queue::enqueue(
            &store,
            &base.enqueue_ranked(0.0, queue::priority::Recency::ChangedFile)?,
            0,
        )?;
        store
            .writer()?
            .execute("UPDATE jobs SET state='done' WHERE id=?1", [id])?;
        assert_eq!(
            queue::enqueue(
                &store,
                &prepared().enqueue_ranked(1.0, queue::priority::Recency::ChangedFile)?,
                1,
            )?,
            id
        );
        let (state, priority): (String, f64) = store.writer()?.query_row(
            "SELECT state,priority FROM jobs WHERE id=?1",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!((state.as_str(), priority), ("done", 2.0));
        let mut model = destination();
        model.model = "replacement".into();
        let mut changed_policy = policy();
        changed_policy.fingerprint = [4; 32];
        for next in [
            Prepared::new(
                occurrence(),
                context("renamed.rs", "code"),
                destination(),
                policy(),
            )?,
            Prepared::new(occurrence(), context("a.rs", "code"), model, policy())?,
            Prepared::new(
                occurrence(),
                context("a.rs", "code"),
                destination(),
                changed_policy,
            )?,
        ] {
            let next = next.unwrap();
            assert_ne!(queue::enqueue(&store, &next.enqueue(1.0), 2)?, id);
        }
        let count: i64 = store
            .writer()?
            .query_row("SELECT COUNT(*) FROM jobs", [], |r| r.get(0))?;
        assert_eq!(count, 4);
        Ok(())
    }

    #[test]
    fn identity_covers_occurrence_input_destination_artifact_and_effective_policy() {
        let base = prepared();
        let mut variants = Vec::new();
        for change in 0..11 {
            let mut occurrence = occurrence();
            let mut destination = destination();
            let mut policy = policy();
            let mut path = "a.rs";
            let mut text = "code";
            match change {
                0 => occurrence.id = 2,
                1 => occurrence.content_hash = [2; 32],
                2 => occurrence.kind = UnitKind::DocSection,
                3 => path = "b.rs",
                4 => text = "changed",
                5 => destination.provider = "embed:ollama".into(),
                6 => destination.endpoint.push_str("/other"),
                7 => destination.model.push_str("-other"),
                8 => destination.artifact_hash = Some([0; 32]),
                9 => destination.dims = 512,
                _ => policy.max_level = 3,
            }
            let next = Prepared::new(occurrence, context(path, text), destination, policy)
                .unwrap()
                .unwrap();
            assert_ne!(base.current().desired_hash, next.current().desired_hash);
            variants.push(next.current().desired_hash);
        }
        variants.sort();
        variants.dedup();
        assert_eq!(variants.len(), 11);
        assert_ne!(
            digest("test", &[b"ab", b"c"]),
            digest("test", &[b"a", b"bc"])
        );
        let input = base.input();
        assert_eq!(
            base.enqueue(1.0).tokens as usize,
            tiktoken_rs::cl100k_base_singleton()
                .encode_with_special_tokens(&format!("{}\n{}", input.header, input.text))
                .len()
        );
        let edited = Prepared::new(
            occurrence(),
            context("a.rs", "changed"),
            destination(),
            policy(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(base.context_hash(), edited.context_hash());
        assert_ne!(
            base.context_hash(),
            Prepared::new(
                occurrence(),
                context("renamed.rs", "code"),
                destination(),
                policy()
            )
            .unwrap()
            .unwrap()
            .context_hash()
        );
    }

    #[test]
    fn protected_and_l1_only_inputs_produce_no_embedding_job() {
        for policy in [
            Policy {
                never_send: true,
                ..policy()
            },
            Policy {
                max_level: 1,
                ..policy()
            },
        ] {
            assert!(
                Prepared::new(
                    occurrence(),
                    context("a.rs", "protected"),
                    destination(),
                    policy
                )
                .unwrap()
                .is_none()
            );
        }
        let mut bad = destination();
        bad.dims = 0;
        assert!(Prepared::new(occurrence(), context("a.rs", "code"), bad, policy()).is_err());
    }
}
