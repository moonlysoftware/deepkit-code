//! Shared lexical/graph/dense retrieval, rerank and cached collapse; v1 §9 steps 1–8.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "public transport wiring follows in T1.7–8")
)]
use super::{
    Error, RankedCandidate, collapse, dense, embedding_query, fusion, lexical_leg_checked,
    materialize, query, rerank,
};
use crate::{config::Config, store::Store};
#[cfg(test)]
use std::time::Duration;
use std::{collections::BTreeSet, sync::Arc};
use tokio::time::Instant;

type Publications =
    std::collections::BTreeMap<i64, Option<crate::store::projection::SourcePublication>>;

/// One leg's original hydration, before a later leg can replace a shared ID.
/// Parent-chain and source changes remain observable even with identical ranks.
pub(crate) struct LegEvidence {
    pub(crate) occurrences: Vec<crate::store::StoredUnit>,
    pub(crate) links: std::collections::BTreeMap<i64, Vec<collapse::ProjectionLink>>,
    pub(crate) publications: Publications,
}
impl LegEvidence {
    fn capture(prepared: &collapse::Prepared) -> Result<Self, Error> {
        Ok(Self {
            occurrences: prepared.occurrence_evidence(&|| Ok(()))?,
            links: prepared.links.clone(),
            publications: prepared.publications.clone(),
        })
    }
}

/// Transport-neutral search inputs; IDs are resolved by the transport adapter.
#[derive(Clone)]
pub(crate) struct Request {
    pub(crate) query: String,
    pub(crate) scope: Option<crate::store::SearchScope>,
    pub(crate) k: usize,
    pub(crate) mode: fusion::Mode,
    pub(crate) exclude_ids: Vec<i64>,
}
/// One explicitly prepared active tier; automatic routing and transition fallback
/// are supplied by the hosting layer, not inferred from unrelated vector tables.
pub(crate) struct Tier {
    pub(crate) factory: Arc<embedding_query::Factory>,
    pub(crate) table: String,
}
impl Tier {
    /// Selects the configured host table. Active model/profile/dimension identity
    /// is rechecked inside dense retrieval before dispatch and in the scan snapshot.
    pub(crate) fn configured(factory: Arc<embedding_query::Factory>) -> Self {
        Self {
            table: factory.table_slug(),
            factory,
        }
    }
}
/// Explicit dense-leg outcome permits lexical fallback without hiding failures.
#[derive(Debug)]
pub(crate) enum DenseState {
    NotRequested,
    Unavailable,
    Complete,
    Failed(dense::Error),
    /// One or more requested spaces failed; successful lists still contribute.
    Partial(Vec<SpaceFailure>),
}
/// Identifies lost dense work without exposing query text or provider responses.
#[derive(Debug)]
pub(crate) struct SpaceFailure {
    pub(crate) table: String,
    pub(crate) error: dense::Error,
}

enum Adaptive<'a> {
    Ready(u64),
    Lazy(&'a rerank::lazy::Host),
}
enum Retrieval {
    Single(Tier),
    Routed(Box<dense::routed::Providers>),
}
#[derive(Default, Debug, PartialEq)]
pub(crate) struct DenseLists {
    pub(crate) single_table: Option<String>,
    pub(crate) single: Vec<i64>,
    pub(crate) spaces: Vec<(String, Vec<i64>)>,
}
/// Lexical/graph retrieval outcome. A completed empty list is distinct from a
/// failed read, and a typed dense-only request does not count as degradation.
#[derive(Debug)]
pub(crate) enum LexicalState {
    NotRequested,
    Complete,
    Failed(Error),
}
/// Fused candidate window, ready for rerank and collapse rather than a final k-hit
/// response. Contributions and occurrence metadata use the shared result shape.
pub(crate) struct Candidates {
    pub(crate) hits: Vec<RankedCandidate>,
    pub(crate) dense: DenseState,
    pub(crate) lexical: LexicalState,
    prepared: collapse::Prepared,
    selection_leg: super::LexicalLeg,
    selection_projections: Vec<crate::store::StoredUnit>,
    selection_legs: [LegEvidence; 2],
    dense_selection: DenseLists,
    pub(crate) rerank: rerank::State,
    pub(crate) rerank_auto: Option<rerank::auto::Status>,
    store: Arc<Store>,
    config: Config,
    query: String,
    control: super::control::Control,
    // Standalone calls own their budget through reranking; hosted calls borrow it.
    _owner: Option<super::control::Owner>,
}

/// Final results with explicit retrieval and rerank outcomes. Lost work remains
/// visible to response adapters even when fallback returns an empty result set.
pub(crate) struct Results {
    /// Original local authority, including candidates hidden by ranking/collapse.
    /// This is evidence for validation, not a claim of current source freshness.
    pub(crate) selection: super::Selection,
    /// Ordered dense contributors before fusion truncation, paired with their
    /// spaces. Validation must not substitute newly embedded query results.
    pub(crate) dense_selection: DenseLists,
    /// Independent lexical/dense snapshots before merge. A shared file or parent
    /// changing between legs must not be disguised by the later hydration.
    pub(crate) selection_legs: [LegEvidence; 2],
    pub(crate) rerank: rerank::State,
    pub(crate) rerank_auto: Option<rerank::auto::Status>,
    pub(crate) hits: Vec<RankedCandidate>,
    pub(crate) dense: DenseState,
    pub(crate) lexical: LexicalState,
}

impl Results {
    /// Reports lost requested retrieval work for the transport's index.degraded
    /// field. Successful zero-hit lists and intentionally disabled legs are healthy.
    pub(crate) fn degraded(&self) -> bool {
        self.rerank.fallback().is_some()
            || matches!(self.lexical, LexicalState::Failed(_))
            || matches!(
                self.dense,
                DenseState::Unavailable | DenseState::Failed(_) | DenseState::Partial(_)
            )
    }
}

/// Retrieves and collapses the configured candidate window without reranking.
/// Parent hydration shares each leg's deadline; final grouping requires no I/O,
/// so a timed-out embedding cannot delay the already prepared lexical fallback.
/// Hosting supplies the selected provider and explicit admission budget via its
/// factory; retrieval derives its canonical host table instead of accepting a name.
pub(crate) async fn search_without_rerank<C>(
    store: Arc<Store>,
    config: Config,
    request: Request,
    factory: Option<Arc<embedding_query::Factory>>,
    clock: C,
) -> Result<Results, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    let k = request.k;
    let settings = config.search.clone();
    candidates(store, config, request, factory.map(Tier::configured), clock)
        .await?
        .finish(k, &settings)
}

/// Produces final unreranked results across prepared routing spaces. The same
/// lexical/graph, RRF and cached collapse stages serve single and routed queries.
pub(crate) async fn search_routed_without_rerank<C>(
    providers: dense::routed::Providers,
    request: Request,
    clock: C,
) -> Result<Results, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    let k = request.k;
    let settings = providers.config.search.clone();
    candidates_routed(providers, request, clock)
        .await?
        .finish(k, &settings)
}

/// Retains routed per-space ranks and failures for the shared rerank boundary.
pub(crate) async fn candidates_routed<C>(
    providers: dense::routed::Providers,
    request: Request,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    candidates_with_retrieval(
        Arc::clone(&providers.store),
        providers.config.clone(),
        request,
        Some(Retrieval::Routed(Box::new(providers))),
        None,
        clock,
        None,
    )
    .await
}

impl Candidates {
    /// Waits for the retained host only within this window's original deadline.
    /// Failed or unfinished preparation keeps the whole RRF window and typed status.
    pub(crate) async fn rerank_lazy<C>(mut self, host: &rerank::lazy::Host, clock: C) -> Self
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        if !matches!(self.rerank, rerank::State::NotRequested) || self.config.rerank.mode == "off" {
            return self;
        }
        if !host.matches(&self.store, &self.config) {
            self.rerank = rerank::State::failure(rerank::Error::Input);
        } else {
            use rerank::lazy::Outcome;
            self.rerank = match self
                .control
                .run(host.get(self.control.deadline()))
                .await
                .unwrap_or_else(|_| host.peek())
            {
                Outcome::Ready(factory) => return self.rerank_with(factory, clock).await,
                Outcome::Pending => rerank::State::Preparing,
                Outcome::Idle => rerank::State::Timeout,
                Outcome::Failed(error) => match error.as_ref() {
                    rerank::Error::Privacy => rerank::State::Privacy,
                    rerank::Error::Deadline => rerank::State::Timeout,
                    _ => rerank::State::PreparationFailed(error),
                },
                Outcome::Off => rerank::State::failure(rerank::Error::Input),
            };
        }
        self.annotate_rerank_fallback();
        self
    }

    /// Reranks this complete window once, within its original retrieval deadline.
    /// Any refusal retains RRF order/scores and annotates every surviving hit;
    /// successful scoring precedes the same cached collapse used by fallback.
    pub(crate) async fn rerank_with<C>(mut self, factory: Arc<rerank::Factory>, clock: C) -> Self
    where
        C: Fn() -> i64 + Send + Sync + 'static,
    {
        if !matches!(self.rerank, rerank::State::NotRequested) || self.config.rerank.mode == "off" {
            return self;
        }
        let control = self.control.clone();
        let result = control
            .run(async {
                if !factory.matches(&self.store, &self.config) {
                    return Err(rerank::Error::Input);
                }
                rerank::validate_ranking(&self.config.rerank, &self.config.search)?;
                let units = self.hits.iter().map(|hit| hit.unit.clone()).collect();
                let prepared = factory
                    .prepare(self.query.clone(), units, self.control.deadline())
                    .await?;
                let scores = prepared.score(self.control.deadline(), clock).await?;
                if Instant::now() >= self.control.deadline() {
                    return Err(rerank::Error::Deadline);
                }
                Ok(scores)
            })
            .await
            .unwrap_or(Err(rerank::Error::Deadline));
        self.rerank_auto = Some(factory.auto_status());
        self.rerank = match result {
            Ok(scores) => match rerank::apply_scores(
                &mut self.hits,
                &scores,
                &self.config.rerank,
                &self.config.search,
            ) {
                Ok(()) => rerank::State::Complete,
                Err(error) => rerank::State::failure(error),
            },
            Err(error) => rerank::State::failure(error),
        };
        self.annotate_rerank_fallback();
        self
    }

    fn annotate_rerank_fallback(&mut self) {
        if let Some(reason) = self.rerank.fallback() {
            for hit in &mut self.hits {
                hit.rerank_fallback = Some(reason);
                for evidence in &mut hit.matches {
                    evidence.why.push(reason);
                }
            }
        }
    }

    /// Finishes a window after its caller has optionally reranked the hits. Uses
    /// the captured parents even if a concurrent index write changes the database;
    /// the subsequent freshness stage must annotate or refresh those results.
    pub(crate) fn finish(
        self,
        k: usize,
        settings: &crate::config::Search,
    ) -> Result<Results, Error> {
        let hits = collapse::finish(self.hits, k, settings, &self.prepared)?;
        let selection = super::Selection::capture(
            self.selection_leg,
            self.selection_legs
                .iter()
                .flat_map(|leg| leg.occurrences.iter().cloned())
                .collect(),
            self.selection_projections,
            self.prepared,
            &hits,
            &|| Ok(()),
        )?;
        Ok(Results {
            selection,
            dense_selection: self.dense_selection,
            selection_legs: self.selection_legs,
            rerank: self.rerank,
            rerank_auto: self.rerank_auto,
            hits,
            dense: self.dense,
            lexical: self.lexical,
        })
    }
}

/// Runs independent retrieval legs concurrently, then fuses their ranked IDs once
/// with the configured candidate limit. Typed clauses control which lists exist;
/// the requested natural-query mode cannot suppress an explicitly typed clause.
/// Failed retrieval legs contribute no votes; their explicit outcomes accompany
/// already hydrated results from surviving legs, without another database read.
/// Read-only blocking work may finish after cancellation, but cannot dispatch or
/// publish anything.
pub(crate) async fn candidates<C>(
    store: Arc<Store>,
    config: Config,
    request: Request,
    tier: Option<Tier>,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    candidates_with_retrieval(
        store,
        config,
        request,
        tier.map(Retrieval::Single),
        None,
        clock,
        None,
    )
    .await
}

/// Selects with the live or saved adaptive bound without waiting for weights.
/// Call `rerank_lazy` on the window to start/wait once retrieval has useful fallback.
pub(crate) async fn candidates_for_lazy<C>(
    store: Arc<Store>,
    config: Config,
    request: Request,
    tier: Option<Tier>,
    host: &rerank::lazy::Host,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if !host.matches(&store, &config) {
        return Err(Error::InvalidQuery {
            reason: "reranker belongs to another store or configuration",
        });
    }
    candidates_with_retrieval(
        store,
        config,
        request,
        tier.map(Retrieval::Single),
        Some(Adaptive::Lazy(host)),
        clock,
        None,
    )
    .await
}

/// Restores the same lazy adaptive selection across desired/previous routed spaces.
pub(crate) async fn candidates_routed_for_lazy<C>(
    providers: dense::routed::Providers,
    request: Request,
    host: &rerank::lazy::Host,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if !host.matches(&providers.store, &providers.config) {
        return Err(Error::InvalidQuery {
            reason: "reranker belongs to another store or configuration",
        });
    }
    candidates_with_retrieval(
        Arc::clone(&providers.store),
        providers.config.clone(),
        request,
        Some(Retrieval::Routed(Box::new(providers))),
        Some(Adaptive::Lazy(host)),
        clock,
        None,
    )
    .await
}

/// Applies the host's effective bound during fusion, before selecting or checking
/// the candidate set. An already selected in-flight window is never truncated.
pub(crate) async fn candidates_for_rerank<C>(
    store: Arc<Store>,
    config: Config,
    request: Request,
    tier: Option<Tier>,
    factory: &Arc<rerank::Factory>,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if !factory.matches(&store, &config) {
        return Err(Error::InvalidQuery {
            reason: "reranker belongs to another store or configuration",
        });
    }
    candidates_with_retrieval(
        store,
        config,
        request,
        tier.map(Retrieval::Single),
        Some(Adaptive::Ready(factory.auto_status().effective_candidates)),
        clock,
        None,
    )
    .await
}

/// Uses the same adaptive selection for desired/previous routed vector spaces.
pub(crate) async fn candidates_routed_for_rerank<C>(
    providers: dense::routed::Providers,
    request: Request,
    factory: &Arc<rerank::Factory>,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if !factory.matches(&providers.store, &providers.config) {
        return Err(Error::InvalidQuery {
            reason: "reranker belongs to another store or configuration",
        });
    }
    candidates_with_retrieval(
        Arc::clone(&providers.store),
        providers.config.clone(),
        request,
        Some(Retrieval::Routed(Box::new(providers))),
        Some(Adaptive::Ready(factory.auto_status().effective_candidates)),
        clock,
        None,
    )
    .await
}

/// Borrows one service invocation's deadline/cancellation across single-tier work.
/// Optional lazy reranking uses the same adaptive window without starting weights.
pub(crate) async fn candidates_with_control<C>(
    store: Arc<Store>,
    config: Config,
    request: Request,
    tier: Option<Tier>,
    host: Option<&rerank::lazy::Host>,
    control: super::control::Control,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if host.is_some_and(|host| !host.matches(&store, &config)) {
        return Err(Error::InvalidQuery {
            reason: "reranker belongs to another store or configuration",
        });
    }
    candidates_with_retrieval(
        store,
        config,
        request,
        tier.map(Retrieval::Single),
        host.map(Adaptive::Lazy),
        clock,
        Some(control),
    )
    .await
}

/// Keeps desired/previous routing spaces inside the original service budget.
pub(crate) async fn candidates_routed_with_control<C>(
    providers: dense::routed::Providers,
    request: Request,
    host: Option<&rerank::lazy::Host>,
    control: super::control::Control,
    clock: C,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if host.is_some_and(|host| !host.matches(&providers.store, &providers.config)) {
        return Err(Error::InvalidQuery {
            reason: "reranker belongs to another store or configuration",
        });
    }
    candidates_with_retrieval(
        Arc::clone(&providers.store),
        providers.config.clone(),
        request,
        Some(Retrieval::Routed(Box::new(providers))),
        host.map(Adaptive::Lazy),
        clock,
        Some(control),
    )
    .await
}

async fn candidates_with_retrieval<C>(
    store: Arc<Store>,
    config: Config,
    request: Request,
    retrieval: Option<Retrieval>,
    adaptive: Option<Adaptive<'_>>,
    clock: C,
    control: Option<super::control::Control>,
) -> Result<Candidates, Error>
where
    C: Fn() -> i64 + Send + Sync + 'static,
{
    if request.k == 0 || request.k as u128 > u128::from(config.search.max_k) {
        return Err(Error::InvalidK {
            max: config.search.max_k,
        });
    }
    let configured_limit = usize::try_from(config.rerank.candidates)
        .ok()
        .filter(|limit| *limit > 0)
        .ok_or(Error::InvalidQuery {
            reason: "invalid candidate window",
        })?;
    if config.limits.query_timeout_ms == 0 {
        return Err(Error::InvalidQuery {
            reason: "zero query timeout",
        });
    }
    if config.search.fusion != "rrf"
        || config.search.rrf_k == 0
        || !config.search.graph_prior_weight.is_finite()
        || config.search.graph_prior_weight < 0.0
    {
        return Err(Error::InvalidFusion);
    }
    let mut parsed = query::parse(&request.query, request.mode)?;
    if parsed.dense.is_some()
        && let Some(retrieval) = &retrieval
        && match retrieval {
            Retrieval::Single(tier) => {
                !Arc::ptr_eq(&store, tier.factory.store())
                    || !tier.factory.matches_configuration(&config)
            }
            Retrieval::Routed(providers) => providers.factories.iter().any(|factory| {
                !Arc::ptr_eq(&store, factory.store()) || !factory.matches_configuration(&config)
            }),
        }
    {
        return Err(Error::InvalidQuery {
            reason: "dense tier belongs to another store or configuration",
        });
    }
    let (owner, control) = match control {
        Some(control) => (None, control),
        None => {
            let owner = super::control::Owner::new(
                std::time::Instant::now(),
                config.limits.query_timeout_ms,
            )?;
            let control = owner.control();
            (Some(owner), control)
        }
    };
    let deadline = control.deadline();
    let adaptation = async {
        control
            .run(async {
                match adaptive {
                    None => Ok((configured_limit, None)),
                    Some(Adaptive::Ready(bound)) => Ok((
                        usize::try_from(bound)
                            .unwrap_or(configured_limit)
                            .min(configured_limit),
                        None,
                    )),
                    Some(Adaptive::Lazy(host)) => {
                        host.selection_status(deadline).await.map(|status| {
                            (
                                usize::try_from(status.effective_candidates)
                                    .unwrap_or(configured_limit)
                                    .min(configured_limit),
                                Some(status),
                            )
                        })
                    }
                }
            })
            .await
            .map_err(|_| rerank::Error::Deadline)?
    };
    let candidate_store = Arc::clone(&store);
    let dense_text = parsed.dense.take();
    let lexical_store = Arc::clone(&store);
    let lexical_settings = config.search.clone();
    let lexical_scope = request.scope.clone();
    let lexical_excludes = request.exclude_ids.clone();
    let lexical_requested = parsed.lexical.is_some();
    let lexical = async {
        if !lexical_requested {
            return Ok((super::LexicalLeg::default(), collapse::Prepared::default()));
        }
        let lexical_control = control.clone();
        control
            .run(tokio::task::spawn_blocking(move || {
                let check = || lexical_control.check();
                check()?;
                let leg = lexical_leg_checked(
                    &lexical_store,
                    &parsed,
                    request.k,
                    lexical_scope.as_ref(),
                    &lexical_settings,
                    &lexical_excludes,
                    Some(&check),
                )?;
                let ids = leg
                    .ordered
                    .iter()
                    .chain(&leg.graph)
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let units = collapse::prepare_with_work(
                    &lexical_store,
                    &ids,
                    &lexical_excludes.into_iter().collect(),
                    Some(&check),
                )?;
                check()?;
                Ok((leg, units))
            }))
            .await
            .map_err(|_| Error::Deadline)?
            .map_err(|_| Error::QueryTask)?
    };
    let dense_scope =
        dense::Scope::from_search(request.scope.as_ref(), request.exclude_ids.clone());
    let dense_excludes = request.exclude_ids.iter().copied().collect();
    let dense = async {
        let Some(text) = dense_text else {
            return (
                DenseLists::default(),
                collapse::Prepared::default(),
                DenseState::NotRequested,
            );
        };
        let Some(retrieval) = retrieval else {
            return (
                DenseLists::default(),
                collapse::Prepared::default(),
                DenseState::Unavailable,
            );
        };
        let result = control
            .run(async {
                let (lists, state) = match retrieval {
                    Retrieval::Single(tier) => {
                        let table = tier.table.clone();
                        let hits = dense::retrieve(
                            tier.factory,
                            tier.table,
                            text,
                            dense_scope,
                            deadline,
                            clock,
                        )
                        .await?;
                        (
                            DenseLists {
                                single_table: Some(table),
                                single: hits.into_iter().map(|hit| hit.unit_id).collect(),
                                spaces: Vec::new(),
                            },
                            DenseState::Complete,
                        )
                    }
                    Retrieval::Routed(providers) => {
                        let result =
                            dense::routed::retrieve(*providers, text, dense_scope, deadline, clock)
                                .await?;
                        let mut lists = DenseLists::default();
                        let mut failures = Vec::new();
                        for space in result.spaces {
                            match space.hits {
                                Ok(hits) => lists.spaces.push((
                                    space.table,
                                    hits.into_iter().map(|hit| hit.unit_id).collect(),
                                )),
                                Err(error) => failures.push(SpaceFailure {
                                    table: space.table,
                                    error,
                                }),
                            }
                        }
                        let state = if failures.is_empty() {
                            DenseState::Complete
                        } else {
                            DenseState::Partial(failures)
                        };
                        (lists, state)
                    }
                };
                let hydrate_ids = lists
                    .single
                    .iter()
                    .chain(lists.spaces.iter().flat_map(|(_, ids)| ids))
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let hydration_control = control.clone();
                let units = control
                    .run(tokio::task::spawn_blocking(move || {
                        hydration_control
                            .check()
                            .map_err(|_| dense::Error::Deadline)?;
                        collapse::prepare_with_work(
                            &store,
                            &hydrate_ids,
                            &dense_excludes,
                            Some(&|| hydration_control.check()),
                        )
                        .map_err(dense::Error::Projection)
                    }))
                    .await
                    .map_err(|_| dense::Error::Deadline)?
                    .map_err(|_| dense::Error::Task)??;
                if Instant::now() >= deadline {
                    return Err(dense::Error::Deadline);
                }
                Ok::<_, dense::Error>((lists, units, state))
            })
            .await
            .unwrap_or(Err(dense::Error::Deadline));
        match result {
            Ok((lists, units, state)) => (lists, units, state),
            Err(error) => (
                DenseLists::default(),
                collapse::Prepared::default(),
                DenseState::Failed(error),
            ),
        }
    };
    let (lexical, (dense_ids, dense_units, dense), adaptation) =
        tokio::join!(lexical, dense, adaptation);
    let (limit, rerank_auto, rerank) = match adaptation {
        Ok((limit, status)) => (limit, status, rerank::State::NotRequested),
        Err(error) => (configured_limit, None, rerank::State::failure(error)),
    };
    let (lexical, lexical_units, lexical_state) = match lexical {
        Ok((leg, units)) => (
            leg,
            units,
            if lexical_requested {
                LexicalState::Complete
            } else {
                LexicalState::NotRequested
            },
        ),
        Err(error) => (
            super::LexicalLeg::default(),
            collapse::Prepared::default(),
            LexicalState::Failed(error),
        ),
    };
    let spaces = dense_ids
        .spaces
        .iter()
        .map(|(table, ids)| fusion::DenseList { table, ids })
        .collect::<Vec<_>>();
    let fused = fusion::merge_spaces(
        fusion::Lists {
            lexical: &lexical.ordered,
            dense: &dense_ids.single,
            graph: &lexical.graph,
        },
        &spaces,
        fusion::Settings {
            mode: fusion::Mode::Hybrid,
            rrf_k: config.search.rrf_k,
            graph_weight: config.search.graph_prior_weight,
            exclude: &request.exclude_ids.into_iter().collect(),
            limit,
        },
    )?;
    // Capture each leg before merge replaces matching IDs, then the effective
    // projection before materialize consumes its unit map. Ladder progress alone
    // is excluded from all of these L1 identities.
    let selection_legs = [
        LegEvidence::capture(&lexical_units)?,
        LegEvidence::capture(&dense_units)?,
    ];
    let mut prepared = lexical_units;
    prepared.extend(dense_units);
    let selection_projections = prepared.occurrence_evidence(&|| Ok(()))?;
    let mut result = Candidates {
        rerank,
        rerank_auto,
        store: candidate_store,
        config,
        query: request.query,
        control,
        _owner: owner,
        hits: materialize(fused, &lexical.exact, &mut prepared.units),
        selection_leg: lexical,
        selection_projections,
        selection_legs,
        dense_selection: dense_ids,
        dense,
        lexical: lexical_state,
        prepared,
    };
    result.annotate_rerank_fallback();
    Ok(result)
}

mod validation;

#[cfg(test)]
mod tests;
