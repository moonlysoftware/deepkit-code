//! Reciprocal-rank fusion shared by search transports; v1 §9 steps 6/10.
use super::Error;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Natural-query retrieval mode; typed queries may override their selected legs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Mode {
    Lexical,
    Dense,
    Hybrid,
}
/// Ranked, occurrence-ID lists. Each leg contributes at most once per occurrence.
#[derive(Default)]
pub(crate) struct Lists<'a> {
    pub(crate) lexical: &'a [i64],
    pub(crate) dense: &'a [i64],
    pub(crate) graph: &'a [i64],
}
/// One independently ranked vector space, before any fusion or truncation.
pub(crate) struct DenseList<'a> {
    pub(crate) table: &'a str,
    pub(crate) ids: &'a [i64],
}

/// Fusion controls applied before the candidate bound, independent of transport.
pub(crate) struct Settings<'a> {
    pub(crate) mode: Mode,
    pub(crate) rrf_k: u64,
    pub(crate) graph_weight: f64,
    pub(crate) exclude: &'a BTreeSet<i64>,
    pub(crate) limit: usize,
}
/// A one-based list position and its contribution to the total RRF score.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Contribution {
    pub(crate) rank: usize,
    pub(crate) score: f64,
}
/// Retained evidence for verbose output and later reranking/collapse.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Scores {
    pub(crate) lexical: Option<Contribution>,
    /// Strongest individual dense contribution, retained for single-leg consumers.
    pub(crate) dense: Option<Contribution>,
    /// Full routed evidence; each named space contributes independently once.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) dense_spaces: BTreeMap<String, Contribution>,
    pub(crate) graph: Option<Contribution>,
}
/// A unique occurrence, sorted by total RRF score then stable numeric ID.
#[derive(Debug)]
pub(crate) struct Hit {
    pub(crate) id: i64,
    pub(crate) score: f64,
    pub(crate) scores: Scores,
}

/// Merges active ranked legs, removes excluded IDs before truncation and preserves
/// diagnostic contributions. Repeated IDs cannot add votes or displace ranks;
/// excluded entries retain their original list positions for other candidates.
pub(crate) fn merge(lists: Lists<'_>, settings: Settings<'_>) -> Result<Vec<Hit>, Error> {
    merge_spaces(lists, &[], settings)
}

/// Fuses lexical, graph and each named dense space directly, preserving original
/// per-space ranks. Rejects duplicate names or mixing named and anonymous dense
/// lists so callers cannot accidentally add the same dense retrieval twice.
pub(crate) fn merge_spaces(
    lists: Lists<'_>,
    spaces: &[DenseList<'_>],
    settings: Settings<'_>,
) -> Result<Vec<Hit>, Error> {
    let names = spaces
        .iter()
        .map(|space| space.table)
        .collect::<BTreeSet<_>>();
    if spaces.len() > 4
        || names.len() != spaces.len()
        || names.contains("")
        || (!spaces.is_empty() && !lists.dense.is_empty())
    {
        return Err(Error::InvalidFusion);
    }
    if settings.rrf_k == 0 || !settings.graph_weight.is_finite() || settings.graph_weight < 0.0 {
        return Err(Error::InvalidFusion);
    }
    let mut hits = BTreeMap::<i64, Hit>::new();
    for (leg, ids, weight, table) in [
        (0, lists.lexical, 1.0, None),
        (1, lists.dense, 1.0, None),
        (2, lists.graph, settings.graph_weight, None),
    ]
    .into_iter()
    .chain(
        spaces
            .iter()
            .map(|space| (1, space.ids, 1.0, Some(space.table))),
    ) {
        if weight == 0.0
            || (settings.mode == Mode::Lexical && leg == 1)
            || (settings.mode == Mode::Dense && leg != 1)
        {
            continue;
        }
        let mut seen = BTreeSet::new();
        let mut rank = 0;
        for &id in ids {
            if !seen.insert(id) {
                continue;
            }
            rank += 1;
            if settings.exclude.contains(&id) {
                continue;
            }
            let score = weight / (settings.rrf_k as f64 + rank as f64);
            let hit = hits.entry(id).or_insert_with(|| Hit {
                id,
                score: 0.0,
                scores: Scores::default(),
            });
            hit.score += score;
            let contribution = Some(Contribution { rank, score });
            match leg {
                0 => hit.scores.lexical = contribution,
                1 => {
                    if hit.scores.dense.is_none_or(|old| old.score < score) {
                        hit.scores.dense = contribution;
                    }
                    if let Some(table) = table {
                        hit.scores
                            .dense_spaces
                            .insert(table.into(), Contribution { rank, score });
                    }
                }
                _ => hit.scores.graph = contribution,
            }
        }
    }
    let mut hits = hits.into_values().collect::<Vec<_>>();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    hits.truncate(settings.limit);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn settings(exclude: &BTreeSet<i64>, mode: Mode, limit: usize) -> Settings<'_> {
        Settings {
            mode,
            rrf_k: 60,
            graph_weight: 0.3,
            exclude,
            limit,
        }
    }
    #[test]
    fn hand_computed_three_leg_scores_and_diagnostics() -> Result<(), Error> {
        let excluded = BTreeSet::new();
        let hits = merge(
            Lists {
                lexical: &[1, 2],
                dense: &[2, 3],
                graph: &[3, 1],
            },
            settings(&excluded, Mode::Hybrid, 50),
        )?;
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![2, 1, 3]);
        assert!((hits[0].score - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-14);
        assert!((hits[1].score - (1.0 / 61.0 + 0.3 / 62.0)).abs() < 1e-14);
        assert!((hits[2].score - (1.0 / 62.0 + 0.3 / 61.0)).abs() < 1e-14);
        assert_eq!(hits[0].scores.lexical.map(|v| v.rank), Some(2));
        assert_eq!(hits[0].scores.dense.map(|v| v.rank), Some(1));
        assert!(hits[0].scores.graph.is_none());
        Ok(())
    }
    #[test]
    fn modes_exclusions_duplicates_and_stable_ties() -> Result<(), Error> {
        let excluded = BTreeSet::from([1]);
        let lists = || Lists {
            lexical: &[1, 1, 2],
            dense: &[3, 2],
            graph: &[4],
        };
        let lexical = merge(lists(), settings(&excluded, Mode::Lexical, 1))?;
        assert_eq!(lexical[0].id, 2);
        assert_eq!(lexical[0].scores.lexical.map(|v| v.rank), Some(2));
        let dense = merge(lists(), settings(&excluded, Mode::Dense, 50))?;
        assert_eq!(dense.iter().map(|h| h.id).collect::<Vec<_>>(), vec![3, 2]);
        assert!(
            dense
                .iter()
                .all(|h| h.scores.graph.is_none() && h.scores.lexical.is_none())
        );
        let ties = merge(
            Lists {
                lexical: &[2],
                dense: &[1],
                graph: &[],
            },
            settings(&BTreeSet::new(), Mode::Hybrid, 50),
        )?;
        assert_eq!(ties.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 2]);
        Ok(())
    }
    #[test]
    fn inactive_legs_and_invalid_settings_cannot_create_votes() -> Result<(), Error> {
        let excluded = BTreeSet::new();
        let mut args = settings(&excluded, Mode::Hybrid, 50);
        args.graph_weight = 0.0;
        assert!(
            merge(
                Lists {
                    graph: &[1],
                    ..Default::default()
                },
                args
            )?
            .is_empty()
        );
        for weight in [f64::NAN, f64::INFINITY, -1.0] {
            let mut args = settings(&excluded, Mode::Hybrid, 50);
            args.graph_weight = weight;
            assert!(merge(Lists::default(), args).is_err());
        }
        let mut args = settings(&excluded, Mode::Hybrid, 50);
        args.rrf_k = 0;
        assert!(merge(Lists::default(), args).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod routed_tests {
    use super::*;
    fn settings(exclude: &BTreeSet<i64>, mode: Mode) -> Settings<'_> {
        Settings {
            mode,
            rrf_k: 60,
            graph_weight: 0.3,
            exclude,
            limit: 20,
        }
    }
    #[test]
    fn independent_space_ranks_contribute_once_without_intermediate_fusion() {
        let lists = || Lists {
            lexical: &[1, 2],
            dense: &[],
            graph: &[1],
        };
        let spaces = [
            DenseList {
                table: "old",
                ids: &[3, 2, 2],
            },
            DenseList {
                table: "new",
                ids: &[4, 2],
            },
        ];
        let excluded = BTreeSet::new();
        let hits = merge_spaces(lists(), &spaces, settings(&excluded, Mode::Hybrid)).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![2, 1, 3, 4]
        );
        assert!((hits[0].score - 3.0 / 62.0).abs() < 1e-12);
        assert!((hits[1].score - 1.3 / 61.0).abs() < 1e-12);
        for table in ["old", "new"] {
            assert_eq!(hits[0].scores.dense_spaces[table].rank, 2);
        }
        assert_eq!(hits[0].scores.dense.unwrap().rank, 2);
        assert!((hits[2].score - 1.0 / 61.0).abs() < 1e-12);
        assert!((hits[3].score - 1.0 / 61.0).abs() < 1e-12);
        let excluded = BTreeSet::from([1, 3]);
        let hits = merge_spaces(lists(), &spaces, settings(&excluded, Mode::Dense)).unwrap();
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![2, 4]);
        assert!((hits[0].score - 2.0 / 62.0).abs() < 1e-12);
        let hits = merge_spaces(lists(), &spaces, settings(&excluded, Mode::Lexical)).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 2);
        assert!(hits[0].scores.dense_spaces.is_empty());
    }
    #[test]
    fn duplicate_or_already_fused_dense_inputs_are_refused() {
        let excluded = BTreeSet::new();
        let spaces = [
            DenseList {
                table: "same",
                ids: &[1],
            },
            DenseList {
                table: "same",
                ids: &[2],
            },
        ];
        assert!(
            merge_spaces(Lists::default(), &spaces, settings(&excluded, Mode::Hybrid)).is_err()
        );
        assert!(
            merge_spaces(
                Lists {
                    dense: &[1],
                    ..Default::default()
                },
                &spaces[..1],
                settings(&excluded, Mode::Hybrid)
            )
            .is_err()
        );
    }
}
