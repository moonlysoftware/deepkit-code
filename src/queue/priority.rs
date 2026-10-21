//! Rank and recency scheduling weights; see `v1.md` §8.3.

use super::{Error, Kind};

/// Distinguishes current-run file changes from aged commit-derived targets.
/// Callers supply the target's age, not the time since the job was enqueued.
#[derive(Clone, Copy)]
pub(crate) enum Recency {
    ChangedFile,
    Commit { age_days: f64 },
}

/// Calculates scheduling priority from a normalized PageRank percentile.
/// Invalid ranks/ages are rejected instead of silently changing claim order.
/// This weight is scheduling metadata and must not enter desired-output hashes.
pub(crate) fn calculate(kind: Kind, rank_pct: f64, recency: Recency) -> Result<f64, Error> {
    if !rank_pct.is_finite() || !(0.0..=1.0).contains(&rank_pct) {
        return Err(Error::Invalid("PageRank percentile"));
    }
    let recency = match recency {
        Recency::ChangedFile => 1.0,
        Recency::Commit { age_days } if age_days.is_finite() && age_days >= 0.0 => {
            (-age_days / 30.0).exp()
        }
        Recency::Commit { .. } => return Err(Error::Invalid("commit age")),
    };
    let level_weight = match kind {
        Kind::Embed => 2.0,
        Kind::DescribeSymbol
        | Kind::SummarizePr
        | Kind::SummarizeModule
        | Kind::DescribeImage
        | Kind::VlmPage => 1.0,
    };
    Ok(level_weight * (0.6 * rank_pct + 0.4 * recency))
}

/// Uses the highest source percentile for a composite, independent of ordering
/// or duplicate sources. Empty or invalid provenance ranks cannot be scheduled.
pub(crate) fn composite(
    kind: Kind,
    source_percentiles: &[f64],
    recency: Recency,
) -> Result<f64, Error> {
    let mut maximum = None::<f64>;
    for &rank in source_percentiles {
        if !rank.is_finite() || !(0.0..=1.0).contains(&rank) {
            return Err(Error::Invalid("source PageRank percentile"));
        }
        maximum = Some(maximum.map_or(rank, |previous| previous.max(rank)));
    }
    calculate(
        kind,
        maximum.ok_or(Error::Invalid("empty source PageRank percentiles"))?,
        recency,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        queue,
        store::{self, StoreLimits},
    };
    use proptest::prelude::*;

    #[test]
    fn specified_weights_decay_and_composite_maximum() -> Result<(), Error> {
        assert_eq!(calculate(Kind::Embed, 0.0, Recency::ChangedFile)?, 0.8);
        assert_eq!(calculate(Kind::Embed, 1.0, Recency::ChangedFile)?, 2.0);
        for kind in [
            Kind::DescribeSymbol,
            Kind::SummarizePr,
            Kind::SummarizeModule,
            Kind::DescribeImage,
            Kind::VlmPage,
        ] {
            assert_eq!(calculate(kind, 1.0, Recency::ChangedFile)?, 1.0);
            assert_eq!(
                calculate(kind, 0.0, Recency::Commit { age_days: 30.0 })?,
                0.4 / std::f64::consts::E
            );
        }
        assert_eq!(
            calculate(Kind::Embed, 0.0, Recency::Commit { age_days: f64::MAX })?,
            0.0
        );
        assert_eq!(
            composite(
                Kind::SummarizePr,
                &[0.2, 0.9, 0.5, 0.9],
                Recency::ChangedFile
            )?,
            calculate(Kind::SummarizePr, 0.9, Recency::ChangedFile)?
        );
        Ok(())
    }

    #[test]
    fn invalid_inputs_cannot_poison_ordering() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
            assert!(calculate(Kind::Embed, bad, Recency::ChangedFile).is_err());
            assert!(composite(Kind::Embed, &[1.0, bad], Recency::ChangedFile).is_err());
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
            assert!(calculate(Kind::Embed, 0.5, Recency::Commit { age_days: bad }).is_err());
        }
        assert!(composite(Kind::Embed, &[], Recency::ChangedFile).is_err());
    }

    proptest! {
        #[test]
        fn rank_and_recency_are_monotone(rank in 0.0f64..=1.0, other in 0.0f64..=1.0, age in 0.0f64..10000.0, later in 0.0f64..10000.0) {
            let score = calculate(Kind::Embed, rank, Recency::Commit { age_days: age })?;
            prop_assert!((0.0..=2.0).contains(&score));
            let higher_rank = calculate(Kind::Embed, rank.max(other), Recency::Commit { age_days: age })?;
            let older = calculate(Kind::Embed, rank, Recency::Commit { age_days: age + later })?;
            prop_assert!(higher_rank >= score);
            prop_assert!(older <= score);
            prop_assert_eq!(score, 2.0 * calculate(Kind::SummarizePr, rank, Recency::Commit { age_days: age })?);
        }
    }

    #[test]
    fn calculated_scores_drive_claims_and_reprioritization_keeps_generation()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store::open(":memory:", StoreLimits::default())?;
        let mut ids = Vec::new();
        for (target, kind, rank, recency) in [
            (1, Kind::Embed, 0.1, Recency::ChangedFile),
            (
                2,
                Kind::SummarizePr,
                0.9,
                Recency::Commit { age_days: 30.0 },
            ),
            (3, Kind::Embed, 0.9, Recency::ChangedFile),
        ] {
            let mut input = queue::tests::input(target, &[1; 32]);
            input.kind = kind;
            input.priority = calculate(kind, rank, recency)?;
            ids.push(queue::enqueue(&store, &input, 0)?);
        }
        let owner = queue::Owner::new(&store)?;
        let leases = queue::claim(&store, "embed:local", &owner, 0, 3, 30)?;
        assert_eq!(
            leases.iter().map(|lease| lease.id).collect::<Vec<_>>(),
            vec![ids[2], ids[0], ids[1]]
        );
        // Updating scheduling rank must not replace a live worker's generation.
        let mut input = queue::tests::input(1, &[1; 32]);
        input.priority = calculate(Kind::Embed, 1.0, Recency::ChangedFile)?;
        assert_eq!(queue::enqueue(&store, &input, 1)?, ids[0]);
        assert!(queue::renew(&store, &leases[1], 1)?);
        assert!(queue::claim(&store, "embed:local", &owner, 1, 3, 30)?.is_empty());
        Ok(())
    }
}
