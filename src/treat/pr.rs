//! Deterministic PR collation and member-derived identity; v1.md §6.6–6.7.
use crate::{config::pr_patterns::Marker, store::prs::Member};
use std::collections::BTreeSet;

/// Bump only when the deterministic PR collation contract changes.
pub(crate) const VERSION: &str = "pr-collator-v2";

pub(crate) struct Collated {
    pub(crate) hash: [u8; 32],
    pub(crate) breadcrumb: String,
    pub(crate) header: String,
    pub(crate) text: String,
}

/// Collates chronological subjects/body lines, capped unions and honest diff totals.
/// The caller captures the symbol rank order once at publication.
pub(crate) fn collate(
    marker: &Marker,
    date: &str,
    members: &[Member],
    symbols: &[String],
) -> Collated {
    let mut ordered = members.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.time.cmp(&b.time).then(a.sha.cmp(&b.sha)));
    let files = members
        .iter()
        .flat_map(|m| m.files.iter())
        .collect::<BTreeSet<_>>();
    let additions: Option<i64> = members.iter().map(|m| m.additions).sum();
    let deletions: Option<i64> = members.iter().map(|m| m.deletions).sum();
    let stats = format!(
        "+{} −{}",
        additions.map_or_else(|| "?".into(), |v| v.to_string()),
        deletions.map_or_else(|| "?".into(), |v| v.to_string())
    );
    let subjects = ordered
        .iter()
        .map(|m| match m.body.lines().next().filter(|s| !s.is_empty()) {
            Some(line) => format!("{}\n{line}", m.subject),
            None => m.subject.clone(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let text = format!(
        "{subjects}\nfiles:\n{}\nsymbols:\n{}\n{stats}",
        files
            .into_iter()
            .take(200)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n"),
        symbols
            .iter()
            .take(100)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    );
    let fingerprint = super::composite::source_fingerprint(
        "pr",
        VERSION,
        members.iter().map(|m| m.sha.as_slice()),
    );
    Collated {
        hash: super::composite::payload_hash(&fingerprint, &text),
        breadcrumb: marker.number.map_or_else(
            || format!("PR › {}", marker.title),
            |n| format!("PR #{n} › {}", marker.title),
        ),
        header: format!("merged {date} › {} commits › {stats}", members.len()),
        text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chronological_capped_collation_and_source_hash_ignore_input_order() {
        let make = |id: u8, time: i64| Member {
            id: i64::from(id),
            sha: vec![id; 20],
            unit: i64::from(id),
            owner: None,
            time,
            subject: format!("subject {id}"),
            body: format!("body {id}\nnot collated"),
            files: (0..250).map(|n| format!("file{n:03}")).collect(),
            symbols: (0..150).map(|n| format!("symbol{n:03}")).collect(),
            additions: Some(4),
            deletions: Some(2),
        };
        let members = vec![make(2, 20), make(1, 10)];
        let symbols = members[0]
            .symbols
            .iter()
            .rev()
            .take(100)
            .cloned()
            .collect::<Vec<_>>();
        let marker = Marker {
            number: Some(7),
            title: "Title".into(),
            strategy: "merge".into(),
        };
        let result = collate(&marker, "2026-01-01", &members, &symbols);
        assert_eq!(result.breadcrumb, "PR #7 › Title");
        assert_eq!(result.header, "merged 2026-01-01 › 2 commits › +8 −4");
        assert!(
            result
                .text
                .starts_with("subject 1\nbody 1\nsubject 2\nbody 2\nfiles:\n")
        );
        assert!(!result.text.contains("not collated"));
        let (_, tail) = result.text.split_once("\nfiles:\n").unwrap();
        let (files, tail) = tail.split_once("\nsymbols:\n").unwrap();
        assert_eq!(files.lines().count(), 200);
        assert_eq!(tail.lines().count(), 101);
        assert_eq!(tail.lines().next(), Some("symbol149"));
        let mut reversed = members.clone();
        reversed.reverse();
        let mut body_change = members.clone();
        body_change[0].subject.push_str(" changed");
        assert_ne!(
            result.hash,
            collate(&marker, "2026-01-01", &body_change, &symbols).hash
        );
        let mut symbol_order = symbols.clone();
        symbol_order.reverse();
        assert_ne!(
            result.hash,
            collate(&marker, "2026-01-01", &members, &symbol_order).hash
        );
        let changed_context = Marker {
            number: Some(8),
            title: "Renamed title".into(),
            strategy: marker.strategy.clone(),
        };
        assert_eq!(
            result.hash,
            collate(&changed_context, "2026-02-01", &members, &symbols).hash
        );
        assert_eq!(
            result.hash,
            collate(&marker, "2026-01-01", &reversed, &symbols).hash
        );
        assert_eq!(
            collate(&marker, "2026-01-01", &reversed, &symbols).text,
            result.text
        );
        reversed[0].sha[0] = 99;
        assert_ne!(
            result.hash,
            collate(&marker, "2026-01-01", &reversed, &symbols).hash
        );
        reversed[0].additions = None;
        assert!(
            collate(&marker, "2026-01-01", &reversed, &symbols)
                .header
                .ends_with("+? −4")
        );
    }
}
