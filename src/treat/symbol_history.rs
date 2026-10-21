//! Deterministic symbol history text and source identity; v1.md §6.6–6.7.
use crate::store::symbol_history::Entry;
/// Change when the history text contract changes, independently of model settings.
pub(crate) const VERSION: &str = "symbol-history-v2";
#[derive(Debug, thiserror::Error)]
#[error("history commit date cannot be represented")]
pub(crate) struct Error;
pub(crate) struct Collated {
    pub(crate) hash: [u8; 32],
    pub(crate) text: String,
}
/// Uses the selected immutable source SHAs; text is oldest-first with a SHA tie-break.
pub(crate) fn collate(entries: &[Entry]) -> Result<Collated, Error> {
    let mut ordered = entries.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.time.cmp(&b.time).then(a.sha.cmp(&b.sha)));
    let fingerprint = super::composite::source_fingerprint(
        "history",
        VERSION,
        entries.iter().map(|e| e.sha.as_slice()),
    );
    let mut lines = Vec::new();
    for entry in ordered {
        let sha = entry
            .sha
            .iter()
            .take(4)
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let date = gix::date::Time::new(entry.time, 0)
            .format(gix::date::time::format::SHORT)
            .map_err(|_| Error)?;
        let stat = |v: Option<i64>| v.map_or_else(|| "?".into(), |v| v.to_string());
        lines.push(format!(
            "{sha} {date} {}: {} (+{} −{})",
            entry.author.replace(['\n', '\r'], " "),
            entry.subject.replace(['\n', '\r'], " "),
            stat(entry.additions),
            stat(entry.deletions)
        ));
    }
    let text = lines.join("\n");
    Ok(Collated {
        hash: super::composite::payload_hash(&fingerprint, &text),
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn history_payload_tracks_canonical_body_and_source_multiset_without_order() {
        let make = |n: u8| Entry {
            id: i64::from(n),
            sha: vec![n; 20],
            unit: i64::from(n),
            author: "fixture".into(),
            time: i64::from(n),
            subject: format!("subject {n}"),
            additions: Some(1),
            deletions: Some(0),
        };
        let mut entries = vec![make(1), make(2)];
        let initial = collate(&entries).unwrap();
        entries.reverse();
        assert_eq!(collate(&entries).unwrap().hash, initial.hash);
        entries[0].author = "changed author".into();
        assert_ne!(collate(&entries).unwrap().hash, initial.hash);
        entries[0].author = "fixture".into();
        entries[0].sha[19] = 99;
        // The visible short SHA is unchanged: source identity still changes.
        let changed = collate(&entries).unwrap();
        assert_eq!(changed.text, initial.text);
        assert_ne!(changed.hash, initial.hash);
    }
}
