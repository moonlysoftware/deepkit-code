//! Deterministic directory collation with file and README provenance; v1.md §6.6–6.7.
use super::composite;
use std::collections::BTreeMap;

/// Change when the canonical module rendering or source-selection contract changes.
pub(crate) const VERSION: &str = "module-collator-v1";

/// Stable symbol presentation; database row ids never break rendering ties.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Symbol {
    pub(crate) name: String,
    pub(crate) qualified: String,
    pub(crate) rank: f64,
    pub(crate) line: i64,
}

/// One file within the directory subtree, named relative to that directory.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct File {
    pub(crate) path: String,
    pub(crate) hash: Vec<u8>,
    pub(crate) symbols: Vec<Symbol>,
}

/// The first completed README section selected from the module directory itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Readme {
    pub(crate) hash: Vec<u8>,
    pub(crate) text: String,
}

/// Complete canonical inputs, captured together with their database provenance.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Input {
    pub(crate) files: Vec<File>,
    pub(crate) readme: Option<Readme>,
    pub(crate) graph_hashes: Vec<Vec<u8>>,
    pub(crate) depends_on: BTreeMap<String, usize>,
    pub(crate) used_by: BTreeMap<String, usize>,
}

/// Immutable body identity and its human-readable fallback for repository maps.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Collated {
    pub(crate) hash: [u8; 32],
    pub(crate) header: String,
    pub(crate) text: String,
}

fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

fn compare_symbols(left: &Symbol, right: &Symbol) -> std::cmp::Ordering {
    right
        .rank
        .total_cmp(&left.rank)
        .then_with(|| left.qualified.cmp(&right.qualified))
        .then_with(|| left.line.cmp(&right.line))
        .then_with(|| left.name.cmp(&right.name))
}

/// Renders subtree files, their three leading symbols, twenty ranked exports,
/// and cross-directory edge counts. Equal input sets render independently of
/// insertion order; filename and graph-body changes receive distinct payloads.
pub(crate) fn collate(input: &Input) -> Collated {
    let mut files = input.files.iter().collect::<Vec<_>>();
    files.sort_by(|left, right| left.path.cmp(&right.path));
    let mut lines = Vec::new();
    if let Some(readme) = &input.readme {
        lines.push(readme.text.clone());
    }
    lines.push("files:".into());
    let mut exports = Vec::new();
    for file in &files {
        let mut symbols = file.symbols.iter().collect::<Vec<_>>();
        symbols.sort_by(|left, right| compare_symbols(left, right));
        let names = symbols
            .iter()
            .take(3)
            .map(|s| label(&s.name))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(if names.is_empty() {
            label(&file.path)
        } else {
            format!("{} — {names}", label(&file.path))
        });
        exports.extend(
            symbols
                .into_iter()
                .map(|symbol| (file.path.as_str(), symbol)),
        );
    }
    let symbol_count = exports.len();
    exports.sort_by(|(left_path, left), (right_path, right)| {
        compare_symbols(left, right).then_with(|| left_path.cmp(right_path))
    });
    lines.push("exports:".into());
    for (path, symbol) in exports.iter().take(20) {
        lines.push(format!("{} — {}", label(&symbol.qualified), label(path)));
    }
    for (name, counts) in [
        ("depends on:", &input.depends_on),
        ("used by:", &input.used_by),
    ] {
        lines.push(name.into());
        for (directory, count) in counts {
            lines.push(format!("{}: {count}", label(directory)));
        }
    }
    let text = lines.join("\n");
    let fingerprint = composite::source_fingerprint(
        "module",
        VERSION,
        files
            .iter()
            .map(|file| file.hash.as_slice())
            .chain(input.readme.iter().map(|readme| readme.hash.as_slice()))
            .chain(input.graph_hashes.iter().map(Vec::as_slice)),
    );
    Collated {
        hash: composite::payload_hash(&fingerprint, &text),
        header: format!("module · {} files · {symbol_count} symbols", files.len()),
        text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Input {
        Input {
            files: (0..5)
                .map(|i| File {
                    path: format!("file{i}.rs"),
                    hash: vec![i; 32],
                    symbols: (0..5)
                        .map(|s| Symbol {
                            name: format!("item{s}"),
                            qualified: format!("file{i}::item{s}"),
                            rank: f64::from(s),
                            line: i64::from(s) + 1,
                        })
                        .collect(),
                })
                .collect(),
            readme: Some(Readme {
                hash: vec![9; 32],
                text: "# Overview\nModule purpose.".into(),
            }),
            graph_hashes: Vec::new(),
            depends_on: BTreeMap::from([("other".into(), 3)]),
            used_by: BTreeMap::from([("tests".into(), 2)]),
        }
    }

    #[test]
    fn counts_sections_rank_prefixes_and_order_are_deterministic() {
        let mut input = fixture();
        let first = collate(&input);
        assert_eq!(first.header, "module · 5 files · 25 symbols");
        assert!(
            first.text.starts_with(
                "# Overview\nModule purpose.\nfiles:\nfile0.rs — item4, item3, item2\n"
            )
        );
        assert!(
            first
                .text
                .ends_with("depends on:\nother: 3\nused by:\ntests: 2")
        );
        assert_eq!(
            first
                .text
                .split_once("exports:\n")
                .unwrap()
                .1
                .split_once("\ndepends on:")
                .unwrap()
                .0
                .lines()
                .count(),
            20
        );
        input.files.reverse();
        for file in &mut input.files {
            file.symbols.reverse();
        }
        assert_eq!(first, collate(&input));
    }

    #[test]
    fn same_sources_different_names_and_graph_bodies_cannot_alias() {
        let input = fixture();
        let first = collate(&input);
        let mut renamed = input.clone();
        renamed.files[0].path = "renamed.rs".into();
        assert_ne!(first.hash, collate(&renamed).hash);
        let mut edges = input.clone();
        edges.depends_on.insert("other".into(), 4);
        assert_ne!(first.hash, collate(&edges).hash);
        let mut ranked = input.clone();
        ranked.files[0].symbols[0].rank = 100.0;
        assert_ne!(first.hash, collate(&ranked).hash);
        let mut source = input.clone();
        source.files[0].hash[0] += 1;
        let changed = collate(&source);
        assert_eq!(first.text, changed.text);
        assert_ne!(first.hash, changed.hash);
    }
}
