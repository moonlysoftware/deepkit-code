//! Ordered built-in and custom PR message patterns; v1.md §4.4 and §11.
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Preserves string presets while allowing an explicit house-style strategy.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged, deny_unknown_fields)]
pub(crate) enum Pattern {
    Builtin(String),
    Custom { regex: String, strategy: String },
}

#[derive(Clone)]
pub(crate) struct Compiled {
    regex: Regex,
    strategy: String,
}

/// Metadata recognized from a marker; ancestry is reconstructed independently.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Marker {
    pub(crate) number: Option<i64>,
    pub(crate) title: String,
    pub(crate) strategy: String,
}

/// Validates patterns before indexing or dispatch; no unknown preset is ignored.
pub(crate) fn compile(patterns: &[Pattern]) -> Result<Vec<Compiled>, String> {
    // Configuration is revalidated at every publication fence. Recompiling four
    // unchanged regex automata there makes history ingestion scale with regex
    // compilation cost rather than changed Git data. Retain only the last exact
    // pattern set; regex clones share immutable compiled state.
    type Cached = Option<(Vec<Pattern>, Result<Vec<Compiled>, String>)>;
    static CACHE: std::sync::Mutex<Cached> = std::sync::Mutex::new(None);
    if let Some((_, result)) = CACHE
        .lock()
        .map_err(|_| "PR pattern cache unavailable")?
        .as_ref()
        .filter(|(previous, _)| previous == patterns)
    {
        return result.clone();
    }
    let result = compile_uncached(patterns);
    *CACHE.lock().map_err(|_| "PR pattern cache unavailable")? =
        Some((patterns.to_owned(), result.clone()));
    result
}

fn compile_uncached(patterns: &[Pattern]) -> Result<Vec<Compiled>, String> {
    patterns
        .iter()
        .map(|pattern| {
            let (pattern, strategy) = match pattern {
                Pattern::Builtin(name) => match name.as_str() {
                    "github-merge" => (r"^Merge pull request #(?P<number>\d+) from .*", "merge"),
                    "github-squash" => (r"\(#(?P<number>\d+)\)\s*$", "squash"),
                    "gitlab-merge" => (
                        r"^Merge branch .* See merge request .*!(?P<number>\d+)",
                        "gitlab",
                    ),
                    "plain-merge" => (r"^Merge branch '(?P<title>.+)' into .*", "plain"),
                    _ => return Err(format!("unknown preset {name}")),
                },
                Pattern::Custom { regex, strategy } => (regex.as_str(), strategy.as_str()),
            };
            if !matches!(strategy, "merge" | "squash" | "gitlab" | "plain") {
                return Err(format!("unknown strategy {strategy}"));
            }
            let regex = Regex::new(pattern).map_err(|e| e.to_string())?;
            Ok(Compiled {
                regex,
                strategy: strategy.to_owned(),
            })
        })
        .collect()
}

/// Applies §4.4 parent counts and pattern order exactly; rebase/octopus markers do not qualify.
pub(crate) fn detect(
    patterns: &[Compiled],
    parents: usize,
    subject: &str,
    body: &str,
) -> Option<Marker> {
    for pattern in patterns {
        if parents != if pattern.strategy == "squash" { 1 } else { 2 } {
            continue;
        }
        let Some(captures) = pattern.regex.captures(subject) else {
            continue;
        };
        let number = captures
            .name("number")
            .or_else(|| {
                (pattern.strategy != "plain")
                    .then(|| captures.get(1))
                    .flatten()
            })
            .and_then(|m| m.as_str().parse::<i64>().ok());
        let title = if let Some(title) = captures.name("title") {
            title.as_str().to_owned()
        } else if pattern.strategy == "squash" {
            subject
                .get(..captures.get(0)?.start())?
                .trim_end()
                .to_owned()
        } else if pattern.strategy == "plain" {
            captures.get(1).map_or(subject, |m| m.as_str()).to_owned()
        } else {
            body.lines()
                .next()
                .filter(|s| !s.is_empty())
                .unwrap_or(subject)
                .to_owned()
        };
        return Some(Marker {
            number,
            title,
            strategy: pattern.strategy.clone(),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn styles_parent_counts_custom_order_and_invalid_configuration() {
        let patterns = compile(&super::super::Commits::default().pr_patterns).unwrap();
        for (parents, subject, number, strategy, title) in [
            (
                2,
                "Merge pull request #42 from team/topic",
                Some(42),
                "merge",
                "Body title",
            ),
            (1, "Fix retry (#9)", Some(9), "squash", "Fix retry"),
            (2, "Merge branch 'topic' into main", None, "plain", "topic"),
            (
                2,
                "Merge branch topic See merge request team/repo!7",
                Some(7),
                "gitlab",
                "Body title",
            ),
        ] {
            assert_eq!(
                detect(&patterns, parents, subject, "Body title\nMore"),
                Some(Marker {
                    number,
                    title: title.into(),
                    strategy: strategy.into()
                })
            );
        }
        assert!(detect(&patterns, 1, "unmarked rebase", "body").is_none());
        assert!(detect(&patterns, 3, "Merge pull request #42 from team/topic", "").is_none());
        assert!(compile(&[Pattern::Builtin("unknown".into())]).is_err());
        assert!(
            compile(&[Pattern::Custom {
                regex: "[".into(),
                strategy: "plain".into()
            }])
            .is_err()
        );
        let custom = compile(&[Pattern::Custom {
            regex: r"^Merge commit '(?P<title>.+)'$".into(),
            strategy: "plain".into(),
        }])
        .unwrap();
        assert_eq!(
            detect(&custom, 2, "Merge commit 'abc'", "").unwrap().title,
            "abc"
        );
    }
}
