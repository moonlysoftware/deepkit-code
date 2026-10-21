//! Observational treatment-pattern lint with shared discovery policy; v1 §§13, 16.
use super::Finding;
use crate::{config::Config, sync};
use globset::Glob;
use std::{
    path::Path,
    time::{Duration, Instant},
};

fn result(status: &'static str, message: &str, evidence: Option<serde_json::Value>) -> Finding {
    Finding {
        id: "config.treatment_globs".into(),
        status,
        message: message.into(),
        evidence,
    }
}
fn incomplete() -> Finding {
    result(
        "not_checked",
        "Treatment glob traversal was incomplete; unmatched patterns cannot be established",
        None,
    )
}

/// Checks explicit patterns against an observed eligible-file walk, never source
/// contents or the Git index. This is not an atomic filesystem snapshot.
pub(super) fn check(root: &Path, config: &Config, configured: Option<bool>) -> Finding {
    inspect(root, config, configured, 100_000, Duration::from_secs(2))
}
fn inspect(
    root: &Path,
    config: &Config,
    configured: Option<bool>,
    max_entries: usize,
    budget: Duration,
) -> Finding {
    match configured {
        Some(false) => {
            return result(
                "not_applicable",
                "Built-in treatment globs are not linted for repository coverage",
                None,
            );
        }
        None => {
            return result(
                "not_checked",
                "Treatment glob lint requires configuration provenance",
                None,
            );
        }
        Some(true) => {}
    }
    // Cover compilation and matching as well as traversal. Individual glob and
    // filesystem operations remain cooperative, rather than preemptible.
    let started = Instant::now();
    let mut matchers = Vec::new();
    for (rule, treatment) in config.treatments.iter().enumerate() {
        if started.elapsed() >= budget {
            return incomplete();
        }
        for (pattern, text) in treatment.patterns.iter().enumerate() {
            if started.elapsed() >= budget {
                return incomplete();
            }
            match Glob::new(text) {
                Ok(glob) => matchers.push((rule + 1, pattern + 1, glob.compile_matcher(), false)),
                Err(_) => {
                    return result(
                        "failed",
                        "A configured treatment glob is invalid",
                        Some(serde_json::json!({"rule": rule + 1, "pattern": pattern + 1})),
                    );
                }
            }
        }
    }
    if started.elapsed() >= budget {
        return incomplete();
    }
    if matchers.is_empty() {
        return result(
            "not_applicable",
            "No configured treatment globs require matching",
            None,
        );
    }
    let Ok(root) = root.join(&config.index.root).canonicalize() else {
        return incomplete();
    };
    if !root.is_dir() {
        return incomplete();
    }
    let mut files = 0u64;
    for (entries, entry) in sync::discovery_builder(&root).build().enumerate() {
        if entries >= max_entries || started.elapsed() >= budget {
            return incomplete();
        }
        let Ok(entry) = entry else {
            return incomplete();
        };
        if entry.error().is_some() {
            return incomplete();
        }
        if entry.path_is_symlink() {
            continue;
        }
        let Some(kind) = entry.file_type() else {
            return incomplete();
        };
        if !kind.is_file() {
            continue;
        }
        if files >= config.limits.max_files {
            return incomplete();
        }
        files += 1;
        let Ok(relative) = entry.path().strip_prefix(&root) else {
            return incomplete();
        };
        // Indexing cannot catalogue non-Unicode paths. Avoid claiming absence
        // when any such entry makes the observed matching domain incomplete.
        if relative.to_str().is_none() {
            return incomplete();
        }
        for (_, _, matcher, matched) in &mut matchers {
            if started.elapsed() >= budget {
                return incomplete();
            }
            *matched |= matcher.is_match(relative);
        }
    }
    if started.elapsed() >= budget {
        return incomplete();
    }
    let mut unmatched = Vec::new();
    for (rule, pattern, _, matched) in &matchers {
        if started.elapsed() >= budget {
            return incomplete();
        }
        if !matched {
            unmatched.push(serde_json::json!({"rule": rule, "pattern": pattern}));
        }
    }
    let evidence = Some(serde_json::json!({
        "files_observed": files, "patterns_checked": matchers.len(), "unmatched": unmatched,
        "scope": "eligible files observed during complete traversal"
    }));
    if started.elapsed() >= budget {
        return incomplete();
    }
    if unmatched.is_empty() {
        result(
            "passed",
            "Every configured treatment glob matched an observed eligible file",
            evidence,
        )
    } else {
        result(
            "warning",
            "Configured treatment globs matched no observed eligible files",
            evidence,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Treatment;
    use std::fs;
    fn config(patterns: &[&str]) -> Config {
        Config {
            treatments: vec![Treatment {
                patterns: patterns.iter().map(|value| (*value).into()).collect(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }
    #[test]
    fn configured_root_and_shared_ignore_rules_define_matching_domain() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("project/src")).unwrap();
        fs::write(root.path().join("project/src/lib.rs"), "").unwrap();
        fs::write(root.path().join("project/hidden.rs"), "").unwrap();
        fs::write(root.path().join("project/.deepkitignore"), "hidden.rs\n").unwrap();
        fs::create_dir(root.path().join("project/.deepkit")).unwrap();
        fs::write(root.path().join("project/.deepkit/private.rs"), "").unwrap();
        let mut config = config(&["src/**", "hidden.rs", ".deepkit/**"]);
        config.index.root = "project".into();
        let report = check(root.path(), &config, Some(true));
        assert_eq!(report.status, "warning");
        assert_eq!(report.evidence.as_ref().unwrap()["files_observed"], 1);
        assert_eq!(
            report.evidence.as_ref().unwrap()["unmatched"],
            serde_json::json!([
                {"rule": 1, "pattern": 2}, {"rule": 1, "pattern": 3}
            ])
        );
    }
    #[test]
    fn capped_or_expired_walk_never_claims_patterns_are_unmatched() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("first.rs"), "").unwrap();
        let mut config = config(&["absent/**"]);
        config.limits.max_files = 0;
        for report in [
            check(root.path(), &config, Some(true)),
            inspect(root.path(), &config, Some(true), 0, Duration::from_secs(2)),
            inspect(root.path(), &config, Some(true), 100, Duration::ZERO),
        ] {
            assert_eq!(report.status, "not_checked");
            assert!(report.evidence.is_none());
        }
    }
    #[test]
    fn exhausted_budget_precedes_pattern_compilation_and_diagnosis() {
        let root = tempfile::tempdir().unwrap();
        // Invalid first pattern would yield failed if compilation starts before
        // checking the budget. Many later patterns model configuration fanout.
        let mut patterns = vec!["["; 10_000];
        patterns.push("absent/**");
        let report = inspect(
            root.path(),
            &config(&patterns),
            Some(true),
            100,
            Duration::ZERO,
        );
        assert_eq!(report.status, "not_checked");
        assert!(report.evidence.is_none());
    }
    #[test]
    fn unreadable_root_and_ignore_parse_error_are_not_unmatched_proof() {
        let root = tempfile::tempdir().unwrap();
        let mut config = config(&["absent/**"]);
        config.index.root = "missing".into();
        assert_eq!(
            check(root.path(), &config, Some(true)).status,
            "not_checked"
        );
        config.index.root = ".".into();
        fs::write(root.path().join(".deepkitignore"), "{a,b\n").unwrap();
        fs::write(root.path().join("file.rs"), "").unwrap();
        assert_eq!(
            check(root.path(), &config, Some(true)).status,
            "not_checked"
        );
    }
    #[test]
    fn defaults_are_not_warnings_and_unknown_provenance_is_explicit() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(
            check(root.path(), &Config::default(), Some(false)).status,
            "not_applicable"
        );
        assert_eq!(
            check(root.path(), &Config::default(), None).status,
            "not_checked"
        );
    }
    #[test]
    fn invalid_and_unmatched_patterns_never_expose_user_text() {
        let root = tempfile::tempdir().unwrap();
        let sentinel = format!(
            "private-{}",
            root.path().file_name().unwrap().to_string_lossy()
        );
        for pattern in [format!("{sentinel}/**"), format!("{sentinel}[")] {
            let report = check(root.path(), &config(&[&pattern]), Some(true));
            assert!(matches!(report.status, "warning" | "failed"));
            assert!(!report.message.contains(&sentinel));
            assert!(!format!("{:?}", report.evidence).contains(&sentinel));
        }
    }
}
