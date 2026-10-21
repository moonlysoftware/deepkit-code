//! CLI-only indexing niceness; v1.md §3.5. Serve never lowers its own priority.

/// Records whether the configured courtesy limit was applied to this index process.
#[derive(Debug, serde::Serialize)]
pub(super) struct Report {
    pub(super) requested: bool,
    pub(super) state: &'static str,
    pub(super) before: Option<i32>,
    pub(super) after: Option<i32>,
}

/// Runs before CLI indexing creates worker threads, which inherit its niceness.
/// Already lower-priority processes are preserved; failure remains observable.
pub(super) fn apply(requested: bool) -> Report {
    if !requested {
        return Report {
            requested,
            state: "disabled",
            before: None,
            after: None,
        };
    }
    #[cfg(unix)]
    {
        let before = rustix::process::getpriority_process(None).ok();
        let target = before.unwrap_or(10).max(10);
        let changed = rustix::process::setpriority_process(None, target).is_ok();
        let after = rustix::process::getpriority_process(None).ok();
        Report {
            requested,
            state: if changed && after.is_some_and(|value| value >= target) {
                "applied"
            } else {
                "unavailable"
            },
            before,
            after,
        }
    }
    #[cfg(not(unix))]
    Report {
        requested,
        state: "unsupported",
        before: None,
        after: None,
    }
}
