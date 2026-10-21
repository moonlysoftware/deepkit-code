//! Bench-machine nightly diagnostics; see `v1.md` §13 and `plan/07-dev-machine.md` §5.

use super::Finding;
#[cfg(any(target_os = "linux", test))]
use serde_json::json;
#[cfg(target_os = "linux")]
use std::process::{Command, Output};

#[cfg(any(target_os = "linux", test))]
const TIMER: &str = "dkc-nightly.timer";

/// Checks only host scheduling state; it never starts or changes a user unit.
pub(super) fn collect() -> Vec<Finding> {
    #[cfg(not(target_os = "linux"))]
    {
        vec![Finding::new(
            "bench.timer",
            "not_applicable",
            "The Phase 4 nightly timer is installed only on the Linux bench machine",
        )]
    }
    #[cfg(target_os = "linux")]
    collect_linux()
}

#[cfg(target_os = "linux")]
fn collect_linux() -> Vec<Finding> {
    let enabled = systemctl(&["is-enabled", TIMER]);
    let active = systemctl(&["is-active", TIMER]);
    let last_trigger = systemctl(&["show", TIMER, "--property=LastTriggerUSec", "--value"]);
    let linger = linger();
    vec![
        state_finding(
            "bench.timer.enabled",
            enabled.as_ref().is_some_and(|output| {
                output.status.success()
                    && matches!(text(output).as_str(), "enabled" | "enabled-runtime")
            }),
            "Nightly user timer is enabled",
            "Nightly user timer is not enabled",
        ),
        state_finding(
            "bench.timer.active",
            active
                .as_ref()
                .is_some_and(|output| output.status.success() && text(output) == "active"),
            "Nightly user timer is active",
            "Nightly user timer is not active",
        ),
        state_finding(
            "bench.timer.fired",
            last_trigger
                .as_ref()
                .is_some_and(|output| output.status.success() && trigger_recorded(&text(output))),
            "Nightly user timer has a recorded firing",
            "Nightly user timer has no recorded firing",
        ),
        state_finding(
            "bench.timer.linger",
            linger
                .as_ref()
                .is_some_and(|output| output.status.success() && text(output) == "yes"),
            "User lingering is enabled for unattended nightly runs",
            "User lingering is not enabled or could not be checked",
        ),
    ]
}

#[cfg(target_os = "linux")]
fn systemctl(arguments: &[&str]) -> Option<Output> {
    Command::new("systemctl")
        .arg("--user")
        .args(arguments)
        .output()
        .ok()
}

#[cfg(target_os = "linux")]
fn linger() -> Option<Output> {
    let uid = Command::new("id").arg("-u").output().ok()?;
    if !uid.status.success() {
        return None;
    }
    let uid = String::from_utf8(uid.stdout).ok()?;
    Command::new("loginctl")
        .args(["show-user", uid.trim(), "--property=Linger", "--value"])
        .output()
        .ok()
}

#[cfg(target_os = "linux")]
fn text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[cfg(any(target_os = "linux", test))]
fn trigger_recorded(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty() && !matches!(value, "n/a" | "0" | "0us")
}

#[cfg(any(target_os = "linux", test))]
fn state_finding(id: &str, passed: bool, success: &str, failure: &str) -> Finding {
    let mut finding = Finding::new(
        id,
        if passed { "passed" } else { "failed" },
        if passed { success } else { failure },
    );
    finding.evidence = Some(json!({"passed": passed, "timer": TIMER}));
    finding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_firing_requires_a_concrete_systemd_timestamp() {
        for absent in ["", "n/a", "0", "0us", "  n/a\n"] {
            assert!(!trigger_recorded(absent));
        }
        assert!(trigger_recorded("Sun 2026-09-20 02:00:01 CEST"));
    }

    #[test]
    fn findings_expose_only_bounded_boolean_evidence() {
        let finding = state_finding("bench.timer.active", false, "yes", "no");
        assert_eq!(finding.status, "failed");
        assert_eq!(finding.message, "no");
        assert_eq!(
            finding.evidence,
            Some(json!({"passed":false,"timer":TIMER}))
        );
    }
}
