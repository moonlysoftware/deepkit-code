#!/usr/bin/env python3
"""Audit Phase 4 local and Moonly reports without weakening release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


GIB = 1024 * 1024 * 1024
NAMED_COLD_LIMITS_MS = {
    "ripgrep": 10_000.0,
    "laravel": 60_000.0,
    "kubernetes": 480_000.0,
}


def issue(kind: str, metric: str, actual: Any, target: Any, detail: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "metric": metric,
        "actual": actual,
        "target": target,
        "detail": detail,
    }


def score_median(paired: dict[str, Any], arm: str, metric: str) -> float | None:
    values = []
    for repetition in paired.get("scores", []):
        value = (
            repetition.get(arm, {})
            .get("commit_derived", {})
            .get(metric)
        )
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.median(values) if values else None


def audit_target(target: dict[str, Any], backend: str = "cuda") -> dict[str, Any]:
    name = target.get("name", "<missing>")
    failures: list[dict[str, Any]] = []
    release_misses: list[dict[str, Any]] = []
    observations: dict[str, Any] = {}

    provenance = target.get("provenance", {})
    pre_index = provenance.get("pre_index", {})
    if backend in ("cuda", "dual", "ollama"):
        if pre_index.get("idle") is not True:
            failures.append(issue("measurement", "pre_index.idle", pre_index.get("idle"), True, "CUDA run did not start from the required CPU/GPU idle preflight"))
    else:
        load = pre_index.get("load_average", [])
        load_1m = load[0] if isinstance(load, list) and load else None
        if not isinstance(load_1m, (int, float)) or load_1m >= 1:
            failures.append(issue("measurement", "pre_index.load_1m", load_1m, "<1", "CPU run did not start from the required load preflight"))

    robustness = target.get("robustness", {})
    for field in ("panics", "dead_jobs"):
        value = robustness.get(field)
        if value != 0:
            failures.append(issue("robustness", field, value, 0, "corpus runs require zero"))
    parse_rates = {}
    for language, row in robustness.get("parse_failures_by_language", {}).items():
        rate = row.get("failure_rate")
        parse_rates[language] = rate
        if not isinstance(rate, (int, float)) or rate > 0.005:
            failures.append(issue("robustness", f"parse_failure_rate.{language}", rate, "<=0.005", "hostile fixtures are excluded from this denominator"))
    observations["parse_failure_rates"] = parse_rates

    model = target.get("model_index", {})
    coverage = model.get("coverage", {})
    indexed = coverage.get("indexed_units")
    covered = coverage.get("covered_units")
    pending = model.get("pending_jobs")
    providers = model.get("index_result", {}).get("embedding", {}).get("providers", [])
    terminal_provider_pending = model.get("terminal_provider_pending")
    if isinstance(terminal_provider_pending, dict):
        provider_pending = sum(
            value for value in terminal_provider_pending.values() if isinstance(value, int)
        )
    else:
        provider_pending = sum(
            provider.get("pending", 0)
            for provider in providers
            if isinstance(provider.get("pending", 0), int)
        )
    capacity_jobs = (
        target.get("paired_search", {})
        .get("capacity_pauses", {})
        .get("jobs")
    )
    ladder = (target.get("status") or {}).get("ladder", {})
    observations["model"] = {
        "indexed_units": indexed,
        "covered_units": covered,
        "pending_jobs": pending,
        "provider_pending_jobs": provider_pending,
        "capacity_paused_jobs": capacity_jobs,
        "ladder": ladder,
        "throughput_units_per_second": model.get("throughput_units_per_second"),
    }
    requested_level = 3 if backend in ("moonly", "dual") else 2
    if model.get("requested_level") != requested_level:
        phase = "T4.3 remote/dual" if backend in ("moonly", "dual") else "T4.2/T4.3 local"
        failures.append(issue("model", "requested_level", model.get("requested_level"), requested_level, f"{phase} run requested the complete configured ladder"))
    expected_state = "quiescent" if pending == 0 else "pending"
    if model.get("embedding_state") != expected_state:
        failures.append(issue("model", "embedding_state", model.get("embedding_state"), expected_state, "embedding state disagrees with the terminal queue snapshot"))
    if (
        not isinstance(pending, int)
        or pending < 0
        or coverage.get("pending_jobs") != pending
        or provider_pending != pending
        or (
            capacity_jobs != pending
            and (pending > 0 or capacity_jobs is not None)
        )
    ):
        failures.append(issue("model", "pending_jobs", [pending, coverage.get("pending_jobs"), provider_pending, capacity_jobs], "report pending == coverage pending == provider pending == capacity pauses", "only visible capacity pauses may remain"))
    ladder_counts = [ladder.get(level) for level in ("l1", "l2", "l3")]
    if (
        not isinstance(indexed, int)
        or indexed <= 0
        or not isinstance(covered, int)
        or covered < 0
        or covered > indexed
        or any(not isinstance(count, int) or count < 0 for count in ladder_counts)
        or sum(ladder_counts) != indexed
        or ladder.get("l2") != covered
    ):
        failures.append(issue("model", "coverage", [covered, indexed, ladder_counts], "nonnegative exact ladder counts sum to indexed and L2 equals covered", "active table coverage disagrees with the persisted ladder"))
    for provider in providers:
        for field in ("dead", "errors", "lease_lost"):
            if provider.get(field) != 0:
                failures.append(issue("model", f"provider.{field}", provider.get(field), 0, str(provider.get("provider"))))

    index = target.get("index", {})
    cold_ms = index.get("cold_ms")
    edit = index.get("content_edit", {})
    observations["index"] = {
        "cold_ms": cold_ms,
        "incremental_ms": index.get("incremental_ms"),
        "content_edit_ms": edit.get("elapsed_ms"),
        "touched_files": index.get("touched_files"),
        "peak_rss_bytes": index.get("peak_rss_bytes"),
    }
    cold_limit = NAMED_COLD_LIMITS_MS.get(name)
    if cold_limit is not None and (not isinstance(cold_ms, (int, float)) or cold_ms >= cold_limit):
        release_misses.append(issue("release_gate", "l1_cold_ms", cold_ms, f"<{cold_limit}", "named v1 latency gate"))
    touched = index.get("touched_files")
    edit_ms = edit.get("elapsed_ms")
    if isinstance(touched, int) and touched >= 20 and (not isinstance(edit_ms, (int, float)) or edit_ms >= 1_000):
        release_misses.append(issue("release_gate", "l1_content_edit_ms", edit_ms, "<1000", "20-touched-file gate"))

    size = target.get("size", {})
    total_db = size.get("accounting", {}).get("total_db_bytes", size.get("db_bytes"))
    ratio = size.get("ratio")
    checkout = size.get("checkout_bytes")
    observations["size"] = {
        "total_db_bytes": total_db,
        "checkout_bytes": checkout,
        "ratio": ratio,
    }
    if not isinstance(total_db, int) or total_db > 2 * GIB:
        release_misses.append(issue("release_gate", "total_db_bytes", total_db, f"<={2 * GIB}", "every default corpus database must remain at most 2 GiB"))
    # Very small fixtures cannot satisfy a multiplicative database ratio because
    # SQLite's fixed schema pages dominate. Apply the corpus ratio to checkouts
    # large enough for the denominator to describe product data.
    if isinstance(checkout, int) and checkout >= 1024 * 1024 and (
        not isinstance(ratio, (int, float)) or ratio > 3.5
    ):
        release_misses.append(issue("release_gate", "l2_footprint_ratio", ratio, "<=3.5", "full L2 vs admitted checkout text"))

    composites = target.get("composites", {}).get("pr_reconstruction", {})
    detected = composites.get("detected")
    reconstructed = composites.get("reconstructed")
    rate = composites.get("rate")
    observations["pr_reconstruction"] = {
        "detected": detected,
        "reconstructed": reconstructed,
        "rate": rate,
    }
    if isinstance(detected, int) and detected > 0 and (
        not isinstance(rate, (int, float)) or rate < 0.95
    ):
        release_misses.append(issue("release_gate", "pr_reconstruction_rate", rate, ">=0.95", f"{reconstructed}/{detected} reconstructed"))

    paired = target.get("paired_search", {})
    paired_state = paired.get("state")
    if paired_state == "skipped":
        if paired.get("reason") != "no_commit_derived_goldens" or paired.get("goldens") != 0:
            failures.append(issue("quality", "paired_search.skip", paired, "explicit zero-golden skip", "missing or inconsistent quality work"))
        observations["paired_search"] = {"state": "skipped", "goldens": 0}
    else:
        goldens = paired.get("goldens", [])
        golden_count = len(goldens) if isinstance(goldens, list) else None
        repeats = paired.get("repeats")
        pairs = paired.get("pairs", [])
        warmups = paired.get("warmup_repetitions")
        expected_pairs = (
            golden_count * (repeats + warmups)
            if isinstance(golden_count, int)
            and isinstance(repeats, int)
            and isinstance(warmups, int)
            else None
        )
        if paired_state != "complete" or paired.get("error") is not None:
            failures.append(issue("quality", "paired_search.state", [paired_state, paired.get("error")], ["complete", None], "paired search did not complete"))
        if repeats != 3 or warmups != 1:
            failures.append(issue("quality", "paired_search.repetitions", [repeats, warmups], [3, 1], "required one warmup plus three measured repetitions"))
        if expected_pairs is None or len(pairs) != expected_pairs:
            failures.append(issue("quality", "paired_search.pairs", len(pairs), expected_pairs, "every golden needs one warmup and three measured pairs"))
        auto_reduced = False
        for pair_index, pair in enumerate(pairs):
            repetition = pair.get("repetition")
            for arm in ("lexical", "hybrid", "dense_only", "reranked"):
                result = pair.get(arm)
                if not isinstance(result, dict):
                    failures.append(issue("quality", f"pair.{pair_index}.{arm}", result, "result", "missing paired arm"))
                elif result.get("error") is not None:
                    failures.append(issue("quality", f"pair.{pair_index}.{arm}", [result.get("error"), result.get("degraded")], [None, False], "paired arm failed or degraded"))
                elif result.get("degraded") is not False:
                    auto = result.get("rerank_auto", {})
                    expected_warmup = (
                        repetition == 0
                        and arm == "reranked"
                        and result.get("rerank_fallback") == "rerank_timeout"
                        and result.get("candidate_count") == 50
                        and auto.get("configured_candidates") == 50
                        and auto.get("effective_candidates") == 20
                        and auto.get("reduced") is True
                    )
                    if not expected_warmup:
                        failures.append(issue("quality", f"pair.{pair_index}.{arm}", [result.get("error"), result.get("degraded")], [None, False], "paired arm failed or degraded"))
                    else:
                        auto_reduced = True
        scores = paired.get("scores", [])
        if len(scores) != 3:
            failures.append(issue("quality", "paired_search.scores", len(scores), 3, "one score block per measured repetition"))
        for repetition, score in enumerate(scores):
            for arm in ("lexical", "hybrid", "dense_only", "reranked"):
                count = score.get(arm, {}).get("commit_derived", {}).get("count")
                if count != golden_count:
                    failures.append(issue("quality", f"scores.{repetition}.{arm}.count", count, golden_count, "quality denominator changed between arms"))
        observations["paired_search"] = {
            "state": paired_state,
            "goldens": golden_count,
            "repeats": repeats,
            "capacity_pauses": paired.get("capacity_pauses"),
            "lexical_hit_file_at_5": score_median(paired, "lexical", "hit_file_at_5"),
            "hybrid_hit_file_at_5": score_median(paired, "hybrid", "hit_file_at_5"),
            "reranked_hit_file_at_5": score_median(paired, "reranked", "hit_file_at_5"),
            "lexical_recall_at_10": score_median(paired, "lexical", "recall_at_10"),
            "hybrid_recall_at_10": score_median(paired, "hybrid", "recall_at_10"),
            "reranked_recall_at_10": score_median(paired, "reranked", "recall_at_10"),
            "hybrid_p95_ms": paired.get("latency", {}).get("hybrid", {}).get("p95_ms"),
            "reranked_p95_ms": paired.get("latency", {}).get("reranked", {}).get("p95_ms"),
        }
        observations["paired_search"]["rerank_auto_reduced"] = auto_reduced
        if auto_reduced and backend in ("cuda", "dual"):
            release_misses.append(issue("release_gate", "rerank_candidate_count", 20, 50, "T4.6 requires exactly 50 CUDA rerank candidates; the T4.2 survey may continue after its documented warmup adaptation"))

    throughput = model.get("throughput_units_per_second")
    throughput_target = {"cuda": 300, "cpu": 20, "moonly": 80}.get(backend)
    if throughput_target is not None and isinstance(indexed, int) and indexed >= 100 and (
        not isinstance(throughput, (int, float)) or throughput < throughput_target
    ):
        target_kind = "Moonly target" if backend == "moonly" else f"local {backend.upper()} target"
        release_misses.append(issue("release_gate", "l2_throughput_units_per_second", throughput, f">={throughput_target}", target_kind))

    return {
        "name": name,
        "accepted": not failures,
        "failures": failures,
        "release_misses": release_misses,
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--backend", choices=("cuda", "cpu", "moonly", "ollama", "dual"), default="cuda")
    args = parser.parse_args()

    audited = []
    report_failures = []
    builds = []
    for path in args.reports:
        report = json.loads(path.read_text())
        expected_profile = {"cuda": "local", "cpu": "cpu", "moonly": "moonly",
                            "ollama": "ollama", "dual": "dual"}[
            args.backend
        ]
        if report.get("profile") != expected_profile:
            report_failures.append(issue("measurement", f"{path}.profile", report.get("profile"), expected_profile, "wrong benchmark profile"))
        build = report.get("build", {})
        builds.append(build)
        features = set(str(build.get("features", "")).split(","))
        expected_features = {"CUDA", "DEFAULT"} if args.backend in ("cuda", "dual") else {"DEFAULT"}
        if build.get("profile") != "release" or build.get("source_dirty") not in (False, "false") or features != expected_features:
            report_failures.append(issue("measurement", f"{path}.build", build, f"clean release {args.backend.upper()}", "release identity refused"))
        failures = report.get("failures")
        if not isinstance(failures, list):
            report_failures.append(issue("measurement", f"{path}.failures", failures,
                                         "failure list", "benchmark report shape is invalid"))
        elif failures:
            report_failures.extend(failures)
        targets = report.get("targets")
        if not isinstance(targets, list) or not targets:
            report_failures.append(issue("measurement", f"{path}.targets", targets,
                                         "nonempty target list", "a lane cannot accept an empty report"))
        else:
            audited.extend(audit_target(target, args.backend) for target in targets)

    if builds and any(build != builds[0] for build in builds[1:]):
        report_failures.append(issue("measurement", "build_identity", builds, "one exact build", "reports are not comparable"))
    accepted = not report_failures and all(row["accepted"] for row in audited)
    audit_format = (
        "t4-moonly-audit-v1" if args.backend == "moonly"
        else f"t4-local-{args.backend}-audit-v1"
        if args.backend in ("cuda", "cpu")
        else f"t4-{args.backend}-audit-v1"
    )
    result = {
        "format": audit_format,
        "backend": args.backend,
        "reports": [str(path) for path in args.reports],
        "build": builds[0] if builds else None,
        "accepted": accepted,
        "report_failures": report_failures,
        "targets": audited,
        "release_misses": [
            {"target": row["name"], **miss}
            for row in audited
            for miss in row["release_misses"]
        ],
    }
    result[f"accepted_for_t4_{2 if args.backend == 'cuda' else 3}"] = accepted
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered)
    sys.stdout.write(rendered)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
