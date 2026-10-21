#!/usr/bin/env python3
"""Reduce collected T4.9 sessions using explicit frozen-rubric grades."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--grades", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def percentile_interval(pairs: list[dict]) -> list[float] | None:
    questions = sorted({pair["question_id"] for pair in pairs})
    if not questions:
        return None
    groups = [[pair for pair in pairs if pair["question_id"] == question] for question in questions]
    randomizer = random.Random(409)
    samples = []
    for _ in range(10_000):
        draw = [
            pair["token_ratio_with_over_without"]
            for _ in groups
            for pair in randomizer.choice(groups)
        ]
        samples.append(statistics.median(draw))
    samples.sort()
    return [samples[249], samples[9_749]]


def arm_summary(rows: list[dict], arm: str) -> dict:
    selected = [row for row in rows if row["condition"] == arm]
    expected_files = sum(len(row["expected_files"]) for row in selected)
    expected_symbols = sum(len(row["expected_symbols"]) for row in selected)
    return {
        "sessions": len(selected),
        "answered": sum(row["state"] == "answered" for row in selected),
        "incomplete": sum(row["state"] == "incomplete" for row in selected),
        "correctness_points": sum(row["correctness_0_to_2"] for row in selected),
        "correctness_possible": len(selected) * 2,
        "evidence_points": sum(row["evidence_0_to_2"] for row in selected),
        "evidence_possible": len(selected) * 2,
        "agent_usage": {
            key: sum(row["agent_usage"][key] for row in selected)
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
        },
        "chat_requests": sum(row["chat_requests"] for row in selected),
        "tool_calls": sum(row["tool_calls"] for row in selected),
        "wall_seconds": sum(row["wall_seconds"] for row in selected),
        "expected_files_found": sum(len(row["expected_files_seen"]) for row in selected),
        "expected_files_possible": expected_files,
        "expected_symbols_found": sum(len(row["expected_symbols_seen"]) for row in selected),
        "expected_symbols_possible": expected_symbols,
        "sessions_with_all_expected_files": sum(
            bool(row["expected_files"])
            and len(row["expected_files_seen"]) == len(row["expected_files"])
            for row in selected
        ),
        "median_agent_tokens": statistics.median(
            row["agent_usage"]["total_tokens"] for row in selected
        ),
        "median_wall_seconds": statistics.median(row["wall_seconds"] for row in selected),
    }


def token_result_sentence(reduction_fraction: float, target_met: bool) -> str:
    """Describe a saving or increase without reversing a negative reduction."""
    if reduction_fraction >= 0:
        change = f"{100 * reduction_fraction:.2f}% fewer"
    else:
        change = f"{-100 * reduction_fraction:.2f}% more"
    verdict = "met" if target_met else "not met"
    return f"Result: **{change} agent tokens** with Deepkit Code; target {verdict}."


def aggregate(rows: list[dict]) -> dict:
    by_id = {row["id"]: row for row in rows}
    pairs = []
    for row in rows:
        if row["condition"] != "without_dkc":
            continue
        prefix = f"{row['question_id']}-r{row['repeat']}"
        without = by_id[f"{prefix}-without_dkc"]
        with_dkc = by_id[f"{prefix}-with_dkc"]
        denominator = without["agent_usage"]["total_tokens"]
        pairs.append(
            {
                "question_id": row["question_id"],
                "repeat": row["repeat"],
                "token_ratio_with_over_without": (
                    with_dkc["agent_usage"]["total_tokens"] / denominator
                    if denominator
                    else None
                ),
                "token_difference_with_minus_without": with_dkc["agent_usage"]["total_tokens"]
                - without["agent_usage"]["total_tokens"],
                "correctness_delta": with_dkc["correctness_0_to_2"]
                - without["correctness_0_to_2"],
                "evidence_delta": with_dkc["evidence_0_to_2"] - without["evidence_0_to_2"],
                "expected_file_discovery_delta": len(with_dkc["expected_files_seen"])
                - len(without["expected_files_seen"]),
                "expected_symbol_discovery_delta": len(with_dkc["expected_symbols_seen"])
                - len(without["expected_symbols_seen"]),
                "wall_ratio_with_over_without": (
                    with_dkc["wall_seconds"] / without["wall_seconds"]
                    if without["wall_seconds"]
                    else None
                ),
                "wall_difference_with_minus_without": with_dkc["wall_seconds"]
                - without["wall_seconds"],
            }
        )
    usable = [pair for pair in pairs if pair["token_ratio_with_over_without"] is not None]
    summary = {
        arm: arm_summary(rows, arm) for arm in ["without_dkc", "with_dkc"]
    }
    without_tokens = summary["without_dkc"]["agent_usage"]["total_tokens"]
    with_tokens = summary["with_dkc"]["agent_usage"]["total_tokens"]
    ratio = with_tokens / without_tokens if without_tokens else None
    return {
        "arms": summary,
        "pairs": pairs,
        "median_paired_token_ratio": statistics.median(
            pair["token_ratio_with_over_without"] for pair in usable
        ),
        "median_paired_token_difference": statistics.median(
            pair["token_difference_with_minus_without"] for pair in pairs
        ),
        "median_paired_wall_difference_seconds": statistics.median(
            pair["wall_difference_with_minus_without"] for pair in pairs
        ),
        "median_paired_correctness_delta": statistics.median(
            pair["correctness_delta"] for pair in pairs
        ),
        "median_paired_evidence_delta": statistics.median(
            pair["evidence_delta"] for pair in pairs
        ),
        "ratio_of_total_tokens": ratio,
        "total_token_reduction_fraction": 1 - ratio if ratio is not None else None,
        "question_cluster_bootstrap_95_percentile_interval": percentile_interval(usable),
        "equal_or_better_correctness": summary["with_dkc"]["correctness_points"]
        >= summary["without_dkc"]["correctness_points"],
        "target_30_percent_fewer_tokens": ratio is not None and ratio <= 0.70,
    }


def main() -> int:
    args = parse()
    protocol_path = args.protocol.resolve()
    grades_path = args.grades.resolve()
    output_path = args.out.resolve()
    if output_path.exists() or output_path.with_suffix(".md").exists():
        raise SystemExit("summary outputs already exist")
    protocol = json.loads(protocol_path.read_text())
    grades = json.loads(grades_path.read_text())
    evidence = Path(protocol["evidence"])
    terminal = json.loads((evidence / "terminal.json").read_text())
    if terminal.get("state") != "collected":
        raise SystemExit("experiment collection is not terminal and complete")
    protocol_sha256 = sha256(protocol_path)
    if terminal.get("protocol_sha256") != protocol_sha256:
        raise SystemExit("terminal collection belongs to a different protocol")
    scheduled_ids = [row["id"] for row in protocol["schedule"]]
    terminal_ids = [row.get("id") for row in terminal.get("sessions", [])]
    if len(scheduled_ids) != len(set(scheduled_ids)) or terminal_ids != scheduled_ids:
        raise SystemExit("terminal collection does not exactly match the schedule")
    if sha256(Path(protocol["rubric"])) != protocol["rubric_sha256"]:
        raise SystemExit("frozen rubric changed after collection")
    if grades.get("protocol_sha256") != protocol_sha256:
        raise SystemExit("grading packet belongs to a different protocol")
    if not grades.get("grader") or not grades.get("method"):
        raise SystemExit("grades must identify the grader and grading method")
    grade_rows = {row["transcript_sha256"]: row for row in grades.get("sessions", [])}
    if len(grade_rows) != len(protocol["schedule"]):
        raise SystemExit("grades must cover every scheduled session exactly once")
    questions = {question["id"]: question for question in protocol["questions"]}
    rows = []
    for scheduled in protocol["schedule"]:
        path = evidence / f"{scheduled['id']}.json"
        session = json.loads(path.read_text())
        transcript_sha256 = sha256(path)
        grade = grade_rows.get(transcript_sha256)
        if grade is None:
            raise SystemExit(f"missing or stale grade for {scheduled['id']}")
        if grade.get("question_id") != scheduled["question_id"]:
            raise SystemExit(f"grade question differs from schedule for {scheduled['id']}")
        correctness = grade.get("correctness_0_to_2")
        support = grade.get("evidence_0_to_2")
        if correctness not in [0, 1, 2] or support not in [0, 1, 2] or not grade.get("reason"):
            raise SystemExit(f"invalid grade for {scheduled['id']}")
        if session["state"] != "answered" and (correctness != 0 or support != 0):
            raise SystemExit(f"non-answer must retain zero grade for {scheduled['id']}")
        if any(session.get(key) != scheduled.get(key) for key in ["id", "question_id", "condition", "repeat"]):
            raise SystemExit(f"session identity differs from schedule for {scheduled['id']}")
        tools = [tool for turn in session["turns"] for tool in turn.get("tools", [])]
        expected_files = questions[scheduled["question_id"]].get("expect_files", [])
        expected_symbols = questions[scheduled["question_id"]].get("expect_symbols", [])
        discovery_text = session.get("answer", "") + "\n" + "\n".join(
            tool["wire_text"] for tool in tools
        )
        rows.append(
            {
                **scheduled,
                "state": session["state"],
                "correctness_0_to_2": correctness,
                "evidence_0_to_2": support,
                "grade_reason": grade["reason"],
                "agent_usage": session["agent_usage"],
                "chat_requests": len(session["turns"]),
                "tool_calls": session["tool_calls"],
                "wall_seconds": session["wall_seconds"],
                "expected_files": expected_files,
                "expected_files_seen": [
                    expected_path for expected_path in expected_files if expected_path in discovery_text
                ],
                "expected_symbols": expected_symbols,
                "expected_symbols_seen": [
                    expected_symbol
                    for expected_symbol in expected_symbols
                    if expected_symbol in discovery_text
                ],
                "dkc_searches": sum(tool["name"] == "dkc_search" for tool in tools),
                "transcript_sha256": transcript_sha256,
            }
        )
    cohorts = {
        cohort: aggregate([row for row in rows if row["cohort"] == cohort])
        for cohort in sorted({row["cohort"] for row in rows})
    }
    reservations = terminal.get("dkc_new_reservations", [])
    provider_usage = {}
    for _identifier, provider, _day, calls, tokens, state in reservations:
        current = provider_usage.setdefault(
            provider,
            {"calls": 0, "tokens": 0, "states": {}, "cost": None},
        )
        current["calls"] += calls
        current["tokens"] += tokens
        current["states"][state] = current["states"].get(state, 0) + 1
    primary = cohorts["primary"]
    output = {
        "state": "graded",
        "protocol_sha256": protocol_sha256,
        "rubric_sha256": protocol["rubric_sha256"],
        "grades_sha256": sha256(grades_path),
        "grader": grades.get("grader"),
        "grading_method": grades.get("method"),
        "cohorts": cohorts,
        "primary_release_result": {
            "questions": 30,
            "paired_repetitions": 3,
            "sessions": 180,
            "token_reduction_fraction": primary["total_token_reduction_fraction"],
            "equal_or_better_correctness": primary["equal_or_better_correctness"],
            "target_met": primary["target_30_percent_fewer_tokens"]
            and primary["equal_or_better_correctness"],
        },
        "sessions": rows,
        "dkc_provider_usage": provider_usage,
        "dkc_changed_old_reservations": terminal.get("dkc_changed_old_reservations", []),
        "agent_cost": None,
        "dkc_cost": None,
        "cost_status": terminal.get("cost_status"),
        "uncertainty": (
            "Question-cluster bootstrap resamples the 30 frozen questions; three model repetitions per "
            "question are correlated. Results describe this model, revision, configuration and tool budget."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    markdown = output_path.with_suffix(".md")
    without = primary["arms"]["without_dkc"]
    with_dkc = primary["arms"]["with_dkc"]
    markdown.write_text(
        "# T4.9 paired agent token experiment\n\n"
        + token_result_sentence(
            primary["total_token_reduction_fraction"],
            output["primary_release_result"]["target_met"],
        )
        + "\n\n"
        "| arm | sessions | tokens | correctness | evidence | tool calls | wall seconds |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| without dkc | {without['sessions']} | {without['agent_usage']['total_tokens']} | "
        f"{without['correctness_points']}/{without['correctness_possible']} | "
        f"{without['evidence_points']}/{without['evidence_possible']} | "
        f"{without['tool_calls']} | {without['wall_seconds']:.1f} |\n"
        f"| with dkc | {with_dkc['sessions']} | {with_dkc['agent_usage']['total_tokens']} | "
        f"{with_dkc['correctness_points']}/{with_dkc['correctness_possible']} | "
        f"{with_dkc['evidence_points']}/{with_dkc['evidence_possible']} | "
        f"{with_dkc['tool_calls']} | {with_dkc['wall_seconds']:.1f} |\n\n"
        f"Median paired ratio: `{primary['median_paired_token_ratio']:.5f}`. "
        f"Question-cluster bootstrap interval: `{primary['question_cluster_bootstrap_95_percentile_interval']}`.\n\n"
        f"Median paired token difference: `{primary['median_paired_token_difference']}`; "
        f"median paired wall-time difference: "
        f"`{primary['median_paired_wall_difference_seconds']:.3f}s`.\n\n"
        f"Expected-file discovery: `{without['expected_files_found']}/{without['expected_files_possible']}` "
        f"without dkc and `{with_dkc['expected_files_found']}/{with_dkc['expected_files_possible']}` "
        "with dkc.\n\n"
        "Rates and monetary costs remain empty by user direction; raw provider usage is retained in JSON.\n"
    )
    print(json.dumps(output["primary_release_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
