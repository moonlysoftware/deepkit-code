#!/usr/bin/env python3
"""Create a condition-blind grading packet from a completed T4.9 collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def build_packet(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
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
    questions = {question["id"]: question for question in protocol["questions"]}
    items = []
    for scheduled in protocol["schedule"]:
        path = evidence / f"{scheduled['id']}.json"
        session = json.loads(path.read_text())
        if session.get("state") not in {"answered", "incomplete", "failed"}:
            raise SystemExit(f"session is not terminal: {scheduled['id']}")
        if any(
            session.get(field) != scheduled.get(field)
            for field in ["id", "question_id", "cohort", "condition", "repeat"]
        ):
            raise SystemExit(f"session identity differs from schedule: {scheduled['id']}")
        question = questions[scheduled["question_id"]]
        items.append(
            {
                "transcript_sha256": sha256(path),
                "question_id": question["id"],
                "cohort": question["cohort"],
                "query": question["query"],
                "rubric": question["rubric"],
                "answer": session.get("answer"),
                "state": session["state"],
                "correctness_0_to_2": None,
                "evidence_0_to_2": None,
                "reason": "",
            }
        )
    seed = int(sha256(protocol_path)[:16], 16)
    random.Random(seed).shuffle(items)
    for index, item in enumerate(items, 1):
        item["blind_id"] = f"item-{index:03}"
    return {
        "task": "T4.9",
        "protocol_sha256": protocol_sha256,
        "grader": "",
        "method": "",
        "blinding": (
            "Condition, repetition and session ID are omitted. Grade only the answer against the "
            "frozen rubric; transcript hashes reconnect grades during reduction."
        ),
        "sessions": items,
    }


def main() -> int:
    args = parse()
    protocol_path = args.protocol.resolve()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit("grading packet output already exists")
    packet = build_packet(protocol_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"packet": str(output), "sessions": len(packet["sessions"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
