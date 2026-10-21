#!/usr/bin/env python3
"""Offline safety and denominator controls for the T4.9 experiment runner."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch


def import_script(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = import_script("token_experiment", "token-experiment.py")
PREPARE = import_script("prepare_token_experiment", "prepare-token-experiment.py")
SUMMARIZE = import_script("summarize_token_experiment", "summarize-token-experiment.py")
GRADES = import_script("prepare_token_grades", "prepare-token-grades.py")


def protocol() -> dict:
    return {
        "binary": "unused",
        "dkc_environment": {},
        "agent": {
            "model": "fixture",
            "endpoint": "https://invalid.example",
            "temperature": 0,
            "extra_body": {},
        },
        "user_agent": "dkc-test/0",
        "system_prompt": "test",
        "limits": {
            "chat_requests_per_session": 3,
            "tools_per_session": 2,
            "force_answer_after_tools": 1,
            "max_output_tokens_per_request": 100,
            "request_token_overhead": 100,
            "conservative_token_admission_per_session": 10_000,
            "chat_request_timeout_seconds": 1,
            "session_timeout_seconds": 3,
            "dkc_tool_timeout_seconds": 1,
            "agent_rpm": 120,
            "agent_tpm_conservative": 100_000,
        },
    }


class Controls(unittest.TestCase):
    def test_schedule_is_complete_reproducible_and_counterbalanced(self):
        questions = [
            {
                "id": f"q-{index:02}",
                "cohort": "primary" if index < 30 else "held_out",
                "query": f"question {index}",
            }
            for index in range(36)
        ]
        schedule = PREPARE.build_schedule(questions)
        self.assertEqual(schedule, PREPARE.build_schedule(questions))
        self.assertEqual(len(schedule), 216)
        for repeat in range(1, 4):
            pairs = [row for row in schedule if row["repeat"] == repeat]
            primary_first = [
                pairs[index]["condition"]
                for index in range(0, len(pairs), 2)
                if pairs[index]["cohort"] == "primary"
            ]
            self.assertEqual(primary_first.count("with_dkc"), 15)
            self.assertEqual(primary_first.count("without_dkc"), 15)

    def test_release_user_agent_comes_from_exact_binary_version(self):
        self.assertEqual(PREPARE.user_agent("dkc 1.0.0"), "dkc/1.0.0")
        self.assertEqual(
            PREPARE.user_agent("dkc 1.0.0-rc.1+release"),
            "dkc/1.0.0-rc.1+release",
        )
        with self.assertRaises(SystemExit):
            PREPARE.user_agent("deepkit-code unknown")

    def test_frozen_rubric_matches_golden_inputs(self):
        repository = Path(__file__).resolve().parents[2]
        rubric = json.loads(
            (repository / "bench/goldens/ripgrep-rubric.json").read_text()
        )
        PREPARE.validate_rubric(
            rubric["questions"],
            [
                repository / "bench/goldens/ripgrep.yaml",
                repository / "bench/goldens/ripgrep-held-out.yaml",
            ],
        )

    def test_copied_index_must_serve_healthy_lexical_search(self):
        head = "a" * 40
        response = {"hits": [{"path": "crates/regex/src/matcher.rs"}],
                    "index": {"partial": False, "degraded": False,
                              "lexical": "complete", "head": head, "indexed_head": head}}
        self.assertEqual(PREPARE.validate_lexical_probe(response, head)["hits"], 1)
        for changed in (
            {**response, "hits": []},
            {**response, "index": {**response["index"], "partial": True}},
            {**response, "index": {**response["index"], "degraded": True}},
            {**response, "index": {**response["index"], "indexed_head": "b" * 40}},
        ):
            with self.assertRaises(SystemExit):
                PREPARE.validate_lexical_probe(changed, head)

    def test_initial_index_fingerprint_refuses_drift_and_live_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / ".deepkit/code.db"
            database.parent.mkdir()
            database.write_bytes(b"frozen index")
            value = {"database": ".deepkit/code.db",
                     "initial_database_sha256": PREPARE.initial_database_sha256(database)}
            RUNNER.verify_initial_database(value, root)
            database.write_bytes(b"changed index")
            with self.assertRaises(RuntimeError):
                RUNNER.verify_initial_database(value, root)
            database.write_bytes(b"frozen index")
            Path(f"{database}-wal").write_bytes(b"unapplied")
            with self.assertRaises(RuntimeError):
                RUNNER.verify_initial_database(value, root)
            with self.assertRaises(SystemExit):
                PREPARE.initial_database_sha256(database)

    def test_paths_and_unavailable_dkc_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.rs").write_text("a\nb\n")
            for name in ["../secret", "/etc/passwd", ".deepkit/config.toml"]:
                with self.assertRaises(ValueError):
                    RUNNER.permitted_file(root, {"a.rs"}, name)
            with self.assertRaises(ValueError):
                RUNNER.execute_tool(
                    "dkc_search",
                    {"query": "x"},
                    "without_dkc",
                    root,
                    {"a.rs"},
                    protocol(),
                    RUNNER.time.monotonic() + 1,
                )

    def test_symlink_and_oversized_output_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").symlink_to("/etc/passwd")
            with self.assertRaises(ValueError):
                RUNNER.permitted_file(root, {"a"}, "a")
        rendered = RUNNER.wire_output({"text": "é" * 9_000})
        self.assertLess(len(rendered.encode()), 8_000)
        self.assertTrue(json.loads(rendered)["truncated"])

    def test_protocol_requires_thirty_primary_complete_rubrics(self):
        questions = [
            {"id": f"q-{index}", "cohort": "primary", "rubric": {"facts": ["x"], "evidence": ["a:1"]}}
            for index in range(30)
        ]
        schedule = [
            {
                "id": f"{question['id']}-r{repeat}-{condition}",
                "question_id": question["id"],
                "cohort": "primary",
                "query": "q",
                "repeat": repeat,
                "condition": condition,
            }
            for question in questions
            for repeat in range(1, 4)
            for condition in ["without_dkc", "with_dkc"]
        ]
        value = {
            "task": "T4.9",
            "state": "frozen_before_dispatch",
            "initial_database_sha256": "a" * 64,
            "binary_version": "dkc 1.0.0",
            "user_agent": "dkc/1.0.0",
            "questions": questions,
            "schedule": schedule,
        }
        RUNNER.validate_protocol(value)
        value["user_agent"] = "dkc/0.1.0"
        with self.assertRaises(RuntimeError):
            RUNNER.validate_protocol(value)
        value["user_agent"] = "dkc/1.0.0"
        value["schedule"].pop()
        with self.assertRaises(RuntimeError):
            RUNNER.validate_protocol(value)

    def test_transport_failure_is_retained_once_without_details_or_retry(self):
        class Broken:
            calls = 0

            def open(self, *_args, **_kwargs):
                self.calls += 1
                raise OSError("private transport details")

        broken = Broken()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            RUNNER.urllib.request, "build_opener", return_value=broken
        ):
            output = Path(directory) / "result.json"
            item = {
                "id": "fixture",
                "question_id": "q",
                "cohort": "primary",
                "query": "test",
                "repeat": 1,
                "condition": "without_dkc",
            }
            result = RUNNER.run_session(
                protocol(), item, Path(directory), set(), "secret", output, RUNNER.RateWindow()
            )
            self.assertEqual(result["state"], "failed")
            self.assertEqual(broken.calls, 1)
            retained = output.read_text()
            self.assertNotIn("private transport", retained)
            self.assertNotIn("secret", retained)

    def test_token_exhaustion_is_an_incomplete_denominator_result(self):
        value = protocol()
        value["system_prompt"] = "x" * 10_000
        with tempfile.TemporaryDirectory() as directory, patch.object(
            RUNNER.urllib.request, "build_opener"
        ) as opener:
            output = Path(directory) / "result.json"
            item = {
                "id": "fixture",
                "question_id": "q",
                "cohort": "primary",
                "query": "test",
                "repeat": 1,
                "condition": "without_dkc",
            }
            result = RUNNER.run_session(
                value, item, Path(directory), set(), "secret", output, RUNNER.RateWindow()
            )
            self.assertEqual(result["state"], "incomplete")
            self.assertEqual(result["turns"], [])
            opener.return_value.open.assert_not_called()

    def test_tool_call_round_trip_is_retained_and_answered(self):
        class Response:
            def __init__(self, payload):
                self.payload = json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return self.payload

        class Opener:
            def __init__(self):
                self.responses = [
                    Response(
                        {
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 4,
                                "total_tokens": 14,
                            },
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "call-1",
                                                "type": "function",
                                                "function": {
                                                    "name": "list_files",
                                                    "arguments": '{"glob":"*.rs"}',
                                                },
                                            }
                                        ],
                                    }
                                }
                            ],
                        }
                    ),
                    Response(
                        {
                            "usage": {
                                "prompt_tokens": 20,
                                "completion_tokens": 5,
                                "total_tokens": 25,
                            },
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "Found a.rs.",
                                    }
                                }
                            ],
                        }
                    ),
                ]
                self.requests = []

            def open(self, request, **_kwargs):
                self.requests.append(json.loads(request.data))
                return self.responses.pop(0)

        opener = Opener()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.rs").write_text("fn main() {}\n")
            output = root / "result.json"
            item = {
                "id": "fixture",
                "question_id": "q",
                "cohort": "primary",
                "query": "find rust",
                "repeat": 1,
                "condition": "without_dkc",
            }
            with patch.object(
                RUNNER.urllib.request, "build_opener", return_value=opener
            ), patch.object(RUNNER.time, "sleep"):
                result = RUNNER.run_session(
                    protocol(), item, root, {"a.rs"}, "secret", output, RUNNER.RateWindow()
                )

            self.assertEqual(result["state"], "answered")
            self.assertEqual(result["answer"], "Found a.rs.")
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(
                result["agent_usage"],
                {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39},
            )
            self.assertEqual(
                result["turns"][0]["tools"][0]["result"],
                {"paths": ["a.rs"], "truncated": False},
            )
            self.assertEqual(
                opener.requests[1]["messages"][-1],
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": '{"paths": ["a.rs"], "truncated": false}',
                },
            )
            self.assertEqual(json.loads(output.read_text())["state"], "answered")

    def test_redirect_handler_never_forwards_credentials(self):
        self.assertIsNone(RUNNER.NoRedirect().redirect_request(None, None, None, None, None, None))

    def test_interrupted_session_is_retained_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            value = {
                "id": "q-r1-with_dkc",
                "question_id": "q",
                "cohort": "primary",
                "condition": "with_dkc",
                "repeat": 1,
                "state": "running",
                "tool_calls": 0,
                "turns": [{"state": "dispatched", "wall_seconds": 1.5}],
                "agent_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            path.write_text(json.dumps(value))
            retained = RUNNER.interrupted_session(path)
            self.assertEqual(retained["state"], "failed")
            self.assertTrue(retained["interrupted"])
            self.assertEqual(retained["wall_seconds"], 1.5)
            self.assertEqual(RUNNER.interrupted_session(path), retained)

    def test_resume_cooldown_ignores_old_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            old = "2000-01-01T00:00:00Z"
            (evidence / "session.json").write_text(
                json.dumps({"turns": [{"started_utc": old}]})
            )
            self.assertEqual(RUNNER.resume_cooldown(evidence), 0.0)

    def test_usage_accounting_refuses_an_unreadable_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.db").write_text("not sqlite")
            with self.assertRaises(sqlite3.DatabaseError):
                RUNNER.usage_reservations(root, "code.db")

    def test_summary_uses_paired_tokens_quality_latency_and_discovery(self):
        rows = []
        for condition, tokens, wall, files in [
            ("without_dkc", 100, 12.0, []),
            ("with_dkc", 70, 8.0, ["source.rs"]),
        ]:
            rows.append(
                {
                    "id": f"q-r1-{condition}",
                    "question_id": "q",
                    "repeat": 1,
                    "condition": condition,
                    "state": "answered",
                    "correctness_0_to_2": 2,
                    "evidence_0_to_2": 2,
                    "agent_usage": {
                        "prompt_tokens": tokens - 10,
                        "completion_tokens": 10,
                        "total_tokens": tokens,
                    },
                    "chat_requests": 1,
                    "tool_calls": 1,
                    "wall_seconds": wall,
                    "expected_files": ["source.rs"],
                    "expected_files_seen": files,
                    "expected_symbols": ["Thing"],
                    "expected_symbols_seen": ["Thing"] if files else [],
                }
            )
        summary = SUMMARIZE.aggregate(rows)
        self.assertEqual(summary["ratio_of_total_tokens"], 0.7)
        self.assertEqual(summary["median_paired_token_difference"], -30)
        self.assertEqual(summary["median_paired_wall_difference_seconds"], -4.0)
        self.assertTrue(summary["equal_or_better_correctness"])
        self.assertTrue(summary["target_30_percent_fewer_tokens"])
        self.assertEqual(summary["arms"]["with_dkc"]["expected_files_found"], 1)

    def test_token_result_sentence_reports_increase_without_negative_fewer(self):
        self.assertEqual(
            SUMMARIZE.token_result_sentence(-0.038500665608668605, False),
            "Result: **3.85% more agent tokens** with Deepkit Code; target not met.",
        )

    def test_grading_packet_omits_condition_repeat_and_session_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            evidence.mkdir()
            session = {
                "id": "q-r1-with_dkc",
                "question_id": "q",
                "cohort": "primary",
                "condition": "with_dkc",
                "repeat": 1,
                "state": "answered",
                "answer": "Supported answer.",
            }
            (evidence / "q-r1-with_dkc.json").write_text(json.dumps(session))
            protocol = {
                "evidence": str(evidence),
                "questions": [
                    {
                        "id": "q",
                        "cohort": "primary",
                        "query": "Question?",
                        "rubric": {"facts": ["fact"], "evidence": ["source:1"]},
                    }
                ],
                "schedule": [
                    {
                        "id": "q-r1-with_dkc",
                        "question_id": "q",
                        "cohort": "primary",
                        "condition": "with_dkc",
                        "repeat": 1,
                    }
                ],
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            terminal = {
                "state": "collected",
                "protocol_sha256": GRADES.sha256(protocol_path),
                "sessions": [{"id": "q-r1-with_dkc"}],
            }
            terminal_path = evidence / "terminal.json"
            terminal_path.write_text(json.dumps(terminal))
            packet = GRADES.build_packet(protocol_path)
            item = packet["sessions"][0]
            self.assertNotIn("condition", item)
            self.assertNotIn("repeat", item)
            self.assertNotIn("id", item)
            self.assertEqual(item["answer"], "Supported answer.")
            self.assertIsNone(item["correctness_0_to_2"])
            terminal["protocol_sha256"] = "0" * 64
            terminal_path.write_text(json.dumps(terminal))
            with self.assertRaises(SystemExit):
                GRADES.build_packet(protocol_path)


if __name__ == "__main__":
    unittest.main()
