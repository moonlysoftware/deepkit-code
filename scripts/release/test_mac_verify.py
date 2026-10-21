#!/usr/bin/env python3
"""Offline regressions for complete, fail-closed Metal evidence collection."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("mac_verify", Path(__file__).with_name("mac_verify.py"))
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class FakeRun:
    environment = {}

    def __init__(self):
        self.calls = []

    def run(self, name, command, **kwargs):
        self.calls.append((name, command, kwargs))
        return False


class MetalVerifierTests(unittest.TestCase):
    def test_failed_probe_does_not_prevent_later_measurements(self):
        run = FakeRun()
        VERIFY.run_probes(run)
        self.assertEqual([call[0] for call in run.calls], [
            "embed-sanity", "throughput", "rerank-1", "rerank-2", "rerank-3", "idle-rss"])
        self.assertEqual(run.calls[1][2]["environment"]["DKC_BENCH_DOCUMENT_COUNT"], "2000")
        for _, command, _ in run.calls:
            self.assertEqual(command[command.index("--bin") + 1], "dkc")

    def test_numerical_miss_cannot_pass_with_zero_stage_exits(self):
        self.assertEqual(VERIFY.verdict({"throughput": False}, [{"exit_code": 0}]), "failed")
        self.assertEqual(VERIFY.verdict({"throughput": True}, [{"exit_code": 101}]), "failed")
        self.assertEqual(VERIFY.verdict({"throughput": True}, [{"exit_code": 0}]), "passed")
        self.assertEqual(VERIFY.verdict({}, []), "failed")

    def test_actual_stage_failure_and_json_stderr_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = VERIFY.Run(root, root / "result.json", dict(os.environ))
            self.assertFalse(run.run("first", [sys.executable, "-c", "raise SystemExit(7)"]))
            self.assertTrue(run.run("next", [sys.executable, "-c",
                'import sys; print("{}" ); print("diagnostic", file=sys.stderr)', "--json"]))
            self.assertEqual(json.loads(run.text("next")), {})
            self.assertEqual([stage["exit_code"] for stage in run.stages], [7, 0])
            stored = json.loads(run.path("stages", "json").read_text())
            self.assertEqual(stored, run.stages)
            self.assertIn("diagnostic", Path(stored[1]["stderr_log"]).read_text())
            with self.assertRaises(FileExistsError):
                run.run("first", [sys.executable, "-c", "pass"])

    def test_timeout_has_terminal_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = VERIFY.Run(root, root / "result.json", dict(os.environ))
            self.assertFalse(run.run("timeout", [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.1))
            self.assertEqual(run.stages[0]["exit_code"], 124)

    def test_three_sets_are_required_and_p95_is_checked_against_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = VERIFY.Run(root, root / "result.json", {})
            for index, value in enumerate((100.0, 200.0, 300.0), 1):
                record = {"candidates": 50, "sample_count": 20, "samples_ms": [value] * 20,
                          "requested_gpu_layers": -1, "p95_nearest_rank_ms": value}
                run.path(f"rerank-{index}").write_text("Metal Rank component diagnostic: " + json.dumps(record))
            values, checks = VERIFY.measurements(run)
            self.assertEqual(values["rerank"]["median_p95_ms"], 200.0)
            self.assertTrue(checks["rerank_50_candidates_p95"])
            record["p95_nearest_rank_ms"] = 0.0
            run.path("rerank-3").write_text("Metal Rank component diagnostic: " + json.dumps(record))
            self.assertFalse(VERIFY.measurements(run)[1]["rerank_50_candidates_p95"])
            run.path("rerank-3").unlink()
            self.assertFalse(VERIFY.measurements(run)[1]["rerank_50_candidates_p95"])
            record["samples_ms"] = None
            run.path("rerank-3").write_text("Metal Rank component diagnostic: " + json.dumps(record))
            self.assertFalse(VERIFY.measurements(run)[1]["rerank_50_candidates_p95"])

    def test_throughput_under_target_fails_even_if_native_test_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = VERIFY.Run(root, root / "result.json", {})
            run.path("throughput").write_text(
                "embedding benchmark: threads=4 metal=true, requested_gpu_layers=-1\n"
                "embedding throughput: 50.00 units/s (2000 documents in 40.000s)\n"
                "test result: ok. 1 passed; 0 failed; 0 ignored\n")
            values, checks = VERIFY.measurements(run)
            self.assertEqual(values["embedding"]["documents"], 2000)
            self.assertFalse(checks["throughput_2000_units"])

    def test_ripgrep_requires_terminal_local_l2_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = VERIFY.Run(root, root / "result.json", {})
            status = {"queue": {"dead": 0, "failed": 0, "queued": 0, "leased": 0},
                      "l2": {"covered_units": 10, "pending_jobs": 0,
                             "tables": [{"configured": {"roles": ["primary"]}, "profile": "local"}]}}
            run.path("ripgrep-status").write_text(json.dumps(status))
            checks = VERIFY.measurements(run)[1]
            self.assertTrue(checks["ripgrep_queue"])
            self.assertTrue(checks["ripgrep_l2"])
            status["queue"]["queued"] = 1
            status["l2"]["pending_jobs"] = 1
            run.path("ripgrep-status").write_text(json.dumps(status))
            checks = VERIFY.measurements(run)[1]
            self.assertFalse(checks["ripgrep_queue"])
            self.assertFalse(checks["ripgrep_l2"])
            for malformed in (
                {"queue": None, "l2": None},
                {"queue": [], "l2": {"covered_units": 10, "pending_jobs": 0, "tables": None}},
                {"queue": status["queue"], "l2": {"covered_units": 10,
                    "pending_jobs": 0, "tables": [None]}},
                {"queue": status["queue"], "l2": {"covered_units": None,
                    "pending_jobs": 0, "tables": []}},
            ):
                run.path("ripgrep-status").write_text(json.dumps(malformed))
                checks = VERIFY.measurements(run)[1]
                self.assertFalse(checks["ripgrep_l2"])

    def test_ripgrep_search_rejects_partial_or_degraded_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = VERIFY.Run(root, root / "result.json", {})
            result = {"hits": [{"path": "src/lib.rs"}],
                      "index": {"partial": True, "degraded": False}}
            run.path("ripgrep-search").write_text(json.dumps(result))
            self.assertFalse(VERIFY.measurements(run)[1]["ripgrep_search"])
            result["index"]["partial"] = False
            result["index"]["degraded"] = True
            run.path("ripgrep-search").write_text(json.dumps(result))
            self.assertFalse(VERIFY.measurements(run)[1]["ripgrep_search"])
            result["index"]["degraded"] = False
            run.path("ripgrep-search").write_text(json.dumps(result))
            self.assertTrue(VERIFY.measurements(run)[1]["ripgrep_search"])
            result["index"] = {"partial": False}
            run.path("ripgrep-search").write_text(json.dumps(result))
            self.assertFalse(VERIFY.measurements(run)[1]["ripgrep_search"])
            result["index"] = "malformed"
            run.path("ripgrep-search").write_text(json.dumps(result))
            self.assertFalse(VERIFY.measurements(run)[1]["ripgrep_search"])


if __name__ == "__main__":
    unittest.main()
