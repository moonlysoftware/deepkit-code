"""The frozen corpus auditor covers all five matrix profile kinds."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


AUDITOR = Path(__file__).resolve().parents[2] / "scripts/bench/audit_matrix.py"


def target(level: int, *, idle: bool = True) -> dict:
    """Minimal complete no-golden lane with consistent queue and ladder state."""
    return {
        "name": "fixture", "provenance": {"pre_index": {"idle": idle}},
        "robustness": {"panics": 0, "dead_jobs": 0},
        "model_index": {
            "requested_level": level, "pending_jobs": 0,
            "embedding_state": "quiescent",
            "coverage": {"indexed_units": 100, "covered_units": 100,
                         "pending_jobs": 0},
            "index_result": {"embedding": {"providers": []}},
        },
        "status": {"ladder": {"l1": 0, "l2": 100, "l3": 0}},
        "paired_search": {"state": "skipped",
                          "reason": "no_commit_derived_goldens", "goldens": 0},
    }


class MatrixAuditTest(unittest.TestCase):
    def test_special_profiles_require_gpu_and_their_actual_ladder(self) -> None:
        audit = runpy.run_path(str(AUDITOR))["audit_target"]
        for backend, level in (("ollama", 2), ("dual", 3)):
            with self.subTest(backend=backend):
                result = audit(target(level), backend)
                self.assertTrue(result["accepted"])
                self.assertFalse(any(row["metric"] == "l2_throughput_units_per_second"
                                     for row in result["release_misses"]))
                self.assertFalse(audit(target(level, idle=False), backend)["accepted"])
                self.assertFalse(audit(target(level + 1), backend)["accepted"])

    def test_special_profile_cli_rejects_empty_target_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend, level, features in (("ollama", 2, "DEFAULT"),
                                             ("dual", 3, "CUDA,DEFAULT")):
                report_path = root / f"{backend}.json"
                audit_path = root / f"{backend}-audit.json"
                report = {
                    "profile": backend, "failures": [],
                    "build": {"profile": "release", "source_dirty": "false",
                              "features": features},
                    "targets": [target(level)],
                }
                report_path.write_text(json.dumps(report))
                command = [sys.executable, str(AUDITOR), str(report_path),
                           "--backend", backend, "--out", str(audit_path)]
                self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
                self.assertTrue(json.loads(audit_path.read_text())["accepted"])
                report["targets"] = []
                report_path.write_text(json.dumps(report))
                self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
                self.assertFalse(json.loads(audit_path.read_text())["accepted"])
