"""Frozen matrix resumes only reports that the real preflight can revalidate."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_preflight as frozen
import matrix_run
import nightly
import test_candidate_preflight as fixtures


class MatrixRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixtures.CandidatePreflightTest(
            "test_complete_clean_candidate_passes_without_output"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.manifest = self.root / "notes/bench/frozen-candidate.json"
        self.manifest.write_text(json.dumps(self.fixture.document))
        self.args = SimpleNamespace(candidate_manifest=self.manifest,
                                    log_dir=self.root / "notes/runs",
                                    candidate_limit=1, dry_run=False)
        original_check = frozen.check
        checker = patch.object(
            matrix_run.frozen, "check",
            side_effect=lambda root, document: original_check(
                root, document, expected_targets=2, expected_lanes=2,
                expected_lane_plan="notes/bench/lane-plan.json",
            ),
        )
        checker.start()
        self.addCleanup(checker.stop)
        self.calls = []
        owner = self

        def fake_observe(_root, lane, receipt, candidate_sha, phase, _cache):
            self.calls.append(("identity", lane["target"], phase))
            return {
                "schema": frozen.LIVE_MODEL_SCHEMA,
                "candidate_sha256": candidate_sha,
                "target": lane["target"], "profile": lane["profile"],
                "phase": phase, "ready": True,
                "identities": [
                    {**{field: row[field] for field in
                        ("role", "model", "provider", "kind", "identity")},
                     "matched": True}
                    for row in receipt["identities"]
                ],
            }

        observer = patch.object(matrix_run.model_identity, "observe",
                                side_effect=fake_observe)
        observer.start()
        self.addCleanup(observer.stop)
        self.fake_observe = fake_observe

        class FakeRunner:
            def __init__(self, log: Path):
                self.log = log
                log.write_text("fake retained attempt log\n")

            def idle(self, require_gpu: bool, timeout: int = 900) -> None:
                owner.calls.append(("idle", require_gpu))

            def run(self, arguments: list, **kwargs) -> int:
                output = Path(arguments[arguments.index("--out") + 1])
                if arguments[1:3] == ["bench", "run"]:
                    target = arguments[arguments.index("--only") + 1]
                    owner.calls.append(("bench", target))
                    output.write_text(json.dumps({
                        "profile": "cpu", "build": owner.fixture.builds["cpu"]["build"],
                        "failures": [], "targets": [{"name": target,
                                                     "provenance": owner.fixture.report_provenance(target)}],
                    }))
                    return 0
                report = json.loads(Path(arguments[2]).read_text())
                owner.calls.append(("audit", report["targets"][0]["name"]))
                output.write_text(json.dumps({
                    "accepted": True, "build": report["build"],
                    "targets": [{"name": report["targets"][0]["name"],
                                 "accepted": True}],
                    "release_misses": [],
                }))
                return 0

        self.runner = FakeRunner

    def run_candidate(self) -> int:
        return matrix_run.execute(self.args, root=self.root, runner_type=self.runner)

    def test_two_lanes_resume_without_rewriting_first_report(self) -> None:
        self.assertEqual(self.run_candidate(), 0)
        first = self.root / "notes/bench/new-sample.json"
        original = first.read_bytes()
        self.assertEqual(self.run_candidate(), 0)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample"), ("bench", "fixtures-docs")])
        state = json.loads(next(self.args.log_dir.glob("matrix-*.state.json")).read_text())
        self.assertEqual(len(state["completed"]), 2)
        self.assertEqual([row["state"] for row in state["attempts"]],
                         ["complete", "complete"])
        self.assertEqual(len([call for call in self.calls if call[0] == "identity"]), 4)

    def test_pre_lane_identity_drift_retains_failed_attempt_without_benchmark(self) -> None:
        self.runner = nightly.Runner
        def drift(root, lane, receipt, candidate_sha, phase, cache):
            result = self.fake_observe(root, lane, receipt, candidate_sha, phase, cache)
            result["ready"] = False
            result["identities"][0]["identity"] = "f" * 64
            result["identities"][0]["matched"] = False
            return result

        with patch.object(matrix_run.model_identity, "observe", side_effect=drift):
            self.assertEqual(self.run_candidate(), 1)
        self.assertFalse(any(call[0] == "bench" for call in self.calls))
        state = json.loads(next(self.args.log_dir.glob("matrix-*.state.json")).read_text())
        self.assertFalse(state["completed"])
        self.assertEqual(state["attempts"][0]["state"], "failed")
        self.assertEqual(set(state["attempts"][0]["identity_observations"]), {"before"})
        self.assertTrue(Path(state["attempts"][0]["log"]).is_file())
        with self.assertRaisesRegex(ValueError, "interrupted or failed attempt"):
            self.run_candidate()
        diagnosis = self.root / "notes/bench/model-admission-diagnosis.md"
        diagnosis.write_text("Frozen model changed before benchmark admission.\n")
        self.args.candidate_skip_lane = "sample:local_cpu"
        self.args.candidate_skip_reason = "Frozen model changed before benchmark admission"
        self.args.candidate_skip_evidence = diagnosis
        self.assertEqual(matrix_run.skip_failed(self.args, root=self.root), 0)

    def test_post_lane_identity_drift_retains_report_without_acceptance(self) -> None:
        def drift(root, lane, receipt, candidate_sha, phase, cache):
            result = self.fake_observe(root, lane, receipt, candidate_sha, phase, cache)
            if phase == "after":
                result["ready"] = False
                result["identities"][0]["identity"] = "f" * 64
                result["identities"][0]["matched"] = False
            return result

        with patch.object(matrix_run.model_identity, "observe", side_effect=drift):
            self.assertEqual(self.run_candidate(), 1)
        self.assertTrue((self.root / "notes/bench/new-sample.json").is_file())
        state = json.loads(next(self.args.log_dir.glob("matrix-*.state.json")).read_text())
        self.assertFalse(state["completed"])
        self.assertEqual(set(state["attempts"][0]["identity_observations"]),
                         {"before", "after"})
        evidence = self.root / "notes/bench/model-drift-diagnosis.md"
        evidence.write_text("Model deployment changed during this lane.\n")
        self.args.candidate_skip_lane = "sample:local_cpu"
        self.args.candidate_skip_reason = "Model deployment changed during the lane"
        self.args.candidate_skip_evidence = evidence
        self.assertEqual(matrix_run.skip_failed(self.args, root=self.root), 0)
        self.assertEqual(self.run_candidate(), 0)
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample"), ("bench", "fixtures-docs")])

    def test_lane_config_is_restored_before_next_lane_and_after_failure(self) -> None:
        target = next(row for row in self.fixture.document["targets"]
                      if row["name"] == "sample")
        local = self.fixture.sample / ".deepkit/config.local.toml"
        repository = self.fixture.sample / ".deepkit/config.toml"
        local.parent.mkdir()
        local.write_text("[embedder.local]\ngpu_layers = 7\n")
        repository.write_text("[limits]\nmax_file_bytes = '1 MiB'\n")
        target["config_baseline"] = {
            "config.local.toml": frozen.sha256(local),
            "config.toml": frozen.sha256(repository),
        }
        for name in ("sample",):
            path = self.root / self.fixture.receipts[name][0]
            receipt = json.loads(path.read_text())
            receipt["starting_config_sha256"] = target["config_baseline"]
            path.write_text(json.dumps(receipt))
            next(row for row in self.fixture.document["inputs"]
                 if row["path"] == self.fixture.receipts[name][0])["sha256"] = frozen.sha256(path)
        self.manifest.write_text(json.dumps(self.fixture.document))
        before = (local.read_bytes(), repository.read_bytes())
        parent = self.runner

        class MutatingRunner(parent):
            def run(self, arguments: list, **kwargs) -> int:
                if arguments[1:3] == ["bench", "run"] and \
                        arguments[arguments.index("--only") + 1] == "sample":
                    local.write_text("[embedder.local]\ngpu_layers = 0\n")
                    repository.write_text("[embedder]\nprofile = 'remote'\n")
                return super().run(arguments, **kwargs)

        self.runner = MutatingRunner
        self.assertEqual(self.run_candidate(), 0)
        self.assertEqual((local.read_bytes(), repository.read_bytes()), before)
        self.assertEqual(self.run_candidate(), 0)
        self.assertEqual((local.read_bytes(), repository.read_bytes()), before)
        self.assertFalse(list(local.parent.glob("bench-config-recovery-*")))

        with self.assertRaisesRegex(RuntimeError, "lane failed"):
            with matrix_run.restored_target_config(target, "f" * 64):
                local.write_text("changed by failed lane\n")
                raise RuntimeError("lane failed")
        self.assertEqual((local.read_bytes(), repository.read_bytes()), before)

    def test_changed_starting_config_refuses_lane_before_idle(self) -> None:
        local = self.fixture.sample / ".deepkit/config.local.toml"
        local.parent.mkdir()
        local.write_text("unplanned config\n")
        self.assertEqual(self.run_candidate(), 2)
        self.assertEqual(self.calls, [])

    def test_unresolved_config_recovery_is_never_overwritten(self) -> None:
        target = self.fixture.document["targets"][0]
        parent = self.fixture.sample / ".deepkit"
        parent.mkdir()
        recovery = parent / ("bench-config-recovery-" + "f" * 16)
        recovery.mkdir()
        (recovery / "manifest.json").write_text("retained recovery\n")
        with self.assertRaisesRegex(ValueError, "unresolved benchmark config recovery"):
            with matrix_run.restored_target_config(target, "f" * 64):
                self.fail("recovery should stop before a lane starts")
        self.assertEqual((recovery / "manifest.json").read_text(), "retained recovery\n")

    def test_drifted_completed_report_refuses_resume(self) -> None:
        self.assertEqual(self.run_candidate(), 0)
        first = self.root / "notes/bench/new-sample.json"
        first.write_text(first.read_text() + "\n")
        self.assertEqual(self.run_candidate(), 2)
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample")])

    def test_drifted_live_observation_refuses_resume(self) -> None:
        self.assertEqual(self.run_candidate(), 0)
        observation = self.root / "notes/bench/new-sample-identity-after.json"
        observation.write_text(observation.read_text() + "\n")
        self.assertEqual(self.run_candidate(), 2)
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample")])

    def test_malformed_attempt_state_refuses_resume(self) -> None:
        self.assertEqual(self.run_candidate(), 0)
        state_path = next(self.args.log_dir.glob("matrix-*.state.json"))
        state = json.loads(state_path.read_text())
        state["attempts"].append(None)
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "malformed interrupted attempt"):
            self.run_candidate()
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample")])

    def test_failed_attempt_is_preserved_and_never_retried(self) -> None:
        parent = self.runner

        class FailedRunner(parent):
            def run(self, arguments: list, **kwargs) -> int:
                if arguments[1:3] == ["bench", "run"]:
                    super().run(arguments, **kwargs)
                    return 1
                return super().run(arguments, **kwargs)

        self.runner = FailedRunner
        self.assertEqual(self.run_candidate(), 1)
        first = self.root / "notes/bench/new-sample.json"
        original = first.read_bytes()
        with self.assertRaisesRegex(ValueError, "interrupted or failed attempt"):
            self.run_candidate()
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample")])

    def test_rejected_audit_retains_report_and_blocks_next_lane(self) -> None:
        parent = self.runner

        class RejectedRunner(parent):
            def run(self, arguments: list, **kwargs) -> int:
                code = super().run(arguments, **kwargs)
                if arguments[1:3] != ["bench", "run"]:
                    audit = Path(arguments[arguments.index("--out") + 1])
                    record = json.loads(audit.read_text())
                    record["accepted"] = False
                    audit.write_text(json.dumps(record))
                    return 1
                return code

        self.runner = RejectedRunner
        self.assertEqual(self.run_candidate(), 1)
        report = self.root / "notes/bench/new-sample.json"
        self.assertTrue(report.is_file())
        self.assertTrue((self.root / "notes/bench/new-sample-audit.json").is_file())
        with self.assertRaisesRegex(ValueError, "interrupted or failed attempt"):
            self.run_candidate()
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample")])

    def test_explicit_evidenced_skip_continues_other_lane_without_acceptance(self) -> None:
        parent = self.runner

        class FailedRunner(parent):
            def run(self, arguments: list, **kwargs) -> int:
                code = super().run(arguments, **kwargs)
                if arguments[1:3] == ["bench", "run"]:
                    local = owner.fixture.sample / ".deepkit/config.local.toml"
                    local.write_text("[embedder.local]\ngpu_layers = 0\n")
                    return 1
                return code

        owner = self
        self.runner = FailedRunner
        self.assertEqual(self.run_candidate(), 1)
        self.assertFalse((self.fixture.sample / ".deepkit/config.local.toml").exists())
        evidence = self.root / "notes/bench/sample-diagnosis.md"
        evidence.write_text("Target-specific benchmark failure under review.\n")
        self.args.candidate_skip_lane = "sample:local_cpu"
        self.args.candidate_skip_reason = "Target-specific failure, pending repair"
        self.args.candidate_skip_evidence = evidence
        with patch.object(nightly, "ROOT", self.root), \
             patch.object(nightly, "SHARED_LOCK",
                          self.args.log_dir / "t4-2-benchmark.lock"), \
             patch.object(sys, "argv", [
                 "nightly.py", "--candidate-manifest", str(self.manifest),
                 "--log-dir", str(self.args.log_dir),
                 "--candidate-skip-lane", self.args.candidate_skip_lane,
                 "--candidate-skip-reason", self.args.candidate_skip_reason,
                 "--candidate-skip-evidence", str(evidence),
             ]):
            self.assertEqual(nightly.main(), 0)
        report = self.root / "notes/bench/new-sample.json"
        report_bytes = report.read_bytes()
        self.runner = parent
        self.assertEqual(self.run_candidate(), 0)
        self.assertEqual(report.read_bytes(), report_bytes)
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample"), ("bench", "fixtures-docs")])
        state = json.loads(next(self.args.log_dir.glob("matrix-*.state.json")).read_text())
        self.assertEqual(set(state["skipped"]), {"sample:local_cpu"})
        self.assertEqual(set(state["completed"]), {"fixtures-docs:local_cpu"})
        receipt = self.root / "notes/bench/new-sample-skip.json"
        self.assertTrue(receipt.is_file())
        self.assertFalse(json.loads(receipt.read_text()).get("accepted", False))
        self.assertEqual(self.run_candidate(), 0)
        receipt.write_text(receipt.read_text() + "changed\n")
        self.assertEqual(self.run_candidate(), 2)

    def test_skip_requires_failed_lane_and_diagnosis(self) -> None:
        evidence = self.root / "notes/bench/sample-diagnosis.md"
        evidence.write_text("No failure has occurred.\n")
        self.args.candidate_skip_lane = "sample:local_cpu"
        self.args.candidate_skip_reason = "This lane has no failed attempt"
        self.args.candidate_skip_evidence = evidence
        with self.assertRaisesRegex(ValueError, "no failed attempt"):
            matrix_run.skip_failed(self.args, root=self.root)
        self.assertFalse((self.root / "notes/bench/new-sample-skip.json").exists())

    def test_interrupted_before_report_can_be_skipped_but_evidence_drift_refuses(self) -> None:
        parent = self.runner

        class InterruptedRunner(parent):
            def idle(self, require_gpu: bool, timeout: int = 900) -> None:
                raise TimeoutError("host never became idle")

        self.runner = InterruptedRunner
        self.assertEqual(self.run_candidate(), 1)
        report = self.root / "notes/bench/new-sample.json"
        self.assertFalse(report.exists())
        evidence = self.root / "notes/bench/host-diagnosis.md"
        evidence.write_text("Host load did not meet the admission condition.\n")
        self.args.candidate_skip_lane = "sample:local_cpu"
        self.args.candidate_skip_reason = "Host admission failed; lane remains open"
        self.args.candidate_skip_evidence = evidence
        self.assertEqual(matrix_run.skip_failed(self.args, root=self.root), 0)
        self.runner = parent
        self.assertEqual(self.run_candidate(), 0)
        self.assertFalse(report.exists())
        evidence.write_text(evidence.read_text() + "changed\n")
        self.assertEqual(self.run_candidate(), 2)

    def test_rejected_audit_is_pinned_by_skip(self) -> None:
        parent = self.runner

        class RejectedRunner(parent):
            def run(self, arguments: list, **kwargs) -> int:
                code = super().run(arguments, **kwargs)
                if arguments[1:3] != ["bench", "run"]:
                    return 1
                return code

        self.runner = RejectedRunner
        self.assertEqual(self.run_candidate(), 1)
        evidence = self.root / "notes/bench/audit-diagnosis.md"
        evidence.write_text("Lane auditor rejected a target-specific result.\n")
        self.args.candidate_skip_lane = "sample:local_cpu"
        self.args.candidate_skip_reason = "Rejected audit needs target-specific repair"
        self.args.candidate_skip_evidence = evidence
        self.assertEqual(matrix_run.skip_failed(self.args, root=self.root), 0)
        receipt = json.loads((self.root / "notes/bench/new-sample-skip.json").read_text())
        self.assertEqual(receipt["rejected_audit"]["path"],
                         "notes/bench/new-sample-audit.json")
        audit = self.root / receipt["rejected_audit"]["path"]
        audit.write_text(audit.read_text() + "changed\n")
        self.runner = parent
        self.assertEqual(self.run_candidate(), 2)
        self.assertEqual([call for call in self.calls if call[0] == "bench"],
                         [("bench", "sample")])

    def test_stop_and_dry_run_never_start_a_lane(self) -> None:
        self.args.dry_run = True
        self.assertEqual(self.run_candidate(), 0)
        self.assertEqual(self.calls, [])
        (self.root / "notes/STOP").touch()
        self.assertEqual(self.run_candidate(), 2)
        self.assertEqual(self.calls, [])

    def test_candidate_cli_refuses_occupied_live_corpus_slot(self) -> None:
        lock_path = self.args.log_dir / "t4-2-benchmark.lock"
        with lock_path.open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(nightly, "ROOT", self.root), \
                 patch.object(nightly, "SHARED_LOCK", lock_path), \
                 patch.object(sys, "argv", ["nightly.py", "--candidate-manifest",
                                            str(self.manifest), "--log-dir",
                                            str(self.args.log_dir)]):
                self.assertEqual(nightly.main(), 75)
        self.assertEqual(self.calls, [])
