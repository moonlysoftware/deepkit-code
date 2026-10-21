"""Candidate assembly only publishes complete, preflighted matrix inputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_preflight as preflight
import freeze_candidate
import test_candidate_preflight as fixtures


class FreezeCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = fixtures.CandidatePreflightTest(
            "test_complete_clean_candidate_passes_without_output"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.spec = self.root / "notes/bench/candidate-inputs.json"
        self.output = self.root / "notes/bench/frozen-candidate.json"
        self.record = {
            "name": "frozen-test",
            "binaries": {kind: entry["path"]
                         for kind, entry in fixture.builds.items()},
            "bindings": [
                {field: lane[field] for field in (
                    "target", "profile", "effective_config_input",
                    "model_identity_input",
                )}
                for lane in fixture.document["lanes"]
            ],
        }
        self.spec.write_text(json.dumps(self.record))
        self.addCleanup(patch.stopall)
        patch.object(preflight, "REQUIRED_LANE_PLAN",
                     "notes/bench/lane-plan.json").start()
        patch.object(preflight, "REQUIRED_MATRIX_BASELINE",
                     "notes/bench/matrix-baseline.json").start()
        original_check = preflight.check
        patch.object(
            freeze_candidate.frozen, "check",
            side_effect=lambda root, document: original_check(
                root, document, expected_targets=2, expected_lanes=2,
                expected_lane_plan="notes/bench/lane-plan.json",
            ),
        ).start()

    def test_assembles_and_freezes_exact_two_lane_fixture(self) -> None:
        document = freeze_candidate.freeze(self.root, self.spec, self.output)
        self.assertEqual(len(document["targets"]), 2)
        self.assertEqual(len(document["lanes"]), 2)
        self.assertEqual(document["binaries"]["cpu"], self.fixture.builds["cpu"])
        self.assertEqual(document["lanes"][0]["output"],
                         "notes/bench/frozen-test-sample-local_cpu.json")
        self.assertEqual(json.loads(self.output.read_text()), document)
        self.assertEqual(freeze_candidate.frozen.check(self.root, document), [])
        with self.assertRaisesRegex(ValueError, "already exists"):
            freeze_candidate.freeze(self.root, self.spec, self.output)
        self.spec.write_text(self.spec.read_text() + "\n")
        self.assertTrue(any("frozen input changed" in issue
                            for issue in freeze_candidate.frozen.check(self.root, document)))

    def test_missing_lane_binding_refuses_without_output(self) -> None:
        self.record["bindings"].pop()
        self.spec.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "every required lane"):
            freeze_candidate.freeze(self.root, self.spec, self.output)
        self.assertFalse(self.output.exists())

    def test_template_lists_every_required_lane_but_is_not_candidate_ready(self) -> None:
        template_path = self.root / "notes/bench/binding-template.json"
        form = freeze_candidate.template(self.root, "next", template_path,
                                         expected_lanes=2)
        self.assertEqual(len(form["bindings"]), 2)
        self.assertEqual(
            {(row["target"], row["profile"]) for row in form["bindings"]},
            {("sample", "local_cpu"), ("fixtures-docs", "local_cpu")},
        )
        self.assertFalse(all(row["effective_config_input"]
                             for row in form["bindings"]))
        with self.assertRaisesRegex(ValueError, "nonempty CPU and CUDA"):
            freeze_candidate.freeze(self.root, template_path, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_receipt_refuses_without_output(self) -> None:
        self.record["bindings"][0]["model_identity_input"] = "notes/bench/missing.json"
        self.spec.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "frozen input is missing"):
            freeze_candidate.freeze(self.root, self.spec, self.output)
        self.assertFalse(self.output.exists())

    def test_placeholder_receipt_refuses_before_creating_candidate(self) -> None:
        config_name = self.fixture.receipts["sample"][0]
        path = self.root / config_name
        config = json.loads(path.read_text())
        config["effective_config_blake3"] = "placeholder"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "effective-config receipt is incomplete"):
            freeze_candidate.freeze(self.root, self.spec, self.output)
        self.assertFalse(self.output.exists())

    def test_wrong_cuda_binary_and_stop_refuse_without_output(self) -> None:
        self.record["binaries"]["cuda"] = self.record["binaries"]["cpu"]
        self.spec.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "cuda: wrong feature selection"):
            freeze_candidate.freeze(self.root, self.spec, self.output)
        self.assertFalse(self.output.exists())
        self.record["binaries"]["cuda"] = self.fixture.builds["cuda"]["path"]
        self.spec.write_text(json.dumps(self.record))
        (self.root / "notes/STOP").touch()
        with self.assertRaisesRegex(ValueError, "notes/STOP"):
            freeze_candidate.freeze(self.root, self.spec, self.output)
        self.assertFalse(self.output.exists())

    def test_generated_fixture_tree_is_pinned(self) -> None:
        document = freeze_candidate.assemble(self.root, self.spec)
        fixture = next(row for row in document["targets"]
                       if row["name"] == "fixtures-docs")
        self.assertEqual(fixture["head"], self.fixture.docs_head)
        self.assertEqual(fixture["tree_sha256"], preflight.tree_sha256(self.fixture.docs))
        (self.fixture.docs / "page.md").write_text("changed\n")
        self.assertNotEqual(freeze_candidate.assemble(self.root, self.spec)["targets"][1]
                            ["tree_sha256"], fixture["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
