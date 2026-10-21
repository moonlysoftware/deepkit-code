"""One-lane capture keeps model evidence incomplete and never overwrites receipts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_lane_receipts as capture


class CaptureLaneReceiptsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "notes/bench").mkdir(parents=True)
        self.document = {
            "target_head": "b" * 40,
            "model_evidence_complete": False,
            "effective_config": {
                "schema": "dkc.phase4.effective-config.v1",
                "target": "sample", "profile": "local_cpu",
                "effective_config_blake3": "a" * 64,
                "config_files_blake3": {"config.toml": "c" * 64},
                "starting_config_sha256": {},
                "capture_binary_sha256": "d" * 64,
                "capture_build": {"commit": "fixture"},
            },
            "model_identity_template": {
                "schema": "dkc.phase4.model-identity.v1",
                "target": "sample", "profile": "local_cpu",
                "model_usage": {"configured_profile": "local",
                                "configured_embedding": {},
                                "configured_reranker": {},
                                "docs_tier_configured": False,
                                "verified_models": {}},
                "identities": [],
            },
        }

    def test_writes_distinct_records_once_without_promoting_model_evidence(self) -> None:
        config, models = capture.write(self.root, "sample-local-cpu", self.document)
        self.assertEqual(json.loads(config.read_text()), self.document["effective_config"])
        self.assertEqual(json.loads(models.read_text())["identities"], [])
        before = (config.read_bytes(), models.read_bytes())
        with self.assertRaisesRegex(ValueError, "already exists"):
            capture.write(self.root, "sample-local-cpu", self.document)
        self.assertEqual((config.read_bytes(), models.read_bytes()), before)

    def test_refuses_a_false_complete_claim_or_wrong_lane(self) -> None:
        self.document["model_evidence_complete"] = True
        with self.assertRaisesRegex(ValueError, "valid config"):
            capture.write(self.root, "sample-local-cpu", self.document)
        self.document["model_evidence_complete"] = False
        self.document["model_identity_template"]["profile"] = "moonly"
        with self.assertRaisesRegex(ValueError, "valid config"):
            capture.write(self.root, "sample-local-cpu", self.document)
        self.assertFalse(list((self.root / "notes/bench").iterdir()))


if __name__ == "__main__":
    unittest.main()
