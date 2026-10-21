"""Live lane identity checks reject changed bytes and deployment selections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_identity as live


class LiveModelIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "notes/bench").mkdir(parents=True)
        (self.root / "Cargo.toml").write_text('[package]\nversion = "1.0.0"\n')
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.file = self.cache / "model.gguf"
        self.file.write_bytes(b"first-model")
        self.expected = hashlib.sha256(b"first-model").hexdigest()
        self.evidence_name = "notes/bench/model-evidence.json"
        (self.root / self.evidence_name).write_text(json.dumps({
            "cache_file": "model.gguf",
        }))
        self.lane = {"target": "sample", "profile": "local_cpu"}
        self.receipt = {"identities": [{
            "role": "embedder", "model": "fixture-embedder", "provider": "local",
            "kind": "sha256", "identity": self.expected,
            "evidence_input": self.evidence_name,
        }]}

    def test_local_bytes_are_hashed_once_until_metadata_changes(self) -> None:
        cache = {}
        with patch.dict("os.environ", {"DKC_MODELS_DIR": str(self.cache)}), \
             patch.object(live.frozen, "sha256", wraps=live.frozen.sha256) as hashed:
            before = live.observe(self.root, self.lane, self.receipt, "a" * 64,
                                  "before", cache)
            after = live.observe(self.root, self.lane, self.receipt, "a" * 64,
                                 "after", cache)
            self.assertTrue(before["ready"] and after["ready"])
            self.assertEqual(hashed.call_count, 1)
            self.file.write_bytes(b"other-model")
            changed = live.observe(self.root, self.lane, self.receipt, "a" * 64,
                                   "after", cache)
            self.assertFalse(changed["ready"])
            self.assertEqual(changed["identities"][0]["issue"], "identity_changed")
            self.assertEqual(hashed.call_count, 2)

    def test_remote_header_is_hashed_and_never_retained_raw(self) -> None:
        token = "marker"
        raw_id = "deployment-private-id"
        headers = {"x-litellm-model-id": raw_id,
                   "x-litellm-model-group": "fixture-model"}
        with patch.object(live, "credential", return_value=token), \
             patch.object(live, "request_json", return_value=(
                 {"data": [{"embedding": [0.1]}]}, headers,
             )) as requested:
            digest = live.moon_credential_probe("embedder", "fixture-model",
                                                user_agent="dkc/1.0.0")
        self.assertEqual(digest, hashlib.sha256(raw_id.encode()).hexdigest())
        self.assertEqual(requested.call_args.kwargs["user_agent"], "dkc/1.0.0")
        self.assertEqual(requested.call_args.kwargs["token"], token)
        self.assertNotIn(raw_id, digest)

    def test_remote_alias_change_is_refused(self) -> None:
        headers = {"x-litellm-model-id": "another-deployment",
                   "x-litellm-model-group": "wrong-alias"}
        with patch.object(live, "credential", return_value="private-test-token"), \
             patch.object(live, "request_json", return_value=(
                 {"results": [{"relevance_score": 0.9}]}, headers,
             )):
            with self.assertRaisesRegex(ValueError, "complete selected-deployment"):
                live.moon_credential_probe("reranker", "fixture-model",
                                           user_agent="dkc/1.0.0")

    def test_ollama_uses_exact_model_tag_digest(self) -> None:
        with patch.object(live, "request_json", return_value=(
            {"models": [{"name": "fixture:latest", "digest": "b" * 64}]}, {},
        )):
            self.assertEqual(live.ollama_digest("fixture:latest",
                                                user_agent="dkc/1.0.0"), "b" * 64)
            with self.assertRaisesRegex(ValueError, "unique content digest"):
                live.ollama_digest("other:latest", user_agent="dkc/1.0.0")

    def test_probe_error_does_not_expose_secret_in_observation(self) -> None:
        self.receipt["identities"][0].update(provider="remote", kind="deployment")
        with patch.object(live, "moon_credential_probe",
                          side_effect=ValueError("private-test-token")):
            observed = live.observe(self.root, self.lane, self.receipt, "a" * 64,
                                    "before", {})
        self.assertFalse(observed["ready"])
        self.assertNotIn("private-test-token", json.dumps(observed))


if __name__ == "__main__":
    unittest.main()
