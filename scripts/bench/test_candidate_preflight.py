"""Static candidate admission must reject drift before a corpus lane starts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_preflight as preflight


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
    ).strip()


def commit(root: Path) -> str:
    subprocess.run(["git", "-C", str(root), "add", "."], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    return git(root, "rev-parse", "HEAD")


class CandidatePreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root = base / "repo"
        self.root.mkdir()
        self.sample = base / "sample"
        self.sample.mkdir()
        for path in (self.root, self.sample):
            subprocess.run(["git", "init", "-q", str(path)], check=True)
            git(path, "config", "user.name", "Candidate Fixture")
            git(path, "config", "user.email", "candidate@example.invalid")
        (self.sample / "README.md").write_text("sample\n")
        self.sample_head = commit(self.sample)
        self.docs = base / "fixtures-docs"
        self.docs.mkdir()
        (self.docs / "page.md").write_text("fixture\n")
        subprocess.run(["git", "init", "-q", str(self.docs)], check=True)
        subprocess.run(["git", "-C", str(self.docs), "-c", "user.name=Candidate Fixture",
                        "-c", "user.email=candidate@example.invalid", "commit",
                        "--allow-empty", "-qm", "fixture seed"], check=True)
        self.docs_head = git(self.docs, "rev-parse", "HEAD")

        for folder in ("bench/profiles", "bench/goldens", "src", "scripts/bench", "notes/bench", "notes/runs", "target"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        (self.root / "src/main.rs").write_text("fn main() {}\n")
        (self.root / "bench/profiles/sample.toml").write_text("# sample\n")
        (self.root / "bench/goldens/sample.yaml").write_text("queries: []\n")
        (self.root / "bench/corpus.toml").write_text(
            '[[repo]]\nname = "sample"\nurl = "local"\n'
            f'path = "{self.sample}"\nrev = "{self.sample_head}"\n'
            'config = "profiles/sample.toml"\n'
            'golden = "goldens/sample.yaml"\n'
            '[[repo]]\nname = "fixtures-docs"\nurl = "local"\n'
            f'path = "{self.docs}"\nrev = ""\n'
        )
        (self.root / preflight.AUDITOR_PATH).write_text("# frozen fixture auditor\n")
        self.revision = commit(self.root)
        self.source_hash = "a" * 64
        self.builds = {}
        for kind, features in (("cpu", "DEFAULT"), ("cuda", "CUDA,DEFAULT")):
            build = {
                "commit": self.revision, "source_blake3": self.source_hash,
                "source_dirty": "false", "profile": "release", "features": features,
            }
            binary = self.root / "target" / f"dkc-{kind}"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                f"print({json.dumps(json.dumps({'build': build}))})\n"
            )
            binary.chmod(0o755)
            self.builds[kind] = {
                "path": str(binary.relative_to(self.root)),
                "sha256": preflight.sha256(binary), "build": build,
            }
        plan = self.root / "notes/bench/lane-plan.json"
        plan.write_text(json.dumps({"lanes": [
            {"target": "sample", "lane": "local_cpu"},
            {"target": "fixtures-docs", "lane": "local_cpu"},
        ]}))
        baseline = self.root / "notes/bench/matrix-baseline.json"
        baseline.write_text(json.dumps({
            "audit_valid": True, "target_count": 2, "required_lane_count": 2,
            "targets": [
                {"name": name, "lanes": [{"profile": "local_cpu"}]}
                for name in ("sample", "fixtures-docs")
            ],
        }))
        paths = ["bench/corpus.toml", "bench/profiles/sample.toml",
                 "bench/goldens/sample.yaml", "notes/bench/lane-plan.json",
                 "notes/bench/matrix-baseline.json", preflight.AUDITOR_PATH]
        self.model_usage = {
            "configured_profile": "local",
            "configured_embedding": {"local": "fixture-embedder", "remote": "remote-embedder",
                                     "ollama": "ollama-embedder"},
            "configured_reranker": {"local": "fixture-reranker", "remote": "remote-reranker"},
            "docs_tier_configured": False,
            "verified_models": {
                "embedder": {"model": "fixture-embedder", "profile": "local"},
                "rerank": {"model": "fixture-reranker", "selected_provider": "local"},
                "docs": None,
                "llm": {"enabled": False, "model": None, "vlm_model": None},
                "source": "configured",
            },
        }
        identities = []
        for role, model, digest in (("embedder", "fixture-embedder", "b" * 64),
                                    ("reranker", "fixture-reranker", "c" * 64)):
            evidence_name = f"notes/bench/{role}-model-evidence.json"
            identity = {"role": role, "model": model, "provider": "local",
                        "kind": "sha256", "identity": digest,
                        "evidence_input": evidence_name}
            (self.root / evidence_name).write_text(json.dumps({
                "schema": preflight.MODEL_EVIDENCE_SCHEMA,
                **{field: identity[field] for field in
                   ("role", "model", "provider", "kind", "identity")},
                "origin": "fixture model verification",
            }))
            paths.append(evidence_name)
            identities.append(identity)
        self.receipts = {}
        for name in ("sample", "fixtures-docs"):
            config_name = f"notes/bench/{name}-config.json"
            model_name = f"notes/bench/{name}-models.json"
            config = {"schema": preflight.CONFIG_RECEIPT_SCHEMA, "target": name,
                      "profile": "local_cpu", "effective_config_blake3": "d" * 64,
                      "config_files_blake3": {"config.toml": "e" * 64,
                                              "config.local.toml": "f" * 64},
                      "starting_config_sha256": {},
                      "capture_binary_sha256": self.builds["cpu"]["sha256"],
                      "capture_build": self.builds["cpu"]["build"]}
            models = {"schema": preflight.MODEL_RECEIPT_SCHEMA, "target": name,
                      "profile": "local_cpu", "model_usage": self.model_usage,
                      "identities": identities}
            (self.root / config_name).write_text(json.dumps(config))
            (self.root / model_name).write_text(json.dumps(models))
            paths.extend((config_name, model_name))
            self.receipts[name] = (config_name, model_name)
        self.document = {
            "schema": preflight.SCHEMA,
            "product": {"commit": self.revision, "source_blake3": self.source_hash},
            "host": {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()},
            "corpus_manifest": {"path": paths[0], "sha256": preflight.sha256(self.root / paths[0])},
            "lane_plan": {"path": paths[3], "sha256": preflight.sha256(plan)},
            "matrix_baseline": {"path": paths[4], "sha256": preflight.sha256(baseline)},
            "audit_tool": {"path": preflight.AUDITOR_PATH,
                           "sha256": preflight.sha256(self.root / preflight.AUDITOR_PATH)},
            "inputs": [{"path": name, "sha256": preflight.sha256(self.root / name)}
                       for name in paths],
            "targets": [
                {"name": "sample", "root": str(self.sample), "head": self.sample_head,
                 "config_baseline": {}},
                {"name": "fixtures-docs", "root": str(self.docs), "head": self.docs_head,
                 "tree_sha256": preflight.tree_sha256(self.docs), "config_baseline": {}},
            ],
            "binaries": self.builds,
            "lanes": [
                {"target": name, "profile": "local_cpu", "binary": "cpu",
                 "state": "pending",
                 "effective_config_input": self.receipts[name][0],
                 "model_identity_input": self.receipts[name][1],
                 "output": f"notes/bench/new-{name}.json"}
                for name in ("sample", "fixtures-docs")
            ],
        }

    def report_provenance(self, target: str) -> dict:
        config = json.loads((self.root / self.receipts[target][0]).read_text())
        head = self.sample_head if target == "sample" else self.docs_head
        return {"head": head,
                "effective_config_blake3": config["effective_config_blake3"],
                "config_files_blake3": config["config_files_blake3"],
                "model_usage": self.model_usage}

    def rehash_input(self, name: str) -> None:
        next(row for row in self.document["inputs"] if row["path"] == name)["sha256"] = \
            preflight.sha256(self.root / name)

    def complete_sample(self) -> Path:
        lane = self.document["lanes"][0]
        path = self.root / lane["output"]
        path.write_text(json.dumps({
            "profile": "cpu", "build": self.builds["cpu"]["build"],
            "failures": [], "targets": [{"name": "sample",
                                         "provenance": self.report_provenance("sample")}],
        }))
        audit_name = "notes/bench/sample-audit.json"
        audit_path = self.root / audit_name
        audit_path.write_text(json.dumps({
            "accepted": True, "build": self.builds["cpu"]["build"],
            "targets": [{"name": "sample", "accepted": True}],
        }))
        self.document["inputs"].append({"path": audit_name,
                                        "sha256": preflight.sha256(audit_path)})
        models = json.loads((self.root / lane["model_identity_input"]).read_text())
        observations = {}
        for phase in ("before", "after"):
            name = f"notes/bench/new-sample-identity-{phase}.json"
            evidence = {
                "schema": preflight.LIVE_MODEL_SCHEMA,
                "candidate_sha256": "a" * 64,
                "target": "sample", "profile": "local_cpu", "phase": phase,
                "ready": True,
                "identities": [
                    {**{field: row[field] for field in
                        ("role", "model", "provider", "kind", "identity")},
                     "matched": True}
                    for row in models["identities"]
                ],
            }
            (self.root / name).write_text(json.dumps(evidence))
            digest = preflight.sha256(self.root / name)
            self.document["inputs"].append({"path": name, "sha256": digest})
            observations[phase] = {"path": name, "sha256": digest}
        lane.update(state="complete", report_sha256=preflight.sha256(path),
                    audit_input=audit_name, candidate_sha256="a" * 64,
                    identity_observations=observations)
        return path

    def test_completed_lane_rejects_changed_live_identity_even_with_rehashed_input(self) -> None:
        self.complete_sample()
        self.assertEqual(self.check(), [])
        name = "notes/bench/new-sample-identity-after.json"
        path = self.root / name
        observed = json.loads(path.read_text())
        observed["identities"][0]["identity"] = "f" * 64
        path.write_text(json.dumps(observed))
        self.rehash_input(name)
        self.document["lanes"][0]["identity_observations"]["after"]["sha256"] = \
            preflight.sha256(path)
        self.assertTrue(any("live model after role differs" in issue
                            for issue in self.check()))

    def test_completed_lane_needs_both_live_observations(self) -> None:
        self.complete_sample()
        self.document["lanes"][0]["identity_observations"].pop("after")
        self.assertTrue(any("live model observations are incomplete" in issue
                            for issue in self.check()))

    def check(self) -> list[str]:
        return preflight.check(
            self.root, self.document, expected_targets=2, expected_lanes=2,
            expected_lane_plan="notes/bench/lane-plan.json",
        )

    def test_complete_clean_candidate_passes_without_output(self) -> None:
        self.assertEqual(self.check(), [])
        self.assertFalse((self.root / "notes/bench/new-sample.json").exists())

    def test_capture_binary_and_starting_config_must_match_candidate(self) -> None:
        name = self.receipts["sample"][0]
        receipt_path = self.root / name
        receipt = json.loads(receipt_path.read_text())
        receipt["capture_binary_sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt))
        self.rehash_input(name)
        self.assertTrue(any("capture binary differs" in issue for issue in self.check()))
        receipt["capture_binary_sha256"] = self.builds["cpu"]["sha256"]
        receipt["starting_config_sha256"] = {"config.local.toml": "1" * 64}
        receipt_path.write_text(json.dumps(receipt))
        self.rehash_input(name)
        self.assertTrue(any("starting configuration differs" in issue for issue in self.check()))

    def test_readme_commit_does_not_invalidate_unchanged_binary_source(self) -> None:
        (self.root / "README.md").write_text("Updated release explanation.\n")
        commit(self.root)
        self.assertEqual(self.check(), [])

    def test_release_defaults_refuse_a_small_cohort(self) -> None:
        issues = preflight.check(self.root, self.document)
        self.assertIn("corpus target count must be 26", issues)
        self.assertIn("required lane count must be 81", issues)

    def test_changed_lane_plan_cannot_preserve_only_the_count(self) -> None:
        baseline_path = self.root / "notes/bench/matrix-baseline.json"
        baseline = json.loads(baseline_path.read_text())
        baseline["targets"][0]["lanes"][0]["profile"] = "moonly"
        baseline_path.write_text(json.dumps(baseline))
        digest = preflight.sha256(baseline_path)
        self.document["matrix_baseline"]["sha256"] = digest
        next(item for item in self.document["inputs"]
             if item["path"] == "notes/bench/matrix-baseline.json")["sha256"] = digest
        self.assertIn("required lane plan differs from audited matrix baseline",
                      self.check())

    def test_binary_input_and_product_drift_refuse(self) -> None:
        binary = self.root / self.builds["cpu"]["path"]
        marker = self.root / "executed-changed-binary"
        original = binary.read_text()
        binary.write_text(original + f"open({str(marker)!r}, 'w').write('unsafe')\n")
        self.assertIn("cpu: binary bytes changed", self.check())
        self.assertFalse(marker.exists())
        binary.write_text(original)
        (self.root / "bench/profiles/sample.toml").write_text("# changed\n")
        issues = self.check()
        self.assertTrue(any("frozen input changed" in issue for issue in issues))
        self.assertTrue(any("product or benchmark tooling" in issue for issue in issues))

    def test_lane_auditor_is_frozen_with_candidate(self) -> None:
        (self.root / preflight.AUDITOR_PATH).write_text("# changed auditor\n")
        self.assertIn("lane auditor bytes changed", self.check())

    def test_checkout_and_fixture_drift_refuse(self) -> None:
        (self.sample / "README.md").write_text("moved\n")
        commit(self.sample)
        self.assertIn("sample: checkout HEAD moved", self.check())
        (self.docs / "page.md").write_text("changed\n")
        self.assertIn("fixtures-docs: generated fixture tree changed",
                      self.check())

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires raw Unix filename bytes")
    def test_hostile_fixture_filename_is_pinned_and_drift_refuses(self) -> None:
        raw_name = b"invalid-\xff.txt"
        raw_path = os.path.join(os.fsencode(self.docs), raw_name)
        with open(raw_path, "wb") as target:
            target.write(b"hostile fixture\n")
        expected = hashlib.sha256()
        expected.update(b"dkc.fixture-tree.v2\0")
        for name, content in ((raw_name, b"hostile fixture\n"),
                              (b"page.md", b"fixture\n")):
            expected.update(len(name).to_bytes(8, "big"))
            expected.update(name)
            expected.update(len(content).to_bytes(8, "big"))
            expected.update(content)
        self.assertEqual(preflight.tree_sha256(self.docs), expected.hexdigest())
        self.document["targets"][1]["tree_sha256"] = expected.hexdigest()
        self.assertEqual(self.check(), [])
        with open(raw_path, "wb") as target:
            target.write(b"changed fixture\n")
        self.assertIn("fixtures-docs: generated fixture tree changed", self.check())

    def test_fixture_hash_frames_files_to_refuse_tree_stream_collision(self) -> None:
        first = self.root / "first-tree"
        second = self.root / "second-tree"
        first.mkdir()
        second.mkdir()
        (first / "x").write_bytes(b"ay\0b")
        (second / "x").write_bytes(b"a")
        (second / "y").write_bytes(b"b")
        # The old path-NUL-content stream was identical for these trees:
        # x\0ay\0b. The candidate pin must distinguish their file boundaries.
        self.assertNotEqual(preflight.tree_sha256(first), preflight.tree_sha256(second))

    def test_generated_fixture_needs_pinned_git_head_for_benchmark(self) -> None:
        self.document["targets"][1]["head"] = "0" * 40
        self.assertIn("fixtures-docs: generated fixture Git HEAD changed", self.check())
        self.document["targets"][1]["head"] = self.docs_head
        subprocess.run(["git", "-C", str(self.docs), "add", "page.md"], check=True)
        self.assertIn("fixtures-docs: generated fixture Git seed is not empty and solitary",
                      self.check())
        subprocess.run(["git", "-C", str(self.docs), "reset", "-q"], check=True)
        subprocess.run(["git", "-C", str(self.docs), "-c", "user.name=Candidate Fixture",
                        "-c", "user.email=candidate@example.invalid", "commit",
                        "--allow-empty", "-qm", "another seed"], check=True)
        self.document["targets"][1]["head"] = git(self.docs, "rev-parse", "HEAD")
        self.assertIn("fixtures-docs: generated fixture Git seed is not empty and solitary",
                      self.check())
        shutil.rmtree(self.docs / ".git")
        self.assertIn("fixtures-docs: generated fixture Git HEAD unavailable", self.check())

    def test_missing_lane_duplicate_output_and_existing_output_refuse(self) -> None:
        self.document["lanes"].pop()
        self.assertIn("candidate lanes differ from required lane plan",
                      self.check())
        self.document["lanes"].append({
            "target": "fixtures-docs", "profile": "local_cpu", "binary": "cpu",
            "state": "pending",
            "effective_config_input": self.receipts["fixtures-docs"][0],
            "model_identity_input": self.receipts["fixtures-docs"][1],
            "output": "notes/bench/new-sample.json",
        })
        self.assertTrue(any("duplicate lane output" in issue
                            for issue in self.check()))
        (self.root / "notes/bench/new-sample.json").write_text("old result\n")
        self.assertTrue(any("already exists" in issue
                            for issue in self.check()))

    def test_lane_report_cannot_occupy_another_lane_audit_path(self) -> None:
        self.document["lanes"][1]["output"] = "notes/bench/new-sample-audit.json"
        self.assertIn("lane report output collides with another lane audit",
                      self.check())

    def test_wrong_backend_and_build_metadata_refuse(self) -> None:
        self.document["lanes"][0]["binary"] = "cuda"
        self.assertTrue(any("wrong binary for profile" in issue
                            for issue in self.check()))
        self.document["lanes"][0]["binary"] = "cpu"
        self.document["binaries"]["cuda"]["build"]["commit"] = "0" * 40
        self.assertTrue(any("cuda: build metadata changed" in issue
                            for issue in self.check()))

    def test_completed_lane_can_resume_only_with_matching_report_and_audit(self) -> None:
        lane = self.document["lanes"][0]
        report_path = self.complete_sample()
        audit_path = self.root / lane["audit_input"]
        self.assertEqual(self.check(), [])
        report_path.write_text(report_path.read_text() + "\n")
        self.assertTrue(any("completed lane report bytes changed" in issue
                            for issue in self.check()))
        report_path.write_text(report_path.read_text().rstrip())
        lane["report_sha256"] = preflight.sha256(report_path)
        audit_path.write_text(json.dumps({
            "accepted": False, "build": self.builds["cpu"]["build"],
            "targets": [{"name": "sample", "accepted": False}],
        }))
        self.document["inputs"][-1]["sha256"] = preflight.sha256(audit_path)
        self.assertTrue(any("audit does not accept target" in issue
                            for issue in self.check()))

    def test_receipt_identity_and_empty_content_refuse_even_when_rehashed(self) -> None:
        config_name, model_name = self.receipts["sample"]
        config_path = self.root / config_name
        config = json.loads(config_path.read_text())
        config["effective_config_blake3"] = "placeholder"
        config_path.write_text(json.dumps(config))
        self.rehash_input(config_name)
        self.assertTrue(any("effective-config receipt is incomplete" in issue
                            for issue in self.check()))
        config_path.write_text(json.dumps({**config, "effective_config_blake3": "d" * 64}))
        self.rehash_input(config_name)
        self.document["lanes"][0]["model_identity_input"] = \
            self.receipts["fixtures-docs"][1]
        self.assertTrue(any("model_identity_input identity differs" in issue
                            for issue in self.check()))
        self.document["lanes"][0]["model_identity_input"] = model_name
        model_path = self.root / model_name
        model = json.loads(model_path.read_text())
        model["identities"] = []
        model_path.write_text(json.dumps(model))
        self.rehash_input(model_name)
        self.assertTrue(any("model evidence roles differ" in issue
                            for issue in self.check()))

    def test_completed_report_must_match_config_and_model_receipts(self) -> None:
        path = self.complete_sample()
        self.assertEqual(self.check(), [])
        report = json.loads(path.read_text())
        report["targets"][0]["provenance"]["effective_config_blake3"] = "0" * 64
        path.write_text(json.dumps(report))
        self.document["lanes"][0]["report_sha256"] = preflight.sha256(path)
        self.assertTrue(any("effective configuration differs" in issue
                            for issue in self.check()))
        report["targets"][0]["provenance"] = self.report_provenance("sample")
        report["targets"][0]["provenance"]["config_files_blake3"]["config.toml"] = "1" * 64
        path.write_text(json.dumps(report))
        self.document["lanes"][0]["report_sha256"] = preflight.sha256(path)
        self.assertTrue(any("effective configuration differs" in issue
                            for issue in self.check()))
        report["targets"][0]["provenance"] = self.report_provenance("sample")
        report["targets"][0]["provenance"]["model_usage"]["configured_profile"] = "remote"
        path.write_text(json.dumps(report))
        self.document["lanes"][0]["report_sha256"] = preflight.sha256(path)
        self.assertTrue(any("model usage differs" in issue for issue in self.check()))

    def test_remote_deployment_needs_frozen_sourced_evidence(self) -> None:
        model_name = self.receipts["sample"][1]
        model_path = self.root / model_name
        model = json.loads(model_path.read_text())
        model["model_usage"]["verified_models"]["embedder"]["profile"] = "remote"
        identity = model["identities"][0]
        identity["evidence_input"] = "notes/bench/sample-remote-embedder-evidence.json"
        identity.update(provider="remote", kind="deployment",
                        identity="moon-gateway-deployment-1")
        evidence_path = self.root / identity["evidence_input"]
        evidence = {"schema": preflight.MODEL_EVIDENCE_SCHEMA,
                    **{field: identity[field] for field in
                       ("role", "model", "provider", "kind", "identity")},
                    "origin": "operator deployment inventory"}
        evidence_path.write_text(json.dumps(evidence))
        self.document["inputs"].append({"path": identity["evidence_input"],
                                        "sha256": preflight.sha256(evidence_path)})
        model_path.write_text(json.dumps(model))
        self.rehash_input(model_name)
        self.rehash_input(identity["evidence_input"])
        self.assertEqual(self.check(), [])
        evidence["identity"] = "different-deployment"
        evidence_path.write_text(json.dumps(evidence))
        self.rehash_input(identity["evidence_input"])
        self.assertTrue(any("embedder model evidence differs" in issue
                            for issue in self.check()))
        evidence["identity"] = identity["identity"]
        evidence_path.write_text(json.dumps(evidence))
        self.rehash_input(identity["evidence_input"])
        identity["identity"] = ""
        model_path.write_text(json.dumps(model))
        self.rehash_input(model_name)
        self.assertTrue(any("embedder model evidence is not frozen" in issue
                            for issue in self.check()))

    def test_malformed_receipt_fields_refuse_without_crashing(self) -> None:
        model_name = self.receipts["sample"][1]
        path = self.root / model_name
        model = json.loads(path.read_text())
        model["model_usage"]["configured_profile"] = []
        model["identities"][0]["provider"] = []
        path.write_text(json.dumps(model))
        self.rehash_input(model_name)
        issues = self.check()
        self.assertTrue(any("model-identity receipt is incomplete" in issue
                            for issue in issues))
        model["model_usage"]["configured_profile"] = "local"
        path.write_text(json.dumps(model))
        self.rehash_input(model_name)
        self.assertTrue(any("embedder model evidence identity differs" in issue
                            for issue in self.check()))

    def test_skipped_lane_retains_failure_without_becoming_complete(self) -> None:
        lane = self.document["lanes"][0]
        report = self.root / lane["output"]
        report.write_text('{"failures":["example"]}\n')
        log_name = "notes/runs/sample-failed.log"
        evidence_name = "notes/bench/sample-diagnosis.md"
        skip_name = "notes/bench/new-sample-skip.json"
        (self.root / log_name).write_text("benchmark failed\n")
        (self.root / evidence_name).write_text("Target-specific failure evidence.\n")
        disposition = {
            "schema": "dkc.phase4.matrix-lane-skip.v1",
            "candidate_sha256": "a" * 64,
            "target": "sample", "profile": "local_cpu",
            "output": lane["output"], "reason": "Target-specific failure under review",
            "report_sha256": preflight.sha256(report),
            "evidence": {"path": evidence_name,
                         "sha256": preflight.sha256(self.root / evidence_name)},
            "attempt_log": {"path": log_name,
                            "sha256": preflight.sha256(self.root / log_name)},
        }
        (self.root / skip_name).write_text(json.dumps(disposition))
        for name in (log_name, evidence_name, skip_name):
            self.document["inputs"].append({"path": name,
                                            "sha256": preflight.sha256(self.root / name)})
        lane.update(state="skipped", report_sha256=preflight.sha256(report),
                    skip_input=skip_name, candidate_sha256="a" * 64)
        self.assertEqual(self.check(), [])
        self.assertEqual(lane["state"], "skipped")
        report.write_text(report.read_text() + "changed\n")
        self.assertIn("skipped lane report bytes changed: ('sample', 'local_cpu')",
                      self.check())

    def test_output_escape_and_missing_model_identity_refuse(self) -> None:
        self.document["lanes"][0]["output"] = "notes/bench/../../../outside.json"
        self.document["lanes"][0]["model_identity_input"] = "notes/bench/absent.json"
        issues = self.check()
        self.assertTrue(any("lane output invalid" in issue for issue in issues))
        self.assertTrue(any("lacks frozen model_identity_input" in issue
                            for issue in issues))

    def test_stop_wrong_boot_and_unpinned_aurora_refuse(self) -> None:
        (self.root / "notes/STOP").touch()
        self.assertTrue(any("notes/STOP" in issue
                            for issue in self.check()))
        (self.root / "notes/STOP").unlink()
        self.document["host"]["boot_id"] = "wrong"
        self.assertTrue(any("host boot differs" in issue
                            for issue in self.check()))
        self.document["host"]["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        corpus = self.root / "bench/corpus.toml"
        corpus.write_text(corpus.read_text().replace('name = "sample"', 'name = "aurora"')
                          .replace(f'rev = "{self.sample_head}"', 'rev = ""'))
        self.document["corpus_manifest"]["sha256"] = preflight.sha256(corpus)
        self.document["inputs"][0]["sha256"] = preflight.sha256(corpus)
        self.document["targets"][0]["name"] = "aurora"
        self.assertTrue(any("aurora: corpus revision is not pinned" in issue
                            for issue in self.check()))

    def test_cli_refuses_malformed_manifest_with_machine_readable_result(self) -> None:
        path = self.root / "notes/bench/malformed.json"
        path.write_text("[]\n")
        result = subprocess.run(
            [sys.executable, str(Path(preflight.__file__)), str(path),
             "--repo-root", str(self.root)],
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(json.loads(result.stdout)["ready"])
        self.assertFalse((self.root / "notes/bench/new-sample.json").exists())


if __name__ == "__main__":
    unittest.main()
