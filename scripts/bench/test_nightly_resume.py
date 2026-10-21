"""Offline identity, interruption and evidence-preservation checks for nightlies."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import nightly
import nightly_resume as resume


def target_report(build, *, failed=False):
    """Return the minimum real-schema fields the nightly validates."""
    return {
        "profile": "local",
        "build": build,
        "targets": [] if failed else [{
            "name": "tiny", "provenance": {"pre_index": {
                "load_average": [0.0], "idle": True,
            }},
        }],
        "failures": [{"name": "tiny"}] if failed else [],
    }


class NightlyResumeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "nightly-test-local.json"
        self.state_path = self.root / "nightly-active.state.json"
        self.recipe = nightly.Recipe("local", ("tiny",), "cuda", True)
        self.commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=nightly.ROOT, text=True
        ).strip()
        self.build = {
            "binary_blake3": "frozen-cuda", "source_dirty": "false",
            "commit": self.commit, "profile": "release", "features": "CUDA,DEFAULT",
        }
        self.state = {"targets": {}, "inputs": {"model_receipt": None}}
        self.runner = nightly.Runner(self.root / "run.log")

    def frozen_binaries(self):
        """Make distinct byte and metadata identities under the allowed target root."""
        base = nightly.ROOT / "target/nightly" / ("resume-test-" + self.root.name)
        cpu = base / "cpu/dkc"
        cuda = base / "cuda/dkc"
        cpu.parent.mkdir(parents=True)
        cuda.parent.mkdir(parents=True)
        cpu.write_bytes(b"frozen-cpu-test-binary")
        cuda.write_bytes(b"frozen-cuda-test-binary")
        self.addCleanup(lambda: shutil.rmtree(base, ignore_errors=True))
        builds = {
            "cpu": {**self.build, "features": "DEFAULT", "binary_blake3": "frozen-cpu"},
            "cuda": self.build,
        }
        return {"cpu": cpu, "cuda": cuda}, builds

    def test_completed_exact_report_is_reused_without_running_or_rewriting(self):
        source = self.root / "prior.json"
        source.write_text(json.dumps(target_report(self.build)))
        before = source.read_bytes()
        self.state["targets"]["local:tiny"] = {
            "state": "complete", "attempt": 1, "path": str(source),
            "sha256": resume.sha256(source),
        }
        with patch.object(self.runner, "idle") as idle, patch.object(self.runner, "run") as run:
            combined = nightly.run_recipe(
                self.recipe, self.root / "dkc", self.root / "corpus.toml",
                self.out, self.runner, self.state, self.state_path, self.build,
            )
        idle.assert_not_called()
        run.assert_not_called()
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(combined["source_reports"], [str(source)])

    def test_changed_completed_report_is_refused_without_overwrite(self):
        source = self.root / "prior.json"
        source.write_text(json.dumps(target_report(self.build)))
        self.state["targets"]["local:tiny"] = {
            "state": "complete", "attempt": 1, "path": str(source),
            "sha256": "wrong",
        }
        with patch.object(self.runner, "run") as run:
            with self.assertRaisesRegex(ValueError, "report bytes changed"):
                nightly.run_recipe(
                    self.recipe, self.root / "dkc", self.root / "corpus.toml",
                    self.out, self.runner, self.state, self.state_path, self.build,
                )
        run.assert_not_called()

    def test_budget_defers_without_reserving_or_repeating_the_next_target(self):
        source = self.root / "prior.json"
        source.write_text(json.dumps(target_report(self.build)))
        self.state["targets"]["local:tiny"] = {
            "state": "complete", "attempt": 1, "path": str(source),
            "sha256": resume.sha256(source),
        }
        recipe = nightly.Recipe("local", ("tiny", "other"), "cuda", True)
        with patch.object(self.runner, "run") as run:
            with self.assertRaisesRegex(nightly.BudgetDeferred, "time budget exhausted"):
                nightly.run_recipe(
                    recipe, self.root / "dkc", self.root / "corpus.toml",
                    self.out, self.runner, self.state, self.state_path, self.build,
                    time.monotonic() - 1,
                )
        run.assert_not_called()
        self.assertNotIn("local:other", self.state["targets"])
        self.assertEqual(source.read_text(), json.dumps(target_report(self.build)))

    def test_budget_limited_child_timeout_becomes_explicit_deferral(self):
        with patch.object(self.runner, "run", side_effect=subprocess.TimeoutExpired("test", 60)):
            with self.assertRaisesRegex(nightly.BudgetDeferred, "budget ended"):
                nightly.run_with_budget(
                    self.runner, ["test"], time.monotonic() + 70, 10800,
                )

    def test_interrupted_attempt_is_preserved_and_next_attempt_completes(self):
        def interrupted(arguments, **kwargs):
            Path(arguments[arguments.index("--out") + 1]).write_text("partial")
            raise RuntimeError("interrupted")

        with patch.object(self.runner, "idle"), patch.object(self.runner, "run", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                nightly.run_recipe(
                    self.recipe, self.root / "dkc", self.root / "corpus.toml",
                    self.out, self.runner, self.state, self.state_path, self.build,
                )
        first = Path(self.state["targets"]["local:tiny"]["path"])
        self.assertEqual(first.read_text(), "partial")

        def completed(arguments, **kwargs):
            Path(arguments[arguments.index("--out") + 1]).write_text(
                json.dumps(target_report(self.build))
            )
            return 0

        with patch.object(self.runner, "idle"), patch.object(self.runner, "run", side_effect=completed):
            nightly.run_recipe(
                self.recipe, self.root / "dkc", self.root / "corpus.toml",
                self.out, self.runner, self.state, self.state_path, self.build,
            )
        self.assertEqual(first.read_text(), "partial")
        self.assertEqual(self.state["targets"]["local:tiny"]["attempt"], 2)
        self.assertEqual(self.state["targets"]["local:tiny"]["state"], "complete")

    def test_structured_failure_is_retained_and_not_repeated(self):
        def failed(arguments, **kwargs):
            Path(arguments[arguments.index("--out") + 1]).write_text(
                json.dumps(target_report(self.build, failed=True))
            )
            return 1

        with patch.object(self.runner, "idle"), patch.object(self.runner, "run", side_effect=failed):
            with self.assertRaisesRegex(ValueError, "contains failures"):
                nightly.run_recipe(
                    self.recipe, self.root / "dkc", self.root / "corpus.toml",
                    self.out, self.runner, self.state, self.state_path, self.build,
                )
        self.assertEqual(self.state["targets"]["local:tiny"]["state"], "failed")
        with patch.object(self.runner, "run") as run:
            with self.assertRaisesRegex(ValueError, "prior failed target"):
                nightly.run_recipe(
                    self.recipe, self.root / "dkc", self.root / "corpus.toml",
                    self.out, self.runner, self.state, self.state_path, self.build,
                )
        run.assert_not_called()

    def test_own_target_timeout_requires_diagnosis_before_retry(self):
        with patch.object(self.runner, "idle"), \
             patch.object(self.runner, "run",
                          side_effect=subprocess.TimeoutExpired("dkc bench run", 10800)):
            with self.assertRaises(subprocess.TimeoutExpired):
                nightly.run_recipe(
                    self.recipe, self.root / "dkc", self.root / "corpus.toml",
                    self.out, self.runner, self.state, self.state_path, self.build,
                    time.monotonic() + 20000,
                )
        self.assertEqual(self.state["targets"]["local:tiny"]["state"], "failed")

    def test_remote_report_without_deployment_receipt_is_remeasured(self):
        prior = self.root / "prior.json"
        old = target_report(self.build)
        old["profile"] = "moonly"
        prior.write_text(json.dumps(old))
        self.state["targets"]["moonly:tiny"] = {
            "state": "complete", "attempt": 1, "path": str(prior),
            "sha256": resume.sha256(prior),
        }
        recipe = nightly.Recipe("moonly", ("tiny",), "cpu", False)
        before = prior.read_bytes()

        def completed(arguments, **kwargs):
            Path(arguments[arguments.index("--out") + 1]).write_text(json.dumps(old))
            return 0

        with patch.object(self.runner, "idle"), patch.object(self.runner, "run",
                                                      side_effect=completed) as run:
            nightly.run_recipe(
                recipe, self.root / "dkc", self.root / "corpus.toml",
                self.out, self.runner, self.state, self.state_path, self.build,
            )
        self.assertEqual(run.call_count, 1)
        self.assertEqual(prior.read_bytes(), before)
        self.assertEqual(self.state["targets"]["moonly:tiny"]["attempt"], 2)

    def test_input_snapshot_refuses_changed_pin_and_hashes_local_inputs(self):
        repository = self.root / "source"
        repository.mkdir()
        for args in (("init",), ("config", "user.email", "test@example.invalid"),
                     ("config", "user.name", "Test")):
            subprocess.run(["git", *args], cwd=repository, check=True,
                           capture_output=True)
        (repository / "file.txt").write_text("first")
        subprocess.run(["git", "add", "file.txt"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "pin"], cwd=repository,
                       check=True, capture_output=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository,
                                       text=True).strip()
        profile = self.root / "profile.toml"
        profile.write_text("value = 1\n")
        corpus = self.root / "corpus.toml"
        corpus.write_text(f'[[repo]]\nname = "tiny"\npath = "{repository}"\n'
                          f'rev = "{head}"\nconfig = "profile.toml"\n')
        with patch.dict(os.environ, {"DKC_MODELS_DIR": str(self.root)}):
            first = resume.input_snapshot(corpus, None)
            profile.write_text("value = 2\n")
            second = resume.input_snapshot(corpus, None)
            self.assertNotEqual(first, second)
            model = self.root / "model.gguf"
            model.write_bytes(b"first")
            third = resume.input_snapshot(corpus, None)
            self.assertNotEqual(second, third)
            model.write_bytes(b"second")
            self.assertNotEqual(third, resume.input_snapshot(corpus, None))
            (repository / "file.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "tracked corpus checkout is dirty"):
                resume.input_snapshot(corpus, None)

    def test_second_invocation_skips_completed_target_and_keeps_old_attempt(self):
        directory = self.root / "evidence"
        logs = self.root / "logs"
        directory.mkdir()
        logs.mkdir()
        binaries, builds_by_kind = self.frozen_binaries()
        corpus = self.root / "corpus.toml"
        corpus.write_text("fixture")
        args = type("Args", (), {
            "corpus": corpus, "log_dir": logs, "out_dir": directory,
            "compare_only": None, "model_identity_receipt": None,
        })()
        recipe = nightly.Recipe("local", ("tiny", "other"), "cuda", False)
        build = self.build
        calls = []
        interrupted = True

        def fake_run(self, command, **kwargs):
            nonlocal interrupted
            if command == nightly.PULL:
                return 0
            target = command[command.index("--only") + 1]
            calls.append(target)
            out = Path(command[command.index("--out") + 1])
            if target == "other" and interrupted:
                out.write_text("partial")
                interrupted = False
                raise RuntimeError("test interruption")
            report = target_report(build)
            report["targets"][0]["name"] = target
            out.write_text(json.dumps(report))
            return 0

        with patch.object(nightly, "recipes", return_value=(recipe,)), \
             patch.object(nightly, "gpu_inventory", return_value=1), \
             patch.object(resume, "input_snapshot", return_value={"files": {}, "targets": {}, "model_receipt": None}), \
             patch.object(nightly, "build_binaries", return_value=binaries) as builds, \
             patch.object(nightly, "binary_metadata",
                          side_effect=lambda path: builds_by_kind["cpu" if path == binaries["cpu"] else "cuda"]), \
             patch.object(nightly.Runner, "idle"), \
             patch.object(nightly.Runner, "run", fake_run):
            self.assertEqual(nightly.execute(args), 1)
            state = resume.read_state(logs / "nightly-active.state.json")
            first = Path(state["targets"]["local:tiny"]["path"])
            partial = Path(state["targets"]["local:other"]["path"])
            first_bytes = first.read_bytes()
            self.assertEqual(partial.read_text(), "partial")
            self.assertEqual(nightly.execute(args), 0)
            self.assertEqual(builds.call_count, 1)
        self.assertEqual(calls, ["tiny", "other", "other"])
        self.assertEqual(first.read_bytes(), first_bytes)
        self.assertEqual(partial.read_text(), "partial")
        self.assertFalse((logs / "nightly-active.state.json").exists())
        self.assertEqual(len(list(logs.glob("nightly-*-complete.state.json"))), 1)

    def test_completed_profile_is_not_compared_to_its_own_latest_report(self):
        directory = self.root / "evidence"
        logs = self.root / "logs"
        directory.mkdir()
        logs.mkdir()
        binaries, builds_by_kind = self.frozen_binaries()
        corpus = self.root / "corpus.toml"
        corpus.write_text("fixture")
        args = type("Args", (), {
            "corpus": corpus, "log_dir": logs, "out_dir": directory,
            "compare_only": None, "model_identity_receipt": None,
        })()
        stamp = "frozen-test"
        local = directory / f"nightly-{stamp}-local.json"
        local.write_text(json.dumps(target_report(self.build)))
        latest = directory / "nightly-latest-local.json"
        latest.write_bytes(local.read_bytes())
        inputs = {"files": {}, "targets": {}, "model_receipt": None}
        frozen = {
            kind: {
                "path": str(binary), "sha256": resume.sha256(binary),
                "build": builds_by_kind[kind],
            }
            for kind, binary in binaries.items()
        }
        state = {
            "schema": resume.SCHEMA, "stamp": stamp,
            "commit": self.commit,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "inputs": inputs,
            "binaries": frozen,
            "targets": {},
            "profiles": {"local": {
                "state": "complete", "report": str(local), "sha256": resume.sha256(local),
            }},
        }
        resume.write_state(logs / "nightly-active.state.json", state)
        calls = []
        expected_build = builds_by_kind["cpu"]

        def fake_run(runner, command, **kwargs):
            if command == nightly.PULL:
                return 0
            target = command[command.index("--only") + 1]
            calls.append(target)
            report = target_report(expected_build)
            report["profile"] = "cpu"
            report["targets"][0]["name"] = target
            Path(command[command.index("--out") + 1]).write_text(json.dumps(report))
            return 0

        recipes = (
            nightly.Recipe("local", ("tiny",), "cuda", True),
            nightly.Recipe("cpu", ("other",), "cpu", False),
        )
        with patch.object(nightly, "recipes", return_value=recipes), \
             patch.object(nightly, "gpu_inventory", return_value=1), \
             patch.object(resume, "input_snapshot", return_value=inputs), \
             patch.object(nightly, "build_binaries") as builds, \
             patch.object(nightly, "binary_metadata",
                          side_effect=lambda path: builds_by_kind["cpu" if path == binaries["cpu"] else "cuda"]), \
             patch.object(nightly.Runner, "idle"), \
             patch.object(nightly.Runner, "run", fake_run):
            self.assertEqual(nightly.execute(args), 0)
        builds.assert_not_called()
        self.assertEqual(calls, ["other"])
        self.assertEqual(latest.read_bytes(), local.read_bytes())
        self.assertNotIn("local · **PASS**", (directory / "NIGHTLY.md").read_text())

    def test_restore_requires_same_inputs_and_binary_bytes(self):
        binaries, builds_by_kind = self.frozen_binaries()
        commit = self.commit
        inputs = {"corpus": "one"}
        records = {
            kind: {"path": str(binary), "sha256": resume.sha256(binary),
                   "build": builds_by_kind[kind]}
            for kind, binary in binaries.items()
        }
        state = {
            "commit": commit, "inputs": inputs,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "binaries": records,
        }
        with patch.object(nightly, "binary_metadata",
                          side_effect=lambda path: builds_by_kind["cpu" if path == binaries["cpu"] else "cuda"]):
            self.assertEqual(nightly.restore_binaries(state, commit, inputs),
                             binaries)
            self.assertIsNone(nightly.restore_binaries(state, commit, {"corpus": "two"}))
            binaries["cpu"].write_bytes(b"changed")
            self.assertIsNone(nightly.restore_binaries(state, commit, inputs))

    def test_completed_remote_profile_requires_deployment_receipt_on_resume(self):
        directory = self.root / "evidence"
        logs = self.root / "logs"
        directory.mkdir()
        logs.mkdir()
        corpus = self.root / "corpus.toml"
        corpus.write_text("fixture")
        args = type("Args", (), {
            "corpus": corpus, "log_dir": logs, "out_dir": directory,
            "compare_only": None, "model_identity_receipt": None,
        })()
        state = {
            "schema": resume.SCHEMA, "stamp": "frozen-test",
            "commit": self.commit,
            "inputs": {"model_receipt": None},
            "targets": {},
            "binaries": {"cpu": {"build": self.build}},
            "profiles": {"moonly": {"state": "complete", "sha256": "old"}},
        }
        with patch.object(nightly, "recipes", return_value=(
            nightly.Recipe("moonly", ("tiny",), "cpu", False),
        )), patch.object(nightly, "gpu_inventory", return_value=1), \
             patch.object(resume, "input_snapshot", return_value=state["inputs"]), \
             patch.object(resume, "read_state", return_value=state), \
             patch.object(nightly, "restore_binaries", return_value={"cpu": self.root / "dkc"}), \
             patch.object(nightly.Runner, "run", return_value=0) as run:
            self.assertEqual(nightly.execute(args), 1)
        self.assertEqual(run.call_count, 1)  # pull only
        self.assertIn("cannot resume without a deployment receipt",
                      (directory / "NIGHTLY.md").read_text())


if __name__ == "__main__":
    unittest.main()
