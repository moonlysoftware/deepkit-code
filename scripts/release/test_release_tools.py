#!/usr/bin/env python3
"""Offline controls for release archive validation."""

import importlib.util
import io
import pathlib
import re
import tarfile
import tempfile
import tomllib
import unittest
import zipfile


PATH = pathlib.Path(__file__).with_name("check-artifact-size.py")
SPEC = importlib.util.spec_from_file_location("check_artifact_size", PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class ArtifactSizeTests(unittest.TestCase):
    def test_native_qualification_skips_plan_prs_but_blocks_release_publication(self):
        root = PATH.parents[2]
        config = tomllib.loads((root / "dist-workspace.toml").read_text())["dist"]
        self.assertEqual(config["pr-run-mode"], "plan")
        self.assertIn("./release-platforms", config["local-artifacts-jobs"])
        for phase in ("plan-jobs", "host-jobs"):
            self.assertNotIn("./release-platforms", config.get(phase, []))

        generated = (root / ".github/workflows/release.yml").read_text()

        def job(name):
            match = re.search(
                rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z][\w-]*:\n|\Z)",
                generated, re.MULTILINE | re.DOTALL,
            )
            self.assertIsNotNone(match, name)
            return match.group(1)

        native = job("custom-release-platforms")
        condition = re.search(r"^    if: (.+)$", native, re.MULTILINE)
        self.assertIsNotNone(condition)
        self.assertEqual(
            condition.group(1),
            "${{ needs.plan.outputs.publishing == 'true' || "
            "fromJson(needs.plan.outputs.val).ci.github.pr_run_mode == 'upload' }}",
        )
        self.assertIn("plan: ${{ needs.plan.outputs.val }}", native)
        for name in ("build-global-artifacts", "custom-release-sizes",
                     "custom-release-acceptance", "host"):
            self.assertIn("- custom-release-platforms", job(name))
        self.assertIn(
            "(needs.custom-release-platforms.result == 'skipped' || "
            "needs.custom-release-platforms.result == 'success')",
            job("host"),
        )
        self.assertIn("gh release create", job("host"))
        reusable = (root / ".github/workflows/release-platforms.yml").read_text()
        self.assertIn("plan:\n        required: true\n        type: string", reusable)
        self.assertNotIn("continue-on-error", reusable)

    def test_github_release_and_registry_publication_require_phase_four_readiness(self):
        root = PATH.parents[2]
        generated = (root / ".github/workflows/release.yml").read_text()
        for name in ("host", "publish-homebrew-formula", "publish-npm"):
            match = re.search(
                rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z][\w-]*:\n|\Z)",
                generated,
                re.MULTILINE | re.DOTALL,
            )
            self.assertIsNotNone(match, name)
            dependency = (
                "- custom-release-acceptance" if name == "host" else "- host"
            )
            self.assertIn(dependency, match.group(1))
            if name == "host":
                self.assertIn("needs.custom-release-acceptance.result", match.group(1))
        config = (root / "dist-workspace.toml").read_text()
        self.assertIn('global-artifacts-jobs = ["./release-sizes", "./release-acceptance"]', config)
        guard = (root / ".github/workflows/release-acceptance.yml").read_text()
        self.assertIn("--require-ready", guard)
        self.assertIn("fetch-depth: 0", guard)
        self.assertIn("!fromJson(inputs.plan).announcement_is_prerelease", guard)

    def test_smoke_script_explicitly_stays_at_model_free_l1(self):
        script = PATH.with_name("smoke.sh").read_text()
        powershell = PATH.with_name("smoke.ps1").read_text()
        self.assertIn('index --level 1 --json', script)
        self.assertNotIn('"$binary" index --json', script)
        self.assertIn('@("index", "--level", "1", "--json")', powershell)

    def test_mac_verifier_records_the_accelerated_embedding_request(self):
        script = PATH.with_name("mac_verify.py").read_text()
        self.assertIn('embedding["metal_feature"] is True', script)
        self.assertIn('embedding["requested_gpu_layers"] == -1', script)

    def test_reads_linux_and_windows_binary_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            tar_path = root / "deepkit-code-linux.tar.gz"
            data = b"linux"
            with tarfile.open(tar_path, "w:gz") as archive:
                member = tarfile.TarInfo("deepkit-code-linux/dkc")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            zip_path = root / "deepkit-code-windows.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("dkc.exe", b"windows")
            self.assertEqual(list(CHECKER.binary_members(tar_path)), [("deepkit-code-linux/dkc", 5)])
            self.assertEqual(list(CHECKER.binary_members(zip_path)), [("dkc.exe", 7)])

    def test_ignores_unrelated_archive_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "deepkit-code-source.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("README.md", b"not a binary")
            self.assertEqual(list(CHECKER.binary_members(path)), [])

    def test_audit_gates_download_bytes_and_records_installed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = "x86_64-unknown-linux-musl"
            path = root / f"deepkit-code-{target}.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo(f"deepkit-code-{target}/dkc")
                payload = b"linux" * 10_000
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            archive_bytes = path.stat().st_size
            self.assertGreater(len(payload), archive_bytes)
            reports, errors = CHECKER.audit(root, archive_bytes, (target,))
            self.assertEqual(errors, [])
            self.assertEqual(reports[0]["target"], target)
            self.assertEqual(reports[0]["bytes"], len(payload))
            self.assertEqual(reports[0]["archive_bytes"], archive_bytes)
            self.assertEqual(reports[0]["size_scope"], "download_archive")
            self.assertTrue(reports[0]["passed"])

            reports, errors = CHECKER.audit(root, archive_bytes - 1,
                                            (target, "aarch64-apple-darwin"))
            self.assertFalse(reports[0]["passed"])
            self.assertIn("one or more release download archives exceed the size limit", errors)
            self.assertIn("aarch64-apple-darwin: expected one dkc binary, found 0", errors)

    def test_archive_limit_includes_other_members_and_duplicate_binaries_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = "x86_64-pc-windows-msvc"
            path = root / f"deepkit-code-{target}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("dkc.exe", b"exe")
                archive.writestr("LICENSE", b"license" * 100)
            reports, errors = CHECKER.audit(root, 500, (target,))
            self.assertEqual(reports[0]["bytes"], 3)
            self.assertFalse(reports[0]["passed"])
            self.assertIn("one or more release download archives exceed the size limit", errors)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("duplicate/dkc.exe", b"exe")
            _, errors = CHECKER.audit(root, path.stat().st_size, (target,))
            self.assertIn(f"{target}: expected one dkc binary, found 2", errors)

    def test_dist_plan_rejects_stale_archive_with_matching_target(self):
        target = "x86_64-unknown-linux-musl"
        planned_name = f"deepkit-code-1.0.0-{target}.tar.gz"
        plan = {"artifacts": {planned_name: {
            "kind": "executable-zip", "target_triples": [target],
        }}}
        expected = CHECKER.planned_archives(plan, (target,))
        self.assertEqual(expected, {target: planned_name})
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "deepkit-code-npm-package.tar.gz").write_bytes(b"installer")
            stale = root / f"deepkit-code-0.9.0-{target}.tar.gz"
            with tarfile.open(stale, "w:gz") as archive:
                member = tarfile.TarInfo(f"deepkit-code-{target}/dkc")
                member.size = 5
                archive.addfile(member, io.BytesIO(b"linux"))
            reports, errors = CHECKER.audit(root, 1_000_000, (target,), expected)
            self.assertEqual(reports, [])
            self.assertTrue(any("unplanned release archive" in error for error in errors))
            self.assertTrue(any("expected one dkc binary, found 0" in error for error in errors))
            stale.rename(root / planned_name)
            reports, errors = CHECKER.audit(root, 1_000_000, (target,), expected)
            self.assertEqual(errors, [])
            self.assertEqual(len(reports), 1)


if __name__ == "__main__":
    unittest.main()
