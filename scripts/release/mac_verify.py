#!/usr/bin/env python3
"""Collect native T4.10 evidence, retaining every failed independent measurement."""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_manifest(root):
    return {
        name: sha256(root / name)
        for name in git(root, "ls-files", "-z").split("\0")
        if name and not name.startswith("notes/") and (root / name).is_file()
    }


def diagnostic(text, label):
    for line in text.splitlines():
        if label in line:
            try:
                value = json.loads(line.split(label, 1)[1].strip())
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return None


def test_passed(text):
    return "test result: ok. 1 passed; 0 failed" in text


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class Run:
    """Persist stage exits immediately; a failed probe does not erase later evidence."""

    def __init__(self, root, out, environment):
        self.root, self.out, self.environment = root, out, environment
        self.prefix = out.with_suffix("")
        self.stages = []

    def path(self, name, suffix="log"):
        return Path(f"{self.prefix}.{name}.{suffix}")

    def run(self, name, command, cwd=None, timeout=1200, environment=None):
        log = self.path(name)
        started = time.monotonic()
        stage = {"name": name, "command": [str(arg) for arg in command],
                 "cwd": str(cwd or self.root), "started_at": now(), "log": str(log)}
        print(f"Metal verification: {name}", flush=True)
        error_log = self.path(name, "stderr.log") if "--json" in command else None
        with contextlib.ExitStack() as stack:
            stream = stack.enter_context(log.open("x"))
            errors = stack.enter_context(error_log.open("x")) if error_log else subprocess.STDOUT
            try:
                process = subprocess.Popen(command, cwd=cwd or self.root,
                                        env=environment or self.environment,
                                        stdout=stream, stderr=errors, start_new_session=True)
                try:
                    stage["exit_code"] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    stage.update(exit_code=124, error="stage timeout; process group terminated")
            except OSError as error:
                stage.update(exit_code=127, error=str(error))
        stage.update(wall_seconds=round(time.monotonic() - started, 3),
                     log_sha256=sha256(log), log_bytes=log.stat().st_size)
        if error_log:
            stage.update(stderr_log=str(error_log), stderr_sha256=sha256(error_log))
        self.stages.append(stage)
        self.path("stages", "json").write_text(json.dumps(self.stages, indent=2) + "\n")
        return stage["exit_code"] == 0

    def text(self, name):
        path = self.path(name)
        return path.read_text(errors="replace") if path.exists() else ""


def probe_command(name):
    # The crate is a binary; targeting it avoids relinking every integration test.
    return ["cargo", "test", "--release", "--locked", "--features", "metal",
            "--bin", "dkc", name, "--", "--exact", "--ignored", "--nocapture"]


def run_probes(run):
    """Run every independent component even when an earlier numerical gate misses."""
    run.run("embed-sanity", probe_command("models::llama::tests::embed_sanity"))
    environment = dict(run.environment, DKC_BENCH_DOCUMENT_COUNT="2000",
                       DKC_BENCH_DOCUMENT_REPEATS="1")
    run.run("throughput", probe_command("models::llama::tests::embed_throughput"),
            environment=environment)
    for index in range(1, 4):
        run.run(f"rerank-{index}", probe_command(
            "models::llama::rerank::tests::verified_metal_rank_fifty_candidate_p95"))
    run.run("idle-rss", probe_command(
        "models::llama::rerank::tests::verified_metal_idle_rss_after_both_models_unload"))


def measurements(run):
    text = run.text("throughput")
    match = re.search(r"embedding throughput: ([\d.]+) units/s \((\d+) documents in ([\d.]+)s\)", text)
    metal_match = re.search(r"embedding benchmark: .* metal=(true|false), requested_gpu_layers=(-?\d+)", text)
    embedding = None
    if match:
        embedding = {"units_per_second": float(match[1]), "documents": int(match[2]),
                     "elapsed_seconds": float(match[3]), "minimum_units_per_second": 200.0,
                     "metal_feature": bool(metal_match and metal_match[1] == "true"),
                     "requested_gpu_layers": int(metal_match[2]) if metal_match else None}
    sets = [diagnostic(run.text(f"rerank-{i}"), "Metal Rank component diagnostic:")
            for i in range(1, 4)]
    valid_sets = all(
        isinstance(item, dict) and item.get("candidates") == 50 and item.get("sample_count") == 20
        and isinstance(item.get("samples_ms"), list) and len(item["samples_ms"]) == 20
        and all(finite(sample) and sample >= 0 for sample in item["samples_ms"])
        and finite(item.get("p95_nearest_rank_ms"))
        and item["p95_nearest_rank_ms"] == sorted(item["samples_ms"])[18]
        and item.get("requested_gpu_layers") == -1
        for item in sets
    )
    median = statistics.median(item["p95_nearest_rank_ms"] for item in sets) if valid_sets else None
    memory = diagnostic(run.text("idle-rss"), "Metal idle RSS diagnostic:")

    def object_from_log(name):
        try:
            value = json.loads(run.text(name))
            return value if isinstance(value, dict) else None
        except ValueError:
            return None

    ripgrep = {key: object_from_log(f"ripgrep-{key}") for key in ("index", "search", "status")}
    search = ripgrep["search"] or {}
    search_index = search.get("index") if isinstance(search.get("index"), dict) else {}
    status = ripgrep["status"] or {}
    jobs = status.get("queue") if isinstance(status.get("queue"), dict) else {}
    l2 = status.get("l2") if isinstance(status.get("l2"), dict) else {}
    tables = l2.get("tables") if isinstance(l2.get("tables"), list) else []
    checks = {
        "embed_sanity": test_passed(run.text("embed-sanity")),
        "throughput_2000_units": bool(embedding and embedding["documents"] == 2000
            and finite(embedding["units_per_second"]) and embedding["units_per_second"] >= 200.0
            and embedding["metal_feature"] is True and embedding["requested_gpu_layers"] == -1
            and test_passed(text)),
        "rerank_50_candidates_p95": bool(valid_sets and median < 400.0),
        "idle_rss_after_unload": bool(memory and memory.get("embedding_unloaded") is True
            and memory.get("reranker_unloaded") is True
            and finite(memory.get("idle_rss_bytes")) and 0 < memory["idle_rss_bytes"] < 300_000_000
            and test_passed(run.text("idle-rss"))),
        "ripgrep_index": ripgrep["index"] is not None,
        "ripgrep_search": bool(search.get("hits")
            and search_index.get("partial") is False
            and search_index.get("degraded") is False),
        "ripgrep_queue": bool(jobs.get("dead") == 0 and jobs.get("failed") == 0
            and jobs.get("queued") == 0 and jobs.get("leased") == 0),
        "ripgrep_l2": bool(finite(l2.get("covered_units")) and l2["covered_units"] > 0
            and l2.get("pending_jobs") == 0
            and any(isinstance(table, dict)
                    and isinstance(table.get("configured"), dict)
                    and table["configured"].get("roles") == ["primary"]
                    and table.get("profile") == "local" for table in tables)),
    }
    return {"embedding": embedding,
            "rerank": {"sets": sets, "median_p95_ms": median, "maximum_p95_ms": 400.0},
            "memory": memory, "ripgrep": ripgrep}, checks


def verdict(checks, stages):
    return "passed" if checks and all(checks.values()) and stages and all(
        stage["exit_code"] == 0 for stage in stages) else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--ripgrep", type=Path, default=Path(os.environ.get(
        "DKC_RIPGREP_DIR", str(Path(os.environ.get("DKC_CORPUS_DIR", str(Path.home() / "dkc-corpus"))) / "ripgrep"))))
    parser.add_argument("--models", type=Path, default=Path(os.environ.get(
        "DKC_MODELS_DIR", str(Path.home() / ".cache/deepkit-code/models"))))
    parser.add_argument("--out", type=Path, default=root / "notes/runs" / f"mac-verify-{time.time_ns()}.json")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11+ is required for the pinned TOML manifests")
    import tomllib
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("mac-verify requires an Apple Silicon Mac")
    args.out = args.out.resolve()
    args.models, args.ripgrep = args.models.resolve(), args.ripgrep.resolve()
    prefix = args.out.with_suffix("")
    if args.out.exists() or any(args.out.parent.glob(prefix.name + ".*")):
        parser.error("output or stage files already exist; choose a fresh --out")
    if git(root, "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)notes/**"):
        parser.error("product inputs must be clean before native release verification")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    manifest = tomllib.loads((root / "bench/corpus.toml").read_text())
    revision = next(repo["rev"] for repo in manifest["repo"] if repo["name"] == "ripgrep")
    git(args.ripgrep, "cat-file", "-e", revision + "^{commit}")
    environment = dict(os.environ, DKC_MODELS_DIR=str(args.models), DKC_RERANKER_CACHE=str(args.models),
                       DKC_LIVE_METAL="1", MACOSX_DEPLOYMENT_TARGET="13.0")
    target = Path(environment.get("CARGO_TARGET_DIR", str(root / "target")))
    if not target.is_absolute():
        target = root / target
    binary = target / "release/dkc"
    run = Run(root, args.out, environment)
    before = source_manifest(root)
    report = {"schema": 2, "task": "T4.10", "status": "running", "started_at": now(),
              "host": {"system": platform.system(), "machine": platform.machine(),
                       "macos": platform.mac_ver()[0], "load_average_start": os.getloadavg()},
              "source": {"commit": git(root, "rev-parse", "HEAD"), "before_sha256": before},
              "models_dir": str(args.models), "ripgrep_source": str(args.ripgrep),
              "ripgrep_revision": revision}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    scratch = None
    worktree = None
    worktree_added = False
    try:
        if not run.run("build", ["cargo", "build", "--release", "--locked", "--features", "metal"], timeout=7200):
            raise RuntimeError("Metal build failed")
        report["binary"] = {"path": str(binary), "bytes": binary.stat().st_size, "sha256": sha256(binary)}
        for name in ("qwen3-embedding-0.6b-q8_0", "qwen3-reranker-0.6b-q8_0"):
            if not run.run(name, [str(binary), "models", "pull", name, "--json"], timeout=1800):
                raise RuntimeError("pinned model verification failed: " + name)
        registry = tomllib.loads((root / "src/models/models.toml").read_text())
        report["models"] = [{"name": item["name"], "bytes": (args.models / item["file"]).stat().st_size,
                             "sha256": sha256(args.models / item["file"]), "expected_sha256": item["sha256"]}
                            for item in registry["models"] if item["name"] in (
                                "qwen3-embedding-0.6b-q8_0", "qwen3-reranker-0.6b-q8_0")]
        if not run.run("probe-build", ["cargo", "test", "--release", "--locked", "--features",
                                        "metal", "--bin", "dkc", "--no-run"], timeout=7200):
            raise RuntimeError("Metal test executable build failed")
        run.run("gpu", ["/usr/sbin/system_profiler", "SPDisplaysDataType"])
        # Separate compilation load from timed inference; preserve a busy-host miss.
        admission = []
        deadline = time.monotonic() + 180
        while True:
            load = os.getloadavg()
            admission.append({"at": now(), "load_average": load})
            if load[0] < 1.0 or time.monotonic() >= deadline:
                break
            print(f"Waiting for benchmark idle admission: load={load[0]:.2f}", flush=True)
            time.sleep(15)
        report["host"]["idle_admission"] = admission
        report["host"]["load_average_before_probes"] = os.getloadavg()
        run_probes(run)
        scratch = Path(tempfile.mkdtemp(prefix="dkc-mac-verify-"))
        worktree = scratch / "ripgrep"
        if not run.run("ripgrep-worktree", ["git", "-C", str(args.ripgrep), "worktree", "add",
                                           "--detach", str(worktree), revision]):
            raise RuntimeError("pinned ripgrep worktree failed")
        worktree_added = True
        if not run.run("ripgrep-init", [str(binary), "init", "--profile", "local", "--yes", "--gitignore", "--json"], cwd=worktree):
            raise RuntimeError("ripgrep initialization failed")
        # Preserve the explicit local preset written by init; add only corpus exceptions.
        with (worktree / ".deepkit/config.toml").open("a") as stream:
            stream.write("\n" + (root / "bench/profiles/ripgrep.toml").read_text())
        for name, command in (
            ("index", ["index", "--level", "2", "--json"]),
            ("search", ["search", "regex matcher", "--mode", "hybrid", "--json"]),
            ("status", ["status", "--json"]),
        ):
            run.run("ripgrep-" + name, [str(binary), *command], cwd=worktree, timeout=7200)
    except Exception as error:
        report["error"] = str(error)
    finally:
        # Never remove unrelated worktrees. A failed cleanup is itself retained.
        if worktree_added:
            removed = run.run("ripgrep-cleanup", ["git", "-C", str(args.ripgrep), "worktree", "remove", "--force", str(worktree)])
            if removed and scratch:
                scratch.rmdir()
        elif scratch:
            report["retained_scratch"] = str(scratch)
        values, checks = measurements(run)
        after = source_manifest(root)
        report["source"]["after_sha256"] = after
        checks.update(native_apple_silicon=True, source_unchanged=before == after,
                      metal_build=bool(report.get("binary")),
                      pinned_models=bool(len(report.get("models", [])) == 2 and all(
                          item["sha256"] == item["expected_sha256"] for item in report.get("models", []))),
                      workflow_complete="error" not in report,
                      idle_admission=report["host"].get("load_average_before_probes", [float("inf")])[0] < 1.0)
        report.update(values, checks=checks, stages=run.stages, finished_at=now(),
                      status=verdict(checks, run.stages))
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "report": str(args.out), "checks": checks}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
