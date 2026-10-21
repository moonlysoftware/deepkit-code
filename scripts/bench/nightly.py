#!/usr/bin/env python3
"""Run the full, resumable Phase 4 nightly benchmark and regression gate."""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import tomllib

import nightly_resume as resume


ROOT = Path(__file__).resolve().parents[2]
PULL = ["git", "pull", "--ff-only", "--no-rebase"]
SHARED_LOCK = ROOT / "notes/runs/t4-2-benchmark.lock"
MAX_RUN_SECONDS = 6 * 60 * 60
BUDGET_MARGIN_SECONDS = 5 * 60


class BudgetDeferred(Exception):
    """The scheduled slot ended before another complete target could finish."""


def remaining_timeout(deadline: float, ceiling: int) -> int:
    """Bound each child by the service's remaining six-hour slot."""
    remaining = int(deadline - time.monotonic())
    if remaining < 60:
        raise BudgetDeferred("nightly time budget exhausted; unfinished targets retained")
    return min(ceiling, remaining)


def run_with_budget(
    runner: "Runner", arguments: list, deadline: float, ceiling: int, **kwargs
) -> int:
    """Run a child within both its own limit and the scheduled slot."""
    timeout = remaining_timeout(deadline, ceiling)
    try:
        return runner.run(arguments, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as error:
        if timeout < ceiling:
            raise BudgetDeferred("nightly budget ended during command; partial evidence retained") from error
        raise


@dataclass(frozen=True)
class Recipe:
    """One comparable nightly profile and its exact target cohort."""

    profile: str
    targets: tuple[str, ...]
    binary_kind: str
    require_gpu: bool


def atomic_write(path: Path, text: str) -> None:
    """Publish complete files so resume never observes half a result."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def nvidia_query(arguments: list[str], timeout: float = 10) -> str:
    """Bound a driver query even when the NVIDIA child stops responding."""
    command = ["nvidia-smi", *arguments]
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            # A driver-stuck child can remain in uninterruptible sleep. Do not
            # wait indefinitely or hold the nightly's output pipes open.
            if child.stdout is not None:
                child.stdout.close()
            if child.stderr is not None:
                child.stderr.close()
        raise TimeoutError("GPU preflight query timed out") from error
    if child.returncode != 0:
        raise RuntimeError(f"GPU preflight exited {child.returncode}: {stderr.strip()}")
    if stderr.strip():
        raise RuntimeError(f"GPU preflight diagnostic: {stderr.strip()}")
    return stdout


def gpu_inventory() -> int:
    """Reject a missing or malformed device before expensive nightly builds."""
    output = nvidia_query(["--query-gpu=index", "--format=csv,noheader,nounits"])
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if not rows or not all(row.isdigit() for row in rows):
        raise RuntimeError("GPU preflight returned no valid device inventory")
    return len(rows)


def manifest_targets(path: Path, group: str | None = None) -> tuple[str, ...]:
    """Read the frozen manifest order, optionally selecting one named group."""
    document = tomllib.loads(path.read_text())
    repositories = document.get("repo", [])
    names = []
    for repository in repositories:
        if group is None or group in repository.get("groups", []):
            names.append(repository["name"])
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{path}: empty or duplicate benchmark target selection")
    return tuple(names)


def recipes(path: Path) -> tuple[Recipe, ...]:
    """Return the T4.5 recipes from plan/07-dev-machine.md section 5."""
    full = manifest_targets(path)
    group_a = manifest_targets(path, "A")
    return (
        Recipe("local", full, "cuda", True),
        Recipe("moonly", full, "cpu", False),
        Recipe("cpu", group_a, "cpu", False),
    )


def read_report(path: Path) -> dict:
    """Read one structurally complete benchmark report."""
    report = json.loads(path.read_text())
    if not isinstance(report.get("targets"), list) or not isinstance(
        report.get("failures"), list
    ):
        raise ValueError(f"{path}: missing target or failure arrays")
    return report


def result_names(report: dict) -> list[str]:
    return [target["name"] for target in report["targets"]] + [
        failure["name"] for failure in report["failures"]
    ]


def validate_combined(path: Path, expected: tuple[str, ...], profile: str) -> dict:
    """Require the complete intended cohort before comparison or promotion."""
    report = read_report(path)
    names = result_names(report)
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f"{path}: incomplete {profile} target cohort")
    if report.get("profile") != profile:
        raise ValueError(f"{path}: expected profile {profile}")
    if report["failures"]:
        raise ValueError(f"{path}: contains {len(report['failures'])} target failures")
    return report


class Runner:
    """Run bounded children with one durable log and process-group cleanup."""

    def __init__(self, log: Path):
        self.log = log
        self.rotation = 0

    def run(
        self,
        args,
        *,
        timeout: int = 3600,
        output: Path | None = None,
        environment: dict[str, str] | None = None,
        check: bool = True,
    ) -> int:
        with self.log.open("a") as log:
            log.write(json.dumps({"command": [str(argument) for argument in args]}) + "\n")
            log.flush()
            if output is None:
                child = subprocess.Popen(
                    [str(argument) for argument in args],
                    cwd=ROOT,
                    stdout=log,
                    stderr=log,
                    env=environment,
                    start_new_session=True,
                )
                destination = None
            else:
                destination = output.open("w")
                child = subprocess.Popen(
                    [str(argument) for argument in args],
                    cwd=ROOT,
                    stdout=destination,
                    stderr=log,
                    env=environment,
                    start_new_session=True,
                )
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            finally:
                if destination is not None:
                    destination.close()
        self.rotate()
        if check and code:
            raise RuntimeError(f"{args[0]} exited {code}; see {self.log}")
        return code

    def rotate(self) -> None:
        if self.log.stat().st_size < 10 * 1024 * 1024:
            return
        self.rotation += 1
        self.log.rename(self.log.with_suffix(f".{self.rotation}.log"))
        self.log.write_text("Previous output rotated to numbered sibling log.\n")

    def idle(self, require_gpu: bool, timeout: int = 900) -> None:
        """Wait for the documented immediate CPU and optional GPU preflight."""
        deadline = time.monotonic() + timeout
        while True:
            gpu_empty = None
            if require_gpu:
                gpu = nvidia_query(
                    [
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                )
                gpu_empty = not gpu.strip()
            load = os.getloadavg()[0]
            ready = load < 1 and gpu_empty is not False
            with self.log.open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "load_1m": load,
                            "gpu_required": require_gpu,
                            "gpu_compute_empty": gpu_empty,
                            "idle": ready,
                            "utc": utc(),
                        }
                    )
                    + "\n"
                )
            if ready:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("idle preflight timed out")
            time.sleep(10)


def build_binaries(
    runner: Runner, stamp: str, deadline: float | None = None
) -> dict[str, Path]:
    """Build and freeze comparable CPU and CUDA release executables."""
    if deadline is None:
        deadline = time.monotonic() + MAX_RUN_SECONDS - BUDGET_MARGIN_SECONDS
    base_environment = dict(os.environ)
    base_environment["CARGO_INCREMENTAL"] = "0"
    cpu_environment = dict(base_environment)
    cpu_environment["CARGO_TARGET_DIR"] = str(ROOT / "target/nightly-cpu")
    cuda_environment = dict(base_environment)
    cuda_environment["CARGO_TARGET_DIR"] = str(ROOT / "target/nightly-cuda")
    run_with_budget(
        runner,
        ["cargo", "build", "--release", "--locked"],
        deadline,
        7200,
        environment=cpu_environment,
    )
    run_with_budget(
        runner,
        ["cargo", "build", "--release", "--locked", "--features", "cuda"],
        deadline,
        7200,
        environment=cuda_environment,
    )
    live_environment = dict(cuda_environment)
    live_environment.update(
        {
            "DKC_LIVE_CUDA": "1",
            "DKC_LIVE_GATEWAY": "1",
            "DKC_LIVE_OLLAMA": "1",
        }
    )
    run_with_budget(
        runner,
        [
            "cargo",
            "test",
            "--locked",
            "--features",
            "cuda",
            "--",
            "--include-ignored",
            "--skip",
            "diagnose_saved_module_once_with_accounting",
        ],
        deadline,
        7200,
        environment=live_environment,
    )
    snapshots = {}
    for kind, target_directory in [
        ("cpu", ROOT / "target/nightly-cpu"),
        ("cuda", ROOT / "target/nightly-cuda"),
    ]:
        source = target_directory / "release/dkc"
        destination = ROOT / "target/nightly" / stamp / kind / "dkc"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination.chmod(0o555)
        snapshots[kind] = destination
    return snapshots


def validate_target_report(
    path: Path, target: str, recipe: Recipe, code: int, build: dict | None
) -> tuple[dict, dict]:
    """Validate one per-target report and preserve its failure evidence."""
    report = read_report(path)
    if report.get("profile") != recipe.profile:
        raise ValueError(f"{path}: wrong profile")
    if build is not None and report.get("build") != build:
        raise ValueError(f"{path}: build identity changed within profile")
    if result_names(report) != [target]:
        raise ValueError(f"{path}: expected exactly target {target}")
    failed = bool(report["failures"])
    if failed == (code == 0):
        raise ValueError(f"{path}: exit status and failure evidence disagree")
    if not failed:
        preflight = report["targets"][0].get("provenance", {}).get("pre_index", {})
        load = preflight.get("load_average", [])
        if not isinstance(load, list) or not load or load[0] >= 1:
            raise ValueError(f"{path}: immediate load preflight was not below 1")
        if recipe.require_gpu and preflight.get("idle") is not True:
            raise ValueError(f"{path}: immediate CUDA preflight was not idle")
    return report, report.get("build")


def slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-" for character in value
    ).strip("-")


def combine_reports(
    path: Path, recipe: Recipe, reports: list[dict], sources: list[Path], build: dict
) -> dict:
    """Publish one profile report and sibling Markdown from per-target evidence."""
    if not reports:
        raise ValueError(f"no {recipe.profile} target report completed")
    combined = reports[0].copy()
    combined["targets"] = [target for report in reports for target in report["targets"]]
    combined["failures"] = [
        failure for report in reports for failure in report["failures"]
    ]
    combined["source_reports"] = [str(source) for source in sources]
    combined["build"] = build
    atomic_write(path, json.dumps(combined, indent=2) + "\n")
    markdown = [f"# Nightly {recipe.profile}", ""]
    for target, source_report in zip(recipe.targets, sources, strict=True):
        source = source_report.with_suffix(".md")
        if source.exists():
            markdown.extend([f"## {target}", "", source.read_text().rstrip(), ""])
    atomic_write(path.with_suffix(".md"), "\n".join(markdown).rstrip() + "\n")
    return combined


def run_recipe(
    recipe: Recipe,
    binary: Path,
    corpus: Path,
    out: Path,
    runner: Runner,
    state: dict,
    state_path: Path,
    expected_build: dict,
    deadline: float | None = None,
) -> dict:
    if deadline is None:
        deadline = time.monotonic() + MAX_RUN_SECONDS - BUDGET_MARGIN_SECONDS
    reports = []
    sources = []
    environment = dict(os.environ)
    environment.setdefault("DKC_MODELS_DIR", "/data/dkc-models")
    for target in recipe.targets:
        key = f"{recipe.profile}:{target}"
        saved = state["targets"].get(key, {})
        if saved.get("state") == "complete" and (
            recipe.profile != "moonly" or state["inputs"]["model_receipt"] is not None
        ):
            source = resume.report_path(saved["path"], out.parent)
            if resume.sha256(source) != saved.get("sha256"):
                raise ValueError(f"{key}: completed report bytes changed")
            report, _ = validate_target_report(
                source, target, recipe, 0, expected_build
            )
            reports.append(report)
            sources.append(source)
            continue
        if saved.get("state") == "failed":
            raise ValueError(f"{key}: prior failed target requires diagnosis or new inputs")
        idle_timeout = remaining_timeout(deadline, 900)
        attempt = int(saved.get("attempt", 0)) + 1
        target_out = out.with_name(
            f"{out.stem}-{slug(target)}-attempt-{attempt}.json"
        )
        if target_out.exists():
            raise ValueError(f"{key}: target output already exists: {target_out}")
        state["targets"][key] = {
            "state": "running", "attempt": attempt, "path": str(target_out)
        }
        resume.write_state(state_path, state)
        try:
            runner.idle(recipe.require_gpu, timeout=idle_timeout)
        except TimeoutError as error:
            if idle_timeout < 900:
                raise BudgetDeferred("nightly budget ended during idle admission") from error
            raise
        try:
            code = run_with_budget(
                runner,
                [
                    binary,
                    "bench",
                    "run",
                    "--corpus",
                    corpus,
                    "--profile",
                    recipe.profile,
                    "--only",
                    target,
                    "--out",
                    target_out,
                    "--keep",
                    "--json",
                ],
                deadline,
                10800,
                environment=environment,
                check=False,
            )
        except subprocess.TimeoutExpired:
            state["targets"][key] = {
                "state": "failed", "attempt": attempt, "path": str(target_out),
                "reason": "target timed out at its own three-hour limit",
            }
            resume.write_state(state_path, state)
            raise
        report, observed_build = validate_target_report(
            target_out, target, recipe, code, expected_build
        )
        if observed_build != expected_build:
            raise ValueError(f"{key}: build identity changed")
        state["targets"][key] = {
            "state": "failed" if report["failures"] else "complete",
            "attempt": attempt,
            "path": str(target_out),
            "sha256": resume.sha256(target_out),
        }
        resume.write_state(state_path, state)
        if report["failures"]:
            raise ValueError(f"{key}: target report contains failures")
        reports.append(report)
        sources.append(target_out)
    return combine_reports(out, recipe, reports, sources, expected_build)


def binary_metadata(binary: Path) -> dict:
    """Read the exact build identity recorded by a frozen executable."""
    output = subprocess.check_output(
        [str(binary), "bench", "metadata", "--json"], text=True, timeout=60
    )
    build = json.loads(output)["build"]
    if build.get("source_dirty") != "false":
        raise ValueError(f"{binary}: nightly binary contains dirty source")
    return build


def valid_binary_build(build: dict, commit: str, kind: str) -> bool:
    """Require the pulled release source and the intended CPU/CUDA feature."""
    features = build.get("features", "").split(",")
    return (
        build.get("commit") == commit
        and build.get("profile") == "release"
        and build.get("source_dirty") == "false"
        and ("CUDA" in features) == (kind == "cuda")
    )


def restore_binaries(state: dict, commit: str, inputs: dict) -> dict[str, Path] | None:
    """Reuse only unchanged inputs and byte-identical frozen executables."""
    if state.get("commit") != commit or state.get("inputs") != inputs:
        return None
    if state.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return None
    frozen = state.get("binaries", {})
    binaries = {}
    for kind in ("cpu", "cuda"):
        record = frozen.get(kind, {})
        path = Path(record.get("path", "")).resolve()
        if not path.is_relative_to((ROOT / "target/nightly").resolve()):
            return None
        if not path.is_file() or resume.sha256(path) != record.get("sha256"):
            return None
        if not valid_binary_build(record.get("build", {}), commit, kind):
            return None
        if binary_metadata(path) != record.get("build"):
            return None
        binaries[kind] = path
    return binaries


def append_verdict(
    path: Path,
    *,
    commit: str,
    profile: str,
    verdict: str,
    stage: str,
    detail: str,
    report: Path,
    baseline: Path | None,
    diff: Path | None,
    log: Path,
) -> str:
    previous = path.read_text() if path.exists() else "# Nightly regression\n\n"
    safe = detail.replace("\n", " ").replace("|", "/")
    entry = (
        f"- {utc()} · {commit[:12]} · {profile} · **{verdict}** · {stage}: {safe}. "
        f"Report `{report}`; baseline `{baseline}`; diff `{diff}`; log `{log}`.\n"
    )
    atomic_write(path, previous + entry)
    print(entry.strip(), flush=True)
    return entry


def dry_run(args) -> int:
    corpus = args.corpus.resolve(strict=True)
    planned = recipes(corpus)
    response = {
        "dry_run": True,
        "corpus": str(corpus),
        "pull": PULL,
        "builds": ["cpu release", "cuda release"],
        "live_cuda_tests": True,
        "recipes": [
            {
                "profile": recipe.profile,
                "binary": recipe.binary_kind,
                "require_gpu": recipe.require_gpu,
                "target_count": len(recipe.targets),
                "targets": recipe.targets,
            }
            for recipe in planned
        ],
    }
    print(json.dumps(response, indent=2))
    return 0


def compare_only(args, runner: Runner, log_path: Path, commit: str) -> int:
    stamp = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        + f"-{os.getpid()}"
    )
    candidate = args.compare_only.resolve()
    saved = args.out_dir / f"nightly-{stamp}-baseline.json"
    diff = args.out_dir / f"nightly-{stamp}-diff.json"
    baseline = args.baseline.resolve() if args.baseline is not None else None
    try:
        if baseline is None:
            raise ValueError("--compare-only requires --baseline")
        baseline = baseline.resolve(strict=True)
        candidate = candidate.resolve(strict=True)
        read_report(baseline)
        read_report(candidate)
        atomic_write(saved, baseline.read_text())
        code = runner.run(
            [
                args.binary.resolve(),
                "bench",
                "diff",
                saved,
                candidate,
                "--fail-on-regression",
                "--json",
            ],
            timeout=60,
            output=diff,
            check=False,
        )
        passed = code == 0 and json.loads(diff.read_text()).get("pass") is True
        detail = "synthetic comparison completed"
    except (Exception, KeyboardInterrupt) as error:
        passed = False
        detail = str(error)
    latest = args.out_dir / "nightly-latest.json"
    if passed:
        atomic_write(latest, candidate.read_text())
    append_verdict(
        args.out_dir / "NIGHTLY.md",
        commit=commit,
        profile="compare-only",
        verdict="PASS" if passed else "FAIL",
        stage="complete" if passed else "diff",
        detail=detail,
        report=candidate,
        baseline=saved if saved.exists() else baseline,
        diff=diff if diff.exists() else None,
        log=log_path,
    )
    return 0 if passed else 1


def execute(args) -> int:
    deadline = time.monotonic() + MAX_RUN_SECONDS - BUDGET_MARGIN_SECONDS
    corpus = args.corpus.resolve(strict=True)
    planned = recipes(corpus)
    stamp = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        + f"-{os.getpid()}"
    )
    day = datetime.datetime.now().astimezone().date().isoformat()
    log_path = args.log_dir / f"nightly-{stamp}.log"
    runner = Runner(log_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.compare_only is not None:
        return compare_only(args, runner, log_path, commit)
    state_path = args.log_dir / "nightly-active.state.json"
    stage = "pull"
    try:
        runner.run(PULL, timeout=remaining_timeout(deadline, 120))
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        stage = "gpu-preflight"
        devices = gpu_inventory()
        with log_path.open("a") as log:
            log.write(json.dumps({"gpu_devices": devices, "utc": utc()}) + "\n")
        stage = "input-preflight"
        inputs = resume.input_snapshot(
            corpus, getattr(args, "model_identity_receipt", None)
        )
        state = resume.read_state(state_path)
        binaries = (
            restore_binaries(state, commit, inputs) if state is not None else None
        )
        if state is not None and binaries is None:
            archive = args.log_dir / f"nightly-{state['stamp']}-abandoned.state.json"
            if archive.exists():
                archive = args.log_dir / f"nightly-{state['stamp']}-{stamp}-abandoned.state.json"
            state_path.replace(archive)
            with log_path.open("a") as log:
                log.write(f"Archived incompatible run state to {archive}\n")
            state = None
        if state is None:
            stage = "build-and-live-tests"
            binaries = build_binaries(runner, stamp, deadline)
            frozen = {
                kind: {
                    "path": str(binary),
                    "sha256": resume.sha256(binary),
                    "build": binary_metadata(binary),
                }
                for kind, binary in binaries.items()
            }
            if any(
                not valid_binary_build(record["build"], commit, kind)
                for kind, record in frozen.items()
            ):
                raise ValueError("nightly binaries do not match pulled release source/features")
            state = {
                "schema": resume.SCHEMA,
                "stamp": stamp,
                "commit": commit,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "inputs": inputs,
                "binaries": frozen,
                "targets": {},
                "profiles": {},
            }
            resume.write_state(state_path, state)
        else:
            stamp = state["stamp"]
            with log_path.open("a") as log:
                log.write(f"Resuming frozen nightly {stamp}\n")
    except BudgetDeferred as error:
        atomic_write(log_path.with_suffix(".exit"), "0\n")
        append_verdict(
            args.out_dir / "NIGHTLY.md",
            commit=commit,
            profile="all",
            verdict="DEFERRED",
            stage=stage,
            detail=str(error),
            report=args.out_dir / f"{day}-all.json",
            baseline=None,
            diff=None,
            log=log_path,
        )
        return 0
    except (Exception, KeyboardInterrupt) as error:
        atomic_write(log_path.with_suffix(".exit"), "1\n")
        append_verdict(
            args.out_dir / "NIGHTLY.md",
            commit=commit,
            profile="all",
            verdict="FAIL",
            stage=stage,
            detail=str(error),
            report=args.out_dir / f"{day}-all.json",
            baseline=None,
            diff=None,
            log=log_path,
        )
        return 1
    failed = False
    deferred = False
    profiles = state.setdefault("profiles", {})
    for recipe in planned:
        candidate = args.out_dir / f"nightly-{stamp}-{recipe.profile}.json"
        latest = args.out_dir / f"nightly-latest-{recipe.profile}.json"
        diff = args.out_dir / f"nightly-{stamp}-{recipe.profile}-diff.json"
        baseline_copy = args.out_dir / f"nightly-{stamp}-{recipe.profile}-baseline.json"
        baseline = baseline_copy if baseline_copy.exists() else None
        stage = "run"
        try:
            prior = profiles.get(recipe.profile, {})
            if prior.get("state") == "complete":
                if (
                    recipe.profile == "moonly"
                    and state["inputs"]["model_receipt"] is None
                ):
                    raise ValueError(
                        "completed Moonly profile cannot resume without a deployment receipt"
                    )
                if resume.sha256(candidate) != prior.get("sha256"):
                    raise ValueError(f"{recipe.profile}: completed profile report changed")
                report = validate_combined(candidate, recipe.targets, recipe.profile)
                if report.get("build") != state["binaries"][recipe.binary_kind]["build"]:
                    raise ValueError(f"{recipe.profile}: completed profile build changed")
                continue
            if prior.get("state") == "failed":
                failed = True
                continue
            if baseline is None and latest.exists():
                atomic_write(baseline_copy, latest.read_text())
                baseline = baseline_copy
            run_recipe(
                recipe, binaries[recipe.binary_kind], corpus, candidate, runner,
                state, state_path, state["binaries"][recipe.binary_kind]["build"],
                deadline,
            )
            validate_combined(candidate, recipe.targets, recipe.profile)
            if baseline is not None:
                stage = "diff"
                if diff.exists():
                    diff = args.out_dir / (
                        f"nightly-{stamp}-{recipe.profile}-diff-retry-{os.getpid()}.json"
                    )
                    if diff.exists():
                        raise ValueError(f"comparison output already exists: {diff}")
                runner.run(
                    [
                        binaries[recipe.binary_kind],
                        "bench",
                        "diff",
                        baseline,
                        candidate,
                        "--fail-on-regression",
                        "--json",
                    ],
                    timeout=60,
                    output=diff,
                )
                if json.loads(diff.read_text()).get("pass") is not True:
                    raise ValueError("diff did not report PASS")
            else:
                diff = None
            atomic_write(latest, candidate.read_text())
            verdict = "PASS"
            detail = (
                "initial baseline"
                if baseline is None
                else "full cohort passed regression diff"
            )
            stage = "complete"
        except BudgetDeferred as error:
            deferred = True
            verdict = "DEFERRED"
            detail = str(error)
            with log_path.open("a") as log:
                log.write(f"DEFERRED {recipe.profile} at {stage}: {detail}\n")
        except (Exception, KeyboardInterrupt) as error:
            failed = True
            verdict = "FAIL"
            detail = str(error)
            with log_path.open("a") as log:
                log.write(f"FAIL {recipe.profile} at {stage}: {detail}\n")
        append_verdict(
            args.out_dir / "NIGHTLY.md",
            commit=commit,
            profile=recipe.profile,
            verdict=verdict,
            stage=stage,
            detail=detail,
            report=candidate,
            baseline=baseline,
            diff=diff,
            log=log_path,
        )
        structured_failure = any(
            state["targets"].get(f"{recipe.profile}:{target}", {}).get("state") == "failed"
            for target in recipe.targets
        )
        if verdict == "PASS" or (
            verdict == "FAIL" and (structured_failure or stage == "diff")
        ):
            profiles[recipe.profile] = {
                "state": "complete" if verdict == "PASS" else "failed",
                "report": str(candidate),
                "sha256": resume.sha256(candidate) if candidate.exists() else None,
            }
            resume.write_state(state_path, state)
        if deferred:
            break
    code = 1 if failed else 0
    if code == 0 and not deferred:
        state_path.replace(args.log_dir / f"nightly-{stamp}-complete.state.json")
    atomic_write(log_path.with_suffix(".exit"), f"{code}\n")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "notes/bench")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "notes/runs")
    parser.add_argument("--corpus", type=Path, default=ROOT / "bench/corpus.toml")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--compare-only", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/dkc")
    parser.add_argument("--model-identity-receipt", type=Path)
    parser.add_argument("--candidate-manifest", type=Path,
                        help="run an immutable Phase 4 matrix candidate")
    parser.add_argument("--candidate-limit", type=int,
                        help="stop after this many accepted candidate lanes")
    parser.add_argument("--candidate-skip-lane",
                        help="explicitly skip one failed candidate target:profile")
    parser.add_argument("--candidate-skip-reason",
                        help="diagnosed reason for an explicit candidate skip")
    parser.add_argument("--candidate-skip-evidence", type=Path,
                        help="retained notes evidence for an explicit candidate skip")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    if args.candidate_manifest is not None and args.compare_only is not None:
        parser.error("--candidate-manifest cannot be combined with --compare-only")
    if args.candidate_manifest is None and args.candidate_limit is not None:
        parser.error("--candidate-limit requires --candidate-manifest")
    skipping = any(value is not None for value in (
        args.candidate_skip_lane, args.candidate_skip_reason,
        args.candidate_skip_evidence,
    ))
    if skipping and (args.candidate_manifest is None or args.dry_run
                     or args.candidate_limit is not None or not all((
                         args.candidate_skip_lane, args.candidate_skip_reason,
                         args.candidate_skip_evidence,
                     ))):
        parser.error("candidate skip requires manifest, lane, reason and evidence only")
    if args.candidate_manifest is not None:
        import matrix_run

        if args.dry_run:
            try:
                return matrix_run.execute(args, root=ROOT, runner_type=Runner)
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(json.dumps({"ready": False, "error": str(error)}))
                return 2
    if args.dry_run:
        return dry_run(args)
    for directory in (args.out_dir, args.log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Offline comparisons only publish under their own output directory. Keep
    # the corpus slot for runs that may build, index, or measure live targets.
    lock_paths = [args.log_dir / "dkc-nightly.lock"]
    if args.compare_only is None:
        lock_paths.append(SHARED_LOCK)
    locks = [path.open("w") for path in lock_paths]
    try:
        for lock in locks:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("another benchmark run owns the slot", flush=True)
                return 75
        if args.candidate_manifest is not None:
            try:
                if skipping:
                    return matrix_run.skip_failed(args, root=ROOT)
                return matrix_run.execute(args, root=ROOT, runner_type=Runner)
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(json.dumps({"ready": False, "error": str(error)}))
                return 2
        return execute(args)
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
