"""Run a frozen Phase 4 candidate through the existing nightly benchmark harness.

The candidate manifest stays immutable. A separate state file records accepted
lane receipts; every resume reconstructs and rechecks those receipts before a
new benchmark can start. See v1.md §14 and plan/06-benchmarks.md §3.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys

import candidate_preflight as frozen
import model_identity
import nightly_resume as resume


SCHEMA = "dkc.phase4.matrix-run-state.v1"
PROFILES = {
    "local_cuda": ("local", "cuda", True),
    "moonly": ("moonly", "moonly", False),
    "local_cpu": ("cpu", "cpu", False),
    "ollama": ("ollama", "ollama", True),
    "dual_local_code_moonly_docs": ("dual", "dual", True),
}


def utc() -> str:
    """Timestamp attempt transitions so an interrupted run remains diagnosable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key(lane: dict) -> str:
    """Identify one manifest lane without depending on its output filename."""
    return f"{lane['target']}:{lane['profile']}"


def publish(path: Path, value: dict) -> None:
    """Atomically save a complete matrix checkpoint after each transition."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_state(path: Path, manifest_sha256: str,
               *, allow_unresolved: bool = False) -> dict:
    """Read only state belonging to these exact immutable candidate bytes."""
    if not path.exists():
        return {"schema": SCHEMA, "manifest_sha256": manifest_sha256,
                "completed": {}, "skipped": {}, "attempts": []}
    state = json.loads(path.read_text())
    if (state.get("schema") != SCHEMA or
            state.get("manifest_sha256") != manifest_sha256 or
            not isinstance(state.get("completed"), dict) or
            not isinstance(state.get("skipped", {}), dict) or
            not isinstance(state.get("attempts"), list)):
        raise ValueError("matrix state is malformed or belongs to another candidate")
    state.setdefault("skipped", {})
    unresolved = [row for row in state["attempts"]
                  if not isinstance(row, dict) or row.get("state") != "complete"]
    if any(not isinstance(row, dict) or row.get("state") not in (
        "failed", "admitting", "running"
    ) for row in unresolved):
        raise ValueError("matrix state has a malformed interrupted attempt")
    if any(not isinstance(lane_key, str) or
           sum(row.get("lane") == lane_key for row in unresolved) != 1
           for lane_key in state["skipped"]):
        raise ValueError("matrix skipped state needs one failed/interrupted attempt")
    if not allow_unresolved and any(
        not isinstance(row, dict) or row.get("lane") not in state["skipped"]
        for row in unresolved
    ):
        raise ValueError("matrix has an interrupted or failed attempt; retain and diagnose it")
    return state


def merged_candidate(base: dict, state: dict) -> dict:
    """Overlay accepted and explicitly skipped receipts for static preflight."""
    document = copy.deepcopy(base)
    lanes = {key(lane): lane for lane in document["lanes"]}
    def append_input(name: str, digest: str) -> None:
        existing = next((row for row in document["inputs"]
                         if row.get("path") == name), None)
        if existing is None:
            document["inputs"].append({"path": name, "sha256": digest})
        elif existing.get("sha256") != digest:
            raise ValueError(f"frozen input hash conflicts with state: {name}")

    for lane_key, entry in state["completed"].items():
        lane = lanes.get(lane_key)
        if lane is None or lane.get("state") != "pending":
            raise ValueError(f"completed state has no pending candidate lane: {lane_key}")
        audit_name = entry.get("audit_input")
        audit_sha = entry.get("audit_sha256")
        report_sha = entry.get("report_sha256")
        if not all(isinstance(item, str) and item for item in
                   (audit_name, audit_sha, report_sha)):
            raise ValueError(f"completed state lacks frozen evidence: {lane_key}")
        observations = entry.get("identity_observations")
        if not isinstance(observations, dict) or set(observations) != {"before", "after"}:
            raise ValueError(f"completed state lacks live model evidence: {lane_key}")
        lane.update(state="complete", report_sha256=report_sha,
                    audit_input=audit_name,
                    candidate_sha256=state["manifest_sha256"],
                    identity_observations=observations)
        append_input(audit_name, audit_sha)
        for row in observations.values():
            append_input(row["path"], row["sha256"])
    for lane_key, entry in state["skipped"].items():
        lane = lanes.get(lane_key)
        if lane is None or lane.get("state") != "pending" or lane_key in state["completed"]:
            raise ValueError(f"skipped state has no pending candidate lane: {lane_key}")
        receipt = entry.get("skip_input")
        report_sha = entry.get("report_sha256")
        if not isinstance(receipt, str) or not receipt:
            raise ValueError(f"skipped state lacks disposition: {lane_key}")
        for name_field, sha_field in (
            ("skip_input", "skip_sha256"),
            ("evidence_input", "evidence_sha256"),
            ("log_input", "log_sha256"),
        ):
            name, digest = entry.get(name_field), entry.get(sha_field)
            if not isinstance(name, str) or not isinstance(digest, str):
                raise ValueError(f"skipped state lacks frozen {name_field}: {lane_key}")
            append_input(name, digest)
        for row in entry.get("identity_observations", {}).values():
            append_input(row["path"], row["sha256"])
        if entry.get("audit_input") is not None:
            audit_name, audit_sha = entry["audit_input"], entry.get("audit_sha256")
            if not isinstance(audit_name, str) or not isinstance(audit_sha, str):
                raise ValueError(f"skipped state lacks frozen rejected audit: {lane_key}")
            append_input(audit_name, audit_sha)
        lane.update(state="skipped", skip_input=receipt,
                    candidate_sha256=state["manifest_sha256"],
                    report_sha256=report_sha)
    return document


def attempt_log(log_dir: Path, manifest_sha256: str, lane: dict) -> Path:
    """Use a stable unique log path without allowing target names to escape."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", key(lane))
    return log_dir / f"matrix-{manifest_sha256[:16]}-{name}.log"


@contextmanager
def restored_target_config(target: dict, manifest_sha256: str):
    """Restore frozen private config bytes after a lane, including a failed lane."""
    root = Path(target["root"])
    baseline = target["config_baseline"]
    names = ("config.toml", "config.local.toml")
    for name in names:
        path = root / ".deepkit" / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"target has an unsafe starting {name}")
        actual = frozen.sha256(path) if path.exists() else None
        if actual != baseline.get(name):
            raise ValueError(f"target starting {name} changed before lane")
    parent = root / ".deepkit"
    if parent.is_symlink():
        raise ValueError("target .deepkit directory is linked")
    parent.mkdir(exist_ok=True)
    recovery = parent / f"bench-config-recovery-{manifest_sha256[:16]}"
    if recovery.exists() or recovery.is_symlink():
        raise ValueError(f"unresolved benchmark config recovery: {recovery}")
    recovery.mkdir(mode=0o700)
    modes = {}
    try:
        for name in names:
            path = parent / name
            if path.exists():
                modes[name] = path.stat().st_mode & 0o777
                backup = recovery / name
                shutil.copyfile(path, backup)
                backup.chmod(0o600)
        (recovery / "manifest.json").write_text(json.dumps({
            "schema": "dkc.phase4.config-recovery.v1",
            "candidate_sha256": manifest_sha256, "baseline": baseline,
            "original_modes": modes,
        }, sort_keys=True) + "\n")
        yield
    finally:
        try:
            for name in names:
                path = parent / name
                if name in baseline:
                    temporary = parent / f"{name}.bench-restore.tmp"
                    shutil.copyfile(recovery / name, temporary)
                    temporary.chmod(modes[name])
                    os.replace(temporary, path)
                elif path.exists() or path.is_symlink():
                    if not path.is_file() and not path.is_symlink():
                        raise ValueError(f"lane left unsafe {name}; recovery retained")
                    path.unlink()
            if any((parent / name).exists() != (name in baseline) or
                   (name in baseline and frozen.sha256(parent / name) != baseline[name])
                   for name in names):
                raise ValueError("benchmark config restoration changed frozen bytes")
        except Exception:
            raise
        else:
            shutil.rmtree(recovery)


def candidate_identity(args, root: Path) -> tuple[Path, str, dict, Path]:
    """Resolve a frozen manifest and its separate SHA-bound progress file."""
    if not args.log_dir.resolve().is_relative_to((root / "notes/runs").resolve()):
        raise ValueError("candidate logs and state must stay under notes/runs")
    manifest = args.candidate_manifest.resolve()
    if not manifest.is_file() or not manifest.is_relative_to((root / "notes/bench").resolve()):
        raise ValueError("candidate manifest must be a file under notes/bench")
    manifest_sha = resume.sha256(manifest)
    base = json.loads(manifest.read_text())
    state_path = args.log_dir / f"matrix-{manifest_sha[:16]}.state.json"
    return manifest, manifest_sha, base, state_path


def skip_failed(args, *, root: Path) -> int:
    """Record an operator-diagnosed blocker without accepting or retrying it."""
    _manifest, manifest_sha, base, state_path = candidate_identity(args, root)
    if not state_path.is_file():
        raise ValueError("matrix has no failed attempt to skip")
    state = read_state(state_path, manifest_sha, allow_unresolved=True)
    unresolved = [row for row in state["attempts"]
                  if not isinstance(row, dict) or row.get("state") != "complete"
                  and row.get("lane") not in state["skipped"]]
    lane_key = args.candidate_skip_lane
    if (len(unresolved) != 1 or not isinstance(unresolved[0], dict)
            or unresolved[0].get("lane") != lane_key
            or unresolved[0].get("state") not in ("failed", "admitting", "running")):
        raise ValueError("skip requires the one retained failed/interrupted attempt for this lane")
    reason = args.candidate_skip_reason
    if not isinstance(reason, str) or len(reason.strip()) < 12:
        raise ValueError("skip reason must describe the diagnosed blocker")
    lane = next((row for row in base["lanes"] if key(row) == lane_key), None)
    if lane is None or lane.get("state") != "pending":
        raise ValueError("skip lane is absent or not pending in frozen candidate")
    evidence = args.candidate_skip_evidence.resolve()
    if not evidence.is_file() or not evidence.is_relative_to((root / "notes").resolve()):
        raise ValueError("skip evidence must be an existing file under notes")
    attempt = unresolved[0]
    log = Path(attempt["log"]).resolve()
    if not log.is_file() or not log.is_relative_to((root / "notes/runs").resolve()):
        raise ValueError("failed attempt log is missing from notes/runs")
    output = frozen.inside(root, lane["output"])
    report_sha = resume.sha256(output) if output.is_file() else None
    if output.exists() and report_sha is None:
        raise ValueError("failed lane output is not a regular report file")
    audit = output.with_name(output.stem + "-audit.json")
    if audit.exists() and not audit.is_file():
        raise ValueError("failed lane audit output is not a regular file")
    audit_name = audit.relative_to(root).as_posix() if audit.exists() else None
    audit_sha = resume.sha256(audit) if audit_name else None
    receipt = output.with_name(output.stem + "-skip.json")
    if receipt.exists():
        raise ValueError(f"skip receipt already exists: {receipt}")
    receipt_name = receipt.relative_to(root).as_posix()
    evidence_name = evidence.relative_to(root).as_posix()
    log_name = log.relative_to(root).as_posix()
    disposition = {
        "schema": "dkc.phase4.matrix-lane-skip.v1",
        "candidate_sha256": manifest_sha, "target": lane["target"],
        "profile": lane["profile"], "output": lane["output"],
        "reason": reason.strip(), "failed_state": attempt["state"],
        "report_sha256": report_sha,
        "rejected_audit": ({"path": audit_name, "sha256": audit_sha}
                           if audit_name else None),
        "evidence": {"path": evidence_name, "sha256": resume.sha256(evidence)},
        "attempt_log": {"path": log_name, "sha256": resume.sha256(log)},
        "identity_observations": attempt.get("identity_observations", {}),
        "recorded_utc": utc(),
    }
    with receipt.open("x") as target:
        target.write(json.dumps(disposition, indent=2, sort_keys=True) + "\n")
    state["skipped"][lane_key] = {
        "skip_input": receipt_name, "skip_sha256": resume.sha256(receipt),
        "evidence_input": evidence_name,
        "evidence_sha256": disposition["evidence"]["sha256"],
        "log_input": log_name, "log_sha256": disposition["attempt_log"]["sha256"],
        "report_sha256": report_sha,
        "audit_input": audit_name, "audit_sha256": audit_sha,
        "identity_observations": attempt.get("identity_observations", {}),
    }
    issues = frozen.check(root, merged_candidate(base, state))
    if issues:
        state["skipped"].pop(lane_key)
        receipt.unlink()
        raise ValueError(f"skip disposition failed candidate preflight: {issues}")
    publish(state_path, state)
    print(json.dumps({"ready": True, "skipped_lane": lane_key,
                      "candidate_complete": False, "state": str(state_path)}))
    return 0


def execute(args, *, root: Path, runner_type) -> int:
    """Validate and run pending lanes, preserving every failure for diagnosis."""
    _manifest, manifest_sha, base, state_path = candidate_identity(args, root)
    state = read_state(state_path, manifest_sha)
    if args.candidate_limit is not None and args.candidate_limit < 1:
        raise ValueError("candidate lane limit must be positive")
    checked = merged_candidate(base, state)
    issues = frozen.check(root, checked)
    if issues:
        print(json.dumps({"ready": False, "issues": issues}, indent=2))
        return 2
    pending = [lane for lane in checked["lanes"] if lane["state"] == "pending"]
    completed = sum(lane["state"] == "complete" for lane in checked["lanes"])
    skipped = [key(lane) for lane in checked["lanes"] if lane["state"] == "skipped"]
    if args.dry_run:
        print(json.dumps({"ready": True, "candidate_sha256": manifest_sha,
                          "candidate_complete": not pending and not skipped,
                          "completed": completed, "skipped": skipped,
                          "pending": [key(lane) for lane in pending]}, indent=2))
        return 0

    corpus = frozen.inside(root, checked["corpus_manifest"]["path"])
    auditor = frozen.inside(root, checked["audit_tool"]["path"])
    admitted = 0
    local_hash_cache: dict = {}
    for lane in pending:
        if args.candidate_limit is not None and admitted >= args.candidate_limit:
            break
        checked = merged_candidate(base, state)
        issues = frozen.check(root, checked)
        if issues:
            print(json.dumps({"ready": False, "issues": issues}, indent=2))
            return 2
        lane_key = key(lane)
        profile, backend, gpu = PROFILES[lane["profile"]]
        binary = frozen.inside(root, checked["binaries"][lane["binary"]]["path"])
        output = frozen.inside(root, lane["output"])
        if output.suffix != ".json":
            raise ValueError(f"pending lane output must be JSON: {lane_key}")
        audit = output.with_name(output.stem + "-audit.json")
        if audit.exists():
            raise ValueError(f"lane audit output already exists: {audit}")
        observation_paths = {
            phase: output.with_name(output.stem + f"-identity-{phase}.json")
            for phase in ("before", "after")
        }
        if any(path.exists() or path.is_symlink() for path in observation_paths.values()):
            raise ValueError(f"lane identity output already exists: {lane_key}")
        log = attempt_log(args.log_dir, manifest_sha, lane)
        if log.exists():
            raise ValueError(f"lane attempt log already exists: {log}")
        attempt = {"lane": lane_key, "state": "admitting", "log": str(log),
                   "output": lane["output"], "started_utc": utc(),
                   "identity_observations": {}}
        state["attempts"].append(attempt)
        publish(state_path, state)
        try:
            runner = runner_type(log)
            log.touch(exist_ok=True)
            model_receipt = json.loads(frozen.inside(root, lane["model_identity_input"]).read_text())

            def record_identity(phase: str) -> None:
                observed = model_identity.observe(root, lane, model_receipt,
                                                  manifest_sha, phase, local_hash_cache)
                path = observation_paths[phase]
                with path.open("x") as destination:
                    destination.write(json.dumps(observed, indent=2, sort_keys=True) + "\n")
                attempt["identity_observations"][phase] = {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": resume.sha256(path),
                }
                publish(state_path, state)
                if observed.get("ready") is not True:
                    raise ValueError(f"live model identity {phase} lane differs or is unavailable")

            record_identity("before")
            runner.idle(gpu, timeout=900)
            attempt["state"] = "running"
            publish(state_path, state)
            target = next(row for row in checked["targets"] if row["name"] == lane["target"])
            with restored_target_config(target, manifest_sha):
                code = runner.run([
                    binary, "bench", "run", "--corpus", corpus, "--profile",
                    profile, "--only", lane["target"], "--out", output,
                    "--keep", "--json",
                ], timeout=10800, check=False)
            attempt["bench_exit"] = code
            record_identity("after")
            if code != 0:
                raise RuntimeError(f"benchmark exited {code}; report retained")
            audit_code = runner.run([
                sys.executable, auditor, output, "--backend", backend,
                "--out", audit,
            ], timeout=300, check=False)
            attempt["audit_exit"] = audit_code
            if audit_code != 0:
                raise RuntimeError(f"lane auditor exited {audit_code}; report retained")
            audit_data = json.loads(audit.read_text())
            if audit_data.get("accepted") is not True:
                raise ValueError("lane auditor did not accept the retained report")
            audit_name = audit.relative_to(root).as_posix()
            state["completed"][lane_key] = {
                "report_sha256": resume.sha256(output),
                "audit_input": audit_name, "audit_sha256": resume.sha256(audit),
                "release_misses": len(audit_data.get("release_misses", [])),
                "identity_observations": attempt["identity_observations"],
            }
            verified = merged_candidate(base, state)
            issues = frozen.check(root, verified)
            if issues:
                state["completed"].pop(lane_key)
                raise ValueError(f"completed lane failed frozen preflight: {issues}")
            attempt["state"] = "complete"
            attempt["finished_utc"] = utc()
            publish(state_path, state)
            admitted += 1
        except Exception as error:
            attempt["state"] = "failed"
            attempt["error"] = str(error)
            attempt["finished_utc"] = utc()
            publish(state_path, state)
            print(json.dumps({"ready": False, "failed_lane": lane_key,
                              "error": str(error), "state": str(state_path)}))
            return 1
    print(json.dumps({"ready": True, "candidate_sha256": manifest_sha,
                      "candidate_complete": not skipped and len(pending) == admitted,
                      "completed": completed + admitted,
                      "skipped": skipped, "pending": len(pending) - admitted,
                      "state": str(state_path)}))
    return 0
