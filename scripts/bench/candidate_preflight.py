#!/usr/bin/env python3
"""Refuse a Phase 4 corpus matrix unless its frozen candidate still matches disk.

This is a read-only boundary check. The benchmark runner must separately take
the shared slot and perform real idle/device admission before each timed lane.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "dkc.phase4.frozen-candidate.v1"
CONFIG_RECEIPT_SCHEMA = "dkc.phase4.effective-config.v1"
MODEL_RECEIPT_SCHEMA = "dkc.phase4.model-identity.v1"
MODEL_EVIDENCE_SCHEMA = "dkc.phase4.model-evidence.v1"
LIVE_MODEL_SCHEMA = "dkc.phase4.live-model-observation.v1"
EMPTY_GIT_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
REQUIRED_TARGETS = 26
REQUIRED_LANES = 81
REQUIRED_LANE_PLAN = "bench/acceptance/corpus-lanes.json"
REQUIRED_MATRIX_BASELINE = "bench/acceptance/corpus-matrix.json"
AUDITOR_PATH = "scripts/bench/audit_matrix.py"
PRODUCT_PATHS = (
    "Cargo.toml", "Cargo.lock", "build.rs", ".cargo", "src", "crates",
    "tests", "fixtures", "bench", "scripts", "vendor",
)
MODEL_USAGE_FIELDS = (
    "configured_profile", "configured_embedding", "configured_reranker",
    "docs_tier_configured", "verified_models",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def valid_digest(value: object) -> bool:
    """Require a complete lowercase hex digest in nonsecret identity receipts."""
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def lane_receipts(root: Path, lane: dict, input_names: set[str]) -> tuple[dict, dict, list[str]]:
    """Check frozen lane receipts before running or resuming its benchmark."""
    key = (lane.get("target"), lane.get("profile"))
    issues: list[str] = []
    receipts: list[dict] = []
    for binding, schema in (("effective_config_input", CONFIG_RECEIPT_SCHEMA),
                            ("model_identity_input", MODEL_RECEIPT_SCHEMA)):
        name = lane.get(binding)
        if not isinstance(name, str) or name not in input_names:
            issues.append(f"lane lacks frozen {binding}: {key}")
            receipts.append({})
            continue
        try:
            value = json.loads(inside(root, name).read_text())
        except (OSError, ValueError, TypeError) as error:
            issues.append(f"lane {binding} unavailable: {key}: {error}")
            receipts.append({})
            continue
        if not isinstance(value, dict) or value.get("schema") != schema or \
                value.get("target") != key[0] or value.get("profile") != key[1]:
            issues.append(f"lane {binding} identity differs: {key}")
            receipts.append({})
            continue
        receipts.append(value)

    config, models = receipts
    if config:
        files = config.get("config_files_blake3")
        if not valid_digest(config.get("effective_config_blake3")) or \
                not isinstance(files, dict) or "config.toml" not in files or \
                not set(files) <= {"config.toml", "config.local.toml"} or \
                not all(valid_digest(digest) for digest in files.values()):
            issues.append(f"lane effective-config receipt is incomplete: {key}")
    if models:
        usage = models.get("model_usage")
        identities = models.get("identities")
        if not isinstance(usage, dict) or not all(field in usage for field in MODEL_USAGE_FIELDS) or \
                usage.get("configured_profile") not in ("local", "remote", "ollama") or \
                not isinstance(usage.get("configured_embedding"), dict) or \
                not all(isinstance(usage["configured_embedding"].get(name), str) and
                        usage["configured_embedding"][name] for name in ("local", "remote", "ollama")) or \
                not isinstance(usage.get("configured_reranker"), dict) or \
                not all(isinstance(usage["configured_reranker"].get(name), str) and
                        usage["configured_reranker"][name] for name in ("local", "remote")) or \
                not isinstance(usage.get("docs_tier_configured"), bool) or \
                not isinstance(usage.get("verified_models"), dict) or \
                not isinstance(identities, list) or not all(isinstance(row, dict) for row in identities):
            issues.append(f"lane model-identity receipt is incomplete: {key}")
        else:
            verified = usage["verified_models"]
            required: dict[str, tuple[str, str | None]] = {}
            for role, entry, provider_key in (
                ("embedder", verified.get("embedder"), "profile"),
                ("reranker", verified.get("rerank"), "selected_provider"),
                ("docs", verified.get("docs"), "profile"),
            ):
                if entry is None and role == "docs":
                    continue
                if not isinstance(entry, dict) or not isinstance(entry.get("model"), str) or \
                        not entry["model"] or not isinstance(entry.get(provider_key), str) or \
                        entry[provider_key] not in ("local", "remote", "ollama"):
                    issues.append(f"lane {role} model identity is incomplete: {key}")
                    continue
                required[role] = (entry["model"], entry[provider_key])
            llm = verified.get("llm")
            if not isinstance(llm, dict) or not isinstance(llm.get("enabled"), bool):
                issues.append(f"lane LLM model identity is incomplete: {key}")
            elif llm["enabled"]:
                for role, field in (("llm", "model"), ("vlm", "vlm_model")):
                    if role == "vlm" and llm.get(field) is None:
                        continue
                    if not isinstance(llm.get(field), str) or not llm[field]:
                        issues.append(f"lane {role} model identity is incomplete: {key}")
                    else:
                        required[role] = (llm[field], None)
            listed = {row.get("role"): row for row in identities
                      if isinstance(row.get("role"), str)}
            if len(listed) != len(identities) or set(listed) != set(required):
                issues.append(f"lane model evidence roles differ: {key}")
            for role, (model, provider) in required.items():
                row = listed.get(role, {})
                kind = row.get("kind")
                actual_provider = row.get("provider")
                identity = row.get("identity")
                evidence_name = row.get("evidence_input")
                if row.get("model") != model or \
                        actual_provider not in ("local", "remote", "ollama") or \
                        (provider is not None and actual_provider != provider) or \
                        (actual_provider == "local" and kind != "sha256") or \
                        (actual_provider in ("remote", "ollama") and kind != "deployment") or \
                        kind not in ("sha256", "deployment"):
                    issues.append(f"lane {role} model evidence identity differs: {key}")
                    continue
                if (kind == "sha256" and not valid_digest(identity)) or \
                        (kind == "deployment" and (not isinstance(identity, str) or len(identity.strip()) < 8)) or \
                        not isinstance(evidence_name, str) or evidence_name not in input_names:
                    issues.append(f"lane {role} model evidence is not frozen: {key}")
                    continue
                try:
                    evidence = json.loads(inside(root, evidence_name).read_text())
                    if not isinstance(evidence, dict) or \
                            evidence.get("schema") != MODEL_EVIDENCE_SCHEMA or \
                            any(evidence.get(field) != row.get(field) for field in
                                ("role", "model", "provider", "kind", "identity")) or \
                            not isinstance(evidence.get("origin"), str) or \
                            len(evidence["origin"].strip()) < 8:
                        issues.append(f"lane {role} model evidence differs: {key}")
                except (OSError, ValueError, TypeError) as error:
                    issues.append(f"lane {role} model evidence unavailable: {key}: {error}")
    return config, models, issues


def live_observation_issues(root: Path, lane: dict, models: dict, references: object,
                            input_hashes: dict[str, str], *, accepted: bool,
                            candidate_sha256: str | None = None) -> list[str]:
    """Bind retained live observations to the frozen role identities."""
    key = (lane.get("target"), lane.get("profile"))
    output = inside(root, lane["output"])
    if not isinstance(references, dict):
        return [f"lane live model observations are incomplete: {key}"]
    phases = {"before", "after"} if accepted else set(references)
    if set(references) != phases or not set(references) <= {"before", "after"}:
        return [f"lane live model observations are incomplete: {key}"]
    expected = {row["role"]: row for row in models.get("identities", [])}
    issues = []
    for phase, reference in references.items():
        name = output.with_name(output.stem + f"-identity-{phase}.json").relative_to(root).as_posix()
        if not isinstance(reference, dict) or reference.get("path") != name or \
                input_hashes.get(name) != reference.get("sha256"):
            issues.append(f"lane live model {phase} is not frozen: {key}")
            continue
        try:
            observation = json.loads(inside(root, name).read_text())
            observed_rows = observation.get("identities", [])
            observed = {row["role"]: row for row in observed_rows}
            if (observation.get("schema") != LIVE_MODEL_SCHEMA or
                    observation.get("target") != key[0] or
                    observation.get("profile") != key[1] or
                    observation.get("phase") != phase or
                    (candidate_sha256 is not None and
                     observation.get("candidate_sha256") != candidate_sha256) or
                    not isinstance(observed_rows, list) or
                    len(observed) != len(observed_rows) or
                    set(observed) != set(expected) or
                    (accepted and observation.get("ready") is not True)):
                issues.append(f"lane live model {phase} identity differs: {key}")
                continue
            for role, frozen_row in expected.items():
                row = observed[role]
                if any(row.get(field) != frozen_row.get(field) for field in
                       ("role", "model", "provider", "kind")) or \
                        (accepted and (row.get("matched") is not True or
                                       row.get("identity") != frozen_row.get("identity"))) or \
                        (not accepted and row.get("matched") is True and
                         row.get("identity") != frozen_row.get("identity")):
                    issues.append(f"lane live model {phase} role differs: {key}: {role}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            issues.append(f"lane live model {phase} unavailable: {key}: {error}")
    return issues


def sha256(path: Path) -> str:
    """Hash a frozen file without loading a release binary into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root: Path, name: str) -> Path:
    """Resolve a manifest path while rejecting escapes from the checkout."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("path must be a nonempty checkout-relative string")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes checkout: {name}")
    return path


def git(root: Path, *args: str) -> str:
    """Read Git state without fetching or changing a corpus checkout."""
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
        timeout=10,
    ).strip()


def tree_sha256(root: Path) -> str:
    """Hash framed fixture paths and bytes without file-boundary ambiguity."""
    digest = hashlib.sha256()
    digest.update(b"dkc.fixture-tree.v2\0")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".deepkit" in relative.parts or ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"fixture tree contains a symlink: {relative}")
        if not path.is_file():
            continue
        # The hostile corpus includes an invalid-UTF-8 filename. Reversible
        # filesystem encoding preserves its raw bytes while ordinary UTF-8
        # names keep the same hash as earlier candidate manifests.
        raw_name = os.fsencode(relative.as_posix())
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            digest.update(len(raw_name).to_bytes(8, "big"))
            digest.update(raw_name)
            digest.update(size.to_bytes(8, "big"))
            read = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                read += len(block)
            if read != size:
                raise ValueError(f"fixture changed while hashing: {relative}")
    return digest.hexdigest()


def check(
    root: Path, document: dict,
    *, expected_targets: int = REQUIRED_TARGETS,
    expected_lanes: int = REQUIRED_LANES,
    expected_lane_plan: str = REQUIRED_LANE_PLAN,
) -> list[str]:
    """Return every static mismatch that would make a matrix candidate unsafe."""
    issues: list[str] = []

    if not isinstance(document, dict):
        return ["candidate manifest must be a JSON object"]
    for key in ("product", "host", "corpus_manifest", "lane_plan",
                "matrix_baseline", "binaries"):
        if not isinstance(document.get(key), dict):
            return [f"candidate {key} must be an object"]
    for key in ("inputs", "targets", "lanes"):
        if not isinstance(document.get(key), list) or not all(
            isinstance(item, dict) for item in document[key]
        ):
            return [f"candidate {key} must be an array of objects"]

    def require(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    if document.get("schema") != SCHEMA:
        return [f"schema must be {SCHEMA}"]
    product = document.get("product", {})
    revision = product.get("commit")
    source_hash = product.get("source_blake3")
    require(isinstance(revision, str) and len(revision) == 40, "product commit missing")
    require(isinstance(source_hash, str) and len(source_hash) == 64,
            "product source hash missing")
    if issues:
        return issues

    if (root / "notes/STOP").exists():
        issues.append("notes/STOP requests no matrix work")
    try:
        require(git(root, "cat-file", "-t", revision) == "commit",
                "product commit is absent")
        require(not git(root, "diff", "--name-only", revision, "--", *PRODUCT_PATHS),
                "product or benchmark tooling changed since frozen commit")
        require(not git(root, "status", "--porcelain", "--untracked-files=all", "--", *PRODUCT_PATHS),
                "product or benchmark tooling has uncommitted changes")
    except (OSError, subprocess.SubprocessError):
        issues.append("product revision or checkout cannot be verified")

    host = document.get("host", {})
    try:
        require(host.get("boot_id") == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "host boot differs from frozen candidate")
    except OSError:
        issues.append("host boot cannot be read")

    corpus_record = document.get("corpus_manifest", {})
    corpus_name = corpus_record.get("path", "")
    corpus: dict = {}
    try:
        corpus_path = inside(root, corpus_name)
        require(corpus_path.is_file() and sha256(corpus_path) == corpus_record.get("sha256"),
                "corpus manifest hash differs")
        corpus = tomllib.loads(corpus_path.read_text())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        issues.append(f"corpus manifest unavailable: {error}")
        return issues
    targets = corpus.get("repo", [])
    if not isinstance(targets, list) or not all(isinstance(row, dict) for row in targets):
        return issues + ["corpus manifest targets must be an array of tables"]
    require(bool(targets) and len({row.get("name") for row in targets}) == len(targets),
            "corpus manifest has no unique targets")
    require(len(targets) == expected_targets,
            f"corpus target count must be {expected_targets}")

    frozen_inputs = document.get("inputs", [])
    input_names: set[str] = set()
    input_hashes: dict[str, str] = {}
    for item in frozen_inputs:
        name = item.get("path", "")
        try:
            path = inside(root, name)
            require(name not in input_names, f"duplicate frozen input: {name}")
            input_names.add(name)
            input_hashes[name] = item.get("sha256")
            require(path.is_file() and sha256(path) == item.get("sha256"),
                    f"frozen input changed: {name}")
        except (OSError, ValueError) as error:
            issues.append(f"frozen input invalid: {name}: {error}")
    required_inputs = {corpus_name}
    for row in targets:
        for key in ("config", "golden"):
            if row.get(key):
                required_inputs.add(str((Path(corpus_name).parent / row[key]).as_posix()))
        for name in row.get("extra_goldens", []):
            required_inputs.add(str((Path(corpus_name).parent / name).as_posix()))
    require(required_inputs <= input_names,
            "corpus configuration or golden input missing from frozen hashes")
    audit_record = document.get("audit_tool", {})
    if not isinstance(audit_record, dict):
        audit_record = {}
    audit_name = audit_record.get("path", "")
    require(audit_name == AUDITOR_PATH, "wrong frozen lane auditor")
    try:
        audit_path = inside(root, audit_name)
        require(audit_path.is_file() and sha256(audit_path) == audit_record.get("sha256"),
                "lane auditor bytes changed")
        require(audit_name in input_names, "lane auditor is not a frozen input")
    except (OSError, ValueError) as error:
        issues.append(f"lane auditor unavailable: {error}")
    plan_record = document.get("lane_plan", {})
    plan_name = plan_record.get("path", "")
    require(plan_name == expected_lane_plan, "wrong required lane plan")
    planned_lanes: set[tuple[str, str]] = set()
    try:
        plan_path = inside(root, plan_name)
        require(plan_path.is_file() and sha256(plan_path) == plan_record.get("sha256"),
                "required lane plan hash differs")
        require(plan_name in input_names, "required lane plan is not a frozen input")
        planned = json.loads(plan_path.read_text()).get("lanes", [])
        planned_lanes = {(row["target"], row["lane"]) for row in planned}
        require(bool(planned_lanes) and len(planned_lanes) == len(planned),
                "required lane plan has no unique lanes")
        require(len(planned_lanes) == expected_lanes,
                f"required lane count must be {expected_lanes}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(f"required lane plan unavailable: {error}")
    baseline_record = document.get("matrix_baseline", {})
    baseline_name = baseline_record.get("path", "")
    require(baseline_name == REQUIRED_MATRIX_BASELINE if expected_lanes == REQUIRED_LANES
            else bool(baseline_name), "wrong matrix baseline")
    try:
        baseline_path = inside(root, baseline_name)
        require(baseline_path.is_file() and
                sha256(baseline_path) == baseline_record.get("sha256"),
                "matrix baseline hash differs")
        require(baseline_name in input_names, "matrix baseline is not a frozen input")
        baseline = json.loads(baseline_path.read_text())
        baseline_lanes = {
            (target["name"], lane["profile"])
            for target in baseline["targets"] for lane in target["lanes"]
        }
        require(baseline.get("audit_valid") is True and
                baseline.get("target_count") == expected_targets and
                baseline.get("required_lane_count") == expected_lanes and
                baseline_lanes == planned_lanes,
                "required lane plan differs from audited matrix baseline")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        issues.append(f"matrix baseline unavailable: {error}")

    target_records = {row.get("name"): row for row in document.get("targets", [])}
    require(len(target_records) == len(targets), "frozen target list is incomplete or duplicated")
    require(set(target_records) == {row.get("name") for row in targets},
            "frozen targets differ from corpus manifest")
    for row in targets:
        name = row.get("name")
        record = target_records.get(name, {})
        raw_root = record.get("root", "")
        path = Path(raw_root) if isinstance(raw_root, str) else Path()
        if not path.is_absolute() or not path.is_dir():
            issues.append(f"{name}: corpus root absent")
            continue
        baseline = record.get("config_baseline")
        require(isinstance(baseline, dict) and
                set(baseline) <= {"config.toml", "config.local.toml"} and
                all(valid_digest(value) for value in baseline.values()),
                f"{name}: starting config pin is malformed")
        if isinstance(baseline, dict):
            for config_name in ("config.toml", "config.local.toml"):
                config_path = path / ".deepkit" / config_name
                require(not config_path.is_symlink(),
                        f"{name}: linked starting {config_name}")
                if config_name in baseline:
                    require(config_path.is_file() and
                            sha256(config_path) == baseline[config_name],
                            f"{name}: starting {config_name} changed")
                else:
                    require(not config_path.exists(),
                            f"{name}: unexpected starting {config_name}")
        if row.get("path"):
            configured = Path(row["path"])
            if not configured.is_absolute():
                configured = corpus_path.parent / configured
            require(path.resolve() == configured.resolve(),
                    f"{name}: corpus root differs from manifest path")
        pinned = row.get("rev", "")
        if name == "fixtures-docs" and not pinned:
            try:
                require(tree_sha256(path) == record.get("tree_sha256"),
                        f"{name}: generated fixture tree changed")
                head = git(path, "rev-parse", "HEAD")
                require(len(head) == 40 and head == record.get("head"),
                        f"{name}: generated fixture Git HEAD changed")
                require(git(path, "rev-parse", "HEAD^{tree}") == EMPTY_GIT_TREE and
                        git(path, "rev-list", "--count", "HEAD") == "1" and
                        not git(path, "ls-files", "--cached"),
                        f"{name}: generated fixture Git seed is not empty and solitary")
            except (OSError, ValueError) as error:
                issues.append(f"{name}: generated fixture tree unavailable: {error}")
            except subprocess.SubprocessError:
                issues.append(f"{name}: generated fixture Git HEAD unavailable")
            continue
        if not isinstance(pinned, str) or len(pinned) != 40:
            issues.append(f"{name}: corpus revision is not pinned")
            continue
        require(record.get("head") == pinned, f"{name}: frozen head differs from manifest")
        try:
            require(git(path, "rev-parse", "HEAD") == pinned,
                    f"{name}: checkout HEAD moved")
            require(not git(path, "status", "--porcelain", "--untracked-files=no"),
                    f"{name}: tracked checkout is dirty")
        except (OSError, subprocess.SubprocessError):
            issues.append(f"{name}: checkout cannot be verified")

    binaries = document.get("binaries", {})
    require(set(binaries) == {"cpu", "cuda"}, "both CPU and CUDA binaries are required")
    for kind in ("cpu", "cuda"):
        record = binaries.get(kind, {})
        try:
            path = inside(root, record.get("path", ""))
            if not path.is_file():
                issues.append(f"{kind}: binary absent")
                continue
            if sha256(path) != record.get("sha256"):
                issues.append(f"{kind}: binary bytes changed")
                continue
            actual = json.loads(subprocess.check_output(
                [str(path), "bench", "metadata", "--json"], text=True,
                stderr=subprocess.DEVNULL, timeout=30,
            ))["build"]
            require(actual == record.get("build"), f"{kind}: build metadata changed")
            require(actual.get("commit") == revision and
                    actual.get("source_blake3") == source_hash and
                    actual.get("source_dirty") in (False, "false") and
                    actual.get("profile") == "release", f"{kind}: not the clean frozen release source")
            features = {item.strip().upper() for item in str(actual.get("features", "")).split(",")}
            require(features == ({"DEFAULT"} if kind == "cpu" else {"CUDA", "DEFAULT"}),
                    f"{kind}: wrong feature selection")
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            issues.append(f"{kind}: binary metadata unavailable: {error}")

    lanes = document.get("lanes", [])
    lane_keys: set[tuple[str, str]] = set()
    outputs: set[str] = set()
    for lane in lanes:
        key = (lane.get("target", ""), lane.get("profile", ""))
        require(key not in lane_keys, f"duplicate lane: {key}")
        lane_keys.add(key)
        require(key[0] in target_records, f"lane has unknown target: {key[0]}")
        require(lane.get("binary") in ("cpu", "cuda"), f"lane has unknown binary: {key}")
        expected_binary = "cuda" if key[1] in (
            "local_cuda", "dual_local_code_moonly_docs",
        ) else "cpu"
        require(lane.get("binary") == expected_binary,
                f"lane uses wrong binary for profile: {key}")
        config_receipt, model_receipt, receipt_issues = lane_receipts(
            root, lane, input_names,
        )
        issues.extend(receipt_issues)
        require(config_receipt.get("starting_config_sha256") ==
                target_records.get(key[0], {}).get("config_baseline"),
                f"lane starting configuration differs: {key}")
        expected_build = binaries.get(lane.get("binary"), {})
        require(config_receipt.get("capture_binary_sha256") ==
                expected_build.get("sha256") and
                config_receipt.get("capture_build") == expected_build.get("build"),
                f"lane capture binary differs: {key}")
        name = lane.get("output", "")
        try:
            path = inside(root, name)
            require(path.is_relative_to((root / "notes/bench").resolve()),
                    f"lane output outside notes/bench: {name}")
            require(name not in outputs, f"duplicate lane output: {name}")
            outputs.add(name)
            state = lane.get("state")
            require(state in ("pending", "complete", "skipped"),
                    f"lane has invalid state: {key}")
            if state == "pending":
                require(not path.exists(), f"pending lane output already exists: {name}")
            elif state == "complete":
                if not path.is_file() or sha256(path) != lane.get("report_sha256"):
                    issues.append(f"completed lane report bytes changed: {key}")
                    continue
                try:
                    opener = gzip.open if path.suffix == ".gz" else open
                    with opener(path, "rt") as source:
                        report = json.load(source)
                    expected_profile = {
                        "local_cuda": "local", "local_cpu": "cpu",
                        "dual_local_code_moonly_docs": "dual",
                    }.get(key[1], key[1])
                    require(report.get("profile") == expected_profile and
                            report.get("build") == binaries[lane["binary"]]["build"] and
                            not report.get("failures") and
                            [row.get("name") for row in report.get("targets", [])] == [key[0]],
                            f"completed lane report identity differs: {key}")
                    rows = report.get("targets")
                    target_report = rows[0] if isinstance(rows, list) and len(rows) == 1 \
                        and isinstance(rows[0], dict) else {}
                    provenance = target_report.get("provenance", {})
                    if not isinstance(provenance, dict):
                        provenance = {}
                    require(provenance.get("head") == target_records[key[0]].get("head") and
                            provenance.get("effective_config_blake3") ==
                            config_receipt.get("effective_config_blake3") and
                            provenance.get("config_files_blake3") ==
                            config_receipt.get("config_files_blake3"),
                            f"completed lane effective configuration differs: {key}")
                    observed_usage = provenance.get("model_usage", {})
                    expected_usage = model_receipt.get("model_usage", {})
                    require(isinstance(observed_usage, dict) and
                            all(observed_usage.get(field) == expected_usage.get(field)
                                and field in expected_usage for field in MODEL_USAGE_FIELDS),
                            f"completed lane model usage differs: {key}")
                    require(valid_digest(lane.get("candidate_sha256")),
                            f"completed lane candidate identity missing: {key}")
                    issues.extend(live_observation_issues(
                        root, lane, model_receipt, lane.get("identity_observations"),
                        input_hashes, accepted=True,
                        candidate_sha256=lane.get("candidate_sha256"),
                    ))
                    audit_name = lane.get("audit_input")
                    require(audit_name in input_names,
                            f"completed lane lacks frozen audit: {key}")
                    if audit_name in input_names:
                        audit = json.loads(inside(root, audit_name).read_text())
                        require(audit.get("accepted") is True and
                                audit.get("build") == report.get("build") and
                                any(row.get("name") == key[0] and row.get("accepted") is True
                                    for row in audit.get("targets", [])),
                                f"completed lane audit does not accept target: {key}")
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    issues.append(f"completed lane evidence unavailable: {key}: {error}")
            elif state == "skipped":
                skip_name = lane.get("skip_input")
                expected_skip = path.with_name(path.stem + "-skip.json").relative_to(root).as_posix()
                require(skip_name == expected_skip and skip_name in input_names,
                        f"skipped lane lacks frozen disposition: {key}")
                if skip_name in input_names:
                    try:
                        skip = json.loads(inside(root, skip_name).read_text())
                        require(skip.get("schema") == "dkc.phase4.matrix-lane-skip.v1"
                                and isinstance(lane.get("candidate_sha256"), str)
                                and re.fullmatch(r"[0-9a-f]{64}", lane["candidate_sha256"]) is not None
                                and skip.get("candidate_sha256") == lane.get("candidate_sha256")
                                and skip.get("target") == key[0]
                                and skip.get("profile") == key[1]
                                and skip.get("output") == name
                                and isinstance(skip.get("reason"), str)
                                and len(skip["reason"].strip()) >= 12,
                                f"skipped lane disposition identity differs: {key}")
                        for field in ("evidence", "attempt_log"):
                            reference = skip.get(field, {})
                            require(isinstance(reference, dict)
                                    and reference.get("path") in input_names
                                    and input_hashes.get(reference.get("path"))
                                    == reference.get("sha256"),
                                    f"skipped lane {field} is not frozen: {key}")
                        issues.extend(live_observation_issues(
                            root, lane, model_receipt,
                            skip.get("identity_observations", {}), input_hashes,
                            accepted=False,
                            candidate_sha256=lane.get("candidate_sha256"),
                        ))
                        rejected_audit = skip.get("rejected_audit")
                        audit_path = path.with_name(path.stem + "-audit.json")
                        if rejected_audit is not None:
                            require(isinstance(rejected_audit, dict)
                                    and rejected_audit.get("path")
                                    == audit_path.relative_to(root).as_posix()
                                    and rejected_audit.get("path") in input_names
                                    and input_hashes.get(rejected_audit.get("path"))
                                    == rejected_audit.get("sha256"),
                                    f"skipped lane rejected audit is not frozen: {key}")
                        else:
                            require(not audit_path.exists(),
                                    f"skipped lane has unpinned rejected audit: {key}")
                        report_sha = skip.get("report_sha256")
                        require(lane.get("report_sha256") == report_sha,
                                f"skipped lane report pin differs: {key}")
                        require((path.is_file() and sha256(path) == report_sha)
                                if report_sha is not None else not path.exists(),
                                f"skipped lane report bytes changed: {key}")
                    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                        issues.append(f"skipped lane evidence unavailable: {key}: {error}")
        except ValueError as error:
            issues.append(f"lane output invalid: {error}")
    require(bool(lanes), "candidate has no lanes")
    audit_outputs = {
        str(Path(name).with_name(Path(name).stem + "-audit.json"))
        for name in outputs if Path(name).suffix == ".json"
    }
    require(not outputs.intersection(audit_outputs),
            "lane report output collides with another lane audit")
    skip_outputs = {
        str(Path(name).with_name(Path(name).stem + "-skip.json"))
        for name in outputs if Path(name).suffix == ".json"
    }
    require(not outputs.intersection(skip_outputs),
            "lane report output collides with another lane skip receipt")
    require(lane_keys == planned_lanes, "candidate lanes differ from required lane plan")
    return issues


def main() -> int:
    """Print a machine-readable refusal or a successful static preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        issues = check(args.repo_root.resolve(), json.loads(args.manifest.read_text()))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        issues = [f"candidate manifest unavailable: {error}"]
    ready = not issues
    print(json.dumps({"ready": ready, "issues": issues}, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    sys.exit(main())
