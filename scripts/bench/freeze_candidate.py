#!/usr/bin/env python3
"""Assemble one immutable Phase 4 matrix candidate; see v1.md §14.

Every receipt is supplied explicitly by the operator. This tool computes file
hashes and target pins, then runs the full static admission check before writing
the manifest. It does not build binaries, fetch models, or run a corpus lane.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import candidate_preflight as frozen


ROOT = Path(__file__).resolve().parents[2]
CORPUS = "bench/corpus.toml"


def safe_name(name: object) -> str:
    """Keep generated report filenames within the frozen evidence directory."""
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError("candidate name must be a safe nonempty filename component")
    return name


def hashed(root: Path, name: str) -> dict[str, str]:
    """Bind one checkout-relative file by its current SHA-256 bytes."""
    path = frozen.inside(root, name)
    if not path.is_file():
        raise ValueError(f"frozen input is missing: {name}")
    return {"path": name, "sha256": frozen.sha256(path)}


def binary(root: Path, name: str) -> dict:
    """Capture exact executable bytes and self-reported release build identity."""
    record = hashed(root, name)
    path = frozen.inside(root, name)
    metadata = json.loads(subprocess.check_output(
        [str(path), "bench", "metadata", "--json"], text=True,
        stderr=subprocess.DEVNULL, timeout=30,
    ))
    build = metadata.get("build")
    if not isinstance(build, dict):
        raise ValueError(f"binary lacks build metadata: {name}")
    if frozen.sha256(path) != record["sha256"]:
        raise ValueError(f"binary changed while reading metadata: {name}")
    return {**record, "build": build}


def corpus_root(corpus_path: Path, row: dict) -> Path:
    """Resolve a target exactly as the corpus fetch tool resolves its roots."""
    if row.get("path"):
        path = Path(os.path.expandvars(row["path"])).expanduser()
        if not path.is_absolute():
            path = corpus_path.parent / path
    else:
        base = Path(os.environ.get("DKC_CORPUS_DIR", "/data/dkc-corpus"))
        path = base / row["name"]
    return path.resolve(strict=True)


def config_baseline(target: Path) -> dict[str, str]:
    """Pin private starting config bytes without copying them into the manifest."""
    result = {}
    for name in ("config.toml", "config.local.toml"):
        path = target / ".deepkit" / name
        if path.is_symlink():
            raise ValueError(f"target has linked {name}: {target}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"target config is not a regular file: {path}")
            result[name] = frozen.sha256(path)
    return result


def assemble(root: Path, spec_path: Path) -> dict:
    """Generate all 81 lanes from the audited plan and explicit per-lane receipts."""
    spec_name = spec_path.resolve().relative_to(root.resolve()).as_posix()
    if not spec_name.startswith("notes/bench/"):
        raise ValueError("candidate input specification must be under notes/bench")
    spec = json.loads(spec_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("candidate input specification must be an object")
    name = safe_name(spec.get("name"))
    binaries = spec.get("binaries")
    if not isinstance(binaries, dict) or set(binaries) != {"cpu", "cuda"}:
        raise ValueError("candidate specification needs CPU and CUDA binaries")
    if not all(isinstance(path, str) and path for path in binaries.values()):
        raise ValueError("candidate needs nonempty CPU and CUDA binary paths")
    builds = {kind: binary(root, binaries[kind]) for kind in ("cpu", "cuda")}
    cpu_build = builds["cpu"]["build"]
    revision = cpu_build.get("commit")
    source_hash = cpu_build.get("source_blake3")
    if not isinstance(revision, str) or not isinstance(source_hash, str):
        raise ValueError("CPU binary lacks product commit or source hash")

    corpus_path = frozen.inside(root, CORPUS)
    corpus = tomllib.loads(corpus_path.read_text())
    rows = corpus.get("repo")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("corpus manifest has no repository table")
    targets = []
    input_names = {CORPUS, frozen.REQUIRED_LANE_PLAN,
                   frozen.REQUIRED_MATRIX_BASELINE, frozen.AUDITOR_PATH, spec_name}
    for row in rows:
        target = corpus_root(corpus_path, row)
        record = {"name": row["name"], "root": str(target),
                  "head": frozen.git(target, "rev-parse", "HEAD"),
                  "config_baseline": config_baseline(target)}
        if row["name"] == "fixtures-docs" and not row.get("rev"):
            record["tree_sha256"] = frozen.tree_sha256(target)
        targets.append(record)
        for field in ("config", "golden"):
            if value := row.get(field):
                input_names.add((Path(CORPUS).parent / value).as_posix())
        for value in row.get("extra_goldens", []):
            input_names.add((Path(CORPUS).parent / value).as_posix())

    plan = json.loads(frozen.inside(root, frozen.REQUIRED_LANE_PLAN).read_text())
    planned = [(lane["target"], lane["lane"]) for lane in plan["lanes"]]
    if len(planned) != len(set(planned)):
        raise ValueError("required lane plan contains duplicate lanes")
    binding_rows = spec.get("bindings")
    if not isinstance(binding_rows, list) or not all(
        isinstance(row, dict) for row in binding_rows
    ):
        raise ValueError("candidate needs explicit per-lane bindings")
    bindings = {}
    for row in binding_rows:
        key = (row.get("target"), row.get("profile"))
        if key in bindings:
            raise ValueError(f"duplicate lane binding: {key}")
        bindings[key] = row
    if set(bindings) != set(planned):
        raise ValueError("candidate bindings must match every required lane exactly")

    lanes = []
    for target, profile in planned:
        binding = bindings[target, profile]
        receipts = {}
        for field in ("effective_config_input", "model_identity_input"):
            value = binding.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{target}:{profile} lacks {field}")
            receipts[field] = value
            input_names.add(value)
        hashed(root, receipts["model_identity_input"])
        model_record = json.loads(frozen.inside(root, receipts["model_identity_input"]).read_text())
        if not isinstance(model_record, dict) or not isinstance(model_record.get("identities"), list):
            raise ValueError(f"{target}:{profile} has no model identity records")
        for identity in model_record["identities"]:
            if not isinstance(identity, dict) or not isinstance(identity.get("evidence_input"), str):
                raise ValueError(f"{target}:{profile} has invalid model evidence binding")
            input_names.add(identity["evidence_input"])
        kind = "cuda" if profile in (
            "local_cuda", "dual_local_code_moonly_docs"
        ) else "cpu"
        lanes.append({"target": target, "profile": profile, "binary": kind,
                      "state": "pending", **receipts,
                      "output": f"notes/bench/{name}-{target}-{profile}.json"})

    inputs = [hashed(root, path) for path in sorted(input_names)]
    records = {record["path"]: record for record in inputs}
    return {
        "schema": frozen.SCHEMA,
        "product": {"commit": revision, "source_blake3": source_hash},
        "host": {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()},
        "corpus_manifest": records[CORPUS],
        "lane_plan": records[frozen.REQUIRED_LANE_PLAN],
        "matrix_baseline": records[frozen.REQUIRED_MATRIX_BASELINE],
        "audit_tool": records[frozen.AUDITOR_PATH],
        "inputs": inputs, "targets": targets, "binaries": builds, "lanes": lanes,
    }


def freeze(root: Path, spec: Path, output: Path) -> dict:
    """Refuse incomplete candidates and publish a fresh manifest exclusively."""
    root = root.resolve()
    output = output.absolute()
    if output.parent.resolve() != (root / "notes/bench").resolve():
        raise ValueError("candidate manifest output must be directly under notes/bench")
    if output.exists() or output.is_symlink():
        raise ValueError(f"candidate manifest already exists: {output}")
    document = assemble(root, spec)
    issues = frozen.check(root, document)
    if issues:
        raise ValueError("candidate preflight refused: " + "; ".join(issues))
    with output.open("x") as target:
        json.dump(document, target, indent=2, sort_keys=True)
        target.write("\n")
    return document


def template(root: Path, name: str, output: Path,
             *, expected_lanes: int = frozen.REQUIRED_LANES) -> dict:
    """Write all planned binding slots as an explicitly incomplete input form."""
    name = safe_name(name)
    root = root.resolve()
    output = output.absolute()
    if output.parent.resolve() != (root / "notes/bench").resolve():
        raise ValueError("candidate input template must be directly under notes/bench")
    plan = json.loads(frozen.inside(root, frozen.REQUIRED_LANE_PLAN).read_text())
    keys = [(row["target"], row["lane"]) for row in plan["lanes"]]
    if len(keys) != expected_lanes or len(set(keys)) != len(keys):
        raise ValueError(f"binding template needs {expected_lanes} unique planned lanes")
    bindings = [
        {"target": row["target"], "profile": row["lane"],
         "effective_config_input": "", "model_identity_input": ""}
        for row in plan["lanes"]
    ]
    document = {"name": name, "binaries": {"cpu": "", "cuda": ""},
                "bindings": bindings}
    with output.open("x") as target:
        json.dump(document, target, indent=2)
        target.write("\n")
    return document


def main() -> int:
    """Require every candidate pin before writing any frozen manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--template", type=Path,
                        help="write the incomplete 81-lane input form")
    parser.add_argument("--name", help="safe candidate name for --template")
    args = parser.parse_args()
    try:
        if args.template is not None:
            if args.spec is not None or args.out is not None or args.name is None:
                parser.error("--template requires --name and excludes --spec/--out")
            document = template(ROOT, args.name, args.template)
            print(json.dumps({"ready": False, "template": str(args.template),
                              "lanes": len(document["bindings"])}))
            return 0
        if args.spec is None or args.out is None or args.name is not None:
            parser.error("freezing requires --spec and --out")
        document = freeze(ROOT, args.spec, args.out)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"ready": False, "error": str(error)}))
        return 2
    print(json.dumps({"ready": True, "manifest": str(args.out),
                      "targets": len(document["targets"]),
                      "lanes": len(document["lanes"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
