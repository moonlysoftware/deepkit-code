#!/usr/bin/env python3
"""Gate v1 download archive bytes and report uncapped installed executable bytes."""

import argparse
import json
from pathlib import Path
import tarfile
import zipfile

EXPECTED_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-pc-windows-msvc",
)


def binary_members(path: Path):
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and Path(member.name).name == "dkc":
                    yield member.name, member.size
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.is_dir() and Path(member.filename).name == "dkc.exe":
                    yield member.filename, member.file_size


def planned_archives(plan: dict, expected_targets=EXPECTED_TARGETS):
    """Map each release target to the exact executable archive named by dist."""
    archives = {}
    for name, artifact in plan["artifacts"].items():
        if artifact.get("kind") != "executable-zip":
            continue
        targets = artifact.get("target_triples", [])
        if len(targets) != 1 or targets[0] not in expected_targets:
            raise ValueError(f"unexpected executable archive target: {name}: {targets}")
        target = targets[0]
        if target in archives:
            raise ValueError(f"duplicate executable archive for {target}")
        if Path(name).name != name or not (name.endswith(".tar.gz") or name.endswith(".zip")):
            raise ValueError(f"invalid executable archive name: {name}")
        archives[target] = name
    missing = set(expected_targets) - archives.keys()
    if missing:
        raise ValueError(f"dist plan lacks executable archives for: {sorted(missing)}")
    return archives


def audit(directory: Path, maximum_bytes: int, expected_targets=EXPECTED_TARGETS, expected_archives=None):
    reports = []
    errors = []
    planned_names = set(expected_archives.values()) if expected_archives is not None else None
    for path in sorted(directory.rglob("deepkit-code-*")):
        if not (path.name.endswith(".tar.gz") or path.suffix == ".zip"):
            continue
        matches = [target for target in expected_targets if target in path.name]
        if not matches:
            continue
        if planned_names is not None and path.name not in planned_names:
            errors.append(f"unplanned release archive: {path}")
            continue
        if len(matches) != 1:
            errors.append(f"release archive has no unique target: {path}")
            continue
        target = matches[0]
        if expected_archives is not None and path.name != expected_archives[target]:
            errors.append(f"release archive does not match dist plan: {path}")
            continue
        archive_bytes = path.stat().st_size
        for member, size in binary_members(path):
            reports.append(
                {
                    "target": target,
                    "archive": str(path),
                    "member": member,
                    "bytes": size,
                    "archive_bytes": archive_bytes,
                    "size_scope": "download_archive",
                    "maximum_bytes": maximum_bytes,
                    "passed": archive_bytes <= maximum_bytes,
                }
            )
    for target in expected_targets:
        count = sum(report["target"] == target for report in reports)
        if count != 1:
            errors.append(f"{target}: expected one dkc binary, found {count}")
    if any(not report["passed"] for report in reports):
        errors.append("one or more release download archives exceed the size limit")
    return reports, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--maximum-bytes", type=int, default=50_000_000)
    parser.add_argument("--plan-file", type=Path, help="exact cargo-dist plan JSON for this release")
    args = parser.parse_args()
    expected_archives = planned_archives(json.loads(args.plan_file.read_text())) if args.plan_file else None
    reports, errors = audit(args.directory, args.maximum_bytes, expected_archives=expected_archives)
    print(json.dumps({"binaries": reports, "errors": errors}, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
