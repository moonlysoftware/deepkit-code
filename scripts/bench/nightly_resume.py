"""Frozen input and report identities for resumable Phase 4 nightlies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

from candidate_preflight import tree_sha256


SCHEMA = "dkc.nightly.resume.v1"


def sha256(path: Path) -> str:
    """Hash a report or input without loading a large artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    """Read a pinned checkout without fetching or changing it."""
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=10,
    ).strip()


def input_snapshot(corpus: Path, model_receipt: Path | None) -> dict:
    """Capture every local corpus input needed to reuse a completed target."""
    manifest = tomllib.loads(corpus.read_text())
    files = {str(corpus.resolve()): sha256(corpus)}
    targets = {}
    corpus_dir = Path(os.environ.get("DKC_CORPUS_DIR", "/data/dkc-corpus"))
    for row in manifest["repo"]:
        name = row["name"]
        for key in ("config", "golden"):
            if value := row.get(key):
                path = (corpus.parent / value).resolve()
                files[str(path)] = sha256(path)
        for value in row.get("extra_goldens", []):
            path = (corpus.parent / value).resolve()
            files[str(path)] = sha256(path)
        if row.get("path"):
            root = Path(row["path"])
            if not root.is_absolute():
                root = corpus.parent / root
        elif row.get("fixture"):
            root = corpus.parent / row["fixture"]
        else:
            root = corpus_dir / name
        root = root.resolve(strict=True)
        if name == "fixtures-docs" and not row.get("rev"):
            source = {"root": str(root), "tree_sha256": tree_sha256(root)}
        else:
            pinned = row.get("rev", "")
            if len(pinned) != 40 or git(root, "rev-parse", "HEAD") != pinned:
                raise ValueError(f"{name}: corpus HEAD is not the pinned revision")
            if git(root, "status", "--porcelain", "--untracked-files=no"):
                raise ValueError(f"{name}: tracked corpus checkout is dirty")
            source = {"root": str(root), "head": pinned}
        targets[name] = source
    receipt = None
    if model_receipt is not None:
        receipt = {"path": str(model_receipt.resolve()), "sha256": sha256(model_receipt)}
    model_root = Path(os.environ.get("DKC_MODELS_DIR", "/data/dkc-models"))
    local_models = {
        str(path.resolve()): sha256(path)
        for path in sorted(model_root.glob("*.gguf"))
        if path.is_file()
    }
    return {
        "files": files,
        "targets": targets,
        "local_models": local_models,
        "model_receipt": receipt,
    }


def read_state(path: Path) -> dict | None:
    """Read a resumable run, refusing malformed state rather than guessing."""
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if state.get("schema") != SCHEMA or not isinstance(state.get("targets"), dict):
        raise ValueError(f"{path}: invalid nightly resume state")
    return state


def write_state(path: Path, state: dict) -> None:
    """Publish a complete progress snapshot after each target boundary."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def report_path(path: str, out_dir: Path) -> Path:
    """Keep a state-supplied report inside the configured evidence directory."""
    candidate = Path(path).resolve()
    if not candidate.is_relative_to(out_dir.resolve()):
        raise ValueError(f"nightly report path escapes {out_dir}: {path}")
    return candidate
