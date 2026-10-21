#!/usr/bin/env python3
"""Capture one Phase 4 lane's real configuration and incomplete model template.

The binary reproduces benchmark setup in a disposable repository. This helper
writes its two nonsecret JSON records under notes/bench without overwriting a
previous capture. Model evidence must still be supplied independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LANES = {"l1", "local_cuda", "local_cpu", "moonly", "ollama",
         "dual_local_code_moonly_docs"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path: Path) -> str:
    """Hash a release binary in bounded memory before and after capture."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(binary: Path, corpus: Path, target: str, profile: str) -> dict:
    """Ask the exact binary for a non-indexing configuration capture."""
    if profile not in LANES:
        raise ValueError(f"unknown lane profile: {profile}")
    binary = binary.resolve(strict=True)
    before = sha256(binary)
    result = subprocess.run(
        [str(binary), "bench", "capture-config", "--corpus",
         str(corpus.resolve()), "--target", target, "--profile", profile, "--json"],
        cwd=ROOT, text=True, capture_output=True, timeout=120, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"configuration capture exited {result.returncode}; run the binary directly for diagnostics")
    document = json.loads(result.stdout)
    if not isinstance(document, dict):
        raise ValueError("configuration capture must be an object")
    metadata = subprocess.run(
        [str(binary), "bench", "metadata", "--json"], cwd=ROOT,
        text=True, capture_output=True, timeout=30, check=False,
    )
    if metadata.returncode != 0:
        raise ValueError(f"capture binary metadata exited {metadata.returncode}")
    build = json.loads(metadata.stdout).get("build")
    if not isinstance(build, dict) or sha256(binary) != before:
        raise ValueError("capture binary changed or lacks build metadata")
    config = document.get("effective_config")
    if not isinstance(config, dict):
        raise ValueError("capture lacks effective configuration")
    config["capture_binary_sha256"] = before
    config["capture_build"] = build
    return document


def write(root: Path, prefix: str, document: dict) -> tuple[Path, Path]:
    """Exclusively retain genuine configuration and visibly incomplete model records."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix) is None:
        raise ValueError("receipt prefix must be a safe filename component")
    config = document.get("effective_config")
    models = document.get("model_identity_template")
    if not isinstance(config, dict) or not isinstance(models, dict) or \
            config.get("schema") != "dkc.phase4.effective-config.v1" or \
            models.get("schema") != "dkc.phase4.model-identity.v1" or \
            config.get("target") != models.get("target") or \
            config.get("profile") != models.get("profile") or \
            not isinstance(config.get("target"), str) or not config["target"] or \
            config.get("profile") not in LANES or \
            not isinstance(config.get("effective_config_blake3"), str) or \
            HEX64.fullmatch(config["effective_config_blake3"]) is None or \
            not isinstance(config.get("config_files_blake3"), dict) or \
            "config.toml" not in config["config_files_blake3"] or \
            not set(config["config_files_blake3"]) <= {"config.toml", "config.local.toml"} or \
            not all(isinstance(value, str) and HEX64.fullmatch(value) is not None
                    for value in config["config_files_blake3"].values()) or \
            not isinstance(config.get("starting_config_sha256"), dict) or \
            not set(config["starting_config_sha256"]) <= {"config.toml", "config.local.toml"} or \
            not all(isinstance(value, str) and HEX64.fullmatch(value) is not None
                    for value in config["starting_config_sha256"].values()) or \
            not isinstance(config.get("capture_binary_sha256"), str) or \
            HEX64.fullmatch(config["capture_binary_sha256"]) is None or \
            not isinstance(config.get("capture_build"), dict) or \
            not isinstance(models.get("model_usage"), dict) or \
            not all(field in models["model_usage"] for field in (
                "configured_profile", "configured_embedding", "configured_reranker",
                "docs_tier_configured", "verified_models")) or \
            models.get("identities") != [] or \
            document.get("model_evidence_complete") is not False or \
            not isinstance(document.get("target_head"), str) or \
            re.fullmatch(r"[0-9a-f]{40}", document["target_head"]) is None:
        raise ValueError("capture lacks a valid config or incomplete model template")
    directory = root / "notes/bench"
    if not directory.is_dir():
        raise ValueError("notes/bench is absent")
    config_path = directory / f"{prefix}-config.json"
    model_path = directory / f"{prefix}-models-template.json"
    if config_path.exists() or model_path.exists():
        raise ValueError("receipt output already exists")
    with config_path.open("x") as output:
        json.dump(config, output, indent=2, sort_keys=True)
        output.write("\n")
    try:
        with model_path.open("x") as output:
            json.dump(models, output, indent=2, sort_keys=True)
            output.write("\n")
    except (OSError, ValueError):
        config_path.unlink(missing_ok=True)
        raise
    return config_path, model_path


def main() -> int:
    """Capture one lane and report which of its two receipts still needs evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--corpus", default=ROOT / "bench/corpus.toml", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--profile", choices=sorted(LANES), required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    try:
        document = capture(args.binary, args.corpus, args.target, args.profile)
        config_path, model_path = write(ROOT, args.prefix, document)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"ready": False, "error": str(error)}))
        return 2
    print(json.dumps({"config_ready": True, "model_ready": False,
                      "target_head": document["target_head"],
                      "config": str(config_path.relative_to(ROOT)),
                      "model_template": str(model_path.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
