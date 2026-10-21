#!/usr/bin/env python3
"""Create the reproducible T4.12 release-readiness receipt.

This audit is deliberately allowed to succeed while the release is not ready.
Use ``--require-ready`` before creating the v1 tag and ``--require-released``
after the release workflow and human release marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "notes/reports/phase-4.md"
V1 = ROOT / "v1.md"
BENCHMARK_PLAN = ROOT / "plan/06-benchmarks.md"
CORPUS_MATRIX_BASELINE = (
    ROOT / "bench/acceptance/corpus-matrix.json"
)
CORPUS_MATRIX_FINAL = ROOT / "notes/bench/t4-12-corpus-evidence-matrix-final.json"
RELEASE_WORKFLOW_FINAL = ROOT / "notes/bench/t4-12-release-workflow-final.json"
CORPUS_MANIFEST = ROOT / "bench/corpus.toml"
BASE_CORPUS_PROFILES = ("local_cuda", "moonly", "local_cpu")
SPECIAL_CORPUS_PROFILES = {
    "ripgrep": ("ollama",),
    "laravel": ("ollama",),
    "mdn-content": ("dual_local_code_moonly_docs",),
}
REQUIRED_CORPUS_METRIC_FIELDS = {
    "l1_cold_ms",
    "incremental_ms",
    "peak_rss_bytes",
    "checkout_text_bytes",
    "database_bytes",
    "footprint_ratio",
    "caps_hit",
    "units_symbols_edges",
    "pr_reconstruction_rate",
    "quality_with_rerank",
    "quality_without_rerank",
    "parse_failure_rates",
    "panics",
    "hostile_outcomes",
    "dead_jobs",
}
RELEASE_TARGETS = {
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-pc-windows-msvc",
}
RELEASE_ARCHIVES = {
    "aarch64-apple-darwin": "deepkit-code-aarch64-apple-darwin.tar.gz",
    "aarch64-unknown-linux-musl": "deepkit-code-aarch64-unknown-linux-musl.tar.gz",
    "x86_64-apple-darwin": "deepkit-code-x86_64-apple-darwin.tar.gz",
    "x86_64-pc-windows-msvc": "deepkit-code-x86_64-pc-windows-msvc.zip",
    "x86_64-unknown-linux-musl": "deepkit-code-x86_64-unknown-linux-musl.tar.gz",
}
RELEASE_CAPTURE_BASE_CHECKS = {
    "workflow_is_release",
    "workflow_completed_successfully",
    "workflow_is_tag_push",
    "workflow_commit_matches_tag",
    "workflow_url_is_https",
    "all_workflow_jobs_terminal_without_failure",
    "release_tag_is_v1_0_0",
    "release_is_published_final",
    "release_url_is_https",
    "unified_checksum_is_release_asset",
    "unified_checksum_downloaded_is_nonempty",
    "unified_checksum_matches_downloads",
    "unified_checksum_url_is_https",
}
RELEASE_CAPTURE_ARTIFACT_CHECK_SUFFIXES = {
    "archive_and_sidecar_are_release_assets",
    "downloaded_archive_is_nonempty",
    "checksum_matches_download",
    "asset_url_is_https",
}
EXPECTED_RELEASE_CAPTURE_CHECKS = RELEASE_CAPTURE_BASE_CHECKS | {
    f"{target}_{suffix}"
    for target in RELEASE_TARGETS
    for suffix in RELEASE_CAPTURE_ARTIFACT_CHECK_SUFFIXES
}
EXPECTED_REPORT_CRITERIA = (
    "Five binaries and installers",
    "All profiles on ripgrep, Laravel; live switch",
    "Moonly preset end to end",
    "mdn/content dual mode",
    "Default reranker and positive delta on four golden repos",
    "Composite reconstruction and newest summaries",
    "Dirty-edit and distant-checkout freshness",
    "Documents and hostile fixtures",
    "Zero setup",
    "Caps and shrink",
    "Zero panics/dead fixture jobs and quality gates",
    "Every §15 performance target",
    "Complete doctor diagnoses",
    "README token number",
)
EXPECTED_V1_CRITERION_SHA256 = (
    "749daabe3920dc32786915b5b8fbaa03026704021e2163949f12d55d9d7e5840",
    "93b74289da290cc2184aee4b19c6e55af55224c4b9b7ee807d005c8378452ff5",
    "370a33238cdc15413e56f1f7800c0ff9f183f9001c6a5af3f466a3add93ce034",
    "faf39d0c4f3ea491e5808906053bccb2df1ee78dad42ba2002cfc87742712a76",
    "bfa6fd219dc3c09dd9aa1c772265fbcc79d85b4db5e84474fd3b7de6dae2b827",
    "2df96b74ff4fb2bd893434e64c65657604a1a574c73544f0053e1106273c418a",
    "b72c49912f66eb9839abb293b3231d75b3a0097c1314440e4f415344f3757cb3",
    "2f1118777b81aa8c3e16ae4039614a8130b896a2291cf15e599a9ed397b5d902",
    "df7486b0e61a63a8f36fb0455809030952c621338773b639cff4016ea802dfdf",
    "3d76bef88ca3af3ab6275e9fcca427ed961393ebd85b2978a80eb98e803bf26a",
    "0b8fdeb38f8a38d4f8b67a21ea5aa44d66ab855f210ed233e6e702b6c1a01ded",
    "557ae12358446c9ed3564b1db3a471c36c2996db1f3c43c490162c0a16b3d8f2",
    "ecc185e7512c166a397ccfc5062b4241a8daea5ffc1df1ae08d7096269b26816",
    "459d7a28ba84afb1954903be0d17c4713b52b8f98a87f0bf73c15014cdbab3d6",
)
EXPECTED_REPORT_METRICS = (
    "ripgrep L1 cold",
    "Aurora-sized L1 cold",
    "Kubernetes L1 cold",
    "20-file incremental",
    "1,000-file branch switch",
    "CUDA L2 throughput",
    "Metal L2 throughput",
    "CPU L2 throughput",
    "Moonly sustained throughput",
    "CUDA hybrid search p95",
    "dual-mode overhead p95",
    "CUDA rerank 50 p95",
    "Metal rerank 50 p95",
    "Moonly rerank 50 p95",
    "map p95",
    "idle RSS after unload",
    "loaded local RSS",
    "L1/L2 footprint",
    "database per corpus repo",
    "300k exact scan",
    "cold zero-setup search",
    "each release download archive",
)
EXPECTED_V1_METRIC_SHA256 = (
    "b9e1241de8404e89a55fc1b688f149052760d398f25c50864f7e579278286b01",
    "19838842faa0e0179adb11d031819ce0d6a3693078ed03d349e820e4c251d322",
    "66e2418eac7f30f4b9ea082c6fd6ff222b06232ca30024ae6b8a19c9827cee6b",
    "6134fb0002a43656b26a0a9ae4f98a98e2ac2e0a57bf8b74b4dc6af93e2ac499",
    "fa853c405bd1805ef24992d504cdc46258f5f29153db6311509c61ef3413ed78",
    "3bb60c3ddc05c6cb0f79de917681d0ca740eb813474e8a46d51a93ee13bed445",
    "c5a3e43bcf93d5b7b51b8f319f5931315afff1729a8dae4783524f7dd94bffd8",
    "84131b117f1be87174ea51286e9d3b023e00dd14b944a5681c3eafb0822ad038",
    "fea52f7827de717e8f812f6cc4151df14633a4d419c3584870c6a2f565dedb43",
    "0bd259b5d70fc0535397cd31cd2fac126a6f3f8781e5e03d7451cc19bd2f8ed8",
    "068bbb529053d423872ab5c808d466c6ef050703a848e877a0b598e0e369a109",
    "b83009cc604f80b389ed5375dfffd59410a6e7ecad3180ea4d3101e38390200c",
    "7727e43e853eab1fcd0961ddcb27b427d696796c4e01f8074ed84d8d4818a887",
    "6f7b334df5764ed00be2e64f79cff099b1b24fbbbfbc30106898a5d58133a553",
    "5e4f7b46488f419358c89d6c6edd6694155a695fec5c288f4e7c102d8806f8df",
    "2b2d69ace96ca370f1d5b095b00bf260c1044ee31f5cede041f6d4adca7917d1",
    "9c7e8fedbfbcf9292b3ed5f89221e97af39b9b21a306ab2dbd559046ab88896a",
    "7ebabbb6a30a7093581d81573e730d4e76edfda080dcdaf964e2003cef1e8776",
    "db6a50a65fa4c4454794aea31892848f963708a772697c85698fa98c28386c75",
    "0fdec7a25838e7ee801e6917829556670439c78a461c5cc9866644de4bbe620c",
    "71b916f98fcbaacd9e9949002a4552a67e3d707a8a34d82f90705ef936a2475b",
    "f949b857347c6701e1ff642f7053ce5eb5e1714ba57ac1ec5255b6aa7544a1a1",
)
EXPECTED_BENCHMARK_REQUIREMENTS_SHA256 = (
    "62902659a18c998a6bea5f0398f616b93cc654261e5bef1705ff4e8c52c72738"
)
EXPECTED_ADDITIONAL_BENCHMARK_LABELS = (
    "Hand-written HitFile@5, Moonly",
    "Hand-written HitFile@5, local",
    "Hand-written vague HitFile@5, Moonly",
    "Hand-written ident HitSymbol@10",
    "Hand-written Recall@10",
    "Hand-written rerank delta",
    "Hand-written hybrid vs lexical",
    "Commit-derived sanity floor",
    "Commit-derived regression",
    "HitPR@3 from PR titles",
    "Compact tokens returned at k=10",
    "Docs subset on FastAPI and MDN",
    "CPU rerank 50 p95",
    "Moonly hybrid + rerank warm p95",
    "`open`, `symbol`, `refs` p95",
    "MCP round-trip overhead",
    "PR reconstruction",
    "Histories for top-K symbols with a touching commit",
    "Module units for qualifying directories",
    "Newest Laravel PR summaries",
    "Dirty-file detection in `watch`",
    "Stale file at query time",
    "Distant checkout line validity",
    "Panics in corpus runs",
    "Dead jobs on fixtures-docs and tiny",
    "Parse failures per language",
    "Hostile files",
    "`never_send`",
    "Agent token number",
)
PLAN_REQUIREMENT_REPORT_MAPPING = (
    ("Hand-written HitFile@5, Moonly",),
    ("Hand-written HitFile@5, local",),
    ("Hand-written vague HitFile@5, Moonly",),
    ("Hand-written ident HitSymbol@10",),
    ("Hand-written Recall@10",),
    ("Hand-written rerank delta",),
    ("Hand-written hybrid vs lexical",),
    ("Commit-derived sanity floor",),
    ("Commit-derived regression",),
    ("HitPR@3 from PR titles",),
    ("Compact tokens returned at k=10",),
    ("Docs subset on FastAPI and MDN",),
    ("CUDA hybrid search p95",),
    ("dual-mode overhead p95",),
    ("CUDA rerank 50 p95",),
    ("CPU rerank 50 p95",),
    ("Moonly rerank 50 p95",),
    ("Moonly hybrid + rerank warm p95",),
    ("map p95",),
    ("`open`, `symbol`, `refs` p95",),
    ("cold zero-setup search",),
    ("MCP round-trip overhead",),
    ("ripgrep L1 cold",),
    ("Aurora-sized L1 cold",),
    ("Kubernetes L1 cold",),
    ("20-file incremental",),
    ("1,000-file branch switch",),
    ("CUDA L2 throughput", "Metal L2 throughput", "CPU L2 throughput"),
    ("Moonly sustained throughput",),
    ("300k exact scan",),
    ("L1/L2 footprint",),
    ("database per corpus repo",),
    ("idle RSS after unload", "loaded local RSS"),
    ("each release download archive",),
    ("PR reconstruction",),
    ("Histories for top-K symbols with a touching commit",),
    ("Module units for qualifying directories",),
    ("Newest Laravel PR summaries",),
    ("Dirty-file detection in `watch`",),
    ("Stale file at query time",),
    ("Distant checkout line validity",),
    ("Panics in corpus runs",),
    ("Dead jobs on fixtures-docs and tiny",),
    ("Parse failures per language",),
    ("Hostile files",),
    ("`never_send`",),
    ("Agent token number",),
)


def run_git(*args: str) -> str:
    """Return bounded Git output from the audited checkout."""
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def markdown_section(text: str, heading: str) -> str:
    """Return one level-two Markdown section without the following section."""
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    body_start = text.find("\n", start)
    if body_start < 0:
        return ""
    end = text.find("\n## ", body_start + 1)
    return text[body_start + 1 :] if end < 0 else text[body_start + 1 : end]


def table_rows(section: str) -> list[list[str]]:
    """Parse ordinary pipe-table data rows from one Markdown section."""
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] in {"Task", "Criterion", "Metric", "Requirement"}:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)
    return rows


def checklist_items(section: str) -> list[str]:
    """Parse wrapped unchecked Markdown checklist entries in source order."""
    items: list[str] = []
    current: list[str] | None = None
    for line in section.splitlines():
        if line.startswith("- [ ] "):
            if current is not None:
                items.append(" ".join(current))
            current = [line[6:].strip()]
        elif current is not None and line.strip() and not line.startswith("---"):
            current.append(line.strip())
    if current is not None:
        items.append(" ".join(current))
    return items


def proposed_pcrs() -> list[dict[str, str | None]]:
    """Return every PCR whose own authoritative status remains proposed."""
    rows: list[dict[str, str | None]] = []
    for path in sorted((ROOT / "notes/pcr").glob("*.md")):
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        status = re.search(r"^\*\*Status:\*\*\s*(.+?)\s*$", text, re.MULTILINE)
        if status is None or status.group(1).strip().lower() != "proposed":
            continue
        identifier = re.match(r"(\d+)-", path.name)
        if identifier is None:
            raise ValueError(f"cannot derive PCR number from {path}")
        blocking = re.search(
            r"^\*\*Blocking:\*\*\s*(.+?)\s*$", text, re.MULTILINE
        )
        rows.append(
            {
                "id": f"PCR-{identifier.group(1)}",
                "path": str(path.relative_to(ROOT)),
                "status": "proposed",
                "blocking": blocking.group(1).strip() if blocking else None,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows


def tracked_product_changes() -> list[dict[str, str]]:
    """Return tracked non-notes changes that prevent a clean release source."""
    changes: list[dict[str, str]] = []
    records = run_git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all"
    ).split("\0")
    skip_rename_source = False
    for line in records:
        if skip_rename_source:
            skip_rename_source = False
            continue
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:]
        if "R" in status or "C" in status:
            skip_rename_source = True
        if path.startswith("notes/"):
            continue
        changes.append({"path": path, "status": status})
    return changes


def corpus_matrix_status(head: str) -> tuple[Path, dict[str, object], dict[str, bool]]:
    """Validate the final corpus matrix, falling back to the explicit baseline."""
    path = CORPUS_MATRIX_FINAL if CORPUS_MATRIX_FINAL.is_file() else CORPUS_MATRIX_BASELINE
    matrix = json.loads(path.read_text())
    targets = matrix.get("targets", [])
    lanes = [lane for target in targets for lane in target.get("lanes", [])]
    required_fields = set(matrix.get("required_metric_fields", []))
    matrix_checks = matrix.get("checks")
    baseline_check_names = {
        "exactly_26_targets",
        "exactly_24_public_pins",
        "names_unique",
        "groups_a11_b10_c4",
        "all_base_profiles_present",
        "special_profiles_exact",
        "exactly_81_required_lanes",
        "all_required_fields_present",
        "all_metric_cells_have_exact_shape",
        "no_final_result_claimed",
    }
    evidence_check_names = baseline_check_names - {"no_final_result_claimed"} | {
        "lane_states_supported",
        "metric_states_supported",
        "metric_state_payloads_well_formed",
    }
    expected_check_names = (
        evidence_check_names if path == CORPUS_MATRIX_FINAL else baseline_check_names
    )
    manifest_raw = CORPUS_MANIFEST.read_bytes()
    manifest = tomllib.loads(manifest_raw.decode("utf-8"))
    expected_targets = []
    for repo in manifest.get("repo", []):
        name = repo["name"]
        expected_targets.append(
            {
                "name": name,
                "url": repo["url"],
                "rev": repo.get("rev", ""),
                "fixture": repo.get("fixture"),
                "groups": repo.get("groups", []),
                "config": repo.get("config"),
                "profiles": [
                    *BASE_CORPUS_PROFILES,
                    *SPECIAL_CORPUS_PROFILES.get(name, ()),
                ],
            }
        )
    actual_targets = [
        {
            "name": target.get("name"),
            "url": target.get("url"),
            "rev": target.get("rev", ""),
            "fixture": target.get("fixture"),
            "groups": target.get("groups", []),
            "config": target.get("config"),
            "profiles": [lane.get("profile") for lane in target.get("lanes", [])],
        }
        for target in targets
    ]

    def retained_evidence(value: object) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        evidence = Path(value)
        if evidence.is_absolute() or not value.startswith("notes/"):
            return False
        resolved = (ROOT / evidence).resolve()
        return resolved.is_relative_to((ROOT / "notes").resolve()) and resolved.is_file()

    digest_cache: dict[str, str] = {}

    def retained_sha256(value: object, claimed: object) -> bool:
        if not retained_evidence(value) or not isinstance(value, str):
            return False
        if re.fullmatch(r"[0-9a-f]{64}", str(claimed)) is None:
            return False
        if value not in digest_cache:
            digest_cache[value] = hashlib.sha256((ROOT / value).read_bytes()).hexdigest()
        return digest_cache[value] == claimed

    lane_by_key = {
        (target.get("name"), lane.get("profile")): lane
        for target in targets
        for lane in target.get("lanes", [])
    }
    fill_history = matrix.get("fill_history")
    history_entries = fill_history if isinstance(fill_history, list) else []
    product_source_revision = matrix.get("product_source_revision")
    product_source_blake3 = matrix.get("product_source_blake3")
    history_keys: list[tuple[object, object]] = []
    history_valid = isinstance(fill_history, list)
    if history_valid:
        for entry in fill_history:
            if not isinstance(entry, dict):
                history_valid = False
                break
            key = (entry.get("target"), entry.get("lane"))
            lane = lane_by_key.get(key)
            if (
                lane is None
                or key in history_keys
                or entry.get("source_revision") != lane.get("source_revision")
                or entry.get("source_revision") != product_source_revision
                or re.fullmatch(r"[0-9a-f]{40}", str(entry.get("build_revision")))
                is None
                or entry.get("source_blake3") != product_source_blake3
                or entry.get("binary_sha256") != lane.get("binary_sha256")
                or entry.get("report") != lane.get("evidence")
                or not retained_sha256(
                    entry.get("report"), entry.get("report_sha256")
                )
                or not retained_sha256(
                    entry.get("build_receipt"),
                    entry.get("build_receipt_sha256"),
                )
            ):
                history_valid = False
                break
            history_keys.append(key)
    if path == CORPUS_MATRIX_BASELINE:
        history_valid = history_valid and not fill_history

    import_metadata = matrix.get("import_manifest")
    if path == CORPUS_MATRIX_BASELINE:
        import_manifest_valid = import_metadata is None
    else:
        import_manifest_valid = isinstance(import_metadata, dict) and retained_sha256(
            import_metadata.get("path"), import_metadata.get("sha256")
        )
        if import_manifest_valid:
            import_document = json.loads((ROOT / import_metadata["path"]).read_text())
            import_rows = import_document.get("lanes", [])
            import_keys = [
                (row.get("target"), row.get("lane"))
                for row in import_rows
                if isinstance(row, dict)
            ]
            history_by_key = {
                (entry.get("target"), entry.get("lane")): entry
                for entry in history_entries
                if isinstance(entry, dict)
            }
            import_manifest_valid = (
                import_document.get("schema")
                == "dkc.t4-12.corpus-evidence-import-manifest.v2"
                and import_document.get("product_source_revision")
                == product_source_revision
                and import_document.get("product_source_blake3")
                == product_source_blake3
                and import_keys == list(lane_by_key)
                and len(import_keys) == 81
                and all(
                    (
                        row.get("report") is None
                        and row.get("build_receipt") is None
                        and (row.get("target"), row.get("lane")) not in history_by_key
                    )
                    or (
                        history_by_key.get(
                            (row.get("target"), row.get("lane")), {}
                        ).get("report")
                        == row.get("report")
                        and history_by_key.get(
                            (row.get("target"), row.get("lane")), {}
                        ).get("build_receipt")
                        == row.get("build_receipt")
                    )
                    for row in import_rows
                )
                and import_metadata.get("required_lanes") == 81
                and import_metadata.get("populated_lanes") == len(history_keys)
                and import_metadata.get("complete") == (len(history_keys) == 81)
            )

    def resolved_metric(cell: object) -> bool:
        if not isinstance(cell, dict):
            return False
        state = cell.get("state")
        if state == "MEASURED":
            return (
                cell.get("value") is not None
                and retained_evidence(cell.get("evidence"))
            )
        if state == "NOT_APPLICABLE":
            return isinstance(cell.get("reason"), str) and bool(cell["reason"].strip())
        return False

    source_revision_valid = (
        isinstance(product_source_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", product_source_revision) is not None
        and isinstance(product_source_blake3, str)
        and re.fullmatch(r"[0-9a-f]{64}", product_source_blake3) is not None
    )
    source_is_ancestor = False
    source_inputs_unchanged = False
    if source_revision_valid:
        source_is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", product_source_revision, head],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode == 0
        source_inputs_unchanged = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                product_source_revision,
                head,
                "--",
                "src",
                "crates",
                "Cargo.toml",
                "Cargo.lock",
                "build.rs",
                ".cargo",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode == 0

    shape = {
        "matrix_audit_valid": matrix.get("audit_valid") is True
        and isinstance(matrix_checks, dict)
        and set(matrix_checks) == expected_check_names
        and all(value is True for value in matrix_checks.values()),
        "matrix_audit_mode_matches_path": matrix.get("audit_mode")
        == ("evidence" if path == CORPUS_MATRIX_FINAL else "baseline"),
        "exactly_26_matrix_targets": len(targets) == 26
        and matrix.get("target_count") == 26,
        "exactly_81_matrix_lanes": len(lanes) == 81
        and matrix.get("required_lane_count") == 81,
        "matrix_target_names_unique": len({target.get("name") for target in targets})
        == 26,
        "matrix_metric_shape_complete": required_fields
        == REQUIRED_CORPUS_METRIC_FIELDS
        and all(set(lane.get("metrics", {})) == required_fields for lane in lanes),
        "matrix_metric_cell_shape_complete": all(
            isinstance(cell, dict)
            and set(cell) == {"state", "value", "evidence", "reason"}
            for lane in lanes
            for cell in lane.get("metrics", {}).values()
        ),
        "matrix_fill_history_valid": history_valid,
        "matrix_import_manifest_valid": import_manifest_valid,
        "matrix_product_source_identity_valid": (
            product_source_revision is None and product_source_blake3 is None
        )
        if path == CORPUS_MATRIX_BASELINE
        else source_revision_valid and source_is_ancestor and source_inputs_unchanged,
        "matrix_contract_matches_current_manifest": matrix.get("schema")
        == "dkc.t4-12.corpus-evidence-matrix.v1"
        and matrix.get("manifest") == str(CORPUS_MANIFEST.relative_to(ROOT))
        and matrix.get("manifest_sha256")
        == hashlib.sha256(manifest_raw).hexdigest()
        and actual_targets == expected_targets,
    }
    final = {
        "all_81_corpus_lanes_pass": len(lanes) == 81
        and all(lane.get("status") == "PASS" for lane in lanes),
        "all_corpus_lanes_use_release_source": len(lanes) == 81
        and source_revision_valid
        and source_is_ancestor
        and source_inputs_unchanged
        and all(
            lane.get("source_revision") == product_source_revision for lane in lanes
        ),
        "all_corpus_lanes_name_binary": len(lanes) == 81
        and all(
            isinstance(lane.get("binary_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", lane["binary_sha256"]) is not None
            for lane in lanes
        ),
        "all_corpus_lanes_name_evidence": len(lanes) == 81
        and all(retained_evidence(lane.get("evidence")) for lane in lanes),
        "all_corpus_metrics_resolved": len(lanes) == 81
        and all(
            resolved_metric(value)
            for lane in lanes
            for value in lane.get("metrics", {}).values()
        ),
        "all_corpus_lanes_have_fill_provenance": len(history_keys) == 81
        and set(history_keys) == set(lane_by_key),
    }
    return path, {"shape": shape, "final": final}, matrix


def release_workflow_status(head: str) -> dict[str, object]:
    """Validate the durable post-tag workflow receipt when it exists."""
    if not RELEASE_WORKFLOW_FINAL.is_file():
        return {"present": False, "pass": False, "path": str(RELEASE_WORKFLOW_FINAL.relative_to(ROOT))}
    receipt = json.loads(RELEASE_WORKFLOW_FINAL.read_text())
    artifacts = receipt.get("artifacts", [])
    artifact_targets = {artifact.get("target") for artifact in artifacts}
    unified_checksum = receipt.get("unified_checksum")
    workflow = receipt.get("workflow")
    workflow_jobs = workflow.get("jobs", []) if isinstance(workflow, dict) else []
    checks = {
        "schema_supported": receipt.get("schema")
        == "dkc.t4-12.release-workflow.v1",
        "captured_live_from_github": receipt.get("capture_mode") == "live_github",
        "all_capture_checks_pass": isinstance(receipt.get("checks"), dict)
        and set(receipt["checks"]) == EXPECTED_RELEASE_CAPTURE_CHECKS
        and all(value is True for value in receipt["checks"].values()),
        "state_passed": receipt.get("state") == "passed",
        "tag_is_v1_0_0": receipt.get("tag") == "v1.0.0",
        "commit_matches_head": receipt.get("commit") == head,
        "workflow_details_match_receipt": isinstance(workflow, dict)
        and isinstance(workflow.get("database_id"), int)
        and workflow["database_id"] > 0
        and workflow.get("name") == "Release"
        and workflow.get("event") == "push"
        and workflow.get("head_branch") == "v1.0.0"
        and workflow.get("head_sha") == head
        and isinstance(workflow.get("attempt"), int)
        and workflow["attempt"] > 0
        and isinstance(workflow.get("url"), str)
        and re.fullmatch(r"https://\S+", workflow["url"]) is not None
        and isinstance(workflow_jobs, list)
        and bool(workflow_jobs)
        and all(
            isinstance(job, dict)
            and isinstance(job.get("name"), str)
            and bool(job["name"].strip())
            and job.get("status") == "completed"
            and job.get("conclusion") in {"success", "skipped"}
            for job in workflow_jobs
        ),
        "release_url_present": isinstance(receipt.get("release_url"), str)
        and re.fullmatch(r"https://\S+", receipt["release_url"]) is not None,
        "five_targets_present": artifact_targets == RELEASE_TARGETS,
        "exact_archive_names_present": len(artifacts) == 5
        and all(
            RELEASE_ARCHIVES.get(artifact.get("target")) == artifact.get("name")
            for artifact in artifacts
        ),
        "artifact_hashes_present": len(artifacts) == 5
        and all(
            isinstance(artifact.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None
            for artifact in artifacts
        ),
        "artifact_names_present": len(artifacts) == 5
        and all(
            isinstance(artifact.get("name"), str) and bool(artifact["name"].strip())
            for artifact in artifacts
        ),
        "artifact_urls_present": len(artifacts) == 5
        and all(
            isinstance(artifact.get("url"), str)
            and re.fullmatch(r"https://\S+", artifact["url"]) is not None
            for artifact in artifacts
        ),
        "artifact_sizes_present": len(artifacts) == 5
        and all(
            isinstance(artifact.get("bytes"), int) and artifact["bytes"] > 0
            for artifact in artifacts
        ),
        "checksum_metadata_present": len(artifacts) == 5
        and all(
            artifact.get("checksum_asset") == artifact.get("name") + ".sha256"
            and isinstance(artifact.get("checksum_url"), str)
            and re.fullmatch(r"https://\S+", artifact["checksum_url"]) is not None
            and isinstance(artifact.get("checksum_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", artifact["checksum_sha256"])
            is not None
            for artifact in artifacts
        ),
        "unified_checksum_metadata_present": isinstance(unified_checksum, dict)
        and unified_checksum.get("name") == "SHA256SUMS"
        and isinstance(unified_checksum.get("url"), str)
        and re.fullmatch(r"https://\S+", unified_checksum["url"]) is not None
        and isinstance(unified_checksum.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", unified_checksum["sha256"]) is not None
        and isinstance(unified_checksum.get("bytes"), int)
        and unified_checksum["bytes"] > 0
        and unified_checksum.get("archive_entries")
        == {
            artifact.get("name"): artifact.get("sha256")
            for artifact in artifacts
        },
    }
    return {
        "present": True,
        "pass": all(checks.values()),
        "path": str(RELEASE_WORKFLOW_FINAL.relative_to(ROOT)),
        "sha256": hashlib.sha256(RELEASE_WORKFLOW_FINAL.read_bytes()).hexdigest(),
        "checks": checks,
    }


def build_receipt() -> dict[str, object]:
    """Inspect the current checkout and assemble the readiness decision."""
    report = REPORT.read_text()
    v1_text = V1.read_text()
    benchmark_plan_text = BENCHMARK_PLAN.read_text()
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    package = cargo["package"]

    task_rows = table_rows(markdown_section(report, "Task ledger"))
    tasks = {row[0].split()[0]: row[1] for row in task_rows if len(row) >= 2}

    acceptance_rows = table_rows(
        markdown_section(report, "v1 §16 acceptance walk-through")
    )
    acceptance = [
        {
            "criterion": row[0],
            "required_proof": row[1],
            "current_evidence": row[2],
            "state": row[3],
        }
        for row in acceptance_rows
        if len(row) >= 4
    ]
    v1_criteria = checklist_items(
        markdown_section(v1_text, "16. Acceptance criteria")
    )
    v1_hashes = [hashlib.sha256(item.encode()).hexdigest() for item in v1_criteria]
    report_labels = [row["criterion"] for row in acceptance]
    acceptance_contract = [
        {
            "position": position,
            "v1_text": source,
            "v1_sha256": source_hash,
            "report_criterion": row["criterion"] if row else None,
            "report_state": row["state"] if row else None,
        }
        for position, (source, source_hash, row) in enumerate(
            zip(
                v1_criteria,
                v1_hashes,
                [*acceptance, *([None] * max(0, len(v1_criteria) - len(acceptance)))],
            ),
            start=1,
        )
    ]
    v1_performance = table_rows(
        markdown_section(v1_text, "15. Performance targets")
    )
    report_performance = table_rows(
        markdown_section(report, "v1 §15 requirements")
    )
    v1_metric_hashes = [
        hashlib.sha256("\n".join(row).encode()).hexdigest()
        for row in v1_performance
    ]
    report_metric_labels = [row[0] for row in report_performance]
    performance_contract = [
        {
            "position": position,
            "v1_metric": source[0],
            "v1_target": source[1],
            "v1_sha256": source_hash,
            "report_metric": row[0] if row else None,
            "report_target": row[1] if row and len(row) > 1 else None,
            "report_value": row[2] if row and len(row) > 2 else None,
            "report_evidence": row[3] if row and len(row) > 3 else None,
            "report_state": row[4] if row and len(row) > 4 else None,
        }
        for position, (source, source_hash, row) in enumerate(
            zip(
                v1_performance,
                v1_metric_hashes,
                [
                    *report_performance,
                    *([None] * max(0, len(v1_performance) - len(report_performance))),
                ],
            ),
            start=1,
        )
    ]
    benchmark_requirements_body = markdown_section(
        benchmark_plan_text, "5. Requirements (release gates)"
    )
    benchmark_requirements_source = (
        "## 5. Requirements (release gates)\n"
        + benchmark_requirements_body.rstrip()
        + "\n"
    )
    benchmark_requirements = table_rows(benchmark_requirements_body)
    additional_benchmark_rows = table_rows(
        markdown_section(report, "Additional `plan/06-benchmarks.md` §5 requirements")
    )
    additional_labels = [row[0] for row in additional_benchmark_rows]
    report_states = {
        **{row[0]: row[4] for row in report_performance if len(row) >= 5},
        **{row[0]: row[3] for row in additional_benchmark_rows if len(row) >= 4},
    }
    benchmark_plan_contract = [
        {
            "position": position,
            "plan_requirement": source[0],
            "plan_target": source[1],
            "report_rows": [
                {"label": label, "state": report_states.get(label)}
                for label in mapping
            ],
        }
        for position, (source, mapping) in enumerate(
            zip(benchmark_requirements, PLAN_REQUIREMENT_REPORT_MAPPING), start=1
        )
    ]

    blockers_section = markdown_section(report, "Known blockers and misses")
    boundaries = sorted(
        set(re.findall(r"^- (Q-\d+):", blockers_section, re.MULTILINE))
    )
    pcrs = proposed_pcrs()
    product_changes = tracked_product_changes()
    tags = sorted(run_git("tag", "--list").splitlines())
    head = run_git("rev-parse", "HEAD").strip()
    matrix_path, matrix_status, _matrix = corpus_matrix_status(head)
    v1_tag_commit = (
        run_git("rev-list", "-n", "1", "v1.0.0").strip()
        if "v1.0.0" in tags
        else None
    )
    workflow_status = release_workflow_status(head)

    review = re.search(r"^\*\*Human review:\*\*\s*(.+?)\s*$", report, re.MULTILINE)
    release = re.search(r"^\*\*Release:\*\*\s*(.+?)\s*$", report, re.MULTILINE)
    review_value = review.group(1).strip().lower() if review else ""
    release_value = release.group(1).strip().lower() if release else ""
    changelog = (ROOT / "CHANGELOG.md").read_text()

    candidate_checks = {
        "all_fourteen_acceptance_rows_pass": len(acceptance) == 14
        and all(
            row["state"].upper() == "PASS"
            and bool(row["required_proof"].strip())
            and bool(row["current_evidence"].strip())
            and row["current_evidence"].strip().upper() != "OPEN"
            for row in acceptance
        ),
        "all_twenty_two_performance_rows_pass": len(report_performance) == 22
        and all(
            len(row) >= 5
            and row[4].upper() == "PASS"
            and bool(row[2].strip())
            and row[2].strip().upper() != "OPEN"
            and bool(row[3].strip())
            for row in report_performance
        ),
        "all_twenty_nine_additional_benchmark_rows_pass": len(additional_benchmark_rows)
        == 29
        and all(
            len(row) >= 4
            and row[3].upper() == "PASS"
            and bool(row[2].strip())
            and row[2].strip().upper() != "OPEN"
            for row in additional_benchmark_rows
        ),
        "final_corpus_matrix_present": matrix_path == CORPUS_MATRIX_FINAL,
        "all_corpus_matrix_lanes_complete": all(matrix_status["shape"].values())
        and all(matrix_status["final"].values()),
        "all_proposed_pcrs_resolved_or_deferred": not pcrs,
        "all_twelve_phase_tasks_complete": len(tasks) == 12
        and all(state.upper() in {"COMPLETE", "DONE", "PASS"} for state in tasks.values()),
        "changelog_has_1_0_0": bool(
            re.search(r"^## \[1\.0\.0\](?:\s|$)", changelog, re.MULTILINE)
        ),
        "crate_publication_enabled": package.get("publish", True) is not False,
        "human_review_marker": review_value == "reviewed",
        "known_blockers_absent": not boundaries,
        "product_source_clean": not product_changes,
        "release_candidate_tag_exists": any(
            re.fullmatch(r"v0\.9\.0-rc(?:\d+)?", tag) for tag in tags
        ),
        "version_is_1_0_0": package["version"] == "1.0.0",
    }
    completion_checks = {
        "human_release_marker": release_value == "released",
        "v1_tag_exists": "v1.0.0" in tags,
        "v1_tag_points_to_audited_head": v1_tag_commit == head,
        "release_workflow_receipt_passes": workflow_status["pass"] is True,
    }
    audit_checks = {
        "twelve_unique_phase_tasks": len(tasks) == 12,
        "task_rows_have_evidence_columns": len(task_rows) == 12
        and all(len(row) >= 3 for row in task_rows),
        "fourteen_unique_report_rows": len(acceptance) == 14
        and len(set(report_labels)) == 14,
        "acceptance_rows_have_proof_and_evidence_columns": len(acceptance_rows) == 14
        and all(len(row) >= 4 for row in acceptance_rows),
        "report_rows_match_reviewed_mapping": tuple(report_labels)
        == EXPECTED_REPORT_CRITERIA,
        "fourteen_v1_checklist_items": len(v1_criteria) == 14,
        "v1_criteria_match_reviewed_source": tuple(v1_hashes)
        == EXPECTED_V1_CRITERION_SHA256,
        "twenty_two_v1_performance_rows": len(v1_performance) == 22,
        "v1_performance_rows_match_reviewed_source": tuple(v1_metric_hashes)
        == EXPECTED_V1_METRIC_SHA256,
        "twenty_two_unique_report_performance_rows": len(report_performance) == 22
        and len(set(report_metric_labels)) == 22,
        "performance_rows_have_value_and_evidence_columns": len(report_performance)
        == 22
        and all(len(row) >= 5 for row in report_performance),
        "report_performance_rows_match_reviewed_mapping": tuple(report_metric_labels)
        == EXPECTED_REPORT_METRICS,
        "forty_seven_benchmark_plan_requirements": len(benchmark_requirements) == 47,
        "benchmark_plan_requirements_match_reviewed_source": hashlib.sha256(
            benchmark_requirements_source.encode()
        ).hexdigest()
        == EXPECTED_BENCHMARK_REQUIREMENTS_SHA256,
        "twenty_nine_unique_additional_benchmark_rows": len(additional_benchmark_rows)
        == 29
        and len(set(additional_labels)) == 29,
        "additional_rows_have_evidence_columns": len(additional_benchmark_rows) == 29
        and all(len(row) >= 4 for row in additional_benchmark_rows),
        "additional_benchmark_rows_match_reviewed_mapping": tuple(additional_labels)
        == EXPECTED_ADDITIONAL_BENCHMARK_LABELS,
        "all_forty_seven_plan_requirements_mapped": len(PLAN_REQUIREMENT_REPORT_MAPPING)
        == 47
        and all(
            label in report_states
            for mapping in PLAN_REQUIREMENT_REPORT_MAPPING
            for label in mapping
        ),
        **matrix_status["shape"],
        "proposed_pcr_ids_unique": len(pcrs) == len({row["id"] for row in pcrs}),
    }
    audit_valid = all(audit_checks.values())
    candidate_ready = audit_valid and all(candidate_checks.values())
    release_complete = candidate_ready and all(completion_checks.values())
    return {
        "schema": "dkc.t4-12.release-readiness.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Readiness audit only; it performs no tag, publication, or release action."
        ),
        "head": head,
        "version": package["version"],
        "publish": package.get("publish", True),
        "tags": tags,
        "phase_task_states": tasks,
        "acceptance_rows": acceptance,
        "acceptance_contract": acceptance_contract,
        "performance_rows": report_performance,
        "performance_contract": performance_contract,
        "additional_benchmark_rows": additional_benchmark_rows,
        "benchmark_plan_contract": benchmark_plan_contract,
        "benchmark_requirements_sha256": hashlib.sha256(
            benchmark_requirements_source.encode()
        ).hexdigest(),
        "corpus_matrix": {
            "path": str(matrix_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            **matrix_status,
        },
        "v1_tag_commit": v1_tag_commit,
        "release_workflow": workflow_status,
        "v1_sha256": hashlib.sha256(v1_text.encode()).hexdigest(),
        "known_open_boundaries": boundaries,
        "open_pcrs": pcrs,
        "tracked_product_changes": product_changes,
        "candidate_checks": candidate_checks,
        "completion_checks": completion_checks,
        "audit_checks": audit_checks,
        "audit_valid": audit_valid,
        "candidate_ready": candidate_ready,
        "release_complete": release_complete,
        "expected_state": "released" if release_complete else (
            "candidate_ready" if candidate_ready else "not_ready"
        ),
    }


def main() -> int:
    """Write one deterministic-shape receipt and enforce readiness on request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--require-ready", action="store_true")
    mode.add_argument("--require-released", action="store_true")
    args = parser.parse_args()

    try:
        receipt = build_receipt()
    except FileNotFoundError:
        # Public beta source excludes development journals. A missing final-v1
        # ledger must fail closed; restoring evidence is not an implicit waiver.
        receipt = {
            "schema": "dkc.t4-12.release-readiness.v2",
            "audit_valid": False,
            "candidate_ready": False,
            "release_complete": False,
            "expected_state": "not_ready",
            "reason": "Required v1 acceptance evidence is absent; restore and review the archived ledger before a stable release.",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_valid": receipt["audit_valid"],
                "candidate_ready": receipt["candidate_ready"],
                "release_complete": receipt["release_complete"],
            }
        )
    )
    if not receipt["audit_valid"]:
        return 2
    if args.require_ready and not receipt["candidate_ready"]:
        return 1
    if args.require_released and not receipt["release_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
