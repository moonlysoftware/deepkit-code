#!/usr/bin/env python3
"""One bounded public-CLI comparison, ten frozen ripgrep vague questions.
No indexing, copying, configuration-file edits, query retries, or label tuning.
Run only in an idle window with a reviewed immutable binary and prepared index.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import tomllib

PIN = "3fce3b5bb0236da2df6d99672afb8a719642eca7"
GOLDEN_SHA256 = "abeca433c437d9abfa7087045b209c9b1ab88f43047d2b7d1a7ee04da7f6a37f"
ARMS = ("lexical", "hybrid", "reranked")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


def source_identity(root):
    h = hashlib.sha256()
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    for relative in sorted(p for p in paths if p):
        path = root / os.fsdecode(relative)
        if path.is_symlink():
            data = os.readlink(path).encode()
        else:
            data = path.read_bytes()
        h.update(relative + b"\0" + hashlib.sha256(data).digest())
    return {"head": git(root, "rev-parse", "HEAD"), "tracked_bytes_sha256": h.hexdigest(),
            "tracked_dirty": bool(git(root, "status", "--porcelain=v1", "--untracked-files=no"))}


def json_cell(value):
    return {"blob_hex": value.hex()} if isinstance(value, bytes) else value


def db_inventory(root):
    path = root / ".deepkit/code.db"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        metadata = dict(db.execute("SELECT key,value FROM meta WHERE key IN ('schema_version','indexed_head')"))
        tables = [dict(zip(("slug", "model", "profile", "dims"), r)) for r in db.execute(
            "SELECT slug,model,profile,dims FROM embedding_tables WHERE active=1 ORDER BY slug")]
        for row in tables:
            name = "vec_" + row["slug"]
            if name not in names or not name.replace("_", "").isalnum():
                raise ValueError("active vector table missing or invalid")
            row["vectors"] = db.execute('SELECT count(*) FROM "' + name + '"').fetchone()[0]
        selected = {"files", "units", "occurrences", "symbols", "edges", "unit_sources",
                    "unit_contexts", "embedding_publications", "embedding_tables",
                    "embedding_bindings", "embedding_transition"} & names
        selected |= {"vec_" + t["slug"] for t in tables}
        logical_hashes = {}
        for name in sorted(selected):
            safe = '"' + name.replace('"', '""') + '"'
            columns = db.execute("PRAGMA table_info(" + safe + ")").fetchall()
            primary = [r[1] for r in sorted(columns, key=lambda r: r[5]) if r[5]]
            order = ",".join('"' + p.replace('"', '""') + '"' for p in primary) or "rowid"
            h = hashlib.sha256()
            count = 0
            for values in db.execute("SELECT * FROM " + safe + " ORDER BY " + order):
                h.update(json.dumps([json_cell(v) for v in values], ensure_ascii=False,
                                    separators=(",", ":")).encode() + b"\n")
                count += 1
            logical_hashes[name] = {"rows": count, "sha256": h.hexdigest()}
        return {"meta": metadata, "files": dict(zip(("count", "l1_pending", "truncated"),
                db.execute("SELECT count(*),coalesce(sum(l1_pending),0),coalesce(sum(truncated),0) FROM files").fetchone())),
                "active_embeddings": tables, "logical_index": logical_hashes,
                "jobs": dict(db.execute("SELECT state,count(*) FROM jobs GROUP BY state"))}


def configuration(root):
    file = root / ".deepkit/config.toml"
    cfg = tomllib.loads(file.read_text())
    embed = cfg.get("embedder", {})
    rerank = cfg.get("rerank", {})
    public = {"embedder_profile": embed.get("profile"), "embedder_model": embed.get("remote", {}).get("model"),
              "dims": embed.get("dims"), "rerank_provider": rerank.get("provider"),
              "rerank_mode": rerank.get("mode"), "rerank_model": rerank.get("remote", {}).get("model"),
              "rerank_candidates": rerank.get("candidates"),
              "timeouts": {k: cfg.get("limits", {}).get(k) for k in ("query_timeout_ms", "rerank_timeout_ms")}}
    config_files = [file, Path.home() / ".config/deepkit-code/config.toml"]
    # Hash configuration, never copy credentials or arbitrary environment values.
    hashes = {str(p): sha(p.read_bytes()) for p in config_files if p.is_file()}
    overrides = {k: v for k, v in os.environ.items() if k.startswith("DKC_") and
                 not any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))}
    return {"selected_settings": public, "file_sha256": hashes,
            "ambient_dkc_env_sha256": sha(json.dumps(overrides, sort_keys=True).encode()),
            "probe_override": {"DKC_SEARCH__LAZY_RESYNC": "false"}}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("/data/dkc-corpus/ripgrep"))
    ap.add_argument("--golden", type=Path, default=Path(__file__).resolve().parents[2] / "bench/goldens/ripgrep.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seconds", type=int, default=600)
    args = ap.parse_args()
    if not 1 <= args.max_seconds <= 900:
        ap.error("max-seconds must be 1..900")
    args.binary, args.root, args.golden, args.out = (p.resolve() for p in (args.binary, args.root, args.golden, args.out))
    if args.out.exists():
        ap.error("refusing existing output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": "beta_vague_query_public_cli_diagnostic", "state": "preflight", "rows": [],
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "root": str(args.root),
              "binary": str(args.binary), "max_query_processes": 30, "max_seconds": args.max_seconds,
              "method": "10 approved seen vague questions; source scope;k10;one pass per arm;rotating order;no retries",
              "not_claimed": ["fresh holdout", "release timing", "full matrix", "unchanged remote weights", "token savings"],
              "usage_note": "Remote query accounting may change SQLite usage rows. Logical index content must remain unchanged."}
    save(args.out, report)
    started = time.monotonic()
    try:
        import yaml
        raw = args.golden.read_bytes()
        if sha(raw) != GOLDEN_SHA256:
            raise ValueError("golden bytes differ from reviewed frozen input")
        questions = [q for q in yaml.safe_load(raw) if q["kind"] == "vague"]
        if [q["id"] for q in questions] != [f"rg-{n:03d}" for n in range(1, 11)]:
            raise ValueError("vague question cohort differs from frozen ten IDs")
        report["questions"] = [{k: q[k] for k in ("id", "query", "expect_files")} for q in questions]
        report["golden_sha256"] = sha(raw)
        report["binary_sha256"] = sha(args.binary.read_bytes())
        report["before"] = {"source": source_identity(args.root), "config": configuration(args.root),
                            "database": db_inventory(args.root)}
        before = report["before"]
        if before["source"]["head"] != PIN or before["source"]["tracked_dirty"]:
            raise ValueError("source pin or tracked cleanliness failed")
        db = before["database"]
        if db["meta"] != {"schema_version": "23", "indexed_head": PIN} or db["files"]["l1_pending"]:
            raise ValueError("index schema/head/L1 admission failed")
        tables = db["active_embeddings"]
        if len(tables) != 1 or tables[0]["profile"] != "remote" or tables[0]["model"] != "qwen3-embedding-8b" or tables[0]["dims"] != 1024 or not tables[0]["vectors"]:
            raise ValueError("expected prepared remote embedding space is unavailable")
        settings = before["config"]["selected_settings"]
        if settings["embedder_profile"] != "remote" or settings["embedder_model"] != "qwen3-embedding-8b" or settings["rerank_model"] != "qwen3-reranker-4b" or settings["rerank_mode"] == "off":
            raise ValueError("configured model path is not the prepared comparison")
        metadata = subprocess.run([str(args.binary), "bench", "metadata", "--json"], cwd=args.root,
                                  capture_output=True, text=True, timeout=20)
        if metadata.returncode:
            raise ValueError("current binary metadata command failed")
        report["build_metadata"] = json.loads(metadata.stdout)
        env = dict(os.environ, DKC_SEARCH__LAZY_RESYNC="false")
        report["state"] = "running"
        save(args.out, report)
        stop_reason = None
        for number, question in enumerate(questions):
            # First operation is lexical admission; thereafter rotate arm order.
            order = ARMS[number % 3:] + ARMS[:number % 3]
            for arm in order:
                row = {"id": question["id"], "arm": arm, "query": question["query"],
                       "expect_files": question["expect_files"], "hit_file_at5": False,
                       "recall_at10": 0.0, "valid_path": False}
                report["rows"].append(row)
                if stop_reason or time.monotonic() - started >= args.max_seconds:
                    stop_reason = stop_reason or "wall_budget_exhausted"
                    row["state"] = "not_run"
                    row["reason"] = stop_reason
                    continue
                command = [str(args.binary), "search", question["query"], "--scope", "source", "-k", "10",
                           "--mode", "lexical" if arm == "lexical" else "hybrid", "--json", "--verbose"]
                if arm != "reranked":
                    command.append("--no-rerank")
                tick = time.monotonic()
                try:
                    proc = subprocess.run(command, cwd=args.root, env=env, capture_output=True,
                                          text=True, timeout=min(25, max(1, args.max_seconds - (tick - started))))
                    row["wall_ms"] = 1000 * (time.monotonic() - tick)
                    row["exit_code"] = proc.returncode
                    row["stderr_sha256"] = sha(proc.stderr.encode())
                    if proc.returncode:
                        raise ValueError("search_process_failed")
                    response = json.loads(proc.stdout)
                    row["response"] = response
                    index = response.get("index", {})
                    if index.get("partial") or index.get("degraded"):
                        raise ValueError("partial_or_degraded_response")
                    if index.get("head") != PIN or index.get("indexed_head") != PIN:
                        raise ValueError("response_head_mismatch")
                    if index.get("lazy_resync", {}).get("attempted_files", 0):
                        raise ValueError("unexpected_lazy_repair")
                    if arm != "lexical" and index.get("dense") != "complete":
                        raise ValueError("dense_path_not_complete")
                    if arm == "reranked" and index.get("reranked") is not True:
                        raise ValueError("reranker_did_not_run")
                    paths = [hit.get("path") for hit in response.get("hits", [])]
                    expected = set(question["expect_files"])
                    row["hit_file_at5"] = bool(set(paths[:5]) & expected)
                    row["recall_at10"] = len(set(paths[:10]) & expected) / len(expected)
                    row["valid_path"] = True
                    row["state"] = "complete"
                except (ValueError, subprocess.TimeoutExpired) as exc:
                    row["state"] = "failed"
                    row["reason"] = str(exc) if isinstance(exc, ValueError) else "subprocess_timeout"
                    stop_reason = row["reason"]
                save(args.out, report)
        report["after"] = {"source": source_identity(args.root), "config": configuration(args.root),
                           "database": db_inventory(args.root)}
        report["binary_unchanged"] = report["binary_sha256"] == sha(args.binary.read_bytes())
        report["logical_inputs_unchanged"] = before == report["after"]
        report["summary"] = {}
        for arm in ARMS:
            rows = [r for r in report["rows"] if r["arm"] == arm]
            report["summary"][arm] = {"scheduled": 10, "completed": sum(r["valid_path"] for r in rows),
                                      "hit_file_at5": sum(r["hit_file_at5"] for r in rows) / 10,
                                      "recall_at10": sum(r["recall_at10"] for r in rows) / 10,
                                      "failed_or_not_run": sum(not r["valid_path"] for r in rows)}
        report["state"] = "complete" if not stop_reason and report["logical_inputs_unchanged"] and report["binary_unchanged"] else "failed"
        report["stop_reason"] = stop_reason
    except Exception as exc:
        report["state"] = "refused"
        report["preflight_error"] = type(exc).__name__ + ": " + str(exc)
    report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(args.out, report)
    print(json.dumps({"state": report["state"], "output": str(args.out), "summary": report.get("summary")}))
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
