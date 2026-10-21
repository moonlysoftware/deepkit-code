#!/usr/bin/env python3
"""Run the frozen T4.9 paired agent experiment without replaying uncertain calls."""

from __future__ import annotations

import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def credential() -> str:
    value = os.environ.get("MOONLY_GATEWAY_API_KEY")
    if value:
        return value
    path = Path.home() / ".config/deepkit-code/credentials.toml"
    if path.stat().st_mode & 0o077:
        raise RuntimeError("credential file permissions must be private")
    document = tomllib.loads(path.read_text())
    for row in document.get("credentials", []):
        if row.get("host") == "gateway.moonly.ai" and row.get("api_key"):
            return row["api_key"]
    raise RuntimeError("Moonly credential is unavailable")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the bearer credential through an HTTP redirect."""

    def redirect_request(self, *_args, **_kwargs):
        return None


def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
BASE_TOOLS = [
    function("list_files", "List tracked repository paths matching a glob.", {"glob": STRING}, ["glob"]),
    function(
        "search_text",
        "Search tracked text using a case-insensitive literal and return numbered lines.",
        {"text": STRING},
        ["text"],
    ),
    function(
        "read_file",
        "Read at most 120 numbered lines from a tracked repository file.",
        {"path": STRING, "start": INTEGER},
        ["path", "start"],
    ),
]
DKC_TOOLS = [
    function(
        "dkc_search",
        (
            "Hybrid repository search with the shipped dense, graph and rerank pipeline. "
            "Returns compact evidence IDs and fresh source locations."
        ),
        {"query": STRING},
        ["query"],
    ),
    function(
        "dkc_open",
        "Open one Deepkit Code evidence ID or tracked repository path.",
        {"target": STRING},
        ["target"],
    ),
]


def bounded_string(value: object, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("invalid string")
    if len(value.encode()) > maximum_bytes:
        raise ValueError("oversized string")
    return value


def permitted_file(root: Path, files: set[str], name: object) -> Path:
    name = bounded_string(name, 512)
    path_name = Path(name)
    if name not in files or path_name.is_absolute() or ".." in path_name.parts:
        raise ValueError("only tracked repository files are readable")
    path = root / path_name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("symlink or outside path refused")
    return path


def subprocess_environment(protocol: dict) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
    }
    for name in ["DKC_MODELS_DIR", "CUDA_VISIBLE_DEVICES"]:
        value = protocol.get("dkc_environment", {}).get(name)
        if isinstance(value, str):
            environment[name] = value
    return environment


def execute_tool(
    name: str,
    arguments: dict,
    condition: str,
    root: Path,
    files: set[str],
    protocol: dict,
    deadline: float,
) -> dict:
    if time.monotonic() >= deadline:
        raise TimeoutError("session deadline")
    if name == "list_files":
        pattern = bounded_string(arguments["glob"], 256)
        matches = [path for path in sorted(files) if fnmatch.fnmatchcase(path, pattern)]
        return {"paths": matches[:200], "truncated": len(matches) > 200}
    if name == "read_file":
        path = permitted_file(root, files, arguments["path"])
        start = arguments["start"]
        if type(start) is not int or start < 1 or path.stat().st_size > 1_000_000:
            raise ValueError("invalid range or oversized file")
        lines = path.read_text(errors="replace").splitlines()
        return {
            "path": arguments["path"],
            "lines": [
                f"{index}: {line}"
                for index, line in enumerate(lines, 1)
                if start <= index < start + 120
            ],
        }
    if name == "search_text":
        needle = bounded_string(arguments["text"], 256).casefold()
        matches = []
        for relative in sorted(files):
            if time.monotonic() >= deadline:
                raise TimeoutError("session deadline")
            path = permitted_file(root, files, relative)
            if path.stat().st_size > 1_000_000:
                continue
            raw = path.read_bytes()
            if b"\0" in raw:
                continue
            for index, line in enumerate(raw.decode(errors="replace").splitlines(), 1):
                if needle in line.casefold():
                    matches.append(f"{relative}:{index}:{line[:500]}")
                    if len(matches) == 80:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}
    if condition != "with_dkc":
        raise ValueError("tool is unavailable in this condition")
    if name == "dkc_search":
        command = [
            protocol["binary"],
            "search",
            bounded_string(arguments["query"], 512),
            "--mode",
            "hybrid",
            "-k",
            "5",
            "--json",
        ]
    elif name == "dkc_open":
        target = bounded_string(arguments["target"], 512)
        if target.startswith("-") or Path(target).is_absolute() or ".." in Path(target).parts:
            raise ValueError("invalid open target")
        if "/" in target:
            permitted_file(root, files, target)
        command = [protocol["binary"], "open", target, "--json"]
    else:
        raise ValueError("unknown tool")
    timeout = min(protocol["limits"]["dkc_tool_timeout_seconds"], max(0.001, deadline - time.monotonic()))
    result = subprocess.run(
        command,
        cwd=root,
        env=subprocess_environment(protocol),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        response = {"error": "non_json_dkc_response"}
    return {"exit": result.returncode, "response": response}


def wire_output(value: object) -> str:
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered.encode()) <= 8_000:
        return rendered
    bounded = rendered.encode()[:7_600].decode(errors="ignore")
    return json.dumps({"truncated": True, "text": bounded}, ensure_ascii=False)


class SessionLimit(RuntimeError):
    """A finite experiment outcome that remains in the denominator."""


class RateWindow:
    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []

    def admit(self, tokens: int, protocol: dict, deadline: float) -> None:
        requests = protocol["limits"]["agent_rpm"]
        token_limit = protocol["limits"]["agent_tpm_conservative"]
        if tokens > token_limit:
            raise SessionLimit("one request exceeds the fixed provider token window")
        while True:
            now = time.monotonic()
            self.samples = [(tick, used) for tick, used in self.samples if now - tick < 60]
            if now >= deadline:
                raise SessionLimit("session rate-admission deadline")
            if len(self.samples) < requests and sum(used for _, used in self.samples) + tokens <= token_limit:
                self.samples.append((now, tokens))
                return
            time.sleep(min(1, deadline - now))


def run_session(
    protocol: dict,
    item: dict,
    root: Path,
    files: set[str],
    key: str,
    output: Path,
    rates: RateWindow,
) -> dict:
    started = time.monotonic()
    limits = protocol["limits"]
    result = {
        **item,
        "state": "running",
        "started_utc": utc(),
        "turns": [],
        "tool_calls": 0,
        "admitted_token_bound": 0,
        "agent_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    messages = [
        {"role": "system", "content": protocol["system_prompt"]},
        {"role": "user", "content": item["query"]},
    ]
    tools = BASE_TOOLS + (DKC_TOOLS if item["condition"] == "with_dkc" else [])
    opener = urllib.request.build_opener(NoRedirect)
    deadline = started + limits["session_timeout_seconds"]
    try:
        for turn in range(limits["chat_requests_per_session"]):
            if time.monotonic() >= deadline:
                raise SessionLimit("session wall budget exhausted")
            allow_tools = result["tool_calls"] < limits["force_answer_after_tools"]
            body = {
                "model": protocol["agent"]["model"],
                "messages": messages,
                "temperature": protocol["agent"]["temperature"],
                "max_tokens": limits["max_output_tokens_per_request"],
                "tools": tools,
                "tool_choice": "auto" if allow_tools else "none",
                **protocol["agent"].get("extra_body", {}),
            }
            payload = json.dumps(body, ensure_ascii=False).encode()
            bound = len(payload) + limits["request_token_overhead"] + limits["max_output_tokens_per_request"]
            if result["admitted_token_bound"] + bound > limits["conservative_token_admission_per_session"]:
                raise SessionLimit("session token admission exhausted")
            rates.admit(bound, protocol, deadline)
            result["admitted_token_bound"] += bound
            turn_record = {
                "turn": turn,
                "state": "dispatched",
                "started_utc": utc(),
                "token_bound": bound,
            }
            result["turns"].append(turn_record)
            save(output, result)
            request = urllib.request.Request(
                protocol["agent"]["endpoint"].rstrip("/") + "/chat/completions",
                data=payload,
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                    "User-Agent": protocol["user_agent"],
                },
            )
            request_started = time.monotonic()
            timeout = min(limits["chat_request_timeout_seconds"], max(0.001, deadline - time.monotonic()))
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise RuntimeError("oversized provider response")
            reply = json.loads(raw)
            turn_record.update(
                state="received",
                response=reply,
                wall_seconds=time.monotonic() - request_started,
            )
            usage = reply.get("usage", {})
            for field in result["agent_usage"]:
                value = usage.get(field)
                if type(value) is not int or value < 0:
                    raise RuntimeError("missing or invalid provider usage")
                result["agent_usage"][field] += value
            message = reply["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            messages.append({key: message[key] for key in ["role", "content", "tool_calls"] if key in message})
            save(output, result)
            if not calls:
                answer = message.get("content", "")
                if not answer.strip():
                    raise RuntimeError("empty model answer; no automatic retry")
                result.update(state="answered", answer=answer)
                break
            if not allow_tools:
                raise SessionLimit("model called a tool after the forced-answer boundary")
            for call in calls:
                if result["tool_calls"] >= limits["tools_per_session"]:
                    raise SessionLimit("model exceeded the fixed tool budget")
                result["tool_calls"] += 1
                tick = time.monotonic()
                function_call = call["function"]
                try:
                    value = execute_tool(
                        function_call["name"],
                        json.loads(function_call["arguments"]),
                        item["condition"],
                        root,
                        files,
                        protocol,
                        deadline,
                    )
                except (KeyError, ValueError, OSError, TimeoutError, subprocess.TimeoutExpired) as error:
                    value = {"error": type(error).__name__}
                text = wire_output(value)
                turn_record.setdefault("tools", []).append(
                    {
                        "name": function_call["name"],
                        "arguments": function_call["arguments"],
                        "result": value,
                        "wire_text": text,
                        "wall_seconds": time.monotonic() - tick,
                    }
                )
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                save(output, result)
            minimum_interval = 60.0 / limits["agent_rpm"]
            time.sleep(max(0, minimum_interval - (time.monotonic() - request_started)))
        if result["state"] == "running":
            result.update(state="incomplete", reason="chat turn budget exhausted")
    except SessionLimit as error:
        result.update(state="incomplete", reason=str(error))
    except Exception as error:
        result.update(state="failed", error_type=type(error).__name__)
    finally:
        result.update(finished_utc=utc(), wall_seconds=time.monotonic() - started)
        save(output, result)
    return result


def validate_protocol(protocol: dict) -> None:
    if protocol.get("task") != "T4.9" or protocol.get("state") != "frozen_before_dispatch":
        raise RuntimeError("protocol is not a frozen T4.9 allocation")
    if not re.fullmatch(r"[0-9a-f]{64}", str(protocol.get("initial_database_sha256"))):
        raise RuntimeError("protocol has no initial index fingerprint")
    binary_version = protocol.get("binary_version")
    match = re.fullmatch(
        r"dkc ([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)",
        str(binary_version),
    )
    if match is None or protocol.get("user_agent") != f"dkc/{match.group(1)}":
        raise RuntimeError("protocol User-Agent does not match frozen binary")
    questions = protocol.get("questions", [])
    ids = [question.get("id") for question in questions]
    if len(ids) != len(set(ids)) or sum(question.get("cohort") == "primary" for question in questions) != 30:
        raise RuntimeError("protocol must contain 30 unique primary questions")
    expected = {
        (question["id"], repeat, condition)
        for question in questions
        for repeat in range(1, 4)
        for condition in ["without_dkc", "with_dkc"]
    }
    observed = {
        (item.get("question_id"), item.get("repeat"), item.get("condition"))
        for item in protocol.get("schedule", [])
    }
    if expected != observed or len(protocol["schedule"]) != len(expected):
        raise RuntimeError("schedule must contain exactly three paired repetitions per question")
    for question in questions:
        rubric = question.get("rubric", {})
        if not rubric.get("facts") or not rubric.get("evidence"):
            raise RuntimeError("every question needs a frozen factual and evidence rubric")


def tracked_files(root: Path) -> tuple[set[str], dict[str, str]]:
    raw = subprocess.check_output(["git", "-C", root, "ls-files", "-z"])
    tracked = set(raw.decode().split("\0")) - {""}
    symlinks = {path: os.readlink(root / path) for path in tracked if (root / path).is_symlink()}
    return tracked - symlinks.keys(), symlinks


def source_manifest(root: Path, files: set[str]) -> dict[str, str]:
    return {path: sha256(permitted_file(root, files, path)) for path in sorted(files)}


def usage_reservations(root: Path, database: str) -> dict[str, list]:
    path = root / database
    if not path.is_file():
        return {}
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return {
            str(row[0]): list(row)
            for row in connection.execute(
                "SELECT id,provider,day,calls,tokens,state FROM usage_reservations ORDER BY id"
            )
        }


def verify_initial_database(protocol: dict, root: Path) -> None:
    """Refuse dispatch when the isolated index changed after protocol freeze."""
    path = root / protocol["database"]
    wal = Path(f"{path}-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("frozen ripgrep index has unapplied SQLite WAL pages")
    if sha256(path) != protocol["initial_database_sha256"]:
        raise RuntimeError("frozen ripgrep index changed before first dispatch")


def session_summary(result: dict) -> dict:
    """Return the durable terminal fields retained by the experiment manifest."""
    return {
        name: result[name]
        for name in [
            "id",
            "question_id",
            "cohort",
            "condition",
            "repeat",
            "state",
            "tool_calls",
            "wall_seconds",
            "agent_usage",
        ]
    }


def interrupted_session(path: Path) -> dict:
    """Make an interrupted in-flight session terminal without replaying its call."""
    result = json.loads(path.read_text())
    if result.get("state") == "running":
        result.update(
            state="failed",
            reason="interrupted in-flight session retained without replay",
            interrupted=True,
            finished_utc=utc(),
            wall_seconds=sum(turn.get("wall_seconds", 0) for turn in result.get("turns", [])),
        )
        save(path, result)
    return result


def resume_cooldown(evidence: Path) -> float:
    """Return seconds needed to clear every pre-resume one-minute rate window."""
    latest = None
    for path in evidence.glob("*.json"):
        if path.name in {
            "terminal.json",
            "reservations-before.json",
            "reservations-after.json",
            "reservations-terminal.json",
        }:
            continue
        try:
            session = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for turn in session.get("turns", []):
            value = turn.get("started_utc")
            if not isinstance(value, str):
                continue
            try:
                stamp = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            latest = stamp if latest is None else max(latest, stamp)
    if latest is None:
        return 0.0
    age = (datetime.datetime.now(datetime.timezone.utc) - latest).total_seconds()
    return max(0.0, 60.0 - age)


def main(arguments: list[str]) -> int:
    if len(arguments) == 2 and arguments[1] in {"-h", "--help"}:
        print("usage: token-experiment.py [--resume] PROTOCOL.json")
        return 0
    resume = len(arguments) == 3 and arguments[1] == "--resume"
    if (resume and len(arguments) != 3) or (not resume and len(arguments) != 2):
        print("usage: token-experiment.py [--resume] PROTOCOL.json", file=sys.stderr)
        return 2
    protocol_path = Path(arguments[-1]).resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    root = Path(protocol["root"]).resolve()
    evidence = Path(protocol["evidence"]).resolve()
    terminal_path = evidence / "terminal.json"
    if resume:
        if not terminal_path.is_file():
            raise SystemExit("resume requested without a retained terminal manifest")
        terminal = json.loads(terminal_path.read_text())
        if terminal.get("state") not in {"preflight", "running"}:
            raise SystemExit("only a preflight or running experiment can resume")
        if terminal.get("protocol_sha256") != sha256(protocol_path):
            raise SystemExit("retained protocol differs from the requested protocol")
        terminal["resumed_utc"] = utc()
    else:
        evidence.mkdir(parents=True, exist_ok=False)
        terminal = {
            "state": "preflight",
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "sessions": [],
            "started_utc": utc(),
        }
    try:
        for name, expected in protocol["immutable_sha256"].items():
            if sha256(Path(name)) != expected:
                raise RuntimeError("frozen prerequisite changed")
        if not resume or terminal.get("initial_database_verified") is not True:
            verify_initial_database(protocol, root)
            terminal["initial_database_verified"] = True
        if subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip() != protocol["head"]:
            raise RuntimeError("corpus pin changed")
        files, symlinks = tracked_files(root)
        source_before = source_manifest(root, files)
        source_digest = hashlib.sha256(json.dumps(source_before, sort_keys=True).encode()).hexdigest()
        reservations_path = evidence / "reservations-before.json"
        if resume:
            if source_digest != terminal.get("source_manifest_sha256"):
                raise RuntimeError("tracked source changed before experiment resume")
            if symlinks != terminal.get("excluded_tracked_symlinks"):
                raise RuntimeError("tracked symlink targets changed before experiment resume")
            reservations_before = json.loads(reservations_path.read_text())
        else:
            reservations_before = usage_reservations(root, protocol["database"])
            save(reservations_path, reservations_before)
        terminal.update(
            state="running",
            tracked_files=len(files),
            excluded_tracked_symlinks=symlinks,
            source_manifest_sha256=source_digest,
        )
        save(terminal_path, terminal)
        key = credential()
        if resume:
            time.sleep(resume_cooldown(evidence))
        rates = RateWindow()
        completed = {row["id"] for row in terminal["sessions"]}
        if len(completed) != len(terminal["sessions"]):
            raise RuntimeError("retained terminal manifest contains duplicate sessions")
        for item in protocol["schedule"]:
            if item["id"] in completed:
                retained = json.loads((evidence / f"{item['id']}.json").read_text())
                if retained.get("state") not in {"answered", "incomplete", "failed"}:
                    raise RuntimeError("completed manifest entry is not terminal on disk")
                continue
            session_path = evidence / f"{item['id']}.json"
            if session_path.exists():
                result = interrupted_session(session_path)
                if any(
                    result.get(name) != item.get(name)
                    for name in ["id", "question_id", "cohort", "condition", "repeat"]
                ):
                    raise RuntimeError("retained session identity differs from schedule")
                if result.get("state") not in {"answered", "incomplete", "failed"}:
                    raise RuntimeError("retained session has an unknown state")
                terminal["sessions"].append(session_summary(result))
                completed.add(item["id"])
                save(terminal_path, terminal)
                continue
            print(f"session {item['id']} {item['condition']}", flush=True)
            result = run_session(
                protocol,
                item,
                root,
                files,
                key,
                session_path,
                rates,
            )
            terminal["sessions"].append(session_summary(result))
            completed.add(item["id"])
            save(terminal_path, terminal)
        if symlinks != {path: os.readlink(root / path) for path in symlinks}:
            raise RuntimeError("tracked symlink targets changed")
        if source_before != source_manifest(root, files):
            raise RuntimeError("tracked source files changed")
        reservations_after = usage_reservations(root, protocol["database"])
        save(evidence / "reservations-after.json", reservations_after)
        added = [value for key, value in reservations_after.items() if key not in reservations_before]
        changed = [key for key, value in reservations_before.items() if reservations_after.get(key) != value]
        terminal.update(
            state="collected",
            source_unchanged=True,
            dkc_new_reservations=added,
            dkc_changed_old_reservations=changed,
            failed_sessions=sum(row["state"] == "failed" for row in terminal["sessions"]),
            incomplete_sessions=sum(row["state"] == "incomplete" for row in terminal["sessions"]),
            grading="pending",
            cost_rates=None,
            cost_status="User authorized unpriced testing; retain usage and leave rates empty.",
        )
    except Exception as error:
        terminal.update(state="failed", error_type=type(error).__name__, reason=str(error))
        if root.exists():
            save(
                evidence / "reservations-terminal.json",
                usage_reservations(root, protocol.get("database", ".deepkit/code.db")),
            )
    finally:
        terminal["finished_utc"] = utc()
        save(terminal_path, terminal)
    return 0 if terminal["state"] == "collected" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
