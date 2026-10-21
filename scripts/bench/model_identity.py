"""Observe model identities at Phase 4 lane admission (v1.md §14).

These checks compare live local bytes or provider deployment identities with a
frozen role receipt. A Moonly deployment ID is not an immutable weights pin.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tomllib
import urllib.request

import candidate_preflight as frozen


SCHEMA = "dkc.phase4.live-model-observation.v1"
MOONLY = "https://gateway.moonly.ai"
OLLAMA = "http://127.0.0.1:11434"
TINY_PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAusB9Y9z7ioAAAAASUVORK5CYII=")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the Moonly bearer token on the intended host."""

    def redirect_request(self, *_args, **_kwargs):
        return None


def credential() -> str:
    """Resolve the gateway secret without writing it into evidence or errors."""
    value = os.environ.get("MOONLY_GATEWAY_API_KEY")
    if value:
        return value
    path = Path.home() / ".config/deepkit-code/credentials.toml"
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("Moonly credential file permissions must be 0600")
    document = tomllib.loads(path.read_text())
    matches = [row.get("api_key") for row in document.get("credentials", [])
               if row.get("host") == "gateway.moonly.ai" and row.get("api_key")]
    if len(matches) != 1:
        raise ValueError("one Moonly gateway credential is required")
    return matches[0]


def request_json(url: str, payload: dict | None, *, token: str | None = None,
                 user_agent: str = "dkc/unknown") -> tuple[dict, object]:
    """Issue a small fixed probe and return its JSON plus response headers."""
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("model identity response exceeds 1 MiB")
        return json.loads(body), response.headers


def moon_credential_probe(role: str, model: str, *, user_agent: str) -> str:
    """Hash the selected deployment header from one successful fixed request."""
    if role in ("embedder", "docs"):
        endpoint = "/v1/embeddings"
        payload = {"model": model, "input": "Deepkit Code identity probe"}
    elif role == "reranker":
        endpoint = "/v1/rerank"
        payload = {"model": model, "query": "identity probe",
                   "documents": ["identity probe"]}
    elif role in ("llm", "vlm"):
        endpoint = "/v1/chat/completions"
        content = "Reply OK." if role == "llm" else [
            {"type": "text", "text": "Reply OK."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}},
        ]
        payload = {"model": model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}
    else:
        raise ValueError("unsupported remote model role")
    document, headers = request_json(MOONLY + endpoint, payload,
                                     token=credential(), user_agent=user_agent)
    if role in ("embedder", "docs"):
        complete = isinstance(document.get("data"), list) and bool(document["data"])
    elif role == "reranker":
        complete = isinstance(document.get("results"), list) and bool(document["results"])
    else:
        choices = document.get("choices")
        complete = isinstance(choices, list) and bool(choices) and bool(
            choices[0].get("message", {}).get("content"))
    selected = headers.get("x-litellm-model-id")
    group = headers.get("x-litellm-model-group")
    if not complete or not selected or group != model:
        raise ValueError("Moonly probe lacks a complete selected-deployment response")
    return hashlib.sha256(selected.encode()).hexdigest()


def ollama_digest(model: str, *, user_agent: str) -> str:
    """Read Ollama's content digest for the exact configured tag."""
    document, _headers = request_json(OLLAMA + "/api/tags", None,
                                      user_agent=user_agent)
    rows = [row for row in document.get("models", [])
            if row.get("name") == model or row.get("model") == model]
    if len(rows) != 1 or not frozen.valid_digest(rows[0].get("digest")):
        raise ValueError("Ollama model tag has no unique content digest")
    return rows[0]["digest"]


def local_digest(evidence: dict, cache: dict, *, cache_root: Path) -> str:
    """Hash changed cache bytes; reuse a full hash only for unchanged inode metadata."""
    basename = evidence.get("cache_file")
    if not isinstance(basename, str) or Path(basename).name != basename or \
            basename in ("", ".", ".."):
        raise ValueError("local evidence has no safe cache filename")
    path = cache_root / basename
    if path.is_symlink() or not path.is_file():
        raise ValueError("local model cache file is missing or linked")
    before = path.stat()
    marker = (before.st_dev, before.st_ino, before.st_size,
              before.st_mtime_ns, before.st_ctime_ns)
    previous = cache.get(str(path))
    if previous is not None and previous[0] == marker:
        return previous[1]
    digest = frozen.sha256(path)
    after = path.stat()
    if marker != (after.st_dev, after.st_ino, after.st_size,
                  after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("local model cache changed while hashing")
    cache[str(path)] = (marker, digest)
    return digest


def observe(root: Path, lane: dict, receipt: dict, candidate_sha256: str,
            phase: str, cache: dict) -> dict:
    """Return a safe, nonsecret receipt even when identity admission fails."""
    if phase not in ("before", "after"):
        raise ValueError("model observation phase must be before or after")
    version = tomllib.loads((root / "Cargo.toml").read_text())["package"]["version"]
    cache_root = Path(os.environ.get("DKC_MODELS_DIR") or
                      Path.home() / ".cache/deepkit-code/models")
    rows = []
    for expected in receipt["identities"]:
        role = expected["role"]
        row = {field: expected[field] for field in ("role", "model", "provider", "kind")}
        row["identity"] = None
        try:
            evidence = json.loads(frozen.inside(root, expected["evidence_input"]).read_text())
            if expected["provider"] == "local":
                actual = local_digest(evidence, cache, cache_root=cache_root)
            elif expected["provider"] == "remote":
                actual = moon_credential_probe(role, expected["model"],
                                               user_agent=f"dkc/{version}")
            elif expected["provider"] == "ollama":
                actual = ollama_digest(expected["model"], user_agent=f"dkc/{version}")
            else:
                raise ValueError("unsupported model provider")
            row["identity"] = actual
            row["matched"] = actual == expected["identity"]
            if not row["matched"]:
                row["issue"] = "identity_changed"
        except Exception:
            row["matched"] = False
            row["issue"] = "identity_unavailable"
        rows.append(row)
    return {"schema": SCHEMA, "candidate_sha256": candidate_sha256,
            "target": lane["target"], "profile": lane["profile"], "phase": phase,
            "observed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ready": bool(rows) and all(row["matched"] for row in rows),
            "identities": rows}
