# Setup and usage

[Deepkit Code](../README.md) · [Documentation](README.md) · [Beta status](beta-release.md)

This beta targets **macOS and Linux**. The source repository is public; the
Homebrew preview still requires private access. Windows remains unsupported.
Models are downloaded separately.

## Private Homebrew preview

The [Moonly tap](https://github.com/moonlysoftware/homebrew-tap) provides an
Apple Silicon build and an x86-64 GNU Linux diagnostic. You need Homebrew,
Python 3.11+, the GitHub CLI, and permission to read both
`moonlysoftware/homebrew-tap` and `moonlysoftware/deepkit-code-history`.
Access to the public source repository alone is not sufficient.

```sh
gh auth login
gh auth setup-git
brew tap moonlysoftware/tap
python3 "$(brew --repository moonlysoftware/tap)/scripts/prepare-private.py"
brew install --formula moonlysoftware/tap/dkc
brew test moonlysoftware/tap/dkc
dkc --version
```

`gh auth setup-git` lets Git use your GitHub CLI credentials for the private tap's
HTTPS clone. If you already use an SSH tap, keep that remote; `brew tap` without
an explicit URL also accepts the existing setup.

The helper verifies the archive, executable, build identity and licenses before
placing the download in Homebrew's cache. The current package is
`0.1.0-beta.1_1`; `dkc --version` reports the upstream version `0.1.0-beta.1`.
Its downloads remain in the history repository and expire on **14 November 2026**.

### Update the Homebrew preview

```sh
brew update
python3 "$(brew --repository moonlysoftware/tap)/scripts/prepare-private.py"
brew upgrade --formula moonlysoftware/tap/dkc
brew test moonlysoftware/tap/dkc
```

Run the helper again if Homebrew's cache has been cleared. For a repair, use
`brew reinstall --formula moonlysoftware/tap/dkc` after preparing the archive.
The tap's [validation record](https://github.com/moonlysoftware/homebrew-tap/blob/main/docs/validation.md)
covers the tested installation and upgrade paths. Public binary releases remain
separate from the public source repository.

## Build from source

You need **Rust 1.98+**, Git, a C/C++ toolchain, CMake and Clang/libclang.
Clone the public repository and install:

```sh
git clone https://github.com/moonlysoftware/deepkit-code.git
cd deepkit-code
cargo install --path . --locked
dkc --version
```

Ensure Cargo's binary directory, usually `~/.cargo/bin`, is on your PATH.
NVIDIA builds need `--features cuda` and the CUDA toolkit. Apple Silicon uses
Metal. Do not use `--all-features`:
GPU backends are platform-specific. [Beta status](beta-release.md) lists checked artifacts.

## First run: no model or API key

Run these commands from your project's Git repository:

```sh
dkc init --profile local --yes
dkc index --level 1
dkc search "retry policy" --mode lexical --no-rerank
dkc status --json
```

Level 1 extracts text, symbols and Git history for local search. It needs no
model. Plain `index` requests level 3 and may run configured model work;
use `--level 1` or `--no-queue` when you want model-free indexing.

The index is `.deepkit/code.db`. Setup writes `.deepkit/config.toml` and an
exclude block in `.git/info/exclude`. `init --gitignore` uses `.gitignore`
instead; linked worktrees currently require that option. Git hooks are opt-in
through `dkc hooks install` and can be removed with `dkc hooks remove`.

## Search and follow the evidence

Replace the sample names and paths with ones in your repository:

```sh
dkc search "retry policy" -k 5 --scope 'src/**' --json
dkc search should_retry --mode lexical --no-rerank
dkc symbol should_retry
dkc open src/retry.rs:1-20 --context 3
dkc refs should_retry --tests-only
dkc map --budget 2000
dkc recent --path src --since 7d --json
dkc recent --symbol should_retry --commits --json
```

Search defaults to `hybrid`; `lexical` and `dense` are also available.
Repeat `--scope` to search a union of path globs. `open` accepts a returned ID
or a path and optional line range. Code ranges expand to their containing
symbol. Pass a returned `continuation_id` to `open` for the next page.
References and graph relationships are approximate. Recent history groups
recognized pull requests, with commits as fallback. `--commits` requests
individual commits. Histories are collated from source records; optional summaries are separate.

JSON responses expose incomplete work. An empty result with `index.partial`
does **not** establish that nothing matches. Changed or unverifiable sources
are suppressed; tools attempt bounded repair when `search.lazy_resync=true`.
If a response remains partial or reports `source_unavailable`, finish
`dkc index --level 1` and retry. After edits, reopen by path if an old ID fails.

## Add models when you need them

For local embeddings and optional local reranking:

```sh
dkc models list --json
dkc models pull qwen3-embedding-0.6b-q8_0
dkc models verify qwen3-embedding-0.6b-q8_0
dkc models pull qwen3-reranker-0.6b-q8_0
dkc index --level 2
dkc search "where do failed requests get another attempt?" --json
```

Downloads resume and verify pinned sizes and SHA-256 hashes. The default cache
is `~/.cache/deepkit-code/models`; `DKC_MODELS_DIR` overrides it. The model
cache cap applies. `models list` shows presence; use `models verify` to check
the complete artifact. `models prune --dry-run` previews unused cache removal.

Select another profile explicitly:

```sh
dkc init --profile moonly
dkc init --profile local --docs moonly
dkc init --profile ollama --url http://localhost:11434 --model qwen3-embedding:0.6b
dkc init --profile remote --url https://provider.example/v1 --model embedding-model
```

Moonly enables remote embedding and enrichment. `--docs moonly` keeps code
embedding local and adds a remote document tier; it also enables enrichment.
Ollama setup neither starts its server nor pulls models. Custom remote setup
accepts `--rerank-model`, `--llm-model` and `--vlm-model`; enrichment requires
both chat and vision models. Remote setup checks advertised model names,
which does not establish inference quality.

## Privacy and configuration

Local embedding and reranking keep that work on your machine. Remote providers
receive the corresponding queries, source text or images. Ollama sends data
to its configured server. Summaries and vision processing use remote services.
Model downloads make network requests even for an otherwise local profile.

Keys come from the configured environment variable first, then
`~/.config/deepkit-code/credentials.toml` (or its `XDG_CONFIG_HOME` location).
Moonly uses `MOONLY_GATEWAY_API_KEY`; the custom remote preset uses
`DKC_REMOTE_API_KEY`. Setup offers a hidden prompt, or `--api-key-stdin` for
one piped key line. Credential files must be private (mode 0600). Keep keys
and HTTP bearer tokens out of repository files and command-line arguments.

Settings override in this order: defaults, user `deepkit-code/config.toml`,
`.deepkit/config.toml`, `.deepkit/config.local.toml`, `DKC_` environment variables,
then supported CLI overrides. Nested environment keys use `__`, for example
`DKC_LIMITS__QUERY_TIMEOUT_MS=5000`. The baseline uses local embedding with
remote enrichment disabled; other configuration layers can override it.

Example repository settings:

```toml
[limits]
db_max_bytes = "2 GiB"
max_file_bytes = "1 MiB"
query_timeout_ms = 5000

[search]
lazy_resync = true

[serve]
agent_writable_config = false
```

Ignore paths with `.gitignore` or `.deepkitignore`. A matching treatment with
`never_send=true` keeps a source at level 1 and excludes it from model work,
while preserving local navigation. Treatment rules use first-match order;
an explicit `treatments` list replaces the inherited list. Review the whole
effective list when adding privacy rules. These rules do not control what
an external coding assistant sends after reading a result.

Optional `[[queue.budgets]]` entries set `profile`, a remote/Ollama base `url`,
and `daily_calls` or `daily_tokens`. Omitted ceilings are unlimited; zero blocks
that work. Embedding budgets reset at UTC midnight and are shared by models
at the same endpoint. They do not impose a general spending cap on every model
role. Inspect queue usage and each role's settings before enabling remote work.

## Connect an MCP client

```sh
dkc init --codex
dkc init --cursor
dkc init --claude
```

These opt-in flags merge an entry into `.codex/config.toml`, `.cursor/mcp.json`
or `.mcp.json`. They preserve other settings and refuse conflicting entries.
For another client,
put this entry inside its `mcpServers` object:

```json
{"deepkit-code": {"command": "dkc", "args": ["serve", "--stdio"]}}
```

Launch DKC in the repository directory, using PATH or an absolute executable path.
Codex setup uses `serve --stdio` and records the repository's absolute working
directory. Each client communicates with its own process without binding a TCP
port, so separate projects can be open together. Codex loads project configuration
only for trusted projects; see its [MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
Multiple clients can also reopen the same existing index. For a repository with
no index yet, finish its first `dkc index --level 1` before starting multiple
clients together. Existing-index startup allows ordinary SQLite sidecar changes
while still rejecting replaced database paths and unsafe files.
The generated `cwd` is machine-specific. After moving or copying a repository,
update that entry's `cwd` to the intended checkout before connecting. An existing
entry pointing elsewhere is reported as a conflict, never silently redirected.
Setup preserves existing TOML bytes and comments; an inline `mcp_servers = {...}`
table may need conversion to regular tables before the entry can be added.
`dkc doctor --json` also checks the project Codex entry and its working directory.
When switching from an existing HTTP setup, remove the old `deepkit-code` HTTP
entry from your user-level Codex configuration before using the project entry.
Codex rejects a stdio entry that also has an inherited HTTP `url`. DKC setup and
doctor inspect the project file; they do not modify or validate Codex's effective
user, profile or managed configuration.

For HTTP, run `dkc serve --http 127.0.0.1:7433`; connect to `http://127.0.0.1:7433/mcp`.
Each HTTP server needs an available address and port. For a second project, for
example, use `--http 127.0.0.1:7434` and configure that project's client to use
`http://127.0.0.1:7434/mcp`. A busy port stops startup before index setup and reports
the occupied address with a suggested remedy; DKC does not silently change the
client's endpoint. Plain `dkc serve` uses stdio, as does `serve --stdio`.
Non-loopback binds require a bearer token from `DKC_SERVE__HTTP_TOKEN`.
MCP configuration tools show effective values and their source.
`serve.agent_writable_config=false` disables agent configuration writes.

`serve` builds an absent index in the background and watches later changes.
Automatic setup requires an ordinary Git checkout; linked-worktree bootstrap
is unsupported. Set `DKC_SERVE__AUTO_INIT=false` to disable automatic setup.
Missing models leave level 1 tools available. `dkc watch` runs the watcher and
queue without MCP. Add `--poll` for filesystems without reliable native events.
Ctrl-C stops the server/watcher; committed index work is retained.

## Queue, maintenance and troubleshooting

```sh
dkc queue status --jobs 20 --json
dkc queue drain --max-passes 4 --json
dkc queue retry --json
dkc doctor --json
```

Indexing and draining use bounded passes without waiting through retry delays.
`pending` means work remains. `quiescent` alone does not prove
complete coverage: also inspect failures and deferrals. Retry preserves usage,
attempt counts and delays. `queue regenerate --job ID` explicitly requeues one
eligible completed text summary while retaining its current published output.

`doctor` checks configuration, index health, caps and models. It does not create
an index or download weights, but configured provider probes can make model
requests using fixed public inputs and recorded usage. A passing model listing
alone does not establish that inference works.

| Symptom | Next step |
| --- | --- |
| Partial or stale results | Run `index --level 1`; inspect pending files and freshness reasons in JSON. |
| Dense/rerank fallback | Check `index.dense`, `index.rerank`, model verification and queue status. |
| Query timeout | Narrow `--scope`, try lexical search without reranking, or raise `limits.query_timeout_ms`. |
| Cap reached | Inspect `status --json`; review ignores or raise the relevant limit, then resume indexing. |
| MCP executable not found | Use an absolute executable path and check the client's PATH and working directory. |
| HTTP address already in use | Use project stdio setup (`init --codex`, `--cursor` or `--claude`), or choose a free HTTP port and update the client's URL. |
| Scanned page deferred | Vision processing needs a configured provider; complex scans also need a PDFium-enabled build and library. |

`index --full` reprocesses level 1; `index --rebuild-fts` rebuilds lexical indexing.
`index --shrink` explicitly evicts generated summaries and vectors toward the
database cap; retained level 1 data can prevent reaching it. `clean` deletes
the admitted database and WAL sidecars, retaining source and configuration.

See [benchmarks](benchmarks.md) and [beta status](beta-release.md). For development, run
`cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`,
`cargo test` and `cargo deny check`; NVIDIA checks also use `--features cuda`.
