# Deepkit Code

Help your coding agent find the right code and understand what changed.

Deepkit Code (`dkc`) searches your **code, docs and Git history**. Your agent can
find an implementation, follow references, look up tests and check what changed
before it starts editing. The index stays in your project, ready for the next
session too.

## Install

**Beta for Apple Silicon Macs running macOS 13 or later.** Install with Homebrew:

```sh
brew install moonlysoftware/tap/dkc
```

Already installed? Run `brew update && brew upgrade moonlysoftware/tap/dkc`.
[Build from source](docs/usage.md#build-from-source) for other supported setups.

## Connect your agent

Pick your client and run its commands from your project's Git repository.
Setup uses local defaults for new projects and keeps existing DKC settings.
The first index needs no model download or API key. MCP is how your agent
connects to Deepkit Code's tools.

### Codex

```sh
dkc init --yes --codex
dkc index --level 1
```

Adds the connection to `.codex/config.toml`. Open the project in Codex and make
sure it's [trusted](https://learn.chatgpt.com/docs/extend/mcp), so Codex can load
the project settings.

### Claude Code

```sh
dkc init --yes --claude
dkc index --level 1
```

Adds the connection to `.mcp.json` for Claude Code in this project.

### Cursor

```sh
dkc init --yes --cursor
dkc index --level 1
```

Adds the connection to `.cursor/mcp.json` for Cursor's agent.

After setup, restart or reconnect your client to load the tools. Each connection
uses stdio, so you can keep multiple projects open without competing for a port.

## What can you ask it?

Start with “Use Deepkit Code…” and give your agent a task:

| Ask your agent | What it can look up |
| --- | --- |
| “Where do we retry failed requests?” | Search for `retry`, then open the matching functions and nearby code. |
| “Before changing `calculate_total`, find its references and tests.” | Follow indexed references and test matches to see what may be affected. |
| “What changed in `src/auth` this week?” | Look up recent Git changes in that folder and inspect the relevant commits. |
| “Find our rate-limit docs and compare them with the code.” | Search docs and source together, then open both to check the details. |

Use names and paths from your own repo. Results point back to files, line ranges
and Git records, so your agent can check the details. DKC watches for changes
while connected; run `dkc index --level 1` to refresh the index yourself.
For searches without model downloads, ask for lexical mode and reranking off.

Prefer the terminal? The same lookups are available there:

```sh
dkc search "retry" --mode lexical --no-rerank
dkc refs calculate_total --tests-only
dkc recent --path src/auth --since 7d
```

Want search by meaning too? See [model setup for semantic search and reranking](docs/usage.md#add-models-when-you-need-them).
Your index lives in `.deepkit/`. Local search needs no external service;
configured remote models receive the content they process. See [privacy and configuration](docs/usage.md#privacy-and-configuration).

## Learn more

- [Setup, models and MCP](docs/usage.md)
- [Benchmarks and their limitations](docs/benchmarks.md)
- [Beta status and supported platforms](docs/beta-release.md)

Windows is not supported yet. Full v1 validation is still in progress.
Report problems with your DKC version and a small reproducible example.

Developed and maintained by **[Moonly](https://moonlysoftware.com)**.

Licensed under [MIT OR Apache-2.0](LICENSE.md). Bundled dependencies retain
their [own licenses](THIRD_PARTY_NOTICES.md).
