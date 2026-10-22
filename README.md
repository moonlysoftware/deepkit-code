# Deepkit Code

Help your coding agent find the right code and understand what changed.

Built and maintained by **[Moonly](https://moonlysoftware.com)**, Deepkit Code
(`dkc`) searches your **code, docs and Git history**. Your agent can find an
implementation, follow references, look up tests and check what changed before
it starts editing. The index stays in your project, ready for the next session.

## Install

**Beta for Apple Silicon Macs running macOS 13 or later.** Install with Homebrew:

```sh
brew install moonlysoftware/tap/dkc
```

Already installed? Run `brew update && brew upgrade moonlysoftware/tap/dkc`.
[Build from source](docs/usage.md#build-from-source) for other supported setups.

## Choose where the models run

The examples use **Moonly Gateway**, where Moonly runs the models for you.
Prefer your own hardware? Choose the local profile with Metal or CUDA.

Embeddings turn code, documents and search questions into numbers that represent
their meaning. DKC compares them to find related content even when the words
differ, alongside ordinary keyword search.

| Option | Embeddings with the default models | What you need |
| --- | --- | --- |
| **Moonly Gateway** | Qwen3 Embedding 8B runs on Moonly's servers. | A Moonly API key and internet. No local model download or GPU. |
| **Metal — local Mac** | Qwen3 Embedding 0.6B runs on your Apple Silicon GPU. | An Apple Silicon Mac and local model downloads. Metal is included in the Mac build. |
| **CUDA — local NVIDIA** | The same local 0.6B model runs on your NVIDIA GPU. | Linux, a compatible NVIDIA GPU, CUDA toolkit and a [source build](docs/usage.md#build-from-source) with `--features cuda`. Models download locally. |

Local CPU execution is supported too.

All three store the index and embeddings locally in `.deepkit/code.db`.
Gateway receives the content and queries it processes, and also handles
reranking, summaries and image descriptions. Local setup keeps embedding and
reranking on your machine, with summaries and image descriptions off by default.

**For Moonly Gateway, [request an API key through Moonly](https://moonlysoftware.com).**
Setup asks for it privately and saves it outside your repository. You can also
provide it through the `MOONLY_GATEWAY_API_KEY` environment variable.

## Connect your agent

Pick your client and run its commands from your project's Git repository.
These examples select Moonly Gateway, including for an already configured
project. MCP is how your agent connects to Deepkit Code's tools.

### Codex

```sh
dkc init --profile moonly --codex
dkc index
```

Adds the connection to `.codex/config.toml`. Open the project in Codex and make
sure it's [trusted](https://learn.chatgpt.com/docs/extend/mcp), so Codex can load
the project settings.

### Claude Code

```sh
dkc init --profile moonly --claude
dkc index
```

Adds the connection to `.mcp.json` for Claude Code in this project.

### Cursor

```sh
dkc init --profile moonly --cursor
dkc index
```

Adds the connection to `.cursor/mcp.json` for Cursor's agent.

After setup, restart or reconnect your client to load the tools. Each connection
uses stdio, so you can keep multiple projects open without competing for a port.

### What do those commands mean?

- `dkc init` sets up DKC for this project.
- `--profile moonly` chooses Moonly Gateway. Use `--profile local` for Metal or
  CUDA, without a Moonly API key.
- `--codex`, `--claude` or `--cursor` adds the connection for that client.
- `dkc index` reads your files and Git history, builds the search index, creates
  embeddings and runs enabled AI summaries and descriptions.

For a lighter run, `--level 1` indexes text, symbols and history without model
calls. `--level 2` adds embeddings. Plain `dkc index` uses level 3 and also runs
enabled summaries and descriptions. These levels control that command;
the connected agent's server can continue configured model work afterward.

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
# --mode lexical searches keywords; --no-rerank skips the extra model ranking step.
dkc search "retry" --mode lexical --no-rerank

# --tests-only shows references to calculate_total found in test files.
dkc refs calculate_total --tests-only

# --path chooses a folder; --since 7d means the last seven days.
dkc recent --path src/auth --since 7d
```

Reranking is the extra model step that sorts the best matches by relevance;
`--no-rerank` skips it. [Model setup](docs/usage.md#add-models-when-you-need-them)
covers local downloads, Ollama and other providers. [Privacy and configuration](docs/usage.md#privacy-and-configuration)
explains what each provider receives.

## Learn more

- [Setup, models and MCP](docs/usage.md)
- [Benchmarks and their limitations](docs/benchmarks.md)
- [Beta status and supported platforms](docs/beta-release.md)

Windows is not supported yet. Full v1 validation is still in progress.
Report problems with your DKC version and a small reproducible example.

Licensed under [MIT OR Apache-2.0](LICENSE.md). Bundled dependencies retain
their [own licenses](THIRD_PARTY_NOTICES.md).
