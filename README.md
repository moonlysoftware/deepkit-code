# Deepkit Code

Search your repository's **code, documents and Git history** from the command
line or your coding assistant.

Deepkit Code (`dkc`) connects search results to files, symbols and changes so
your assistant can follow the context. One Rust binary, one local SQLite index,
with CLI and MCP interfaces.

## Install

**Beta for Apple Silicon Macs.** Install with Homebrew:

```sh
brew install moonlysoftware/tap/dkc
```

Already installed? Run `brew update && brew upgrade moonlysoftware/tap/dkc`.
[Build from source](docs/usage.md#build-from-source) for other supported setups.

## Try it

Inside your project's Git repository:

```sh
dkc init --profile local --yes --codex
dkc index --level 1
dkc search "retry policy" --mode lexical --no-rerank
```

This works without a model download or API key. Use `--cursor` or `--claude`
instead of `--codex` for those clients, or omit the flag for CLI use.
Generated MCP configurations use stdio, so projects do not compete for a port.

## What you get

- **Code and documents:** search source files, Markdown, notebooks, PDFs and
  supported Office files; open results at their source locations.
- **Repository context:** navigate symbols and references, inspect changes,
  and search commits and reconstructed pull requests.
- **Optional semantic search:** add local or remote embeddings and reranking.
  Images and scanned pages need a configured vision provider.

Your index stays in `.deepkit/`. Local search needs no external service;
configured remote models receive the content they process. See
[models and privacy](docs/usage.md#privacy-and-configuration).

## Learn more

- [Setup, models and MCP](docs/usage.md)
- [Benchmarks and their limitations](docs/benchmarks.md)
- [Beta status and supported platforms](docs/beta-release.md)

Windows is not supported yet. Full v1 validation is still in progress.
Report problems with your DKC version and a small reproducible example.

Licensed under [MIT OR Apache-2.0](LICENSE.md). Bundled dependencies retain
their [own licenses](THIRD_PARTY_NOTICES.md).
