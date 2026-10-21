# Changelog

All notable changes to Deepkit Code are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Preparing `0.1.0-beta.1`, the first Unix beta. Transfer and public distribution
are pending. The full v1 corpus, performance and platform acceptance is separate.

### Beta preparation

- Confine configured database opening and cleanup to admitted paths inside
  `.deepkit`, including symlink and hardlink checks.
- Bound credential reads to regular files no larger than 1 MiB and refuse
  linked or publicly readable credential files on Unix.
- Improve natural-language lexical fallback while preserving query syntax and
  identifiers; retain the measured controls and limitations.
- Update rustls to 0.23.46 to resolve the current dependency advisories.
- Record Apache-licensed sqlite-vector source provenance and ship a verified
  bundle of complete third-party notices with release archives.
- Prepare a checksum-bound Homebrew formula and native install-test workflow.
- Replace development logs in the tracked tree with a compact evidence set,
  a user guide, benchmark methods and a beta handover.

### Added

- One `dkc` binary for repository indexing, CLI search, and MCP over stdio or
  streamable HTTP, backed by one `.deepkit/code.db` SQLite file.
- Lexical, dense, graph-assisted, and reranked search with source opening,
  symbols, approximate references, repository maps, recent pull requests, and
  status/configuration tools.
- Incremental indexing, repository watching, Git `HEAD` reconciliation, visible
  freshness, dirty-file repair, and recoverable branch-switch handling.
- Local, Moonly-compatible remote, Ollama, and opt-in dual document embedding
  profiles with queued work, rate limits, durable usage records, and explicit
  privacy treatments.
- PDF, Office, notebook, image, and scanned-page treatment, plus symbol, pull
  request, module, image, and VLM-page enrichment jobs.
- Pull-request, symbol-history, and module composites with retained source
  provenance.
- Database, provider, model, queue, configuration, IDE, and benchmark-host
  diagnostics through `dkc doctor`.
- Reproducible corpus, regression, freshness, MCP, release, Metal, and paired
  agent-token evaluation tooling.

### Safety and limits

- Repository writes are confined to `.deepkit/` and one managed block in
  `.git/info/exclude` unless another location is explicitly requested.
- Storage and work caps degrade new material to a lower treatment level and are
  visible in status; retained vectors and summaries are removed for space only
  by explicit `dkc index --shrink`.
- `never_send` material stays out of remote embeddings, reranking, summaries,
  captions, and derived outbound context.

[Unreleased]: https://github.com/moonlysoftware/deepkit-code/commits/main
