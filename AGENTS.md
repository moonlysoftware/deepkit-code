# Deepkit Code — contributor and coding-agent guide

Deepkit Code (`dkc`) indexes repository content and Git history in one SQLite
file and exposes search through CLI and MCP. Start with README.md, docs/usage.md,
notes/STATE.md and notes/QUESTIONS.md. Read docs/beta-audit.md and
docs/beta-release.md before changing release behavior.

## Product boundaries

- Keep one `dkc` crate and `sqlite-vector-sys`. Propose architectural changes
  before adding crates, data stores or services.
- Store the index in `.deepkit/code.db`: exact FLOAT16 vector search and FTS5.
  No ANN index, daemon, REST API or metrics endpoint.
- Preserve CLI JSON and MCP stdio/streamable HTTP interfaces.
- Preserve the five L3 jobs: describe_symbol, summarize_pr, summarize_module,
  describe_image and vlm_page. Histories are collated source records.
- Enforce limits and expose partial/degraded work. Do not delete retained work
  for space without an explicit shrink operation.
- Repository writes stay in `.deepkit/` and one managed Git exclude block unless
  the user explicitly requests another setup action. Hooks remain opt-in.
- Credentials belong in environment variables or private user configuration,
  never repository files, fixtures, logs or commits. Preserve outbound privacy
  rules and destination authentication. Remote calls identify the DKC version.
- A second document embedder exists only when configured. Single mode never
  reads a second vector table. No in-process generative model, audio or video.
- Rust stable, edition 2024; original code is MIT OR Apache-2.0. Dependencies
  retain their own licenses. Keep the complete notice bundle current.

## Checks and changes

Inspect Git status and recent commits; preserve unrelated work. Before product
changes establish a passing `cargo test` baseline. Before committing run:

```
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo deny check
```

On the NVIDIA bench host, also run Clippy and tests with `--features cuda`.
Never use `--all-features`; GPU backends are platform-specific. Validate changed
benchmark/release tools and the offline license bundle. Do not bypass the hook.
No unwrap/expect/panic/todo/unimplemented outside tests, build.rs and bench.
Keep unsafe at the permitted FFI edges, with SAFETY comments. Document public
functions and keep generated fixtures and individual tracked files below 1 MB.

Keep commits coherent and use `type(module): summary`. Update the state map and
relevant public docs when behavior changes. Local logs and journals are ignored;
retain terminal exit status and resume information for long runs. Serialize
Cargo writers and use immutable binaries during measurements. Never invent a
measurement, conceal a failed check or call an ignored test passed.

## Evidence and release

The beta does not complete the archived v1 acceptance plan. Consult its preserved
Git history/external archive before resuming full v1 qualification; missing
acceptance evidence must continue to block stable release. Do not publish,
transfer ownership or change repository visibility without owner authorization.
Keep this contributor policy owner-reviewed. Current beta evidence is in
bench/results/, with methods and limits in docs/benchmarks.md.
