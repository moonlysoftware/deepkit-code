# Deepkit Code

Deepkit Code (`dkc`) gives your coding assistant a searchable view of a repository:
code, docs, supported documents and images, and the Git history behind them.
One Rust binary. One SQLite index. CLI and MCP.

**Private beta:** Deepkit Code now lives under `moonlysoftware`. Invited testers
can use the [private Homebrew tap](https://github.com/moonlysoftware/homebrew-tap)
or [build from source](docs/usage.md). Public installers and full v1 acceptance
remain unfinished.

Development before 21 October 2026, including earlier commits, pull requests
and build artifacts, is preserved in the private
[deepkit-code-history repository](https://github.com/moonlysoftware/deepkit-code-history).
This repository starts from the current code with a fresh Git history.

## The Deepkit idea, brought into a repository

Deepkit Code adapts Aurora's Deepkit knowledge pipeline: extract useful pieces,
keep their context, connect related pieces, and spend model work where it helps.

A function is more useful beside its file, containing class, nearby docs and
change history than as an anonymous text chunk. DKC keeps those connections and
checks returned source locations against the repository.

When several matches point to the same symbol, DKC groups their evidence into
one result. A pull request groups its commits; a symbol history collects its
changes. Histories are assembled from source records rather than invented by a
model. Optional summaries are stored separately and retain their sources.

That is the point of collating results: let the assistant open a useful piece of
context, then follow its evidence. It still needs to read and reason about the
code; a search result is not an answer by itself.

## What is inside

```text
repository + Git history
        ↓
code / document extraction → symbols, sections, relationships, changes
        ↓
.deepkit/code.db
  exact words + optional vectors + an approximate code graph
        ↓
combine rankings → optional reranking → group overlapping matches
        ↓
CLI or MCP → search, open, symbol, refs, map, recent, status
```

Indexing starts with a file catalogue, then adds searchable text and structure.
Embeddings and optional descriptions follow within your configured limits.
You can use plain search without downloading a model or adding an API key.

- **Code and context:** source, Markdown, notebooks, digital PDFs and supported
  Office files. Images and scanned pages need a configured vision provider.
- **Three retrieval signals:** exact words and names, vector similarity, and
  nearby graph relationships. Reranking can reorder the combined candidates.
- **Traceable results:** source ranges, freshness information, and evidence
  retained when several matches collapse into one result.
- **Repository history:** searchable commits, reconstructed pull requests,
  module context and collated symbol histories.
- **One local index:** no separate database server or indexing daemon.

## A few measured results

These are retained development measurements, with their limitations attached.
They describe the tested inputs and builds, not every repository.

| Check | Result |
| --- | --- |
| 200 ripgrep commit-derived questions, three repetitions | Expected file in the top five: **76.0%** with words + graph, **76.5%** with added vectors, **87.5%** after reranking. |
| 10 vague, human-reviewed ripgrep questions | Expected file in the top five: **6/10** with words + graph, **9/10** with added vectors, **8/10** after reranking. |
| 40 vague questions across four repositories | Neighbor graph votes found an expected file in the top five for **16/40** questions, versus **14/40** with those votes disabled. Top-ten recall was unchanged. |
| 30 ripgrep questions, three paired agent repetitions | Graded correctness rose from **71/180 to 86/180 points**, while agent token use rose **3.85%**. We do not claim token savings. |

The first comparison used Moonly's Qwen3 8B embedder and 4B reranker on a
September 21 build. Reranked p95 latency was **1.24 seconds**, above our target.
Those queries came from commits; they do not establish a gain on vague questions.

The October 14 vague-query check used the same model aliases on the beta
preparation build. Vectors helped this small, previously seen set; reranking
lost one top-five match. All 30 calls completed. It is a diagnostic, not a fresh
holdout or a promise of the same gain in your repository.

The graph comparison used lexical search only, on a September 25 prototype.
Three of its four question sets still await human review. It isolates neighbor
graph votes, not every use of structure. A separate lexical fallback experiment
improved results, but that is not evidence of better semantic retrieval.

[Benchmark methods, exact inputs, results and misses](docs/benchmarks.md)

## Models and privacy

Choose local models through llama.cpp, remote Moonly models, or an Ollama server.
A second document embedder is optional. Local embedding and reranking keep that
work on your machine; configured remote services receive the corresponding
query and repository text. Summaries and vision processing use remote providers.

The index lives in `.deepkit/`. Credentials belong in your environment or the
user credentials file, never in a repository config. A `never_send` treatment
keeps matching sources out of remote model requests. Caps and incomplete work
are visible in `status`; lexical results remain available without full vectors.

## Trying the beta

Install the private Homebrew preview or build using the [setup guide](docs/usage.md), then connect
with `dkc init --codex`, `dkc init --cursor` or `dkc init --claude`. Other MCP clients can start
`dkc serve --stdio`; streamable HTTP is also supported.

The private Homebrew preview needs access to both Moonly repositories and an
authenticated download step. Its tap README has the commands. Public Homebrew,
shell, PowerShell, npm, MSI and `cargo binstall` installs are not published yet.

Some performance targets and native installer checks remain open. The full
corpus matrix is unfinished, and Windows source verification still needs work.
Graph relationships are approximate. Beta feedback on your own repository is
valuable; include the DKC version and a small reproducible example.

Deepkit Code grew from Deepkit's treatment and retrieval design. DKC's code is
MIT OR Apache-2.0; bundled dependencies retain their own licenses.
