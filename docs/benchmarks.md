# What we measured

These are development measurements, not a promise that every repository will
score the same way. DKC's lexical, vector, graph and reranking contributions
are separated below. The [original results](../bench/results/beta-20261014/)
include per-query outputs and a SHA-256 manifest.

**HitFile@5** means at least one expected file appeared among the first five
returned hits. **Recall@10** measures the fraction of expected files found in
the first ten hits, averaged across questions. Ten hits can include more than
one hit from the same file. Neither metric measures answer correctness.

## Words, vectors and reranking

The September 21 ripgrep experiment used 200 commit-derived source queries,
three repetitions, pinned revision `3fce3b5bb0236da2df6d99672afb8a719642eca7`,
and clean DKC source `572a0ed712365a70620fd1e459fd10550de7111d`.

| Retrieval | HitFile@5 | Recall@10 |
| --- | ---: | ---: |
| Lexical + graph | 76.0% | 71.86% |
| Hybrid, adding vectors | 76.5% | 72.47% |
| Hybrid + reranker | 87.5% | 81.67% |

The CPU release executable ran on Linux, an Intel i7-14700K and 128 GiB RAM.
Moonly supplied Qwen3-Embedding-8B (1,024 dimensions) and Qwen3-Reranker-4B
(50 candidates). Model aliases were recorded; immutable remote weight revisions
were unavailable. Reranked p95 was **1,235.6 ms**. The measured corpus lane also
missed throughput and footprint targets and retained seven token-cap pauses.
This is evidence of better ranking for this cohort, mainly from reranking;
it does not establish a vector-search gain on vague questions.

Source: [full report](../bench/results/beta-20261014/2026-09-21-final-A-moonly-ripgrep.json.gz)
and [audit receipt](../bench/results/beta-20261014/t4-final-moonly-ripgrep-evidence-1.json).

## Vague questions and graph context

### Current semantic-search diagnostic

On October 14 we replayed ten previously seen, human-reviewed ripgrep questions
through three retrieval settings. The expected files and question bytes were
frozen before this check; no ranking adjustment followed the results.

| Retrieval | HitFile@5 | Recall@10 | Median CLI wall time |
| --- | ---: | ---: | ---: |
| Lexical + graph | 6/10 (60%) | 60% | 71 ms |
| Hybrid, adding vectors | 9/10 (90%) | 85% | 242 ms |
| Hybrid + reranker | 8/10 (80%) | 75% | 1,077 ms |

All 30 calls completed without partial or degraded responses. The source,
configuration, index and executable stayed unchanged. Calls rotated through
the three settings once per question, with source scope, ten results, and no
reindex, retry or tuning. Moonly supplied Qwen3-Embedding-8B (1,024 dimensions)
and Qwen3-Reranker-4B (50 candidates); immutable remote weight revisions were
not available. The prepared index had 9,112 vectors among 9,948 units, so this
is not a claim of complete vector coverage.

The optimized CPU executable included the beta preparation changes and recorded
dirty-source provenance. This small, previously seen set shows a vector gain
and a reranking regression on those questions. It is neither a fresh holdout
nor a release latency qualification. Times include CLI startup and any remote
request. The host was idle at admission; there was one repetition per setting.

See the [complete diagnostic](../bench/results/beta-20261014/vague-probe.json),
[host receipt](../bench/results/beta-20261014/probe-host-admission.json) and
[reproduction tool](../scripts/bench/vague_probe.py).

### Earlier graph and lexical controls

The September 25 experiment included 40 questions marked `vague`: ten each
from ripgrep, Laravel, Django and Hugo. These ask about behavior without naming
the answer symbol. The exact questions are in [bench/goldens](../bench/goldens/).
Ripgrep's questions were [human-reviewed](../bench/results/beta-20261014/t1-15-golden-review.json);
the other three sets remain drafts. The approved ripgrep bytes intentionally
retain their historical DRAFT header so earlier hashes remain valid.

Examples include “Which code decides whether hidden files are skipped?” and
“Where does the program decide between streaming a file and mapping it into
memory?” The expected files and ambiguity notes are published with each query.

| Lexical-search setting | HitFile@5 | Recall@10 |
| --- | ---: | ---: |
| Default graph contribution | 16/40 (40%) | 33.75% |
| Neighbor graph votes disabled | 14/40 (35%) | 33.75% |
| Improved plain-question fallback | 17/40 (42.5%) | 45.0% |

These calls used source scope, ten results, no embeddings and no reranker.
Disabling neighbor votes leaves other stored structural ranks in place: this
is not a measurement of removing every graph feature. A two-question top-five
difference is small, and top-ten recall did not improve from those votes.

The fallback change removes common grammar words only when natural-language
lexical matching needs to broaden. It preserves explicit query syntax and
identifier terms. Across all 120 development questions, recall improved from
64.58% to 70.42%. Twelve separate fresh diagnostic questions improved from
29.17% to 37.50%; that small, low-scoring sample does not prove broad performance.
Those twelve questions were agent-authored and are not human-reviewed release goldens.
An identifier-preservation bug found after selection was fixed separately.

The original prototype was based on `b3fc30ab` and records dirty-source build
identity. The [frozen protocol](../bench/results/beta-20261014/protocol.md),
[all-rule comparison](../bench/results/beta-20261014/quality-screen.json),
[graph subgroup audit](../bench/results/beta-20261014/claim-audit.json), and
[fresh-question result](../bench/results/beta-20261014/final-validation.json)
retain the controls, rejected rules and limitations. These are lexical gains,
not a semantic model benchmark.

The October 14 optimized preparation build reproduced the selected rule's
ordered hit identities, paths and ranges on all 120 development questions,
12 syntax controls and 12 previously exposed final-check questions. The initial
strict harness failed because SQLite removed an empty WAL file; a separate
audit found the database, sources and configuration unchanged. Returned token
estimates differed, so this is ranking parity, not byte-identical responses.
One separately reconstructed `CAN` identifier control also passed. See the
[complete replay receipt](../bench/results/beta-20261014/shipping-parity.json),
which preserves the original failed attempt. This was not a fresh holdout.

## Does it save agent tokens?

Our completed experiment used 30 ripgrep questions, three paired repetitions
and 180 primary agent sessions. Another 36 auxiliary sessions were separate.
The agent was Qwen3.5-35B-A3B through Moonly at temperature zero, with thinking
disabled, fixed budgets, counterbalanced order and condition-blind Codex grading.

| Condition | Agent tokens | Correctness points |
| --- | ---: | ---: |
| Without DKC | 1,548,207 | 71/180 |
| With DKC | 1,607,814 | 86/180 |

DKC improved the graded answers in this sample but used **3.85% more tokens**.
Scores are points on a 0–2 scale per answer, not counts of correct sessions.
The question-cluster uncertainty interval for the token ratio crosses one.
We do not claim token savings. Model work inside DKC is accounted separately.

See the [result](../bench/results/beta-20261014/t4-9-final-result-2.json),
[protocol](../bench/results/beta-20261014/t4-9-protocol-final-2.json), and
[frozen rubric](../bench/goldens/ripgrep-rubric.json). Full transcripts and
development journals remain in the verified development archive.

## What remains open

The complete corpus/profile matrix and several latency, model-throughput and
database-footprint targets are unfinished. Apple Silicon Homebrew installation
passed on October 14; Intel Mac, portable Linux packages and Windows runtime
still need qualification. See the [native receipts](../bench/results/beta-20261014/native-homebrew.json).
The earlier M1 performance run missed its throughput/rerank targets and timed
out on its pinned-ripgrep search. A successful build does not resolve those
misses, and the new Homebrew smoke did not exercise model inference. Beta
preparation does not mark the v1 acceptance plan complete.

Use the [benchmark tools](../scripts/bench/README.md) for repeatable tests.
Keep query inputs, source revision, model settings, failures and partial results
when comparing versions; do not retune questions to improve a headline.
