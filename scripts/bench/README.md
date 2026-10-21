# Early nightly regression

For published measurements and their scope, read [the benchmark guide](../../docs/benchmarks.md).
Selected original receipts live in `bench/results/`; `notes/` is now ignored
local run storage. The dated v1 acceptance ledger remains in the external
development archive and Git history. This tooling does not imply that its
complete matrix has passed.

`vague_probe.py --binary /absolute/immutable/dkc --out /new/report.json`
performs one bounded comparison of ten frozen, previously reviewed ripgrep
questions. It requires Python with PyYAML and the prepared remote index at
`/data/dkc-corpus/ripgrep` (override with `--root`). It does not index, retry
failed queries or tune labels. Run it in an idle window; retain incomplete arms
and all source/model identities. It is a diagnostic, not a fresh holdout.

## Frozen Phase 4 candidate preflight

Before launching the complete Phase 4 matrix, commit and gate one clean product
revision, build separate immutable CPU and CUDA release binaries, and record a
`dkc.phase4.frozen-candidate.v1` JSON manifest. Run
`python3 scripts/bench/candidate_preflight.py MANIFEST.json`. Exit 0 means its
static inputs still match; exit 2 refuses the run and prints all detected
mismatches. This command is read-only and does not index or call models.

The manifest contains:

- `product`: full Git `commit` and the build's `source_blake3`.
- `host.boot_id`: the boot intended for the matrix. A reboot needs a new
  admission receipt and manifest before any timed lane.
- `corpus_manifest`, `lane_plan` and `matrix_baseline`: checkout-relative
  `path` and `sha256`. The 81-lane import skeleton must exactly match the
  separately audited 26-target matrix baseline. Aurora qualification was
  deferred beyond v1 on 2026-09-25; its original entry remains in
  `bench/corpus-deferred.toml`.
- `inputs`: hashed checkout-relative paths for the corpus TOML, its referenced
  configs/goldens, both matrix templates, and each lane's effective-config and
  model-identity receipts. Keep credentials out of these receipts.
- `targets`: every corpus target's absolute `root` and pinned `head`. The
  generated `fixtures-docs` target pins both its empty local Git seed `head`
  (needed by `dkc bench run`) and its versioned, file-framed `tree_sha256`
  (which pins untracked content). Changing the hash version invalidates
  earlier generated-fixture candidate pins and nightly resume state.
  Its hash includes relative filesystem path bytes (including the deliberate
  invalid-UTF-8 name) and file contents, while excluding `.deepkit/` and
  `.git/`.
- `binaries.cpu` and `binaries.cuda`: checkout-relative immutable `path`, file
  `sha256`, and the complete `build` object returned by that binary's
  `dkc bench metadata --json` command.
- `audit_tool`: `path` exactly `scripts/bench/audit_matrix.py` and its
  `sha256`; include the same path and hash in `inputs` so an audit change
  invalidates the candidate before another lane starts.
- `lanes`: every required `(target, profile)` pair, its `binary` (`cpu` or
  `cuda`), distinct `output` under `notes/bench/`, `state` (`pending` or
  `complete`), and references to its `effective_config_input` and
  `model_identity_input` in `inputs`. Pending outputs must not exist. A
  completed lane additionally records `report_sha256` and `audit_input`; the
  retained report and accepted audit must still match the frozen build and
  target before the remaining lanes can resume.

The current blank matrix and import skeleton are
`bench/acceptance/corpus-matrix.json` and
`bench/acceptance/corpus-lanes.json`.
Earlier baselines retain their historical scope and do not admit a new run.

Prepare the 81 per-lane bindings with
`python3 scripts/bench/freeze_candidate.py --template
notes/bench/CANDIDATE-inputs.json --name CANDIDATE`. This creates an incomplete
input form; its empty binary and receipt paths are deliberate. Fill the two
immutable release-binary paths and, for every lane, the checkout-relative
`effective_config_input` and `model_identity_input` paths. These must be
existing nonsecret receipts for the exact target/profile settings and model
deployment or local model bytes, not invented labels. Once all inputs and
corpus roots are present and `notes/STOP` is absent, run
`python3 scripts/bench/freeze_candidate.py --spec
notes/bench/CANDIDATE-inputs.json --out notes/bench/CANDIDATE.json`. The assembler
hashes each input, reads both binaries' self-reported build metadata, pins
every corpus HEAD and the generated fixture tree, and runs the full static
preflight before writing the manifest exclusively. A missing receipt, drifted
binary, incomplete plan or failed preflight leaves no candidate output. The
input form itself is one of the frozen hashed inputs.

Each lane needs two versioned JSON receipts whose `target` and `profile` equal
the lane key. The `dkc.phase4.effective-config.v1` receipt contains
`effective_config_blake3` (the 64-character BLAKE3 value reported after the
benchmark has applied its profile and target configuration) and
`config_files_blake3` (a map of `config.toml` and, when present,
`config.local.toml` to their BLAKE3 values). Capture these from a real
equivalent configuration preparation, not from the input TOML alone: `dkc
bench run` writes a target's generated `.deepkit` configuration before it
computes the effective settings hash. A completed lane's report must reproduce
both hashes and the frozen target HEAD exactly.

Use the exact intended binary to capture one lane without indexing or
changing its corpus checkout:

```sh
python3 scripts/bench/capture_lane_receipts.py \
  --binary target/release/dkc --target tiny --profile local_cpu \
  --prefix CANDIDATE-tiny-local_cpu
```

`dkc bench capture-config --corpus bench/corpus.toml --target tiny --profile
local_cpu --json` is the underlying command. It copies the target's current
configuration layers into a private disposable Git repository and runs the
benchmark's own configuration setup there. It emits the target HEAD, SHA-256
hashes of both starting configuration files when present, exact
effective/config-file hashes, and model fields; it does not index the target.
The helper writes an exclusive `-config.json` receipt and a
`-models-template.json` under `notes/bench/`. Candidate preflight accepts the
config receipt only when its starting hashes and capture binary SHA-256/build
match the frozen target and selected CPU/CUDA binary. The model template has empty
`identities` and is deliberately **not**
candidate-ready until independent local-file/deployment evidence is added.
Remote profile initialization may still check its configured provider; capture
is not an offline claim of provider health.

A candidate freezes the starting `config.toml` and `config.local.toml` hashes
for every target without copying their possibly private contents into the
manifest. The matrix runner saves those files under that target's `.deepkit/`
with mode 0600, runs one lane, and restores their exact bytes even when the
lane fails normally. Preflight rejects changed starting files before another
lane or resume. An abrupt runner crash leaves a named
`.deepkit/bench-config-recovery-<candidate>` directory; inspect and restore
its pinned files before continuing. Never discard that recovery copy to force
the matrix past a mismatch. A skipped lane remains unaccepted.

The `dkc.phase4.model-identity.v1` receipt contains `model_usage` with the
report's static `configured_profile`, `configured_embedding`,
`configured_reranker`, `docs_tier_configured` and `verified_models` fields.
Its `identities` array has one entry for each active `embedder`, `reranker`,
`docs`, `llm` and `vlm` role. Each entry names `role`, `model`, `provider`,
`kind`, `identity` and a checkout-relative `evidence_input`. `kind=sha256`
requires a lowercase 64-character file digest for a local model;
`kind=deployment` requires a nonempty deployment identity for remote or
Ollama service models. The evidence input uses
`dkc.phase4.model-evidence.v1` with matching `role`, `model`, `provider`,
`kind`, `identity` and a nonempty `origin` naming the independent source of
that observation. The assembler hashes each referenced evidence input into
the candidate. A completed report must match the receipt's static model-usage
fields. Neither a configured model string nor an operator-supplied evidence
file proves which remote deployment answered a request; retain the actual
provider observation and live-model admission separately. Missing or
unverifiable deployment identity keeps that lane out of the candidate.

The matrix runner writes `-identity-before.json` and `-identity-after.json`
beside each lane report. Each observation must match every active role in the
frozen receipt before the lane can be accepted. Local models are fully hashed
on first use in a runner process; later checks reuse the digest only while
device, inode, size and modification/change timestamps are unchanged. Moonly
roles use a small successful request and hash the selected `x-litellm-model-id`
header; Ollama uses the exact tag's `/api/tags` content digest. The gateway
credential and raw deployment header are never retained. A mismatch or probe
failure leaves a failed attempt and blocks blind retry. A diagnosed skip pins
any observations but never accepts the lane. Static preflight rechecks the
observation bytes and frozen identity matches on every resume. A Moonly proxy
deployment ID can still hide a weights change behind an unchanged ID. The
human deferred immutable operator-side revisions on 2026-09-25 (Q-033);
the sourced deployment receipts and before/after drift checks remain required.

The preflight checks exact binary bytes and build identity, committed product
and benchmark tooling, corpus HEADs or fixture-tree hash, all input hashes,
lane-plan coverage, output uniqueness, boot identity and `notes/STOP`. It does
not prove idle load, live CUDA inference, model service health, benchmark
quality or release acceptance. The runner must still take the shared benchmark
slot and enforce each lane's own immediate admission and report audit. The old
`notes/runs/t4-post-reboot-matrix-resume-1.py` launcher is retired; its dated
outputs and product revision cannot enter a new candidate matrix.

Run `scripts/bench/nightly.sh --candidate-manifest
notes/bench/CANDIDATE.json --dry-run` to validate the frozen candidate and
list pending lanes without launching work. After host admission, remove
`--dry-run` to run it, optionally adding `--candidate-limit 1` for a bounded
first invocation. This route takes the same live-corpus lock as nightly,
rechecks preflight before every lane, waits for idle load/device admission,
checks model identity before that idle wait and after the benchmark,
and uses the frozen binary to run exactly one target/profile into its unique
output. It runs the pinned auditor and checkpoints report/audit hashes in a
manifest-bound state file under `notes/runs/`. The manifest itself stays
unchanged. A failed or interrupted attempt retains its evidence and blocks
automatic retry. After diagnosing a target-specific blocker, use
`scripts/bench/nightly.sh --candidate-manifest notes/bench/CANDIDATE.json
--candidate-skip-lane TARGET:PROFILE --candidate-skip-reason 'specific diagnosis'
--candidate-skip-evidence notes/bench/DIAGNOSIS.md` to record one explicit
disposition under the same live-corpus lock. The diagnosis file and failed
attempt log must exist under `notes/`; a retained partial report and rejected
audit are pinned if present. The runner then continues other pending lanes. The skipped lane stays
open, is never an accepted completion, and requires a new candidate/output
path for a repaired rerun. Drift in skip evidence refuses resume. An audit can
preserve numerical release misses without declaring the Phase 4 acceptance
row passed. Neither this command nor `--dry-run` replaces native checks or
the final nightly verdict.

T1.R4 runs CPU release L1 benchmarks for pinned ripgrep and Laravel. It does not call models. T4.5 expands the same entry point to the full corpus, GPU builds and live profiles.

Run `scripts/bench/nightly.sh`. It fast-forwards the working checkout, builds the CPU release and copies the executable to a unique `target/nightly/` directory so later builds cannot replace its child executable. The pull explicitly disables rebase while requiring fast-forward, so local pull.rebase settings do not force a clean worktree. Existing work is not stashed. An overlapping nightly exits 75 without changing the active run's verdict.

Each repository waits up to 15 minutes for load below 1 and no GPU compute process. The benchmark's immediate pre-index idle check must also pass. Results compare through `dkc bench diff --fail-on-regression`, using the last PASS or the pinned corrected release report on the first run. Use `--corpus /absolute/path/corpus.toml` to select verified benchmark checkouts; the default is `bench/corpus.toml`. A missing manifest fails before pulling or building. Keep the manifest and pinned source policy consistent with the baseline, including timestamps for capped files. Every baseline is archived before comparison. Failures never replace the last-PASS report. A PASS means no regression, not satisfaction of all absolute release targets.

An interrupted full run resumes from `notes/runs/nightly-active.state.json` only
when the pulled commit, boot, pinned corpus HEADs, manifest, referenced
configs/goldens, local GGUF bytes and both frozen release binaries still match.
Completed target reports are checked by hash and build identity before reuse;
unfinished attempts and incompatible runs keep their old files. A structured
target failure is retained for diagnosis and is not retried blindly. Moonly
targets need `--model-identity-receipt /path/to/nonsecret-receipt.json` for
cross-invocation reuse, because a local cache hash cannot identify a remote
deployment. Without that receipt, a partial Moonly cohort remeasures prior
remote targets in distinct attempts; a previously complete Moonly profile
cannot be reused on a later invocation. Keep credentials out of the receipt.

The service has a six-hour limit. The runner reserves its last five minutes
for cleanup, bounds build/target children by the remaining slot, and records
`DEFERRED` when it must continue on a later timer firing. Only a complete
profile is compared or promoted to `nightly-latest-<profile>.json`; a deferred
cohort is never a PASS. The active state is archived after all profiles pass or
when frozen inputs change. Its report and comparison paths include a unique
run stamp, so a later run cannot overwrite earlier evidence.

Evidence is under `notes/bench/`, with append-only verdict entries in `NIGHTLY.md`. Logs and exit codes are under `notes/runs/`; logs rotate to numbered siblings after commands exceed 10 MiB. Commands have timeouts and child process groups are terminated on cancellation. The nightly does not commit results. The next development session investigates any new FAIL first.

The systemd templates in `systemd/` use a user timer at **02:00 local**, with missed runs caught up when the timer is active again. Replace `@ROOT@` with the checkout path, install both files into `~/.config/systemd/user/`, run `systemctl --user daemon-reload`, then `systemctl --user enable --now dkc-nightly.timer`. Check `loginctl show-user "$USER" -p Linger`: without linger, execution requires a user session. Test the actual service with `systemctl --user start --no-block dkc-nightly.service`; inspect `systemctl --user status dkc-nightly.service` and the run logs.

`python3 scripts/bench/test_nightly.py` uses the actual Rust `dkc bench diff` from `target/debug/dkc` in a temporary directory to verify PASS, a synthetic quality regression and missing-report failure. Run it after `cargo test` has built the binary. `--compare-only`, `--out-dir`, `--log-dir` and `--baseline` allow replay without indexing; use an isolated output directory for synthetic results. Comparison-only runs keep their output-directory lock but do not reserve the shared live-corpus benchmark slot.

## Bounded embedding preparation

After a completed `dkc index --level 1`, prepare the selected embedding spaces with explicit exercise limits:

```sh
dkc bench embed --root /data/dkc-corpus/ripgrep \
  --model-cache /data/dkc-models --out /tmp/ripgrep-embedding.json \
  --max-calls 10000 --max-tokens 10000000 \
  --attempt-timeout-ms 120000 --heartbeat-ms 10000 \
  --max-schedule 32 --max-jobs 8 --max-passes 1000 --json
```

These are example run bounds, not public queue defaults. Calls/tokens are per-provider daily ceilings including already charged usage. Local models must already be cached and verified; this command does not download weights. Only providers for eligible indexed kinds are prepared. It refuses incomplete L1 or a changed HEAD.

The report is refreshed atomically after each pass and retains build/config identity, bounds, queue state, transition state, coverage and publication counts. `bounded`, `waiting` and `failed` exit nonzero with the report retained. Rerun to resume durable jobs; already published generations are not embedded again. `quiescent` means no remaining work observed for the selected providers, not full corpus coverage: inspect coverage and job states separately. The explicit catalogue sweep assigns changed-file priority; this is not watcher freshness evidence. Use `scripts/run-detached.sh` for long runs and retain its log/pid/exit files.

## Paired search diagnostic

After embedding has stopped and the queue has no pending embeddings:

```sh
dkc bench search --root /data/dkc-corpus/ripgrep-t17v \
  --model-cache /data/dkc-models --out /tmp/ripgrep-paired-search.json \
  --max-calls 20000 --max-tokens 20000000 --repeats 5 --json
```

This derives up to 200 commit-message queries using the existing pinned-tree eligibility rules. Lexical and hybrid modes use the same transport-neutral search/collapse pipeline, source scope and ten-result window, with reranking disabled. It preserves both raw and eligible HitFile@5/Recall@10 scores for each measured repetition. The existing lexical benchmark retains its original retrieval path and shares only scoring/eligibility code.

One full warmup workload precedes measured repetitions; the mode order alternates across queries and repetitions. Reports retain warmup and measured paths, timings and degradation, and are atomically refreshed after each pair. Only measured samples enter latency percentiles. Model loading, report serialization and CLI process startup are outside these timings. This is an in-process CPU/GPU-build diagnostic, not the comparable nightly, CLI round-trip latency or a GPU runtime certification. The build identity and coverage before/after are recorded; run against an idle index. Observed HEAD, L1 generation, config, routing or coverage changes fail the run, but these checks do not promise a frozen database across the whole workload.

Budgets include warmup/retries and usage already charged today. Local models must be cached. Missing routes, warmup degradation and setup failures exit nonzero with retained partial evidence when a report has been initialized. Measured degradation also exits nonzero; inspect each pair before comparing scores. A complete diagnostic does not establish full L2 coverage or hybrid superiority. Unlike embedding preparation, interrupted measurements restart the workload; they do not merge timing samples from different invocations. Reports can exceed 1 MB and should stay outside commits; retain a concise evidence summary instead.


Add `--rerank --rerank-rpm 10000 --rerank-tpm 100000000` to measure a third hybrid-plus-rerank arm. These are explicit example rate ceilings, not defaults; the same per-provider daily call/token bounds still apply. The configured local/remote selection, score threshold, document cap, deadlines and auto mode are honored. Local reranker weights must already be verified in `--model-cache` or a separate `--rerank-model-cache`; no download occurs. `mode=off` refuses requested rerank setup. Setup and smoke loading are outside measured query times.

With reranking enabled, arm order rotates across queries/repetitions. Each arm runs its own retrieval; the unreranked hybrid baseline keeps its configured candidate window while the reranked arm may adapt50→20. Per-sample candidate counts, outcome/fallback and auto state expose that difference. Reports add separate reranked quality and measured-only latency, safe configuration details, rate bounds, and auto state before/after. A degraded warmup query retains all its arms and stops; measured degradation retains evidence but exits nonzero. This commit-derived diagnostic does not replace the handwritten-golden rerank evaluation or GPU runtime/CLI latency gates.


Add `--dense-only` to include a separate dense-only arm, optionally alongside
`--rerank`. For natural-text queries it uses the same queries, scopes, exclusions and
ten-result window, with lexical and graph contributions disabled by the existing
dense retrieval mode. Explicit `lex:` and `vec:` clauses keep their existing
routing in every arm; inspect the reported leg states before describing a
workload as a pure dense-only comparison. Every enabled arm rotates through execution order. Reports add
`dense_only` samples, quality scores and measured-only latency; the default
lexical/hybrid workload remains unchanged when the flag is absent.

Dense-only makes its own embedding requests under the same configured budgets,
including warmup and retry accounting. It is not a free reuse of the hybrid
query embedding. Provider refusal makes the diagnostic non-successful and
retains the failed sample; it must not be presented as a successful empty search.
Handwritten dense-only scores use formatted response evidence with the original
positive, scoped and no-answer denominators. This diagnostic does not alter
production search defaults.

For handwritten evaluation, add `--golden /path/to/primary.yaml` and repeat `--extra-golden /path/to/extras.yaml` for held-out/scoped/no-answer cohorts. This replaces the commit-derived workload. Optional `--golden-reviewed-blake3 HASH` must match the human-reviewed primary bytes; omit it for drafts. Input files are size-bounded and validated before model preparation, including expected indexed symbols. Declared scopes are retained; absent scopes default to source scope.

All arms share the existing handwritten scorer: per-cohort positive, symbol and no-answer denominators stay separate, with no-answer cases excluded from positive recall. Symbol hits require matching file and containing line ranges. Each measured repetition records its own cohort report; warmup is excluded. Raw evidence contains locations and formatted-response token counts, not snippets. Formatting/token counting is outside the shared query timing. A first degraded warmup query saves all of its arms and stops immediately, rather than continuing a workload that cannot pass. Reviewed primary input does not mark extra cohorts reviewed, and draft runs do not satisfy human-review gates.

## Fixed 50-candidate rerank measurement

`dkc bench rerank` scores the same 50 existing unit IDs for every warmup and
measurement. Configure `rerank.mode='on'`, at least 50 candidates, and the intended
provider, document cap and timeout on an isolated, current index first.

```sh
dkc bench rerank --root /path/to/isolated-corpus \
  --query-file /tmp/query.txt --units-file /tmp/units.json \
  --model-cache /path/to/verified-reranker-cache --out /tmp/new-rerank-report.json \
  --max-calls 21 --max-tokens 2000000 --rerank-rpm 60 --rerank-tpm 2000000 \
  --warmups 1 --repeats 20 --json
```

These example daily ceilings include existing usage and warmups; choose bounds
for the actual inputs. `units.json` is an ordered JSON array of exactly 50 distinct
positive SQLite unit IDs. The query is UTF-8. Local weights must already be cached;
the command does not download them. Protected/stale candidates refuse the whole
set. Failures remain in the report, with no retry, subset reduction or passing
percentile. The report retains digests rather than query/document text.

Timing covers admitted scoring, source checks and settlement; preparation is
reported separately. It is not pure model-kernel latency. Actual GPU placement
needs independent process/native evidence. The new report file is created
exclusively and updated through its retained handle; existing files, links and
changed path identities are refused. Progress writes are synchronous, not atomic
snapshots, so wait for command completion before consuming the final report.

For a bounded `dkc bench search` diagnostic, `--continue-degraded` retains the
remaining queries after a degraded warmup. Default behavior still stops at the
first degraded warmup. The option is recorded in the JSON report; all fallback
samples stay in the quality denominator and timing arrays. A degraded run still
exits nonzero and cannot become a passing regression baseline. Provider budgets,
deadlines, source checks and requested repetitions remain enforced. Missing
prerequisites or invalid source/configuration still abort the exercise.


### PR rank-weight diagnostic

`dkc bench pr-ranking --root REPO --queries FROZEN_QUERIES.json
--ranks FROZEN_RANKS.json --out NEW_REPORT.json` compares current lexical ranking,
a uniform1.20 PR boost, and fixed date bands on one shared candidate list per
question. Query input retains `original` and `older_controls` arrays containing
`rank`, `merge`, and `title`; the rank file supplies `pin` and `prs` with `merge`,
`rank`, and `candidate_score_multiplier`. Both inputs must identify the same HEAD.
The fixed multipliers are1.20/1.15/1.10/1.05 for ranks1–100/101–200/201–500/501–1000.
Other ranks retain1.0. Summary occurrences use their projected PR parent.

This command is an internal lexical diagnostic. It preserves requested k3 and
the direct lexical200-plus-graph100 admission window. It does not invoke models,
change production scoring, verify each public source, or apply the service's
response-token limit. `raw_hit_pr_at_3` and `response_before_service_limits` must
therefore remain separate from delivered public HitPR@3. Use external frozen
snapshot/accounting guards and public-response checks before drawing an exit
conclusion. Missing targets and failed questions stay in their cohort denominators;
the output must be new, and partial progress is retained.
