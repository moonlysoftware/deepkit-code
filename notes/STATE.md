# Current development state

## Current handoff — 21 October 2026

The owner requested a fresh private `moonlysoftware/deepkit-code` with one
initial commit. The original private repository is renamed to
`moonlysoftware/deepkit-code-history`, preserving commits, PRs, issues and
Actions artifacts. Product source is carried forward from `b4e7b289`; only
repository-history documentation and the parentless CI baseline handling change.

PR #4 was merged after source CI passed with 1,416 tests passing and 17 ignored
per Rust run. The private Homebrew preview is revision `0.1.0-beta.1_1`, pinned
to native run `38807128022` in the history repository. Both native artifact
lifecycles passed. Earlier run failures and the records below remain historical
evidence; they do not supersede the final PR #4 result.

The current repository has no old Git ancestry. Use the separate history
repository for archived specifications and evidence, following notes/README.md.
Public release, visibility changes and full v1 acceptance remain separate work.

## Previous development record

- **Active follow-up:** `codex/codex-mcp-setup` adds opt-in project Codex stdio
  setup and actionable HTTP bind diagnostics. It includes bounded regressions
  for separate-project startup, same-project clients and independent shutdown.
  Baseline `5d084d2` passed hosted CI run `38633657916`, including Rust tests,
  strict Clippy and dependency checks. The owner requested minimal testing load
  on the active Mac; full Rust validation of this follow-up is deferred to the
  draft PR's hosted CI. Local formatting, diff checks and the offline notice
  bundle verification passed; no local Rust build or model download was run.
  This is not evidence that the colleague's exact setup
  has been reproduced, nor qualification of simultaneous same-index creation.
  Initial PR/push CI runs `38798021823` / `38797993747` stopped at Clippy's
  `enum_variant_names` lint after the third `ExtraFile` variant was added;
  tests did not run. The follow-up removes the common variant suffix rather
  than suppressing the lint. Both corrected runs passed Clippy. PR CI
  `38799646315` then hit the existing image correctness test's declared 2-second
  query deadline; push CI `38799637361` passed all 1,174 unit tests, then exposed
  incorrect bootstrap-progress expectations in the new reconnect tests.
  Reconnect assertions now check persisted completion and successful searches;
  initial-startup progress assertions remain strict. Only the image correctness
  fixture gets a 10-second query budget; production limits are unchanged.
  Native Homebrew run `38799646335` passed on macOS ARM and GNU Linux. Hosted
  runs `38803502288` / `38803494672` then passed all unit tests and the separate-
  project regression, but both exposed an existing-index startup guard race in
  the same-project regression. Startup's strict identity snapshot rejected normal
  SQLite sidecar changes by the other opener. Bootstrap and repository opens now
  reuse the existing sidecar-tolerant check only when the captured database was
  present; fresh creation and destructive cleanup retain strict checks. Deterministic
  regressions retain rejection of changed database/ancestry and hostile sidecars.
  Hosted validation of this correction is pending; prior failures remain evidence.

- **Phase:** 4 — archived `plan/phases/phase-4-corpus-bench-ship.md` at `a7e45b0`; full v1 acceptance remains open.
- **Task:** T4.7 / T4.8 / T4.12 — user-requested Unix beta preparation.
- **Branch:** `main`; PR #3 merged as `e1a1777feb1cbf99e29acef25e7723c6092690f7`, preserving all 97 preparation commits.
- **Next action:** run the prepared private tap's documented installation on the invited Mac. Source integration is complete; public visibility and release/tag publication remain owner actions. Preserve the earlier unresolved watcher CI latency miss.

## Main integration

The owner requested that main receive the latest work on 14 October. PR #3
merged after all four existing runs at `c2477b6` passed: PR and push Rust CI
(each 1,399 passed, 17 ignored), native ARM Mac/GNU Linux Homebrew lifecycles,
and release planning. The watcher test passed in both current CPU jobs; that
does not establish a repair for the earlier intermittent miss. Release build
and publication jobs were skipped. The merge has the exact reviewed tree,
retains both parents and includes the older PR #2 head. Local main is synced;
the private tap retains its original artifact pins. See
bench/results/beta-20261014/main-integration.json for the terminal check receipt.
The handover follow-up changes documentation/evidence only and reuses these
unchanged product checks; automatically triggered main CI is separate.

## Private tap preparation

The source repository is now `moonlysoftware/deepkit-code`, still private; SSH
and admin access were verified and origin updated. The new private tap checkout
is `/home/lucky93/homebrew-tap`. It pins the previously verified native PR run
38604704711; archive expiration is 13 November 2026. Its authenticated `gh`
helper checks archive/build/binary/license identities and seeds Homebrew's cache.
The tracked formula contains no credentials or machine-specific paths. ARM Mac
is the planned public platform; GNU Linux is a private diagnostic only. No new
release/tag, public visibility or full-v1 acceptance is included in private setup.
The source merge was subsequently authorized and completed as recorded above.

Tap implementation `d9d53b6` is committed and pushed. The real SSH tap clone,
authenticated artifact download, install/search/test/license checks, uninstall,
fresh install, reinstall/test/uninstall passed on Linux GNU. All 25 helper tests
passed locally and in hosted Ubuntu/macOS15 checks (run 38628582047). The new
Mac private-tap install and a distinct-version upgrade remain untested. Evidence:
bench/results/beta-20261014/private-tap.json; full logs are under the external
archive's private-tap/ directory. No DKC binary remains installed in the isolated
Linux test prefix.

The earlier source-head `cee7a68` CI/native runs were cancelled (38625799493,
38625785479, 38625799402); release planning passed. Cancellation is not a pass,
and the earlier watcher miss remains open. Use the original qualified artifacts
for private preview and rebuild reviewed main before public distribution.

## Current scope

Read [the beta audit](../docs/beta-audit.md), [handover](../docs/beta-release.md)
and [benchmarks](../docs/benchmarks.md). The user requested licensing, README,
complete reachable-history review, journal cleanup and a concrete transfer/tap
TODO list. This is a finite beta batch, not a restart of the full corpus matrix.

Confirmed path-confinement and credential-read defects are repaired with
regressions. rustls is updated to 0.23.46. sqlite-vector provenance now points
to the Apache revision with byte-identical relevant upstream sources; complete
third-party notices are packaged. The Sep25 lexical fallback and MSVC
mapping are preserved. Windows runtime is still unsupported.

Final checks passed in `notes/runs/beta-20261014-final-gates-2.*` and
`beta-20261014-final-cpu-build.*`: CPU and CUDA each 1,399 passed; 17/19
ignored. Formatting, strict Clippy, cargo deny and the optimized CPU build passed.
The immutable preparation binary is `target/beta-20261014/frozen/dkc`;
its dirty-source provenance is diagnostic, not a distributable release identity.
Earlier failed attempts remain local: the first deny found patched rustls
advisories; the first full test run exposed two hardcoded version expectations,
now corrected to use the package version. Benchmark tooling: 100 tests passed;
release tooling: 37 passed. The real local Linux Homebrew install/test/reinstall
cycle passed. Archive size: 29,279,198 bytes. A 30-call vague-search diagnostic
completed: top-five expected files 6/10 lexical, 9/10 hybrid, 8/10 reranked.
The selected lexical rule's 144 ordered-result projections matched; the original
strict harness failure on removal of an empty WAL is preserved separately.
No additional corpus or tuning loop is needed for this preparation.

Preparation commit `ae9f1be` is included in merged PR #3. Native Homebrew run
38604704711 passed on Apple Silicon and Linux GNU. Both archives match the
locally tested source hash and contain verified notices. Their embedded commit
is the PR merge `41efee5b879fb2353a5407516aa90a957fba78ee`; rebuild on reviewed
main before publishing. Archive sizes are 27,565,669 / 28,959,976 bytes.
Release planning run 38604704675 passed; publishing/native full-v1 jobs were
skipped, so it contributes no full-v1 qualification. Both CPU CI runs passed:
PR 38604704620 and push 38604671146. Public terminal receipts are retained in
bench/results/beta-20261014/hosted-ci.json. The final follow-up records only
documentation/evidence; product and packaging source remains identical to ae9f1be.

Cleanup validation passed with 100 benchmark-tool and 37 release-tool tests in a
clean snapshot without the archived documents or ignored notes. Offline licenses,
relative documentation links and both stable-release refusal modes passed.
The first snapshot setup lacked Git identity, then a built DKC executable;
those harness setup failures are retained with the corrected terminal result
under the external archive's `approved-cleanup/` directory.

Later `c5f69eb` push CI 38611067348 failed only the polling delete-publication
wait in tests/watch.rs (1.5 s); PR CI 38611079412 and native checks 38611079472
passed. The prior green receipts are not overwritten. No threshold was widened,
no test skipped and no product fix inferred from the parallel pass. See
bench/results/beta-20261014/followup-ci.json for the failure and bounded diagnostic.
Five focused CPU watcher tests passed without changes (polling delete 743–757 ms),
so the hosted miss is unreproduced locally, not repaired. No further retries run.
Full logs and hashes are in the external archive under approved-cleanup/.

## Archive and local files

Verified backup: `/data/dkc-corpus/beta-archive-20261014-a7e45b0/` contains the
all-ref Git bundle, pre-audit patch, development archive and audit inventories.
The original archive manifest records hashes. Routine logs and journals remain
on this machine but are removed from the tracked tree. Only this map,
QUESTIONS.md and README.md remain tracked in notes. Public evidence lives under
`bench/results/beta-20261014/`; reusable inputs/tools moved to fixtures/bench,
bench/acceptance and scripts. Do not delete or stage `notes/STOP`.

The owner approved the protected-document proposal on 14 October. Applied the
exact reviewed patch: 33 internal plan/spec/reference files archived, AGENTS.md
replaced with the supplied contributor guide. No history rewrite. The patch is
`protected-doc-cleanup-with-guide.patch` in the backup directory, SHA-256
`115224bc124e0a6ef84d456f1dc6c2c0b01a9e003b63cfaa433284561dda2046`.
Historical source comments citing v1.md refer to that preserved specification;
see notes/README.md for retrieval. The protected-file hook remains enabled.

The old unattended coding loop is retired. No corpus campaign is authorized
by this cleanup. The v1 release gate deliberately fails closed without its
archived acceptance ledger; no missing historical result becomes a pass.

## Preserved v1 ledger (not beta claims)

| Task | State | Remaining work |
| --- | --- | --- |
| T4.1 | PARTIAL | Complete active 26-target fixture/corpus verification; Aurora deferred beyond v1 |
| T4.2 | OPEN | Fresh host/candidate admission, shared rerank/parser misses, then full CUDA groups A/B/C |
| T4.3 | PARTIAL; timing blocked | Lexical provider-switch development check closed; Moonly/CPU matrix, live Ollama/profile acceptance and dual mode remain |
| T4.4 | PARTIAL | Formal paired scoring, general ranking review and public human goldens |
| T4.5 | PARTIAL | Final harness evidence, timer firing and full nightly verdict |
| T4.6 | OPEN | Measure all requirements; repair or explicitly dispose of misses |
| T4.7 | PARTIAL | Beta licensing and Apple Silicon Homebrew artifact qualified; other distribution targets and publication remain |
| T4.8 | DOCUMENTATION PREPARED | Keep README/doctor claims aligned with final evidence |
| T4.9 | MEASURED MISS | Preserve completed 216-session result; any follow-up separately diagnosed |
| T4.10 | FAILED/OPEN | Native Metal performance, L2 coverage and hybrid-search verification |
| T4.11 | OPEN | Native Mac/Windows/Linux installer evidence per required form |
| T4.12 | OPEN | All acceptance rows, PCR disposition, human review/release and tags |


The full historical Phase 4 report and PCRs are in the external archive and
Git history at the starting commit. Their conclusions remain authoritative for
v1; consult them before resuming that acceptance effort. New beta receipts do
not silently close those criteria. User decisions are summarized in QUESTIONS.
