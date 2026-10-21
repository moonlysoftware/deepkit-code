# Ten bounded search-quality iterations — 2026-09-25

Authorized by Bart: ten iterations collecting data and making small changes, aiming for roughly two hours of experimentation. This is T4.4/T4.9 development evidence, not a replacement for Phase 4 acceptance or human golden review. The old paused Codex goal could not be replaced by the tool because it is unfinished; Bart has the short replacement objective in chat. Work proceeds independently of that UI limitation.

## Frozen method (before scores)

- Product base: b3fc30ab8d167f0e2a0781069d9bfa5e111161aa. Prototype source is isolated at `/data/dkc-corpus/dkc-search-tuning-source-20260925`; experimental switches never ship.
- Fresh L1 indexes at `/data/dkc-corpus/search-tuning-20260925/{ripgrep,laravel,django,hugo}`, pinned by `bench/corpus.toml`. No queue, external models, index copying, or corpus matrix. `notes/STOP` stays untouched.
- Preparation binary: `target/phase4-provisional-cuda-1476005/dkc`, SHA256 `e63bd6457cd4445fcc3d426a3d3d01f748fc7948f04f3d81ebbf0d2faa46357e`. Clean 1476005 Rust source is identical to base; only the MSVC-only CPUID include differs. GNU L1/search paths are unchanged. This is diagnostic evidence, not an exact-current-release artifact.
- Development cohort: all 120 existing questions from four `bench/goldens/*.yaml` files. These have been seen before. Keep their exact bytes/labels. Uniform `--scope source --mode lexical --no-rerank -k 10 --json`; this explicit source-only protocol differs from some earlier reports.
- Baseline arm 0 and each arm use the same prototype build and complete public pipeline. Compare arm 0 with the immutable preparation binary before using it. Freeze binary, source patch, queries, root HEADs, config and DB hashes. Every run retains failures and partial responses. No post-hoc label fixes.
- Primary: per-question fraction of expected files delivered at top 10 (Recall@10); secondary: any expected file in top 5 (HitFile@5). Duplicate chunks count once per expected file. Report each repository separately. Failed requests remain in denominators.
- Time is CLI-inclusive monotonic elapsed, including process start and freshness checks. Median/p95 are diagnostic, not warm MCP or idle release timing. Repeat baseline/candidate in opposite order for finalists; do not time while compiling or indexing.
- Final untouched holdout: 12 independently source-grounded agent-authored questions, four each Laravel/Django/Hugo; six cross-file. SHA256 `29bbfc2f3b4988ca097c942d166e8bce76dd5443949d2c37a16c07d373bd15b3`. Root does not inspect questions/labels until iteration 10. These are diagnostic, not human-approved goldens.

## Predeclared iterations

Each arm is independent against baseline, not cumulative. Preserve original dense text, typed/quoted query semantics, source scopes, privacy, budgets, freshness and exact-identifier priority. Exclude typed or quoted queries from every experimental arm, including rank-only arm 3. Before any variant scores, the implementation guard was fixed to require at least two lexical words and at least one non-identifier word for natural-text treatments; untyped arm 3 may also retain precise matches for standalone terms.

| Iteration | Hypothesis | Fixed treatment |
| --- | --- | --- |
| 1 | Conversational framing obscures intent | Strip a leading Where is/does/are, How do/does/are, Which or What on natural questions |
| 2 | Repeated lexical clauses distort BM25 | Deduplicate equivalent unquoted lexical clauses, retaining first order |
| 3 | Broad fallback discards precise matches | Keep nonempty strict-AND matches first, append unseen OR candidates when strict has fewer than k |
| 4 | Identifier fields should carry exact spelling | Exact identifier spelling in idents; grouped subtokens in body; ordinary terms unchanged |
| 5 | Coverage of meaningful terms beats one rare word | OR candidates ordered by distinct substantive clause coverage, stable BM25 ties; no fitted weights |
| 6 | Adjacent words express relationships | A boolean adjacent-query-phrase preference on natural unquoted queries; baseline order within ties |
| 7 | Named code should anchor mixed questions | When explicit identifier groups match, prefer their matches in mixed natural/identifier fallback |
| 8 | Graph expansion may dilute natural questions | Disable existing graph weight for natural questions with no explicit identifier; diagnostic ablation only |
| 9 | Grammar words add unrelated matches | Remove frozen grammatical words only during natural-language OR fallback: a an the is are was were be been being do does did can could would should where which what how; preserve negation and from/to |
| 10 | A gain survives new questions | Evaluate the single strongest eligible rule on the frozen final holdout, with baseline; no further tuning |

## Keep/reject rule

An eligible rule must improve aggregate Recall@10, avoid a drop in either metric on every repository, and improve recall on at least two repositories. It must introduce no errors/partial responses, preserve index/config bytes, and pass exact identifier, quotes, typed-query, no-match and scope controls. Finalist repeated CLI p95 must not increase by more than 10% or 10 ms (whichever is larger). Among eligible rules choose highest aggregate recall, then HitFile@5, then simpler/faster code. No combinations or fitted weights in this batch. A diagnostic graph ablation cannot alone justify removing specified behavior.

At iteration 10, retain only if holdout aggregate recall and HitFile@5 do not decrease and no repository loses recall. If no rule is eligible, record nine rejections and validate baseline repeatability on the holdout. Zero accepted changes is a valid outcome; do not disguise rejected experiments as improvements. If promising but below these criteria, document a separate future hypothesis rather than extending this batch.

Integrate only the minimal winning implementation; meaningful focused tests, then all required CPU/CUDA fmt/Clippy/test/deny gates once per integrated batch. Record every trial, update STATE/journal, commit and push evidence and any verified code. Full release acceptance remains open.
