# sqlite-vector-sys

This crate statically compiles the sqlite-vector extension used by Deepkit Code.

The vendored source is sqlite-vector **1.1.0**, pinned to the Apache-2.0
upstream revision `cfe78ae17c108938b3c87713775b94f7ffe24b87`:
<https://github.com/sqliteai/sqlite-vector/tree/cfe78ae17c108938b3c87713775b94f7ffe24b87>.

Upstream relicensed the project in September 2026. Every vendored upstream
source/header and `API.md` is byte-identical between the previous revision
`235a7c51b18858e1fea9fb2895e115e4cbbd9f7d` and this Apache revision. The
revision refresh changes license/provenance, not upstream runtime behavior;
DKC's local changes below remain in place. The accompanying provenance
inventory records the Git blob and SHA-256 identities. The three locally
modified upstream C files carry explicit DKC modification notices.

The C translation units are compiled at `-O2`. AVX2/FMA and AVX-512 flags are
applied only to their respective x86_64 translation units, matching upstream's
Makefile while preserving runtime CPU dispatch and a portable baseline.

The upstream Apache-2.0 license is included at
`vendor/sqlite-vector/LICENSE.md` (SHA-256
`421717de661f6707b9829bd7d459994af9fc5c7b400335da822a817ffd8a42a1`).
The included FP16 headers independently use the MIT license. They are
byte-identical to Maratyszcza/FP16 revision
`98b0a46bce017382a6351a19577ec43a715b6835`; retain its copyright and license
notice in the distribution's third-party notices bundle.

Local patches to that revision:

- T1.1 adds x86 F16C conversion/accumulation in `distance-avx2.c`.
- T4.7 selects MSVC-compatible CPU intrinsics headers in `distance-cpu.c`.
  The wrapper build also maps POSIX case-insensitive comparisons to the
  corresponding MSVC names on Windows.
- T1.R3 removes the normalized-FLOAT16 shortcut: rounding changes both vector
  norms, so FLOAT16 always uses true cosine, including older registrations with
  `normalized=1`. The AVX2/F16C cosine kernel accumulates dot and both squared
  norms in one pass, with independent accumulators to avoid serial FMA latency.
  Runtime dispatch and non-F16C fallback remain intact. Rust tests compare actual
  decoded FLOAT16 values with an independent scalar f64 cosine oracle (1e-5),
  including self/near ties, negative/extreme inputs and SIMD boundary dimensions.
- T1.R3 makes the exact full-scan heap break distance ties by ascending rowid,
  including ties crossing the top-k boundary. Unscoped Rust scans use its bounded
  native heap; scoped scans use the existing streaming full-scan interface and
  apply SQL predicates before ORDER/LIMIT. A global prefetch cannot guarantee
  scoped top-k. Streaming computes all row distances; no approximate index or
  quantization is used. AVX2/F16C is tested here; AVX-512 and ARM are not claimed.
- T1.R1a installs dkc's 5-second SQLite busy timeout at extension initialization,
  before its internal table is created. Auto-extensions run inside connection
  opening, before the Rust store can set its connection PRAGMAs; without this,
  an observer opening a fresh database can make index startup fail immediately.

- T1.5al adds a deferred DKC initialization entry point: automatic connection
  opening registers functions without creating `_sqliteai_vector`. Store invokes
  the native `dkc_vector_metadata_init()` SQL function inside admission; the
  schema remains defined by the existing native macro. The original eager ABI
  remains available. A final read-only version check clears the expected error
  from the initial missing-function probe, which SQLite checks after automatic
  extension loading. The explicit metadata function is DIRECTONLY.

`scripts/check-vector-init.sh <bundled-sqlite-source-directory>` checks both entry points and 128 open/scan/close lifetimes under
ASan/UBSan on the x86_64 bench machine. Rust storage tests cover read-only opens,
metadata rollback at a native WAL ceiling, capacity refusal and reopening.

DKC adds `dkc_vector_prune()` (DIRECTONLY) for connection-local registrations
whose main-database tables no longer exist. It refuses explicit transactions and
active vector cursors, frees retired resources, and reuses sparse slots without
moving retained contexts. Store calls it after committed table retirement and
before registering newly prepared tables. This is a DKC main-database API, not a
general attached/temporary-table cleanup interface. Native lifecycle tests also
cover active-cursor refusal, rolled-back drops, and 256 retire/reuse cycles per
entry point under ASan/UBSan.
