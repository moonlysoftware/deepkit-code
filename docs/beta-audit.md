# Beta preparation audit — 14 October 2026

The audit started at `a7e45b06ce1c40c7648c44359bb248c83f7b232b` on the
private `bartscheffer/deepkit-code` repository, using preparation branch
`prepare/beta-20261014`. The owner subsequently transferred it to
`moonlysoftware/deepkit-code` and authorized private tap preparation and the
source merge into main. See the [current handover](beta-release.md) for their
completed checks and the remaining public launch steps. Visibility, release
and public package-manager publication remain separate owner actions.

## Coverage

All **734 reachable commits** were inventoried, including branches and tags;
723 were ancestors of the starting HEAD. Automated review covered all **439
commits changing product/tooling paths**, processing 15,657,404 bytes of diffs.
All **9,740 reachable blobs** (277,349,027 bytes) were scanned for common secret
formats, including 143 binary blobs. No reachable blob exceeded 1 MB.

The 20 credential-pattern matches were synthetic URL parsing/redaction cases
in three test locations. No match was found outside those fixture contexts.
That does not establish that every possible secret format is absent. The
[aggregate receipt](../bench/results/beta-20261014/history-scan-summary.json)
records the scope; full inventories, redacted candidates and scanner source
are retained with the external development archive.

This is complete automated coverage of reachable history plus selected deeper
code review, **not a manual review of every historical line**. The deeper review
focused on filesystem writes/deletion, credentials, outbound privacy policy,
HTTP authentication, caps, freshness, release tooling and their tests.
Unreachable/reflog-only objects were outside the history inventory.

## Issues fixed before beta

| Finding | Change and regression coverage |
| --- | --- |
| Configured `clean` paths could delete files outside `.deepkit`. | Shared destination validation and descriptor-relative Unix cleanup. CLI tests cover absolute/traversal paths, linked ancestry/sidecars, valid custom paths and missing directories. |
| Ordinary database opens could initialize or migrate an external SQLite file. | Validate repository destinations before parent creation, SQLite opening or benchmark-ledger access. CLI tests verify external files remain byte-identical and no external directory/report is created. |
| Credential reads could block on a FIFO or allocate an unbounded file. | Regular-file, no-follow, nonblocking Unix open; permission checks and a 1 MiB bounded read. Tests cover links, FIFO, oversize and public permissions. |
| The locked TLS library had four current RustSec advisories. | Update rustls from 0.23.45 to 0.23.46; regenerate and verify dependency notices. |
| sqlite-vector's old licensing provenance blocked redistribution. | Adopt the Apache-2.0 upstream revision whose 19 relevant source/header/API files match the old pin byte for byte. Preserve local modifications and independent FP16 MIT notices. |
| Packaging could use stale/mismatched diagnostic artifacts. | Require clean matching build provenance for distribution, complete verified notices, archive size checks, and checksum-bound formulas. Formula validation rejects unsafe archives and inconsistent embedded build metadata. |

The deletion and external-database issues were reproduced against disposable
files before repair. Credentials were hardened from code review without running
an intentional blocking or resource-exhaustion attack. No additional confirmed
privacy, HTTP-authentication or cap-enforcement bypass emerged from this review.

The existing plain-question lexical fallback work was also finished, preserving
explicit query syntax and identifiers. Its retained experiments and limitations
are described in [the benchmark guide](benchmarks.md).

## Validation receipts

The preparation's final check results and artifact identities are recorded in
[validation.json](../bench/results/beta-20261014/validation.json). Failed attempts
remain in the external/local development records. Baseline results from before
the repairs are not counted as validation of the repairs. Native Homebrew jobs
must pass on the actual advertised platform; Linux does not establish Mac
inference performance or Windows support.

A subsequent push CI run on documentation-only `c5f69eb` missed the polling
watcher's 1.5-second delete-publication deadline; its parallel PR CI and native
Homebrew runs passed. This remains a recorded miss, not evidence of a confirmed
lost event or a repaired defect. Five unchanged focused tests passed locally
(polling deletes 743–757 ms), so the hosted failure remains unreproduced.
See the [follow-up receipt](../bench/results/beta-20261014/followup-ci.json).
The approved document cleanup separately passes 100 benchmark-tool and 37
release-tool tests in a clean snapshot without the archived files; license
verification, local documentation links and the stable-release refusal also pass.
See [cleanup checks](../bench/results/beta-20261014/cleanup-verification.json).

## Cleanup and remaining boundaries

The development archive contains a verified all-ref Git bundle, the pre-audit
working-tree patch and the development documents/logs. Selected original
benchmark evidence is retained under `bench/results/`; reusable tools and test
fixtures were moved out of `notes/`. Routine local run output stays ignored.
Historical Git commits are preserved. Removing current-tree files does not
make historical private material safe to publish.

On 14 October the owner approved review of PR #3, retained history and the
prepared protected-document cleanup. The exact approved patch removes 33
internal plan/specification/reference files from the current tree and replaces
AGENTS.md with the reviewed contributor guide. These documents remain in Git
history and the verified external archive; [retrieval instructions](../notes/README.md)
explain historical source references. No history rewrite, merge, transfer or
publication is included in that approval. Review Actions logs/artifacts before
public visibility. The beta does not waive the unfinished full v1 matrix,
native-platform checks or missed performance targets.

Static path escapes are now refused, but SQLite preflight is not an atomic
reservation against a hostile process concurrently swapping filesystem paths.
Windows repository operations remain unsupported. Large or malicious compressed
documents also deserve further resource-bound testing; no parser exploit was
confirmed here. These limits must not be described as a complete security audit.

The exact transfer and release sequence is in [the beta handover](beta-release.md).
