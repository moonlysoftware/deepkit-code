# Release engineering

For the current Unix beta, start with [the beta handover](../../docs/beta-release.md).
`beta-homebrew.yml` builds and tests native archives without publishing them.
`package-beta.py` checks binary provenance and the complete notice bundle;
`brew-formula.py` binds the formula to actual archive checksums. The commands
below describe the separate, unfinished five-platform v1 release path.

Release PRs run cargo-dist planning only. The separate source CI and native
Homebrew workflow perform beta development checks. Full native qualification is
a `local-artifacts-jobs` gate on every automated tag publication attempt,
including prereleases. Global artifacts, archive-size checks, v1 acceptance and
GitHub Release publication wait for that gate; registry publication waits for
GitHub hosting. The full gate still includes Windows runtime and macOS Metal
checks, so an unqualified platform blocks that publication path. The v1 evidence
guard remains required for stable releases. A successful Unix beta PR does not
claim that the five-platform release gate has passed.

This directory contains the target-independent checks and native smoke scripts
for the five v1 release targets. The release itself is configured in
`dist-workspace.toml`; `dist generate` owns `.github/workflows/release.yml` and
uses the reusable workflows in `.github/workflows/` for native builds, the
50 MB download-archive gate, and the conventional `SHA256SUMS` asset.

The target matrix is:

| Target | Inference feature | Native host |
| --- | --- | --- |
| `aarch64-apple-darwin` | `metal` | Apple Silicon macOS 13+ |
| `x86_64-apple-darwin` | CPU | Intel macOS 13+ |
| `x86_64-unknown-linux-musl` | CPU | x86-64 Linux |
| `aarch64-unknown-linux-musl` | CPU | ARM64 Linux |
| `x86_64-pc-windows-msvc` | CPU | Windows 10+ |

`release-platforms.yml` also builds the native GNU/Linux development target.
The Linux/NVIDIA CUDA build remains a development and benchmark gate; it is not
a v1 distribution target.

## Local checks

Run the checks that do not require another operating system:

```sh
bash -n scripts/release/assert-rustls.sh \
  scripts/release/mac-verify.sh scripts/release/smoke.sh \
  scripts/release/zig-musl-tool.sh
python3 scripts/release/test_release_tools.py
python3 scripts/release/test_mac_verify.py
python3 scripts/release/test_zig_musl_tool.py
python3 scripts/release/test_brew_formula.py
python3 scripts/release/test_license_bundle.py
python3 scripts/release/generate-notices.py --check licenses/third-party-licenses.tar.gz
scripts/release/assert-rustls.sh
dist generate --check
dist plan
```

The Linux musl jobs use Zig 0.15.1 for native C/C++ compilation and final
Rust linking. `.github/build-setup.yml` creates target-specific command names
for `scripts/release/zig-musl-tool.sh` and exports `CC`, `CXX` and Cargo linker
bindings for each musl target. The wrapper translates Rust's musl target
spelling for Zig and removes Rust's duplicate self-contained CRT arguments
at the final link. Historical x86-64 artifact and ARM64 cross-build/emulated-smoke
receipts are retained in the development archive under their original
`notes/bench/` paths; see [local records](../../notes/README.md). Those receipts
do not establish current native ARM64 release qualification.

`check-artifact-size.py` requires exactly one `dkc` executable per target and
rejects missing, duplicate, unplanned, or oversized download archives. It
matches each archive to the exact executable artifact in the current
`cargo-dist` plan and measures the full compressed file against the
50,000,000-byte v1 limit. Reports retain uncompressed executable `bytes`
without an installed-size cap, plus `archive_bytes` and an explicit
`size_scope` for the gate. This implements Bart's September 25 acceptance of
PCR-026. The release workflow passes the plan to the
size gate after collecting platform archives and before creating a GitHub
Release. For a local artifact set, pass `--plan-file` with JSON from
`dist plan --output-format=json`.

The generated GitHub Release host job waits for the custom
`release-acceptance.yml` job, and Homebrew/npm publication waits for the host.
On a final tag, the acceptance job requires the complete Phase 4 readiness
audit, including reviewed corpus, benchmark, installer and human evidence.
Prerelease tags can still build candidate artifacts without publishing to
those registries. `dist generate --check` verifies that the generated workflow
matches this dependency order.

`smoke.sh` accepts either an installed `dkc` on `PATH` or `DKC_BIN=/path/to/dkc`;
`smoke.ps1` accepts the installed command or `-DkcPath`. They create an isolated
copy of `fixtures/tiny`, initialize the local profile, explicitly index only L1,
run lexical search, and verify status and queue health. They do not download a
model or use a provider key.

`mac-verify.sh` is the native Apple Silicon acceptance script (Python 3.11+). It builds with
Metal, verifies both pinned GGUFs, records the Metal feature and full GPU-layer
request, measures 2,000 embedding units, gates exactly 50 rerank candidates,
checks post-unload RSS, and runs pinned ripgrep end to end. Three rerank sets
retain their individual results and median p95. Independent probes continue
after a failure; any failed stage or acceptance check makes the script exit
nonzero. A fresh `--out` is required, and the report includes source manifests,
model and binary hashes, and every stage's exit status and log hash. Its output
belongs in the Phase 4 evidence report. `CARGO_TARGET_DIR` is supported.

## Release boundary

The sqlite-vector licensing issue was resolved by adopting an Apache-2.0
upstream revision with byte-identical relevant source; see the provenance in
`crates/sqlite-vector-sys/`. Original DKC code remains MIT OR Apache-2.0.
The crate stays `publish = false`; preparing a beta tap does not authorize
crates.io, npm or a final v1 publication.

When those boundaries are closed, the Phase 4 release sequence is:

1. Set the approved release version and decide each registry separately.
2. Run the full CPU and CUDA quality gates and `dist generate --check`.
3. Push a `v0.9.0-rc.*` tag and inspect every artifact, installer smoke result,
   checksum, attestation, and native Metal result.
4. Restore and review the archived `v1.md`, `plan/06-benchmarks.md` and Phase 4
   evidence record (`notes/reports/phase-4.md`) for final acceptance, or explicitly
   migrate their bound contracts and evidence. The checker refuses missing inputs;
   [archive retrieval](../../notes/README.md#archived-specifications-and-reference-documents)
   explains where they are preserved.
5. Only after human review, tag `v1.0.0` and verify the public installers.

The older `probe.py` and `targets.toml` remain useful feasibility evidence. A
successful probe means its JSON was written; each recorded command exit code is
the build result.
