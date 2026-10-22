# Beta release and Homebrew handoff

[Setup and usage](usage.md) · [Measured results](benchmarks.md)

The next candidate version is **0.1.0-beta.2**. This is a Mac/Linux beta, with
Windows setup, database access and source opening still unqualified. A working
build does not complete the v1 performance, corpus or installer requirements.

**22 October 2026:** the owner made the current source repository public.
The Homebrew tap and history repository remain private; no public binary release
is published yet. Use the [current installation instructions](usage.md) for
source builds or the access-required Homebrew preview. The records below retain
the earlier preparation and release plan.

## Public Homebrew candidate

The owner requested preparation of public Homebrew distribution. The first
public formula targets **Apple Silicon macOS**; Linux GNU remains diagnostic.
Version `0.1.0-beta.2` sorts after the private `0.1.0-beta.1_1` preview.

The candidate updates OfficeOxide to 0.1.15, ZIP to 9.0.5 and rustls-webpki to
0.103.16 to address the dependency advisories reported by the first public-repo
CI run. Office extraction fingerprints change so existing Office documents are
reprocessed with the updated parser. Dependency notices are regenerated from
the exact lockfile. The native Apple Silicon build now runs on macOS 15; its
compiled deployment floor remains macOS 13.

Before publication, require passing source checks and a clean native build at
the exact candidate commit, then verify the archive's provenance, checksums,
licenses and Homebrew install/upgrade behavior. Generate the public formula
from those verified bytes using GitHub release URLs. Keep the history repository
private and the existing preview available until the new release is ready.
Publishing the prerelease assets and changing the tap's visibility require
owner approval of the prepared release. Do not advertise anonymous installation
until the published download and tap work without credentials.

## Repository split — 21 October 2026

The original private repository is now `moonlysoftware/deepkit-code-history`.
Its commits, PRs and Actions artifacts remain there. The new private
`moonlysoftware/deepkit-code` starts with one snapshot commit of the current
code; this is not a public release or a new qualification result. Existing
Homebrew preview artifacts are still downloaded from the history repository.
The preparation record below describes the earlier development and receipts.

## Earlier preparation record

| Item | State |
| --- | --- |
| User documentation and retained benchmark evidence | Prepared; see [benchmarks](benchmarks.md). |
| External development backup | Prepared and verified; retain the archive and its checksums outside the public repository. |
| Owner review and protected-document cleanup | Approved 14 October: 33 internal documents archived and the prepared contributor guide installed. Git history retained. |
| Source integration | PR #3 merged into `main` as `e1a1777` after all final checks; all 97 preparation commits preserved. [Integration receipt](../bench/results/beta-20261014/main-integration.json). |
| Final CPU/CUDA source checks | Passed: 1,399 tests in each configuration; ignored tests remain excluded. Formatting, strict Clippy and dependency checks passed. |
| Hosted Rust quality checks | PR and push CI passed at reviewed head `c2477b6`: each 1,399 passed, 17 ignored, zero failed. The earlier `c5f69eb` watcher deadline miss remains recorded in the [follow-up receipt](../bench/results/beta-20261014/followup-ci.json); current passes do not establish a repair. |
| Bounded vague-query diagnostic | Complete: 30 calls, ten reviewed questions; [results and limits](benchmarks.md#current-semantic-search-diagnostic). |
| Local Linux archive and Homebrew install/test | Passed install, formula test, uninstall, install, reinstall, test and uninstall. Diagnostic archive: 29,279,198 bytes. |
| Native Apple Silicon Homebrew check | Passed the complete install/test/reinstall cycle in [PR #3's native workflow](https://github.com/moonlysoftware/deepkit-code-history/actions/runs/38604704711). Clean archive: 27,565,669 bytes. |
| Hosted Linux GNU Homebrew diagnostic | Passed the same lifecycle; clean archive: 28,959,976 bytes. This does not qualify the portable musl package. |
| Repository transfer | Completed by the owner; `moonlysoftware/deepkit-code` remains private and Git/SSH/admin access is verified. |
| Private Homebrew tap | [Prepared and tested](../bench/results/beta-20261014/private-tap.json): actual Linux install/search/reinstall passed; 25 helper tests passed on Ubuntu/macOS CI. Invited Mac installation remains next. |
| Public release and public tap installation | Pending; no public release/tag or visibility change. |

The beta packaging helpers bind real archives to their SHA-256 checksums and
include provenance and dependency notices. A distributable archive must come
from the reviewed clean commit and pass its native smoke test. The complete
download archive limit is 50,000,000 bytes; installed executable size is measured
without a separate cap. Dirty-build diagnostics are not release artifacts.

The owner approved review of [PR #3](https://github.com/moonlysoftware/deepkit-code-history/pull/3),
retained history and the prepared internal-document cleanup on 14 October.
That cleanup is applied; the validated product and packaging code stays
unchanged. Cleanup checks pass, including 100 benchmark-tool and 37 release-tool
tests in a checkout without historical documents. The owner subsequently
transferred the private repository, requested private tap preparation and
authorized updating main. PR #3 merged as `e1a1777` after both Rust CI runs,
both native Homebrew lifecycles and release planning passed at `c2477b6`.
The merge tree exactly matches that reviewed head. Earlier cancelled `cee7a68`
runs contribute no qualification; the earlier watcher miss remains unresolved.
Public publication remains an owner action.

The new `beta-homebrew.yml` workflow passed for native Apple Silicon and the
Linux GNU diagnostic. Both archives' checksums, embedded clean commit, source
hash and complete notices were independently verified. The tested checkout was
PR merge commit `41efee5b879fb2353a5407516aa90a957fba78ee`, not a release tag.
The Mac runner used macOS 14; Homebrew emitted an unsupported-OS warning and
classified it as Tier 3. The install smoke passed, but a fresh test on a currently
supported Mac remains part of the public launch check below.

Start the public tap with **Apple Silicon only**. Add Intel Mac or Linux after
their actual distribution packages pass. These jobs did not test model inference.
Exact local results and artifact identities are in the
[validation receipt](../bench/results/beta-20261014/validation.json) and
[native artifact receipt](../bench/results/beta-20261014/native-homebrew.json).

## Your remaining steps

1. **Source integration is complete.** PR #3 merged on 14 October with its final
   checks green and all preparation history retained. The [integration receipt](../bench/results/beta-20261014/main-integration.json)
   records the reviewed head and merge. Historical files remain in older commits.
   Review Actions logs and artifacts before public
   visibility, and keep the verified external backup. See
   [GitHub's visibility guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility).
2. **Private repository split is complete.** Current development uses the fresh
   `moonlysoftware/deepkit-code`; earlier history and preview artifacts are in
   `moonlysoftware/deepkit-code-history`. The private tap remains at
   `moonlysoftware/homebrew-tap`. Preview installation requires access to the
   tap and history repositories; follow the tap README for invited testing.
3. **Verify organization settings and the native beta build.** Update local
   remotes, check collaborator access, Actions permissions and any required
   secrets. Run the new beta workflow manually on the reviewed `main` commit.
   PR checks can build a temporary merge commit; use the resulting manual-run
   artifacts whose embedded commit is the exact commit you will tag.
   Record the actual supported target(s), complete archive hashes and install
   results. Do not advertise untested targets.
4. **Publish the reviewed beta assets after owner approval.** The source
   repository is already public. Temporarily disable the generated **Release** workflow
   before creating the beta tag: its default five-platform route still includes
   unsupported Windows smoke tests. Create `v0.1.0-beta.2` at the exact reviewed
   commit as a GitHub **prerelease**, upload the qualified archives and checksums,
   then re-enable that workflow. This is a manual beta publication; keep the
   full v1 checks intact. GitHub documents [disabling and enabling workflows](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows).
5. **Prepare the existing private tap for public installation.** Download the
   final uploaded release archives, verify their checksums against the retained
   build receipts, and generate `Formula/dkc.rb` from those exact bytes. The
   example below selects Apple Silicon only. Replace the private cache-based
   preview formula with the reviewed release formula in the tap
   and run its installation test on each advertised native platform before
   merging it. See [Homebrew's tap guide](https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap).
6. **Check the public installation and announce the beta.** From a fresh native
   machine, install from the published tap, run `brew test`, and try the
   [model-free first run](usage.md#first-run-no-model-or-api-key) in a small
   repository. Then publish the installation command with the supported
   platforms and [known benchmark limitations](benchmarks.md).

For an existing checkout that still contains the old history, point its remote
at the history repository. Clone current development separately; do not push or
merge the old ancestry into the fresh repository:

```sh
git remote set-url origin git@github.com:moonlysoftware/deepkit-code-history.git
git clone git@github.com:moonlysoftware/deepkit-code.git ../deepkit-code-fresh
```

The temporary workflow switch is explicit; run it only for the reviewed manual
beta publication, leaving ordinary CI enabled:

```sh
gh workflow disable release.yml --repo moonlysoftware/deepkit-code
# Create the reviewed beta tag/prerelease and upload its verified assets.
gh workflow enable release.yml --repo moonlysoftware/deepkit-code
```

From the DKC checkout, with final uploaded archives downloaded to
`./release-assets` (Python 3.11+):

```sh
python3 scripts/release/brew-formula.py \
  --artifacts ./release-assets \
  --version 0.1.0-beta.2 \
  --targets aarch64-apple-darwin \
  --out /tmp/dkc-beta-formula.rb
```

Copy that reviewed file to `Formula/dkc.rb` in the existing tap. The generator refuses
missing selected archives and existing output files. It computes real hashes;
there are no placeholder checksums to replace. `--local-test` uses local file
URLs for preparation and must not be used for the published formula.

Only after the release assets and tap formula are public:

```sh
brew install moonlysoftware/tap/dkc
brew test moonlysoftware/tap/dkc
dkc --version
```

This is a project-maintained tap. A submission to Homebrew's central repository
is a separate step. Automatic prerelease registry publication remains disabled
with `publish-prereleases=false`; publishing this beta does not change that.

## Licensing and model downloads

Original DKC code remains **MIT OR Apache-2.0**. Dependencies retain their own
licenses; distribute the accompanying [notices](../THIRD_PARTY_NOTICES.md) and
license bundle with every package. The original project license does not
relicense reference material or dependencies. Model weights are not bundled;
their downloads and terms remain separate from the binary.
