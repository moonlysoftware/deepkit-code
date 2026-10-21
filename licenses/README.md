# Release license bundle

`third-party-licenses.tar.gz` contains complete license and attribution texts,
plus `third-party-licenses/MANIFEST.json`. It is included in release archives and
retained by the Homebrew formula in its `pkgshare` directory. To read the texts:

```sh
tar -xzf third-party-licenses.tar.gz
```

The inventory covers every registry package in `Cargo.lock`, patched gix-index,
vendored sqlite-vector, native library notices and copied grammar queries.
Optional, build, platform and test dependencies are included conservatively;
this is not a claim that every package is linked into every binary. Original
Deepkit Code licensing is documented in the repository's `LICENSE.md`.

## Verify before packaging

From the repository root, with Python 3.11 or later:

```sh
python3 scripts/release/generate-notices.py --check licenses/third-party-licenses.tar.gz
python3 scripts/release/test_license_bundle.py
```

Verification is offline and read-only. It checks the exact lockfile digest,
archive membership and file hashes, safe regular-file paths, package inventory,
and current license/provenance inputs. It also compares copied native, grammar,
gix and supplement notices directly with their current source bytes. A changed
or missing input requires regeneration; changing an archive's own manifest does
not bypass the source checks. No Cargo registry cache is needed for verification.

## Regenerate after dependency or notice changes

Generation needs the checksum-matching `.crate` archives for all locked packages.
Fetch them, then choose an output directory that does not already exist:

```sh
cargo fetch --locked
python3 scripts/release/generate-notices.py \
  --out target/license-notices-next \
  --bundle licenses/third-party-licenses.tar.gz
```

Review changed dependencies and their terms before regenerating. Keep the new
bundle and its source supplements in the same commit as the relevant lockfile or
notice changes. The generated plain-text directory stays outside version control.
The checked-in archive must remain below 1,000,000 bytes. Fixed tar ownership,
permissions, ordering and timestamps, plus a fixed gzip timestamp, make the
archive reproducible from identical inputs.

`notice-overrides/` restores texts omitted from published crate archives.
`notice-overrides-provenance.json` records pinned upstream sources or the explicit
Apache-2.0 license choice where a dual-licensed crate omitted a standalone text.
`native-notices/` retains native library licenses and provenance, including
sqlite-vector's independently licensed FP16 headers. `model-license-evidence.json`
identifies separately downloaded models; model weights are not packaged here.

The manifest binds all files under `notice-overrides/` and `native-notices/`, their
provenance metadata, and the repository's copied license sources. This README and
the compressed output are not source inputs, avoiding a self-referential hash.
The verifier prints the final archive SHA-256 for release evidence.
