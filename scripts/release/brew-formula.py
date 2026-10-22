#!/usr/bin/env python3
"""Prepare a checksum-bound beta tap formula from actual native archives.

This writes a local file only. It does not create a tap, upload an artifact or
publish a release. --local-test uses file URLs for testing an unpublished build.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

PLATFORMS = {
    "aarch64-apple-darwin": ("on_macos", "on_arm"),
    "x86_64-apple-darwin": ("on_macos", "on_intel"),
    "aarch64-unknown-linux-musl": ("on_linux", "on_arm"),
    "x86_64-unknown-linux-musl": ("on_linux", "on_intel"),
    "x86_64-unknown-linux-gnu": ("on_linux", "on_intel"),
}


def quote(value: str) -> str:
    """Use literal Ruby strings so paths cannot interpolate Ruby expressions."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def inspect_archive(path: Path, version: str | None = None, target: str | None = None,
                    local_test: bool = False) -> str:
    """Validate archive layout and any explicit build provenance before publication."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if not members or any(
            PurePosixPath(member.name).is_absolute()
            or ".." in PurePosixPath(member.name).parts
            or not (member.isfile() or member.isdir())
            for member in members
        ):
            raise ValueError(f"unsafe or empty archive: {path.name}")
        names = [PurePosixPath(member.name) for member in members]
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        binaries = [member for member in members if PurePosixPath(member.name).name == "dkc"]
        if len(binaries) != 1 or not binaries[0].isfile() or not binaries[0].mode & 0o111:
            raise ValueError("archive must contain one executable dkc")
        parent = PurePosixPath(binaries[0].name).parent
        if len(parent.parts) > 1 or any(
            parent != PurePosixPath('.') and PurePosixPath(member.name) != parent
            and parent not in PurePosixPath(member.name).parents
            for member in members
        ):
            raise ValueError("archive must be flat or use one shared top-level directory")
        files = {PurePosixPath(member.name) for member in members if member.isfile()}
        for name in ("LICENSE-MIT", "LICENSE-APACHE", "LICENSE.md", "THIRD_PARTY_NOTICES.md"):
            if parent / name not in files:
                raise ValueError(f"archive is missing {name}")
        if parent / "third-party-licenses.tar.gz" not in files:
            raise ValueError("archive is missing the third-party license bundle")
        build_member = next((member for member in members
                             if PurePosixPath(member.name) == parent / "BUILD.json"), None)
        if build_member is not None:
            if not build_member.isfile() or build_member.size > 1_000_000:
                raise ValueError("BUILD.json must be a bounded regular file")
            try:
                metadata = json.loads(archive.extractfile(build_member).read())
            except (ValueError, UnicodeError) as error:
                raise ValueError("BUILD.json is invalid JSON") from error
            if not isinstance(metadata, dict) or not isinstance(metadata.get("build"), dict):
                raise ValueError("BUILD.json is missing build provenance")
            build = metadata["build"]
            if version is not None and metadata.get("version") != version:
                raise ValueError("BUILD.json version differs from requested formula version")
            if target is not None and build.get("target") != target:
                raise ValueError("BUILD.json target differs from selected archive target")
            if not local_test and (
                metadata.get("diagnostic") is not False
                or str(build.get("source_dirty")).lower() != "false"
            ):
                raise ValueError("diagnostic or dirty build cannot be used by a public formula")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def generate(directory: Path, version: str, targets: list[str], local_test: bool) -> str:
    """Bind only explicitly selected, present platforms; missing archives fail."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("version must be a release SemVer without a v prefix")
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("select at least one unique target")
    rows = []
    seen = set()
    for target in targets:
        if target not in PLATFORMS or (target.endswith("-gnu") and not local_test):
            raise ValueError(f"unsupported public target: {target}")
        platform = PLATFORMS[target]
        if platform in seen:
            raise ValueError("multiple archives for the same operating system and CPU")
        seen.add(platform)
        archive = directory / f"deepkit-code-{target}.tar.gz"
        digest = inspect_archive(archive, version, target, local_test)
        url = archive.resolve().as_uri() if local_test else (
            f"https://github.com/moonlysoftware/deepkit-code/releases/download/v{version}/{archive.name}"
        )
        rows.append((platform, url, digest))
    lines = [
        '# Generated by scripts/release/brew-formula.py; review before publishing.',
        'require "json"', '', 'class Dkc < Formula',
        '  desc "Local hybrid code and document search for IDE agents"',
        '  homepage "https://github.com/moonlysoftware/deepkit-code"',
        f'  version {quote(version)}',
        '  license any_of: ["MIT", "Apache-2.0"]', '',
    ]
    if targets == ["aarch64-apple-darwin"]:
        # Match the native beta's MACOSX_DEPLOYMENT_TARGET=13.0 and avoid
        # presenting an Apple Silicon-only archive to other platforms.
        lines.extend([
            '  depends_on macos: :ventura',
            '  depends_on arch: :arm64', '',
        ])
    for (os_block, cpu_block), url, digest in rows:
        lines.extend([
            f'  {os_block} do', f'    {cpu_block} do',
            f'      url {quote(url)}', f'      sha256 "{digest}"',
            '    end', '  end', '',
        ])
    lines.extend([
        '  def install', '    bin.install "dkc"',
        '    pkgshare.install "THIRD_PARTY_NOTICES.md", "third-party-licenses.tar.gz"',
        '  end', '',
        '  test do',
        '    (testpath/"retry.rs").write "pub fn should_retry(attempt: u32) -> bool { attempt < 3 }\\n"',
        '    system "git", "init", "--quiet"',
        '    system "git", "add", "retry.rs"',
        '    system "git", "-c", "user.name=DKC Homebrew Test", "-c",',
        '           "user.email=homebrew@example.invalid", "commit", "--quiet", "-m", "fixture"',
        '    ENV["DKC_MODELS_DIR"] = (testpath/"models").to_s',
        '    system bin/"dkc", "init", "--profile", "local", "--yes", "--json"',
        '    system bin/"dkc", "index", "--level", "1", "--json"',
        '    result = JSON.parse(shell_output("#{bin}/dkc search should_retry --mode lexical --no-rerank --json"))',
        '    assert_equal false, result.fetch("index").fetch("partial")',
        '    assert result.fetch("hits").any? { |hit| hit.fetch("path") == "retry.rs" }',
        '    status = JSON.parse(shell_output("#{bin}/dkc status --json"))',
        '    assert_operator status.fetch("files"), :>=, 1',
        '    assert_operator status.fetch("units"), :>=, 1',
        '    assert_equal 0, status.fetch("queue").fetch("dead")',
        '    assert_path_exists testpath/".deepkit/code.db"',
        '  end', 'end', '',
    ])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--targets', nargs='+', required=True, choices=PLATFORMS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--local-test', action='store_true')
    args = parser.parse_args()
    rendered = generate(args.artifacts, args.version, args.targets, args.local_test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        output.write(rendered)
    print(args.out)


if __name__ == '__main__':
    main()
