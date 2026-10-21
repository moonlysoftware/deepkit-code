#!/usr/bin/env python3
"""Package one actual Unix beta binary with provenance and full license texts.

No upload or publication occurs. Dirty builds require an explicit diagnostic flag
and retain their dirty marker; they are not release candidates.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-dirty-diagnostic', action='store_true')
    args = parser.parse_args()
    binary = args.binary.resolve()
    package = tomllib.loads((ROOT/'Cargo.toml').read_text())['package']
    version = subprocess.check_output([str(binary), '--version'], text=True).strip()
    if version != f'dkc {package["version"]}':
        raise SystemExit('binary version differs from Cargo.toml')
    metadata = json.loads(subprocess.check_output([str(binary), 'bench', 'metadata', '--json'], text=True))
    build = metadata['build']
    target = build['target']
    if target not in ('aarch64-apple-darwin', 'x86_64-apple-darwin',
                      'aarch64-unknown-linux-musl', 'x86_64-unknown-linux-musl',
                      'x86_64-unknown-linux-gnu'):
        raise SystemExit('this beta packaging path supports native Unix targets only')
    if build['profile'] != 'release':
        raise SystemExit('package an optimized release/dist binary')
    if str(build['source_dirty']).lower() != 'false' and not args.allow_dirty_diagnostic:
        raise SystemExit('dirty or unknown source; use a clean build before distribution')
    if not args.allow_dirty_diagnostic:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=all', '--',
             'src', 'crates', 'vendor', 'Cargo.toml', 'Cargo.lock', 'build.rs', '.cargo'],
            cwd=ROOT, text=True).strip()
        if build['commit'] != commit or dirty:
            raise SystemExit('binary and clean checkout must identify the same commit')
    subprocess.run([sys.executable, str(ROOT/'scripts/release/generate-notices.py'),
                    '--root', str(ROOT),
                    '--check', str(ROOT/'licenses/third-party-licenses.tar.gz')], check=True)
    data = binary.read_bytes()
    metadata['binary_sha256'] = hashlib.sha256(data).hexdigest()
    metadata['version'] = package['version']
    metadata['diagnostic'] = args.allow_dirty_diagnostic
    entries = [('dkc', data, 0o755),
               ('BUILD.json', (json.dumps(metadata, indent=2)+'\n').encode(), 0o644)]
    for name in ['README.md', 'LICENSE.md', 'LICENSE-MIT', 'LICENSE-APACHE', 'THIRD_PARTY_NOTICES.md']:
        entries.append((name, (ROOT/name).read_bytes(), 0o644))
    entries.append(('third-party-licenses.tar.gz', (ROOT/'licenses/third-party-licenses.tar.gz').read_bytes(), 0o644))
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out/f'deepkit-code-{target}.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for name, content, mode in entries:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = mode
                tar.addfile(member, io.BytesIO(content))
    if archive.stat().st_size > 50_000_000:
        archive.unlink()
        raise SystemExit('compressed archive exceeds the 50 MB release limit')
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps({'archive': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size,
                      'build': build, 'diagnostic': args.allow_dirty_diagnostic}))


if __name__ == '__main__':
    main()
