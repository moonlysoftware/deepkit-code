#!/usr/bin/env python3
"""Generate or verify the release notice bundle; see v1.md §12.

Generation reads checksum-verified Cargo.lock crate archives. Verification needs
only the checkout and bundle, so release jobs can reject stale notices offline.
The inventory includes optional, platform, build and test dependencies; it does
not claim that every listed package is linked into every binary.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
import tomllib

NOTICE = re.compile(r'^(?:licen[sc]e|copying|copyright|notice)(?:[._-].*)?$', re.I)
APACHE_SQLITE_SHA = '421717de661f6707b9829bd7d459994af9fc5c7b400335da822a817ffd8a42a1'
SQLITE_LICENSE = 'crates/sqlite-vector-sys/vendor/sqlite-vector/LICENSE.md'
REGISTRY = 'registry+https://github.com/rust-lang/crates.io-index'
BUNDLE_ROOT = 'third-party-licenses'
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 64_000_000
MAX_FILES = 10_000


def digest(data):
    """Return the content identity used for both source and packaged notices."""
    return hashlib.sha256(data).hexdigest()


def safe_path(value):
    """Reject ambiguous or escaping archive paths before reading or writing them."""
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or '\0' in value:
        raise ValueError(f'unsafe notice path {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) != value or value == '.':
        raise ValueError(f'unsafe notice path {value!r}')
    return path


def source_inputs(root, supplements):
    """Bind copied notices and their reviewed provenance, excluding bundle/README."""
    repo_files = [root/SQLITE_LICENSE]
    repo_files.extend(root/'vendor/gix-index'/name for name in ('LICENSE-APACHE', 'LICENSE-MIT', 'PATCH.md', 'Cargo.toml'))
    repo_files.extend(sorted((root/'src/treat/grammar_queries').glob('*.LICENSE')))
    # These inventories describe the reviewed source revisions when present.
    for name in ('crates/sqlite-vector-sys/vendor-provenance.json',
                 'src/treat/grammar_queries/sources.json'):
        if (root/name).exists():
            repo_files.append(root/name)
    supplement_files = [supplements/name for name in
                        ('notice-overrides-provenance.json', 'model-license-evidence.json')]
    for name in ('notice-overrides', 'native-notices'):
        directory = supplements/name
        if not directory.is_dir():
            raise ValueError(f'missing supplement directory: {name}')
        supplement_files.extend(file for file in directory.rglob('*') if file.is_file() or file.is_symlink())
    records = []
    for kind, base, files in (('repository', root, repo_files),
                              ('supplement', supplements, supplement_files)):
        for file in files:
            if file.is_symlink() or not file.is_file():
                raise ValueError(f'notice source must be a regular file: {file}')
            records.append({'kind': kind, 'path': file.relative_to(base).as_posix(),
                            'sha256': digest(file.read_bytes())})
    return sorted(records, key=lambda row: (row['kind'], row['path']))


def registry_packages(lock):
    """Require explicit review for dependency sources outside crates.io."""
    result = []
    for package in sorted(lock['package'], key=lambda p: (p['name'], p['version'])):
        source = package.get('source')
        if not source:
            continue
        if source != REGISTRY:
            raise ValueError(f'new non-registry dependency needs review: {package["name"]} {source}')
        result.append(package)
    return result


def generate(root, out, supplements, cargo_home):
    """Copy complete notice texts with portable provenance into a fresh directory."""
    if out.exists():
        raise ValueError('output exists; choose a fresh output directory')
    lock_bytes = (root/'Cargo.lock').read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    inputs = source_inputs(root, supplements)
    manifest = {'schema_version': 1,
                'scope': 'all Cargo.lock packages; includes optional/build/test dependencies',
                'cargo_lock_sha256': digest(lock_bytes), 'source_inputs': inputs,
                'packages': [], 'files': []}
    out.mkdir(parents=True)

    def emit(relative, data, source, source_input=None):
        path = safe_path(relative)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f'notice exceeds file limit: {relative}')
        target = out.joinpath(*path.parts)
        if target.exists():
            raise ValueError(f'duplicate notice path {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        record = {'path': str(path), 'sha256': digest(data), 'source': source}
        if source_input:
            record['source_input'] = source_input
        manifest['files'].append(record)

    def emit_source(relative, file, kind, base, source):
        emit(relative, file.read_bytes(), source,
             {'kind': kind, 'path': file.relative_to(base).as_posix()})

    for package in registry_packages(lock):
        key = f"{package['name']}-{package['version']}"
        archives = sorted((cargo_home/'registry/cache').glob(f'*/{key}.crate'))
        matching = [p for p in archives if digest(p.read_bytes()) == package['checksum']]
        if not matching:
            raise ValueError(f'missing checksum-matching archive {key}; cargo fetch --locked first')
        notice_count = 0
        with tarfile.open(fileobj=io.BytesIO(matching[0].read_bytes()), mode='r:gz') as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            cargo = tomllib.loads(tar.extractfile(members[f'{key}/Cargo.toml']).read().decode())['package']
            declared = cargo.get('license')
            if not declared:
                raise ValueError(f'license expression missing for {key}; review before bundling')
            if 'MPL' in declared:
                raise ValueError(f'new source-disclosure license requires review: {key}: {declared}')
            for name, member in sorted(members.items()):
                rel = safe_path(name).relative_to(key)
                if NOTICE.fullmatch(rel.name) or str(rel) == cargo.get('license-file'):
                    if member.size > MAX_FILE_BYTES:
                        raise ValueError(f'oversized notice {key}/{rel}')
                    emit(f'crates/{key}/{rel}', tar.extractfile(member).read(),
                         f'crates.io/{key} sha256:{package["checksum"]}')
                    notice_count += 1
        override = supplements/'notice-overrides'/key
        if override.is_dir():
            for file in sorted(override.rglob('*')):
                if file.is_file():
                    emit_source(f'crates/{key}/upstream/{file.relative_to(override).as_posix()}',
                                file, 'supplement', supplements,
                                f'reviewed override; see notice-overrides-provenance.json: {key}')
                    notice_count += 1
        if notice_count == 0:
            raise ValueError(f'no license text for {key}; supply reviewed pinned override')
        manifest['packages'].append({'name': package['name'], 'version': package['version'],
                                     'license': declared, 'crate_sha256': package['checksum']})

    for file in sorted((root/'src/treat/grammar_queries').glob('*.LICENSE')):
        emit_source(f'grammar-queries/{file.name}', file, 'repository', root,
                    file.relative_to(root).as_posix())
    for name in ('LICENSE-APACHE', 'LICENSE-MIT', 'PATCH.md'):
        file = root/'vendor/gix-index'/name
        emit_source(f'vendored/gix-index/{name}', file, 'repository', root,
                    file.relative_to(root).as_posix())
    gix = tomllib.loads((root/'vendor/gix-index/Cargo.toml').read_text())['package']
    manifest['packages'].append({'name': 'gix-index', 'version': gix['version'],
                                 'license': gix['license'], 'source': 'vendor/gix-index/PATCH.md'})
    if digest((root/SQLITE_LICENSE).read_bytes()) != APACHE_SQLITE_SHA:
        raise ValueError('sqlite-vector Apache provenance/license does not match approved revision')
    emit_source('vendored/sqlite-vector/LICENSE.md', root/SQLITE_LICENSE, 'repository', root,
                'sqliteai/sqlite-vector@cfe78ae17c108938b3c87713775b94f7ffe24b87/LICENSE.md')
    for file in sorted((supplements/'native-notices').rglob('*')):
        if file.is_file() and file.name != 'Pear-README.md':
            emit_source(f'native/{file.relative_to(supplements/"native-notices").as_posix()}',
                        file, 'supplement', supplements,
                        'reviewed native source; see native-provenance metadata')
    for name in ('notice-overrides-provenance.json', 'model-license-evidence.json'):
        emit_source(name, supplements/name, 'supplement', supplements, 'reviewed pinned upstream metadata')
    if inputs != source_inputs(root, supplements) or lock_bytes != (root/'Cargo.lock').read_bytes():
        raise ValueError('notice inputs changed during generation')
    (out/'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest


def write_bundle(directory, destination):
    """Write a deterministic archive atomically, with no timestamps or host names."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', filename='', mtime=0, compresslevel=9) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            for file in sorted(directory.rglob('*')):
                if file.is_symlink():
                    raise ValueError(f'notice source must not be a symlink: {file}')
                if not file.is_file():
                    continue
                relative = safe_path(file.relative_to(directory).as_posix())
                data = file.read_bytes()
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError(f'oversized bundle member: {relative}')
                member = tarfile.TarInfo(f'{BUNDLE_ROOT}/{relative}')
                member.size = len(data)
                member.mode = 0o644
                member.mtime = member.uid = member.gid = 0
                member.uname = member.gname = ''
                archive.addfile(member, io.BytesIO(data))
    data = buffer.getvalue()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('compressed notice bundle exceeds 1 MB repository limit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.chmod(0o644)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest(data)


def check_bundle(bundle, root, supplements):
    """Verify membership, hashes and current source binding without Cargo or writes."""
    if bundle.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('compressed notice bundle exceeds 1 MB repository limit')
    contents = {}
    total = 0
    with tarfile.open(bundle, mode='r:gz') as archive:
        for member in archive:
            path = safe_path(member.name)
            if len(path.parts) < 2 or path.parts[0] != BUNDLE_ROOT or not member.isfile():
                raise ValueError(f'unsafe bundle member: {member.name}')
            relative = path.relative_to(BUNDLE_ROOT).as_posix()
            if relative in contents:
                raise ValueError(f'duplicate bundle member: {relative}')
            total += member.size
            if member.size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES or len(contents) >= MAX_FILES:
                raise ValueError('notice bundle exceeds inspection limits')
            contents[relative] = archive.extractfile(member).read()
    if 'MANIFEST.json' not in contents:
        raise ValueError('missing MANIFEST.json')
    manifest = json.loads(contents.pop('MANIFEST.json'))
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported notice manifest version')
    lock_bytes = (root/'Cargo.lock').read_bytes()
    if manifest.get('cargo_lock_sha256') != digest(lock_bytes):
        raise ValueError('Cargo.lock changed; regenerate the notice bundle')
    inputs = source_inputs(root, supplements)
    if manifest.get('source_inputs') != inputs:
        raise ValueError('notice source inputs changed; regenerate the notice bundle')
    input_hashes = {(row['kind'], row['path']): row['sha256'] for row in inputs}
    records = manifest.get('files', [])
    listed = {}
    for row in records:
        path = safe_path(row['path']).as_posix()
        if path in listed:
            raise ValueError(f'duplicate manifest member: {path}')
        listed[path] = row
    if set(listed) != set(contents):
        raise ValueError('bundle membership differs from the notice manifest')
    for path, row in listed.items():
        if digest(contents[path]) != row['sha256']:
            raise ValueError(f'notice hash mismatch: {path}')
        if 'source_input' in row:
            source = row['source_input']
            if input_hashes.get((source['kind'], source['path'])) != row['sha256']:
                raise ValueError(f'notice differs from current source: {path}')
    expected = {(p['name'], p['version'], p['checksum']) for p in
                registry_packages(tomllib.loads(lock_bytes.decode()))}
    actual = [(p['name'], p['version'], p['crate_sha256']) for p in
              manifest.get('packages', []) if 'crate_sha256' in p]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('package inventory differs from Cargo.lock')
    # Check source copies independently of the archive's own manifest: changing
    # both a license and its recorded digest must not bypass current-source checks.
    required_sources = {
        'vendored/sqlite-vector/LICENSE.md': ('repository', SQLITE_LICENSE),
    }
    for name in ('LICENSE-APACHE', 'LICENSE-MIT', 'PATCH.md'):
        required_sources[f'vendored/gix-index/{name}'] = ('repository', f'vendor/gix-index/{name}')
    for file in (root/'src/treat/grammar_queries').glob('*.LICENSE'):
        required_sources[f'grammar-queries/{file.name}'] = ('repository', file.relative_to(root).as_posix())
    for name in ('notice-overrides-provenance.json', 'model-license-evidence.json'):
        required_sources[name] = ('supplement', name)
    for file in (supplements/'native-notices').rglob('*'):
        if file.is_file() and file.name != 'Pear-README.md':
            required_sources[f'native/{file.relative_to(supplements/"native-notices").as_posix()}'] = (
                'supplement', file.relative_to(supplements).as_posix())
    for p in registry_packages(tomllib.loads(lock_bytes.decode())):
        key = f"{p['name']}-{p['version']}"
        prefix = f'crates/{key}/'
        if not any(name.startswith(prefix) for name in listed):
            raise ValueError(f'package has no included notice: {key}')
        override = supplements/'notice-overrides'/key
        if override.is_dir():
            for file in override.rglob('*'):
                if file.is_file():
                    required_sources[f'{prefix}upstream/{file.relative_to(override).as_posix()}'] = (
                        'supplement', file.relative_to(supplements).as_posix())
    for path, source in required_sources.items():
        if path not in contents:
            raise ValueError('required source notice missing from bundle')
        if digest(contents[path]) != input_hashes[source]:
            raise ValueError(f'notice differs from current source: {path}')
    if digest(contents['vendored/sqlite-vector/LICENSE.md']) != APACHE_SQLITE_SHA:
        raise ValueError('sqlite-vector Apache license does not match approved revision')
    return {'packages': len(manifest['packages']), 'notice_files': len(contents),
            'cargo_lock_sha256': digest(lock_bytes), 'bundle_sha256': digest(bundle.read_bytes()),
            'bundle_bytes': bundle.stat().st_size}


def main():
    """Expose generation and read-only verification as separate CLI modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path, help='Fresh directory for generated plain-text notices')
    mode.add_argument('--check', type=Path, metavar='BUNDLE', help='Verify an existing bundle without generation')
    parser.add_argument('--bundle', type=Path, help='Also write a deterministic tar.gz after generation')
    parser.add_argument('--supplements', type=Path, default=Path(__file__).resolve().parents[2]/'licenses')
    parser.add_argument('--cargo-home', type=Path,
                        default=Path(os.environ.get('CARGO_HOME', str(Path.home()/'.cargo'))))
    args = parser.parse_args()
    if args.check and args.bundle:
        parser.error('--bundle requires --out')
    try:
        if args.check:
            report = check_bundle(args.check, args.root, args.supplements)
        else:
            manifest = generate(args.root, args.out, args.supplements, args.cargo_home)
            report = {'packages': len(manifest['packages']), 'notice_files': len(manifest['files']),
                      'output': str(args.out), 'cargo_lock_sha256': manifest['cargo_lock_sha256']}
            if args.bundle:
                write_bundle(args.out, args.bundle)
                report.update(check_bundle(args.bundle, args.root, args.supplements))
        print(json.dumps(report, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError, EOFError, tarfile.TarError) as error:
        parser.exit(1, f'notice bundle: {error}\n')


if __name__ == '__main__':
    main()
