#!/usr/bin/env python3
"""Provision pinned full Git clones without changing existing corpus worktrees."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          stdout=subprocess.PIPE, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--only', help='comma-separated manifest names')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[2] / 'bench/corpus.toml')
    args = parser.parse_args()
    manifest = tomllib.loads(args.manifest.read_text())
    repos = manifest['repo']
    wanted = set(args.only.split(',')) if args.only else None
    names = [repo['name'] for repo in repos]
    if len(names) != len(set(names)):
        raise ValueError('duplicate repository names in manifest')
    if wanted is not None and (not all(wanted) or wanted - set(names)):
        raise ValueError(f'unknown/empty repository selection: {sorted(wanted - set(names))}')
    base = Path(os.environ.get('DKC_CORPUS_DIR', '/data/dkc-corpus')).expanduser().resolve()
    results = []
    for repo in repos:
        name = repo['name']
        if wanted is not None and name not in wanted:
            continue
        fixture = repo.get('fixture')
        if repo['url'] == 'local' and 'path' not in repo and fixture is None:
            results.append({'name': name, 'status': 'local-not-cloned'})
            continue
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
            raise ValueError(f'unsafe repository name: {name}')
        rev = repo['rev']
        if not re.fullmatch(r'[0-9a-f]{40}', rev):
            raise ValueError(f'{name}: expected a full pinned SHA')
        root = Path(os.path.expandvars(repo.get('path', str(base / name)))).expanduser()
        if not root.exists():
            if fixture is not None:
                source = Path(os.path.expandvars(fixture)).expanduser()
                if not source.is_absolute():
                    source = (args.manifest.resolve().parent / source).resolve()
                if not source.is_dir():
                    raise ValueError(f'{name}: fixture source is not a directory: {source}')
                if args.verify:
                    raise ValueError(f'{name}: missing {root}')
                root.parent.mkdir(parents=True, exist_ok=True)
                temporary = root.with_name(root.name + '.fetching')
                if temporary.exists():
                    raise ValueError(f'{name}: interrupted fixture at {temporary}; inspect it before retrying')
                shutil.copytree(source, temporary)
                subprocess.run(['git', '-C', str(temporary), 'init', '-q'], check=True)
                subprocess.run(['git', '-C', str(temporary), 'add', '-A'], check=True)
                environment = dict(os.environ)
                environment.update({
                    'GIT_AUTHOR_DATE': '2000-01-01T00:00:00Z',
                    'GIT_COMMITTER_DATE': '2000-01-01T00:00:00Z',
                })
                subprocess.run([
                    'git', '-C', str(temporary), '-c', 'user.name=Deepkit Corpus',
                    '-c', 'user.email=corpus@deepkit.invalid', 'commit', '-qm',
                    f'Pin {name} corpus fixture'
                ], check=True, env=environment)
                actual = git(temporary, 'rev-parse', 'HEAD')
                if actual != rev:
                    raise ValueError(f'{name}: generated fixture is {actual}, expected {rev}')
                temporary.rename(root)
            elif repo['url'] == 'local':
                raise ValueError(f'{name}: local repository must be provisioned at {root}')
            elif args.verify:
                raise ValueError(f'{name}: missing {root}')
            else:
                root.parent.mkdir(parents=True, exist_ok=True)
                # A temporary sibling never replaces a pre-existing checkout. Interrupted
                # clone directories remain explicit evidence rather than being reused.
                temporary = root.with_name(root.name + '.fetching')
                if temporary.exists():
                    raise ValueError(f'{name}: interrupted clone at {temporary}; inspect it before retrying')
                subprocess.run(['git', 'clone', '--no-checkout', '--', repo['url'], str(temporary)], check=True)
                git(temporary, 'checkout', '--detach', rev)
                temporary.rename(root)
        if git(root, 'rev-parse', '--show-toplevel') != str(root.resolve()):
            raise ValueError(f'{name}: path is not its own repository root')
        if git(root, 'rev-parse', '--is-shallow-repository') != 'false':
            raise ValueError(f'{name}: shallow clone is not an accepted corpus')
        partial = subprocess.run(['git', '-C', str(root), 'config', '--get-regexp',
                                  r'^(extensions\.partialClone|remote\..*\.(promisor|partialclonefilter))$'],
                                 capture_output=True, text=True)
        if partial.returncode not in [0, 1]:
            raise ValueError(f'{name}: unable to inspect partial clone configuration')
        if partial.returncode == 0:
            raise ValueError(f'{name}: partial/promisor clone is not an accepted corpus')
        head = git(root, 'rev-parse', 'HEAD')
        if head != rev:
            raise ValueError(f'{name}: existing checkout is {head}, expected {rev}; refusing to move it')
        git(root, 'fsck', '--connectivity-only', '--no-dangling')
        results.append({'name': name, 'path': str(root), 'rev': head,
                        'status': 'verified', 'commits': int(git(root, 'rev-list', '--count', 'HEAD'))})
        print(json.dumps(results[-1]), flush=True)
    print(json.dumps({'repositories': results, 'verified': sum(r['status'] == 'verified' for r in results)}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'corpus fetch: {error}', file=sys.stderr)
        sys.exit(1)
