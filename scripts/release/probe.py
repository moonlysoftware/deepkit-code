#!/usr/bin/env python3
"""Measure available native release and probe musl feasibility; v1.md §12, T1.R5."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    results = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'commands': [], 'tools': {}}

    def run(command):
        started = time.monotonic()
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        record = {'command': command, 'exit_code': p.returncode,
                  'elapsed_s': time.monotonic() - started, 'stdout': p.stdout, 'stderr': p.stderr}
        results['commands'].append(record)
        print(json.dumps({'command': command, 'exit_code': p.returncode, 'elapsed_s': record['elapsed_s']}), flush=True)
        # Evidence survives a later failed command; callers decide which outcomes are expected.
        args.out.write_text(json.dumps(results, indent=2) + '\n')
        return p

    for tool in ('rustc', 'cargo', 'cmake', 'clang', 'musl-gcc', 'x86_64-linux-musl-gcc', 'musl-g++', 'zig', 'cargo-zigbuild'):
        results['tools'][tool] = shutil.which(tool)
    results['checkout_commit'] = run(['git', 'rev-parse', 'HEAD']).stdout.strip()
    results['working_tree'] = run(['git', 'status', '--short']).stdout
    run(['rustc', '-vV'])
    run(['cargo', '--version'])
    run(['cmake', '--version'])
    run(['clang', '--version'])
    run(['rustup', 'target', 'list', '--installed'])
    native = run(['cargo', 'build', '--release', '--locked'])
    if native.returncode:
        raise SystemExit('native release build failed; partial evidence retained')
    binary = ROOT / 'target/release/dkc'
    results['native_artifact'] = {'path': str(binary), 'bytes': binary.stat().st_size,
                                  'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    run([str(binary), '--version'])
    run(['file', str(binary)])
    run(['readelf', '-l', str(binary)])
    run(['readelf', '-d', str(binary)])
    # A feasibility attempt: missing target/toolchain is an observed failure, never a pass.
    musl = run(['cargo', 'build', '--release', '--locked', '--target', 'x86_64-unknown-linux-musl'])
    results['musl_build'] = 'built' if musl.returncode == 0 else 'failed_see_command_evidence'
    results['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    args.out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
