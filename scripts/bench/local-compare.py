#!/usr/bin/env python3
"""Compare two prebuilt CPU test binaries on identical T1.4 fixtures, retaining failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path, help='JSON report; raw logs use the same prefix')
    args = parser.parse_args()
    model = Path(os.environ['DKC_MODELS_DIR']) / 'Qwen3-Embedding-0.6B-Q8_0.gguf'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'scope': 'single paired CPU component diagnostic, not a release gate',
        'kernel': platform.release(),
        'affinity': sorted(os.sched_getaffinity(0)),
        'throughput_threads': 8,
        'sanity_threads': 'provider defaults within the recorded affinity',
        'model_sha256': digest(model),
        'artifacts': {},
        'runs': [],
        'status': 'running',
    }
    try:
        for name in ('baseline', 'candidate'):
            path = getattr(args, name).resolve(strict=True)
            report['artifacts'][name] = {'path': str(path), 'sha256': digest(path)}
        for case, repeats in (('sanity', 1), ('short', 1), ('long', 16)):
            for name, artifact in report['artifacts'].items():
                test = 'embed_sanity' if case == 'sanity' else 'embed_throughput'
                command = [artifact['path'], f'models::llama::tests::{test}',
                           '--exact', '--ignored', '--nocapture']
                env = dict(os.environ, DKC_BENCH_CPU_THREADS='8',
                           DKC_BENCH_DOCUMENT_REPEATS=str(repeats),
                           DKC_BENCH_BATCH_TIMINGS='1')
                log = args.output.with_suffix(f'.{case}.{name}.log')
                run = {'case': case, 'artifact': name, 'command': command,
                       'load_before': os.getloadavg(), 'log': str(log)}
                report['runs'].append(run)
                print(f'Running {case} / {name}; log: {log}', flush=True)
                with log.open('w') as stream:
                    process = subprocess.run(command, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, timeout=600)
                run['exit_code'] = process.returncode
                output = log.read_text()
                if process.returncode or 'test result: ok. 1 passed' not in output:
                    raise RuntimeError(f'{case}/{name} failed; see {log}')
                if case != 'sanity':
                    counts = re.search(r'embedding benchmark: (.+)', output)
                    run['input'] = dict(re.findall(r'(\w+)=(\w+)', counts[1]))
                    if run['input'].get('cuda') != 'false' or run['input'].get('threads') != '8':
                        raise RuntimeError('comparison requires CPU inference with eight threads')
                    result = re.search(r'([\d.]+) units/s \((\d+) documents in ([\d.]+)s\)', output)
                    run['units_per_second'] = float(result[1])
                    run['documents'] = int(result[2])
                    run['elapsed_seconds_rounded'] = float(result[3])
                args.output.write_text(json.dumps(report, indent=2) + '\n')
        for case in ('short', 'long'):
            pair = [run for run in report['runs'] if run['case'] == case]
            if pair[0]['input'] != pair[1]['input']:
                raise RuntimeError(f'{case} input or feature metadata differs')
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['status']}: {args.output}", flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
