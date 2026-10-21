"""Exercise publication and regression failure using the actual Rust benchmark diff."""
from contextlib import redirect_stdout
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import nightly

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / 'fixtures/bench/regression-baseline.json'
DEBUG_BINARY = ROOT / 'target/debug/dkc'


class NightlyTest(unittest.TestCase):
    def test_gpu_stderr_with_zero_exit_cannot_be_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = nightly.Runner(Path(tmp) / 'run.log')
            with patch.object(nightly, 'nvidia_query',
                              side_effect=RuntimeError('GPU preflight diagnostic: Unable to determine device handle for GPU0')):
                with self.assertRaisesRegex(RuntimeError, 'GPU preflight diagnostic'):
                    runner.idle(True)

    def test_gpu_inventory_requires_a_real_device_row(self):
        with patch.object(nightly, 'nvidia_query', return_value='0\n1\n'):
            self.assertEqual(nightly.gpu_inventory(), 2)
        for output in ('', 'No devices were found\n'):
            with self.subTest(output=output), patch.object(nightly, 'nvidia_query', return_value=output):
                with self.assertRaisesRegex(RuntimeError, 'no valid device inventory'):
                    nightly.gpu_inventory()

    def test_stuck_gpu_query_has_a_finite_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'nvidia-smi'
            fake.write_text('#!/bin/sh\nsleep 5\n')
            fake.chmod(0o755)
            started = time.monotonic()
            with patch.dict(os.environ, {'PATH': tmp + os.pathsep + os.environ['PATH']}):
                with self.assertRaisesRegex(TimeoutError, 'GPU preflight query timed out'):
                    nightly.nvidia_query(['-L'], timeout=0.1)
            self.assertLess(time.monotonic() - started, 3)

    def test_missing_gpu_fails_before_build_and_keeps_terminal_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = type('Args', (), {
                'corpus': ROOT / 'bench/corpus.toml',
                'log_dir': directory / 'logs',
                'out_dir': directory / 'reports',
                'compare_only': None,
            })()
            args.log_dir.mkdir()
            args.out_dir.mkdir()
            with patch.object(nightly.Runner, 'run', return_value=0) as run, \
                 patch.object(nightly, 'gpu_inventory', side_effect=RuntimeError('GPU unavailable')), \
                 patch.object(nightly, 'build_binaries') as build:
                self.assertEqual(nightly.execute(args), 1)
            self.assertEqual(run.call_count, 1)  # git pull only
            build.assert_not_called()
            self.assertIn('all · **FAIL** · gpu-preflight: GPU unavailable',
                          (args.out_dir / 'NIGHTLY.md').read_text())
            self.assertEqual(next(args.log_dir.glob('nightly-*.exit')).read_text(), '1\n')

    def test_expired_six_hour_slot_records_deferred_without_starting_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = type('Args', (), {
                'corpus': ROOT / 'bench/corpus.toml',
                'log_dir': directory / 'logs',
                'out_dir': directory / 'reports',
                'compare_only': None,
            })()
            args.log_dir.mkdir()
            args.out_dir.mkdir()
            with patch.object(nightly, 'MAX_RUN_SECONDS', nightly.BUDGET_MARGIN_SECONDS + 1), \
                 patch.object(nightly.Runner, 'run') as run:
                self.assertEqual(nightly.execute(args), 0)
            run.assert_not_called()
            self.assertIn('all · **DEFERRED** · pull: nightly time budget exhausted',
                          (args.out_dir / 'NIGHTLY.md').read_text())
            self.assertEqual(next(args.log_dir.glob('nightly-*.exit')).read_text(), '0\n')

    def test_fast_forward_preserves_dirty_work_when_pull_rebase_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin = Path(tmp) / 'origin'
            checkout = Path(tmp) / 'checkout'
            origin.mkdir()
            def git(root, *args):
                subprocess.run(['git', *args], cwd=root, check=True, capture_output=True)
            git(origin, 'init')
            git(origin, 'config', 'user.name', 'Bench Fixture')
            git(origin, 'config', 'user.email', 'bench@example.invalid')
            (origin / 'work.txt').write_text('original')
            git(origin, 'add', '.')
            git(origin, 'commit', '-m', 'initial')
            git(Path(tmp), 'clone', str(origin), str(checkout))
            git(checkout, 'config', 'pull.rebase', 'true')
            (checkout / 'work.txt').write_text('preserved unfinished work')
            (origin / 'incoming.txt').write_text('incoming')
            git(origin, 'add', '.')
            git(origin, 'commit', '-m', 'incoming')
            old = subprocess.run(['git', 'pull', '--ff-only'], cwd=checkout, capture_output=True)
            self.assertNotEqual(old.returncode, 0)
            fixed = subprocess.run(nightly.PULL, cwd=checkout, capture_output=True)
            self.assertEqual(fixed.returncode, 0, fixed.stderr)
            self.assertEqual((checkout / 'work.txt').read_text(), 'preserved unfinished work')
            self.assertEqual((checkout / 'incoming.txt').read_text(), 'incoming')

    def test_dry_run_lists_full_profiles_without_mutating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'outputs'
            command = [
                str(ROOT / 'scripts/bench/nightly.sh'),
                '--out-dir', str(directory),
                '--log-dir', str(directory / 'logs'),
                '--corpus', str(ROOT / 'bench/corpus.toml'),
                '--dry-run',
            ]
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['builds'], ['cpu release', 'cuda release'])
            self.assertEqual(
                [(row['profile'], row['target_count']) for row in plan['recipes']],
                [('local', 26), ('moonly', 26), ('cpu', 11)],
            )
            self.assertTrue(plan['recipes'][0]['require_gpu'])
            self.assertFalse(plan['recipes'][1]['require_gpu'])
            self.assertFalse(directory.exists())

    def test_missing_corpus_dry_run_fails_before_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'outputs'
            command = [
                str(ROOT / 'scripts/bench/nightly.sh'),
                '--out-dir', str(directory),
                '--log-dir', str(directory / 'logs'),
                '--corpus', str(directory / 'absent.toml'),
                '--dry-run',
            ]
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(directory.exists())

    def test_build_failure_writes_terminal_verdict_and_exit_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = type(
                'Args',
                (),
                {
                    'corpus': ROOT / 'bench/corpus.toml',
                    'log_dir': directory / 'logs',
                    'out_dir': directory / 'reports',
                    'compare_only': None,
                },
            )()
            args.log_dir.mkdir()
            args.out_dir.mkdir()
            with patch.object(nightly.Runner, 'run', side_effect=RuntimeError('build refused')):
                self.assertEqual(nightly.execute(args), 1)
            verdict = (args.out_dir / 'NIGHTLY.md').read_text()
            self.assertIn('all · **FAIL** · pull: build refused', verdict)
            receipt = next(args.log_dir.glob('nightly-*.exit'))
            self.assertEqual(receipt.read_text(), '1\n')

    def test_pass_regression_and_missing_report_preserve_last_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            command = [str(ROOT / 'scripts/bench/nightly.sh'), '--out-dir', str(directory),
                       '--log-dir', str(directory / 'logs'), '--baseline', str(SEED),
                       '--binary', str(DEBUG_BINARY)]
            good = subprocess.run(command + ['--compare-only', str(SEED)], capture_output=True)
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            latest = (directory / 'nightly-latest.json').read_bytes()
            candidate = json.loads(SEED.read_text())
            candidate['targets'][0]['quality']['commit_derived']['hit_file_at_5'] -= 0.03
            bad = directory / 'regression.json'
            bad.write_text(json.dumps(candidate))
            failed = subprocess.run(command + ['--compare-only', str(bad)], capture_output=True)
            self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
            self.assertEqual((directory / 'nightly-latest.json').read_bytes(), latest)
            diffs = [json.loads(p.read_text()) for p in directory.glob('*-diff.json')]
            self.assertTrue(any('HitFile@5 dropped' in str(d.get('failures')) for d in diffs))
            missing = subprocess.run(command + ['--compare-only', str(directory / 'absent.json')], capture_output=True)
            self.assertEqual(missing.returncode, 1)
            self.assertEqual((directory / 'nightly-latest.json').read_bytes(), latest)
            verdicts = (directory / 'NIGHTLY.md').read_text()
            self.assertEqual(verdicts.count('**PASS**'), 1)
            self.assertEqual(verdicts.count('**FAIL**'), 2)

    def test_compare_only_does_not_take_the_live_corpus_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shared = directory / 'shared.lock'
            args = ['nightly.py', '--out-dir', str(directory / 'reports'),
                    '--log-dir', str(directory / 'logs'), '--baseline', str(SEED),
                    '--binary', str(DEBUG_BINARY)]
            with shared.open('w') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(nightly, 'SHARED_LOCK', shared), redirect_stdout(io.StringIO()):
                    with patch.object(sys, 'argv', args + ['--compare-only', str(SEED)]):
                        self.assertEqual(nightly.main(), 0)
                    with patch.object(sys, 'argv', args):
                        self.assertEqual(nightly.main(), 75)
            verdicts = (directory / 'reports/NIGHTLY.md').read_text()
            self.assertEqual(verdicts.count('**PASS**'), 1)
            self.assertNotIn('**FAIL**', verdicts)


if __name__ == '__main__':
    unittest.main()
