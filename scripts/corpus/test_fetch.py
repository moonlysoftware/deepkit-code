#!/usr/bin/env python3
"""Exercise full-clone selection, exact pins and protection of existing checkouts."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('fetch.sh').resolve()


class FetchTests(unittest.TestCase):
    def test_subset_pin_and_existing_checkout_refusals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            def git(where, *args):
                return subprocess.check_output(['git', '-C', str(where), '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', *args], text=True).strip()
            git(source, 'init', '-q')
            (source / 'a.txt').write_text('first\n')
            git(source, 'add', 'a.txt')
            git(source, 'commit', '-q', '-m', 'first')
            first = git(source, 'rev-parse', 'HEAD')
            (source / 'a.txt').write_text('second\n')
            git(source, 'commit', '-qam', 'second')
            second = git(source, 'rev-parse', 'HEAD')
            manifest = root / 'corpus.toml'
            manifest.write_text(''.join(f'[[repo]]\nname="{name}"\nurl="{source.as_uri()}"\nrev="{first}"\n' for name in ['one', 'two']))
            env = dict(os.environ, DKC_CORPUS_DIR=str(root / 'corpus'))
            def fetch(*args):
                return subprocess.run([str(SCRIPT), '--manifest', str(manifest), *args],
                    env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(fetch('--verify', '--only', 'one').returncode, 0)
            result = fetch('--only', 'one')
            self.assertEqual(result.returncode, 0, result.stderr)
            checkout = root / 'corpus/one'
            self.assertEqual(git(checkout, 'rev-parse', 'HEAD'), first)
            self.assertFalse((root / 'corpus/two').exists())
            (checkout / 'a.txt').write_text('local work\n')
            self.assertEqual(fetch('--verify', '--only', 'one').returncode, 0)
            self.assertEqual((checkout / 'a.txt').read_text(), 'local work\n')
            self.assertNotEqual(fetch('--only', 'unknown').returncode, 0)
            git(checkout, 'config', 'remote.origin.promisor', 'true')
            self.assertNotEqual(fetch('--verify', '--only', 'one').returncode, 0)
            git(checkout, 'config', '--unset', 'remote.origin.promisor')
            git(checkout, 'checkout', '--', 'a.txt')
            git(checkout, 'checkout', '-q', '--detach', second)
            self.assertNotEqual(fetch('--only', 'one').returncode, 0)
            self.assertEqual(git(checkout, 'rev-parse', 'HEAD'), second)
            # Explicit local paths are verified, never silently skipped or cloned.
            manifest.write_text(f'[[repo]]\nname="local"\nurl="local"\npath="{source}"\nrev="{second}"\n')
            self.assertEqual(fetch('--verify').returncode, 0)
            manifest.write_text(f'[[repo]]\nname="local"\nurl="local"\npath="{root / "missing"}"\nrev="{second}"\n')
            self.assertNotEqual(fetch().returncode, 0)
            self.assertFalse((root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
