"""A beta formula must use real archives, safe paths and complete notices."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('brew_formula', Path(__file__).with_name('brew-formula.py'))
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)


class BrewFormulaTests(unittest.TestCase):
    def archive(self, directory, extra=None, skip=None, prefix='', link=False, build=None):
        path = directory / 'deepkit-code-x86_64-unknown-linux-musl.tar.gz'
        files = ['dkc', 'LICENSE-MIT', 'LICENSE-APACHE', 'LICENSE.md',
                 'THIRD_PARTY_NOTICES.md', 'third-party-licenses.tar.gz']
        if extra:
            files.append(extra)
        with tarfile.open(path, 'w:gz') as archive:
            for name in files:
                if name == skip:
                    continue
                member = tarfile.TarInfo(prefix + name)
                member.size = 4
                member.mode = 0o755 if name == 'dkc' else 0o644
                archive.addfile(member, io.BytesIO(b'test'))
            if build is not None:
                data = build if isinstance(build, bytes) else json.dumps(build).encode()
                member = tarfile.TarInfo(prefix + 'BUILD.json')
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if link:
                member = tarfile.TarInfo(prefix + 'linked')
                member.type = tarfile.SYMTYPE
                member.linkname = '/outside'
                archive.addfile(member)
        return path

    def test_complete_archive_binds_hash_and_real_smoke_test(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            archive = self.archive(directory)
            value = formula.generate(directory, '0.1.0-beta.1', ['x86_64-unknown-linux-musl'], False)
            self.assertIn(formula.inspect_archive(archive), value)
            self.assertIn('/v0.1.0-beta.1/', value)
            self.assertIn('system bin/"dkc", "index", "--level", "1"', value)
            self.assertNotIn('file://', value)

    def test_missing_notice_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for options in ({'skip': 'LICENSE.md'}, {'extra': '../outside'}):
                with self.subTest(options=options):
                    path = self.archive(directory, **options)
                    with self.assertRaises(ValueError):
                        formula.inspect_archive(path)

    def test_missing_platform_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.archive(directory)
            with self.assertRaises(FileNotFoundError):
                formula.generate(directory, '0.1.0-beta.1',
                                 ['x86_64-unknown-linux-musl', 'aarch64-apple-darwin'], False)

    def test_cargo_dist_layout_is_supported_but_deep_or_linked_layouts_are_not(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = self.archive(directory, prefix='deepkit-code-linux/')
            self.assertEqual(len(formula.inspect_archive(path)), 64)
            for options in ({'prefix': 'outer/inner/'}, {'link': True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    formula.inspect_archive(self.archive(directory, **options))

    def test_public_formula_rejects_diagnostic_dirty_or_mismatched_build(self):
        valid = {'version': '0.1.0-beta.1', 'diagnostic': False,
                 'build': {'target': 'x86_64-unknown-linux-musl', 'source_dirty': 'false'}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.archive(directory, build=valid)
            formula.generate(directory, '0.1.0-beta.1', ['x86_64-unknown-linux-musl'], False)
            variants = [dict(valid, diagnostic=True), dict(valid, version='0.1.0-beta.2'),
                        dict(valid, build=dict(valid['build'], source_dirty='true')),
                        dict(valid, build=dict(valid['build'], source_dirty='unknown')),
                        dict(valid, build=dict(valid['build'], target='aarch64-apple-darwin')),
                        b'{broken json']
            for metadata in variants:
                with self.subTest(metadata=metadata):
                    self.archive(directory, build=metadata)
                    with self.assertRaises(ValueError):
                        formula.generate(directory, '0.1.0-beta.1', ['x86_64-unknown-linux-musl'], False)

    def test_local_formula_allows_explicit_diagnostic_but_not_wrong_target(self):
        metadata = {'version': '0.1.0-beta.1', 'diagnostic': True,
                    'build': {'target': 'x86_64-unknown-linux-musl', 'source_dirty': 'true'}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.archive(directory, build=metadata)
            value = formula.generate(directory, '0.1.0-beta.1', ['x86_64-unknown-linux-musl'], True)
            self.assertIn('file://', value)
            metadata['build']['target'] = 'aarch64-apple-darwin'
            self.archive(directory, build=metadata)
            with self.assertRaisesRegex(ValueError, 'target differs'):
                formula.generate(directory, '0.1.0-beta.1', ['x86_64-unknown-linux-musl'], True)

    def test_duplicate_archive_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for extra in ('LICENSE.md', './LICENSE.md'):
                with self.subTest(extra=extra):
                    path = self.archive(Path(temp), extra=extra)
                    with self.assertRaisesRegex(ValueError, 'duplicate member'):
                        formula.inspect_archive(path)

    def test_gnu_diagnostic_cannot_be_published_as_portable_linux(self):
        with self.assertRaises(ValueError):
            formula.generate(Path('/unused'), '0.1.0-beta.1', ['x86_64-unknown-linux-gnu'], False)

    def test_apple_silicon_only_formula_declares_architecture_and_deployment_floor(self):
        target = 'aarch64-apple-darwin'
        metadata = {'version': '0.1.0-beta.2', 'diagnostic': False,
                    'build': {'target': target, 'source_dirty': 'false'}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.archive(directory, build=metadata).rename(
                directory / f'deepkit-code-{target}.tar.gz')
            for local_test in (False, True):
                with self.subTest(local_test=local_test):
                    value = formula.generate(directory, '0.1.0-beta.2', [target], local_test)
                    self.assertIn('  depends_on macos: :ventura\n', value)
                    self.assertIn('  depends_on arch: :arm64\n', value)
                    self.assertNotIn('on_linux', value)
                    self.assertEqual('file://' in value, local_test)

    def test_mixed_platform_formula_keeps_its_existing_platform_support(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            linux = self.archive(directory)
            (directory / 'deepkit-code-aarch64-apple-darwin.tar.gz').write_bytes(linux.read_bytes())
            value = formula.generate(directory, '0.1.0-beta.1',
                                     ['x86_64-unknown-linux-musl', 'aarch64-apple-darwin'], False)
            self.assertIn('  on_linux do\n', value)
            self.assertIn('  on_macos do\n', value)
            self.assertNotIn('depends_on macos:', value)
            self.assertNotIn('depends_on arch:', value)


if __name__ == '__main__':
    unittest.main()
