#!/usr/bin/env python3
"""Offline controls for release notice corruption, stale inputs and reproduction."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

PATH = Path(__file__).with_name('generate-notices.py')
SPEC = importlib.util.spec_from_file_location('generate_notices', PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class LicenseBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base/'repo'
        self.supplements = self.root/'licenses'
        self.cache = self.base/'cargo-home'
        self.output = self.base/'notices'
        self.bundle = self.base/'notices.tar.gz'
        fixtures = {
            HELPER.SQLITE_LICENSE: (PATH.parents[2]/HELPER.SQLITE_LICENSE).read_bytes(),
            'vendor/gix-index/LICENSE-APACHE': b'Apache fixture\n',
            'vendor/gix-index/LICENSE-MIT': b'MIT fixture\n',
            'vendor/gix-index/PATCH.md': b'Local patch provenance\n',
            'vendor/gix-index/Cargo.toml': b'[package]\nname="gix-index"\nversion="0.55.0"\nlicense="MIT OR Apache-2.0"\n',
            'src/treat/grammar_queries/c.LICENSE': b'Grammar copyright and MIT grant\n',
            'licenses/notice-overrides-provenance.json': b'[]\n',
            'licenses/model-license-evidence.json': b'[]\n',
            'licenses/native-notices/LICENSE': b'Native copyright and grant\n',
            'licenses/native-notices/provenance.json': b'[]\n',
        }
        for name, data in fixtures.items():
            file = self.root/name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
        (self.supplements/'notice-overrides').mkdir()
        archive = self.cache/'registry/cache/example/fixture-1.0.0.crate'
        archive.parent.mkdir(parents=True)
        self.write_tar(archive, [
            ('fixture-1.0.0/Cargo.toml', b'[package]\nname="fixture"\nversion="1.0.0"\nlicense="MIT"\n'),
            ('fixture-1.0.0/LICENSE', b'Fixture copyright and MIT grant\n'),
        ])
        checksum = HELPER.digest(archive.read_bytes())
        (self.root/'Cargo.lock').write_text(
            f'version = 4\n[[package]]\nname="fixture"\nversion="1.0.0"\n'
            f'source="{HELPER.REGISTRY}"\nchecksum="{checksum}"\n')
        self.manifest = HELPER.generate(self.root, self.output, self.supplements, self.cache)
        HELPER.write_bundle(self.output, self.bundle)

    @staticmethod
    def write_tar(path, entries):
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in entries:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def members(self):
        with tarfile.open(self.bundle, 'r:gz') as archive:
            return [(item.name, archive.extractfile(item).read()) for item in archive]

    def check(self):
        return HELPER.check_bundle(self.bundle, self.root, self.supplements)

    def test_verified_bundle_is_reproducible_across_directory_and_timestamp(self):
        before = self.bundle.read_bytes()
        for path in self.output.rglob('*'):
            os.utime(path, (123456789, 123456789))
        second = self.base/'different-output'
        HELPER.generate(self.root, second, self.supplements, self.cache)
        HELPER.write_bundle(second, self.base/'second.tar.gz')
        self.assertEqual(before, (self.base/'second.tar.gz').read_bytes())
        self.assertEqual(self.check()['packages'], 2)
        self.assertEqual(self.check()['bundle_sha256'], HELPER.digest(before))

    def test_changed_lockfile_is_rejected_without_registry_cache(self):
        (self.root/'Cargo.lock').write_text((self.root/'Cargo.lock').read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'Cargo.lock changed'):
            self.check()

    def test_missing_extra_and_duplicate_members_are_rejected(self):
        original = self.members()
        variants = [original[1:], original+[('third-party-licenses/unlisted.txt', b'extra')], original+[original[-1]]]
        for entries in variants:
            with self.subTest(entries=len(entries)):
                self.write_tar(self.bundle, entries)
                with self.assertRaisesRegex(ValueError, 'missing MANIFEST|membership|duplicate'):
                    self.check()

    def test_corrupt_notice_is_rejected(self):
        entries = [(name, b'corrupt' if name.endswith('/fixture-1.0.0/LICENSE') else data)
                   for name, data in self.members()]
        self.write_tar(self.bundle, entries)
        with self.assertRaisesRegex(ValueError, 'notice hash mismatch'):
            self.check()

    def test_changed_source_or_added_supplement_requires_regeneration(self):
        file = self.root/'src/treat/grammar_queries/c.LICENSE'
        original = file.read_bytes()
        file.write_bytes(original+b'changed')
        with self.assertRaisesRegex(ValueError, 'source inputs changed'):
            self.check()
        file.write_bytes(original)
        (self.supplements/'native-notices/NEW-NOTICE').write_bytes(b'new attribution')
        with self.assertRaisesRegex(ValueError, 'source inputs changed'):
            self.check()

    def test_current_source_check_survives_rewritten_archive_hash(self):
        entries = dict(self.members())
        path = 'grammar-queries/c.LICENSE'
        entries['third-party-licenses/'+path] = b'different grant'
        manifest = json.loads(entries['third-party-licenses/MANIFEST.json'])
        for row in manifest['files']:
            if row['path'] == path:
                row['sha256'] = HELPER.digest(b'different grant')
                row.pop('source_input')
        entries['third-party-licenses/MANIFEST.json'] = json.dumps(manifest).encode()
        self.write_tar(self.bundle, entries.items())
        with self.assertRaisesRegex(ValueError, 'differs from current source'):
            self.check()

    def test_removing_required_notice_and_manifest_record_is_rejected(self):
        entries = dict(self.members())
        path = 'native/LICENSE'
        entries.pop('third-party-licenses/'+path)
        manifest = json.loads(entries['third-party-licenses/MANIFEST.json'])
        manifest['files'] = [row for row in manifest['files'] if row['path'] != path]
        entries['third-party-licenses/MANIFEST.json'] = json.dumps(manifest).encode()
        self.write_tar(self.bundle, entries.items())
        with self.assertRaisesRegex(ValueError, 'required source notice missing'):
            self.check()

    def test_escaping_path_and_symlink_are_rejected_without_extraction(self):
        entries = self.members()
        self.write_tar(self.bundle, entries+[('third-party-licenses/../escape', b'bad')])
        with self.assertRaisesRegex(ValueError, 'unsafe notice path'):
            self.check()
        with tarfile.open(self.bundle, 'w:gz') as archive:
            member = tarfile.TarInfo('third-party-licenses/link')
            member.type = tarfile.SYMTYPE
            member.linkname = '/outside'
            archive.addfile(member)
        with self.assertRaisesRegex(ValueError, 'unsafe bundle member'):
            self.check()
        self.assertFalse((self.base/'escape').exists())

    def test_package_inventory_cannot_drop_locked_dependency(self):
        entries = dict(self.members())
        manifest = json.loads(entries['third-party-licenses/MANIFEST.json'])
        manifest['packages'] = [p for p in manifest['packages'] if p['name'] != 'fixture']
        entries['third-party-licenses/MANIFEST.json'] = json.dumps(manifest).encode()
        self.write_tar(self.bundle, entries.items())
        with self.assertRaisesRegex(ValueError, 'inventory differs from Cargo.lock'):
            self.check()


if __name__ == '__main__':
    unittest.main()
