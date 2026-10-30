"""Regression checks for the source-only collaborator handoff (standard library)."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/package_identity_playground.py'
SPEC = importlib.util.spec_from_file_location('package_identity_playground', SCRIPT)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class SourceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='playground test ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stylegan = self.root / 'generator-checkout'
        self.insightface = self.root / 'matcher-checkout'
        for filename in package.STYLEGAN_FILES:
            self.write(self.stylegan / filename, 'upstream source')
        for tree in package.STYLEGAN_TREES:
            self.write(self.stylegan / tree / '__init__.py', '# source')
        for filename in package.INSIGHTFACE_FILES:
            self.write(self.insightface / filename, 'upstream notices')
        self.write(self.insightface / 'python-package/insightface/__init__.py', '# matcher')

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)

    def test_only_sources_are_exported_in_stable_sibling_layout(self):
        self.write(self.stylegan / 'experiments/ui/app.js', 'uncommitted UI')
        self.write(self.stylegan / 'scripts/start.sh', '#!/bin/bash')
        private_paths = ('models/generator.pkl', 'outputs/ui/run/metadata.json',
                         '.venv/config.py', '.git/config', '.env',
                         'experiments/__pycache__/module.pyc', 'experiments/.private/settings.json',
                         'experiments/upload.png', 'notebooks/research.ipynb')
        for name in private_paths:
            self.write(self.stylegan / name, 'DO NOT SHARE')
        self.write(self.insightface / 'python-package/insightface/data/private.pkl', 'DO NOT SHARE')
        files = package.collect_files(self.stylegan, self.insightface)
        self.assertEqual(files['stylegan3/experiments/ui/app.js'][0], b'uncommitted UI')
        self.assertEqual(files['stylegan3/scripts/start.sh'][1], 0o755)
        self.assertIn('insightface/python-package/insightface/__init__.py', files)
        self.assertFalse(any(b'DO NOT SHARE' in data for data, _ in files.values()))

    def test_source_file_symlink_is_refused(self):
        secret = self.root / 'private.txt'
        secret.write_text('DO NOT SHARE')
        (self.stylegan / 'experiments/leak.py').symlink_to(secret)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            package.collect_files(self.stylegan, self.insightface)

    def test_source_directory_symlink_is_refused(self):
        (self.stylegan / 'experiments/linked').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            package.collect_files(self.stylegan, self.insightface)

    def test_manifest_checksums_and_archive_safety(self):
        output = self.root / 'handoff.tar.gz'
        with contextlib.redirect_stdout(io.StringIO()):
            package.build(output, self.stylegan, self.insightface)
        with tarfile.open(output) as archive:
            members = archive.getmembers()
            self.assertTrue(all(member.isfile() and '..' not in Path(member.name).parts for member in members))
            self.assertTrue(all(member.uid == member.gid == 0 and member.uname == '' for member in members))
            content = {member.name.removeprefix('identity-playground/'): archive.extractfile(member).read()
                       for member in members}
        manifest = json.loads(content['MANIFEST.json'])
        for name, entry in manifest['files'].items():
            self.assertEqual(hashlib.sha256(content[name]).hexdigest(), entry['sha256'])
            self.assertEqual(len(content[name]), entry['bytes'])
        for line in content['SHA256SUMS'].decode().splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(hashlib.sha256(content[name]).hexdigest(), digest)
        digest, name = output.with_name(output.name + '.sha256').read_text().strip().split('  ', 1)
        self.assertEqual(name, output.name)
        self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
        with self.assertRaises(FileExistsError):
            package.build(output, self.stylegan, self.insightface)


if __name__ == '__main__':
    unittest.main()
