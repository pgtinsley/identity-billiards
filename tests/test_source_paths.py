"""Dependency discovery must work in Git clones and exported sibling bundles."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.source_paths import insightface_source


class InsightFaceSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('INSIGHTFACE_REPO', None)

    def checkout(self, path):
        (path / 'python-package/insightface').mkdir(parents=True)
        return path.resolve()

    def test_git_clone_prefers_vendored_dependency(self):
        vendor = self.checkout(self.root / 'vendor/insightface')
        self.checkout(self.root.parent / 'insightface')
        self.assertEqual(insightface_source(self.root), vendor)

    def test_source_bundle_uses_sibling_dependency(self):
        (self.root / 'vendor/insightface').mkdir(parents=True)
        sibling = self.checkout(self.root.parent / 'insightface')
        self.assertEqual(insightface_source(self.root), sibling)

    def test_explicit_source_overrides_environment_and_default(self):
        self.checkout(self.root / 'vendor/insightface')
        os.environ['INSIGHTFACE_REPO'] = str(self.root / 'environment')
        explicit = self.root / 'explicit'
        self.assertEqual(insightface_source(self.root, explicit), explicit.resolve())

    def test_environment_override_is_not_silently_replaced(self):
        self.checkout(self.root / 'vendor/insightface')
        override = self.root / 'not-yet-installed'
        os.environ['INSIGHTFACE_REPO'] = str(override)
        self.assertEqual(insightface_source(self.root), override.resolve())

    def test_missing_dependency_reports_the_submodule_location(self):
        self.assertEqual(insightface_source(self.root), (self.root / 'vendor/insightface').resolve())
