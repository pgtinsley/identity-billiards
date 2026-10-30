"""Locate the pinned dependency or a sibling checkout in portable bundles."""

import os
from pathlib import Path


def insightface_source(root, source=None):
    override = source if source is not None else os.environ.get('INSIGHTFACE_REPO')
    if override:
        return Path(override).expanduser().resolve()
    root = Path(root)
    for candidate in (root / 'vendor/insightface', root.parent / 'insightface'):
        if (candidate / 'python-package/insightface').is_dir():
            return candidate.resolve()
    return (root / 'vendor/insightface').resolve()
