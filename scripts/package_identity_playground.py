#!/usr/bin/env python3
"""Build a source-only collaborator bundle from these two local checkouts.

Uses an allowlist, includes uncommitted experiment code, and never copies the
virtual environment, model weights, generated outputs, or Git history.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


STYLEGAN_FILES = ('LICENSE.txt', 'README.md', 'README.stylegan3.md', 'THIRD_PARTY.md',
                 'CONTRIBUTING.md', 'legacy.py', '.gitignore')
STYLEGAN_TREES = {
    'dnnlib': {'.py'},
    'torch_utils': {'.py', '.cpp', '.cu', '.h'},
    'training': {'.py'},
    'experiments': {'.py', '.md', '.txt', '.html', '.css', '.js'},
    'scripts': {'.py', '.sh'},
    'tests': {'.py'},
    'docs': {'.md', '.txt', '.png'},
}
INSIGHTFACE_FILES = ('README.md', 'python-package/README.md',
                     'python-package/docs/model_zoo.md', 'python-package/docs/person_analysis.md',
                     'python-package/docs/commercial_evaluation.md')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def collect_files(stylegan, insightface):
    """Select only source/runtime files; reject links instead of dereferencing them."""
    result = {}

    def add(root, relative, destination):
        path = root / relative
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f'Refusing symlink in source bundle: {path}')
        if not path.is_file():
            raise FileNotFoundError(path)
        result[f'{destination}/{relative.as_posix()}'] = (path.read_bytes(), 0o755 if path.suffix == '.sh' else 0o644)

    for relative in STYLEGAN_FILES:
        add(stylegan, Path(relative), 'stylegan3')
    for tree, suffixes in STYLEGAN_TREES.items():
        base = stylegan / tree
        if base.is_symlink() or not base.is_dir():
            raise ValueError(f'Expected source directory, without symlinks: {base}')
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(stylegan)
            if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f'Refusing symlink in source bundle: {path}')
            if path.is_file() and path.suffix in suffixes:
                add(stylegan, relative, 'stylegan3')
    for relative in INSIGHTFACE_FILES:
        add(insightface, Path(relative), 'insightface')
    package = insightface / 'python-package/insightface'
    if package.is_symlink() or not package.is_dir():
        raise ValueError(f'Expected InsightFace source directory: {package}')
    for path in sorted(package.rglob('*')):
        relative = path.relative_to(insightface)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Refusing symlink in source bundle: {path}')
        if path.is_file() and path.suffix in {'.py', '.json', '.pem', '.md', '.txt'}:
            add(insightface, relative, 'insightface')
    return result


def repository_info(root):
    def git(*arguments):
        process = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None

    return {'revision': git('rev-parse', 'HEAD'),
            'has_local_changes': bool(git('status', '--porcelain'))}


def build(output, stylegan, insightface):
    files = collect_files(stylegan, insightface)
    files['README.md'] = (b'# Identity Playground\n\n'
                          b'Start with the [collaborator guide](stylegan3/experiments/SHARING.md).\n\n'
                          b'This archive contains source code and pinned requirements. Model weights, '
                          b'Python environments and experiment data are excluded.\n', 0o644)
    manifest = {
        'format': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'repositories': {'stylegan3': repository_info(stylegan), 'insightface': repository_info(insightface)},
        'notes': 'Snapshot of selected working-tree files, including local experiments. No weights or user data.',
        'files': {name: {'sha256': sha256(data), 'bytes': len(data)}
                  for name, (data, _) in sorted(files.items())},
    }
    files['MANIFEST.json'] = ((json.dumps(manifest, indent=2) + '\n').encode(), 0o644)
    files['SHA256SUMS'] = (''.join(f'{sha256(data)}  {name}\n' for name, (data, _) in sorted(files.items())).encode(), 0o644)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing handoff from accidental replacement.
    with output.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, (data, mode) in sorted(files.items()):
                    info = tarfile.TarInfo(f'identity-playground/{name}')
                    info.size, info.mode, info.mtime = len(data), mode, 0
                    archive.addfile(info, io.BytesIO(data))
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    output.with_name(output.name + '.sha256').write_text(f'{digest}  {output.name}\n')
    print(f'Created {output} ({output.stat().st_size / 1024:.0f} KiB; {len(files)} files)')
    print(f'SHA256 {digest}')
    return manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = root / 'vendor/insightface'
    if not (default_source / 'python-package/insightface').is_dir():
        default_source = root.parent / 'insightface'
    parser.add_argument('--insightface-repo', type=Path, default=default_source)
    parser.add_argument('--output', type=Path, default=root / 'dist/identity-playground.tar.gz')
    args = parser.parse_args()
    try:
        build(args.output.expanduser().absolute(), root, args.insightface_repo.expanduser().resolve())
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot build bundle: {error}\n')


if __name__ == '__main__':
    main()
