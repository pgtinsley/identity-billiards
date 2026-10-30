"""Check the collaborator environment; optionally exercise the models on a GPU."""

import argparse
import importlib
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--smoke', action='store_true',
                        help='Load/download models, render and match a face, run projection and encode video')
    args = parser.parse_args()
    errors = []

    def check(label, operation):
        try:
            detail = operation()
            print(f'PASS {label}' + (f': {detail}' if detail is not None else ''), flush=True)
        except Exception as error:
            errors.append(label)
            print(f'FAIL {label}: {error}', flush=True)

    def python_version():
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError('Use Python 3.11 via scripts/setup_identity_playground.sh')
        return sys.version.split()[0]

    check('Python', python_version)
    for module in ('torch', 'numpy', 'fastapi', 'uvicorn', 'multipart', 'onnx',
                   'onnxruntime', 'cv2', 'scipy', 'skimage', 'matplotlib', 'imageio_ffmpeg'):
        check(module, lambda module=module: getattr(importlib.import_module(module), '__version__', 'imported'))
    if errors:
        return 1

    from experiments.face_boundary import DEFAULT_NETWORK, MODEL_ROOT, ROOT
    from experiments.face_projection import PERCEPTUAL_PATH

    from experiments.source_paths import insightface_source

    source = insightface_source(ROOT)

    def insightface_import():
        package = source / 'python-package'
        if not (package / 'insightface').is_dir():
            raise FileNotFoundError(f'{package}; run git submodule update --init --recursive '
                                    'or set INSIGHTFACE_REPO to the checkout root')
        sys.path.insert(0, str(package))
        from insightface.app import FaceAnalysis
        return str(source)

    check('InsightFace source import', insightface_import)

    def output_directory():
        directory = Path(os.environ.get('FACE_UI_OUTPUT_DIR', ROOT / 'outputs/ui')).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass
        return str(directory.resolve())

    check('Writable outputs', output_directory)

    def cuda():
        import torch
        import onnxruntime as ort
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA device visible. Run this check inside your GPU allocation.')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX CUDA provider missing; rerun setup with the pinned GPU dependencies')
        return torch.cuda.get_device_name()

    if args.device == 'cuda':
        check('CUDA availability', cuda)
    for label, path in [('StyleGAN', DEFAULT_NETWORK), ('ArcFace', MODEL_ROOT / 'insightface/models/buffalo_l'),
                        ('VGG16', PERCEPTUAL_PATH)]:
        print(f'INFO {label}: {path} ({"present" if path.exists() else "downloads on first use"})')
    if args.smoke and not errors:
        def smoke():
            import imageio.v2 as imageio
            import numpy as np
            import torch
            from experiments.face_boundary import download_network, face_embedding, load_generator, load_matcher
            from experiments.face_projection import load_perceptual_model, project_photo

            torch.set_num_threads(4)
            download_network()
            generator = load_generator(device=args.device)
            matcher = load_matcher(device=args.device)
            with torch.inference_mode():
                styles = generator.mapping.w_avg[None, None].repeat(1, generator.num_ws, 1)
                face = generator.synthesis(styles, noise_mode='const', force_fp32=True)
                rgb = (face[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            observation = face_embedding(matcher, rgb)
            if not observation.valid:
                raise RuntimeError(f'Mean face failed detection: {observation.reason}')
            perceptual = load_perceptual_model(args.device)
            anchor, info = project_photo(generator, rgb, perceptual, steps=1)
            if not np.isfinite(anchor).all() or not np.isfinite(info['loss']):
                raise RuntimeError('Projection produced nonfinite values')
            with tempfile.TemporaryDirectory() as directory:
                video = Path(directory) / 'smoke.mp4'
                imageio.mimwrite(video, [rgb, rgb], fps=2, codec='libx264')
                if video.stat().st_size == 0:
                    raise RuntimeError('Video encoder produced an empty file')
            return 'render, detection/embedding, one projection step and MP4 encoding passed'

        check('Model smoke check', smoke)
    elif not args.smoke:
        print('INFO Model execution was not checked; use --smoke on the allocated GPU node.')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
