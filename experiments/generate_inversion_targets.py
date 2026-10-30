"""Generate reproducible shared-W faces with known inversion ground truth."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from experiments.face_boundary import (
    DEFAULT_NETWORK, DEFAULT_SEED_TRUNCATION_PSI, ROOT, face_embedding,
    file_sha256, load_generator, load_matcher,
)
from gen_images import parse_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=parse_range, default='0-9,23,42')
    parser.add_argument('--truncation-psi', type=float, default=DEFAULT_SEED_TRUNCATION_PSI)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/inversion-targets-psi075')
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Provide at least one seed, without duplicates')
    if any(not 0 <= seed <= 2147483647 for seed in args.seeds):
        parser.error('Seeds must be between 0 and 2147483647')
    if not 0 < args.truncation_psi <= 1:
        parser.error('Truncation must be finite and in (0, 1]')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    generator = load_generator(device=args.device)
    device = next(generator.parameters()).device
    matcher = load_matcher(device=device.type)
    images_dir = args.output / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    resolution = generator.img_resolution
    columns = min(4, len(args.seeds))
    rows = (len(args.seeds) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * resolution, rows * (resolution + 26)), 'white')
    labels = ImageDraw.Draw(sheet)
    records, latents_z, latents_w, raw_latents_w, styles_all = [], [], [], [], []
    with torch.inference_mode():
        for index, seed in enumerate(args.seeds):
            latent_z = np.random.RandomState(seed).randn(1, generator.z_dim).astype(np.float32)
            tensor_z = torch.as_tensor(latent_z, device=device)
            conditioning = torch.zeros((1, 0), device=device)
            raw_styles = generator.mapping(tensor_z, conditioning, truncation_psi=1)
            styles = generator.mapping(tensor_z, conditioning, truncation_psi=args.truncation_psi)
            shared = styles[:, :1].repeat(1, generator.num_ws, 1)
            torch.testing.assert_close(shared, styles, rtol=0, atol=0)
            synthetic = generator.synthesis(shared, noise_mode='const', force_fp32=True)
            pixels = (synthetic.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            image = Image.fromarray(pixels)
            filename = f'seed{seed:04d}.png'
            image.save(images_dir / filename)
            observation = face_embedding(matcher, pixels)
            records.append({'seed': seed, 'file': f'images/{filename}',
                            'sha256': file_sha256(images_dir / filename),
                            'face_valid': observation.valid, 'face_reason': observation.reason,
                            'detection_score': float(observation.detection_score) if observation.valid else None})
            latents_z.append(latent_z[0])
            latents_w.append(styles[0, 0].cpu().numpy())
            raw_latents_w.append(raw_styles[0, 0].cpu().numpy())
            styles_all.append(styles[0].cpu().numpy())
            column, row = index % columns, index // columns
            sheet.paste(image, (column * resolution, row * (resolution + 26) + 26))
            labels.text((column * resolution + 8, row * (resolution + 26) + 7),
                        f'Seed {seed} | psi {args.truncation_psi:g}', fill='black')
            print(f'{filename}: {observation.reason}', flush=True)
    np.savez_compressed(args.output / 'latents.npz', seeds=np.array(args.seeds),
                        z=np.stack(latents_z), w=np.stack(latents_w),
                        w_untruncated=np.stack(raw_latents_w), ws=np.stack(styles_all),
                        w_avg=generator.mapping.w_avg.cpu().numpy())
    with np.load(args.output / 'latents.npz') as saved, torch.inference_mode():
        for index, record in enumerate(records):
            shared = torch.as_tensor(saved['w'][index][None, None].copy(), device=device).repeat(1, generator.num_ws, 1)
            synthetic = generator.synthesis(shared, noise_mode='const', force_fp32=True)
            restored = (synthetic.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            with Image.open(args.output / record['file']) as image:
                np.testing.assert_array_equal(restored, np.asarray(image))
    sheet.save(args.output / 'contact-sheet.png')
    metadata = {'created': datetime.now(timezone.utc).isoformat(),
                'network': str(DEFAULT_NETWORK), 'network_sha256': file_sha256(DEFAULT_NETWORK),
                'truncation_psi': args.truncation_psi, 'noise_mode': 'const', 'force_fp32': True,
                'deterministic_algorithms': True, 'tf32': False,
                'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'torch': torch.__version__, 'resolution': resolution, 'latent_space': 'shared W',
                'num_ws': generator.num_ws, 'w_dim': generator.w_dim,
                'additional_alignment': False, 'selection': 'explicit seeds; no detection-based filtering',
                'saved_w_rerender': 'all PNGs reproduced pixel-for-pixel from saved shared W',
                'images': records}
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    (args.output / 'README.txt').write_text(
        'Synthetic inversion targets\n\n'
        'Upload individual images/*.png files, NOT the contact sheet.\n'
        'Check "Photo is already an FFHQ-aligned crop" in the UI.\n'
        'Do not re-align, resize, edit or JPEG-compress these targets for the controlled test.\n'
        'The fixed generator and constant synthesis noise are the same as the UI.\n'
        'latents.npz stores seeds, z, truncated shared w, untruncated w, repeated ws and w_avg.\n'
        'Array rows correspond to manifest.json image order.\n'
        'Saved W rerenders were verified pixel-for-pixel against every PNG.\n'
        'PNG targets are 8-bit quantized, so float-image fitting loss need not reach zero.\n'
        'Different latent codes may reconstruct similarly; judge image and identity recovery too.\n')
    archive_path = args.output / 'synthetic-inversion-targets.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob('*')):
            if path.is_file() and path != archive_path:
                archive.write(path, path.relative_to(args.output))
    print(f'Verified {len(records)} exact shared-W rerenders; '
          f'{sum(record["face_valid"] for record in records)} valid single-face detections.', flush=True)
    print(f'Contact sheet: {args.output / "contact-sheet.png"}\nBundle: {archive_path}', flush=True)


if __name__ == '__main__':
    main()
