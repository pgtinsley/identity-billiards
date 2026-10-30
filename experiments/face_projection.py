"""Five-landmark FFHQ-style alignment and perceptual inversion into shared W."""

from pathlib import Path
import hashlib
import io
import json

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from experiments.face_boundary import MODEL_ROOT, face_embedding, file_sha256


PERCEPTUAL_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
PERCEPTUAL_PATH = MODEL_ROOT / 'vgg16.pt'


def read_photo(data):
    if len(data) > 10 * 1024 * 1024:
        raise ValueError('Choose a photo smaller than 10 MB')
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError('Choose a photo smaller than 25 megapixels')
            image = ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('Choose a readable JPEG, PNG, or WebP image') from error
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image


def align_photo(image, matcher, resolution=256, already_aligned=False):
    import cv2

    image = image.convert('RGB')
    rgb = np.asarray(image)
    faces = matcher.get(np.ascontiguousarray(rgb[:, :, ::-1]))
    if len(faces) != 1:
        raise ValueError('The starting photo must contain exactly one detectable face')
    if already_aligned:
        if image.width != image.height:
            raise ValueError('An already-aligned FFHQ image must be square')
        return np.asarray(image.resize((resolution, resolution), Image.Resampling.LANCZOS))
    landmarks = np.asarray(faces[0].kps, dtype=np.float64)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        raise ValueError('Could not locate five reliable facial landmarks')
    eye_center = landmarks[:2].mean(axis=0)
    eye_axis = landmarks[1] - landmarks[0]
    mouth_axis = landmarks[3:5].mean(axis=0) - eye_center
    horizontal = eye_axis - mouth_axis[::-1] * np.array([-1, 1])
    length = np.linalg.norm(horizontal)
    if length < 1e-6:
        raise ValueError('Facial landmarks are degenerate; choose another photo')
    horizontal *= max(np.linalg.norm(eye_axis) * 2, np.linalg.norm(mouth_axis) * 1.8) / length
    vertical = horizontal[::-1] * np.array([-1, 1])
    center = eye_center + mouth_axis * 0.1
    quad = np.stack([center - horizontal - vertical, center - horizontal + vertical,
                     center + horizontal + vertical, center + horizontal - vertical])
    padding = int(max(0, -quad.min(), quad[:, 0].max() - image.width,
                      quad[:, 1].max() - image.height)) + 2
    padded = cv2.copyMakeBorder(rgb, padding, padding, padding, padding, cv2.BORDER_REFLECT_101)
    quad += padding + 0.5
    aligned = Image.fromarray(padded).transform((resolution, resolution), Image.Transform.QUAD,
                                               tuple(quad.flatten()), Image.Resampling.BICUBIC)
    result = np.asarray(aligned)
    if not face_embedding(matcher, result).valid:
        raise ValueError('The aligned crop is not a valid face; try an already-aligned FFHQ crop')
    return result


def load_perceptual_model(device, progress=None):
    import requests
    import torch

    if not PERCEPTUAL_PATH.is_file():
        temporary = PERCEPTUAL_PATH.with_suffix('.pt.part')
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(PERCEPTUAL_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if progress:
                        progress(0, 1)
                    stream.write(chunk)
        temporary.replace(PERCEPTUAL_PATH)
    model = torch.jit.load(str(PERCEPTUAL_PATH), map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def project_photo(generator, aligned, perceptual, steps=250, progress=None):
    import torch
    import torch.nn.functional as functional

    if not 1 <= steps <= 1000 or aligned.shape != (generator.img_resolution, generator.img_resolution, 3):
        raise ValueError('Invalid projection steps or target image size')
    device = next(generator.parameters()).device
    target = torch.as_tensor(aligned.copy(), device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    with torch.no_grad():
        target_features = perceptual(target, resize_images=False, return_lpips=True)
        target_small = functional.interpolate(target / 255, size=(64, 64), mode='area')
    midpoint = generator.mapping.w_avg.detach()[None]
    shared = torch.nn.Parameter(midpoint.clone())
    optimizer = torch.optim.Adam([shared], lr=0.08)
    best_loss = float('inf')
    best_w = shared.detach().clone()
    history = []
    for index in range(steps):
        if progress:
            progress(index, steps)
        fraction = (index + 1) / steps
        ramp = 0.5 - 0.5 * np.cos(min(1, (1 - fraction) / 0.25) * np.pi)
        optimizer.param_groups[0]['lr'] = 0.08 * ramp * min(1, fraction / 0.05)
        styles = shared[:, None].repeat(1, generator.num_ws, 1)
        synthetic = (generator.synthesis(styles, noise_mode='const', force_fp32=True) * 127.5 + 128).clamp(0, 255)
        features = perceptual(synthetic, resize_images=False, return_lpips=True)
        perceptual_loss = (features - target_features).square().sum()
        small = functional.interpolate(synthetic / 255, size=(64, 64), mode='area')
        pixel_loss = (small - target_small).square().mean()
        loss = perceptual_loss + 0.5 * pixel_loss + 1e-4 * (shared - midpoint).square().mean()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise RuntimeError('Projection diverged; choose a different photo')
        if value < best_loss:
            best_loss = value
            best_w = shared.detach().clone()
        if index % 10 == 0 or index == steps - 1:
            history.append({'step': index + 1, 'loss': value})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if progress:
        progress(steps, steps)
    return best_w[0].cpu().numpy(), {'loss': best_loss, 'steps': steps, 'history': history,
                                    'method': 'shared W; VGG16 LPIPS + pixel loss + weak W regularization',
                                    'noise_mode': 'const', 'perceptual_model_sha256': file_sha256(PERCEPTUAL_PATH)}


def save_projection(directory, aligned, reconstructed, anchor_w, info, source_data):
    directory = Path(directory)
    Image.fromarray(aligned).save(directory / 'aligned.png')
    Image.fromarray(reconstructed).save(directory / 'reconstruction.png')
    np.savez_compressed(directory / 'projection.npz', w=anchor_w)
    info = dict(info, source_sha256=hashlib.sha256(source_data).hexdigest(),
                alignment='five_landmark_ffhq_approximation' if not info['already_aligned'] else 'supplied_ffhq_crop')
    (directory / 'projection.json').write_text(json.dumps(info, indent=2) + '\n')
