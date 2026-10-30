"""Synthetic-face identity regions, finite-difference billiards, and video output."""

from dataclasses import dataclass
import copy
from functools import lru_cache, partial
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NETWORK_URL = (
    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/'
    'files/stylegan2-ffhq-256x256.pkl'
)
MODEL_ROOT = Path(os.environ.get('FACE_MODELS_DIR', ROOT / 'models')).expanduser().resolve()
DEFAULT_NETWORK = MODEL_ROOT / 'stylegan2-ffhq-256x256.pkl'
DEFAULT_MAX_COSINE_DISTANCE = 0.55
DEFAULT_IDENTITY_SIMILARITY = 0.45
DEFAULT_SEED_TRUNCATION_PSI = 0.75
IDENTITY_THRESHOLD_SOURCES = [
    'https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/person/config.py',
    'https://github.com/serengil/deepface/blob/master/deepface/config/threshold.py',
]
REFLECTION_POLICIES = {
    'mirror': {'label': 'Mirror', 'description': 'Reflect around the local identity-boundary normal; preserves the tangential component.'},
    'random_inward': {'label': 'Random inward', 'description': 'Choose a seeded random direction from the inward hemisphere at each collision.'},
    'center': {'label': 'Toward center', 'description': 'Aim toward the fixed starting point; uses the inward normal if that direction is locally outward.'},
    'grazing': {'label': 'Glide along boundary', 'description': 'Favor the incoming tangent with a small inward component; explores near the boundary.'},
}


@dataclass
class Observation:
    score: float
    valid: bool = True
    reason: str = 'ok'
    embedding: np.ndarray | None = None
    detection_score: float = float('nan')


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Expected a finite, nonzero direction')
    return vector / length


def download_network(path=DEFAULT_NETWORK):
    import requests

    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pkl.part')
    with requests.get(NETWORK_URL, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temporary.replace(path)
    return path


def _reference_operation(operation, *args, **kwargs):
    kwargs['impl'] = 'ref'
    return operation(*args, **kwargs)


def load_generator(network=DEFAULT_NETWORK, device='auto', reference_ops=True):
    import torch
    import legacy
    from torch_utils.ops import bias_act, upfirdn2d

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if reference_ops:
        for module, name in [(bias_act, 'bias_act'), (upfirdn2d, 'upfirdn2d')]:
            original = getattr(module, name)
            if not isinstance(original, partial):
                setattr(module, name, partial(_reference_operation, original))
    elif any(isinstance(function, partial) for function in
             [bias_act.bias_act, upfirdn2d.upfirdn2d]):
        raise RuntimeError('Restart Python before switching from reference to CUDA ops')
    with Path(network).open('rb') as stream:
        generator = legacy.load_network_pkl(stream)['G_ema']
    generator = generator.eval().requires_grad_(False).to(device)
    if generator.c_dim:
        raise ValueError('These experiments require an unconditional face generator')
    return generator


def load_matcher(name='buffalo_l', device='auto', root=None, source=None):
    import torch

    from experiments.source_paths import insightface_source

    source = insightface_source(ROOT, source)
    package = source / 'python-package'
    if not (package / 'insightface').is_dir():
        raise FileNotFoundError(f'InsightFace source not found: {package}. '
                                'Run git submodule update --init --recursive or set INSIGHTFACE_REPO.')
    sys.path.insert(0, str(package))
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    use_cuda = device == 'cuda' or (device == 'auto' and torch.cuda.is_available())
    if use_cuda and sys.platform == 'linux':
        import ctypes

        library_root = Path(torch.__file__).resolve().parents[1] / 'nvidia'
        for relative, pattern in [('cuda_nvrtc/lib', 'libnvrtc.so.*'), ('cudnn/lib', 'libcudnn*.so.9')]:
            for library in sorted((library_root / relative).glob(pattern)):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    matcher = FaceAnalysis(
        name=name, root=str(root or MODEL_ROOT / 'insightface'),
        allowed_modules=['detection', 'recognition'], providers=providers, sess_options=options,
    )
    matcher.prepare(ctx_id=0 if use_cuda else -1, det_size=(320, 320), det_thresh=0.5)
    if 'recognition' not in matcher.models:
        raise RuntimeError('Selected package has no recognition model')
    actual = {task: model.session.get_providers() for task, model in matcher.models.items()}
    if use_cuda and any('CUDAExecutionProvider' not in entries for entries in actual.values()):
        raise RuntimeError(f'CUDA requested but ONNX Runtime fell back: {actual}')
    print(f'InsightFace source: {source}; providers: {actual}')
    return matcher


def face_embedding(matcher, rgb):
    faces = matcher.get(np.ascontiguousarray(rgb[:, :, ::-1]))
    if len(faces) != 1:
        return Observation(float('nan'), False, 'no_face' if not faces else 'multiple_faces')
    embedding = np.asarray(faces[0].normed_embedding, dtype=np.float64)
    if not np.isfinite(embedding).all() or np.linalg.norm(embedding) < 1e-8:
        return Observation(float('nan'), False, 'invalid_embedding')
    return Observation(1.0, embedding=unit(embedding), detection_score=float(faces[0].det_score))


class LatentSlice:
    """Affine slice of shared W or Z. W axes use empirical principal SD units."""

    def __init__(self, generator, seed=23, dimensions=2, space='w', basis_seed=7,
                 basis_kind='pca', mapping_samples=2048, truncation_psi=None, center='mean'):
        import torch

        if truncation_psi is None:
            truncation_psi = DEFAULT_SEED_TRUNCATION_PSI if center == 'seed' else 1.0
        if space not in ('w', 'z') or not 1 <= dimensions <= generator.z_dim:
            raise ValueError('Choose w or z and a valid slice dimension')
        if basis_kind not in ('pca', 'random') or not 0 < truncation_psi <= 1:
            raise ValueError('Choose pca/random basis and truncation in (0, 1]')
        if center not in ('mean', 'seed'):
            raise ValueError('Choose mean or seed center')
        self.generator = generator
        self.device = next(generator.parameters()).device
        self.center = center
        self.seed = seed if center == 'seed' else None
        self.center_definition = 'seeded_gaussian_z'
        if center == 'mean':
            self.center_definition = 'mapping.w_avg' if space == 'w' else 'gaussian_prior_zero'
        self.space = space
        self.truncation_psi = truncation_psi
        self.basis_truncation_psi = 1.0 if space == 'w' else None
        self.basis_kind = basis_kind if space == 'w' else 'orthonormal_random'
        self.basis_seed = basis_seed
        self.mapping_samples = mapping_samples
        self.dimensions = dimensions
        self.anchor_z = None
        random = np.random.default_rng(basis_seed)
        with torch.inference_mode():
            if center == 'mean' and space == 'w':
                learned_mean = getattr(generator.mapping, 'w_avg', None)
                if learned_mean is None:
                    raise ValueError('Generator has no learned mapping.w_avg for mean centering')
                self.anchor = learned_mean.detach().cpu().numpy().copy()
            else:
                self.anchor_z = (np.zeros(generator.z_dim, dtype=np.float32) if center == 'mean' else
                                 np.random.RandomState(seed).randn(generator.z_dim).astype(np.float32))
                self.anchor = (self._map(self.anchor_z[None])[0, 0].cpu().numpy()
                               if space == 'w' else self.anchor_z.copy())
            if space == 'w':
                if mapping_samples <= generator.w_dim:
                    raise ValueError('Use more mapping samples than W dimensions')
                mapped = []
                for start in range(0, mapping_samples, 128):
                    batch = random.standard_normal((min(128, mapping_samples - start), generator.z_dim))
                    mapped.append(self._map(batch, truncation_psi=self.basis_truncation_psi)[:, 0].cpu().numpy())
                covariance = np.cov(np.concatenate(mapped), rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                transform = np.sqrt(np.maximum(eigenvalues[::-1], 0))[:, None] * eigenvectors[:, ::-1].T
                if basis_kind == 'pca':
                    self.basis = transform[:dimensions]
                else:
                    rotation, _ = np.linalg.qr(random.standard_normal((generator.w_dim, dimensions)))
                    self.basis = rotation.T @ transform
            else:
                rotation, _ = np.linalg.qr(random.standard_normal((generator.z_dim, dimensions)))
                self.basis = rotation.T

    @property
    def center_label(self):
        if self.center == 'seed':
            return f'seed {self.seed}'
        if self.center == 'projected':
            return 'projected upload'
        return 'learned W mean' if self.space == 'w' else 'Z prior mean'

    def with_center_w(self, anchor, center='projected', seed=None, anchor_z=None):
        anchor = np.asarray(anchor, dtype=np.float32)
        if self.space != 'w' or center not in ('projected', 'seed'):
            raise ValueError('A replacement center requires shared W and projected/seed mode')
        if anchor.shape != (self.generator.w_dim,) or not np.isfinite(anchor).all():
            raise ValueError('Invalid shared W center')
        result = copy.copy(self)
        result.anchor = anchor.copy()
        result.center = center
        result.center_definition = 'projected_upload' if center == 'projected' else 'seeded_gaussian_z'
        result.seed = seed if center == 'seed' else None
        result.anchor_z = None if anchor_z is None else np.asarray(anchor_z, dtype=np.float32).copy()
        if center == 'projected':
            result.truncation_psi = 1.0
        return result

    def with_seed(self, seed, truncation_psi=DEFAULT_SEED_TRUNCATION_PSI):
        import torch

        if self.space != 'w' or not 0 < truncation_psi <= 1:
            raise ValueError('A seeded W center requires truncation in (0, 1]')
        anchor_z = np.random.RandomState(seed).randn(self.generator.z_dim).astype(np.float32)
        with torch.inference_mode():
            anchor = self._map(anchor_z[None], truncation_psi=truncation_psi)[0, 0].cpu().numpy()
        result = self.with_center_w(anchor, center='seed', seed=seed, anchor_z=anchor_z)
        result.truncation_psi = truncation_psi
        return result

    def _map(self, latents, truncation_psi=None):
        import torch

        latents = torch.as_tensor(latents, dtype=torch.float32, device=self.device)
        labels = torch.zeros((len(latents), 0), device=self.device)
        return self.generator.mapping(latents, labels,
                                      truncation_psi=self.truncation_psi if truncation_psi is None else truncation_psi)

    def render(self, coordinates):
        import torch

        coordinates = np.atleast_2d(coordinates)
        if coordinates.shape[1] != self.dimensions or not np.isfinite(coordinates).all():
            raise ValueError('Invalid slice coordinates')
        latent = self.anchor + coordinates @ self.basis
        with torch.inference_mode():
            if self.space == 'w':
                shared = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
                styles = shared[:, None, :].repeat(1, self.generator.num_ws, 1)
            else:
                styles = self._map(latent)
            images = self.generator.synthesis(styles, noise_mode='const', force_fp32=True)
            images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return images.cpu().numpy()


class IdentityOracle:
    def __init__(self, latent_slice, matcher, reference_image=None):
        self.slice = latent_slice
        self.matcher = matcher
        self.reference = (latent_slice.render(np.zeros(latent_slice.dimensions))[0]
                          if reference_image is None else np.asarray(reference_image, dtype=np.uint8).copy())
        self.reference_kind = 'generated_center' if reference_image is None else 'aligned_upload'
        self.reference_observation = face_embedding(matcher, self.reference)
        if not self.reference_observation.valid:
            raise ValueError(f'Anchor at {latent_slice.center_label}: {self.reference_observation.reason}; '
                             'adjust the matcher or explicitly select a different center')

    def __call__(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.shape != (self.slice.dimensions,) or not np.isfinite(coordinates).all():
            raise ValueError('Invalid oracle coordinates')
        return self._evaluate(tuple(coordinates))

    @lru_cache(maxsize=4096)
    def _evaluate(self, coordinates):
        image = self.slice.render(coordinates)[0]
        observation = face_embedding(self.matcher, image)
        if observation.valid:
            observation.score = float(np.clip(
                observation.embedding @ self.reference_observation.embedding, -1, 1))
        return observation


def inside(observation, threshold):
    return observation.valid and np.isfinite(observation.score) and observation.score >= threshold


def similarity_grid(oracle, extent=3.0, size=25, progress=None):
    if oracle.slice.dimensions != 2 or size < 2 or extent <= 0:
        raise ValueError('A positive extent, size >= 2, and a 2D slice are required')
    axis = np.linspace(-extent, extent, size)
    scores = np.full((size, size), np.nan)
    reasons = np.empty((size, size), dtype='U24')
    from tqdm.auto import tqdm

    for row, vertical in enumerate(tqdm(axis, desc='Identity map')):
        for column, horizontal in enumerate(axis):
            observation = oracle([horizontal, vertical])
            scores[row, column] = observation.score
            reasons[row, column] = observation.reason
        if progress:
            progress(row + 1, size)
    return {'axis': axis, 'scores': scores, 'reasons': reasons}


def first_exit(oracle, direction, threshold=DEFAULT_IDENTITY_SIMILARITY, increment=0.2, max_radius=6.0,
               bisections=10):
    if not -1 < threshold < 1 or increment <= 0 or max_radius <= 0:
        raise ValueError('Invalid first-exit parameters')
    direction = unit(direction)
    if not inside(oracle(np.zeros_like(direction)), threshold):
        raise ValueError('The anchor must be valid and inside the boundary')
    lower = 0.0
    for upper in np.append(np.arange(increment, max_radius, increment), max_radius):
        observation = oracle(upper * direction)
        if not inside(observation, threshold):
            for _ in range(bisections):
                middle = (lower + upper) / 2
                if inside(oracle(middle * direction), threshold):
                    lower = middle
                else:
                    upper = middle
            rejected = oracle(upper * direction)
            reason = 'identity' if rejected.valid else rejected.reason
            return {'radius': float(lower), 'censored': False, 'reason': reason}
        lower = upper
    return {'radius': float(max_radius), 'censored': True, 'reason': 'radius_limit'}


def radial_scan(oracle, directions, threshold=DEFAULT_IDENTITY_SIMILARITY, **kwargs):
    from tqdm.auto import tqdm

    return [first_exit(oracle, direction, threshold, **kwargs)
            for direction in tqdm(directions, desc='First exits')]


def local_metric(oracle, dimensions, epsilon=0.05):
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    derivatives = []
    for direction in np.eye(dimensions):
        positive, negative = oracle(epsilon * direction), oracle(-epsilon * direction)
        if not positive.valid or not negative.valid:
            raise ValueError('Face detection failed during local metric estimation')
        derivatives.append((positive.embedding - negative.embedding) / (2 * epsilon))
    jacobian = np.stack(derivatives)
    return jacobian @ jacobian.T


def reflection_direction(velocity, normal, position, policy, random):
    if policy not in REFLECTION_POLICIES:
        raise ValueError('Unknown reflection policy')
    if np.linalg.norm(normal) <= 1e-8 or np.dot(velocity, normal) >= 0:
        return -velocity, 'reverse_fallback'
    normal = unit(normal)
    if policy == 'mirror':
        return unit(velocity - 2 * np.dot(velocity, normal) * normal), 'normal'
    if policy == 'random_inward':
        direction = unit(random.standard_normal(len(velocity)))
        return (direction if np.dot(direction, normal) >= 0 else -direction), policy
    if policy == 'center':
        if np.linalg.norm(position) > 1e-8:
            direction = unit(-position)
            if np.dot(direction, normal) > 0:
                return direction, policy
        return normal, 'center_normal_fallback'
    tangent = velocity - np.dot(velocity, normal) * normal
    if np.linalg.norm(tangent) < 1e-8:
        tangent = random.standard_normal(len(velocity))
        tangent -= np.dot(tangent, normal) * normal
    if np.linalg.norm(tangent) < 1e-8:
        return normal, 'grazing_normal_fallback'
    return unit(unit(tangent) + 0.2 * normal), policy


def billiard(oracle, dimensions=2, threshold=DEFAULT_IDENTITY_SIMILARITY, frames=240, step_size=0.07,
             seed=11, gradient_epsilon=0.03, max_radius=6.0, substeps=4, bisections=10,
             reflection_policy='mirror', progress=None):
    if not -1 < threshold < 1 or frames < 1 or step_size <= 0 or substeps < 1:
        raise ValueError('Invalid billiard parameters')
    if gradient_epsilon <= 0 or max_radius <= 0:
        raise ValueError('Gradient epsilon and radius must be positive')
    if reflection_policy not in REFLECTION_POLICIES:
        raise ValueError('Unknown reflection policy')
    position = np.zeros(dimensions, dtype=np.float64)
    measured = oracle(position)
    if not inside(measured, threshold):
        raise ValueError('The anchor must be valid and inside the boundary')
    random = np.random.default_rng(seed)
    velocity = unit(random.standard_normal(dimensions))
    positions, observations, events = [position.copy()], [measured], []

    def evaluate(candidate):
        if np.linalg.norm(candidate) > max_radius:
            return None, 'radius_limit'
        observation = oracle(candidate)
        reason = None if inside(observation, threshold) else ('identity' if observation.valid else observation.reason)
        return observation, reason

    for frame in range(1, frames):
        remaining = step_size
        collisions = 0
        while remaining > 1e-8:
            if progress:
                progress(frame, frames)
            distance = min(remaining, step_size / substeps)
            candidate = position + distance * velocity
            candidate_measurement, reason = evaluate(candidate)
            if reason is None:
                position, measured = candidate, candidate_measurement
                remaining -= distance
                continue
            lower, upper = 0.0, distance
            contact, contact_measurement = position, measured
            for _ in range(bisections):
                middle = (lower + upper) / 2
                probe = position + middle * velocity
                probe_measurement, probe_reason = evaluate(probe)
                if probe_reason is None:
                    lower = middle
                    contact, contact_measurement = probe, probe_measurement
                else:
                    upper = middle
                    reason = probe_reason
            travel = max(0, lower - min(1e-4, lower / 2))
            backoff_rejected = False
            if travel > 0:
                inset = position + travel * velocity
                inset_measurement, inset_reason = evaluate(inset)
                if inset_reason is None:
                    position, measured = inset, inset_measurement
                else:
                    position, measured = contact, contact_measurement
                    travel = lower
                    backoff_rejected = True
            remaining -= travel
            normal = np.zeros(dimensions)
            if reason == 'radius_limit':
                normal = -position
            elif reason == 'identity':
                for axis in range(dimensions):
                    offset = np.eye(1, dimensions, axis)[0] * gradient_epsilon
                    positive, negative = oracle(position + offset), oracle(position - offset)
                    if positive.valid and negative.valid:
                        normal[axis] = (positive.score - negative.score) / (2 * gradient_epsilon)
            velocity, reflection = reflection_direction(velocity, normal, position, reflection_policy, random)
            events.append({'frame': frame, 'reason': reason, 'reflection': reflection,
                           'position': position.tolist(), 'backoff_rejected': backoff_rejected})
            collisions += 1
            if collisions >= 8:
                events.append({'frame': frame, 'reason': 'collision_limit', 'reflection': 'hold',
                               'position': position.tolist()})
                break
        if not inside(measured, threshold) or np.linalg.norm(position) > max_radius:
            raise RuntimeError('Billiard produced a frame outside the accepted region')
        positions.append(position.copy())
        observations.append(measured)
    if progress:
        progress(frames, frames)
    return {'positions': np.stack(positions), 'scores': np.array([item.score for item in observations]),
            'embeddings': np.stack([item.embedding for item in observations])
            if all(item.embedding is not None for item in observations) else None,
            'detection_scores': np.array([item.detection_score for item in observations]),
            'events': events, 'threshold': threshold, 'step_size': step_size,
            'trajectory_seed': seed, 'max_radius': max_radius, 'substeps': substeps,
            'reflection_policy': reflection_policy,
            'gradient_epsilon': gradient_epsilon, 'bisections': bisections}


def diversity_summary(embeddings, distance=0.05):
    if embeddings is None or distance <= 0:
        raise ValueError('Provide embeddings and a positive cosine-distance cutoff')
    embeddings = np.asarray(embeddings)
    selected = [0]
    for index in range(1, len(embeddings)):
        if np.min(1 - embeddings[selected] @ embeddings[index]) >= distance:
            selected.append(index)
    similarities = embeddings @ embeddings.T
    upper = similarities[np.triu_indices(len(embeddings), 1)]
    return {'frames': len(embeddings), 'greedy_representatives': len(selected),
            'representative_indices': selected, 'cosine_distance_cutoff': distance,
            'minimum_pairwise_similarity': float(upper.min()) if len(upper) else 1.0,
            'median_pairwise_similarity': float(np.median(upper)) if len(upper) else 1.0}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(oracle, network):
    import torch
    import insightface

    def revision(directory):
        result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    latent = oracle.slice
    package_dir = Path(insightface.__file__).resolve().parents[2]
    model_files = [Path(network)] + sorted(Path(oracle.matcher.model_dir).glob('*.onnx'))
    return {'stylegan_revision': revision(ROOT), 'insightface_revision': revision(package_dir),
            'insightface_source': str(package_dir), 'python': sys.version,
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch', 'numpy', 'onnxruntime-gpu', 'opencv-python', 'imageio']},
            'device': str(latent.device), 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'anchor_seed': latent.seed, 'space': latent.space, 'dimensions': latent.dimensions,
            'center': latent.center, 'center_definition': latent.center_definition,
            'identity_reference': oracle.reference_kind,
            'basis_kind': latent.basis_kind, 'basis_seed': latent.basis_seed,
            'mapping_samples': latent.mapping_samples, 'truncation_psi': latent.truncation_psi,
            'basis_truncation_psi': latent.basis_truncation_psi,
            'noise_mode': 'const', 'force_fp32': True,
            'providers': {task: model.session.get_providers() for task, model in oracle.matcher.models.items()},
            'detector_size': [320, 320], 'detector_threshold': 0.5,
            'models': {str(path): file_sha256(path) for path in model_files}}


def save_run(directory, oracle, trace, network=DEFAULT_NETWORK, grid=None, extra=None):
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {key: trace[key] for key in ['positions', 'scores', 'detection_scores', 'embeddings']
              if trace[key] is not None}
    arrays.update(anchor=oracle.slice.anchor, basis=oracle.slice.basis,
                  reference_embedding=oracle.reference_observation.embedding)
    if oracle.slice.anchor_z is not None:
        arrays['anchor_z'] = oracle.slice.anchor_z
    if grid:
        arrays.update({f'grid_{key}': value for key, value in grid.items()})
    np.savez_compressed(directory / 'trajectory.npz', **arrays)
    metadata = provenance(oracle, network)
    metadata.update({key: value for key, value in trace.items() if not isinstance(value, np.ndarray)})
    metadata['identity_threshold'] = {
        'score_metric': 'cosine_similarity', 'minimum_similarity': trace['threshold'],
        'maximum_cosine_distance': 1.0 - trace['threshold'], 'comparison': 'similarity >= threshold',
        'default_reference': {'model': 'buffalo_l', 'minimum_similarity': DEFAULT_IDENTITY_SIMILARITY,
                              'maximum_cosine_distance': DEFAULT_MAX_COSINE_DISTANCE,
                              'basis': 'upstream defaults; not calibrated on these generated faces',
                              'sources': IDENTITY_THRESHOLD_SOURCES},
    }
    metadata['diversity'] = diversity_summary(trace['embeddings'])
    metadata['extra'] = extra or {}
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    Image.fromarray(oracle.reference).save(directory / 'reference.png')
    return directory


def plot_map(grid, threshold=DEFAULT_IDENTITY_SIMILARITY, positions=None):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 5))
    axis, scores = grid['axis'], grid['scores']
    mesh = axes.pcolormesh(axis, axis, np.ma.masked_invalid(scores), shading='auto',
                           cmap='magma', vmin=0, vmax=1)
    if np.isfinite(scores).any() and np.nanmin(scores) < threshold < np.nanmax(scores):
        axes.contour(axis, axis, scores, levels=[threshold], colors=['#66ffcc'], linewidths=2)
    if positions is not None:
        axes.plot(positions[:, 0], positions[:, 1], color='#78d5ff', linewidth=0.8)
    axes.scatter([0], [0], color='white', marker='*', s=90)
    axes.set(xlabel='Slice coordinate 1', ylabel='Slice coordinate 2', aspect='equal',
             title=f'Anchor similarity; contour = {threshold:g}; blank = invalid face')
    figure.colorbar(mesh, ax=axes, label='ArcFace cosine similarity')
    figure.tight_layout()
    return figure


def write_video(path, oracle, trace, grid=None, fps=24, progress=None):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from tqdm.auto import tqdm

    matplotlib.use('Agg')
    if trace['positions'].shape[1] != 2:
        raise ValueError('The video map currently requires a 2D slice')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(5, 4.8), dpi=100, facecolor='#101525')
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111, facecolor='#101525')
    if grid is not None:
        axes.pcolormesh(grid['axis'], grid['axis'], np.ma.masked_invalid(grid['scores']),
                        shading='auto', cmap='magma', vmin=0, vmax=1)
        if np.nanmin(grid['scores']) < trace['threshold'] < np.nanmax(grid['scores']):
            axes.contour(grid['axis'], grid['axis'], grid['scores'], levels=[trace['threshold']],
                         colors=['#66ffcc'], linewidths=2)
        bound = float(grid['axis'][-1])
    else:
        bound = max(1.0, float(np.abs(trace['positions']).max()) * 1.15)
    bound = max(bound, float(np.abs(trace['positions']).max()) * 1.1)
    axes.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect='equal',
             xlabel='Latent coordinate 1', ylabel='Latent coordinate 2', title='Identity boundary / trajectory')
    axes.tick_params(colors='white')
    for label in [axes.xaxis.label, axes.yaxis.label, axes.title]:
        label.set_color('white')
    axes.scatter([0], [0], color='white', marker='*', s=80)
    trail, = axes.plot([], [], color='#87e4ff', linewidth=1.5)
    ball, = axes.plot([], [], 'o', color='#66ffcc', markersize=8)
    figure.tight_layout()
    collision_frames = {event['frame'] for event in trace['events']}
    reference = Image.fromarray(oracle.reference).resize((256, 256))
    with imageio.get_writer(str(path), fps=fps, codec='libx264', pixelformat='yuv420p',
                            macro_block_size=16, ffmpeg_log_level='error') as writer:
        for index, coordinates in enumerate(tqdm(trace['positions'], desc='Rendering video')):
            if progress:
                progress(index, len(trace['positions']))
            frame = Image.new('RGB', (1024, 640), '#101525')
            draw = ImageDraw.Draw(frame)
            current = Image.fromarray(oracle.slice.render(coordinates)[0]).resize((384, 384))
            frame.paste(current, (32, 88))
            frame.paste(reference.resize((112, 112)), (32, 496))
            trail.set_data(trace['positions'][max(0, index - 100):index + 1, 0],
                           trace['positions'][max(0, index - 100):index + 1, 1])
            ball.set_data([coordinates[0]], [coordinates[1]])
            ball.set_color('#ffcc66' if index in collision_frames else '#66ffcc')
            canvas.draw()
            frame.paste(Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3]), (496, 80))
            draw.text((32, 24), 'LATENT BILLIARDS  /  SYNTHETIC IDENTITY', fill='white', font_size=24)
            draw.text((32, 62), f'{oracle.slice.space.upper()} slice | {oracle.slice.center_label} | frame {index + 1}',
                      fill='#97b3cb', font_size=17)
            draw.text((160, 508), f'Similarity: {trace["scores"][index]:.3f} (min {trace["threshold"]:.3f})\n'
                      f'Distance: {1.0 - trace["scores"][index]:.3f} (max {1.0 - trace["threshold"]:.3f})',
                      fill='#66ffcc', font_size=19)
            draw.text((160, 570), 'Fixed reference', fill='#97b3cb', font_size=17)
            draw.text((520, 580), 'Green: identity contour   Gold: collision', fill='#97b3cb', font_size=17)
            writer.append_data(np.asarray(frame))
    if progress:
        progress(len(trace['positions']), len(trace['positions']))
    return path
