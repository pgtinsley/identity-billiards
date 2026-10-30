"""Run a reproducible pilot or download its generator without opening Jupyter."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from experiments.face_boundary import (
    DEFAULT_IDENTITY_SIMILARITY, DEFAULT_NETWORK, REFLECTION_POLICIES, ROOT, IdentityOracle, LatentSlice, billiard, diversity_summary,
    download_network, load_generator, load_matcher, plot_map, radial_scan,
    save_run, similarity_grid, write_video,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, default=DEFAULT_NETWORK)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--matcher', default='buffalo_l')
    parser.add_argument('--space', choices=['w', 'z'], default='w')
    parser.add_argument('--basis', choices=['pca', 'random'], default='pca')
    parser.add_argument('--center', choices=['mean', 'seed'], default='mean')
    parser.add_argument('--reflection-policy', choices=REFLECTION_POLICIES, default='mirror')
    parser.add_argument('--seed', type=int, default=23, help='Anchor seed when --center seed is selected')
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument('--threshold', type=float,
                            help=f'Minimum cosine similarity (default: {DEFAULT_IDENTITY_SIMILARITY:.2f})')
    thresholds.add_argument('--max-cosine-distance', type=float,
                            help='Maximum cosine distance, equivalent to 1 minus minimum similarity')
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--step-size', type=float, default=0.07)
    parser.add_argument('--grid-size', type=int, default=25)
    parser.add_argument('--extent', type=float, default=3.0)
    parser.add_argument('--max-radius', type=float, default=6.0)
    parser.add_argument('--rays', type=int, default=24)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.max_cosine_distance is not None:
        if not 0 < args.max_cosine_distance < 2:
            parser.error('--max-cosine-distance must be finite and between 0 and 2, exclusive')
        args.threshold = 1.0 - args.max_cosine_distance
    elif args.threshold is None:
        args.threshold = DEFAULT_IDENTITY_SIMILARITY
    if not -1 < args.threshold < 1:
        parser.error('--threshold must be finite and between -1 and 1, exclusive')
    import torch

    torch.set_num_threads(4)
    print(f'Identity boundary: cosine similarity >= {args.threshold:.3f} '
          f'(cosine distance <= {1.0 - args.threshold:.3f})', flush=True)
    if not args.network.exists():
        if args.network != DEFAULT_NETWORK:
            parser.error('Custom network does not exist; only the default is downloaded automatically')
        download_network(args.network)
    if args.download_only:
        print(args.network)
        return
    started = time.perf_counter()
    generator = load_generator(args.network, device=args.device)
    print(f'Generator device: {next(generator.parameters()).device}', flush=True)
    matcher = load_matcher(args.matcher, device=args.device)
    latent = LatentSlice(generator, seed=args.seed, space=args.space, basis_kind=args.basis, center=args.center)
    oracle = IdentityOracle(latent, matcher)
    print(f'Anchor at {latent.center_label}; self-similarity = {oracle([0, 0]).score:.6f}', flush=True)
    grid = similarity_grid(oracle, extent=args.extent, size=args.grid_size) if args.grid_size else None
    angles = np.linspace(0, 2 * np.pi, args.rays, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    rays = radial_scan(oracle, directions, args.threshold, max_radius=args.max_radius) if args.rays else []
    print('Tracing billiard...', flush=True)
    trace = billiard(oracle, threshold=args.threshold, frames=args.frames, step_size=args.step_size,
                     max_radius=args.max_radius, reflection_policy=args.reflection_policy)
    output = args.output or ROOT / 'outputs' / datetime.now(timezone.utc).strftime('pilot-%Y%m%d-%H%M%S')
    save_run(output, oracle, trace, args.network, grid=grid,
             extra={'rays': rays, 'ray_directions': directions.tolist(), 'fps': args.fps,
                    'elapsed_before_video_seconds': time.perf_counter() - started})
    if grid:
        figure = plot_map(grid, args.threshold, trace['positions'])
        figure.savefig(output / 'identity_map.png', dpi=160)
        import matplotlib.pyplot as plt

        plt.close(figure)
    video = write_video(output / 'billiard.mp4', oracle, trace, grid, fps=args.fps)
    print(diversity_summary(trace['embeddings']), flush=True)
    print(f'{len(trace["events"])} events; elapsed {time.perf_counter() - started:.1f}s; video: {video}', flush=True)


if __name__ == '__main__':
    main()
