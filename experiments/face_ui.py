"""Local browser UI and serialized GPU jobs for face identity experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import traceback
from typing import Literal
from urllib.parse import urlparse
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from experiments.face_boundary import (
    DEFAULT_IDENTITY_SIMILARITY, DEFAULT_MAX_COSINE_DISTANCE, DEFAULT_NETWORK, DEFAULT_SEED_TRUNCATION_PSI,
    REFLECTION_POLICIES, ROOT, IdentityOracle, LatentSlice, billiard, download_network,
    inside, load_generator, load_matcher, plot_map, save_run, similarity_grid, write_video,
)
from experiments.face_projection import align_photo, load_perceptual_model, project_photo, read_photo, save_projection


UI_ROOT = Path(os.environ.get('FACE_UI_OUTPUT_DIR', ROOT / 'outputs/ui')).expanduser().resolve()
STATIC_ROOT = Path(__file__).with_name('ui')
ASSETS = {'reference.png', 'aligned.png', 'reconstruction.png', 'projection.npz', 'projection.json',
          'billiard.mp4', 'identity_map.png', 'trajectory.npz', 'metadata.json'}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='face-gpu')
guard = threading.Lock()
jobs = {}
access_token = os.environ.get('FACE_UI_ACCESS_TOKEN')


class Cancelled(Exception):
    pass


@dataclass
class Job:
    kind: str
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = 'queued'
    stage: str = 'Queued'
    progress: float = 0
    result: dict | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def directory(self):
        return UI_ROOT / self.identifier

    def update(self, stage, progress):
        if self.cancel.is_set():
            raise Cancelled()
        with guard:
            self.stage, self.progress = stage, float(progress)

    def callback(self, stage, start, length):
        return lambda completed, total: self.update(stage, start + length * completed / max(total, 1))

    def snapshot(self):
        return {'id': self.identifier, 'kind': self.kind, 'status': self.status, 'stage': self.stage,
                'progress': self.progress, 'result': self.result, 'error': self.error, 'created': self.created}


class Runtime:
    def __init__(self):
        self.base = None
        self.matcher = None
        self.generator = None
        self.perceptual = None
        self.centers = {}
        self.device = 'Not loaded'

    def ensure(self, job):
        if self.base is not None:
            return
        import torch

        torch.set_num_threads(4)
        job.update('Loading StyleGAN and ArcFace', 0.02)
        if not DEFAULT_NETWORK.exists():
            download_network()
        self.generator = load_generator(device=os.environ.get('FACE_UI_DEVICE', 'auto'))
        self.matcher = load_matcher(device=str(next(self.generator.parameters()).device).split(':')[0])
        self.device = torch.cuda.get_device_name() if torch.cuda.is_available() and next(self.generator.parameters()).is_cuda else 'CPU'
        job.update('Fitting shared W slice', 0.06)
        self.base = LatentSlice(self.generator)


runtime = Runtime()


@asynccontextmanager
async def lifespan(application):
    try:
        yield
    finally:
        with guard:
            for job in jobs.values():
                if job.status in ('queued', 'running'):
                    job.cancel.set()
        executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title='Identity Playground', docs_url=None, redoc_url=None, lifespan=lifespan)


class RunSettings(BaseModel):
    center_id: str = Field(pattern='^[0-9a-f]{32}$')
    step_size: float = Field(default=0.07, ge=0.01, le=0.4)
    max_cosine_distance: float = Field(default=DEFAULT_MAX_COSINE_DISTANCE, ge=0.05, le=0.9)
    reflection_policy: Literal['mirror', 'random_inward', 'center', 'grazing'] = 'mirror'
    duration: int = Field(default=6, ge=2, le=30)
    trajectory_seed: int = Field(default=11, ge=0, le=2147483647)
    grid_size: int = Field(default=25, ge=9, le=51)
    extent: float = Field(default=3, ge=1, le=6)


def asset_url(identifier, filename):
    return f'/api/files/{identifier}/{filename}'


def submit(kind, operation):
    with guard:
        if sum(job.status in ('queued', 'running') for job in jobs.values()) >= 4:
            raise HTTPException(429, 'The GPU queue is full; wait for a job to finish')
        job = Job(kind)
        jobs[job.identifier] = job
    job.directory.mkdir(parents=True, exist_ok=True)

    def execute():
        try:
            job.update('Starting', 0)
            with guard:
                job.status = 'running'
            result = operation(job)
            job.update('Complete', 1)
            with guard:
                job.result, job.status = result, 'complete'
        except Cancelled:
            with guard:
                job.status, job.stage = 'cancelled', 'Cancelled'
        except Exception as error:
            traceback.print_exc()
            with guard:
                job.status, job.stage, job.error = 'failed', 'Failed', str(error)

    executor.submit(execute)
    return {'job_id': job.identifier}


def prepare_center(job, mode, seed, steps, data, already_aligned):
    runtime.ensure(job)
    if len(runtime.centers) >= 50:
        raise ValueError('This session has 50 prepared centers; restart the server for a new session')
    job.update('Preparing starting image', 0.1)
    reference = None
    info = {}
    if mode == 'mean':
        latent = runtime.base
    elif mode == 'seed':
        latent = runtime.base.with_seed(seed)
    else:
        job.update('Detecting and aligning the uploaded face', 0.12)
        reference = align_photo(read_photo(data), runtime.matcher, runtime.generator.img_resolution, already_aligned)
        Image.fromarray(reference).save(job.directory / 'aligned.png')
        job.update('Loading projection model', 0.15)
        if runtime.perceptual is None:
            runtime.perceptual = load_perceptual_model(runtime.base.device, job.callback('Downloading projection model', 0.15, 0.02))
        anchor, info = project_photo(runtime.generator, reference, runtime.perceptual, steps,
                                     job.callback('Projecting image into shared W', 0.18, 0.73))
        latent = runtime.base.with_center_w(anchor)
    job.update('Checking the reconstructed identity', 0.94)
    oracle = IdentityOracle(latent, runtime.matcher, reference_image=reference)
    reconstructed = latent.render([0, 0])[0]
    measured = oracle([0, 0])
    score = measured.score if measured.valid else None
    if mode == 'image':
        info.update(already_aligned=already_aligned, similarity=score,
                    reference='aligned uploaded image', center='projected shared W')
        save_projection(job.directory, reference, reconstructed, latent.anchor, info, data)
    else:
        Image.fromarray(oracle.reference).save(job.directory / 'aligned.png')
        Image.fromarray(reconstructed).save(job.directory / 'reconstruction.png')
    Image.fromarray(oracle.reference).save(job.directory / 'reference.png')
    runtime.centers[job.identifier] = {'slice': latent, 'reference': reference, 'mode': mode,
                                      'projection': info, 'directory': job.directory}
    return {'center_id': job.identifier, 'mode': mode, 'label': latent.center_label, 'seed': latent.seed,
            'truncation_psi': latent.truncation_psi,
            'similarity': score, 'distance': None if score is None else 1 - score,
            'device': runtime.device, 'projection': info,
            'reference_url': asset_url(job.identifier, 'aligned.png'),
            'reconstruction_url': asset_url(job.identifier, 'reconstruction.png'),
            'projection_url': asset_url(job.identifier, 'projection.npz') if mode == 'image' else None}


def run_experiment(job, settings):
    center = runtime.centers.get(settings.center_id)
    if center is None:
        raise ValueError('Prepare the starting image again before running')
    oracle = IdentityOracle(center['slice'], runtime.matcher, reference_image=center['reference'])
    threshold = 1 - settings.max_cosine_distance
    initial = oracle([0, 0])
    if not inside(initial, threshold):
        raise ValueError('The projected center does not meet this identity threshold. '
                         'Try more projection steps, a better aligned photo, or a looser threshold.')
    grid = similarity_grid(oracle, extent=settings.extent, size=settings.grid_size,
                           progress=job.callback('Mapping the identity boundary', 0, 0.35))
    trace = billiard(oracle, frames=settings.duration * 24, threshold=threshold,
                     step_size=settings.step_size, reflection_policy=settings.reflection_policy,
                     seed=settings.trajectory_seed, progress=job.callback('Tracing the trajectory', 0.35, 0.35))
    job.update('Saving measurements', 0.71)
    save_run(job.directory, oracle, trace, grid=grid,
             extra={'ui_settings': settings.model_dump(), 'center_job_id': settings.center_id,
                    'projection': center['projection'], 'fps': 24})
    figure = plot_map(grid, threshold, trace['positions'])
    figure.savefig(job.directory / 'identity_map.png', dpi=150)
    import matplotlib.pyplot as plt

    plt.close(figure)
    write_video(job.directory / 'billiard.mp4', oracle, trace, grid, fps=24,
                progress=job.callback('Rendering the video', 0.73, 0.26))
    metadata = json.loads((job.directory / 'metadata.json').read_text())
    return {'video_url': asset_url(job.identifier, 'billiard.mp4'),
            'map_url': asset_url(job.identifier, 'identity_map.png'),
            'trajectory_url': asset_url(job.identifier, 'trajectory.npz'),
            'metadata_url': asset_url(job.identifier, 'metadata.json'),
            'settings': settings.model_dump(), 'label': center['slice'].center_label,
            'minimum_similarity': float(trace['scores'].min()), 'collisions': len(trace['events']),
            'fallbacks': sum('fallback' in event['reflection'] for event in trace['events']),
            'holds': sum(event['reflection'] == 'hold' for event in trace['events']),
            'radius_collisions': sum(event['reason'] == 'radius_limit' for event in trace['events']),
            'invalid_cells': int(np.count_nonzero(grid['reasons'] != 'ok')),
            'diversity': metadata['diversity']}


@app.middleware('http')
async def same_origin(request: Request, call_next):
    supplied_token = request.query_params.get('token', '')
    authorized = not access_token or secrets.compare_digest(request.cookies.get('face_ui_session', ''), access_token)
    token_login = bool(access_token and supplied_token and secrets.compare_digest(supplied_token, access_token))
    if not authorized and not token_login:
        return JSONResponse({'detail': 'Open the token-bearing URL printed by the GPU server to authorize this browser.'}, status_code=401)
    origin = request.headers.get('origin')
    if request.method == 'POST' and origin and urlparse(origin).netloc != request.headers.get('host'):
        return JSONResponse({'detail': 'Use the UI from the same origin'}, status_code=403)
    response = await call_next(request)
    if token_login:
        response.set_cookie('face_ui_session', access_token, httponly=True, samesite='strict')
    return response


@app.get('/api/config')
def config():
    return {'policies': REFLECTION_POLICIES, 'default_distance': DEFAULT_MAX_COSINE_DISTANCE,
            'seed_truncation_psi': DEFAULT_SEED_TRUNCATION_PSI,
            'default_similarity': DEFAULT_IDENTITY_SIMILARITY, 'device': runtime.device,
            'space': 'shared W', 'dimensions': 2}


@app.post('/api/centers')
async def prepare(mode: Literal['mean', 'seed', 'image'] = Form('mean'),
                  seed: int = Form(23, ge=0, le=2147483647),
                  steps: int = Form(250, ge=50, le=1000),
                  already_aligned: bool = Form(False), photo: UploadFile | None = File(None)):
    data = None
    if photo is not None:
        try:
            data = await photo.read(10 * 1024 * 1024 + 1)
        finally:
            await photo.close()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(413, 'Choose a photo smaller than 10 MB')
    if mode == 'image' and not data:
        raise HTTPException(422, 'Choose a starting photo')
    return submit('prepare', lambda job: prepare_center(job, mode, seed, steps, data, already_aligned))


@app.post('/api/runs')
def run(settings: RunSettings):
    if settings.center_id not in runtime.centers:
        raise HTTPException(404, 'Prepare a starting image first')
    return submit('run', lambda job: run_experiment(job, settings))


@app.get('/api/jobs')
def list_jobs():
    with guard:
        return [job.snapshot() for job in list(jobs.values())[-30:]][::-1]


@app.get('/api/jobs/{identifier}')
def get_job(identifier: str):
    with guard:
        job = jobs.get(identifier)
        if job is None:
            raise HTTPException(404, 'Job not found')
        return job.snapshot()


@app.post('/api/jobs/{identifier}/cancel')
def cancel_job(identifier: str):
    with guard:
        job = jobs.get(identifier)
        if job is None:
            raise HTTPException(404, 'Job not found')
        if job.status in ('queued', 'running'):
            job.cancel.set()
        return {'status': job.status}


@app.get('/api/files/{identifier}/{filename}')
def get_asset(identifier: str, filename: str):
    if len(identifier) != 32 or any(character not in '0123456789abcdef' for character in identifier) or filename not in ASSETS:
        raise HTTPException(404, 'File not found')
    path = UI_ROOT / identifier / filename
    if not path.is_file():
        raise HTTPException(404, 'File not found')
    return FileResponse(path, filename=filename if path.suffix in ('.npz', '.json') else None)


app.mount('/', StaticFiles(directory=str(STATIC_ROOT), html=True), name='ui')


def main():
    global access_token, UI_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=UI_ROOT,
                        help='Directory for this server\'s uploads, projections and runs')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    UI_ROOT = args.output_dir.expanduser().resolve()
    UI_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ['FACE_UI_DEVICE'] = args.device
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not access_token:
        access_token = secrets.token_urlsafe(32)
    print(f'GPU node: {socket.getfqdn()}; keep this allocation open.', flush=True)
    print(f'Outputs: {UI_ROOT}', flush=True)
    print(f'After forwarding port {args.port}, open http://127.0.0.1:{args.port}/'
          + (f'?token={access_token}' if access_token else ''), flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
