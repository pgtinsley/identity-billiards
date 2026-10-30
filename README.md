# Identity Billiards

Explore how a face generator's latent space changes appearance while a face
recognizer continues to match a fixed reference. **Identity Playground** is the
browser interface: choose a generated face or project an uploaded photo, then
watch a trajectory bounce off an identity-similarity boundary.

The current experiment uses NVIDIA's **StyleGAN2 FFHQ 256×256** checkpoint in the
StyleGAN3 runtime, with InsightFace `buffalo_l` detection and recognition. It is
a research prototype for exploring a matcher's acceptance region.

## What it does

- Starts from the learned W mean, a seeded face, or a photo projected into shared W.
- Moves through a 2D PCA slice with mirror, random-inward, toward-center, or
  boundary-gliding reflection.
- Shows the reference, reconstruction, identity boundary, and trajectory.
- Exports MP4 videos, latent coordinates, embeddings, and reproducibility metadata.
- Includes a CLI, research notebooks, and known-latent synthetic inversion controls.

Every position is compared with the **same reference face**. The default match
rule is cosine similarity ≥ **0.45** (cosine distance ≤ **0.55**), with exactly one
detected face. Seeded starts use truncation ψ = **0.75**. These settings define an
operational region; they do not establish ground-truth identity or a calibrated
false-match rate.

## Quick start

The pinned UI environment targets **Linux x86_64, Python 3.11, and NVIDIA CUDA**.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git first.
An NVIDIA GPU is recommended for interactive use; the check command also supports
CPU execution.

```bash
git clone --recurse-submodules https://github.com/pgtinsley/identity-billiards.git
cd identity-billiards
bash scripts/setup_identity_playground.sh

# Run inside your GPU allocation or on your GPU workstation:
bash scripts/check_identity_playground.sh --smoke
bash scripts/start_identity_playground.sh --host 127.0.0.1 --port 7860
```

Open the localhost URL printed by the server. Model weights download on first
use and are stored in ignored `models/`; outputs go to ignored `outputs/ui/`.
The setup script installs the UI environment; the smoke check loads the real
models, renders and embeds a face, performs one projection step, and encodes video.

If you cloned without dependencies, run `git submodule update --init --recursive`.
For a CPU smoke check, use
`bash scripts/check_identity_playground.sh --device cpu --smoke`.
The server accepts `--device cpu` too; projection and rendering will be slower.

For a remote cluster, start the server with its default network binding and
follow the [SSH tunneling guide](experiments/UI.md#connect-from-your-laptop).
Network binding enables session-token authentication. Each collaborator should
run their own environment and GPU allocation; the
[sharing guide](experiments/SHARING.md) covers source bundles and shared model caches.

## Documentation

| Guide | Contents |
| --- | --- |
| [Using the UI](experiments/UI.md) | Controls, remote access, photo projection, synthetic controls |
| [Experiment methodology](experiments/README.md) | Geometry, thresholds, CLI, outputs, interpretation |
| [Sharing and installation](experiments/SHARING.md) | Portable bundles, model caches, cluster troubleshooting |
| [Contributing](CONTRIBUTING.md) | Development checks and GPU validation |
| [Roadmap](docs/ROADMAP.md) | Implemented features and next experiments |
| [Upstream provenance](THIRD_PARTY.md) | Source revisions, local modifications, license scope |
| [Original StyleGAN3 README](README.stylegan3.md) | Upstream training and architecture documentation |

For an automated run after setup:

```bash
.venv/bin/python -m experiments.run_face_boundary \
  --device cuda --grid-size 17 --frames 120 --rays 16 \
  --output outputs/first-run
```

Use a new output directory for each run. Notebooks are optional; install their
dependencies separately as described in the experiment guide.

## Repository layout

```text
experiments/         Geometry, projection, CLI, and browser UI
notebooks/           Optional research examples with cleared outputs
scripts/             Environment setup, smoke check, launcher, source bundling
tests/               CPU-only source-bundle regression tests
vendor/insightface/  Pinned upstream Git submodule
dnnlib/, torch_utils/, training/, legacy.py
                    NVIDIA runtime retained for checkpoint compatibility
```

The other original StyleGAN3 utilities remain available, but the playground's
pinned environment targets inference rather than reproducing upstream training.
`INSIGHTFACE_REPO` can point to another InsightFace checkout;
`FACE_MODELS_DIR` and `FACE_UI_OUTPUT_DIR` override local model and result paths.

## Research limits and licensing

Shared-W inversion cannot reconstruct every photo faithfully. The UI shows the
reference-to-reconstruction match so that this error can be examined before
motion. Trajectories are correlated explorations of a 2D slice, not independent
or uniform samples of the full latent region. Detector alignment, finite steps,
and rendering quantization can affect the measured boundary.

This distribution retains the [NVIDIA Source Code License](LICENSE.txt), including
its non-commercial research/evaluation limitation. InsightFace code has its own
license, and its pretrained models have separate usage terms. See
[THIRD_PARTY.md](THIRD_PARTY.md). Model weights, uploaded photos, generated runs,
and Python environments are excluded from version control.
