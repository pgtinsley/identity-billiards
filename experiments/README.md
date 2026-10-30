# Face identity billiards

**Start with the browser UI:** [Identity Playground setup and remote SSH access](UI.md).
For external collaborators, use the [source bundle and sharing guide](SHARING.md).
It provides starting-face alignment/projection, step size, identity threshold,
four reflection policies, cancellation, and video/data downloads. The CLI remains
available for automated runs; notebooks are optional research examples.

A fixed synthetic anchor defines an operational identity region:

`R(tau) = {u : exactly one face is detected and cosine(E(G(u)), E(G(0))) >= tau}`.

By default the fixed center is the generator's stored **learned W mean**,
`generator.mapping.w_avg`, repeated across synthesis layers. The face rendered at
that point is the identity reference. The center stays fixed while the ball moves.
Input-image alignment and projection are available in the browser UI; random
initialization remains an explicit opt-in with `--center seed --seed 23` rather
than the default. Seeded centers now use truncation **psi = 0.75**, with
untruncated W samples defining PCA step-size units. Mean and projected centers
are unchanged; see the [UI guide](UI.md) for inversion alternatives and caveats.

The score is similarity, so a collision happens when it falls **below** the
threshold. Every frame is compared with the fixed anchor. Comparing only with
the previous frame would allow identity to drift. This is a matcher's acceptance
region, not ground truth about personhood.

## Identity threshold survey

The default is now **cosine distance <= 0.55**, equivalently **cosine similarity
>= 0.45**, for our `buffalo_l` ResNet50@WebFace600K recognizer. This replaces the
initial exploratory similarity threshold of 0.60. Existing videos keep their
original thresholds; lowering the threshold expands the accepted region.

There is no universal ArcFace identity-match threshold: ArcFace describes a
training objective, while recognition weights, preprocessing, data and desired
false-match rate determine an operating point. A quick primary-source survey:

| Implementation / setting | Published rule | Equivalent rule in our score convention |
| --- | --- | --- |
| [InsightFace PersonConfig](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/person/config.py) | Face cosine similarity >= 0.45 | Similarity >= 0.45; distance <= 0.55 |
| [DeepFace Buffalo_L defaults](https://github.com/serengil/deepface/blob/master/deepface/config/threshold.py) | Cosine distance <= 0.55 | Similarity >= 0.45 |
| [DeepFace generic ArcFace defaults](https://github.com/serengil/deepface/blob/master/deepface/config/threshold.py) | Cosine distance <= 0.68 | Similarity >= 0.32; different model configuration |
| [InsightFace buffalo_l example](https://github.com/deepinsight/insightface/blob/master/examples/face_recognition/insightface_app.py) | Cosine similarity > 0.65; explicitly adjustable | A stricter example, not a universal recommendation |

InsightFace's [model guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/person_analysis.md)
identifies the Cheetah L face weights as the same files as `buffalo_l`, making
its PersonConfig default relevant here. The two model-matched defaults agree;
we adopt their operating point without calling it a calibrated consensus across
all ArcFace models. Their other quality gates and matching policies differ.
InsightFace's [verification evaluation](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval/verification.py)
selects thresholds on training folds or at a target false-accept rate rather
than declaring a fixed universal cutoff.

For our L2-normalized embeddings, `distance = 1 - similarity` and squared
Euclidean distance is `2 * (1 - similarity)`. Thus the chosen cutoff also means
normalized Euclidean distance <= sqrt(1.10), approximately 1.049. Keep raw
Euclidean distance, normalized Euclidean distance, cosine distance and cosine
similarity distinct; ArcFace's training margin is another different quantity.
This setting has no measured false-match guarantee on our generated faces or
on a different recognition model. Equality passes; detection validity remains
a separate requirement.

The CLI and both notebooks use the shared default. `--threshold` continues to
mean **minimum cosine similarity**. Alternatively use
`--max-cosine-distance 0.55`; the two flags are mutually exclusive. Saved metadata
records the effective similarity and distance thresholds and default sources.
Override either convention explicitly for controlled sweeps or other matchers.

## CLI and optional notebooks

Run from the repository root. The pinned `vendor/insightface` submodule is
imported directly, without modifying it or installing its unrelated GUI and
face reconstruction dependencies. Run `git submodule update --init --recursive`
after cloning. Portable source bundles use a sibling InsightFace directory
instead. Set `INSIGHTFACE_REPO` to use another checkout.

```bash
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r experiments/requirements.txt
.venv/bin/python -m experiments.run_face_boundary --device cuda --grid-size 17 --frames 120 --rays 16
```

For a Z-space video, enlarge the coordinate range and step size:

```bash
.venv/bin/python -m experiments.run_face_boundary --device cuda --space z --grid-size 21 --extent 10 --max-radius 16 --step-size 0.3 --frames 144 --rays 16
```

These W and Z videos illustrate different parameterizations; their coordinate
units, directions, and travel distances differ, so their diversity counts are not
a controlled sampler-performance comparison.

For the browser UI and remote SSH forwarding, use [the UI guide](UI.md).
A CPU-only CLI session can use
`--device cpu`; the installed ONNX GPU distribution also contains its CPU provider.
GPU commands need access to NVIDIA devices; a restricted agent sandbox may hide
them even on an allocated GPU node.

The environment pins PyTorch 2.5.1 (CUDA 12.4 on Linux) and ONNX Runtime GPU 1.20.2
(CUDA 12/cuDNN 9). The loader imports PyTorch and explicitly preloads its bundled
cuDNN components and NVRTC before creating ONNX sessions, so no system CUDA
library-path setup is needed. Do not install both `onnxruntime` and `onnxruntime-gpu`.
`requirements.txt` lists core dependencies. `requirements-ui-lock.txt` pins the
UI runtime; `requirements-lock.txt` retains the original notebook-enabled pilot.
The old upstream training environment remains untouched. This environment targets
inference, not reproducing the original training stack. Plain PyTorch reference
implementations of the StyleGAN2 bias/filter operations run on the GPU without
requiring nvcc; FP32 and constant synthesis noise reduce distracting variation.
FFmpeg comes from `imageio-ffmpeg`; a system FFmpeg install is unnecessary.

The runner downloads NVIDIA's **StyleGAN2 FFHQ 256x256** network and the
`buffalo_l` model pack into ignored `models/`. Only load trusted network pickles.
The generator is the StyleGAN2 architecture running in the compatible StyleGAN3
repository. Higher-resolution and StyleGAN3 architecture comparisons can follow.
The starter's reference-op configuration is tested with StyleGAN2.

## Optional notebooks

Install `experiments/requirements-notebooks.txt` to use these examples, then
register an ipykernel and launch Jupyter. They are not needed for the UI.

- `01_identity_billiards.ipynb`: generate an anchor, map a 2D slice, scan first-exit
  radii, reflect a trajectory, measure diversity, and render an annotated MP4.
- `02_geometry_and_controls.ipynb`: compare thresholds and anchors, probe an 8D
  slice, inspect the local embedding metric, and measure unrelated-seed scores.

Outputs include `reference.png`, `identity_map.png`, `billiard.mp4`,
`trajectory.npz`, and `metadata.json`. Archives contain coordinates, embeddings,
scores, detector confidence, the exact basis and anchor. Metadata includes model
hashes, repository revisions, package versions, settings, and collision reasons.
`anchor_seed` is null for a mean center; `trajectory_seed` controls only motion.
Mean-centered W archives omit `anchor_z` because `w_avg` has no stored source Z.
Use a new output directory for each run. Reconstruct an archived slice by loading
the same generator/settings and assigning its saved `anchor` and `basis`; rendering
`positions` reproduces its image path without fitting PCA again.

## What the geometry means

The default is shared **W**: one 512D style vector is repeated across all synthesis
layers. The default anchor is the stored `mapping.w_avg`, not the mean of the
sampled PCA-fitting batch. Changing the seed cannot move this center. In Z-space,
mean centering uses the Gaussian prior mean, zero; this is not a learned W mean,
and mapping zero does not generally reproduce `w_avg`. Explicit `center='seed'`
instead maps a seeded Gaussian Z sample for W or uses that Z vector directly.
The mean is a fixed latent point, not a guarantee of a circular similarity contour.
A PCA fit to 2048
mapped samples supplies axes scaled by their empirical standard deviations.
Two coordinates span the first two PCs. W perturbations can leave the mapping
network's image, especially far from the anchor. `basis_kind='random'` rotates
directions in standardized W; `space='z'` uses an orthonormal affine slice of Z.
Z coordinates have Euclidean units. Their Gaussian prior density decreases as
the norm changes along an affine slice. Record space, center, basis, truncation, and seed:
radii in different coordinate systems are not directly comparable.

The ball follows straight segments. Small substeps find a rejected segment,
bisection locates a collision, and central finite differences estimate its inward
similarity normal. Reflection uses `v_new = v - 2 * dot(v, n) * n` and consumes
the remaining frame distance. Detection failures are separate collision types;
an unavailable normal reverses velocity and records the fallback. A radius cap
is a computational guard, recorded separately from identity collisions. Eight
collisions in a frame cause a recorded hold. Inspect these diagnostics before
interpreting a trajectory statistically.

A finite step can miss thin rejected bands. Detector landmark changes and uint8
rendering also make this a nonsmooth measurement. Repeat with smaller steps,
more substeps, and different gradient epsilons to check stability. First-exit ray
scans report the first sampled exit and refine its bracket; they cannot establish
convexity, rule out narrow holes, or measure disconnected reentry regions.
Scan full score profiles or the heatmap to investigate those questions.

A continuous latent region does not supply a meaningful finite face count.
Report the number of frames and greedy representatives at an explicit embedding cosine
distance cutoff, together with the smallest and median pairwise similarity.
This count depends on sample budget, order, temporal correlation, and cutoff.
Two frames can both match the anchor yet fail to match each other. A billiard
path is an exploration tool and is not an independent or uniform region sample.
An 8D random-ray radius distribution is also not a full 512D volume estimate.

Near the anchor, normalized embeddings suggest
`1 - cosine ≈ 0.5 * delta.T @ (J.T @ J) @ delta`.
The local metric estimates `J.T @ J` by finite differences, revealing directions
that change identity quickly or slowly. This predicts a local ellipsoid; larger
threshold regions can bend, develop holes, and depend strongly on the matcher
and alignment. Check sensitivity to epsilon and compare standardized random W
slices before interpreting anisotropy as intrinsic identity geometry.

## Useful next experiments

1. Sweep `tau = 0.45, 0.6, 0.8` on identical slices and compare contours, exit
   radii, censoring, and diversity at a fixed frame budget. These are exploratory
   thresholds, not calibrated false-match operating points.
2. Repeat across seeds and random planes. Plot distributions with invalid/censored
   outcomes separate. Compare PCA and random standardized W, then Z.
3. Measure anchor-to-unrelated-seed scores as a synthetic negative control.
   Use held-out real identity data for a calibrated false-match threshold;
   generated seeds are not guaranteed distinct identities. Validate interesting
   trajectories with a second matcher instead of optimizing only one.
4. Compare free motion, specular reflection, and randomized inward redirects
   under matched distance and synthesis-call budgets. The included simulator
   provides mirror, random-inward, center-directed and tangent-biased policies,
   with explicit reversal fallbacks when the boundary normal is unreliable.
5. For visual variety, explore 8D dynamics and project the trajectory to 2D;
   a projection can show crossings that do not occur in the actual space.
   Keep a grid only for true 2D slices. W+ and layer-restricted changes are later
   controls for whether fine texture adds apparent diversity without identity change.
6. Compare anchor matching with requiring every retained pair to match. The first
   defines an anchor-centered set; pairwise matching defines a stricter cluster.
   Centroid matching and graph connectivity give still different groupings.

## Model choice and sources

Start with [`buffalo_l`](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md):
SCRFD-10GF face detection with five landmarks and ResNet50@WebFace600K recognition.
Detection finds/aligns the face; ArcFace embeddings define the identity boundary.
The current documentation also lists Raccoon packages, but does not establish
that they improve recognition accuracy here. `antelopev2` provides a larger
ResNet100@Glint360K recognizer with the same SCRFD-10GF detector and needs manual
model installation. Calling a detector state of the art requires a defined
benchmark; larger detection models need not help on centered FFHQ faces.

Primary references: [StyleGAN3 compatibility and model links](https://github.com/NVlabs/stylegan3),
[SCRFD models and detection benchmarks](https://github.com/deepinsight/insightface/tree/master/detection/scrfd),
[ArcFace paper](https://arxiv.org/abs/1801.07698), and
[ONNX CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
InsightFace pretrained weights are licensed for non-commercial research.
