# Sharing Identity Playground

Each collaborator uses their own cluster account, extracted copy, Python
environment and GPU allocation. The browser runs on their laptop; the frontend
relays an SSH tunnel to the UI in the GPU allocation. Generation, photo
projection, matching and video encoding all run on that GPU node.

For collaborators with GitHub access, cloning the repository is the usual path:

```bash
git clone --recurse-submodules https://github.com/pgtinsley/identity-billiards.git
cd identity-billiards
bash scripts/setup_identity_playground.sh
```

Continue with the GPU check and launcher below, from the repository root.
The remaining archive instructions provide an alternative for collaborators
receiving a portable source snapshot. Archives retain the historical sibling
`stylegan3/` and `insightface/` layout; Git clones use `vendor/insightface/`.

## Prepare the handoff

From the maintainer's Identity Billiards repository root:

```bash
python3 scripts/package_identity_playground.py
```

Send `dist/identity-playground.tar.gz` and its `.sha256` file using your usual
file sharing method. The archive captures the current experiment code, including
uncommitted files, the required StyleGAN runtime, the sibling InsightFace Python
source, upstream notices, pinned UI dependencies and these instructions. It
excludes model weights, photos, generated outputs, virtual environments, caches,
notebooks and Git history. `MANIFEST.json` records the project and InsightFace
checkout revisions and SHA256 hashes for the included source; the
[provenance notes](../THIRD_PARTY.md) record the NVIDIA base revision.
`SHA256SUMS` checks the extracted contents.
To build another snapshot, use `--output dist/identity-playground-v2.tar.gz`.

## Install once, under your own account

On a cluster frontend with internet access, put both files in your own project
directory on storage available to the GPU nodes. Allow several GB for Python,
CUDA dependencies and models; the source archive itself is small.

```bash
sha256sum -c identity-playground.tar.gz.sha256
tar -xzf identity-playground.tar.gz
cd identity-playground
sha256sum -c SHA256SUMS
cd stylegan3
bash scripts/setup_identity_playground.sh
```

Setup requires `uv` on PATH and creates Python 3.11 plus `.venv` at this location.
Load the site's uv module or install uv using your usual software setup if it is
missing. The lock file targets Linux x86_64 and PyTorch CUDA 12.4 / ONNX Runtime
GPU 1.20.2. Jupyter, Node.js, nvcc and a system FFmpeg are unnecessary. Do not copy
someone else's `.venv`: its interpreter paths can point to their home directory.
If you move an installation, move the old `.venv` aside and rerun setup.

Setup synchronizes the environment to the pinned UI requirements. Use this
environment only for the playground; install notebook tools in a separate one.

### Model files

The default model directory is `stylegan3/models/`. Models download on first use.
To reuse an existing complete model cache on these same machines, set the
following in **each terminal where you check or start the UI**:

```bash
export FACE_MODELS_DIR=/absolute/path/to/readable/models
```

The directory must have this layout:

```text
models/
  stylegan2-ffhq-256x256.pkl
  vgg16.pt
  insightface/models/buffalo_l/
    ... model pack's .onnx files ...
```

Ask the project owner for the cache path and read access using the normal
cluster sharing mechanism. A complete cache can be read-only. If any models
are missing, use your own writable cache. Avoid concurrent first downloads into
the same cache. For GPU nodes without internet, populate your cache in advance
on a network-enabled host with:

```bash
bash scripts/check_identity_playground.sh --device cpu --smoke
```

This loads all three model families and can be slow on CPU. Use an appropriate
CPU job if the site's frontend policy restricts compute work. Model weights are
not redistributed in the source archive. Upstream notices remain in
`stylegan3/LICENSE.txt` and `insightface/README.md` in the extracted bundle;
the latter also describes the pretrained model usage terms.

## Start inside your GPU allocation

Use your usual cluster workflow to request an interactive NVIDIA GPU allocation.
On the **allocated GPU node**, change to your extracted `identity-playground/stylegan3`
directory. Set `FACE_MODELS_DIR` here too if you use a shared cache. Then run:

```bash
bash scripts/check_identity_playground.sh --smoke
bash scripts/start_identity_playground.sh --port 7860
```

The check verifies imports, GPU visibility, ONNX CUDA availability and output
permissions, then loads the real models, renders and embeds a face, runs one
projection step and encodes a short MP4. A successful import-only check does not
verify GPU model execution. The `--smoke` check does not run a full trajectory
or measure projection quality.

The server prints its GPU hostname, output directory and a localhost URL with a
session token. Keep this terminal and the allocation alive. The launcher binds
to the cluster network for frontend tunneling and enables token authentication.
An example URL is `http://127.0.0.1:7860/?token=...`.

Use a different server port if another process occupies 7860. To put this
server's results in a separate directory, add
`--output-dir /absolute/path/to/my-playground-results`. The equivalent environment
variable is `FACE_UI_OUTPUT_DIR`. The launcher creates new files with private
permissions; existing directory permissions and site ACLs still apply.

## Open the UI on your laptop

In a separate **laptop terminal**, replace the three uppercase placeholders:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 7860:GPU_HOST:7860 \
  YOUR_USERNAME@FRONTEND_HOST
```

`GPU_HOST` is the hostname printed by your server. `FRONTEND_HOST` is the
externally reachable frontend you normally SSH to. Use your usual VPN/bastion
configuration if required. Leave the quiet tunnel terminal running. Open the
**exact token-bearing localhost URL** printed by your server in your laptop
browser. The token authorizes the browser with an HttpOnly session cookie.

If local port 7860 is occupied, use `-L 7862:GPU_HOST:7860` and replace the URL's
port with 7862, keeping its token. If you changed the server port, change the
rightmost tunnel port to match. A new allocation can have a different GPU
hostname, and a server restart normally creates a new token.

If your cluster allows SSH directly to compute nodes through a jump host, the
loopback-only alternative in [the UI guide](UI.md) is also available. The launcher
accepts the UI flags, so `--host 127.0.0.1` selects that mode.

## First experiment and saved results

1. Wait for the learned W mean face to load; the first load also fits the PCA
   slice. If models are uncached, downloads add time.
2. Generate a short trajectory with the default settings, then download the
   MP4, trajectory archive and metadata.
3. Try a seeded starting face and each reflection policy. For a photo, choose
   Upload, select a single face, then click **Align & project photo**. Inspect
   the reference/reconstruction match before generating a trajectory.

The [UI guide](UI.md) explains controls, alignment, projection limitations and
data formats. Outputs persist under `outputs/ui/<job-id>/` by default. Uploads
produce aligned face images, reconstruction images and latent vectors; run
archives contain embeddings and provenance. Share only the intended results.
Prepared centers and interactive history live in memory and must be prepared
again after a restart; saved files remain on disk.

Each server is one shared session with a single GPU worker and a four-job queue.
Anyone holding that server's token can view its jobs/files and cancel its work.
Use separate installations/allocations for independent collaborator sessions.
Two servers sharing an installation should use separate output directories.

Stop with Ctrl-C, close the laptop tunnel and release the allocation using your
usual cluster workflow. Cancellation takes effect at the next compute checkpoint;
it can take time during a model load or first download.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing/broken Python interpreter | Run setup at the final install path; move a broken `.venv` aside first. |
| CUDA check fails | Run inside the GPU allocation; confirm GPU visibility and rerun setup if packages differ. |
| ONNX falls back to CPU | Keep only `onnxruntime-gpu` in this environment; rerun setup and the smoke check. |
| Download fails | Use a network-enabled host to prefill the cache, or select a complete readable cache. |
| InsightFace source missing | Keep the sibling bundle layout, or set `INSIGHTFACE_REPO` to your InsightFace checkout root. |
| Browser says unauthorized | Reopen the current server's URL with its full token. |
| Browser cannot connect | Check allocation/server are alive, tunnel hostname/ports match, and VPN is connected if required. |
| Address already in use | Choose another server/local port and update the tunnel and browser URL accordingly. |
| GPU queue full | Wait or cancel a queued job; use your own server for an independent session. |

For a useful issue report, include the source bundle SHA256, failing check or
job error, GPU type, settings and relevant metadata. Keep tokens and uploaded
face data out of the report unless you intend to share them.
