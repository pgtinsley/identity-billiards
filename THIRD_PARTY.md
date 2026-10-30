# Source provenance and third-party notices

## NVIDIA StyleGAN3

The repository retains the history and runtime of
[NVlabs/stylegan3](https://github.com/NVlabs/stylegan3), based on commit
`c233a919a6faee6e36a316ddd4eddababad1adf9`.
The original README is preserved as [README.stylegan3.md](README.stylegan3.md),
and the complete original license remains in [LICENSE.txt](LICENSE.txt).
Existing copyright notices are retained in source files.

The Identity Billiards additions are `experiments/`, `notebooks/`, `scripts/`,
`tests/`, and the project documentation and repository configuration. The
inference loader selects PyTorch reference implementations for StyleGAN2
operations, avoiding an nvcc build. It loads the StyleGAN2 FFHQ 256px checkpoint;
this is distinct from using the StyleGAN3 generator architecture.

This distribution and its project additions are provided under the retained
NVIDIA Source Code License. It limits use to non-commercial research/evaluation.
No broader license is asserted over upstream code or pretrained models.

## InsightFace

[deepinsight/insightface](https://github.com/deepinsight/insightface) is an
unmodified Git submodule at `vendor/insightface`, pinned to
`c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52`.
The playground imports its Python source directly instead of installing the
unrelated GUI and reconstruction packages. The optional `INSIGHTFACE_REPO`
environment variable selects an alternate checkout.

The upstream [license statement](https://github.com/deepinsight/insightface/tree/c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52#license) describes
the code as MIT-licensed and pretrained models as restricted to non-commercial
research. Those source and model terms remain separate from NVIDIA's license.
The portable source-bundle script includes upstream notices with selected
InsightFace source files.

## Model weights and other dependencies

The generator checkpoint, `buffalo_l` ONNX models, and NVIDIA VGG16 perceptual
model are downloaded separately and are not committed or included in source
bundles. Their upstream terms continue to apply. Runtime metadata records model
hashes and package versions; pinned UI dependencies live in
`experiments/requirements-ui-lock.txt`. Python packages and any third-party
components embedded in upstream source retain their respective notices.
