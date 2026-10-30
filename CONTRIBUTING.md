# Contributing

Clone with `--recurse-submodules` and follow the [quick start](README.md#quick-start).
Keep project-specific changes in `experiments/`, `scripts/`, `notebooks/`, and
`tests/` where possible. Retain upstream copyright and license notices.

## Checks without models or a GPU

From the repository root, with Python 3.11 or newer:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q experiments scripts tests
for script in scripts/*.sh; do bash -n "$script"; done
```

The bundle tests exercise source allowlisting, symlink rejection, archive paths,
and content hashes using temporary fixtures. GitHub Actions runs these checks
without downloading weights or installing the CUDA environment.

## Changes involving model execution

Run `bash scripts/check_identity_playground.sh --smoke` in a GPU allocation.
For UI or trajectory changes, also generate a short mean-centered trajectory and
exercise the affected photo/seed workflow. Inspect minimum similarity, collision
reasons, fallbacks, and saved metadata. For projection changes, compare against
known-latent controls from `experiments.generate_inversion_targets`.
The smoke check verifies execution; it does not establish inversion quality or
trajectory correctness. Document the actual checks and their limitations in the PR.

## Experiment hygiene

- Keep model weights, uploads, local credentials, logs, and generated results
  outside commits. Use `models/` and `outputs/` for local data.
- Clear notebook outputs and execution counts before committing.
- Record coordinate space, anchor, threshold, truncation, seeds, and model hashes
  when reporting experimental results. Compare samplers using matched budgets.
- Update documentation when changing defaults or the output format.
- Update a dependency pin deliberately and rerun model checks; do not track a
  moving upstream branch as a reproducibility substitute.

Use issues for reproducible bugs and focused experiment proposals. Include the
revision, relevant settings, and a small description of expected versus observed
behavior. Share face images or result archives only when you intend to make them
public, and remove session tokens from reports.
