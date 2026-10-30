# Project status and next experiments

## Implemented

- Shared-W and Z-space CLI experiments, fixed-reference matching, boundary maps,
  first-exit scans, trajectory metadata, and MP4 export.
- Browser UI with learned-mean, seeded, and uploaded-photo starts; FFHQ-style
  alignment; shared-W perceptual projection; cancellation and session history.
- Four reflection policies, collision diagnostics, and verified accepted frames.
- Default minimum cosine similarity 0.45 and seed truncation psi 0.75.
- Synthetic shared-W inversion targets with exact saved latents and deterministic
  pixel checks, plus portable collaborator source bundles and cluster instructions.

## Next comparisons

1. Compare shared-W projection against regularized W+ on the same 256px generator,
   fixed noise, images, and comparable compute budgets. Report LPIPS, identity
   similarity, runtime, visual reconstruction, and local motion naturalness.
2. Sweep thresholds on identical slices and repeat across anchors and planes.
   Separate detector failures, radius censoring, and identity exits.
3. Compare reflection policies using matched travel and synthesis-call budgets.
4. Validate interesting trajectories with a second recognizer and examine
   sensitivity to step size and finite-difference settings.

W+ inversion, pretrained encoder integration, per-image generator tuning, and
source-to-target planning are not implemented. The
[UI guide](../experiments/UI.md#alternatives-to-the-current-inversion) records
the compatibility constraints and proposed inversion comparisons.
