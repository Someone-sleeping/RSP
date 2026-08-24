# Experiment Branches

Each major method version is preserved as an independent branch. New work
should branch from the closest completed version and should not overwrite an
older experiment branch.

| Branch | Base commit | Scope |
| --- | --- | --- |
| `main` | `d226e1e` | Upstream GrowSP baseline snapshot |
| `unsup-refiner-interactive4d` | `4b16540` | Pre-existing remote interactive-refiner version |
| `archive/unsup-refiner-interactive4d-local` | `2a2d463` | Preserved local interactive-refiner version with divergent history |
| `exp/self-interactive-error-verifier` | `95a9660` | Label-free error verifier |
| `exp/meta-error-optimizer` | `77c1411` | Meta error optimization prototype |
| `exp/four-stage-upper-bound` | `9980ed7` | Four-stage upper-bound analysis and tensor flow |
| `exp/meta-adapted-refiner` | `fb28b01` | Episodic Meta-Refiner adaptation |
| `exp/efficiency-qualitative` | `1af178c` | Efficiency and original-point qualitative diagnostics |
| `exp/decomposition-meta-refiner-verifier` | `16e976d` | Unified decomposition, Meta-Refiner, and verifier pipeline |
| `exp/learnable-superpoint-structure` | `5a01ee1` | Training-integrated learnable superpoint assignment |
| `exp/cross-dataset-meta-split-verifier` | `b8d644d` | Cross-dataset/backbone evaluation, full-residual verification, and original-point qualitative results |
| `exp/four-stage-dataset-specific-ckpts` | `7068609` | Correct 45.84 four-stage protocol with dataset-specific Refiner checkpoints and hash binding |
| `exp/scannet-data-logosp-meta-calibration` | `0f90005` | ScanNet data audit and LogoSP candidate-gating calibration |
| `exp/stage3-split-candidate-refiner-verifier` | `0f90005` | Training-integrated Stage 3 with superpoint split, Candidate-based Refiner, and Conservative Verifier |

The learnable-structure branch starts from
`exp/decomposition-meta-refiner-verifier`, preserving the earlier unified
pipeline as a reproducible parent version.

The cross-dataset branch starts from the validated learnable-structure branch
and preserves the S3DIS-only implementation before adding dataset adapters and
the stricter cross-backbone verification protocol.

The dataset-specific branch corrects the earlier cross-dataset diagnostic: it
uses the 43.8588 S3DIS backbone associated with the 45.84 result and refuses
silent zero-residual or mismatched-checkpoint evaluation.

The two interactive-refiner commits had the same subject but different Git
histories. The remote branch was left untouched; the local variant was pushed
under `archive/` instead of being force-updated.

The Stage-3 branch starts from the calibrated cross-dataset history, removes
Meta adaptation from the primary method, and preserves the old frozen
evaluation path only for numerical regression. Its primary path starts after
GrowSP growing and feeds verified region structure and targets back into joint
unsupervised training.
