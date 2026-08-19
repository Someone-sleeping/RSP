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

The learnable-structure branch starts from
`exp/decomposition-meta-refiner-verifier`, preserving the earlier unified
pipeline as a reproducible parent version.

The two interactive-refiner commits had the same subject but different Git
histories. The remote branch was left untouched; the local variant was pushed
under `archive/` instead of being force-updated.
