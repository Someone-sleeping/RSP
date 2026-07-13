# Self-Interactive Error Verifier

## Task

This experiment converts iterative interactive correction into a label-free error-guided task:

1. frozen historical references propose alternative point predictions;
2. reference confidence, entropy, temporal agreement, and region consistency diagnose proposal reliability;
3. point and region candidates are accepted only by label-free rules;
4. unreliable temporal proposals fall back to region projection or the existing split/refiner output.

Ground-truth labels are read only by the final S3DIS metric computation. They are not used for semantic alignment, candidate construction, thresholding, reliability selection, or rollback.

## Method Components

- **Temporal correction hypotheses:** semantic predictions from frozen checkpoints are aligned to the current reference with classifier-center matching.
- **Point verifier:** a correction needs temporal vote support and a minimum mean target probability.
- **Region verifier:** region projection is accepted only when temporal region consensus supports the same target.
- **Reliable historical anchor:** temporal correction is enabled only if at least one historical checkpoint has no lower mean confidence and no higher mean entropy than the current reference. Otherwise the method falls back without temporal correction.
- **Joint correction:** the existing split/refiner prediction can be overridden by a high-confidence temporal hypothesis. This is permutation invariant and does not use class-index tie breaking.

The implementation and diagnostic sweeps are in `tools_eval_error_verifier.py`.

## Area 5 Results

Reference: epoch 1270. Historical checkpoints: 1170, 1180, 1190.

| Variant | mIoU | Delta vs frozen reference | Delta vs old best refiner |
|---|---:|---:|---:|
| Frozen reference | 43.8588 | - | - |
| Region projection | 44.2971 | +0.4383 | - |
| Split/projection no-op | 44.6369 | +0.7782 | - |
| Existing best split/refiner | 45.1113 | +1.2526 | - |
| Independent point/region verifier, threshold 0.64 | 45.1714 | +1.3127 | - |
| Joint refiner + temporal verifier, scale 0.8, threshold 0.64 | 45.5944 | +1.7356 | +0.4831 |

The independent verifier exceeds the requested +1 mIoU target without split proposals or a learned refiner. The joint result improves the previous best refiner by about +0.48 mIoU, which is close to the requested +0.5 target.

## Cross-Area Diagnostics

Fold-specific references were used for every held-out area.

| Area | Frozen reference | Raw temporal behavior | Label-free reliability decision | Selected fallback/result |
|---|---:|---|---|---:|
| Area 1, epoch 110 | 32.4662 | temporal candidates degrade | no reliable anchor | region projection, 32.7405 |
| Area 2, epoch 80 | 35.3385 | temporal candidates are unstable | no reliable anchor | region projection, 36.5774 |
| Area 2 + existing refiner | 35.3385 | temporal candidates rejected | keep split/refiner | 36.8375 |

These diagnostics do not support a claim that fixed temporal consensus always improves segmentation. They support a narrower claim: historical predictions provide useful error hypotheses when a label-free reliability test identifies a credible anchor, while explicit fallback prevents applying the mechanism when that condition is absent.

## Reproduction

Independent Area 5 verifier:

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python tools_eval_error_verifier.py \
  --reference_epochs 1170,1180,1190 \
  --thresholds 0.64 \
  --output_json ckpt/S3DIS/error_verifier/area5_independent.json
```

Joint Area 5 verifier:

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python tools_eval_error_verifier.py \
  --reference_epochs 1170,1180,1190 \
  --thresholds 0.64 \
  --refiner_checkpoint ckpt/S3DIS/refiner_projectloss02_e10/refiner_best_checkpoint.pth \
  --refiner_scale 0.8 \
  --output_json ckpt/S3DIS/error_verifier/area5_joint.json
```

## Current Limitations

- Area 5 is the only fold where the available historical sequence contains a reliable anchor under the current proxy.
- The reliability decision is computed over the held-out collection rather than independently per scene or region.
- The method adds inference cost because several frozen checkpoints are evaluated.
- The +0.48 joint gain still needs additional seeds or independently trained references before it should be described as stable.
