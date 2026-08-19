# Training-Integrated Learnable Superpoint Structure

## Positioning

This module is not prediction post-processing. It is inserted into the
unsupervised segmentation training loop. Backbone point features, current
semantic predictions, geometry, and color jointly determine a dynamic local
partition. The verified partition changes both current training supervision
and the superpoint structure used by the next clustering round.

## Data Flow

1. The backbone extracts point features. The current primitive classifier is
   mapped to semantic centers to produce point semantic logits.
2. Low-purity or high-entropy parent superpoints are selected as semantic
   difference candidates. Stable regions are left unchanged.
3. Each candidate is represented by feature, normalized XYZ, RGB, and current
   semantic probability. A learnable affinity network performs iterative soft
   point-to-child assignment.
4. The structure verifier checks child size, distinct semantic consensus,
   confidence, confidence gain, and semantic separation. A rejected proposal
   preserves its parent region.
5. Accepted children provide dynamic region IDs and region-consensus semantic
   targets. Their weighted semantic loss and differentiable compactness losses
   are optimized together with the backbone objective.
6. If the Error-Query Refiner is enabled, it receives dynamic region IDs and
   verified targets when constructing queries and region supervision.
7. At the next GrowSP clustering round, the trained assignment module first
   rebuilds candidate regions. These regions are then used for superpoint
   feature aggregation, progressive merging, and point-wise pseudo-label
   generation.

The first clustering round keeps the initial geometric superpoints because no
historical classifier exists yet. Learned structure enters reclustering from
the second round onward.

## Objectives

The structure learner uses label-free feature compactness, geometric
compactness, semantic compactness, assignment entropy, and child balance.
Structural gradients update the assignment module without directly pulling the
backbone away from its primitive-clustering objective. Only verified child
consensus can be converted into optional semantic supervision. Ground-truth
labels are never read by candidate discovery, assignment, verification, or
training loss.

## Code Map

- `models/learnable_superpoint.py`: soft assignment, structural objectives,
  verifier, and verified region supervision.
- `train_S3DIS.py`: module construction, joint optimization, dynamic Refiner
  input, logging, and CLI configuration.
- `lib/utils.py`: learned structure injection before GrowSP aggregation and
  pseudo-label generation.
- `lib/my_utils.py`: checkpoint save/resume support.
- `tools_smoke_learnable_superpoint.py`: real-scene forward/backward and
  reclustering contract check.
- `tests/test_learnable_superpoint.py`: homogeneous, mixed-region, and gradient
  tests.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 conda run -n cm_growsp \
  python train_S3DIS.py \
  --save_path ckpt/S3DIS/learnable_superpoint_structure \
  --pseudo_label_path pseudo_label_s3dis/learnable_superpoint_structure \
  --learnable_sp_enable \
  --refine_enable
```

The structure-only ablation removes `--refine_enable`. The GrowSP baseline
removes both flags.

## Verification Snapshot

The real-scene smoke test on `Area_1_WC_1` with the epoch-350 reference model
reported 48,223 training points, 12 accepted splits, 2.99% verified supervision
coverage, 30,465 trainable parameters, and about 356 MB peak allocated GPU
memory for the complete diagnostic process. The clustering-path check produced
48,199 valid points and exactly 48,199 point-region indices.

These values verify implementation behavior and interface compatibility; they
are not segmentation accuracy results. Full Area-5 training and ablation are
still required to report mIoU improvement.
