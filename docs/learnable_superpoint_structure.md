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

The default structure configuration is deliberately conservative: at most
four candidate parents are considered per scene, verified child supervision is
disabled, and structural gradients update only the assignment module. The
learned partition affects the backbone indirectly through the superpoint
aggregation and pseudo-labels generated at the next clustering round.

## Area-5 Controlled Experiment

All runs below resume from `baseline/ckpts/model_1250_resume.pth`, train for 20
epochs, evaluate at epochs 1260 and 1270, and use the same S3DIS Area-5 split.
The baseline was rerun in the current environment rather than copied from an
older log because progressive KMeans introduces measurable run variation.

| Configuration | Epoch 1260 mIoU | Epoch 1270 mIoU | Delta vs. rerun baseline at 1270 |
| --- | ---: | ---: | ---: |
| GrowSP rerun baseline | 44.58 | 42.14 | - |
| Aggressive split plus direct supervision | 40.71 | stopped | - |
| Conservative split plus direct supervision | 41.86 | stopped | - |
| Conservative structure-only feedback | 43.39 | **44.39** | **+2.25** |

The effective run accepted 367 splits across 204 training scenes at the epoch
1261 reclustering boundary and changed 0.61% of training points. Independent
evaluation of the saved epoch-1270 checkpoints reproduced 44.39 mIoU, 56.74
mAcc, and 80.95 oAcc for structure-only feedback versus 42.14 mIoU, 54.15
mAcc, and 78.72 oAcc for the rerun baseline.

The comparison also exposes an important optimization constraint. Letting the
structure or child-consensus loss directly update the backbone conflicts with
GrowSP's primitive-clustering objective and degrades accuracy. The validated
variant therefore learns the partition on detached backbone features and
feeds accepted structure changes back through the next unsupervised
aggregation and pseudo-label cycle. This is a single controlled run; final
paper numbers should report repeated seeds.

## Verification Snapshot

The real-scene smoke test on `Area_1_WC_1` with the epoch-350 reference model
reported 48,223 training points, 12 accepted splits, 2.99% verified supervision
coverage, 30,465 trainable parameters, and about 356 MB peak allocated GPU
memory for the complete diagnostic process. The clustering-path check produced
48,199 valid points and exactly 48,199 point-region indices.

These values verify implementation behavior and interface compatibility. The
Area-5 experiment above reports segmentation accuracy separately.
