# Unsupervised Interactive Refiner

This branch treats the refiner as an unsupervised adaptation of Interactive4D for GrowSP semantic refinement.

## Design Boundary

- The query/click is not an object click and does not provide a class label.
- A query only marks a candidate error region: mixed, inconsistent, uncertain, or likely over-merged.
- The refiner must not receive ground-truth semantic labels during training or evaluation.
- Split targets are derived from frozen-teacher/prototype predictions inside geometrically and visually consistent sub-regions.
- Final predictions are accepted through no-label split/region projection and compared against a no-op projection baseline.

## Effective Components

- Frozen GrowSP teacher/backbone.
- Query-conditioned `ErrorQueryRefiner` trained separately from the frozen teacher.
- Geometry/color/feature/semantic split-region proposal in `lib/split_regions.py`.
- Region projection and split projection in `eval_S3DIS.py`.
- Dedicated refiner training in `train_refiner_S3DIS.py`.
- Best checkpoint selection by `delta_vs_no_op_projection`, not by raw baseline delta.

## No-Label-Leakage Rule

The only semantic sources allowed for refiner training are:

- frozen teacher logits,
- frozen teacher pseudo labels,
- teacher/prototype region aggregation,
- optional temporal teacher predictions from another checkpoint,
- geometry/color/feature consistency.

S3DIS ground-truth labels are used only by the held-out evaluation code to report metrics.

## Verified S3DIS Area 5 Result

Using the trained refiner in `ckpt/S3DIS/refiner_split80_project_e8/`:

| Setting | mIoU |
| --- | ---: |
| Raw frozen GrowSP baseline | 43.86 |
| No-op split/region projection | 44.64 |
| Trained refiner + projection | 45.10 |

Refiner-only gain over no-op projection: `+0.46 mIoU`, about `+1.03%` relative improvement.

## Train

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python train_refiner_S3DIS.py \
  --teacher_ckpt_dir /home/magic/magic/cm/repositories/GrowSP/ckpt/S3DIS/1baseline/ckpts \
  --teacher_epoch 1270 \
  --save_path ckpt/S3DIS/refiner_split80_project_e8_repro/
```

## Evaluate Trained Refiner

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python eval_S3DIS.py \
  --save_path ckpt/S3DIS/refiner_split80_project_e8/ \
  --eval_epoch best \
  --refine_enable \
  --refine_split_enable \
  --refine_residual_scale 0.9
```

## Evaluate No-Op Projection

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python eval_S3DIS.py \
  --save_path ckpt/S3DIS/refiner_split80_project_e8/ \
  --eval_epoch 0 \
  --refine_enable \
  --refine_split_enable \
  --refine_residual_scale 0.9
```
