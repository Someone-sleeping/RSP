# Stage-2 Split-Aware Feature Refiner

Branch: `exp/stage2-split-feature-refiner`

This experiment inserts structure correction into GrowSP Stage 2. It is not a
post-processing step: verified superpoint structure and refined point features
are used to regenerate region features, grow clusters, primitive assignments,
and the pseudo-label supervision that updates the backbone.

## Training flow

1. Resume the trained Stage-1 backbone and primitive classifier.
2. Run the original GrowSP merge to obtain the scheduled coarse superpoints for
   the current Stage-2 round.
3. Detect mixed candidates inside those grown superpoints from the current
   semantic state, point features, coordinates, and colors.
4. Produce candidate-only feature residuals with bidirectional Query-Scene
   attention and a point-wise prior branch.
5. Accept a split only when child compactness/separation is not degraded and
   the residual magnitude remains bounded; otherwise restore the parent region
   and its original features.
6. Use the verified split regions directly for primitive clustering and pseudo
   labels; no second GrowSP merge is performed after decomposition.
7. Jointly optimize the backbone and
   feature Refiner with primitive, split-semantic, feature-structure, and
   residual regularization losses.

The Refiner output therefore affects both the current training loss and the
next clustering round through updated backbone features.

Each clustering round also stores the grown, pre-decomposition region map.
Training batches reload this map so candidate discovery uses the same coarse
regions that produced the current pseudo labels. Final validation reports the
jointly optimized backbone prediction; it does not append the Refiner as an
inference-time post-processing module.

## Start from the completed Stage 1

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. PYTHONNOUSERSITE=1 \
conda run --no-capture-output -n cm_growsp \
python train_S3DIS.py \
  --resume ckpt/S3DIS/baseline/ckpts/model_470_resume.pth \
  --start_stage2_from_resume \
  --stage2_split_refine_enable \
  --stage2_split_start_ratio 0.0 \
  --max_epoch 500 800 \
  --max_iter 10000 30000 \
  --stage2_feature_refiner_lr 1e-4 \
  --stage2_feature_residual_scale 0.1 \
  --stage2_split_lambda 0.2 \
  --stage2_feature_lambda 0.1 \
  --stage2_residual_lambda 0.01 \
  --save_path ckpt/S3DIS/stage2_split_feature_refiner/ \
  --pseudo_label_path pseudo_label_s3dis/stage2_split_feature_refiner/
```

Use `--start_stage2_from_resume` only for the Stage-1 checkpoint. To continue
an interrupted run from this branch's `model_*_resume.pth`, omit that flag so
the stored `start_grow_epoch` and Refiner optimizer state are retained.

## Verification

The end-to-end smoke test resumes epoch 470 and checks clustering, one joint
training step, checkpoint save/reload, and Area-5 evaluation. All 204 training
scenes produced pre-decomposition grown-region maps. With the smoke-only target
of 20 grown regions per scene, the method found 2,387 candidates and accepted
669 splits before primitive clustering. Its 39.46 Area-5 mIoU is not a reported
accuracy result because only one Stage-2 batch is optimized.
