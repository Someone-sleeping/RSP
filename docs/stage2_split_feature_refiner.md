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
5. Accept a split only when it improves child compactness over the unsplit
   parent, maps the children to distinct and semantically compatible GrowSP
   primitives, preserves child separation after refinement, and keeps the
   residual magnitude bounded; otherwise restore the parent region and features.
6. Fit global primitives on the merged parent regions, then assign verified
   children to those primitives as local pseudo-label overrides. This preserves
   GrowSP's global clustering state while allowing local supervision to change.
7. Preserve the original GrowSP primitive gradient for the backbone. The
   detached local branch optimizes the Feature Refiner with primitive-improvement,
   split-semantic, feature-structure, and residual losses. Verified structures
   still update the backbone indirectly through the next pseudo-label round.

The Refiner output therefore affects its detached local objective immediately
and can affect the backbone only through verified primitive overrides in a
later pseudo-label round.

## Paired validation

Both runs resume `baseline/ckpts/model_1070_resume.pth`, use seed 2022 and
single-process loading, and stop at epoch 1080. The control reaches 42.93 Area-5
mIoU; the RNG-isolated conservative Stage-2 path reaches 43.26 (+0.33). An
independent reload of the saved epoch-1080 checkpoint reproduces 43.26.

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
  --stage2_split_start_ratio 0.75 \
  --max_epoch 500 800 \
  --max_iter 10000 30000 \
  --stage2_feature_refiner_lr 1e-4 \
  --stage2_feature_residual_scale 0.1 \
  --stage2_split_lambda 0.2 \
  --stage2_feature_lambda 0.1 \
  --stage2_residual_lambda 0.01 \
  --stage2_refiner_primitive_lambda 0.5 \
  --stage2_min_split_conf 0.35 \
  --stage2_min_structure_gain 0.01 \
  --stage2_min_child_separation 0.05 \
  --stage2_min_primitive_gain 0.005 \
  --stage2_min_primitive_margin 0.01 \
  --stage2_override_min_gain 0.05 \
  --stage2_override_min_margin 0.02 \
  --stage2_max_residual_norm 1.0 \
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
