# Stage-2 Split-Aware Feature Refiner

Branch: `feat/stage2-query-context-feature-refiner`

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
4. Pool each proposed child into a query token. The Candidate-based Refiner
   reads detached non-candidate scene context and writes a feature residual only
   to candidate points. Candidate features retain their backbone gradient.
5. The decomposition gate accepts a split from feature compactness, sibling
   separation, and Top-K semantic-group primitive support. A separate residual
   gate accepts the refined feature only when it preserves the verified
   structure and keeps the residual bounded. A rejected residual therefore does
   not discard an otherwise reliable decomposition.
6. Train the backbone and Refiner jointly on accepted decomposition cores using
   child semantic, semantic-group primitive, sibling-structure, and residual
   objectives. Query discovery and both verifier decisions are stop-gradient.
7. Fit global primitives on the merged parent regions, then assign verified
   children to those primitives as local pseudo-label overrides. This preserves
   GrowSP's global clustering state while allowing local supervision to change.

The Refiner is thus a training-integrated feature adapter after the backbone and
before region aggregation. It is not applied to final predictions at inference.

## Archived conservative baseline

Both runs resume `baseline/ckpts/model_1070_resume.pth`, use seed 2022 and
single-process loading, and stop at epoch 1080. The control reaches 42.93 Area-5
mIoU; the RNG-isolated conservative Stage-2 path reaches 43.26 (+0.33). An
independent reload of the saved epoch-1080 checkpoint reproduces 43.26.

That detached implementation is preserved at
`archive/stage2-conservative-feature-refiner-45p24`. Results for the joint
feature-adaptation branch must be reported separately.

## Joint feature-adaptation validation

The calibrated branch resumes the same epoch-1070 checkpoint and runs
continuously to epoch 1080. It accepts 114 decompositions (0.355% point
coverage), applies 53 local primitive overrides (0.064%), and reaches 43.28
Area-5 mIoU. Reloading the saved checkpoint independently reproduces 43.28.
The matched GrowSP control is 42.93 and the detached conservative branch is
43.26. Increasing `stage2_backbone_gradient_scale` from 0.1 to 0.2 reduces the
result to 42.79, so 0.1 remains the conservative default.

These ten-epoch diagnostics validate the training data flow and short-range
stability. They do not replace the full Stage-2 result or multi-seed reporting.

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
  --stage2_backbone_gradient_scale 0.1 \
  --stage2_split_lambda 0.2 \
  --stage2_feature_lambda 0.1 \
  --stage2_residual_lambda 0.01 \
  --stage2_refiner_primitive_lambda 0.05 \
  --stage2_min_split_conf 0.35 \
  --stage2_min_structure_gain 0.01 \
  --stage2_min_child_separation 0.05 \
  --stage2_min_primitive_gain 0.005 \
  --stage2_min_primitive_margin 0.01 \
  --stage2_primitive_top_k 3 \
  --stage2_primitive_support_tolerance 0.01 \
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
