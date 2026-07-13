# Meta Error Optimizer

## Motivation

The split/refiner and temporal verifier use globally fixed correction strengths. Different scenes, semantic slots, and error-query distributions need different update magnitudes. This experiment treats each scene as a label-free task and introduces a small support/query meta-optimization loop instead of another segmentation backbone.

## Method

The implementation has three stages:

1. **Meta initialization:** combine the split/refiner probability and temporal consensus with a conservative product-of-experts (PoE) initialization.
2. **Task adaptation:** split scene points deterministically into support and query subsets. The support subset adapts either one scalar or class-wise PoE weights. High-confidence temporal consensus provides correction targets, while the frozen refiner distribution provides preservation targets.
3. **Query verification and fast update:** accept the adapted parameters only when the independent query objective improves. High-confidence error queries can then apply a sparse temporal update. If no reliable historical anchor exists, the complete meta path is disabled and prediction falls back to the existing split/refiner.

The inner objective is

```text
L_inner = lambda_correct * CE(P_fused, y_temporal)
        + lambda_keep * KL(P_fused, P_refiner)
        + lambda_entropy * H(P_fused).
```

No S3DIS ground-truth label is used by alignment, support/query construction, adaptation, verification, or fallback. Ground truth is used only by the final metric computation.

Code:

- `lib/meta_optimizer.py`: differentiable PoE adaptation and query rollback.
- `tools_eval_error_verifier.py`: candidate construction, meta integration, evaluation, and diagnostics.

## Area 5 Result

Reference checkpoints: base 1270 and historical references 1170/1180/1190. Existing refiner checkpoint: `refiner_projectloss02_e10/refiner_best_checkpoint.pth`.

| Variant | mIoU | Delta vs frozen reference | Delta vs old split/refiner | Delta vs previous verifier |
|---|---:|---:|---:|---:|
| Frozen reference | 43.8588 | - | - | - |
| Existing split/refiner | 45.1113 | +1.2526 | - | - |
| Previous verifier best | 45.5944 | +1.7356 | +0.4831 | - |
| Fixed PoE, weight 0.10 | 45.6926 | +1.8338 | +0.5813 | +0.0982 |
| Scalar meta adaptation | 45.6831 | +1.8243 | +0.5718 | +0.0887 |
| Class-wise meta adaptation | 45.6911 | +1.8324 | +0.5798 | +0.0967 |
| Meta adaptation + fast error-query update | 45.8386 | +1.9798 | +0.7273 | +0.2442 |

The strict target of another +0.5 mIoU over the previous verifier was not reached. The achieved incremental gain is +0.2442 mIoU. Relative to the original split/refiner, the meta-optimized path adds +0.7273 mIoU.

## Ablation Findings

- A scalar inner loop with weak preservation over-adapts: the average temporal weight grows to 0.28--0.49 and mIoU falls to 45.16--45.54.
- Increasing preservation keeps the adapted weight near 0.10 and recovers 45.68.
- Class-wise adaptation is stable but does not materially beat the fixed initialization.
- The main additional gain comes from coupling the conservative meta-adapted PoE state with sparse, high-confidence error-query updates at threshold 0.80.
- A class-index-based voting tie produced an invalid optimistic result in an earlier diagnostic and was removed. All reported meta results are permutation invariant.

## Cross-Area Safety Check

On fold-specific Area 2, the historical references do not pass the label-free reliable-anchor test. The final selector therefore keeps the existing split/refiner result:

| Variant | Area 2 mIoU |
|---|---:|
| Frozen reference | 35.3385 |
| Existing split/refiner | 36.8375 |
| Selected meta strategy with fallback | 36.8375 |

Raw meta candidates reach similar values but are not selected because the temporal source is unreliable under the no-label confidence/entropy test.

## Reproduction

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python tools_eval_error_verifier.py \
  --reference_epochs 1170,1180,1190 \
  --thresholds 0.80 \
  --selection_threshold 0.80 \
  --blend_weights 0.08 \
  --refiner_checkpoint ckpt/S3DIS/refiner_projectloss02_e10/refiner_best_checkpoint.pth \
  --refiner_scale 1.0 \
  --meta_optimize \
  --meta_classwise \
  --meta_initial_weight 0.08 \
  --meta_inner_steps 5 \
  --meta_inner_lr 0.1 \
  --meta_correction_weight 1.0 \
  --meta_keep_weight 5.0 \
  --output_json ckpt/S3DIS/meta_optimizer/area5_final.json
```

## Limitations

- The current method meta-optimizes a very small fusion parameter set; it is a bilevel meta-optimizer, not a full MAML-trained neural selector.
- The meta initialization is selected from the current experimental sequence and needs independent-fold or training-scene meta-training for a stronger generalization claim.
- Area 1 has no compatible trained refiner checkpoint for the same joint diagnostic.
- The additional inference cost still includes multiple frozen historical references.
