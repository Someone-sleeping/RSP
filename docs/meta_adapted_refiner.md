# Episodically Meta-Adapted Error-Query Refiner

## Scope

This experiment moves adaptation from post-Refiner PoE fusion into the Refiner residual path. It implements a label-free, scene-wise support/query inner loop. It does not yet learn the initialization across training scenes, so it should be described as an episodic Meta-Refiner prototype rather than full MAML.

The frozen GrowSP backbone, error-guided decomposition, trained Error-Query Refiner, and historical checkpoints remain unchanged. S3DIS labels are used only after prediction for metrics.

## Data Flow

1. Error-guided decomposition identifies suspicious regions and proposes split targets.
2. The frozen Refiner exposes point, context, and optional region residual components.
3. Superpoint parity creates spatially disjoint support and query subsets.
4. Reliable temporal and structural agreement supplies label-free correction targets. Agreement with the original Refiner supplies preservation targets.
5. The support subset adapts lightweight point/context gates and an optional zero-initialized class-bias adapter.
6. The query objective accepts the adapted state or rolls back to the original Refiner.
7. The Error Verifier checks Meta-Refiner changes against temporal votes, confidence, and region/no-op predictions. Rejected changes return to the original Refiner prediction.

Split points are excluded from semantic adapter optimization because their hard targets belong to the decomposition stage.

## Implementation

- `models/query_refiner.py`: optionally returns point, context, region, and total residuals without changing the default output.
- `lib/meta_refiner.py`: differentiable region/split projection, region-level support/query construction, branch-gate adaptation, class-bias adaptation, and query rollback.
- `tools_eval_error_verifier.py`: Meta-Refiner arguments, candidate construction, verifier variants, statistics, and JSON output.

## Area 5 Results

All experiments use base epoch 1270, historical epochs 1170/1180/1190, the existing `refiner_projectloss02_e10` checkpoint, and GPU 1.

| Variant | mIoU | Change from matching baseline |
|---|---:|---:|
| Frozen GrowSP | 43.8588 | - |
| Split + original Refiner | 45.1113 | - |
| Scalar branch gates | 45.1112 | -0.0001 |
| Class-wise branch gates | 45.1113 | +0.0000 |
| Conservative class-bias Meta-Refiner | 45.1585 | +0.0471 |
| Original Refiner + temporal override | 45.5326 | - |
| Meta-Refiner + candidate verifier + temporal override | 45.5618 | +0.0292 |
| Existing fixed PoE + override | **45.8402** | - |
| Meta-Refiner + existing PoE + joint verifier | 45.8391 | -0.0011 |

The stronger bias adapter improves its label-free query objective but lowers raw mIoU to 44.7344. Candidate verification recovers this path to 45.5591, showing that rollback is necessary but cannot repair a systematically misaligned adaptation target.

## Reproduction

```bash
env CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n cm_growsp \
  python tools_eval_error_verifier.py \
  --reference_epochs 1170,1180,1190 \
  --thresholds 0.80 \
  --selection_threshold 0.80 \
  --blend_weights 0.08 \
  --refiner_checkpoint ckpt/S3DIS/refiner_projectloss02_e10/refiner_best_checkpoint.pth \
  --refiner_scale 1.0 \
  --meta_initial_weight 0.08 \
  --meta_refiner \
  --meta_refiner_adapt_bias \
  --meta_refiner_inner_steps 5 \
  --meta_refiner_inner_lr 0.1 \
  --meta_refiner_correction_weight 1.0 \
  --meta_refiner_keep_weight 5.0 \
  --meta_refiner_scale_reg 0.1 \
  --meta_refiner_bias_reg 1.0 \
  --meta_refiner_verify_threshold 0.64 \
  --output_json ckpt/S3DIS/meta_refiner/area5_bias_conservative.json
```

## Conclusion

Adapting the Refiner is technically viable and contributes up to +0.0471 mIoU to the raw split/Refiner path in this study. It does not improve the strongest existing Area 5 result: the best joint result is effectively tied but 0.0011 mIoU lower. The current bottleneck is the alignment of historical-consensus proxy targets with semantic mIoU, not adaptation capacity.

A stronger meta-learning claim requires a cross-scene outer loop on non-test S3DIS areas, a learned adapter initialization, and evaluation on an untouched fold. The current result does not justify replacing the fixed PoE path in the reported best method.
