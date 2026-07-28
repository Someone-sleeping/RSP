# Decomposition + Meta-based Refiner + Result Verifier

## Scope

This implementation turns the previously scattered diagnostic branches into a
single reusable inference pipeline. It uses a frozen unsupervised backbone, a
trained query-conditioned Refiner, and aligned historical checkpoints. Ground
truth is read only after the pipeline has produced all predictions and is used
only to compute evaluation metrics.

The Meta-based Refiner is scene-wise episodic meta-optimization: support
regions adapt lightweight residual gates and class bias, while held-out query
regions decide whether to accept the adapted state. It is not cross-scene MAML
and does not claim a learned outer-loop initialization.

## Data Flow

### Decomposition

The stage receives frozen point features, logits, coordinates, colors, and
initial superpoints. Mixed or inconsistent regions produce semantic-difference
queries. Suspected under-segmented superpoints are locally split, and the split
targets plus region consensus are projected back to point logits.

Output: decomposition-only logits, split targets, query indices, refine/keep
masks, and split statistics.

### Meta-based Refiner

The frozen Refiner exposes point and context residual components. Region parity
constructs disjoint support and query sets. Reliable temporal and structural
agreement provides label-free correction targets, while agreement with the
original Refiner provides preservation targets. The support set adapts branch
gates and a conservative class bias. The query objective either accepts the
adapted state or rolls the whole scene back to the original Refiner.

Output: Meta-adapted probability, scene acceptance state, adapted scales, bias
norm, support/query correction counts, and query gain.

### Result Verifier

The verifier first checks every Meta-induced label change. A candidate is
accepted only when it has enough historical votes, sufficient temporal
confidence, and support from temporal, region, or decomposition predictions.
Unsupported Meta changes roll back to the original Refiner result. Finally, a
high-confidence temporal candidate may override the verified result.

The point-wise decision code is:

| Code | Decision |
|---:|---|
| 0 | Keep the current result |
| 1 | Accept the Meta-Refiner change |
| 2 | Roll back to the original Refiner |
| 3 | Apply the verified temporal override |

## Implementation

- `lib/semantic_difference_pipeline.py`: structured stage outputs and unified
  label-free pipeline.
- `tools_eval_semantic_difference_pipeline.py`: S3DIS model loading,
  stage-wise evaluation, statistics, and JSON output.
- `tests/test_semantic_difference_pipeline.py`: deterministic verifier decision
  test covering accept, keep, rollback, and temporal override.

## Area 5 Results

The fixed configuration uses base epoch 1270, historical epochs
1170/1180/1190, Refiner scale 1.0, Meta target confidence 0.80, Meta candidate
verification confidence 0.64, and final result verification confidence 0.80.
These values reproduce the previously recorded conservative Meta-Refiner run;
no new Area 5 threshold sweep was performed.

| Stage | mIoU | Delta vs base | Delta vs previous |
|---|---:|---:|---:|
| Frozen backbone | 43.8588 | - | - |
| Decomposition | 44.6369 | +0.7782 | +0.7782 |
| Query Refiner | 45.1113 | +1.2526 | +0.4744 |
| Meta-based Refiner | 45.1585 | +1.2997 | +0.0471 |
| Meta result verification | 45.1642 | +1.3054 | +0.0058 |
| Final Result Verifier | **45.5618** | **+1.7030** | **+0.3975** |

The unified implementation is numerically identical to the corresponding six
outputs in `area5_bias_conservative.json`; all mIoU differences are exactly
zero at the stored floating-point precision.

## Decision Statistics

Across 68 Area 5 scenes:

- 56.28 semantic-difference queries are generated per scene;
- 7.46% of voxel points enter refinement masks;
- 4.65% receive explicit split targets;
- scene-level Meta adaptation is accepted in 17.65% of scenes;
- Meta changes 0.161% of voxel predictions;
- 0.106% are accepted and 0.055% are rolled back;
- temporal result verification overrides 2.53% of voxel predictions;
- the final output changes 9.58% of voxel predictions relative to the base.

The final gain must therefore not be attributed entirely to Meta adaptation.
Meta contributes a small positive adjustment, candidate rollback prevents a
subset of unsupported updates, and the largest final increment comes from the
verified temporal override.

## Reproduction

```bash
env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n cm_growsp \
  python tools_eval_semantic_difference_pipeline.py \
  --workers 4 \
  --output_json \
    ckpt/S3DIS/semantic_difference_pipeline/area5_full.json
```

The reported full run was executed on GPU0. A one-scene smoke test is available
by adding `--max_scenes 1 --workers 0`.
