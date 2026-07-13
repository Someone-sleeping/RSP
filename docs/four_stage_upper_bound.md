# Four-Stage Error-Guided Correction Upper Bound

## Scope and Protocol

This study explores the upper bound of four label-free correction modules for frozen GrowSP. ScanNet is intentionally excluded. All proposal construction, reliability scoring, support/query adaptation, rollback, and fallback decisions use only frozen predictions, features, superpoints, and confidence statistics. S3DIS ground truth is used only for final metrics and explicitly named oracle diagnostics.

The four modules are:

1. hierarchical superpoint split and merge;
2. error-conditioned residual refinement and region-wise scale selection;
3. local or region-level correction verification;
4. cross-scene episodic meta-optimization.

Area 5 scale sweeps are upper-bound diagnostics because the scale was inspected on the evaluation fold. The strongest unchanged Area 5 result remains the previously fixed `meta_adapt_override_t80` result, 45.8386 mIoU. The 45.8979 result should not be presented as an unbiased final score.

## Actual Results

| Fold | Frozen | Split only | Old split/refiner | Best explored actual | Gain vs frozen |
|---|---:|---:|---:|---:|---:|
| Area 1 diagnostic | 32.2083 | 33.0012 | 33.0012 | 33.0253 candidate vote | +0.8170 |
| Area 2 diagnostic | 35.3385 | 36.7322 | 36.8375 | 37.0599 region-risk rollback | +1.7214 |
| Area 5 | 43.8588 | 44.6369 | 45.1113 | 45.8386 fixed meta/verifier | +1.9798 |
| Area 5 tuned upper bound | 43.8588 | 44.6369 | 45.1113 | 45.8979 scale 1.20 + verifier | +2.0392 |

Area 2 gains are useful cross-area diagnostics, not a full fold-retrained benchmark. A formal cross-fold claim still requires retraining the GrowSP backbone and refiner for every held-out fold.

## Module Ablation

| Module | Area 2 best | Area 2 region oracle | Area 5 best explored | Area 5 region oracle |
|---|---:|---:|---:|---:|
| Split | 36.7322 | 38.2820 | 44.6369 | 45.9473 |
| Refiner | 37.0286 | 39.3103 | 45.8979 | 48.8868 |
| Verifier | 37.0599 | 38.8502 | 45.8124 | 47.9866 |
| Meta | 36.8671 | 38.0835 | 45.8402 | 47.2205 |

Region oracle selects one complete candidate per superpoint-like region. Invalid points are grouped conservatively per scene rather than treated as singleton regions. Point oracle values are higher still: the refiner reaches 39.8302 on Area 2 and 49.3381 on Area 5. Both diagnostics use labels and are potential estimates only.

### 1. Hierarchical Split

- The original conservative split is robust: +0.7929 on Area 1, +1.3937 on Area 2, and +0.7782 on Area 5.
- Multiple split proposals do not improve the result. Area 2 changes from 36.7322 to 36.7229 and Area 5 from 44.6369 to 44.5749.
- Merging small children using temporal support also degrades slightly. The current historical predictions are not calibrated enough to decide split topology.
- The remaining oracle gap is real, but another hand-designed split rule is unlikely to capture it. Future split work should predict region risk jointly with the verifier.

### 2. Error-Conditioned Refiner

- A global residual scale of 0.60 gives 37.0286 on Area 2, exceeding the previous scale-0.70 result.
- Label-free region-wise scale selection reaches 36.9842 on the strict Area 2 pool and was competitive but not uniformly superior.
- Directly adding residual updates outside split regions is harmful on both folds. Split and projection are therefore necessary structural constraints.
- On Area 5, a stronger residual proposal combined with conservative PoE verification reaches the tuned upper bound of 45.8979. The raw refiner itself deteriorates at strong scales, so the gain comes from selecting a useful subset of residual changes.

The strong-scale effect is stable across seeds, although the best scale drifts:

| Run | Fixed scale-meta | Best scale-meta | Gain | Best scale |
|---|---:|---:|---:|---:|
| Main | 45.8402 | 45.8979 | +0.0578 | 1.20 |
| Seed 2022 | 45.6181 | 45.6825 | +0.0644 | 1.25 |
| Seed 2023 | 45.5869 | 45.6527 | +0.0658 | 1.25 |
| Seed 2024 | 45.7450 | 45.7881 | +0.0431 | 1.00 |

Mean gain is +0.0578 mIoU with population standard deviation 0.0090. This supports the mechanism, not a universal fixed scale.

### 3. Region Verifier

- Region-risk rollback is the strongest Area 2 actual strategy at 37.0599, +0.2224 over the old split/refiner.
- The same fixed rule does not beat the fixed Area 5 meta/verifier result; its best Area 5 value is 45.8124.
- The strict region oracle gaps are large: +2.2504 above the best actual Area 2 result for the refiner pool and +2.9889 above the tuned Area 5 result.

The verifier is therefore the main remaining bottleneck. The next model should operate on region statistics and jointly predict `accept`, `rollback`, and residual scale. It should be calibrated on non-test scenes rather than tuned per point or per held-out fold.

### 4. Episodic Meta-Optimization

`tools_meta_outer_s3dis.py` implements a label-free first-order MAML-style loop. Areas 1/2/3/4/6 provide 102 scene episodes; superpoint parity creates spatial support/query subsets. The shared temporal PoE initialization is adapted on support and updated from query gradients. Area 5 labels are read only after prediction.

The learned temporal weight moves from 0.0800 to 0.1207. Area 5 results are 45.6452 before task adaptation, 45.6503 after adaptation, and 45.6793 after sparse override. This is below the existing 45.8386 method. The outer loop is operational, but its label-free proxy is misaligned with mIoU: mean Area 5 query gain is negative even though 58.8% of tasks pass the local acceptance test.

The next meta variant should optimize a region-level risk/calibration model, not only one global PoE scalar. Non-Area5 episodes should meta-learn how confidence, temporal agreement, projection consistency, region size, and residual magnitude map to correction reliability.

## Interpretation

The experiments do not support four independent additive gains. Split, refiner, verifier, and meta optimization act on the same small set of corrected points, so their gains overlap. The defensible claim is a staged error-guided correction framework with four complementary roles:

1. split exposes internally conflicting regions;
2. the refiner proposes semantic corrections;
3. the verifier prevents harmful updates;
4. meta-optimization adapts the verifier across scene episodes.

The clearest path toward the potential upper bound is a learned region-level gate with episodic calibration. Split variants should remain conservative, while candidate diversity should come mainly from residual scales and temporal anchors.

## Reproduction

Area 5 four-stage diagnostic:

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python tools_eval_error_verifier.py \
  --base_epoch 1270 \
  --reference_epochs 1170,1180,1190 \
  --test_area Area_5 \
  --thresholds 0.80 \
  --selection_threshold 0.80 \
  --refiner_checkpoint ckpt/S3DIS/refiner_projectloss02_e10/refiner_best_checkpoint.pth \
  --refiner_scale 1.0 \
  --refiner_scales 1.0,1.2,1.25 \
  --meta_optimize \
  --meta_classwise \
  --meta_initial_weight 0.08 \
  --meta_keep_weight 5.0 \
  --four_stage_diagnostic \
  --output_json ckpt/S3DIS/four_stage_upper_bound/area5_strict_oracle.json
```

Cross-scene meta outer loop:

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n cm_growsp python tools_meta_outer_s3dis.py \
  --meta_train_areas Area_1,Area_2,Area_3,Area_4,Area_6 \
  --test_area Area_5 \
  --reference_epochs 1170,1180,1190 \
  --refiner_checkpoint ckpt/S3DIS/refiner_projectloss02_e10/refiner_best_checkpoint.pth \
  --refiner_scale 0.7 \
  --output_checkpoint ckpt/S3DIS/four_stage_upper_bound/meta_outer_s3dis.pth \
  --output_json ckpt/S3DIS/four_stage_upper_bound/meta_outer_area5.json
```

Summarize strict ablations:

```bash
conda run -n cm_growsp python tools_summarize_four_stage_ablation.py \
  ckpt/S3DIS/four_stage_upper_bound/area2_strict_oracle.json \
  ckpt/S3DIS/four_stage_upper_bound/area5_strict_oracle.json
```
