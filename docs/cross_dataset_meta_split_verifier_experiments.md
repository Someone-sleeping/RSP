# Cross-Dataset Split + Meta-Refiner + Verify Experiments

## Scope

This branch evaluates a common semantic-difference pipeline on GrowSP S3DIS,
ScanNet, SemanticKITTI, and LogoSP S3DIS checkpoints. Ground truth is accessed
only after every stage prediction has been produced and is used only for
Hungarian matching and metrics.

The experiment has two distinct scopes that must not be conflated:

- S3DIS uses the training-integrated learnable-superpoint checkpoint and its
  learned assignment module, followed by the trained Query Refiner, episodic
  gate/bias adaptation, and full-residual verification.
- ScanNet, SemanticKITTI, and LogoSP have no dataset-specific learned
  superpoint or Refiner checkpoint. They therefore use the fixed
  semantic-difference decomposition and a zero-residual conservative start;
  only label-free episodic gate/bias adaptation is allowed. These runs test
  checkpoint and backbone portability, not a fully retrained method.

The checkpoint-level runner is an experimental diagnostic. It does not change
the method's paper positioning: accepted structure and semantic corrections
are intended to alter superpoint aggregation and training supervision inside
the unsupervised training process, rather than being described as a generic
final-prediction post-processing method.

## Verification Correction

The previous verifier only checked changes introduced by Meta relative to the
Refiner. A stale Refiner could therefore damage a new backbone and bypass
rollback. The new full-residual verifier uses accepted decomposition as the
structural anchor and verifies every Refiner or Meta change. Unsupported
changes return to decomposition. Cross-backbone temporal override is disabled
because independently clustered semantic centers can otherwise produce large
but semantically misaligned updates.

## Quantitative Results

| Dataset / backbone | Evaluation scope | Base mIoU | Decomposition | Meta before verify | Final verified | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S3DIS / GrowSP | Area 5, 68 scenes | 44.3876 | 44.8699 | 41.9415 | **44.8905** | **+0.5029** |
| ScanNet / GrowSP | validation, 312 scenes | 3.5364 | 3.5772 | 3.5772 | **3.5772** | **+0.0408** |
| SemanticKITTI / GrowSP | sequence 08, uniform stride 40, 102 frames | 14.1501 | 13.7745 | 13.7745 | **13.7745** | **-0.3756** |
| S3DIS / LogoSP | Area 5, 68 scenes | 43.7671 | 43.7526 | 43.7526 | **43.7526** | **-0.0145** |

S3DIS is the only run with all learned components. Its trained Refiner does not
transfer cleanly to the learned-structure backbone by itself: Refiner mIoU is
41.9270. Meta raises this by 0.0145, and the verifier rolls unsupported
residuals back to reach 44.8905. The net gain over the 44.3876
training-integrated structure checkpoint is 0.5029 mIoU.

ScanNet is a valid negative-quality audit: the supplied epoch-930 checkpoint
has only 3.5364 mIoU and triggers no split or Meta episodes under the fixed
thresholds. SemanticKITTI is evaluated on 102 frames sampled uniformly from
sequence 08 without consulting labels. It triggers 35.72 semantic-difference
queries per frame, but no split or Meta episode is accepted; fixed region
projection lowers mIoU by 0.3756. LogoSP remains effectively neutral without a
LogoSP-trained Refiner. These results establish that the runner supports all
three datasets and a second backbone, but they do not support a claim of
cross-dataset learned-module generalization. Dataset-specific structure and
Refiner training is required for that claim.

## Efficiency

| Dataset / backbone | Scenes | Mean seconds / scene | Peak allocated memory |
| --- | ---: | ---: | ---: |
| S3DIS / GrowSP | 68 | 2.61 | 1803 MB |
| ScanNet / GrowSP | 312 | 1.42 | 679 MB |
| SemanticKITTI / GrowSP | 102 | 19.00 | 1062 MB |
| S3DIS / LogoSP | 68 | 4.04 | 1871 MB |

For S3DIS, the Query Refiner has 286,756 parameters, the learnable structure
module has 30,465, and the episodic adapter has 14. Total additional trainable
state is 317,235 parameters; verification is parameter-free. The reported
end-to-end time includes the current backbone, three historical references,
decomposition, Meta adaptation, and verification.

## Qualitative Results

The Area-5 qualitative comparison maps all predictions and region IDs back to
the original point cloud. Four room types are selected by point-accuracy gain:
WC, hallway, office, and storage. Each selected scene contains full-resolution
PLY files for:

1. original RGB point cloud;
2. ground truth;
3. frozen prediction;
4. initial superpoints;
5. split superpoints;
6. verified refinement;
7. difference diagnosis, where blue is changed, green is corrected, and red
   is harmed.

The combined figure is generated at
`ckpt/cross_dataset/s3dis_qualitative/qualitative_comparison.png`. Full point
clouds remain at original resolution; only the PNG renderer is deterministically
subsampled to avoid redundant overdraw.

![S3DIS qualitative comparison](assets/cross_dataset_s3dis_qualitative.png)

## Reproduction

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=12 \
  conda run -n cm_growsp python tools_eval_cross_dataset_pipeline.py \
  --dataset s3dis \
  --workers 4 \
  --output_json ckpt/cross_dataset/s3dis_area5_verified.json

env CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=12 \
  conda run -n cm_growsp python tools_eval_cross_dataset_pipeline.py \
  --dataset scannet \
  --workers 4 \
  --output_json ckpt/cross_dataset/scannet_full.json

env CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=12 \
  conda run -n cm_growsp python tools_eval_cross_dataset_pipeline.py \
  --dataset semantickitti \
  --scene_stride 40 \
  --workers 4 \
  --output_json ckpt/cross_dataset/semantickitti_seq08_stride40.json

env CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=12 \
  conda run -n cm_growsp python tools_eval_cross_dataset_pipeline.py \
  --dataset logosp_s3dis \
  --workers 4 \
  --output_json ckpt/cross_dataset/logosp_s3dis_area5_full.json
```
