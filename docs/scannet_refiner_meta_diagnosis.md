# ScanNet and Refiner/Meta Diagnosis

## ScanNet data audit

`tools_audit_scannet_data.py` performed a full read of every scene used by the
GrowSP loaders. The audit passed:

- 1,201 training scenes and 312 validation scenes, with no overlap or missing files;
- 223,890,677 points, all finite and carrying only labels `-1` or `0..19`;
- point and initial-superpoint arrays agree in length for every scene;
- 100 additional processed scenes are ScanNet test scenes beginning at
  `scene0707_00`; explicit split files prevent them from entering training or validation;
- initial superpoints contain 17 to 687 regions per scene (mean 132.32).

The local epoch-930 failure is therefore not explained by missing scenes,
invalid labels, split leakage, or mismatched superpoints. Its training log
shows a complete but collapsed run: validation mIoU stays near 3--5 throughout
training and ends at 3.54, while the superpoint oracle remains 58.81 mIoU at
epoch 921. The network loss is still 4.39 for 300 primitive classes at epoch
930, and primitive semantic coverage remains dominated by a few large classes.
The local run is not a valid reproduction of the paper's 25.4 +/- 2.3 result.

The evaluator blocks the known invalid checkpoint hash. ScanNet method results
must remain withdrawn until an official checkpoint is obtained or a fresh run
passes the native baseline check. `train_ScanNet.py` has also been repaired to
consume the current extended `get_pseudo` return contract before retraining.

## Refiner and Meta diagnosis

The negative LogoSP result was primarily a support mismatch. Refiner targets
cover only about 0.7% of training points, but the old inference path added the
learned point residual to the entire scene. The shared point MLP therefore
shifted predictions outside any Semantic Difference candidate region.

The corrected path masks every point/context/region residual by the candidate
support before projection. With the same epoch-20 backbone and the same old
Refiner checkpoint on all 68 Area 5 scenes:

| Path | mIoU | Delta from frozen |
| --- | ---: | ---: |
| Frozen LogoSP | 45.9827 | 0.0000 |
| Superpoint decomposition | 46.7690 | +0.7864 |
| Candidate-gated Refiner | 46.3475 | +0.3649 |
| Fixed Meta fusion | 46.7272 | +0.7445 |
| Online episodic Meta | 46.6315 | +0.6489 |
| Residual verifier | **46.7749** | **+0.7922** |

LogoSP supplies only one earlier checkpoint (epoch 10), and it is weaker and
less confident than epoch 20. A two-epoch experiment that explicitly allowed
that single checkpoint to create temporal training targets reached 45.9566 raw
and 46.7430 after verification, below the retained configuration. Training now
keeps an explicit two-vote default; automatic majority voting is available only
when requested. Runtime Meta can use the single reference as a proposal, but
the verifier remains responsible for rejecting unsupported changes.

All reported method decisions are label-free. Area 5 ground truth is used only
after prediction for Hungarian matching and diagnostics. Reproduction commands
should set `PYTHONNOUSERSITE=1` so the environment uses its compatible NumPy
and scikit-learn versions.
