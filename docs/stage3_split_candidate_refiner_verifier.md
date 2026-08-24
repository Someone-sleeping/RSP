# Stage-3 Split + Candidate-based Refiner + Conservative Verifier

## Method Positioning

本方法不是对最终预测进行后处理，也不是冻结 GrowSP 后再运行一次独立细化。
完整训练过程由三个连续阶段组成：

1. **Stage 1: Initial Superpoint Training.** 使用初始固定超点聚合点特征，执行特征聚类、伪标签生成和网络训练。
2. **Stage 2: Progressive Superpoint Growing.** 按 GrowSP 的原始策略逐步合并超点，并周期性更新聚类中心与伪标签。
3. **Stage 3: Split, Refine, and Verify.** 在 growing 完成后，基于当前点级特征、预测状态与区域结构拆分可能被错误合并的超点；随后仅对候选区域生成语义残差，并由保守验证器控制候选更新。验证后的结构和训练目标继续参与特征聚合、伪标签更新及 backbone 训练。

因此，Stage 3 的输入不是已经完成的最终分割图，而是无监督训练过程中的当前点特征、当前语义状态、已 grow 的超点以及当前伪标签。其输出也不是一次性的可视化结果，而是更新后的超点结构和训练监督。

## Components

### Superpoint Split

模块在每个已 grow 超点内部统计预测纯度和熵，仅选择包含稳定竞争模式的候选区域。候选超点由坐标、颜色、backbone 特征和语义概率共同构成多模态描述，并通过确定性的二簇分解产生两个子区域。只有当两个子区域均具有足够点数、可靠类别共识且共识类别不同，拆分才被接受。

输出包括候选点、查询锚点、子区域目标及重新编号后的动态超点。动态超点在后续聚类轮次中替换原父超点，直接改变区域特征聚合和伪标签生成的基本单元。

### Candidate-based Refiner

Refiner 以 backbone 点特征、空间坐标和拆分产生的候选查询为输入。Query-to-Scene Attention 允许查询读取全场景上下文，Scene-to-Query Attention 将与查询相关的上下文写回点表示。全场景仅作为信息源，最终语义残差通过 candidate mask 限制在候选点内，非候选区域保持当前状态。

### Conservative Verifier

Verifier 检查候选残差是否增强了拆分子区域的结构共识，同时限制异常大的更新。获得结构支持的更新被接受；降低目标支持度或残差幅度异常的更新被回滚到当前预测。结构验证通过的拆分目标用于约束 backbone，只有 verifier 接受的候选才用于监督 Refiner；回滚候选仅保留残差回零约束。

## Joint Optimization

Stage 3 保留 GrowSP 的 primitive pseudo-label 损失，并增加两类无标签监督：拆分目标对 backbone 语义特征的约束，以及候选目标对 Refiner 残差的约束。两部分在同一个训练步中反向传播，backbone 与 Candidate-based Refiner 联合更新。随着 backbone 特征变化，下一轮聚类重新计算拆分结构和伪标签，形成训练内闭环。

## Compatibility

`CandidateBasedRefiner` 保留 `ErrorQueryRefiner` 类别名，参数名与历史 checkpoint 完全一致。旧的冻结模型评测链用于复现既有消融结果；它不再作为本文主方法的训练定义。Meta adaptation 也保留在历史实验分支中，但不属于这条精简主路径。

## Implementation

- `lib/stage3_pipeline.py`: 超点拆分、候选残差、保守验证及联合训练损失。
- `models/query_refiner.py`: `CandidateBasedRefiner` 与候选区域残差门控。
- `lib/utils.py`: 拆分结构参与超点特征聚合和重新聚类的接口。
- `train_S3DIS.py`: GrowSP Stage 1/2 之后的显式 Stage 3 训练入口。
- `eval_S3DIS.py`: 与训练一致的拆分、候选修正和接受/回滚推理路径。
- `tests/test_stage3_pipeline.py`: 拆分、联合梯度、接受/回滚和 checkpoint 兼容测试。

启动方式：

```bash
env CUDA_VISIBLE_DEVICES=1 conda run -n cm_growsp python train_S3DIS.py \
  --stage3_enable \
  --stage3_epochs 20 \
  --save_path ckpt/S3DIS/stage3_candidate_refiner_verifier/
```

从完成 Stage 2 的 resume checkpoint 启动时，前两个循环自动跳过，训练直接进入 Stage 3。

## Validation Status

基于 `ckpt/S3DIS/baseline/ckpts/model_1270_resume.pth` 的端到端 smoke 已在 GPU 1 完成。Stage 3 首轮重新聚类覆盖 204 个训练场景，发现 11,961 个候选超点并接受 2,486 次拆分，动态结构覆盖 3.97% 的点；随后一个真实 batch 完成 backbone 与 Refiner 联合反向传播并保存 epoch 1271 checkpoint。backbone-only mIoU 为 44.00，一致的 Stage 3 verified prediction 为 43.99，二者均仅用于确认训练与推理链可运行，不作为正式精度结果。

历史 Refiner checkpoint 可由重命名后的类直接加载，state-dict 键完全一致。当前父分支的候选门控评测得到 45.4917 mIoU；仓库中 45.8402 的存档 JSON 来自更早的非同构候选门控协议，不能用来声称当前 Stage 3 已达到 45.84。新主方法的正式结果需要完成 Stage 3 全量训练后单独报告。
