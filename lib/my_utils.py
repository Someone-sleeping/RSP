import matplotlib.patches as mpatches
import torch
import os
import re
import math
import json
import wandb
import shutil
import logging
import datetime
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from torch.optim.lr_scheduler import LambdaLR
from torchdiffeq import odeint_adjoint as odeint

try:
    from lib.vis import VisualizationThreadPool
except ImportError:
    VisualizationThreadPool = None

import seaborn as sns
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
try:
    import umap
except Exception:
    umap = None
from models.fpn import Res16FPN18
from models.res16unet import Res16UNet14


def compute_multiscale_geometry(coords, normals, region, scales=[0.1, 1.0]):
    """
    coords: [N, 3], float tensor
    normals: [N, 3], float tensor
    region: [N], long tensor, 每个点的超点索引
    return: [num_regions, num_features]
    """
    import numpy as np
    from sklearn.neighbors import KDTree

    coords_np = coords.cpu().numpy()
    normals_np = normals.cpu().numpy()
    region_np = region.cpu().numpy()

    kdtree = KDTree(coords_np)
    region_feats = []
    unique_regions = np.unique(region_np)

    for r in unique_regions:
        mask = (region_np == r)
        center = coords_np[mask].mean(0)
        cur_feats = []
        for radius in scales:
            idx = kdtree.query_radius([center], r=radius)[0]
            if len(idx) < 3:
                cur_feats.extend([0, 0, 0])
                continue
            pts = coords_np[idx]
            ns = normals_np[idx]
            z_range = pts[:, 2].max() - pts[:, 2].min()
            z_std = np.std(pts[:, 2])
            norm_var = np.var(ns, axis=0).sum()
            cur_feats.extend([z_range, z_std, norm_var])
        region_feats.append(cur_feats)

    return torch.tensor(region_feats).float().cuda()


def masked_kmeans_consensus(region_feats, n_segments, n_rounds=10, mask_prob=0.3, seed=0):
    """
    随机mask特征 + 多次KMeans聚类 + 一致性矩阵融合
    """
    N, D = region_feats.shape
    all_labels = []
    consensus_matrix = torch.zeros(N, N, device=region_feats.device)

    for r in range(n_rounds):
        # 1. 随机mask部分维度
        mask = (torch.rand(D, device=region_feats.device) > mask_prob).float()
        masked_feats = region_feats * mask

        # 2. KMeans聚类（CPU上跑 sklearn）
        kmeans = KMeans(n_clusters=n_segments, n_init=5, random_state=seed + r)
        labels = kmeans.fit_predict(masked_feats.cpu().numpy())
        labels = torch.from_numpy(labels).to(region_feats.device).long()
        all_labels.append(labels)

        # 3. 更新一致性矩阵（用矩阵乘法替代双循环）
        one_hot = F.one_hot(labels, num_classes=n_segments).float()  # [N, K]
        consensus_matrix += one_hot @ one_hot.T   # [N, N]

    # 归一化
    consensus_matrix /= n_rounds

    # 4. 在一致性矩阵上再聚类
    kmeans_final = KMeans(n_clusters=n_segments, n_init=10, random_state=seed)
    final_labels = kmeans_final.fit_predict(consensus_matrix.cpu().numpy())
    return torch.from_numpy(final_labels).long()


def compute_color_consistency(rgb_feats, mode="variance", normalize=True):
    """
    计算一个超点的颜色一致性 (color consistency)

    Args:
        rgb_feats (torch.Tensor): [N, 3] 的颜色特征
        mode (str): "variance" 用方差衡量一致性, "dominant" 用dominant color, "both"结合
        normalize (bool): 是否归一化到 [0,1] (如果原始是 [-1,1] 或 [0,255] 都会处理)

    Returns:
        consistency_score (float): 一致性分数 (0~1，越高表示越一致)
        stats (dict): 包含均值/方差/众数等统计信息
    """
    if rgb_feats.size(0) == 0:
        return 0.0, {}

    feats = rgb_feats.clone().detach().cpu().numpy()

    # 归一化 ([-1,1] -> [0,1])
    if normalize:
        feats = (feats - feats.min()) / (feats.max() - feats.min() + 1e-8)

    mean_color = feats.mean(axis=0)
    var_color = feats.var(axis=0).mean()   # 各通道方差的平均
    std_color = np.sqrt(var_color)

    # 基于方差的一致性 (小方差 -> 高一致性)
    consistency_var = np.exp(-std_color * 10)  # 指数映射到 0~1

    # 基于 dominant color 的一致性
    # 这里用简单的聚类方式: 如果某个点离均值最近，就当 dominant
    dists = np.linalg.norm(feats - mean_color, axis=1)
    dominant_ratio = (dists < dists.mean()).sum() / len(dists)

    if mode == "variance":
        score = consistency_var
    elif mode == "dominant":
        score = dominant_ratio
    else:  # both
        score = 0.5 * (consistency_var + dominant_ratio)

    stats = {
        "mean_color": mean_color,
        "std_color": std_color,
        "consistency_var": float(consistency_var),
        "dominant_ratio": float(dominant_ratio),
    }
    return dists < dists.mean(), stats


def augment_pointcloud(coords, features=None, rotation_range=math.pi, jitter_scale=0, drop_prob=0.0):
    """
    简单的点云增强：随机绕 z 轴旋转 + 小幅 jitter + 随机 dropout 点
    coords: Tensor (N,3)
    features: Tensor (N, C) or None
    """
    device = coords.device
    N = coords.shape[0]

    # 1. 分离 batch_idx 和坐标
    batch_idx = coords[:, 0:1]   # (N,1)
    xyz = coords[:, 1:4]         # (N,3)

    # 2. 随机绕 z 轴旋转
    theta = (torch.randn(1, device=device) * rotation_range).item()
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = torch.tensor([[cos_t, -sin_t, 0.],
                      [sin_t,  cos_t, 0.],
                      [0.,     0.,    1.]], device=device, dtype=coords.dtype)

    xyz2 = torch.matmul(xyz, R.t())  # (N,3)

    # 3. jitter
    xyz2 = xyz2 + torch.randn_like(xyz2) * jitter_scale

    # 拼回 coords2
    coords2 = torch.cat([batch_idx, xyz2], dim=1)  # (N,4)

    colors = features[:, :3]             # (N,3)
    norm_xyz = features[:, 3:6]          # (N,3)

    # 对 norm_coords 做相同的旋转 & jitter
    norm_xyz2 = torch.matmul(norm_xyz, R.t())
    norm_xyz2 = norm_xyz2 + torch.randn_like(norm_xyz2) * jitter_scale

    # 拼回新的 features
    features2 = torch.cat([colors, norm_xyz2], dim=-1)  # (N,6)

    return coords2, features2


def iic_loss_from_probs(p, p_aug, eps=1e-8):
    """
    # 计算 IIC 损失的核心函数
    # 输入：p (B, K), p_aug (B, K) 代表两个视图的 softmax 概率
    # 返回：标量 iic_loss（max mutual information => 我们最小化 -I）

    p : (B, K) 视图1的 softmax 概率
    p_aug : (B, K) 视图2的 softmax 概率
    计算 joint matrix: P_ij = (1/B) * sum_n p_n_i * p'_n_j
    然后计算 mutual information I(P) = sum_ij P_ij * log( P_ij / (P_i * P_j) )
    我们返回 -I(P) 即为要最小化的损失
    """
    assert p.dim() == 2 and p_aug.dim() == 2 and p.shape == p_aug.shape
    B, K = p.shape

    # 计算联合计数（batch 内求和）
    # joint unnormalized: K x K
    # torch.einsum 比较直观： sum_n p[n,i] * p_aug[n,j]
    joint = torch.einsum('nk,nj->kj', p, p_aug)  # (K, K)
    # 对称化（可选）：让 joint 矩阵更稳定
    joint = (joint + joint.t()) / 2.0

    # 归一化为概率
    joint = joint / (joint.sum() + eps)

    # 边缘分布
    pi = joint.sum(dim=1, keepdim=True)  # (K,1)
    pj = joint.sum(dim=0, keepdim=True)  # (1,K)

    # 互信息 (数值稳定)
    # I = sum_ij joint_ij * (log joint_ij - log(pi_i) - log(pj_j))
    log_joint = torch.log(joint + eps)
    log_pi = torch.log(pi + eps)
    log_pj = torch.log(pj + eps)

    mi = (joint * (log_joint - log_pi - log_pj)).sum()
    iic_loss = -mi  # 我们要最小化 -I
    return iic_loss


def backup_selected(args):
    files = ["train_S3DIS.py", "eval_S3DIS.py", "models", "lib", "datasets"]

    save_dir = Path(args.save_path) / "backup"
    save_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        src = Path(f)
        dst = save_dir / src.name
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)  # 避免旧文件残留
            shutil.copytree(src, dst)

    args_path = save_dir / "args.json"
    if hasattr(args, "__dict__"):
        args = vars(args)
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4, ensure_ascii=False)


def load_resume_checkpoint(
    args, model, optimizer, scheduler, logger, refiner=None,
    learnable_sp=None, learnable_sp_optimizer=None,
):
    """处理模型断点续传，返回当前 epoch 和阶段控制参数"""
    if not args.resume:
        return 0, 0, False  # start_epoch, start_grow_epoch, is_Growing

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if refiner is not None:
        if 'refiner_state_dict' in checkpoint:
            try:
                refiner.load_state_dict(checkpoint['refiner_state_dict'])
            except RuntimeError as exc:
                logger.info(f"Refiner state was not restored: {exc}")
        else:
            logger.info("Resume checkpoint has no refiner_state_dict; initializing refiner from scratch.")
    if learnable_sp is not None:
        if 'learnable_sp_state_dict' in checkpoint:
            try:
                learnable_sp.load_state_dict(checkpoint['learnable_sp_state_dict'])
            except RuntimeError as exc:
                logger.info(f"Learnable superpoint state was not restored: {exc}")
        else:
            logger.info("Resume checkpoint has no learnable_sp_state_dict; initializing it from scratch.")
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as exc:
            logger.info(f"Optimizer state was not restored: {exc}")
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except ValueError as exc:
            logger.info(f"Scheduler state was not restored: {exc}")
    if learnable_sp_optimizer is not None and 'learnable_sp_optimizer_state_dict' in checkpoint:
        try:
            learnable_sp_optimizer.load_state_dict(checkpoint['learnable_sp_optimizer_state_dict'])
        except ValueError as exc:
            logger.info(f"Learnable superpoint optimizer state was not restored: {exc}")

    start_epoch = checkpoint.get('epoch', 0)
    is_Growing = checkpoint.get('is_Growing', False)
    start_grow_epoch = checkpoint.get('start_grow_epoch', 0)

    logger.info(f"Checkpoint resumed from {args.resume}, epoch {start_epoch}, "
                f"is_Growing: {is_Growing}, start_grow_epoch: {start_grow_epoch}")

    return start_epoch, start_grow_epoch, is_Growing


def save_checkpoints(
    args, epoch, model, optimizer, scheduler, classifier, is_Growing, start_grow_epoch, logger,
    refiner=None, refiner_optimizer=None, learnable_sp=None, learnable_sp_optimizer=None,
):
    """一键保存所有 Checkpoints，并自动兼容处理不存在的路径"""

    # 自动检查并创建 ckpts 文件夹
    ckpt_dir = os.path.join(args.save_path, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. 保存用于 Resume 的完整状态
    state = {
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'epoch': epoch,
        'is_Growing': is_Growing,
        'start_grow_epoch': start_grow_epoch,
        'training_stage': getattr(args, 'training_stage', 'growsp'),
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if refiner is not None:
        state['refiner_state_dict'] = refiner.state_dict()
    if refiner_optimizer is not None:
        state['refiner_optimizer_state_dict'] = refiner_optimizer.state_dict()
    if learnable_sp is not None:
        state['learnable_sp_state_dict'] = learnable_sp.state_dict()
    if learnable_sp_optimizer is not None:
        state['learnable_sp_optimizer_state_dict'] = learnable_sp_optimizer.state_dict()

    # 使用上一步定义好的 ckpt_dir，代码更干净
    resume_path = os.path.join(ckpt_dir, f'model_{epoch}_resume.pth')
    torch.save(state, resume_path)
    logger.info(f"Checkpoint saved to {resume_path}")

    # 2. 保存纯净版权重
    model_path = os.path.join(ckpt_dir, f'model_{epoch}_checkpoint.pth')
    cls_path = os.path.join(ckpt_dir, f'cls_{epoch}_checkpoint.pth')
    torch.save(model.state_dict(), model_path)
    torch.save(classifier.state_dict(), cls_path)
    # eval_S3DIS.py reads from args.save_path directly; keep these copies for
    # compatibility with the existing evaluation call in train_S3DIS.py.
    torch.save(model.state_dict(), os.path.join(args.save_path, f'model_{epoch}_checkpoint.pth'))
    torch.save(classifier.state_dict(), os.path.join(args.save_path, f'cls_{epoch}_checkpoint.pth'))
    if refiner is not None:
        torch.save(refiner.state_dict(), os.path.join(ckpt_dir, f'refiner_{epoch}_checkpoint.pth'))
        torch.save(refiner.state_dict(), os.path.join(args.save_path, f'refiner_{epoch}_checkpoint.pth'))
    if learnable_sp is not None:
        torch.save(learnable_sp.state_dict(), os.path.join(ckpt_dir, f'learnable_sp_{epoch}_checkpoint.pth'))
        torch.save(learnable_sp.state_dict(), os.path.join(args.save_path, f'learnable_sp_{epoch}_checkpoint.pth'))


def compute_type1_centers(sp_feats, primitive_labels, primitive_centers, args, logger, sp_feats_rgb, sp_feats_region_num=None):
    """处理 Type 1 逻辑：基于颜色一致性的中心计算与权重分配"""
    primitive_loss_weight = torch.zeros((args.primitive_num))
    count = 0

    # 记录用于可视化的列表
    centers_origin_list = []
    centers_new_list = []
    valid_cluster_indices = []

    for cluster_idx in range(args.primitive_num):
        indices = primitive_labels == cluster_idx
        domin_ind, s1 = compute_color_consistency(sp_feats_rgb[indices])

        if s1['consistency_var'] > 0.14:
            cluster_avg = sp_feats[indices].mean(0, keepdims=True)
            weight = 1.0
        elif s1['dominant_ratio'] >= 0.5:
            count += 1
            cluster_avg = sp_feats[indices][domin_ind].mean(0, keepdims=True)
            weight = 1.0
        else:
            cluster_avg = sp_feats[indices].mean(0, keepdims=True)
            weight = 0.5

        primitive_centers[cluster_idx] = cluster_avg
        primitive_loss_weight[cluster_idx] = weight

        valid_cluster_indices.append(cluster_idx)
        centers_origin_list.append(sp_feats[indices].mean(0, keepdims=True))
        centers_new_list.append(cluster_avg)

    logger.info(f'unmatched center num == {count}')
    save_path = f'{args.pseudo_label_path}/primitive_loss_weight.pt'
    torch.save(primitive_loss_weight, save_path)
    logger.info(f"Saved primitive info to {save_path}")
    # visualize_cluster_refinement(sp_feats, primitive_labels, valid_cluster_indices, centers_origin_list, centers_new_list, args)

    return primitive_centers


def compute_type2_centers(sp_feats, primitive_labels, primitive_centers, args, logger, sp_feats_rgb=None, sp_feats_region_num=None):
    """处理 Type 2 逻辑：基于 Attention 的中心去噪"""
    temperature = getattr(args, 'temperature', 0.1)

    for cluster_idx in range(args.primitive_num):
        indices = primitive_labels == cluster_idx
        cluster_feats = sp_feats[indices]

        # 第一阶段：粗聚类初始化
        cluster_avg = cluster_feats.mean(dim=0, keepdim=True).detach()

        # 第二阶段：注意力机制去噪
        cluster_avg_norm = F.normalize(cluster_avg, p=2, dim=1)
        cluster_feats_norm = F.normalize(cluster_feats, p=2, dim=1)

        similarity = torch.mm(cluster_feats_norm, cluster_avg_norm.t())
        weights = F.softmax(similarity / temperature, dim=0)
        refined_center = torch.sum(weights * cluster_feats, dim=0)

        primitive_centers[cluster_idx] = refined_center

    return primitive_centers


def visualize_cluster_refinement(sp_feats, primitive_labels, valid_cluster_indices, centers_origin_list, centers_new_list, args):
    """执行 t-SNE 降维并保存可视化结果"""
    print("正在准备可视化数据...")

    # 修复了原代码的 bug：现在使用全局特征而不是最后一个簇的特征
    features_np = sp_feats.numpy()
    centers_origin_np = torch.cat(centers_origin_list, dim=0).numpy()
    centers_new_np = torch.cat(centers_new_list, dim=0).numpy()

    # S3DIS 特征点极多，强烈建议在这里加入随机采样以防止 t-SNE 内存溢出或卡死
    max_points = 5000
    if features_np.shape[0] > max_points:
        print(f"点数过多 ({features_np.shape[0]})，随机采样 {max_points} 个点用于背景显示...")
        sample_idx = np.random.choice(features_np.shape[0], max_points, replace=False)
        features_np = features_np[sample_idx]
        primitive_labels_np = primitive_labels[sample_idx].numpy()
    else:
        primitive_labels_np = primitive_labels.numpy()

    X_all = np.vstack([features_np, centers_origin_np, centers_new_np])

    print("正在运行 t-SNE (这可能需要一点时间)...")
    reducer = TSNE(n_components=2, init='pca', random_state=42)
    X_embedded = reducer.fit_transform(X_all)

    num_points = features_np.shape[0]
    num_clusters = centers_origin_np.shape[0]

    pts_2d = X_embedded[:num_points]
    centers_origin_2d = X_embedded[num_points: num_points + num_clusters]
    centers_new_2d = X_embedded[num_points + num_clusters:]

    plt.figure(figsize=(20, 20))
    colors = matplotlib.cm.nipy_spectral(np.linspace(0, 1, args.primitive_num))
    np.random.seed(42)
    np.random.shuffle(colors)

    print("正在绘制图像...")
    # 画背景点
    for i in range(num_clusters):
        real_id = valid_cluster_indices[i]
        mask = primitive_labels_np == real_id
        plt.scatter(pts_2d[mask, 0], pts_2d[mask, 1], color=colors[real_id], alpha=0.15, s=6, marker='.')

    # 画中心点与连线
    for i in range(num_clusters):
        real_id = valid_cluster_indices[i]
        color = colors[real_id]

        plt.scatter(centers_origin_2d[i, 0], centers_origin_2d[i, 1], marker='o', s=10, color=color, alpha=0.6, linewidths=0)
        plt.scatter(centers_new_2d[i, 0], centers_new_2d[i, 1], marker='*', s=10, edgecolors='k', facecolors=color, linewidths=0.3, zorder=10)

        dist = np.linalg.norm(centers_origin_2d[i] - centers_new_2d[i])
        if dist > 0.5:
            plt.plot([centers_origin_2d[i, 0], centers_new_2d[i, 0]],
                     [centers_origin_2d[i, 1], centers_new_2d[i, 1]],
                     color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Features', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='x', color='black', label='Original Center (Mean)', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='*', color='w', label='Refined Center', markerfacecolor='gray', markeredgecolor='black', markersize=10, linestyle='None'),
        Line2D([0], [0], color='black', lw=1, linestyle='--', label='Shift Path')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Cluster Refinement Visualization (t-SNE)")
    plt.axis('off')

    save_path = './visualization_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至: {save_path}")


def setup_loss_weight(args, loss):
    """阶段一：如果开启 region_weight_enable，动态加载并替换 loss 的权重"""
    if getattr(args, 'region_weight_enable', False):
        weight_path = f'{args.pseudo_label_path}/primitive_loss_weight.pt'
        primitive_loss_weight = torch.load(weight_path, map_location="cpu").cuda()
        loss.weight = primitive_loss_weight
    return loss


def compute_total_loss(args, ssl_loss, losses, ssl_loss_lambda=0.1):
    """阶段二：计算总 Loss，并返回需要记录的自定义 loss 字典"""

    if getattr(args, 'double_ssl', False):
        # 局部导入，避免未开启该功能时因为找不到模型文件而报错
        loss_ssl_func = MultiModalInfoNCELoss()
        loss_ssl = loss_ssl_func(ssl_loss[0], ssl_loss[1])

        # 【关键】依然保留 .item() 修复显存泄漏的问题
        losses['loss_ssl'] = loss_ssl.item() * ssl_loss_lambda

    return losses


def log_training_info(args, logger, epoch, batch_idx, num_batches, iteration, loss_display, losses_display, lr, time_used):
    # 动态构建各分项 loss 的字符串
    dynamic_loss_str = ""
    if losses_display:
        loss_strs = []
        for k, v in losses_display.items():
            avg_val = v / args.log_interval
            # 格式化拼接，如 "loss_custom: 0.123456"
            loss_strs.append(f"{k}: {avg_val:.6f}")
        
        # 将所有项用逗号连接，并在末尾加上逗号和空格以衔接后面的信息
        dynamic_loss_str = ", ".join(loss_strs) + ", "

    logger.info(
        'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.10f}, {}lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
            epoch, batch_idx, num_batches, 100. * batch_idx / num_batches,
            iteration, loss_display, dynamic_loss_str, lr, time_used, args.log_interval))
   


def setup_custom_env(args, project):
    """配置伪标签路径与 wandb 初始化"""
    proj_name = args.save_path.split('/')[-1]

    args.pseudo_label_path = os.path.join(args.pseudo_label_path, proj_name)
    os.makedirs(args.pseudo_label_path, exist_ok=True)
    wandb.login(key="3c02e03bad1d238fd2ebe43287bed3dcf9715810")

    wandb.init(
        entity="njucuimo",
        project=project,
        name=proj_name
    )


class WandbHandler(logging.Handler):
    # 定义类别顺序
    SEMANTIC_CLASSES = [
        "ceiling", "floor", "wall", "beam", "column",
        "window", "door", "table", "chair", "sofa", "bookcase", "board"
    ]

    def __init__(self):
        super().__init__()

    def emit(self, record):
        msg = self.format(record)
        logs = {}

        # --- 匹配 Epoch ---
        m = re.search(r'Train Epoch:\s*(\d+)', msg)
        if m:
            logs['Epoch'] = int(m.group(1))
            # --- 匹配 loss、lr ---
            for key, val in re.findall(r'([A-Za-z_]+):\s*([0-9.eE+-]+)', msg):
                try:
                    logs[key] = float(val)
                except ValueError:
                    pass

        # --- 匹配 Superpoints / Primitives 前缀 ---
        prefix = None

        # --- 每类 IoU ---
        m_iou = re.search(r'IoUs\|.*?\|\s*(.*)', msg)
        if m_iou:
            prefix = 'Val'
            if "Superpoints" in msg:
                prefix = "Superpoints"
            elif "Primitives" in msg:
                prefix = "Primitives"
            for key, val in re.findall(r'(\boAcc|\bmAcc|\bmIoU)\s+([0-9.]+)', msg):
                try:
                    key_name = f"{prefix}/{key}"
                    logs[key_name] = float(val)
                except ValueError:
                    pass
            m = re.search(r'Epoch:\s*(\d+)', msg)
            if m:
                logs['Epoch'] = int(m.group(1))

            iou_str = m_iou.group(1).strip()
            iou_list = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+', iou_str)]
            for idx, val in enumerate(iou_list):
                if idx < len(self.SEMANTIC_CLASSES):
                    key_name = f"{prefix}/{self.SEMANTIC_CLASSES[idx]}_IoU" if prefix else f"class_{idx}_IoU"
                    logs[key_name] = val

        if logs:
            wandb.log(logs)


def build_model(model_name='res16fpn18', **kwargs):
    if model_name == 'res16fpn18':
        return Res16FPN18(**kwargs)
    elif model_name == 'res16unet14':
        return Res16UNet14(**kwargs)
    else:
        raise ValueError(f'Unknown model: {model_name}')


# 定义空间平滑函数 (可放置在训练循环外部)


import torch
import torch.nn.functional as F

def spatial_consistency_loss(logits, sp_centroids, k=5, temperature=1.0, chunk_size=2048):
    """
    终极显存优化版的空间一致性损失。
    1. chunked KNN 解决距离矩阵的 OOM。
    2. k-loop 拆解解决 KL Divergence 时的中间缓存 OOM。
    """
    N = logits.shape[0]
    
    # 1. 无梯度计算 KNN 索引
    with torch.no_grad():
        knn_indices = torch.zeros((N, k), dtype=torch.long, device=sp_centroids.device)
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_centroids = sp_centroids[i:end_idx]
            dist_chunk = torch.cdist(chunk_centroids, sp_centroids, p=2)
            _, topk_idx = torch.topk(-dist_chunk, k=k+1, dim=1)
            knn_indices[i:end_idx] = topk_idx[:, 1:]

    # 2. 计算 Logits 概率
    pred_probs = F.softmax(logits / temperature, dim=-1) 
    pred_log_probs = F.log_softmax(logits / temperature, dim=-1) 
    
    # 作为目标的中心点概率不需要计算梯度
    target_probs = pred_probs.detach() # 形状: (N, M)
    
    loss_smooth = 0.0
    
    # 🌟 优化核心：通过循环 k 次，彻底消灭 (N, k, M) 的巨型 Tensor 分配
    for i in range(k):
        # 每次只取第 i 个邻居，形状变为 (N, M)
        neighbor_log_i = pred_log_probs[knn_indices[:, i]] 
        
        # 此时参与 kl_div 的张量形状全都是轻量级的 (N, M)
        loss_smooth += F.kl_div(neighbor_log_i, target_probs, reduction='batchmean')
        
    # 求 k 个邻居的平均损失
    return loss_smooth / k


# =====================================================================
# S3DIS 专属配色字典 (13类)，保证所有图表的颜色绝对一致，图例清晰
# =====================================================================
S3DIS_COLORS = {
    0: '#1f77b4',  # ceiling (蓝)
    1: '#ff7f0e',  # floor (橙)
    2: '#2ca02c',  # wall (绿)
    3: '#d62728',  # beam (红)
    4: '#9467bd',  # column (紫)
    5: '#8c564b',  # window (棕)
    6: '#e377c2',  # door (粉)
    7: '#7f7f7f',  # table (灰)
    8: '#bcbd22',  # chair (黄绿)
    9: '#17becf',  # sofa (青)
    10: '#aec7e8',  # bookcase (浅蓝)
    11: '#ffbb78',  # board (浅橙)
    12: '#98df8a'  # clutter (浅绿)
}

S3DIS_NAMES = {
    0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column',
    5: 'window', 6: 'door', 7: 'table', 8: 'chair', 9: 'sofa',
    10: 'bookcase', 11: 'board', 12: 'clutter'
}


# =====================================================================
# 1、「超点的真实语义中心 vs 超点特征的聚类中心」位置对比图
# =====================================================================
def plot_1_gt_vs_cluster_center_offset(
    sp_feats_np, sp_gt_labels, sp_pseudo_labels, primitive_centers_np,
    save_dir, logger, epoch=0
):
    """
    含义：真实语义中心在哪里？K-means把它聚到哪里去了？【补充】：为什么会被聚到那里？
    做法：每个真实类别生成两张图。一张看偏移距离，另一张“全景大图”画出该聚类簇的完整真实成分，解释偏移原因。
    """
    out_dir = os.path.join(save_dir, "1_Center_Offsets")
    os.makedirs(out_dir, exist_ok=True)

    unique_gt = np.unique(sp_gt_labels)

    for gt_label in unique_gt:
        mask = (sp_gt_labels == gt_label)
        if np.sum(mask) < 10:
            continue

        gt_name = S3DIS_NAMES.get(gt_label, f"C{gt_label}")
        gt_color = S3DIS_COLORS.get(gt_label, 'green')

        # 1. 计算真实语义中心
        class_feats = sp_feats_np[mask]
        gt_center = class_feats.mean(axis=0, keepdims=True)

        # 2. 找到最主要的聚类簇 (Top-1)
        assigned_protos = Counter(sp_pseudo_labels[mask]).most_common(2)
        if not assigned_protos:
            continue
        top1_pid = assigned_protos[0][0]
        top1_center_high_dim = primitive_centers_np[top1_pid].reshape(1, -1)

        # ==========================================================
        # 补充的大图：提取这个聚类簇(Top-1)内部的所有真实超点
        # ==========================================================
        cluster_mask = (sp_pseudo_labels == top1_pid)
        cluster_feats = sp_feats_np[cluster_mask]
        cluster_gts = sp_gt_labels[cluster_mask]

        # 将 GT点、Cluster所有点、GT中心、Cluster中心 拼在一起做一次统一的 PCA
        pca_data_big = np.vstack([class_feats, cluster_feats, gt_center, top1_center_high_dim])
        pca_2d_big = PCA(n_components=2).fit_transform(pca_data_big)

        # 拆分降维后的坐标
        feats_gt_2d = pca_2d_big[:len(class_feats)]
        feats_cluster_2d = pca_2d_big[len(class_feats): len(class_feats)+len(cluster_feats)]
        gt_center_2d = pca_2d_big[-2]
        cluster_center_2d = pca_2d_big[-1]

        # 计算高维原空间的真实欧氏距离
        true_distance = np.linalg.norm(gt_center.flatten() - top1_center_high_dim.flatten())

        # ==========================================================
        # 绘图：成因解释大图 (Explanation Big Picture)
        # ==========================================================
        plt.figure(figsize=(14, 10))

        # 背景：用浅灰色把该类的真实所有点画出来，展示“理想的分布范围”
        plt.scatter(feats_gt_2d[:, 0], feats_gt_2d[:, 1], c='lightgray', s=30, alpha=0.3,
                    label=f'Background: All True "{gt_name}" Points')

        # 前景：画出这个 K-means 簇里到底都有些什么东西（按真实类别上色）
        for g_val in np.unique(cluster_gts):
            g_mask = (cluster_gts == g_val)
            g_name = S3DIS_NAMES.get(g_val, f"C{g_val}")
            g_color = S3DIS_COLORS.get(g_val, 'gray')
            plt.scatter(feats_cluster_2d[g_mask, 0], feats_cluster_2d[g_mask, 1],
                        c=g_color, s=80, alpha=0.8, edgecolors='white', linewidths=0.5,
                        label=f'Pulled by: {g_name} (n={np.sum(g_mask)})')

        # 画真实语义中心
        plt.scatter(gt_center_2d[0], gt_center_2d[1], c='black', s=400, marker='o',
                    edgecolors='white', linewidths=3, label='Ground Truth Center', zorder=5)

        # 画聚类中心
        plt.scatter(cluster_center_2d[0], cluster_center_2d[1], c='red', s=600, marker='*',
                    edgecolors='black', linewidths=2, label=f'K-means Center (P{top1_pid})', zorder=5)

        # 画偏移箭头
        plt.annotate('', xy=(cluster_center_2d[0], cluster_center_2d[1]),
                     xytext=(gt_center_2d[0], gt_center_2d[1]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=3, ls='--'))

        # 距离文本
        mid_x = (gt_center_2d[0] + cluster_center_2d[0]) / 2
        mid_y = (gt_center_2d[1] + cluster_center_2d[1]) / 2
        plt.text(mid_x, mid_y, f' Offset Dist: {true_distance:.2f} ', color='red', fontsize=14, fontweight='bold',
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.85, edgecolor='red', boxstyle='round,pad=0.3'))

        plt.title(f'Why did the center shift?\nBecause Cluster P{top1_pid} is dragged away by other semantic points',
                  fontsize=18, fontweight='bold', y=1.02)

        # 将图例放在外侧，因为内容可能比较多
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, title="Cluster P{} Composition".format(top1_pid), title_fontsize=14)
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"offset_{gt_name}_explanation_big_picture_{epoch}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================================
        # (保留原来的基础图作为比对：仅显示类内偏移)
        # ==========================================================
        plt.figure(figsize=(10, 8))
        plt.scatter(feats_gt_2d[:, 0], feats_gt_2d[:, 1], c=gt_color, s=10, alpha=0.15, label=f'{gt_name} Superpoints')
        plt.scatter(gt_center_2d[0], gt_center_2d[1], c='black', s=300, marker='o', edgecolors='white', linewidths=3, label='Ground Truth Center', zorder=5)
        plt.scatter(cluster_center_2d[0], cluster_center_2d[1], c='red', s=400, marker='*', edgecolors='black', label=f'Cluster P{top1_pid}', zorder=5)
        plt.annotate('', xy=(cluster_center_2d[0], cluster_center_2d[1]), xytext=(gt_center_2d[0], gt_center_2d[1]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2.5, ls='--'))
        plt.text(mid_x, mid_y, f' Dist: {true_distance:.2f} ', color='red', fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.85, edgecolor='red'))
        plt.title(f'Center Offset (Basic View): True "{gt_name}" vs Assigned Center', fontsize=15, fontweight='bold')
        plt.legend(loc='best')
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"offset_{gt_name}_basic_{epoch}.png"), dpi=300)
        plt.close()


# =====================================================================
# 2、所有聚类中心的 “语义纯度” 堆叠条形图 (分块画全)
# =====================================================================
def plot_2_all_clusters_semantic_purity(
    sp_gt_labels, sp_pseudo_labels, num_clusters, save_dir, logger, epoch=0, chunk_size=50
):
    """
    含义：K-means出来的这M个簇，每个簇里面到底是由哪些真实类别构成的？
    做法：画出【所有】聚类簇，计算平均纯度并输出Log。同时将整体纯度指标打在生成的图片标题上。
    """
    out_dir = os.path.join(save_dir, "2_Purity_Stacks")
    os.makedirs(out_dir, exist_ok=True)

    unique_gt = np.unique(sp_gt_labels)

    # 统计每个簇的成分
    composition_matrix = np.zeros((num_clusters, len(unique_gt)))
    cluster_sizes = np.zeros(num_clusters)

    for pid in range(num_clusters):
        mask = (sp_pseudo_labels == pid)
        cluster_sizes[pid] = np.sum(mask)
        if cluster_sizes[pid] > 0:
            gt_counts = Counter(sp_gt_labels[mask])
            for i, gt in enumerate(unique_gt):
                composition_matrix[pid, i] = gt_counts.get(gt, 0)

    # 转换为百分比 (归一化)，即“纯度”
    with np.errstate(divide='ignore', invalid='ignore'):
        purity_matrix = np.nan_to_num(composition_matrix / composition_matrix.sum(axis=1, keepdims=True))

    # 过滤掉空簇
    valid_ids = np.where(cluster_sizes > 0)[0]

    # 初始化纯度指标（用于标题）
    macro_avg_purity = 0.0
    micro_avg_purity = 0.0

    # =====================================================================
    # 计算纯度指标并输出 Log
    # =====================================================================
    if len(valid_ids) > 0:
        # 每个有效簇的最大占比，即该簇的纯度
        cluster_purities = np.max(purity_matrix[valid_ids], axis=1)
        valid_sizes = cluster_sizes[valid_ids]

        # 计算两种平均值
        macro_avg_purity = np.mean(cluster_purities)
        micro_avg_purity = np.sum(cluster_purities * valid_sizes) / np.sum(valid_sizes)

        # 计算纯度分布区间
        num_valid = len(valid_ids)
        p90_plus = np.sum(cluster_purities >= 0.90)
        p70_90 = np.sum((cluster_purities >= 0.70) & (cluster_purities < 0.90))
        p50_70 = np.sum((cluster_purities >= 0.50) & (cluster_purities < 0.70))
        p50_minus = np.sum(cluster_purities < 0.50)

        # 构建日志信息
        log_msg = (
            f"\n" + "="*50 + "\n"
            f"📊 [Epoch {epoch}] 聚类语义纯度评估报告\n"
            f"--------------------------------------------------\n"
            f"有效聚类簇数量 : {num_valid} / {num_clusters}\n"
            f"Macro 平均纯度 : {macro_avg_purity * 100:.2f}% (按簇平均)\n"
            f"Micro 平均纯度 : {micro_avg_purity * 100:.2f}% (按点加权平均)\n"
            f"--------------------------------------------------\n"
            f"纯度分布区间:\n"
            f"  - 极高纯度 (≥ 90%) : {p90_plus:3d} 个簇 ({p90_plus/num_valid*100:.1f}%)\n"
            f"  - 较高纯度 (70-90%): {p70_90:3d} 个簇 ({p70_90/num_valid*100:.1f}%)\n"
            f"  - 中等纯度 (50-70%): {p50_70:3d} 个簇 ({p50_70/num_valid*100:.1f}%)\n"
            f"  - 混杂/低纯度 (< 50%): {p50_minus:3d} 个簇 ({p50_minus/num_valid*100:.1f}%)\n"
            + "="*50
        )

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

    # =====================================================================
    # 继续画图逻辑，并将指标拼接到标题
    # =====================================================================
    # 按超点数量从大到小排
    valid_ids = valid_ids[np.argsort(cluster_sizes[valid_ids])[::-1]]

    num_chunks = int(np.ceil(len(valid_ids) / chunk_size))
    legend_patches = [mpatches.Patch(color=S3DIS_COLORS.get(gt, 'gray'), label=S3DIS_NAMES.get(gt, f"C{gt}")) for gt in unique_gt]

    for c in range(num_chunks):
        chunk_ids = valid_ids[c*chunk_size: (c+1)*chunk_size]
        chunk_purities = purity_matrix[chunk_ids]

        plt.figure(figsize=(20, 8))
        bottoms = np.zeros(len(chunk_ids))
        x_labels = [f"P{pid}\n(n={int(cluster_sizes[pid])})" for pid in chunk_ids]

        for i, gt in enumerate(unique_gt):
            heights = chunk_purities[:, i]
            plt.bar(x_labels, heights, bottom=bottoms, color=S3DIS_COLORS.get(gt, 'gray'), edgecolor='white', width=0.8)
            bottoms += heights

        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% Purity Threshold')

        # 🌟 修改点：在标题中直接加入 Epoch 和 纯度指标
        title_str = (
            f'Semantic Composition of All Clusters (Part {c+1}/{num_chunks} | Epoch {epoch})\n'
            f'Overall Micro Purity: {micro_avg_purity*100:.2f}%  |  Overall Macro Purity: {macro_avg_purity*100:.2f}%'
        )
        plt.title(title_str, fontsize=18, fontweight='bold', pad=15)

        plt.ylabel('Proportion (Purity %)', fontsize=14)
        plt.xlabel('Cluster ID (and point count)', fontsize=14)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha='right')

        # 将全局图例放在外侧
        plt.legend(handles=legend_patches + [plt.Line2D([0], [0], color='red', lw=2, ls='--', label='50% Purity')],
                   bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"purity_stack_part_{c+1}_epoch_{epoch}.png"), dpi=300)
        plt.close()


# =====================================================================
# 3、同一聚类簇的 “内部真实语义构成” (散点图 + 柱状图) 解决看不清构成的问题
# =====================================================================
def plot_3_single_cluster_composition(
    sp_feats_np, sp_gt_labels, sp_pseudo_labels, primitive_centers_np,
    save_dir, logger, epoch=0, num_show=5
):
    """
    含义：我框出了这一个簇，它内部的特征是怎么分布的？到底是由哪几类杂交组成的？
    做法：挑选几个最杂乱的簇。生成 1x2 子图：
         左图：该簇内部的 PCA 散点图，严格按 S3DIS 颜色着色。
         右图：横向柱状图，明明白白写出该簇内各类别的精确数量。
    """
    out_dir = os.path.join(save_dir, "3_Cluster_Internal_Mix")
    os.makedirs(out_dir, exist_ok=True)

    num_clusters = primitive_centers_np.shape[0]

    # 找最混杂的几个簇 (香农熵最大)
    mixed_scores = []
    for pid in range(num_clusters):
        mask = (sp_pseudo_labels == pid)
        if np.sum(mask) > 50:  # 找稍微大一点的簇才有意义
            counts = np.array(list(Counter(sp_gt_labels[mask]).values()))
            probs = counts / np.sum(counts)
            entropy = -np.sum(probs * np.log2(probs + 1e-8))
            mixed_scores.append((pid, entropy))

    mixed_scores.sort(key=lambda x: x[1], reverse=True)
    target_ids = [item[0] for item in mixed_scores[:num_show]]

    for pid in target_ids:
        mask = (sp_pseudo_labels == pid)
        cluster_feats = sp_feats_np[mask]
        cluster_gts = sp_gt_labels[mask]

        # 计算内部成分
        gt_counter = Counter(cluster_gts)
        sorted_gts = sorted(gt_counter.items(), key=lambda x: x[1], reverse=False)  # 为横向柱状图准备

        # 降维
        pca_2d = PCA(n_components=2).fit_transform(cluster_feats)

        # 开启 1x2 画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [2, 1]})

        # ============ 左图：散点图 ============
        for gt_val, count in gt_counter.items():
            gt_mask = (cluster_gts == gt_val)
            gt_name = S3DIS_NAMES.get(gt_val, f"C{gt_val}")
            gt_color = S3DIS_COLORS.get(gt_val, 'gray')
            ax1.scatter(pca_2d[gt_mask, 0], pca_2d[gt_mask, 1], c=gt_color, s=80, alpha=0.8, edgecolors='white', label=f"{gt_name} ({count})")

        ax1.set_title(f'Feature Space of Cluster P{pid} (Colored by GT)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # ============ 右图：横向柱形图 (极其清晰的构成) ============
        bar_names = [S3DIS_NAMES.get(g[0], f"C{g[0]}") for g in sorted_gts]
        bar_counts = [g[1] for g in sorted_gts]
        bar_colors = [S3DIS_COLORS.get(g[0], 'gray') for g in sorted_gts]

        bars = ax2.barh(bar_names, bar_counts, color=bar_colors, edgecolor='black')
        ax2.set_title('Exact Composition Count', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Superpoints', fontsize=12)

        # 在柱子上标上数字和百分比
        total_pts = sum(bar_counts)
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + total_pts*0.02, bar.get_y() + bar.get_height()/2,
                     f"{int(width)} ({width/total_pts*100:.1f}%)",
                     va='center', fontsize=11, fontweight='bold')

        # 大标题
        fig.suptitle(f'Deep Dive: Why Cluster P{pid} is Semantically Mixed', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cluster_{pid}_composition_{epoch}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def plot_global_feature_space_comparison(
    sp_feats_np, sp_gt_labels, sp_pseudo_labels, primitive_centers_np,
    save_dir, logger, epoch=0
):
    """
    生成两张全局大图 (上帝视角)：
    所有超点 + 所有聚类中心(五角星) + 所有GT中心(圆圈)
    图A：超点按 K-means 聚类ID 着色 (展现几何聚类的边界)
    图B：超点按 真实语义标签 着色 (展现真实的语义流形分布)
    """
    out_dir = os.path.join(save_dir, "0_Global_Feature_Space")
    os.makedirs(out_dir, exist_ok=True)

    unique_gt = np.unique(sp_gt_labels)

    # 1. 计算所有 12 (或13) 个真实语义中心
    gt_centers_dict = {}
    for gt_label in unique_gt:
        mask = (sp_gt_labels == gt_label)
        if np.sum(mask) > 0:
            gt_centers_dict[gt_label] = sp_feats_np[mask].mean(axis=0)

    gt_center_labels = list(gt_centers_dict.keys())
    gt_centers_array = np.array(list(gt_centers_dict.values()))

    # 2. 全局 PCA 降维 (将所有超点、聚类中心、GT中心一起降维映射到同一个2D平面)
    # 为了保证空间一致性，必须用所有超点 fit PCA
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(sp_feats_np)
    protos_2d = pca.transform(primitive_centers_np)
    gt_centers_2d = pca.transform(gt_centers_array)

    # =====================================================================
    # 图 A：按 K-means 聚类 ID 着色 (The Geometric Partitions)
    # =====================================================================
    plt.figure(figsize=(16, 12))

    # 画所有超点 (按聚类簇着色，使用 tab20 循环色板以区分相邻的簇)
    plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=sp_pseudo_labels, cmap='tab20',
                s=10, alpha=0.4, edgecolors='none')

    # 画所有 GT 中心 (黑色大圆圈)
    plt.scatter(gt_centers_2d[:, 0], gt_centers_2d[:, 1], c='black', s=400, marker='o',
                edgecolors='white', linewidths=3, label='Ground Truth Centers (Ideal)', zorder=5)

    # 画所有 聚类中心 (红色大五角星)
    plt.scatter(protos_2d[:, 0], protos_2d[:, 1], c='red', s=150, marker='*',
                edgecolors='black', linewidths=1, label='K-means Cluster Centers', zorder=6)

    plt.title(f'Global Feature Space: Colored by K-means Clusters (Epoch {epoch})', fontsize=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=14)
    plt.axis('equal')
    plt.axis('off')  # 关掉坐标轴刻度，让图更纯粹
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"global_space_colored_by_clusters_{epoch}.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =====================================================================
    # 图 B：按 真实语义标签 (Ground Truth) 着色 (The Semantic Manifold)
    # =====================================================================
    plt.figure(figsize=(16, 12))

    # 准备图例
    legend_patches = []

    # 画所有超点 (按真实 S3DIS 标签着色)
    for gt_label in unique_gt:
        mask = (sp_gt_labels == gt_label)
        gt_name = S3DIS_NAMES.get(gt_label, f"C{gt_label}")
        gt_color = S3DIS_COLORS.get(gt_label, 'gray')

        plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=gt_color,
                    s=10, alpha=0.4, edgecolors='none')

        legend_patches.append(mpatches.Patch(color=gt_color, label=gt_name))

    # 画所有 GT 中心 (黑色大圆圈)
    for i, gt_label in enumerate(gt_center_labels):
        gt_name = S3DIS_NAMES.get(gt_label, f"C{gt_label}")
        plt.scatter(gt_centers_2d[i, 0], gt_centers_2d[i, 1], c='black', s=400, marker='o',
                    edgecolors='white', linewidths=3, zorder=5)
        # # 可选：在GT中心旁边打上类别名称
        plt.text(gt_centers_2d[i, 0], gt_centers_2d[i, 1], gt_name,
                 color='black', fontsize=12, fontweight='bold', ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 画所有 聚类中心 (红色大五角星)
    plt.scatter(protos_2d[:, 0], protos_2d[:, 1], c='red', s=150, marker='*',
                edgecolors='black', linewidths=1, zorder=6)

    # 添加黑圈和红星到图例
    legend_patches.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markeredgecolor='white', markersize=15, label='Ground Truth Centers'))
    legend_patches.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='black', markersize=20, label='K-means Cluster Centers'))

    plt.title(f'Global Feature Space: Colored by Ground Truth Semantics (Epoch {epoch})', fontsize=20, fontweight='bold')
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"global_space_colored_by_GT_{epoch}.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 全局特征空间大图已生成 (0_Global_Feature_Space)")

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def evaluate_and_plot_1to1_centers(sp_feats, sp_gt_labels, primitive_centers, args, logger, epoch=0):
    """
    独立计算 GT 中心和伪标签中心，通过匈牙利算法进行严格的 1对1 匹配，
    只使用 logger 输出信息，并将可视化匹配图保存到 4_1to1_Center_Matching。
    【优化版】修复索引匹配bug、优化标签位置、新增距离标注、防文字重叠
    """
    # ==========================================
    # 0. 准备保存目录与格式转换
    # ==========================================
    out_dir = os.path.join(args.save_path, "4_1to1_Center_Matching")
    os.makedirs(out_dir, exist_ok=True)

    # 统一转为numpy数组
    if isinstance(sp_feats, torch.Tensor):
        sp_feats = sp_feats.detach().cpu().numpy()
    if isinstance(sp_gt_labels, torch.Tensor):
        sp_gt_labels = sp_gt_labels.detach().cpu().numpy()
    if isinstance(primitive_centers, torch.Tensor):
        primitive_centers = primitive_centers.detach().cpu().numpy()

    sem_num = args.semantic_class  # S3DIS 中通常是 13
    s3dis_names = {
        0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column',
        5: 'window', 6: 'door', 7: 'table', 8: 'chair', 9: 'sofa',
        10: 'bookcase', 11: 'board', 12: 'clutter'
    }

    # ==========================================
    # 1. 独立计算 GT 语义中心
    # ==========================================
    gt_centers = np.full((sem_num, args.feats_dim), np.nan)  # 用nan替代inf，更安全
    unique_gt = np.unique(sp_gt_labels)

    for gt_label in unique_gt:
        mask = (sp_gt_labels == gt_label)
        if np.sum(mask) < 10:  # 过滤样本过少的类别
            continue
        class_feats = sp_feats[mask]
        gt_centers[gt_label] = class_feats.mean(axis=0)  # 去掉keepdims，维度对齐

    # ==========================================
    # 2. 独立计算 伪标签(宏观) 中心
    # ==========================================
    kmeans = KMeans(n_clusters=sem_num, n_init=10, random_state=0)
    kmeans.fit(primitive_centers)
    pseudo_macro_centers = kmeans.cluster_centers_

    # ==========================================
    # 3. 构建代价矩阵 & 匈牙利算法1对1匹配
    # ==========================================
    cost_matrix = np.full((sem_num, sem_num), 1e6)  # 无效类别默认高代价
    for i in range(sem_num):
        if not np.isnan(gt_centers[i]).any():  # 仅对有效GT类别计算代价
            for j in range(sem_num):
                cost_matrix[i, j] = np.linalg.norm(gt_centers[i] - pseudo_macro_centers[j])

    # 匈牙利算法匹配：row_ind=GT类别索引，col_ind=匹配的伪中心索引，一一对应
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # 【修复核心bug】建立GT类别到伪中心的正确映射，避免原代码索引错位问题
    gt_to_pseudo_map = {gt_idx: pseudo_idx for gt_idx, pseudo_idx in zip(row_ind, col_ind)}

    # ==========================================
    # 4. Logger 输出匹配结果
    # ==========================================
    logger.info("-" * 80)
    logger.info(f"{'GT Class':<12} | {'Matched Pseudo ID':<20} | {'L2 Distance':<12} | {'Cosine Sim':<12}")
    logger.info("-" * 80)

    # 预存有效匹配的完整信息，避免后续重复计算
    valid_matches = []
    total_l2 = 0.0
    total_cos = 0.0
    valid_class_count = 0

    for gt_idx, pseudo_idx in zip(row_ind, col_ind):
        if np.isnan(gt_centers[gt_idx]).any():
            continue
            
        gt_name = s3dis_names.get(gt_idx, f"Class_{gt_idx}")
        gt_c = gt_centers[gt_idx]
        ps_c = pseudo_macro_centers[pseudo_idx]

        # 计算匹配指标
        l2_dist = np.linalg.norm(gt_c - ps_c)
        cos_sim = np.dot(gt_c, ps_c) / (np.linalg.norm(gt_c) * np.linalg.norm(ps_c) + 1e-8)

        # 保存有效匹配信息，可视化直接用
        valid_matches.append({
            "gt_idx": gt_idx,
            "pseudo_idx": pseudo_idx,
            "gt_name": gt_name,
            "gt_center": gt_c,
            "pseudo_center": ps_c,
            "l2_dist": l2_dist,
            "cos_sim": cos_sim
        })

        # 统计全局指标
        total_l2 += l2_dist
        total_cos += cos_sim
        valid_class_count += 1

        logger.info(f"{gt_name:<12} | ID: {pseudo_idx:<16} | {l2_dist:<12.4f} | {cos_sim:<12.4f}")

    logger.info("-" * 80)
    if valid_class_count > 0:
        avg_l2 = total_l2 / valid_class_count
        avg_cos = total_cos / valid_class_count
        logger.info(f"[Epoch {epoch}] Global 1-to-1 Match Result -> Avg L2 Dist: {avg_l2:.4f}, Avg Cosine Sim: {avg_cos:.4f}")
        logger.info("-" * 80)
    else:
        logger.warning(f"[Epoch {epoch}] No valid GT classes found, skip visualization")
        return

    # ==========================================
    # 5. PCA降维 + 优化版2D可视化
    # ==========================================
    # 提取有效GT中心和所有伪中心，在同一空间做PCA降维
    valid_gt_centers = np.array([m["gt_center"] for m in valid_matches])
    combined_centers = np.vstack([valid_gt_centers, pseudo_macro_centers])
    
    # PCA降维到2D
    pca = PCA(n_components=2, random_state=0)
    pca_2d = pca.fit_transform(combined_centers)
    
    # 拆分GT和伪中心的2D坐标
    gt_2d_all = pca_2d[:len(valid_gt_centers)]
    pseudo_2d_all = pca_2d[len(valid_gt_centers):]

    # 【动态适配坐标】计算全局坐标范围，避免固定偏移导致的标签错位
    all_x = pca_2d[:, 0]
    all_y = pca_2d[:, 1]
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 动态计算标签偏移量（适配不同尺度的PCA结果）
    text_offset_y = y_range * 0.03  # y轴偏移为总范围的3%
    text_offset_x = x_range * 0.02  # x轴微调偏移
    # 坐标加padding，避免标签跑出画布
    pad_x = x_range * 0.1
    pad_y = y_range * 0.1

    # 初始化画布
    plt.figure(figsize=(12, 10), dpi=150)
    ax = plt.gca()

    # 1. 绘制所有伪宏观中心（红星，底层）
    plt.scatter(pseudo_2d_all[:, 0], pseudo_2d_all[:, 1], 
                c='red', s=120, marker='*', edgecolors='black', 
                linewidth=0.5, label='Macro Pseudo Centers', zorder=3)

    # 2. 遍历有效匹配，绘制GT点、连线、标签、距离标注
    for i, match in enumerate(valid_matches):
        gt_2d = gt_2d_all[i]
        pseudo_2d = pseudo_2d_all[match["pseudo_idx"]]
        gt_name = match["gt_name"]
        l2_dist = match["l2_dist"]

        # 绘制GT中心（黑圈，顶层）
        plt.scatter(gt_2d[0], gt_2d[1], 
                    c='black', s=100, marker='o', edgecolors='white', 
                    linewidth=1, zorder=5)

        # 【优化标签位置】动态调整标签位置，避免跑出画布/重叠
        # 标签默认在点上方，若点在画布顶部则放在下方
        if gt_2d[1] > (y_max - y_range * 0.15):
            text_y = gt_2d[1] - text_offset_y * 1.5
            va_align = 'top'
        else:
            text_y = gt_2d[1] + text_offset_y
            va_align = 'bottom'

        # 绘制类别名称标签，加半透明白底保证可读性
        plt.text(gt_2d[0], text_y, gt_name, 
                 color='black', fontsize=9, fontweight='bold',
                 ha='center', va=va_align,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

        # 绘制GT到伪中心的匹配虚线
        line, = plt.plot([gt_2d[0], pseudo_2d[0]], [gt_2d[1], pseudo_2d[1]],
                         color='royalblue', linestyle='--', linewidth=1, zorder=2)

        # 【新增核心功能】在连线中点标注L2距离
        mid_x = (gt_2d[0] + pseudo_2d[0]) / 2
        mid_y = (gt_2d[1] + pseudo_2d[1]) / 2
        plt.text(mid_x, mid_y, f"{l2_dist:.2f}",
                 color='darkblue', fontsize=7, fontweight='medium',
                 ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='royalblue', 
                           boxstyle='round,pad=0.2', alpha=0.9, linewidth=0.5))

    # 3. 补充GT图例、全局样式优化
    plt.scatter([], [], c='black', s=100, marker='o', edgecolors='white', 
                linewidth=1, label='Ground Truth Centers')
    plt.title(f'1-to-1 Hungarian Matching: GT vs Pseudo Macro Centers (Epoch {epoch})', 
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    # 设置坐标范围，加padding
    plt.xlim(x_min - pad_x, x_max + pad_x)
    plt.ylim(y_min - pad_y, y_max + pad_y)
    plt.grid(True, linestyle=':', alpha=0.5, zorder=0)
    plt.axis('equal')
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(out_dir, f"1to1_matching_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 可选：保存匹配结果到csv，方便后续分析
    if valid_class_count > 0:
        import pandas as pd
        df = pd.DataFrame(valid_matches)
        df.to_csv(os.path.join(out_dir, f"matching_result_epoch_{epoch}.csv"), index=False)
# =====================================================================
# 一键调用入口
# =====================================================================
def generate_all_s3dis_clustering_plots(
    sp_feats_np, sp_gt_labels, primitive_centers_np,
    args, logger, epoch=0
):

    dist_matrix = cdist(sp_feats_np, primitive_centers_np)  # (32071, 300)
    sp_pseudo_labels = np.argmin(dist_matrix, axis=1)  # (32071,)
    print(f"🚀 [Epoch {epoch}] 正在生成核心论证图表...")
    evaluate_and_plot_1to1_centers(sp_feats_np, sp_gt_labels, primitive_centers_np, args, logger, epoch)
    plot_global_feature_space_comparison(sp_feats_np, sp_gt_labels, sp_pseudo_labels, primitive_centers_np, args.save_path, logger, epoch)

    # 1. 位置对比图
    plot_1_gt_vs_cluster_center_offset(sp_feats_np, sp_gt_labels, sp_pseudo_labels, primitive_centers_np, args.save_path, logger, epoch)
    print("✅ 图1: 真实中心偏移图 (1_Center_Offsets) 已生成")

    # 2. 全量纯度图
    num_clusters = primitive_centers_np.shape[0]
    plot_2_all_clusters_semantic_purity(sp_gt_labels, sp_pseudo_labels, num_clusters, args.save_path, logger, epoch)
    print("✅ 图2: 全量聚类纯度堆叠图 (2_Purity_Stacks) 已生成")

    # 3. 散点+柱状图
    plot_3_single_cluster_composition(sp_feats_np, sp_gt_labels, sp_pseudo_labels, primitive_centers_np, args.save_path, logger, epoch)
    print("✅ 图3: 单簇内部真实构成剖析图 (3_Cluster_Internal_Mix) 已生成")


def update_prototypes_with_metaode(initial_prototypes, metaode_func, t_span):
    initial_prototypes = initial_prototypes.unsqueeze(0)  # (1, num_classes, feat_dim)
    t = torch.linspace(0, 1, steps=t_span).float().to(initial_prototypes.device)
    updated = odeint(metaode_func, initial_prototypes, t, method='rk4')[-1]
    updated = F.normalize(updated, dim=-1)
    return updated.squeeze(0)  # (num_classes, feat_dim)


def updated_prototypes(in_channel, centroids_num, centroids):
    metaode_func = OptimizedPrototypeMetaODEFunc(feat_dim=in_channel, hidden_dim=128, num_heads=4, dropout=0.1).to(centroids.device)
    updated_prototypes = update_prototypes_with_metaode(centroids, metaode_func, t_span=4)

    return updated_prototypes


class OptimizedPrototypeMetaODEFunc(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        """
        优化版的 MetaODE 模块
        参数:
            feat_dim: 超点原型的特征维度（例如 300）
            hidden_dim: 内部隐藏层维度，用于时间嵌入和前馈网络（例如 128）
            num_heads: 多头自注意力中的头数（例如 4）
            dropout: dropout 概率，防止过拟合
        """
        super(OptimizedPrototypeMetaODEFunc, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # 时间嵌入模块：将标量时间 t 映射到 hidden_dim 维度
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 拼接原型和时间嵌入后投影回 feat_dim
        self.proj = nn.Linear(feat_dim + hidden_dim, feat_dim)

        # 多头自注意力模块，用于捕捉各超点之间的交互信息
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        # 前馈网络进一步提取非线性特征
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        # 使用 LayerNorm 和 Dropout 进行残差连接后的归一化
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.norm3 = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, prototypes):
        """
        Args:
            t: 标量时间参数（例如 0.5）
            prototypes: Tensor, 尺寸为 (batch, n_way, feat_dim)，表示当前超点原型
        Returns:
            输出同样为 (batch, n_way, feat_dim)，代表原型的更新量 d(prototypes)/dt
        """
        batch, n_way, feat_dim = prototypes.shape

        # 1. 时间嵌入：将标量 t 映射为 (1, hidden_dim)，并扩展至 (batch, n_way, hidden_dim)
        t_tensor = torch.tensor([[t]], device=prototypes.device, dtype=prototypes.dtype)
        t_embed = self.time_embed(t_tensor)  # (1, hidden_dim)
        t_embed = t_embed.expand(batch, n_way, self.hidden_dim)

        # 2. 拼接原型特征与时间嵌入，再投影回 feat_dim
        prot_concat = torch.cat([prototypes, t_embed], dim=-1)  # (batch, n_way, feat_dim + hidden_dim)
        d_prototypes_time = self.proj(prot_concat)  # (batch, n_way, feat_dim)

        # 第一部分更新：加上基于时间的信息（残差连接）
        out = prototypes + d_prototypes_time
        out = self.norm1(out)

        # 3. 多头自注意力：各超点之间信息交流，捕捉全局关系
        attn_out, _ = self.attn(out, out, out)
        out = out + self.dropout(attn_out)
        out = self.norm2(out)

        # 4. 前馈网络：进一步非线性变换
        ffn_out = self.ffn(out)
        out = out + self.dropout(ffn_out)
        out = self.norm3(out)

        return out
