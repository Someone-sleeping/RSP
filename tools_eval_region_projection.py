import argparse
import os

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from eval_S3DIS import compute_unsupervised_metrics
from lib.split_regions import build_region_consistency_queries, build_split_region_queries
from lib.utils import get_fixclassifier
from models.fpn import Res16FPN18


class Args:
    data_path = "data/S3DIS/input"
    sp_path = "data/S3DIS/initial_superpoints/"
    save_path = "ckpt/S3DIS/refiner_split_best_smoke/"
    bn_momentum = 0.02
    conv1_kernel_size = 5
    workers = 8
    cluster_workers = 4
    seed = 2022
    voxel_size = 0.05
    input_dim = 6
    primitive_num = 300
    semantic_class = 12
    feats_dim = 128
    ignore_label = 12


def parse_args():
    parser = argparse.ArgumentParser("Evaluate projection and smoothing baselines on S3DIS")
    parser.add_argument("--save_path", default="ckpt/S3DIS/refiner_projectloss02_e10/")
    parser.add_argument("--eval_epoch", default="best", help='checkpoint epoch or "best"')
    parser.add_argument("--test_area", default="Area_5", help="S3DIS held-out area, or comma-separated areas")
    parser.add_argument("--data_path", default=Args.data_path)
    parser.add_argument("--sp_path", default=Args.sp_path)
    parser.add_argument("--workers", type=int, default=Args.workers)
    parser.add_argument("--cluster_workers", type=int, default=Args.cluster_workers)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--voxel_size", type=float, default=Args.voxel_size)
    parser.add_argument("--input_dim", type=int, default=Args.input_dim)
    parser.add_argument("--primitive_num", type=int, default=Args.primitive_num)
    parser.add_argument("--semantic_class", type=int, default=Args.semantic_class)
    parser.add_argument("--feats_dim", type=int, default=Args.feats_dim)
    parser.add_argument("--ignore_label", type=int, default=Args.ignore_label)
    parser.add_argument("--bn_momentum", type=float, default=Args.bn_momentum)
    parser.add_argument("--conv1_kernel_size", type=int, default=Args.conv1_kernel_size)
    return parser.parse_args(namespace=Args())


def parse_test_areas(test_area):
    areas = [area.strip() for area in str(test_area).split(",") if area.strip()]
    if not areas:
        raise ValueError("test_area must contain at least one S3DIS area")
    return areas


def checkpoint_paths(save_path, eval_epoch):
    model_name = "model_best_checkpoint.pth" if str(eval_epoch) == "best" else f"model_{eval_epoch}_checkpoint.pth"
    cls_name = "cls_best_checkpoint.pth" if str(eval_epoch) == "best" else f"cls_{eval_epoch}_checkpoint.pth"
    return os.path.join(save_path, model_name), os.path.join(save_path, cls_name)


def main():
    args = parse_args()
    model = Res16FPN18(
        in_channels=args.input_dim,
        out_channels=args.primitive_num,
        conv1_kernel_size=args.conv1_kernel_size,
        config=args,
    ).cuda()
    model_path, cls_path = checkpoint_paths(args.save_path, args.eval_epoch)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(cls_path))
    cls.eval()

    primitive_centers = cls.weight.data
    cluster_pred = KMeans(
        n_clusters=args.semantic_class,
        n_init=10,
        random_state=0,
        n_jobs=10,
    ).fit_predict(primitive_centers.cpu().numpy())
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        centroids[cluster_idx] = primitive_centers[cluster_pred == cluster_idx].mean(0, keepdims=True)
    classifier = get_fixclassifier(args.feats_dim, args.semantic_class, F.normalize(centroids, dim=1)).cuda()
    classifier.eval()

    loader = DataLoader(
        S3DIStest(args, areas=parse_test_areas(args.test_area)),
        batch_size=1,
        collate_fn=cfl_collate_fn_test(),
        num_workers=4,
        pin_memory=True,
    )

    split_configs = [
        ("split_s1_m80_p92_e25", 1.0, 80, 0.92, 0.25, 0.15, "score"),
        ("random_split_s1_m80_p92_e25", 1.0, 80, 0.92, 0.25, 0.15, "random"),
        ("split_s3_m80_p92_e25", 3.0, 80, 0.92, 0.25, 0.15, "score"),
        ("split_s10_m80_p92_e25", 10.0, 80, 0.92, 0.25, 0.15, "score"),
        ("split_s10_m120_p92_e25", 10.0, 120, 0.92, 0.25, 0.15, "score"),
        ("split_s10_m120_p95_e30", 10.0, 120, 0.95, 0.30, 0.15, "score"),
        ("split_s10_m120_p98_e40", 10.0, 120, 0.98, 0.40, 0.10, "score"),
        ("split_s5_m160_p95_e40", 5.0, 160, 0.95, 0.40, 0.10, "score"),
    ]
    base_region_names = [
        "region_logit",
        "region_prob",
        "region_confprob",
        "region_feat",
        "region_vote",
    ]
    strategy_names = ["base", "graph_smooth", "consistency"]
    strategy_names.extend(base_region_names)
    for region_name in base_region_names:
        strategy_names.append("consistency_" + region_name)
    for name, *_ in split_configs:
        strategy_names.append(name)
        for region_name in base_region_names:
            strategy_names.append(name + "_" + region_name)
    all_preds = {name: [] for name in strategy_names}
    all_labels = []
    for coords, features, inverse_map, labels, index, region in loader:
        with torch.no_grad():
            feats = F.normalize(model(ME.TensorField(features, coords, device=0)), dim=1)
            scores = F.linear(feats, F.normalize(classifier.weight))
            base = scores.argmax(dim=1).cpu()
            all_preds["base"].append(base[inverse_map.long()][labels != args.ignore_label])

            region_preds = {name: base.clone() for name in base_region_names}
            region = region.squeeze()
            probs = F.softmax(scores, dim=1)
            point_conf = probs.max(dim=1)[0]
            for region_id in torch.unique(region):
                if int(region_id) == -1:
                    continue
                mask = region == region_id
                region_preds["region_logit"][mask] = scores[mask].mean(dim=0).argmax().cpu()
                region_preds["region_prob"][mask] = probs[mask].mean(dim=0).argmax().cpu()
                weights = point_conf[mask].clamp_min(1e-6)
                region_preds["region_confprob"][mask] = ((probs[mask] * weights[:, None]).sum(dim=0) / weights.sum()).argmax().cpu()
                feat_score = F.linear(F.normalize(feats[mask].mean(dim=0, keepdim=True), dim=1), F.normalize(classifier.weight))
                region_preds["region_feat"][mask] = feat_score.argmax(dim=1).cpu()[0]
                vote_labels, vote_counts = torch.unique(base[mask.cpu()], return_counts=True)
                region_preds["region_vote"][mask] = vote_labels[torch.argmax(vote_counts)]
            for region_name, pred in region_preds.items():
                all_preds[region_name].append(pred[inverse_map.long()][labels != args.ignore_label])

            point_batch_ids = coords[:, 0].long().cuda()
            point_coords = coords[:, 1:].float().cuda()
            point_colors = features[:, :3].float().cuda()
            for split_name, logit_scale, max_regions, purity_th, entropy_th, min_conf, selection_mode in split_configs:
                split_pred = base.clone()
                (
                    _split_queries,
                    _split_mask,
                    split_targets,
                    _split_conf,
                    _split_keep,
                    _split_stats,
                ) = build_split_region_queries(
                    scores * logit_scale,
                    feats,
                    point_coords,
                    point_colors,
                    region.cuda(),
                    point_batch_ids,
                    min_region_points=30,
                    min_child_points=8,
                    max_split_regions_per_scene=max_regions,
                    split_purity_threshold=purity_th,
                    split_entropy_threshold=entropy_th,
                    split_min_conf=min_conf,
                    xyz_weight=1.0,
                    rgb_weight=0.5,
                    feat_weight=0.25,
                    semantic_weight=1.0,
                    selection_mode=selection_mode,
                    random_seed=args.seed,
                )
                split_valid = (split_targets >= 0).cpu()
                split_pred[split_valid] = split_targets.cpu()[split_valid]
                all_preds[split_name].append(split_pred[inverse_map.long()][labels != args.ignore_label])
                for region_name, region_pred in region_preds.items():
                    split_region_pred = region_pred.clone()
                    split_region_pred[split_valid] = split_targets.cpu()[split_valid]
                    all_preds[split_name + "_" + region_name].append(
                        split_region_pred[inverse_map.long()][labels != args.ignore_label]
                    )

            consistency_pred = base.clone()
            (
                _consistency_queries,
                _consistency_mask,
                consistency_targets,
                _consistency_conf,
                _consistency_keep,
                _consistency_stats,
            ) = build_region_consistency_queries(
                scores * 10.0,
                region.cuda(),
                point_batch_ids,
                min_region_points=20,
                max_regions_per_scene=80,
                min_region_conf=0.35,
                min_disagree_ratio=0.02,
                point_conf_threshold=0.55,
                point_entropy_threshold=0.55,
            )
            consistency_valid = (consistency_targets >= 0).cpu()
            consistency_pred[consistency_valid] = consistency_targets.cpu()[consistency_valid]
            all_preds["consistency"].append(consistency_pred[inverse_map.long()][labels != args.ignore_label])
            for region_name, region_pred in region_preds.items():
                consistency_region_pred = region_pred.clone()
                consistency_region_pred[consistency_valid] = consistency_targets.cpu()[consistency_valid]
                all_preds["consistency_" + region_name].append(
                    consistency_region_pred[inverse_map.long()][labels != args.ignore_label]
                )

            graph_projected = base.clone()
            region_ids = [r for r in torch.unique(region).tolist() if int(r) != -1]
            if len(region_ids) > 1:
                region_centers, region_scores = [], []
                for region_id in region_ids:
                    mask = region == int(region_id)
                    xyz = coords[mask, 1:].float().mean(dim=0).cpu()
                    rgb = features[mask, :3].float().mean(dim=0).cpu() / 255.0
                    region_centers.append(torch.cat([xyz / coords[:, 1:].float().std(dim=0).cpu().clamp_min(1.0), rgb], dim=0))
                    region_scores.append(scores[mask].mean(dim=0).cpu())
                region_centers = torch.stack(region_centers, dim=0).numpy()
                region_scores = torch.stack(region_scores, dim=0)
                k = min(8, len(region_ids))
                region_knn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(region_centers).kneighbors(return_distance=False)
                smooth_scores = region_scores[region_knn].mean(dim=1)
                smooth_preds = smooth_scores.argmax(dim=1)
                for idx, region_id in enumerate(region_ids):
                    graph_projected[region == int(region_id)] = smooth_preds[idx]
            all_preds["graph_smooth"].append(graph_projected[inverse_map.long()][labels != args.ignore_label])

            valid = labels != args.ignore_label
            all_labels.append(labels[valid])

    labels = torch.cat(all_labels).numpy()
    base_np = torch.cat(all_preds["base"]).numpy()
    base_miou = compute_unsupervised_metrics(base_np, labels, args.semantic_class)[3]
    for name in strategy_names:
        pred_np = torch.cat(all_preds[name]).numpy()
        metrics = compute_unsupervised_metrics(pred_np, labels, args.semantic_class)[:4]
        print(name, metrics, "delta", metrics[3] - base_miou, "changed", float((base_np != pred_np).mean()))


if __name__ == "__main__":
    main()
