import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStrain, cfl_collate_fn as s3dis_collate
from datasets.ScanNet import Scannettrain, cfl_collate_fn as scannet_collate
from datasets.SemanticKITTI import KITTItrain, cfl_collate_fn as kitti_collate
from lib.split_regions import build_region_consistency_queries, build_split_region_queries
from models.query_refiner import (
    ErrorQueryRefiner,
    delta_l2,
    gate_refiner_residual,
    refinement_keep_kl,
    resolve_min_temporal_votes,
)
from tools_eval_error_verifier import (
    KNOWN_INVALID_BASE_CHECKPOINTS,
    load_reference,
    model_input,
    project_region_and_split,
    region_candidates,
    split_colors,
)


PRESETS = {
    "scannet": {
        "checkpoint_dir": "ckpt/ScanNet/baseline",
        "base_epoch": 930,
        "reference_epochs": "900,910,920",
        "data_path": "data/ScanNet/processed/",
        "sp_path": "data/ScanNet/initial_superpoints/",
        "input_dim": 6,
        "primitive_num": 300,
        "semantic_class": 20,
        "feats_dim": 128,
        "ignore_label": -1,
        "voxel_size": 0.05,
        "batch_size": 4,
    },
    "semantickitti": {
        "checkpoint_dir": "ckpt/SemanticKITTI/baseline/ckpts",
        "base_epoch": 400,
        "reference_epochs": "370,380,390",
        "data_path": "data/SemanticKITTI/dataset/sequences",
        "sp_path": "data/SemanticKITTI/initial_superpoints/sequences/",
        "input_dim": 3,
        "primitive_num": 500,
        "semantic_class": 19,
        "feats_dim": 128,
        "ignore_label": -1,
        "voxel_size": 0.15,
        "batch_size": 8,
    },
    "logosp_s3dis": {
        "checkpoint_dir": "/home/magic/magic/cm/repositories/LogoSP/ckpt/S3DIS/seg",
        "base_epoch": 20,
        "reference_epochs": "10",
        "data_path": "/home/magic/magic/cm/repositories/LogoSP/data/S3DIS/input_0.010",
        "sp_path": "/home/magic/magic/cm/repositories/LogoSP/data/S3DIS/initial_superpoints/",
        "input_dim": 3,
        "primitive_num": 12,
        "semantic_class": 12,
        "feats_dim": 384,
        "ignore_label": 12,
        "voxel_size": 0.05,
        "batch_size": 4,
    },
}


@dataclass
class TrainStats:
    steps: int = 0
    loss: float = 0.0
    optimize_points: int = 0
    points: int = 0


def parse_args():
    parser = argparse.ArgumentParser("Train a dataset-specific label-free Error-Query Refiner")
    parser.add_argument("--dataset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--base_epoch", type=int, default=None)
    parser.add_argument("--reference_epochs", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--sp_path", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--training_scenes", type=int, default=800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument(
        "--min_temporal_votes",
        type=int,
        default=2,
        help="Required historical checkpoint votes; zero selects a majority dynamically.",
    )
    parser.add_argument("--temporal_confidence_threshold", type=float, default=0.80)
    parser.add_argument("--allow_invalid_checkpoint", action="store_true", default=False)
    return parser.parse_args()


def apply_preset(args):
    preset = PRESETS[args.dataset]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.bn_momentum = 0.02
    args.conv1_kernel_size = 5
    args.drop_threshold = 10
    args.r_crop = 50
    args.pseudo_label_path = os.path.join(args.save_path, "unused_pseudo")
    args.allow_missing_pseudo = True
    return args


def parse_epochs(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_train_dataset(args):
    if args.dataset == "scannet":
        return Scannettrain(args), scannet_collate()
    if args.dataset == "semantickitti":
        total = 19130
        count = min(int(args.training_scenes), total)
        indices = np.linspace(0, total - 1, num=count, dtype=np.int64)
        return KITTItrain(args, indices), kitti_collate()
    areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]
    return S3DIStrain(args, areas=areas), s3dis_collate()


def temporal_statistics(reference_models, field, inds, semantic_classes):
    probabilities = []
    with torch.no_grad():
        for model, centers in reference_models:
            features = F.normalize(model(field)[inds], dim=1)
            probabilities.append(F.softmax(F.linear(features, centers) * 10.0, dim=1))
    probabilities = torch.stack(probabilities)
    mean_probability = probabilities.mean(dim=0)
    predictions = probabilities.argmax(dim=2)
    vote_count = F.one_hot(predictions, num_classes=semantic_classes).sum(dim=0)
    votes, _ = vote_count.max(dim=1)
    return mean_probability, votes


def build_targets(args, base_scores, base_features, coords, features, regions, batch_ids, temporal_probability, temporal_votes):
    point_coords = coords[:, 1:].float().cuda()
    colors = split_colors(args, coords, features).float().cuda()
    query_indices, refine_mask, split_targets, split_conf, keep_mask, _ = build_split_region_queries(
        base_scores,
        base_features,
        point_coords,
        colors,
        regions,
        batch_ids,
        min_region_points=30,
        min_child_points=8,
        max_split_regions_per_scene=80,
        split_purity_threshold=0.92,
        split_entropy_threshold=0.25,
        split_min_conf=0.15,
        xyz_weight=1.0,
        rgb_weight=0.5,
        feat_weight=0.25,
        semantic_weight=1.0,
    )
    consistency = build_region_consistency_queries(
        base_scores * 10.0,
        regions,
        batch_ids,
        min_region_points=20,
        max_regions_per_scene=40,
        min_region_conf=0.35,
        min_disagree_ratio=0.02,
        point_conf_threshold=0.55,
        point_entropy_threshold=0.55,
    )
    consistency_queries, consistency_mask, consistency_targets, consistency_conf, consistency_keep, _ = consistency
    if consistency_queries.numel() > 0:
        query_indices = torch.unique(torch.cat((query_indices, consistency_queries)))
    refine_mask = refine_mask | consistency_mask
    keep_mask = keep_mask | consistency_keep

    target = torch.full_like(base_scores.argmax(dim=1), -1)
    target_confidence = base_scores.new_zeros(target.shape)
    split_valid = split_targets >= 0
    target[split_valid] = split_targets[split_valid]
    target_confidence[split_valid] = split_conf[split_valid]
    temporal_confidence, temporal_target = temporal_probability.max(dim=1)
    reference_count = len(parse_epochs(args.reference_epochs))
    required_votes = resolve_min_temporal_votes(args.min_temporal_votes, reference_count)
    temporal_valid = (
        (target < 0)
        & (temporal_votes >= required_votes)
        & (temporal_confidence >= args.temporal_confidence_threshold)
        & (temporal_target != base_scores.argmax(dim=1))
    )
    target[temporal_valid] = temporal_target[temporal_valid]
    target_confidence[temporal_valid] = temporal_confidence[temporal_valid]
    consistency_valid = (target < 0) & (consistency_targets >= 0) & (consistency_conf >= 0.75)
    target[consistency_valid] = consistency_targets[consistency_valid]
    target_confidence[consistency_valid] = consistency_conf[consistency_valid]
    optimize_mask = refine_mask & (target >= 0) & (target != base_scores.argmax(dim=1))
    keep_mask = keep_mask | (refine_mask & ~optimize_mask)
    return query_indices, refine_mask, split_targets, target, target_confidence, optimize_mask, keep_mask


def main():
    args = apply_preset(parse_args())
    os.makedirs(args.save_path, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_model_path = os.path.join(
        args.checkpoint_dir, f"model_{args.base_epoch}_checkpoint.pth"
    )
    base_model_hash = checkpoint_sha256(base_model_path)
    if (
        base_model_hash in KNOWN_INVALID_BASE_CHECKPOINTS
        and not args.allow_invalid_checkpoint
    ):
        raise ValueError(KNOWN_INVALID_BASE_CHECKPOINTS[base_model_hash])

    base_model, base_centers = load_reference(args, args.base_epoch)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    reference_models = []
    for epoch in parse_epochs(args.reference_epochs):
        model, centers = load_reference(args, epoch, reference_centers=base_centers)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        reference_models.append((model, centers))

    dataset, collate = build_train_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )
    refiner = ErrorQueryRefiner(
        feat_dim=args.feats_dim,
        num_classes=args.semantic_class,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=0.0,
    ).cuda()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        stats = TrainStats()
        dataset.mode = "train"
        for step, data in enumerate(loader, start=1):
            if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                break
            coords, features, _normals, _labels, _inverse, _pseudo, inds, region, _index = data
            selected_coords = coords[inds.long()]
            selected_features = features[inds.long()]
            field = ME.TensorField(model_input(args, coords, features), coords, device=0)
            with torch.no_grad():
                base_features = F.normalize(base_model(field)[inds.long()], dim=1)
                base_scores = F.linear(base_features, base_centers)
                temporal_probability, temporal_votes = temporal_statistics(
                    reference_models, field, inds.long(), args.semantic_class
                )
            regions = region.squeeze(-1).long().cuda()
            batch_ids = selected_coords[:, 0].long().cuda()
            targets = build_targets(
                args,
                base_scores,
                base_features,
                selected_coords,
                selected_features,
                regions,
                batch_ids,
                temporal_probability,
                temporal_votes,
            )
            query_indices, refine_mask, split_targets, target, target_confidence, optimize_mask, keep_mask = targets
            if query_indices.numel() == 0 or not optimize_mask.any():
                continue
            delta = refiner(
                base_features.detach(),
                selected_coords[:, 1:].float().cuda(),
                batch_ids,
                query_indices,
                regions,
                use_region_branch=False,
            )
            delta = gate_refiner_residual(delta, refine_mask)
            refined_scores = base_scores + delta
            projected_scores = project_region_and_split(refined_scores, regions, split_targets)
            weights = target_confidence[optimize_mask].clamp_min(0.05)
            point_ce = F.cross_entropy(refined_scores[optimize_mask] * 3.0, target[optimize_mask], reduction="none")
            project_ce = F.cross_entropy(projected_scores[optimize_mask] * 3.0, target[optimize_mask], reduction="none")
            loss_target = ((point_ce + 0.2 * project_ce) * weights).sum() / weights.sum()
            loss_keep = refinement_keep_kl(refined_scores * 3.0, base_scores * 3.0, keep_mask)
            probability = F.softmax(refined_scores[optimize_mask], dim=1)
            loss_entropy = -(probability * torch.log(probability.clamp_min(1e-6))).sum(dim=1).mean()
            loss = loss_target + loss_keep + 0.005 * delta_l2(delta, optimize_mask) + 0.02 * loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats.steps += 1
            stats.loss += float(loss.item())
            stats.optimize_points += int(optimize_mask.sum().item())
            stats.points += int(base_scores.size(0))
            if stats.steps % 20 == 0:
                print(
                    f"[{args.dataset}] epoch {epoch} step {stats.steps} "
                    f"loss {stats.loss / stats.steps:.4f} optimize {100.0 * stats.optimize_points / max(stats.points, 1):.2f}%",
                    flush=True,
                )
        epoch_summary = {
            "epoch": epoch,
            "steps": stats.steps,
            "mean_loss": stats.loss / max(stats.steps, 1),
            "optimize_ratio": stats.optimize_points / max(stats.points, 1),
        }
        history.append(epoch_summary)
        torch.save(refiner.state_dict(), os.path.join(args.save_path, f"refiner_{epoch}_checkpoint.pth"))
        print(json.dumps(epoch_summary), flush=True)

    refiner_path = os.path.join(args.save_path, "refiner_final_checkpoint.pth")
    torch.save(refiner.state_dict(), refiner_path)
    model_path = os.path.join(args.checkpoint_dir, f"model_{args.base_epoch}_checkpoint.pth")
    classifier_path = os.path.join(args.checkpoint_dir, f"cls_{args.base_epoch}_checkpoint.pth")
    metadata = {
        "dataset": args.dataset,
        "base_epoch": args.base_epoch,
        "reference_epochs": parse_epochs(args.reference_epochs),
        "base_model_sha256": checkpoint_sha256(model_path),
        "base_classifier_sha256": checkpoint_sha256(classifier_path),
        "refiner_checkpoint": refiner_path,
        "refiner_sha256": checkpoint_sha256(refiner_path),
        "history": history,
        "target_configuration": {
            "min_temporal_votes": args.min_temporal_votes,
            "effective_min_temporal_votes": resolve_min_temporal_votes(
                args.min_temporal_votes, len(parse_epochs(args.reference_epochs))
            ),
            "temporal_confidence_threshold": args.temporal_confidence_threshold,
        },
        "label_usage": "No ground-truth labels are used by the Refiner optimizer.",
    }
    with open(os.path.join(args.save_path, "training_metadata.json"), "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
