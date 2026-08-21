import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from types import SimpleNamespace

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import DataLoader, Subset

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from datasets.ScanNet import Scannetval, cfl_collate_fn_val as scannet_collate
from datasets.SemanticKITTI import KITTIval, cfl_collate_fn_val as kitti_collate
from eval_S3DIS import compute_unsupervised_metrics
from lib.semantic_difference_pipeline import (
    SemanticDifferencePipelineConfig,
    run_semantic_difference_pipeline,
)
from models.fpn import Res16FPN18
from models.learnable_superpoint import SemanticDifferenceSuperpointLearner
from models.query_refiner import ErrorQueryRefiner


STAGE_ORDER = (
    "base",
    "decomposition",
    "refiner",
    "meta_refiner",
    "meta_verified",
    "final_verified",
)


PRESETS = {
    "s3dis": {
        "checkpoint_dir": "ckpt/S3DIS/learnable_sp_structureonly_from1250_e20",
        "reference_checkpoint_dir": "ckpt/S3DIS/ckpts",
        "base_epoch": 1270,
        "reference_epochs": "1170,1180,1190",
        "data_path": "data/S3DIS/input",
        "sp_path": "data/S3DIS/initial_superpoints/",
        "input_dim": 6,
        "primitive_num": 300,
        "semantic_class": 12,
        "feats_dim": 128,
        "ignore_label": 12,
        "voxel_size": 0.05,
        "refiner_checkpoint": "ckpt/S3DIS/refiner_projectloss02_e10/refiner_best_checkpoint.pth",
        "structure_checkpoint": "ckpt/S3DIS/learnable_sp_structureonly_from1250_e20/learnable_sp_1270_checkpoint.pth",
    },
    "scannet": {
        "checkpoint_dir": "ckpt/ScanNet/baseline",
        "reference_checkpoint_dir": "ckpt/ScanNet/baseline",
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
        "refiner_checkpoint": "",
        "structure_checkpoint": "",
    },
    "semantickitti": {
        "checkpoint_dir": "ckpt/SemanticKITTI/baseline/ckpts",
        "reference_checkpoint_dir": "ckpt/SemanticKITTI/baseline/ckpts",
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
        "refiner_checkpoint": "",
        "structure_checkpoint": "",
    },
    "logosp_s3dis": {
        "checkpoint_dir": "/home/magic/magic/cm/repositories/LogoSP/ckpt/S3DIS/seg",
        "reference_checkpoint_dir": "/home/magic/magic/cm/repositories/LogoSP/ckpt/S3DIS/seg",
        "base_epoch": 100,
        "reference_epochs": "70,80,90",
        "data_path": "/home/magic/magic/cm/repositories/LogoSP/data/S3DIS/input_0.010",
        "sp_path": "/home/magic/magic/cm/repositories/LogoSP/data/S3DIS/initial_superpoints/",
        "input_dim": 3,
        "primitive_num": 12,
        "semantic_class": 12,
        "feats_dim": 384,
        "ignore_label": 12,
        "voxel_size": 0.05,
        "refiner_checkpoint": "",
        "structure_checkpoint": "",
    },
}


class ZeroResidualRefiner(nn.Module):
    """Conservative start for backbone transfer without Refiner weights."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(
        self,
        point_features,
        point_coordinates,
        batch_ids,
        query_indices,
        regions=None,
        use_region_branch=False,
        return_components=False,
    ):
        del point_coordinates, batch_ids, query_indices, regions, use_region_branch
        zeros = point_features.new_zeros((point_features.size(0), self.num_classes))
        if return_components:
            return {
                "point": zeros,
                "context": zeros.clone(),
                "region": zeros.clone(),
                "total": zeros.clone(),
            }
        return zeros


def parse_args():
    parser = argparse.ArgumentParser(
        "Cross-dataset Semantic Difference Split + Meta-Refiner + Verify"
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(PRESETS),
        required=True,
    )
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--reference_checkpoint_dir", default=None)
    parser.add_argument("--base_epoch", type=int, default=None)
    parser.add_argument("--reference_epochs", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--sp_path", default=None)
    parser.add_argument("--refiner_checkpoint", default=None)
    parser.add_argument("--structure_checkpoint", default=None)
    parser.add_argument("--test_area", default="Area_5")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--scene_output_dir", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--scene_stride", type=int, default=1)
    parser.add_argument("--prob_scale", type=float, default=10.0)
    parser.add_argument("--refiner_scale", type=float, default=1.0)
    parser.add_argument("--min_temporal_votes", type=int, default=2)
    parser.add_argument("--meta_target_confidence", type=float, default=0.80)
    parser.add_argument("--meta_verify_confidence", type=float, default=0.64)
    parser.add_argument("--result_verify_confidence", type=float, default=0.80)
    parser.add_argument("--enable_temporal_override", action="store_true")
    parser.add_argument("--meta_inner_steps", type=int, default=5)
    parser.add_argument("--meta_inner_lr", type=float, default=0.1)
    parser.add_argument("--meta_keep_weight", type=float, default=5.0)
    parser.add_argument("--meta_scale_reg", type=float, default=0.1)
    parser.add_argument("--meta_bias_reg", type=float, default=1.0)
    parser.add_argument("--disable_meta_bias", action="store_true")
    parser.add_argument("--refiner_hidden_dim", type=int, default=128)
    parser.add_argument("--refiner_num_heads", type=int, default=4)
    parser.add_argument("--bn_momentum", type=float, default=0.02)
    parser.add_argument("--conv1_kernel_size", type=int, default=5)
    parser.add_argument("--structure_hidden_dim", type=int, default=64)
    parser.add_argument("--structure_iterations", type=int, default=3)
    parser.add_argument("--structure_temperature", type=float, default=0.2)
    return parser.parse_args()


def apply_preset(args):
    preset = PRESETS[args.dataset]
    for name, value in preset.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.input_dim = preset["input_dim"]
    args.primitive_num = preset["primitive_num"]
    args.semantic_class = preset["semantic_class"]
    args.feats_dim = preset["feats_dim"]
    args.ignore_label = preset["ignore_label"]
    args.voxel_size = preset["voxel_size"]
    return args


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def semantic_centers(classifier, semantic_classes):
    primitive_centers = classifier.weight.detach()
    if primitive_centers.size(0) == semantic_classes:
        return F.normalize(primitive_centers, dim=1)
    assignment = KMeans(
        n_clusters=semantic_classes,
        n_init=10,
        random_state=0,
        n_jobs=10,
    ).fit_predict(primitive_centers.cpu().numpy())
    centers = primitive_centers.new_zeros((semantic_classes, primitive_centers.size(1)))
    for class_id in range(semantic_classes):
        mask = torch.as_tensor(assignment == class_id, device=primitive_centers.device)
        centers[class_id] = primitive_centers[mask].mean(dim=0)
    return F.normalize(centers, dim=1)


def align_centers(source, reference):
    similarity = torch.mm(F.normalize(source, dim=1), F.normalize(reference, dim=1).t())
    match = linear_assignment(similarity.detach().cpu().numpy().max() - similarity.detach().cpu().numpy())
    aligned = source.new_zeros(source.shape)
    for source_id, reference_id in match:
        aligned[int(reference_id)] = source[int(source_id)]
    return F.normalize(aligned, dim=1)


def logo_model_class():
    package_name = "cross_eval_logosp_models"
    if package_name not in sys.modules:
        package_path = "/home/magic/magic/cm/repositories/LogoSP/models"
        spec = importlib.util.spec_from_file_location(
            package_name,
            os.path.join(package_path, "__init__.py"),
            submodule_search_locations=[package_path],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.fpn").Res16FPN18


def load_reference(args, checkpoint_dir, epoch, reference_centers=None):
    model_class = logo_model_class() if args.dataset == "logosp_s3dis" else Res16FPN18
    model = model_class(
        in_channels=args.input_dim,
        out_channels=args.primitive_num,
        conv1_kernel_size=args.conv1_kernel_size,
        config=args,
    ).cuda()
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, f"model_{epoch}_checkpoint.pth"),
            map_location="cpu",
        )
    )
    model.eval()
    classifier = nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    classifier.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, f"cls_{epoch}_checkpoint.pth"),
            map_location="cpu",
        )
    )
    classifier.eval()
    centers = semantic_centers(classifier, args.semantic_class)
    if reference_centers is not None:
        centers = align_centers(centers, reference_centers)
    return model, centers


def build_dataset(args):
    if args.dataset in ("s3dis", "logosp_s3dis"):
        areas = [item.strip() for item in args.test_area.split(",") if item.strip()]
        dataset = S3DIStest(args, areas=areas)
        collate = cfl_collate_fn_test()
    elif args.dataset == "scannet":
        dataset = Scannetval(args)
        collate = scannet_collate()
    else:
        dataset = KITTIval(args)
        collate = kitti_collate()
    stride = max(int(args.scene_stride), 1)
    indices = list(range(0, len(dataset), stride))
    if args.max_scenes > 0:
        indices = indices[: args.max_scenes]
    selected = Subset(dataset, indices)
    return dataset, selected, collate


def model_input(args, coords, features):
    if args.dataset == "semantickitti":
        return coords[:, 1:].float() * float(args.voxel_size)
    if args.dataset == "logosp_s3dis":
        return features[:, :3]
    return features


def pipeline_colors(args, coords, features):
    if features.size(1) >= 3:
        return features[:, :3]
    if args.dataset == "semantickitti":
        intensity = features[:, :1]
        return intensity.repeat(1, 3)
    return coords[:, 1:].float()


def load_refiner(args):
    if not args.refiner_checkpoint:
        return ZeroResidualRefiner(args.semantic_class).cuda().eval(), "episodic-zero-residual"
    refiner = ErrorQueryRefiner(
        feat_dim=args.feats_dim,
        num_classes=args.semantic_class,
        hidden_dim=args.refiner_hidden_dim,
        num_heads=args.refiner_num_heads,
        dropout=0.0,
    ).cuda()
    refiner.load_state_dict(torch.load(args.refiner_checkpoint, map_location="cpu"), strict=False)
    return refiner.eval(), "trained-query-refiner"


def load_structure(args):
    if not args.structure_checkpoint:
        return None
    learner = SemanticDifferenceSuperpointLearner(
        feat_dim=args.feats_dim,
        num_classes=args.semantic_class,
        hidden_dim=args.structure_hidden_dim,
        iterations=args.structure_iterations,
        temperature=args.structure_temperature,
    ).cuda()
    learner.load_state_dict(torch.load(args.structure_checkpoint, map_location="cpu"))
    return learner.eval()


def build_config(args):
    return SemanticDifferencePipelineConfig(
        semantic_classes=args.semantic_class,
        probability_scale=args.prob_scale,
        refiner_scale=args.refiner_scale,
        meta_inner_steps=args.meta_inner_steps,
        meta_inner_lr=args.meta_inner_lr,
        meta_keep_weight=args.meta_keep_weight,
        meta_scale_regularization=args.meta_scale_reg,
        meta_bias_regularization=args.meta_bias_reg,
        meta_adapt_bias=not args.disable_meta_bias,
        meta_target_confidence=args.meta_target_confidence,
        meta_verify_confidence=args.meta_verify_confidence,
        result_verify_confidence=args.result_verify_confidence,
        min_temporal_votes=args.min_temporal_votes,
        enable_temporal_override=args.enable_temporal_override,
    )


def stage_metrics(predictions, labels, semantic_classes):
    labels = torch.cat(labels).numpy()
    arrays = {name: torch.cat(values).numpy() for name, values in predictions.items()}
    results = {}
    previous = None
    for name in STAGE_ORDER:
        metrics = compute_unsupervised_metrics(arrays[name], labels, semantic_classes)
        results[name] = {
            "oAcc": float(metrics[0]),
            "mAcc": float(metrics[1]),
            "mIoU": float(metrics[3]),
            "delta_mIoU_from_base": 0.0 if name == "base" else float(metrics[3] - results["base"]["mIoU"]),
            "delta_mIoU_from_previous": 0.0 if previous is None else float(metrics[3] - results[previous]["mIoU"]),
            "changed_from_base": float((arrays[name] != arrays["base"]).mean()),
        }
        previous = name
    return results


def normalized_statistics(sums):
    points = max(sums["points"], 1.0)
    scenes = max(sums["scenes"], 1.0)
    return {
        "scenes": int(sums["scenes"]),
        "voxel_points": int(sums["points"]),
        "queries_per_scene": sums["queries"] / scenes,
        "split_queries_per_scene": sums["split_queries"] / scenes,
        "refine_point_ratio": sums["refine_points"] / points,
        "split_point_ratio": sums["split_points"] / points,
        "learned_structure_candidate_per_scene": sums["structure_candidates"] / scenes,
        "learned_structure_accept_per_scene": sums["structure_accepted"] / scenes,
        "learned_structure_point_ratio": sums["structure_points"] / points,
        "meta_scene_accept_ratio": sums["meta_scene_accepted"] / scenes,
        "meta_candidate_accept_ratio": sums["meta_accepted"] / points,
        "meta_candidate_rollback_ratio": sums["meta_rollback"] / points,
        "temporal_override_ratio": sums["temporal_override"] / points,
        "final_changed_from_base_ratio": sums["final_changed_from_base"] / points,
        "mean_scene_seconds": sums["scene_seconds"] / scenes,
        "peak_gpu_memory_mb": sums["peak_gpu_memory_bytes"] / (1024.0 ** 2),
    }


def save_scene_output(args, dataset, source_index, batch, output, dynamic_regions):
    if not args.scene_output_dir:
        return
    os.makedirs(args.scene_output_dir, exist_ok=True)
    coords, features, inverse_map, labels, _index, original_regions = batch
    inverse = inverse_map.long().cuda()
    if hasattr(dataset, "name"):
        scene_name = str(dataset.name[source_index]).lstrip("/").replace("/", "_")
    else:
        scene_name = f"scene_{source_index:05d}"
    np.savez_compressed(
        os.path.join(args.scene_output_dir, f"{scene_name}.npz"),
        voxel_xyz=(coords[:, 1:].numpy() * float(args.voxel_size)),
        voxel_rgb=pipeline_colors(args, coords, features).numpy(),
        original_labels=labels.numpy(),
        inverse_map=inverse_map.numpy(),
        initial_regions=original_regions.squeeze().numpy(),
        dynamic_regions=dynamic_regions.detach().cpu().numpy(),
        base=output.base_prediction.detach().cpu().numpy(),
        decomposition=output.decomposition.no_op_prediction.detach().cpu().numpy(),
        refiner=output.decomposition.refined_prediction.detach().cpu().numpy(),
        meta=output.meta_refiner.prediction.detach().cpu().numpy(),
        final=output.verifier.final_prediction.detach().cpu().numpy(),
        split_targets=output.decomposition.split_targets.detach().cpu().numpy(),
        decision=output.verifier.decision.detach().cpu().numpy(),
    )


def main():
    args = apply_preset(parse_args())
    config = build_config(args)
    base_model, base_centers = load_reference(
        args, args.checkpoint_dir, args.base_epoch
    )
    temporal_references = []
    for epoch in parse_int_list(args.reference_epochs):
        model, centers = load_reference(
            args,
            args.reference_checkpoint_dir,
            epoch,
            reference_centers=base_centers,
        )
        temporal_references.append((epoch, model, centers))
    refiner, refiner_source = load_refiner(args)
    structure = load_structure(args)
    raw_dataset, selected_dataset, collate = build_dataset(args)
    loader = DataLoader(
        selected_dataset,
        batch_size=1,
        collate_fn=collate,
        num_workers=args.workers,
        pin_memory=True,
    )

    predictions = {name: [] for name in STAGE_ORDER}
    labels_all = []
    sums = defaultdict(float)
    torch.cuda.reset_peak_memory_stats()
    selected_indices = selected_dataset.indices
    total_scenes = len(selected_indices)
    evaluation_started = time.time()
    for local_scene_id, batch in enumerate(loader):
        started = time.time()
        coords, features, inverse_map, labels, _index, regions = batch
        field = ME.TensorField(model_input(args, coords, features), coords, device=0)
        with torch.no_grad():
            base_features = F.normalize(base_model(field), dim=1)
            base_scores = F.linear(base_features, base_centers)
            temporal_probabilities = []
            for _epoch, model, centers in temporal_references:
                reference_features = F.normalize(model(field), dim=1)
                temporal_probabilities.append(
                    F.softmax(
                        F.linear(reference_features, centers) * float(args.prob_scale),
                        dim=1,
                    )
                )
            temporal_probabilities = torch.stack(temporal_probabilities, dim=0)

        dynamic_regions = regions.squeeze().long().cuda()
        if structure is not None:
            with torch.no_grad():
                structure_output = structure(
                    base_features,
                    coords[:, 1:].float().cuda(),
                    pipeline_colors(args, coords, features).float().cuda(),
                    base_scores * float(args.prob_scale),
                    dynamic_regions,
                    coords[:, 0].long().cuda(),
                    min_region_points=20,
                    min_child_points=6,
                    max_regions_per_scene=4,
                    purity_threshold=0.8,
                    entropy_threshold=0.4,
                    min_child_confidence=0.5,
                    min_confidence_gain=0.05,
                    min_semantic_separation=0.5,
                )
            dynamic_regions = structure_output.dynamic_regions
            sums["structure_candidates"] += structure_output.stats["selected_regions"]
            sums["structure_accepted"] += structure_output.stats["accepted_splits"]
            sums["structure_points"] += (
                structure_output.stats["supervised_ratio"] * dynamic_regions.numel()
            )

        output = run_semantic_difference_pipeline(
            config,
            refiner,
            base_scores,
            base_features,
            coords,
            pipeline_colors(args, coords, features),
            dynamic_regions,
            temporal_probabilities,
        )
        inverse = inverse_map.long().cuda()
        valid = labels != args.ignore_label
        for name, prediction in output.stage_predictions().items():
            predictions[name].append(prediction[inverse].cpu()[valid])
        labels_all.append(labels[valid])

        sums["scenes"] += 1
        for name, value in output.decomposition.statistics.items():
            if isinstance(value, (int, float)):
                sums[name] += float(value)
        sums["meta_scene_accepted"] += int(output.meta_refiner.scene_accepted)
        for name in (
            "meta_accepted",
            "meta_rollback",
            "temporal_override",
            "final_changed_from_base",
        ):
            sums[name] += float(output.verifier.statistics[name])
        sums["scene_seconds"] += time.time() - started
        save_scene_output(
            args,
            raw_dataset,
            selected_indices[local_scene_id],
            batch,
            output,
            dynamic_regions,
        )
        completed = local_scene_id + 1
        if completed % 20 == 0 or completed == total_scenes:
            elapsed = time.time() - evaluation_started
            mean_seconds = elapsed / completed
            remaining_minutes = mean_seconds * (total_scenes - completed) / 60.0
            print(
                f"[{args.dataset}] {completed}/{total_scenes} scenes, "
                f"{mean_seconds:.2f}s/scene, ETA {remaining_minutes:.1f} min",
                flush=True,
            )

    sums["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    results = stage_metrics(predictions, labels_all, args.semantic_class)
    statistics = normalized_statistics(sums)
    payload = {
        "dataset": args.dataset,
        "method": "Semantic Difference Split + Episodic Meta-Refiner + Full Residual Verify",
        "base_checkpoint": os.path.join(args.checkpoint_dir, f"model_{args.base_epoch}_checkpoint.pth"),
        "reference_checkpoint_dir": args.reference_checkpoint_dir,
        "reference_epochs": parse_int_list(args.reference_epochs),
        "refiner_source": refiner_source,
        "structure_checkpoint": args.structure_checkpoint,
        "evaluation_scope": {
            "test_area": (
                args.test_area
                if args.dataset in ("s3dis", "logosp_s3dis")
                else None
            ),
            "scene_stride": int(args.scene_stride),
            "max_scenes": int(args.max_scenes),
            "evaluated_scenes": int(statistics["scenes"]),
        },
        "label_usage": "Ground truth is read only after all predictions for metric computation.",
        "pipeline_config": asdict(config),
        "results": results,
        "statistics": statistics,
    }
    output_directory = os.path.dirname(args.output_json)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
