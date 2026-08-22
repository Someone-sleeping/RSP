import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import DataLoader, Subset

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from datasets.ScanNet import Scannetval, cfl_collate_fn_val as scannet_collate
from datasets.SemanticKITTI import KITTIval, cfl_collate_fn_val as kitti_collate
from eval_S3DIS import compute_unsupervised_metrics
from lib.correction_strategies import (
    error_type_conditioned_refinement,
    hierarchical_split_merge,
    local_consensus_verifier,
    region_risk_rollback,
    regionwise_refiner_scale_selection,
)
from lib.meta_optimizer import meta_optimize_poe_weight
from lib.meta_refiner import meta_adapt_refiner_gates
from lib.split_regions import build_region_consistency_queries, build_split_region_queries
from models.fpn import Res16FPN18
from models.query_refiner import (
    ErrorQueryRefiner,
    gate_refiner_residual,
    resolve_min_temporal_votes,
)


KNOWN_INVALID_BASE_CHECKPOINTS = {
    "f910f3295f6773965ca5d671791368f551137a9380e0ad895bf8c72c2fb0351a": (
        "The supplied ScanNet epoch-930 run reproduces 3.54 mIoU instead of "
        "the reported 25.4 +/- 2.3 and must not be used as a formal baseline."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser("Evaluate label-free temporal/region correction verification")
    parser.add_argument(
        "--dataset",
        choices=("s3dis", "scannet", "semantickitti", "logosp_s3dis"),
        default="s3dis",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/magic/magic/cm/repositories/GrowSP/ckpt/S3DIS/1baseline/ckpts",
    )
    parser.add_argument("--base_epoch", type=int, default=1270)
    parser.add_argument("--reference_epochs", default="1170,1250,1260")
    parser.add_argument("--test_area", default="Area_5")
    parser.add_argument("--data_path", default="data/S3DIS/input")
    parser.add_argument("--sp_path", default="data/S3DIS/initial_superpoints/")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--primitive_num", type=int, default=300)
    parser.add_argument("--semantic_class", type=int, default=12)
    parser.add_argument("--feats_dim", type=int, default=128)
    parser.add_argument("--ignore_label", type=int, default=12)
    parser.add_argument("--bn_momentum", type=float, default=0.02)
    parser.add_argument("--conv1_kernel_size", type=int, default=5)
    parser.add_argument("--prob_scale", type=float, default=10.0)
    parser.add_argument("--thresholds", default="0.60,0.64")
    parser.add_argument("--selection_threshold", type=float, default=0.64)
    parser.add_argument(
        "--min_temporal_votes",
        type=int,
        default=0,
        help="Required checkpoint votes; zero selects a majority from available references.",
    )
    parser.add_argument("--refiner_checkpoint", default="")
    parser.add_argument("--refiner_scale", type=float, default=1.0)
    parser.add_argument("--refiner_scales", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--refiner_hidden_dim", type=int, default=128)
    parser.add_argument("--refiner_num_heads", type=int, default=4)
    parser.add_argument("--blend_weights", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--meta_optimize", action="store_true", default=False)
    parser.add_argument("--meta_initial_weight", type=float, default=0.1)
    parser.add_argument("--meta_inner_steps", type=int, default=5)
    parser.add_argument("--meta_inner_lr", type=float, default=0.5)
    parser.add_argument("--meta_correction_weight", type=float, default=1.0)
    parser.add_argument("--meta_keep_weight", type=float, default=1.0)
    parser.add_argument("--meta_entropy_weight", type=float, default=0.01)
    parser.add_argument("--meta_query_tolerance", type=float, default=0.0)
    parser.add_argument("--meta_classwise", action="store_true", default=False)
    parser.add_argument("--meta_refiner", action="store_true", default=False)
    parser.add_argument("--meta_refiner_classwise", action="store_true", default=False)
    parser.add_argument("--meta_refiner_initial_scale", type=float, default=1.0)
    parser.add_argument("--meta_refiner_min_scale", type=float, default=0.25)
    parser.add_argument("--meta_refiner_max_scale", type=float, default=1.75)
    parser.add_argument("--meta_refiner_inner_steps", type=int, default=5)
    parser.add_argument("--meta_refiner_inner_lr", type=float, default=0.1)
    parser.add_argument("--meta_refiner_correction_weight", type=float, default=1.0)
    parser.add_argument("--meta_refiner_keep_weight", type=float, default=5.0)
    parser.add_argument("--meta_refiner_entropy_weight", type=float, default=0.01)
    parser.add_argument("--meta_refiner_scale_reg", type=float, default=0.1)
    parser.add_argument("--meta_refiner_query_tolerance", type=float, default=0.0)
    parser.add_argument("--meta_refiner_adapt_bias", action="store_true", default=False)
    parser.add_argument("--meta_refiner_bias_reg", type=float, default=0.1)
    parser.add_argument("--meta_refiner_verify_threshold", type=float, default=0.64)
    parser.add_argument("--four_stage_diagnostic", action="store_true", default=False)
    parser.add_argument("--diagnostic_multi_proposal", action="store_true", default=False)
    parser.add_argument("--skip_region_oracle", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scene_stride", type=int, default=1)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--scene_output_dir", default="")
    parser.add_argument("--allow_invalid_checkpoint", action="store_true", default=False)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_areas(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_refiner_binding(args):
    if not args.refiner_checkpoint:
        return {"verified": False, "reason": "No Refiner checkpoint supplied."}
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "docs",
        "four_stage_checkpoint_bindings.json",
    )
    manifest_binding = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest_binding = json.load(manifest_file).get(args.dataset, {})
    metadata_path = os.path.join(
        os.path.dirname(args.refiner_checkpoint),
        "training_metadata.json",
    )
    training_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            training_metadata = json.load(metadata_file)
    metadata = {**manifest_binding, **training_metadata}
    if not metadata:
        return {
            "verified": False,
            "reason": f"No binding found in {metadata_path} or {manifest_path}.",
        }
    metadata_dataset = training_metadata.get("dataset", args.dataset)
    if metadata_dataset != args.dataset:
        raise ValueError(
            f"Refiner dataset mismatch: {metadata_dataset} != {args.dataset}"
        )
    if int(metadata.get("base_epoch", -1)) != int(args.base_epoch):
        raise ValueError(
            f"Refiner base epoch mismatch: {metadata.get('base_epoch')} != {args.base_epoch}"
        )
    checkpoint_specs = (
        (
            "backbone",
            os.path.join(args.checkpoint_dir, f"model_{args.base_epoch}_checkpoint.pth"),
            "base_model_sha256",
        ),
        (
            "classifier",
            os.path.join(args.checkpoint_dir, f"cls_{args.base_epoch}_checkpoint.pth"),
            "base_classifier_sha256",
        ),
        ("Refiner", args.refiner_checkpoint, "refiner_sha256"),
    )
    actual_hashes = {}
    for component, checkpoint_path, metadata_key in checkpoint_specs:
        actual_hash = checkpoint_sha256(checkpoint_path)
        if (
            component == "backbone"
            and actual_hash in KNOWN_INVALID_BASE_CHECKPOINTS
            and not getattr(args, "allow_invalid_checkpoint", False)
        ):
            raise ValueError(KNOWN_INVALID_BASE_CHECKPOINTS[actual_hash])
        expected_hash = metadata.get(metadata_key)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{component} checkpoint mismatch: {actual_hash} != {expected_hash}"
            )
        actual_hashes[metadata_key] = actual_hash
    return {
        "verified": True,
        "training_metadata": metadata_path if training_metadata else None,
        "binding_manifest": manifest_path if manifest_binding else None,
        **actual_hashes,
    }


def semantic_centers(primitive_classifier, semantic_class):
    primitive_centers = primitive_classifier.weight.detach()
    cluster_pred = KMeans(
        n_clusters=semantic_class,
        n_init=10,
        random_state=0,
        n_jobs=10,
    ).fit_predict(primitive_centers.cpu().numpy())
    centers = primitive_centers.new_zeros((semantic_class, primitive_centers.size(1)))
    for semantic_id in range(semantic_class):
        mask = torch.as_tensor(cluster_pred == semantic_id, device=primitive_centers.device)
        centers[semantic_id] = primitive_centers[mask].mean(dim=0)
    return F.normalize(centers, dim=1)


def align_centers(source_centers, reference_centers):
    similarity = torch.mm(
        F.normalize(source_centers, dim=1),
        F.normalize(reference_centers, dim=1).t(),
    ).detach().cpu().numpy()
    match = linear_assignment(similarity.max() - similarity)
    aligned = source_centers.new_zeros(source_centers.shape)
    for source_id, reference_id in match:
        aligned[int(reference_id)] = source_centers[int(source_id)]
    return F.normalize(aligned, dim=1)


def logo_model_class():
    package_name = "four_stage_logosp_models"
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


def load_reference(args, epoch, reference_centers=None):
    model_class = logo_model_class() if args.dataset == "logosp_s3dis" else Res16FPN18
    model = model_class(
        in_channels=args.input_dim,
        out_channels=args.primitive_num,
        conv1_kernel_size=args.conv1_kernel_size,
        config=args,
    ).cuda()
    model_path = os.path.join(args.checkpoint_dir, f"model_{epoch}_checkpoint.pth")
    classifier_path = os.path.join(args.checkpoint_dir, f"cls_{epoch}_checkpoint.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    classifier = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    classifier.eval()
    centers = semantic_centers(classifier, args.semantic_class)
    if reference_centers is not None:
        centers = align_centers(centers, reference_centers)
    return model, centers


def build_eval_dataset(args):
    if args.dataset in ("s3dis", "logosp_s3dis"):
        dataset = S3DIStest(args, areas=parse_areas(args.test_area))
        collate = cfl_collate_fn_test()
    elif args.dataset == "scannet":
        dataset = Scannetval(args)
        collate = scannet_collate()
    else:
        dataset = KITTIval(args)
        collate = kitti_collate()
    indices = list(range(0, len(dataset), max(int(args.scene_stride), 1)))
    if args.max_scenes > 0:
        indices = indices[: args.max_scenes]
    return Subset(dataset, indices), collate


def model_input(args, coords, features):
    if args.dataset == "semantickitti":
        return coords[:, 1:].float() * float(args.voxel_size)
    if args.dataset == "logosp_s3dis":
        return features[:, :3]
    return features


def split_colors(args, coords, features):
    if args.dataset == "semantickitti":
        if features.size(1) == 1:
            return features.repeat(1, 3)
        return coords[:, 1:].float() * float(args.voxel_size)
    return features[:, :3]


def save_scene_predictions(
    args,
    eval_dataset,
    local_index,
    batch,
    scene_predictions,
    split_targets,
):
    if not args.scene_output_dir:
        return
    source_index = eval_dataset.indices[local_index]
    source_dataset = eval_dataset.dataset
    if hasattr(source_dataset, "name"):
        scene_name = str(source_dataset.name[source_index]).lstrip("/").replace("/", "_")
    else:
        scene_name = f"scene_{source_index:05d}"
    coords, features, inverse_map, labels, _index, regions = batch
    final_name = f"meta_adapt_override_t{int(round(100 * args.selection_threshold))}"
    final_prediction = scene_predictions.get(final_name, scene_predictions["base"])
    decision = torch.zeros_like(final_prediction)
    decision[final_prediction != scene_predictions["base"]] = 1
    os.makedirs(args.scene_output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.scene_output_dir, f"{scene_name}.npz"),
        voxel_xyz=coords[:, 1:].numpy() * float(args.voxel_size),
        voxel_rgb=split_colors(args, coords, features).numpy(),
        original_labels=labels.numpy(),
        inverse_map=inverse_map.numpy(),
        initial_regions=regions.squeeze().numpy(),
        dynamic_regions=regions.squeeze().numpy(),
        base=scene_predictions["base"].detach().cpu().numpy(),
        decomposition=scene_predictions["split_noop"].detach().cpu().numpy(),
        refiner=scene_predictions["split_refiner"].detach().cpu().numpy(),
        meta=final_prediction.detach().cpu().numpy(),
        final=final_prediction.detach().cpu().numpy(),
        split_targets=split_targets.detach().cpu().numpy(),
        decision=decision.detach().cpu().numpy(),
    )


def region_candidates(base_scores, regions):
    raw_probs = F.softmax(base_scores, dim=1)
    point_conf = raw_probs.max(dim=1)[0]
    projected = base_scores.argmax(dim=1).clone()
    for region_id in torch.unique(regions):
        if int(region_id.item()) == -1:
            continue
        mask = regions == region_id
        weights = point_conf[mask].clamp_min(1e-6)
        region_prob = (raw_probs[mask] * weights[:, None]).sum(dim=0) / weights.sum()
        projected[mask] = region_prob.argmax()
    return projected


def temporal_region_support(probabilities, regions):
    num_references, num_points, num_classes = probabilities.shape
    region_target = torch.zeros(num_points, dtype=torch.long, device=probabilities.device)
    region_conf = probabilities.new_zeros(num_points)
    region_votes = torch.zeros(num_points, dtype=torch.long, device=probabilities.device)
    for region_id in torch.unique(regions):
        mask = regions == region_id
        if int(region_id.item()) == -1:
            continue
        mean_prob = probabilities[:, mask].mean(dim=(0, 1))
        confidence, target = mean_prob.max(dim=0)
        per_reference_target = []
        for reference_id in range(num_references):
            ref_prob = probabilities[reference_id, mask].mean(dim=0)
            per_reference_target.append(ref_prob.argmax())
        per_reference_target = torch.stack(per_reference_target)
        support = (per_reference_target == target).sum()
        region_target[mask] = target
        region_conf[mask] = confidence
        region_votes[mask] = support
    return region_target, region_conf, region_votes


def project_region_and_split(scores, regions, split_targets):
    projected = scores.clone()
    probabilities = F.softmax(projected, dim=1)
    point_conf = probabilities.max(dim=1)[0]
    for region_id in torch.unique(regions):
        if int(region_id.item()) == -1:
            continue
        mask = regions == region_id
        weights = point_conf[mask].clamp_min(1e-6)
        region_prob = (probabilities[mask] * weights[:, None]).sum(dim=0, keepdim=True) / weights.sum()
        projected[mask] = torch.log(region_prob.clamp_min(1e-6))
    valid_split = split_targets >= 0
    if valid_split.any():
        projected[valid_split] = projected.new_full(
            (int(valid_split.sum().item()), projected.size(1)),
            -20.0,
        )
        projected[valid_split, split_targets[valid_split].long()] = 20.0
    return projected


def run_split_refiner(
    args,
    refiner,
    base_scores,
    base_feats,
    coords,
    features,
    regions,
    multi_proposal=False,
    delta_override=None,
    return_components=False,
):
    batch_ids = coords[:, 0].long().cuda()
    point_coords = coords[:, 1:].float().cuda()
    point_colors = split_colors(args, coords, features).float().cuda()
    (
        query_indices,
        refine_mask,
        split_targets,
        _split_conf,
        keep_mask,
        _split_stats,
    ) = build_split_region_queries(
        base_scores,
        base_feats,
        point_coords,
        point_colors,
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
        multi_proposal=multi_proposal,
    )
    (
        consistency_queries,
        consistency_mask,
        _consistency_targets,
        _consistency_conf,
        consistency_keep,
        _consistency_stats,
    ) = build_region_consistency_queries(
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
    if consistency_queries.numel() > 0:
        query_indices = torch.unique(torch.cat([query_indices, consistency_queries], dim=0))
    refine_mask = refine_mask | consistency_mask
    keep_mask = keep_mask | consistency_keep
    if delta_override is None:
        with torch.no_grad():
            refiner_output = refiner(
                base_feats,
                point_coords,
                batch_ids,
                query_indices,
                regions,
                use_region_branch=False,
                return_components=return_components,
            )
            if return_components:
                delta_components = refiner_output
                delta_scores = refiner_output["total"]
            else:
                delta_components = None
                delta_scores = refiner_output
    else:
        delta_components = None
        delta_scores = delta_override
    # The Refiner is trained from candidate-region supervision. Applying its
    # residual outside that support turns a local correction into a global
    # classifier shift, especially when pseudo targets are sparse.
    delta_scores = gate_refiner_residual(delta_scores, refine_mask)
    if delta_components is not None:
        delta_components = {
            name: gate_refiner_residual(value, refine_mask)
            for name, value in delta_components.items()
        }
    no_op_scores = project_region_and_split(base_scores, regions, split_targets)
    refined_scores = project_region_and_split(
        base_scores + args.refiner_scale * delta_scores,
        regions,
        split_targets,
    )
    outputs = (no_op_scores, refined_scores, delta_scores, split_targets)
    if return_components:
        return outputs + (delta_components, refine_mask, keep_mask)
    return outputs


def prediction_mapping(predictions, labels, num_classes):
    valid = (labels >= 0) & (labels < num_classes)
    histogram = np.bincount(
        num_classes * labels[valid] + predictions[valid],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    match = linear_assignment(histogram.max() - histogram)
    pred_to_gt = np.full(num_classes, -1, dtype=np.int64)
    for gt_id, pred_id in match:
        pred_to_gt[int(pred_id)] = int(gt_id)
    return pred_to_gt


def oracle_diagnostic(base, labels, pred_to_gt, candidates, group_ids=None):
    """Label-using upper bound for diagnostics only; never used for selection."""
    oracle = base.copy()
    if group_ids is None:
        base_wrong = pred_to_gt[base] != labels
        for candidate in candidates:
            candidate_correct = pred_to_gt[candidate] == labels
            oracle[base_wrong & candidate_correct] = candidate[base_wrong & candidate_correct]
        return oracle

    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    for start, end in zip(starts, ends):
        indices = order[start:end]
        best_correct = int((pred_to_gt[base[indices]] == labels[indices]).sum())
        best_candidate = None
        for candidate in candidates:
            correct = int((pred_to_gt[candidate[indices]] == labels[indices]).sum())
            if correct > best_correct:
                best_correct = correct
                best_candidate = candidate
        if best_candidate is not None:
            oracle[indices] = best_candidate[indices]
    return oracle


def main():
    args = parse_args()
    evaluation_started = time.time()
    checkpoint_binding = verify_refiner_binding(args)
    reference_epochs = parse_int_list(args.reference_epochs)
    args.min_temporal_votes = resolve_min_temporal_votes(
        args.min_temporal_votes, len(reference_epochs)
    )
    thresholds = parse_float_list(args.thresholds)
    blend_weights = parse_float_list(args.blend_weights)
    refiner_scales = parse_float_list(args.refiner_scales)

    base_model, base_centers = load_reference(args, args.base_epoch)
    temporal_references = []
    for epoch in reference_epochs:
        model, centers = load_reference(args, epoch, reference_centers=base_centers)
        temporal_references.append((epoch, model, centers))

    refiner = None
    if args.refiner_checkpoint:
        refiner = ErrorQueryRefiner(
            feat_dim=args.feats_dim,
            num_classes=args.semantic_class,
            hidden_dim=args.refiner_hidden_dim,
            num_heads=args.refiner_num_heads,
            dropout=0.0,
        ).cuda()
        refiner.load_state_dict(torch.load(args.refiner_checkpoint, map_location="cpu"), strict=False)
        refiner.eval()

    eval_dataset, eval_collate = build_eval_dataset(args)
    loader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=eval_collate,
        num_workers=args.workers,
        pin_memory=True,
    )

    names = ["base", "region", "temporal_mean", "temporal_confweighted", "temporal_vote", "all_mean"]
    names.extend([f"epoch_{epoch}" for epoch in reference_epochs])
    if refiner is not None:
        names.extend(["split_noop", "split_refiner", "refiner_candidate_vote"])
        if args.meta_refiner:
            names.append("meta_refiner")
        if args.meta_optimize:
            names.append("meta_scene_poe")
        for blend_weight in blend_weights:
            blend_suffix = int(round(100 * blend_weight))
            names.extend(
                [
                    f"refiner_temporal_mix_w{blend_suffix}",
                    f"refiner_temporal_poe_w{blend_suffix}",
                ]
            )
    for threshold in thresholds:
        suffix = int(round(100 * threshold))
        names.extend(
            [
                f"point_verified_t{suffix}",
                f"confpoint_verified_t{suffix}",
                f"region_verified_t{suffix}",
                f"cross_verified_t{suffix}",
                f"point_region_union_t{suffix}",
                f"arbitrated_t{suffix}",
            ]
        )
        if refiner is not None:
            names.extend(
                [
                    f"refiner_temporal_override_t{suffix}",
                    f"meta_init_poe_override_t{suffix}",
                    f"meta_init_poe_region_t{suffix}",
                    f"refiner_override_region_t{suffix}",
                    f"refiner_compatible_union_t{suffix}",
                    f"residual_verified_t{suffix}",
                    f"refiner_verified_union_t{suffix}",
                ]
            )
            if args.meta_optimize:
                names.append(f"meta_adapt_override_t{suffix}")
            if args.meta_refiner:
                names.extend(
                    [
                        f"meta_refiner_override_t{suffix}",
                        f"meta_refiner_residual_verified_t{suffix}",
                        f"meta_refiner_verified_union_t{suffix}",
                        f"meta_refiner_poe_override_t{suffix}",
                        f"meta_refiner_candidate_gate_t{suffix}",
                        f"meta_refiner_verified_override_t{suffix}",
                        f"meta_refiner_verified_poe_override_t{suffix}",
                        f"meta_refiner_joint_structural_t{suffix}",
                        f"meta_refiner_joint_temporal_t{suffix}",
                    ]
                )
    diagnostic_names = []
    if args.four_stage_diagnostic and refiner is not None:
        if args.diagnostic_multi_proposal:
            diagnostic_names.extend(["split_multi_noop", "split_multi_refiner"])
        diagnostic_names.extend([f"refiner_scale_s{int(round(100 * scale))}" for scale in refiner_scales])
        diagnostic_names.extend([f"refiner_local_s{int(round(100 * scale))}" for scale in refiner_scales])
        diagnostic_names.extend([f"scale_meta_s{int(round(100 * scale))}" for scale in refiner_scales])
        diagnostic_names.extend(
            [f"scale_select_tw{int(round(10 * weight))}" for weight in (0.0, 0.5, 1.0, 2.0)]
        )
        for merge_ratio in (0.15, 0.25, 0.40):
            diagnostic_names.append(f"split_merge_r{int(round(100 * merge_ratio))}")
            for support in (1, 2):
                diagnostic_names.append(
                    f"typed_refine_r{int(round(100 * merge_ratio))}_s{support}"
                )
                for local_support in (3, 4, 5):
                    diagnostic_names.append(
                        f"local_verify_r{int(round(100 * merge_ratio))}_s{support}_v{local_support}"
                    )
        diagnostic_names.extend([f"meta_local_v{support}" for support in (3, 4, 5)])
        diagnostic_names.extend([f"meta_region_r{int(round(100 * value))}" for value in (0.4, 0.5, 0.6, 0.7)])
    names.extend(diagnostic_names)
    all_predictions = {name: [] for name in names}
    all_labels = []
    all_region_ids = []
    region_offset = 0
    proxy_sums = {"points": 0.0}
    meta_stats = {
        "scenes": 0,
        "accepted_scenes": 0,
        "selected_weight_sum": 0.0,
        "adapted_weight_sum": 0.0,
        "query_gain_sum": 0.0,
        "correction_ratio_sum": 0.0,
    }
    meta_refiner_stats = {
        "scenes": 0,
        "accepted_scenes": 0,
        "query_gain_sum": 0.0,
        "correction_ratio_sum": 0.0,
        "support_corrections": 0,
        "query_corrections": 0,
        "scale_sums": None,
        "bias_norm_sum": 0.0,
    }

    def add_proxy(name, value, weight):
        proxy_sums[name] = proxy_sums.get(name, 0.0) + float(value) * float(weight)

    for local_scene_index, batch in enumerate(loader):
        coords, features, inverse_map, labels, _index, region = batch
        in_field = ME.TensorField(model_input(args, coords, features), coords, device=0)
        with torch.no_grad():
            base_feats = F.normalize(base_model(in_field), dim=1)
            base_scores = F.linear(base_feats, base_centers)
            temporal_scores = []
            for _epoch, model, centers in temporal_references:
                feats = F.normalize(model(in_field), dim=1)
                temporal_scores.append(F.linear(feats, centers))

        base_pred = base_scores.argmax(dim=1)
        regions = region.squeeze().long().cuda()
        region_pred = region_candidates(base_scores, regions)
        temporal_probabilities = torch.stack(
            [F.softmax(scores * args.prob_scale, dim=1) for scores in temporal_scores],
            dim=0,
        )
        temporal_predictions = temporal_probabilities.argmax(dim=2)
        temporal_mean_prob = temporal_probabilities.mean(dim=0)
        temporal_conf, temporal_mean_pred = temporal_mean_prob.max(dim=1)
        reference_confidence = temporal_probabilities.max(dim=2)[0].pow(2).clamp_min(1e-6)
        temporal_confweighted_prob = (
            temporal_probabilities * reference_confidence.unsqueeze(2)
        ).sum(dim=0) / reference_confidence.sum(dim=0, keepdim=False).unsqueeze(1)
        confweighted_conf, temporal_confweighted_pred = temporal_confweighted_prob.max(dim=1)
        temporal_vote_count = F.one_hot(
            temporal_predictions,
            num_classes=args.semantic_class,
        ).sum(dim=0)
        temporal_votes, temporal_vote_pred = temporal_vote_count.max(dim=1)
        mean_vote_support = temporal_vote_count.gather(1, temporal_mean_pred.unsqueeze(1)).squeeze(1)
        temporal_vote_pred[mean_vote_support == temporal_votes] = temporal_mean_pred[
            mean_vote_support == temporal_votes
        ]

        base_probability = F.softmax(base_scores * args.prob_scale, dim=1)
        all_mean_pred = torch.cat(
            [base_probability.unsqueeze(0), temporal_probabilities],
            dim=0,
        ).mean(dim=0).argmax(dim=1)
        temporal_region_pred, temporal_region_conf, temporal_region_votes = temporal_region_support(
            temporal_probabilities,
            regions,
        )

        num_scene_points = int(base_pred.numel())
        proxy_sums["points"] += num_scene_points
        base_entropy = -(base_probability * torch.log(base_probability.clamp_min(1e-6))).sum(dim=1)
        base_region_pred = region_candidates(base_scores, regions)
        add_proxy("base_confidence", base_probability.max(dim=1)[0].mean().item(), num_scene_points)
        add_proxy("base_entropy", base_entropy.mean().item(), num_scene_points)
        add_proxy("base_region_disagreement", (base_pred != base_region_pred).float().mean().item(), num_scene_points)
        add_proxy("temporal_vote_strength", (temporal_votes.float() / len(reference_epochs)).mean().item(), num_scene_points)
        add_proxy("temporal_unanimity", (temporal_votes == len(reference_epochs)).float().mean().item(), num_scene_points)
        add_proxy("temporal_base_disagreement", (temporal_mean_pred != base_pred).float().mean().item(), num_scene_points)
        for reference_index, epoch in enumerate(reference_epochs):
            reference_probability = temporal_probabilities[reference_index]
            reference_prediction = temporal_predictions[reference_index]
            reference_entropy = -(
                reference_probability * torch.log(reference_probability.clamp_min(1e-6))
            ).sum(dim=1)
            reference_region = region_candidates(temporal_scores[reference_index], regions)
            add_proxy(
                f"epoch_{epoch}_confidence",
                reference_probability.max(dim=1)[0].mean().item(),
                num_scene_points,
            )
            add_proxy(f"epoch_{epoch}_entropy", reference_entropy.mean().item(), num_scene_points)
            add_proxy(
                f"epoch_{epoch}_base_agreement",
                (reference_prediction == base_pred).float().mean().item(),
                num_scene_points,
            )
            add_proxy(
                f"epoch_{epoch}_region_disagreement",
                (reference_prediction != reference_region).float().mean().item(),
                num_scene_points,
            )

        scene_predictions = {
            "base": base_pred,
            "region": region_pred,
            "temporal_mean": temporal_mean_pred,
            "temporal_confweighted": temporal_confweighted_pred,
            "temporal_vote": temporal_vote_pred,
            "all_mean": all_mean_pred,
        }
        for reference_index, epoch in enumerate(reference_epochs):
            scene_predictions[f"epoch_{epoch}"] = temporal_predictions[reference_index]

        no_op_pred = None
        refined_pred = None
        meta_adapted_pred = None
        meta_refiner_pred = None
        meta_refiner_probability = None
        diagnostic_scale_probabilities = []
        if refiner is not None:
            split_refiner_outputs = run_split_refiner(
                args,
                refiner,
                base_scores,
                base_feats,
                coords,
                features,
                regions,
                return_components=args.meta_refiner,
            )
            if args.meta_refiner:
                (
                    no_op_scores,
                    refined_scores,
                    delta_scores,
                    split_targets,
                    delta_components,
                    meta_refine_mask,
                    _meta_keep_mask,
                ) = split_refiner_outputs
            else:
                no_op_scores, refined_scores, delta_scores, split_targets = split_refiner_outputs
                delta_components = None
            no_op_pred = no_op_scores.argmax(dim=1)
            refined_pred = refined_scores.argmax(dim=1)
            candidate_votes = F.one_hot(
                torch.cat(
                    [
                        refined_pred.unsqueeze(0),
                        no_op_pred.unsqueeze(0),
                        region_pred.unsqueeze(0),
                        temporal_predictions,
                    ],
                    dim=0,
                ),
                num_classes=args.semantic_class,
            ).sum(dim=0)
            candidate_max_votes, candidate_vote_pred = candidate_votes.max(dim=1)
            for preferred_prediction in (base_pred, no_op_pred, refined_pred):
                preferred_votes = candidate_votes.gather(
                    1,
                    preferred_prediction.unsqueeze(1),
                ).squeeze(1)
                prefer_mask = preferred_votes == candidate_max_votes
                candidate_vote_pred[prefer_mask] = preferred_prediction[prefer_mask]
            scene_predictions["split_noop"] = no_op_pred
            scene_predictions["split_refiner"] = refined_pred
            scene_predictions["refiner_candidate_vote"] = candidate_vote_pred
            refined_probability = F.softmax(refined_scores, dim=1)
            if args.meta_refiner:
                (
                    meta_refiner_probability,
                    meta_refiner_accepted,
                    scene_meta_refiner_stats,
                ) = meta_adapt_refiner_gates(
                    base_scores.detach(),
                    {name: value.detach() for name, value in delta_components.items()},
                    refined_scores.detach(),
                    temporal_mean_prob.detach(),
                    temporal_vote_count.detach(),
                    region_pred.detach(),
                    no_op_pred.detach(),
                    base_pred.detach(),
                    regions.detach(),
                    split_targets.detach(),
                    residual_scale=args.refiner_scale,
                    initial_scale=args.meta_refiner_initial_scale,
                    min_scale=args.meta_refiner_min_scale,
                    max_scale=args.meta_refiner_max_scale,
                    confidence_threshold=args.selection_threshold,
                    min_votes=args.min_temporal_votes,
                    inner_steps=args.meta_refiner_inner_steps,
                    inner_lr=args.meta_refiner_inner_lr,
                    correction_weight=args.meta_refiner_correction_weight,
                    keep_weight=args.meta_refiner_keep_weight,
                    entropy_weight=args.meta_refiner_entropy_weight,
                    scale_regularization=args.meta_refiner_scale_reg,
                    query_tolerance=args.meta_refiner_query_tolerance,
                    classwise=args.meta_refiner_classwise,
                    refine_mask=meta_refine_mask.detach(),
                    adapt_bias=args.meta_refiner_adapt_bias,
                    bias_regularization=args.meta_refiner_bias_reg,
                )
                meta_refiner_pred = meta_refiner_probability.argmax(dim=1)
                scene_predictions["meta_refiner"] = meta_refiner_pred
                meta_refiner_stats["scenes"] += 1
                meta_refiner_stats["accepted_scenes"] += int(meta_refiner_accepted)
                meta_refiner_stats["query_gain_sum"] += float(
                    scene_meta_refiner_stats["query_gain"]
                )
                meta_refiner_stats["correction_ratio_sum"] += float(
                    scene_meta_refiner_stats["correction_ratio"]
                )
                meta_refiner_stats["support_corrections"] += int(
                    scene_meta_refiner_stats["support_corrections"]
                )
                meta_refiner_stats["query_corrections"] += int(
                    scene_meta_refiner_stats["query_corrections"]
                )
                scene_scales = scene_meta_refiner_stats["scales"]
                if meta_refiner_stats["scale_sums"] is None:
                    meta_refiner_stats["scale_sums"] = [0.0] * len(scene_scales)
                for scale_id, scale in enumerate(scene_scales):
                    meta_refiner_stats["scale_sums"][scale_id] += float(scale)
                meta_refiner_stats["bias_norm_sum"] += float(
                    scene_meta_refiner_stats["bias_norm"]
                )
            if args.four_stage_diagnostic:
                if args.diagnostic_multi_proposal:
                    multi_no_op_scores, multi_refined_scores, _multi_delta, _multi_split = run_split_refiner(
                        args,
                        refiner,
                        base_scores,
                        base_feats,
                        coords,
                        features,
                        regions,
                        multi_proposal=True,
                        delta_override=delta_scores,
                    )
                    scene_predictions["split_multi_noop"] = multi_no_op_scores.argmax(dim=1)
                    scene_predictions["split_multi_refiner"] = multi_refined_scores.argmax(dim=1)
                scale_probabilities = []
                for scale in refiner_scales:
                    scale_scores = project_region_and_split(
                        base_scores + scale * delta_scores,
                        regions,
                        split_targets,
                    )
                    scale_probability = F.softmax(scale_scores, dim=1)
                    scale_probabilities.append(scale_probability)
                    diagnostic_scale_probabilities.append((scale, scale_probability))
                    scene_predictions[f"refiner_scale_s{int(round(100 * scale))}"] = (
                        scale_probability.argmax(dim=1)
                    )
                    local_prediction = no_op_pred.clone()
                    non_split = split_targets < 0
                    local_prediction[non_split] = (
                        base_scores + scale * delta_scores
                    ).argmax(dim=1)[non_split]
                    scene_predictions[f"refiner_local_s{int(round(100 * scale))}"] = local_prediction
                for temporal_weight in (0.0, 0.5, 1.0, 2.0):
                    name = f"scale_select_tw{int(round(10 * temporal_weight))}"
                    scene_predictions[name] = regionwise_refiner_scale_selection(
                        scale_probabilities,
                        base_pred,
                        temporal_mean_prob,
                        temporal_vote_count,
                        regions,
                        temporal_weight=temporal_weight,
                    )
                no_op_probability = F.softmax(no_op_scores, dim=1)
                for merge_ratio in (0.15, 0.25, 0.40):
                    merge_suffix = int(round(100 * merge_ratio))
                    split_merge = hierarchical_split_merge(
                        no_op_pred,
                        split_targets,
                        region_pred,
                        temporal_probabilities,
                        temporal_mean_prob,
                        regions,
                        child_agreement_threshold=0.5,
                        child_confidence_threshold=0.5,
                        merge_max_child_ratio=merge_ratio,
                    )
                    scene_predictions[f"split_merge_r{merge_suffix}"] = split_merge
                    for support in (1, 2):
                        typed_name = f"typed_refine_r{merge_suffix}_s{support}"
                        typed_prediction = error_type_conditioned_refinement(
                            split_merge,
                            no_op_pred,
                            refined_pred,
                            refined_probability,
                            no_op_probability,
                            region_pred,
                            split_targets,
                            temporal_mean_prob,
                            temporal_vote_count,
                            min_temporal_votes=args.min_temporal_votes,
                            temporal_confidence_threshold=0.5,
                            min_independent_support=support,
                            confidence_gain=0.01,
                        )
                        scene_predictions[typed_name] = typed_prediction
                        for local_support in (3, 4, 5):
                            local_name = (
                                f"local_verify_r{merge_suffix}_s{support}_v{local_support}"
                            )
                            scene_predictions[local_name] = local_consensus_verifier(
                                typed_prediction,
                                base_pred,
                                region_pred,
                                no_op_pred,
                                temporal_predictions,
                                temporal_mean_prob,
                                min_support=local_support,
                                temporal_confidence_threshold=0.5,
                            )
            meta_initial_poe_score = (
                (1.0 - args.meta_initial_weight) * torch.log(refined_probability.clamp_min(1e-6))
                + args.meta_initial_weight * torch.log(temporal_mean_prob.clamp_min(1e-6))
            )
            meta_initial_poe_pred = meta_initial_poe_score.argmax(dim=1)
            for blend_weight in blend_weights:
                blend_suffix = int(round(100 * blend_weight))
                mixed_probability = (
                    (1.0 - blend_weight) * refined_probability
                    + blend_weight * temporal_mean_prob
                )
                poe_score = (
                    (1.0 - blend_weight) * torch.log(refined_probability.clamp_min(1e-6))
                    + blend_weight * torch.log(temporal_mean_prob.clamp_min(1e-6))
                )
                scene_predictions[f"refiner_temporal_mix_w{blend_suffix}"] = mixed_probability.argmax(dim=1)
                scene_predictions[f"refiner_temporal_poe_w{blend_suffix}"] = poe_score.argmax(dim=1)
            if args.meta_optimize:
                meta_probability, selected_weight, meta_accepted, scene_meta_stats = meta_optimize_poe_weight(
                    refined_probability.detach(),
                    temporal_mean_prob.detach(),
                    temporal_votes.detach(),
                    region_pred.detach(),
                    no_op_pred.detach(),
                    base_pred.detach(),
                    initial_weight=args.meta_initial_weight,
                    confidence_threshold=args.selection_threshold,
                    min_votes=args.min_temporal_votes,
                    inner_steps=args.meta_inner_steps,
                    inner_lr=args.meta_inner_lr,
                    correction_weight=args.meta_correction_weight,
                    keep_weight=args.meta_keep_weight,
                    entropy_weight=args.meta_entropy_weight,
                    query_tolerance=args.meta_query_tolerance,
                    classwise=args.meta_classwise,
                )
                scene_predictions["meta_scene_poe"] = meta_probability.argmax(dim=1)
                meta_adapted_pred = scene_predictions["meta_scene_poe"]
                meta_stats["scenes"] += 1
                meta_stats["accepted_scenes"] += int(meta_accepted)
                meta_stats["selected_weight_sum"] += float(selected_weight)
                meta_stats["adapted_weight_sum"] += float(scene_meta_stats.get("adapted_weight", selected_weight))
                meta_stats["query_gain_sum"] += float(scene_meta_stats.get("query_gain", 0.0))
                meta_stats["correction_ratio_sum"] += float(scene_meta_stats.get("correction_ratio", 0.0))
        for threshold in thresholds:
            suffix = int(round(100 * threshold))
            point_accept = (
                (temporal_votes >= args.min_temporal_votes)
                & (temporal_conf >= threshold)
                & (temporal_mean_pred != base_pred)
            )
            point_verified = base_pred.clone()
            point_verified[point_accept] = temporal_mean_pred[point_accept]

            confpoint_accept = (
                (temporal_votes >= args.min_temporal_votes)
                & (confweighted_conf >= threshold)
                & (temporal_confweighted_pred != base_pred)
            )
            confpoint_verified = base_pred.clone()
            confpoint_verified[confpoint_accept] = temporal_confweighted_pred[confpoint_accept]

            region_accept = (
                (temporal_region_pred == region_pred)
                & (temporal_region_votes >= args.min_temporal_votes)
                & (temporal_region_conf >= threshold)
                & (region_pred != base_pred)
            )
            region_verified = base_pred.clone()
            region_verified[region_accept] = region_pred[region_accept]

            cross_accept = point_accept & (temporal_mean_pred == region_pred)
            cross_verified = base_pred.clone()
            cross_verified[cross_accept] = temporal_mean_pred[cross_accept]

            point_region_union = point_verified.clone()
            compatible_region_accept = region_accept & (
                (~point_accept) | (region_pred == temporal_mean_pred)
            )
            point_region_union[compatible_region_accept] = region_pred[compatible_region_accept]

            strong_temporal = (
                (temporal_votes == len(reference_epochs))
                & (temporal_conf >= threshold)
                & (temporal_mean_pred != base_pred)
            )
            arbitrated = base_pred.clone()
            arbitrated[strong_temporal] = temporal_mean_pred[strong_temporal]
            arbitrated[cross_accept] = temporal_mean_pred[cross_accept]

            scene_predictions[f"point_verified_t{suffix}"] = point_verified
            scene_predictions[f"confpoint_verified_t{suffix}"] = confpoint_verified
            scene_predictions[f"region_verified_t{suffix}"] = region_verified
            scene_predictions[f"cross_verified_t{suffix}"] = cross_verified
            scene_predictions[f"point_region_union_t{suffix}"] = point_region_union
            scene_predictions[f"arbitrated_t{suffix}"] = arbitrated
            if refiner is not None:
                temporal_override = refined_pred.clone()
                temporal_override[point_accept] = temporal_mean_pred[point_accept]

                meta_init_override = meta_initial_poe_pred.clone()
                meta_init_override[point_accept] = temporal_mean_pred[point_accept]

                if args.four_stage_diagnostic and abs(threshold - args.selection_threshold) < 1e-6:
                    for scale, scale_probability in diagnostic_scale_probabilities:
                        scale_meta_score = (
                            (1.0 - args.meta_initial_weight)
                            * torch.log(scale_probability.clamp_min(1e-6))
                            + args.meta_initial_weight
                            * torch.log(temporal_mean_prob.clamp_min(1e-6))
                        )
                        scale_meta_prediction = scale_meta_score.argmax(dim=1)
                        scale_meta_prediction[point_accept] = temporal_mean_pred[point_accept]
                        scene_predictions[f"scale_meta_s{int(round(100 * scale))}"] = (
                            scale_meta_prediction
                        )

                if args.four_stage_diagnostic and abs(threshold - args.selection_threshold) < 1e-6:
                    for support in (3, 4, 5):
                        scene_predictions[f"meta_local_v{support}"] = local_consensus_verifier(
                            meta_init_override,
                            base_pred,
                            region_pred,
                            no_op_pred,
                            temporal_predictions,
                            temporal_mean_prob,
                            min_support=support,
                            temporal_confidence_threshold=threshold,
                        )
                    for reliability in (0.4, 0.5, 0.6, 0.7):
                        scene_predictions[f"meta_region_r{int(round(100 * reliability))}"] = (
                            region_risk_rollback(
                                meta_init_override,
                                refined_pred,
                                region_pred,
                                no_op_pred,
                                temporal_mean_prob,
                                temporal_vote_count,
                                regions,
                                min_region_reliability=reliability,
                            )
                        )

                meta_init_region = meta_init_override.clone()
                meta_region_completion = region_accept & ~point_accept & (
                    (meta_init_override == base_pred)
                    | (meta_init_override == region_pred)
                    | (no_op_pred == region_pred)
                )
                meta_init_region[meta_region_completion] = region_pred[meta_region_completion]

                override_region = temporal_override.clone()
                region_completion = region_accept & ~point_accept & (
                    (refined_pred == base_pred)
                    | (refined_pred == region_pred)
                    | (no_op_pred == region_pred)
                )
                override_region[region_completion] = region_pred[region_completion]

                compatible_union = refined_pred.clone()
                verifier_changed = point_region_union != base_pred
                compatible = verifier_changed & (
                    (refined_pred == base_pred)
                    | (refined_pred == point_region_union)
                    | (no_op_pred == point_region_union)
                )
                compatible_union[compatible] = point_region_union[compatible]

                refiner_support = temporal_vote_count.gather(1, refined_pred.unsqueeze(1)).squeeze(1)
                refiner_temporal_prob = temporal_mean_prob.gather(1, refined_pred.unsqueeze(1)).squeeze(1)
                residual_changed = refined_pred != no_op_pred
                residual_accept = (
                    (refiner_support >= args.min_temporal_votes)
                    & (refiner_temporal_prob >= threshold)
                )
                residual_verified = refined_pred.clone()
                residual_verified[residual_changed & ~residual_accept] = no_op_pred[
                    residual_changed & ~residual_accept
                ]

                verified_union = residual_verified.clone()
                union_compatible = verifier_changed & (
                    (verified_union == base_pred)
                    | (verified_union == point_region_union)
                    | (no_op_pred == point_region_union)
                )
                verified_union[union_compatible] = point_region_union[union_compatible]

                scene_predictions[f"refiner_temporal_override_t{suffix}"] = temporal_override
                scene_predictions[f"meta_init_poe_override_t{suffix}"] = meta_init_override
                scene_predictions[f"meta_init_poe_region_t{suffix}"] = meta_init_region
                scene_predictions[f"refiner_override_region_t{suffix}"] = override_region
                scene_predictions[f"refiner_compatible_union_t{suffix}"] = compatible_union
                scene_predictions[f"residual_verified_t{suffix}"] = residual_verified
                scene_predictions[f"refiner_verified_union_t{suffix}"] = verified_union
                if meta_adapted_pred is not None:
                    meta_adapt_override = meta_adapted_pred.clone()
                    meta_adapt_override[point_accept] = temporal_mean_pred[point_accept]
                    scene_predictions[f"meta_adapt_override_t{suffix}"] = meta_adapt_override
                if meta_refiner_pred is not None:
                    meta_refiner_override = meta_refiner_pred.clone()
                    meta_refiner_override[point_accept] = temporal_mean_pred[point_accept]

                    meta_refiner_changed = meta_refiner_pred != no_op_pred
                    meta_refiner_residual_verified = meta_refiner_pred.clone()
                    meta_refiner_residual_verified[
                        meta_refiner_changed & ~residual_accept
                    ] = no_op_pred[meta_refiner_changed & ~residual_accept]

                    meta_refiner_verified_union = meta_refiner_residual_verified.clone()
                    meta_refiner_union_compatible = verifier_changed & (
                        (meta_refiner_verified_union == base_pred)
                        | (meta_refiner_verified_union == point_region_union)
                        | (no_op_pred == point_region_union)
                    )
                    meta_refiner_verified_union[meta_refiner_union_compatible] = (
                        point_region_union[meta_refiner_union_compatible]
                    )

                    meta_refiner_poe_score = (
                        (1.0 - args.meta_initial_weight)
                        * torch.log(meta_refiner_probability.clamp_min(1e-6))
                        + args.meta_initial_weight
                        * torch.log(temporal_mean_prob.clamp_min(1e-6))
                    )
                    meta_refiner_poe_override = meta_refiner_poe_score.argmax(dim=1)
                    meta_refiner_poe_override[point_accept] = temporal_mean_pred[point_accept]

                    meta_candidate_support = temporal_vote_count.gather(
                        1, meta_refiner_pred.unsqueeze(1)
                    ).squeeze(1)
                    meta_candidate_confidence = temporal_mean_prob.gather(
                        1, meta_refiner_pred.unsqueeze(1)
                    ).squeeze(1)
                    meta_candidate_changed = meta_refiner_pred != refined_pred
                    meta_candidate_accept = (
                        (meta_candidate_support >= args.min_temporal_votes)
                        & (
                            meta_candidate_confidence
                            >= args.meta_refiner_verify_threshold
                        )
                        & (
                            (meta_refiner_pred == temporal_mean_pred)
                            | (meta_refiner_pred == region_pred)
                            | (meta_refiner_pred == no_op_pred)
                        )
                    )
                    meta_candidate_gate = meta_refiner_pred.clone()
                    meta_candidate_gate[
                        meta_candidate_changed & ~meta_candidate_accept
                    ] = refined_pred[meta_candidate_changed & ~meta_candidate_accept]
                    meta_refiner_verified_override = meta_candidate_gate.clone()
                    meta_refiner_verified_override[point_accept] = temporal_mean_pred[point_accept]

                    verified_meta_probability = torch.where(
                        (meta_candidate_changed & meta_candidate_accept)[:, None],
                        meta_refiner_probability,
                        refined_probability,
                    )
                    verified_meta_poe_score = (
                        (1.0 - args.meta_initial_weight)
                        * torch.log(verified_meta_probability.clamp_min(1e-6))
                        + args.meta_initial_weight
                        * torch.log(temporal_mean_prob.clamp_min(1e-6))
                    )
                    meta_refiner_verified_poe_override = verified_meta_poe_score.argmax(dim=1)
                    meta_refiner_verified_poe_override[point_accept] = temporal_mean_pred[point_accept]

                    meta_refiner_joint_structural = meta_init_override.clone()
                    meta_joint_available = (
                        meta_candidate_changed
                        & meta_candidate_accept
                        & (meta_init_override == refined_pred)
                    )
                    meta_refiner_joint_structural[meta_joint_available] = meta_refiner_pred[
                        meta_joint_available
                    ]
                    meta_refiner_joint_temporal = meta_init_override.clone()
                    meta_temporal_available = meta_joint_available & (
                        meta_refiner_pred == temporal_mean_pred
                    )
                    meta_refiner_joint_temporal[meta_temporal_available] = meta_refiner_pred[
                        meta_temporal_available
                    ]

                    scene_predictions[f"meta_refiner_override_t{suffix}"] = meta_refiner_override
                    scene_predictions[
                        f"meta_refiner_residual_verified_t{suffix}"
                    ] = meta_refiner_residual_verified
                    scene_predictions[
                        f"meta_refiner_verified_union_t{suffix}"
                    ] = meta_refiner_verified_union
                    scene_predictions[
                        f"meta_refiner_poe_override_t{suffix}"
                    ] = meta_refiner_poe_override
                    scene_predictions[
                        f"meta_refiner_candidate_gate_t{suffix}"
                    ] = meta_candidate_gate
                    scene_predictions[
                        f"meta_refiner_verified_override_t{suffix}"
                    ] = meta_refiner_verified_override
                    scene_predictions[
                        f"meta_refiner_verified_poe_override_t{suffix}"
                    ] = meta_refiner_verified_poe_override
                    scene_predictions[
                        f"meta_refiner_joint_structural_t{suffix}"
                    ] = meta_refiner_joint_structural
                    scene_predictions[
                        f"meta_refiner_joint_temporal_t{suffix}"
                    ] = meta_refiner_joint_temporal

        if refiner is not None:
            save_scene_predictions(
                args,
                eval_dataset,
                local_scene_index,
                batch,
                scene_predictions,
                split_targets,
            )
        valid = labels != args.ignore_label
        inverse = inverse_map.long().cuda()
        for name, prediction in scene_predictions.items():
            all_predictions[name].append(prediction[inverse].cpu()[valid])
        all_labels.append(labels[valid])
        full_regions = regions[inverse].cpu()[valid].long()
        invalid_region = full_regions < 0
        full_regions = full_regions + region_offset
        if invalid_region.any():
            start = int(full_regions.max().item()) + 1
            full_regions[invalid_region] = start
        all_region_ids.append(full_regions)
        region_offset = int(full_regions.max().item()) + 1
        completed = local_scene_index + 1
        if completed % 20 == 0 or completed == len(eval_dataset):
            elapsed = time.time() - evaluation_started
            eta_minutes = (elapsed / completed) * (len(eval_dataset) - completed) / 60.0
            print(
                f"[{args.dataset}] {completed}/{len(eval_dataset)} scenes, ETA {eta_minutes:.1f} min",
                flush=True,
            )

    labels_np = torch.cat(all_labels).numpy()
    base_np = torch.cat(all_predictions["base"]).numpy()
    base_metrics = compute_unsupervised_metrics(base_np, labels_np, args.semantic_class)[:4]
    results = {}
    for name in names:
        prediction_np = torch.cat(all_predictions[name]).numpy()
        metrics = compute_unsupervised_metrics(prediction_np, labels_np, args.semantic_class)[:4]
        results[name] = {
            "oAcc": float(metrics[0]),
            "mAcc": float(metrics[1]),
            "mIoU": float(metrics[3]),
            "delta_mIoU": float(metrics[3] - base_metrics[3]),
            "changed_ratio": float((prediction_np != base_np).mean()),
        }

    pred_to_gt = prediction_mapping(base_np, labels_np, args.semantic_class)
    region_ids_np = torch.cat(all_region_ids).numpy()
    mapped_base = pred_to_gt[base_np]
    oracle = base_np.copy()
    for candidate_name in ["region", "temporal_mean", "temporal_vote"]:
        candidate = torch.cat(all_predictions[candidate_name]).numpy()
        candidate_correct = pred_to_gt[candidate] == labels_np
        base_wrong = mapped_base != labels_np
        oracle[base_wrong & candidate_correct] = candidate[base_wrong & candidate_correct]
    oracle_metrics = compute_unsupervised_metrics(oracle, labels_np, args.semantic_class)[:4]
    results["candidate_oracle_diagnostic"] = {
        "oAcc": float(oracle_metrics[0]),
        "mAcc": float(oracle_metrics[1]),
        "mIoU": float(oracle_metrics[3]),
        "delta_mIoU": float(oracle_metrics[3] - base_metrics[3]),
        "changed_ratio": float((oracle != base_np).mean()),
    }
    if diagnostic_names:
        diagnostic_oracle = base_np.copy()
        for candidate_name in diagnostic_names:
            candidate = torch.cat(all_predictions[candidate_name]).numpy()
            candidate_correct = pred_to_gt[candidate] == labels_np
            base_wrong = mapped_base != labels_np
            diagnostic_oracle[base_wrong & candidate_correct] = candidate[base_wrong & candidate_correct]
        diagnostic_metrics = compute_unsupervised_metrics(
            diagnostic_oracle, labels_np, args.semantic_class
        )[:4]
        results["four_stage_point_oracle_diagnostic"] = {
            "oAcc": float(diagnostic_metrics[0]),
            "mAcc": float(diagnostic_metrics[1]),
            "mIoU": float(diagnostic_metrics[3]),
            "delta_mIoU": float(diagnostic_metrics[3] - base_metrics[3]),
            "changed_ratio": float((diagnostic_oracle != base_np).mean()),
        }
        oracle_pools = {
            "split": [
                name for name in diagnostic_names if name.startswith("split_merge_")
            ] + ["split_noop", "split_multi_noop"],
            "refiner": [
                name
                for name in diagnostic_names
                if name.startswith("refiner_scale_")
                or name.startswith("refiner_local_")
                or name.startswith("scale_select_")
                or name.startswith("scale_meta_")
                or name.startswith("typed_refine_")
            ] + ["split_refiner", "split_multi_refiner"],
            "verifier": [
                name
                for name in diagnostic_names
                if name.startswith("local_verify_")
                or name.startswith("meta_local_")
                or name.startswith("meta_region_")
            ] + [
                f"point_region_union_t{int(round(100 * args.selection_threshold))}",
                f"refiner_verified_union_t{int(round(100 * args.selection_threshold))}",
            ],
            "meta": [
                f"meta_init_poe_override_t{int(round(100 * args.selection_threshold))}",
                f"refiner_temporal_poe_w{int(round(100 * args.meta_initial_weight))}",
            ],
        }
        for module_name, pool_names in oracle_pools.items():
            pool_names = [name for name in pool_names if name in all_predictions]
            pool_predictions = [
                torch.cat(all_predictions[name]).numpy() for name in pool_names
            ]
            granularities = [("point", None)]
            if not args.skip_region_oracle:
                granularities.append(("region", region_ids_np))
            for granularity, groups in granularities:
                module_oracle = oracle_diagnostic(
                    base_np,
                    labels_np,
                    pred_to_gt,
                    pool_predictions,
                    group_ids=groups,
                )
                module_metrics = compute_unsupervised_metrics(
                    module_oracle, labels_np, args.semantic_class
                )[:4]
                results[f"{module_name}_{granularity}_oracle_diagnostic"] = {
                    "oAcc": float(module_metrics[0]),
                    "mAcc": float(module_metrics[1]),
                    "mIoU": float(module_metrics[3]),
                    "delta_mIoU": float(module_metrics[3] - base_metrics[3]),
                    "changed_ratio": float((module_oracle != base_np).mean()),
                }

    total_proxy_points = max(proxy_sums.pop("points"), 1.0)
    proxies = {name: value / total_proxy_points for name, value in sorted(proxy_sums.items())}
    reliable_anchor_epochs = [
        epoch
        for epoch in reference_epochs
        if proxies[f"epoch_{epoch}_confidence"] >= proxies["base_confidence"]
        and proxies[f"epoch_{epoch}_entropy"] <= proxies["base_entropy"]
    ]
    selected_threshold = min(thresholds, key=lambda value: abs(value - args.selection_threshold))
    selected_suffix = int(round(100 * selected_threshold))
    independent_candidate = f"point_region_union_t{selected_suffix}"
    selected_independent_strategy = independent_candidate if reliable_anchor_epochs else "base"
    results["selected_independent"] = dict(results[selected_independent_strategy])
    results["selected_independent"]["strategy"] = selected_independent_strategy
    if refiner is not None:
        if args.meta_optimize:
            joint_candidate = f"meta_adapt_override_t{selected_suffix}"
        else:
            joint_candidate = f"meta_init_poe_override_t{selected_suffix}"
        selected_joint_strategy = joint_candidate if reliable_anchor_epochs else "base"
        results["selected_joint"] = dict(results[selected_joint_strategy])
        results["selected_joint"]["strategy"] = selected_joint_strategy
    selection = {
        "requested_threshold": args.selection_threshold,
        "selected_threshold": selected_threshold,
        "reliable_anchor_epochs": reliable_anchor_epochs,
        "temporal_enabled": bool(reliable_anchor_epochs),
        "independent_strategy": selected_independent_strategy,
    }
    if refiner is not None:
        selection["joint_strategy"] = selected_joint_strategy
    if args.meta_optimize and meta_stats["scenes"] > 0:
        scene_count = float(meta_stats["scenes"])
        meta_summary = {
            "scenes": int(meta_stats["scenes"]),
            "accepted_scenes": int(meta_stats["accepted_scenes"]),
            "accept_ratio": meta_stats["accepted_scenes"] / scene_count,
            "mean_selected_weight": meta_stats["selected_weight_sum"] / scene_count,
            "mean_adapted_weight": meta_stats["adapted_weight_sum"] / scene_count,
            "mean_query_gain": meta_stats["query_gain_sum"] / scene_count,
            "mean_correction_ratio": meta_stats["correction_ratio_sum"] / scene_count,
        }
    else:
        meta_summary = {}
    if args.meta_refiner and meta_refiner_stats["scenes"] > 0:
        scene_count = float(meta_refiner_stats["scenes"])
        meta_refiner_summary = {
            "scenes": int(meta_refiner_stats["scenes"]),
            "accepted_scenes": int(meta_refiner_stats["accepted_scenes"]),
            "accept_ratio": meta_refiner_stats["accepted_scenes"] / scene_count,
            "mean_query_gain": meta_refiner_stats["query_gain_sum"] / scene_count,
            "mean_correction_ratio": meta_refiner_stats["correction_ratio_sum"] / scene_count,
            "support_corrections": int(meta_refiner_stats["support_corrections"]),
            "query_corrections": int(meta_refiner_stats["query_corrections"]),
            "mean_scales": [
                value / scene_count for value in (meta_refiner_stats["scale_sums"] or [])
            ],
            "mean_bias_norm": meta_refiner_stats["bias_norm_sum"] / scene_count,
        }
    else:
        meta_refiner_summary = {}
    elapsed_seconds = time.time() - evaluation_started
    efficiency = {
        "scenes": len(eval_dataset),
        "total_seconds": elapsed_seconds,
        "seconds_per_scene": elapsed_seconds / max(len(eval_dataset), 1),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
    }

    for name, result in sorted(results.items(), key=lambda item: item[1]["mIoU"], reverse=True):
        print(
            f"{name:30s} mIoU {result['mIoU']:.4f} "
            f"delta {result['delta_mIoU']:+.4f} changed {100.0 * result['changed_ratio']:.2f}%"
        )
    print("label_free_proxies", json.dumps(proxies, sort_keys=True))
    print("label_free_selection", json.dumps(selection, sort_keys=True))
    if meta_summary:
        print("meta_optimizer", json.dumps(meta_summary, sort_keys=True))
    if meta_refiner_summary:
        print("meta_refiner", json.dumps(meta_refiner_summary, sort_keys=True))
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "config": vars(args),
                    "dataset": args.dataset,
                    "checkpoint_dir": args.checkpoint_dir,
                    "refiner_checkpoint": args.refiner_checkpoint,
                    "checkpoint_binding": checkpoint_binding,
                    "base_epoch": args.base_epoch,
                    "reference_epochs": reference_epochs,
                    "test_area": args.test_area,
                    "results": results,
                    "label_free_proxies": proxies,
                    "label_free_selection": selection,
                    "meta_optimizer": meta_summary,
                    "meta_refiner": meta_refiner_summary,
                    "efficiency": efficiency,
                },
                output_file,
                indent=2,
            )


if __name__ == "__main__":
    main()
