import argparse
import json
import os

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from eval_S3DIS import compute_unsupervised_metrics
from lib.split_regions import build_region_consistency_queries, build_split_region_queries
from models.fpn import Res16FPN18
from models.query_refiner import ErrorQueryRefiner


def parse_args():
    parser = argparse.ArgumentParser("Evaluate label-free temporal/region correction verification")
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
    parser.add_argument("--min_temporal_votes", type=int, default=2)
    parser.add_argument("--refiner_checkpoint", default="")
    parser.add_argument("--refiner_scale", type=float, default=1.0)
    parser.add_argument("--refiner_hidden_dim", type=int, default=128)
    parser.add_argument("--refiner_num_heads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_areas(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


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


def load_reference(args, epoch, reference_centers=None):
    model = Res16FPN18(
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


def run_split_refiner(args, refiner, base_scores, base_feats, coords, features, regions):
    batch_ids = coords[:, 0].long().cuda()
    point_coords = coords[:, 1:].float().cuda()
    point_colors = features[:, :3].float().cuda()
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
    del refine_mask, keep_mask

    delta_scores = refiner(
        base_feats,
        point_coords,
        batch_ids,
        query_indices,
        regions,
        use_region_branch=False,
    )
    no_op_scores = project_region_and_split(base_scores, regions, split_targets)
    refined_scores = project_region_and_split(
        base_scores + args.refiner_scale * delta_scores,
        regions,
        split_targets,
    )
    return no_op_scores, refined_scores


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


def main():
    args = parse_args()
    reference_epochs = parse_int_list(args.reference_epochs)
    thresholds = parse_float_list(args.thresholds)

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

    loader = DataLoader(
        S3DIStest(args, areas=parse_areas(args.test_area)),
        batch_size=1,
        collate_fn=cfl_collate_fn_test(),
        num_workers=args.workers,
        pin_memory=True,
    )

    names = ["base", "region", "temporal_mean", "temporal_confweighted", "temporal_vote", "all_mean"]
    names.extend([f"epoch_{epoch}" for epoch in reference_epochs])
    if refiner is not None:
        names.extend(["split_noop", "split_refiner", "refiner_candidate_vote"])
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
                    f"refiner_override_region_t{suffix}",
                    f"refiner_compatible_union_t{suffix}",
                    f"residual_verified_t{suffix}",
                    f"refiner_verified_union_t{suffix}",
                ]
            )
    all_predictions = {name: [] for name in names}
    all_labels = []
    proxy_sums = {"points": 0.0}

    def add_proxy(name, value, weight):
        proxy_sums[name] = proxy_sums.get(name, 0.0) + float(value) * float(weight)

    for coords, features, inverse_map, labels, _index, region in loader:
        in_field = ME.TensorField(features, coords, device=0)
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
        if refiner is not None:
            no_op_scores, refined_scores = run_split_refiner(
                args,
                refiner,
                base_scores,
                base_feats,
                coords,
                features,
                regions,
            )
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
                scene_predictions[f"refiner_override_region_t{suffix}"] = override_region
                scene_predictions[f"refiner_compatible_union_t{suffix}"] = compatible_union
                scene_predictions[f"residual_verified_t{suffix}"] = residual_verified
                scene_predictions[f"refiner_verified_union_t{suffix}"] = verified_union

        valid = labels != args.ignore_label
        inverse = inverse_map.long().cuda()
        for name, prediction in scene_predictions.items():
            all_predictions[name].append(prediction[inverse].cpu()[valid])
        all_labels.append(labels[valid])

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
    selected_independent_strategy = independent_candidate if reliable_anchor_epochs else "region"
    results["selected_independent"] = dict(results[selected_independent_strategy])
    results["selected_independent"]["strategy"] = selected_independent_strategy
    if refiner is not None:
        joint_candidate = f"refiner_temporal_override_t{selected_suffix}"
        selected_joint_strategy = joint_candidate if reliable_anchor_epochs else "split_refiner"
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

    for name, result in sorted(results.items(), key=lambda item: item[1]["mIoU"], reverse=True):
        print(
            f"{name:30s} mIoU {result['mIoU']:.4f} "
            f"delta {result['delta_mIoU']:+.4f} changed {100.0 * result['changed_ratio']:.2f}%"
        )
    print("label_free_proxies", json.dumps(proxies, sort_keys=True))
    print("label_free_selection", json.dumps(selection, sort_keys=True))
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "base_epoch": args.base_epoch,
                    "reference_epochs": reference_epochs,
                    "test_area": args.test_area,
                    "results": results,
                    "label_free_proxies": proxies,
                    "label_free_selection": selection,
                },
                output_file,
                indent=2,
            )


if __name__ == "__main__":
    main()
