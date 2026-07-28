import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from eval_S3DIS import compute_unsupervised_metrics
from lib.semantic_difference_pipeline import (
    SemanticDifferencePipelineConfig,
    run_semantic_difference_pipeline,
)
from models.query_refiner import ErrorQueryRefiner
from tools_eval_error_verifier import load_reference, parse_areas, parse_int_list


STAGE_ORDER = (
    "base",
    "decomposition",
    "refiner",
    "meta_refiner",
    "meta_verified",
    "final_verified",
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate Decomposition + Meta-based Refiner + Result Verifier"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=(
            "/home/magic/magic/cm/repositories/GrowSP/"
            "ckpt/S3DIS/1baseline/ckpts"
        ),
    )
    parser.add_argument("--base_epoch", type=int, default=1270)
    parser.add_argument("--reference_epochs", default="1170,1180,1190")
    parser.add_argument("--test_area", default="Area_5")
    parser.add_argument("--data_path", default="data/S3DIS/input")
    parser.add_argument("--sp_path", default="data/S3DIS/initial_superpoints/")
    parser.add_argument(
        "--refiner_checkpoint",
        default=(
            "ckpt/S3DIS/refiner_projectloss02_e10/"
            "refiner_best_checkpoint.pth"
        ),
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_scenes", type=int, default=0)

    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--primitive_num", type=int, default=300)
    parser.add_argument("--semantic_class", type=int, default=12)
    parser.add_argument("--feats_dim", type=int, default=128)
    parser.add_argument("--ignore_label", type=int, default=12)
    parser.add_argument("--bn_momentum", type=float, default=0.02)
    parser.add_argument("--conv1_kernel_size", type=int, default=5)
    parser.add_argument("--prob_scale", type=float, default=10.0)
    parser.add_argument("--refiner_scale", type=float, default=1.0)
    parser.add_argument("--refiner_hidden_dim", type=int, default=128)
    parser.add_argument("--refiner_num_heads", type=int, default=4)

    parser.add_argument("--min_temporal_votes", type=int, default=2)
    parser.add_argument("--meta_target_confidence", type=float, default=0.80)
    parser.add_argument("--meta_verify_confidence", type=float, default=0.64)
    parser.add_argument("--result_verify_confidence", type=float, default=0.80)
    parser.add_argument("--meta_inner_steps", type=int, default=5)
    parser.add_argument("--meta_inner_lr", type=float, default=0.1)
    parser.add_argument("--meta_keep_weight", type=float, default=5.0)
    parser.add_argument("--meta_scale_reg", type=float, default=0.1)
    parser.add_argument("--meta_bias_reg", type=float, default=1.0)
    parser.add_argument("--disable_meta_bias", action="store_true")
    return parser.parse_args()


def build_pipeline_config(args):
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
    )


def load_models(args):
    reference_epochs = parse_int_list(args.reference_epochs)
    base_model, base_centers = load_reference(args, args.base_epoch)
    temporal_references = []
    for epoch in reference_epochs:
        model, centers = load_reference(
            args,
            epoch,
            reference_centers=base_centers,
        )
        temporal_references.append((epoch, model, centers))

    refiner = ErrorQueryRefiner(
        feat_dim=args.feats_dim,
        num_classes=args.semantic_class,
        hidden_dim=args.refiner_hidden_dim,
        num_heads=args.refiner_num_heads,
        dropout=0.0,
    ).cuda()
    refiner.load_state_dict(
        torch.load(args.refiner_checkpoint, map_location="cpu"),
        strict=False,
    )
    refiner.eval()
    return base_model, base_centers, temporal_references, refiner


def summarize_metrics(all_predictions, labels, semantic_classes):
    labels_array = torch.cat(labels).numpy()
    concatenated = {
        name: torch.cat(values).numpy()
        for name, values in all_predictions.items()
    }
    base_prediction = concatenated["base"]
    base_metrics = compute_unsupervised_metrics(
        base_prediction,
        labels_array,
        semantic_classes,
    )[:4]
    results = {}
    previous_name = None
    for name in STAGE_ORDER:
        prediction = concatenated[name]
        metrics = compute_unsupervised_metrics(
            prediction,
            labels_array,
            semantic_classes,
        )[:4]
        if previous_name is None:
            changed_from_previous = 0.0
        else:
            changed_from_previous = float(
                (prediction != concatenated[previous_name]).mean()
            )
        results[name] = {
            "oAcc": float(metrics[0]),
            "mAcc": float(metrics[1]),
            "mIoU": float(metrics[3]),
            "delta_mIoU_from_base": float(metrics[3] - base_metrics[3]),
            "delta_mIoU_from_previous": (
                0.0
                if previous_name is None
                else float(metrics[3] - results[previous_name]["mIoU"])
            ),
            "changed_from_base": float(
                (prediction != base_prediction).mean()
            ),
            "changed_from_previous": changed_from_previous,
        }
        previous_name = name
    return results


def normalized_statistics(sums):
    point_count = max(sums.get("points", 0.0), 1.0)
    scene_count = max(sums.get("scenes", 0.0), 1.0)
    return {
        "scenes": int(sums.get("scenes", 0.0)),
        "voxel_points": int(sums.get("points", 0.0)),
        "queries_per_scene": sums.get("queries", 0.0) / scene_count,
        "split_queries_per_scene": (
            sums.get("split_queries", 0.0) / scene_count
        ),
        "refine_point_ratio": sums.get("refine_points", 0.0) / point_count,
        "split_point_ratio": sums.get("split_points", 0.0) / point_count,
        "meta_scene_accept_ratio": (
            sums.get("meta_scene_accepted", 0.0) / scene_count
        ),
        "mean_meta_query_gain": (
            sums.get("meta_query_gain", 0.0) / scene_count
        ),
        "meta_candidate_change_ratio": (
            sums.get("meta_changed", 0.0) / point_count
        ),
        "meta_candidate_accept_ratio": (
            sums.get("meta_accepted", 0.0) / point_count
        ),
        "meta_candidate_rollback_ratio": (
            sums.get("meta_rollback", 0.0) / point_count
        ),
        "temporal_override_ratio": (
            sums.get("temporal_override", 0.0) / point_count
        ),
        "final_changed_from_base_ratio": (
            sums.get("final_changed_from_base", 0.0) / point_count
        ),
    }


def print_results(results, statistics):
    print("\nStage-wise Area evaluation")
    print("{:<18} {:>9} {:>10} {:>10}".format(
        "stage", "mIoU", "delta-base", "delta-prev"
    ))
    for name in STAGE_ORDER:
        values = results[name]
        print(
            "{:<18} {:>9.4f} {:>+10.4f} {:>+10.4f}".format(
                name,
                values["mIoU"],
                values["delta_mIoU_from_base"],
                values["delta_mIoU_from_previous"],
            )
        )
    print("\nPipeline statistics")
    print(json.dumps(statistics, indent=2))


def main():
    args = parse_args()
    config = build_pipeline_config(args)
    base_model, base_centers, temporal_references, refiner = load_models(args)
    dataset = S3DIStest(args, areas=parse_areas(args.test_area))
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=cfl_collate_fn_test(),
        num_workers=args.workers,
        pin_memory=True,
    )

    all_predictions = {name: [] for name in STAGE_ORDER}
    all_labels = []
    statistic_sums = defaultdict(float)
    for scene_id, batch in enumerate(loader):
        if args.max_scenes > 0 and scene_id >= args.max_scenes:
            break
        coords, features, inverse_map, labels, _index, region = batch
        in_field = ME.TensorField(features, coords, device=0)
        with torch.no_grad():
            base_features = F.normalize(base_model(in_field), dim=1)
            base_scores = F.linear(base_features, base_centers)
            temporal_probabilities = []
            for _epoch, model, centers in temporal_references:
                reference_features = F.normalize(model(in_field), dim=1)
                reference_scores = F.linear(reference_features, centers)
                temporal_probabilities.append(
                    F.softmax(
                        reference_scores * float(config.probability_scale),
                        dim=1,
                    )
                )
            temporal_probabilities = torch.stack(
                temporal_probabilities,
                dim=0,
            )

        regions = region.squeeze().long().cuda()
        output = run_semantic_difference_pipeline(
            config,
            refiner,
            base_scores,
            base_features,
            coords,
            features,
            regions,
            temporal_probabilities,
        )
        inverse = inverse_map.long().cuda()
        valid = labels != args.ignore_label
        for name, prediction in output.stage_predictions().items():
            all_predictions[name].append(
                prediction[inverse].cpu()[valid]
            )
        all_labels.append(labels[valid])

        statistic_sums["scenes"] += 1
        for name, value in output.decomposition.statistics.items():
            statistic_sums[name] += float(value)
        statistic_sums["meta_scene_accepted"] += int(
            output.meta_refiner.scene_accepted
        )
        statistic_sums["meta_query_gain"] += float(
            output.meta_refiner.statistics["query_gain"]
        )
        for name in (
            "meta_changed",
            "meta_accepted",
            "meta_rollback",
            "temporal_override",
            "final_changed_from_base",
        ):
            statistic_sums[name] += float(output.verifier.statistics[name])

    results = summarize_metrics(
        all_predictions,
        all_labels,
        args.semantic_class,
    )
    statistics = normalized_statistics(statistic_sums)
    print_results(results, statistics)
    payload = {
        "method": (
            "Decomposition + Meta-based Refiner + Result Verifier"
        ),
        "label_usage": (
            "Ground truth is used only after inference for evaluation metrics."
        ),
        "test_area": args.test_area,
        "base_epoch": args.base_epoch,
        "reference_epochs": parse_int_list(args.reference_epochs),
        "pipeline_config": asdict(config),
        "results": results,
        "statistics": statistics,
    }
    if args.output_json:
        output_directory = os.path.dirname(args.output_json)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
        print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
