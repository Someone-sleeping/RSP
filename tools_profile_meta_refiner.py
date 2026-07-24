import argparse
import json
import os
import time

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from lib.meta_refiner import meta_adapt_refiner_gates
from models.query_refiner import ErrorQueryRefiner
from tools_eval_error_verifier import (
    load_reference,
    parse_areas,
    parse_int_list,
    region_candidates,
    run_split_refiner,
)


def parse_args():
    parser = argparse.ArgumentParser("Profile the label-free Meta-Refiner pipeline")
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/magic/magic/cm/repositories/GrowSP/ckpt/S3DIS/1baseline/ckpts",
    )
    parser.add_argument("--base_epoch", type=int, default=1270)
    parser.add_argument("--reference_epochs", default="1170,1180,1190")
    parser.add_argument("--test_area", default="Area_5")
    parser.add_argument("--data_path", default="data/S3DIS/input")
    parser.add_argument("--sp_path", default="data/S3DIS/initial_superpoints/")
    parser.add_argument(
        "--refiner_checkpoint",
        default="ckpt/S3DIS/refiner_projectloss02_e10/refiner_best_checkpoint.pth",
    )
    parser.add_argument("--output_json", default="ckpt/S3DIS/meta_refiner/efficiency_area5.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup_scenes", type=int, default=3)
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
    parser.add_argument("--selection_threshold", type=float, default=0.8)
    parser.add_argument("--min_temporal_votes", type=int, default=2)
    parser.add_argument("--meta_initial_weight", type=float, default=0.08)
    parser.add_argument("--meta_refiner_initial_scale", type=float, default=1.0)
    parser.add_argument("--meta_refiner_min_scale", type=float, default=0.25)
    parser.add_argument("--meta_refiner_max_scale", type=float, default=1.75)
    parser.add_argument("--meta_refiner_inner_steps", type=int, default=5)
    parser.add_argument("--meta_refiner_inner_lr", type=float, default=0.1)
    parser.add_argument("--meta_refiner_correction_weight", type=float, default=1.0)
    parser.add_argument("--meta_refiner_keep_weight", type=float, default=5.0)
    parser.add_argument("--meta_refiner_entropy_weight", type=float, default=0.01)
    parser.add_argument("--meta_refiner_scale_reg", type=float, default=0.1)
    parser.add_argument("--meta_refiner_bias_reg", type=float, default=1.0)
    parser.add_argument("--meta_refiner_query_tolerance", type=float, default=0.0)
    parser.add_argument("--meta_refiner_verify_threshold", type=float, default=0.64)
    return parser.parse_args()


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def timed_call(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    return output, 1000.0 * (time.perf_counter() - start)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "std": float(array.std()),
    }


def main():
    args = parse_args()
    reference_epochs = parse_int_list(args.reference_epochs)
    base_model, base_centers = load_reference(args, args.base_epoch)
    temporal_references = []
    for epoch in reference_epochs:
        model, centers = load_reference(args, epoch, reference_centers=base_centers)
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

    backbone_parameters = parameter_count(base_model)
    refiner_parameters = parameter_count(refiner)
    adapter_parameters = 2 + args.semantic_class
    parameter_summary = {
        "pretrained_backbone": int(backbone_parameters),
        "error_query_refiner": int(refiner_parameters),
        "episodic_adapter": int(adapter_parameters),
        "semantic_difference_verifier": 0,
        "additional_trainable_total": int(refiner_parameters + adapter_parameters),
        "additional_trainable_percent_of_backbone": float(
            100.0 * (refiner_parameters + adapter_parameters) / backbone_parameters
        ),
        "online_frozen_reference_copies": len(temporal_references),
        "online_frozen_reference_parameters": int(
            len(temporal_references) * backbone_parameters
        ),
    }

    loader = DataLoader(
        S3DIStest(args, areas=parse_areas(args.test_area)),
        batch_size=1,
        collate_fn=cfl_collate_fn_test(),
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident_memory = torch.cuda.memory_allocated()
    records = []

    for scene_id, (coords, features, _inverse_map, _labels, _index, region) in enumerate(loader):
        if args.max_scenes > 0 and scene_id >= args.warmup_scenes + args.max_scenes:
            break
        torch.cuda.reset_peak_memory_stats()
        scene_start = time.perf_counter()
        in_field = ME.TensorField(features, coords, device=0)

        def run_backbone():
            with torch.no_grad():
                features_out = F.normalize(base_model(in_field), dim=1)
                scores_out = F.linear(features_out, base_centers)
            return features_out, scores_out

        (base_feats, base_scores), backbone_ms = timed_call(run_backbone)

        def run_temporal():
            probabilities = []
            with torch.no_grad():
                for _epoch, model, centers in temporal_references:
                    temporal_feats = F.normalize(model(in_field), dim=1)
                    temporal_scores = F.linear(temporal_feats, centers)
                    probabilities.append(
                        F.softmax(temporal_scores * args.prob_scale, dim=1)
                    )
            stacked = torch.stack(probabilities, dim=0)
            mean_probability = stacked.mean(dim=0)
            predictions = stacked.argmax(dim=2)
            vote_count = F.one_hot(
                predictions,
                num_classes=args.semantic_class,
            ).sum(dim=0)
            votes, _vote_prediction = vote_count.max(dim=1)
            return stacked, mean_probability, vote_count, votes

        (
            temporal_probabilities,
            temporal_mean_probability,
            temporal_vote_count,
            temporal_votes,
        ), temporal_ms = timed_call(run_temporal)

        regions = region.squeeze().long().cuda()
        region_prediction = region_candidates(base_scores, regions)
        base_prediction = base_scores.argmax(dim=1)

        def run_refinement():
            return run_split_refiner(
                args,
                refiner,
                base_scores,
                base_feats,
                coords,
                features,
                regions,
                return_components=True,
            )

        (
            no_op_scores,
            refined_scores,
            _delta_scores,
            split_targets,
            delta_components,
            refine_mask,
            _keep_mask,
        ), split_refiner_ms = timed_call(run_refinement)
        no_op_prediction = no_op_scores.argmax(dim=1)

        def run_meta():
            return meta_adapt_refiner_gates(
                base_scores.detach(),
                {
                    name: value.detach()
                    for name, value in delta_components.items()
                },
                refined_scores.detach(),
                temporal_mean_probability.detach(),
                temporal_vote_count.detach(),
                region_prediction.detach(),
                no_op_prediction.detach(),
                base_prediction.detach(),
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
                classwise=False,
                refine_mask=refine_mask.detach(),
                adapt_bias=True,
                bias_regularization=args.meta_refiner_bias_reg,
            )

        (
            meta_probability,
            meta_accepted,
            meta_stats,
        ), meta_adaptation_ms = timed_call(run_meta)

        def run_verifier():
            refined_probability = F.softmax(refined_scores, dim=1)
            refined_prediction = refined_probability.argmax(dim=1)
            meta_prediction = meta_probability.argmax(dim=1)
            temporal_confidence, temporal_prediction = (
                temporal_mean_probability.max(dim=1)
            )
            point_accept = (
                (temporal_votes >= args.min_temporal_votes)
                & (temporal_confidence >= args.selection_threshold)
                & (temporal_prediction != base_prediction)
            )
            initial_poe_score = (
                (1.0 - args.meta_initial_weight)
                * torch.log(refined_probability.clamp_min(1e-6))
                + args.meta_initial_weight
                * torch.log(temporal_mean_probability.clamp_min(1e-6))
            )
            selected = initial_poe_score.argmax(dim=1)
            selected[point_accept] = temporal_prediction[point_accept]

            candidate_support = temporal_vote_count.gather(
                1, meta_prediction.unsqueeze(1)
            ).squeeze(1)
            candidate_confidence = temporal_mean_probability.gather(
                1, meta_prediction.unsqueeze(1)
            ).squeeze(1)
            candidate_changed = meta_prediction != refined_prediction
            candidate_accept = (
                (candidate_support >= args.min_temporal_votes)
                & (
                    candidate_confidence
                    >= args.meta_refiner_verify_threshold
                )
                & (
                    (meta_prediction == temporal_prediction)
                    | (meta_prediction == region_prediction)
                    | (meta_prediction == no_op_prediction)
                )
            )
            use_meta = (
                candidate_changed
                & candidate_accept
                & (selected == refined_prediction)
            )
            selected[use_meta] = meta_prediction[use_meta]
            return selected

        _final_prediction, verifier_ms = timed_call(run_verifier)
        torch.cuda.synchronize()
        total_ms = 1000.0 * (time.perf_counter() - scene_start)
        measured_stage_ms = (
            backbone_ms
            + temporal_ms
            + split_refiner_ms
            + meta_adaptation_ms
            + verifier_ms
        )
        peak_memory = torch.cuda.max_memory_allocated()
        peak_reserved_memory = torch.cuda.max_memory_reserved()
        meta_active = (
            meta_stats["support_corrections"] > 0
            and meta_stats["query_corrections"] > 0
        )
        record = {
            "scene_id": int(scene_id),
            "voxel_points": int(base_scores.size(0)),
            "backbone_ms": backbone_ms,
            "temporal_evidence_ms": temporal_ms,
            "split_refiner_ms": split_refiner_ms,
            "meta_adaptation_ms": meta_adaptation_ms,
            "verifier_ms": verifier_ms,
            "other_pipeline_ms": max(total_ms - measured_stage_ms, 0.0),
            "end_to_end_without_meta_ms": max(
                total_ms - meta_adaptation_ms,
                0.0,
            ),
            "end_to_end_ms": total_ms,
            "peak_memory_bytes": int(peak_memory),
            "peak_reserved_memory_bytes": int(peak_reserved_memory),
            "incremental_peak_memory_bytes": int(max(peak_memory - resident_memory, 0)),
            "meta_accepted": bool(meta_accepted),
            "meta_active": bool(meta_active),
        }
        if scene_id >= args.warmup_scenes:
            records.append(record)

    if not records:
        raise RuntimeError("No profiled scenes remain after warmup")

    timing_keys = [
        "backbone_ms",
        "temporal_evidence_ms",
        "split_refiner_ms",
        "meta_adaptation_ms",
        "verifier_ms",
        "other_pipeline_ms",
        "end_to_end_without_meta_ms",
        "end_to_end_ms",
    ]
    timing_summary = {
        key: summarize([record[key] for record in records])
        for key in timing_keys
    }
    memory_summary = {
        "model_resident_mib": float(resident_memory / (1024**2)),
        "mean_peak_allocated_mib": float(
            np.mean([record["peak_memory_bytes"] for record in records])
            / (1024**2)
        ),
        "max_peak_allocated_mib": float(
            max(record["peak_memory_bytes"] for record in records)
            / (1024**2)
        ),
        "max_incremental_peak_mib": float(
            max(record["incremental_peak_memory_bytes"] for record in records)
            / (1024**2)
        ),
        "max_peak_reserved_mib": float(
            max(record["peak_reserved_memory_bytes"] for record in records)
            / (1024**2)
        ),
    }
    active_meta_times = [
        record["meta_adaptation_ms"]
        for record in records
        if record["meta_active"]
    ]
    output = {
        "config": vars(args),
        "profiled_scenes": len(records),
        "mean_voxel_points": float(
            np.mean([record["voxel_points"] for record in records])
        ),
        "parameter_count": parameter_summary,
        "timing_ms_per_scene": timing_summary,
        "memory": memory_summary,
        "meta_accept_ratio": float(
            np.mean([record["meta_accepted"] for record in records])
        ),
        "meta_active_ratio": float(
            np.mean([record["meta_active"] for record in records])
        ),
        "meta_adaptation_active_ms": (
            summarize(active_meta_times) if active_meta_times else {}
        ),
        "records": records,
        "measurement_scope": {
            "model_loading": "excluded",
            "dataloader_and_disk_io": "excluded",
            "cuda_synchronization": "enabled around every stage",
            "historical_predictions": "computed online with three frozen references",
        },
    }
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)

    print(json.dumps(
        {
            "profiled_scenes": output["profiled_scenes"],
            "mean_voxel_points": output["mean_voxel_points"],
            "parameter_count": parameter_summary,
            "timing_ms_per_scene": timing_summary,
            "memory": memory_summary,
            "meta_accept_ratio": output["meta_accept_ratio"],
            "meta_active_ratio": output["meta_active_ratio"],
            "meta_adaptation_active_ms": output[
                "meta_adaptation_active_ms"
            ],
        },
        indent=2,
    ))
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
