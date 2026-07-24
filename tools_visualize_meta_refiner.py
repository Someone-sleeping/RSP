import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from lib.helper_ply import write_ply
from lib.meta_refiner import meta_adapt_refiner_gates
from models.query_refiner import ErrorQueryRefiner
from tools_eval_error_verifier import (
    load_reference,
    parse_areas,
    parse_int_list,
    region_candidates,
    run_split_refiner,
)


CLASS_COLORS = np.asarray(
    [
        [170, 170, 170],
        [95, 95, 95],
        [126, 174, 214],
        [116, 92, 65],
        [190, 132, 76],
        [88, 160, 205],
        [194, 112, 100],
        [65, 151, 91],
        [221, 166, 56],
        [164, 100, 177],
        [141, 121, 76],
        [80, 133, 190],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser("Visualize Meta-Refiner qualitative results")
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
    parser.add_argument(
        "--output_dir",
        default="ckpt/S3DIS/meta_refiner/qualitative",
    )
    parser.add_argument("--selection", choices=["success", "mixed", "failure"], default="mixed")
    parser.add_argument("--num_scenes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
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
    return parser.parse_args()


def label_mapping(prediction, label, num_classes):
    valid = (label >= 0) & (label < num_classes)
    histogram = np.bincount(
        num_classes * label[valid] + prediction[valid],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    match = linear_assignment(histogram.max() - histogram)
    mapping = np.arange(num_classes, dtype=np.int64)
    for ground_truth_id, prediction_id in match:
        mapping[int(prediction_id)] = int(ground_truth_id)
    return mapping


def mapped_prediction(prediction, mapping):
    return mapping[prediction.clip(0, len(mapping) - 1)]


def point_accuracy(prediction, label):
    valid = (label >= 0) & (label < len(CLASS_COLORS))
    return float((prediction[valid] == label[valid]).mean())


def infer_scene(args, models, refiner, batch, scene_name):
    base_model, base_centers, temporal_references = models
    coords, features, inverse_map, labels, _index, region = batch
    in_field = ME.TensorField(features, coords, device=0)
    with torch.no_grad():
        base_feats = F.normalize(base_model(in_field), dim=1)
        base_scores = F.linear(base_feats, base_centers)
        temporal_probabilities = []
        for _epoch, model, centers in temporal_references:
            temporal_feats = F.normalize(model(in_field), dim=1)
            temporal_scores = F.linear(temporal_feats, centers)
            temporal_probabilities.append(
                F.softmax(temporal_scores * args.prob_scale, dim=1)
            )
        temporal_probabilities = torch.stack(temporal_probabilities, dim=0)

    regions = region.squeeze().long().cuda()
    region_prediction = region_candidates(base_scores, regions)
    base_prediction = base_scores.argmax(dim=1)
    temporal_mean_probability = temporal_probabilities.mean(dim=0)
    temporal_predictions = temporal_probabilities.argmax(dim=2)
    temporal_vote_count = F.one_hot(
        temporal_predictions,
        num_classes=args.semantic_class,
    ).sum(dim=0)

    (
        no_op_scores,
        refined_scores,
        _delta_scores,
        split_targets,
        delta_components,
        refine_mask,
        _keep_mask,
    ) = run_split_refiner(
        args,
        refiner,
        base_scores,
        base_feats,
        coords,
        features,
        regions,
        return_components=True,
    )
    no_op_prediction = no_op_scores.argmax(dim=1)
    meta_probability, meta_accepted, meta_stats = meta_adapt_refiner_gates(
        base_scores.detach(),
        {name: value.detach() for name, value in delta_components.items()},
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

    inverse = inverse_map.long().cuda()
    valid = (labels >= 0) & (labels < args.semantic_class)
    full_xyz = coords[:, 1:].cpu().numpy()[inverse_map.long().numpy()]
    full_rgb = np.clip(
        (features[:, :3].cpu().numpy()[inverse_map.long().numpy()] + 0.5)
        * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return {
        "name": scene_name,
        "xyz": full_xyz[valid.numpy()],
        "rgb": full_rgb[valid.numpy()],
        "label": labels[valid].numpy(),
        "base": base_prediction[inverse][valid.cuda()].cpu().numpy(),
        "split_refiner": refined_scores.argmax(dim=1)[inverse][valid.cuda()].cpu().numpy(),
        "meta": meta_probability.argmax(dim=1)[inverse][valid.cuda()].cpu().numpy(),
        "refine_mask": refine_mask[inverse][valid.cuda()].cpu().numpy().astype(bool),
        "split_mask": (split_targets >= 0)[inverse][valid.cuda()].cpu().numpy().astype(bool),
        "split_targets": split_targets[inverse][valid.cuda()].cpu().numpy(),
        "meta_accepted": bool(meta_accepted),
        "meta_stats": meta_stats,
    }


def scene_score(scene, selection):
    xyz_range = scene["xyz"].max(axis=0) - scene["xyz"].min(axis=0)
    aspect = xyz_range[:2].max() / max(float(xyz_range[:2].min()), 1e-6)
    visual_penalty = max(aspect - 3.5, 0.0)
    structure = min(float(scene["split_mask"].mean()) * 10.0, 1.0)
    meta_change = min(scene["meta_changed_ratio"] * 100.0, 1.0)
    if selection == "failure":
        return -100.0 * scene["meta_gain"] + 0.2 * structure + meta_change - visual_penalty
    return 100.0 * scene["meta_gain"] + 0.2 * structure + meta_change - visual_penalty


def select_scenes(scenes, selection, count):
    count = max(int(count), 1)
    candidates = [
        scene
        for scene in scenes
        if (
            scene["xyz"].shape[0] > 2000
            and scene["split_mask"].any()
            and scene["meta_changed_ratio"] > 0
        )
    ]
    if not candidates:
        candidates = scenes
    if selection != "mixed":
        return sorted(
            candidates,
            key=lambda scene: scene_score(scene, selection),
            reverse=True,
        )[:count]

    positive_count = (count + 1) // 2
    negative_count = count - positive_count
    positive = sorted(
        candidates,
        key=lambda scene: scene_score(scene, "success"),
        reverse=True,
    )
    negative = sorted(
        candidates,
        key=lambda scene: scene_score(scene, "failure"),
        reverse=True,
    )
    selected = []
    selected_ids = set()
    for pool, target in ((positive, positive_count), (negative, negative_count)):
        added = 0
        for scene in pool:
            if id(scene) in selected_ids:
                continue
            selected.append(scene)
            selected_ids.add(id(scene))
            added += 1
            if added >= target:
                break
    return selected[:count]


def projected_view(xyz):
    return np.stack(
        [xyz[:, 0] + 0.32 * xyz[:, 1], xyz[:, 2] + 0.13 * xyz[:, 1]],
        axis=1,
    )


def draw_points(axis, xyz, colors, title, overlay=None):
    view = projected_view(xyz)
    order = np.argsort(xyz[:, 1])
    size = float(np.clip(70000.0 / max(len(xyz), 1), 0.25, 1.4))
    axis.scatter(
        view[order, 0],
        view[order, 1],
        c=colors[order].astype(np.float32) / 255.0,
        s=size,
        linewidths=0,
        rasterized=True,
    )
    if overlay is not None and overlay.any():
        overlay_view = view[overlay]
        axis.scatter(
            overlay_view[:, 0],
            overlay_view[:, 1],
            facecolors="none",
            edgecolors="#16a36a",
            s=max(size * 2.5, 1.0),
            linewidths=0.25,
            alpha=0.7,
            rasterized=True,
        )
    axis.set_title(title, fontsize=10, fontweight="bold", pad=5)
    axis.set_aspect("equal")
    axis.axis("off")


def difference_colors(scene, mapping):
    colors = np.full((len(scene["xyz"]), 3), 205, dtype=np.uint8)
    suspicious = scene["refine_mask"]
    colors[suspicious] = np.asarray([224, 90, 70], dtype=np.uint8)
    split = scene["split_mask"]
    if split.any():
        split_targets = scene["split_targets"][split].clip(0, len(mapping) - 1)
        mapped_targets = mapping[split_targets]
        colors[split] = CLASS_COLORS[mapped_targets]
    return colors


def export_scene(output_dir, scene, mapping):
    scene_dir = os.path.join(output_dir, scene["name"].lstrip("/"))
    os.makedirs(scene_dir, exist_ok=True)
    variants = {
        "frozen_prediction": mapped_prediction(scene["base"], mapping),
        "split_refiner": mapped_prediction(scene["split_refiner"], mapping),
        "meta_adaptation": mapped_prediction(scene["meta"], mapping),
    }
    for name, prediction in variants.items():
        colors = CLASS_COLORS[prediction]
        write_ply(
            os.path.join(scene_dir, f"{name}.ply"),
            [scene["xyz"].astype(np.float32), colors, prediction.astype(np.int32)],
            ["x", "y", "z", "red", "green", "blue", "prediction"],
        )
    difference = difference_colors(scene, mapping)
    write_ply(
        os.path.join(scene_dir, "semantic_difference_split.ply"),
        [
            scene["xyz"].astype(np.float32),
            difference,
            scene["refine_mask"].astype(np.int32),
            scene["split_mask"].astype(np.int32),
        ],
        [
            "x",
            "y",
            "z",
            "red",
            "green",
            "blue",
            "suspicious",
            "split",
        ],
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    reference_epochs = parse_int_list(args.reference_epochs)
    base_model, base_centers = load_reference(args, args.base_epoch)
    temporal_references = []
    for epoch in reference_epochs:
        model, centers = load_reference(args, epoch, reference_centers=base_centers)
        temporal_references.append((epoch, model, centers))
    models = (base_model, base_centers, temporal_references)

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

    dataset = S3DIStest(args, areas=parse_areas(args.test_area))
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=cfl_collate_fn_test(),
        num_workers=args.workers,
        pin_memory=True,
    )
    scenes = []
    for batch in loader:
        scene_index = int(batch[4][0])
        scenes.append(
            infer_scene(args, models, refiner, batch, dataset.name[scene_index])
        )

    all_labels = np.concatenate([scene["label"] for scene in scenes])
    all_base = np.concatenate([scene["base"] for scene in scenes])
    mapping = label_mapping(all_base, all_labels, args.semantic_class)
    for scene in scenes:
        mapped_base = mapped_prediction(scene["base"], mapping)
        mapped_refiner = mapped_prediction(scene["split_refiner"], mapping)
        mapped_meta = mapped_prediction(scene["meta"], mapping)
        scene["base_acc"] = point_accuracy(mapped_base, scene["label"])
        scene["refiner_acc"] = point_accuracy(mapped_refiner, scene["label"])
        scene["meta_acc"] = point_accuracy(mapped_meta, scene["label"])
        scene["refiner_gain"] = scene["refiner_acc"] - scene["base_acc"]
        scene["meta_gain"] = scene["meta_acc"] - scene["refiner_acc"]
        scene["meta_changed_ratio"] = float(
            (scene["meta"] != scene["split_refiner"]).mean()
        )

    selected = select_scenes(scenes, args.selection, args.num_scenes)
    rows = len(selected)
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(13.2, 2.45 * rows),
        dpi=240,
    )
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, scene in enumerate(selected):
        mapped_base = mapped_prediction(scene["base"], mapping)
        mapped_refiner = mapped_prediction(scene["split_refiner"], mapping)
        mapped_meta = mapped_prediction(scene["meta"], mapping)
        draw_points(
            axes[row, 0],
            scene["xyz"],
            CLASS_COLORS[mapped_base],
            "Frozen prediction",
        )
        draw_points(
            axes[row, 1],
            scene["xyz"],
            difference_colors(scene, mapping),
            "Semantic difference & split",
        )
        draw_points(
            axes[row, 2],
            scene["xyz"],
            CLASS_COLORS[mapped_refiner],
            "Split + Refiner",
        )
        draw_points(
            axes[row, 3],
            scene["xyz"],
            CLASS_COLORS[mapped_meta],
            "Meta adaptation",
            overlay=scene["meta"] != scene["split_refiner"],
        )
        axes[row, 0].text(
            -0.04,
            0.5,
            (
                f"{scene['name'].lstrip('/')}\n"
                f"Refiner {100.0 * scene['refiner_gain']:+.2f} pp\n"
                f"Meta {100.0 * scene['meta_gain']:+.2f} pp"
            ),
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
            fontweight="bold",
        )
        export_scene(args.output_dir, scene, mapping)

    figure.suptitle(
        "S3DIS Area 5: Semantic Difference-Guided Refinement",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0.025, 0, 1, 0.965], w_pad=0.25, h_pad=0.55)
    figure_path = os.path.join(
        args.output_dir,
        f"qualitative_{args.selection}_{rows}.png",
    )
    figure.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    summary = []
    for scene in selected:
        summary.append(
            {
                "scene": scene["name"],
                "points": int(len(scene["xyz"])),
                "base_point_accuracy": scene["base_acc"],
                "split_refiner_point_accuracy": scene["refiner_acc"],
                "meta_point_accuracy": scene["meta_acc"],
                "refiner_gain_percentage_points": 100.0 * scene["refiner_gain"],
                "meta_gain_percentage_points": 100.0 * scene["meta_gain"],
                "suspicious_ratio": float(scene["refine_mask"].mean()),
                "split_ratio": float(scene["split_mask"].mean()),
                "meta_changed_ratio": float(
                    scene["meta_changed_ratio"]
                ),
                "meta_accepted": scene["meta_accepted"],
                "meta_query_gain": scene["meta_stats"]["query_gain"],
            }
        )
    summary_path = os.path.join(
        args.output_dir,
        f"qualitative_{args.selection}_{rows}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "selection": args.selection,
                "ground_truth_usage": (
                    "Evaluation-only global class alignment and scene analysis; "
                    "not used by prediction or adaptation."
                ),
                "scenes": summary,
            },
            summary_file,
            indent=2,
        )
    all_scene_path = os.path.join(args.output_dir, "all_scene_analysis.json")
    with open(all_scene_path, "w", encoding="utf-8") as all_scene_file:
        json.dump(
            [
                {
                    "scene": scene["name"],
                    "points": int(len(scene["xyz"])),
                    "base_point_accuracy": scene["base_acc"],
                    "split_refiner_point_accuracy": scene["refiner_acc"],
                    "meta_point_accuracy": scene["meta_acc"],
                    "refiner_gain_percentage_points": 100.0
                    * scene["refiner_gain"],
                    "meta_gain_percentage_points": 100.0 * scene["meta_gain"],
                    "meta_changed_ratio": scene["meta_changed_ratio"],
                    "meta_accepted": scene["meta_accepted"],
                    "meta_query_gain": scene["meta_stats"]["query_gain"],
                }
                for scene in sorted(
                    scenes,
                    key=lambda item: item["meta_gain"],
                    reverse=True,
                )
            ],
            all_scene_file,
            indent=2,
        )
    print(json.dumps(summary, indent=2))
    print(f"Saved {figure_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {all_scene_path}")


if __name__ == "__main__":
    main()
