import argparse
import colorsys
import json
import os
import re
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from lib.helper_ply import read_ply, write_ply


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
    parser = argparse.ArgumentParser("Original-point S3DIS qualitative comparison")
    parser.add_argument(
        "--scene_cache_dir",
        default="ckpt/cross_dataset/s3dis_area5_scenes",
    )
    parser.add_argument("--data_path", default="data/S3DIS/input")
    parser.add_argument(
        "--output_dir",
        default="ckpt/cross_dataset/s3dis_qualitative",
    )
    parser.add_argument("--num_scenes", type=int, default=4)
    return parser.parse_args()


def scene_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def room_type(name):
    match = re.match(r"^Area_\d+_(.+)_\d+$", name)
    return match.group(1) if match else name


def prediction_mapping(prediction, labels, classes=12):
    valid = (labels >= 0) & (labels < classes)
    histogram = np.bincount(
        classes * labels[valid] + prediction[valid],
        minlength=classes ** 2,
    ).reshape(classes, classes)
    match = linear_assignment(histogram.max() - histogram)
    mapping = np.arange(classes, dtype=np.int64)
    for label_id, prediction_id in match:
        mapping[int(prediction_id)] = int(label_id)
    return mapping


def meta_verified(cache):
    prediction = cache["meta"].copy()
    rollback = cache["decision"] == 2
    prediction[rollback] = cache["decomposition"][rollback]
    return prediction


def split_region_ids(cache):
    parent = cache["dynamic_regions"].astype(np.int64)
    target = cache["split_targets"].astype(np.int64)
    result = parent.copy()
    next_region = int(parent[parent >= 0].max()) + 1 if (parent >= 0).any() else 0
    for parent_id in np.unique(parent[parent >= 0]):
        mask = parent == parent_id
        child_targets = np.unique(target[mask & (target >= 0)])
        if child_targets.size <= 1:
            continue
        for child_target in child_targets[1:]:
            result[mask & (target == child_target)] = next_region
            next_region += 1
    return result


def region_colors(region_ids):
    colors = np.full((len(region_ids), 3), 205, dtype=np.uint8)
    for region_id in np.unique(region_ids[region_ids >= 0]):
        hue = (int(region_id) * 0.61803398875) % 1.0
        saturation = 0.58 + 0.16 * ((int(region_id) % 3) / 2.0)
        value = 0.72 + 0.20 * ((int(region_id) % 5) / 4.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[region_ids == region_id] = np.asarray(rgb) * 255
    return colors


def projected_view(xyz):
    return np.stack(
        [xyz[:, 0] + 0.32 * xyz[:, 1], xyz[:, 2] + 0.13 * xyz[:, 1]],
        axis=1,
    )


def draw_points(axis, xyz, colors, title):
    if len(xyz) > 25000:
        sample = np.linspace(0, len(xyz) - 1, 25000, dtype=np.int64)
        xyz = xyz[sample]
        colors = colors[sample]
    view = projected_view(xyz)
    order = np.argsort(xyz[:, 1])
    size = float(np.clip(75000.0 / max(len(xyz), 1), 0.22, 1.35))
    axis.scatter(
        view[order, 0],
        view[order, 1],
        c=colors[order].astype(np.float32) / 255.0,
        s=size,
        linewidths=0,
        rasterized=True,
    )
    axis.set_title(title, fontsize=9, fontweight="bold", pad=4)
    axis.set_aspect("equal")
    axis.axis("off")


def diagnostic_colors(base, refined, labels):
    valid = (labels >= 0) & (labels < len(CLASS_COLORS))
    colors = np.full((len(labels), 3), 205, dtype=np.uint8)
    changed = valid & (base != refined)
    corrected = changed & (base != labels) & (refined == labels)
    harmed = changed & (base == labels) & (refined != labels)
    colors[changed] = np.asarray([54, 115, 190], dtype=np.uint8)
    colors[corrected] = np.asarray([38, 160, 92], dtype=np.uint8)
    colors[harmed] = np.asarray([211, 67, 62], dtype=np.uint8)
    return colors, corrected, harmed, changed


def load_scenes(args):
    scenes = []
    for cache_path in sorted(glob(os.path.join(args.scene_cache_dir, "*.npz"))):
        name = scene_name(cache_path)
        cache = np.load(cache_path)
        raw = read_ply(os.path.join(args.data_path, f"{name}.ply"))
        xyz = np.vstack([raw["x"], raw["y"], raw["z"]]).T.astype(np.float32)
        rgb = np.vstack([raw["red"], raw["green"], raw["blue"]]).T.astype(np.uint8)
        inverse = cache["inverse_map"].astype(np.int64)
        labels = cache["original_labels"].astype(np.int64)
        if len(xyz) != len(labels) or len(inverse) != len(labels):
            raise ValueError(f"Original point mapping mismatch for {name}")
        scenes.append(
            {
                "name": name,
                "cache": cache,
                "xyz": xyz,
                "rgb": rgb,
                "labels": labels,
                "inverse": inverse,
                "base": cache["base"][inverse],
                "verified": meta_verified(cache)[inverse],
                "initial_regions": cache["initial_regions"][inverse],
                "split_regions": split_region_ids(cache)[inverse],
            }
        )
    return scenes


def select_scenes(scenes, mapping, count):
    for scene in scenes:
        valid = (scene["labels"] >= 0) & (scene["labels"] < len(mapping))
        mapped_base = mapping[scene["base"]]
        mapped_verified = mapping[scene["verified"]]
        scene["mapped_base"] = mapped_base
        scene["mapped_verified"] = mapped_verified
        scene["base_acc"] = float((mapped_base[valid] == scene["labels"][valid]).mean())
        scene["verified_acc"] = float((mapped_verified[valid] == scene["labels"][valid]).mean())
        scene["gain"] = scene["verified_acc"] - scene["base_acc"]
        scene["change_ratio"] = float((mapped_base[valid] != mapped_verified[valid]).mean())
        scene["split_ratio"] = float(
            (scene["initial_regions"] != scene["split_regions"]).mean()
        )
    ranked = sorted(
        scenes,
        key=lambda item: (item["gain"], item["change_ratio"]),
        reverse=True,
    )
    selected = []
    used_types = set()
    for scene in ranked:
        current_type = room_type(scene["name"])
        if current_type in used_types or scene["change_ratio"] <= 0:
            continue
        selected.append(scene)
        used_types.add(current_type)
        if len(selected) >= count:
            return selected
    selected_names = {scene["name"] for scene in selected}
    for scene in ranked:
        if scene["name"] not in selected_names:
            selected.append(scene)
            selected_names.add(scene["name"])
        if len(selected) >= count:
            break
    return selected


def write_colored_cloud(path, xyz, colors, values, value_name):
    write_ply(
        path,
        [xyz.astype(np.float32), colors.astype(np.uint8), values.astype(np.int32)],
        ["x", "y", "z", "red", "green", "blue", value_name],
    )


def export_scene(output_dir, scene):
    directory = os.path.join(output_dir, scene["name"])
    os.makedirs(directory, exist_ok=True)
    gt_colors = np.full((len(scene["labels"]), 3), 205, dtype=np.uint8)
    valid = (scene["labels"] >= 0) & (scene["labels"] < len(CLASS_COLORS))
    gt_colors[valid] = CLASS_COLORS[scene["labels"][valid]]
    diagnostic, corrected, harmed, changed = diagnostic_colors(
        scene["mapped_base"], scene["mapped_verified"], scene["labels"]
    )
    variants = {
        "original_rgb": (scene["rgb"], np.zeros(len(valid)), "rgb_source"),
        "ground_truth": (gt_colors, scene["labels"], "label"),
        "frozen_prediction": (
            CLASS_COLORS[scene["mapped_base"]],
            scene["mapped_base"],
            "prediction",
        ),
        "initial_superpoints": (
            region_colors(scene["initial_regions"]),
            scene["initial_regions"],
            "region",
        ),
        "split_superpoints": (
            region_colors(scene["split_regions"]),
            scene["split_regions"],
            "region",
        ),
        "verified_refinement": (
            CLASS_COLORS[scene["mapped_verified"]],
            scene["mapped_verified"],
            "prediction",
        ),
        "difference_diagnostic": (
            diagnostic,
            changed.astype(np.int32),
            "changed_01",
        ),
    }
    for name, (colors, values, value_name) in variants.items():
        write_colored_cloud(
            os.path.join(directory, f"{name}.ply"),
            scene["xyz"],
            colors,
            np.asarray(values),
            value_name,
        )
    return {
        "corrected_ratio": float(corrected.mean()),
        "harmed_ratio": float(harmed.mean()),
        "changed_ratio": float(changed.mean()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scenes = load_scenes(args)
    all_labels = np.concatenate([scene["labels"] for scene in scenes])
    all_base = np.concatenate([scene["base"] for scene in scenes])
    mapping = prediction_mapping(all_base, all_labels)
    selected = select_scenes(scenes, mapping, max(args.num_scenes, 1))

    figure, axes = plt.subplots(
        len(selected),
        7,
        figsize=(20.5, 2.55 * len(selected)),
        dpi=140,
    )
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)
    summary = []
    for row, scene in enumerate(selected):
        gt_colors = np.full((len(scene["labels"]), 3), 205, dtype=np.uint8)
        valid = (scene["labels"] >= 0) & (scene["labels"] < len(CLASS_COLORS))
        gt_colors[valid] = CLASS_COLORS[scene["labels"][valid]]
        diagnostic, _corrected, _harmed, _changed = diagnostic_colors(
            scene["mapped_base"], scene["mapped_verified"], scene["labels"]
        )
        panels = (
            (scene["rgb"], "Original RGB"),
            (gt_colors, "Ground truth"),
            (CLASS_COLORS[scene["mapped_base"]], "Frozen prediction"),
            (region_colors(scene["initial_regions"]), "Initial superpoints"),
            (region_colors(scene["split_regions"]), "Split superpoints"),
            (CLASS_COLORS[scene["mapped_verified"]], "Verified refinement"),
            (diagnostic, "Difference diagnostic"),
        )
        for column, (colors, title) in enumerate(panels):
            draw_points(axes[row, column], scene["xyz"], colors, title)
        axes[row, 0].text(
            -0.04,
            0.5,
            f"{scene['name']}\npoint Acc {100.0 * scene['gain']:+.2f} pp",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
            fontweight="bold",
        )
        diagnostic_stats = export_scene(args.output_dir, scene)
        summary.append(
            {
                "scene": scene["name"],
                "room_type": room_type(scene["name"]),
                "original_points": int(len(scene["xyz"])),
                "base_point_accuracy": scene["base_acc"],
                "verified_point_accuracy": scene["verified_acc"],
                "gain_percentage_points": 100.0 * scene["gain"],
                "prediction_change_ratio": scene["change_ratio"],
                **diagnostic_stats,
            }
        )
    figure.suptitle(
        "S3DIS Area 5 Original-Point Qualitative Comparison",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.995,
        0.006,
        "Difference: blue = changed, green = corrected, red = harmed",
        ha="right",
        fontsize=8,
    )
    figure.tight_layout(rect=[0.025, 0.018, 1, 0.965], w_pad=0.2, h_pad=0.5)
    figure_path = os.path.join(args.output_dir, "qualitative_comparison.png")
    figure.savefig(figure_path, facecolor="white")
    plt.close(figure)
    with open(
        os.path.join(args.output_dir, "qualitative_comparison.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "semantic_mapping": mapping.tolist(),
                "legend": {
                    "changed": "blue",
                    "corrected": "green",
                    "harmed": "red",
                },
                "selected_scenes": summary,
            },
            output_file,
            indent=2,
        )
    print(json.dumps(summary, indent=2))
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
