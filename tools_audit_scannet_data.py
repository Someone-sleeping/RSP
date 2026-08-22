#!/usr/bin/env python3
"""Audit the local ScanNet preprocessing output used by GrowSP."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from lib.helper_ply import read_ply


EXPECTED_FIELDS = {"x", "y", "z", "red", "green", "blue", "class"}
VALID_LABELS = set(range(20)) | {-1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/ScanNet/processed")
    parser.add_argument("--sp_path", default="data/ScanNet/initial_superpoints")
    parser.add_argument(
        "--train_split", default="data_prepare/ScanNet_splits/scannetv2_train.txt"
    )
    parser.add_argument(
        "--val_split", default="data_prepare/ScanNet_splits/scannetv2_val.txt"
    )
    parser.add_argument("--output", default="ckpt/ScanNet/data_audit.json")
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=0,
        help="Audit at most this many listed scenes; zero audits every scene.",
    )
    return parser.parse_args()


def read_split(path):
    return [
        Path(line.strip()).stem[:12]
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    sp_path = Path(args.sp_path)
    split_names = {
        "train": read_split(args.train_split),
        "val": read_split(args.val_split),
    }
    listed = split_names["train"] + split_names["val"]
    selected = listed[: args.max_scenes] if args.max_scenes > 0 else listed

    errors = []
    warnings = []
    label_counts = Counter()
    total_points = 0
    ignored_points = 0
    superpoint_counts = []

    duplicate_train = sorted(
        name for name, count in Counter(split_names["train"]).items() if count > 1
    )
    duplicate_val = sorted(
        name for name, count in Counter(split_names["val"]).items() if count > 1
    )
    overlap = sorted(set(split_names["train"]) & set(split_names["val"]))
    if duplicate_train:
        errors.append(f"duplicate train scenes: {duplicate_train[:10]}")
    if duplicate_val:
        errors.append(f"duplicate val scenes: {duplicate_val[:10]}")
    if overlap:
        errors.append(f"train/val overlap: {overlap[:10]}")

    for scene_name in selected:
        ply_path = data_path / f"{scene_name}.ply"
        superpoint_path = sp_path / f"{scene_name}_superpoint.npy"
        if not ply_path.exists():
            errors.append(f"missing point cloud: {ply_path}")
            continue
        if not superpoint_path.exists():
            errors.append(f"missing superpoints: {superpoint_path}")
            continue
        try:
            points = read_ply(str(ply_path))
            superpoints = np.load(superpoint_path, mmap_mode="r")
        except Exception as exc:
            errors.append(f"failed to read {scene_name}: {exc}")
            continue

        fields = set(points.dtype.names or ())
        missing_fields = EXPECTED_FIELDS - fields
        if missing_fields:
            errors.append(f"{scene_name}: missing PLY fields {sorted(missing_fields)}")
            continue
        if len(points) != len(superpoints):
            errors.append(
                f"{scene_name}: point/superpoint length mismatch "
                f"{len(points)} != {len(superpoints)}"
            )
            continue

        xyz = np.column_stack([points[name] for name in ("x", "y", "z")])
        rgb = np.column_stack([points[name] for name in ("red", "green", "blue")])
        labels = np.asarray(points["class"])
        rounded_labels = np.rint(labels).astype(np.int64)
        unique_labels, counts = np.unique(rounded_labels, return_counts=True)
        label_counts.update(dict(zip(unique_labels.tolist(), counts.tolist())))
        total_points += len(points)
        ignored_points += int((rounded_labels == -1).sum())
        superpoint_counts.append(int(np.unique(superpoints[superpoints >= 0]).size))

        invalid_labels = sorted(set(unique_labels.tolist()) - VALID_LABELS)
        if not np.isfinite(xyz).all():
            errors.append(f"{scene_name}: non-finite XYZ values")
        if not np.isfinite(labels).all() or not np.allclose(labels, rounded_labels):
            errors.append(f"{scene_name}: non-finite or non-integral labels")
        if invalid_labels:
            errors.append(f"{scene_name}: invalid labels {invalid_labels}")
        if rgb.min() < 0 or rgb.max() > 255:
            errors.append(f"{scene_name}: RGB outside [0, 255]")
        if superpoints.ndim != 1 or not np.issubdtype(superpoints.dtype, np.integer):
            errors.append(f"{scene_name}: invalid superpoint array {superpoints.shape}/{superpoints.dtype}")
        if np.any(superpoints < -1):
            errors.append(f"{scene_name}: superpoint IDs below -1")

    processed_names = {path.stem for path in data_path.glob("*.ply")}
    listed_names = set(listed)
    extra_processed = sorted(processed_names - listed_names)
    missing_processed = sorted(listed_names - processed_names)
    if extra_processed:
        warnings.append(
            f"{len(extra_processed)} processed scenes are outside train/val lists "
            "(normally ScanNet test scenes and ignored by the loaders)."
        )

    report = {
        "status": "pass" if not errors else "fail",
        "paths": {
            "data": str(data_path.resolve()),
            "superpoints": str(sp_path.resolve()),
        },
        "split": {
            "train_scenes": len(split_names["train"]),
            "val_scenes": len(split_names["val"]),
            "overlap": overlap,
            "missing_processed": missing_processed,
            "extra_processed_count": len(extra_processed),
            "extra_processed_examples": extra_processed[:10],
        },
        "audited_scenes": len(selected),
        "total_points": total_points,
        "ignored_points": ignored_points,
        "ignored_ratio": ignored_points / max(total_points, 1),
        "label_counts": {str(key): label_counts[key] for key in sorted(label_counts)},
        "superpoints_per_scene": {
            "min": min(superpoint_counts) if superpoint_counts else None,
            "mean": float(np.mean(superpoint_counts)) if superpoint_counts else None,
            "max": max(superpoint_counts) if superpoint_counts else None,
        },
        "errors": errors,
        "warnings": warnings,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
