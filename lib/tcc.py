from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F


@dataclass
class TCCSummary:
    mean_center_shift: float
    mean_effective_ratio: float
    mean_reliability: float
    refined_clusters: int


def _effective_sample_ratio(weights):
    return (1.0 / weights.square().sum().clamp_min(1e-12)) / weights.numel()


def refine_primitive_centers(
    features,
    primitive_labels,
    primitive_num,
    colors=None,
    feature_temperature=0.10,
    color_sigma=0.20,
    color_weight=0.25,
    center_strength=0.50,
    min_effective_ratio=0.25,
    loss_weight_strength=0.25,
    min_loss_weight=0.50,
):
    """Estimate robust primitive centers from feature and color reliability.

    Colors are expected in a stable scene-level range such as [-0.5, 0.5].
    The routine never uses semantic labels or ground truth.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    labels = torch.as_tensor(
        primitive_labels, dtype=torch.long, device=features.device
    ).view(-1)
    if labels.numel() != features.size(0):
        raise ValueError("primitive_labels must contain one label per feature")
    if colors is not None:
        colors = colors.to(features.device).float()
        if colors.shape != (features.size(0), 3):
            raise ValueError("colors must have shape [N, 3]")

    feature_temperature = max(float(feature_temperature), 1e-4)
    color_sigma = max(float(color_sigma), 1e-4)
    color_weight = min(max(float(color_weight), 0.0), 1.0)
    center_strength = min(max(float(center_strength), 0.0), 1.0)
    min_effective_ratio = min(max(float(min_effective_ratio), 0.0), 0.99)
    loss_weight_strength = min(max(float(loss_weight_strength), 0.0), 1.0)
    min_loss_weight = min(max(float(min_loss_weight), 0.0), 1.0)

    centers = features.new_zeros((int(primitive_num), features.size(1)))
    loss_weights = features.new_ones((int(primitive_num),))
    shifts = []
    effective_ratios = []
    reliabilities = []

    for primitive_id in range(int(primitive_num)):
        mask = labels == primitive_id
        if not mask.any():
            continue
        cluster_features = features[mask].float()
        normalized_features = F.normalize(cluster_features, dim=1)
        mean_center = F.normalize(
            normalized_features.mean(dim=0, keepdim=True), dim=1
        ).squeeze(0)

        feature_similarity = F.cosine_similarity(
            normalized_features, mean_center.unsqueeze(0), dim=1
        )
        feature_reliability = torch.exp(
            (feature_similarity - 1.0) / feature_temperature
        )

        if colors is None or color_weight == 0.0:
            color_reliability = torch.ones_like(feature_reliability)
        else:
            cluster_colors = colors[mask]
            color_center = cluster_colors.median(dim=0).values
            color_distance = torch.linalg.vector_norm(
                cluster_colors - color_center.unsqueeze(0), dim=1
            )
            color_reliability = torch.exp(
                -0.5 * (color_distance / color_sigma).square()
            )

        point_reliability = feature_reliability * (
            (1.0 - color_weight) + color_weight * color_reliability
        )
        weights = point_reliability.clamp_min(1e-8)
        weights = weights / weights.sum()
        effective_ratio = _effective_sample_ratio(weights)

        # Concentrated weights indicate that the evidence is too weak for a
        # large center move. Shrink smoothly toward the ordinary mean.
        support = (
            (effective_ratio - min_effective_ratio)
            / max(1.0 - min_effective_ratio, 1e-6)
        ).clamp(0.0, 1.0)
        adaptive_strength = center_strength * support
        robust_center = F.normalize(
            (weights.unsqueeze(1) * normalized_features).sum(
                dim=0, keepdim=True
            ),
            dim=1,
        ).squeeze(0)
        refined_center = F.normalize(
            (1.0 - adaptive_strength) * mean_center
            + adaptive_strength * robust_center,
            dim=0,
        )
        centers[primitive_id] = refined_center

        feature_quality = feature_similarity.mean().clamp(0.0, 1.0)
        color_quality = color_reliability.mean().clamp(0.0, 1.0)
        evidence_quality = (
            (1.0 - color_weight) * feature_quality
            + color_weight * color_quality
        )
        reliability = (0.75 * evidence_quality + 0.25 * effective_ratio).clamp(
            0.0, 1.0
        )
        loss_weights[primitive_id] = max(
            float(min_loss_weight),
            1.0 - loss_weight_strength * (1.0 - float(reliability.item())),
        )

        shifts.append(float((1.0 - torch.dot(mean_center, refined_center)).item()))
        effective_ratios.append(float(effective_ratio.item()))
        reliabilities.append(float(reliability.item()))

    count = len(shifts)
    summary = TCCSummary(
        mean_center_shift=sum(shifts) / max(count, 1),
        mean_effective_ratio=sum(effective_ratios) / max(count, 1),
        mean_reliability=sum(reliabilities) / max(count, 1),
        refined_clusters=count,
    )
    return F.normalize(centers, dim=1), loss_weights, summary


def load_primitive_reliability(args, device):
    if not (
        getattr(args, "tcc_enable", False)
        or getattr(args, "region_weight_enable", False)
    ):
        return None
    path = os.path.join(
        getattr(args, "pseudo_label_path", ""), "primitive_loss_weight.pt"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "TCC primitive reliability is missing: {}. Run clustering before "
            "the training epoch.".format(path)
        )
    return torch.load(path, map_location="cpu").to(device=device, dtype=torch.float32)
