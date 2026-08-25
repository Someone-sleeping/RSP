from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lib.stage3_pipeline import Stage3Config, reindex_split_regions, split_superpoints


@dataclass
class Stage2FeatureConfig:
    """Split-and-grow feature refinement used during GrowSP Stage 2."""

    residual_scale: float = 0.1
    query_scale: float = 10.0
    min_region_points: int = 20
    min_child_points: int = 8
    max_regions_per_scene: int = 20
    purity_threshold: float = 0.92
    entropy_threshold: float = 0.25
    min_split_confidence: float = 0.15
    verifier_tolerance: float = 0.02
    max_residual_norm: float = 5.0
    separation_weight: float = 0.25

    def split_config(self):
        return Stage3Config(
            query_scale=self.query_scale,
            min_region_points=self.min_region_points,
            min_child_points=self.min_child_points,
            max_regions_per_scene=self.max_regions_per_scene,
            purity_threshold=self.purity_threshold,
            entropy_threshold=self.entropy_threshold,
            min_split_confidence=self.min_split_confidence,
        )


@dataclass
class Stage2FeatureOutput:
    base_features: torch.Tensor
    refined_features: torch.Tensor
    residual_features: torch.Tensor
    dynamic_regions: torch.Tensor
    supervision_targets: torch.Tensor
    supervision_confidence: torch.Tensor
    supervision_mask: torch.Tensor
    candidate_mask: torch.Tensor
    accept_mask: torch.Tensor
    rollback_mask: torch.Tensor
    query_indices: torch.Tensor
    original_regions: torch.Tensor
    batch_ids: torch.Tensor
    cannot_link_pairs: torch.Tensor
    stats: dict


def _child_structure_score(features, parent_mask, targets, separation_weight):
    child_ids = torch.unique(targets[parent_mask & (targets >= 0)])
    if child_ids.numel() < 2:
        return features.new_tensor(float("-inf"))
    centers = []
    compactness = features.new_tensor(0.0)
    for child_id in child_ids:
        child_mask = parent_mask & (targets == child_id)
        child_features = F.normalize(features[child_mask], dim=1)
        center = F.normalize(child_features.mean(dim=0, keepdim=True), dim=1)
        centers.append(center)
        compactness = compactness + F.cosine_similarity(
            child_features, center.expand_as(child_features), dim=1
        ).mean()
    compactness = compactness / len(centers)
    separation = 1.0 - F.cosine_similarity(centers[0], centers[1], dim=1).mean()
    return compactness + float(separation_weight) * separation


def _verify_feature_splits(config, base_features, proposed_features, residual, split, regions, batch_ids):
    accepted_targets = split.targets.clone()
    accept_mask = torch.zeros_like(split.candidate_mask)
    rollback_mask = torch.zeros_like(split.candidate_mask)
    accepted_regions = 0
    rejected_regions = 0
    regions = regions.view(-1).long().to(base_features.device)
    batch_ids = batch_ids.view(-1).long().to(base_features.device)

    for batch_id in torch.unique(batch_ids):
        scene_mask = batch_ids == batch_id
        for region_id in torch.unique(regions[scene_mask]):
            if int(region_id.item()) == -1:
                continue
            parent_mask = scene_mask & (regions == region_id)
            candidate = parent_mask & (split.targets >= 0)
            if torch.unique(split.targets[candidate]).numel() < 2:
                continue
            base_score = _child_structure_score(
                base_features.detach(), parent_mask, split.targets, config.separation_weight
            )
            refined_score = _child_structure_score(
                proposed_features.detach(), parent_mask, split.targets, config.separation_weight
            )
            residual_norm = residual.detach()[candidate].norm(dim=1).mean()
            accepted = (
                refined_score + float(config.verifier_tolerance) >= base_score
                and residual_norm <= float(config.max_residual_norm)
            )
            if accepted:
                accept_mask[candidate] = True
                accepted_regions += 1
            else:
                rollback_mask[candidate] = True
                accepted_targets[candidate] = -1
                rejected_regions += 1

    return accepted_targets, accept_mask, rollback_mask, accepted_regions, rejected_regions


def run_stage2_feature_pipeline(
    config,
    feature_refiner,
    point_features,
    coordinates,
    colors,
    semantic_logits,
    regions,
    batch_ids,
):
    """Refine candidate features and commit only structurally safe splits."""
    split = split_superpoints(
        config.split_config(),
        point_features.detach(),
        coordinates,
        colors,
        semantic_logits.detach(),
        regions,
        batch_ids,
    )
    proposed_features, residual = feature_refiner.refine(
        point_features,
        coordinates,
        batch_ids,
        split.query_indices,
        split.candidate_mask,
        regions=split.dynamic_regions,
        residual_scale=config.residual_scale,
    )
    targets, accept_mask, rollback_mask, accepted, rejected = _verify_feature_splits(
        config,
        point_features,
        proposed_features,
        residual,
        split,
        regions,
        batch_ids,
    )
    refined_features = point_features.clone()
    refined_features[accept_mask] = proposed_features[accept_mask]
    dynamic_regions = reindex_split_regions(regions, batch_ids, targets)
    cannot_link_pairs = []
    flat_regions = regions.view(-1).long().to(point_features.device)
    flat_batch_ids = batch_ids.view(-1).long().to(point_features.device)
    for batch_id in torch.unique(flat_batch_ids):
        scene_mask = flat_batch_ids == batch_id
        for region_id in torch.unique(flat_regions[scene_mask]):
            parent = scene_mask & (flat_regions == region_id) & (targets >= 0)
            child_regions = torch.unique(dynamic_regions[parent])
            if child_regions.numel() == 2:
                cannot_link_pairs.append(child_regions)
    if cannot_link_pairs:
        cannot_link_pairs = torch.stack(cannot_link_pairs).long()
    else:
        cannot_link_pairs = torch.empty(
            (0, 2), dtype=torch.long, device=point_features.device
        )
    confidence = torch.zeros_like(split.target_confidence)
    confidence[accept_mask] = split.target_confidence[accept_mask]
    stats = {
        "selected_regions": split.statistics["split_candidate_regions"],
        "accepted_splits": accepted,
        "rejected_splits": rejected,
        "supervised_ratio": float(accept_mask.float().mean().item()),
    }
    return Stage2FeatureOutput(
        base_features=point_features,
        refined_features=refined_features,
        residual_features=residual,
        dynamic_regions=dynamic_regions,
        supervision_targets=targets,
        supervision_confidence=confidence,
        supervision_mask=accept_mask,
        candidate_mask=split.candidate_mask,
        accept_mask=accept_mask,
        rollback_mask=rollback_mask,
        query_indices=split.query_indices,
        original_regions=regions.view(-1).long(),
        batch_ids=batch_ids.view(-1).long(),
        cannot_link_pairs=cannot_link_pairs,
        stats=stats,
    )


def stage2_feature_losses(output, semantic_centers, semantic_scale=3.0):
    """Feature-space objectives for accepted child regions."""
    mask = output.supervision_mask
    zero = output.refined_features.sum() * 0.0
    if mask.any():
        logits = F.linear(F.normalize(output.refined_features[mask], dim=1), semantic_centers)
        point_loss = F.cross_entropy(
            logits * semantic_scale,
            output.supervision_targets[mask],
            reduction="none",
        )
        weights = output.supervision_confidence[mask].detach().clamp_min(1e-3)
        semantic_loss = (point_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        semantic_loss = zero

    structure_terms = []
    for batch_id in torch.unique(output.batch_ids):
        scene_mask = output.batch_ids == batch_id
        for region_id in torch.unique(output.original_regions[scene_mask]):
            parent_mask = scene_mask & (output.original_regions == region_id)
            if torch.unique(output.supervision_targets[parent_mask & mask]).numel() < 2:
                continue
            structure_terms.append(
                -_child_structure_score(
                    output.refined_features,
                    parent_mask,
                    output.supervision_targets,
                    separation_weight=0.25,
                )
            )
    structure_loss = torch.stack(structure_terms).mean() if structure_terms else zero
    residual_loss = (
        output.residual_features[output.candidate_mask].pow(2).mean()
        if output.candidate_mask.any()
        else zero
    )
    return {
        "semantic": semantic_loss,
        "structure": structure_loss,
        "residual": residual_loss,
    }


class Stage2FeatureModule:
    """Shared adapter for Stage-2 training-time and clustering-time refinement."""

    def __init__(self, feature_refiner, config):
        self.feature_refiner = feature_refiner
        self.config = config

    def eval(self):
        self.feature_refiner.eval()
        return self

    def __call__(self, point_features, coordinates, colors, semantic_logits, regions, batch_ids, **_):
        return run_stage2_feature_pipeline(
            self.config,
            self.feature_refiner,
            point_features,
            coordinates,
            colors,
            semantic_logits,
            regions,
            batch_ids,
        )
