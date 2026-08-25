from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lib.split_regions import build_split_region_queries
from models.query_refiner import delta_l2, refinement_keep_kl


@dataclass
class Stage3Config:
    """Configuration for the training-integrated third stage."""

    residual_scale: float = 1.0
    query_scale: float = 10.0
    min_region_points: int = 20
    min_child_points: int = 8
    max_regions_per_scene: int = 20
    purity_threshold: float = 0.92
    entropy_threshold: float = 0.25
    min_split_confidence: float = 0.15
    xyz_weight: float = 1.0
    rgb_weight: float = 0.5
    feature_weight: float = 0.25
    semantic_weight: float = 1.0
    verifier_tolerance: float = 0.02
    max_residual_norm: float = 5.0


@dataclass
class SuperpointSplitOutput:
    query_indices: torch.Tensor
    candidate_mask: torch.Tensor
    keep_mask: torch.Tensor
    targets: torch.Tensor
    target_confidence: torch.Tensor
    dynamic_regions: torch.Tensor
    statistics: dict

    @property
    def supervision_mask(self):
        return self.targets >= 0


@dataclass
class CandidateRefinementOutput:
    base_logits: torch.Tensor
    refined_logits: torch.Tensor
    residual_logits: torch.Tensor


@dataclass
class VerificationOutput:
    verified_logits: torch.Tensor
    accept_mask: torch.Tensor
    rollback_mask: torch.Tensor
    keep_mask: torch.Tensor
    supervision_targets: torch.Tensor
    supervision_confidence: torch.Tensor
    statistics: dict

    @property
    def supervision_mask(self):
        return self.supervision_targets >= 0


@dataclass
class Stage3Output:
    split: SuperpointSplitOutput
    refinement: CandidateRefinementOutput
    verification: VerificationOutput


def reindex_split_regions(regions, batch_ids, split_targets):
    """Turn accepted child targets into contiguous superpoint identifiers."""
    regions = regions.view(-1).long()
    batch_ids = batch_ids.view(-1).long().to(regions.device)
    split_targets = split_targets.view(-1).long().to(regions.device)
    dynamic_regions = torch.full_like(regions, -1)
    next_region = 0

    for batch_id in torch.unique(batch_ids):
        scene_mask = batch_ids == batch_id
        for region_id in torch.unique(regions[scene_mask]):
            if int(region_id.item()) == -1:
                continue
            parent_mask = scene_mask & (regions == region_id)
            child_targets = torch.unique(split_targets[parent_mask & (split_targets >= 0)])
            if child_targets.numel() < 2:
                dynamic_regions[parent_mask] = next_region
                next_region += 1
                continue

            assigned = torch.zeros_like(parent_mask)
            for target in child_targets:
                child_mask = parent_mask & (split_targets == target)
                dynamic_regions[child_mask] = next_region
                assigned |= child_mask
                next_region += 1
            # Defensive fallback for points omitted by a future split proposal.
            remainder = parent_mask & ~assigned
            if remainder.any():
                dynamic_regions[remainder] = next_region
                next_region += 1
    return dynamic_regions


def split_superpoints(config, point_features, coordinates, colors, semantic_logits, regions, batch_ids):
    """Discover and structurally validate over-merged superpoints without GT."""
    queries, candidate_mask, targets, confidence, keep_mask, statistics = (
        build_split_region_queries(
            logits=semantic_logits * float(config.query_scale),
            point_feats=point_features,
            coords=coordinates,
            colors=colors,
            regions=regions,
            batch_ids=batch_ids,
            min_region_points=config.min_region_points,
            min_child_points=config.min_child_points,
            max_split_regions_per_scene=config.max_regions_per_scene,
            split_purity_threshold=config.purity_threshold,
            split_entropy_threshold=config.entropy_threshold,
            split_min_conf=config.min_split_confidence,
            xyz_weight=config.xyz_weight,
            rgb_weight=config.rgb_weight,
            feat_weight=config.feature_weight,
            semantic_weight=config.semantic_weight,
        )
    )
    dynamic_regions = reindex_split_regions(regions, batch_ids, targets)
    return SuperpointSplitOutput(
        query_indices=queries,
        candidate_mask=candidate_mask,
        keep_mask=keep_mask,
        targets=targets,
        target_confidence=confidence,
        dynamic_regions=dynamic_regions,
        statistics=statistics,
    )


def refine_candidates(config, refiner, point_features, coordinates, batch_ids, semantic_logits, split):
    """Read scene context globally and write residuals only to candidates."""
    residual = refiner(
        point_features,
        coordinates,
        batch_ids,
        split.query_indices,
        regions=split.dynamic_regions,
        candidate_mask=split.candidate_mask,
    )
    refined = semantic_logits + float(config.residual_scale) * residual
    return CandidateRefinementOutput(semantic_logits, refined, residual)


def verify_candidates(config, split, refinement):
    """Accept supported residuals and roll unsupported changes back to the base state."""
    targets = split.targets
    candidate_mask = split.candidate_mask & (targets >= 0)
    base_probability = F.softmax(refinement.base_logits.detach(), dim=1)
    refined_probability = F.softmax(refinement.refined_logits.detach(), dim=1)

    gather_targets = targets.clamp_min(0).unsqueeze(1)
    base_support = base_probability.gather(1, gather_targets).squeeze(1)
    refined_support = refined_probability.gather(1, gather_targets).squeeze(1)
    residual_norm = refinement.residual_logits.detach().norm(dim=1)
    accept_mask = candidate_mask & (
        refined_support + float(config.verifier_tolerance) >= base_support
    ) & (residual_norm <= float(config.max_residual_norm))
    rollback_mask = candidate_mask & ~accept_mask

    verified_logits = refinement.base_logits.clone()
    verified_logits[accept_mask] = refinement.refined_logits[accept_mask]
    # Split validation is sufficient to train the backbone structure, whereas
    # the Refiner may learn from a proposal only after this second gate accepts it.
    supervision_targets = torch.full_like(targets, -1)
    supervision_targets[accept_mask] = targets[accept_mask]
    supervision_confidence = torch.zeros_like(split.target_confidence)
    supervision_confidence[accept_mask] = split.target_confidence[accept_mask]
    count = max(int(candidate_mask.sum().item()), 1)
    statistics = {
        "candidate_points": int(candidate_mask.sum().item()),
        "accepted_points": int(accept_mask.sum().item()),
        "rollback_points": int(rollback_mask.sum().item()),
        "accept_ratio": float(accept_mask.sum().item()) / count,
    }
    return VerificationOutput(
        verified_logits=verified_logits,
        accept_mask=accept_mask,
        rollback_mask=rollback_mask,
        keep_mask=split.keep_mask,
        supervision_targets=supervision_targets,
        supervision_confidence=supervision_confidence,
        statistics=statistics,
    )


def run_stage3_pipeline(
    config,
    refiner,
    point_features,
    coordinates,
    colors,
    semantic_logits,
    regions,
    batch_ids,
):
    """Run split, candidate refinement, and verification inside training."""
    split = split_superpoints(
        config,
        point_features,
        coordinates,
        colors,
        semantic_logits.detach(),
        regions,
        batch_ids,
    )
    refinement = refine_candidates(
        config,
        refiner,
        point_features,
        coordinates,
        batch_ids,
        semantic_logits,
        split,
    )
    verification = verify_candidates(config, split, refinement)
    return Stage3Output(split, refinement, verification)


def confidence_weighted_cross_entropy(logits, targets, confidence):
    mask = targets >= 0
    if not mask.any():
        return logits.sum() * 0.0
    point_loss = F.cross_entropy(logits[mask], targets[mask], reduction="none")
    weights = confidence[mask].detach().clamp_min(1e-3)
    return (point_loss * weights).sum() / weights.sum().clamp_min(1e-6)


def stage3_training_losses(config, output, semantic_scale=3.0, keep_weight=1.0, residual_weight=0.01):
    """Return backbone and refiner objectives derived only from verified proxies."""
    verified = output.verification
    # Structurally valid child targets update the backbone even when a learned
    # residual is rolled back. Refiner supervision is verifier-gated below.
    backbone_loss = confidence_weighted_cross_entropy(
        output.refinement.base_logits * semantic_scale,
        output.split.targets,
        output.split.target_confidence,
    )
    refiner_loss = confidence_weighted_cross_entropy(
        output.refinement.refined_logits * semantic_scale,
        verified.supervision_targets,
        verified.supervision_confidence,
    )
    keep_loss = refinement_keep_kl(
        output.refinement.refined_logits * semantic_scale,
        output.refinement.base_logits * semantic_scale,
        verified.keep_mask,
    )
    residual_loss = delta_l2(
        output.refinement.residual_logits,
        output.split.candidate_mask | verified.keep_mask,
    )
    refiner_loss = refiner_loss + keep_weight * keep_loss + residual_weight * residual_loss
    return {
        "backbone": backbone_loss,
        "refiner": refiner_loss,
        "keep": keep_loss,
        "residual": residual_loss,
    }


class SuperpointDecomposer:
    """Adapter used while GrowSP regenerates superpoint-level pseudo labels."""

    def __init__(self, config):
        self.config = config

    def eval(self):
        return self

    def __call__(self, point_features, coordinates, colors, semantic_logits, regions, batch_ids, **_):
        output = split_superpoints(
            self.config,
            point_features,
            coordinates,
            colors,
            semantic_logits,
            regions,
            batch_ids,
        )
        output.stats = {
            "selected_regions": output.statistics["split_candidate_regions"],
            "accepted_splits": output.statistics["split_regions"],
            "supervised_ratio": float(output.supervision_mask.float().mean().item()),
        }
        output.supervision_targets = output.targets
        return output
