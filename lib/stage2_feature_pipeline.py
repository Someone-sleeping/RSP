from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lib.stage3_pipeline import Stage3Config, reindex_split_regions, split_superpoints


@dataclass
class Stage2FeatureConfig:
    """Grow-then-decompose direct feature updates used during Stage 2."""

    backbone_gradient_scale: float = 0.1
    query_scale: float = 10.0
    min_region_points: int = 20
    min_child_points: int = 8
    max_regions_per_scene: int = 20
    purity_threshold: float = 0.92
    entropy_threshold: float = 0.25
    min_split_confidence: float = 0.35
    verifier_tolerance: float = 0.0
    max_feature_update_norm: float = 1.0
    min_structure_gain: float = 0.01
    min_child_separation: float = 0.05
    min_primitive_gain: float = 0.005
    min_primitive_margin: float = 0.01
    primitive_top_k: int = 3
    primitive_support_tolerance: float = 0.01
    separation_weight: float = 0.25

    @classmethod
    def from_args(cls, args):
        """Build one shared configuration for training, clustering and eval."""
        return cls(
            backbone_gradient_scale=getattr(
                args, "stage2_backbone_gradient_scale", 0.1
            ),
            min_region_points=getattr(args, "stage2_min_region_points", 20),
            min_child_points=getattr(args, "stage2_min_child_points", 8),
            max_regions_per_scene=getattr(args, "stage2_max_regions", 20),
            purity_threshold=getattr(args, "stage2_purity_th", 0.92),
            entropy_threshold=getattr(args, "stage2_entropy_th", 0.25),
            min_split_confidence=getattr(args, "stage2_min_split_conf", 0.35),
            verifier_tolerance=getattr(args, "stage2_verifier_tolerance", 0.0),
            max_feature_update_norm=getattr(
                args, "stage2_max_feature_update_norm", 1.0
            ),
            min_structure_gain=getattr(args, "stage2_min_structure_gain", 0.01),
            min_child_separation=getattr(
                args, "stage2_min_child_separation", 0.05
            ),
            min_primitive_gain=getattr(args, "stage2_min_primitive_gain", 0.005),
            min_primitive_margin=getattr(
                args, "stage2_min_primitive_margin", 0.01
            ),
            primitive_top_k=getattr(args, "stage2_primitive_top_k", 3),
            primitive_support_tolerance=getattr(
                args, "stage2_primitive_support_tolerance", 0.01
            ),
        )

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
    candidate_features: torch.Tensor
    verified_features: torch.Tensor
    feature_update: torch.Tensor
    dynamic_regions: torch.Tensor
    supervision_targets: torch.Tensor
    supervision_confidence: torch.Tensor
    supervision_mask: torch.Tensor
    candidate_mask: torch.Tensor
    decomposition_mask: torch.Tensor
    feature_accept_mask: torch.Tensor
    rollback_mask: torch.Tensor
    feature_rollback_mask: torch.Tensor
    query_indices: torch.Tensor
    original_regions: torch.Tensor
    batch_ids: torch.Tensor
    stats: dict


def _partition_statistics(features, parent_mask, targets):
    child_ids = torch.unique(targets[parent_mask & (targets >= 0)])
    if child_ids.numel() < 2:
        invalid = features.new_tensor(float("-inf"))
        return invalid, invalid, invalid
    parent_features = F.normalize(features[parent_mask], dim=1)
    parent_center = F.normalize(parent_features.mean(dim=0, keepdim=True), dim=1)
    parent_compactness = F.cosine_similarity(
        parent_features, parent_center.expand_as(parent_features), dim=1
    ).mean()
    centers = []
    compactness = features.new_tensor(0.0)
    child_points = 0
    for child_id in child_ids:
        child_mask = parent_mask & (targets == child_id)
        child_features = F.normalize(features[child_mask], dim=1)
        center = F.normalize(child_features.mean(dim=0, keepdim=True), dim=1)
        centers.append(center)
        count = int(child_features.size(0))
        compactness = compactness + count * F.cosine_similarity(
            child_features, center.expand_as(child_features), dim=1
        ).mean()
        child_points += count
    compactness = compactness / max(child_points, 1)
    separation = 1.0 - F.cosine_similarity(centers[0], centers[1], dim=1).mean()
    return parent_compactness, compactness, separation


def _child_structure_score(features, parent_mask, targets, separation_weight):
    _, compactness, separation = _partition_statistics(features, parent_mask, targets)
    return compactness + float(separation_weight) * separation


def _primitive_partition_is_consistent(
    config, features, parent_mask, targets, primitive_centers, primitive_to_semantic
):
    if primitive_centers is None or primitive_to_semantic is None:
        return True

    centers = F.normalize(primitive_centers.detach().to(features.device), dim=1)
    mapping = primitive_to_semantic.detach().long().to(features.device)
    parent_center = F.normalize(features[parent_mask].mean(dim=0, keepdim=True), dim=1)
    parent_support = F.linear(parent_center, centers).max()
    child_support = features.new_tensor(0.0)
    child_points = 0
    primitive_ids = []

    for child_target in torch.unique(targets[parent_mask & (targets >= 0)]):
        child_mask = parent_mask & (targets == child_target)
        child_center = F.normalize(features[child_mask].mean(dim=0, keepdim=True), dim=1)
        scores = F.linear(child_center, centers).squeeze(0)
        global_top_scores, global_top_ids = scores.topk(k=min(2, scores.numel()))
        global_margin = global_top_scores[0] - global_top_scores[-1]
        nearest_matches_target = mapping[global_top_ids[0]] == child_target
        if nearest_matches_target:
            if global_margin < float(config.min_primitive_margin):
                return False
        elif global_margin > float(config.primitive_support_tolerance):
            return False

        semantic_support = scores.new_full(
            (int(mapping.max().item()) + 1,), float("-inf")
        )
        for semantic_id in torch.unique(mapping):
            group_scores = scores[mapping == semantic_id]
            top_group_scores = group_scores.topk(
                k=min(int(config.primitive_top_k), group_scores.numel())
            ).values
            semantic_support[semantic_id] = top_group_scores.mean()

        target_id = int(child_target.item())
        if target_id >= semantic_support.numel():
            return False
        target_support = semantic_support[target_id]
        best_support = semantic_support.max()
        if target_support + float(config.primitive_support_tolerance) < best_support:
            return False

        target_primitive_scores = scores[mapping == target_id]
        top_scores, local_top_ids = target_primitive_scores.topk(
            k=min(2, target_primitive_scores.numel())
        )
        target_primitive_ids = torch.nonzero(mapping == target_id, as_tuple=False).flatten()
        primitive_id = target_primitive_ids[local_top_ids[0]]
        count = int(child_mask.sum().item())
        child_support = child_support + count * top_scores[0]
        child_points += count
        primitive_ids.append(int(primitive_id.item()))

    if len(set(primitive_ids)) < 2:
        return False
    child_support = child_support / max(child_points, 1)
    return bool(
        (child_support - parent_support >= float(config.min_primitive_gain)).item()
    )


def _verify_feature_updates(
    config,
    base_features,
    candidate_features,
    feature_update,
    split,
    regions,
    batch_ids,
    primitive_centers=None,
    primitive_to_semantic=None,
):
    accepted_targets = split.targets.clone()
    decomposition_mask = torch.zeros_like(split.candidate_mask)
    feature_accept_mask = torch.zeros_like(split.candidate_mask)
    rollback_mask = torch.zeros_like(split.candidate_mask)
    feature_rollback_mask = torch.zeros_like(split.candidate_mask)
    accepted_regions = 0
    rejected_regions = 0
    feature_accepted_regions = 0
    feature_rejected_regions = 0
    feature_score_rejections = 0
    feature_norm_rejections = 0
    primitive_rejections = 0
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
            parent_compactness, child_compactness, child_separation = _partition_statistics(
                base_features.detach(), parent_mask, split.targets
            )
            candidate_score = _child_structure_score(
                candidate_features.detach(), parent_mask, split.targets, config.separation_weight
            )
            update_norm = feature_update.detach()[candidate].norm(dim=1).mean()
            primitive_consistent = _primitive_partition_is_consistent(
                config,
                base_features.detach(),
                parent_mask,
                split.targets,
                primitive_centers,
                primitive_to_semantic,
            )
            decomposition_accepted = (
                child_compactness - parent_compactness >= float(config.min_structure_gain)
                and child_separation >= float(config.min_child_separation)
                and primitive_consistent
            )
            if decomposition_accepted:
                decomposition_mask[candidate] = True
                accepted_regions += 1
                score_accepted = (
                    candidate_score + float(config.verifier_tolerance) + 1e-6
                    >= base_score
                )
                norm_accepted = update_norm <= float(config.max_feature_update_norm)
                feature_accepted = score_accepted and norm_accepted
                if feature_accepted:
                    feature_accept_mask[candidate] = True
                    feature_accepted_regions += 1
                else:
                    feature_score_rejections += int(not score_accepted)
                    feature_norm_rejections += int(not norm_accepted)
                    feature_rollback_mask[candidate] = True
                    feature_rejected_regions += 1
            else:
                primitive_rejections += int(not primitive_consistent)
                rollback_mask[candidate] = True
                accepted_targets[candidate] = -1
                rejected_regions += 1

    return (
        accepted_targets,
        decomposition_mask,
        feature_accept_mask,
        rollback_mask,
        feature_rollback_mask,
        accepted_regions,
        rejected_regions,
        feature_accepted_regions,
        feature_rejected_regions,
        feature_score_rejections,
        feature_norm_rejections,
        primitive_rejections,
    )


def run_stage2_feature_pipeline(
    config,
    feature_refiner,
    point_features,
    coordinates,
    colors,
    semantic_logits,
    regions,
    batch_ids,
    primitive_centers=None,
    primitive_to_semantic=None,
):
    """Update candidate features and commit only structurally safe features."""
    # Training already supplies normalized embeddings, while clustering obtains
    # raw backbone activations. A shared scale is required for verifier update
    # norms and prevents normalization itself from looking like a correction.
    point_features = F.normalize(point_features, dim=1)
    # Candidate discovery is a non-differentiable decision made from the current
    # state. Once selected, its feature objective jointly trains the backbone and
    # Refiner; non-candidate context is detached inside the Refiner.
    discovery_features = point_features.detach()
    split = split_superpoints(
        config.split_config(),
        discovery_features,
        coordinates,
        colors,
        semantic_logits.detach(),
        regions,
        batch_ids,
    )
    if not hasattr(feature_refiner, "update_candidate_features"):
        raise TypeError(
            "Stage-2 requires a direct feature model with "
            "update_candidate_features()."
        )
    candidate_features = feature_refiner.update_candidate_features(
        point_features,
        coordinates,
        batch_ids,
        split.query_indices,
        split.candidate_mask,
        regions=split.dynamic_regions,
        backbone_gradient_scale=config.backbone_gradient_scale,
    )
    # This displacement is only a verifier/regularizer measurement. The model
    # predicts candidate features directly, never semantic-logit deltas.
    feature_update = candidate_features - point_features
    (
        targets,
        decomposition_mask,
        feature_accept_mask,
        rollback_mask,
        feature_rollback_mask,
        accepted,
        rejected,
        feature_accepted,
        feature_rejected,
        feature_score_rejections,
        feature_norm_rejections,
        primitive_rejections,
    ) = _verify_feature_updates(
        config,
        discovery_features,
        candidate_features,
        feature_update,
        split,
        regions,
        batch_ids,
        primitive_centers,
        primitive_to_semantic,
    )
    verified_features = point_features.clone()
    verified_features[feature_accept_mask] = candidate_features[feature_accept_mask]
    dynamic_regions = reindex_split_regions(regions, batch_ids, targets)
    confidence = torch.zeros_like(split.target_confidence)
    confidence[decomposition_mask] = split.target_confidence[decomposition_mask]
    stats = {
        "selected_regions": split.statistics["split_candidate_regions"],
        "proposed_splits": split.statistics["split_regions"],
        "accepted_splits": accepted,
        "rejected_splits": rejected,
        "feature_updates_accepted": feature_accepted,
        "feature_updates_rejected": feature_rejected,
        "feature_score_rejections": feature_score_rejections,
        "feature_norm_rejections": feature_norm_rejections,
        "primitive_rejections": primitive_rejections,
        "supervised_ratio": float(decomposition_mask.float().mean().item()),
        "feature_update_ratio": float(feature_accept_mask.float().mean().item()),
    }
    return Stage2FeatureOutput(
        base_features=point_features,
        candidate_features=candidate_features,
        verified_features=verified_features,
        feature_update=feature_update,
        dynamic_regions=dynamic_regions,
        supervision_targets=targets,
        supervision_confidence=confidence,
        supervision_mask=decomposition_mask,
        candidate_mask=split.candidate_mask,
        decomposition_mask=decomposition_mask,
        feature_accept_mask=feature_accept_mask,
        rollback_mask=rollback_mask,
        feature_rollback_mask=feature_rollback_mask,
        query_indices=split.query_indices,
        original_regions=regions.view(-1).long(),
        batch_ids=batch_ids.view(-1).long(),
        stats=stats,
    )


def stage2_feature_losses(
    output,
    semantic_centers,
    primitive_centers=None,
    primitive_to_semantic=None,
    semantic_scale=3.0,
    primitive_loss_weight=None,
):
    """Feature-space objectives for accepted child regions."""
    # A candidate feature update contributes gradients only after verification.
    # Accepted decomposition without an accepted feature update can still alter the
    # region structure, but cannot train an unsupported feature transformation.
    mask = output.feature_accept_mask
    zero = output.verified_features.sum() * 0.0
    if mask.any():
        logits = F.linear(
            F.normalize(output.verified_features[mask], dim=1), semantic_centers
        )
        point_loss = F.cross_entropy(
            logits * semantic_scale,
            output.supervision_targets[mask],
            reduction="none",
        )
        weights = output.supervision_confidence[mask].detach().clamp_min(1e-3)
        semantic_loss = (point_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        semantic_loss = zero

    primitive_loss = zero
    if mask.any() and primitive_centers is not None and primitive_to_semantic is not None:
        primitive_centers = F.normalize(
            primitive_centers.detach().to(output.candidate_features.device), dim=1
        )
        primitive_to_semantic = primitive_to_semantic.detach().long().to(
            output.candidate_features.device
        )
        base_scores = F.linear(
            F.normalize(output.base_features.detach()[mask], dim=1), primitive_centers
        )
        semantic_targets = output.supervision_targets[mask]
        primitive_targets = torch.full_like(semantic_targets, -1)
        for semantic_id in torch.unique(semantic_targets):
            point_mask = semantic_targets == semantic_id
            group_ids = torch.nonzero(
                primitive_to_semantic == semantic_id, as_tuple=False
            ).flatten()
            if group_ids.numel() == 0:
                continue
            local_ids = base_scores[point_mask][:, group_ids].argmax(dim=1)
            primitive_targets[point_mask] = group_ids[local_ids]
        valid_primitive = primitive_targets >= 0
        if valid_primitive.any():
            primitive_logits = F.linear(
                F.normalize(output.verified_features[mask][valid_primitive], dim=1),
                primitive_centers,
            )
            point_loss = F.cross_entropy(
                primitive_logits * semantic_scale,
                primitive_targets[valid_primitive],
                weight=primitive_loss_weight,
                reduction="none",
            )
            weights = output.supervision_confidence[mask][valid_primitive].detach().clamp_min(1e-3)
            primitive_loss = (point_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    structure_terms = []
    for batch_id in torch.unique(output.batch_ids):
        scene_mask = output.batch_ids == batch_id
        for region_id in torch.unique(output.original_regions[scene_mask]):
            parent_mask = scene_mask & (output.original_regions == region_id)
            if torch.unique(output.supervision_targets[parent_mask & mask]).numel() < 2:
                continue
            structure_terms.append(
                -_child_structure_score(
                    output.verified_features,
                    parent_mask,
                    output.supervision_targets,
                    separation_weight=0.25,
                )
            )
    structure_loss = torch.stack(structure_terms).mean() if structure_terms else zero
    feature_update_loss = (
        output.feature_update[mask].pow(2).mean()
        if mask.any()
        else zero
    )
    return {
        "semantic": semantic_loss,
        "primitive": primitive_loss,
        "structure": structure_loss,
        "feature_update": feature_update_loss,
    }


class Stage2FeatureModule:
    """Run the same integrated Stage-2 feature path in training and clustering."""

    apply_after_grow = True
    commits_dynamic_regions = True

    def __init__(self, feature_refiner, config):
        self.feature_refiner = feature_refiner
        self.config = config
        self.primitive_centers = None
        self.primitive_to_semantic = None

    def set_reference_primitives(self, primitive_centers, primitive_to_semantic):
        self.primitive_centers = primitive_centers.detach()
        self.primitive_to_semantic = primitive_to_semantic.detach()

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
            self.primitive_centers,
            self.primitive_to_semantic,
        )
