from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from lib.meta_refiner import (
    meta_adapt_refiner_gates,
    project_region_and_split_differentiable,
)
from lib.split_regions import (
    build_region_consistency_queries,
    build_split_region_queries,
)


@dataclass(frozen=True)
class SemanticDifferencePipelineConfig:
    semantic_classes: int = 12
    probability_scale: float = 10.0
    refiner_scale: float = 1.0
    meta_initial_scale: float = 1.0
    meta_min_scale: float = 0.25
    meta_max_scale: float = 1.75
    meta_inner_steps: int = 5
    meta_inner_lr: float = 0.1
    meta_correction_weight: float = 1.0
    meta_keep_weight: float = 5.0
    meta_entropy_weight: float = 0.01
    meta_scale_regularization: float = 0.1
    meta_bias_regularization: float = 1.0
    meta_query_tolerance: float = 0.0
    meta_classwise: bool = False
    meta_adapt_bias: bool = True
    meta_target_confidence: float = 0.80
    meta_verify_confidence: float = 0.64
    result_verify_confidence: float = 0.80
    min_temporal_votes: int = 2
    enable_temporal_override: bool = True


@dataclass
class DecompositionOutput:
    no_op_scores: torch.Tensor
    refined_scores: torch.Tensor
    delta_scores: torch.Tensor
    delta_components: Dict[str, torch.Tensor]
    query_indices: torch.Tensor
    refine_mask: torch.Tensor
    keep_mask: torch.Tensor
    split_targets: torch.Tensor
    statistics: Dict[str, float]

    @property
    def no_op_prediction(self):
        return self.no_op_scores.argmax(dim=1)

    @property
    def refined_prediction(self):
        return self.refined_scores.argmax(dim=1)


@dataclass
class MetaRefinerOutput:
    probability: torch.Tensor
    scene_accepted: bool
    statistics: Dict[str, object]

    @property
    def prediction(self):
        return self.probability.argmax(dim=1)


@dataclass
class ResultVerifierOutput:
    meta_verified_prediction: torch.Tensor
    final_prediction: torch.Tensor
    decision: torch.Tensor
    meta_accept_mask: torch.Tensor
    rollback_mask: torch.Tensor
    temporal_override_mask: torch.Tensor
    keep_mask: torch.Tensor
    statistics: Dict[str, float]


@dataclass
class SemanticDifferencePipelineOutput:
    base_prediction: torch.Tensor
    region_prediction: torch.Tensor
    temporal_mean_probability: torch.Tensor
    temporal_vote_count: torch.Tensor
    decomposition: DecompositionOutput
    meta_refiner: MetaRefinerOutput
    verifier: ResultVerifierOutput

    def stage_predictions(self):
        return {
            "base": self.base_prediction,
            "decomposition": self.decomposition.no_op_prediction,
            "refiner": self.decomposition.refined_prediction,
            "meta_refiner": self.meta_refiner.prediction,
            "meta_verified": self.verifier.meta_verified_prediction,
            "final_verified": self.verifier.final_prediction,
        }


def region_consensus_prediction(scores, regions):
    probability = F.softmax(scores, dim=1)
    point_confidence = probability.max(dim=1)[0]
    prediction = scores.argmax(dim=1).clone()
    for region_id in torch.unique(regions):
        if int(region_id.item()) < 0:
            continue
        mask = regions == region_id
        weights = point_confidence[mask].clamp_min(1e-6)
        region_probability = (
            (probability[mask] * weights[:, None]).sum(dim=0)
            / weights.sum().clamp_min(1e-6)
        )
        prediction[mask] = region_probability.argmax()
    return prediction


def decompose_and_refine(
    config,
    refiner,
    base_scores,
    base_features,
    sparse_coordinates,
    input_features,
    regions,
):
    device = base_scores.device
    batch_ids = sparse_coordinates[:, 0].long().to(device)
    point_coordinates = sparse_coordinates[:, 1:].float().to(device)
    point_colors = input_features[:, :3].float().to(device)
    (
        split_queries,
        split_refine_mask,
        split_targets,
        _split_confidence,
        split_keep_mask,
        split_statistics,
    ) = build_split_region_queries(
        base_scores,
        base_features,
        point_coordinates,
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
        multi_proposal=False,
    )
    (
        consistency_queries,
        consistency_mask,
        _consistency_targets,
        _consistency_confidence,
        consistency_keep_mask,
        consistency_statistics,
    ) = build_region_consistency_queries(
        base_scores * float(config.probability_scale),
        regions,
        batch_ids,
        min_region_points=20,
        max_regions_per_scene=40,
        min_region_conf=0.35,
        min_disagree_ratio=0.02,
        point_conf_threshold=0.55,
        point_entropy_threshold=0.55,
    )
    query_indices = split_queries
    if consistency_queries.numel() > 0:
        query_indices = torch.unique(
            torch.cat([split_queries, consistency_queries], dim=0)
        )
    refine_mask = split_refine_mask | consistency_mask
    keep_mask = split_keep_mask | consistency_keep_mask

    with torch.no_grad():
        delta_components = refiner(
            base_features,
            point_coordinates,
            batch_ids,
            query_indices,
            regions,
            use_region_branch=False,
            return_components=True,
        )
    delta_scores = delta_components["total"]
    no_op_scores = project_region_and_split_differentiable(
        base_scores,
        regions,
        split_targets,
    ).detach()
    refined_scores = project_region_and_split_differentiable(
        base_scores + float(config.refiner_scale) * delta_scores,
        regions,
        split_targets,
    ).detach()

    statistics = {
        "queries": int(query_indices.numel()),
        "split_queries": int(split_queries.numel()),
        "consistency_queries": int(consistency_queries.numel()),
        "refine_points": int(refine_mask.sum().item()),
        "split_points": int((split_targets >= 0).sum().item()),
        "keep_points": int(keep_mask.sum().item()),
        "points": int(base_scores.size(0)),
    }
    for prefix, values in (
        ("split", split_statistics),
        ("consistency", consistency_statistics),
    ):
        for name, value in values.items():
            if isinstance(value, (int, float)):
                statistics[f"{prefix}_{name}"] = float(value)
    return DecompositionOutput(
        no_op_scores=no_op_scores,
        refined_scores=refined_scores,
        delta_scores=delta_scores.detach(),
        delta_components={
            name: value.detach() for name, value in delta_components.items()
        },
        query_indices=query_indices,
        refine_mask=refine_mask,
        keep_mask=keep_mask,
        split_targets=split_targets,
        statistics=statistics,
    )


def adapt_meta_refiner(
    config,
    base_scores,
    base_prediction,
    region_prediction,
    regions,
    temporal_mean_probability,
    temporal_vote_count,
    decomposition,
):
    probability, accepted, statistics = meta_adapt_refiner_gates(
        base_scores.detach(),
        decomposition.delta_components,
        decomposition.refined_scores.detach(),
        temporal_mean_probability.detach(),
        temporal_vote_count.detach(),
        region_prediction.detach(),
        decomposition.no_op_prediction.detach(),
        base_prediction.detach(),
        regions.detach(),
        decomposition.split_targets.detach(),
        residual_scale=config.refiner_scale,
        initial_scale=config.meta_initial_scale,
        min_scale=config.meta_min_scale,
        max_scale=config.meta_max_scale,
        confidence_threshold=config.meta_target_confidence,
        min_votes=config.min_temporal_votes,
        inner_steps=config.meta_inner_steps,
        inner_lr=config.meta_inner_lr,
        correction_weight=config.meta_correction_weight,
        keep_weight=config.meta_keep_weight,
        entropy_weight=config.meta_entropy_weight,
        scale_regularization=config.meta_scale_regularization,
        query_tolerance=config.meta_query_tolerance,
        classwise=config.meta_classwise,
        refine_mask=decomposition.refine_mask.detach(),
        adapt_bias=config.meta_adapt_bias,
        bias_regularization=config.meta_bias_regularization,
    )
    return MetaRefinerOutput(
        probability=probability,
        scene_accepted=bool(accepted),
        statistics=statistics,
    )


def verify_result(
    config,
    base_prediction,
    region_prediction,
    temporal_mean_probability,
    temporal_vote_count,
    decomposition,
    meta_refiner,
):
    no_op_prediction = decomposition.no_op_prediction
    meta_prediction = meta_refiner.prediction
    temporal_confidence, temporal_prediction = temporal_mean_probability.max(dim=1)

    candidate_support = temporal_vote_count.gather(
        1,
        meta_prediction[:, None],
    ).squeeze(1)
    candidate_confidence = temporal_mean_probability.gather(
        1,
        meta_prediction[:, None],
    ).squeeze(1)
    candidate_changed = meta_prediction != no_op_prediction
    structural_support = (
        (meta_prediction == temporal_prediction)
        | (meta_prediction == region_prediction)
    )
    meta_accept_mask = (
        candidate_changed
        & (candidate_support >= int(config.min_temporal_votes))
        & (candidate_confidence >= float(config.meta_verify_confidence))
        & structural_support
    )
    # Decomposition is the verified structural anchor. Validate the complete
    # Refiner/Meta residual against it so a stale Refiner cannot bypass rollback.
    rollback_mask = candidate_changed & ~meta_accept_mask
    meta_verified_prediction = meta_prediction.clone()
    meta_verified_prediction[rollback_mask] = no_op_prediction[rollback_mask]

    temporal_support = temporal_vote_count.gather(
        1,
        temporal_prediction[:, None],
    ).squeeze(1)
    temporal_accept = (
        (temporal_support >= int(config.min_temporal_votes))
        & (temporal_confidence >= float(config.result_verify_confidence))
        & (temporal_prediction != base_prediction)
        & (
            (temporal_prediction == region_prediction)
            | (temporal_prediction == no_op_prediction)
            | decomposition.refine_mask
        )
    )
    if not config.enable_temporal_override:
        temporal_accept = torch.zeros_like(temporal_accept)
    temporal_override_mask = temporal_accept & (
        temporal_prediction != meta_verified_prediction
    )
    final_prediction = meta_verified_prediction.clone()
    final_prediction[temporal_accept] = temporal_prediction[temporal_accept]

    decision = torch.zeros_like(base_prediction, dtype=torch.uint8)
    decision[meta_accept_mask] = 1
    decision[rollback_mask] = 2
    decision[temporal_override_mask] = 3
    keep_mask = decision == 0
    point_count = max(int(base_prediction.numel()), 1)
    statistics = {
        "points": point_count,
        "meta_changed": int(candidate_changed.sum().item()),
        "meta_accepted": int(meta_accept_mask.sum().item()),
        "meta_rollback": int(rollback_mask.sum().item()),
        "temporal_override": int(temporal_override_mask.sum().item()),
        "kept": int(keep_mask.sum().item()),
        "final_changed_from_base": int(
            (final_prediction != base_prediction).sum().item()
        ),
    }
    for name in (
        "meta_changed",
        "meta_accepted",
        "meta_rollback",
        "temporal_override",
        "kept",
        "final_changed_from_base",
    ):
        statistics[f"{name}_ratio"] = statistics[name] / point_count
    return ResultVerifierOutput(
        meta_verified_prediction=meta_verified_prediction,
        final_prediction=final_prediction,
        decision=decision,
        meta_accept_mask=meta_accept_mask,
        rollback_mask=rollback_mask,
        temporal_override_mask=temporal_override_mask,
        keep_mask=keep_mask,
        statistics=statistics,
    )


def run_semantic_difference_pipeline(
    config,
    refiner,
    base_scores,
    base_features,
    sparse_coordinates,
    input_features,
    regions,
    temporal_probabilities,
):
    base_prediction = base_scores.argmax(dim=1)
    region_prediction = region_consensus_prediction(base_scores, regions)
    temporal_mean_probability = temporal_probabilities.mean(dim=0)
    temporal_predictions = temporal_probabilities.argmax(dim=2)
    temporal_vote_count = F.one_hot(
        temporal_predictions,
        num_classes=config.semantic_classes,
    ).sum(dim=0)

    decomposition = decompose_and_refine(
        config,
        refiner,
        base_scores,
        base_features,
        sparse_coordinates,
        input_features,
        regions,
    )
    meta_refiner = adapt_meta_refiner(
        config,
        base_scores,
        base_prediction,
        region_prediction,
        regions,
        temporal_mean_probability,
        temporal_vote_count,
        decomposition,
    )
    verifier = verify_result(
        config,
        base_prediction,
        region_prediction,
        temporal_mean_probability,
        temporal_vote_count,
        decomposition,
        meta_refiner,
    )
    return SemanticDifferencePipelineOutput(
        base_prediction=base_prediction,
        region_prediction=region_prediction,
        temporal_mean_probability=temporal_mean_probability,
        temporal_vote_count=temporal_vote_count,
        decomposition=decomposition,
        meta_refiner=meta_refiner,
        verifier=verifier,
    )
