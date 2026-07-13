import torch
import torch.nn.functional as F


def _target_values(values, targets):
    return values.gather(1, targets.long().unsqueeze(1)).squeeze(1)


def hierarchical_split_merge(
    no_op_prediction,
    split_targets,
    region_prediction,
    temporal_probabilities,
    temporal_mean_probability,
    regions,
    child_agreement_threshold=0.5,
    child_confidence_threshold=0.5,
    min_child_points=8,
    merge_max_child_ratio=0.25,
):
    """Keep a proposed split child only when history supports it as a group.

    Unsupported children are merged back into the original region consensus.
    All decisions use frozen predictions and the original superpoint partition.
    """
    result = no_op_prediction.clone()
    split_valid = split_targets >= 0
    if not split_valid.any():
        return result

    temporal_predictions = temporal_probabilities.argmax(dim=2)
    for region_id in torch.unique(regions):
        if int(region_id.item()) == -1:
            continue
        region_mask = regions == region_id
        local_split = region_mask & split_valid
        if not local_split.any():
            continue
        for target in torch.unique(split_targets[local_split]):
            child_mask = local_split & (split_targets == target)
            child_ratio = child_mask.sum().float() / local_split.sum().float().clamp_min(1.0)
            if float(child_ratio.item()) > float(merge_max_child_ratio):
                continue
            if int(child_mask.sum().item()) < int(min_child_points):
                result[child_mask] = region_prediction[child_mask]
                continue
            agreement = (temporal_predictions[:, child_mask] == target).float().mean()
            confidence = temporal_mean_probability[child_mask, target.long()].mean()
            accepted = (
                float(agreement.item()) >= float(child_agreement_threshold)
                and float(confidence.item()) >= float(child_confidence_threshold)
            )
            if not accepted:
                result[child_mask] = region_prediction[child_mask]
    return result


def error_type_conditioned_refinement(
    hierarchical_prediction,
    no_op_prediction,
    refined_prediction,
    refined_probability,
    no_op_probability,
    region_prediction,
    split_targets,
    temporal_mean_probability,
    temporal_vote_count,
    min_temporal_votes=2,
    temporal_confidence_threshold=0.5,
    min_independent_support=1,
    confidence_gain=0.0,
):
    """Accept residual changes according to split, region, and temporal support."""
    result = hierarchical_prediction.clone()
    residual_changed = refined_prediction != no_op_prediction
    non_split_residual = residual_changed & (split_targets < 0)
    if not non_split_residual.any():
        return result

    temporal_target_votes = _target_values(temporal_vote_count, refined_prediction)
    temporal_target_confidence = _target_values(temporal_mean_probability, refined_prediction)
    refined_target_confidence = _target_values(refined_probability, refined_prediction)
    no_op_target_confidence = _target_values(no_op_probability, refined_prediction)
    support = (
        (temporal_target_votes >= int(min_temporal_votes))
        & (temporal_target_confidence >= float(temporal_confidence_threshold))
    ).long()
    support = support + (region_prediction == refined_prediction).long()
    support = support + (
        refined_target_confidence >= no_op_target_confidence + float(confidence_gain)
    ).long()
    accepted = (
        non_split_residual
        & (support >= int(min_independent_support))
        & (refined_target_confidence >= 0.25)
    )
    result[accepted] = refined_prediction[accepted]
    return result


def local_consensus_verifier(
    proposal_prediction,
    base_prediction,
    region_prediction,
    no_op_prediction,
    temporal_predictions,
    temporal_mean_probability,
    min_support=3,
    temporal_confidence_threshold=0.5,
    proposal_weight=1.0,
):
    """Accept local corrections supported by independent structural sources."""
    num_classes = temporal_mean_probability.size(1)
    sources = torch.cat(
        [
            base_prediction.unsqueeze(0),
            region_prediction.unsqueeze(0),
            no_op_prediction.unsqueeze(0),
            temporal_predictions,
        ],
        dim=0,
    )
    votes = F.one_hot(sources, num_classes=num_classes).sum(dim=0).float()
    votes.scatter_add_(
        1,
        proposal_prediction.unsqueeze(1),
        votes.new_full((votes.size(0), 1), float(proposal_weight)),
    )
    max_support, candidate = votes.max(dim=1)
    for preferred in (base_prediction, no_op_prediction, proposal_prediction):
        preferred_support = _target_values(votes, preferred)
        tie = preferred_support == max_support
        candidate[tie] = preferred[tie]

    temporal_candidate_confidence = _target_values(temporal_mean_probability, candidate)
    temporal_supported = temporal_candidate_confidence >= float(temporal_confidence_threshold)
    structural_supported = (candidate == region_prediction) | (candidate == no_op_prediction)
    accepted = (
        (candidate != base_prediction)
        & (max_support >= float(min_support))
        & (temporal_supported | structural_supported)
    )
    result = proposal_prediction.clone()
    proposal_support = _target_values(votes, proposal_prediction)
    proposal_temporal_confidence = _target_values(
        temporal_mean_probability, proposal_prediction
    )
    proposal_structural_support = (
        (proposal_prediction == region_prediction)
        | (proposal_prediction == no_op_prediction)
    )
    rollback = (
        (proposal_prediction != base_prediction)
        & (proposal_support < float(min_support))
        & (proposal_temporal_confidence < float(temporal_confidence_threshold))
        & ~proposal_structural_support
    )
    result[rollback] = base_prediction[rollback]
    result[accepted] = candidate[accepted]
    return result


def region_risk_rollback(
    proposal_prediction,
    fallback_prediction,
    region_prediction,
    no_op_prediction,
    temporal_mean_probability,
    temporal_vote_count,
    regions,
    min_region_reliability=0.5,
):
    """Rollback proposal changes in regions with weak multi-source reliability."""
    result = proposal_prediction.clone()
    changed = proposal_prediction != fallback_prediction
    if not changed.any():
        return result
    num_references = max(int(temporal_vote_count.sum(dim=1).max().item()), 1)
    temporal_confidence = _target_values(
        temporal_mean_probability, proposal_prediction
    )
    temporal_support = _target_values(
        temporal_vote_count.float(), proposal_prediction
    ) / float(num_references)
    structural_support = (
        (proposal_prediction == region_prediction)
        | (proposal_prediction == no_op_prediction)
    ).float()
    point_reliability = (
        temporal_confidence + temporal_support + structural_support
    ) / 3.0
    for region_id in torch.unique(regions):
        region_changed = changed & (regions == region_id)
        if not region_changed.any():
            continue
        reliability = point_reliability[region_changed].mean()
        if float(reliability.item()) < float(min_region_reliability):
            result[region_changed] = fallback_prediction[region_changed]
    return result


def regionwise_refiner_scale_selection(
    candidate_probabilities,
    base_prediction,
    temporal_mean_probability,
    temporal_vote_count,
    regions,
    temporal_weight=1.0,
    preservation_weight=0.1,
):
    """Select a residual scale per region using a label-free reliability score."""
    candidate_predictions = [probability.argmax(dim=1) for probability in candidate_probabilities]
    selected = candidate_predictions[0].clone()
    num_references = max(int(temporal_vote_count.sum(dim=1).max().item()), 1)
    for region_id in torch.unique(regions):
        if int(region_id.item()) == -1:
            continue
        mask = regions == region_id
        best_score = None
        best_prediction = None
        for probability, prediction in zip(candidate_probabilities, candidate_predictions):
            confidence = _target_values(probability[mask], prediction[mask]).mean()
            temporal_confidence = _target_values(
                temporal_mean_probability[mask], prediction[mask]
            ).mean()
            vote_support = _target_values(
                temporal_vote_count[mask].float(), prediction[mask]
            ).mean() / float(num_references)
            preservation = (prediction[mask] == base_prediction[mask]).float().mean()
            score = (
                confidence
                + float(temporal_weight) * (temporal_confidence + vote_support)
                + float(preservation_weight) * preservation
            )
            if best_score is None or float(score.item()) > best_score:
                best_score = float(score.item())
                best_prediction = prediction
        selected[mask] = best_prediction[mask]
    return selected
