import math

import torch
import torch.nn.functional as F


def _poe_log_probability(reference_probability, temporal_probability, blend_weight):
    score = (
        (1.0 - blend_weight) * torch.log(reference_probability.clamp_min(1e-6))
        + blend_weight * torch.log(temporal_probability.clamp_min(1e-6))
    )
    return F.log_softmax(score, dim=1)


def _masked_nll(log_probability, targets, mask, weights=None):
    if not mask.any():
        return log_probability.sum() * 0.0
    loss = F.nll_loss(log_probability[mask], targets[mask], reduction="none")
    if weights is None:
        return loss.mean()
    selected_weights = weights[mask].clamp_min(1e-4)
    return (loss * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)


def _meta_objective(
    log_probability,
    reference_probability,
    reference_prediction,
    temporal_prediction,
    temporal_confidence,
    correction_mask,
    keep_mask,
    subset_mask,
    correction_weight,
    keep_weight,
    entropy_weight,
):
    correction_subset = correction_mask & subset_mask
    keep_subset = keep_mask & subset_mask
    correction_loss = _masked_nll(
        log_probability,
        temporal_prediction,
        correction_subset,
        temporal_confidence,
    )
    if keep_subset.any():
        keep_loss = F.kl_div(
            log_probability[keep_subset],
            reference_probability[keep_subset],
            reduction="batchmean",
        )
    else:
        keep_loss = log_probability.sum() * 0.0
    if subset_mask.any():
        probability = log_probability[subset_mask].exp()
        entropy = -(probability * log_probability[subset_mask]).sum(dim=1).mean()
    else:
        entropy = log_probability.sum() * 0.0
    return (
        correction_weight * correction_loss
        + keep_weight * keep_loss
        + entropy_weight * entropy
    )


def meta_optimize_poe_weight(
    reference_probability,
    temporal_probability,
    temporal_votes,
    region_prediction,
    no_op_prediction,
    base_prediction,
    initial_weight=0.1,
    confidence_threshold=0.64,
    min_votes=2,
    inner_steps=5,
    inner_lr=0.5,
    correction_weight=1.0,
    keep_weight=1.0,
    entropy_weight=0.01,
    query_tolerance=0.0,
    classwise=False,
):
    """Adapt one PoE fusion weight on support points and verify it on query points.

    The targets are label-free. Temporal consensus supervises candidate corrections,
    while the reference distribution supervises preservation. Even/odd point hashes
    form deterministic support/query partitions for each scene.
    """
    device = reference_probability.device
    num_points = reference_probability.size(0)
    temporal_confidence, temporal_prediction = temporal_probability.max(dim=1)
    reference_prediction = reference_probability.argmax(dim=1)
    region_supported = temporal_prediction == region_prediction
    structural_support = region_supported | (temporal_prediction == no_op_prediction)
    correction_mask = (
        (temporal_votes >= min_votes)
        & (temporal_confidence >= confidence_threshold)
        & (temporal_prediction != reference_prediction)
        & structural_support
    )
    keep_mask = (
        (reference_prediction == temporal_prediction)
        | (reference_prediction == no_op_prediction)
        | (reference_prediction == base_prediction)
    ) & ~correction_mask

    indices = torch.arange(num_points, device=device)
    support_mask = (indices % 2) == 0
    query_mask = ~support_mask
    if not (correction_mask & support_mask).any() or not query_mask.any():
        return reference_probability, float(initial_weight), False, {
            "correction_ratio": float(correction_mask.float().mean().item()),
            "query_gain": 0.0,
        }

    clipped_initial = min(max(float(initial_weight), 1e-4), 1.0 - 1e-4)
    initial_logit = math.log(clipped_initial / (1.0 - clipped_initial))
    blend_shape = (reference_probability.size(1),) if classwise else ()
    blend_logit = reference_probability.new_full(blend_shape, initial_logit, requires_grad=True)

    for _ in range(max(int(inner_steps), 1)):
        blend_weight = torch.sigmoid(blend_logit)
        log_probability = _poe_log_probability(
            reference_probability,
            temporal_probability,
            blend_weight,
        )
        support_loss = _meta_objective(
            log_probability,
            reference_probability,
            reference_prediction,
            temporal_prediction,
            temporal_confidence,
            correction_mask,
            keep_mask,
            support_mask,
            correction_weight,
            keep_weight,
            entropy_weight,
        )
        gradient = torch.autograd.grad(support_loss, blend_logit, create_graph=False)[0]
        blend_logit = (blend_logit - inner_lr * gradient).detach().requires_grad_(True)

    adapted_weight = torch.sigmoid(blend_logit).detach()
    adapted_log_probability = _poe_log_probability(
        reference_probability,
        temporal_probability,
        adapted_weight,
    )
    initial_weight_tensor = reference_probability.new_full(blend_shape, clipped_initial)
    initial_log_probability = _poe_log_probability(
        reference_probability,
        temporal_probability,
        initial_weight_tensor,
    )
    adapted_query_loss = _meta_objective(
        adapted_log_probability,
        reference_probability,
        reference_prediction,
        temporal_prediction,
        temporal_confidence,
        correction_mask,
        keep_mask,
        query_mask,
        correction_weight,
        keep_weight,
        entropy_weight,
    )
    initial_query_loss = _meta_objective(
        initial_log_probability,
        reference_probability,
        reference_prediction,
        temporal_prediction,
        temporal_confidence,
        correction_mask,
        keep_mask,
        query_mask,
        correction_weight,
        keep_weight,
        entropy_weight,
    )
    query_gain = float((initial_query_loss - adapted_query_loss).item())
    accepted = query_gain >= float(query_tolerance)
    selected_probability = adapted_log_probability.exp() if accepted else initial_log_probability.exp()
    mean_adapted_weight = float(adapted_weight.mean().item())
    selected_weight = mean_adapted_weight if accepted else clipped_initial
    stats = {
        "correction_ratio": float(correction_mask.float().mean().item()),
        "adapted_weight": mean_adapted_weight,
        "selected_weight": selected_weight,
        "query_gain": query_gain,
        "classwise": bool(classwise),
    }
    return selected_probability, selected_weight, accepted, stats
