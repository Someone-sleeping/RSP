import math

import torch
import torch.nn.functional as F


def project_region_and_split_differentiable(scores, regions, split_targets):
    """Apply the evaluation projection while retaining gradients to input scores."""
    probability = F.softmax(scores, dim=1)
    point_confidence = probability.max(dim=1)[0]
    projected_parts = []
    for region_id in torch.unique(regions):
        if int(region_id.item()) == -1:
            continue
        mask = regions == region_id
        weights = point_confidence[mask].clamp_min(1e-6)
        region_probability = (
            (probability[mask] * weights[:, None]).sum(dim=0, keepdim=True)
            / weights.sum().clamp_min(1e-6)
        )
        projected_parts.append((mask, torch.log(region_probability.clamp_min(1e-6))))

    projected = scores.clone()
    for mask, region_score in projected_parts:
        projected[mask] = region_score

    valid_split = split_targets >= 0
    if valid_split.any():
        forced = scores.new_full(
            (int(valid_split.sum().item()), scores.size(1)),
            -20.0,
        )
        forced.scatter_(1, split_targets[valid_split, None].long(), 20.0)
        projected[valid_split] = forced
    return projected


def _masked_nll(log_probability, targets, mask, weights=None):
    if not mask.any():
        return log_probability.sum() * 0.0
    loss = F.nll_loss(log_probability[mask], targets[mask], reduction="none")
    if weights is None:
        return loss.mean()
    selected_weights = weights[mask].clamp_min(1e-4)
    return (loss * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)


def _region_support_query_masks(regions):
    support = torch.zeros_like(regions, dtype=torch.bool)
    valid_regions = torch.unique(regions[regions >= 0])
    if valid_regions.numel() > 0:
        support_region = (valid_regions % 2) == 0
        for region_id, is_support in zip(valid_regions, support_region):
            support[regions == region_id] = is_support
    invalid = regions < 0
    if invalid.any():
        invalid_indices = torch.nonzero(invalid, as_tuple=False).flatten()
        support[invalid_indices[::2]] = True
    return support, ~support


def _gate_from_logit(gate_logit, min_scale, max_scale):
    return min_scale + (max_scale - min_scale) * torch.sigmoid(gate_logit)


def _initial_gate_logit(initial_scale, min_scale, max_scale):
    normalized = (float(initial_scale) - min_scale) / (max_scale - min_scale)
    normalized = min(max(normalized, 1e-4), 1.0 - 1e-4)
    return math.log(normalized / (1.0 - normalized))


def _compose_scores(
    base_scores,
    component_deltas,
    gate_logit,
    regions,
    split_targets,
    residual_scale,
    min_scale,
    max_scale,
    residual_bias=None,
    adapt_mask=None,
):
    gates = _gate_from_logit(gate_logit, min_scale, max_scale)
    delta = base_scores.new_zeros(base_scores.shape)
    for component_id, component in enumerate(component_deltas):
        component_gate = gates[component_id]
        if component_gate.ndim > 0:
            component_gate = component_gate.unsqueeze(0)
        delta = delta + component * component_gate
    raw_scores = base_scores + float(residual_scale) * delta
    if residual_bias is not None and adapt_mask is not None:
        raw_scores = raw_scores + adapt_mask[:, None].to(raw_scores.dtype) * residual_bias[None, :]
    return project_region_and_split_differentiable(raw_scores, regions, split_targets)


def _objective(
    log_probability,
    original_probability,
    temporal_prediction,
    temporal_confidence,
    correction_mask,
    keep_mask,
    subset_mask,
    correction_weight,
    keep_weight,
    entropy_weight,
    gate_scale,
    scale_regularization,
    initial_scale,
    residual_bias=None,
    bias_regularization=0.0,
):
    correction_loss = _masked_nll(
        log_probability,
        temporal_prediction,
        correction_mask & subset_mask,
        temporal_confidence,
    )
    selected_keep = keep_mask & subset_mask
    if selected_keep.any():
        keep_loss = F.kl_div(
            log_probability[selected_keep],
            original_probability[selected_keep],
            reduction="batchmean",
        )
    else:
        keep_loss = log_probability.sum() * 0.0
    if subset_mask.any():
        selected_log_probability = log_probability[subset_mask]
        probability = selected_log_probability.exp()
        entropy = -(probability * selected_log_probability).sum(dim=1).mean()
    else:
        entropy = log_probability.sum() * 0.0
    scale_loss = (gate_scale - float(initial_scale)).pow(2).mean()
    if residual_bias is None:
        bias_loss = log_probability.sum() * 0.0
    else:
        bias_loss = residual_bias.pow(2).mean()
    return (
        float(correction_weight) * correction_loss
        + float(keep_weight) * keep_loss
        + float(entropy_weight) * entropy
        + float(scale_regularization) * scale_loss
        + float(bias_regularization) * bias_loss
    )


def meta_adapt_refiner_gates(
    base_scores,
    component_deltas,
    original_refined_scores,
    temporal_probability,
    temporal_votes,
    region_prediction,
    no_op_prediction,
    base_prediction,
    regions,
    split_targets,
    residual_scale=1.0,
    initial_scale=1.0,
    min_scale=0.25,
    max_scale=1.75,
    confidence_threshold=0.8,
    min_votes=2,
    inner_steps=5,
    inner_lr=0.1,
    correction_weight=1.0,
    keep_weight=5.0,
    entropy_weight=0.01,
    scale_regularization=0.1,
    query_tolerance=0.0,
    classwise=False,
    refine_mask=None,
    adapt_bias=False,
    bias_regularization=0.1,
):
    """Adapt lightweight Refiner branch gates and validate them on held-out regions."""
    components = [component_deltas["point"], component_deltas["context"]]
    if component_deltas.get("region") is not None and component_deltas["region"].abs().max() > 0:
        components.append(component_deltas["region"])

    original_probability = F.softmax(original_refined_scores.detach(), dim=1)
    original_prediction = original_probability.argmax(dim=1)
    temporal_confidence, temporal_prediction = temporal_probability.max(dim=1)
    temporal_support = temporal_votes.gather(1, temporal_prediction[:, None]).squeeze(1)
    structural_support = (
        (temporal_prediction == region_prediction)
        | (temporal_prediction == no_op_prediction)
    )
    correction_mask = (
        (temporal_support >= int(min_votes))
        & (temporal_confidence >= float(confidence_threshold))
        & (temporal_prediction != original_prediction)
        & structural_support
        & (split_targets < 0)
    )
    if refine_mask is not None:
        correction_mask = correction_mask & refine_mask
        adapt_mask = refine_mask & (split_targets < 0)
    else:
        adapt_mask = split_targets < 0
    keep_mask = (
        (original_prediction == temporal_prediction)
        | (original_prediction == no_op_prediction)
        | (original_prediction == base_prediction)
    ) & ~correction_mask
    support_mask, query_mask = _region_support_query_masks(regions)

    support_corrections = correction_mask & support_mask
    query_corrections = correction_mask & query_mask
    if not support_corrections.any() or not query_corrections.any():
        return original_probability, False, {
            "query_gain": 0.0,
            "correction_ratio": float(correction_mask.float().mean().item()),
            "support_corrections": int(support_corrections.sum().item()),
            "query_corrections": int(query_corrections.sum().item()),
            "scales": [float(initial_scale)] * len(components),
            "bias_norm": 0.0,
        }

    class_count = base_scores.size(1)
    gate_shape = (len(components), class_count) if classwise else (len(components),)
    initial_logit = _initial_gate_logit(initial_scale, min_scale, max_scale)
    gate_logit = base_scores.new_full(gate_shape, initial_logit, requires_grad=True)
    residual_bias = base_scores.new_zeros(class_count, requires_grad=True) if adapt_bias else None

    for _ in range(max(int(inner_steps), 1)):
        adapted_scores = _compose_scores(
            base_scores,
            components,
            gate_logit,
            regions,
            split_targets,
            residual_scale,
            min_scale,
            max_scale,
            residual_bias,
            adapt_mask,
        )
        adapted_log_probability = F.log_softmax(adapted_scores, dim=1)
        gate_scale = _gate_from_logit(gate_logit, min_scale, max_scale)
        support_loss = _objective(
            adapted_log_probability,
            original_probability,
            temporal_prediction,
            temporal_confidence,
            correction_mask,
            keep_mask,
            support_mask,
            correction_weight,
            keep_weight,
            entropy_weight,
            gate_scale,
            scale_regularization,
            initial_scale,
            residual_bias,
            bias_regularization,
        )
        optimized_parameters = [gate_logit]
        if residual_bias is not None:
            optimized_parameters.append(residual_bias)
        gradients = torch.autograd.grad(support_loss, optimized_parameters)
        gate_logit = (gate_logit - float(inner_lr) * gradients[0]).detach().requires_grad_(True)
        if residual_bias is not None:
            residual_bias = (
                residual_bias - float(inner_lr) * gradients[1]
            ).detach().requires_grad_(True)

    adapted_scores = _compose_scores(
        base_scores,
        components,
        gate_logit,
        regions,
        split_targets,
        residual_scale,
        min_scale,
        max_scale,
        residual_bias,
        adapt_mask,
    )
    adapted_log_probability = F.log_softmax(adapted_scores, dim=1)
    adapted_scale = _gate_from_logit(gate_logit, min_scale, max_scale)
    adapted_query_loss = _objective(
        adapted_log_probability,
        original_probability,
        temporal_prediction,
        temporal_confidence,
        correction_mask,
        keep_mask,
        query_mask,
        correction_weight,
        keep_weight,
        entropy_weight,
        adapted_scale,
        scale_regularization,
        initial_scale,
        residual_bias,
        bias_regularization,
    )
    original_log_probability = torch.log(original_probability.clamp_min(1e-6))
    initial_scale_tensor = adapted_scale.new_full(adapted_scale.shape, float(initial_scale))
    original_query_loss = _objective(
        original_log_probability,
        original_probability,
        temporal_prediction,
        temporal_confidence,
        correction_mask,
        keep_mask,
        query_mask,
        correction_weight,
        keep_weight,
        entropy_weight,
        initial_scale_tensor,
        scale_regularization,
        initial_scale,
        residual_bias=original_probability.new_zeros(class_count) if adapt_bias else None,
        bias_regularization=bias_regularization,
    )
    query_gain = float((original_query_loss - adapted_query_loss).detach().item())
    accepted = query_gain >= float(query_tolerance)
    selected_probability = adapted_log_probability.detach().exp() if accepted else original_probability
    scale_values = adapted_scale.detach().mean(dim=-1) if adapted_scale.ndim == 2 else adapted_scale.detach()
    return selected_probability, accepted, {
        "query_gain": query_gain,
        "correction_ratio": float(correction_mask.float().mean().item()),
        "support_corrections": int(support_corrections.sum().item()),
        "query_corrections": int(query_corrections.sum().item()),
        "scales": [float(value.item()) for value in scale_values],
        "bias_norm": float(residual_bias.detach().norm().item()) if residual_bias is not None else 0.0,
        "classwise": bool(classwise),
    }
