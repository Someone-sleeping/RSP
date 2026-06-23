import torch
import torch.nn.functional as F


def _safe_top2(probs):
    if probs.size(1) == 1:
        top1 = probs[:, 0]
        top2 = torch.zeros_like(top1)
        pred = torch.zeros_like(top1, dtype=torch.long)
        return top1, top2, pred
    top_values, top_indices = torch.topk(probs, k=2, dim=1)
    return top_values[:, 0], top_values[:, 1], top_indices[:, 0]


def build_error_queries(
    logits,
    pseudo_labels,
    regions,
    batch_ids,
    coords,
    colors=None,
    prob_threshold=0.7,
    margin_threshold=0.2,
    region_purity_threshold=0.8,
    color_consistency_threshold=0.35,
    geometry_consistency_threshold=0.65,
    min_region_points=10,
    max_queries_per_scene=20,
):
    """Select conservative suspect-region query anchors for refinement.

    A suspect region is geometrically/color coherent but semantically unstable:
    mixed pseudo labels, high uncertainty, low support for the pseudo label, or
    disagreement between the model prediction and teacher pseudo label. Stable
    high-confidence regions are returned separately as keep regions so the
    refiner can be regularized toward a no-op there.
    """
    device = logits.device
    pseudo_labels = pseudo_labels.long().to(device)
    regions = regions.squeeze(-1).long().to(device)
    batch_ids = batch_ids.long().to(device)
    coords = coords.to(device)
    if colors is not None:
        colors = colors.to(device).float()

    valid_label = pseudo_labels >= 0
    probs = F.softmax(logits.detach(), dim=1)
    top1, top2, pred = _safe_top2(probs)
    margin = top1 - top2

    pseudo_prob = torch.zeros_like(top1)
    pseudo_valid = valid_label & (pseudo_labels < probs.size(1))
    pseudo_prob[pseudo_valid] = probs[pseudo_valid, pseudo_labels[pseudo_valid]]

    model_high_conf_point = pseudo_valid & (pseudo_prob >= prob_threshold) & (margin >= margin_threshold)
    pred_match_point = pseudo_valid & (pred == pseudo_labels)

    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
    entropy = entropy / max(float(torch.log(torch.tensor(probs.size(1), device=device))), 1e-6)

    query_indices = []
    selected_refine_masks = []
    selected_keep_masks = []
    candidate_region_mask = torch.zeros_like(model_high_conf_point)
    keep_region_mask = torch.zeros_like(model_high_conf_point)
    regions_after_size = 0
    regions_after_purity = 0
    regions_after_attr = 0
    regions_suspect = 0
    regions_keep = 0
    unique_batches = torch.unique(batch_ids)

    for batch_id in unique_batches:
        scene_mask = batch_ids == batch_id
        scene_regions = torch.unique(regions[scene_mask])
        suspect_candidates = []
        keep_candidates = []

        for region_id in scene_regions:
            if region_id.item() == -1:
                continue
            mask = scene_mask & (regions == region_id) & pseudo_valid
            count = int(mask.sum().item())
            if count < min_region_points:
                continue
            regions_after_size += 1

            region_labels = pseudo_labels[mask]
            _, label_counts = torch.unique(region_labels, return_counts=True)
            purity = label_counts.max().float() / float(count)
            if purity >= region_purity_threshold:
                regions_after_purity += 1

            region_coords = coords[mask].float()
            coord_center = region_coords.mean(dim=0, keepdim=True)
            coord_dist = torch.norm(region_coords - coord_center, dim=1)
            coord_scale = torch.norm(
                coords[scene_mask].float().max(dim=0)[0] - coords[scene_mask].float().min(dim=0)[0],
                p=2,
            ).clamp_min(1.0)
            geometry_spread = (coord_dist.mean() / coord_scale).clamp(0.0, 1.0)
            geometry_consistency = 1.0 - geometry_spread

            if colors is not None:
                region_colors = colors[mask].float()
                color_std = region_colors.std(dim=0, unbiased=False).mean()
                color_consistency = torch.exp(-6.0 * color_std).clamp(0.0, 1.0)
            else:
                color_consistency = torch.tensor(1.0, device=device)

            if color_consistency < color_consistency_threshold:
                continue
            if geometry_consistency < geometry_consistency_threshold:
                continue
            regions_after_attr += 1
            candidate_region_mask[mask] = True

            disagreement = (pred[mask] != pseudo_labels[mask]).float().mean()
            uncertainty = entropy[mask].mean()
            model_support = model_high_conf_point[mask].float().mean()
            instability = 1.0 - model_support
            attr_risk = (1.0 - color_consistency) + (1.0 - geometry_consistency)
            semantic_mixing = 1.0 - purity
            score = 2.0 * semantic_mixing + disagreement + uncertainty + instability + 0.5 * attr_risk

            local_indices = torch.nonzero(mask, as_tuple=False).flatten()
            local_coords = coords[local_indices].float()
            centroid = local_coords.mean(dim=0, keepdim=True)
            distances = torch.norm(local_coords - centroid, dim=1)

            is_keep_region = (
                purity >= region_purity_threshold
                and model_support >= 0.8
                and disagreement <= 0.05
                and uncertainty <= 0.35
            )
            if is_keep_region:
                regions_keep += 1
                keep_region_mask[mask] = True
                keep_candidates.append((float(model_support.item()), mask))
                continue

            is_suspect_region = (
                semantic_mixing > (1.0 - region_purity_threshold)
                or disagreement > 0.05
                or uncertainty > 0.35
                or model_support < 0.8
            )
            if not is_suspect_region:
                continue
            regions_suspect += 1

            if disagreement > 0:
                wrong_local = torch.nonzero(pred[local_indices] != pseudo_labels[local_indices], as_tuple=False).flatten()
                wrong_indices = local_indices[wrong_local]
                wrong_coords = coords[wrong_indices].float()
                wrong_centroid = wrong_coords.mean(dim=0, keepdim=True)
                wrong_distances = torch.norm(wrong_coords - wrong_centroid, dim=1)
                selected = wrong_indices[torch.argmin(wrong_distances)]
            else:
                # Within a trusted region, the most uncertain point is the best
                # lightweight proxy for a user/query anchor.
                selected = local_indices[torch.argmax(entropy[local_indices] - 0.01 * distances)]

            suspect_candidates.append((float(score.item()), int(selected.item()), mask))

        suspect_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, idx, mask in suspect_candidates[:max_queries_per_scene]:
            query_indices.append(idx)
            selected_refine_masks.append(mask)
        keep_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, mask in keep_candidates[:max_queries_per_scene]:
            selected_keep_masks.append(mask)

    if query_indices:
        query_indices = torch.tensor(query_indices, dtype=torch.long, device=device)
    else:
        query_indices = torch.empty(0, dtype=torch.long, device=device)

    if selected_refine_masks:
        refine_mask = torch.stack(selected_refine_masks, dim=0).any(dim=0)
    else:
        refine_mask = torch.zeros_like(model_high_conf_point)

    if selected_keep_masks:
        keep_mask = torch.stack(selected_keep_masks, dim=0).any(dim=0)
    else:
        keep_mask = torch.zeros_like(model_high_conf_point)

    stats = {
        "num_queries": int(query_indices.numel()),
        "trusted_ratio": float(refine_mask.float().mean().item()) if refine_mask.numel() > 0 else 0.0,
        "refine_ratio": float(refine_mask.float().mean().item()) if refine_mask.numel() > 0 else 0.0,
        "keep_ratio": float(keep_mask.float().mean().item()) if keep_mask.numel() > 0 else 0.0,
        "candidate_ratio": float(candidate_region_mask.float().mean().item()) if candidate_region_mask.numel() > 0 else 0.0,
        "keep_candidate_ratio": float(keep_region_mask.float().mean().item()) if keep_region_mask.numel() > 0 else 0.0,
        "point_high_conf_ratio": float(model_high_conf_point.float().mean().item()) if model_high_conf_point.numel() > 0 else 0.0,
        "pred_match_ratio": float(pred_match_point.float().mean().item()) if pred_match_point.numel() > 0 else 0.0,
        "regions_after_size": regions_after_size,
        "regions_after_purity": regions_after_purity,
        "regions_after_attr": regions_after_attr,
        "regions_suspect": regions_suspect,
        "regions_keep": regions_keep,
    }
    return query_indices, refine_mask, keep_mask, stats
