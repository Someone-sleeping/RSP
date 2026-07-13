import torch
import torch.nn.functional as F


def _safe_norm(x):
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)


def _two_means(features, iterations=6):
    center = features.mean(dim=0, keepdim=True)
    distances = torch.norm(features - center, dim=1)
    c0 = features[torch.argmax(distances)].clone()
    distances = torch.norm(features - c0.unsqueeze(0), dim=1)
    c1 = features[torch.argmax(distances)].clone()
    labels = torch.zeros(features.size(0), dtype=torch.long, device=features.device)

    for _ in range(iterations):
        d0 = torch.norm(features - c0.unsqueeze(0), dim=1)
        d1 = torch.norm(features - c1.unsqueeze(0), dim=1)
        labels = (d1 < d0).long()
        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            break
        c0 = features[labels == 0].mean(dim=0)
        c1 = features[labels == 1].mean(dim=0)
    return labels


def _region_split_features(point_feats, coords, colors, probs, mask, xyz_weight, rgb_weight, feat_weight, semantic_weight):
    pieces = []
    region_coords = coords[mask].float()
    pieces.append(xyz_weight * _safe_norm(region_coords))
    if colors is not None and rgb_weight > 0:
        pieces.append(rgb_weight * _safe_norm(colors[mask].float()))
    if point_feats is not None and feat_weight > 0:
        pieces.append(feat_weight * F.normalize(point_feats[mask].float(), dim=1))
    if semantic_weight > 0:
        pieces.append(semantic_weight * probs[mask].float())
    return torch.cat(pieces, dim=1)


def _cluster_target(probs, mask):
    mean_prob = probs[mask].mean(dim=0)
    conf, target = mean_prob.max(dim=0)
    return target.long(), conf


def _score_split(probs, coords, global0, global1, conf0, conf1):
    prob0 = probs[global0].mean(dim=0)
    prob1 = probs[global1].mean(dim=0)
    semantic_sep = torch.abs(prob0 - prob1).sum()
    coord0 = coords[global0].float().mean(dim=0)
    coord1 = coords[global1].float().mean(dim=0)
    coord_scale = coords[torch.cat([global0, global1])].float().std(dim=0, unbiased=False).norm().clamp_min(1e-6)
    geometry_sep = torch.norm(coord0 - coord1) / coord_scale
    balance = min(float(global0.numel()), float(global1.numel())) / max(float(global0.numel() + global1.numel()), 1.0)
    return float((conf0 + conf1 + semantic_sep + 0.1 * geometry_sep).item()) + 0.2 * balance


def build_split_region_queries(
    logits,
    point_feats,
    coords,
    colors,
    regions,
    batch_ids,
    min_region_points=20,
    min_child_points=8,
    max_split_regions_per_scene=20,
    split_purity_threshold=0.92,
    split_entropy_threshold=0.25,
    split_min_conf=0.15,
    xyz_weight=1.0,
    rgb_weight=0.5,
    feat_weight=0.25,
    semantic_weight=1.0,
    multi_proposal=False,
    selection_mode="score",
    random_seed=0,
):
    """Split suspect GrowSP regions into semantic sub-regions.

    Returns query anchors, a refine mask, point targets for accepted sub-regions,
    a keep mask for stable regions, and diagnostics. The split is deliberately
    conservative: a region is used as split supervision only when the two child
    clusters have different semantic consensus targets.
    """
    device = logits.device
    coords = coords.to(device)
    regions = regions.squeeze(-1).long().to(device)
    batch_ids = batch_ids.long().to(device)
    if colors is not None:
        colors = colors.to(device).float()
    if point_feats is not None:
        point_feats = point_feats.to(device)

    probs = F.softmax(logits.detach(), dim=1)
    pred = probs.argmax(dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
    entropy = entropy / max(float(torch.log(torch.tensor(probs.size(1), device=device))), 1e-6)

    refine_mask = torch.zeros((logits.size(0),), dtype=torch.bool, device=device)
    keep_mask = torch.zeros_like(refine_mask)
    split_targets = torch.full((logits.size(0),), -1, dtype=torch.long, device=device)
    split_target_conf = logits.new_zeros((logits.size(0),))
    query_indices = []
    split_regions = 0
    split_subregions = 0
    candidate_regions = 0
    stable_regions = 0

    for batch_id in torch.unique(batch_ids):
        scene_mask = batch_ids == batch_id
        candidates = []
        stable = []

        for region_id in torch.unique(regions[scene_mask]):
            if region_id.item() == -1:
                continue
            mask = scene_mask & (regions == region_id)
            count = int(mask.sum().item())
            if count < min_region_points:
                continue

            labels, counts = torch.unique(pred[mask], return_counts=True)
            purity = counts.max().float() / mask.sum().float().clamp_min(1.0)
            region_entropy = entropy[mask].mean()
            score = (1.0 - purity) + region_entropy

            if purity >= split_purity_threshold and region_entropy < split_entropy_threshold:
                stable.append((float(purity.item()), mask))
                continue

            candidate_regions += 1
            candidates.append((float(score.item()), mask))

        if selection_mode == "random" and len(candidates) > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_seed) + int(batch_id.item()))
            order = torch.randperm(len(candidates), generator=generator).tolist()
            candidates = [candidates[idx] for idx in order]
        else:
            candidates.sort(key=lambda item: item[0], reverse=True)
        for _, mask in candidates[:max_split_regions_per_scene]:
            local_indices = torch.nonzero(mask, as_tuple=False).flatten()
            proposals = [(xyz_weight, rgb_weight, feat_weight, semantic_weight)]
            if multi_proposal:
                proposals.extend(
                    [
                        (max(xyz_weight, 1.0), max(rgb_weight, 0.5), 0.0, 0.0),
                        (max(xyz_weight, 1.0), 0.0, 0.0, max(semantic_weight, 1.0)),
                        (0.0, 0.0, max(feat_weight, 0.25), max(semantic_weight, 1.0)),
                    ]
                )
            best = None
            for proposal in proposals:
                split_features = _region_split_features(
                    point_feats,
                    coords,
                    colors,
                    probs,
                    mask,
                    proposal[0],
                    proposal[1],
                    proposal[2],
                    proposal[3],
                )
                child_labels = _two_means(split_features)
                child0 = child_labels == 0
                child1 = child_labels == 1
                if int(child0.sum().item()) < min_child_points or int(child1.sum().item()) < min_child_points:
                    continue

                global0 = local_indices[child0]
                global1 = local_indices[child1]
                target0, conf0 = _cluster_target(probs, global0)
                target1, conf1 = _cluster_target(probs, global1)
                if conf0.item() < split_min_conf or conf1.item() < split_min_conf:
                    continue
                if target0.item() == target1.item():
                    continue
                score = _score_split(probs, coords, global0, global1, conf0, conf1)
                if best is None or score > best[0]:
                    best = (score, global0, global1, target0, target1, conf0, conf1)

            if best is None:
                continue

            _, global0, global1, target0, target1, conf0, conf1 = best
            refine_mask[global0] = True
            refine_mask[global1] = True
            split_targets[global0] = target0
            split_targets[global1] = target1
            split_target_conf[global0] = conf0
            split_target_conf[global1] = conf1
            query_indices.append(int(global0[torch.argmax(entropy[global0])].item()))
            query_indices.append(int(global1[torch.argmax(entropy[global1])].item()))
            split_regions += 1
            split_subregions += 2

        stable.sort(key=lambda item: item[0], reverse=True)
        for _, mask in stable[:max_split_regions_per_scene]:
            keep_mask[mask] = True
            stable_regions += 1

    if query_indices:
        query_indices = torch.tensor(query_indices, dtype=torch.long, device=device)
    else:
        query_indices = torch.empty(0, dtype=torch.long, device=device)

    stats = {
        "split_queries": int(query_indices.numel()),
        "split_refine_ratio": float(refine_mask.float().mean().item()) if refine_mask.numel() else 0.0,
        "split_keep_ratio": float(keep_mask.float().mean().item()) if keep_mask.numel() else 0.0,
        "split_candidate_regions": candidate_regions,
        "split_regions": split_regions,
        "split_subregions": split_subregions,
        "split_stable_regions": stable_regions,
    }
    return query_indices, refine_mask, split_targets, split_target_conf, keep_mask, stats


def build_uncertain_region_queries(logits, regions, batch_ids, min_region_points=10, max_queries_per_scene=20):
    device = logits.device
    regions = regions.squeeze(-1).long().to(device)
    batch_ids = batch_ids.long().to(device)
    probs = F.softmax(logits.detach(), dim=1)
    pred = probs.argmax(dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
    entropy = entropy / max(float(torch.log(torch.tensor(probs.size(1), device=device))), 1e-6)

    query_indices = []
    masks = []
    selected_regions = 0
    for batch_id in torch.unique(batch_ids):
        scene_mask = batch_ids == batch_id
        candidates = []
        for region_id in torch.unique(regions[scene_mask]):
            if region_id.item() == -1:
                continue
            mask = scene_mask & (regions == region_id)
            if int(mask.sum().item()) < min_region_points:
                continue
            _, counts = torch.unique(pred[mask], return_counts=True)
            purity = counts.max().float() / mask.sum().float().clamp_min(1.0)
            score = entropy[mask].mean() + (1.0 - purity)
            local_indices = torch.nonzero(mask, as_tuple=False).flatten()
            selected = local_indices[torch.argmax(entropy[local_indices])]
            candidates.append((float(score.item()), int(selected.item()), mask))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, selected, mask in candidates[:max_queries_per_scene]:
            query_indices.append(selected)
            masks.append(mask)
            selected_regions += 1

    if query_indices:
        query_indices = torch.tensor(query_indices, dtype=torch.long, device=device)
        refine_mask = torch.stack(masks, dim=0).any(dim=0)
    else:
        query_indices = torch.empty(0, dtype=torch.long, device=device)
        refine_mask = torch.zeros((logits.size(0),), dtype=torch.bool, device=device)

    return query_indices, refine_mask, {"fallback_queries": int(query_indices.numel()), "fallback_regions": selected_regions}


def build_region_consistency_queries(
    logits,
    regions,
    batch_ids,
    min_region_points=20,
    max_regions_per_scene=40,
    min_region_conf=0.35,
    min_disagree_ratio=0.02,
    point_conf_threshold=0.55,
    point_entropy_threshold=0.55,
):
    """Use region-level semantic consensus as self-supervised correction targets.

    This complements split-region supervision. Split queries handle mixed
    superpoints; consistency queries handle noisy points inside otherwise
    coherent regions. It never uses GT labels: targets come from the frozen
    teacher's region-averaged semantic distribution.
    """
    device = logits.device
    regions = regions.squeeze(-1).long().to(device)
    batch_ids = batch_ids.long().to(device)

    probs = F.softmax(logits.detach(), dim=1)
    pred = probs.argmax(dim=1)
    point_conf = probs.max(dim=1)[0]
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
    entropy = entropy / max(float(torch.log(torch.tensor(probs.size(1), device=device))), 1e-6)

    refine_mask = torch.zeros((logits.size(0),), dtype=torch.bool, device=device)
    keep_mask = torch.zeros_like(refine_mask)
    targets = torch.full((logits.size(0),), -1, dtype=torch.long, device=device)
    target_conf = logits.new_zeros((logits.size(0),))
    query_indices = []
    selected_regions = 0
    corrected_points = 0
    kept_points = 0

    for batch_id in torch.unique(batch_ids):
        scene_mask = batch_ids == batch_id
        candidates = []
        for region_id in torch.unique(regions[scene_mask]):
            if region_id.item() == -1:
                continue
            mask = scene_mask & (regions == region_id)
            count = int(mask.sum().item())
            if count < min_region_points:
                continue

            region_prob = probs[mask].mean(dim=0)
            region_conf, region_target = region_prob.max(dim=0)
            if region_conf.item() < min_region_conf:
                continue

            local_pred = pred[mask]
            disagree = local_pred != region_target
            local_entropy = entropy[mask]
            local_conf = point_conf[mask]
            noisy = disagree | (local_conf < point_conf_threshold) | (local_entropy > point_entropy_threshold)
            disagree_ratio = disagree.float().mean()
            noisy_ratio = noisy.float().mean()
            if disagree_ratio.item() < min_disagree_ratio and noisy_ratio.item() < min_disagree_ratio:
                keep_mask[mask] = True
                kept_points += count
                continue

            score = float((region_conf * (disagree_ratio + noisy_ratio)).item())
            candidates.append((score, mask, region_target.long(), region_conf))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, mask, region_target, region_conf in candidates[:max_regions_per_scene]:
            local_indices = torch.nonzero(mask, as_tuple=False).flatten()
            correction_mask = mask & (
                (pred != region_target)
                | (point_conf < point_conf_threshold)
                | (entropy > point_entropy_threshold)
            )
            if correction_mask.sum() == 0:
                continue
            correction_indices = torch.nonzero(correction_mask, as_tuple=False).flatten()
            refine_mask[correction_indices] = True
            targets[correction_indices] = region_target
            target_conf[correction_indices] = region_conf
            query_indices.append(int(correction_indices[torch.argmax(entropy[correction_indices])].item()))
            stable = mask & ~correction_mask
            keep_mask[stable] = True
            selected_regions += 1
            corrected_points += int(correction_indices.numel())
            kept_points += int(stable.sum().item())

    if query_indices:
        query_indices = torch.tensor(query_indices, dtype=torch.long, device=device)
    else:
        query_indices = torch.empty(0, dtype=torch.long, device=device)

    stats = {
        "consistency_queries": int(query_indices.numel()),
        "consistency_refine_ratio": float(refine_mask.float().mean().item()) if refine_mask.numel() else 0.0,
        "consistency_keep_ratio": float(keep_mask.float().mean().item()) if keep_mask.numel() else 0.0,
        "consistency_regions": selected_regions,
        "consistency_points": corrected_points,
        "consistency_keep_points": kept_points,
    }
    return query_indices, refine_mask, targets, target_conf, keep_mask, stats
