import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LearnableSuperpointOutput:
    dynamic_regions: torch.Tensor
    supervision_targets: torch.Tensor
    supervision_confidence: torch.Tensor
    supervision_mask: torch.Tensor
    candidate_mask: torch.Tensor
    structure_loss: torch.Tensor
    feature_loss: torch.Tensor
    geometry_loss: torch.Tensor
    semantic_loss: torch.Tensor
    entropy_loss: torch.Tensor
    balance_loss: torch.Tensor
    stats: dict


def _standardize(values):
    return (values - values.mean(dim=0, keepdim=True)) / values.std(
        dim=0, keepdim=True, unbiased=False
    ).clamp_min(1e-5)


def verified_region_supervision_loss(logits, output, ignore_index=-1):
    """Cross entropy induced by verified dynamic subregions."""
    if not output.supervision_mask.any():
        return logits.sum() * 0.0
    targets = output.supervision_targets.clone()
    targets[~output.supervision_mask] = ignore_index
    point_loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")
    weights = output.supervision_confidence.detach() * output.supervision_mask.float()
    return (point_loss * weights).sum() / weights.sum().clamp_min(1.0)


class SemanticDifferenceSuperpointLearner(nn.Module):
    """Learn a local superpoint split inside the unsupervised training loop.

    Candidate selection and acceptance are deliberately non-differentiable gates.
    The local association itself is differentiable and is optimized by structural
    objectives; accepted child regions modify region aggregation and pseudo
    supervision in the same training iteration.
    """

    def __init__(
        self,
        feat_dim,
        num_classes,
        hidden_dim=64,
        iterations=3,
        temperature=0.2,
        feature_weight=1.0,
        geometry_weight=0.2,
        semantic_weight=1.0,
        entropy_weight=0.02,
        balance_weight=0.05,
    ):
        super().__init__()
        self.iterations = iterations
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.geometry_weight = geometry_weight
        self.semantic_weight = semantic_weight
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attribute_encoder = nn.Sequential(
            nn.Linear(6 + num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.affinity = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.affinity[-1].weight)
        nn.init.zeros_(self.affinity[-1].bias)

    def _encode_region(self, point_feats, coords, colors, probabilities, indices):
        xyz = _standardize(coords[indices].float())
        rgb = _standardize(colors[indices].float())
        semantic = probabilities[indices].detach()
        feature_tokens = self.feature_encoder(point_feats[indices])
        attribute_tokens = self.attribute_encoder(torch.cat([xyz, rgb, semantic], dim=1))
        return F.normalize(feature_tokens + attribute_tokens, dim=1), xyz, semantic

    def _associate(self, tokens, semantic):
        confidence, prediction = semantic.max(dim=1)
        first = torch.argmax(confidence)
        alternatives = prediction != prediction[first]
        if alternatives.any():
            second = torch.argmax(confidence.masked_fill(~alternatives, -1.0))
        else:
            second = torch.argmax(1.0 - F.cosine_similarity(tokens, tokens[first].unsqueeze(0), dim=1))
        seed_indices = torch.stack([first, second])
        centers = tokens.detach()[seed_indices]
        semantic_centers = semantic[seed_indices]
        assignment = None
        for _ in range(self.iterations):
            point_tokens = tokens[:, None, :].expand(-1, 2, -1)
            center_tokens = centers[None, :, :].expand(tokens.size(0), -1, -1)
            pair = torch.cat([point_tokens, center_tokens, torch.abs(point_tokens - center_tokens)], dim=2)
            learned_affinity = self.affinity(pair).squeeze(-1)
            cosine_affinity = (point_tokens * center_tokens).sum(dim=2)
            semantic_affinity = semantic @ F.normalize(semantic_centers, dim=1).transpose(0, 1)
            assignment = F.softmax(
                (cosine_affinity + semantic_affinity + learned_affinity) / self.temperature,
                dim=1,
            )
            denominator = assignment.sum(dim=0).unsqueeze(1).clamp_min(1e-5)
            centers = F.normalize(assignment.transpose(0, 1) @ tokens / denominator, dim=1)
            semantic_centers = assignment.transpose(0, 1) @ semantic / denominator
        return assignment, centers

    def _candidate_regions(
        self,
        probabilities,
        regions,
        batch_ids,
        min_region_points,
        purity_threshold,
        entropy_threshold,
        max_regions_per_scene,
    ):
        predictions = probabilities.argmax(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1)
        entropy = entropy / max(math.log(probabilities.size(1)), 1e-6)
        selected = []
        candidate_mask = torch.zeros_like(regions, dtype=torch.bool)
        candidate_count = 0
        for batch_id in torch.unique(batch_ids):
            scene_mask = batch_ids == batch_id
            candidates = []
            for region_id in torch.unique(regions[scene_mask]):
                if int(region_id.item()) < 0:
                    continue
                mask = scene_mask & (regions == region_id)
                if int(mask.sum().item()) < min_region_points:
                    continue
                _, counts = torch.unique(predictions[mask], return_counts=True)
                purity = counts.max().float() / mask.sum().float()
                region_entropy = entropy[mask].mean()
                if purity >= purity_threshold and region_entropy <= entropy_threshold:
                    continue
                score = (1.0 - purity) + region_entropy
                candidates.append((float(score.item()), mask))
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidate_count += len(candidates)
            for _, mask in candidates[:max_regions_per_scene]:
                candidate_mask[mask] = True
                selected.append(mask)
        return selected, candidate_mask, candidate_count

    @staticmethod
    def _weighted_compactness(values, assignment, centers=None):
        if centers is None:
            denominator = assignment.sum(dim=0).unsqueeze(1).clamp_min(1e-5)
            centers = assignment.transpose(0, 1) @ values / denominator
        distance = (values[:, None, :] - centers[None, :, :]).pow(2).mean(dim=2)
        return (assignment * distance).sum() / assignment.sum().clamp_min(1.0)

    def forward(
        self,
        point_feats,
        coords,
        colors,
        semantic_logits,
        regions,
        batch_ids,
        min_region_points=20,
        min_child_points=6,
        max_regions_per_scene=12,
        purity_threshold=0.9,
        entropy_threshold=0.3,
        min_child_confidence=0.2,
        min_confidence_gain=0.01,
        min_semantic_separation=0.15,
    ):
        device = point_feats.device
        coords = coords.to(device).float()
        colors = colors.to(device).float()
        regions = regions.to(device).long().view(-1)
        batch_ids = batch_ids.to(device).long().view(-1)
        probabilities = F.softmax(semantic_logits.detach(), dim=1)

        selected, candidate_mask, candidate_count = self._candidate_regions(
            probabilities,
            regions,
            batch_ids,
            min_region_points,
            purity_threshold,
            entropy_threshold,
            max_regions_per_scene,
        )

        dynamic_regions = torch.full_like(regions, -1)
        supervision_targets = torch.full_like(regions, -1)
        supervision_confidence = semantic_logits.new_zeros(regions.shape)
        supervision_mask = torch.zeros_like(regions, dtype=torch.bool)
        accepted_children = []
        losses = {name: semantic_logits.sum() * 0.0 for name in ("feature", "geometry", "semantic", "entropy", "balance")}
        accepted = 0
        proposed = 0
        rejected_size = 0
        rejected_same_target = 0
        rejected_confidence = 0
        rejected_gain = 0
        rejected_separation = 0

        for mask in selected:
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            tokens, normalized_xyz, semantic = self._encode_region(
                point_feats, coords, colors, probabilities, indices
            )
            assignment, centers = self._associate(tokens, semantic)
            proposed += 1

            semantic_centers = assignment.transpose(0, 1) @ semantic
            semantic_centers = semantic_centers / assignment.sum(dim=0).unsqueeze(1).clamp_min(1e-5)
            xyz_centers = assignment.transpose(0, 1) @ normalized_xyz
            xyz_centers = xyz_centers / assignment.sum(dim=0).unsqueeze(1).clamp_min(1e-5)
            losses["feature"] = losses["feature"] + self._weighted_compactness(tokens, assignment, centers)
            losses["geometry"] = losses["geometry"] + self._weighted_compactness(
                normalized_xyz, assignment, xyz_centers
            )
            losses["semantic"] = losses["semantic"] + self._weighted_compactness(
                semantic, assignment, semantic_centers
            )
            losses["entropy"] = losses["entropy"] + (
                -(assignment * assignment.clamp_min(1e-6).log()).sum(dim=1).mean()
            )
            losses["balance"] = losses["balance"] + (assignment.mean(dim=0) - 0.5).pow(2).mean()

            hard_assignment = assignment.detach().argmax(dim=1)
            child_sizes = torch.bincount(hard_assignment, minlength=2)
            if not bool((child_sizes >= min_child_points).all().item()):
                rejected_size += 1
                continue
            hard_semantic_centers = torch.stack(
                [semantic[hard_assignment == child_id].mean(dim=0) for child_id in range(2)], dim=0
            )
            child_confidence, child_targets = hard_semantic_centers.max(dim=1)
            parent_confidence = semantic.mean(dim=0).max()
            child_weight = child_sizes.float() / child_sizes.sum().clamp_min(1)
            confidence_gain = (child_weight * child_confidence).sum() - parent_confidence
            semantic_separation = torch.abs(hard_semantic_centers[0] - hard_semantic_centers[1]).sum()
            if int(child_targets[0].item()) == int(child_targets[1].item()):
                rejected_same_target += 1
                continue
            if not bool((child_confidence >= min_child_confidence).all().item()):
                rejected_confidence += 1
                continue
            if float(confidence_gain.item()) < min_confidence_gain:
                rejected_gain += 1
                continue
            if float(semantic_separation.item()) < min_semantic_separation:
                rejected_separation += 1
                continue

            accepted += 1
            accepted_children.append((indices, hard_assignment))
            for child_id in range(2):
                child_indices = indices[hard_assignment == child_id]
                supervision_targets[child_indices] = child_targets[child_id]
                supervision_confidence[child_indices] = child_confidence[child_id]
                supervision_mask[child_indices] = True

        normalizer = max(proposed, 1)
        for name in losses:
            losses[name] = losses[name] / normalizer
        structure_loss = (
            self.feature_weight * losses["feature"]
            + self.geometry_weight * losses["geometry"]
            + self.semantic_weight * losses["semantic"]
            + self.entropy_weight * losses["entropy"]
            + self.balance_weight * losses["balance"]
        )

        next_region = 0
        accepted_by_point = {}
        for indices, hard_assignment in accepted_children:
            for point_index, child_id in zip(indices.tolist(), hard_assignment.tolist()):
                accepted_by_point[point_index] = child_id
        for batch_id in torch.unique(batch_ids):
            scene_mask = batch_ids == batch_id
            for region_id in torch.unique(regions[scene_mask]):
                mask = scene_mask & (regions == region_id)
                if int(region_id.item()) < 0:
                    continue
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                child_values = [accepted_by_point.get(int(index.item()), -1) for index in indices]
                if child_values and child_values[0] >= 0:
                    child_tensor = torch.tensor(child_values, device=device)
                    dynamic_regions[indices[child_tensor == 0]] = next_region
                    next_region += 1
                    dynamic_regions[indices[child_tensor == 1]] = next_region
                    next_region += 1
                else:
                    dynamic_regions[indices] = next_region
                    next_region += 1

        stats = {
            "candidate_regions": candidate_count,
            "selected_regions": len(selected),
            "proposed_splits": proposed,
            "accepted_splits": accepted,
            "rejected_size": rejected_size,
            "rejected_same_target": rejected_same_target,
            "rejected_confidence": rejected_confidence,
            "rejected_gain": rejected_gain,
            "rejected_separation": rejected_separation,
            "dynamic_regions": next_region,
            "supervised_ratio": float(supervision_mask.float().mean().item()) if regions.numel() else 0.0,
            "candidate_ratio": float(candidate_mask.float().mean().item()) if regions.numel() else 0.0,
        }
        return LearnableSuperpointOutput(
            dynamic_regions=dynamic_regions,
            supervision_targets=supervision_targets,
            supervision_confidence=supervision_confidence,
            supervision_mask=supervision_mask,
            candidate_mask=candidate_mask,
            structure_loss=structure_loss,
            feature_loss=losses["feature"],
            geometry_loss=losses["geometry"],
            semantic_loss=losses["semantic"],
            entropy_loss=losses["entropy"],
            balance_loss=losses["balance"],
            stats=stats,
        )
