import torch
import torch.nn as nn
import torch.nn.functional as F


def gate_refiner_residual(delta_logits, candidate_mask):
    """Restrict a learned semantic residual to its pseudo-supervised support."""
    if delta_logits.ndim != 2:
        raise ValueError("delta_logits must have shape [N, C]")
    candidate_mask = candidate_mask.to(device=delta_logits.device, dtype=torch.bool).view(-1)
    if candidate_mask.numel() != delta_logits.size(0):
        raise ValueError("candidate_mask must contain one value per point")
    return delta_logits * candidate_mask.unsqueeze(1).to(delta_logits.dtype)


def resolve_min_temporal_votes(requested_votes, reference_count):
    """Resolve automatic majority voting without weakening an explicit threshold."""
    if reference_count < 1:
        raise ValueError("At least one historical reference checkpoint is required")
    if requested_votes <= 0:
        return max(1, (reference_count + 1) // 2)
    return int(requested_votes)


class LegacyLogitResidualRefiner(nn.Module):
    """Legacy semantic-logit residual model kept for archived experiments.

    Query tokens may read global scene context, but the returned residual can be
    gated to candidate points. This keeps contextual reasoning global while
    preventing an uncertain query from rewriting unrelated scene predictions.
    """

    def __init__(self, feat_dim, num_classes, hidden_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.query_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_to_scene = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.scene_to_query = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_classes)
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.region_mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._zero_output()

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self._zero_output()

    def _zero_output(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.point_mlp[-1].weight)
        nn.init.zeros_(self.point_mlp[-1].bias)
        nn.init.zeros_(self.region_mlp[-1].weight)
        nn.init.zeros_(self.region_mlp[-1].bias)

    def forward(
        self,
        point_feats,
        point_coords,
        batch_ids,
        query_indices,
        regions=None,
        use_region_branch=False,
        return_components=False,
        candidate_mask=None,
        pool_query_regions=False,
    ):
        point_coords = point_coords.float()
        if point_coords.size(1) > 3:
            point_coords = point_coords[:, :3]

        batch_ids = batch_ids.long().to(point_feats.device)
        if regions is not None:
            regions = regions.long().to(point_feats.device).view(-1)

        point_delta = self.point_mlp(point_feats)
        region_delta = point_delta.new_zeros(point_delta.shape)
        if use_region_branch and regions is not None:
            for batch_id in torch.unique(batch_ids):
                scene_mask = batch_ids == batch_id
                for region_id in torch.unique(regions[scene_mask]):
                    if int(region_id.item()) == -1:
                        continue
                    mask = scene_mask & (regions == region_id)
                    if mask.any():
                        region_feat = point_feats[mask].mean(dim=0, keepdim=True)
                        region_delta[mask] = self.region_mlp(region_feat)
        context_delta = point_delta.new_zeros(point_delta.shape)
        if query_indices.numel() == 0:
            if candidate_mask is not None:
                point_delta = gate_refiner_residual(point_delta, candidate_mask)
                region_delta = gate_refiner_residual(region_delta, candidate_mask)
            if return_components:
                return {
                    "point": point_delta,
                    "context": context_delta,
                    "region": region_delta,
                    "total": point_delta + region_delta,
                }
            return point_delta + region_delta

        for batch_id in torch.unique(batch_ids):
            scene_mask = batch_ids == batch_id
            scene_indices = torch.nonzero(scene_mask, as_tuple=False).flatten()
            scene_query_indices = query_indices[batch_ids[query_indices] == batch_id]
            if scene_query_indices.numel() == 0:
                continue

            scene_feats = point_feats[scene_indices]
            scene_coords = point_coords[scene_indices]
            if pool_query_regions and regions is not None:
                query_feats = []
                query_coords = []
                for query_index in scene_query_indices:
                    query_region = regions[query_index]
                    query_mask = scene_mask & (regions == query_region)
                    query_feats.append(point_feats[query_mask].mean(dim=0))
                    query_coords.append(point_coords[query_mask].mean(dim=0))
                query_feats = torch.stack(query_feats, dim=0)
                query_coords = torch.stack(query_coords, dim=0)
            else:
                query_feats = point_feats[scene_query_indices]
                query_coords = point_coords[scene_query_indices]

            coord_min = scene_coords.min(dim=0, keepdim=True)[0]
            coord_max = scene_coords.max(dim=0, keepdim=True)[0]
            coord_range = (coord_max - coord_min).clamp_min(1e-6)
            scene_pos = self.pos_proj((scene_coords - coord_min) / coord_range)
            query_pos = self.pos_proj((query_coords - coord_min) / coord_range)

            scene_tokens = self.feat_proj(scene_feats) + scene_pos
            query_tokens = self.query_proj(query_feats) + query_pos

            query_ctx, _ = self.query_to_scene(
                query_tokens.unsqueeze(0),
                scene_tokens.unsqueeze(0),
                scene_tokens.unsqueeze(0),
                need_weights=False,
            )
            query_tokens = query_tokens + query_ctx.squeeze(0)

            scene_ctx, _ = self.scene_to_query(
                scene_tokens.unsqueeze(0),
                query_tokens.unsqueeze(0),
                query_tokens.unsqueeze(0),
                need_weights=False,
            )
            scene_tokens = scene_tokens + scene_ctx.squeeze(0)
            scene_tokens = scene_tokens + self.ffn(scene_tokens)

            context_delta[scene_indices] = self.out_proj(self.out_norm(scene_tokens))

        if candidate_mask is not None:
            point_delta = gate_refiner_residual(point_delta, candidate_mask)
            context_delta = gate_refiner_residual(context_delta, candidate_mask)
            region_delta = gate_refiner_residual(region_delta, candidate_mask)
        total_delta = point_delta + context_delta + region_delta
        if return_components:
            return {
                "point": point_delta,
                "context": context_delta,
                "region": region_delta,
                "total": total_delta,
            }
        return total_delta


# Historical checkpoints only store parameter names, so these aliases keep old
# experiment commands loadable. The current Stage-2 method does not use them.
CandidateBasedRefiner = LegacyLogitResidualRefiner
ErrorQueryRefiner = LegacyLogitResidualRefiner


def refined_cross_entropy(refined_logits, pseudo_labels, trusted_mask, ignore_index=-1):
    if trusted_mask.sum() == 0:
        return refined_logits.sum() * 0.0
    targets = pseudo_labels.long().clone()
    targets[~trusted_mask] = ignore_index
    return F.cross_entropy(refined_logits, targets, ignore_index=ignore_index)


def refinement_keep_kl(refined_logits, base_logits, keep_mask):
    if keep_mask.sum() == 0:
        return refined_logits.sum() * 0.0
    return F.kl_div(
        F.log_softmax(refined_logits[keep_mask], dim=1),
        F.softmax(base_logits.detach()[keep_mask], dim=1),
        reduction="batchmean",
    )


def delta_l2(delta_logits, mask=None):
    if mask is not None:
        if mask.sum() == 0:
            return delta_logits.sum() * 0.0
        delta_logits = delta_logits[mask]
    return delta_logits.pow(2).mean()
