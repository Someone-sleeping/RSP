import torch
import torch.nn as nn
import torch.nn.functional as F

from models.query_refiner import CandidateBasedRefiner


class CandidateFeatureContextBlock(nn.Module):
    """Query-conditioned context layer that returns updated point features.

    The block reads the complete scene, but writes only candidate points. Its
    output is a feature representation rather than a semantic-logit residual.
    Identity initialization lets an existing backbone checkpoint adopt the
    layer without changing its initial predictions.
    """

    def __init__(self, feat_dim, hidden_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.query_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_to_scene = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.scene_to_query = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.context_to_feature = nn.Linear(hidden_dim, feat_dim)
        self.local_feature = nn.Linear(feat_dim, feat_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self._identity_initialize(feat_dim)

    def _identity_initialize(self, feat_dim):
        nn.init.zeros_(self.context_to_feature.weight)
        nn.init.zeros_(self.context_to_feature.bias)
        nn.init.eye_(self.local_feature.weight)
        nn.init.zeros_(self.local_feature.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    @staticmethod
    def _scaled_backbone_features(point_features, gradient_scale):
        return point_features.detach() + float(gradient_scale) * (
            point_features - point_features.detach()
        )

    def forward(
        self,
        point_features,
        coordinates,
        batch_ids,
        query_indices,
        candidate_mask,
        regions=None,
        backbone_gradient_scale=0.1,
    ):
        candidate_mask = candidate_mask.to(
            device=point_features.device, dtype=torch.bool
        ).view(-1)
        batch_ids = batch_ids.to(point_features.device).long().view(-1)
        query_indices = query_indices.to(point_features.device).long().view(-1)
        coordinates = coordinates.to(point_features.device).float()
        if coordinates.size(1) > 3:
            coordinates = coordinates[:, :3]
        if regions is not None:
            regions = regions.to(point_features.device).long().view(-1)

        base_features = self._scaled_backbone_features(
            point_features, backbone_gradient_scale
        )
        # Non-candidate tokens provide context without receiving gradients from
        # a local candidate objective.
        context_input = torch.where(
            candidate_mask[:, None], base_features, point_features.detach()
        )
        refined_features = base_features.clone()
        if query_indices.numel() == 0 or not candidate_mask.any():
            return F.normalize(refined_features, dim=1)

        for batch_id in torch.unique(batch_ids):
            scene_mask = batch_ids == batch_id
            scene_indices = torch.nonzero(scene_mask, as_tuple=False).flatten()
            scene_queries = query_indices[batch_ids[query_indices] == batch_id]
            if scene_queries.numel() == 0:
                continue

            scene_features = context_input[scene_indices]
            scene_coordinates = coordinates[scene_indices]
            query_features = []
            query_coordinates = []
            for query_index in scene_queries:
                if regions is None:
                    query_mask = torch.zeros_like(scene_mask)
                    query_mask[query_index] = True
                else:
                    query_mask = scene_mask & (regions == regions[query_index])
                query_features.append(context_input[query_mask].mean(dim=0))
                query_coordinates.append(coordinates[query_mask].mean(dim=0))
            query_features = torch.stack(query_features, dim=0)
            query_coordinates = torch.stack(query_coordinates, dim=0)

            coord_min = scene_coordinates.min(dim=0, keepdim=True).values
            coord_max = scene_coordinates.max(dim=0, keepdim=True).values
            coord_range = (coord_max - coord_min).clamp_min(1e-6)
            scene_tokens = self.feat_proj(scene_features) + self.pos_proj(
                (scene_coordinates - coord_min) / coord_range
            )
            query_tokens = self.query_proj(query_features) + self.pos_proj(
                (query_coordinates - coord_min) / coord_range
            )

            query_context, _ = self.query_to_scene(
                query_tokens[None], scene_tokens[None], scene_tokens[None],
                need_weights=False,
            )
            query_tokens = query_tokens + query_context.squeeze(0)
            scene_context, _ = self.scene_to_query(
                scene_tokens[None], query_tokens[None], query_tokens[None],
                need_weights=False,
            )
            scene_tokens = scene_tokens + scene_context.squeeze(0)
            scene_tokens = scene_tokens + self.ffn(scene_tokens)
            normalized_tokens = self.output_norm(scene_tokens)

            proposal = F.normalize(
                self.local_feature(base_features[scene_indices])
                + self.context_to_feature(normalized_tokens),
                dim=1,
            )
            blend = torch.sigmoid(self.gate(normalized_tokens))
            scene_refined = F.normalize(
                (1.0 - blend) * base_features[scene_indices] + blend * proposal,
                dim=1,
            )
            scene_candidates = candidate_mask[scene_indices]
            refined_features[scene_indices[scene_candidates]] = scene_refined[
                scene_candidates
            ]

        return F.normalize(refined_features, dim=1)


class CandidateFeatureRefiner(CandidateBasedRefiner):
    """Contextual residual adapter operating in backbone feature space."""

    def __init__(self, feat_dim, hidden_dim=128, num_heads=4, dropout=0.0):
        super().__init__(
            feat_dim=feat_dim,
            num_classes=feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def refine(
        self,
        point_features,
        coordinates,
        batch_ids,
        query_indices,
        candidate_mask,
        regions=None,
        residual_scale=0.1,
        backbone_gradient_scale=0.1,
    ):
        candidate_mask = candidate_mask.to(
            device=point_features.device, dtype=point_features.dtype
        ).view(-1, 1)
        # Candidate losses may update their backbone features. Scene context is
        # still readable, but detached non-candidate tokens prevent a local proxy
        # from rewriting unrelated backbone features through attention.
        scaled_features = (
            point_features.detach()
            + float(backbone_gradient_scale)
            * (point_features - point_features.detach())
        )
        refiner_input = (
            candidate_mask * scaled_features
            + (1.0 - candidate_mask) * point_features.detach()
        )
        residual = self(
            refiner_input,
            coordinates,
            batch_ids,
            query_indices,
            regions=regions,
            candidate_mask=candidate_mask.bool().view(-1),
            pool_query_regions=True,
        )
        refined = F.normalize(
            scaled_features + float(residual_scale) * residual,
            dim=1,
        )
        return refined, residual
