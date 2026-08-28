import torch
import torch.nn as nn
import torch.nn.functional as F

class CandidateFeatureRefiner(nn.Module):
    """Query-conditioned module that directly returns updated point features.

    The block reads the complete scene, but writes only candidate points. Its
    output is the feature representation consumed by GrowSP classification and
    superpoint aggregation; it has no semantic-logit output head.
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
        self.feature_projection = nn.Linear(feat_dim + hidden_dim, feat_dim)
        self._identity_initialize(feat_dim)

    def _identity_initialize(self, feat_dim):
        # Start from the backbone representation while leaving the contextual
        # columns trainable from the first optimization step.
        nn.init.zeros_(self.feature_projection.weight)
        nn.init.eye_(self.feature_projection.weight[:, :feat_dim])
        nn.init.zeros_(self.feature_projection.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate the first direct-feature checkpoint layout when possible."""
        projection_key = prefix + "feature_projection.weight"
        local_key = prefix + "local_feature.weight"
        context_key = prefix + "context_to_feature.weight"
        if projection_key not in state_dict and (
            local_key in state_dict or context_key in state_dict
        ):
            projection = self.feature_projection.weight.detach().clone()
            feat_dim = self.feature_projection.out_features
            if local_key in state_dict:
                projection[:, :feat_dim] = state_dict.pop(local_key)
            if context_key in state_dict:
                projection[:, feat_dim:] = state_dict.pop(context_key)
            state_dict[projection_key] = projection
            local_bias = state_dict.pop(prefix + "local_feature.bias", None)
            context_bias = state_dict.pop(prefix + "context_to_feature.bias", None)
            if prefix + "feature_projection.bias" not in state_dict:
                bias = self.feature_projection.bias.detach().clone()
                if local_bias is not None:
                    bias.add_(local_bias)
                if context_bias is not None:
                    bias.add_(context_bias)
                state_dict[prefix + "feature_projection.bias"] = bias
            state_dict.pop(prefix + "gate.weight", None)
            state_dict.pop(prefix + "gate.bias", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        updated_features = base_features.clone()
        if query_indices.numel() == 0 or not candidate_mask.any():
            return F.normalize(updated_features, dim=1)

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

            # The head predicts the candidate representation itself. It does
            # not emit class logits or an additive feature residual.
            scene_updated = F.normalize(
                self.feature_projection(
                    torch.cat(
                        (base_features[scene_indices], normalized_tokens), dim=1
                    )
                ),
                dim=1,
            )
            scene_candidates = candidate_mask[scene_indices]
            updated_features[scene_indices[scene_candidates]] = scene_updated[
                scene_candidates
            ]

        return F.normalize(updated_features, dim=1)


# Compatibility for source files archived with the first unified checkpoint.
# Both names refer to the same direct feature-output implementation.
CandidateFeatureContextBlock = CandidateFeatureRefiner
