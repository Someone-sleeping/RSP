import torch.nn as nn

from models.feature_refiner import CandidateFeatureContextBlock


def extract_backbone_state_dict(state_dict):
    """Return backbone weights from either legacy or unified checkpoints."""
    if any(key.startswith("backbone.") for key in state_dict):
        return {
            key[len("backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
    return state_dict


class UnifiedBackboneFeatureModel(nn.Module):
    """Backbone and candidate-conditioned feature context in one model."""

    def __init__(
        self,
        backbone,
        feat_dim,
        hidden_dim=128,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_context = CandidateFeatureContextBlock(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, sparse_input):
        return self.backbone(sparse_input)

    def refine_candidate_features(
        self,
        point_features,
        coordinates,
        batch_ids,
        query_indices,
        candidate_mask,
        regions=None,
        backbone_gradient_scale=0.1,
    ):
        refined = self.feature_context(
            point_features,
            coordinates,
            batch_ids,
            query_indices,
            candidate_mask,
            regions=regions,
            backbone_gradient_scale=backbone_gradient_scale,
        )
        return refined, refined - point_features

    def backbone_parameters(self):
        return self.backbone.parameters()

    def context_parameters(self):
        return self.feature_context.parameters()

    def load_state_dict(self, state_dict, strict=True):
        unified = any(
            key.startswith("backbone.") or key.startswith("feature_context.")
            for key in state_dict
        )
        if not unified:
            state_dict = {"backbone." + key: value for key, value in state_dict.items()}

        incompatible = super().load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key for key in incompatible.missing_keys
            if key.startswith("feature_context.")
        }
        disallowed_missing = set(incompatible.missing_keys) - allowed_missing
        if strict and (disallowed_missing or incompatible.unexpected_keys):
            raise RuntimeError(
                "Unified model checkpoint mismatch: missing={}, unexpected={}".format(
                    sorted(disallowed_missing), sorted(incompatible.unexpected_keys)
                )
            )
        return incompatible
