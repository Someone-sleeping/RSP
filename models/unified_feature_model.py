import torch.nn as nn

from models.feature_refiner import CandidateFeatureRefiner


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
    """One trainable model for backbone extraction and direct feature refinement."""

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
        self.feature_refiner = CandidateFeatureRefiner(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, sparse_input):
        return self.backbone(sparse_input)

    def update_candidate_features(
        self,
        point_features,
        coordinates,
        batch_ids,
        query_indices,
        candidate_mask,
        regions=None,
        backbone_gradient_scale=0.1,
    ):
        return self.feature_refiner(
            point_features,
            coordinates,
            batch_ids,
            query_indices,
            candidate_mask,
            regions=regions,
            backbone_gradient_scale=backbone_gradient_scale,
        )

    def backbone_parameters(self):
        return self.backbone.parameters()

    def feature_refiner_parameters(self):
        return self.feature_refiner.parameters()

    def load_state_dict(self, state_dict, strict=True):
        unified = any(
            key.startswith("backbone.")
            or key.startswith("feature_refiner.")
            or key.startswith("feature_context.")
            for key in state_dict
        )
        if not unified:
            state_dict = {"backbone." + key: value for key, value in state_dict.items()}
        else:
            state_dict = {
                (
                    "feature_refiner." + key[len("feature_context."):]
                    if key.startswith("feature_context.")
                    else key
                ): value
                for key, value in state_dict.items()
            }

        incompatible = super().load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key for key in incompatible.missing_keys
            if key.startswith("feature_refiner.")
        }
        disallowed_missing = set(incompatible.missing_keys) - allowed_missing
        if strict and (disallowed_missing or incompatible.unexpected_keys):
            raise RuntimeError(
                "Unified model checkpoint mismatch: missing={}, unexpected={}".format(
                    sorted(disallowed_missing), sorted(incompatible.unexpected_keys)
                )
            )
        return incompatible
