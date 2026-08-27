import torch.nn.functional as F

from models.query_refiner import CandidateBasedRefiner


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
