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
    ):
        residual = self(
            point_features,
            coordinates,
            batch_ids,
            query_indices,
            regions=regions,
            candidate_mask=candidate_mask,
        )
        refined = F.normalize(
            point_features + float(residual_scale) * residual,
            dim=1,
        )
        return refined, residual
