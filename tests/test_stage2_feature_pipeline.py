import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.stage2_feature_pipeline import (
    Stage2FeatureConfig,
    run_stage2_feature_pipeline,
    stage2_feature_losses,
)
from lib.utils import enforce_cannot_link
from models.feature_refiner import CandidateFeatureRefiner


def _mixed_region():
    left = torch.stack(
        [torch.linspace(-2, -1, 8), torch.zeros(8), torch.zeros(8)], dim=1
    )
    right = torch.stack(
        [torch.linspace(1, 2, 8), torch.zeros(8), torch.zeros(8)], dim=1
    )
    point_features = torch.cat(
        [torch.tensor([[1.0, 0.0]]).repeat(8, 1), torch.tensor([[0.0, 1.0]]).repeat(8, 1)]
    ).requires_grad_()
    semantic_logits = 4.0 * point_features
    return (
        point_features,
        torch.cat([left, right]),
        torch.zeros(16, 3),
        semantic_logits,
        torch.zeros(16, dtype=torch.long),
        torch.zeros(16, dtype=torch.long),
    )


def _config():
    return Stage2FeatureConfig(
        min_region_points=8,
        min_child_points=4,
        max_regions_per_scene=2,
        purity_threshold=0.9,
        entropy_threshold=0.2,
    )


def test_feature_refiner_updates_features_and_preserves_split_structure():
    inputs = _mixed_region()
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)
    output = run_stage2_feature_pipeline(_config(), refiner, *inputs)
    centers = torch.eye(2)
    losses = stage2_feature_losses(output, centers)
    (losses['semantic'] + 0.1 * losses['structure']).backward()

    assert output.stats['accepted_splits'] == 1
    assert torch.unique(output.dynamic_regions).numel() == 2
    assert output.supervision_mask.all()
    assert inputs[0].grad is not None
    assert refiner.out_proj.weight.grad is not None


class CollapsingFeatureRefiner(nn.Module):
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
        proposed = F.normalize(torch.ones_like(point_features), dim=1)
        return proposed, proposed - point_features


def test_feature_verifier_rolls_back_a_split_that_collapses_child_separation():
    output = run_stage2_feature_pipeline(
        _config(), CollapsingFeatureRefiner(), *_mixed_region()
    )

    assert output.stats['accepted_splits'] == 0
    assert output.stats['rejected_splits'] == 1
    assert output.rollback_mask.all()
    assert torch.unique(output.dynamic_regions).numel() == 1
    assert torch.allclose(output.refined_features, output.base_features)


def test_cannot_link_prevents_split_children_from_immediately_remerging():
    assignments = torch.tensor([0, 0, 1, 2])
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    )
    sizes = torch.tensor([8, 6, 10, 12])

    constrained, prevented = enforce_cannot_link(
        assignments,
        features,
        sizes,
        torch.tensor([[0, 1]]),
    )

    assert prevented == 1
    assert constrained[0] != constrained[1]
    assert torch.unique(constrained).numel() == torch.unique(assignments).numel()
