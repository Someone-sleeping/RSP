import torch

from models.learnable_superpoint import (
    SemanticDifferenceSuperpointLearner,
    verified_region_supervision_loss,
)


def _learner():
    torch.manual_seed(7)
    return SemanticDifferenceSuperpointLearner(
        feat_dim=8,
        num_classes=3,
        hidden_dim=16,
        iterations=2,
    )


def test_homogeneous_region_is_kept():
    learner = _learner()
    features = torch.randn(20, 8)
    coords = torch.randn(20, 3)
    colors = torch.randn(20, 3)
    logits = torch.full((20, 3), -4.0)
    logits[:, 0] = 4.0
    output = learner(features, coords, colors, logits, torch.zeros(20), torch.zeros(20))

    assert output.stats["selected_regions"] == 0
    assert output.dynamic_regions.unique().tolist() == [0]
    assert not output.supervision_mask.any()


def test_mixed_region_produces_verified_dynamic_superpoints():
    learner = _learner()
    features = torch.cat([torch.ones(12, 8), -torch.ones(12, 8)], dim=0)
    coords = torch.cat([torch.randn(12, 3) * 0.02 - 1, torch.randn(12, 3) * 0.02 + 1], dim=0)
    colors = torch.cat([torch.zeros(12, 3), torch.ones(12, 3)], dim=0)
    logits = torch.full((24, 3), -5.0)
    logits[:12, 0] = 5.0
    logits[12:, 1] = 5.0
    output = learner(
        features,
        coords,
        colors,
        logits,
        torch.zeros(24),
        torch.zeros(24),
        min_region_points=10,
        min_child_points=4,
        min_confidence_gain=0.0,
    )

    assert output.stats["accepted_splits"] == 1
    assert output.dynamic_regions.unique().numel() == 2
    assert output.supervision_mask.all()
    assert output.supervision_targets.unique().numel() == 2


def test_structure_and_supervision_losses_backpropagate():
    learner = _learner()
    features = torch.cat([torch.ones(10, 8), -torch.ones(10, 8)], dim=0).requires_grad_()
    coords = torch.cat([torch.randn(10, 3) - 1, torch.randn(10, 3) + 1], dim=0)
    colors = torch.cat([torch.zeros(10, 3), torch.ones(10, 3)], dim=0)
    logits = torch.full((20, 3), -3.0, requires_grad=True)
    with torch.no_grad():
        logits[:10, 0] = 3.0
        logits[10:, 1] = 3.0
    output = learner(
        features,
        coords,
        colors,
        logits,
        torch.zeros(20),
        torch.zeros(20),
        min_region_points=8,
        min_child_points=3,
        min_confidence_gain=0.0,
    )
    total = output.structure_loss + verified_region_supervision_loss(logits, output)
    total.backward()

    assert learner.affinity[-1].weight.grad is not None
    assert learner.affinity[-1].weight.grad.abs().sum() > 0
    assert features.grad is not None
    assert logits.grad is not None
