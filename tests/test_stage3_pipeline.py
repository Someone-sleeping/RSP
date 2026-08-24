import torch
import torch.nn as nn

from lib.stage3_pipeline import (
    CandidateRefinementOutput,
    Stage3Config,
    SuperpointSplitOutput,
    run_stage3_pipeline,
    stage3_training_losses,
    verify_candidates,
)
from models.query_refiner import CandidateBasedRefiner, ErrorQueryRefiner


class TargetSupportingRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        point_features,
        coordinates,
        batch_ids,
        query_indices,
        regions=None,
        candidate_mask=None,
    ):
        residual = point_features[:, :2] * self.scale
        return residual * candidate_mask[:, None].to(residual.dtype)


def _mixed_superpoint_inputs():
    coordinates = torch.cat(
        [
            torch.stack([torch.linspace(-2, -1, 8), torch.zeros(8), torch.zeros(8)], dim=1),
            torch.stack([torch.linspace(1, 2, 8), torch.zeros(8), torch.zeros(8)], dim=1),
        ]
    )
    semantic_logits = torch.cat(
        [torch.tensor([[4.0, -2.0]]).repeat(8, 1), torch.tensor([[-2.0, 4.0]]).repeat(8, 1)]
    ).requires_grad_()
    point_features = torch.softmax(semantic_logits.detach(), dim=1)
    colors = torch.zeros(16, 3)
    regions = torch.zeros(16, dtype=torch.long)
    batch_ids = torch.zeros(16, dtype=torch.long)
    return point_features, coordinates, colors, semantic_logits, regions, batch_ids


def test_stage3_splits_region_and_joint_loss_updates_both_network_parts():
    inputs = _mixed_superpoint_inputs()
    refiner = TargetSupportingRefiner()
    config = Stage3Config(
        min_region_points=8,
        min_child_points=4,
        max_regions_per_scene=2,
        purity_threshold=0.9,
        entropy_threshold=0.2,
    )

    output = run_stage3_pipeline(config, refiner, *inputs)
    losses = stage3_training_losses(config, output)
    (losses["backbone"] + losses["refiner"]).backward()

    assert output.split.statistics["split_regions"] == 1
    assert torch.unique(output.split.dynamic_regions).numel() == 2
    assert output.split.candidate_mask.all()
    assert output.verification.rollback_mask.sum().item() == 0
    assert inputs[3].grad is not None
    assert refiner.scale.grad is not None


def test_conservative_verifier_rolls_back_unsupported_candidate_change():
    split = SuperpointSplitOutput(
        query_indices=torch.tensor([0]),
        candidate_mask=torch.tensor([True, True, False]),
        keep_mask=torch.tensor([False, False, True]),
        targets=torch.tensor([0, 1, -1]),
        target_confidence=torch.tensor([0.8, 0.8, 0.0]),
        dynamic_regions=torch.tensor([0, 1, 2]),
        statistics={},
    )
    base = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    residual = torch.tensor([[-5.0, 5.0], [-1.0, 1.0], [0.0, 0.0]])
    refinement = CandidateRefinementOutput(base, base + residual, residual)

    output = verify_candidates(Stage3Config(max_residual_norm=10.0), split, refinement)

    assert output.accept_mask.tolist() == [False, True, False]
    assert output.rollback_mask.tolist() == [True, False, False]
    assert output.supervision_targets.tolist() == [-1, 1, -1]
    assert torch.equal(output.verified_logits[0], base[0])
    assert torch.equal(output.verified_logits[2], base[2])


def test_candidate_refiner_rename_preserves_legacy_checkpoint_layout():
    assert ErrorQueryRefiner is CandidateBasedRefiner
    legacy = ErrorQueryRefiner(4, 3, hidden_dim=8, num_heads=2)
    renamed = CandidateBasedRefiner(4, 3, hidden_dim=8, num_heads=2)
    renamed.load_state_dict(legacy.state_dict())
    assert list(legacy.state_dict()) == list(renamed.state_dict())


def test_candidate_refiner_empty_query_cannot_modify_non_candidates():
    refiner = CandidateBasedRefiner(4, 3, hidden_dim=8, num_heads=2)
    with torch.no_grad():
        refiner.point_mlp[-1].bias.fill_(1.0)
    output = refiner(
        torch.randn(5, 4),
        torch.randn(5, 3),
        torch.zeros(5, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        candidate_mask=torch.zeros(5, dtype=torch.bool),
    )
    assert torch.count_nonzero(output).item() == 0
