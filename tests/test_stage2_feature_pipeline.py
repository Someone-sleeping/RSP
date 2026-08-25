from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.stage2_feature_pipeline import (
    Stage2FeatureConfig,
    Stage2FeatureModule,
    run_stage2_feature_pipeline,
    stage2_feature_losses,
)
from lib.utils import build_split_primitive_overrides, get_pseudo
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
    assert inputs[0].grad is None
    assert refiner.out_proj.weight.grad is not None


def test_feature_refiner_proxy_losses_do_not_update_backbone_features():
    inputs = _mixed_region()
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)
    output = run_stage2_feature_pipeline(_config(), refiner, *inputs)
    losses = stage2_feature_losses(output, torch.eye(2))

    (losses['semantic'] + losses['structure'] + losses['residual']).backward()

    assert inputs[0].grad is None
    assert refiner.point_mlp[-1].weight.grad is not None


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


def test_feature_verifier_rejects_semantic_split_without_feature_separation():
    point_features, coordinates, colors, semantic_logits, regions, batch_ids = _mixed_region()
    point_features = torch.tensor([[1.0, 0.0]]).repeat(16, 1).requires_grad_()
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)

    output = run_stage2_feature_pipeline(
        _config(), refiner, point_features, coordinates, colors,
        semantic_logits, regions, batch_ids,
    )

    assert output.stats['accepted_splits'] == 0
    assert output.stats['rejected_splits'] == 1
    assert output.rollback_mask.all()


def test_feature_verifier_rejects_children_inconsistent_with_global_primitives():
    inputs = _mixed_region()
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)
    primitive_centers = torch.eye(2)
    primitive_to_semantic = torch.tensor([0, 0])

    output = run_stage2_feature_pipeline(
        _config(), refiner, *inputs, primitive_centers, primitive_to_semantic
    )

    assert output.stats['accepted_splits'] == 0
    assert output.stats['primitive_rejections'] == 1
    assert output.rollback_mask.all()


def test_stage2_module_is_applied_after_grow():
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)
    module = Stage2FeatureModule(refiner, _config())

    assert module.apply_after_grow is True


def test_grown_regions_are_saved_for_the_training_round(tmp_path):
    args = SimpleNamespace(
        pseudo_label_path=str(tmp_path),
        stage2_split_refine_enable=True,
    )
    labels = torch.tensor([0, 1, 1, -1])
    initial_regions = torch.tensor([0, 0, 1, -1])
    grown_regions = torch.tensor([0, 0, 1])
    final_regions = [torch.tensor([0, 1, 2])]
    context = [('scene', labels, initial_regions, grown_regions)]

    get_pseudo(args, context, np.array([2, 3, 4]), final_regions)

    saved = np.load(tmp_path / 'scene_grown_region.npy')
    assert np.array_equal(saved, np.array([0, 0, 1, -1]))


def test_verified_children_override_labels_after_parent_primitive_clustering(tmp_path):
    args = SimpleNamespace(
        pseudo_label_path=str(tmp_path),
        stage2_split_refine_enable=True,
    )
    labels = torch.tensor([0, 0, 1, -1])
    initial_regions = torch.tensor([0, 0, 1, -1])
    grown_regions = torch.tensor([0, 0, 1])
    split_data = {
        'dynamic_regions': torch.tensor([0, 1, 2]),
        'refined_features': torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        'accept_mask': torch.tensor([False, True, True]),
    }
    context = [('scene', labels, initial_regions, grown_regions, split_data)]
    parent_regions = [torch.tensor([0, 0, 1])]
    primitive_centers = torch.eye(2)

    overrides, stats = build_split_primitive_overrides(
        context,
        primitive_centers,
        np.array([1, 0]),
        parent_regions,
    )
    get_pseudo(
        args,
        context,
        np.array([1, 0]),
        parent_regions,
        primitive_overrides=overrides,
    )

    saved = np.load(tmp_path / 'scene.npy')
    assert np.array_equal(saved, np.array([1, 0, 1, -1]))
    assert np.array_equal(np.load(tmp_path / 'scene_split_region.npy'), np.array([0, 1, 2, -1]))
    assert stats['children'] == 2
    assert stats['points'] == 2
