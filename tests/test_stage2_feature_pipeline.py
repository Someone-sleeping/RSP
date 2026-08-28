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
from lib.my_utils import load_resume_checkpoint
from lib.utils import build_split_primitive_overrides, get_pseudo
from models.feature_refiner import CandidateFeatureContextBlock, CandidateFeatureRefiner
from models.unified_feature_model import UnifiedBackboneFeatureModel
from eval_S3DIS import grow_eval_regions, resolve_eval_growsp


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
    assert output.accept_mask.all()
    assert inputs[0].grad is not None
    assert inputs[0].grad.abs().sum() > 0
    assert refiner.out_proj.weight.grad is not None


def test_feature_refiner_proxy_losses_update_candidate_backbone_features():
    inputs = _mixed_region()
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)
    output = run_stage2_feature_pipeline(_config(), refiner, *inputs)
    losses = stage2_feature_losses(output, torch.eye(2))

    (losses['semantic'] + losses['structure'] + losses['residual']).backward()

    assert inputs[0].grad is not None
    assert inputs[0].grad.abs().sum() > 0
    assert refiner.point_mlp[-1].weight.grad is not None


def test_feature_refiner_reads_but_does_not_backpropagate_to_non_candidates():
    torch.manual_seed(0)
    features = torch.randn(6, 4, requires_grad=True)
    coordinates = torch.randn(6, 3)
    batch_ids = torch.zeros(6, dtype=torch.long)
    regions = torch.tensor([0, 0, 1, 1, 2, 2])
    candidate_mask = torch.tensor([True, True, False, False, False, False])
    refiner = CandidateFeatureRefiner(4, hidden_dim=8, num_heads=2)
    with torch.no_grad():
        refiner.out_proj.weight.fill_(0.1)
        refiner.point_mlp[-1].weight.fill_(0.1)

    proposed, _ = refiner.refine(
        features,
        coordinates,
        batch_ids,
        torch.tensor([0]),
        candidate_mask,
        regions=regions,
    )
    proposed[candidate_mask].sum().backward()

    assert features.grad[candidate_mask].abs().sum() > 0
    assert torch.equal(
        features.grad[~candidate_mask], torch.zeros_like(features.grad[~candidate_mask])
    )


def test_direct_feature_context_is_identity_initialized_and_trainable():
    torch.manual_seed(0)
    features = F.normalize(torch.randn(6, 4), dim=1).requires_grad_()
    coordinates = torch.randn(6, 3)
    batch_ids = torch.zeros(6, dtype=torch.long)
    regions = torch.tensor([0, 0, 1, 1, 2, 2])
    candidate_mask = torch.tensor([True, True, False, False, False, False])
    context = CandidateFeatureContextBlock(4, hidden_dim=8, num_heads=2)

    refined = context(
        features,
        coordinates,
        batch_ids,
        torch.tensor([0]),
        candidate_mask,
        regions=regions,
    )

    assert torch.allclose(refined, features, atol=1e-6)
    refined[candidate_mask, 0].sum().backward()
    assert context.context_to_feature.weight.grad is not None
    assert context.context_to_feature.weight.grad.abs().sum() > 0
    assert torch.equal(
        features.grad[~candidate_mask], torch.zeros_like(features.grad[~candidate_mask])
    )


def test_unified_model_loads_legacy_backbone_state_and_owns_context():
    legacy_backbone = nn.Linear(4, 4)
    legacy_state = legacy_backbone.state_dict()
    model = UnifiedBackboneFeatureModel(
        nn.Linear(4, 4), feat_dim=4, hidden_dim=8, num_heads=2
    )

    model.load_state_dict(legacy_state)

    assert torch.equal(model.backbone.weight, legacy_state['weight'])
    assert any(key.startswith('feature_context.') for key in model.state_dict())


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
        backbone_gradient_scale=0.1,
    ):
        proposed = F.normalize(torch.ones_like(point_features), dim=1)
        return proposed, proposed - point_features


class DirectionPreservingLargeRawResidual(nn.Module):
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
        residual = 2.0 * point_features
        proposed = F.normalize(
            point_features + float(residual_scale) * residual, dim=1
        )
        return proposed, residual


def test_feature_verifier_keeps_split_but_rolls_back_collapsing_residual():
    output = run_stage2_feature_pipeline(
        _config(), CollapsingFeatureRefiner(), *_mixed_region()
    )

    assert output.stats['accepted_splits'] == 1
    assert output.stats['refinement_accepted_splits'] == 0
    assert output.stats['refinement_rejected_splits'] == 1
    assert output.decomposition_mask.all()
    assert output.refinement_rollback_mask.all()
    assert not output.rollback_mask.any()
    assert torch.unique(output.dynamic_regions).numel() == 2
    assert torch.allclose(output.refined_features, output.base_features)


def test_feature_verifier_bounds_the_applied_not_raw_residual():
    config = _config()
    config.max_residual_norm = 0.5
    output = run_stage2_feature_pipeline(
        config, DirectionPreservingLargeRawResidual(), *_mixed_region()
    )

    assert output.stats['accepted_splits'] == 1
    assert output.stats['refinement_accepted_splits'] == 1
    assert output.stats['residual_norm_rejections'] == 0


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


def test_feature_verifier_uses_soft_topk_primitive_group_support():
    inputs = _mixed_region()
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)
    primitive_centers = torch.tensor(
        [[1.0, 0.0], [0.999, 0.04], [0.995, 0.10], [0.0, 1.0]]
    )
    primitive_to_semantic = torch.tensor([1, 0, 0, 1])

    output = run_stage2_feature_pipeline(
        _config(), refiner, *inputs, primitive_centers, primitive_to_semantic
    )

    assert output.stats['accepted_splits'] == 1
    assert output.stats['primitive_rejections'] == 0


def test_stage2_module_is_applied_after_grow():
    refiner = CandidateFeatureRefiner(2, hidden_dim=8, num_heads=2)
    module = Stage2FeatureModule(refiner, _config())

    assert module.apply_after_grow is True


def test_resume_checkpoint_restores_stage2_refiner_activation_state(tmp_path):
    model = nn.Linear(2, 2)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 1075,
        'is_Growing': True,
        'start_grow_epoch': 470,
        'training_stage': 'stage2_split_feature_refiner',
        'feature_refiner_state_dict': {},
    }
    path = tmp_path / 'resume.pth'
    torch.save(checkpoint, path)
    args = SimpleNamespace(resume=str(path))
    logger = SimpleNamespace(info=lambda *_: None)

    state = load_resume_checkpoint(args, model, None, None, logger)

    assert state == (1075, 470, True)
    assert args.resume_has_feature_refiner is True
    assert args.resume_training_stage == 'stage2_split_feature_refiner'


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


def test_named_unified_checkpoint_uses_final_growsp_size(tmp_path):
    args = SimpleNamespace(
        save_path=str(tmp_path),
        max_epoch=[500, 800],
        growsp_start=80,
        growsp_end=20,
    )

    assert resolve_eval_growsp(args, 'best') == 20


def test_eval_region_growing_drops_tiny_regions_and_compacts_ids():
    args = SimpleNamespace(
        drop_threshold=3,
        w_rgb=1.0,
        w_xyz=0.2,
        w_norm=0.8,
        voxel_size=0.05,
    )
    features = F.normalize(torch.randn(10, 4), dim=1)
    coordinates = torch.stack(
        [torch.arange(10).float(), torch.zeros(10), torch.zeros(10)], dim=1
    )
    colors = torch.zeros(10, 3)
    regions = torch.tensor([4, 4, 8, 8, 8, 8, 12, 12, 12, 12])

    grown = grow_eval_regions(
        args, features, coordinates, colors, regions, target=1
    )

    assert grown[:2].tolist() == [-1, -1]
    assert torch.unique(grown[2:]).tolist() == [0]


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
