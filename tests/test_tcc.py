import torch
import torch.nn.functional as F
from types import SimpleNamespace

from lib.tcc import load_primitive_reliability, refine_primitive_centers


def test_tcc_moves_center_away_from_feature_and_color_outliers():
    clean = F.normalize(
        torch.tensor([[1.0, 0.05], [1.0, -0.05], [0.98, 0.02], [1.0, 0.0]]),
        dim=1,
    )
    outlier = F.normalize(torch.tensor([[-0.3, 1.0]]), dim=1)
    features = torch.cat((clean, outlier), dim=0)
    colors = torch.tensor(
        [[0.0, 0.0, 0.0]] * 4 + [[0.5, 0.5, 0.5]], dtype=torch.float32
    )
    labels = torch.zeros(features.size(0), dtype=torch.long)

    centers, weights, summary = refine_primitive_centers(
        features,
        labels,
        primitive_num=1,
        colors=colors,
        center_strength=1.0,
        color_weight=0.5,
        min_effective_ratio=0.0,
    )
    ordinary = F.normalize(features.mean(dim=0), dim=0)
    clean_center = F.normalize(clean.mean(dim=0), dim=0)

    assert torch.dot(centers[0], clean_center) > torch.dot(ordinary, clean_center)
    assert 0.5 <= weights[0] <= 1.0
    assert summary.refined_clusters == 1


def test_tcc_shrinks_center_move_when_effective_support_is_too_small():
    features = F.normalize(
        torch.tensor([[1.0, 0.0], [-1.0, 0.1], [-1.0, -0.1]]), dim=1
    )
    labels = torch.zeros(3, dtype=torch.long)
    mean_center = F.normalize(features.mean(dim=0), dim=0)

    centers, _, summary = refine_primitive_centers(
        features,
        labels,
        primitive_num=1,
        feature_temperature=0.01,
        center_strength=1.0,
        min_effective_ratio=0.9,
    )

    assert torch.allclose(centers[0], mean_center, atol=1e-5)
    assert summary.mean_effective_ratio < 0.9


def test_tcc_is_deterministic_and_keeps_all_weights_bounded():
    torch.manual_seed(4)
    features = F.normalize(torch.randn(12, 4), dim=1)
    colors = torch.rand(12, 3) - 0.5
    labels = torch.tensor([0] * 6 + [1] * 6)

    first = refine_primitive_centers(features, labels, 2, colors=colors)
    second = refine_primitive_centers(features, labels, 2, colors=colors)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.all(first[1] >= 0.5)
    assert torch.all(first[1] <= 1.0)


def test_tcc_clamps_invalid_loss_weight_bound():
    features = F.normalize(torch.randn(6, 4), dim=1)
    labels = torch.zeros(6, dtype=torch.long)

    _, weights, _ = refine_primitive_centers(
        features,
        labels,
        primitive_num=1,
        loss_weight_strength=1.0,
        min_loss_weight=2.0,
    )

    assert torch.equal(weights, torch.ones_like(weights))


def test_tcc_reports_missing_reliability_file(tmp_path):
    args = SimpleNamespace(
        tcc_enable=True,
        region_weight_enable=False,
        pseudo_label_path=str(tmp_path),
    )

    try:
        load_primitive_reliability(args, device='cpu')
    except FileNotFoundError as error:
        assert 'Run clustering before the training epoch' in str(error)
    else:
        raise AssertionError('missing reliability file should fail explicitly')
