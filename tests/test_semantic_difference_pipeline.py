import torch

from models.query_refiner import gate_refiner_residual, resolve_min_temporal_votes
from lib.semantic_difference_pipeline import (
    DecompositionOutput,
    MetaRefinerOutput,
    SemanticDifferencePipelineConfig,
    verify_result,
)


def test_refiner_residual_is_zero_outside_candidate_support():
    residual = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    gated = gate_refiner_residual(residual, torch.tensor([False, True, False]))

    assert gated.tolist() == [[0.0, -0.0], [2.0, -2.0], [0.0, -0.0]]


def test_temporal_vote_resolution_preserves_conservative_explicit_threshold():
    assert resolve_min_temporal_votes(0, 1) == 1
    assert resolve_min_temporal_votes(0, 3) == 2
    assert resolve_min_temporal_votes(2, 1) == 2


def _scores(prediction, classes=3):
    scores = torch.full((len(prediction), classes), -4.0)
    scores.scatter_(1, torch.tensor(prediction)[:, None], 4.0)
    return scores


def test_result_verifier_accept_keep_rollback_and_temporal_override():
    config = SemanticDifferencePipelineConfig(
        semantic_classes=3,
        min_temporal_votes=2,
        meta_verify_confidence=0.60,
        result_verify_confidence=0.80,
    )
    base_prediction = torch.tensor([0, 0, 0, 1, 2])
    refined_prediction = torch.tensor([0, 1, 1, 1, 2])
    no_op_prediction = torch.tensor([0, 1, 1, 1, 2])
    region_prediction = torch.tensor([1, 1, 1, 1, 2])
    meta_prediction = torch.tensor([1, 2, 1, 1, 2])
    temporal_probability = torch.tensor(
        [
            [0.05, 0.90, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ]
    )
    temporal_votes = torch.tensor(
        [
            [0, 3, 0],
            [3, 0, 0],
            [0, 0, 3],
            [0, 3, 0],
            [0, 0, 3],
        ]
    )
    decomposition = DecompositionOutput(
        no_op_scores=_scores(no_op_prediction.tolist()),
        refined_scores=_scores(refined_prediction.tolist()),
        delta_scores=torch.zeros(5, 3),
        delta_components={},
        query_indices=torch.empty(0, dtype=torch.long),
        refine_mask=torch.tensor([False, False, True, False, False]),
        keep_mask=torch.ones(5, dtype=torch.bool),
        split_targets=torch.full((5,), -1, dtype=torch.long),
        statistics={},
    )
    meta_probability = torch.softmax(
        _scores(meta_prediction.tolist()),
        dim=1,
    )
    meta_refiner = MetaRefinerOutput(
        probability=meta_probability,
        scene_accepted=True,
        statistics={},
    )

    output = verify_result(
        config,
        base_prediction,
        region_prediction,
        temporal_probability,
        temporal_votes,
        decomposition,
        meta_refiner,
    )

    assert output.meta_accept_mask.tolist() == [True, False, False, False, False]
    assert output.rollback_mask.tolist() == [False, True, False, False, False]
    assert output.temporal_override_mask.tolist() == [False, False, True, False, False]
    assert output.keep_mask.tolist() == [False, False, False, True, True]
    assert output.meta_verified_prediction.tolist() == [1, 1, 1, 1, 2]
    assert output.final_prediction.tolist() == [1, 1, 2, 1, 2]
    assert output.decision.tolist() == [1, 2, 3, 0, 0]


def test_result_verifier_rolls_back_unsupported_refiner_change():
    config = SemanticDifferencePipelineConfig(
        semantic_classes=3,
        min_temporal_votes=2,
        meta_verify_confidence=0.60,
        result_verify_confidence=0.90,
    )
    base_prediction = torch.tensor([0, 0])
    no_op_prediction = torch.tensor([1, 1])
    refined_prediction = torch.tensor([2, 1])
    temporal_probability = torch.tensor(
        [[0.80, 0.15, 0.05], [0.05, 0.90, 0.05]]
    )
    temporal_votes = torch.tensor([[3, 0, 0], [0, 3, 0]])
    decomposition = DecompositionOutput(
        no_op_scores=_scores(no_op_prediction.tolist()),
        refined_scores=_scores(refined_prediction.tolist()),
        delta_scores=torch.zeros(2, 3),
        delta_components={},
        query_indices=torch.empty(0, dtype=torch.long),
        refine_mask=torch.ones(2, dtype=torch.bool),
        keep_mask=torch.zeros(2, dtype=torch.bool),
        split_targets=torch.full((2,), -1, dtype=torch.long),
        statistics={},
    )
    meta_refiner = MetaRefinerOutput(
        probability=torch.softmax(_scores(refined_prediction.tolist()), dim=1),
        scene_accepted=False,
        statistics={},
    )

    output = verify_result(
        config,
        base_prediction,
        no_op_prediction,
        temporal_probability,
        temporal_votes,
        decomposition,
        meta_refiner,
    )

    assert output.rollback_mask.tolist() == [True, False]
    assert output.meta_verified_prediction.tolist() == [1, 1]
    assert output.final_prediction.tolist() == [1, 1]
