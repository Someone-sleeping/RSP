import argparse
import json
import math
import os
import random

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
from eval_S3DIS import compute_unsupervised_metrics
from lib.meta_optimizer import _meta_objective, _poe_log_probability
from models.query_refiner import ErrorQueryRefiner
from tools_eval_error_verifier import (
    load_reference,
    parse_areas,
    region_candidates,
    run_split_refiner,
)


def parse_args():
    parser = argparse.ArgumentParser("Cross-scene label-free FOMAML for GrowSP correction")
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/magic/magic/cm/repositories/GrowSP/ckpt/S3DIS/1baseline/ckpts",
    )
    parser.add_argument("--base_epoch", type=int, default=1270)
    parser.add_argument("--reference_epochs", default="1170,1180,1190")
    parser.add_argument("--meta_train_areas", default="Area_1,Area_2,Area_3,Area_4,Area_6")
    parser.add_argument("--test_area", default="Area_5")
    parser.add_argument("--data_path", default="data/S3DIS/input")
    parser.add_argument("--sp_path", default="data/S3DIS/initial_superpoints/")
    parser.add_argument("--refiner_checkpoint", required=True)
    parser.add_argument("--refiner_scale", type=float, default=0.7)
    parser.add_argument("--initial_weight", type=float, default=0.08)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--inner_lr", type=float, default=0.1)
    parser.add_argument("--outer_epochs", type=int, default=3)
    parser.add_argument("--outer_lr", type=float, default=0.02)
    parser.add_argument("--correction_weight", type=float, default=1.0)
    parser.add_argument("--keep_weight", type=float, default=5.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--min_temporal_votes", type=int, default=2)
    parser.add_argument("--query_tolerance", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--output_checkpoint", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--primitive_num", type=int, default=300)
    parser.add_argument("--semantic_class", type=int, default=12)
    parser.add_argument("--feats_dim", type=int, default=128)
    parser.add_argument("--ignore_label", type=int, default=12)
    parser.add_argument("--bn_momentum", type=float, default=0.02)
    parser.add_argument("--conv1_kernel_size", type=int, default=5)
    parser.add_argument("--refiner_hidden_dim", type=int, default=128)
    parser.add_argument("--refiner_num_heads", type=int, default=4)
    return parser.parse_args()


def make_loader(args, areas):
    return DataLoader(
        S3DIStest(args, areas=areas),
        batch_size=1,
        collate_fn=cfl_collate_fn_test(),
        num_workers=args.workers,
        pin_memory=True,
    )


def infer_episode(args, models, refiner, batch, keep_evaluation_data=False):
    base_model, base_centers, temporal_references = models
    coords, features, inverse_map, labels, _index, region = batch
    in_field = ME.TensorField(features, coords, device=0)
    with torch.no_grad():
        base_feats = F.normalize(base_model(in_field), dim=1)
        base_scores = F.linear(base_feats, base_centers)
        temporal_probabilities = []
        for _epoch, model, centers in temporal_references:
            temporal_feats = F.normalize(model(in_field), dim=1)
            temporal_scores = F.linear(temporal_feats, centers)
            temporal_probabilities.append(F.softmax(temporal_scores * 10.0, dim=1))
        temporal_probabilities = torch.stack(temporal_probabilities, dim=0)
        regions = region.squeeze().long().cuda()
        region_prediction = region_candidates(base_scores, regions)
        no_op_scores, refined_scores, _delta, _split = run_split_refiner(
            args,
            refiner,
            base_scores,
            base_feats,
            coords,
            features,
            regions,
        )

    temporal_mean_probability = temporal_probabilities.mean(dim=0)
    temporal_predictions = temporal_probabilities.argmax(dim=2)
    temporal_vote_count = F.one_hot(
        temporal_predictions, num_classes=args.semantic_class
    ).sum(dim=0)
    episode = {
        "reference_probability": F.softmax(refined_scores, dim=1).half().cpu(),
        "temporal_probability": temporal_mean_probability.half().cpu(),
        "temporal_votes": temporal_vote_count.to(torch.uint8).cpu(),
        "region_prediction": region_prediction.to(torch.uint8).cpu(),
        "no_op_prediction": no_op_scores.argmax(dim=1).to(torch.uint8).cpu(),
        "base_prediction": base_scores.argmax(dim=1).to(torch.uint8).cpu(),
        "regions": regions.to(torch.int32).cpu(),
    }
    if keep_evaluation_data:
        episode["inverse_map"] = inverse_map.to(torch.int32)
        episode["labels"] = labels.to(torch.int16)
    return episode


def move_episode(episode):
    return {
        name: value.cuda().float() if "probability" in name else value.cuda().long()
        for name, value in episode.items()
        if name not in ("inverse_map", "labels")
    }


def episode_masks(episode, confidence_threshold, min_votes):
    temporal_confidence, temporal_prediction = episode["temporal_probability"].max(dim=1)
    reference_prediction = episode["reference_probability"].argmax(dim=1)
    correction_mask = (
        (episode["temporal_votes"].max(dim=1)[0] >= int(min_votes))
        & (temporal_confidence >= float(confidence_threshold))
        & (temporal_prediction != reference_prediction)
        & (
            (temporal_prediction == episode["region_prediction"])
            | (temporal_prediction == episode["no_op_prediction"])
        )
    )
    keep_mask = (
        (reference_prediction == temporal_prediction)
        | (reference_prediction == episode["region_prediction"])
        | (reference_prediction == episode["no_op_prediction"])
        | (reference_prediction == episode["base_prediction"])
    ) & ~correction_mask
    valid_regions = episode["regions"].clamp_min(0)
    support_mask = (valid_regions % 2) == 0
    support_mask[episode["regions"] < 0] = (
        torch.arange(support_mask.numel(), device=support_mask.device)[episode["regions"] < 0] % 2
    ) == 0
    return temporal_prediction, temporal_confidence, correction_mask, keep_mask, support_mask, ~support_mask


def objective(args, episode, blend_logit, subset_mask, masks):
    temporal_prediction, temporal_confidence, correction_mask, keep_mask, _support, _query = masks
    blend_weight = torch.sigmoid(blend_logit)
    log_probability = _poe_log_probability(
        episode["reference_probability"], episode["temporal_probability"], blend_weight
    )
    loss = _meta_objective(
        log_probability,
        episode["reference_probability"],
        episode["reference_probability"].argmax(dim=1),
        temporal_prediction,
        temporal_confidence,
        correction_mask,
        keep_mask,
        subset_mask,
        args.correction_weight,
        args.keep_weight,
        args.entropy_weight,
    )
    return loss, log_probability


def adapt_task(args, episode, initial_logit):
    masks = episode_masks(
        episode, args.confidence_threshold, args.min_temporal_votes
    )
    support_mask, query_mask = masks[-2:]
    adapted_logit = initial_logit.detach().clone().requires_grad_(True)
    for _ in range(max(args.inner_steps, 1)):
        support_loss, _ = objective(args, episode, adapted_logit, support_mask, masks)
        gradient = torch.autograd.grad(support_loss, adapted_logit)[0]
        adapted_logit = (
            adapted_logit - args.inner_lr * gradient
        ).detach().clamp(-8.0, 4.0).requires_grad_(True)
    query_loss, adapted_log_probability = objective(
        args, episode, adapted_logit, query_mask, masks
    )
    initial_query_loss, initial_log_probability = objective(
        args, episode, initial_logit, query_mask, masks
    )
    gain = float((initial_query_loss - query_loss).detach().item())
    accepted = gain >= args.query_tolerance
    selected = adapted_log_probability if accepted else initial_log_probability
    return adapted_logit, query_loss, selected, accepted, gain, masks


def metrics(predictions, labels, semantic_class):
    prediction_np = torch.cat(predictions).numpy()
    label_np = torch.cat(labels).numpy()
    values = compute_unsupervised_metrics(prediction_np, label_np, semantic_class)[:4]
    return {"oAcc": float(values[0]), "mAcc": float(values[1]), "mIoU": float(values[3])}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    reference_epochs = [int(value) for value in args.reference_epochs.split(",") if value]
    base_model, base_centers = load_reference(args, args.base_epoch)
    temporal_references = []
    for epoch in reference_epochs:
        model, centers = load_reference(args, epoch, reference_centers=base_centers)
        temporal_references.append((epoch, model, centers))
    models = (base_model, base_centers, temporal_references)

    refiner = ErrorQueryRefiner(
        feat_dim=args.feats_dim,
        num_classes=args.semantic_class,
        hidden_dim=args.refiner_hidden_dim,
        num_heads=args.refiner_num_heads,
        dropout=0.0,
    ).cuda()
    refiner.load_state_dict(torch.load(args.refiner_checkpoint, map_location="cpu"), strict=False)
    refiner.eval()

    train_episodes = []
    train_loader = make_loader(args, parse_areas(args.meta_train_areas))
    for batch in train_loader:
        train_episodes.append(infer_episode(args, models, refiner, batch))
    clipped_initial = min(max(args.initial_weight, 1e-4), 1.0 - 1e-4)
    initial_logit = torch.tensor(
        math.log(clipped_initial / (1.0 - clipped_initial)), device="cuda"
    )
    outer_history = []
    for outer_epoch in range(args.outer_epochs):
        order = list(range(len(train_episodes)))
        random.Random(args.seed + outer_epoch).shuffle(order)
        losses = []
        accepted = 0
        for episode_index in order:
            episode = move_episode(train_episodes[episode_index])
            adapted_logit, query_loss, _selected, task_accepted, _gain, _masks = adapt_task(
                args, episode, initial_logit
            )
            query_gradient = torch.autograd.grad(query_loss, adapted_logit)[0]
            initial_logit = (
                initial_logit - args.outer_lr * query_gradient
            ).detach().clamp(-8.0, 4.0)
            losses.append(float(query_loss.detach().item()))
            accepted += int(task_accepted)
        summary = {
            "epoch": outer_epoch + 1,
            "query_loss": float(np.mean(losses)),
            "initial_weight": float(torch.sigmoid(initial_logit).item()),
            "task_accept_ratio": accepted / max(len(order), 1),
        }
        outer_history.append(summary)
        print("outer", json.dumps(summary, sort_keys=True))

    del train_episodes
    prediction_names = ["base", "split_refiner", "meta_initial", "meta_adapt", "meta_override"]
    predictions = {name: [] for name in prediction_names}
    all_labels = []
    test_stats = {"scenes": 0, "accepted": 0, "query_gain": 0.0}
    test_loader = make_loader(args, parse_areas(args.test_area))
    for batch in test_loader:
        stored = infer_episode(args, models, refiner, batch, keep_evaluation_data=True)
        episode = move_episode(stored)
        adapted_logit, _query_loss, selected_log_probability, accepted, gain, masks = adapt_task(
            args, episode, initial_logit
        )
        initial_probability = _poe_log_probability(
            episode["reference_probability"],
            episode["temporal_probability"],
            torch.sigmoid(initial_logit),
        ).exp()
        meta_prediction = selected_log_probability.argmax(dim=1)
        temporal_prediction = masks[0]
        temporal_confidence = masks[1]
        point_accept = (
            (episode["temporal_votes"].max(dim=1)[0] >= args.min_temporal_votes)
            & (temporal_confidence >= args.confidence_threshold)
            & (temporal_prediction != meta_prediction)
        )
        override_prediction = meta_prediction.clone()
        override_prediction[point_accept] = temporal_prediction[point_accept]

        inverse = stored["inverse_map"].long().cuda()
        labels = stored["labels"]
        valid = labels != args.ignore_label
        scene_predictions = {
            "base": episode["base_prediction"],
            "split_refiner": episode["reference_probability"].argmax(dim=1),
            "meta_initial": initial_probability.argmax(dim=1),
            "meta_adapt": meta_prediction,
            "meta_override": override_prediction,
        }
        for name, prediction in scene_predictions.items():
            predictions[name].append(prediction[inverse].cpu()[valid])
        all_labels.append(labels[valid])
        test_stats["scenes"] += 1
        test_stats["accepted"] += int(accepted)
        test_stats["query_gain"] += gain

    results = {
        name: metrics(value, all_labels, args.semantic_class)
        for name, value in predictions.items()
    }
    base_miou = results["base"]["mIoU"]
    for result in results.values():
        result["delta_mIoU"] = result["mIoU"] - base_miou
    test_stats["accept_ratio"] = test_stats["accepted"] / max(test_stats["scenes"], 1)
    test_stats["mean_query_gain"] = test_stats["query_gain"] / max(test_stats["scenes"], 1)
    output = {
        "config": vars(args),
        "meta_train_areas": parse_areas(args.meta_train_areas),
        "test_area": args.test_area,
        "reference_epochs": reference_epochs,
        "learned_initial_weight": float(torch.sigmoid(initial_logit).item()),
        "outer_history": outer_history,
        "test_meta_stats": test_stats,
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output_checkpoint:
        output_dir = os.path.dirname(args.output_checkpoint)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save({"initial_logit": initial_logit.cpu(), "config": vars(args)}, args.output_checkpoint)
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2)


if __name__ == "__main__":
    main()
