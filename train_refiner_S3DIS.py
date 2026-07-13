import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStrain, cfl_collate_fn
from eval_S3DIS import eval as eval_s3dis
from lib.error_query import build_error_queries
from lib.my_utils import build_model
from lib.split_regions import build_region_consistency_queries, build_split_region_queries, build_uncertain_region_queries
from lib.utils import get_pseudo, get_sp_feature
from models.query_refiner import ErrorQueryRefiner, delta_l2, refinement_keep_kl


def parse_args():
    parser = argparse.ArgumentParser("Train only the S3DIS semantic refiner with a frozen GrowSP teacher")
    parser.add_argument("--data_path", type=str, default="data/S3DIS/input")
    parser.add_argument("--sp_path", type=str, default="data/S3DIS/initial_superpoints/")
    parser.add_argument("--save_path", type=str, default="ckpt/S3DIS/refiner_only/")
    parser.add_argument("--pseudo_label_path", type=str, default="")
    parser.add_argument("--teacher_ckpt_dir", type=str, required=True)
    parser.add_argument("--teacher_epoch", type=int, default=-1)
    parser.add_argument("--test_area", type=str, default="Area_5", help="S3DIS held-out area, or comma-separated areas")
    parser.add_argument("--teacher_growsp", type=int, default=20)
    parser.add_argument("--target_teacher_ckpt_dir", type=str, default="")
    parser.add_argument("--target_teacher_epoch", type=int, default=-1)
    parser.add_argument("--rebuild_pseudo", action="store_true", default=False)

    parser.add_argument("--model", type=str, default="res16fpn18")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--primitive_num", type=int, default=300)
    parser.add_argument("--semantic_class", type=int, default=12)
    parser.add_argument("--feats_dim", type=int, default=128)
    parser.add_argument("--ignore_label", type=int, default=12)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--bn_momentum", type=float, default=0.02)
    parser.add_argument("--conv1_kernel_size", type=int, default=5)
    parser.add_argument("--drop_threshold", type=int, default=10)
    parser.add_argument("--w_rgb", type=float, default=1.0)
    parser.add_argument("--w_xyz", type=float, default=0.2)
    parser.add_argument("--w_norm", type=float, default=0.8)
    parser.add_argument("--c_rgb", type=float, default=3.0)
    parser.add_argument("--c_shape", type=float, default=3.0)
    parser.add_argument("--z_enable", action="store_true", default=False)

    parser.add_argument("--max_epoch", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cluster_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)

    parser.add_argument("--refine_lr", type=float, default=1.5e-4)
    parser.add_argument("--refine_weight_decay", type=float, default=1e-4)
    parser.add_argument("--refine_hidden_dim", type=int, default=128)
    parser.add_argument("--refine_num_heads", type=int, default=4)
    parser.add_argument("--refine_dropout", type=float, default=0.0)
    parser.add_argument("--refine_residual_scale", type=float, default=0.7)
    parser.add_argument("--refine_rounds", type=int, default=1)
    parser.add_argument("--refine_query_scale", type=float, default=10.0)
    parser.add_argument("--refine_apply_all", action="store_true", default=True)
    parser.add_argument("--no_refine_apply_all", dest="refine_apply_all", action="store_false")
    parser.add_argument("--refine_region_project", action="store_true", default=True)
    parser.add_argument("--no_refine_region_project", dest="refine_region_project", action="store_false")
    parser.add_argument("--refine_region_project_mode", type=str, default="confprob", choices=["logit", "prob", "confprob", "feat", "vote"])
    parser.add_argument("--refine_region_branch", action="store_true", default=False)
    parser.add_argument("--refine_split_project", action="store_true", default=True)
    parser.add_argument("--no_refine_split_project", dest="refine_split_project", action="store_false")
    parser.add_argument("--refine_split_project_logit", type=float, default=20.0)
    parser.add_argument("--refine_point_accept_gate", action="store_true", default=False)
    parser.add_argument("--refine_point_accept_conf_gain", type=float, default=0.05)
    parser.add_argument("--refine_point_accept_min_conf", type=float, default=0.50)
    parser.add_argument("--refine_point_accept_min_margin", type=float, default=0.10)
    parser.add_argument("--refine_point_accept_entropy_gain", type=float, default=0.0)
    parser.add_argument("--refine_region_accept_gate", action="store_true", default=False)
    parser.add_argument("--refine_region_accept_conf_gain", type=float, default=0.02)
    parser.add_argument("--refine_region_accept_entropy_gain", type=float, default=0.0)
    parser.add_argument("--refine_conf_th", type=float, default=0.55)
    parser.add_argument("--refine_margin_th", type=float, default=0.05)
    parser.add_argument("--refine_region_purity_th", type=float, default=0.75)
    parser.add_argument("--refine_color_consistency_th", type=float, default=0.2)
    parser.add_argument("--refine_geometry_consistency_th", type=float, default=0.55)
    parser.add_argument("--refine_min_region_points", type=int, default=10)
    parser.add_argument("--refine_max_queries", type=int, default=20)
    parser.add_argument("--refine_min_ratio", type=float, default=0.02)
    parser.add_argument("--consensus_conf_th", type=float, default=0.15)
    parser.add_argument("--consensus_temp", type=float, default=0.7)
    parser.add_argument("--refine_ce_lambda", type=float, default=1.0)
    parser.add_argument("--refine_project_ce_lambda", type=float, default=0.2)
    parser.add_argument("--refine_accept_lambda", type=float, default=0.0)
    parser.add_argument("--refine_accept_gain", type=float, default=0.02)
    parser.add_argument("--refine_noop_kl_lambda", type=float, default=0.0)
    parser.add_argument("--refine_noop_kl_conf", type=float, default=0.75)
    parser.add_argument("--refine_pseudo_lambda", type=float, default=1.0)
    parser.add_argument("--refine_pseudo_conf", type=float, default=0.55)
    parser.add_argument(
        "--refine_optimize_mode",
        type=str,
        default="flip",
        choices=["flip", "trusted", "split", "split_or_multi"],
        help="which self-supervised targets are allowed to train residual updates",
    )
    parser.add_argument("--refine_optimize_min_conf", type=float, default=0.0)
    parser.add_argument("--refine_optimize_split_min_conf", type=float, default=0.0)
    parser.add_argument("--refine_optimize_min_support", type=int, default=2)
    parser.add_argument("--refine_margin_lambda", type=float, default=1.0)
    parser.add_argument("--refine_target_margin", type=float, default=0.3)
    parser.add_argument("--refine_keep_lambda", type=float, default=1.0)
    parser.add_argument("--refine_delta_lambda", type=float, default=0.005)
    parser.add_argument("--refine_entropy_lambda", type=float, default=0.02)
    parser.add_argument("--split_target_weight", type=float, default=1.0)
    parser.add_argument("--consistency_target_weight", type=float, default=0.1)
    parser.add_argument("--consistency_train_conf", type=float, default=0.75)
    parser.add_argument("--region_target_weight", type=float, default=0.25)
    parser.add_argument("--temporal_target_weight", type=float, default=0.5)
    parser.add_argument("--temporal_target_conf", type=float, default=0.70)
    parser.add_argument("--temporal_logit_scale", type=float, default=10.0)
    parser.add_argument("--temporal_align_mode", type=str, default="center", choices=["center", "batch"])
    parser.add_argument("--refine_split_enable", action="store_true", default=True)
    parser.add_argument("--no_refine_split_enable", dest="refine_split_enable", action="store_false")
    parser.add_argument("--refine_consistency_enable", action="store_true", default=True)
    parser.add_argument("--no_refine_consistency_enable", dest="refine_consistency_enable", action="store_false")
    parser.add_argument("--consistency_min_region_points", type=int, default=20)
    parser.add_argument("--consistency_max_regions", type=int, default=40)
    parser.add_argument("--consistency_min_conf", type=float, default=0.35)
    parser.add_argument("--consistency_min_disagree", type=float, default=0.02)
    parser.add_argument("--consistency_point_conf", type=float, default=0.55)
    parser.add_argument("--consistency_point_entropy", type=float, default=0.55)
    parser.add_argument("--consistency_logit_scale", type=float, default=10.0)
    parser.add_argument("--split_logit_scale", type=float, default=1.0)
    parser.add_argument("--split_min_region_points", type=int, default=30)
    parser.add_argument("--split_min_child_points", type=int, default=8)
    parser.add_argument("--split_max_regions", type=int, default=80)
    parser.add_argument("--split_purity_th", type=float, default=0.92)
    parser.add_argument("--split_entropy_th", type=float, default=0.25)
    parser.add_argument("--split_min_conf", type=float, default=0.15)
    parser.add_argument("--split_xyz_weight", type=float, default=1.0)
    parser.add_argument("--split_rgb_weight", type=float, default=0.5)
    parser.add_argument("--split_feat_weight", type=float, default=0.25)
    parser.add_argument("--split_semantic_weight", type=float, default=1.0)
    parser.add_argument("--split_multi_proposal", action="store_true", default=False)
    parser.add_argument("--split_selection_mode", type=str, default="score", choices=["score", "random"])
    parser.add_argument("--split_random_seed", type=int, default=0)
    return parser.parse_args()


def parse_test_areas(test_area):
    areas = [area.strip() for area in str(test_area).split(",") if area.strip()]
    if not areas:
        raise ValueError("test_area must contain at least one S3DIS area")
    return areas


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(save_path):
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger("train_refiner")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    file_handler = logging.FileHandler(os.path.join(save_path, "train_refiner.log"))
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_teacher_epoch(ckpt_dir, requested_epoch):
    if requested_epoch >= 0:
        return requested_epoch
    pattern = re.compile(r"^(model|cls)_(\d+)_checkpoint\.pth$")
    model_epochs, cls_epochs = set(), set()
    for name in os.listdir(ckpt_dir):
        match = pattern.match(name)
        if match is None:
            continue
        (model_epochs if match.group(1) == "model" else cls_epochs).add(int(match.group(2)))
    shared = sorted(model_epochs & cls_epochs)
    if not shared:
        raise FileNotFoundError(f"No paired model/cls checkpoints in {ckpt_dir}")
    return shared[-1]


def load_teacher_checkpoint(args, logger, ckpt_dir, requested_epoch, name="teacher"):
    teacher_epoch = resolve_teacher_epoch(ckpt_dir, requested_epoch)
    model_path = os.path.join(ckpt_dir, f"model_{teacher_epoch}_checkpoint.pth")
    cls_path = os.path.join(ckpt_dir, f"cls_{teacher_epoch}_checkpoint.pth")
    if not os.path.exists(model_path) or not os.path.exists(cls_path):
        raise FileNotFoundError(f"Missing teacher checkpoint pair for epoch {teacher_epoch}")

    model = build_model(
        args.model,
        in_channels=args.input_dim,
        out_channels=args.primitive_num,
        conv1_kernel_size=args.conv1_kernel_size,
        config=args,
    ).cuda()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    classifier = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    classifier.load_state_dict(torch.load(cls_path, map_location="cpu"))
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)

    logger.info(f"Loaded frozen {name} model: {model_path}")
    logger.info(f"Loaded frozen {name} classifier: {cls_path}")
    return model, classifier, teacher_epoch


def load_teacher(args, logger):
    return load_teacher_checkpoint(args, logger, args.teacher_ckpt_dir, args.teacher_epoch, "teacher")


def build_semantic_centers(primitive_classifier, semantic_class):
    primitive_centers = primitive_classifier.weight.detach()
    cluster_pred = KMeans(
        n_clusters=semantic_class,
        n_init=10,
        random_state=0,
        n_jobs=10,
    ).fit_predict(primitive_centers.cpu().numpy())

    semantic_centers = primitive_centers.new_zeros((semantic_class, primitive_centers.size(1)))
    for semantic_id in range(semantic_class):
        mask = torch.as_tensor(cluster_pred == semantic_id, device=primitive_centers.device)
        if mask.any():
            semantic_centers[semantic_id] = primitive_centers[mask].mean(dim=0)
    return F.normalize(semantic_centers, dim=1), cluster_pred


def align_semantic_centers(source_centers, reference_centers):
    source_centers = F.normalize(source_centers, dim=1)
    reference_centers = F.normalize(reference_centers, dim=1)
    sim = torch.mm(source_centers, reference_centers.t()).detach().cpu().numpy()
    match = linear_assignment(sim.max() - sim)
    aligned = source_centers.new_zeros(source_centers.shape)
    for source_id, reference_id in match:
        aligned[int(reference_id)] = source_centers[int(source_id)]
    return F.normalize(aligned, dim=1)


def align_logits_by_batch_predictions(source_logits, reference_logits):
    source_pred = source_logits.detach().argmax(dim=1).cpu().numpy()
    reference_pred = reference_logits.detach().argmax(dim=1).cpu().numpy()
    num_classes = reference_logits.size(1)
    histogram = np.bincount(
        num_classes * reference_pred + source_pred,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    match = linear_assignment(histogram.max() - histogram)
    aligned = source_logits.new_full(source_logits.shape, float("nan"))
    used_source = set()
    for reference_id, source_id in match:
        aligned[:, int(reference_id)] = source_logits[:, int(source_id)]
        used_source.add(int(source_id))
    unused_source = [idx for idx in range(num_classes) if idx not in used_source]
    for reference_id in range(num_classes):
        if not torch.isfinite(aligned[:, reference_id]).all():
            source_id = unused_source.pop(0) if unused_source else reference_id
            aligned[:, reference_id] = source_logits[:, source_id]
    return aligned


def pseudo_marker_path(args, teacher_epoch):
    return os.path.join(args.pseudo_label_path, f".teacher_{teacher_epoch}_growsp_{args.teacher_growsp}.json")


def checkpoint_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_pseudo_signature(args, teacher_epoch):
    ckpt_dir = os.path.abspath(args.teacher_ckpt_dir)
    model_path = os.path.join(ckpt_dir, f"model_{teacher_epoch}_checkpoint.pth")
    cls_path = os.path.join(ckpt_dir, f"cls_{teacher_epoch}_checkpoint.pth")
    if not os.path.exists(model_path) or not os.path.exists(cls_path):
        raise FileNotFoundError(f"Missing teacher checkpoint pair for epoch {teacher_epoch}")
    return {
        "teacher_epoch": int(teacher_epoch),
        "teacher_ckpt_dir": ckpt_dir,
        "teacher_growsp": int(args.teacher_growsp),
        "model_checkpoint": os.path.basename(model_path),
        "cls_checkpoint": os.path.basename(cls_path),
        "model_sha256": checkpoint_sha256(model_path),
        "cls_sha256": checkpoint_sha256(cls_path),
    }


def pseudo_signature_matches(marker, expected):
    for key, value in expected.items():
        if marker.get(key) != value:
            return False, key
    return True, ""


def ensure_frozen_teacher_pseudo(args, logger, model, primitive_classifier, teacher_epoch, training_areas):
    marker_path = pseudo_marker_path(args, teacher_epoch)
    expected_signature = teacher_pseudo_signature(args, teacher_epoch)
    if os.path.exists(marker_path) and not args.rebuild_pseudo:
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
            matched, mismatch_key = pseudo_signature_matches(marker, expected_signature)
            if matched:
                logger.info(f"Using cached frozen-teacher pseudo labels: {args.pseudo_label_path}")
                return
            logger.info(
                "Cached pseudo labels do not match current teacher on key '{}'; rebuilding {}.".format(
                    mismatch_key,
                    args.pseudo_label_path,
                )
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.info(f"Could not read pseudo marker {marker_path}: {exc}; rebuilding pseudo labels.")

    logger.info("Building frozen-teacher pseudo labels once.")
    clusterset = S3DIStrain(args, areas=training_areas)
    cluster_loader = DataLoader(
        clusterset,
        batch_size=1,
        collate_fn=cfl_collate_fn(),
        num_workers=args.cluster_workers,
        pin_memory=True,
    )
    cluster_loader.dataset.mode = "cluster"

    feats, labels, sp_index, context = get_sp_feature(args, cluster_loader, model, args.teacher_growsp)
    sp_feats = torch.cat(feats, dim=0)[:, :args.feats_dim]
    neural_sp_feats = F.normalize(sp_feats, dim=1)
    teacher_centers = F.normalize(primitive_classifier.weight.detach().cpu(), dim=1)
    primitive_labels = F.linear(neural_sp_feats, teacher_centers).argmax(dim=1).cpu().numpy()
    all_pseudo, _, _, _, _ = get_pseudo(args, context, primitive_labels, sp_index)

    Path(args.pseudo_label_path).mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w", encoding="utf-8") as f:
        marker = dict(expected_signature)
        marker.update(
            {
                "labelled_ratio": float((all_pseudo != -1).sum() / max(all_pseudo.shape[0], 1)),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        json.dump(
            marker,
            f,
            indent=2,
        )
    logger.info(f"Pseudo labels cached at {args.pseudo_label_path}")


def region_consensus_targets(logits, regions, refine_mask, batch_ids=None, conf_th=0.25, temperature=0.7, ignore_index=-1):
    probs = F.softmax(logits.detach() / max(temperature, 1e-6), dim=1)
    targets = torch.full((logits.size(0),), ignore_index, dtype=torch.long, device=logits.device)
    target_conf = logits.new_zeros((logits.size(0),))
    if batch_ids is None:
        batch_ids = torch.zeros((logits.size(0),), dtype=torch.long, device=logits.device)
    batch_ids = batch_ids.long().to(logits.device)
    valid = refine_mask & (regions >= 0)
    if valid.sum() == 0:
        return targets, target_conf

    for batch_id in torch.unique(batch_ids[valid]):
        scene_valid = valid & (batch_ids == batch_id)
        for region_id in torch.unique(regions[scene_valid]):
            mask = scene_valid & (regions == region_id)
            if mask.sum() == 0:
                continue
            region_prob = probs[mask].mean(dim=0)
            conf, target = region_prob.max(dim=0)
            if conf >= conf_th:
                targets[mask] = target
                target_conf[mask] = conf
    return targets, target_conf


def masked_entropy(logits, mask):
    if mask.sum() == 0:
        return logits.sum() * 0.0
    probs = F.softmax(logits[mask], dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1).mean()


def region_project_logits(logits, feats, classifier_weight, regions, batch_ids=None, mode="confprob"):
    projected = logits.clone()
    probs = F.softmax(logits, dim=1)
    point_conf = probs.max(dim=1)[0]
    if batch_ids is None:
        batch_ids = torch.zeros((logits.size(0),), dtype=torch.long, device=logits.device)
    batch_ids = batch_ids.long().to(logits.device)
    for batch_id in torch.unique(batch_ids):
        scene_mask = batch_ids == batch_id
        for region_id in torch.unique(regions[scene_mask]):
            if int(region_id.item()) == -1:
                continue
            mask = scene_mask & (regions == region_id)
            if not mask.any():
                continue
            if mode == "prob":
                region_prob = probs[mask].mean(dim=0, keepdim=True)
                projected[mask] = torch.log(region_prob.clamp_min(1e-6))
            elif mode == "confprob":
                weights = point_conf[mask].clamp_min(1e-6)
                region_prob = (probs[mask] * weights[:, None]).sum(dim=0, keepdim=True) / weights.sum().clamp_min(1e-6)
                projected[mask] = torch.log(region_prob.clamp_min(1e-6))
            elif mode == "feat":
                region_score = F.linear(
                    F.normalize(feats[mask].mean(dim=0, keepdim=True), dim=1),
                    F.normalize(classifier_weight),
                )
                projected[mask] = region_score
            elif mode == "vote":
                vote_labels, vote_counts = torch.unique(torch.argmax(logits[mask], dim=1), return_counts=True)
                vote_target = vote_labels[torch.argmax(vote_counts)]
                projected[mask] = logits.new_full((int(mask.sum().item()), logits.size(1)), -20.0)
                projected[mask, vote_target] = 20.0
            else:
                projected[mask] = logits[mask].mean(dim=0, keepdim=True)
    return projected


def split_project_logits(logits, split_targets, project_logit=20.0):
    if split_targets is None:
        return logits
    valid = split_targets >= 0
    if not valid.any():
        return logits
    projected = logits.clone()
    projected[valid] = logits.new_full((int(valid.sum().item()), logits.size(1)), -project_logit)
    projected[valid, split_targets[valid].long()] = project_logit
    return projected


def target_margin_loss(logits, targets, mask, margin=0.3, weights=None):
    valid = mask & (targets >= 0)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits_valid = logits[valid]
    targets_valid = targets[valid]
    target_logits = logits_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
    other_logits = logits_valid.clone()
    other_logits.scatter_(1, targets_valid.unsqueeze(1), -1e6)
    strongest_other = other_logits.max(dim=1)[0]
    loss = F.relu(margin - (target_logits - strongest_other))
    if weights is not None:
        w = weights[valid].clamp_min(0.05)
        return (loss * w).sum() / w.sum().clamp_min(1e-6)
    return loss.mean()


def train_one_epoch(
    args,
    logger,
    epoch,
    train_loader,
    model,
    semantic_centers,
    semantic_cluster_map,
    refiner,
    optimizer,
    target_model=None,
    target_semantic_centers=None,
):
    train_loader.dataset.mode = "train"
    model.eval()
    refiner.train()
    meters = {}
    t0 = time.time()

    for batch_idx, data in enumerate(train_loader):
        coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data
        in_field = ME.TensorField(features, coords, device=0)
        with torch.no_grad():
            feats = model(in_field)
            feats = F.normalize(feats[inds.long()], dim=1)
            base_logits = F.linear(feats, semantic_centers)
            if target_model is not None and target_semantic_centers is not None:
                target_feats = target_model(in_field)
                target_feats = F.normalize(target_feats[inds.long()], dim=1)
                target_logits = F.linear(target_feats, target_semantic_centers)
                if args.temporal_align_mode == "batch":
                    target_logits = align_logits_by_batch_predictions(target_logits, base_logits)
            else:
                target_logits = None

        point_coords = coords[inds.long(), 1:].float().cuda()
        point_batch_ids = coords[inds.long(), 0].long().cuda()
        point_colors = features[inds.long(), :3].float().cuda()
        point_regions = region.squeeze(-1).long().cuda()
        base_pseudo = base_logits.argmax(dim=1)
        primitive_pseudo = pseudo_labels.long().cuda()

        split_targets = None
        split_stats = {
            "split_queries": 0,
            "split_refine_ratio": 0.0,
            "split_regions": 0,
            "split_subregions": 0,
        }
        consistency_targets = None
        consistency_target_conf = None
        consistency_stats = {
            "consistency_queries": 0,
            "consistency_refine_ratio": 0.0,
            "consistency_regions": 0,
            "consistency_points": 0,
        }
        if args.refine_split_enable:
            query_indices, refine_mask, split_targets, split_target_conf, keep_mask, split_stats = build_split_region_queries(
                base_logits * args.split_logit_scale,
                feats.detach(),
                point_coords,
                point_colors,
                point_regions,
                point_batch_ids,
                min_region_points=args.split_min_region_points,
                min_child_points=args.split_min_child_points,
                max_split_regions_per_scene=args.split_max_regions,
                split_purity_threshold=args.split_purity_th,
                split_entropy_threshold=args.split_entropy_th,
                split_min_conf=args.split_min_conf,
                xyz_weight=args.split_xyz_weight,
                rgb_weight=args.split_rgb_weight,
                feat_weight=args.split_feat_weight,
                semantic_weight=args.split_semantic_weight,
                multi_proposal=args.split_multi_proposal,
                selection_mode=args.split_selection_mode,
                random_seed=args.split_random_seed,
            )
            query_stats = {
                "num_queries": split_stats["split_queries"],
                "refine_ratio": split_stats["split_refine_ratio"],
                "keep_ratio": split_stats["split_keep_ratio"],
            }
        else:
            split_target_conf = base_logits.new_zeros((base_logits.size(0),))
            query_indices, refine_mask, keep_mask, query_stats = build_error_queries(
                base_logits * args.consistency_logit_scale,
                base_pseudo,
                point_regions,
                point_batch_ids,
                point_coords,
                colors=point_colors,
                prob_threshold=args.refine_conf_th,
                margin_threshold=args.refine_margin_th,
                region_purity_threshold=args.refine_region_purity_th,
                color_consistency_threshold=args.refine_color_consistency_th,
                geometry_consistency_threshold=args.refine_geometry_consistency_th,
                min_region_points=args.refine_min_region_points,
                max_queries_per_scene=args.refine_max_queries,
            )

        if args.refine_consistency_enable:
            (
                consistency_queries,
                consistency_mask,
                consistency_targets,
                consistency_target_conf,
                consistency_keep_mask,
                consistency_stats,
            ) = build_region_consistency_queries(
                base_logits * args.consistency_logit_scale,
                point_regions,
                point_batch_ids,
                min_region_points=args.consistency_min_region_points,
                max_regions_per_scene=args.consistency_max_regions,
                min_region_conf=args.consistency_min_conf,
                min_disagree_ratio=args.consistency_min_disagree,
                point_conf_threshold=args.consistency_point_conf,
                point_entropy_threshold=args.consistency_point_entropy,
            )
            if consistency_queries.numel() > 0:
                query_indices = torch.unique(torch.cat([query_indices, consistency_queries], dim=0))
            refine_mask = refine_mask | consistency_mask
            keep_mask = keep_mask | consistency_keep_mask
            query_stats["num_queries"] = int(query_indices.numel())
            query_stats["refine_ratio"] = float(refine_mask.float().mean().item())
            query_stats["keep_ratio"] = float(keep_mask.float().mean().item())

        if query_indices.numel() == 0 or float(refine_mask.float().mean().item()) < args.refine_min_ratio:
            fallback_queries, fallback_mask, fallback_stats = build_uncertain_region_queries(
                base_logits,
                point_regions,
                point_batch_ids,
                min_region_points=args.refine_min_region_points,
                max_queries_per_scene=args.refine_max_queries,
            )
            if fallback_queries.numel() > 0:
                query_indices = torch.unique(torch.cat([query_indices, fallback_queries], dim=0))
                refine_mask = refine_mask | fallback_mask
            else:
                fallback_stats = {"fallback_queries": 0, "fallback_regions": 0}
        else:
            fallback_stats = {"fallback_queries": 0, "fallback_regions": 0}

        refined_logits = base_logits
        delta_logits = base_logits.new_zeros(base_logits.shape)
        for _ in range(max(int(args.refine_rounds), 1)):
            delta_logits = refiner(
                feats.detach(),
                point_coords,
                point_batch_ids,
                query_indices,
                point_regions,
                use_region_branch=args.refine_region_branch,
            )
            refined_logits = refined_logits + args.refine_residual_scale * delta_logits
        consensus_targets = torch.full((base_logits.size(0),), -1, dtype=torch.long, device=base_logits.device)
        target_conf = base_logits.new_zeros((base_logits.size(0),))
        target_weight = base_logits.new_zeros((base_logits.size(0),))
        split_support_mask = torch.zeros((base_logits.size(0),), dtype=torch.bool, device=base_logits.device)
        temporal_support_mask = torch.zeros((base_logits.size(0),), dtype=torch.bool, device=base_logits.device)
        consistency_support_mask = torch.zeros((base_logits.size(0),), dtype=torch.bool, device=base_logits.device)
        region_support_mask = torch.zeros((base_logits.size(0),), dtype=torch.bool, device=base_logits.device)
        temporal_targets = torch.full((base_logits.size(0),), -1, dtype=torch.long, device=base_logits.device)
        if split_targets is not None and (split_targets >= 0).any():
            valid_split = split_targets >= 0
            consensus_targets[valid_split] = split_targets[valid_split]
            target_conf[valid_split] = split_target_conf[valid_split]
            target_weight[valid_split] = args.split_target_weight
            split_support_mask = valid_split
        if target_logits is not None:
            temporal_probs = F.softmax(target_logits.detach() * args.temporal_logit_scale, dim=1)
            temporal_conf, temporal_targets = temporal_probs.max(dim=1)
            valid_temporal = (
                (temporal_conf >= args.temporal_target_conf)
                & (temporal_targets != base_pseudo)
                & (consensus_targets < 0)
            )
            consensus_targets[valid_temporal] = temporal_targets[valid_temporal]
            target_conf[valid_temporal] = temporal_conf[valid_temporal]
            target_weight[valid_temporal] = args.temporal_target_weight
            temporal_support_mask = temporal_conf >= args.temporal_target_conf
        if consistency_targets is not None and (consistency_targets >= 0).any():
            valid_consistency = (
                (consistency_targets >= 0)
                & (consensus_targets < 0)
                & (consistency_target_conf >= args.consistency_train_conf)
            )
            consensus_targets[valid_consistency] = consistency_targets[valid_consistency]
            target_conf[valid_consistency] = consistency_target_conf[valid_consistency]
            target_weight[valid_consistency] = args.consistency_target_weight
            consistency_support_mask = (consistency_targets >= 0) & (consistency_target_conf >= args.consistency_train_conf)
        region_targets, region_target_conf = region_consensus_targets(
            base_logits,
            point_regions,
            refine_mask,
            batch_ids=point_batch_ids,
            conf_th=args.consensus_conf_th,
            temperature=args.consensus_temp,
        )
        missing_targets = (consensus_targets < 0) & (region_targets >= 0)
        consensus_targets[missing_targets] = region_targets[missing_targets]
        target_conf[missing_targets] = region_target_conf[missing_targets]
        target_weight[missing_targets] = args.region_target_weight
        region_support_mask = region_targets >= 0
        consensus_mask = (consensus_targets >= 0) & (target_weight > 0)
        flip_mask = consensus_mask & (consensus_targets != base_pseudo)
        support_count = torch.zeros((base_logits.size(0),), dtype=torch.long, device=base_logits.device)
        if split_targets is not None:
            support_count += (split_support_mask & (split_targets == consensus_targets)).long()
        support_count += (temporal_support_mask & (temporal_targets == consensus_targets)).long()
        if consistency_targets is not None:
            support_count += (consistency_support_mask & (consistency_targets == consensus_targets)).long()
        support_count += (region_support_mask & (region_targets == consensus_targets)).long()
        if args.refine_optimize_mode == "trusted":
            candidate_optimize_mask = flip_mask & (target_conf >= args.refine_optimize_min_conf)
        elif args.refine_optimize_mode == "split":
            candidate_optimize_mask = flip_mask & split_support_mask & (target_conf >= args.refine_optimize_split_min_conf)
        elif args.refine_optimize_mode == "split_or_multi":
            split_ok = split_support_mask & (target_conf >= args.refine_optimize_split_min_conf)
            multi_ok = (support_count >= args.refine_optimize_min_support) & (target_conf >= args.refine_optimize_min_conf)
            candidate_optimize_mask = flip_mask & (split_ok | multi_ok)
        else:
            candidate_optimize_mask = flip_mask
        if candidate_optimize_mask.any():
            optimize_mask = candidate_optimize_mask
        elif args.refine_optimize_mode == "flip":
            optimize_mask = consensus_mask
        else:
            optimize_mask = torch.zeros_like(consensus_mask)
        keep_mask = keep_mask | (consensus_mask & ~optimize_mask)

        projected_logits = region_project_logits(
            refined_logits,
            feats.detach(),
            semantic_centers,
            point_regions,
            batch_ids=point_batch_ids,
            mode=args.refine_region_project_mode,
        ) if args.refine_region_project else refined_logits
        no_op_projected_logits = region_project_logits(
            base_logits.detach(),
            feats.detach(),
            semantic_centers,
            point_regions,
            batch_ids=point_batch_ids,
            mode=args.refine_region_project_mode,
        ) if args.refine_region_project else base_logits.detach()
        if args.refine_split_project:
            projected_logits = split_project_logits(projected_logits, split_targets, args.refine_split_project_logit)
            no_op_projected_logits = split_project_logits(no_op_projected_logits, split_targets, args.refine_split_project_logit)

        if optimize_mask.sum() > 0:
            ce_each = F.cross_entropy(refined_logits[optimize_mask] * 3.0, consensus_targets[optimize_mask], reduction="none")
            ce_weight = (target_conf[optimize_mask] * target_weight[optimize_mask]).clamp_min(0.01)
            loss_ce = (ce_each * ce_weight).sum() / ce_weight.sum().clamp_min(1e-6)
            project_ce_each = F.cross_entropy(projected_logits[optimize_mask] * 3.0, consensus_targets[optimize_mask], reduction="none")
            loss_project_ce = (project_ce_each * ce_weight).sum() / ce_weight.sum().clamp_min(1e-6)
        else:
            loss_ce = refined_logits.sum() * 0.0
            loss_project_ce = refined_logits.sum() * 0.0
        if args.refine_accept_lambda > 0 and optimize_mask.sum() > 0:
            non_forced_mask = optimize_mask
            if split_targets is not None:
                non_forced_mask = non_forced_mask & (split_targets < 0)
            if non_forced_mask.sum() > 0:
                target_ids = consensus_targets[non_forced_mask].long()
                refined_target_prob = F.softmax(projected_logits[non_forced_mask], dim=1).gather(1, target_ids.unsqueeze(1)).squeeze(1)
                noop_target_prob = F.softmax(no_op_projected_logits[non_forced_mask], dim=1).gather(1, target_ids.unsqueeze(1)).squeeze(1)
                accept_weight = (target_conf[non_forced_mask] * target_weight[non_forced_mask]).clamp_min(0.01)
                accept_each = F.relu(noop_target_prob + args.refine_accept_gain - refined_target_prob)
                loss_accept = (accept_each * accept_weight).sum() / accept_weight.sum().clamp_min(1e-6)
            else:
                loss_accept = refined_logits.sum() * 0.0
        else:
            loss_accept = refined_logits.sum() * 0.0
        if args.refine_noop_kl_lambda > 0:
            allow_change_mask = optimize_mask & (target_conf >= args.refine_noop_kl_conf)
            noop_kl_mask = (refine_mask | keep_mask | consensus_mask) & ~allow_change_mask
            loss_noop_kl = refinement_keep_kl(projected_logits, no_op_projected_logits.detach(), noop_kl_mask)
        else:
            loss_noop_kl = refined_logits.sum() * 0.0

        if primitive_pseudo.numel() == base_logits.size(0):
            pseudo_valid = (primitive_pseudo >= 0) & (primitive_pseudo < semantic_cluster_map.numel())
            scaled_base_prob = F.softmax(base_logits.detach() * args.consistency_logit_scale, dim=1)
            pseudo_conf_mask = scaled_base_prob.max(dim=1)[0] >= args.refine_pseudo_conf
            pseudo_mask = pseudo_valid & pseudo_conf_mask
        else:
            pseudo_mask = torch.zeros((base_logits.size(0),), dtype=torch.bool, device=base_logits.device)
        if pseudo_mask.sum() > 0:
            pseudo_semantic = semantic_cluster_map[primitive_pseudo[pseudo_mask]].long()
            loss_pseudo = F.cross_entropy(refined_logits[pseudo_mask] * 3.0, pseudo_semantic)
        else:
            loss_pseudo = refined_logits.sum() * 0.0

        loss_keep = refinement_keep_kl(refined_logits * 3.0, base_logits * 3.0, keep_mask)
        loss_margin = target_margin_loss(
            refined_logits * 3.0,
            consensus_targets,
            optimize_mask,
            args.refine_target_margin,
            target_conf * target_weight,
        )
        loss_delta = delta_l2(delta_logits, refine_mask | keep_mask)
        loss_entropy = masked_entropy(refined_logits, optimize_mask)
        loss = (
            args.refine_ce_lambda * loss_ce
            + args.refine_project_ce_lambda * loss_project_ce
            + args.refine_accept_lambda * loss_accept
            + args.refine_noop_kl_lambda * loss_noop_kl
            + args.refine_pseudo_lambda * loss_pseudo
            + args.refine_margin_lambda * loss_margin
            + args.refine_keep_lambda * loss_keep
            + args.refine_delta_lambda * loss_delta
            + args.refine_entropy_lambda * loss_entropy
        )

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            changed = (refined_logits.argmax(dim=1) != base_pseudo) & refine_mask
            changed_within_refine = changed.float().sum() / refine_mask.float().sum().clamp_min(1.0)
            values = {
                "loss_refine": loss.item(),
                "loss_ce": loss_ce.item(),
                "loss_project_ce": loss_project_ce.item(),
                "loss_accept": loss_accept.item(),
                "loss_noop_kl": loss_noop_kl.item(),
                "loss_pseudo": loss_pseudo.item(),
                "loss_keep": loss_keep.item(),
                "loss_margin": loss_margin.item(),
                "loss_delta": loss_delta.item(),
                "loss_entropy": loss_entropy.item(),
                "refine_ratio": query_stats["refine_ratio"],
                "keep_ratio": float(keep_mask.float().mean().item()),
                "queries": float(query_stats["num_queries"]),
                "fallback_queries": float(fallback_stats["fallback_queries"]),
                "split_regions": float(split_stats["split_regions"]),
                "split_subregions": float(split_stats["split_subregions"]),
                "consistency_regions": float(consistency_stats["consistency_regions"]),
                "consistency_points": float(consistency_stats["consistency_points"]),
                "consensus_ratio": float(consensus_mask.float().mean().item()),
                "flip_ratio": float(flip_mask.float().mean().item()),
                "optimize_ratio": float(optimize_mask.float().mean().item()),
                "support_ratio": float((support_count >= args.refine_optimize_min_support).float().mean().item()),
                "pseudo_ratio": float(pseudo_mask.float().mean().item()),
                "changed_all": float(changed.float().mean().item()),
                "changed_refine": float(changed_within_refine.item()),
            }
            for key, value in values.items():
                meters[key] = meters.get(key, 0.0) + value

        if (batch_idx + 1) % args.log_interval == 0:
            denom = float(args.log_interval)
            logger.info(
                "Epoch {:03d} [{:04d}/{:04d}] "
                "loss {:.4f} ce {:.4f} pce {:.4f} accept {:.4f} noopkl {:.4f} pseudo {:.4f} margin {:.4f} keep {:.4f} delta {:.4f} entropy {:.4f} "
                "refine {:.2f}% keep {:.2f}% consensus {:.2f}% flip {:.2f}% opt {:.2f}% support {:.2f}% pseudo {:.2f}% changed {:.2f}% changed@refine {:.2f}% "
                "queries {:.1f} fallback {:.1f} split {:.1f}/{:.1f} consistency {:.1f}/{:.1f}".format(
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    meters["loss_refine"] / denom,
                    meters["loss_ce"] / denom,
                    meters["loss_project_ce"] / denom,
                    meters["loss_accept"] / denom,
                    meters["loss_noop_kl"] / denom,
                    meters["loss_pseudo"] / denom,
                    meters["loss_margin"] / denom,
                    meters["loss_keep"] / denom,
                    meters["loss_delta"] / denom,
                    meters["loss_entropy"] / denom,
                    100 * meters["refine_ratio"] / denom,
                    100 * meters["keep_ratio"] / denom,
                    100 * meters["consensus_ratio"] / denom,
                    100 * meters["flip_ratio"] / denom,
                    100 * meters["optimize_ratio"] / denom,
                    100 * meters["support_ratio"] / denom,
                    100 * meters["pseudo_ratio"] / denom,
                    100 * meters["changed_all"] / denom,
                    100 * meters["changed_refine"] / denom,
                    meters["queries"] / denom,
                    meters["fallback_queries"] / denom,
                    meters["split_regions"] / denom,
                    meters["split_subregions"] / denom,
                    meters["consistency_regions"] / denom,
                    meters["consistency_points"] / denom,
                )
            )
            meters = {}

    logger.info(f"Epoch {epoch:03d} time {time.time() - t0:.1f}s")


def save_for_eval(args, epoch, model, primitive_classifier, refiner, optimizer=None):
    ckpt_dir = os.path.join(args.save_path, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": primitive_classifier.state_dict(),
        "refiner_state_dict": refiner.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, os.path.join(ckpt_dir, f"model_{epoch}_resume.pth"))
    for root in (args.save_path, ckpt_dir):
        torch.save(model.state_dict(), os.path.join(root, f"model_{epoch}_checkpoint.pth"))
        torch.save(primitive_classifier.state_dict(), os.path.join(root, f"cls_{epoch}_checkpoint.pth"))
        torch.save(refiner.state_dict(), os.path.join(root, f"refiner_{epoch}_checkpoint.pth"))


def save_best_for_eval(args, epoch, stats, model, primitive_classifier, refiner, optimizer=None):
    save_for_eval(args, "best", model, primitive_classifier, refiner, optimizer)
    summary = {
        "best_epoch": int(epoch),
        "baseline_mIoU": float(stats["baseline_mIoU"]),
        "no_op_refined_mIoU": float(stats.get("no_op_refined_mIoU", stats["baseline_mIoU"])),
        "refined_mIoU": float(stats["refined_mIoU"]),
        "delta_mIoU": float(stats["delta_mIoU"]),
        "delta_vs_no_op_projection": float(stats.get("delta_vs_no_op_projection", stats["delta_mIoU"])),
        "changed_ratio": float(stats["changed_ratio"]),
        "trusted_ratio": float(stats["trusted_ratio"]),
        "changed_trusted_ratio": float(stats["changed_trusted_ratio"]),
        "queries": int(stats["queries"]),
        "split_regions": int(stats.get("split_regions", 0)),
    }
    with open(os.path.join(args.save_path, "best_refiner.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()
    if not args.pseudo_label_path:
        args.pseudo_label_path = os.path.join(args.save_path, "pseudo_labels")
    args.refine_enable = True
    args.refine_split_enable = True

    set_seed(args.seed)
    logger = set_logger(args.save_path)
    max_key_len = max(len(k) for k in vars(args))
    for key, value in vars(args).items():
        logger.info(f"{key:<{max_key_len}} : {value}")

    all_areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
    test_areas = parse_test_areas(args.test_area)
    unknown_areas = sorted(set(test_areas) - set(all_areas))
    if unknown_areas:
        raise ValueError("Unknown S3DIS test_area values: {}".format(", ".join(unknown_areas)))
    training_areas = sorted(list(set(all_areas) - set(test_areas)))

    model, primitive_classifier, teacher_epoch = load_teacher(args, logger)
    semantic_centers, cluster_pred = build_semantic_centers(primitive_classifier, args.semantic_class)
    semantic_cluster_map = torch.as_tensor(cluster_pred, dtype=torch.long, device="cuda")
    target_model = None
    target_semantic_centers = None
    if args.target_teacher_epoch >= 0 or args.target_teacher_ckpt_dir:
        target_ckpt_dir = args.target_teacher_ckpt_dir or args.teacher_ckpt_dir
        target_model, target_classifier, target_epoch = load_teacher_checkpoint(
            args,
            logger,
            target_ckpt_dir,
            args.target_teacher_epoch,
            "target teacher",
        )
        raw_target_centers, _ = build_semantic_centers(target_classifier, args.semantic_class)
        if args.temporal_align_mode == "batch":
            target_semantic_centers = raw_target_centers
        else:
            target_semantic_centers = align_semantic_centers(raw_target_centers, semantic_centers)
        logger.info(
            f"Temporal target teacher enabled: epoch {target_epoch} aligned to teacher epoch {teacher_epoch} "
            f"with {args.temporal_align_mode} alignment"
        )
    ensure_frozen_teacher_pseudo(args, logger, model, primitive_classifier, teacher_epoch, training_areas)

    trainset = S3DIStrain(args, areas=training_areas)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cfl_collate_fn(),
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )

    refiner = ErrorQueryRefiner(
        feat_dim=args.feats_dim,
        num_classes=args.semantic_class,
        hidden_dim=args.refine_hidden_dim,
        num_heads=args.refine_num_heads,
        dropout=args.refine_dropout,
    ).cuda()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.refine_lr, weight_decay=args.refine_weight_decay)

    save_for_eval(args, 0, model, primitive_classifier, refiner, optimizer)
    logger.info("Saved epoch 0 no-op refiner baseline.")
    no_op_refined_miou = None
    with torch.no_grad():
        o_acc, m_acc, s = eval_s3dis(0, args, test_areas)
        logger.info(f"Epoch: 00, oAcc {o_acc:.2f}  mAcc {m_acc:.2f} IoUs{s}")
        stats = getattr(args, "eval_refine_stats", None)
        if stats:
            no_op_refined_miou = float(stats["refined_mIoU"])
            logger.info(
                "Epoch: 00, Refined oAcc {:.2f}  mAcc {:.2f}  delta_mIoU {:+.2f}  "
                "changed {:.2f}% trusted {:.2f}% keep {:.2f}% changed@trusted {:.2f}% queries {} split_regions {}{}".format(
                    stats["refined_oAcc"],
                    stats["refined_mAcc"],
                    stats["delta_mIoU"],
                    100 * stats["changed_ratio"],
                    100 * stats["trusted_ratio"],
                    100 * stats.get("keep_ratio", 0.0),
                    100 * stats["changed_trusted_ratio"],
                    stats["queries"],
                    stats.get("split_regions", 0),
                    stats["refined_s"],
                )
            )
    if no_op_refined_miou is None:
        no_op_refined_miou = 0.0

    best_refiner_gain = 0.0
    best_summary = None
    for epoch in range(1, args.max_epoch + 1):
        train_one_epoch(
            args,
            logger,
            epoch,
            train_loader,
            model,
            semantic_centers,
            semantic_cluster_map,
            refiner,
            optimizer,
            target_model=target_model,
            target_semantic_centers=target_semantic_centers,
        )
        if epoch % args.save_interval == 0 or epoch == args.max_epoch:
            save_for_eval(args, epoch, model, primitive_classifier, refiner, optimizer)
        if epoch % args.eval_interval == 0 or epoch == args.max_epoch:
            with torch.no_grad():
                o_acc, m_acc, s = eval_s3dis(epoch, args, test_areas)
                logger.info(f"Epoch: {epoch:02d}, oAcc {o_acc:.2f}  mAcc {m_acc:.2f} IoUs{s}")
                stats = getattr(args, "eval_refine_stats", None)
                if stats:
                    stats["no_op_refined_mIoU"] = no_op_refined_miou
                    stats["delta_vs_no_op_projection"] = stats["refined_mIoU"] - no_op_refined_miou
                    logger.info(
                        "Epoch: {:02d}, Refined oAcc {:.2f}  mAcc {:.2f}  delta_mIoU {:+.2f}  delta_vs_noop {:+.2f}  "
                        "changed {:.2f}% trusted {:.2f}% keep {:.2f}% changed@trusted {:.2f}% queries {} split_regions {}{}".format(
                            epoch,
                            stats["refined_oAcc"],
                            stats["refined_mAcc"],
                            stats["delta_mIoU"],
                            stats["delta_vs_no_op_projection"],
                            100 * stats["changed_ratio"],
                            100 * stats["trusted_ratio"],
                            100 * stats.get("keep_ratio", 0.0),
                            100 * stats["changed_trusted_ratio"],
                            stats["queries"],
                            stats.get("split_regions", 0),
                            stats["refined_s"],
                        )
                    )
                    if stats["delta_vs_no_op_projection"] > best_refiner_gain:
                        best_refiner_gain = stats["delta_vs_no_op_projection"]
                        best_summary = save_best_for_eval(args, epoch, stats, model, primitive_classifier, refiner, optimizer)
                        logger.info(
                            "Best refiner updated: epoch {} refined_mIoU {:.2f} delta_mIoU {:+.2f} delta_vs_noop {:+.2f} "
                            "changed {:.2f}% changed@trusted {:.2f}%".format(
                                epoch,
                                best_summary["refined_mIoU"],
                                best_summary["delta_mIoU"],
                                best_summary["delta_vs_no_op_projection"],
                                100 * best_summary["changed_ratio"],
                                100 * best_summary["changed_trusted_ratio"],
                            )
                        )

    if best_summary is None:
        logger.info("No positive refined checkpoint found; epoch 0 no-op refiner remains the safest fallback.")
    else:
        logger.info(
            "Training finished. Best refiner: epoch {} refined_mIoU {:.2f} delta_mIoU {:+.2f} delta_vs_noop {:+.2f}. "
            "Evaluate it with eval_S3DIS.py --eval_epoch best --refine_enable --refine_split_enable.".format(
                best_summary["best_epoch"],
                best_summary["refined_mIoU"],
                best_summary["delta_mIoU"],
                best_summary["delta_vs_no_op_projection"],
            )
        )


if __name__ == "__main__":
    main()
