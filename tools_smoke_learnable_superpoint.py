"""Real-scene forward/backward smoke test for training-integrated superpoints."""

import argparse
import json
import os
import random
from types import SimpleNamespace

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from datasets.S3DIS import S3DIStrain, cfl_collate_fn
from lib.my_utils import build_model
from lib.utils import get_sp_feature
from models.learnable_superpoint import (
    SemanticDifferenceSuperpointLearner,
    verified_region_supervision_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="ckpt/S3DIS/refiner_frozen_conservative")
    parser.add_argument("--epoch", type=int, default=350)
    parser.add_argument("--area", default="Area_1")
    parser.add_argument("--scene_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify_reclustering", action="store_true")
    parser.add_argument("--verify_train_step", action="store_true")
    return parser.parse_args()


def load_state(path, key):
    state = torch.load(path, map_location="cpu")
    return state[key] if isinstance(state, dict) and key in state else state


def main():
    args = parse_args()
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    config = SimpleNamespace(
        data_path="data/S3DIS/input",
        sp_path="data/S3DIS/initial_superpoints",
        pseudo_label_path="pseudo_label_s3dis",
        voxel_size=0.05,
        ignore_label=12,
        drop_threshold=10,
        bn_momentum=0.02,
        feats_dim=128,
        w_rgb=1.0,
        w_xyz=0.2,
        w_norm=0.8,
        c_rgb=3.0,
        c_shape=3.0,
        z_enable=False,
        region_weight_enable=False,
        semantic_class=12,
        log_interval=1,
        learnable_sp_query_scale=10.0,
        learnable_sp_min_region_points=20,
        learnable_sp_min_child_points=6,
        learnable_sp_max_regions=12,
        learnable_sp_purity_th=0.9,
        learnable_sp_entropy_th=0.3,
        learnable_sp_child_conf_th=0.2,
        learnable_sp_conf_gain=0.01,
        learnable_sp_semantic_sep=0.15,
        learnable_sp_structure_lambda=0.1,
        learnable_sp_supervision_lambda=0.2,
    )
    dataset = S3DIStrain(config, areas=[args.area])
    dataset.mode = "train"
    sample_index = args.scene_index % len(dataset)
    data = cfl_collate_fn()([dataset[sample_index]])
    coords, features, _, _, _, _, inds, regions, _ = data

    model = build_model(
        "res16fpn18",
        in_channels=6,
        out_channels=300,
        conv1_kernel_size=5,
        config=config,
    ).to(device)
    model_path = os.path.join(args.checkpoint_dir, f"model_{args.epoch}_checkpoint.pth")
    classifier_path = os.path.join(args.checkpoint_dir, f"cls_{args.epoch}_checkpoint.pth")
    model.load_state_dict(load_state(model_path, "model_state_dict"))
    classifier_weight = load_state(classifier_path, "classifier_state_dict")["weight"].to(device)
    model.eval()

    with torch.no_grad():
        field = ME.TensorField(features, coords, device=device.index)
        point_feats = F.normalize(model(field)[inds.long()], dim=1)
    primitive_to_semantic = KMeans(n_clusters=12, n_init=10, random_state=0).fit_predict(
        classifier_weight.cpu().numpy()
    )
    mapping = torch.from_numpy(primitive_to_semantic).long().to(device)
    semantic_centers = classifier_weight.new_zeros((12, classifier_weight.size(1)))
    for semantic_id in range(12):
        mask = mapping == semantic_id
        semantic_centers[semantic_id] = classifier_weight[mask].mean(dim=0)
    semantic_centers = F.normalize(semantic_centers, dim=1)
    semantic_logits = F.linear(point_feats, semantic_centers).detach().requires_grad_(True)

    point_coords = coords[inds.long(), 1:].to(device)
    point_colors = features[inds.long(), :3].to(device)
    batch_ids = coords[inds.long(), 0].long().to(device)
    regions = regions.squeeze(1).long().to(device)
    learner = SemanticDifferenceSuperpointLearner(128, 12).to(device)
    output = learner(
        point_feats.detach(),
        point_coords,
        point_colors,
        semantic_logits * 10.0,
        regions,
        batch_ids,
        min_confidence_gain=0.0,
    )
    supervision_loss = verified_region_supervision_loss(semantic_logits * 3, output)
    total_loss = output.structure_loss + supervision_loss
    total_loss.backward()

    report = dict(output.stats)
    report.update(
        {
            "scene": dataset.name[sample_index],
            "points": int(point_feats.size(0)),
            "structure_loss": float(output.structure_loss.detach().item()),
            "supervision_loss": float(supervision_loss.detach().item()),
            "trainable_parameters": sum(parameter.numel() for parameter in learner.parameters()),
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            "affinity_grad_norm": float(learner.affinity[-1].weight.grad.norm().item()),
        }
    )
    if args.verify_reclustering:
        cluster_dataset = S3DIStrain(config, areas=[args.area])
        cluster_dataset.file = [cluster_dataset.file[sample_index]]
        cluster_dataset.name = [cluster_dataset.name[sample_index]]
        cluster_loader = DataLoader(
            cluster_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=cfl_collate_fn(),
        )
        _, _, point_region_indices, context = get_sp_feature(
            config,
            cluster_loader,
            model,
            current_growsp=None,
            learnable_sp=learner,
            semantic_centers=semantic_centers,
        )
        valid_points = int((context[0][2] != -1).sum().item())
        report["recluster_valid_points"] = valid_points
        report["recluster_index_points"] = int(point_region_indices[0].numel())
        report["recluster_regions"] = int(point_region_indices[0].unique().numel())
        report["recluster_index_contract"] = valid_points == point_region_indices[0].numel()
        report["recluster_structure_stats"] = config.cluster_learnable_sp_stats
    if args.verify_train_step:
        import train_S3DIS as train_module

        class SmokeLogger:
            def info(self, *unused_args, **unused_kwargs):
                pass

        train_dataset = S3DIStrain(config, areas=[args.area])
        train_dataset.file = [train_dataset.file[sample_index]]
        train_dataset.name = [train_dataset.name[sample_index]]
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=cfl_collate_fn(),
        )
        primitive_classifier = torch.nn.Linear(128, 300, bias=False).to(device)
        primitive_classifier.weight.data.copy_(classifier_weight)
        primitive_classifier.weight.requires_grad_(False)
        train_learner = SemanticDifferenceSuperpointLearner(128, 12).to(device)
        structure_optimizer = torch.optim.AdamW(train_learner.parameters(), lr=1e-3)
        before = train_learner.affinity[0].weight.detach().clone()
        train_module.args = config
        train_module.train(
            train_loader,
            SmokeLogger(),
            model,
            None,
            torch.nn.CrossEntropyLoss(ignore_index=-1).to(device),
            1,
            None,
            primitive_classifier,
            mapping.cpu(),
            freeze_backbone=True,
            learnable_sp=train_learner,
            learnable_sp_optimizer=structure_optimizer,
        )
        parameter_change = (train_learner.affinity[0].weight.detach() - before).abs().sum()
        report["train_step_parameter_change"] = float(parameter_change.item())
        report["train_step_updated"] = bool(parameter_change.item() > 0)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
