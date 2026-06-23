import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from models.query_refiner import ErrorQueryRefiner
from lib.error_query import build_error_queries
from lib.split_regions import build_region_consistency_queries, build_split_region_queries, build_uncertain_region_queries
from lib.utils import get_fixclassifier
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/S3DIS/input',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/S3DIS/initial_superpoints',
                        help='initial superpoint path')
    parser.add_argument('--save_path', type=str, default='trained_models/S3DIS/',
                        help='model savepath')
    parser.add_argument('--eval_epoch', type=str, default='1270',
                        help='checkpoint epoch or "best" for direct evaluation')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=8, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=12, help='invalid label')
    parser.add_argument('--refine_enable', action='store_true', default=False, help='Enable error-query semantic refinement')
    parser.add_argument('--refine_hidden_dim', type=int, default=128, help='hidden dimension for query refinement')
    parser.add_argument('--refine_num_heads', type=int, default=4, help='attention heads for query refinement')
    parser.add_argument('--refine_dropout', type=float, default=0.0, help='dropout for query refinement')
    parser.add_argument('--refine_residual_scale', type=float, default=0.9, help='scale of refinement residual logits')
    parser.add_argument('--refine_rounds', type=int, default=1, help='number of repeated refinement residual applications')
    parser.add_argument('--refine_gate_enable', action='store_true', default=False, help='apply refinement residual only when confidence/margin improves')
    parser.add_argument('--refine_gate_min_conf', type=float, default=0.30, help='minimum refined confidence for gated changed predictions')
    parser.add_argument('--refine_gate_min_margin', type=float, default=0.02, help='minimum refined probability margin for gated changed predictions')
    parser.add_argument('--refine_gate_conf_gain', type=float, default=0.0, help='minimum confidence gain for keeping same-label refinements')
    parser.add_argument('--refine_region_accept_gate', action='store_true', default=False, help='fallback to no-op region projection unless refined projected scores improve region confidence')
    parser.add_argument('--refine_region_accept_conf_gain', type=float, default=0.02, help='minimum region confidence gain for accepting refined projected scores')
    parser.add_argument('--refine_region_accept_entropy_gain', type=float, default=0.0, help='minimum entropy reduction for accepting refined projected scores')
    parser.add_argument('--refine_apply_all', action='store_true', default=True, help='apply refiner residual to all points instead of only query-selected masks')
    parser.add_argument('--refine_region_project', action='store_true', default=True, help='project refined logits to GrowSP region-average predictions')
    parser.add_argument('--refine_region_project_mode', type=str, default='confprob', choices=['logit', 'prob', 'confprob', 'feat', 'vote'], help='region projection aggregation used after refinement')
    parser.add_argument('--refine_region_branch', action='store_true', default=False, help='enable region-level residual branch in the refiner')
    parser.add_argument('--refine_split_project', action='store_true', default=True, help='force accepted split sub-regions to their self-supervised semantic targets')
    parser.add_argument('--refine_split_project_logit', type=float, default=20.0, help='logit value used for split sub-region target projection')
    parser.add_argument('--refine_split_project_gate', action='store_true', default=False, help='only apply split projection when refined logits support the split target')
    parser.add_argument('--refine_split_project_gate_conf', type=float, default=0.18, help='minimum refined target probability for gated split projection')
    parser.add_argument('--refine_split_project_gate_gain', type=float, default=0.02, help='minimum refined target probability gain over the base score for gated split projection')
    parser.add_argument('--refine_query_scale', type=float, default=10.0, help='logit scale used only for refinement query confidence')
    parser.add_argument('--refine_conf_th', type=float, default=0.55, help='model confidence threshold for query diagnostics')
    parser.add_argument('--refine_margin_th', type=float, default=0.05, help='model margin threshold for query diagnostics')
    parser.add_argument('--refine_region_purity_th', type=float, default=0.75, help='minimum primitive prediction purity for inference regions')
    parser.add_argument('--refine_color_consistency_th', type=float, default=0.2, help='minimum color consistency for refinement candidate regions')
    parser.add_argument('--refine_geometry_consistency_th', type=float, default=0.55, help='minimum geometry compactness for refinement candidate regions')
    parser.add_argument('--refine_min_region_points', type=int, default=10, help='minimum points in a trusted region')
    parser.add_argument('--refine_max_queries', type=int, default=20, help='maximum query anchors per scene')
    parser.add_argument('--semantic_logit_source', type=str, default='centroid', choices=['centroid', 'primitive_reduce'], help='semantic logit source for eval/refinement')
    parser.add_argument('--refine_semantic_reduce', type=str, default='max', choices=['max', 'mean', 'logsumexp'], help='primitive-to-semantic logit reduction')
    parser.add_argument('--refine_split_enable', action='store_true', default=False, help='use split-region queries for refinement')
    parser.add_argument('--refine_consistency_enable', action='store_true', default=True)
    parser.add_argument('--consistency_min_region_points', type=int, default=20)
    parser.add_argument('--consistency_max_regions', type=int, default=40)
    parser.add_argument('--consistency_min_conf', type=float, default=0.35)
    parser.add_argument('--consistency_min_disagree', type=float, default=0.02)
    parser.add_argument('--consistency_point_conf', type=float, default=0.55)
    parser.add_argument('--consistency_point_entropy', type=float, default=0.55)
    parser.add_argument('--consistency_logit_scale', type=float, default=10.0)
    parser.add_argument('--split_logit_scale', type=float, default=1.0)
    parser.add_argument('--split_min_region_points', type=int, default=30)
    parser.add_argument('--split_min_child_points', type=int, default=8)
    parser.add_argument('--split_max_regions', type=int, default=80)
    parser.add_argument('--split_purity_th', type=float, default=0.92)
    parser.add_argument('--split_entropy_th', type=float, default=0.25)
    parser.add_argument('--split_min_conf', type=float, default=0.15)
    parser.add_argument('--split_xyz_weight', type=float, default=1.0)
    parser.add_argument('--split_rgb_weight', type=float, default=0.5)
    parser.add_argument('--split_feat_weight', type=float, default=0.25)
    parser.add_argument('--split_semantic_weight', type=float, default=1.0)
    parser.add_argument('--split_multi_proposal', action='store_true', default=False)
    return parser.parse_args()


def reduce_primitive_logits(primitive_logits, cluster_pred, semantic_class, mode='max'):
    semantic_logits = primitive_logits.new_full((primitive_logits.size(0), semantic_class), -1e6)
    cluster_pred = torch.as_tensor(cluster_pred, dtype=torch.long, device=primitive_logits.device)
    for semantic_id in range(semantic_class):
        primitive_mask = cluster_pred == semantic_id
        if primitive_mask.sum() == 0:
            continue
        logits = primitive_logits[:, primitive_mask]
        if mode == 'mean':
            semantic_logits[:, semantic_id] = logits.mean(dim=1)
        elif mode == 'logsumexp':
            semantic_logits[:, semantic_id] = torch.logsumexp(logits, dim=1)
        else:
            semantic_logits[:, semantic_id] = logits.max(dim=1)[0]
    return semantic_logits


def apply_refinement_projection(args, scores, base_scores, feats, classifier, region, split_targets=None):
    if getattr(args, 'refine_region_project', False):
        region_project_mode = getattr(args, 'refine_region_project_mode', 'logit')
        probs = F.softmax(scores, dim=1)
        point_conf = probs.max(dim=1)[0]
        region_for_project = region.to(scores.device)
        for region_id in torch.unique(region_for_project):
            if int(region_id.item()) == -1:
                continue
            region_mask = region_for_project == region_id
            if not region_mask.any():
                continue
            if region_project_mode == 'prob':
                region_prob = probs[region_mask].mean(dim=0, keepdim=True)
                scores[region_mask] = torch.log(region_prob.clamp_min(1e-6))
            elif region_project_mode == 'confprob':
                weights = point_conf[region_mask].clamp_min(1e-6)
                region_prob = (probs[region_mask] * weights[:, None]).sum(dim=0, keepdim=True) / weights.sum().clamp_min(1e-6)
                scores[region_mask] = torch.log(region_prob.clamp_min(1e-6))
            elif region_project_mode == 'feat':
                region_score = F.linear(
                    F.normalize(feats[region_mask].mean(dim=0, keepdim=True), dim=1),
                    F.normalize(classifier.weight),
                )
                scores[region_mask] = region_score
            elif region_project_mode == 'vote':
                vote_labels, vote_counts = torch.unique(torch.argmax(scores[region_mask], dim=1), return_counts=True)
                vote_target = vote_labels[torch.argmax(vote_counts)]
                scores[region_mask] = scores.new_full((int(region_mask.sum().item()), scores.size(1)), -getattr(args, 'refine_split_project_logit', 20.0))
                scores[region_mask, vote_target] = getattr(args, 'refine_split_project_logit', 20.0)
            else:
                scores[region_mask] = scores[region_mask].mean(dim=0, keepdim=True)

    if getattr(args, 'refine_split_project', False) and getattr(args, 'refine_split_enable', False) and split_targets is not None:
        split_valid = split_targets >= 0
        if split_valid.any() and getattr(args, 'refine_split_project_gate', False):
            split_device = split_targets.to(scores.device).long()
            row = torch.arange(scores.size(0), device=scores.device)
            refined_probs = F.softmax(scores, dim=1)
            base_probs = F.softmax(base_scores, dim=1)
            target_prob = refined_probs[row, split_device.clamp_min(0)]
            base_target_prob = base_probs[row, split_device.clamp_min(0)]
            split_gate = (target_prob >= getattr(args, 'refine_split_project_gate_conf', 0.18)) | (
                target_prob >= base_target_prob + getattr(args, 'refine_split_project_gate_gain', 0.02)
            )
            split_valid = split_valid & split_gate
        if split_valid.any():
            scores[split_valid] = scores.new_full(
                (int(split_valid.sum().item()), scores.size(1)),
                -getattr(args, 'refine_split_project_logit', 20.0),
            )
            scores[split_valid, split_targets[split_valid].long()] = getattr(args, 'refine_split_project_logit', 20.0)
    return scores


def apply_region_accept_gate(args, candidate_scores, no_op_scores, region):
    region_for_gate = region.to(candidate_scores.device)
    accepted = torch.zeros((candidate_scores.size(0),), dtype=torch.bool, device=candidate_scores.device)
    for region_id in torch.unique(region_for_gate):
        if int(region_id.item()) == -1:
            continue
        region_mask = region_for_gate == region_id
        if not region_mask.any():
            continue
        cand_prob = F.softmax(candidate_scores[region_mask], dim=1).mean(dim=0)
        noop_prob = F.softmax(no_op_scores[region_mask], dim=1).mean(dim=0)
        cand_conf = cand_prob.max()
        noop_conf = noop_prob.max()
        cand_entropy = -(cand_prob * torch.log(cand_prob.clamp_min(1e-6))).sum()
        noop_entropy = -(noop_prob * torch.log(noop_prob.clamp_min(1e-6))).sum()
        accept = (
            cand_conf >= noop_conf + getattr(args, 'refine_region_accept_conf_gain', 0.02)
            and cand_entropy <= noop_entropy - getattr(args, 'refine_region_accept_entropy_gain', 0.0)
        )
        if accept:
            accepted[region_mask] = True
        else:
            candidate_scores[region_mask] = no_op_scores[region_mask]
    return candidate_scores, accepted


def eval_once(args, model, test_loader, classifier, primitive_classifier=None, cluster_pred=None, refiner=None, use_sp=False):

    all_preds, all_refined_preds, all_label = [], [], []
    stats = {
        "changed_points": 0,
        "trusted_points": 0,
        "keep_points": 0,
        "total_points": 0,
        "queries": 0,
        "split_regions": 0,
        "consistency_regions": 0,
        "changed_trusted_points": 0,
    }
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            #
            if refiner is not None:
                if getattr(args, 'semantic_logit_source', 'centroid') == 'primitive_reduce' and primitive_classifier is not None and cluster_pred is not None:
                    primitive_scores = F.linear(F.normalize(feats), F.normalize(primitive_classifier.weight))
                    base_scores = reduce_primitive_logits(
                        primitive_scores,
                        cluster_pred,
                        args.semantic_class,
                        mode=getattr(args, 'refine_semantic_reduce', 'max'),
                    )
                else:
                    base_scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                base_preds = torch.argmax(base_scores, dim=1).cpu()
                semantic_pseudo = torch.argmax(base_scores, dim=1)
                point_batch_ids = coords[:, 0].long().cuda()
                point_coords = coords[:, 1:].float().cuda()
                point_colors = features[:, :3].float().cuda()
                if getattr(args, 'refine_split_enable', False):
                    query_indices, refine_mask, split_targets, split_target_conf, keep_mask, query_stats = build_split_region_queries(
                        base_scores * getattr(args, 'split_logit_scale', 1.0),
                        feats,
                        point_coords,
                        point_colors,
                        region.cuda(),
                        point_batch_ids,
                        min_region_points=getattr(args, 'split_min_region_points', 30),
                        min_child_points=getattr(args, 'split_min_child_points', 8),
                        max_split_regions_per_scene=getattr(args, 'split_max_regions', 20),
                        split_purity_threshold=getattr(args, 'split_purity_th', 0.92),
                        split_entropy_threshold=getattr(args, 'split_entropy_th', 0.25),
                        split_min_conf=getattr(args, 'split_min_conf', 0.15),
                        xyz_weight=getattr(args, 'split_xyz_weight', 1.0),
                        rgb_weight=getattr(args, 'split_rgb_weight', 0.5),
                        feat_weight=getattr(args, 'split_feat_weight', 0.25),
                        semantic_weight=getattr(args, 'split_semantic_weight', 1.0),
                        multi_proposal=getattr(args, 'split_multi_proposal', False),
                    )
                    if getattr(args, 'refine_consistency_enable', False):
                        (
                            consistency_queries,
                            consistency_mask,
                            consistency_targets,
                            consistency_target_conf,
                            consistency_keep_mask,
                            consistency_stats,
                        ) = build_region_consistency_queries(
                            base_scores * getattr(args, 'consistency_logit_scale', 10.0),
                            region.cuda(),
                            point_batch_ids,
                            min_region_points=getattr(args, 'consistency_min_region_points', 20),
                            max_regions_per_scene=getattr(args, 'consistency_max_regions', 40),
                            min_region_conf=getattr(args, 'consistency_min_conf', 0.35),
                            min_disagree_ratio=getattr(args, 'consistency_min_disagree', 0.02),
                            point_conf_threshold=getattr(args, 'consistency_point_conf', 0.55),
                            point_entropy_threshold=getattr(args, 'consistency_point_entropy', 0.55),
                        )
                        if consistency_queries.numel() > 0:
                            query_indices = torch.unique(torch.cat([query_indices, consistency_queries], dim=0))
                        refine_mask = refine_mask | consistency_mask
                        keep_mask = keep_mask | consistency_keep_mask
                    else:
                        consistency_stats = {"consistency_regions": 0}
                    if query_indices.numel() == 0 or float(refine_mask.float().mean().item()) < 0.02:
                        fallback_queries, fallback_mask, fallback_stats = build_uncertain_region_queries(
                            base_scores,
                            region.cuda(),
                            point_batch_ids,
                            min_region_points=args.refine_min_region_points,
                            max_queries_per_scene=args.refine_max_queries,
                        )
                        if fallback_queries.numel() > 0:
                            query_indices = torch.unique(torch.cat([query_indices, fallback_queries], dim=0))
                            refine_mask = refine_mask | fallback_mask
                    num_queries = int(query_indices.numel())
                    split_regions = query_stats["split_regions"]
                    consistency_regions = consistency_stats["consistency_regions"]
                else:
                    query_indices, refine_mask, keep_mask, query_stats = build_error_queries(
                        base_scores * args.refine_query_scale,
                        semantic_pseudo,
                        region.cuda(),
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
                    num_queries = query_stats["num_queries"]
                    split_regions = 0
                    consistency_regions = 0
                scores = base_scores.clone()
                no_op_scores = None
                if getattr(args, 'refine_region_accept_gate', False):
                    no_op_scores = apply_refinement_projection(
                        args,
                        base_scores.clone(),
                        base_scores,
                        feats,
                        classifier,
                        region,
                        split_targets if getattr(args, 'refine_split_enable', False) else None,
                    )
                apply_mask = torch.ones_like(refine_mask) if getattr(args, 'refine_apply_all', False) else refine_mask
                for _ in range(max(int(getattr(args, 'refine_rounds', 1)), 1)):
                    delta_scores = refiner(
                        feats,
                        point_coords,
                        point_batch_ids,
                        query_indices,
                        region.cuda(),
                        use_region_branch=getattr(args, 'refine_region_branch', False),
                    )
                    if apply_mask.any():
                        candidate_scores = scores + args.refine_residual_scale * delta_scores
                        if getattr(args, 'refine_gate_enable', False):
                            base_probs = F.softmax(scores.detach(), dim=1)
                            candidate_probs = F.softmax(candidate_scores.detach(), dim=1)
                            base_conf, base_pred_round = base_probs.max(dim=1)
                            top2 = torch.topk(candidate_probs, k=2, dim=1).values
                            candidate_conf, candidate_pred_round = candidate_probs.max(dim=1)
                            candidate_margin = top2[:, 0] - top2[:, 1]
                            changed_round = candidate_pred_round != base_pred_round
                            gate_mask = apply_mask & (
                                ((~changed_round) & (candidate_conf >= base_conf + getattr(args, 'refine_gate_conf_gain', 0.0)))
                                | (
                                    changed_round
                                    & (candidate_conf >= getattr(args, 'refine_gate_min_conf', 0.30))
                                    & (candidate_margin >= getattr(args, 'refine_gate_min_margin', 0.02))
                                )
                            )
                            scores[gate_mask] = candidate_scores[gate_mask]
                        else:
                            scores[apply_mask] = candidate_scores[apply_mask]
                scores = apply_refinement_projection(
                    args,
                    scores,
                    base_scores,
                    feats,
                    classifier,
                    region,
                    split_targets if getattr(args, 'refine_split_enable', False) else None,
                )
                if no_op_scores is not None:
                    scores, accepted_mask = apply_region_accept_gate(args, scores, no_op_scores, region)
                    stats["accepted_points"] = stats.get("accepted_points", 0) + int(accepted_mask.sum().item())
                preds = torch.argmax(scores, dim=1).cpu()
                trusted_mask_cpu = apply_mask.cpu()
                changed = preds != base_preds
                stats["changed_points"] += int(changed.sum().item())
                stats["trusted_points"] += int(trusted_mask_cpu.sum().item())
                stats["keep_points"] += int(keep_mask.cpu().sum().item())
                stats["total_points"] += int(preds.numel())
                stats["queries"] += int(num_queries)
                stats["split_regions"] += int(split_regions)
                stats["consistency_regions"] += int(consistency_regions)
                stats["changed_trusted_points"] += int((changed & trusted_mask_cpu).sum().item())
            elif use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
                #
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

                region_scores = F.linear(F.normalize(region_feats), F.normalize(classifier.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()
                base_preds = preds

            base_preds = base_preds[inverse_map.long()]
            preds = preds[inverse_map.long()]
            valid = labels != args.ignore_label
            all_preds.append(base_preds[valid])
            all_refined_preds.append(preds[valid])
            all_label.append(labels[valid])

    return all_preds, all_refined_preds, all_label, stats


def compute_unsupervised_metrics(all_preds, all_labels, sem_num):
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1)) * 100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    return o_Acc, m_Acc, s, 100 * m_IoU



def eval(epoch, args, test_areas = ['Area_5']):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()

    primitive_centers = cls.weight.data###[300, 128]
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.cpu().numpy())#.astype(np.float64))

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    refiner = None
    if getattr(args, 'refine_enable', False):
        refiner_path = os.path.join(args.save_path, 'refiner_' + str(epoch) + '_checkpoint.pth')
        if not os.path.exists(refiner_path):
            refiner_path = os.path.join(args.save_path, 'ckpts', 'refiner_' + str(epoch) + '_checkpoint.pth')
        if os.path.exists(refiner_path):
            refiner = ErrorQueryRefiner(
                feat_dim=args.feats_dim,
                num_classes=args.semantic_class,
                hidden_dim=getattr(args, 'refine_hidden_dim', 128),
                num_heads=getattr(args, 'refine_num_heads', 4),
                dropout=getattr(args, 'refine_dropout', 0.0),
            ).cuda()
            try:
                refiner.load_state_dict(torch.load(refiner_path, map_location='cpu'), strict=False)
                refiner.eval()
                print('Loaded refiner from {}'.format(refiner_path))
            except RuntimeError as exc:
                print('Refiner checkpoint is incompatible; evaluating without refinement. {}'.format(exc))
                refiner = None
        else:
            print('Refiner checkpoint not found; evaluating without refinement.')

    test_dataset = S3DIStest(args, areas=test_areas)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=4, pin_memory=True)

    preds, refined_preds, labels, refine_stats = eval_once(args, model, test_loader, classifier, primitive_classifier=cls, cluster_pred=cluster_pred, refiner=refiner)
    all_preds = torch.cat(preds).numpy()
    all_refined_preds = torch.cat(refined_preds).numpy()
    all_labels = torch.cat(labels).numpy()

    sem_num = args.semantic_class
    o_Acc, m_Acc, s, m_IoU = compute_unsupervised_metrics(all_preds, all_labels, sem_num)
    refined_o_Acc, refined_m_Acc, refined_s, refined_m_IoU = compute_unsupervised_metrics(all_refined_preds, all_labels, sem_num)

    total_points = max(refine_stats["total_points"], 1)
    trusted_points = max(refine_stats["trusted_points"], 1)
    refine_stats.update({
        "baseline_oAcc": o_Acc,
        "baseline_mAcc": m_Acc,
        "baseline_mIoU": m_IoU,
        "refined_oAcc": refined_o_Acc,
        "refined_mAcc": refined_m_Acc,
        "refined_mIoU": refined_m_IoU,
        "delta_mIoU": refined_m_IoU - m_IoU,
        "changed_ratio": refine_stats["changed_points"] / total_points,
        "trusted_ratio": refine_stats["trusted_points"] / total_points,
        "keep_ratio": refine_stats["keep_points"] / total_points,
        "changed_trusted_ratio": refine_stats["changed_trusted_points"] / trusted_points,
        "split_regions": refine_stats["split_regions"],
        "consistency_regions": refine_stats["consistency_regions"],
        "refined_s": refined_s,
    })
    args.eval_refine_stats = refine_stats

    return o_Acc, m_Acc, s


if __name__ == '__main__':

    args = parse_args()
    epoch = args.eval_epoch
    if epoch.isdigit():
        epoch = int(epoch)
    o_Acc, m_Acc, s = eval(epoch, args)
    print('Epoch: {}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
    stats = getattr(args, 'eval_refine_stats', None)
    if stats and args.refine_enable:
        print(
            'Epoch: {}, Refined oAcc {:.2f}  mAcc {:.2f}  delta_mIoU {:+.2f}  '
            'changed {:.2f}% trusted {:.2f}% keep {:.2f}% changed@trusted {:.2f}% queries {} split_regions {} consistency_regions {}'.format(
                epoch,
                stats['refined_oAcc'],
                stats['refined_mAcc'],
                stats['delta_mIoU'],
                100 * stats['changed_ratio'],
                100 * stats['trusted_ratio'],
                100 * stats.get('keep_ratio', 0.0),
                100 * stats['changed_trusted_ratio'],
                stats['queries'],
                stats.get('split_regions', 0),
                stats.get('consistency_regions', 0),
            ),
            stats['refined_s'],
        )
