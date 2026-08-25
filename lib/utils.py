import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import KMeans
import MinkowskiEngine as ME

from tqdm import tqdm
from .my_utils import VisualizationThreadPool


def enforce_cannot_link(sp_index, region_features, region_sizes, cannot_link_pairs):
    """Prevent verified split siblings from immediately merging into one cluster."""
    if cannot_link_pairs is None or cannot_link_pairs.numel() == 0:
        return sp_index, 0
    sp_index = sp_index.clone()
    features = region_features.detach().cpu().float()
    sizes = region_sizes.detach().cpu().view(-1)
    prevented = 0
    for pair in cannot_link_pairs.detach().cpu().long():
        first, second = int(pair[0].item()), int(pair[1].item())
        if sp_index[first] != sp_index[second]:
            continue
        moving = first if sizes[first] <= sizes[second] else second
        forbidden_cluster = int(sp_index[first].item())
        alternatives = torch.unique(sp_index)
        alternatives = alternatives[alternatives != forbidden_cluster]
        if alternatives.numel() == 0:
            continue
        centers = []
        for cluster_id in alternatives:
            centers.append(features[sp_index == cluster_id].mean(dim=0))
        centers = torch.stack(centers)
        nearest = torch.argmin(torch.norm(centers - features[moving], dim=1))
        sp_index[moving] = alternatives[nearest]
        prevented += 1
    return sp_index, prevented

def get_sp_feature(
    args, loader, model, current_growsp, vis=False,
    learnable_sp=None, semantic_centers=None, superpoint_module=None,
):
    # ``learnable_sp`` is retained for old experiment commands. New code passes
    # the deterministic Stage-3 decomposer through the neutral module name.
    legacy_learnable_module = superpoint_module is None and learnable_sp is not None
    superpoint_module = superpoint_module or learnable_sp
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    model.eval()
    context = []
    if vis:
        vis_pool = VisualizationThreadPool(max_threads=4)
    structure_stats = {
        'scenes': 0,
        'candidate_regions': 0,
        'accepted_splits': 0,
        'supervised_points': 0,
        'valid_points': 0,
        'prevented_remerges': 0,
    }
    if superpoint_module is not None:
        superpoint_module.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()
            cannot_link_pairs = None

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            # feats = F.normalize(feats, dim=-1)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:] * args.voxel_size
            if superpoint_module is not None and semantic_centers is not None:
                semantic_logits = F.linear(F.normalize(feats, dim=1), semantic_centers)
                if legacy_learnable_module:
                    semantic_logits = semantic_logits * getattr(args, 'learnable_sp_query_scale', 10.0)
                structure_output = superpoint_module(
                    feats,
                    pc_xyz,
                    pc_rgb,
                    semantic_logits,
                    region,
                    torch.zeros(region.size(0), dtype=torch.long, device=feats.device),
                    min_region_points=getattr(args, 'learnable_sp_min_region_points', 20),
                    min_child_points=getattr(args, 'learnable_sp_min_child_points', 6),
                    max_regions_per_scene=getattr(args, 'learnable_sp_max_regions', 12),
                    purity_threshold=getattr(args, 'learnable_sp_purity_th', 0.9),
                    entropy_threshold=getattr(args, 'learnable_sp_entropy_th', 0.3),
                    min_child_confidence=getattr(args, 'learnable_sp_child_conf_th', 0.2),
                    min_confidence_gain=getattr(args, 'learnable_sp_conf_gain', 0.01),
                    min_semantic_separation=getattr(args, 'learnable_sp_semantic_sep', 0.15),
                )
                region = structure_output.dynamic_regions.cpu()
                if hasattr(structure_output, 'refined_features'):
                    feats = structure_output.refined_features
                cannot_link_pairs = getattr(structure_output, 'cannot_link_pairs', None)
                structure_stats['scenes'] += 1
                structure_stats['candidate_regions'] += structure_output.stats['selected_regions']
                structure_stats['accepted_splits'] += structure_output.stats['accepted_splits']
                structure_stats['supervised_points'] += int(structure_output.supervision_mask.sum().item())
                structure_stats['valid_points'] += int(region.numel())
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            # region_num为region标签的数目
            # region标签和label标签对应 数量相等 值不等
            # region标签来自于region合并后的重赋值
            # region_corr是 01矩阵 N*M 通过 region_feats为region的特征均值
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num  # 特征原型中心 region_corr对应位置为region 其特征feats
            if current_growsp is not None:
                region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
                region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                    
                # region_sizes = per_region_num.squeeze()  # [num_regions]
                # sp_idx_bf = torch.from_numpy(KMeans(n_clusters=n_segments + 1, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                # sp_idx = masked_kmeans_consensus(region_feats, n_segments, n_rounds=10, mask_prob=0.3)
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                sp_idx, prevented = enforce_cannot_link(
                    sp_idx,
                    region_feats,
                    per_region_num,
                    cannot_link_pairs,
                )
                structure_stats['prevented_remerges'] += prevented
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]  # 每个点的region标签 基于region的
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)  # 可以看出one hot 编码的region标签
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t())/per_neural_region_num  # rgb原型中心
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)

            # coords_xyz = pc_xyz  # 已经是 voxel 尺度的
            # ms_geo = compute_multiscale_geometry(coords_xyz, normals, neural_region)
            # #
            # final_rgb = F.normalize(final_rgb, dim=-1)
            # pfh = F.normalize(pfh, dim=-1)
            # 128 + 3 + 10

            if getattr(args, 'z_enable', False):
                feats = torch.cat((feats, args.c_shape*pfh), dim=-1)
                
                region_z = pc_xyz[:, 2:3]

                if current_growsp is not None:
                    out = F.linear(neural_region_corr.t(), region_z.t())
                    region_z_mean = out / per_neural_region_num
                    region_z_max = out.max(dim=1, keepdim=True)[0]
                    region_z_min = out.min(dim=1, keepdim=True)[0]
                else:
                    out = F.linear(region_corr.t(), region_z.t())
                    region_z_mean = out / per_neural_region_num
                    region_z_max = out.max(dim=1, keepdim=True)[0]
                    region_z_min = out.min(dim=1, keepdim=True)[0]
                region_z = torch.cat((region_z_mean, region_z_max, region_z_min), dim=-1)

                region_z = F.normalize(region_z, dim=-1)
                
                feats = torch.cat((feats, region_z), dim=-1)
            else:
                feats = torch.cat((feats, args.c_rgb*final_rgb, args.c_shape*pfh), dim=-1)

            feats = F.normalize(feats, dim=-1)

            # if args.tcc_enable:
            #     feats = torch.cat((feats, per_neural_region_num), dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)

            if vis:
                context.append((scene_name, gt, raw_region, coords, inverse_map))
                vis_path = '/home/magic/magic/cm/repositories/GrowSP/data/S3DIS/sp_vis'
                vis_pool.submit_task(coords, scene_name, valid_mask, inverse_map, neural_region.numpy(), vis_path)
                vis_pool.clean_up()
            else:
                context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    args.cluster_superpoint_stats = structure_stats
    # Compatibility for existing logging and external scripts.
    args.cluster_learnable_sp_stats = structure_stats
    return point_feats_list, point_labels_list, all_sp_index, context



def get_kittisp_feature(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    model.eval()
    context = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)

            feats = model(in_field)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_remission = features
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_feats = F.normalize(region_feats, dim=-1)
                #
                if region_feats.size(0) < current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            final_remission = F.linear(neural_region_corr.t(), pc_remission.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)
            #
            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask]).unsqueeze(0))

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            feats = torch.cat((feats, args.c_rgb*final_remission, args.c_shape*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return point_feats_list, point_labels_list, all_sp_index, context



def get_pseudo(args, context, cluster_pred, all_sp_index=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder):
        os.makedirs(pseudo_label_folder)
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0
    
    pe_gt_labels = -np.ones_like(cluster_pred).astype(np.int32) # 核心：存储每个超点的真实标签（按超点顺序） 
    sp_gt_labels = []  # 核心：存储每个超点的真实标签（按超点顺序）

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sp_index[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        region_tmp = region[valid_mask]       # 点级：有效点对应的超点ID（场景内局部）
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]
        scene_sp_gt = []
        for local_sp_id in np.unique(region_tmp):
            sp_point_mask = (region_tmp == local_sp_id)
            single_sp_gt = torch.mode(labels_tmp[sp_point_mask]).values
            scene_sp_gt.append(single_sp_gt.item()) 

        sp_gt_labels.extend(scene_sp_gt)

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
                
                if pe_gt_labels[p] == -1:
                    pe_gt_labels[p] = sub_cluster_gt
                else:
                    if pe_gt_labels[p] != sub_cluster_gt:
                        print(f"[Warning] conflict at cluster {p}: {pe_gt_labels[p]} vs {sub_cluster_gt}")

        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)
    # 超点级真实标签转numpy
    sp_gt_labels = np.array(sp_gt_labels, dtype=np.int32)
    
    return all_pseudo, all_gt, all_pseudo_gt, sp_gt_labels, pe_gt_labels


def get_pseudo_kitti(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in tqdm(range(len(context))):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_folder = args.pseudo_label_path + '/' + scene_name[0:3]
        if not os.path.exists(pseudo_label_folder):
            os.makedirs(pseudo_label_folder)

        pseudo_label_file = args.pseudo_label_path + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier


def compute_hist(normal, bins=10, min=-1, max=1):
    ## normal : [N, 3]
    normal = F.normalize(normal)
    relation = torch.mm(normal, normal.t())
    relation = torch.triu(relation, diagonal=0) # top-half matrix
    hist = torch.histc(relation, bins, min, max)
    # hist = torch.histogram(relation, bins, range=(-1, 1))
    hist /= hist.sum()

    return hist
