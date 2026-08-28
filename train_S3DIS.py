import argparse
import time
import os
import re
import numpy as np
import random
from datasets.S3DIS import S3DIStrain, cfl_collate_fn
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18
from models.query_refiner import (
    LegacyLogitResidualRefiner,
    refined_cross_entropy,
    refinement_keep_kl,
    delta_l2,
)
from models.learnable_superpoint import (
    SemanticDifferenceSuperpointLearner,
    verified_region_supervision_loss,
)
from eval_S3DIS import eval
from lib.utils import (
    get_pseudo,
    get_sp_feature,
    get_fixclassifier,
)
from lib.error_query import build_error_queries
from lib.stage3_pipeline import (
    Stage3Config,
    SuperpointDecomposer,
    run_stage3_pipeline,
    stage3_training_losses,
)
from lib.stage2_feature_pipeline import (
    Stage2FeatureConfig,
    Stage2FeatureModule,
    run_stage2_feature_pipeline,
    stage2_feature_losses,
)
from sklearn.cluster import KMeans
import logging
from os.path import join
import warnings
warnings.filterwarnings('ignore')

### model import 
from lib.my_utils import backup_selected, load_resume_checkpoint, save_checkpoints, log_training_info
from lib.my_utils import compute_type1_centers, compute_type2_centers
from lib.my_utils import setup_custom_env, WandbHandler
from lib.my_utils import setup_loss_weight, compute_total_loss
from lib.my_utils import build_model
from lib.my_utils import generate_all_s3dis_clustering_plots

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/S3DIS/input',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/S3DIS/initial_superpoints/',
                        help='initial superpoint path')
    parser.add_argument('--save_path', type=str, default='ckpt/S3DIS/',
                        help='model savepath')
    parser.add_argument('--test_area', type=str, default='Area_5',
                        help='S3DIS held-out area, or comma-separated areas')
    ###
    parser.add_argument('--max_epoch', type=int, nargs=2, default=[500, 800], help='max epoch for non-growing and growing stage')
    parser.add_argument('--max_iter', type=int, nargs='+', default=[10000, 30000], help='max iter for non-growing and growing stage')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD parameters')
    parser.add_argument('--dampening', type=float, default=0.1, help='SGD parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='SGD parameters')
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data in training')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--log-interval', type=int, default=20, help='log interval')
    parser.add_argument('--batch_size', type=int, default=10, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--pseudo_label_path', default='pseudo_label_s3dis/', type=str, help='pseudo label save path')
    parser.add_argument('--ignore_label', type=int, default=12, help='invalid label')
    parser.add_argument('--growsp_start', type=int, default=80, help='the start number of growing superpoint')
    parser.add_argument('--growsp_end', type=int, default=20, help='the end number of grwoing superpoint')
    parser.add_argument('--drop_threshold', type=int, default=10, help='ignore superpoints with few points')
    parser.add_argument('--w_rgb', type=float, default=5/5, help='weight for RGB in merging superpoint')
    parser.add_argument('--w_xyz', type=float, default=1/5, help='weight for XYZ in merging superpoint')
    parser.add_argument('--w_norm', type=float, default=4/5, help='weight for Normal in merging superpoint')
    parser.add_argument('--c_rgb', type=float, default=3, help='weight for RGB in clustering primitives')
    parser.add_argument('--c_shape', type=float, default=3, help='weight for PFH in clustering primitives')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint for resume')
    parser.add_argument('--model', type=str, default='res16fpn18', help='res16fpn18 res16unet14')
    parser.add_argument('--wandb', action='store_true', default=False, help='Enable fixed weight')
    parser.add_argument('--z_enable', action='store_true', default=False, help='Enable fixed weight')
    parser.add_argument('--fixed_weight', action='store_true', default=False, help='Enable fixed weight')
    parser.add_argument('--tcc_enable', action='store_true', default=False, help='Enable TCC')
    parser.add_argument('--region_weight_enable', action='store_true', default=False, help='Enable region weight')
    parser.add_argument('--double_ssl', action='store_true', default=False, help='Enable double SSL')
    parser.add_argument('--plot', action='store_true', default=False, help='Enable double SSL')
    parser.add_argument('--refine_enable', action='store_true', default=False,
                        help='enable the legacy semantic-logit residual experiment')
    parser.add_argument('--refine_lambda', type=float, default=0.3, help='loss weight for query refinement')
    parser.add_argument('--refine_lr', type=float, default=1e-3, help='learning rate for the refinement optimizer')
    parser.add_argument('--refine_weight_decay', type=float, default=1e-4, help='weight decay for the refinement optimizer')
    parser.add_argument('--refine_reset_each_cluster', action='store_true', default=False, help='reset refiner when teacher pseudo labels are regenerated')
    parser.add_argument('--refine_hidden_dim', type=int, default=128, help='hidden dimension for query refinement')
    parser.add_argument('--refine_num_heads', type=int, default=4, help='attention heads for query refinement')
    parser.add_argument('--refine_dropout', type=float, default=0.0, help='dropout for query refinement')
    parser.add_argument('--refine_residual_scale', type=float, default=0.1, help='scale of refinement residual logits')
    parser.add_argument('--refine_keep_lambda', type=float, default=1.0, help='KL weight that keeps stable regions close to baseline logits')
    parser.add_argument('--refine_delta_lambda', type=float, default=0.01, help='L2 weight for refinement residual logits')
    parser.add_argument('--refine_query_scale', type=float, default=10.0, help='logit scale used only for refinement query confidence')
    parser.add_argument('--refine_conf_th', type=float, default=0.7, help='teacher pseudo confidence threshold')
    parser.add_argument('--refine_margin_th', type=float, default=0.2, help='teacher pseudo margin threshold')
    parser.add_argument('--refine_region_purity_th', type=float, default=0.8, help='minimum pseudo purity for trusted regions')
    parser.add_argument('--refine_color_consistency_th', type=float, default=0.35, help='minimum color consistency for refinement candidate regions')
    parser.add_argument('--refine_geometry_consistency_th', type=float, default=0.65, help='minimum geometry compactness for refinement candidate regions')
    parser.add_argument('--refine_min_region_points', type=int, default=10, help='minimum points in a trusted region')
    parser.add_argument('--refine_max_queries', type=int, default=5, help='maximum query anchors per scene')
    parser.add_argument('--refine_teacher_ckpt_dir', type=str, default='', help='directory containing frozen GrowSP model/cls checkpoints')
    parser.add_argument('--refine_teacher_epoch', type=int, default=-1, help='teacher checkpoint epoch; -1 uses the latest epoch with both model and cls checkpoints')
    parser.add_argument('--refine_freeze_backbone', action='store_true', default=False, help='freeze GrowSP backbone and optimize only the refiner')
    parser.add_argument('--refine_teacher_growsp', type=int, default=-1, help='superpoint target used to generate frozen-teacher pseudo labels; -1 uses growsp_end')
    parser.add_argument('--learnable_sp_enable', action='store_true', default=False,
                        help='learn candidate superpoint structure inside unsupervised training')
    parser.add_argument('--learnable_sp_lr', type=float, default=1e-3, help='learning rate for superpoint assignment')
    parser.add_argument('--learnable_sp_weight_decay', type=float, default=1e-4, help='weight decay for superpoint assignment')
    parser.add_argument('--learnable_sp_hidden_dim', type=int, default=64, help='hidden dimension for superpoint assignment')
    parser.add_argument('--learnable_sp_iterations', type=int, default=3, help='soft center update iterations')
    parser.add_argument('--learnable_sp_temperature', type=float, default=0.2, help='soft assignment temperature')
    parser.add_argument('--learnable_sp_query_scale', type=float, default=10.0,
                        help='logit scale used only to discover and verify dynamic regions')
    parser.add_argument('--learnable_sp_structure_lambda', type=float, default=0.1, help='structural objective weight')
    parser.add_argument('--learnable_sp_supervision_lambda', type=float, default=0.0, help='verified region supervision weight')
    parser.add_argument('--learnable_sp_min_region_points', type=int, default=20, help='minimum candidate parent size')
    parser.add_argument('--learnable_sp_min_child_points', type=int, default=6, help='minimum verified child size')
    parser.add_argument('--learnable_sp_max_regions', type=int, default=4, help='maximum candidate parents per scene')
    parser.add_argument('--learnable_sp_purity_th', type=float, default=0.8, help='candidate semantic purity threshold')
    parser.add_argument('--learnable_sp_entropy_th', type=float, default=0.4, help='candidate normalized entropy threshold')
    parser.add_argument('--learnable_sp_child_conf_th', type=float, default=0.5, help='minimum child consensus confidence')
    parser.add_argument('--learnable_sp_conf_gain', type=float, default=0.05, help='minimum child confidence gain')
    parser.add_argument('--learnable_sp_semantic_sep', type=float, default=0.5, help='minimum child semantic separation')
    # Primary paper path: GrowSP Stage 1 -> GrowSP Stage 2 -> our Stage 3.
    parser.add_argument('--stage3_enable', action='store_true', default=False,
                        help='run split + candidate refiner + verifier after GrowSP growing')
    parser.add_argument('--stage3_epochs', type=int, default=20, help='epochs in the third training stage')
    parser.add_argument('--stage3_lr', type=float, default=1e-2, help='backbone learning rate in Stage 3')
    parser.add_argument('--stage3_refiner_lr', type=float, default=1e-4, help='Candidate-based Refiner learning rate')
    parser.add_argument('--stage3_structure_lambda', type=float, default=0.3,
                        help='weight for verified split supervision on the backbone')
    parser.add_argument('--stage3_refiner_lambda', type=float, default=0.3,
                        help='weight for Candidate-based Refiner training')
    parser.add_argument('--stage3_residual_scale', type=float, default=1.0,
                        help='semantic residual scale before conservative verification')
    parser.add_argument('--stage3_max_steps', type=int, default=-1,
                        help='optional batches per epoch for smoke tests; -1 uses all batches')
    parser.add_argument('--start_stage2_from_resume', action='store_true', default=False,
                        help='treat the resumed Stage-1 epoch as the start of GrowSP Stage 2')
    parser.add_argument('--stage2_split_refine_enable', action='store_true', default=False,
                        help='enable split-aware contextual feature refinement during Stage 2')
    parser.add_argument('--stage2_split_start_ratio', type=float, default=0.75,
                        help='fraction of Stage 2 completed before split-aware refinement starts')
    parser.add_argument('--stage2_feature_refiner_lr', type=float, default=1e-2,
                        help='SGD learning rate for the context group inside the unified model')
    parser.add_argument('--stage2_backbone_gradient_scale', type=float, default=0.1)
    parser.add_argument('--stage2_split_lambda', type=float, default=0.2)
    parser.add_argument('--stage2_feature_lambda', type=float, default=0.1)
    parser.add_argument('--stage2_feature_update_lambda', '--stage2_residual_lambda',
                        dest='stage2_feature_update_lambda', type=float, default=0.01,
                        help='regularization weight for accepted feature displacement')
    parser.add_argument('--stage2_refiner_primitive_lambda', type=float, default=0.05)
    parser.add_argument('--stage2_min_region_points', type=int, default=20)
    parser.add_argument('--stage2_min_child_points', type=int, default=8)
    parser.add_argument('--stage2_max_regions', type=int, default=20)
    parser.add_argument('--stage2_purity_th', type=float, default=0.92)
    parser.add_argument('--stage2_entropy_th', type=float, default=0.25)
    parser.add_argument('--stage2_min_split_conf', type=float, default=0.35)
    parser.add_argument('--stage2_verifier_tolerance', type=float, default=0.0)
    parser.add_argument('--stage2_max_feature_update_norm', '--stage2_max_residual_norm',
                        dest='stage2_max_feature_update_norm', type=float, default=1.0)
    parser.add_argument('--stage2_min_structure_gain', type=float, default=0.01)
    parser.add_argument('--stage2_min_child_separation', type=float, default=0.05)
    parser.add_argument('--stage2_min_primitive_gain', type=float, default=0.005)
    parser.add_argument('--stage2_min_primitive_margin', type=float, default=0.01)
    parser.add_argument('--stage2_primitive_top_k', type=int, default=3)
    parser.add_argument('--stage2_primitive_support_tolerance', type=float, default=0.01)
    parser.add_argument('--stage2_max_steps', type=int, default=-1,
                        help='optional batches per Stage-2 epoch for smoke tests; -1 uses all batches')
    parser.add_argument('--stage2_stop_epoch', type=int, default=-1,
                        help='optional absolute epoch for short Stage-2 diagnostics; -1 runs full Stage 2')
    return parser.parse_args()


def parse_test_areas(test_area):
    areas = [area.strip() for area in str(test_area).split(',') if area.strip()]
    if not areas:
        raise ValueError('test_area must contain at least one S3DIS area')
    return areas


def build_training_optimizer(args, model, backbone_lr=None):
    """Build one optimizer for both backbone and integrated feature context."""
    backbone_lr = args.lr if backbone_lr is None else float(backbone_lr)
    if hasattr(model, 'feature_refiner_parameters'):
        parameter_groups = [
            {
                'params': list(model.backbone_parameters()),
                'lr': backbone_lr,
                'poly_schedule': True,
                'name': 'backbone',
            },
            {
                'params': list(model.feature_refiner_parameters()),
                'lr': args.stage2_feature_refiner_lr,
                'poly_schedule': False,
                'name': 'candidate_feature_refiner',
            },
        ]
    else:
        parameter_groups = model.parameters()
    return torch.optim.SGD(
        parameter_groups,
        lr=backbone_lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
    )


def main(args, logger):
    if args.stage3_enable and (args.refine_freeze_backbone or args.refine_teacher_ckpt_dir):
        raise ValueError('Stage 3 jointly trains the backbone and cannot use the frozen-teacher mode.')
    if args.stage3_enable and args.learnable_sp_enable:
        raise ValueError('Choose the Stage-3 decomposer or the legacy learnable superpoint experiment, not both.')
    if args.stage2_split_refine_enable and args.learnable_sp_enable:
        raise ValueError('Stage-2 split refinement and the legacy learnable superpoint module are mutually exclusive.')
    if args.stage2_split_refine_enable and (args.refine_enable or args.stage3_enable):
        raise ValueError(
            'Direct Stage-2 feature refinement cannot be combined with a legacy '
            'semantic-logit residual experiment.'
        )
    if args.stage2_split_refine_enable and (
        args.refine_freeze_backbone or args.refine_teacher_ckpt_dir
    ):
        raise ValueError(
            'Direct Stage-2 feature refinement jointly trains the backbone and '
            'feature Refiner; frozen-teacher mode is unsupported.'
        )
    if args.start_stage2_from_resume and not args.resume:
        raise ValueError('--start_stage2_from_resume requires --resume.')
    if args.refine_teacher_ckpt_dir:
        args.refine_enable = True
        args.refine_freeze_backbone = True

    '''Prepare Data'''
    max_len = max(len(k) for k in vars(args).keys())
    for key, value in vars(args).items():
        logger.info(f"{key:<{max_len}} : {value}")
        
    logger.info("--- cuimo ---")
    logger.info(f"FIXED_WEIGHT(固定grow) set to: {args.fixed_weight}")
    logger.info(f"CC_ENABLE(簇中心矫正) set to: {args.tcc_enable}")
    logger.info(f"REGION_WEGHT_ENABLE(基于区域的loss) set to: {args.region_weight_enable}")
    logger.info(f"DOUBEL_SSL(多模自监督) set to: {args.double_ssl}")
    logger.info(f"REFINE_ENABLE(error-query refinement) set to: {args.refine_enable}")
    logger.info(f"REFINE_FREEZE_BACKBONE set to: {args.refine_freeze_backbone}")
    logger.info(f"LEARNABLE_SP_ENABLE(training-integrated structure) set to: {args.learnable_sp_enable}")
    logger.info(f"STAGE3_ENABLE(split + Candidate-based Refiner + Conservative Verifier) set to: {args.stage3_enable}")
    logger.info(f"STAGE2_SPLIT_REFINE_ENABLE(direct candidate feature refinement) set to: {args.stage2_split_refine_enable}")
    logger.info("------------------------------")
    backup_selected(args)
    all_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
    test_areas = parse_test_areas(args.test_area)
    unknown_areas = sorted(set(test_areas) - set(all_areas))
    if unknown_areas:
        raise ValueError('Unknown S3DIS test_area values: {}'.format(', '.join(unknown_areas)))
    training_areas = sorted(list(set(all_areas) - set(test_areas)))
    logger.info(f"Test Areas: {test_areas}")
    logger.info(f"Training Areas: {training_areas}")

    trainset = S3DIStrain(args, areas=training_areas)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=cfl_collate_fn(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))
    clusterset = S3DIStrain(args, areas=training_areas)
    cluster_loader = DataLoader(clusterset, batch_size=1, collate_fn=cfl_collate_fn(), num_workers=args.cluster_workers, pin_memory=True)

    '''Prepare Model/Optimizer'''
    # model = Res16UNet14(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args)
    model = build_model(args.model, in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args)
    # logger.info(model)
    model = model.cuda()
    candidate_refiner = None
    if args.refine_enable or args.stage3_enable:
        candidate_refiner = LegacyLogitResidualRefiner(
            feat_dim=args.feats_dim,
            num_classes=args.semantic_class,
            hidden_dim=args.refine_hidden_dim,
            num_heads=args.refine_num_heads,
            dropout=args.refine_dropout,
        ).cuda()
        logger.info(candidate_refiner)
    learnable_sp = None
    if args.learnable_sp_enable:
        learnable_sp = SemanticDifferenceSuperpointLearner(
            feat_dim=args.feats_dim,
            num_classes=args.semantic_class,
            hidden_dim=args.learnable_sp_hidden_dim,
            iterations=args.learnable_sp_iterations,
            temperature=args.learnable_sp_temperature,
        ).cuda()
        logger.info(learnable_sp)
    stage2_feature_config = None
    stage2_feature_module = None
    if args.stage2_split_refine_enable:
        if not hasattr(model, 'update_candidate_features'):
            raise TypeError('Stage-2 feature refinement requires the unified model wrapper.')
        stage2_feature_config = Stage2FeatureConfig.from_args(args)
        stage2_feature_module = Stage2FeatureModule(model, stage2_feature_config)
        logger.info(model.feature_refiner)

    optimizer = None
    if not args.refine_freeze_backbone:
        optimizer = build_training_optimizer(args, model)
    refiner_optimizer = None
    if args.refine_enable:
        refiner_optimizer = torch.optim.AdamW(
            candidate_refiner.parameters(), lr=args.refine_lr, weight_decay=args.refine_weight_decay
        )
    learnable_sp_optimizer = None
    if learnable_sp is not None:
        learnable_sp_optimizer = torch.optim.AdamW(
            learnable_sp.parameters(),
            lr=args.learnable_sp_lr,
            weight_decay=args.learnable_sp_weight_decay,
        )
    scheduler = None if optimizer is None else PolyLR(optimizer, max_iter=args.max_iter[0])
    start_epoch, start_grow_epoch, is_Growing = load_resume_checkpoint(
        args, model, optimizer, scheduler, logger, refiner=candidate_refiner,
        learnable_sp=learnable_sp, learnable_sp_optimizer=learnable_sp_optimizer,
    )
    teacher_classifier = None
    if args.refine_teacher_ckpt_dir:
        teacher_classifier, teacher_epoch = load_frozen_teacher(args, model, logger)
        logger.info(f"Frozen teacher loaded from epoch {teacher_epoch}; backbone gradients are disabled.")
    if args.refine_freeze_backbone:
        freeze_backbone(model)
        is_Growing = True
        start_grow_epoch = 0
    if is_Growing and optimizer is not None:
        scheduler = PolyLR(optimizer, max_iter=args.max_iter[1])
        start_epoch, start_grow_epoch, is_Growing = load_resume_checkpoint(
            args, model, optimizer, scheduler, logger, refiner=candidate_refiner,
            learnable_sp=learnable_sp, learnable_sp_optimizer=learnable_sp_optimizer,
        )
    if args.start_stage2_from_resume:
        is_Growing = True
        start_grow_epoch = start_epoch
        optimizer = build_training_optimizer(args, model)
        scheduler = PolyLR(optimizer, max_iter=args.max_iter[1])
        logger.info(
            'Resume epoch {} is used as the explicit Stage-2 growing start.'.format(start_epoch)
        )

    loss = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()
    classifier = None
    primitive_to_semantic = None
    if (args.stage3_enable or args.stage2_split_refine_enable) and args.resume:
        classifier, primitive_to_semantic = restore_structure_state(
            args, model, logger
        )

    '''Train and Cluster'''
    '''Superpoints will not Grow in 1st Stage'''
    stage1_end = 0 if is_Growing else args.max_epoch[0]
    for epoch in range(start_epoch + 1, stage1_end + 1):
        '''Take 10 epochs as a round'''
        if (epoch - 1) % 10 == 0:
            classifier, primitive_to_semantic = cluster(
                args, logger, cluster_loader, model, epoch, start_grow_epoch, is_Growing,
                teacher_classifier=teacher_classifier,
                structure_classifier=classifier, structure_mapping=primitive_to_semantic,
                learnable_sp=learnable_sp,
            )
            refiner_optimizer = maybe_reset_refiner(args, candidate_refiner, logger)
        train(
            train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier, primitive_to_semantic,
            refiner=candidate_refiner if args.refine_enable else None,
            refiner_optimizer=refiner_optimizer, freeze_backbone=args.refine_freeze_backbone,
            learnable_sp=learnable_sp, learnable_sp_optimizer=learnable_sp_optimizer,
        )

        if epoch % 10 == 0:
            save_checkpoints(
                args, epoch, model, optimizer, scheduler, classifier, is_Growing, start_grow_epoch, logger,
                refiner=candidate_refiner if args.refine_enable else None,
                learnable_sp=learnable_sp, learnable_sp_optimizer=learnable_sp_optimizer,
            )
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args, test_areas)
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)
                log_refine_eval_stats(args, logger, epoch)

            iterations = (epoch + 10) * len(train_loader)
            if iterations > args.max_iter[0] or epoch == args.max_epoch[0]:
                start_grow_epoch = epoch
                is_Growing = True
                logger.info('#################################')
                logger.info('### Superpoints Begin Grwoing ###')
                logger.info('#################################')
                if optimizer is not None:
                    optimizer = build_training_optimizer(args, model)
                    scheduler = PolyLR(optimizer, max_iter=args.max_iter[1])
                break

    '''Superpoints will grow in 2nd Stage'''
    current_epoch = max(start_epoch, start_grow_epoch)
    stage2_end = start_grow_epoch + args.max_epoch[1]
    stage2_loop_end = stage2_end
    if args.stage2_stop_epoch > 0:
        stage2_loop_end = min(stage2_loop_end, args.stage2_stop_epoch)
    split_refine_was_active = bool(
        args.stage2_split_refine_enable
        and getattr(args, 'resume_has_feature_refiner', False)
        and getattr(args, 'resume_training_stage', '') in (
            'stage2_split_feature_refiner',
            'stage2_direct_feature_refiner',
        )
    )
    for epoch in range(current_epoch + 1, stage2_loop_end + 1):
        stage2_progress = (epoch - start_grow_epoch) / max(float(args.max_epoch[1]), 1.0)
        split_refine_active = (
            args.stage2_split_refine_enable
            and stage2_progress >= args.stage2_split_start_ratio
        )
        '''Take 10 epochs as a round'''
        first_resumed_stage2_epoch = (
            args.start_stage2_from_resume and epoch == current_epoch + 1
        )
        split_refine_activated = split_refine_active and not split_refine_was_active
        if (epoch - 1) % 10 == 0 or first_resumed_stage2_epoch or split_refine_activated:
            classifier, primitive_to_semantic = cluster(
                args, logger, cluster_loader, model, epoch, start_grow_epoch, is_Growing,
                teacher_classifier=teacher_classifier,
                structure_classifier=classifier, structure_mapping=primitive_to_semantic,
                learnable_sp=learnable_sp,
                superpoint_module=stage2_feature_module if split_refine_active else None,
            )
            refiner_optimizer = maybe_reset_refiner(args, candidate_refiner, logger)
        if split_refine_active:
            train_stage2_feature_model(
                train_loader, logger, model, optimizer, scheduler, classifier,
                primitive_to_semantic, stage2_feature_config, epoch,
            )
        else:
            train(
                train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier, primitive_to_semantic,
                refiner=candidate_refiner if args.refine_enable else None,
                refiner_optimizer=refiner_optimizer, freeze_backbone=args.refine_freeze_backbone,
                learnable_sp=learnable_sp, learnable_sp_optimizer=learnable_sp_optimizer,
            )

        if epoch % 10 == 0 or epoch == stage2_loop_end:
            args.training_stage = (
                'stage2_direct_feature_refiner'
                if split_refine_active else 'growsp_stage2'
            )
            save_checkpoints(
                args, epoch, model, optimizer, scheduler, classifier, is_Growing, start_grow_epoch, logger,
                refiner=candidate_refiner if args.refine_enable else None,
                learnable_sp=learnable_sp, learnable_sp_optimizer=learnable_sp_optimizer,
            )
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args, test_areas)
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)
                log_refine_eval_stats(args, logger, epoch)
        split_refine_was_active = split_refine_active

    if stage2_loop_end < stage2_end:
        logger.info(
            'Stopped Stage 2 diagnostic at epoch %d (full Stage 2 ends at %d).',
            stage2_loop_end, stage2_end,
        )
        return

    # Stage 3 starts only after GrowSP has completed progressive growing. The
    # decomposed regions participate in feature aggregation and pseudo-label
    # regeneration; verified candidate targets then update both network parts.
    if args.stage3_enable:
        logger.info('############################################################')
        logger.info('### Stage 3: Split + Candidate Refiner + Conservative Verify')
        logger.info('############################################################')
        stage3_config = Stage3Config(residual_scale=args.stage3_residual_scale)
        superpoint_decomposer = SuperpointDecomposer(stage3_config)
        optimizer = build_training_optimizer(args, model, backbone_lr=args.stage3_lr)
        refiner_optimizer = torch.optim.AdamW(
            candidate_refiner.parameters(), lr=args.stage3_refiner_lr,
            weight_decay=args.refine_weight_decay,
        )
        scheduler = PolyLR(
            optimizer,
            max_iter=max(args.stage3_epochs * len(train_loader), 1),
        )
        stage3_end = stage2_end + args.stage3_epochs
        stage3_start = max(start_epoch, stage2_end)
        for epoch in range(stage3_start + 1, stage3_end + 1):
            if classifier is None or (epoch - stage2_end - 1) % 10 == 0:
                classifier, primitive_to_semantic = cluster(
                    args, logger, cluster_loader, model, epoch, start_grow_epoch, True,
                    structure_classifier=classifier,
                    structure_mapping=primitive_to_semantic,
                    superpoint_module=superpoint_decomposer,
                )
            train_stage3(
                train_loader,
                logger,
                model,
                optimizer,
                scheduler,
                classifier,
                primitive_to_semantic,
                candidate_refiner,
                refiner_optimizer,
                stage3_config,
                epoch,
            )

            if epoch % 10 == 0 or epoch == stage3_end:
                args.training_stage = 'stage3'
                save_checkpoints(
                    args, epoch, model, optimizer, scheduler, classifier, True,
                    start_grow_epoch, logger, refiner=candidate_refiner,
                    refiner_optimizer=refiner_optimizer,
                )
                with torch.no_grad():
                    o_Acc, m_Acc, s = eval(epoch, args, test_areas)
                    logger.info(
                        'Stage 3 Epoch: {:02d}, oAcc {:.2f} mAcc {:.2f} IoUs'.format(
                            epoch, o_Acc, m_Acc
                        ) + s
                    )


def resolve_teacher_epoch(ckpt_dir, requested_epoch):
    if requested_epoch >= 0:
        model_path = os.path.join(ckpt_dir, f'model_{requested_epoch}_checkpoint.pth')
        cls_path = os.path.join(ckpt_dir, f'cls_{requested_epoch}_checkpoint.pth')
        if not os.path.exists(model_path) or not os.path.exists(cls_path):
            raise FileNotFoundError(f"Missing model/cls checkpoint pair for epoch {requested_epoch} in {ckpt_dir}")
        return requested_epoch

    model_epochs = set()
    cls_epochs = set()
    pattern = re.compile(r'^(model|cls)_(\d+)_checkpoint\.pth$')
    for filename in os.listdir(ckpt_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        target = model_epochs if match.group(1) == 'model' else cls_epochs
        target.add(int(match.group(2)))
    shared_epochs = sorted(model_epochs & cls_epochs)
    if not shared_epochs:
        raise FileNotFoundError(f"No paired model/cls checkpoints found in {ckpt_dir}")
    return shared_epochs[-1]


def _extract_state_dict(checkpoint, key):
    if isinstance(checkpoint, dict) and key in checkpoint:
        return checkpoint[key]
    return checkpoint


def restore_structure_state(args, model, logger):
    """Restore the classifier needed by the first structure-refinement round."""
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint.get('classifier_state_dict')
    if state is None:
        logger.info('Resume checkpoint has no classifier; structure refinement will initialize it by clustering.')
        return None, None
    classifier = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False)
    classifier.load_state_dict(state)
    classifier = classifier.to(next(model.parameters()).device)
    mapping = KMeans(
        n_clusters=args.semantic_class,
        n_init=10,
        random_state=0,
        n_jobs=5,
    ).fit_predict(classifier.weight.detach().cpu().numpy())
    logger.info('Restored classifier and semantic mapping for structure refinement.')
    return classifier, torch.from_numpy(mapping).long()


def load_frozen_teacher(args, model, logger):
    teacher_epoch = resolve_teacher_epoch(args.refine_teacher_ckpt_dir, args.refine_teacher_epoch)
    model_path = os.path.join(args.refine_teacher_ckpt_dir, f'model_{teacher_epoch}_checkpoint.pth')
    cls_path = os.path.join(args.refine_teacher_ckpt_dir, f'cls_{teacher_epoch}_checkpoint.pth')

    model_state = _extract_state_dict(torch.load(model_path, map_location='cpu'), 'model_state_dict')
    model.load_state_dict(model_state)

    classifier = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False)
    cls_state = _extract_state_dict(torch.load(cls_path, map_location='cpu'), 'classifier_state_dict')
    classifier.load_state_dict(cls_state)
    classifier.weight.requires_grad_(False)
    classifier = classifier.to(next(model.parameters()).device)

    logger.info(f"Loaded frozen GrowSP model from {model_path}")
    logger.info(f"Loaded frozen GrowSP classifier from {cls_path}")
    return classifier, teacher_epoch


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def cluster(
    args, logger, cluster_loader, model, epoch, start_grow_epoch=None, is_Growing=False,
    teacher_classifier=None, structure_classifier=None, structure_mapping=None,
    learnable_sp=None, superpoint_module=None,
):
    time_start = time.time()
    cluster_loader.dataset.mode = 'cluster'

    current_growsp = None
    if teacher_classifier is not None and args.refine_freeze_backbone:
        current_growsp = args.growsp_end if args.refine_teacher_growsp < 0 else args.refine_teacher_growsp
        logger.info('Epoch: {}, Frozen teacher pseudo labels use Superpoints Grow to {}'.format(epoch, current_growsp))
    elif is_Growing:
        current_growsp = int(args.growsp_start - ((epoch - start_grow_epoch)/args.max_epoch[1])*(args.growsp_start - args.growsp_end))
        if current_growsp < args.growsp_end:
            current_growsp = args.growsp_end
        if getattr(args, 'fixed_weight', False):
            current_growsp = 59
        logger.info('Epoch: {}, Superpoints Grow to {}'.format(epoch, current_growsp))

    '''Extract Superpoints Feature'''
    active_superpoint_module = superpoint_module or learnable_sp
    structure_centers = None
    if active_superpoint_module is not None and structure_classifier is not None and structure_mapping is not None:
        structure_centers = build_semantic_classifier(
            structure_classifier, structure_mapping, args.semantic_class
        )
        if hasattr(active_superpoint_module, 'set_reference_primitives'):
            active_superpoint_module.set_reference_primitives(
                structure_classifier.weight, structure_mapping
            )
    feats, labels, sp_index, context = get_sp_feature(
        args,
        cluster_loader,
        model,
        current_growsp,
        superpoint_module=active_superpoint_module,
        semantic_centers=structure_centers,
    )
    structure_stats = getattr(args, 'cluster_superpoint_stats', None)
    if structure_stats and structure_stats['scenes'] > 0:
        coverage = structure_stats['supervised_points'] / max(structure_stats['valid_points'], 1)
        feature_coverage = structure_stats['feature_updated_points'] / max(
            structure_stats['valid_points'], 1
        )
        logger.info(
            'Epoch: {}, grow-then-split clustering: scenes {}, candidates {}, proposals {}, '
            'decomposition accepted {}, rejected {} (primitive {}), coverage {:.3f}%, '
            'feature updates accepted {}, rejected {} (score {}, norm {}), update coverage {:.3f}%'.format(
                epoch,
                structure_stats['scenes'],
                structure_stats['candidate_regions'],
                structure_stats['proposed_splits'],
                structure_stats['accepted_splits'],
                structure_stats['rejected_splits'],
                structure_stats['primitive_rejections'],
                100.0 * coverage,
                structure_stats['feature_updates_accepted'],
                structure_stats['feature_updates_rejected'],
                structure_stats['feature_score_rejections'],
                structure_stats['feature_norm_rejections'],
                100.0 * feature_coverage,
            )
        )
    sp_feats = torch.cat(feats, dim=0)### will do Kmeans with geometric distance
    sp_feats_rgb = sp_feats[:, args.feats_dim:args.feats_dim+3]
    if teacher_classifier is not None:
        neural_sp_feats = F.normalize(sp_feats[:, 0:args.feats_dim], dim=1)
        teacher_centers = F.normalize(teacher_classifier.weight.detach().cpu(), dim=1)
        primitive_labels = F.linear(neural_sp_feats, teacher_centers).argmax(dim=1).cpu().numpy()
        classifier = teacher_classifier
        logger.info('Epoch: {}, pseudo labels assigned by frozen teacher classifier.'.format(epoch))
    else:
        # primitive_labels = masked_kmeans_consensus(sp_feats, args.primitive_num, n_rounds=5, mask_prob=0.3)
        primitive_labels = KMeans(n_clusters=args.primitive_num, n_init=5, random_state=0, n_jobs=5).fit_predict(sp_feats.numpy())
    sp_feats_rgb = sp_feats[:, args.feats_dim:args.feats_dim+3]
    sp_feats = sp_feats[:,0:args.feats_dim]### drop geometric feature

    '''Compute Primitive Centers'''
    if teacher_classifier is not None:
        primitive_centers = teacher_classifier.weight.detach().cpu()
    else:
        primitive_centers = torch.zeros((args.primitive_num, args.feats_dim))
        if getattr(args, 'tcc_enable', False):
            primitive_centers = compute_type1_centers(sp_feats, primitive_labels, primitive_centers, args, logger, sp_feats_rgb)
        else:
            for cluster_idx in range(args.primitive_num):
                indices = primitive_labels == cluster_idx
                cluster_avg = sp_feats[indices].mean(0, keepdims=True)
                primitive_centers[cluster_idx] = cluster_avg
        primitive_centers = F.normalize(primitive_centers, dim=1)
        classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.primitive_num, centroids=primitive_centers)
    sp_feats_np = sp_feats.cpu().numpy()
    primitive_centers_np = primitive_centers.cpu().numpy()

    '''Compute and Save Pseudo Labels'''
    all_pseudo, all_gt, all_pseudo_gt, sp_gt_labels, pe_gt_labels = get_pseudo(
        args,
        context,
        primitive_labels,
        sp_index,
    )
    logger.info('labelled points ratio %.2f clustering time: %.2fs', (all_pseudo!=-1).sum()/all_pseudo.shape[0], time.time() - time_start)
    if (pe_gt_labels < 0).any():
        print("存在未赋值标签")
    if args.plot:
        generate_all_s3dis_clustering_plots(sp_feats_np, pe_gt_labels, primitive_centers_np, args, logger, epoch)

    '''Check Superpoint/Primitive Acc in Training'''
    sem_num = args.semantic_class
    mask = (all_pseudo_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Epoch: {}, Superpoints oAcc {:.2f} IoUs'.format(epoch, o_Acc) + s)

    pseudo_class2gt = -np.ones_like(all_gt)
    for i in range(args.primitive_num):
        mask = all_pseudo==i
        if not mask.any():
            continue
        pseudo_class2gt[mask] = torch.mode(torch.from_numpy(all_gt[mask])).values
    mask = (pseudo_class2gt!=-1)&(all_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + pseudo_class2gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    primitive_to_semantic = KMeans(
        n_clusters=args.semantic_class,
        n_init=10,
        random_state=0,
        n_jobs=5,
    ).fit_predict(classifier.weight.detach().cpu().numpy())

    logger.info('Epoch: {}, Primitives oAcc {:.2f} IoUs'.format(epoch, o_Acc) + s)
    return classifier.to(next(model.parameters()).device), torch.from_numpy(primitive_to_semantic).long()


def maybe_reset_refiner(args, refiner, logger):
    if refiner is None:
        return None
    if getattr(args, 'refine_reset_each_cluster', False):
        refiner.reset_parameters()
        logger.info('Refiner reset after teacher pseudo label update.')
    return torch.optim.AdamW(refiner.parameters(), lr=args.refine_lr, weight_decay=args.refine_weight_decay)


def log_refine_eval_stats(args, logger, epoch):
    stats = getattr(args, 'eval_refine_stats', None)
    refinement_enabled = (
        getattr(args, 'refine_enable', False)
        or getattr(args, 'stage3_enable', False)
        or getattr(args, 'stage2_split_refine_enable', False)
    )
    if not stats or not refinement_enabled:
        return
    logger.info(
        'Epoch: {:02d}, Refined oAcc {:.2f}  mAcc {:.2f}  delta_mIoU {:+.2f}  '
        'changed {:.2f}% trusted {:.2f}% keep {:.2f}% changed@trusted {:.2f}% queries {}'.format(
            epoch,
            stats['refined_oAcc'],
            stats['refined_mAcc'],
            stats['delta_mIoU'],
            100 * stats['changed_ratio'],
            100 * stats['trusted_ratio'],
            100 * stats.get('keep_ratio', 0.0),
            100 * stats['changed_trusted_ratio'],
            stats['queries'],
        ) + stats['refined_s']
    )
    if getattr(args, 'stage2_split_refine_enable', False):
        logger.info(
            'Epoch: {:02d}, direct feature model decomposition accepted {}, '
            'feature updates accepted {}, rejected {}, accepted points {}'.format(
                epoch,
                stats.get('decomposition_accepted_regions', 0),
                stats.get('feature_updates_accepted', 0),
                stats.get('feature_updates_rejected', 0),
                stats.get('accepted_points', 0),
            )
        )


def build_semantic_classifier(classifier, primitive_to_semantic, semantic_class):
    device = classifier.weight.device
    primitive_to_semantic = primitive_to_semantic.to(device)
    semantic_centers = classifier.weight.new_zeros((semantic_class, classifier.weight.size(1)))
    for semantic_id in range(semantic_class):
        mask = primitive_to_semantic == semantic_id
        if mask.any():
            semantic_centers[semantic_id] = classifier.weight[mask].mean(dim=0)
    return F.normalize(semantic_centers, dim=1)


def primitive_targets_to_semantic(pseudo_labels, primitive_to_semantic, ignore_index=-1):
    device = pseudo_labels.device
    primitive_to_semantic = primitive_to_semantic.to(device)
    semantic_targets = torch.full_like(pseudo_labels.long(), ignore_index)
    valid = (pseudo_labels >= 0) & (pseudo_labels < primitive_to_semantic.numel())
    semantic_targets[valid] = primitive_to_semantic[pseudo_labels[valid].long()]
    return semantic_targets


def smooth_targets_by_region(targets, regions, refine_mask, ignore_index=-1):
    smoothed = targets.clone()
    valid = refine_mask & (targets != ignore_index)
    if valid.sum() == 0:
        return smoothed
    for region_id in torch.unique(regions[valid]):
        if region_id.item() == -1:
            continue
        mask = valid & (regions == region_id)
        if mask.sum() == 0:
            continue
        labels, counts = torch.unique(targets[mask], return_counts=True)
        smoothed[mask] = labels[torch.argmax(counts)]
    return smoothed


def train(
    train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier, primitive_to_semantic,
    refiner=None, refiner_optimizer=None, freeze_backbone=False,
    learnable_sp=None, learnable_sp_optimizer=None,
):
    train_loader.dataset.mode = 'train'
    if freeze_backbone:
        model.eval()
    else:
        model.train()
    if refiner is not None:
        refiner.train()
    if learnable_sp is not None:
        learnable_sp.train()
    loss_display = 0
    time_curr = time.time()

    losses_display = {}
    loss = setup_loss_weight(args, loss)
    semantic_centers = None
    if refiner is not None or learnable_sp is not None:
        semantic_centers = build_semantic_classifier(classifier, primitive_to_semantic, args.semantic_class)
    for batch_idx, data in enumerate(train_loader):
        losses = {}
        iteration = (epoch - 1) * len(train_loader) + batch_idx+1

        coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

        in_field = ME.TensorField(features, coords, device=0)
        if freeze_backbone:
            with torch.no_grad():
                model_out = model(in_field)
        else:
            model_out = model(in_field)
        feats, ssl_loss = model_out if isinstance(model_out, tuple) else (model_out, None)

        feats = feats[inds.long()]
        feats = F.normalize(feats, dim=-1)
        #
        pseudo_labels_comp = pseudo_labels.long().cuda()
        logits = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        loss_sem = loss(logits * 3, pseudo_labels_comp).mean()

        point_coords = None
        point_batch_ids = None
        point_colors = None
        point_regions = region.squeeze(-1).long().cuda()
        semantic_targets = None
        semantic_logits = None
        dynamic_regions = point_regions
        learnable_sp_output = None
        if refiner is not None or learnable_sp is not None:
            point_coords = coords[inds.long(), 1:].float().cuda()
            point_batch_ids = coords[inds.long(), 0].long().cuda()
            point_colors = features[inds.long(), :3].float().cuda()
            semantic_targets = primitive_targets_to_semantic(pseudo_labels_comp, primitive_to_semantic)
            semantic_logits = F.linear(F.normalize(feats), semantic_centers)

        if learnable_sp is not None:
            learnable_sp_output = learnable_sp(
                feats.detach(),
                point_coords,
                point_colors,
                semantic_logits * args.learnable_sp_query_scale,
                point_regions,
                point_batch_ids,
                min_region_points=args.learnable_sp_min_region_points,
                min_child_points=args.learnable_sp_min_child_points,
                max_regions_per_scene=args.learnable_sp_max_regions,
                purity_threshold=args.learnable_sp_purity_th,
                entropy_threshold=args.learnable_sp_entropy_th,
                min_child_confidence=args.learnable_sp_child_conf_th,
                min_confidence_gain=args.learnable_sp_conf_gain,
                min_semantic_separation=args.learnable_sp_semantic_sep,
            )
            dynamic_regions = learnable_sp_output.dynamic_regions
            loss_sp_supervision = verified_region_supervision_loss(semantic_logits * 3, learnable_sp_output)
            losses['loss_learnable_sp_structure'] = (
                args.learnable_sp_structure_lambda * learnable_sp_output.structure_loss
            )
            losses['loss_learnable_sp_supervision'] = (
                args.learnable_sp_supervision_lambda * loss_sp_supervision
            )
            losses['learnable_sp_feature'] = learnable_sp_output.feature_loss
            losses['learnable_sp_geometry'] = learnable_sp_output.geometry_loss
            losses['learnable_sp_semantic'] = learnable_sp_output.semantic_loss
            losses['learnable_sp_candidates'] = logits.new_tensor(
                float(learnable_sp_output.stats['selected_regions'])
            )
            losses['learnable_sp_accepted'] = logits.new_tensor(
                float(learnable_sp_output.stats['accepted_splits'])
            )
            losses['learnable_sp_supervised'] = logits.new_tensor(
                learnable_sp_output.stats['supervised_ratio']
            )
            if learnable_sp_output.supervision_mask.any():
                semantic_targets = semantic_targets.clone()
                semantic_targets[learnable_sp_output.supervision_mask] = (
                    learnable_sp_output.supervision_targets[learnable_sp_output.supervision_mask]
                )

        loss_refine = None
        if refiner is not None:
            logits_for_refine = semantic_logits.detach()
            feats_for_refine = feats.detach()
            query_indices, refine_mask, keep_mask, query_stats = build_error_queries(
                logits_for_refine * args.refine_query_scale,
                semantic_targets,
                dynamic_regions,
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
            legacy_logit_update = refiner(
                feats_for_refine, point_coords, point_batch_ids, query_indices
            )
            legacy_candidate_logits = (
                logits_for_refine
                + args.refine_residual_scale * legacy_logit_update
            )
            refine_targets = smooth_targets_by_region(semantic_targets, dynamic_regions, refine_mask)
            loss_refine_ce = refined_cross_entropy(
                legacy_candidate_logits * 3, refine_targets, refine_mask
            )
            loss_keep = refinement_keep_kl(
                legacy_candidate_logits * 3, logits_for_refine * 3, keep_mask
            )
            loss_update = delta_l2(
                legacy_logit_update, refine_mask | keep_mask
            )
            loss_refine = (
                loss_refine_ce
                + args.refine_keep_lambda * loss_keep
                + args.refine_delta_lambda * loss_update
            )
            losses['loss_refine'] = args.refine_lambda * loss_refine
            losses['refine_ce'] = loss_refine_ce
            losses['refine_keep_loss'] = args.refine_keep_lambda * loss_keep
            losses['refine_delta_loss'] = args.refine_delta_lambda * loss_update
            losses['refine_trusted'] = logits.new_tensor(query_stats['trusted_ratio'])
            losses['refine_keep'] = logits.new_tensor(query_stats['keep_ratio'])
            losses['refine_candidate'] = logits.new_tensor(query_stats['candidate_ratio'])
            losses['refine_point_high_conf'] = logits.new_tensor(query_stats['point_high_conf_ratio'])
            losses['refine_pred_match'] = logits.new_tensor(query_stats['pred_match_ratio'])
            losses['refine_regions'] = logits.new_tensor(float(query_stats['regions_suspect']))
            losses['refine_queries'] = logits.new_tensor(float(query_stats['num_queries']))
        # 
        if getattr(args, 'loss_opti', False):
            logits = F.linear(F.normalize(feats, p=2, dim=-1), F.normalize(classifier.weight, p=2, dim=-1))
            loss_sem = loss(logits * 10, pseudo_labels_comp).mean()

            from lib.my_utils import spatial_consistency_loss
            sp_centroids = coords[inds.long(), 1:].float().cuda()
            loss_smooth = spatial_consistency_loss(logits * 10, sp_centroids, k=5, temperature=1.0)
            lambda_smooth = 1.0 
            losses['loss_smooth'] = lambda_smooth * loss_smooth
            losses = compute_total_loss(args, ssl_loss, losses)

        loss_display += loss_sem.item()
        loss_all = loss_sem
        for k, v in losses.items():
            if k.startswith('loss_') and k != 'loss_refine':
                loss_all += v
            if k in losses_display:
                losses_display[k] += losses[k].item()
            else:
                losses_display[k] = losses[k].item()

        if optimizer is not None:
            optimizer.zero_grad()
        if learnable_sp_optimizer is not None:
            learnable_sp_optimizer.zero_grad()
        if (not freeze_backbone and optimizer is not None) or learnable_sp_optimizer is not None:
            loss_all.backward() 
        if not freeze_backbone and optimizer is not None:
            optimizer.step()
        if learnable_sp_optimizer is not None:
            learnable_sp_optimizer.step()
        if refiner is not None and refiner_optimizer is not None and loss_refine is not None:
            refiner_optimizer.zero_grad()
            (args.refine_lambda * loss_refine).backward()
            refiner_optimizer.step()
        if scheduler is not None:
            scheduler.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if (batch_idx+1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            log_training_info(
                args, logger, epoch, batch_idx+1, len(train_loader), 
                iteration, loss_display, losses_display, 
                get_current_lr(scheduler, refiner_optimizer), time_used
            )
            time_curr = time.time()
            loss_display = 0
            losses_display.clear()


def train_stage2_feature_model(
    train_loader,
    logger,
    model,
    optimizer,
    scheduler,
    classifier,
    primitive_to_semantic,
    feature_config,
    epoch,
):
    """Train GrowSP directly on verified candidate-conditioned features."""
    train_loader.dataset.mode = 'train'
    model.train()
    semantic_centers = build_semantic_classifier(
        classifier, primitive_to_semantic, args.semantic_class
    ).detach()
    running = {
        'total': 0.0,
        'primitive': 0.0,
        'refiner_primitive': 0.0,
        'split': 0.0,
        'feature': 0.0,
        'accepted': 0,
        'feature_updated': 0,
        'feature_score_rejected': 0,
        'feature_norm_rejected': 0,
        'candidates': 0,
        'steps': 0,
    }

    for batch_idx, data in enumerate(train_loader):
        if args.stage2_max_steps > 0 and batch_idx >= args.stage2_max_steps:
            break
        coords, features, _, _, _, pseudo_labels, inds, region, _ = data
        in_field = ME.TensorField(features, coords, device=0)
        model_out = model(in_field)
        point_features = model_out[0] if isinstance(model_out, tuple) else model_out
        point_features = F.normalize(point_features[inds.long()], dim=1)
        pseudo_labels = pseudo_labels.long().cuda()
        point_coords = coords[inds.long(), 1:].float().cuda()
        point_batch_ids = coords[inds.long(), 0].long().cuda()
        point_colors = features[inds.long(), :3].float().cuda()
        point_regions = region.view(-1).long().cuda()
        semantic_logits = F.linear(point_features, semantic_centers)

        output = run_stage2_feature_pipeline(
            feature_config,
            model,
            point_features,
            point_coords,
            point_colors,
            semantic_logits,
            point_regions,
            point_batch_ids,
            classifier.weight,
            primitive_to_semantic,
        )
        # The main GrowSP objective consumes verified features. Accepted
        # candidates therefore train the feature Refiner and the backbone in the
        # same primitive classification path; rejected points remain unchanged.
        primitive_logits = F.linear(
            output.verified_features, F.normalize(classifier.weight)
        )
        primitive_loss = F.cross_entropy(
            primitive_logits * 3, pseudo_labels, ignore_index=-1
        )
        feature_losses = stage2_feature_losses(
            output,
            semantic_centers,
            classifier.weight,
            primitive_to_semantic,
        )
        total_loss = (
            primitive_loss
            + args.stage2_refiner_primitive_lambda * feature_losses['primitive']
            + args.stage2_split_lambda * feature_losses['semantic']
            + args.stage2_feature_lambda * feature_losses['structure']
            + args.stage2_feature_update_lambda * feature_losses['feature_update']
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        running['total'] += float(total_loss.detach().item())
        running['primitive'] += float(primitive_loss.detach().item())
        running['refiner_primitive'] += float(feature_losses['primitive'].detach().item())
        running['split'] += float(feature_losses['semantic'].detach().item())
        running['feature'] += float(feature_losses['structure'].detach().item())
        running['accepted'] += int(output.decomposition_mask.sum().item())
        running['feature_updated'] += int(output.feature_accept_mask.sum().item())
        running['feature_score_rejected'] += int(
            output.stats['feature_score_rejections']
        )
        running['feature_norm_rejected'] += int(
            output.stats['feature_norm_rejections']
        )
        running['candidates'] += int(output.candidate_mask.sum().item())
        running['steps'] += 1

        if (batch_idx + 1) % args.log_interval == 0:
            steps = max(running['steps'], 1)
            candidates = max(running['candidates'], 1)
            logger.info(
                'Stage 2 Direct-Feature Epoch {:02d} [{}/{}] loss {:.4f} '
                'primitive {:.4f} candidate_primitive {:.4f} split {:.4f} feature {:.4f} '
                'decomposition {:.2f}% feature_update {:.2f}% rejected(score/norm) {}/{}'.format(
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    running['total'] / steps,
                    running['primitive'] / steps,
                    running['refiner_primitive'] / steps,
                    running['split'] / steps,
                    running['feature'] / steps,
                    100.0 * running['accepted'] / candidates,
                    100.0 * running['feature_updated'] / candidates,
                    running['feature_score_rejected'],
                    running['feature_norm_rejected'],
                )
            )
            for key in running:
                running[key] = (
                    0.0
                    if key not in (
                        'accepted', 'feature_updated', 'feature_score_rejected',
                        'feature_norm_rejected', 'candidates', 'steps',
                    )
                    else 0
                )


def train_stage3(
    train_loader,
    logger,
    model,
    optimizer,
    scheduler,
    classifier,
    primitive_to_semantic,
    candidate_refiner,
    refiner_optimizer,
    stage3_config,
    epoch,
):
    """Jointly train GrowSP and the candidate refiner after superpoint growing."""
    train_loader.dataset.mode = 'train'
    model.train()
    candidate_refiner.train()
    semantic_centers = build_semantic_classifier(
        classifier, primitive_to_semantic, args.semantic_class
    ).detach()
    running = {
        'total': 0.0,
        'primitive': 0.0,
        'split': 0.0,
        'refiner': 0.0,
        'candidate_points': 0,
        'accepted_points': 0,
        'steps': 0,
    }

    for batch_idx, data in enumerate(train_loader):
        if args.stage3_max_steps > 0 and batch_idx >= args.stage3_max_steps:
            break
        coords, features, _, _, _, pseudo_labels, inds, region, _ = data
        in_field = ME.TensorField(features, coords, device=0)
        model_out = model(in_field)
        feats = model_out[0] if isinstance(model_out, tuple) else model_out
        feats = F.normalize(feats[inds.long()], dim=-1)

        pseudo_labels = pseudo_labels.long().cuda()
        primitive_logits = F.linear(feats, F.normalize(classifier.weight))
        primitive_loss = F.cross_entropy(
            primitive_logits * 3, pseudo_labels, ignore_index=-1
        )
        semantic_logits = F.linear(feats, semantic_centers)
        point_coords = coords[inds.long(), 1:].float().cuda()
        point_batch_ids = coords[inds.long(), 0].long().cuda()
        point_colors = features[inds.long(), :3].float().cuda()
        point_regions = region.view(-1).long().cuda()

        output = run_stage3_pipeline(
            stage3_config,
            candidate_refiner,
            feats,
            point_coords,
            point_colors,
            semantic_logits,
            point_regions,
            point_batch_ids,
        )
        stage3_losses = stage3_training_losses(
            stage3_config,
            output,
            keep_weight=args.refine_keep_lambda,
            residual_weight=args.refine_delta_lambda,
        )
        total_loss = (
            primitive_loss
            + args.stage3_structure_lambda * stage3_losses['backbone']
            + args.stage3_refiner_lambda * stage3_losses['refiner']
        )

        optimizer.zero_grad()
        refiner_optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        refiner_optimizer.step()
        scheduler.step()

        verifier_stats = output.verification.statistics
        running['total'] += float(total_loss.detach().item())
        running['primitive'] += float(primitive_loss.detach().item())
        running['split'] += float(stage3_losses['backbone'].detach().item())
        running['refiner'] += float(stage3_losses['refiner'].detach().item())
        running['candidate_points'] += verifier_stats['candidate_points']
        running['accepted_points'] += verifier_stats['accepted_points']
        running['steps'] += 1

        if (batch_idx + 1) % args.log_interval == 0:
            steps = max(running['steps'], 1)
            candidates = max(running['candidate_points'], 1)
            logger.info(
                'Stage 3 Epoch {:02d} [{}/{}] loss {:.4f} primitive {:.4f} '
                'split {:.4f} refiner {:.4f} accepted {:.2f}%'.format(
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    running['total'] / steps,
                    running['primitive'] / steps,
                    running['split'] / steps,
                    running['refiner'] / steps,
                    100.0 * running['accepted_points'] / candidates,
                )
            )
            for key in running:
                running[key] = 0.0 if key not in ('candidate_points', 'accepted_points', 'steps') else 0


from torch.optim.lr_scheduler import LambdaLR

class LambdaStepLR(LambdaLR):
  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""
  def __init__(self, optimizer, max_iter=30000, power=0.9, last_step=-1):
    poly = lambda s: (1 - s / (max_iter + 1))**power
    schedules = [
        poly if group.get('poly_schedule', True) else (lambda _s: 1.0)
        for group in optimizer.param_groups
    ]
    super(PolyLR, self).__init__(optimizer, schedules, last_step)


def get_current_lr(scheduler, refiner_optimizer=None):
    if scheduler is not None:
        return scheduler.get_lr()[0]
    if refiner_optimizer is not None:
        return refiner_optimizer.param_groups[0]['lr']
    return 0.0

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)


def set_logger(log_path, use_wandb=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 避免重复添加

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    # Logging to wandb
    if use_wandb:
        wandb_handler = WandbHandler()
        wandb_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(wandb_handler)

    return logger

def set_seed(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    args = parse_args()
    if args.wandb:
        setup_custom_env(args, project='GrowSP-S3DIS') 

    '''Setup logger'''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, 'train.log'), use_wandb=args.wandb)

    '''Random Seed'''
    seed = args.seed
    set_seed(seed)

    main(args, logger)
