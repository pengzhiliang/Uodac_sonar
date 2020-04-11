# model settings
model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained='pretrain_model/cascade_rcnn_dconv_c3-c5_r101_dufpn_1x.pth',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='DuFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.4, 0.9, 1.6, 2.9, 4.2, 6.0, 8.6],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='UncertainSharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.1, 0.1],
            with_uncertain=True,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,),# smoothing=0.01
            loss_bbox=dict(type='UncertainKLLoss', beta=1.0, loss_weight=1.0)),
        dict(
            type='UncertainSharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.05, 0.05],
            with_uncertain=True,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,),# smoothing=0.01
            loss_bbox=dict(type='UncertainKLLoss', beta=1.0, loss_weight=1.0)),
        dict(
            type='UncertainSharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.033, 0.033],
            with_uncertain=True,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,),
            loss_bbox=dict(type='UncertainKLLoss', beta=1.0, loss_weight=1.0))
    ])
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),        
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(score_thr=0.001,
              nms=dict(type='soft_nms_variance_voting',
                       sigma_t=0.01, iou_thr=0.5, min_score=0.001,
                      ),     
              return_variance=False,
              max_per_img=80),)
# dataset settings
dataset_type = 'SonarDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Grafting_SideScan', image_root_dir=data_root, use_side=True),
    dict(type='Stretch', stretch_prob=0.75, stretch_ratios=[(0.95, 1.05),(0.95, 1.05)]),
    dict(type='MinIoURandomCrop', min_ious=(0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.5),
    dict(type='AbjointResize', ratio_scales=[[0.8, 0.9, 1.0, 1.1, 1.2],
                                         [0.8, 0.9, 1.0, 1.1, 1.2]], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.66, direction=['horizontal', 'vertical'], filter_prefix=['gx',]), # 'vertical'
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        ratio_scale=[[0.8, 0.9, 1.0, 1.1, 1.2], # 1.0, 1.1, 1.2, 1.3, 1.4
                     [0.8, 0.9, 1.0, 1.1, 1.2]],
        flip=True,
        ver_flip=True,
        filter_prefix=['gx'],
        transforms=[
            dict(type='AbjointResize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotation/instances_train.json',
        ann_file=data_root + 'annotation/instances_all.json',
        img_prefix=data_root + 'image/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/instances_val.json',
        img_prefix=data_root + 'image/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/a-test.json', # b-test
        img_prefix=data_root + 'a-test-image/', # b-test-image
#         ann_file=data_root + 'annotation/instances_val.json',
#         img_prefix=data_root + 'image/',
        pipeline=test_pipeline))

evaluation = dict(interval=4, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=6)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_rcnn_dconv_c3-c5_r101_dufpn_1x'
load_from = 'pretrain_model/cascade_rcnn_dconv_c3-c5_r101_dufpn_1x.pth'
resume_from = None
workflow = [('train', 1)]
