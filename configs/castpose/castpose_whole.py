

_base_ = ['/home/txy/code/CastPose/configs/_base_/default_runtime.py']
max_epochs = 300
stage2_num_epochs = 10
base_lr = 0.0001

train_cfg = dict(max_epochs=max_epochs, val_interval=2)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Lion', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),##梯度裁剪
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(288, 384),
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='/home/txy/code/CastPose/pth/dwpose/dw-ll-ucoco-384.pth'  # noqa
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='neck.',
            checkpoint='/home/txy/code/CastPose/work_dirs/wholebody_impove/best_coco_AP_epoch_40.pth'  # noqa
        )
        ),
  
    head=dict(
            type='RTMWHead',
            in_channels=1024,
            out_channels=53,
            input_size=(288,384),
            in_featuremap_size=(9,12),
            simcc_split_ratio=codec['simcc_split_ratio'],
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False),
            loss=dict(
                type='KLDiscretLoss',
                use_target_weight=True,
                beta=10.,
                label_softmax=True),
            decoder=codec,
            init_cfg=dict(
            type='Pretrained',
            prefix='head.',
            checkpoint='/home/txy/code/CastPose/work_dirs/pose4_new/best_coco_AP_epoch_300.pth'  # noqa
        )),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'QXCastPoseDatasets'
data_mode = 'topdown'
data_root = '/home/txy/data/'

backend_args = dict(backend='local')


# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.75, 1.25], rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.75],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=2,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]



###数据集加载合并
# datasets = []
dataset_coco1=dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='qx_datasets/coco_json_body_1/train_coco_new_1.json',
    data_prefix=dict(img='qx_datasets/images/'),
    pipeline=[],##我们自己的数据集不需要转换
)

dataset_coco2=dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='qx_datasets/coco_json_body/train_coco_new.json',
    data_prefix=dict(img='qx_datasets/images/'),
    pipeline=[],##我们自己的数据集不需要转换
)

dataset_coco3=dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='qx_datasets/coco_json_body_2/train_coco_2.json',
    data_prefix=dict(img='qx_datasets/images_20231026/'),
    pipeline=[],##我们自己的数据集不需要转换
)
dataset_coco4=dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='qx_datasets/coco_json_body_2_2/train_coco_2_2.json',
    data_prefix=dict(img='qx_datasets/images_20231026/'),
    pipeline=[],##我们自己的数据集不需要转换
)
dataset_coco5=dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='qx_datasets/coco_json_new_2/train_coco_hand_new.json',
    data_prefix=dict(img='qx_datasets/images_20231026/'),
    pipeline=[],##我们自己的数据集不需要转换
)
dataset_coco6=dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='qx_datasets/coco_json_new/train_coco_hand_new.json',
    data_prefix=dict(img='qx_datasets/images_20231026/'),
    pipeline=[],##我们自己的数据集不需要转换
)

###将cocowholebody转为我们自己的数据集
dataset_wholebody = dict(
    type='CocoWholeBodyDataset',
    data_mode=data_mode,
    data_root='/home/txy/data/coco/',
    ann_file='annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='train2017/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=53,  # 与 我们的 数据集关键点数一致
            mapping=[  # 需要列出所有带转换关键点的序号
                (0, 0),  
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (91, 11),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                (92, 12),
                (93, 13),
                (94, 14),
                (95, 15),
                (96, 16),
                (97, 17),
                (98, 18),
                (99, 19),
                (100, 20),
                (101, 21),
                (102, 22),
                (103, 23),
                (104, 24),
                (105, 25),
                (106, 26),
                (107, 27),
                (108, 28),
                (109, 29),
                (110, 30),
                (111, 31),
                (112, 32),
                (113, 33),
                (114, 34),
                (115, 35),
                (116, 36),
                (117, 37),
                (118, 38),
                (119, 39),
                (120, 40),
                (121, 41),
                (122, 42),
                (123, 43),
                (124, 44),
                (125, 45),
                (126, 46),
                (127, 47),
                (128, 48),
                (129, 49),
                (130, 50),
                (131, 51),
                (132, 52),
            ])
    ],
)

###u-body
ubody_datasets = []
u_scene = ['Magic_show', 'ConductMusic', 
         'TalkShow', 'Fitness', 'Interview', 'Olympic', 'TVShow', 
         'SignLanguage', ]


for i in range(len(u_scene)):
    each = dict(
        type='UBody2dDataset',
        data_root='/mnt/P40_NFS/',
        data_mode=data_mode,
        ann_file='20_Research/10_公共数据集/10_Pose/UBody/annotations_new/'+u_scene[i]+'/train_annotations.json',
        data_prefix=dict(img='20_Research/10_公共数据集/10_Pose/UBody/images/'),
        pipeline=[
            dict(
                type='KeypointConverter',
                num_keypoints=53,  # 与 我们的 数据集关键点数一致
                mapping=[  # 需要列出所有带转换关键点的序号
                    (0, 0),  
                    (1, 1),
                    (2, 2),
                    (3, 3),
                    (4, 4),
                    (5, 5),
                    (6, 6),
                    (7, 7),
                    (8, 8),
                    (9, 9),
                    (10, 10),
                    (91, 11),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                    (92, 12),
                    (93, 13),
                    (94, 14),
                    (95, 15),
                    (96, 16),
                    (97, 17),
                    (98, 18),
                    (99, 19),
                    (100, 20),
                    (101, 21),
                    (102, 22),
                    (103, 23),
                    (104, 24),
                    (105, 25),
                    (106, 26),
                    (107, 27),
                    (108, 28),
                    (109, 29),
                    (110, 30),
                    (111, 31),
                    (112, 32),
                    (113, 33),
                    (114, 34),
                    (115, 35),
                    (116, 36),
                    (117, 37),
                    (118, 38),
                    (119, 39),
                    (120, 40),
                    (121, 41),
                    (122, 42),
                    (123, 43),
                    (124, 44),
                    (125, 45),
                    (126, 46),
                    (127, 47),
                    (128, 48),
                    (129, 49),
                    (130, 50),
                    (131, 51),
                    (132, 52),
                ])
    ],
        
)
    ubody_datasets.append(each)

dataset_ubody = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='/home/txy/code/CastPose/configs/_base_/datasets/qx_castpose.py'),
    datasets=ubody_datasets,
    pipeline=[],
    test_mode=False,
)


# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True,),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='/home/txy/code/CastPose/configs/_base_/datasets/qx_castpose.py'),
        datasets=[dataset_coco1,
                  dataset_coco2, 
                  dataset_coco3,
                  dataset_coco4,
                  
                  dataset_coco5,
                  dataset_coco6,
                  ],
        pipeline=train_pipeline,
        # sample_ratio_factor=[2, 1],
        test_mode=False,
    ))

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False,),
    dataset=dict(
        type='QXCastPoseDatasets',
        data_root=data_root,
        # metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='qx_datasets/coco_json_body_1/val_coco_new_1.json',
        data_prefix=dict(img='qx_datasets/images/'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = {
    'checkpoint': {'save_best':'coco/AP','rule': 'greater','max_keep_ckpts': 100},
    'logger': {'interval': 200}
}

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'qx_datasets/coco_json_body_1/val_coco_new_1.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
]
test_evaluator = val_evaluator