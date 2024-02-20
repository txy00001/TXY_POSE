

_base_ = ['/home/txy/code/CastPose/configs/_base_/default_runtime.py']
max_epochs = 300
stage2_num_epochs = 10
base_lr = 0.0001

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
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
    ##加入可见性预测
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
            decoder=codec),
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
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
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
                p=1.0),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
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
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
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
                  dataset_coco6,],
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