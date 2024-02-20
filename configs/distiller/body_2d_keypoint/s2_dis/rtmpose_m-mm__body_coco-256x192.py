_base_ = ['../../../body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py']

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=80, val_interval=10)

# method details
model = dict(
    _delete_ = True,
    type='PoseEstimatorDistiller',
    two_dis = second_dis,
    teacher_pretrained = '/home/yangzhendong/Projects/mmpose/work_dirs/rtmpose_l_dis_m__body_coco-256x192/rtm-m_74.66.pth',
    teacher_cfg = 'configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py',
    student_cfg = 'configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py',
    distill_cfg = [
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_logit',
                                       use_this = logit,
                                       weight = 1,
                                       )
                                ]
                        ),
                    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    train_cfg=train_cfg,
)

optim_wrapper = dict(
    clip_grad=dict(max_norm=1., norm_type=2))