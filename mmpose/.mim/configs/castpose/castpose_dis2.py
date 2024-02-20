_base_ = ['/home/txy/code/CastPose/configs/castpose/pose4.py']

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=200, val_interval=3)

# method details
model = dict(
    _delete_ = True,
    type='PoseEstimatorDistiller',
    two_dis = second_dis,
    teacher_pretrained = '/home/txy/code/CastPose/pth/castpose/dis_s2_pose4_2.pth',
    teacher_cfg = '/home/txy/code/CastPose/configs/castpose/pose4.py',
    student_cfg = '/home/txy/code/CastPose/configs/castpose/pose4.py',
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