# Copyright (c) OpenMMLab. All rights reserved.
from .dekr_head import DEKRHead
from .vis_head import VisPredictHead
from .yolox_head import YOLOXPoseHead

__all__ = ['DEKRHead', 'VisPredictHead', 'YOLOXPoseHead']

###vispred:用于实现可视化预测的功能。它的主要作用是生成姿态关键点的可视化结果，以便在训练或推理过程中对姿态估计进行可视化展示
###dekr:预测姿态关键点的位置和可见性

