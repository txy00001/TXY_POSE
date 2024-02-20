# Copyright (c) OpenMMLab. All rights reserved.
from .base_head import BaseHead
from .coord_cls_heads import RTMCCHead, SimCCHead

from .regression_heads import *
from .dense_reppoints_v2_head import DenseRepPointsV2Head
from .hybrid_heads import DEKRHead, VisPredictHead
from .heatmap_heads import (AssociativeEmbeddingHead, CIDHead, CPMHead,
                            HeatmapHead, MSPNHead, ViPNASHead)
from .vis_head_2 import VisPredictHead_2
__all__ = [
    'BaseHead',
    'RegressionHead', 'SimCCHead',
    'DSNTHead', 'RTMCCHead', 
    'TrajectoryRegressionHead','DenseRepPointsV2Head','DEKRHead', 'VisPredictHead',
    'AssociativeEmbeddingHead','CIDHead','CPMHead','HeatmapHead','MSPNHead','ViPNASHead',
    'VisPredictHead_2',

]
