# Copyright (c) OpenMMLab. All rights reserved.


from .resnet import ResNet, ResNetV1d

from .vanillanet import VanillaNet
from .kw_mobilenetv2 import kw_mobilenetv2

from .hrnet import HRNet
from .hourglass import HourglassNet
from .txynet import TXYPalm
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .csp_darknet_2 import CSPDarknet_2
from .cspnext_2 import CSPNeXt_2


__all__ = [

    'ResNet', 'ResNetV1d',
    'kw_mobilenetv2',
    'HRNet','HourglassNet','TXYPalm','VanillaNet','CSPDarknet','CSPNeXt',
    'CSPDarknet_2','CSPNeXt_2'
]

