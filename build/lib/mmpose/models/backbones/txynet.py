import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmpose.models.backbones.base_backbone import BaseBackbone

from mmengine.model import BaseModule
from mmpose.registry import MODELS

class ConvUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bn: bool = True,
                 act=nn.ReLU
                 ):
        super(ConvUnit, self).__init__()
        if bn:
            bias = False
        else:
            bias = True

        self.layer = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Sequential(),
            act(inplace=True) if isinstance(act, nn.Module) else nn.Sequential()
        )

    def forward(self, x):
        return self.layer(x)


class SEModule(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(SEModule, self).__init__()
        hidden_channels = in_channels // ratio
        # self.squeeze_layer = nn.AdaptiveAvgPool2d(1)
        self.excitation_layer = nn.Sequential(
            ConvUnit(in_channels, hidden_channels, kernel_size=1, stride=1),
            ConvUnit(hidden_channels, in_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        out_ = ConvUnit(c, c, (h, w), stride=1)(x)
        return self.excitation_layer(out_) * x


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        self.convs = nn.Sequential(
           ConvUnit(in_channels=in_channels, out_channels=in_channels,
                   kernel_size=kernel_size, stride=stride, padding=0,
                   groups=in_channels),
            ConvUnit(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        if self.stride > 1:
            h = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            h = F.pad(h, (0, 0, 0, 0, 0, self.channel_pad))

        return self.convs(x) + h



class InvertedResidualDS(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = BlazeBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None)
        
        self._se = SEModule(out_channels)

        self._conv_linear_1 = BlazeBlock(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

        # branch2
        self._conv_pw_2 = BlazeBlock(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        
        self._conv_dw_2 = BlazeBlock(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)

        self._se = SEModule(mid_channels // 2)

        self._conv_linear_2 = BlazeBlock(
            in_channels=mid_channels // 2,

            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_mv1 = BlazeBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            act="hard_swish")
        self._conv_pw_mv1 = BlazeBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish")


    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], axis=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out




class BackBone1(nn.Module):
    def __init__(self):
        super(BackBone1, self).__init__()
        self.backbone1 = nn.ModuleList([
            BlazeBlock(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2)),
            nn.PReLU(32),  # [1,32,96,96]

            BlazeBlock(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.PReLU(32),
            BlazeBlock(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.PReLU(32),
            BlazeBlock(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.PReLU(32),
            BlazeBlock(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.PReLU(64),  # [1,64,48,48]

            BlazeBlock(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.PReLU(64),
            BlazeBlock(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.PReLU(64),
            BlazeBlock(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.PReLU(64),
            BlazeBlock(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.PReLU(128),  # [1,128,24,24]

            BlazeBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.PReLU(128),
            BlazeBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.PReLU(128),
            BlazeBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.PReLU(128),  # 该层需要输出 #[1,128,24,24]
        ])

    def forward(self, x):
        y = x
        for fn in self.backbone1:
            y = fn(y)
        return y


class BackBone2(nn.Module):
    def __init__(self):
        super(BackBone2, self).__init__()
        self.path1 = nn.Sequential(
            InvertedResidualDS(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            nn.PReLU(256),
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256),
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256),
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256))
        self.path2 = nn.Sequential(
            InvertedResidualDS(256, 256, 5, 2),
            nn.PReLU(256),
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256),
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256),
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.conv2d(256, 256, 1),
            nn.PReLU(256))
        self.path3 = nn.Sequential(
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256),
            InvertedResidualDS(256, 256, 5, 1),
            nn.PReLU(256))
        self.path4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.conv2d(256, 128, 1),
            nn.PReLU(128))
        self.path5 = nn.Sequential(
            InvertedResidualDS(128, 128, 5),
            nn.PReLU(128),
            InvertedResidualDS(128, 128, 5),
            nn.PReLU(128))

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(p1)
        p3 = self.path3(p1 + p2)
        p4 = self.path4(p3)
        p5 = self.path5(x + p4)
        return p3, p5



@MODELS.register_module(name="TXYPalm",force=True)
class TXYPalm(BaseBackbone):
    KEY_POINTS_NUMBER = 53
    NUM_PER_KEYPOINT = 2
    NUM_PER_BOX = 4

    def __init__(self):
        super(TXYPalm, self).__init__()

        self.backbone1 = BackBone1()
        self.backbone2 = BackBone2()
        self.classifier1 = nn.Conv2d(256, 6, 1)
        self.classifier2 = nn.Conv2d(128, 2, 1)
        self.regressor1 = nn.Conv2d(256, 108, 1)
        self.regressor2 = nn.Conv2d(128, 36, 1)

    def forward(self, image):
        b1 = self.backbone1(image)
        # print("b1.shape:",b1.shape)

        f1, f2 = self.backbone2(b1)
        # print("f1:",f1.shape)   #torch.Size([1, 256, 12, 12])
        # print("f2:",f2.shape)   #torch.Size([1, 128, 24, 24])
        c1 = self.classifier1(f1)
        c2 = self.classifier2(f2)
        r1 = self.regressor1(f1)
        r2 = self.regressor2(f2)
        # print("c1.shape:",c1.shape)#torch.Size([1, 6, 12, 12])
        # print("c2.shape:",c2.shape)#c2.shape: torch.Size([1, 2, 24, 24])
        # print("r1.shape:",r1.shape)#r1.shape: torch.Size([1, 108, 12, 12])
        # print("r2.shape:",r2.shape)#r2.shape: torch.Size([1, 36, 24, 24])

        regression_channels = self.NUM_PER_BOX + self.KEY_POINTS_NUMBER * self.NUM_PER_KEYPOINT###59
        c1 = c1.permute(0, 2, 3, 1).reshape(-1, c1.shape[1] * c1.shape[2] * c1.shape[3], 1)  # 864
        c2 = c2.permute(0, 2, 3, 1).reshape(-1, c2.shape[1] * c2.shape[2] * c2.shape[3], 1)
        r1 = r1.permute(0, 2, 3, 1).reshape(-1, r1.shape[2] * r1.shape[3] * 6, regression_channels)
        r2 = r2.permute(0, 2, 3, 1).reshape(-1, r2.shape[2] * r2.shape[3] * 2, regression_channels)

        c = torch.cat((c2, c1), dim=1)
        r = torch.cat((r2, r1), dim=1)
        # print("r1.shape:", r1.shape)
        # print("r2.shape:", r2.shape)
        #
        # print("c.shape:", c.shape)
          # torch.Size([1, 2016, 18])
        # print("r:", r)
        return c, r
    

def main():
    dummpy_input = torch.randn(size = (1,3,320,320))
    model = TXYPalm()
    output = model(dummpy_input)
    # torch.Size([1, 128, 40, 40]) h/8
    # torch.Size([1, 256, 20, 20]) h/16
    # torch.Size([1, 512, 10, 10])  h/32


if __name__ == "__main__":
    main()