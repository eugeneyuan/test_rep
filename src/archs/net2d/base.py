# -*- coding: utf-8 -*-
"""
Used keywords:
norm_type, gn_groups, act_type, elu_alpha, leaky_slope
"""
from __future__ import division, print_function, absolute_import

import torch
from torch import nn
import torch.nn.functional as F

from src.archs.net3d.base import _act_layer, Mish, Swish


def _norm_layer(out_chs, **kwargs):
    norm_type = kwargs.get('norm_type') if kwargs.get('norm_type') is not None else 'bn'

    if norm_type == 'bn':
        return nn.BatchNorm2d(out_chs)
    elif norm_type == 'gn':
        gn_groups = kwargs.get('gn_groups') if kwargs.get('gn_groups') is not None else None
        if gn_groups is not None and gn_groups < out_chs:
            return nn.GroupNorm(gn_groups, out_chs)
        else:
            return nn.GroupNorm(out_chs, out_chs)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(out_chs, affine=True)  # learnable parameter
    else:
        raise ValueError("Param norm_type only support 'bn', 'gn', 'in'!")


# no bias conv
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# --------
# building blocks
class DoubleConv(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, downsample=None, groups=1, dilation=1, **kwargs):
        super(DoubleConv, self).__init__()
        if downsample is None:
            self.seq = nn.Sequential(
                conv3x3(in_chs, out_chs),
                _norm_layer(out_chs, **kwargs),
                _act_layer(**kwargs),
                conv3x3(out_chs, out_chs),
                _norm_layer(out_chs, **kwargs),
                _act_layer(**kwargs),
            )
        else:
            self.seq = nn.Sequential(
                downsample,
                _act_layer(**kwargs),
                conv3x3(out_chs, out_chs),
                _norm_layer(out_chs, **kwargs),
                _act_layer(**kwargs),
            )

    def forward(self, x):
        return self.seq(x)


# res family
# use in_chs, out_chs to control the input, output channels instead of expansion
class BasicBlock(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, downsample=None, groups=1, dilation=1, **kwargs):
        super(BasicBlock, self).__init__()
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_chs, out_chs, stride)
        self.norm1 = _norm_layer(out_chs, **kwargs)
        self.conv2 = conv3x3(out_chs, out_chs)
        self.norm2 = _norm_layer(out_chs, **kwargs)
        self.act = _act_layer(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, downsample=None, groups=1, dilation=1, **kwargs):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_chs, in_chs)
        self.norm1 = _norm_layer(in_chs, **kwargs)
        self.conv2 = conv3x3(in_chs, in_chs, stride, groups, dilation)
        self.norm2 = _norm_layer(in_chs, **kwargs)
        self.conv3 = conv1x1(in_chs, out_chs)
        self.norm3 = _norm_layer(out_chs, **kwargs)
        self.act = _act_layer(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


# se family
# here we remain the original activation layer type
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        return module_input * x


class SEBasicBlock(BasicBlock):

    def __init__(self, in_chs, out_chs, reduction=4, stride=1, downsample=None, groups=1, dilation=1, **kwargs):
        super(SEBasicBlock, self).__init__(in_chs=in_chs, out_chs=out_chs, stride=stride, downsample=downsample,
                                           groups=groups, dilation=dilation, **kwargs)
        self.se_module = SEModule(out_chs, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se_module(out) + identity
        out = self.act(out)

        return out


class SEBottleneck(Bottleneck):

    def __init__(self, in_chs, out_chs, reduction=4, stride=1, downsample=None, groups=1, dilation=1, **kwargs):
        super(SEBottleneck, self).__init__(in_chs=in_chs, out_chs=out_chs, stride=stride, downsample=downsample,
                                           groups=groups, dilation=dilation, **kwargs)
        self.se_module = SEModule(out_chs, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se_module(out) + identity
        out = self.act(out)

        return out


# https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/modules/scse.py
class ChannelGate(nn.Module):  # Channel Squeeze

    def __init__(self, channels):
        super(ChannelGate, self).__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x):
        return x * torch.sigmoid(self.squeeze(x))


class SpatialGate(nn.Module):  # Spatial Squeeze

    def __init__(self, channels, reduction=None, squeeze_channels=None):
        """
        Instantiate module
        :param channels: Number of input channels
        :param reduction: Reduction factor
        :param squeeze_channels: Number of channels in squeeze block.
        """
        super(SpatialGate, self).__init__()
        assert (reduction or squeeze_channels), "One of 'reduction' and 'squeeze_channels' must be set"
        assert not (reduction and squeeze_channels), "'reduction' and 'squeeze_channels' are mutually exclusive"

        if squeeze_channels is None:
            squeeze_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.squeeze.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.expand.weight, nonlinearity="sigmoid")

    def forward(self, x):
        module_input = x
        x = torch.sigmoid(self.expand(F.relu(self.squeeze(self.avg_pool(x)), inplace=True)))
        # print(module_input.mean().item(), module_input.std().item(), x.mean().item(), x.std().item())
        return module_input * x


class ChannelSpatialGate(nn.Module):

    def __init__(self, channels, reduction=4):
        super(ChannelSpatialGate, self).__init__()
        self.channel_gate = ChannelGate(channels)
        self.spatial_gate = SpatialGate(channels, reduction=reduction)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)


class SpatialGateV2(nn.Module):  # Spatial Squeeze and Channel Excitation

    def __init__(self, channels, reduction=4):
        super(SpatialGateV2, self).__init__()
        squeeze_channels = max(1, channels // reduction)
        self.squeeze = nn.Sequential(
            nn.Conv2d(channels, squeeze_channels, kernel_size=1, padding=0),
            nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=7, dilation=3, padding=3 * 3),
        )
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        module_input = x
        x = torch.sigmoid(self.expand(F.relu(self.squeeze(x), inplace=True)))
        return module_input * x


class ChannelSpatialGateV2(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelSpatialGateV2, self).__init__()
        self.channel_gate = ChannelGate(channels)
        self.spatial_gate = SpatialGateV2(channels, reduction)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)


def block_plus_scse(b, scse, reduction=4):
    class BlockSC(nn.Module):
        def __init__(self, in_chs, out_chs, stride=1, downsample=None, groups=1, dilation=1, **kwargs):
            super(BlockSC, self).__init__()
            self.b = b(in_chs, out_chs, stride=stride, downsample=downsample, groups=groups, dilation=dilation, **kwargs)
            self.scse = scse(out_chs, reduction)

        def forward(self, x):
            return self.scse(self.b(x))
    return BlockSC
