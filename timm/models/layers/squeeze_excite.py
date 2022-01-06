""" Squeeze-and-Excitation Channel Attention

An SE implementation originally based on PyTorch SE-Net impl.
Has since evolved with additional functionality / configuration.

Paper: `Squeeze-and-Excitation Networks` - https://arxiv.org/abs/1709.01507

Also included is Effective Squeeze-Excitation (ESE).
Paper: `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667

Hacked together by / Copyright 2021 Ross Wightman
"""
import torch
from torch import nn as nn
import torch.nn.functional as F

from .create_act import create_act_layer
from .helpers import make_divisible


def _set_layer_builders(self, dims):
    assert dims in (1, 2, 3), dims
    setattr(self, 'dims', dims)
    setattr(self, '_conv', getattr(nn, f'Conv{dims}d'))


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=None, gate_layer='sigmoid', dims=2):
        super(SEModule, self).__init__()
        assert dims == 2
        _set_layer_builders(self, dims)
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = self._conv(channels, rd_channels, kernel_size=1, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = self._conv(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        dim = tuple(range(-self.dims, 0))[::-1]
        x_se = x.mean(dim, keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax(dim, keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


SqueezeExcite = SEModule  # alias


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """
    def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid', **_):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


EffectiveSqueezeExcite = EffectiveSEModule  # alias



class ChannelSELayerNd(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=16, dims=3):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayerNd, self).__init__()
        self.dims = dims
        self.avg_pool = getattr(nn, f'AdaptiveAvgPool{dims}d')(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, *spatial = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(
            squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        view_shape = (batch_size, num_channels) + (1,) * len(spatial)
        output_tensor = torch.mul(input_tensor, fc_out_2.view(*view_shape))

        return output_tensor


class SpatialSELayerNd(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels, dims=3):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayerNd, self).__init__()
        self.conv = getattr(nn, f'Conv{dims}d')(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, num_channels, *spatial = input_tensor.size()

        out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor,
                                  squeeze_tensor.view(batch_size, 1, *spatial))

        return output_tensor


class ChannelSpatialSELayerNd(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=16, dims=3):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayerNd, self).__init__()
        self.cSE = ChannelSELayerNd(num_channels, reduction_ratio, dims)
        self.sSE = SpatialSELayerNd(num_channels, dims)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class SEModuleNd(ChannelSpatialSELayerNd):
    def __init__(self, channels, rd_ratio=16, dims=3):
        super(SEModuleNd, self).__init__(channels, rd_ratio, dims)


ChannelSpatialSqueezeExciteLayerNd = ChannelSpatialSELayerNd  # alias
