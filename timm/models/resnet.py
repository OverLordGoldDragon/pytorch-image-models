"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py'
(BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants,
tiered stems added by Ross Wightman
Copyright 2020 Ross Wightman
"""
import math, warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import (DropBlock2d, DropPath, AvgPool2dSame, AvgPool3dSame,
                     BlurPool2d, GroupNorm, create_attn, get_attn,
                     create_classifier)
from .layers.custom import (MaxPoolNd, ConvPadNd, BatchNormNd,
                            _set_layer_builders, select_conv)
from .registry import register_model

 # model_registry will add each entrypoint fn to this
__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _weight_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_shape, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, reduce_first=1, dilation=1,
                 first_dilation=None, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None,
                 drop_block=None, drop_path=None, groups=1, kernel_size=3,
                 use_mp=False, residual=True, se_ratio=16):
        super(BasicBlock, self).__init__()
        dims = len(in_shape) - 2
        _set_layer_builders(self, dims)
        self._conv = select_conv
        self.in_shape = in_shape
        self.residual = residual

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or
                                           first_dilation != dilation)
        if use_aa:
            raise NotImplementedError

        if use_mp and (stride != 1 or (isinstance(stride, tuple) and
                                       not all(s == 1 for s in stride))):
            self._max_pool = MaxPoolNd(in_shape, stride, stride)
        else:
            self._max_pool = None

        conv1_in_shape = (self.in_shape if self._max_pool is None else
                          self._max_pool.out_shape)
        conv1_stride = (stride if self._max_pool is None else
                        1)

        self.conv1 = self._conv(
            conv1_in_shape, first_planes,
            kernel_size=kernel_size, stride=conv1_stride,
            dilation=first_dilation, groups=groups, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.aa = aa_layer(channels=first_planes, stride=stride
                           ) if use_aa else None

        self.conv2 = self._conv(
            self.conv1.out_shape, outplanes,
            kernel_size=kernel_size,
            dilation=dilation, groups=groups, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, in_shape=self.conv2.out_shape,
                              se_ratio=se_ratio)

        self.act2 = act_layer(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

        self.out_shape = self.conv2.out_shape

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        # print()
        # print(x.shape)
        if self._max_pool is not None:
            x = self._max_pool(x)
        if self.residual:
            shortcut = x
        # print(x.shape)

        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv2(x)
        # print(x.shape)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.residual and self.downsample is not None:
            shortcut = self.downsample(shortcut)
        # print(x.shape, shortcut.shape)
        if self.residual:
            x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_shape, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, reduce_first=1, dilation=1,
                 groups=1, first_dilation=None, act_layer=nn.ReLU,
                 norm_layer=None, attn_layer=None, aa_layer=None, drop_block=None,
                 drop_path=None, kernel_size=3, use_mp=False, residual=True,
                 se_ratio=16):
        super(Bottleneck, self).__init__()
        dims = len(in_shape) - 2
        _set_layer_builders(self, dims)
        self._conv = select_conv
        self.in_shape = in_shape
        self.residual = residual
        norm_layer = norm_layer or BatchNormNd

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or
                                           first_dilation != dilation)

        if use_aa:
            raise NotImplementedError

        if use_mp and (stride != 1 or (isinstance(stride, tuple) and
                                       not all(s == 1 for s in stride))):
            self._max_pool = MaxPoolNd(in_shape, stride, stride)
        else:
            self._max_pool = None

        conv1_in_shape = (self.in_shape if self._max_pool is None else
                          self._max_pool.out_shape)
        conv2_stride = (stride if self._max_pool is None else
                        1)

        self.conv1 = self._conv(conv1_in_shape, first_planes, kernel_size=1,
                                bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = self._conv(
            self.conv1.out_shape, width, kernel_size=kernel_size,
            stride=conv2_stride, dilation=first_dilation, groups=groups,
            bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = self._conv(self.conv2.out_shape, outplanes,
                                kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, in_shape=self.conv3.out_shape,
                              se_ratio=se_ratio)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

        self.out_shape = self.conv3.out_shape

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        if self._max_pool is not None:
            x = self._max_pool(x)
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


from collections import OrderedDict


def downsample_conv(in_shape, in_channels, out_channels, kernel_size, stride=1,
                    dilation=1, first_dilation=None, norm_layer=None):
    dims = len(in_shape) - 2
    norm_layer = norm_layer or (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                                )[dims - 1]
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size,) * dims
    if not isinstance(dilation, tuple):
        dilation = (dilation,) * dims

    first_dilation = [[]] * dims
    for i, ks in enumerate(kernel_size):
        first_dilation[i] = (first_dilation[i] or dilation[i]) if ks > 1 else 1
    first_dilation = tuple(first_dilation)

    # p = get_padding(kernel_size, stride, first_dilation)

    # _conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dims - 1]
    _conv = select_conv
    return nn.Sequential(
        OrderedDict(
            [('ds-conv',
              _conv(in_shape, out_channels, kernel_size,
                    stride=stride, dilation=first_dilation, bias=False,
                    #padding=p
                    )),
             ('ds-norm',
              norm_layer(out_channels)),]
        )
    )

    # return nn.Sequential(*[
    #     _conv(
    #         in_channels, out_channels, kernel_size, stride=stride, padding=p,
    #         dilation=first_dilation, bias=False),
    #     norm_layer(out_channels)
    # ])


def downsample_avg(in_channels, out_channels, kernel_size, stride=1, dilation=1,
                   first_dilation=None, norm_layer=None):
    1/0
    dims = 0
    norm_layer = norm_layer or (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                                )[dims - 1]
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        if dims == 2:
            avg_pool_fn = (AvgPool2dSame if avg_stride == 1 and dilation > 1 else
                           nn.AvgPool2d)
        elif dims == 3:
            avg_pool_fn = (AvgPool3dSame if avg_stride == 1 and dilation > 1 else
                           nn.AvgPool3d)
        avg_kernel_size = (2 if avg_stride in (1, 2) else
                           avg_stride + 1)
        pool = avg_pool_fn(avg_kernel_size, avg_stride, ceil_mode=True,
                           count_include_pad=False)

    _conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dims - 1]
    1/0
    return nn.Sequential(*[
        pool,
        _conv(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


# def drop_blocks(drop_block_rate=0.):
#     return [
#         None, None,
#         DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
#         DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, in_shape, reduce_first=1,
        down_kernel_size=(1, 1, 1), avg_down=False,
        drop_path_rate=0., layer_groups=(1, 1, 1), layer_kernel_sizes=(3, 3, 3),
        layer_stride=(1, 1, 1), layer_dilation=(1, 1, 1), layer_use_mp=(0, 0, 0),
        layer_residual=(1, 1, 1), layer_se_ratio=(16, 16, 16),
        drop_block_rate=(0, 0, 0),
        drop_block_size=(3, 3, 3),
        drop_block_gamma_scale=(1, 1, 1),
        drop_block_fast=(1, 1, 1),
        attn_layer=None, **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(zip(
            channels, block_repeats)):
        # never liked this name, but weight compat requires it
        stage_name = f'layer{stage_idx + 1}'
        s, use_mp, dilation = [var[stage_idx] for var in
                               (layer_stride, layer_use_mp, layer_dilation)]
        downsample = None

        if s != 1 or inplanes != planes * block_fn.expansion:
            ds_s = 1 if use_mp else s
            down_kwargs = dict(
                in_shape=in_shape, in_channels=inplanes,  # TODO in_shape
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size[stage_idx], stride=ds_s,
                dilation=dilation, first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'))
            downsample = (downsample_avg(**down_kwargs) if avg_down else
                          downsample_conv(**down_kwargs))

        if stage_idx < len(drop_block_rate) and drop_block_rate[stage_idx] != 0:
            _get = lambda ls: ls[stage_idx] if stage_idx < len(ls) else ls[-1]
            drop_block = DropBlock2d(drop_block_rate[stage_idx],
                                     fast=drop_block_fast[stage_idx],
                                     block_size=_get(drop_block_size),
                                     gamma_scale=_get(drop_block_gamma_scale))
        else:
            drop_block = None
        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation,
                            groups=layer_groups[stage_idx],
                            kernel_size=layer_kernel_sizes[stage_idx],
                            use_mp=use_mp, residual=layer_residual[stage_idx],
                            se_ratio=layer_se_ratio[stage_idx],
                            attn_layer=attn_layer, drop_block=drop_block,
                            **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            s = s if block_idx == 0 else 1

            blocks.append(block_fn(
                in_shape, planes, s, downsample,
                first_dilation=prev_dilation, drop_path=None,
                **block_kwargs))
            inplanes = planes * block_fn.expansion
            in_shape = blocks[-1].out_shape
            net_block_idx += 1
        prev_dilation = dilation

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride,
                                 module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet
    that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on
    the v1c, v1d, v1e, and v1s variants included in the MXNet Gluon ResNetV1b
    model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187.
    The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models
    as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA
        ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool
            in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool
            in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool
            in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool
             in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64,
                cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample
        convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels.
        `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width,
          stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3,
          stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs,
        3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between
        stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, block, layers, in_shape, num_classes=1000,
                 cardinality=1, base_width=64, stem_width=64, stem_kernel_size=7,
                 stem_groups=1, block_reduce_first=1,
                 down_kernel_size=(1, 1, 1, 1), avg_down=False,
                 act_layer=nn.ReLU, norm_layer=None,
                 aa_layer=None, in_drop_rate=0., out_drop_rate=0.,
                 drop_path_rate=0., drop_block_rate=(0, 0, 0, 0),
                 drop_block_size=(3, 3, 3, 3),
                 drop_block_gamma_scale=(1, 1, 1, 1),
                 drop_block_fast=(1, 1, 1, 1), in_spatial_dropout=True,
                 global_pool='avg', zero_init_last_bn=True, block_args=None,
                 channels=(64, 128, 256, 512), stem_stride=2, stem_pool=2,
                 stem_pool_kernel_size=3, layer_groups=(1, 1, 1, 1),
                 layer_kernel_sizes=(3, 3, 3, 3),
                 stride=(1, 1, 1, 1), layer_dilation=(1, 1, 1, 1),
                 layer_use_mp=(0, 0, 0, 0), layer_residual=(1, 1, 1, 1),
                 layer_se_ratio=(16, 16, 16, 16), layer_split_idx=None,
                 split_siamese=True, stem_pseudo_4d=False, fuse_op=None,
                 attn_layer='se', fc_post_gap=None, include_classifier=True):
        block_args = block_args or dict()
        self.layers = layers
        self.num_classes = num_classes
        self.in_drop_rate = in_drop_rate
        self.out_drop_rate = out_drop_rate
        self.in_spatial_dropout = in_spatial_dropout
        self.fc_post_gap = fc_post_gap
        self.stem_pool = (stem_pool if isinstance(stem_pool, tuple) else
                          (stem_pool,))
        self.use_stem_pool = bool(not all(p == 1 for p in self.stem_pool))
        self.stem_pseudo_4d = stem_pseudo_4d

        if layer_split_idx is not None:
            assert layer_split_idx >= 0, layer_split_idx
        self.layer_split_idx = layer_split_idx
        self.split_siamese = split_siamese

        # in_sahpe
        if isinstance(in_shape[0], tuple):
            if (layer_split_idx is None or
                in_shape[0] != in_shape[1] or
                len(in_shape) > 2):
                raise ValueError("whatdo with in_shape=%s" % str(in_shape))
            in_shape = in_shape[0]
        self.in_shape = in_shape

        super(ResNet, self).__init__()

        dims = len(in_shape) - 2
        _set_layer_builders(self, dims)
        self._conv = select_conv
        norm_layer = BatchNormNd

        self.sanity_checks = True
        self.n_sanity_checked = 0
        self.n_sanity_checks = 20
        self.n_layers = len(self.layers)

        assert not (dims != 2 and any(d != 0 for d in drop_block_rate)
                    ), (dims, drop_block_rate)

        # Stem ################################################################
        inplanes = stem_width
        def make_stem():
            conv1 = self._conv(in_shape, inplanes, kernel_size=stem_kernel_size,
                               stride=stem_stride, bias=False,
                               groups=stem_groups)
            bn1 = norm_layer(inplanes)
            act1 = act_layer(inplace=True)
            feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
            conv1_out_shape = conv1.out_shape

            stem = nn.Sequential(conv1, bn1, act1)
            return stem, feature_info, conv1_out_shape

        self.stem, self.feature_info, conv1_out_shape = make_stem()

        if self.use_stem_pool:
            self._max_pool = MaxPoolNd(conv1_out_shape, stem_pool_kernel_size,
                                       stem_pool)
            layer1_in_shape = self._max_pool.out_shape
        else:
            layer1_in_shape = conv1_out_shape

        # Pseudo-4d stem ######################################################
        if self.stem_pseudo_4d:
            class Reshape(nn.Module):
                def __init__(self, shape):
                    super(Reshape, self).__init__()
                    self.shape = shape

                def forward(self, x):
                    return x.view(self.shape)

            assert len(in_shape) == 5, in_shape

            def make_stem_pseudo_4d():
                # TODO replace with one modified stem layer
                """- global average `n1` & collapse
                   - convolve over `(n2, n1_fr, t)`
                   - compute SE attention map to scale original `C = 16 * n2`
                """
                avg_pool1 = nn.AdaptiveAvgPool3d((in_shape[2], 1, in_shape[4]))
                # (B, C, n2, n1_fr, 1, t)
                # -- but no 1 implicitly like `keepdim=False`
                #    so we conv `(n2, n1_fr, t)`
                reshape = (in_shape[0], 16, in_shape[1]//16, in_shape[2],
                           in_shape[4])
                reshape10 = Reshape(reshape)

                conv1 = self._conv(reshape, inplanes,
                                   kernel_size=stem_kernel_size,
                                   stride=stem_stride, bias=False,
                                   groups=stem_groups)
                bn1 = norm_layer(inplanes)
                act1 = act_layer(inplace=True)


                stem = nn.Sequential(avg_pool1, reshape10, conv1, bn1, act1)
                return stem

            self.stem_pseudo_4d = make_stem_pseudo_4d()

        # Feature Blocks ######################################################
        def do_make_blocks():
            stage_modules, stage_feature_info = make_blocks(
                block, channels, layers, inplanes, in_shape=layer1_in_shape,
                cardinality=cardinality, base_width=base_width,
                reduce_first=block_reduce_first,
                avg_down=avg_down, down_kernel_size=down_kernel_size,
                act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
                drop_block_rate=drop_block_rate, drop_block_size=drop_block_size,
                drop_block_gamma_scale=drop_block_gamma_scale,
                drop_path_rate=drop_path_rate, drop_block_fast=drop_block_fast,
                layer_groups=layer_groups, layer_kernel_sizes=layer_kernel_sizes,
                layer_stride=stride, layer_dilation=layer_dilation,
                layer_use_mp=layer_use_mp, layer_residual=layer_residual,
                layer_se_ratio=layer_se_ratio, attn_layer=attn_layer,
                **block_args)
            return stage_modules, stage_feature_info

        stage_modules, stage_feature_info = do_make_blocks()
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = channels[-1] * block.expansion
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool,
            include_classifier=include_classifier, dims=dims)

        # non-classifier dense
        if self.fc_post_gap is not None:
            self.fc_post_gap = nn.Sequential(
                nn.Linear(self.num_features, self.fc_post_gap, bias=False),
                nn.BatchNorm1d(self.fc_post_gap),
                nn.ReLU(inplace=True)
            )

        # Fuse op
        self.fuse_op = fuse_op
        if fuse_op is not None:
            fo = {'sum':  torch.add,
                  'diff': torch.sub,
                  None:   None}[fuse_op.split('-')[0]]
            if 'bn' in fuse_op:
                l = getattr(self, f'layer{layer_split_idx}')[-1]
                self._fo_norm = norm_layer(l.out_shape[1])
                self._fuse_op = lambda x0, x1: self._fo_norm(fo(x0, x1))
            else:
                self._fuse_op = fo

        self.init_weights(zero_init_last_bn=zero_init_last_bn)

        # make duplicates & init weights same as original
        if layer_split_idx is not None and not split_siamese:
            stage_modules, stage_feature_info = do_make_blocks()
            for i, stage in enumerate(stage_modules):
                if i > layer_split_idx:
                    break
                name, module = stage
                name += '_dual'
                self.add_module(name, module)
            self.add_module('stem_dual', make_stem()[0])
            if self.stem_pseudo_4d:
                self.add_module('stem_pseudo_4d_dual', make_stem_pseudo_4d())

            names_orig, modules_orig = [], []
            for n, m in self.named_modules():
                if not n.split('.')[0].endswith('_dual') and hasattr(m, 'weight'):
                    names_orig.append(n)
                    modules_orig.append(m)

            for n, m in self.named_modules():
                if n.split('.')[0].endswith('_dual') and hasattr(m, 'weight'):
                    no = n.replace('_dual', '')
                    mo = modules_orig[names_orig.index(no)]
                    with torch.no_grad():
                        m.weight.set_(mo.weight)
                        if m.bias is not None:
                            m.bias.set_(mo.bias)

    def init_weights(self, zero_init_last_bn=True):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features,
                                                      self.num_classes,
                                                      pool_type=global_pool,
                                                      dims=self.dims)

    def save(self, x, i):
        if not hasattr(self, 'save_outs'):
            self.save_outs = False
        if self.save_outs:
            import numpy as np
            np.save(f'{i}.npy', x.detach().cpu().numpy())

    def ashape(self, a, b):
        if self.sanity_checks:
            a = tuple(a.shape)
            b = (b[-1].out_shape if isinstance(b, nn.Sequential) else
                 b.out_shape)
            a = tuple([int(g) for g in a])
            assert a == b, (a, b)
            self.n_sanity_checked += 1
            if self.n_sanity_checked >= self.n_sanity_checks:
                self.sanity_checks = False

    def split_cond(self, second_half, current_idx):
        return (
            # always run
            (second_half is None)  or
            # second half
            (second_half and current_idx >= self.layer_split_idx) or
            # first half
            (not second_half and current_idx < self.layer_split_idx)
        )

    def forward_features(self, x, second_half=None, first_net=True):
        if not second_half:
            x0 = (self.stem(x) if first_net else
                  self.stem_dual(x))
            self.save(x0, 1)

            if self.stem_pseudo_4d:
                x1 = (self.stem_pseudo_4d(x) if first_net else
                      self.stem_pseudo_4d_dual(x))
                x = x0 + x1
            else:
                x = x0

            if self.use_stem_pool:
                x = self._max_pool(x)
            self.save(x, 2)

        if self.n_layers >= 1 and self.split_cond(second_half, 0):
            x = (self.layer1(x) if first_net else
                 self.layer1_dual(x))
            self.ashape(x, self.layer1)
            self.save(x, 3)

        if self.n_layers >= 2 and self.split_cond(second_half, 1):
            x = (self.layer2(x) if first_net else
                 self.layer2_dual(x))
            self.ashape(x, self.layer2)
            self.save(x, 4)

        if self.n_layers >= 3 and self.split_cond(second_half, 2):
            x = (self.layer3(x) if first_net else
                 self.layer3_dual(x))
            self.ashape(x, self.layer3)
            self.save(x, 5)

        if self.n_layers >= 4 and self.split_cond(second_half, 3):
            x = (self.layer4(x) if first_net else
                 self.layer4_dual(x))
            self.ashape(x, self.layer4)
            self.save(x, 6)

        if self.n_layers >= 5 and self.split_cond(second_half, 4):
            x = self.layer5(x)
            self.ashape(x, self.layer5)
            self.save(x, 7)

        return x

    def forward(self, *x):
        if len(x) == 1:
            x = x[0]
        if not isinstance(x, (list, tuple)):
            if self.in_drop_rate:
                if self.in_spatial_dropout:
                    if x.ndim in (3, 4):
                        # 1d, 2d  # TODO torch may fix bug in 1.11+
                        l = F.dropout2d
                    elif x.ndim == 5:
                        l = F.dropout3d
                else:
                    l = F.dropout
                x = l(x, p=float(self.in_drop_rate), training=self.training)
                self.save(x, -1)

            x = self.forward_features(x)
        else:
            x0, x1 = x
            if self.split_siamese:
                x0 = self.forward_features(x0, second_half=False)
                x1 = self.forward_features(x1, second_half=False)
            else:
                x0 = self.forward_features(x0, second_half=False, first_net=True)
                x1 = self.forward_features(x1, second_half=False, first_net=False)
            x = self._fuse_op(x0, x1)
            x = self.forward_features(x, second_half=True, first_net=True)

        x = self.global_pool(x)
        self.save(x, 9)

        if self.out_drop_rate:
            x = F.dropout(x, p=float(self.out_drop_rate), training=self.training)
            self.save(x, 10)

        if self.fc_post_gap is not None:
            x = self.fc_post_gap(x)
            self.save(x, 11)

        if self.fc is not None:
            x = self.fc(x)
            self.save(x, 12)
        return x


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, pretrained,
        # default_cfg=default_cfgs[variant],
        **kwargs)


@register_model
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('resnet18', pretrained, **model_args)


@register_model
def resnet18d(pretrained=False, **kwargs):
    """Constructs a ResNet-18-D model.
    """
    model_args = dict(
        block=BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep',
        avg_down=True, **kwargs)
    return _create_resnet('resnet18d', pretrained, **model_args)


@register_model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet34', pretrained, **model_args)


@register_model
def resnet34d(pretrained=False, **kwargs):
    """Constructs a ResNet-34-D model.
    """
    model_args = dict(
        block=BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep',
        avg_down=True, **kwargs)
    return _create_resnet('resnet34d', pretrained, **model_args)


@register_model
def resnet26(pretrained=False, **kwargs):
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('resnet26', pretrained, **model_args)


@register_model
def resnet26t(pretrained=False, **kwargs):
    """Constructs a ResNet-26-T model.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32,
        stem_type='deep_tiered', avg_down=True, **kwargs)
    return _create_resnet('resnet26t', pretrained, **model_args)


@register_model
def resnet26d(pretrained=False, **kwargs):
    """Constructs a ResNet-26-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32,
                      stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet26d', pretrained, **model_args)


@register_model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)


@register_model
def resnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet50d', pretrained, **model_args)


@register_model
def resnet50t(pretrained=False, **kwargs):
    """Constructs a ResNet-50-T model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, **kwargs)
    return _create_resnet('resnet50t', pretrained, **model_args)


@register_model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('resnet101', pretrained, **model_args)


@register_model
def resnet101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32,
                      stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet101d', pretrained, **model_args)


@register_model
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('resnet152', pretrained, **model_args)


@register_model
def resnet152d(pretrained=False, **kwargs):
    """Constructs a ResNet-152-D model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, **kwargs)
    return _create_resnet('resnet152d', pretrained, **model_args)


@register_model
def resnet200(pretrained=False, **kwargs):
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], **kwargs)
    return _create_resnet('resnet200', pretrained, **model_args)


@register_model
def resnet200d(pretrained=False, **kwargs):
    """Constructs a ResNet-200-D model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, **kwargs)
    return _create_resnet('resnet200d', pretrained, **model_args)


@register_model
def tv_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model with original Torchvision weights.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('tv_resnet34', pretrained, **model_args)


@register_model
def tv_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with original Torchvision weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('tv_resnet50', pretrained, **model_args)


@register_model
def tv_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model w/ Torchvision pretrained weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('tv_resnet101', pretrained, **model_args)


@register_model
def tv_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model w/ Torchvision pretrained weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('tv_resnet152', pretrained, **model_args)


@register_model
def wide_resnet50_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128,
                      **kwargs)
    return _create_resnet('wide_resnet50_2', pretrained, **model_args)


@register_model
def wide_resnet101_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128,
                      **kwargs)
    return _create_resnet('wide_resnet101_2', pretrained, **model_args)


@register_model
def resnet50_gn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model w/ GroupNorm
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50_gn', pretrained, norm_layer=GroupNorm,
                          **model_args)


@register_model
def resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet('resnext50_32x4d', pretrained, **model_args)


@register_model
def resnext50d_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool
    downsample"""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnext50d_32x4d', pretrained, **model_args)


@register_model
def resnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet('resnext101_32x4d', pretrained, **model_args)


@register_model
def resnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=8, **kwargs)
    return _create_resnet('resnext101_32x8d', pretrained, **model_args)


@register_model
def resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt101-64x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64,
                      base_width=4, **kwargs)
    return _create_resnet('resnext101_64x4d', pretrained, **model_args)


@register_model
def tv_resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model with original Torchvision weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet('tv_resnext50_32x4d', pretrained, **model_args)


@register_model
def ig_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=8, **kwargs)
    return _create_resnet('ig_resnext101_32x8d', pretrained, **model_args)


@register_model
def ig_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=16, **kwargs)
    return _create_resnet('ig_resnext101_32x16d', pretrained, **model_args)


@register_model
def ig_resnext101_32x32d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=32, **kwargs)
    return _create_resnet('ig_resnext101_32x32d', pretrained, **model_args)


@register_model
def ig_resnext101_32x48d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=48, **kwargs)
    return _create_resnet('ig_resnext101_32x48d', pretrained, **model_args)


@register_model
def ssl_resnet18(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNet-18 model pre-trained on YFCC100M
    dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from
    https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('ssl_resnet18', pretrained, **model_args)


@register_model
def ssl_resnet50(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNet-50 model pre-trained on YFCC100M
    dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from
    https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('ssl_resnet50', pretrained, **model_args)


@register_model
def ssl_resnext50_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-50 32x4 model pre-trained on
    YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from
    https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet('ssl_resnext50_32x4d', pretrained, **model_args)


@register_model
def ssl_resnext101_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x4 model pre-trained on
    YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from
    https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet('ssl_resnext101_32x4d', pretrained, **model_args)


@register_model
def ssl_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x8 model pre-trained on
    YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from
    https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=8, **kwargs)
    return _create_resnet('ssl_resnext101_32x8d', pretrained, **model_args)


@register_model
def ssl_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x16 model pre-trained on
    YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from
    https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=16, **kwargs)
    return _create_resnet('ssl_resnext101_32x16d', pretrained, **model_args)


@register_model
def swsl_resnet18(pretrained=True, **kwargs):
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('swsl_resnet18', pretrained, **model_args)


@register_model
def swsl_resnet50(pretrained=True, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('swsl_resnet50', pretrained, **model_args)


@register_model
def swsl_resnext50_32x4d(pretrained=True, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet('swsl_resnext50_32x4d', pretrained, **model_args)


@register_model
def swsl_resnext101_32x4d(pretrained=True, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet('swsl_resnext101_32x4d', pretrained, **model_args)


@register_model
def swsl_resnext101_32x8d(pretrained=True, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=8, **kwargs)
    return _create_resnet('swsl_resnext101_32x8d', pretrained, **model_args)


@register_model
def swsl_resnext101_32x16d(pretrained=True, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32,
                      base_width=16, **kwargs)
    return _create_resnet('swsl_resnext101_32x16d', pretrained, **model_args)


@register_model
def ecaresnet26t(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'),
        **kwargs)
    return _create_resnet('ecaresnet26t', pretrained, **model_args)


@register_model
def ecaresnet50d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet50d', pretrained, **model_args)


@register_model
def resnetrs50(pretrained=False, **kwargs):
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep',
        replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs50', pretrained, **model_args)


@register_model
def resnetrs101(pretrained=False, **kwargs):
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep',
        replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs101', pretrained, **model_args)


@register_model
def resnetrs152(pretrained=False, **kwargs):
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep',
        replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs152', pretrained, **model_args)


@register_model
def resnetrs200(pretrained=False, **kwargs):
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep',
        replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs200', pretrained, **model_args)


@register_model
def resnetrs270(pretrained=False, **kwargs):
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 29, 53, 4], stem_width=32, stem_type='deep',
        replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs270', pretrained, **model_args)



@register_model
def resnetrs350(pretrained=False, **kwargs):
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 36, 72, 4], stem_width=32, stem_type='deep',
        replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs350', pretrained, **model_args)


@register_model
def resnetrs420(pretrained=False, **kwargs):
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 44, 87, 4], stem_width=32, stem_type='deep',
        replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs420', pretrained, **model_args)


@register_model
def ecaresnet50d_pruned(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model pruned with eca.
        The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet50d_pruned', pretrained, pruned=True,
                          **model_args)


@register_model
def ecaresnet50t(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'),
        **kwargs)
    return _create_resnet('ecaresnet50t', pretrained, **model_args)


@register_model
def ecaresnetlight(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D light model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[1, 1, 11, 3], stem_width=32, avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnetlight', pretrained, **model_args)


@register_model
def ecaresnet101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet101d', pretrained, **model_args)


@register_model
def ecaresnet101d_pruned(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model pruned with eca.
       The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet101d_pruned', pretrained, pruned=True,
                          **model_args)


@register_model
def ecaresnet200d(pretrained=False, **kwargs):
    """Constructs a ResNet-200-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet200d', pretrained, **model_args)


@register_model
def ecaresnet269d(pretrained=False, **kwargs):
    """Constructs a ResNet-269-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet269d', pretrained, **model_args)


@register_model
def ecaresnext26t_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'),
        **kwargs)
    return _create_resnet('ecaresnext26t_32x4d', pretrained, **model_args)


@register_model
def ecaresnext50t_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'),
        **kwargs)
    return _create_resnet('ecaresnext50t_32x4d', pretrained, **model_args)


@register_model
def resnetblur18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model with blur anti-aliasing
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], aa_layer=BlurPool2d,
                      **kwargs)
    return _create_resnet('resnetblur18', pretrained, **model_args)


@register_model
def resnetblur50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with blur anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d,
                      **kwargs)
    return _create_resnet('resnetblur50', pretrained, **model_args)


@register_model
def seresnet18(pretrained=False, **kwargs):
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2],
                      block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet18', pretrained, **model_args)


@register_model
def seresnet34(pretrained=False, **kwargs):
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3],
                      block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet34', pretrained, **model_args)


@register_model
def seresnet50(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],
                      block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet50', pretrained, **model_args)


@register_model
def seresnet50t(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  stem_width=32,
        stem_type='deep_tiered', avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet50t', pretrained, **model_args)


@register_model
def seresnet101(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3],
                      block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet101', pretrained, **model_args)


@register_model
def seresnet152(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3],
                      block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet152', pretrained, **model_args)


@register_model
def seresnet152d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet152d', pretrained, **model_args)


@register_model
def seresnet200d(pretrained=False, **kwargs):
    """Constructs a ResNet-200-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet200d', pretrained, **model_args)


@register_model
def seresnet269d(pretrained=False, **kwargs):
    """Constructs a ResNet-269-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep',
        avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet269d', pretrained, **model_args)


@register_model
def seresnext26d_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32,
        stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'),
        **kwargs)
    return _create_resnet('seresnext26d_32x4d', pretrained, **model_args)


@register_model
def seresnext26t_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='se'),
        **kwargs)
    return _create_resnet('seresnext26t_32x4d', pretrained, **model_args)


@register_model
def seresnext26tn_32x4d(pretrained=False, **kwargs):
    return seresnext26t_32x4d(pretrained=pretrained, **kwargs)


@register_model
def seresnext50_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext50_32x4d', pretrained, **model_args)


@register_model
def seresnext101_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext101_32x4d', pretrained, **model_args)


@register_model
def seresnext101_32x8d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext101_32x8d', pretrained, **model_args)


@register_model
def senet154(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], cardinality=64, base_width=4,
        stem_type='deep',
        down_kernel_size=3, block_reduce_first=2,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('senet154', pretrained, **model_args)
