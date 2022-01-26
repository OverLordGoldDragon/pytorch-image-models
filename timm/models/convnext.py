""" ConvNeXt

Paper: `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

Original code and weights from https://github.com/facebookresearch/ConvNeXt,
original copyright below

Modifications and additions for timm hacked together by / Copyright 2022,
Ross Wightman
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .fx_features import register_notrace_module
from .helpers import named_apply, build_model_with_cfg
from .layers import (trunc_normal_, ClassifierHead, DropPath,
                     ConvMlp, Mlp, get_select_adaptive_pool)
from .layers.custom import (MaxPoolNd, ConvPadNd, LayerNormNd,
                            _set_layer_builders, _max_pool_maker, ashape)
from .registry import register_model


__all__ = ['ConvNeXt']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv;
          all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear
          -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be
    faster with appropriate choice of LayerNorm impl, however as model size
    increases the tradeoffs appear to change and nn.Linear is a better choice.
    This was observed with PyTorch 1.10 on 3090 GPU, it could change over time
    & w/ different HW.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_shape, dim, kernel_size=7, drop_path=0.,
                 ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None):
        super().__init__()

        self.ndim = len(in_shape) - 2
        if not norm_layer:
            norm_layer = (partial(LayerNormNd, ndim=self.ndim, eps=1e-6)
                          if conv_mlp else partial(nn.LayerNorm, eps=1e-6))
        assert conv_mlp
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        # depthwise conv
        assert in_shape[1] == dim, (in_shape, dim)
        self.conv_dw = ConvPadNd(in_shape, dim, kernel_size=kernel_size,
                                 groups=dim, bias=True)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(self.conv_dw.out_shape, int(mlp_ratio * dim),
                             act_layer=nn.GELU)
        self.gamma = (nn.Parameter(ls_init_value * torch.ones(dim))
                      if ls_init_value > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.out_shape = self.mlp.out_shape


    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            1/0
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, *(1,) * self.ndim))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):

    def __init__(
            self, in_shape, out_chs, kernel_size=7, stride=2, depth=2,
            dp_rates=None, ls_init_value=1.0, conv_mlp=False, norm_layer=None,
            cl_norm_layer=None):
        super().__init__()

        if not isinstance(stride, tuple):
            stride = (stride,) * (len(in_shape) - 2)

        in_chs = in_shape[1]
        if in_chs != out_chs or any(s > 1 for s in stride):
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                ConvPadNd(in_shape, out_chs, kernel_size=stride, stride=stride,
                          bias=True),
            )
            self.downsample.out_shape = self.downsample[-1].out_shape
        else:
            self.downsample = nn.Identity()
            self.downsample.out_shape = in_shape
        block_in_shape = self.downsample.out_shape

        dp_rates = dp_rates or [0.] * depth
        blocks = [ConvNeXtBlock(
            in_shape=block_in_shape, dim=out_chs, kernel_size=kernel_size,
            drop_path=dp_rates[j], ls_init_value=ls_init_value,
            conv_mlp=conv_mlp,
            norm_layer=norm_layer if conv_mlp else cl_norm_layer)
            for j in range(depth)]
        self.blocks = nn.Sequential(*blocks)

        self.blocks.out_shape = blocks[-1].out_shape
        self.out_shape = blocks[-1].out_shape

        self.sanity_checks = True
        self.n_sanity_checked = 0
        self.n_sanity_checks = 100

    def forward(self, x):
        x = self.downsample(x)
        ashape(self, x, self.downsample)
        x = self.blocks(x)
        ashape(self, x, self.blocks)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s` -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head.
                           Default: 1000
        depths (tuple(int)): Number of blocks at each stage.
                             Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage.
                           Default: [96, 192, 384, 768]
        out_drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and
                                 biases. Default: 1.
    """

    def __init__(
            self, in_shape, num_classes=2, global_pool='avg',
            layers=(3, 3, 9, 3), channels=(96, 192, 384, 768),
            stride=(1, 2, 2, 2),
            layer_kernel_sizes=(7, 7, 7, 7),
            stem_width=64, stem_kernel_size=7, stem_stride=4, stem_groups=1,
            ls_init_value=1e-6,
            conv_mlp=False, head_init_scale=1., head_norm_first=False,
            norm_layer=None, out_drop_rate=0., drop_path_rate=0.,
            include_classifier=True,
    ):
        super().__init__()

        self.ndim = len(in_shape) - 2
        if norm_layer is None:
            norm_layer = partial(LayerNormNd, ndim=self.ndim, eps=1e-6)
            cl_norm_layer = (norm_layer if conv_mlp else
                             partial(nn.LayerNorm, eps=1e-6))
        else:
            assert conv_mlp, (
                'If a norm_layer is specified, conv MLP must be used so all norm '
                'expect rank-4, channels-first input'
                )
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.out_drop_rate = out_drop_rate
        self.include_classifier = include_classifier

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in
        # SwinTransformer w/ patch_size = 4 (stem_kernel_size)
        stem_width = stem_width  # NOTE: `stem_width = channels[0]` is standard
        self.stem = nn.Sequential(
            ConvPadNd(in_shape, stem_width, kernel_size=stem_kernel_size,
                      stride=stem_stride, groups=stem_groups, bias=True),
            norm_layer(stem_width)
        )
        self.stem.out_shape = self.stem[0].out_shape

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(
            0, drop_path_rate, sum(layers)).split(layers)]
        prev_chs = channels[0]
        stages = []
        stage_in_shape = self.stem.out_shape
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(len(channels)):
            # FIXME support dilation / output_stride
            out_chs = channels[i]
            stages.append(ConvNeXtStage(
                stage_in_shape, out_chs, kernel_size=layer_kernel_sizes[i],
                stride=stride[i], depth=layers[i], dp_rates=dp_rates[i],
                ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                norm_layer=norm_layer, cl_norm_layer=cl_norm_layer)
            )
            stage_in_shape = stages[-1].out_shape
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)

        assert out_chs == stage_in_shape[1], (out_chs, stage_in_shape)
        self.num_features = prev_chs

        if head_norm_first:
            # norm -> global pool -> fc ordering, like most other nets
            # (not compat with FB weights)
            # # final norm layer, before pooling
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(self.num_features, num_classes,
                                       pool_type=global_pool,
                                       drop_rate=out_drop_rate,
                                       include_classifier=include_classifier)
        else:
            # pool -> norm -> fc, the default ConvNeXt ordering
            # (pretrained FB weights)
            self.norm_pre = nn.Identity()
            self.head = nn.Sequential(OrderedDict([
                ('global_pool', get_select_adaptive_pool(self.ndim)(
                    pool_type=global_pool)),
                ('norm', norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.out_drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes)
                 if (num_classes > 0 and include_classifier) else nn.Identity())
            ]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

        self.sanity_checks = True
        self.n_sanity_checked = 0
        self.n_sanity_checks = 100

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool='avg'):
        1/0

    def forward_features(self, x):
        x = self.stem(x)
        ashape(self, x, self.stem)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def _create_convnext(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(ConvNeXt, variant, **kwargs)
    return model


@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model_args = dict(layers=(3, 3, 9, 3), channels=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny_hnf(pretrained=False, **kwargs):
    model_args = dict(layers=(3, 3, 9, 3), channels=(96, 192, 384, 768),
                      head_norm_first=True, **kwargs)
    model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_small(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_base(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[128, 256, 512, 1024],
                      **kwargs)
    model = _create_convnext('convnext_base', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[192, 384, 768, 1536],
                      **kwargs)
    model = _create_convnext('convnext_large', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_base_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[128, 256, 512, 1024],
                      **kwargs)
    model = _create_convnext('convnext_base_in22ft1k', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_large_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[192, 384, 768, 1536],
                      **kwargs)
    model = _create_convnext('convnext_large_in22ft1k', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_xlarge_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[256, 512, 1024, 2048],
                      **kwargs)
    model = _create_convnext('convnext_xlarge_in22ft1k', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_base_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[128, 256, 512, 1024],
                      **kwargs)
    model = _create_convnext('convnext_base_384_in22ft1k', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_large_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[192, 384, 768, 1536],
                      **kwargs)
    model = _create_convnext('convnext_large_384_in22ft1k', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_xlarge_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[256, 512, 1024, 2048],
                      **kwargs)
    model = _create_convnext('convnext_xlarge_384_in22ft1k',
                             pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base_in22k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[128, 256, 512, 1024],
                      **kwargs)
    model = _create_convnext('convnext_base_in22k', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_large_in22k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[192, 384, 768, 1536],
                      **kwargs)
    model = _create_convnext('convnext_large_in22k', pretrained=pretrained,
                             **model_args)
    return model


@register_model
def convnext_xlarge_in22k(pretrained=False, **kwargs):
    model_args = dict(layers=[3, 3, 27, 3], channels=[256, 512, 1024, 2048],
                      **kwargs)
    model = _create_convnext('convnext_xlarge_in22k', pretrained=pretrained,
                             **model_args)
    return model
