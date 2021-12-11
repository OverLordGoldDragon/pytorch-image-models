""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
from torch.nn import functional as F

from .adaptive_avgmax_pool import (SelectAdaptivePool1d,
                                   SelectAdaptivePool2d,
                                   SelectAdaptivePool3d)
from .linear import Linear


def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False, dims=2):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    fn = (SelectAdaptivePool1d, SelectAdaptivePool2d, SelectAdaptivePool3d
          )[dims - 1]
    global_pool = fn(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False, dims=2):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fn = (None, nn.Conv2d, nn.Conv3d)[dims - 1]
        fc = fn(num_features, num_classes, 1, bias=True)
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting issue
        if num_classes == 2:
            fc = Linear(num_features, 1, bias=True)
        else:
            fc = Linear(num_features, num_classes, bias=True)
    return fc


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False,
                      include_classifier=True, dims=2):
    global_pool, num_pooled_features = _create_pool(
        num_features, num_classes, pool_type, use_conv=use_conv, dims=dims)
    if include_classifier:
        fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv,
                        dims=dims)
    else:
        fc = None
    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0., use_conv=False):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        x = self.flatten(x)
        return x
