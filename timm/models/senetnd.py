"""
SEResNet implementation from Cadene's pretrained models
https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
Additional credit to https://github.com/creafz

Original model: https://github.com/hujie-frank/SENet

ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import math
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import create_classifier
from .registry import register_model

__all__ = ['SENetND']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }


default_cfgs = {
    'legacy_senet154_nd':
        _cfg(url='http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth'),
    'legacy_seresnet18_nd': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet18-4bb0ce65.pth',
        interpolation='bicubic'),
    'legacy_seresnet34_nd': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet34-a4004e63.pth'),
    'legacy_seresnet50_nd': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet50-ce0d4300.pth'),
    'legacy_seresnet101_nd': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet101-7e38fcc6.pth'),
    'legacy_seresnet152_nd': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet152-d17c99b7.pth'),
    'legacy_seresnext26_32x4d_nd': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26_32x4d-65ebdb501.pth',
        interpolation='bicubic'),
    'legacy_seresnext50_32x4d_nd':
        _cfg(url='http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'),
    'legacy_seresnext101_32x4d_nd':
        _cfg(url='http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'),
}


def _weight_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)


def _set_layer_builders(self, dims):
    assert dims in (1, 2, 3), dims
    setattr(self, 'dims', dims)
    setattr(self, '_conv', getattr(nn, f'Conv{dims}d'))
    setattr(self, '_norm', getattr(nn, f'BatchNorm{dims}d'))
    setattr(self, '_max_pool', getattr(nn, f'MaxPool{dims}d'))
    setattr(self, 'mean', lambda x: x.mean([-1, (-1, -2), (-1, -2, -3)][dims],
                                           keepdim=True))


class SEModule(nn.Module):

    def __init__(self, channels, reduction, dims=2):
        super(SEModule, self).__init__()
        _set_layer_builders(self, dims)

        # layers to reuse
        self.fc1 = self._conv(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = self._conv(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.mean(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, dims=2):
        super(SEBottleneck, self).__init__()
        _set_layer_builders(self, dims)

        self.conv1 = self._conv(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = self._norm(planes * 2)
        self.conv2 = self._conv(
            planes * 2, planes * 4, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
        self.bn2 = self._norm(planes * 4)
        self.conv3 = self._conv(
            planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self._norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction, dims=dims)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, dims=2):
        super(SEResNetBottleneck, self).__init__()
        _set_layer_builders(self, dims)
        self.conv1 = self._conv(
            inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = self._norm(planes)
        self.conv3 = self._conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self._norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction, dims=dims)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4, dims=2):
        super(SEResNeXtBottleneck, self).__init__()
        _set_layer_builders(self, dims)
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = self._conv(
            inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = self._norm(width)
        self.conv2 = self._conv(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = self._norm(width)
        self.conv3 = self._conv(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self._norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction, dims=dims)
        self.downsample = downsample
        self.stride = stride


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None,
                 dims=2):
        super(SEResNetBlock, self).__init__()
        _set_layer_builders(self, dims)
        self.conv1 = self._conv(
            inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = self._norm(planes)
        self.conv2 = self._conv(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = self._norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction, dims=dims)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out


class SENetND(nn.Module):

    def __init__(self, block, layers, groups, reduction, drop_rate=0.2,
                 in_chans=3, inplanes=64, input_3x3=False, downsample_kernel_size=1,
                 downsample_padding=0, num_classes=1000, global_pool='avg', dims=2):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        dims (int): whether to use Conv1d, Conv2d, or Conv3d (`1`, `2`, `3`)
        """
        super(SENetND, self).__init__()
        _set_layer_builders(self, dims)
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if input_3x3:
            layer0_modules = [
                ('conv1', self._conv(in_chans, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', self._norm(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', self._conv(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', self._norm(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', self._conv(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', self._norm(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', self._conv(
                    in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', self._norm(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        self.pool0 = self._max_pool(3, stride=2, ceil_mode=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='layer0')]
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.feature_info += [dict(num_chs=64 * block.expansion, reduction=4, module='layer1')]
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.feature_info += [dict(num_chs=128 * block.expansion, reduction=8, module='layer2')]
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.feature_info += [dict(num_chs=256 * block.expansion, reduction=16, module='layer3')]
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.feature_info += [dict(num_chs=512 * block.expansion, reduction=32, module='layer4')]
        self.num_features = 512 * block.expansion
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self._conv(
                    self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size,
                    stride=stride, padding=downsample_padding, bias=False),
                self._norm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.layer0(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.logits(x)
        return x


def _create_senet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        SENetND, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)


@register_model
def legacy_seresnet18_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNetBlock, layers=[2, 2, 2, 2], groups=1, reduction=16, **kwargs)
    return _create_senet('legacy_seresnet18_nd', pretrained, **model_args)


@register_model
def legacy_seresnet34_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNetBlock, layers=[3, 4, 6, 3], groups=1, reduction=16, **kwargs)
    return _create_senet('legacy_seresnet34_nd', pretrained, **model_args)


@register_model
def legacy_seresnet50_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, **kwargs)
    return _create_senet('legacy_seresnet50_nd', pretrained, **model_args)


@register_model
def legacy_seresnet101_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 4, 23, 3], groups=1, reduction=16, **kwargs)
    return _create_senet('legacy_seresnet101_nd', pretrained, **model_args)


@register_model
def legacy_seresnet152_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 8, 36, 3], groups=1, reduction=16, **kwargs)
    return _create_senet('legacy_seresnet152_nd', pretrained, **model_args)


@register_model
def legacy_senet154_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEBottleneck, layers=[3, 8, 36, 3], groups=64, reduction=16,
        downsample_kernel_size=3, downsample_padding=1,  inplanes=128, input_3x3=True, **kwargs)
    return _create_senet('legacy_senet154_nd', pretrained, **model_args)


@register_model
def legacy_seresnext26_32x4d_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNeXtBottleneck, layers=[2, 2, 2, 2], groups=32, reduction=16, **kwargs)
    return _create_senet('legacy_seresnext26_32x4d_nd', pretrained, **model_args)


@register_model
def legacy_seresnext50_32x4d_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16, **kwargs)
    return _create_senet('legacy_seresnext50_32x4d_nd', pretrained, **model_args)


@register_model
def legacy_seresnext101_32x4d_nd(pretrained=False, **kwargs):
    model_args = dict(
        block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16, **kwargs)
    return _create_senet('legacy_seresnext101_32x4d_nd', pretrained, **model_args)
