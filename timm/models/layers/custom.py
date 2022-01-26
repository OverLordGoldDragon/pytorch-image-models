# -*- coding: utf-8 -*-
import torch, math
import torch.nn as nn
import torch.nn.functional as F
# from ..fx_features import register_notrace_module


# @register_notrace_module
class LayerNormNd(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions
    (ie N, C, H, W).
    """
    def __init__(self, normalized_shape, ndim=2, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)
        self.ndim = ndim

        self.permute_shape = {1: (0, 2, 1),
                              2: (0, 2, 3, 1),
                              3: (0, 2, 3, 4, 1)}[ndim]
        self.unpermute_shape = {1: (0, 2, 1),
                                2: (0, 3, 1, 2),
                                3: (0, 4, 1, 2, 3)}[ndim]

    def expdim(self, x):
        if self.ndim == 1:
            return x[:, None]
        if self.ndim == 2:
            return x[:, None, None]
        return x[:, None, None, None]

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(self.permute_shape), self.normalized_shape, self.weight,
                self.bias, self.eps).permute(self.unpermute_shape)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.expdim(self.weight) + self.expdim(self.bias)
            return x


def _max_pool_maker(dims):
    mp_layer = getattr(nn, f'MaxPool{dims}d')
    return lambda padding, in_shape=None, **kwargs: MaxPoolNd(
        mp_layer, padding, in_shape, **kwargs)


class MaxPoolNd(nn.Module):
    def __init__(self, mp_layer, padding=0, in_shape=None, **kwargs):
        super().__init__()
        self.mp_layer = mp_layer(**kwargs)
        self.padding = padding
        self.in_shape = in_shape
        self.kwargs = kwargs

        if in_shape is not None:
            self.out_shape = tuple(ConvPadNd.compute_out_shape(
                in_shape, ks=kwargs.get('kernel_size', 1),
                s=kwargs.get('stride', 1), d=1, pad=padding))
        else:
            self.out_shape = None

    def forward(self, x):
        if self.padding != 0:
            x = F.pad(x, self.padding)
        return self.mp_layer(x)


def _set_layer_builders(self, dims):
    assert dims in (1, 2, 3), dims
    setattr(self, 'dims', dims)
    # setattr(self, '_conv', getattr(nn, f'Conv{dims}d'))
    setattr(self, '_norm', getattr(nn, f'BatchNorm{dims}d'))
    setattr(self, 'mean', lambda x: x.mean(tuple(range(-dims, 0))[::-1],
                                           keepdim=True))
    self._max_pool = _max_pool_maker(dims)


class ConvPadNd(nn.Module):
    """If `stride=1`, implements 'same', else pads enough so that first
    kernel op centers at first point of input (`ceil(kernel_size/2)` on
    each side).
    """
    def __init__(self, in_shape, ch_out, kernel_size=3, stride=1,
                 groups=1, dilation=1, bias=True):
        super().__init__()
        self.in_shape = in_shape
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.bias = bias

        self.dims = len(in_shape) - 2  # minus `batch_size`, `channels`
        self.pad_shape = self.compute_pad_shape(in_shape, kernel_size, stride,
                                                dilation)
        self.out_shape = self.compute_out_shape(in_shape, kernel_size, stride,
                                                dilation, self.pad_shape, ch_out)
        self.do_pad = any(pad > 0 for pad in self.pad_shape)

        ch_in = self.in_shape[1]
        self.conv = getattr(nn, f'Conv{self.dims}d')(
            ch_in, ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, x):
        if self.do_pad:
            x = F.pad(x, self.pad_shape)
        return self.conv(x)

    @staticmethod
    def compute_pad_shape(in_shape, ks, s, d):
        dims, ks, s, d, shape_spatial = ConvPadNd._process_args(
            in_shape, ks, s, d)

        pad_shape = []
        for i, L in enumerate(shape_spatial):
            pad = ConvPadNd.compute_pad_amount(L, ks[i], s[i], d[i])
            pad_shape += [pad//2, pad - pad//2]
        pad_shape = pad_shape[::-1]  # `F.pad` specifies backwards
        return pad_shape

    @staticmethod
    def compute_pad_amount(L, ks, s, d):
        if 0:#s > 1:
            pad = (math.ceil(ks / 2) if ks != 1 else
                   0)
        else:
            pad = (math.ceil(L / s) - 1) * s + (ks - 1) * d + 1 - L
        return int(max(pad, 0))

    @staticmethod
    def compute_out_shape(in_shape, ks, s, d, pad, ch_out=None):
        pad = pad[::-1]  # since it was inverted
        if s == 1:
            out_shape = in_shape
        else:
            dims, ks, s, d, shape_spatial = ConvPadNd._process_args(
                in_shape, ks, s, d)
            out_shape = in_shape[:2]
            for i, L in enumerate(shape_spatial):
                pad_total = pad[2*i] + pad[2*i + 1]
                out_shape += (ConvPadNd.compute_out_amount(
                    L, ks[i], s[i], d[i], pad_total),)
        out_shape = list(out_shape)
        if ch_out is None:
            ch_out = in_shape[1]  # ch_in
        out_shape[1] = ch_out
        return tuple(out_shape)

    @staticmethod
    def compute_out_amount(L, ks, s, d, pad_total):
        return int(math.floor((L + pad_total - d * (ks - 1) - 1) / s + 1))

    @staticmethod
    def _process_args(in_shape, ks, s, d):
        dims = len(in_shape) - 2
        if isinstance(ks, int):
            ks = dims * (ks,)
        if isinstance(s, int):
            s = dims * (s,)
        if isinstance(d, int):
            d = dims * (d,)
        shape_spatial = in_shape[-dims:]
        return dims, ks, s, d, shape_spatial


def ashape(self, a, b):
    if self.sanity_checks:
        a = tuple(a.shape)
        if isinstance(b, nn.Sequential) and not hasattr(b, 'out_shape'):
            b = b[-1].out_shape
        else:
            b = b.out_shape
        a = tuple([int(g) for g in a])
        assert a == b, (a, b)
        self.n_sanity_checked += 1
        if self.n_sanity_checked >= self.n_sanity_checks:
            self.sanity_checks = False


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)

