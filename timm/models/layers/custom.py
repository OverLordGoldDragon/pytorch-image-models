# -*- coding: utf-8 -*-
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Tuple, Callable
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
                              3: (0, 2, 3, 4, 1),
                              4: (0, 2, 3, 4, 5, 1)}[ndim]
        self.unpermute_shape = {1: (0, 2, 1),
                                2: (0, 3, 1, 2),
                                3: (0, 4, 1, 2, 3),
                                4: (0, 5, 1, 2, 3, 4)}[ndim]

    def expdim(self, x):
        if self.ndim == 1:
            return x[:, None]
        if self.ndim == 2:
            return x[:, None, None]
        if self.ndim == 3:
            return x[:, None, None, None]
        return x[:, None, None, None, None]

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


def _set_layer_builders(self, dims):
    assert dims in (1, 2, 3, 4), dims
    setattr(self, 'dims', dims)
    setattr(self, 'mean', lambda x: x.mean(tuple(range(-dims, 0))[::-1],
                                           keepdim=True))


def select_conv(in_shape, *args, **kwargs):
    if len(in_shape) >= 6:  # 4D+
        return convNd(in_shape, *args, **kwargs)
    return ConvPadNd(in_shape, *args, **kwargs)


class ConvPadNd(nn.Module):
    """If `stride=1`, implements 'same', else pads enough so that first
    kernel op centers at first point of input (`ceil(kernel_size/2)` on
    each side).
    """
    def __init__(self, in_shape, ch_out=None, kernel_size=3, stride=1,
                 groups=1, dilation=1, padding=None, bias=True):
        super().__init__()
        self.in_shape = in_shape
        self.ch_out = ch_out or in_shape[1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.bias = bias

        self.dims = len(in_shape) - 2  # minus `batch_size`, `channels`

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * (2 * self.dims)
            self.pad_shape = padding[::-1]
        else:
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

##############################################################################

class convNd(nn.Module):
    """N-dimensional convolution with "same" padding.

    Code modified from and based on
    https://github.com/pvjosue/pytorch_convNd/blob/master/convNd.py
    """
    def __init__(self, in_shape: Tuple,
                 ch_out = None,
                 kernel_size = 1,
                 stride = 1,
                 groups: int = 1,
                 dilation: int = 1,
                 padding=None,
                 bias: bool = True,
                 rank: int = 0,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(convNd, self).__init__()

        ndim = len(in_shape) - 2
        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, Tuple):
            kernel_size = (kernel_size,) * ndim
        if not isinstance(stride, Tuple):
            stride = tuple(stride for _ in range(ndim))
        if not isinstance(dilation, Tuple):
            dilation = tuple(dilation for _ in range(ndim))

        # This parameter defines which Pytorch convolution to use as a base,
        # for 3 Conv2D is used
        if rank == 0 and ndim <= 3:
            max_dims = ndim - 1
        else:
            max_dims = 3

        assert len(kernel_size) == ndim, (len(kernel_size), ndim)
        assert len(stride) == ndim, (len(stride), ndim)
        assert sum(dilation) == ndim, (dilation, ndim)

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.in_shape = in_shape
        self.in_channels = in_shape[1]
        self.ch_out = ch_out or in_shape[1]
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.rank = rank
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.ch_out))
        else:
            self.register_parameter('bias', None)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # compute padding #####################################################
        if padding is None:
            F_pad_padding = ConvPadNd.compute_pad_shape(
                self.in_shape, kernel_size, stride, dilation)
            padding = F_pad_padding[::-1]
        elif isinstance(padding, int):
            padding = (padding,) * (2 * self.ndim)
            F_pad_padding = padding[::-1]
        else:
            F_pad_padding = padding[::-1]
        self.padding = padding
        self.F_pad_padding = F_pad_padding

        self.out_shape = ConvPadNd.compute_out_shape(
            self.in_shape, self.kernel_size, self.stride, self.dilation,
            self.F_pad_padding, self.ch_out)

        self.conv_f = ConvPadNd

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------
        if self.use_bias and self.bias_initializer is not None:
            self.bias_initializer(self.bias)
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        # Compute the next dimension, so for a conv4D, get index 3
        next_dim_len = self.kernel_size[0]

        sub_in_shape = self.in_shape[:2] + self.in_shape[3:]
        for _ in range(next_dim_len):
            if self.ndim - 1 > max_dims:
                # Initialize a Conv_n-1_D layer
                conv_layer = convNd(in_shape=sub_in_shape,
                                    ch_out=self.ch_out,
                                    use_bias=self.use_bias,
                                    rank=self.rank - 1,
                                    kernel_size=self.kernel_size[1:],
                                    stride=self.stride[1:],
                                    groups=self.groups,
                                    dilation=self.dilation[1:],
                                    padding=0,#self.padding[2:],
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

            else:
                # Initialize a Conv layer
                # bias should only be applied by the top most layer,
                # so we disable bias in the internal convs

                # keep batch & ch dim
                conv_layer = self.conv_f(sub_in_shape,
                                         ch_out=self.ch_out,
                                         kernel_size=self.kernel_size[1:],
                                         stride=self.stride[1:],
                                         groups=self.groups,
                                         dilation=self.dilation[1:],
                                         padding=0,#self.padding[2:],
                                         bias=False)

                # Apply initializer functions to weight and bias tensor
                if self.kernel_initializer is not None:
                    self.kernel_initializer(conv_layer.conv.weight)

            # Store the layer
            self.conv_layers.append(conv_layer)

    # -------------------------------------------------------------------------

    def forward(self, x):
        # padding = list(self.padding)

        # Pad dim0 of x if this is the parent convolution ie rank=0
        if self.rank == 0:
            x = F.pad(x, self.F_pad_padding)
            # xShape = list(x.shape)
            # xShape[2] += (self.padding[0] + self.padding[1])
            # padSize = (0, 0, self.padding[0], self.padding[0])
            # padding[0] = 0  # since already padded (just now)
            # x = F.pad(x.view(x.shape[0], x.shape[1], x.shape[2], -1),
            #           padSize, 'constant', 0).view(xShape)

        # Define shortcut names for dimensions of x and kernel
        (b, c_i) = tuple(x.shape[:2])
        size_i = tuple(x.shape[2:])
        size_k = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        # size_o = tuple(math.floor(
        #     (size_i[i] + 2 * padding[i] - size_k[i]) / self.stride[i] + 1)
        #     for i in range(len(size_i)))
        # Compute size of the output without stride
        size_ons = tuple(size_i[i] - size_k[i] + 1 for i in range(len(size_i)))

        # Output tensors for each 3D frame
        # frame_results = torch.zeros((size_o[0], b, self.ch_out,
        #                              *size_o[1:]), device=x.device)
        frame_results = self.out_shape[2] * [
            torch.zeros(self.out_shape[:2] + self.out_shape[3:], device=x.device)]
        empty_frames = self.out_shape[2] * [None]
        # frame_results = size_o[0] * [
        #     torch.zeros((b, self.ch_out, *size_o[1:]), device=x.device)]
        # empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):
            # iterate input's first dimension
            for j in range(size_i[0]):

                # Add results to this output frame
                out_frame = (
                    j -
                    (i - size_k[0] // 2) -
                    (size_i[0] - size_ons[0]) // 2 -
                    (1 - size_k[0] % 2)
                )
                k_center_position = out_frame % self.stride[0]
                out_frame = math.floor(out_frame / self.stride[0])
                if k_center_position != 0:
                    continue

                if out_frame < 0 or out_frame >= self.out_shape[2]:#size_o[0]:
                    continue

                # Prepate x for next dimension
                conv_x = x.view(b, c_i, size_i[0], -1)
                conv_x = conv_x[:, :, j, :].view((b, c_i) + size_i[1:])

                # Convolve
                frame_conv = self.conv_layers[i](conv_x)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        result = torch.stack(frame_results, dim=2)
        # result = frame_results

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b, resultShape[1], -1)
            for k in range(self.ch_out):
                result[:, k, :] += self.bias[k]
            return result.view(resultShape)
        else:
            return result


class MaxPoolNd(nn.modules.pooling._MaxPoolNd):
    def __init__(self, in_shape, kernel_size, stride = None, dilation = 1,
                 return_indices = False, ceil_mode = False):
        super(MaxPoolNd, self).__init__(kernel_size, stride, 0, dilation,
                                        return_indices, ceil_mode)
        self.in_shape = in_shape
        self._ndim = len(self.in_shape) - 2
        self._torch_ndim = min(3, self.ndim)
        assert self.ndim != 1

        self.mp_op = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d
                      }[self.torch_ndim]

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * self.ndim
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = kernel_size
        elif isinstance(stride, int):
            self.stride = (stride,) * self.ndim
        else:
            self.stride = stride

        if self.ndim > 3:
            assert self.stride[0] == 1 and self.kernel_size[0] == 1, (
                self.stride, self.kernel_size)

        if isinstance(dilation, int):
            self.dilation = (dilation,) * self.ndim
        else:
            self.dilation = dilation

        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        # compute pad_shape & out_shape ######################################
        self.pad_shape = ConvPadNd.compute_pad_shape(
            in_shape, kernel_size, stride, dilation)

        self.out_shape = ConvPadNd.compute_out_shape(
            in_shape, kernel_size, stride, dilation, self.pad_shape)

    @property
    def ndim(self):
        return self._ndim

    @property
    def torch_ndim(self):
        return self._torch_ndim

    def forward(self, input):
        input = F.pad(input, self.pad_shape)
        if self.ndim > 3:
            input = input.view(input.shape[0], -1,
                               *input.shape[-self.torch_ndim:])
        out = self.mp_op(input, self.kernel_size[-self.torch_ndim:],
                         self.stride[-self.torch_ndim:], 0,
                         self.dilation[-self.torch_ndim:], self.ceil_mode,
                         self.return_indices)

        if self.ndim > 3:
            out = out.view(*self.out_shape)
        return out


class BatchNormNd(_BatchNorm):
    def forward(self, input):
        if input.ndim >= 6:
            return _BatchNorm.forward(self, input.view(*input.shape[:2], -1)
                                      ).view(*input.shape)
        return _BatchNorm.forward(self, input)

    r"""Ndim BN"""
    def _check_input_dim(self, input):
        pass


class GlobalAveragePooling(nn.Module):
    def __init__(self, flatten=True):
        super(GlobalAveragePooling, self).__init__()
        self.flatten = flatten
        self.keepdim = bool(not self.flatten)

    def forward(self, x):
        return x.mean(dim=tuple(range(2, x.ndim)), keepdim=self.keepdim)
