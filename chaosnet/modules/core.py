#
#   Core Module
#   Copyright Tao Hu
#

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SmartLinear', 'Conv2dBatchReLU', 'GlobalAvgPool2d',
           'Identity', 'SumLayer', 'ConcatLayer', 'PaddedMaxPool2d']
log = logging.getLogger(__name__)


class SmartLinear(nn.Linear):
    r"""Applies a linear transformation to the incoming dataset: :math:`y = xA^T + b`
        It only automatically set input as (Batch, all-dimension)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SmartLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.linear(x, self.weight, self.bias)


class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.01**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.LeakyReLU`

    Note:
        If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:

        >>> conv = ln.layer.Conv2dBatchReLU(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.LeakyReLU, 0.1, inplace=True)
        ... )   # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 relu=lambda: nn.LeakyReLU(0.1, inplace = True), batch_normalize=1, momentum=0.01):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_normalize = batch_normalize

        # Layer
        if batch_normalize:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
                nn.BatchNorm2d(self.out_channels, momentum=momentum),
                relu()
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
                relu()
            )


    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, batch_normalize={batch_normalize} relu={relu})'
        return s.format(name=self.__class__.__name__,
                        relu=self.layers[2 if self.batch_normalize else 1],
                        **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class Identity(nn.Module):
    """ This is used for linear activation
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(B, C)
        return x


class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """
    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}'

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.kernel_size, self.stride, 0, self.dilation)
        return x


class SumLayer(nn.Module):
    """SumLayer used to implement shortcut layer or residual layer"""

    def __init__(self, froms):
        super(SumLayer, self).__init__()
        self.froms = froms

    def forward(self, prev_outputs):
        f = self.froms[0]
        x = prev_outputs[f]
        for f in self.froms[1:]:
            x = x + prev_outputs[f]
        return x

    def __repr__(self):
        s = '{name}({froms})'
        return s.format(name=self.__class__.__name__, froms=self.froms)


class ConcatLayer(nn.Module):
    """ConcatLayer concate outputs from several layers.
    It is used to implement route in darknet
    """

    def __init__(self, froms):
        super(ConcatLayer, self).__init__()
        self.froms = froms

    def forward(self, prev_outputs):
        x = torch.cat([prev_outputs[i] for i in self.froms], 1)
        return x
