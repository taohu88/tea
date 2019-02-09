import warnings
import torch.nn as nn

from ..modules.core import SumLayer, ConcatLayer
from .init import _init_params


class BasicModel(nn.Module):
    """Basic model for many use cases such as classification"""

    def __init__(self, module_list, init_params=True):
        super(BasicModel, self).__init__()
        self.module_list = module_list

        if init_params:
            self.init_params()

        self.context=None

    def init_params(self, initrange=0.01):
        _init_params(self, initrange)

    def tie_weights(self):
        in_module = self.module_list[0]
        if isinstance(in_module, nn.Sequential):
            in_module = in_module[0]

        out_module = self.module_list[-1]
        if isinstance(out_module, nn.Sequential):
            out_module = out_module[0]

        if in_module.weight.size() != out_module.weight.size():
            warnings.warn(f"Try to tie input weights shape {in_module.weight.size()} with output weight shape {out_module.weight.size()}")
            return
        out_module.weight = in_module.weight

    def reset_context(self):
        self.context = None

    def forward(self, x):
        layer_outputs = []
        for i, module in enumerate(self.module_list):
            if isinstance(module, nn.RNNBase):
                x, self.context = module(x, self.context)
            elif isinstance(module, (SumLayer, ConcatLayer)):
                x = module(layer_outputs)
            else:
                x = module(x)
            layer_outputs.append(x)
        # return last as an output
        return x
