import torch.nn as nn

from ._bricks import create_module_list
from ..modules.core import SumLayer, ConcatLayer
from .init import _initialize_weights


class BasicModel(nn.Module):
    """Basic model for many use cases such as classification"""

    def __init__(self, module_defs, input_sz, init_weights=True):
        super(BasicModel, self).__init__()
        self.module_list = create_module_list(module_defs, input_sz)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        _initialize_weights(self)

    def forward(self, x):
        layer_outputs = []
        for i, module in enumerate(self.module_list):
            if (type(module) == SumLayer) or \
               (type(module) == ConcatLayer):
                x = module(layer_outputs)
            else:
                x = module(x)
            layer_outputs.append(x)
        # return last as an output
        return x
