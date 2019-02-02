from __future__ import division

import torch.nn as nn

from .module_factory import create_module_list
from ..modules.core import SumLayer, ConcatLayer


class BasicModel(nn.Module):
    """Basic model for many use cases such as classification"""

    def __init__(self, module_defs, input_sz, init_weights=True):
        super(BasicModel, self).__init__()
        self.module_list = create_module_list(module_defs, input_sz)

        if init_weights:
            self._initialize_weights()            

    # TODO this is relu only, please make it more sutiable for leakyrelu
    def _initialize_weights(self):
        i = 0
        for m in self.modules():
            i += 1
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer_outputs = []
        for i, module in enumerate(self.module_list):
            if (type(module) == SumLayer) or \
               (type(module) == ConcatLayer):
                x = module(layer_outputs)
            elif (type(module)):
                x = module(x)
            layer_outputs.append(x)
        # return last as an output
        return x
