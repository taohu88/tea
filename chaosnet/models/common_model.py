from __future__ import division

import torch.nn as nn

from models.builder import create_module_list
from modules.darkmodules import SumLayer, ConcatLayer, YOLO3Layer


class CommonModel(nn.Module):
    """Common classification model"""

    def __init__(self, module_defs, input_sz):
        super(CommonModel, self).__init__()
        self.module_list = create_module_list(module_defs, input_sz)

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
