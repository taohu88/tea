from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from modules.darkmodules import *


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            if module_def["activation"] == "relu":
                relu = lambda: nn.ReLU(inplace = True)
            else:
                relu = lambda: nn.LeakyReLU(0.1, inplace = True)
            module = Conv2dBatchReLU(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    relu=relu,
                    batch_normalize=bn,
                    momentum=0.01
                )
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = (0, 1, 0, 1)
            else:
                padding = (0, 0, 0, 0)
            module = PaddedMaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=padding
            )

        elif module_def["type"] == "upsample":
            module = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            module = ConcatLayer(layers)

        elif module_def["type"] == "shortcut":
            froms = [-1, int(module_def["from"])]
            filters = output_filters[int(module_def["from"])]
            module = SumLayer(froms)

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLO3Layer(anchors, num_classes, img_height)
            module = yolo_layer

        # Register module list and number of output filters
        module_list.append(module)
        output_filters.append(filters)

    return hyperparams, module_list
