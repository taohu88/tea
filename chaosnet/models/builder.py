from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from modules.darkmodules import *


def make_activation(act_name, module_def):
    if act_name == "relu":
        act = nn.ReLU(True)
    # Darknet by default use 0.1
    elif act_name == "leaky":
        leaky_slope = 0.1
        if "leaky_slope" in module_def:
            leaky_slope = float(module_def["leaky_slope"])
        act = nn.LeakyReLU(leaky_slope, inplace=True)
    elif act_name == "linear":
        act = Identity()
    else:
        raise Exception(f"Unknown {activation} in {module_def}")
    return act


def make_dropout_layer(module_def, input_sz=None, layer_num=None):
    prob = float(0.5) if "probability" not in module_def else float(module_def["probability"])
    return nn.Dropout(prob), None


def make_fc_layer(module_def, input_sz, layer_num=None):
    out_sz = int(module_def["output"])
    act = make_activation(module_def["activation"], module_def)

    module = nn.Sequential(
        nn.Linear(input_sz, out_sz),
        act)
    return module, out_sz


def make_conv_layer(module_def, input_sz, layer_num=None):
    bn = int(module_def["batch_normalize"])
    filters = int(module_def["filters"])
    kernel_size = int(module_def["size"])
    stride = int(module_def["stride"])
    pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
    act = make_activation(module_def["activation"], module_def)

    if bn:
        module = nn.Sequential(
            nn.Conv2d(input_sz, filters, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(filters, momentum=0.01),
            act
        )
    else:
        module = nn.Sequential(
            nn.Conv2d(input_sz, filters, kernel_size, stride, pad, bias=True),
            act
        )
    return module, filters


def make_maxpool_layer(module_def, input_sz=None, layer_num=None):
    kernel_size = int(module_def["size"])
    stride = int(module_def["stride"])
    if kernel_size == 2 and stride == 1:
        # I don't understand this bull shit of padding
        raise Exception(f"Meeting strange kernel size {module_def}")
        padding = (0, 1, 0, 1)
    else:
        padding = (0, 0, 0, 0)
    module = PaddedMaxPool2d(
        kernel_size=int(module_def["size"]),
        stride=int(module_def["stride"]),
        padding=padding
    )
    return module, None


def make_upsample_layer(module_def, input_sz=None, layer_num=None):
    module = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
    return module, None


def make_route_layer(module_def, input_sz, layer_num=None):
    layers = [int(x) for x in module_def["layers"].split(",")]
    filters = sum([input_sz[layer_i] for layer_i in layers])
    module = ConcatLayer(layers)
    return module, filters


def make_sum_layer(module_def, input_sz, layer_num=None):
    froms = [-1, int(module_def["from"])]
    filters = input_sz[int(module_def["from"])]
    module = SumLayer(froms)
    return module, filters


def make_yolo3_layer(module_def, input_sz, layer_num, **hyperparams):
    anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
    # Extract anchors
    anchors = [int(x) for x in module_def["anchors"].split(",")]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in anchor_idxs]
    num_classes = int(module_def["classes"])
    img_height = int(hyperparams["height"])
    # Define detection layer
    module = YOLO3Layer(anchors, num_classes, img_height)
    return module, None


_BUILDERS_ = {
    "convolutional": make_conv_layer,
    "connected": make_fc_layer,
    "maxpool": make_maxpool_layer,
    "upsample": make_upsample_layer,
    "route": make_route_layer,
    "shortcut": make_sum_layer,
    "yolo": make_yolo3_layer,
}

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        if module_def["type"] == "convolutional":
            creat_fun = _BUILDERS_[module_def["type"]]
            module, filters = creat_fun(module_def, output_filters[-1], i)

        elif module_def["type"] == "maxpool":
            creat_fun = _BUILDERS_[module_def["type"]]
            module, _ = creat_fun(module_def, output_filters[-1], i)

        elif module_def["type"] == "upsample":
            creat_fun = _BUILDERS_[module_def["type"]]
            module, _ = creat_fun(module_def, output_filters[-1], i)

        elif module_def["type"] == "route":
            creat_fun = _BUILDERS_[module_def["type"]]
            module, filters = creat_fun(module_def, output_filters, i)

        elif module_def["type"] == "shortcut":
            creat_fun = _BUILDERS_[module_def["type"]]
            module, _ = creat_fun(module_def, output_filters, i)

        elif module_def["type"] == "yolo":
            creat_fun = _BUILDERS_[module_def["type"]]
            module, _ = creat_fun(module_def, output_filters[-1], i, **hyperparams)

        # Register module list and number of output filters
        module_list.append(module)
        output_filters.append(filters)

    return hyperparams, module_list
