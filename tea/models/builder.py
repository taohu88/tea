from __future__ import division

import re
import torch.nn as nn
from functools import reduce
from .cal_sizes import conv2d_out_shape
from ..modules.core import SmartLinear, SumLayer, ConcatLayer
from ..yolo.yolo3_layer import YOLO3Layer


def get_int(module_def, key):
    return int(module_def.get(key, 0))


def is_true(module_def, key):
    return get_int(module_def, key) > 0


def has_batch_normalize(module_def):
    return is_true(module_def, "batch_normalize")


def get_input_size(hyperparams):
    """
    :param hyperparams:
    :return (Batch, Channel, H, W) as used in pytorch
    """
    c = int(hyperparams["channels"])
    h = int(hyperparams["crop_height"]) if "crop_height" in hyperparams else int(hyperparams["height"])
    w = int(hyperparams["crop_width"]) if "crop_width" in hyperparams else int(hyperparams["width"])
    return (None, c, h, w)


def make_activation(module_def):
    act_name = module_def.get("activation", None)
    if act_name is None:
        act = None
    elif act_name == "relu":
        act = nn.ReLU(True)
    # Darknet by default use 0.1
    elif act_name == "leaky":
        leaky_slope = 0.1
        if "leaky_slope" in module_def:
            leaky_slope = float(module_def["leaky_slope"])
        act = nn.LeakyReLU(leaky_slope, inplace=True)
    elif act_name == "linear":
        act = None
    else:
        raise Exception(f"Unknown {act_name} in {module_def}")
    return act


def make_dropout_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    prob = float(0.5) if "probability" not in module_def else float(module_def["probability"])
    # same as last input size
    return nn.Dropout(prob), tuple(prev_out_sz)


def make_fc_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    out_sz = int(module_def["output"])
    act = make_activation(module_def)
    # from last layer without batch dimension (Batch, F/C, H, W)
    in_sz = reduce(lambda x, y: x * y, prev_out_sz[1:])
    if is_true(module_def, "reshape"):
        module_l = [SmartLinear(in_sz, out_sz)]
    else:
        module_l = [nn.Linear(in_sz, out_sz)]

    if act:
        module_l.append(act)

    module = nn.Sequential(*module_l)
    # Batch will be in the first dimension
    return module, (None, out_sz)


def make_conv2d_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    in_filters = prev_out_sz[1]
    bn = has_batch_normalize(module_def)
    filters = int(module_def["filters"])
    kernel_size = int(module_def["size"])
    stride = int(module_def["stride"])
    pad = (kernel_size - 1) // 2 if is_true(module_def, "pad") else 0
    act = make_activation(module_def)

    module_l = [nn.Conv2d(in_filters, filters, kernel_size, stride, pad, bias= not bn)]
    if bn:
        module_l.append(nn.BatchNorm2d(filters, momentum=0.01))
    if act:
        module_l.append(act)

    module = nn.Sequential(*module_l)
    out_h, out_w = conv2d_out_shape(prev_out_sz[2:], kernel_size, stride, pad)
    return module, (None, filters, out_h, out_w)


def make_maxpool2d_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    kernel_size = int(module_def["size"])
    stride = int(module_def["stride"])
    pad = (kernel_size - 1) // 2 if is_true(module_def, "pad") else 0
    module = nn.MaxPool2d(kernel_size, stride, pad)
    out_h, out_w = conv2d_out_shape(prev_out_sz[2:], kernel_size, stride, pad)
    return module, (None, prev_out_sz[1], out_h, out_w)


def make_upsample_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    scale_factor = int(module_def["stride"])
    module = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    out_sz = [None, prev_out_sz[1]] + [i*scale_factor for i in prev_out_sz[2:]]
    return module, tuple(out_sz)


def make_route_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    layers = [int(x) for x in module_def["layers"].split(",")]
    filters = sum([in_sizes[layer_i][1] for layer_i in layers])
    module = ConcatLayer(layers)
    out_sz = [None, filters] + list(prev_out_sz[2:])
    return module, tuple(out_sz)


def make_sum_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    froms = [-1, int(module_def["from"])]
    module = SumLayer(froms)
    return module, tuple(prev_out_sz)


def make_yolo3_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
    # Extract anchors
    anchors = [int(x) for x in module_def["anchors"].split(",")]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in anchor_idxs]
    num_classes = int(module_def["classes"])
    # original img size
    img_height = in_sizes[0][2]
    # Define detection layer
    module = YOLO3Layer(anchors, num_classes, img_height)
    return module, tuple(prev_out_sz)


_BUILDERS_ = {
    "convolutional": make_conv2d_layer,
    "connected": make_fc_layer,
    "dropout": make_dropout_layer,
    "maxpool": make_maxpool2d_layer,
    "upsample": make_upsample_layer,
    "route": make_route_layer,
    "shortcut": make_sum_layer,
    "yolo": make_yolo3_layer,
}


def get_builders():
    return _BUILDERS_


def create_module_list(module_defs, input_sz):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    output_sizes = [input_sz]
    module_list = nn.ModuleList()
    builders = get_builders()
    for i, module_def in enumerate(module_defs):
        type_name = re.split(r'-|_', module_def["type"])[0]
        creat_fun = builders[type_name]
        module, out_sz = creat_fun(module_def, output_sizes, i)

        # Save module_list, and output_sz
        module_list.append(module)
        output_sizes.append(out_sz)
        print(f"Layer {i} {module_def['type']} in_sz {output_sizes[-2][1:]} out_sz {output_sizes[-1][1:]}")

    return module_list
