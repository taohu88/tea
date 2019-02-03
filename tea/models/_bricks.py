"""
This is internal file acting as building block for creating models
It is not for external use
"""
import re
from enum import Enum

import re
import torch.nn as nn
from functools import reduce
from .cal_sizes import conv2d_out_shape
from ..modules.core import SmartLinear, SumLayer, ConcatLayer


class _ModuleEnum(Enum):
    type = "type"
    # valid for input
    size = "size"

    #valid input for conv/pad
    conv2d = "conv2d"
    kernel = "kernel"
    stride = "stride"
    has_pad = "has_pad"

    # valid input for fc
    fc = "fc"
    out_sz = "out_sz"
    reshape = "reshape"

    # valid enums for dropout
    dropout = "dropout"
    prob = "prob"

    # valid enums for upsample
    upsample = "upsample"
    mode = "mode"

    # valid enums for route
    route = "route"
    layers = "layers"

    shortcut = "shortcut"
    from_layer = "from"

    # valid enums for quickconvs
    quickconvs = "quickconvs"
    filters = "filters"
    pool_kernel = "pool_kernel"
    pool_stride = "pool_stride"
    pool_pad = "pool_pad"

    # has batch normal
    has_bn = "has_bn"
    momentum = "momentum"

    # valid actions
    activation = "activation"
    relu = "relu"
    leaky = "leaky"
    linear = "linear"
    leaky_slope = "leaky_slope"



def get_int(module_def, key):
    return int(module_def.get(key, 0))


def is_true(module_def, key):
    return get_int(module_def, key) > 0


def has_batch_normalize(module_def):
    return is_true(module_def, _ModuleEnum.has_bn)


def get_input_size(input_cfg):
    sizes_str = input_cfg[_ModuleEnum.size]
    sizes_str = sizes_str.split('x')

    sizes = [int(s) for s in sizes_str]
    sizes = [None] + sizes
    return tuple(sizes)


def make_activation(module_def):
    act_name = module_def.get(_ModuleEnum.activation, None)
    if act_name is None:
        act = None
    elif act_name == _ModuleEnum.relu.value:
        act = nn.ReLU(True)
    # Darknet by default use 0.1
    elif act_name == _ModuleEnum.leaky.value:
        leaky_slope = 0.1
        if _ModuleEnum.leaky_slope in module_def:
            leaky_slope = float(module_def[_ModuleEnum.leaky_slope])
        act = nn.LeakyReLU(leaky_slope, inplace=True)
    elif act_name == _ModuleEnum.linear:
        act = None
    else:
        raise Exception(f"Unknown {act_name} in {module_def}")
    return act


def make_dropout_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    prob = float(0.5) if _ModuleEnum.prob not in module_def else float(module_def[_ModuleEnum.prob])
    # same as last input size
    return nn.Dropout(prob), tuple(prev_out_sz)


def make_fc_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    out_sz = int(module_def[_ModuleEnum.out_sz])
    act = make_activation(module_def)
    # from last layer without batch dimension (Batch, F/C, H, W)
    in_sz = reduce(lambda x, y: x * y, prev_out_sz[1:])
    if is_true(module_def, _ModuleEnum.reshape):
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
    filters = int(module_def[_ModuleEnum.filters])
    kernel_size = int(module_def[_ModuleEnum.kernel])
    stride = int(module_def[_ModuleEnum.stride])
    pad = (kernel_size - 1) // 2 if is_true(module_def, _ModuleEnum.has_pad) else 0
    act = make_activation(module_def)

    module_l = [nn.Conv2d(in_filters, filters, kernel_size, stride, pad, bias= not bn)]
    if bn:
        momentum = float(module_def.get(_ModuleEnum.momentum, 0.01))
        module_l.append(nn.BatchNorm2d(filters, momentum=momentum))
    if act:
        module_l.append(act)

    module = nn.Sequential(*module_l)
    out_h, out_w = conv2d_out_shape(prev_out_sz[2:], kernel_size, stride, pad)
    return module, (None, filters, out_h, out_w)


def make_maxpool2d_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    kernel_size = int(module_def[_ModuleEnum.kernel])
    stride = int(module_def[_ModuleEnum.stride])
    pad = (kernel_size - 1) // 2 if is_true(module_def, _ModuleEnum.has_pad) else 0
    module = nn.MaxPool2d(kernel_size, stride, pad)
    out_h, out_w = conv2d_out_shape(prev_out_sz[2:], kernel_size, stride, pad)
    return module, (None, prev_out_sz[1], out_h, out_w)


def make_upsample_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    scale_factor = int(module_def[_ModuleEnum.stride])
    mode = module_def.get(_ModuleEnum.mode, "nearest")
    module = nn.Upsample(scale_factor=scale_factor, mode=mode)
    out_sz = [None, prev_out_sz[1]] + [i*scale_factor for i in prev_out_sz[2:]]
    return module, tuple(out_sz)


def make_route_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    layers = [int(x) for x in module_def[_ModuleEnum.layers].split(",")]
    filters = sum([in_sizes[layer_i][1] for layer_i in layers])
    module = ConcatLayer(layers)
    out_sz = [None, filters] + list(prev_out_sz[2:])
    return module, tuple(out_sz)


def make_sum_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    froms = [-1, int(module_def[_ModuleEnum.from_layer])]
    module = SumLayer(froms)
    return module, tuple(prev_out_sz)


def make_quick_convs(module_def, in_sizes, layer_num=None):
    """
    A general method for making conv layer from string
    look like '64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    :param module_def: module_def is a dictionary
    :param in_sizes: input sizes from previous layers
    :param layer_num: optional, what is the layer num of it
    :return : module, and its output size
    """
    prev_out_sz = in_sizes[-1]
    filters = [x.strip() for x in module_def[_ModuleEnum.filters].split(',')]

    layers = []
    in_channels = 3
    act = make_activation(module_def)

    for v in filters:
        if v == 'M':
            pool_kernel = module_def.get(_ModuleEnum.pool_kernel, 2)
            pool_stride = module_def.get(_ModuleEnum.pool_stride, 2)
            layers += [nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)]
        else:
            kernel_size = module_def.get(_ModuleEnum.kernel, 3)
            padding = (kernel_size - 1) // 2 if is_true(module_def, _ModuleEnum.has_pad) else 0
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=padding)
            if has_batch_normalize(module_def):
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)


_BUILDERS_ = {
    _ModuleEnum.quickconvs: make_quick_convs,
    _ModuleEnum.conv2d: make_conv2d_layer,
    _ModuleEnum.fc: make_fc_layer,
    _ModuleEnum.dropout: make_dropout_layer,
    _ModuleEnum.maxpool: make_maxpool2d_layer,
    _ModuleEnum.upsample: make_upsample_layer,
    _ModuleEnum.route: make_route_layer,
    _ModuleEnum.shortcut: make_sum_layer,
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
        type_name = re.split(r'-|_', module_def[_ModuleEnum.type])[0]
        creat_fun = builders[_ModuleEnum(type_name)]
        module, out_sz = creat_fun(module_def, output_sizes, i)

        # Save module_list, and output_sz
        module_list.append(module)
        output_sizes.append(out_sz)
        print(f"Layer {i} {module_def[_ModuleEnum.type]} in_sz {output_sizes[-2][1:]} out_sz {output_sizes[-1][1:]}")

    return module_list
