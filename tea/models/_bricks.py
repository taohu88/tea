"""
This is internal file acting as building block for creating models
It is not for external use
"""
import re
from functools import reduce
import torch.nn as nn

from ..config.module_enum import ModuleEnum
from ..config.app_cfg import get_int, is_true
from .cal_sizes import conv2d_out_shape
from ..modules.core import Conv2dBatchReLU, SmartLinear, SumLayer, ConcatLayer


def has_batch_normalize(module_def):
    return is_true(module_def, ModuleEnum.has_bn)


def make_activation(module_def):
    act_name = module_def.get(ModuleEnum.activation, None)
    if act_name is None:
        act = None
    elif act_name == ModuleEnum.relu.value:
        act = nn.ReLU(True)
    # Darknet by default use 0.1
    elif act_name == ModuleEnum.leaky.value:
        leaky_slope = 0.1
        if ModuleEnum.leaky_slope in module_def:
            leaky_slope = float(module_def[ModuleEnum.leaky_slope])
        act = nn.LeakyReLU(leaky_slope, inplace=True)
    elif act_name == ModuleEnum.linear.value:
        act = None
    else:
        raise Exception(f"Unknown {act_name} in {module_def}")
    return act


def make_dropout_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    prob = float(0.5) if ModuleEnum.prob not in module_def else float(module_def[ModuleEnum.prob])
    # same as last input size
    return nn.Dropout(prob), tuple(prev_out_sz)


def make_fc_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    out_sz = int(module_def[ModuleEnum.size])
    act = make_activation(module_def)
    # from last layer without batch dimension (Batch, F/C, H, W)
    in_sz = reduce(lambda x, y: x * y, prev_out_sz[1:])
    if is_true(module_def, ModuleEnum.reshape):
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
    filters = int(module_def[ModuleEnum.filters])
    kernel_size = int(module_def[ModuleEnum.kernel])
    stride = int(module_def[ModuleEnum.stride])
    pad = (kernel_size - 1) // 2 if is_true(module_def, ModuleEnum.has_pad) else 0
    act = make_activation(module_def)
    # it is ok for fc not have activation layer, but not this conv2d
    if not act:
        act = nn.ReLU(True)

    momentum = float(module_def.get(ModuleEnum.momentum, 0.01)) if bn else 0.01
    module = Conv2dBatchReLU(in_filters, filters, kernel_size, stride, pad, act, bn, momentum)
    out_h, out_w = conv2d_out_shape(prev_out_sz[2:], kernel_size, stride, pad)
    return module, (None, filters, out_h, out_w)


def make_maxpool2d_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    kernel_size = int(module_def[ModuleEnum.kernel])
    stride = int(module_def[ModuleEnum.stride])
    pad = (kernel_size - 1) // 2 if is_true(module_def, ModuleEnum.has_pad) else 0
    module = nn.MaxPool2d(kernel_size, stride, pad)
    out_h, out_w = conv2d_out_shape(prev_out_sz[2:], kernel_size, stride, pad)
    return module, (None, prev_out_sz[1], out_h, out_w)


def make_upsample_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    scale_factor = int(module_def[ModuleEnum.stride])
    mode = module_def.get(ModuleEnum.mode, "nearest")
    module = nn.Upsample(scale_factor=scale_factor, mode=mode)
    out_sz = [None, prev_out_sz[1]] + [i*scale_factor for i in prev_out_sz[2:]]
    return module, tuple(out_sz)


def make_route_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    layers = [int(x) for x in module_def[ModuleEnum.layers].split(",")]
    filters = sum([in_sizes[layer_i][1] for layer_i in layers])
    module = ConcatLayer(layers)
    out_sz = [None, filters] + list(prev_out_sz[2:])
    return module, tuple(out_sz)


def make_sum_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    froms = [-1, int(module_def[ModuleEnum.from_layer])]
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
    prev_out_sz = list(in_sizes[-1])
    filters_str = [x.strip() for x in module_def[ModuleEnum.filters].split(',')]

    layers = []
    for v in filters_str:
        if v == 'M':
            pool_kernel = get_int(module_def, ModuleEnum.pool_kernel, 2)
            pool_stride = get_int(module_def, ModuleEnum.pool_stride, 2)
            has_pool_pad = is_true(module_def, ModuleEnum.has_pool_pad)
            padding = (pool_kernel - 1) // 2 if has_pool_pad else 0

            layers += [nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=padding)]
            # update the HxW
            prev_out_sz[2], prev_out_sz[3] = conv2d_out_shape(prev_out_sz[2:], pool_kernel, pool_stride, padding)
        else:
            filter_size = int(v)
            kernel_size = get_int(module_def, ModuleEnum.kernel, 3)
            padding = (kernel_size - 1) // 2 if is_true(module_def, ModuleEnum.has_pad) else 0
            stride = get_int(module_def, ModuleEnum.stride, 1)
            bn = has_batch_normalize(module_def)
            momentum = float(module_def.get(ModuleEnum.momentum, 0.01)) if bn else 0.01

            act = make_activation(module_def)
            # it is ok for fc not have activation layer, but not this conv2d
            if not act:
                act = nn.ReLU(True)

            layers += [Conv2dBatchReLU(prev_out_sz[1], filter_size, kernel_size, stride, padding, act, bn, momentum)]
            # update Channel/filter
            prev_out_sz[1] = filter_size

    return nn.Sequential(*layers), tuple(prev_out_sz)


_BUILDERS_ = {
    ModuleEnum.quickconvs: make_quick_convs,
    ModuleEnum.conv2d: make_conv2d_layer,
    ModuleEnum.fc: make_fc_layer,
    ModuleEnum.dropout: make_dropout_layer,
    ModuleEnum.maxpool: make_maxpool2d_layer,
    ModuleEnum.upsample: make_upsample_layer,
    ModuleEnum.route: make_route_layer,
    ModuleEnum.shortcut: make_sum_layer,
}


def _get_builders():
    return _BUILDERS_


def create_module_list(module_defs, input_sz):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    output_sizes = [input_sz]
    module_list = nn.ModuleList()
    builders = _get_builders()
    for i, module_def in enumerate(module_defs):
        type_name = re.split(r'-|_', module_def[ModuleEnum.type])[0]
        creat_fun = builders[ModuleEnum(type_name)]
        module, out_sz = creat_fun(module_def, output_sizes, i)

        # Save module_list, and output_sz
        module_list.append(module)
        output_sizes.append(out_sz)
        print(f"Layer {i} {module_def[ModuleEnum.type]} in_sz {output_sizes[-2][1:]} out_sz {output_sizes[-1][1:]}")

    return module_list
