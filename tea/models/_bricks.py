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


def get_input_size(module_def):
    sizes_str = module_def[ModuleEnum.size]
    sizes_str = re.split(r'\s*x\s*', sizes_str)
    sizes = []
    for s in sizes_str:
        if s == "None":
            sizes.append(None)
        else:
            sizes.append(int(s))
    return tuple(sizes)


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


def make_fc_layer(module_def, in_sizes, layer_num=None, reshape=None, override_sz=None):
    prev_out_sz = in_sizes[-1]
    out_sz = override_sz if override_sz else int(module_def[ModuleEnum.size])
    act = make_activation(module_def)

    # check whether we need to do reshape
    start_dim = 0
    for i in range(len(prev_out_sz)):
        if prev_out_sz[i] is not None:
            start_dim = i
            break

    reshape = reshape if reshape is not None else is_true(module_def, ModuleEnum.reshape)
    if reshape:
        # start with start dim such as 1 for (Batch, F/C, H, W)
        in_sz = reduce(lambda x, y: x * y, prev_out_sz[start_dim:])
        module_l = [SmartLinear(in_sz, out_sz, end_dim=start_dim)]
        out_dim = list(prev_out_sz[:start_dim]) + [out_sz]
    else:
        in_sz = prev_out_sz[-1]
        module_l = [nn.Linear(in_sz, out_sz)]
        out_dim = list(prev_out_sz)
        out_dim[-1] = out_sz

    if act:
        module_l.append(act)

    module = nn.Sequential(*module_l)
    # Batch will be in the first dimension
    return module, tuple(out_dim)


def make_conv2d_layer(module_def, in_sizes, layer_num=None, override_sz=None):
    prev_out_sz = in_sizes[-1]
    in_filters = prev_out_sz[1]
    bn = has_batch_normalize(module_def)
    filters = override_sz if override_sz else int(module_def[ModuleEnum.filters])
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
            module, prev_out_sz = make_conv2d_layer(module_def, [prev_out_sz], layer_num=None, override_sz=filter_size)
            layers += [module]
            prev_out_sz = list(prev_out_sz)

    return nn.Sequential(*layers), tuple(prev_out_sz)


def make_quick_fc(module_def, in_sizes, layer_num=None):
    """
    A general method for making conv layer from string
    look like '64, 'D', 128,
    where D mean drop out
    :param module_def: module_def is a dictionary
    :param in_sizes: input sizes from previous layers
    :param layer_num: optional, what is the layer num of it
    :return : module, and its output size
    """
    prev_out_sz = list(in_sizes[-1])
    sizes_str = [x.strip() for x in module_def[ModuleEnum.size].split(',')]

    layers = []
    reshape = module_def[ModuleEnum.reshape]
    for v in sizes_str:
        if v == 'D':
            module, prev_out_sz = make_dropout_layer(module_def, [prev_out_sz], layer_num=None)
            layers += [module]
        else:
            out_sz = int(v)
            # only need to reshape in first time
            module, prev_out_sz = make_fc_layer(module_def, [prev_out_sz],
                                                layer_num=None, override_sz=out_sz, reshape=reshape)
            layers += [module]
            reshape = False

    return nn.Sequential(*layers), prev_out_sz


def make_embedding_layer(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    ninp = prev_out_sz[-1]
    size = int(module_def[ModuleEnum.size])
    encoder = nn.Embedding(ninp, size)

    out_sz = list(prev_out_sz)
    out_sz[-1] = size
    return encoder, tuple(out_sz)


def make_rnn_layers(module_def, in_sizes, layer_num=None):
    prev_out_sz = in_sizes[-1]
    ninp = prev_out_sz[-1]
    cell = module_def[ModuleEnum.cell]
    nhid = int(module_def[ModuleEnum.size])
    nlayers = get_int(module_def, ModuleEnum.nlayers, fallback=1)
    prob = float(0.5) if ModuleEnum.prob not in module_def else float(module_def[ModuleEnum.prob])

    if cell in ['LSTM', 'GRU']:
        rnn = getattr(nn, cell)(ninp, nhid, nlayers, dropout=prob)
    else:
        try:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[cell]
        except KeyError:
            raise ValueError("""An invalid option for `--model` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=prob)

    out_sz = list(prev_out_sz)
    out_sz[-1] = nhid
    return rnn, tuple(out_sz)


_BUILDERS_ = {
    ModuleEnum.quickfcs: make_quick_fc,
    ModuleEnum.quickconvs: make_quick_convs,
    ModuleEnum.conv2d: make_conv2d_layer,
    ModuleEnum.fc: make_fc_layer,
    ModuleEnum.dropout: make_dropout_layer,
    ModuleEnum.embedding: make_embedding_layer,
    ModuleEnum.rnn: make_rnn_layers,
    ModuleEnum.maxpool: make_maxpool2d_layer,
    ModuleEnum.upsample: make_upsample_layer,
    ModuleEnum.route: make_route_layer,
    ModuleEnum.shortcut: make_sum_layer,
}


def _get_builders():
    return _BUILDERS_


def create_module_list(module_defs, input_sz, context=None):
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
