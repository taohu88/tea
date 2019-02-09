from enum import Enum


class ModuleEnum(Enum):
    type = "type"
    # valid for input
    size = "size"

    #valid input for conv/pad
    conv2d = "conv2d"
    kernel = "kernel"
    stride = "stride"
    has_pad = "has_pad"

    # valid input for embedding
    embedding = "embedding"

    # rnn
    rnn = "rnn"
    # rnn cell type
    cell = "cell"
    nlayers = "nlayers"

    # valid input for fc
    quickfcs = "quickfcs"
    fc = "fc"
    reshape = "reshape"

    # maxpool
    maxpool = "maxpool"

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
    has_pool_pad = "has_pool_pad"

    # has batch normal
    has_bn = "has_bn"
    momentum = "momentum"

    # valid actions
    activation = "activation"
    relu = "relu"
    leaky = "leaky"
    linear = "linear"
    leaky_slope = "leaky_slope"
