import warnings
import torch.nn as nn


def _init_conv(conv, act):

    nonlinearity = 'relu'
    if isinstance(act, nn.ReLU):
        nonlinearity = 'relu'
    elif isinstance(act, nn.LeakyReLU):
        nonlinearity = 'leaky_relu'
    else:
        warnings.WarningMessage(f"Don't understand the activation {act}, use relu for init_conv")
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity=nonlinearity)


def _initialize_weights(model):
    prev_conv = None
    gap_to_act = 0
    for m in model.modules():
        if prev_conv:
            gap_to_act += 1
        if isinstance(m, nn.Conv2d):
            prev_conv = m
            gap_to_act = 0
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        else:
            if prev_conv:
                if gap_to_act > 2:
                    raise Exception(f"We can't initialize prev conv layer {prev_conv}")
                else:
                    _init_conv(prev_conv, m)
                # clear it up
                prev_conv = None
                gap_to_act = 0
            params_sz = len(list(m.parameters(recurse=False)))
            if params_sz > 0:
                raise Exception(f"We have a layer {m} not initialized")

