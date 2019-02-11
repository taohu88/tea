# -*- coding: utf-8 -*-
"""
Some common useful functions, which belong no particular package
"""
import torch


def islist(x):
    return isinstance(x, (list, tuple))


def istuple(x):
    return isinstance(x, tuple)


def self_or_first(x):
    if islist(x):
        return x[0]
    return x


def discouont_rewards(raw_rewards, gamma):
    """
    Discount rewards in episode
    :param raw_rewards: raw rewards
    :param gamma: discount rate
    :return: discounted rewards
    """
    R = 0
    rewards = []
    for r in reversed(raw_rewards):
        R = r + gamma * R
        rewards.append(R)
    rewards = rewards[::-1]
    return rewards


def detach_all(x):
    """
    detach all tensors from h
    :param x: tensor or a list/tuple of tensor
    :return:
    """
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif islist(x):
        return tuple(detach_all(v) for v in x)
    else:
        return x


#
# TODO make it generic
def export_rnn_onnx(path, model, device, batch_size, seq_len):
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    torch.onnx.export(model, dummy_input, path)



