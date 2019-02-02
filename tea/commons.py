# -*- coding: utf-8 -*-
"""
Some common useful functions, which belong no particular package
"""


def islist(x):
    return isinstance(x, (list, tuple))


def istuple(x):
    return isinstance(x, tuple)


def self_or_first(x):
    if islist(x):
        return x[0]
    return x


def is_loss_too_large(l):
    return l > 10000.0
