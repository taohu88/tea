import re
from .module_enum import ModuleEnum


def parse_config(path):
    conf = {}
    with open(path, 'r') as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue
            key, val = re.split(r'\s*=\s*', line)
            conf[key.strip()] = val.strip()
    return conf


def parse_model_value(value, context):
    """
    do interpolation first from context,
    "x is {size}" with size = 5 will be interpolated to "x is 5"
    then return interpolated string
    :param value:
    :param context:
    :return:
    """
    return value.format(**context)


def parse_model_config(path, context):
    module_defs = []
    with open(path, 'r') as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith('['):  # This marks the start of a new block
                module_defs.append({})
                module_defs[-1][ModuleEnum.type] = line[1:-1].strip()
            else:
                key, value = re.split(r'\s*=\s*', line)
                module_defs[-1][ModuleEnum(key.strip())] = parse_model_value(value.strip(), context)

    return module_defs


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names