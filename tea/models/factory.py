from ..config.parser import parse_model_config
from tea.config.module_enum import ModuleEnum
from .basic_model import BasicModel


def get_input_size(module_def):
    sizes_str = module_def[ModuleEnum.size]
    sizes_str = sizes_str.split('x')

    sizes = [int(s) for s in sizes_str]
    sizes = [None] + sizes
    return tuple(sizes)


def create_model(cfg):
    model_cfg = cfg.get_model_cfg()
    module_defs = parse_model_config(model_cfg)
    input_cfg = module_defs.pop(0)
    input_sz = get_input_size(input_cfg)

    return BasicModel(module_defs, input_sz)
