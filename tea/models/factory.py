from ..config.parser import parse_model_config
from ._bricks import get_input_size, create_module_list
from .basic_model import BasicModel


def create_model(cfg, init_params=True):
    model_cfg = cfg.get_model_cfg()
    module_defs = parse_model_config(model_cfg, cfg.conf)
    input_cfg = module_defs.pop(0)
    input_sz = get_input_size(input_cfg)
    module_list = create_module_list(module_defs, input_sz)

    return BasicModel(module_list, init_params=init_params)
