from ..config.helper import get_model_cfg
from .parser import parse_model_config
from .module_factory import get_input_size
from .basic_model import BasicModel


def create_model(cfg):
    model_cfg = get_model_cfg(cfg)
    module_defs = parse_model_config(model_cfg)
    input_cfg = module_defs.pop(0)
    input_sz = get_input_size(input_cfg)

    return BasicModel(module_defs, input_sz)
