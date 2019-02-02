import os
from enum import Enum
import configparser
import torch
import torch.nn.functional as F


#
# TODO should we think of wrap this in a config class
#

class CfgKeys(Enum):
    """
    This is used to enforce configuration keys are valids
    """
    data = "data"
    model = "model"
    hypers = "hypers"

    data_in_dir = "data_in_dir"
    model_cfg = "model_cfg"
    model_out_dir = "model_out_dir"
    epochs = "epochs"
    lr = "lr"
    momentum = "momentum"
    log_freq = "log_freq"
    use_gpu = "use_gpu"
    batch_sz = "batch_sz"
    train_batch_sz = "train_batch_sz"
    val_batch_sz = "val_batch_sz"
    test_batch_sz = "test_batch_sz"
    num_workers = "num_workers"


_loss_name_2_fn = {
    # TODO add more later
    "cross_entropy": F.cross_entropy,
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
    "binary_cross_entropy": F.binary_cross_entropy,
    "nll_loss": F.nll_loss
}


def print_cfg(cfg):
    print("Configurations:")
    for each_section in cfg.sections():
        print(f"[{each_section}]")
        for (k, v) in cfg.items(each_section):
            print(f"\t{k} = {v}")


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def get_data_in_dir(cfg):
    return cfg.get(CfgKeys.data.value, CfgKeys.data_in_dir.value)


def get_model_cfg(cfg):
    return cfg.get(CfgKeys.model.value, CfgKeys.model_cfg.value)


def get_model_out_dir(cfg, create_no_exists=True):
    model_out_dir = cfg.get(CfgKeys.model.value, CfgKeys.model_out_dir.value)
    if create_no_exists and (not os.path.exists(model_out_dir)):
        os.makedirs(model_out_dir)
    return model_out_dir


def merge_to_section(cfg, section, a_dict):
    # TODO convert to string is a little bit hack, but I don't know how to do it better
    for k, v in a_dict.items():
        cfg.set(section, str(k), str(v))


def parse_cfg(ini_file, **kwargs):
    config = configparser.ConfigParser()
    config.read(ini_file)
    merge_to_section(config, CfgKeys.hypers.value, kwargs)
    return config


def get_epochs(cfg):
    return cfg.getint(CfgKeys.hypers.value, CfgKeys.epochs.value)


def get_device(cfg):
    use_cuda = cfg.getboolean(CfgKeys.hypers.value, CfgKeys.use_gpu.value, fallback=False)
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    return None


#TODO do we need to refactor this out of here, like some factory class
def get_loss_fn(cfg):
    loss_name = cfg.get("model", "loss")
    return _loss_name_2_fn[loss_name]


def get_lr(cfg):
    return cfg['hypers'].getfloat('lr')


def get_momentum(cfg):
    return cfg['hypers'].getfloat('momentum')


def get_log_freq(cfg):
    return cfg['hypers'].getint('log_freq', -1)


def get_batch_sz(cfg):
    return cfg['hypers'].getint('batch_sz')


def get_train_batch_sz(cfg):
    return cfg['hypers'].getint('train_batch_sz', fallback=get_batch_sz(cfg))


def get_val_batch_sz(cfg):
    return cfg['hypers'].getint('val_batch_sz', fallback=get_batch_sz(cfg))


def get_test_batch_sz(cfg):
    return cfg['hypers'].getint('test_batch_sz', fallback=get_batch_sz(cfg))


def get_num_workers(cfg):
    return cfg['hypers'].getint('num_workers', 1)
