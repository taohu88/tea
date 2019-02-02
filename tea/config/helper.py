import configparser
import torch
import torch.nn.functional as F


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

def merge_to_section(cfg, section, a_dict):
    # TODO convert to string is a little bit hack, but I don't know how to do it better
    for k, v in a_dict.items():
        cfg.set(section, str(k), str(v))


def parse_cfg(ini_file, **kwargs):
    config = configparser.ConfigParser()
    config.read(ini_file)
    merge_to_section(config, 'hypers', kwargs)
    return config


def get_epochs(cfg):
    return cfg['hypers'].getint('epochs')


def get_device(cfg):
    use_cuda = cfg.getint('hypers', 'gpu_flag', fallback=0)
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    return None


#TODO refactor this out of here
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
