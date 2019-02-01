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


def merge_to_section(cfg, section, a_dict):
    return cfg._unify_values(section, a_dict)


def parse_cfg(ini_file, **kwargs):
    config = configparser.ConfigParser()
    config.read(ini_file)
    merge_to_section(config, 'hypers', kwargs)
    return config



def get_device(cfg):
    use_cuda = cfg['hypers'].getboolean('use_gpu', False)
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
    return cfg['hypers'].getint('log_freq')


def get_batch_sz(cfg):
    return cfg['hypers'].getint('batch_sz')


def get_train_batch_sz(cfg):
    return cfg['hypers'].getint('train_batch_sz', fallback=get_batch_sz(cfg))


def get_val_batch_sz(cfg):
    return cfg['hypers'].getint('val_batch_sz', fallback=get_batch_sz(cfg))


def get_test_batch_sz(cfg):
    return cfg['hypers'].getint('test_batch_sz', fallback=get_batch_sz(cfg))


def get_num_workers(cfg):
    return cfg['data'].getint('num_workers', 1)


#TODO move it out of here
def find_min_lr(lrs, losses, skip_start=10, skip_end=5):
    lrs_t = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
    losses_t = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]

    _, idx = min((val, idx) for (idx, val) in enumerate(losses_t))

    return lrs_t[idx]