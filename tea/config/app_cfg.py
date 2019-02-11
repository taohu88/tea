import os
import torch

from .cfg_enum import CfgEnum
from .parser import parse_config
from .loss_fn_map import get_loss_fn_maps


def get_int(cfg, key, fallback=0):
    return int(cfg.get(key, fallback))


def is_true(cfg, key):
    return get_int(cfg, key) > 0


class AppConfig:

    def __init__(self, conf):
        self.conf = conf

    @classmethod
    def from_file(cls, path, **kwargs):
        conf = parse_config(path)
        conf.update(kwargs)
        return cls(conf)

    def update(self, **kwargs):
        self.conf.update(**kwargs)
        return self

    def print(self):
        print("Configurations:")
        for (k, v) in self.conf.items():
            print(f"\t{k} = {v}")

    def get_str(self, key, fallback=None):
        return self.conf.get(key, fallback)

    def get_data_in_dir(self):
        return self.conf.get(CfgEnum.data_in_dir.value)

    def get_model_cfg(self):
        return self.conf.get(CfgEnum.model_cfg.value)

    def get_model_out_dir(self, create_no_exists=True):
        model_out_dir = self.conf.get(CfgEnum.model_out_dir.value)
        if create_no_exists and (not os.path.exists(model_out_dir)):
            os.makedirs(model_out_dir)
        return model_out_dir

    def get_optim(self):
        return self.conf.get(CfgEnum.optim.value, "AdamW")

    def get_epochs(self):
        return int(self.conf[CfgEnum.epochs.value])

    def get_device(self):
        use_cuda = get_int(self.conf, CfgEnum.use_gpu.value, fallback=0)
        if use_cuda and torch.cuda.is_available():
            return "cuda"
        return None

    def get_loss_fn(self):
        loss_name = self.conf[CfgEnum.loss.value]
        return get_loss_fn_maps()[loss_name]

    def get_lr(self):
        return float(self.conf[CfgEnum.lr.value])

    def get_eps(self):
        return float(self.conf[CfgEnum.eps.value])

    def get_clip(self):
        return float(self.conf.get(CfgEnum.clip.value, -1))

    def get_momentum(self):
        return float(self.conf.get(CfgEnum.momentum.value, 0.0))

    def get_weight_decay(self):
        return float(self.conf.get(CfgEnum.weight_decay.value, 0.0))
    
    def get_log_freq(self):
        return get_int(self.conf, CfgEnum.log_freq.value, fallback=-1)

    def get_batch_sz(self):
        return int(self.conf[CfgEnum.batch_sz.value])

    def get_bptt(self):
        return get_int(self.conf, CfgEnum.bptt.value)

    def get_train_batch_sz(self):
        return int(self.conf.get(CfgEnum.train_batch_sz.value, self.get_batch_sz()))
    
    def get_val_batch_sz(self):
        return int(self.conf.get(CfgEnum.val_batch_sz.value, self.get_batch_sz()))

    def get_test_batch_sz(self):
        return int(self.conf.get(CfgEnum.test_batch_sz.value, self.get_batch_sz()))
    
    def get_num_workers(self):
        return get_int(self.conf, CfgEnum.num_workers.value, fallback=1)
