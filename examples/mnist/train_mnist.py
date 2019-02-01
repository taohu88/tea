
import fire

from torchvision import datasets

from tea.vision.cv import transforms
from tea.config.helper import parse_cfg
from tea.data.helper import build_train_val_dataloader
from tea.models.basic_model import build_model
from tea.trainer.classifier import Classifier


def build_train_val_datasets(cfg):
    save_path = cfg.get('data', 'save_path')
    train_ds = datasets.MNIST(save_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    valid_ds = datasets.MNIST(save_path, train=False,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    return train_ds, valid_ds


def build_trainer(cfg, model, train_loader, val_loader):
    return Classifier(cfg, model, train_loader, val_loader)


def run(ini_file='mnist.ini', lr=0.01, batch_sz=256, log_freq=10, use_gpu=True):
    cfg = parse_cfg(ini_file, lr=lr, batch_sz=batch_sz, log_freq=log_freq, use_gpu=use_gpu)
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader, val_loader = build_train_val_dataloader(cfg, train_ds, val_ds)
    model = build_model(cfg)
    classifier = build_trainer(cfg, model, train_loader, val_loader)
    classifier.fit(10)


if __name__ == '__main__':
  fire.Fire(run)
