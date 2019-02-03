from __future__ import division
import fire
from pathlib import Path

import torch
from torchvision import transforms
from tea.config.helper import parse_cfg, print_cfg, get_epochs, get_data_in_dir, get_model_out_dir, get_device
import tea.data.data_loader_factory as DLFactory
import tea.models.factory as MFactory
from tea.trainer.base_learner import find_max_lr, build_trainer
from tea.plot.commons import plot_lr_losses

from tea.data.tiny_imageset import TinyImageSet
import matplotlib.pyplot as plt

from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.train import lr_find, fit_one_cycle
from fastai.vision import accuracy


def build_train_val_datasets(cfg, in_memory=False):
    data_in_dir = get_data_in_dir(cfg)

    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])

    val_aug = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(56)
    ])

    training_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        train_aug,
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        val_aug,
        transforms.ToTensor(),
        normalize
    ])

    train_ds = TinyImageSet(data_in_dir, 'train', transform=training_transform, in_memory=in_memory)
    valid_ds = TinyImageSet(data_in_dir, 'val', transform=valid_transform, in_memory=in_memory)
    return train_ds, valid_ds


"""
Like anything in life, it is good to follow pattern.
In this case, any application starts with cfg file, 
with optional override arguments like the following: 
    data_dir/path
    model_cfg
    model_out_dir
    epochs, lr, batch etc
"""
def run(ini_file='tinyimg.ini',
        data_in_dir='./../../dataset',
        model_cfg='../cfg/vgg-tiny.cfg',
        model_out_dir='./models',
        epochs=30,
        lr=1.0e-5,
        batch_sz=256,
        num_worker=4,
        log_freq=20,
        use_gpu=True):
    # Step 1: parse config
    cfg = parse_cfg(ini_file,
                    data_in_dir=data_in_dir,
                    model_cfg=model_cfg,
                    model_out_dir=model_out_dir,
                    epochs=epochs, lr=lr, batch_sz=batch_sz, log_freq=log_freq,
                    num_worker=num_worker, use_gpu=use_gpu)
    print_cfg(cfg)

    # Step 2: create data sets and loaders
    train_ds, val_ds = build_train_val_datasets(cfg, in_memory=False)
    train_loader, val_loader = DLFactory.create_train_val_dataloader(cfg, train_ds, val_ds)

    # Step 3: create model
    model = MFactory.create_model(cfg)

    # Step 4: train/valid
    learner = build_trainer(cfg, model, train_loader, val_loader)

    # Step 5: optionally find the best lr
    # lr = find_max_lr(learner, train_loader)/10.0
    # print(f"Ideal learning rate {lr}")

    # path = get_model_out_dir(learner.cfg)
    # path = Path(path) / 'lr_tmp.pch'
    # lrs = []
    # r = learner.find_lr(train_loader, start_lr=1.0e-7, end_lr=1.0, batches=100, path=path)
    # plot_lr_losses(r.lr_losses[10:-5])
    # lrs.append(r.get_lr_with_min_loss()[0])
    # lr = sum(lrs) / len(lrs)
    # print('AAA', lrs)
    # print('lr', lr)
    # plt.show()

    # epochs = get_epochs(cfg)
    # learner.fit(train_loader, val_loader, epochs=epochs, lr=lr)

    device = get_device(cfg)
    data = DataBunch(train_loader, val_loader, device=device)
    learn = Learner(data, model, loss_func=torch.nn.CrossEntropyLoss(),
                    metrics=accuracy)
                  #  callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=2)])

    # lr_find(learn, start_lr=1e-7, end_lr=10)
    # learn.recorder.plot()
    # lrs_losses = [(lr, loss) for lr, loss in zip(learn.recorder.lrs, learn.recorder.losses)]
    # min_lr = min(lrs_losses[10:-5], key=lambda x: x[1])[0]
    # lr = min_lr/10.0
    # plt.show()
    # print(f'Minimal lr rate is {min_lr} propose init lr {lr}')
    # fit_one_cycle(learn, epochs, lr)

    learn.fit(epochs, lr)


if __name__ == '__main__':
    fire.Fire(run)

