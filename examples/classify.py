from __future__ import division

import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from chaos.models.builder import get_input_size
from chaos.models.core import BasicModel
from chaos.dataset.tiny_imageset import TinyImageSet
from chaos.models.cfg_parser import parse_data_config, parse_model_config
from chaos.trainer.core import find_min_lr


from fastai.script import call_parse
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.train import lr_find, fit_one_cycle
from fastai.vision import accuracy
from fastai.callbacks.tracker import EarlyStoppingCallback

import matplotlib.pyplot as plt


@call_parse
def main(cfg_file='../cfg/my-vgg.cfg',
         cfg_data='../cfg/tinyimage.data',
         batch_size=256, num_worker=4, cuda=True):
    os.makedirs('output', exist_ok=True)

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    module_defs = parse_model_config(cfg_file)
    hyperparams = module_defs.pop(0)
    input_sz = get_input_size(hyperparams)

    # Set up model
    model = BasicModel(module_defs, input_sz)
    model = model.to(device)

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

    in_memory = True
    # Get dataset configuration
    data_config = parse_data_config(cfg_data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]

    training_set = TinyImageSet(train_path, 'train', transform=training_transform, in_memory=in_memory)
    valid_set = TinyImageSet(valid_path, 'val', transform=valid_transform, in_memory=in_memory)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_worker)


    data = DataBunch(train_loader, test_loader, device=device)
    learn = Learner(data, model, loss_func=torch.nn.CrossEntropyLoss(),
                    metrics=accuracy)
                  #  callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=2)])

    lr_find(learn)
    # learn.recorder.plot()
    # plt.show()
    min_lr = find_min_lr(learn.recorder.lrs, learn.recorder.losses)
    lr = min_lr/10.0
    print(f'Minimal lr rate is {min_lr} propose init lr {lr}')
    fit_one_cycle(learn, 30, lr)
