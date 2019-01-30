from functools import partial
import torch
from torchvision import datasets, transforms

from models.builder import get_input_size
from models.core import BasicModel
from models.cfg_parser import parse_model_config
from engine.core import find_min_lr

from fastai.script import call_parse
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.train import fit_one_cycle, lr_find
from fastai.vision import accuracy
from fastai.callbacks.tracker import EarlyStoppingCallback

import matplotlib.pyplot as plt


@call_parse
def main(cfg_file='../cfg/lecnn.cfg', batch_size=256, cuda=True):

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    module_defs = parse_model_config(cfg_file)
    hyperparams = module_defs.pop(0)
    input_sz = get_input_size(hyperparams)

    # Set up model
    model = BasicModel(module_defs, input_sz)
    model = model.to(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../dataset', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../dataset', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    data = DataBunch(train_loader, test_loader, device=device)
    learn = Learner(data, model, loss_func=torch.nn.CrossEntropyLoss(),
                    metrics=accuracy,
                    callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=2)])

    lr_find(learn)
    # learn.recorder.plot()
    # plt.show()
    min_lr = find_min_lr(learn.recorder.lrs, learn.recorder.losses)
    lr = min_lr/10.0
    print(f'Minimal lr rate is {min_lr} propose init lr {lr}')
    fit_one_cycle(learn, 20, lr)

