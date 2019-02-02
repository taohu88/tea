import math
import copy
import random
from pathlib import Path

import torch
from torch.optim import SGD

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm
from tea.config.helper import get_model_out_dir, get_epochs, get_device, get_loss_fn, get_lr, get_momentum, get_log_freq
from .handlers import LogIterationLoss, LogValidationMetrics, RecordLrAndLoss
from .schedulers import create_lr_finder_scheduler, create_scheduler


def create_optimizer(cfg, model, lr):
    momentum = get_momentum(cfg)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    return optimizer


def create_trainer(cfg, model, optimizer):
    device = get_device(cfg)
    loss_fn = get_loss_fn(cfg)
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    return trainer


def create_evaluator(cfg, model):
    device = get_device(cfg)
    loss_fn = get_loss_fn(cfg)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'loss': Loss(loss_fn)},
                                            device=device)
    return evaluator


def build_trainer(cfg, model, train_loader, val_loader):
    return BaseLearner(cfg, model, train_loader, val_loader)


def find_max_lr(learner, train_loader):
    path = get_model_out_dir(learner.cfg)
    path = Path(path)/'lr_tmp.pch'
    lrs = []
    for i in range(5):
        batches = random.randint(90, 100)
        r = learner.find_lr(train_loader, batches=batches, path=path)
        lrs.append(r.get_lr_with_min_loss()[0])

    lr = sum(lrs)/len(lrs)
    return lr


def find_lr(learner, train_dl, start_lr=1.0e-5, end_lr=10, batches=100, path='/tmp/lr_tmp.pch'):
    learner.save_model(path, with_optimizer=False)

    lr = get_lr(learner.cfg)
    optimizer = create_optimizer(learner.cfg, learner.model, lr)
    trainer = create_trainer(learner.cfg, learner.model, optimizer)
    scheduler = create_lr_finder_scheduler(optimizer, lr, start_lr, end_lr, batches)
    recorder = RecordLrAndLoss(trainer, scheduler, batches)

    epochs = math.ceil(batches/len(train_dl))
    trainer.run(train_dl, max_epochs=epochs)

    learner.load_model(path)

    return recorder


def fit(learner, train_dl, valid_dl=None, epochs=None, lr=None):
    if not epochs:
        epochs = get_epochs(learner.cfg)
    if not lr:
        lr = get_lr(learner.cfg)

    optimizer = create_optimizer(learner.cfg, learner.model, lr)
    trainer = create_trainer(learner.cfg, learner.model, optimizer)
    evaluator = None if not valid_dl else create_evaluator(learner.cfg, learner.model)
    step_size = epochs // 2
    step_size = step_size if step_size > 0 else 1
    scheduler = create_scheduler(learner.cfg, optimizer, step_size)

    @trainer.on(Events.EPOCH_STARTED)
    def scheduler_step(engine):
        scheduler.step()

    pbar = tqdm(
        initial=0, leave=False, total=len(learner.train_dl),
        desc="Batch - loss: {:.3f}".format(0)
    )

    log_freq = get_log_freq(learner.cfg)
    if log_freq > 0:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, LogIterationLoss(log_freq, pbar))

    if learner.valid_dl:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, LogValidationMetrics(evaluator, valid_dl, pbar))
    if not epochs:
        epochs = get_epochs(learner.cfg)
    trainer.run(train_dl, max_epochs=epochs)

    pbar.close()


class BaseLearner(object):
    """
    This is just plain supervised learner(trainer/evalulator)
    """
    def __init__(self, cfg, model, train_dl, valid_dl=None):
        self.cfg = cfg
        # explicitily set model to device
        device = get_device(self.cfg)
        self.model = model.to(device)
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        BaseLearner.fit = fit
        BaseLearner.find_lr = find_lr

    def save_model(self, path, with_optimizer=False):
        if with_optimizer:
            state = {'model': self.model.state_dict(), 'opt':self.optimizer.state_dict()}
        else:
            state = {'model': self.model.state_dict()}
        torch.save(state, path)

    def load_model(self, path):
        device = get_device(self.cfg)
        state = torch.load(path, map_location=device)

        if 'model' in state:
            self.model.load_state_dict(state['model'])

        if 'opt' in state:
            self.optimizer.load_state_dict(state['opt'])

    def copy_model_state(self, with_optimizer=False):
        """
        Warning this will waste your gpu space, so it is not recommended
        :param with_optimizer:
        :return model_state and optimizer state

        """
        model_state = copy.deepcopy(self.model.state_dict())
        if with_optimizer:
            opt_state = copy.deepcopy(self.optimizer.state_dict())
        return model_state, opt_state

    def restore_model_state(self, model_state, opt_state=None):
        self.model.load_state_dict(model_state)
        if opt_state:
            self.optimizer.load_state_dict(opt_state)


