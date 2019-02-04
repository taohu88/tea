import math
import copy
import random
from pathlib import Path

import torch
from torch.optim import SGD, Adam

from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from ignite._utils import convert_tensor

from tqdm import tqdm

from .handlers import LogIterationLoss, LogValidationMetrics, RecordLrAndLoss
from .schedulers import create_lr_finder_scheduler, create_scheduler
from .base_engine import BaseEngine
from ..optimizer.adamw import AdamW


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options
    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


# TODO fix this, not just use Adam
def create_optimizer(cfg, model, lr):
    momentum = cfg.get_momentum()
    weight_decay = cfg.get_weight_decay()
#    optimizer = Adam(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    return optimizer


def create_trainer(cfg, model, optimizer):
    device = cfg.get_device()
    loss_fn = cfg.get_loss_fn()

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=False)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    return BaseEngine(_update)


def create_evaluator(cfg, model):
    device = cfg.get_device()
    loss_fn = cfg.get_loss_fn()

    metrics = {'accuracy': Accuracy(),
               'loss': Loss(loss_fn)}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=False)
            y_pred = model(x)
            return y_pred, y

    engine = BaseEngine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def build_trainer(cfg, model, train_loader, val_loader):
    return BaseLearner(cfg, model, train_loader, val_loader)


def find_max_lr(learner, train_loader):
    path = learner.cfg.get_model_out_dir()
    path = Path(path)/'lr_tmp.pch'
    lrs = []
    for i in range(5):
        batches = random.randint(90, 100)
        r = learner.find_lr(train_loader, batches=batches, path=path)
        lrs.append(r.get_lr_with_min_loss()[0])

    lr = sum(lrs)/len(lrs)
    return lr


def find_lr(learner, train_dl, start_lr=1.0e-7, end_lr=10, batches=100, path='/tmp/lr_tmp.pch'):
    learner.save_model(path, with_optimizer=False)

    lr = learner.cfg.get_lr()
    optimizer = create_optimizer(learner.cfg, learner.model, lr)
    trainer = create_trainer(learner.cfg, learner.model, optimizer)
    scheduler = create_lr_finder_scheduler(optimizer, lr, start_lr, end_lr, batches)
    log_freq = learner.cfg.get_log_freq()
    pbar = tqdm(
        initial=0, leave=False, total=batches,
        desc="Batch - loss: {:.3f}".format(0)
    )

    recorder = RecordLrAndLoss(trainer, scheduler, batches, log_freq, pbar)

    epochs = math.ceil(batches/len(train_dl))
    trainer.run(train_dl, max_epochs=epochs)

    learner.load_model(path)

    return recorder


def fit(learner, train_dl, valid_dl=None, epochs=None, lr=None):
    if not epochs:
        epochs = learner.cfg.get_epochs()
    if not lr:
        lr = learner.cfg.get_lr()

    optimizer = create_optimizer(learner.cfg, learner.model, lr)
    trainer = create_trainer(learner.cfg, learner.model, optimizer)
    evaluator = None if not valid_dl else create_evaluator(learner.cfg, learner.model)
    step_size = epochs // 3
    step_size = step_size if step_size > 0 else 1
    scheduler = create_scheduler(learner.cfg, optimizer, step_size)

    # @trainer.on(Events.EPOCH_STARTED)
    # def scheduler_step(engine):
    #     scheduler.step()

    pbar = tqdm(
        initial=0, leave=False, total=len(learner.train_dl),
        desc="Batch - loss: {:.3f}".format(0)
    )

    log_freq = learner.cfg.get_log_freq()
    if log_freq > 0:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, LogIterationLoss(log_freq, pbar))

    if learner.valid_dl:
        # trainer.add_event_handler(Events.EPOCH_COMPLETED, LogValidationMetrics(evaluator, valid_dl, pbar))
        @trainer.on(Events.EPOCH_COMPLETED)
        def on_epoch_completed(engine):
            " It could be re-entried multiple times"
            evaluator.reset()
            evaluator.run(learner.valid_dl)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            tqdm.write(
                f"Validation - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f} Avg loss: {avg_loss:.3f}")

            if pbar:
                pbar.n = pbar.last_print_n = 0
            scheduler.step(avg_loss)

    if not epochs:
        epochs = learner.cfg.get_epochs()
    trainer.run(train_dl, max_epochs=epochs)

    pbar.close()


class BaseLearner(object):
    """
    This is just plain supervised learner(trainer/evalulator)
    """
    def __init__(self, cfg, model, train_dl, valid_dl=None):
        self.cfg = cfg
        # explicitily set model to device
        device = self.cfg.get_device()
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
        device = self.cfg.get_device()
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
