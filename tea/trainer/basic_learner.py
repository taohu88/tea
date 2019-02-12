import math
import copy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite._utils import convert_tensor

from ..metrics._metric_enum import _MetricEnum
from ..metrics.snapshot import Snapshot
from ..metrics.lr_snapshot import LrSnapshot
from ..config.cfg_enum import CfgEnum

from .callbacks.callback_src_enum import CallbackSrcEnum
from .callbacks._def_batch_printer import DefBatchPrinter
from .callbacks._def_evaluation_printer import DefEvaluationPrinter
from .callbacks.record_lr_loss import RecordLrAndLoss
from .callbacks.scheduler_listener import SchedulerListener
from .callbacks.metric_adapter import MetricAdapter
from .callbacks.run_evaluator import RunEvaluator

from .schedulers import create_lr_finder_scheduler, create_scheduler
from .engine import TeaEngine
from ..optimizer.adamw import AdamW
from ..optimizer.sgdw import SGDW


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options
    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_optimizer(cfg, model, lr):
    optim_str = cfg.get_optim()

    weight_decay = cfg.get_weight_decay()
    if optim_str == "AdamW":
        betas_str = cfg.get_str(CfgEnum.betas.value, "0.9, 0.99")
        betas_str = betas_str.split(",")
        betas = tuple(float(b) for b in betas_str)
        optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_str == "SGDW":
        momentum = cfg.get_momentum()
        optimizer = SGDW(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise Exception(f"Unexcept optimization {optim_str}")

    return optimizer


def create_trainer(cfg, model, optimizer, loss_fn):
    device = cfg.get_device()
    clip = cfg.get_clip()

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        model.reset_context()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=False)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        return y_pred, y, loss.item()

    return TeaEngine(_update)


def create_evaluator(cfg, model, metrics = {}):
    device = cfg.get_device()
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=False)
            y_pred = model(x)
            return y_pred, y

    engine = TeaEngine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def build_trainer(cfg, model):
    return BasicLearner(cfg, model)


def _attach_callbacks(callbacks, trainer, evaluator=None, predictor=None):
    for cb in callbacks:
        listen_to = cb.listen_to
        if listen_to == CallbackSrcEnum.train:
            cb.attach(trainer)
        elif listen_to == CallbackSrcEnum.validation:
            cb.attach(evaluator)
        elif listen_to == CallbackSrcEnum.test:
            cb.attach(predictor)
        else: #either
            def_engine = evaluator if evaluator else trainer
            cb.attach(def_engine)


"""
by default those callbacks all listen to trainer
"""
def _create_def_cbs(cfg, evaluator, train_dl, valid_dl, scheduler):
    cbs = []
    log_freq = cfg.get_log_freq()
    if log_freq > 0:
        log_train_loss = DefBatchPrinter(log_freq, len(train_dl))
        cbs.append(log_train_loss)

    if not isinstance(scheduler, ReduceLROnPlateau):
        metric_name = None
    else:
        if evaluator:
            cbs.append(RunEvaluator(evaluator, valid_dl))
            metric_name = _MetricEnum.valid_loss.value
        else:
            metric_name = _MetricEnum.train_loss.value

    cbs.append(SchedulerListener(scheduler, monitor_metric=metric_name))
    cbs.append(DefEvaluationPrinter())
    return cbs


def _get_train_loss(output):
    return output[2]


def _create_def_metrics(evaluator, opt, loss_fn):
    cbs = []
    if evaluator:
        cbs.append(MetricAdapter(_MetricEnum.valid_loss.value, Loss(loss_fn), listen_to=CallbackSrcEnum.validation))
    else:
        cbs.append(MetricAdapter(_MetricEnum.train_loss.value,
                                 RunningAverage(output_transform=_get_train_loss),
                                 listen_to=CallbackSrcEnum.train))

    cbs.append(MetricAdapter(_MetricEnum.batch_loss.value,
                             Snapshot(output_transform=_get_train_loss,
                                      when=Events.ITERATION_COMPLETED),
                             CallbackSrcEnum.train))
    cbs.append(MetricAdapter(_MetricEnum.lrs.value, LrSnapshot(opt), CallbackSrcEnum.train))
    return cbs


def _create_metrics_callbacks(trainer, evaluator, loss_fn, metrics, opt, use_def_print):
    cbs = []
    for name, m in metrics.items():
        cbs.append(MetricAdapter(name, m))

    if use_def_print:
        cbs += _create_def_metrics(evaluator, opt, loss_fn)
    _attach_callbacks(cbs, trainer, evaluator)


def _create_call_backs(cfg, trainer, evaluator, train_dl, valid_dl, scheduler, callbacks, use_def_print):
    cbs = []
    if callbacks:
        for c in callbacks:
            cbs.append(c)

    if use_def_print:
        cbs += _create_def_cbs(cfg, evaluator, train_dl, valid_dl, scheduler)
    _attach_callbacks(cbs, trainer, evaluator)


def find_lr(learner, train_dl,
            loss_fn=None,
            start_lr=1.0e-7, end_lr=10, batches=100, path='/tmp/lr_tmp.pch'):
    learner.save_model(path, with_optimizer=False)

    lr = learner.cfg.get_lr()
    loss_fn = loss_fn if loss_fn else learner.cfg.get_loss_fn()

    optimizer = create_optimizer(learner.cfg, learner.model, lr)
    trainer = create_trainer(learner.cfg, learner.model, optimizer, loss_fn)
    scheduler = create_lr_finder_scheduler(optimizer, lr, start_lr, end_lr, batches)
    recorder = RecordLrAndLoss(scheduler, batches)
    recorder.attach(trainer)

    max_epochs = math.ceil(batches/len(train_dl))
    trainer.run(train_dl, max_epochs=max_epochs)
    learner.load_model(path)

    return recorder


def fit(learner, train_dl, valid_dl=None,
        loss_fn=None,
        start_epoch=0,
        metrics={},
        callbacks=None,
        use_def_print=True):
    lr = learner.cfg.get_lr()
    loss_fn = loss_fn if loss_fn else learner.cfg.get_loss_fn()

    optimizer = create_optimizer(learner.cfg, learner.model, lr)
    trainer = create_trainer(learner.cfg, learner.model, optimizer, loss_fn)
    scheduler = create_scheduler(learner.cfg, optimizer)
    evaluator = create_evaluator(learner.cfg, learner.model) if valid_dl else None

    _create_metrics_callbacks(trainer, evaluator, loss_fn, metrics, optimizer, use_def_print)
    _create_call_backs(learner.cfg, trainer, evaluator, train_dl, valid_dl, scheduler, callbacks, use_def_print)

    max_epochs = learner.cfg.get_epochs()
    trainer.run(train_dl, start_epoch=start_epoch, max_epochs=max_epochs)


class BasicLearner(object):
    """
    This is just plain supervised learner(trainer/evalulator)
    """
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

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


BasicLearner.fit = fit
BasicLearner.find_lr = find_lr
