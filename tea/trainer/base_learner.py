import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Metric, Accuracy, Loss, RunningAverage

from tqdm import tqdm
from tea.config.helper import get_epochs, get_device, get_loss_fn, get_lr, get_momentum, get_log_freq
from .handlers import LogIterationLoss, LogValidationMetrics
from .lr_finder import LRFinderScheduler
from tea.commons import self_or_first, is_loss_too_large

from ignite._utils import convert_tensor


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


class MyLoss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).

    """

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: x.shape[0]):
        super(MyLoss, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0


    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        average_loss = self._loss_fn(y_pred, y, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss')

        N = self._batch_size(y)
        self._sum += average_loss.item() * N
        self._num_examples += N
        print('FFF', self._sum, self._num_examples)

    def compute(self):
        if self._num_examples == 0:
            raise Exception(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples


class BaseLearner(object):
    """
    This is just plain supervised learner(trainer/evalulator)
    """
    def __init__(self, cfg, model, train_dl, valid_dl=None):
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        lr = get_lr(self.cfg)
        momentum = get_momentum(self.cfg)
        model = self.model
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
        return optimizer

    def _create_trainer(self):
        optimizer = self.optimizer
        device = get_device(self.cfg)
        loss_fn = get_loss_fn(self.cfg)
        trainer = create_supervised_trainer(self.model, optimizer, loss_fn, device=device)
        return trainer

    def _create_evaluator(self):
        device = get_device(self.cfg)
        loss_fn = get_loss_fn(self.cfg)

        evaluator = create_supervised_evaluator(self.model,
                                                metrics={'accuracy': Accuracy(),
                                                         'loss': Loss(loss_fn)},
                                              device=device)
        return evaluator

    def create_scheduler(self):
        #TODO add scheduler
        pass

    def fit(self, train_dl, valid_dl=None):
        trainer = self._create_trainer()
        evaluator = None if not valid_dl else self._create_evaluator()

        pbar = tqdm(
            initial=0, leave=False, total=len(self.train_dl),
            desc="Batch - loss: {:.3f}".format(0)
        )

        log_freq = get_log_freq(self.cfg)
        if log_freq > 0:
            trainer.add_event_handler(Events.ITERATION_COMPLETED, LogIterationLoss(log_freq, pbar))

        if self.valid_dl:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, LogValidationMetrics(evaluator, valid_dl, pbar))

        epochs = get_epochs(self.cfg)
        trainer.run(train_dl, max_epochs=epochs)

        pbar.close()

    def save_model_optimizer_state(self):
        path = "./tmp.pch"
        state = {'model': self.model.state_dict(), 'opt':self.optimizer.state_dict()}
        torch.save(state, path)

    def restore_model_optimizer_state(self):
        "Load model and optimizer state (if `with_opt`) `name` from `self.model_dir` using `device`."
        path = "./tmp.pch"
        device = get_device(self.cfg)
        state = torch.load(path, map_location=device)

        if set(state.keys()) == {'model', 'opt'}:
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['opt'])


    def create_lr_find_trainer(self, metrics={},
                               non_blocking=False,
                               prepare_batch=_prepare_batch):

        model = self.model
        optimizer = self.optimizer
        device = get_device(self.cfg)
        loss_fn = get_loss_fn(self.cfg)

        if device:
            model.to(device)

        def _update(engine, batch):
            model.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            return y_pred, y

        engine = Engine(_update)

        return engine


    def find_lr(self, train_dl, start_lr=1.0e-5, end_lr=10, batches=100, skip_start=5, skip_end=5):
        self.save_model_optimizer_state()

        lr = get_lr(self.cfg)
        scaler = start_lr/lr
        gamma = (end_lr/start_lr)**(1/batches)

        print('AAA', start_lr, end_lr, end_lr/start_lr, gamma, scaler)

        scheduler =  LRFinderScheduler(self.optimizer, gamma=gamma, scaler=scaler)
        trainer = self._create_trainer()

        alpha = 0.98
        avg_output = RunningAverage(output_transform=lambda x: x, alpha=alpha)
        avg_output.attach(trainer, 'running_avg_loss')


        lr_losses = []
        def scheduler_step(engine):
            scheduler.step()
            lrs = scheduler.get_lr()


        def log_loss(engine):
            iter = engine.state.iteration
            if iter >= batches:
                engine.terminate()

            metrics = trainer.state.metrics
            if 'running_avg_loss' not in metrics:
                return

            avg_loss = metrics['running_avg_loss']

            import math
            if math.isnan(avg_loss) or (is_loss_too_large(avg_loss)):
                engine.terminate()

            print(f"Validation - Iter: {iter}  Avg loss: {avg_loss:.3f}")

            lrs = scheduler.get_lr()
            lr_losses.append((self_or_first(lrs), avg_loss))


        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_step)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_loss)

        for k, v in trainer._event_handlers.items():
            print(k, v)

        epochs = get_epochs(self.cfg)
        trainer.run(train_dl, max_epochs=epochs)

        print(lr_losses)

        t = min(lr_losses[skip_start:-skip_end], key = lambda t: t[1])
        print('AAA min',t)

        self.restore_model_optimizer_state()

        return t[0]

