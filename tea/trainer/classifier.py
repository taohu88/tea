
from torch.optim import SGD
import torch
import torch.nn.functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from tea.core.hyper_params import HyperParams

from tqdm import tqdm


class Classifier():
    """
    This is a common classifer
    """
    def __init__(self, model, train_dl, valid_dl, hyper_params):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.hyper_params = hyper_params

    def _create_trainer_evaluator(self):
        lr = self.hyper_params.get_lr()
        momentum = self.hyper_params.get_momentum()
        device = HyperParams.get_device(self.hyper_params.uses_gpu())
        loss_fn = HyperParams.get_loss_fn(self.hyper_params.get_loss_name())
        model = self.model

        self._log_freq = self.hyper_params.get_log_freq()

        optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum)
        trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
        evaluator = create_supervised_evaluator(model,
                                                metrics={'accuracy': Accuracy(),
                                                         'loss': Loss(loss_fn)},
                                                device=device)
        return trainer, evaluator

    def create_scheduler(self):
        #TODO add scheduler
        pass

    def log_training_loss(self, engine, pbar=None):
        log_freq = self._log_freq
        if log_freq < 0:
            return

        iter = engine.state.iteration

        if iter % log_freq != 0:
            return

        if pbar:
            pbar.desc = "Batch - loss: {:.2f}".format(engine.state.output)
            pbar.update(log_freq)
        else:
            tqdm.write("Batch - loss: {:.2f}".format(engine.state.output))

    def log_validation_metrics(self, engine, evaluator, pbar=None):
        evaluator.run(self.valid_dl)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write(
            f"Validation - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f} Avg loss: {avg_loss:.3f}")
        if pbar:
            pbar.n = pbar.last_print_n = 0

    def fit(self, epochs):
        trainer, evaluator = self._create_trainer_evaluator()

        pbar = tqdm(
            initial=0, leave=False, total=len(self.train_dl),
            desc="Batch - loss: {:.3f}".format(0)
        )

        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.log_training_loss, pbar)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_validation_metrics, evaluator, pbar)
        trainer.run(self.train_dl, max_epochs=epochs)

        pbar.close()
