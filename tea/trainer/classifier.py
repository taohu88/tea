from torch.optim import SGD
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm
from tea.config.helper import get_device, get_loss_fn, get_lr, get_momentum, get_log_freq


class Classifier():
    """
    This is a common classifer
    """
    def __init__(self, cfg, model, train_dl, valid_dl):
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl


    def _create_trainer_evaluator(self):
        lr = get_lr(self.cfg)
        momentum = get_momentum(self.cfg)
        device = get_device(self.cfg)
        loss_fn = get_loss_fn(self.cfg)
        model = self.model
        self._log_freq = get_log_freq(self.cfg)

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
