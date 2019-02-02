from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm


class LogIterationLoss(object):

    def __init__(self, log_freq, pbar=None):
        if log_freq < 0:
            raise Exception(f"log freq {log_freq} is negative")

        self.log_freq = log_freq
        self.pbar = pbar

    def __call__(self, engine):
        log_freq = self.log_freq
        iter = engine.state.iteration

        if iter % log_freq != 0:
            return

        pbar = self.pbar
        if pbar:
            pbar.desc = "Batch - loss: {:.2f}".format(engine.state.output)
            pbar.update(log_freq)
        else:
            tqdm.write("Batch - loss: {:.2f}".format(engine.state.output))


class LogValidationMetrics(object):

    def __init__(self, evaluator, valid_dl, pbar=None):
        self.evaluator = evaluator
        self.valid_dl = valid_dl
        self.pbar = pbar

    def __call__(self, engine):
        self.evaluator.run(self.valid_dl)
        metrics = self.evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write(
            f"Validation - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f} Avg loss: {avg_loss:.3f}")

        pbar = self.pbar
        if pbar:
            pbar.n = pbar.last_print_n = 0