import math
from ignite.engine import Events
from ignite.metrics import RunningAverage
from tqdm import tqdm

from tea.commons import self_or_first


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
            tqdm.write("loss: {:.2f}".format(engine.state.output))


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


class RecordLrAndLoss():

    def __init__(self, trainer, scheduler, batches, log_freq, pbar=None):
        self.trainer = trainer
        self.scheduler = scheduler
        self.batches = batches
        self.log_freq = log_freq
        self.pbar = pbar
        self.lr_losses = []
        self.best_loss = None

        alpha = 0.10
        avg_output = RunningAverage(output_transform=lambda x: x, alpha=alpha)
        avg_output.attach(trainer, 'running_avg_loss')

        trainer.add_event_handler(Events.ITERATION_STARTED, self.scheduler_step)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.log_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_complete)

    def scheduler_step(self, engine):
        self.scheduler.step()

    def log_loss(self, engine):
        iter = engine.state.iteration

        if iter >= self.batches:
            engine.terminate()

        metrics = engine.state.metrics
        if 'running_avg_loss' not in metrics:
            return


        avg_loss = metrics['running_avg_loss']
        if math.isnan(avg_loss):
            self.stop()

        pbar = self.pbar
        if pbar:
            pbar.desc = "Batch - loss: {:.2f}".format(engine.state.output)
            pbar.update(self.log_freq)
        else:
            tqdm.write("Batch - loss: {:.2f}".format(engine.state.output))

        if not self.best_loss:
            self.best_loss = avg_loss
        elif self.best_loss > avg_loss:
            self.best_loss = avg_loss

        if avg_loss > 4 * self.best_loss:
            self.stop(engine)

        lrs = self.scheduler.get_lr()
        self.lr_losses.append((self_or_first(lrs), avg_loss))

    def get_records(self):
        return self.lr_losses

    def get_lr_with_min_loss(self):
        return min(self.lr_losses, key = lambda t: t[1])

    def stop(self, engine):
        engine.terminate()

        pbar = self.pbar
        if pbar:
            pbar.n = pbar.last_print_n = 0

    def epoch_complete(self, engine):
        pbar = self.pbar
        if pbar:
            pbar.n = pbar.last_print_n = 0