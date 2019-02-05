import math
from ignite.metrics import RunningAverage
from ignite.engine import Events
from tqdm import tqdm

from tea.utils.commons import self_or_first
from .callback import Callback


class RecordLrAndLoss(Callback):

    def __init__(self, trainer, scheduler, batches, log_freq):
        self.trainer = trainer
        self.scheduler = scheduler
        self.batches = batches
        self.log_freq = log_freq
        self.lr_losses = []
        self.best_loss = None
        self.desc = "Batch loss: {:.3f}"

        self.pbar = tqdm(
            initial=0, leave=False, total=batches,
            desc=self.desc.format(0)
        )

        alpha = 0.10
        avg_output = RunningAverage(output_transform=lambda x: x, alpha=alpha)
        avg_output.attach(trainer, 'running_avg_loss')

        trainer.add_event_handler(Events.ITERATION_STARTED, self.on_iteration_started)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.on_iteration_completed)
        trainer.add_event_handler(Events.COMPLETED, self.on_completed)

    def on_iteration_started(self, engine):
        self.scheduler.step()

    def on_iteration_completed(self, engine):
        iter = engine.state.iteration

        if iter >= self.batches:
            self.stop(engine)
            return

        metrics = engine.state.metrics
        if 'running_avg_loss' not in metrics:
            return

        avg_loss = metrics['running_avg_loss']
        if math.isnan(avg_loss):
            self.stop(engine)
            return

        pbar = self.pbar
        if pbar:
            pbar.desc = self.desc.format(engine.state.output)
            pbar.update(self.log_freq)
        else:
            print(self.desc.format(engine.state.output))

        if not self.best_loss:
            self.best_loss = avg_loss
        elif self.best_loss > avg_loss:
            self.best_loss = avg_loss

        if avg_loss > 4 * self.best_loss:
            self.stop(engine)
            return

        lrs = self.scheduler.get_lr()
        self.lr_losses.append((self_or_first(lrs), avg_loss))

    def on_completed(self, engine):
        self.pbar.close()

    def get_lr_with_min_loss(self):
        return min(self.lr_losses, key = lambda t: t[1])

    def stop(self, engine):
        engine.terminate()


