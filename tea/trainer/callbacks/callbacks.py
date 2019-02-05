from .callback import Callback
from ignite.engine import Events
from tqdm import tqdm


class LogTrainLoss(Callback):
    def __init__(self, engine, log_freq, max_batches):
        if log_freq < 0:
            raise Exception(f"log freq {log_freq} is negative")
        self.log_freq = log_freq
        self.desc = "Batch loss: {:7.3f}"

        self.pbar = tqdm(
            initial=0, leave=False, total=max_batches,
            desc=self.desc.format(0)
        )
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.on_iteration_completed)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_completed)

    def on_iteration_completed(self, engine):
        log_freq = self.log_freq
        iter = engine.state.iteration

        if iter % log_freq != 0:
            return

        pbar = self.pbar
        if pbar:
            pbar.desc = self.desc.format(engine.state.output)
            pbar.update(log_freq)
        else:
            print(self.desc.format(engine.state.output))

    def on_epoch_completed(self, engine):
        pbar = self.pbar
        if pbar:
            pbar.n = pbar.last_print_n = 0


def get_lr(scheduler):
    return [float(param_group['lr']) for param_group in scheduler.optimizer.param_groups]


class LogValidationMetrics(Callback):

    def __init__(self, engine, evaluator, valid_dl, scheduler):
        self.evaluator = evaluator
        self.valid_dl = valid_dl
        self.scheduler = scheduler
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_completed)

    def on_epoch_completed(self, engine):
        " It could be re-entried multiple times"
        self.evaluator.reset()
        self.evaluator.run(self.valid_dl)
        metrics = self.evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print(f"Epoch: {engine.state.epoch:8d}  accuracy: {avg_accuracy:5.3f} loss: {avg_loss:8.3f} lrs: {str(get_lr(self.scheduler))}")
        self.scheduler.step(avg_loss)
