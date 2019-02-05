from .callback import Callback
from ignite.engine import Events
from tqdm import tqdm


class LogOutput(Callback):
    def __init__(self, log_freq, max_batches):
        if log_freq < 0:
            raise Exception(f"log freq {log_freq} is negative")
        self.log_freq = log_freq
        self.desc = "Batch loss: {:7.3f}"

        self.pbar = tqdm(
            initial=0, leave=False, total=max_batches,
            desc=self.desc.format(0)
        )

    def events_to_attach(self):
        return [Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED]

    def iteration_completed(self, engine):
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

    def epoch_completed(self, engine):
        pbar = self.pbar
        if pbar:
            pbar.n = pbar.last_print_n = 0
