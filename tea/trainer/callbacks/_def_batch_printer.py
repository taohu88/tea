from .callback import Callback
from ignite.engine import Events
from tqdm import tqdm
from tea.utils.commons import islist
from tea.metrics._metric_enum import _MetricEnum


class DefBatchPrinter(Callback):
    """
    This is for internal use only
    """
    def __init__(self, log_freq, max_batches):
        super().__init__()
        if log_freq < 0:
            raise Exception(f"log freq {log_freq} is negative")

        self.log_freq = log_freq
        self.last_iter = 0

        self.pbar = tqdm(
            initial=0, leave=False, total=max_batches)

    def events_to_attach(self):
        return [Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED, Events.COMPLETED]

    def iteration_completed(self, engine):
        log_freq = self.log_freq
        iter = engine.state.iteration
        if (iter - self.last_iter) % log_freq != 0:
            return

        pbar = self.pbar
        if pbar:
            pbar.desc = self._generate_desc_str(engine)
            pbar.update(log_freq)
        else:
            print(self.desc.format(engine.state.output))

    def epoch_completed(self, engine):
        pbar = self.pbar
        if pbar:
            pbar.n = pbar.last_print_n = 0
        self.last_iter = engine.state.iteration

    def completed(self, engine):
        if self.pbar:
            self.pbar.close()

    @staticmethod
    def _generate_desc_str(engine):
        metrics = engine.state.metrics

        str_l = []
        for k, v in metrics.items():
            # not batch metrics
            if not _MetricEnum.is_batch_metric(k):
                continue
            if islist(v):
                str_l.append(f"{k:>10s}: {','.join([str(d) for d in v]):>12s}")
            elif isinstance(v, float):
                str_l.append(f"{k:>10s}: {v:>8.3f}")
            elif isinstance(v, int):
                str_l.append(f"{k:>10s}: {v:>4d}")
            else:
                str_l.append(f"{k:>10s}: {str(v):>16s}")

        return ' '.join(str_l)