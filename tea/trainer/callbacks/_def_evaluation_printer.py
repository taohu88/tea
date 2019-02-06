from .callback import Callback
from ignite.engine import Events
from tea.utils.commons import islist
from tea.metrics._metric_enum import _MetricEnum


class DefEvaluationPrinter(Callback):
    """
    This is for internal use only
    """

    def __init__(self):
        super().__init__()

    def events_to_attach(self):
        return [Events.EPOCH_COMPLETED]

    def epoch_completed(self, engine):
        metrics = engine.state.metrics
        str_l = [f"Epoch: {engine.state.epoch:>4d}"]
        for k, v in metrics.items():
            # batch metrics
            if _MetricEnum.is_batch_metric(k):
                continue
            if islist(v):
                str_l.append(f"{k:>10s}: {','.join([str(d) for d in v]):>12s}")
            elif isinstance(v, float):
                str_l.append(f"{k:>10s}: {v:>8.3f}")
            elif isinstance(v, int):
                str_l.append(f"{k:>10s}: {v:>4d}")
            else:
                str_l.append(f"{k:>10s}: {str(v):>16s}")
        print(' '.join(str_l))
