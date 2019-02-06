from .callback import Callback
from ignite.engine import Events
from tea.metrics.metric_enum import MetricEnum
from tea.utils.commons import islist


class PrintEvaluationMetrics(Callback):

    def __init__(self, priority=Callback._LOWER_TIER):
        super().__init__(priority)

    def events_to_attach(self):
        return [Events.EPOCH_COMPLETED]

    def epoch_completed(self, engine):
        metrics = engine.state.metrics
        str_l = [f"Epoch: {engine.state.epoch:>4d}"]
        for k, v in metrics.items():
            if islist(v):
                str_l.append(f"{k:>10s}: {str(v):>12s}")
            elif isinstance(v, float):
                str_l.append(f"{k:>10s}: {v:>8.3f}")
            elif isinstance(v, int):
                str_l.append(f"{k:>10s}: {v:>4d}")
            else:
                str_l.append(f"{k:>10s}: {str(v):>16s}")
        print(' '.join(str_l))
