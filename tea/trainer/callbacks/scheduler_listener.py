from .callback import Callback
from ignite.engine import Events
from tea.metrics.metric_enum import MetricEnum


def get_lrs(scheduler):
    return [str(param_group['lr']) for param_group in scheduler.optimizer.param_groups]


class SchedulerListener(Callback):

    def __init__(self, scheduler, metric_name=MetricEnum.lrs.value,
                 monitor_metric=None, priority=Callback._LOWER_TIER):
        super().__init__(priority)
        self.scheduler = scheduler
        self.metric_name = metric_name
        self.monitor_metric = monitor_metric

    def events_to_attach(self):
        return [Events.EPOCH_COMPLETED]

    def epoch_completed(self, engine):
        # epoch in engine starts with 1
        # while schedule epoch conceptually starting with 0 (as init lr == lr at epoch 0)
        # magically, it is next epoch for schedule
        next_epoch = engine.state.epoch
        metrics = engine.state.metrics
        if self.monitor_metric:
            # could be thrown exception here
            # which is what we want, fail fast
            metric = metrics[self.monitor_metric]
            self.scheduler.step(metric, epoch=next_epoch)
        else:
            self.scheduler.step(epoch=next_epoch)

        lrs = get_lrs(self.scheduler)
        # We also produce metric
        engine.state.metrics[self.metric_name] = lrs
