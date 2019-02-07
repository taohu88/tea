from .callback_src_enum import CallbackSrcEnum
from .callback import Callback
from ignite.engine import Events


class SchedulerListener(Callback):

    def __init__(self, scheduler,
                 monitor_metric=None,
                 listen_to=CallbackSrcEnum.train,
                 event=Events.EPOCH_COMPLETED):
        super().__init__(listen_to)
        self.scheduler = scheduler
        self.monitor_metric = monitor_metric
        self.event = event

    def events_to_attach(self):
        return [self.event]

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
