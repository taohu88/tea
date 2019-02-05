from .callback import Callback
from ignite.engine import Events
from tea.metrics.metric_enum import MetricEnum


class PrintEvaluationMetrics(Callback):

    def __init__(self, priority=Callback._LOWER_TIER):
        super().__init__(priority)

    def events_to_attach(self):
        return [Events.EPOCH_COMPLETED]

    # TODO Print all metrics nicely
    def epoch_completed(self, engine):
        metrics = engine.state.metrics
        accuracy = metrics[MetricEnum.accuracy.value]
        val_loss = metrics[MetricEnum.valid_loss.value]
        print(f"Epoch: {engine.state.epoch:4d}  accuracy: {accuracy:5.3f} loss: {val_loss:8.3f}")
