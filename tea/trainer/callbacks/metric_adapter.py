from .callback import Callback


class MetricAdapter(Callback):
    """
    Adpater class for metrics defined ignite
    """

    def __init__(self, name, metric, priority=None):
        super().__init__(priority)
        self.name = name
        self.metric = metric

    def attach(self, engine):
        self.metric.attach(engine, self.name)
