from .callback import Callback


class MetricAdapter(Callback):
    """
    Adpater class for metrics defined ignite
    """

    def __init__(self, name, metric):
        super().__init__()
        self.name = name
        self.metric = metric

    def attach(self, engine):
        self.metric.attach(engine, self.name)
