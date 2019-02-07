from .callback import Callback


class MetricAdapter(Callback):
    """
    Adpater class for metrics defined ignite
    """

    def __init__(self, name, metric, intent_engine=None):
        super().__init__()
        self.name = name
        self.metric = metric
        self.intent_engine = intent_engine

    def attach(self, engine):
        if self.intent_engine:
            engine = self.intent_engine
        self.metric.attach(engine, self.name)
