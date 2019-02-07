from .callback_src_enum import CallbackSrcEnum
from .callback import Callback


class MetricAdapter(Callback):
    """
    Adpater class for metrics defined ignite
    """

    def __init__(self, name, metric, listen_to=CallbackSrcEnum.either):
        super().__init__(listen_to=listen_to)
        self.name = name
        self.metric = metric
        self.listen_to = listen_to

    def attach(self, engine):
        self.metric.attach(engine, self.name)
