from .callback import Callback
from ignite.engine import Events


class RunEvaluator(Callback):

    def __init__(self, evaluator, valid_dl, priority=Callback._LOW_TIER+1):
        super().__init__(priority)
        self.evaluator = evaluator
        self.valid_dl = valid_dl

    def events_to_attach(self):
        return [Events.EPOCH_COMPLETED]

    def epoch_completed(self, engine):
        self.evaluator.run(self.valid_dl, max_epochs=1)
        # transfer evaluation metrics back
        engine.state.metrics.update(self.evaluator.state.metrics)
