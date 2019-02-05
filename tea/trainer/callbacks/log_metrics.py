from .callback import Callback
from ignite.engine import Events

def get_lr(scheduler):
    return [float(param_group['lr']) for param_group in scheduler.optimizer.param_groups]

#
# TODO refactor it to general one
#
class LogValidationMetrics(Callback):

    def __init__(self, evaluator, valid_dl, scheduler):
        self.evaluator = evaluator
        self.valid_dl = valid_dl
        self.scheduler = scheduler

    def events_to_attach(self):
        return [Events.EPOCH_COMPLETED]

    def epoch_completed(self, engine):
        " It could be re-entried multiple times"
        self.evaluator.reset()
        self.evaluator.run(self.valid_dl)
        metrics = self.evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print(f"Epoch: {engine.state.epoch:4d}  accuracy: {avg_accuracy:5.3f} loss: {avg_loss:8.3f} lrs: {str(get_lr(self.scheduler))}")
        #TODO move schedule out of here
        self.scheduler.step(avg_loss)
