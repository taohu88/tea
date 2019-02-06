from ignite.engine import Events


def get_lrs(opt):
    return [param_group['lr'] for param_group in opt.param_groups]


class LrSnapshot():
    """
    Snapshot lrs
    """
    def __init__(self, trainer, opt):
        self.trainer = trainer
        self.opt = opt

    def epoch_started(self, engine, name):
        lrs = get_lrs(self.opt)
        engine.state.metrics[name] = lrs

    def attach(self, engine, name):
        # Lr snapshot always associate with trainer
        self.trainer.add_event_handler(Events.EPOCH_STARTED, self.epoch_started, name)