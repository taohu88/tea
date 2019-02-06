from ignite.engine import Events


class Snapshot():
    """
    Snapshot any values
    """

    def __init__(self, output_transform=lambda x: x,
                 when=Events.EPOCH_COMPLETED):
        self.output_transform = output_transform
        self.when = when

    def completed(self, engine, name):
        v = self.output_transform(engine.state.output)
        engine.state.metrics[name] = v

    def attach(self, engine, name):
        engine.add_event_handler(self.when, self.completed, name)
