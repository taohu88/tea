class Callback():
    def __init__(self):
        pass

    def epoch_started(self, engine):
        pass

    def epoch_completed(self, engine):
        pass

    def started(self, engine):
        pass

    def completed(self, engine):
        pass

    def iteration_started(self, engine):
        pass

    def iteration_completed(self, engine):
        pass

    def exception_raised(self, engine):
        pass

    def events_to_attach(self):
        pass

    def attach(self, engine):
        events = self.events_to_attach()
        for e in events:
            method_name = e.value
            method_ = getattr(self, method_name)
            if not engine.has_event_handler(method_, e):
                engine.add_event_handler(e, method_)
