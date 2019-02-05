class Callback():
    """
    (_HIGH_TIER, _LOW_TIER) are only usable for general callbacks
    The system reserves priority outside of it.
    It is a little bit of hacky for now
    """
    _HIGH_TIER = 100
    _LOW_TIER = 1000
    _DEF_TIER = 500

    #those are general out of reach for programming
    _HIGHER_TIER = _HIGH_TIER // 10
    _LOWER_TIER = _LOW_TIER * 10

    def __init__(self, priority=None):
        if priority is None:
            priority = Callback._DEF_TIER
        self.priority = priority

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
            engine.add_event_handler(e, method_)
