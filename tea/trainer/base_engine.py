import time
from ignite.engine import Engine, Events
from ignite._utils import _to_hours_mins_secs


class BaseState(object):
    """An object that is used to pass internal and user-defined state between event handlers"""
    def __init__(self, **kwargs):
        # state, add epoch variable specifically
        self.epoch = 0
        self.iteration = 0

        # volatiles
        self.max_epochs = 0
        self.output = None
        self.metrics = {}
        for k, v in kwargs.items():
            setattr(self, k, v)


class BaseEngine(Engine):

    def __init__(self, process_function):
        super(BaseEngine, self).__init__(process_function)
        self.reset()

    def state_dict(self):
        return {
            'epoch': self.state.epoch,
            'iteration': self.state.iteration
        }

    def load_state_dict(self, old_state):
        epoch = old_state.get('epoch', 0)
        iteration = old_state.get('iteration', 0)
        self.state = BaseState(epoch=epoch, iteration=iteration)

    def reset(self):
        self.state = BaseState()

    def _run_once_on_dataset(self, dataloader):
        start_time = time.time()

        try:
            for batch in dataloader:
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                if self.should_terminate or self.should_terminate_single_epoch:
                    self.should_terminate_single_epoch = False
                    break

        except BaseException as e:
            self._logger.error("Current run is terminating due to exception: %s", str(e))
            self._handle_exception(e)

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)

        return hours, mins, secs

    def run(self, dataloader, max_epochs=1):
        """Runs the process_function over the passed data.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`)
            max_epochs (int, optional): max epochs to run for (default: 1)

        Returns:
            State: output state
        """
        self.state.max_epochs = max_epochs
        try:
            self._logger.info("Engine run starting with max_epochs={}".format(max_epochs))
            start_time = time.time()
            self._fire_event(Events.STARTED)
            while self.state.epoch < max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self._fire_event(Events.EPOCH_STARTED)
                hours, mins, secs = self._run_once_on_dataset(dataloader)
                self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.state.epoch, hours, mins, secs)
                if self.should_terminate:
                    break
                self._fire_event(Events.EPOCH_COMPLETED)

            self._fire_event(Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info("Engine run complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._logger.error("Engine run is terminating due to exception: %s", str(e))
            self._handle_exception(e)

        return self.state
