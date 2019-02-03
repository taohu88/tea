from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau


#TODO fix it
def create_scheduler(cfg, optimizer, step_size=5, gamma=0.2):
    scheduler = ReduceLROnPlateau(optimizer, factor=gamma)
    return scheduler


def create_lr_finder_scheduler(optimizer, lr, start_lr, end_lr, batches):
    scaler = start_lr/lr
    gamma = (end_lr/start_lr)**(1/batches)
    return LRFinderScheduler(optimizer, gamma=gamma, scaler=scaler)


class LRFinderScheduler(ExponentialLR):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        scaler (float): adjust the base_lr by scaler
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, scaler, last_epoch=-1):
        self.gamma = gamma
        self.scaler = scaler
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.scaler * base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]
