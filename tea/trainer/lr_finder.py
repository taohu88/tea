from torch.optim.lr_scheduler import ExponentialLR


class LRFinderScheduler(ExponentialLR):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, scaler, last_epoch=-1):
        self.gamma = gamma
        self.scaler = scaler
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.scaler * base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]
