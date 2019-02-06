from enum import Enum


class _MetricEnum(Enum):
    """
    This class is for internal use only
    """
    # this is snapshot of lrs
    lrs = "lrs"
    batch_loss = "batch_loss"
    train_loss = "train_loss"
    valid_loss = "valid_loss"
    accuracy = "accuracy"

    @classmethod
    def is_batch_metric(cls, name):
        return str(name).startswith("batch")

    #TODO add more