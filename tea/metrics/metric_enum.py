from enum import Enum


class MetricEnum(Enum):
    # this is snapshot of lrs
    lrs = "lrs"
    train_loss = "train_loss"
    valid_loss = "valid_loss"
    accuracy = "accuracy"

    #TODO add more