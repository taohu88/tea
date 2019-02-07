from enum import Enum


class CallbackSrcEnum(Enum):
    """
    Define callback source, since we could be in train/test/validation
    """
    train = "train"
    test = "test"
    validation = "validation"
    either = "either"
