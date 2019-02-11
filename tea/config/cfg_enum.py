from enum import Enum


class CfgEnum(Enum):
    """
    This is used to enforce configuration keys are valids
    """
    # Data section
    data_in_dir = "data_in_dir"
    train_dir = "train_dir"
    valid_dir = "valid_dir"
    test_dir  = "test_dir"

    # model section
    model_cfg = "model_cfg"
    model_out_dir = "model_out_dir"
    loss = "loss"

    optim = "optim"
    betas = "betas"

    # hypers sections
    epochs = "epochs"
    vocab_sz = "vocab_sz"
    lr = "lr"
    clip = "clip"
    momentum = "momentum"
    weight_decay = "weight_decay"
    log_freq = "log_freq"
    use_gpu = "use_gpu"
    batch_sz = "batch_sz"
    bptt = "bppt"
    train_batch_sz = "train_batch_sz"
    val_batch_sz = "val_batch_sz"
    test_batch_sz = "test_batch_sz"
    num_workers = "num_workers"

    eps = "eps"
