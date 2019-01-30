

def find_min_lr(lrs, losses, skip_start=10, skip_end=5):
    lrs_t = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
    losses_t = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]

    _, idx = min((val, idx) for (idx, val) in enumerate(losses_t))

    return lrs_t[idx]