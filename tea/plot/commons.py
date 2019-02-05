import matplotlib.pyplot as plt


def plot_lr_losses(lrs_losses):
    lrs = [l for l, _ in lrs_losses]
    losses = [l for _, l in lrs_losses]

    _, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
