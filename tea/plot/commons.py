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


def explore_lr_and_plot(learner, train_loader, path, start_lr=1.0e-5, end_lr=1.0, batches=100):
    r = learner.find_lr(train_loader, path=path, start_lr=start_lr, end_lr=end_lr, batches=batches)
    lr = r.get_lr_with_min_loss()[0]
    plot_lr_losses(r.lr_losses)
    return lr

