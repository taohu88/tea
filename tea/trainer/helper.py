import random
from tea.plot.commons import plot_lr_losses


def explore_lr_and_plot(learner, train_loader, path, start_lr=1.0e-5, end_lr=1.0, batches=100):
    r = learner.find_lr(train_loader, path=path, start_lr=start_lr, end_lr=end_lr, batches=batches)
    lr = r.get_lr_with_min_loss()[0]
    plot_lr_losses(r.lr_losses)
    return lr


def find_max_lr(learner, train_loader, path, tries=5, batches=100):
    lrs = []
    for i in range(tries):
        batches = random.randint(int(batches/100*90), int(batches/100*110))
        r = learner.find_lr(train_loader, batches=batches, path=path)
        lrs.append(r.get_lr_with_min_loss()[0])

    lr = sum(lrs)/len(lrs)
    return lr