import random
import fire

from torchvision import datasets

from tea.vision.cv import transforms
from tea.config.helper import parse_cfg, print_cfg, get_epochs
from tea.data.helper import build_train_val_dataloader
from tea.models.basic_model import build_model
from tea.trainer.base_learner import BaseLearner


def build_train_val_datasets(cfg):
    save_path = cfg.get('data', 'save_path')
    train_ds = datasets.MNIST(save_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    valid_ds = datasets.MNIST(save_path, train=False,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    return train_ds, valid_ds


def build_trainer(cfg, model, train_loader, val_loader):
    return BaseLearner(cfg, model, train_loader, val_loader)


def find_best_lr(classifier, train_loader):
    lrs = []
    for i in range(5):
        batches = random.randint(90, 100)
        r = classifier.find_lr(train_loader, batches=batches)
        lrs.append(r.get_lr_with_min_loss()[0])

    lr = sum(lrs)/len(lrs)/10.0
    return lr


def run(ini_file='mnist.ini', epochs=10, lr=0.01, batch_sz=256, log_freq=10, gpu_flag=1):
    # Step 1: parse config
    cfg = parse_cfg(ini_file, epochs=epochs, lr=lr, batch_sz=batch_sz, log_freq=log_freq, gpu_flag=gpu_flag)
    print_cfg(cfg)

    # Step 2: create data sets and loaders
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader, val_loader = build_train_val_dataloader(cfg, train_ds, val_ds)

    # Step 3: create model
    model = build_model(cfg)

    # Step 4: train/valid
    classifier = build_trainer(cfg, model, train_loader, val_loader)

    # Step 5: optionally find the best lr
    lr = find_best_lr(classifier, train_loader)
    print(f"Ideal learning rate {lr}")

    lr = 0.01
    epochs = get_epochs(cfg)
    classifier.fit(train_loader, val_loader, epochs=epochs, lr=lr)


if __name__ == '__main__':
  fire.Fire(run)
