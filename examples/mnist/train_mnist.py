import fire
from pathlib import Path
from torchvision import datasets

from tea.vision.cv import transforms
from tea.config.app_cfg import AppConfig
import tea.data.data_loader_factory as DLFactory
import tea.models.factory as MFactory
from tea.trainer.base_learner import build_trainer
from tea.plot.commons import explore_lr_and_plot
import matplotlib.pyplot as plt


def build_train_val_datasets(cfg):
    data_in_dir = cfg.get_data_in_dir()
    train_ds = datasets.MNIST(data_in_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    valid_ds = datasets.MNIST(data_in_dir, train=False,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    return train_ds, valid_ds


"""
It is good to follow pattern.
In this case, any application starts with cfg file, 
with optional override arguments like the following: 
    data_dir/path
    model_cfg
    model_out_dir
    epochs, lr, batch etc
"""
def run(ini_file='mnist.ini',
        data_in_dir='../../../dataset',
        model_cfg='../cfg/lecnn.cfg',
        model_out_dir='./models',
        epochs=10,
        lr=0.01, batch_sz=256, log_freq=10, use_gpu=True,
        explore_lr=True):
    # Step 1: parse config
    cfg = AppConfig.from_file(ini_file,
                    data_in_dir=data_in_dir,
                    model_cfg=model_cfg,
                    model_out_dir=model_out_dir,
                    epochs=epochs, lr=lr, batch_sz=batch_sz, log_freq=log_freq, use_gpu=use_gpu)
    cfg.print()

    # Step 2: create data sets and loaders
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader, val_loader = DLFactory.create_train_val_dataloader(cfg, train_ds, val_ds)

    # Step 3: create model
    model = MFactory.create_model(cfg)

    # Step 4: train/valid
    learner = build_trainer(cfg, model, train_loader, val_loader)

    # Step 5: optionally find the best lr
    if explore_lr:
        path = learner.cfg.get_model_out_dir()
        path = Path(path) / 'lr_tmp.pch'
        lr = explore_lr_and_plot(learner, train_loader, path, start_lr=1.0e-5, end_lr=1.0, batches=100)
        print(f'Idea lr {lr}')
        plt.show()
    else:
        learner.fit(train_loader, val_loader)

if __name__ == '__main__':
    fire.Fire(run)
