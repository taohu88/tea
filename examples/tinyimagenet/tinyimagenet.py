from __future__ import division
import fire
from pathlib import Path

from torchvision import transforms
from tea.config.app_cfg import AppConfig
import tea.data.data_loader_factory as DLFactory
import tea.models.factory as MFactory
from tea.trainer.base_learner import build_trainer
from tea.plot.commons import plot_lr_losses

from tea.data.tiny_imageset import TinyImageSet
import matplotlib.pyplot as plt


def build_train_val_datasets(cfg, in_memory=False):
    data_in_dir = cfg.get_data_in_dir()

    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(56),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10)
    ])

    val_aug = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(56)
    ])

    training_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        train_aug,
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        val_aug,
        transforms.ToTensor(),
        normalize
    ])

    train_ds = TinyImageSet(data_in_dir, 'train', transform=training_transform, in_memory=in_memory)
    valid_ds = TinyImageSet(data_in_dir, 'val', transform=valid_transform, in_memory=in_memory)
    return train_ds, valid_ds


"""
Like anything in life, it is good to follow pattern.
In this case, any application starts with cfg file, 
with optional override arguments like the following: 
    data_dir/path
    model_cfg
    model_out_dir
    epochs, lr, batch etc
"""
def run(ini_file='tinyimg.ini',
        data_in_dir='../../../dataset/tiny-imagenet-200',
        model_cfg='../cfg/vgg-tiny-simple.cfg',
        model_out_dir='./models',
        epochs=50,
        lr=3.0e-5,
        batch_sz=256,
        num_workers=4,
        log_freq=20,
        use_gpu=True,
        explore_lr=False):
    # Step 1: parse config
    cfg = AppConfig.from_file(ini_file,
                        data_in_dir=data_in_dir,
                        model_cfg=model_cfg,
                        model_out_dir=model_out_dir,
                        epochs=epochs, lr=lr, batch_sz=batch_sz, log_freq=log_freq,
                        num_workers=num_workers, use_gpu=use_gpu)
    cfg.print()

    # Step 2: create data sets and loaders
    train_ds, val_ds = build_train_val_datasets(cfg, in_memory=False)
    train_loader, val_loader = DLFactory.create_train_val_dataloader(cfg, train_ds, val_ds)

    # Step 3: create model
    model = MFactory.create_model(cfg)
    print(model)

    # Step 4: train/valid
    learner = build_trainer(cfg, model, train_loader, val_loader)

    # Step 5: optionally find the best lr
    if explore_lr:
        path = learner.cfg.get_model_out_dir()
        path = Path(path) / 'lr_tmp.pch'
        lrs = []
        r = learner.find_lr(train_loader, start_lr=1.0e-7, end_lr=1.0, batches=100, path=path)
        plot_lr_losses(r.lr_losses[10:-5])
        lr = r.get_lr_with_min_loss()[0]
        print('lr', lr)
        plt.show()
    else:
        epochs = cfg.get_epochs()
        learner.fit(train_loader, val_loader, epochs=epochs, lr=lr)


if __name__ == '__main__':
    fire.Fire(run)

