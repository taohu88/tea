import fire
from pathlib import Path

from torchvision import transforms
from tea.metrics.accuracy import Accuracy
from tea.config.app_cfg import AppConfig
import tea.data.data_loader_factory as DLFactory
import tea.models.factory as MFactory
from tea.trainer.basic_learner import build_trainer
from tea.trainer.helper import explore_lr_and_plot
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


def run(ini_file='tinyimg.ini',
        data_in_dir='../../../dataset/tiny-imagenet-200',
        model_cfg='../cfg/vgg-tiny-simple.cfg',
        model_out_dir='./models',
        epochs=40,
        lr=1e-3,
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
    train_ds, val_ds = build_train_val_datasets(cfg, in_memory=True)
    train_loader, val_loader = DLFactory.create_train_val_dataloader(cfg, train_ds, val_ds)

    # Step 3: create model
    model = MFactory.create_model(cfg)
    print(model)

    # Step 4: train/valid
    learner = build_trainer(cfg, model)

    # Step 5: optionally find the best lr
    if explore_lr:
        path = learner.cfg.get_model_out_dir()
        path = Path(path) / 'lr_tmp.pch'
        lr = explore_lr_and_plot(learner, train_loader, path, start_lr=1.0e-5, end_lr=1.0, batches=100)
        print(f'Idea lr {lr}')
        plt.show()
    else:
        # accuracy is a classification metric
        metrics = {"accuracy": Accuracy()}
        learner.fit(train_loader, val_loader, metrics=metrics)


if __name__ == '__main__':
    fire.Fire(run)

