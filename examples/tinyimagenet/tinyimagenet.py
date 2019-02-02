from __future__ import division
import fire

from torchvision import transforms
from tea.config.helper import parse_cfg, print_cfg, get_data_in_dir, get_epochs
import tea.data.data_loader_factory as DLFactory
import tea.models.factory as MFactory
from tea.trainer.base_learner import find_max_lr, build_trainer

from tea.data.tiny_imageset import TinyImageSet


def build_train_val_datasets(cfg):
    data_in_dir = get_data_in_dir(cfg)
    in_memory = True

    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
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
        data_in_dir='./../../dataset',
        model_cfg='../cfg/vgg-tiny.cfg',
        model_out_dir='./models',
        epochs=30,
        lr=0.001,
        batch_sz=256,
        num_worker=4,
        log_freq=50,
        use_gpu=True):
    # Step 1: parse config
    cfg = parse_cfg(ini_file,
                    data_in_dir=data_in_dir,
                    model_cfg=model_cfg,
                    model_out_dir=model_out_dir,
                    epochs=epochs, lr=lr, batch_sz=batch_sz, log_freq=log_freq,
                    num_worker=num_worker, use_gpu=use_gpu)
    print_cfg(cfg)

    # Step 2: create data sets and loaders
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader, val_loader = DLFactory.create_train_val_dataloader(cfg, train_ds, val_ds)

    # Step 3: create model
    model = MFactory.create_model(cfg)

    # Step 4: train/valid
    classifier = build_trainer(cfg, model, train_loader, val_loader)

    # # Step 5: optionally find the best lr
    # lr = find_max_lr(classifier, train_loader)/10.0
    # print(f"Ideal learning rate {lr}")

    epochs = get_epochs(cfg)
    classifier.fit(train_loader, val_loader, epochs=epochs, lr=lr)


if __name__ == '__main__':
    fire.Fire(run)

