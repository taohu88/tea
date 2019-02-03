import fire

from torchvision import datasets

from tea.vision.cv import transforms
from tea.config.helper import parse_cfg, print_cfg, get_data_in_dir, get_epochs
import tea.data.data_loader_factory as DLFactory
import tea.models.factory as MFactory
from tea.trainer.base_learner import find_max_lr, build_trainer


def build_train_val_datasets(cfg):
    data_in_dir = get_data_in_dir(cfg)
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
Like anything in life, it is good to follow pattern.
In this case, any application starts with cfg file, 
with optional override arguments like the following: 
    data_dir/path
    model_cfg
    model_out_dir
    epochs, lr, batch etc
"""
def run(ini_file='mnist.ini',
        data_in_dir='./../../dataset',
        model_cfg='../cfg/lecnn.cfg',
        model_out_dir='./models',
        epochs=10, lr=0.01, batch_sz=256, log_freq=10, use_gpu=True):
    # Step 1: parse config
    cfg = parse_cfg(ini_file,
                    data_in_dir=data_in_dir,
                    model_cfg=model_cfg,
                    model_out_dir=model_out_dir,
                    epochs=epochs, lr=lr, batch_sz=batch_sz, log_freq=log_freq, use_gpu=use_gpu)
    print_cfg(cfg)

    # Step 2: create data sets and loaders
    train_ds, val_ds = build_train_val_datasets(cfg)
    train_loader, val_loader = DLFactory.create_train_val_dataloader(cfg, train_ds, val_ds)

    # Step 3: create model
    model = MFactory.create_model(cfg)

    # Step 4: train/valid
    classifier = build_trainer(cfg, model, train_loader, val_loader)

    # # Step 5: optionally find the best lr
    # lr = find_max_lr(classifier, train_loader)/2.0
    # print(f"Ideal learning rate {lr}")

    epochs = get_epochs(cfg)
    classifier.fit(train_loader, val_loader, epochs=epochs, lr=lr)


if __name__ == '__main__':
    fire.Fire(run)
