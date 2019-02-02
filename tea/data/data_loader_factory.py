import torch
from tea.config.helper import get_batch_sz, get_num_workers


def create_train_val_dataloader(cfg, train_dataset, valid_dataset):
    batch_sz = get_batch_sz(cfg)
    num_workers = get_num_workers(cfg)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_sz, shuffle=True, num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=batch_sz, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader