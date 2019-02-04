import torch


def create_train_val_dataloader(cfg, train_dataset, valid_dataset):
    batch_sz = cfg.get_batch_sz()
    num_workers = cfg.get_num_workers()

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_sz, shuffle=True, num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=batch_sz, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader