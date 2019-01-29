from __future__ import division

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torchvision.models.vgg import *

import torch.optim as optim
from data.tiny_image_set import TinyImageSet
from models.builder import get_input_size
from models.common_model import CommonModel
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=374, help="size of each image batch")
    parser.add_argument("--cfg_path", type=str, default="cfg/my-vgg.cfg", help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="cfg/tinyimage.data", help="path to data config file")
    parser.add_argument("--workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")

    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda
    os.makedirs('output', exist_ok=True)

    module_defs = parse_model_config(opt.cfg_path)
    hyperparams = module_defs.pop(0)
    input_sz = get_input_size(hyperparams)

    # Set up model
    model = CommonModel(module_defs, input_sz)
    print(f'Model \n {model}')

    if cuda:
        model.cuda()
    device = torch.device("cuda:0" if cuda else "cpu")

    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    num_classes = int(data_config["classes"])
    
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
    
    in_memory = True
    training_set = TinyImageSet(train_path, 'train', transform=training_transform, in_memory=in_memory)
    valid_set = TinyImageSet(valid_path, 'val', transform=valid_transform, in_memory=in_memory)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.2)
    
    max_epochs = 20
    
    workers = opt.workers
    batch_size = opt.batch_size
    trainloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    validloader = DataLoader(valid_set, batch_size=batch_size, num_workers=workers)
    
    ce_loss = nn.CrossEntropyLoss()
    
    try:
        for epoch in range(max_epochs):
            start = time.time()
            lr_scheduler.step()
            epoch_loss = 0.0
            model.train()
            for idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                batch_loss = ce_loss(outputs, target)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
    
                if idx % 10 == 0:
                    print('{:.1f}% of epoch'.format(idx / float(len(trainloader)) * 100), end='\r')
    
    
            # evaluate on validation set
            num_hits = 0
            num_instances = len(valid_set)
    
            with torch.no_grad():
                model.eval()
                for idx, (data, target) in enumerate(validloader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, pred = torch.max(output, 1) # output.topk(1) *1 = top1
    
                    num_hits += (pred == target).sum().item()
    #                 print('{:.1f}% of validation'.format(idx / float(len(validloader)) * 100), end='\r')
    
            valid_acc = num_hits / num_instances * 100
    
            epoch_loss /= float(len(trainloader))
            print(f'Epoch {epoch} loss {epoch_loss:3f} validation acc: {valid_acc:3f}%')
    #         print("Time used in one epoch: {:.1f}".format(time.time() - start))
    
            # save model
            torch.save(model.state_dict(), 'output/weight.pth')
    
    
    except KeyboardInterrupt:
        print("Interrupted. Releasing resources...")
    
    finally:
        # this is only required for old GPU
        torch.cuda.empty_cache()
