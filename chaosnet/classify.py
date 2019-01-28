from __future__ import division

from models.yolo3 import Darknet
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

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


class MyVGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(MyVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 64, 'M', 128, 128, 256, 256, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def myvgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=160, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="cfg/vgg-16.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="cfg/tinyimage.data", help="path to data config file")
parser.add_argument("--workers", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda


# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]
valid_path = data_config["valid"]
num_classes = int(data_config["classes"])

# normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

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
    #normalize
    ])

valid_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    val_aug,
    transforms.ToTensor(),
    #normalize
    ])

in_memory = True
training_set = TinyImageSet(train_path, 'train', transform=training_transform, in_memory=in_memory)
valid_set = TinyImageSet(valid_path, 'val', transform=valid_transform, in_memory=in_memory)

vgg = myvgg11_bn(num_classes=num_classes)
print('Model', vgg)
device = torch.device("cuda")
vgg = vgg.to(device)

optimizer = torch.optim.SGD(vgg.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)

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
        vgg.train()
        for idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = vgg(data)            
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
            resnet.eval()
            for idx, (data, target) in enumerate(validloader):
                data, target = data.to(device), target.to(device)
                output = vgg(data)
                _, pred = torch.max(output, 1) # output.topk(1) *1 = top1

                num_hits += (pred == target).sum().item()
#                 print('{:.1f}% of validation'.format(idx / float(len(validloader)) * 100), end='\r')

        valid_acc = num_hits / num_instances * 100
        print(f' Validation acc: {valid_acc}%')
        sw.add_scalar('Validation Accuracy(%)', valid_acc, epoch + 1)
            
        epoch_loss /= float(len(trainloader))
#         print("Time used in one epoch: {:.1f}".format(time.time() - start))
        
        # save model
        torch.save(resnet.state_dict(), 'models/weight.pth')
        
        # record loss
        sw.add_scalar('Running Loss', epoch_loss, epoch + 1)
        
        
except KeyboardInterrupt:
    print("Interrupted. Releasing resources...")
    
finally:
    # this is only required for old GPU
    torch.cuda.empty_cache()
