from __future__ import print_function, division
import os
import cv2
import torch
# import pandas as pd
from skimage import transform
import skimage.io as io


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
# import crash_on_ipy
import torch.optim as optim
import time
from torch.optim.lr_scheduler import *
from numpy import genfromtxt

batch_size=100


transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../cifar-10-batches-py', train=True, download=True, transform=transform_train)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='../cifar-10-batches-py', train=False, download=True, transform=transform_test)
dataloader_test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)



class GroupNorm2d(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = GroupNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = GroupNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = GroupNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                GroupNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = GroupNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

net = ResNet50()
print(net)




criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=50000, gamma=0.1)




if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)

if torch.cuda.is_available():
   net.cuda()




total_step=0

while True:
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):

        total_step=total_step+1
        start_time = time.time()

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        labels = labels.view(labels.size(0))

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        duration = time.time() - start_time
        examples_per_sec = labels.size(0) / duration

        running_loss += loss.item()
        
        scheduler.step()
        if total_step % 100 == 99:    # print every 2000 mini-batches
            net.eval()
            if not os.path.exists('./save_model'):
                os.mkdir('./save_model')
            torch.save(net,'./save_model/res50_GN_16groups'+str(total_step))
            firt_para=optimizer.param_groups[0]
            lr_sample = firt_para['lr']
            print('[%5d] loss: %.3f , examples_per_sec:%.4f , lr: %f' %
                  (total_step + 1, running_loss / 20, examples_per_sec, lr_sample))
            running_loss = 0.0

            correct = 0
            total = 0
            for _, (images_acc, labels_acc) in enumerate(dataloader):

                images_acc, labels_acc = Variable(images_acc.cuda()), labels_acc.cuda()
                labels_acc = labels_acc.view(labels_acc.size(0))

                outputs_ = net(images_acc)
                _, predicted = torch.max(outputs_.data, 1)
                total += labels_acc.size(0)
                correct += (predicted == labels_acc).sum()
                break

            print('Train Accuracy : %d %%' % (
                100 * correct / total))

            f_out = open('res50_GN_16groups.txt','a')
            f_out.write('['+str(loss.cpu().detach().numpy())+','+str(correct.cpu().detach().numpy() / total)+',')


            correct = 0
            total = 0
            for _, (images_acc, labels_acc) in enumerate(dataloader_test):
                images_acc, labels_acc = Variable(images_acc.cuda()), labels_acc.cuda()
                labels_acc = labels_acc.view(labels_acc.size(0))

                outputs_ = net(images_acc)
                _, predicted = torch.max(outputs_.data, 1)
                total += labels_acc.size(0)
                correct += (predicted == labels_acc).sum()

            print('Test Accuracy : %d %%' % (
                100 * correct.cpu().detach().numpy() / total))
            f_out.write(str(correct.cpu().detach().numpy() / total)+'],')
            f_out.close()
            net.train()
