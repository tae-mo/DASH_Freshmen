
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import json



# Preparing the training data
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

transforms_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)

trainset = torchvision.datasets.ImageNet('/home/data/Imagenet', split='train', download=None, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# testset = torchvision.datasets.ImageNet('/home/data/Imagenet', split='val', download=None, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)