import os
from pathlib import Path

import torch

import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import re

device = 'cuda' if torch.cuda.is_available else 'cpu'

root = Path('/home/data/Imagenet')
train_path = root / 'train'
val_path = root / 'val'
synset_path = root / 'label' / 'synset_words.txt'

def ImgNetData(train_path=train_path, val_path=val_path):
    train_transforms = transforms.Compose([
        transforms.RandomChoice([transforms.Resize(256), transforms.Resize(480)]),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
        ### 이걸로 사용해볼 순 없나?
        # transforms.RandomResizedCrop((224,224)),
        # transforms.Resize((256,256)),
        # transforms.CenterCrop((224,224)),
        
        transforms.RandomHorizontalFlip(0.5),
    ])
    val_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_data = ImageFolder(root=train_path, transform=train_transforms)
    val_data = ImageFolder(root=val_path, transform=val_transforms)
    
    return train_data, val_data


def synset2word(synset_path=synset_path):
    label_dict = {}
    with open(synset_path, 'r') as f:
        synset_word = f.readlines()
        for i in range(len(synset_word)):
            synset = synset_word[i].split()[0]
            word = re.sub(r'[^a-zA-Z]', '', synset_word[i].split()[1])
            label_dict[synset] = word
            
    return label_dict