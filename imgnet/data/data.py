import os
from pathlib import Path

import torch

import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import re


device = 'cuda' if torch.cuda.is_available else 'cpu'

root = Path('/home/data/Imagenet')
train_path = root / 'train'
val_path = root / 'val'
synset_path = root / 'label' / 'synset_words.txt'

def ImgNetData(rank, world_size, args):
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
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if args.data_path == 'cifar10':
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms) 
        val_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transforms)
    else:
        train_data = ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transforms)
        val_data = ImageFolder(root=os.path.join(args.data_path, "val"), transform=val_transforms)
    
    if args.dummy:
        print("=> Dummy data is used!")
        train_data = datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor())
        val_data = datasets.FakeData(1000, (3, 224, 224), 1000, transforms.ToTensor())
        
    train_sampler = DistributedSampler(train_data,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=True,
                                        drop_last=True) # DistributedSampler에 shuffle option을 주면 DataLoader에는 주면 안됨
    val_sampler = DistributedSampler(val_data,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=False,
                                        drop_last=True)
    
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              num_workers=args.workers,
                              drop_last=True,
                              sampler=train_sampler)
    
    val_loader = DataLoader(val_data,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              num_workers=args.workers,
                              drop_last=True,
                              sampler=val_sampler)
    
    return train_loader, val_loader

if __name__ == "__main__":
    imagenet = ImgNetData("/home/data/imagenet")
    sample = next(iter(imagenet))
    
    print(f"sample:\n{sample[0].size, sample[1]}")