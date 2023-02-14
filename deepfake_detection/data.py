import os
from pathlib import Path
from glob import glob

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, path, data_name, train=True, transform=None):
        '''
        ds_name
        DeepFake   4channel
        DeepFakeDetection 3channel (240, 240)
        Face2Face 3channel
        FaceSwap  4channel
        NeuralTextures  4channel
        '''
        self.path = os.path.join(path, data_name)
        if train:
            self.real_path = os.path.join(self.path, 'train/real')
            self.fake_path = os.path.join(self.path, 'train/fake')
        else:
            self.real_path = os.path.join(self.path, 'test/real')
            self.fake_path = os.path.join(self.path, 'test/fake')
            
        self.real_list = glob(os.path.join(self.real_path, '**/*.png'))
        self.fake_list = glob(os.path.join(self.fake_path, '**/*.png'))
        
        self.transform = transform
        
        self.img_list = self.real_list + self.fake_list
        self.class_list = [0]*len(self.real_list) + [1]*len(self.fake_list)
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path).convert('RGB')
        
        if not self.transform == None:
            img = self.transform(img)
            
        return img, label
    
def get_dataloader(args):
    mean = []
    stdv = []
    transform_train = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=stdv)
    ])
    transforms_val = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=stdv)
    ])
    
    train_data = DeepfakeDataset(args.data_path, args.data_name, train=True, transform=transform_train)
    valid_data = DeepfakeDataset(args.data_path, args.data_name, train=False, transform=transforms_val)
    
    if args.DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        print("[!] [Rank {}] Distributed Sampler Data Loading Done".format(args.local_rank))
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, pin_memory=True,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.n_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, pin_memory=True,
                                               batch_size=args.batch_size,
                                               sampler=None,
                                               shuffle=False,
                                               num_workers=args.n_workers)
    
    return train_loader, valid_loader, train_sampler
    
# if __name__ == "__main__":
#     import sys
#     sys.path.append("/home/jeonghokim/ICML_2022/src")
#     from main import build_args
#     args = build_args()
#     if args.data_type == "imagenet":
#         import torch.distributed as dist
#         dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank)
#     train_loader, valid_loader, train_sampler = get_dataloader(args)
#     img, label, _ = next(iter(train_loader))
#     print(img.shape)
