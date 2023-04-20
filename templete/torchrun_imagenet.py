import os
import tqdm
import argparse
import datetime
import gc
from collections import OrderedDict
import wandb
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

from utils import *
from model.vit import ViTscratch as ViT

# https://discuss.pytorch.kr/t/cuda-out-of-memory/216/5 << out of memory problem 

def parse_args():
    parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
    parser.add_argument('--save_every', '-i', default=1)
    parser.add_argument('--total_epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--scale', '-s', type=float, default=1.0)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    parser.add_argument('--model', '-m', type=str, default='resnet')
    parser.add_argument('--depth', '-d', type=int, default=50)
    parser.add_argument('--rand_seed', '-rs', type=int, default=11)
    parser.add_argument('--drop_out', '-o', type=float, default=.1)
    parser.add_argument('--vit_head', '-vh', type=int, default=12)
    parser.add_argument('--name', '-n', type=str, default='resnet_snapshot.pt')
    parser.add_argument('--resume', '-r', type=bool, default=False)
    parser.add_argument('--factor', '-f', type=float, default=.1, help='optimizer scheduler reducing factor')
    parser.add_argument('--device', default="cuda")
    # parser.add_argument('--keep', '-k', default='not_use_best')
    '''    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()'''
    return parser.parse_args()
#

def ddp_setup(): # torchrun
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200)) # 2시간으로 늘려주기
# end of ddp_setup ft

def main(save_every: int, total_epochs: int, batch_size: int, lr: float, model_depth: int, model:str, args,
         snapshot_path: str = "snapshot.pt", scale: float = 1.0, proj_name : str = 'resnet'):
    
    gc.collect()
    torch.cuda.empty_cache()
    
    ddp_setup() # distributed data parallel
    dataset = load_train_objs()
    train_data, test_data = prepare_dataloader(dataset, batch_size, scale)
    
    trainer = Trainer(model_depth=model_depth, model_type=model, train_data=train_data, test_data=test_data, save_every=save_every, lr=lr, snapshot_file=snapshot_path, proj_name=proj_name, args=args)
    trainer.train(total_epochs)
    destroy_process_group() # Destroy a given process group, and deinitialize the distributed package
#

if __name__ == "__main__":
    args = parse_args()                 # world_size = 2 if torch.cuda.device_count() > 2 else torch.cuda.device_count()
    torch.manual_seed(args.rand_seed)
    main(save_every=args.save_every, total_epochs=args.total_epochs, batch_size=args.batch_size, lr=args.lr,
         model_depth=args.depth, model=args.model, scale=args.scale, proj_name=args.name, args=args)
    
