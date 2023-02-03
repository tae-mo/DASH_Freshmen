import torch
#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import utils
import train 
import validation
import dataset

import torchvision.transforms as tr
from torchvision.datasets import ImageFolder
import torchvision.datasets

#import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os

import model as MD

from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='simple distributed training job')
parser.add_argument('--epoch', default=20, type=int, help='Total epochs to train the model')
parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
parser.add_argument('--train', '-tr', required=False, default='/home/data/Imagenet/train', help='Root of Trainset')
parser.add_argument('--test', '-ts', required=False, default='/home/data/Imagenet/validation', help='Root of Testset')
parser.add_argument('--snapshot_path', '-sn', required=False, default='/home/syon1203/DASH_Freshmen/ViT_classification/snapshot.pt', help='Root of Saving Path')
parser.add_argument('--model', '-m', required=False, default='resnet50', help='Model')
parser.add_argument('--lr', '-l', required=False, default=1e-4, help='Learning Rate')
parser.add_argument('--num_workers', '-w', required=False, default=8, help='Workers')
parser.add_argument('--load_model', '-lm', required=False, default=False, help='Load model')


def main(rank, batch_size: int, world_size):

    torch.cuda.set_device(rank) # set gpu id for each process

    train_loader, test_loader = dataset.dataloader(batch_size, args)

    model = MD.ViT().to(rank)
    
    if args.load_model:

        checkpoint = torch.load(args.snapshot_path)
        model.load_state_dict(checkpoint['MODEL_STATE'])
        epoch = checkpoint['EPOCHS_RUN']

        print(f'Start from epoch:{epoch}')


    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        train_loader.sampler.set_epoch(epoch)

        train_loss, train_accuracy = train.train_fn(train_loader, model, optimizer, criterion, rank)
        test_loss, test_accuracy = validation.test_fn(test_loader, model, criterion, rank)
        
        print(f'Epoch:{epoch} Train loss:{train_loss} Train accuracy:{100*train_accuracy:.4f}% Test loss:{test_loss} Test accuracy:{100*test_accuracy:.4f}%')

        if rank == 0:
            utils._save_snapshot(model, epoch, args.snapshot_path)

        scheduler.step()
        dist.barrier()
    
    utils.cleanup()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    args = parser.parse_args()
    utils.ddp_setup()
    main(rank=local_rank, batch_size=args.batch_size, world_size=dist.get_world_size())

    



