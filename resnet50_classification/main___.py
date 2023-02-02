import torch
#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import utils

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
import logging
import argparse


parser = argparse.ArgumentParser(description='simple distributed training job')
parser.add_argument('--epoch', default=20, type=int, help='Total epochs to train the model')
parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
parser.add_argument('--train', '-tr', required=False, default='/home/data/Imagenet/train', help='Root of Trainset')
parser.add_argument('--test', '-ts', required=False, default='/home/data/Imagenet/validation', help='Root of Testset')
parser.add_argument('--snapshot_path', '-sn', required=False, default='/home/syon1203/DASH_Freshmen/resnet50_classification/snapshot.pt', help='Root of Saving Path')
parser.add_argument('--model', '-m', required=False, default='resnet50', help='Model')
parser.add_argument('--lr', '-l', required=False, default=1e-4, help='Learning Rate')
parser.add_argument('--num_workers', '-w', required=False, default=8, help='Workers')


def ddp_setup():
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

class AverageMeter (object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dataloader(batch_size: int):
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #imagenet mean std 

    #transf_train = tr.Compose([tr.Resize((224, 224)),tr.ToTensor(),tr.RandomHorizontalFlip(), tr.Normalize(*stats, inplace=True)])

    transforms = tr.Compose([
        tr.Resize((224, 224)),
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),
        tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        tr.ToTensor(),
        tr.Normalize(*stats, inplace=True)])

    #train_set = torchvision.datasets.FakeData(1000, (3, 224, 224), 10, transform=transforms,target_transform=None) #dummy
    #test_set = torchvision.datasets.FakeData(200, (3, 224, 224), 10, transform=transforms,target_transform=None) #dummy

    train_set = ImageFolder(args.train, transform=transforms,target_transform=None)
    test_set = ImageFolder(args.test, transform=transforms,target_transform=None)


    train_sampler=DistributedSampler(dataset=train_set,shuffle=True) #Default true
    test_sampler=DistributedSampler(dataset=test_set,shuffle=False) 

    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size, 
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  sampler=train_sampler)
    test_dataloader = DataLoader(test_set,
                                  batch_size=batch_size, 
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  sampler=test_sampler)

    return train_dataloader, test_dataloader
    
def train_fn(train_loader, model, optimizer, criterion, rank):

    model.train()
    loop = tqdm(train_loader, leave=True) #진행
    
    train_loss = []
    correct = []
    count = 0

    for batch_idx, (x, y) in enumerate(loop):
    #for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(rank, non_blocking=True), y.to(rank, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        train_loss.append(loss.item())
        optimizer.zero_grad() #backprop 전 초기화해서 방향
        loss.backward()
        optimizer.step()

        _, prediction = torch.max(out.data, 1) #텐서의 최대값
        correct.append(int(torch.sum(prediction == y.data)))

        count += x.size(0)

        # update progress bar
        loop.set_postfix(loss=loss.item(), accuracy=100*(sum(correct) / count))

        # if batch_idx % 10 == 0 :
        #     print(f'Batch:{batch_idx} Batch loss:{sum(train_loss)/len(train_loss)} Batch accuracy:{100*(sum(correct) / count):.4f}%')
    
    train_accuracy = sum(correct) / count
    train_loss = sum(train_loss)/len(train_loss)

    return train_loss, train_accuracy


def test_fn(test_loader, model, criterion, rank):
    model.eval()
    loop = tqdm(test_loader, leave=True) #진행률

    test_loss = []
    correct = []
    count = 0
  
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(rank), y.to(rank)
        out = model(x)
        loss = criterion(out, y)
        _, prediction = torch.max(out.data, 1)
        
        #print("data   ", prediction,"real answer   " y)
        test_loss.append(loss.item())
        correct.append(int(torch.sum(prediction == y.data)))
        count += x.size(0)

        # update progress bar
        loop.set_postfix(loss=loss.item(), accuracy=100*(sum(correct) / count))

    test_accuracy = sum(correct) / count
    test_loss = sum(test_loss)/len(test_loss)

    return test_loss, test_accuracy


def _save_snapshot(model, epoch):
        snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, args.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {args.snapshot_path}")


def main(rank, batch_size: int, world_size):

    torch.cuda.set_device(rank) # set gpu id for each process

    train_loader, test_loader = dataloader(batch_size)

    model = MD.ResNet50().to(rank)

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        train_loader.sampler.set_epoch(epoch)

        train_loss, train_accuracy = train_fn(train_loader, model, optimizer, criterion, rank)
        test_loss, test_accuracy = test_fn(test_loader, model, criterion, rank)
        
        print(f'Epoch:{epoch} Train loss:{train_loss} Train accuracy:{100*train_accuracy:.4f}% Test loss:{test_loss} Test accuracy:{100*test_accuracy:.4f}%')

        #total_accuracy = [torch.ones_like(test_accuracy) for _ in range(world_size)]
        #total_loss = [torch.ones_like(test_loss) for _ in range(world_size)]

        #dist.all_gather(total_accuracy, test_accuracy)
        #dist.all_gather(total_loss, test_loss)

        if rank == 0:
            _save_snapshot(model, epoch)

        scheduler.step()
        dist.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    args = parser.parse_args()
    ddp_setup()
    main(rank=local_rank, batch_size=args.batch_size, world_size=dist.get_world_size())

    
