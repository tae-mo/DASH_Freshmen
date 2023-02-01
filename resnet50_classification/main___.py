import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as tr
from torchvision.datasets import ImageFolder
import torchvision.datasets

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import model as MD


from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='simple distributed training job')
parser.add_argument('--epoch', default=10, type=int, help='Total epochs to train the model')
parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
parser.add_argument('--train', '-tr', required=False, default='/home/data/Imagenet/train', help='Root of Trainset')
parser.add_argument('--test', '-ts', required=False, default='/home/data/Imagenet/validation', help='Root of Testset')
parser.add_argument('--model', '-m', required=False, default='resnet50', help='Model')
parser.add_argument('--lr', '-l', required=False, default=0.1, help='Learning Rate')


def ddp_setup():
    init_process_group(backend="nccl")

def cleanup():
    destroy_process_group()

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

    transf_train = tr.Compose([
        tr.Resize((224, 224)),
        tr.ToTensor(),
        tr.RandomHorizontalFlip(),
        tr.Normalize(*stats, inplace=True)])

    transf_test = tr.Compose([
        tr.Resize((224, 224)),
        tr.ToTensor(), 
        tr.Normalize(*stats, inplace=True)])

    train_set = torchvision.datasets.FakeData(1000, (3, 224, 224), 10, transform=transf_train,target_transform=None) #dummy
    test_set = torchvision.datasets.FakeData(100, (3, 224, 224), 10, transform=transf_test,target_transform=None) #dummy

    #train_set = ImageFolder(args.train, transform=transf_train,target_transform=None)
    #test_set = ImageFolder(args.test, transform=transf_test,target_transform=None)


    train_sampler=DistributedSampler(dataset=train_set,shuffle=True) #Default true 
    test_sampler=DistributedSampler(dataset=test_set,shuffle=False) 

    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size, 
                                  pin_memory=True,
                                  shuffle=False,
                                  sampler=train_sampler)
    test_dataloader = DataLoader(test_set,
                                  batch_size=batch_size, 
                                  pin_memory=True,
                                  shuffle=False,
                                  sampler=test_sampler)

    return train_dataloader, test_dataloader
    
def train_fn(train_loader, model, optimizer, criterion, rank):
    model.train()
    loop = tqdm(train_loader, leave=True) #진행률
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(rank), y.to(rank)
        out = model(x)
        loss = criterion(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad() #backprop 전 초기화해서 방향
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return sum(mean_loss)/len(mean_loss)
  

def test_fn(test_loader, model, criterion, rank):
    model.eval()
    loop = tqdm(test_loader, leave=True) #진행률
    test_loss, test_accuracy = 0, 0
  
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(rank), y.to(rank)
        out = model(x)
        loss = criterion(out, y)
        _, prediction = torch.max(out.data, 1)

        test_accuracy += int(torch.sum(prediction == y))
        
        # update progress bar
        loop.set_postfix(loss=loss.item())
        test_loss += loss.item()

    test_accuracy = test_accuracy / len(test_loader)
    test_loss = test_loss / len(test_loader)

    return test_loss, test_accuracy


def main(rank, batch_size: int):
    ddp_setup()

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(rank) # set gpu id for each process

    train_loader, test_loader = dataloader(batch_size)

    model = MD.ResNet50().to(rank)


    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        train_loader.sampler.set_epoch(epoch)
        train_fn(train_loader, model, optimizer, criterion, rank)
        val_acc, val_loss = test_fn(test_loader, model, criterion, rank)
        print(val_acc, val_loss)

        scheduler.step()

    

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    args = parser.parse_args()

    main(rank=local_rank, batch_size=args.batch_size)

    #save

    
