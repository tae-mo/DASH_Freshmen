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

import model2 as MD

from tqdm import tqdm
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='simple distributed training job')
parser.add_argument('--epoch', default=30, type=int, help='Total epochs to train the model')
parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
parser.add_argument('--train', '-tr', required=False, default='/home/data/deepfake_privacy/FF_original/FaceForensics++/DeepFake', help='Root of Trainset')
parser.add_argument('--test', '-ts', required=False, default='/home/data/deepfake_privacy/FF_original/FaceForensics++/DeepFake', help='Root of Testset')
parser.add_argument('--snapshot_path', '-sn', required=False, default='/home/syon1203/DASH_Freshmen/xceptionnet/snapshot.pt', help='Root of Saving Path')
parser.add_argument('--snapshot_path_acc', '-sna', required=False, default='/home/syon1203/DASH_Freshmen/xceptionnet/snapshot_acc.pt', help='Root of Saved Path')
parser.add_argument('--snapshot_path_loss', '-snl', required=False, default='/home/syon1203/DASH_Freshmen/xceptionnet/snapshot_loss.pt', help='Root of Saved Path')
parser.add_argument('--model', '-m', required=False, default='VisionTransformer', help='Model')
parser.add_argument('--lr', '-l', required=False, default=3e-3, help='Learning Rate')
parser.add_argument('--num_workers', '-w', required=False, default=8, help='Workers')
parser.add_argument('--load_model', '-lm', required=False, default=False, help='Load model')


def main(rank, batch_size: int, world_size):
    torch.cuda.empty_cache()

    torch.cuda.set_device(rank) # set gpu id for each process

    train_loader, test_loader = dataset.dataloader(batch_size, args)

    top_acc = 0
    top_loss = 0

    model = MD.Xception(num_classes=2).to(rank)
    
    if args.load_model:

        checkpoint = torch.load(args.snapshot_path)
        model.load_state_dict(checkpoint['MODEL_STATE'])
        epoch = checkpoint['EPOCHS_RUN']

        print(f'Start from epoch:{epoch}')


    model = DDP(model, device_ids=[rank])

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        train_loader.sampler.set_epoch(epoch)

        train_loss, train_accuracy = train.train_fn(train_loader, model, optimizer, criterion, rank)
        test_loss, test_accuracy = validation.test_fn(test_loader, model, criterion, rank)
        
        print(f'Epoch:{epoch} Train loss:{train_loss} Train accuracy:{100*train_accuracy:.4f}% Test loss:{test_loss} Test accuracy:{100*test_accuracy:.4f}%')

        top_acc = max(test_accuracy,top_acc)
        top_loss = max(test_loss,top_loss)


        if rank == 0 and top_acc == test_accuracy:
            utils._save_snapshot(model, epoch, args.snapshot_path_acc)
        if rank == 0 and top_loss == test_loss:
            utils._save_snapshot(model, epoch, args.snapshot_path_loss)


        scheduler.step(test_loss)
        dist.barrier()
    
    utils.cleanup()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    args = parser.parse_args()
    utils.ddp_setup()
    main(rank=local_rank, batch_size=args.batch_size, world_size=dist.get_world_size())

    



