import os
import argparse
import builtins
import shutil
import wandb

import torch
import torch.nn as nn
from datetime import datetime

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import StepLR
from model.resnet50 import BottleNeck, ResNet
from model.vit import ViT
from data.data import ImgNetData
from Train import train, validate
import utils
from utils import AverageMeter, Logger, print_args, save_args, load_args, Accuracy

def parse_arg():
    parser = argparse.ArgumentParser(description='Imagenet classifictation')
    # config
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--is_wandb', type=bool, default=True)
    
    # training
    parser.add_argument('--data-path', nargs='?', default='/home/data/Imagenet',
                    help='kinds of dataset') # metavar:인자의 이름지정, nargs 값 개수 지정
    parser.add_argument('-m','--model', default='resnet50',
                    help='kinds of model')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs of run')
    parser.add_argument('-b','--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128), this is the total batch size of all GPUs on the current node whe using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--beta1', default=0.9, type=float, metavar='BETA1',
                        help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='BETA2',
                        help='beta2 for Adam')
    parser.add_argument('--wd', default=0., type=float, metavar='weight_decay',
                        help='weight decay for Adam')
    parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 1000)')
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set') # dest: 적용 위치 지정, '-e'값이 args.evaluate에 저장되는것
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch for resume training')
    
    # data loader
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark") # 제대로 돌아가는지 보기위한 Fake데이터
    
    return parser.parse_args()

def cleanup():
    dist.destroy_process_group()

def main(args):
    now = datetime.now()
    # 옵션1: resume 방법
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint: '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # MAP model to be loaded to specific single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict((checkpoint['state_dict']))
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]['capturable'] = True
            id = checkpoint['wandb_id']
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.is_wandb:
        if args.resume:
            wandb.init(project="ImageNet", name=args.model, notes=' runed at ' + now.strftime('%Y-%m-%d %H:%M'), entity='sang8961', resume='allow', id=id)
        else:
            id = wandb.util.generate_id()
            wandb.init(project="ImageNet", name=args.model, notes=' runed at ' + now.strftime('%Y-%m-%d %H:%M'), entity='sang8961')
        wandb.config.update(args)
    
    utils.init_distributed_mode(args)
    
    torch.cuda.set_device(args.gpu) # 각 프로세스에 gpu id setting
    
    args.batch_size = args.batch_size // torch.cuda.device_count()
    args.workers = args.workers // torch.cuda.device_count()
    
    train_loader, val_loader = ImgNetData(args.gpu, args.world_size, args)
    if args.model == 'resnet50':
        model = ResNet(BottleNeck).to(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.model == 'vit':
        model = ViT().to(args.gpu)
    model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
    criterion = nn.CrossEntropyLoss()
    
    wandb.watch(model, criterion, log="all", log_freq=10)
            
    if args.evaluate: # eval mode
        validate(val_loader, model, criterion, args)
        return
    
    if args.gpu == 0: print(f"Start Training")
    
    best_acc1, best_loss = 0., float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.gpu, args)
        
        print("====================Validate Start==============")
        val_loss, top1_acc = validate(val_loader, model, criterion, args.gpu, args)
        
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)
        best_loss = min(val_loss, best_loss)
        
        if args.gpu == 0:
            wandb.log({"train loss": train_loss, "train acc": train_acc, "val loss": val_loss,
                "val_acc": top1_acc, "epoch": epoch})
        
        if args.gpu == 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'wandb_id': id
            }, is_best, args=args)
            print("checkpoint is saved!!")
        scheduler.step()
        dist.barrier()
        
    cleanup()
        
def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.model)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.model) + 'model_best.pth.tar')
    save_args(args, os.path.join(directory, 'argument.json'))

if __name__ == "__main__":
    args = parse_arg()
    main(args)