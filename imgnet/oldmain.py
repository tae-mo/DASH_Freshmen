import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn # cudnn.benchmark = True 최적의 backend 연산을 찾는 flag를 True로 함. ex)입력크기가 고정된 모델 등에 유효

# for DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed

import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from model.resnet50 import BottleNeck, ResNet
from data.data import ImgNetData
from utils import AverageMeter, Logger, print_args, save_args, load_args, Accuracy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Imagenet classifictation')
parser.add_argument('data', metavar='S', nargs='?', default='imagenet',
                    help='kinds of dataset') # metavar:인자의 이름지정, nargs 값 개수 지정
parser.add_argument('-m','--model', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (defalut: resnet50')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs of run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b','--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node whe using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set') # dest: 적용 위치 지정, '-e'값이 args.evaluate에 저장되는것
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://10.201.134.133:8892', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark") # 제대로 돌아가는지 보기위한 Fake데이터

best_acc1 = 0



def main():
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True # seed를 정했으므로 nondeterministic하지 않게 작업
        cudnn.benchmark = False # 지금 환경에 가장 적합한 알고리즘을 찾을 필요가 없다.
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism')
        
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"]) # local를 가정한건가??
    print('world_size : ', args.world_size)
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count() # GPU 0, 1번만 쓰기로 해서
    else:
        ngpus_per_node = 1
        
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size # 각 노드(machine)별 쓸 수 있는 gpu 다 합치기
        # distributed processing 시작
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) # (fn:작동시킬 함수, nprocs:프로세서 갯수, args:fn에 넣을 args) / fn(i, *args) i is the process index
    else:
        # 단순하게 작동시킬 때
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    # GPU 설정
    print('gpu & ngpus_per_node',gpu, ngpus_per_node)
    args.gpu = gpu
    
    if args.gpu is not None:
        print("use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            print('rank :', args.rank)
        if args.multiprocessing_distributed:
            # gpu = 0,1,2 ... ngpus_per_node-1
            args.rank=args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        
    # Model 설정
    print("Creatin model '{}'".format(args.model))
    if args.data == 'imagenet':
        num_classes = 1000
        img_dir = '/home/data/Imagenet'
    if args.model == 'resnet50':
        model = ResNet(BottleNeck, num_classes=num_classes)
    elif args.model == 'vit':
        pass
    else:
        raise Exception('unknown model: {}'.format(args.model))
    
    # multiprocessing 설정
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow') #gpu는 되는데 multiprocessing이 안될때?
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # when using a single GPU per process and per DDP, we need to divide tha batch size ourselves based on the total number of GPUs we have 왜지??
            args.batch_size = int(args.batch_size / ngpus_per_node) # gpu에 나눠주기 위함이겠지?
            args.workers = int((args.workers+ngpus_per_node-1)/ngpus_per_node) # 왜 이렇게해주지
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # 만약에 device_ids를 따로 설정해주지 않으면, 가능한 모든 gpu를 기준으로 ddp가 알아서 배치사이즈와 workers를 나눠준다는 뜻.
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataparallel is supported.")
    
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # criterion & optimizer 정의
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
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
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(500, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        train_path = os.path.join(img_dir, 'train')
        val_path = os.path.join(img_dir, 'val')
    
        train_dataset, val_dataset = ImgNetData(train_path, val_path)
        
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    if args.evaluate: # eval mode
        validate(val_loader, model, criterion, args)
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # 내용 6-1: train_sampler.set_epoch
            # In distributed mode, calling the set_eopch() method at the beggining of 
            # each epoch before creating the "dataloader" iterator is necessary to make
            # suffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
            train_sampler.set_epoch(epoch)  # 매 에폭마다 train_sampler.set_epoch(epoch)를 해주어야 shuffle이 잘 사용된다고 한다.
            
            train(train_loader, model, criterion, optimizer, epoch, device, args)
            
            top1_acc, losses = validate(val_loader, model, criterion, args)
            
            scheduler.step(epoch)
            
            is_best = top1_acc > best_acc1
            best_acc1 = max(top1_acc, best_acc1)
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, args=args)

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    
    model.train()
    
    end = time.time()
    for i, (input, target) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)
        
        # 또한 텐서 및 스토리지를 고정하면 비동기(asynchronous) GPU 복사본을 사용할 수 있습니다.
        # 비동기식으로 GPU에 데이터 전달 기능을 추가하려면 non_blocking = True 인수를 to() 또는 cuda() 호출 시 argument로 전달하면 됩니다.
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        output = model(input)
        loss = criterion(output, target)
        
        acc1 = Accuracy(output.data, target, topk=(1,))
        
        losses.update(loss.item(), input.size(0))
        top1_acc.update(acc1[0], input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if 1 & args.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val[0]:.4f} ({top1.avg[0]:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1_acc))
            
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg[0]:.3f} Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1_acc, loss=losses))

    return losses.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    
    model.eval()
    
    end = time.time()
    for i, (input, target) in tqdm(enumerate(val_loader)):
        input = input.cuda()
        target = target.cuda()
        
        output = model(input)
        loss = criterion(output, target)
        
        acc1 = Accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1_acc.update(acc1[0], input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if 1 & args.print_freq == 0:
            print('Test (on val set): [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val[0]:.4f} ({top1.avg[0]:.4f})'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses, top1=top1_acc))
    print('* Top 1-err {top1.avg[0]:.3f}  Test Loss {loss.avg:.3f}'.format(
        top1=top1_acc, loss=losses))
    return top1_acc.avg, losses.avg

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')
    save_args(args, os.path.join(directory, 'argument.json'))

if __name__ == '__main__':
    main()