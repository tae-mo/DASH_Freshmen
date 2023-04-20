import os
import logging
import wandb
import csv
import gc
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import barrier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict

import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms

from model.resnet import *
from model.vit import ViTscratch as ViT


class Trainer:
    def __init__( self,
        train_data: DataLoader,
        test_data: DataLoader,
        save_every: int,
        model_type: str,
        args,
        model_depth: int = 50,
        lr: float = 0.001,
        proj_name : str = 'imagenet',
        snapshot_file: str = 'snapshot.pt'
    ) -> None:
        self.losses, self.acces = AverageMeter(), AverageMeter()
        self.dataloader = [train_data, test_data]
        self.logger = logging.getLogger()
        self.gpu_id = int(os.environ["LOCAL_RANK"]) # torchrun provide 
        self.save_every, self.lr = save_every, lr
        self.epochs_run, self.proj_name, self.snapshot_file = 0, proj_name, snapshot_file
        self.logs = [{},{}] # log[0] : train, log[1] : val
        
        if snapshot_file[-3:] != '.pt': self.snapshot_file = snapshot_file+'.pt'
        
        self.create_model(model_depth, model_type, args.vit_head, args.drop_out)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.gpu_id], 
                         output_device=torch.cuda.current_device()) # self.model_without_ddp = self.model.module

        self.loss_fn = nn.CrossEntropyLoss().to(self.gpu_id)  #F.cross_entropy()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=args.factor, patience=1)
        
        # .../result/proj_name/snapshot_file(.pt)
        resume, self.snapshot_path = self._load_snapshot(self.snapshot_file, resume=args.resume)
        if self.gpu_id == 0 : self._load_wandb(args)
        
    def _load_wandb(self, args):
        wandb.login()
        wandb.init(project="dash", entity="merily", name=self.proj_name, id=self.proj_name, resume='allow')
        wandb.config.update(args)
        
    def create_model(self, model_depth, model_type, heads:int=8, drop_out:float=0.1) :
        if model_type == 'resnet' :
            assert model_depth in [0, 18, 34, 50, 101, 152], f'ResNet{model_depth}: Unknown architecture!'
            if model_depth == 0 :
                self.model = models.__dict__['resnet50'](pretrained=False)
            else :
                self.model = myResnet(num_layers=model_depth, block=Block)
        elif model_type == 'vit' :
            assert model_depth in [8, 12], f'ViT{model_depth}: Unknown architecture!'
            self.model = ViT(depth = model_depth, drop_p=drop_out, forward_drop_p=.0, num_heads=heads)
        self.model = self.model.to(self.gpu_id) # "cuda:{}".format(self.gpu_id)) #f"Resuming training from snapshot at Epoch {self.epochs_run}"
        self.model.eval()
        
    def _load_snapshot(self, snapshot_file, resume): # snapshot.pt
        loc = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id} # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        dir = os.path.realpath(__file__) # current_file_path = os.path.abspath(os.path.join(dir, os.pardir))
        _, self.save_dir = make_sure_dir(os.path.join(os.path.abspath(os.path.join(dir, os.pardir)), "result", self.proj_name))
        snapshot_path = os.path.join(self.save_dir, snapshot_file) # print(snapshot_path, loc)
        
        if resume and os.path.exists(snapshot_path): # 기존에 저장한 모델이 있고, 이어서 하고 싶을 때
            print_once(self.gpu_id, "Loading snapshot file ... ")
            snapshot = torch.load(snapshot_path, map_location=loc) # cuda 맞춰주기  print(snapshot["MODEL_STATE"])
            '''new_state_dict = OrderedDict()
            for k, v in snapshot["MODEL_STATE"].items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v'''
            self.model.load_state_dict(snapshot["MODEL_STATE"]) # new_state_dict
            self.epochs_run = snapshot["EPOCHS_RUN"] # epoch를 0에서 이전까지 기록의 숫자로 변경 
            print_once(self.gpu_id, f"Resuming training from snapshot at Epoch {self.epochs_run}")
            return True, snapshot
        else  :
            print_once(self.gpu_id, "Starting training at Epoch 0... ")
            return False, snapshot_path

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "Loss": self.losses.avg,
            "Acc": self.acces.avg,
        }
        torch.save(snapshot, self.snapshot_path) # 저장 방식에 대한 고찰 요구됨 
        print_once(self.gpu_id, f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
     
    def _save_log(self, epoch):
        log = {"Epoch": epoch, "Train Loss":self.logs[0]['loss'], "Train Acc":self.logs[1]['acc'], 
                        "Val Loss":self.logs[1]['loss'], "Val Acc":self.logs[1]['acc']}
        with open(self.save_dir+'/log_csv.csv','a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(log.keys())
            w.writerow(log.values())
        self.logger.info(log)
        wandb.log(log)
 
    # inner for
    def _run_batch(self, source, targets, mode='train'):
        self.optimizer.zero_grad()
        
        output = self.model(source)
        loss = F.cross_entropy(output, targets) # .to(self.gpuid)
        acc = Accuracy(output, targets) # .to(self.gpuid)
        
        self.losses.update(val=loss.item(), batch_sz=source.size(0))
        self.acces.update(val=acc, batch_sz=source.size(0))
        
        if mode == 'train' :
            loss.backward()
            self.optimizer.step()

    # outer for
    def _run_epoch(self, epoch, mode='train', m = 0):
        self.losses, self.acces = AverageMeter(), AverageMeter()
        if mode != 'train' : m = 1
        self.dataloader[m].sampler.set_epoch(epoch)            # for suffle data between gpus
        batch_size = len(next(iter(self.dataloader[m]))[0])    # take batch size from distributed data loader's shape
        tq = tqdm.tqdm(total=(len(self.dataloader[m]) * batch_size))
        tq.set_description(f'>> {self.gpu_id} {mode} epoch {epoch}')
        for _, (source, targets) in enumerate(self.dataloader[m]): # for source, targets in self.dataloader[m]:
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            self._run_batch(source, targets, mode=mode) 
            tq.set_postfix(Loss ='{:.5f}'.format(self.losses.avg), Acc ='{:.5f}'.format(self.acces.avg))
            tq.update(batch_size) 
        tq.close()
        
        if self.gpu_id == 0 :self.logs[m] = {"loss": self.losses.avg, "acc" : self.acces.avg}

    def train(self, max_epochs: int, mode:str="train"):
        self.logger = get_logger(self.save_dir, mode) #.../result/proj print('Current cuda device:', torch.cuda.current_device())
        
        for epoch in range(self.epochs_run, max_epochs): # defualt : 0~max_epoches
            self.model.train()
            self._run_epoch(epoch, mode=mode) # 한 epoch마다 train 
            self._val(current_epoch = epoch)
            
            if (self.gpu_id == 0) : 
                self._save_snapshot(epoch) # 매 epoch마다 학습완료된 모델 저장 
                self._save_log(epoch)
            barrier()
            
            gc.collect()
            torch.cuda.empty_cache()
        #
        if (self.gpu_id == 0) : wandb.finish()
        print_once(self.gpu_id, "Done Every Train and Validation.")
        
    def _val(self, current_epoch):
        self.model.eval()
        with torch.no_grad(): # https://github.com/pytorch/pytorch/issues/16417#issuecomment-566654504 
            self._run_epoch(epoch=current_epoch, mode='val') # validate
# end of Trainer clss #####################################################################

class AverageMeter (object):
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, batch_sz=1):
        self.val = val
        self.sum += val * batch_sz
        self.count += batch_sz # 예제 수를 세어서 배치크기만큼 증가시킨다. 
        self.avg = self.sum / self.count
#

def Accuracy(output, targets):
    preds = output.argmax(dim=1)    # 가장 높은 값을 가진 인덱스를 출력한다. 
    return int( preds.eq(targets).sum() )
#

def get_logger(path, mode='train'):
    logger = logging.getLogger()
    if len(logger.handlers) > 0 : return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(os.path.join(path, mode+'.log' )) # 'train.log'
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
# end of get_logger ft

def make_sure_dir(dir):
    if not os.path.exists(dir):
        print (os.path.exists(dir))
        os.mkdir(dir)
    return os.path.exists(dir), dir
#

def prepare_dataloader(dataset: Dataset, batch_size: int, scale=1, prop=.9, shuffle=True):
    origin_sz = int(len(dataset))
    use_sz = int(origin_sz* scale)
    if scale < 1 : dataset, _ = random_split(dataset, [use_sz, origin_sz-use_sz])
    train, test = random_split(dataset, [int(use_sz*prop), use_sz-int(use_sz*prop)])
    return DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8,
        sampler=DistributedSampler(train, shuffle=shuffle) 
        ), DataLoader(test, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8,
        sampler=DistributedSampler(test, shuffle=False) )
# end of prepare_dataloader ft

def load_train_objs(mode='train'):
    data_path = "/home/data/Imagenet"
    traindir = os.path.join(data_path, mode)
    # valdir = os.path.join(args.data, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomVerticalFlip(),
                                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                        transforms.ToTensor(),
                                                        normalize]))
    return train_dataset
# end of load_train_objs ft

def print_once(gpu_id, message):
    if gpu_id == 0:
        print(message)
#


