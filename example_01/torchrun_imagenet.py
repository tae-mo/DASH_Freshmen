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
    parser.add_argument('--scale', '-s', default=1)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    parser.add_argument('--model', '-m', type=int, default=50)
    parser.add_argument('--rand_seed', '-rs', type=int, default=11)
    parser.add_argument('--name', '-n', type=str, default='resnet_snapshot.pt')
    # parser.add_argument('--train', '-t', default=True)
    parser.add_argument('--device', '-d', default="cuda")
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

class Trainer:
    def __init__( self,
        model_depth: int,
        train_data: DataLoader,
        test_data: DataLoader,
        save_every: int,
        lr: float,
        proj_name : str,
        snapshot_file: str = 'snapshot.pt',
    ) -> None:
        self.losses = AverageMeter()
        self.acces = AverageMeter()
        self.logger = logging.getLogger()
        self.train_data, self.test_data = train_data, test_data
        self.gpu_id = int(os.environ["LOCAL_RANK"]) #torchrun provide 
        self.save_every = save_every
        self.lr = lr
        self.epochs_run = 0     # before get shpapshot, 0 is default
        self.proj_name = proj_name
        self.logs = [{},{}] # log[0] : train, log[1] : val
        
        if snapshot_file[-3:] != '.pt': 
            self.snapshot_file = snapshot_file+'.pt'
        else :
            self.snapshot_file = snapshot_file
        
        self.create_model(model_depth)

        # Initializes internal Module state, shared by both nn.Module and ScriptModule.
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=torch.cuda.current_device()) 
        self.model_without_ddp = self.model.module
        
        wandb.watch(self.model)
        self.loss_fn = nn.CrossEntropyLoss().to(self.gpu_id)  #F.cross_entropy()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        
        # .../result/proj_name/snapshot_file(.pt)
        self.result_dir = ""
        self.snapshot_path = self._load_snapshot(self.snapshot_file)
        
    def create_model(self, model_depth) :
        assert model_depth in [0, 1, 18, 34, 50, 101, 152], f'ResNet{model_depth}: Unknown architecture!'
        if model_depth == 0 :
            self.model = models.__dict__['resnet50'](pretrained=False)
        elif model_depth == 1 :
            self.model = ViT(depth = 12, drop_p=.1, forward_drop_p=.0, num_heads=8)
        else :
            self.model = myResnet(num_layers=model_depth, block=Block)
        self.model = self.model.to(self.gpu_id) # "cuda:{}".format(self.gpu_id)) #f"Resuming training from snapshot at Epoch {self.epochs_run}"
        self.model.eval()
        
    def _load_snapshot(self, snapshot_file): # snapshot.pt
        loc = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id} # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        dir = os.path.realpath(__file__)
        current_file_path = os.path.abspath(os.path.join(dir, os.pardir))
        _, self.result_dir = make_sure_dir(os.path.join(current_file_path, "result"))
        _, self.result_dir = make_sure_dir(os.path.join(self.result_dir, self.proj_name))
        snapshot_path = os.path.join(self.result_dir, snapshot_file)
        # print(snapshot_path, loc)
        
        if os.path.exists(snapshot_path): # 기존에 저장한 모델이 있을 때 
            print_once(self.gpu_id, "Loading snapshot file ... ")
            snapshot = torch.load(snapshot_path, map_location=loc)
            print(snapshot["MODEL_STATE"])
            new_state_dict = OrderedDict()
            for k, v in snapshot["MODEL_STATE"].items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
                        
            self.model.load_state_dict(new_state_dict) #snapshot["MODEL_STATE"])
        
            self.epochs_run = snapshot["EPOCHS_RUN"] # epoch를 0에서 이전까지 기록의 숫자로 변경 
            print_once(self.gpu_id, f"Resuming training from snapshot at Epoch {self.epochs_run}")
        else  :
            print_once(self.gpu_id, "Starting training at Epoch 0... ")
        return snapshot_path

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "Loss": self.losses.avg,
            "Acc": self.acces.avg,
        }
        torch.save(snapshot, self.snapshot_path)
        print_once(self.gpu_id, f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
     
    def _save_log(self, epoch):
        log = {"Epoch": epoch, "Train Loss":self.logs[0]['loss'], "Train Acc":self.logs[1]['acc'], 
                        "Val Loss":self.logs[1]['loss'], "Val Acc":self.logs[1]['acc']}
        self.logger.info(log)
        wandb.log(log)
        with open(self.result_dir+'/log_csv.csv','a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(log.keys())
                w.writerow(log.values())
            else:
                w.writerow(log.values())
 
    # 111111111111111111111111111111111111111111
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

    # 222222222222222222222222222222222222222222
    def _run_epoch(self, epoch, mode='train'):
        self.losses      = AverageMeter()
        self.acces       = AverageMeter()
        if mode == 'train' :
            self.dataloader = self.train_data
            self.model.train()
            m = 0
        else :
            self.dataloader = self.test_data
            self.model.eval()
            m = 1
        
        batch_size = len(next(iter(self.dataloader))[0])    # take batch size from distributed data loader's shape
        self.dataloader.sampler.set_epoch(epoch)            # for suffle data between gpus
        
        tq = tqdm.tqdm(total=(len(self.dataloader) * batch_size))
        tq.set_description(f'>> {self.gpu_id} {mode} epoch {epoch}')
        for _, (source, targets) in enumerate(self.dataloader): # for source, targets in self.dataloader:
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            self._run_batch(source, targets, mode=mode) 
            tq.set_postfix(Loss ='{:.5f}'.format(self.losses.avg), Acc ='{:.5f}'.format(self.acces.avg))
            tq.update(batch_size) 
        tq.close()
        if (self.gpu_id == 0) :
            self.logs[m] = {"loss": self.losses.avg, "acc" : self.acces.avg }

    # 333333333333333333333333333333333333333333333
    def train(self, max_epochs: int):
        mode = "train"
        
        make_sure_dir(os.path.join(self.result_dir, mode))  #.../result/train/
        self.logger = get_logger(self.result_dir, mode)     #.../result/train/
        print('Current cuda device:', torch.cuda.current_device())
        
        for epoch in range(self.epochs_run, max_epochs): # defualt : 0~max_epoches
            
            self._run_epoch(epoch, mode=mode) # 한 epoch마다 train 
            if self.gpu_id == 0: self._save_snapshot(epoch) # 매 epoch마다 학습완료된 모델 저장  
            barrier()
            
            # if epoch % self.save_every == 0: # save_every 마다 검증 시행 
            self._val(current_epoch = epoch)
            if (self.gpu_id == 0) : self._save_log(epoch)
            
            gc.collect()
            torch.cuda.empty_cache()
            barrier()
        #
        print_once(self.gpu_id, "Done Every Train and Validation.")

    def _val(self, current_epoch):
        with torch.no_grad(): # https://github.com/pytorch/pytorch/issues/16417#issuecomment-566654504 
            self._run_epoch(epoch=current_epoch, mode='val') # validate
# end of Trainer clss #####################################################################

def main(save_every: int, total_epochs: int, batch_size: int, lr: float, model_depth: int,
         snapshot_path: str = "snapshot.pt", scale: float = 1.0, proj_name : str = 'resnet'):
    
    gc.collect()
    torch.cuda.empty_cache()
    
    ddp_setup() # distributed data parallel
    dataset = load_train_objs()
    train_data, test_data = prepare_dataloader(dataset, batch_size, scale)
    
    trainer = Trainer(model_depth=model_depth, train_data=train_data, test_data=test_data, save_every=save_every, lr=lr, snapshot_file=snapshot_path, proj_name=proj_name)
    trainer.train(total_epochs)
    destroy_process_group() # Destroy a given process group, and deinitialize the distributed package
#

if __name__ == "__main__":
    args = parse_args()                 # world_size = 2 if torch.cuda.device_count() > 2 else torch.cuda.device_count()
    torch.manual_seed(args.rand_seed)
    wandb.login()
    wandb.init(project="dash", entity="merily", name=args.name)
    wandb.config.update(args) # adds all of the arguments as config variables
    main(save_every=args.save_every, total_epochs=args.total_epochs, batch_size=args.batch_size, lr=args.lr,
         model_depth=args.model, scale=args.scale, proj_name=args.name)
    wandb.finish()
