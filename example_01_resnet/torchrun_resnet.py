import os
import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
    parser.add_argument('--save_every', '-i', default=1)
    parser.add_argument('--total_epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--scale', '-s', default=1)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    parser.add_argument('--model', '-m', type=int, default=50)
    parser.add_argument('--rand_seed', '-rs', type=int, default=11)
    # parser.add_argument('--train', '-t', default=True)
    # parser.add_argument('--project_name', '-n', default='crack_seg')
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
    init_process_group(backend="nccl")
# end of ddp_setup ft

class Trainer:
    def __init__( self,
        model_depth: int,
        train_data: DataLoader,
        test_data: DataLoader,
        save_every: int,
        snapshot_file: str = 'snapshot.pt'
    ) -> None:
        self.losses = AverageMeter()
        self.acces = AverageMeter()
        self.logger = logging.getLogger()
        self.train_data, self.test_data = train_data, test_data
        self.gpu_id = int(os.environ["LOCAL_RANK"]) #torchrun provide 
        
        assert model_depth in [0, 18, 34, 50, 101, 152], f'ResNet{model_depth}: Unknown architecture!'
        if model_depth == 0 :
            self.model = models.__dict__['resnet50'](pretrained=True)
        else :
            self.model = myResnet(num_layers=model_depth, block=Block)
        self.model = self.model.to(self.gpu_id) # "cuda:{}".format(self.gpu_id)) #f"Resuming training from snapshot at Epoch {self.epochs_run}"
    
        
        self.save_every = save_every
        self.epochs_run = 0     # before get shpapshot, 0 is default
        self.snapshot_path = self._load_snapshot(snapshot_file)

        # Initializes internal Module state, shared by both nn.Module and ScriptModule.
        self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=torch.cuda.current_device()) 
        self.loss_fn = nn.CrossEntropyLoss().to(self.gpu_id)  #F.cross_entropy()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        

    
    def _load_snapshot(self, snapshot_file): # snapshot.pt
        loc = f"cuda:{self.gpu_id}"
        
        # current_file_path = os.getcwd() 
        dir = os.path.realpath(__file__)
        current_file_path = os.path.abspath(os.path.join(dir, os.pardir))
        _, self.result_dir = make_sure_dir(os.path.join(current_file_path, "result"))
        snapshot_path = os.path.join(self.result_dir, snapshot_file)
        
        if os.path.exists(snapshot_path): # 기존에 저장한 모델이 있을 때 
            print_once(self.gpu_id, "Loading snapshot file ... ")
            snapshot = torch.load(snapshot_path, map_location=loc)
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        
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
        }
        torch.save(snapshot, self.snapshot_path)
        print_once(self.gpu_id, f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
 
    
    # 111111111111111111111111111111111111111111
    def _run_batch(self, source, targets, mode='train'):
        self.optimizer.zero_grad()
        
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        acc = Accuracy(output, targets)
        
        self.losses.update(val=loss.item(), batch_sz=source.size(0))
        self.acces.update(val=acc, batch_sz=source.size(0))
        
        if mode == 'train' :
            loss.backward()
            self.optimizer.step()


    # 222222222222222222222222222222222222222222
    def _run_epoch(self, epoch, mode='train'):
        self.losses      = AverageMeter()
        self.acces       = AverageMeter()
    
        batch_size = len(next(iter(self.train_data))[0])    # take batch size from distributed data loader's shape
        self.train_data.sampler.set_epoch(epoch)            # for suffle data between gpus
        self.model.train()
                
        print_once(self.gpu_id, f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")

        tq = tqdm.tqdm(total=(len(self.train_data) * batch_size))
        tq.set_description(f'Train Epoch {epoch}')
        for batch_idx, (source, targets) in enumerate(self.train_data): # for source, targets in self.train_data:
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            
            self._run_batch(source, targets) 
            
            tq.set_postfix(Loss ='{:.5f}'.format(self.losses.avg), Acc ='{:.5f}'.format(self.acces.avg))
            tq.update(batch_size) # can not get it...
        tq.close()
        
        if self.gpu_id == 0:
            # log train score
            train_log = { "mode": mode, "epoch": epoch, "loss": self.losses.avg, "acc" : self.acces.avg }
            # loss 값 저장 
            self.logger.info(train_log)


    # 333333333333333333333333333333333333333333333
    def train(self, max_epochs: int):
        mode = "train"
        
        make_sure_dir(os.path.join(self.result_dir, mode))  #.../result/train/
        self.logger = get_logger(self.result_dir, mode)     #.../result/train/
        
        print('Current cuda device:', torch.cuda.current_device())
        
        for epoch in range(self.epochs_run, max_epochs): # defualt : 0~max_epoches
            # 한 epoch마다 train 
            self._run_epoch(epoch) 
            # 매 epoch마다 학습완료된 모델 저장  
            if self.gpu_id == 0: self._save_snapshot(epoch)
            
        #     # save_every 마다 검증 시행 
        #     if epoch % self.save_every == 0: 
        #         self._val(current_epoch = epoch)    
        # #
        print_once(self.gpu_id, "Train Done!")


    def _val(self, current_epoch):
        self._run_epoch(epoch=1, mode='val') # validate

# end of Trainer clss #####################################################################

def main(
    save_every: int, 
    total_epochs: int, 
    batch_size: int,
    lr: float,
    model_depth: int,
    snapshot_path: str = "snapshot.pt",
    scale: float = 1.0
    ):
    ddp_setup() # distributed data parallel
    dataset = load_train_objs()
    train_data, test_data = prepare_dataloader(dataset, batch_size, scale)
    
    trainer = Trainer(model_depth=model_depth, train_data=train_data, test_data=test_data, save_every=save_every, snapshot_file=snapshot_path)
    trainer.train(total_epochs)
    
    destroy_process_group() # Destroy a given process group, and deinitialize the distributed package

#


if __name__ == "__main__":
    args = parse_args()
    # world_size = 2 if torch.cuda.device_count() > 2 else torch.cuda.device_count()
    torch.manual_seed(args.rand_seed)
    
    main(save_every=args.save_every, 
         total_epochs=args.total_epochs, 
         batch_size=args.batch_size,
         lr=args.lr,
         model_depth=args.model,
         scale=args.scale,
         )
    
    