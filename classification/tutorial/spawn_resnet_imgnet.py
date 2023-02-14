import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms

# from torchmetrics.classification import MulticlassPrecisionRecallCurve

import os
import tqdm
import logging


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1" 
    print("ddp_setup ft >>> rank:", rank, ", world_size:", world_size)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
# end of ddp_setup ft

class Trainer:
    def __init__( self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        snapshot_path: str
    ) -> None:
        self.train_loss = 0
        self.losses = []
        
        self.acc_per_batch = 0
        self.temp_acc = 0
        self.acces = []
        
        self.model = model.to(gpu_id)
        self.gpu_id = gpu_id # int(os.environ["LOCAL_RANK"])
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot file ... ")
            self._load_snapshot(snapshot_path)
            
        # Initializes internal Module state, shared by both nn.Module and ScriptModule.
        self.model = DDP(model, device_ids=[gpu_id]) 
    #
    
    def _load_snapshot(self, snapshot_path): # snapshot.pth
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    #    
    
    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        PATH = "result/" + self.snapshot_path
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    #    
    
    
    # 111111111111111111111111111111111111111111
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        
        output = self.model(source)
        
        loss = F.cross_entropy(output, targets)
        self.losses.append(loss.item())
        
        loss.backward()
        self.optimizer.step()
        
        preds = output.argmax(dim=1)
        self.temp_acc += preds.eq(targets).sum()
        self.acces.append(self.temp_acc)
    #

    # 222222222222222222222222222222222222222222
    def _run_epoch(self, epoch):
        self.temp_acc = 0
        batch_size = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        
        if self.gpu_id == 0 : 
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        
        # train per epoch with batch
        tq = tqdm.tqdm(total=(len(self.train_data) * batch_size))
        tq.set_description(f'Train Epoch {epoch}')
        
        for batch_idx, (source, targets) in enumerate(self.train_data): # for source, targets in self.train_data:
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            
            self._run_batch(source, targets) 
            self.loss_per_batch = (sum(self.losses)/len(self.losses))
            self.acc_per_batch  = (sum(self.acces)/len(self.acces))
            
            acc = 100. * self.temp_acc / len(self.train_data.dataset)
            tq.set_postfix( Loss ='{:.5f} per batch'.format(self.loss_per_batch), 
                            Acc  ='{:.5f} per batch'.format(acc), temp ='{:.2f}'.format(self.temp_acc))
            tq.update(batch_size) # can not get it...
        tq.close()
        
        # train score
    #

    # 333333333333333333333333333333333333333333333
    def train(self, max_epochs: int):
        
        if not os.path.exists("/home/jiwon/DASH_Freshmen/example/tutorial/result"):
            os.mkdir("/home/jiwon/DASH_Freshmen/example/tutorial/result")
        if not os.path.exists("/home/jiwon/DASH_Freshmen/example/tutorial/result/train"):
            os.mkdir("/home/jiwon/DASH_Freshmen/example/tutorial/result/train")
        
        logger = get_logger('/home/jiwon/DASH_Freshmen/example/tutorial/result/train')
        print('Current cuda device:', torch.cuda.current_device())
        
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch) # train
            
            # loss 값 저장 
            train_log = {"epoch": epoch, "Train Loss": self.loss_per_batch, "Train Acc": self.acc_per_batch}
            print(train_log)
            logger.info(train_log)
            
            # save_every일 때 마다 체크 포인트 저장 
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                # self._save_checkpoint(epoch) #
                self._save_snapshot(epoch)
        return self.loss_per_batch, self.acc_per_batch
    #

# end of Trainer clss

def get_logger(path):
    logger = logging.getLogger()
    
    if len(logger.handlers) > 0 : return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(path+'.log') # 'train.log'
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
# end of get_logger ft


def load_train_objs():
    traindir = os.path.join("/home/data/Imagenet", 'val')
    # valdir = os.path.join(args.data, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                    )

    train_dataset = datasets.ImageFolder( traindir, transforms.Compose([
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        normalize])
                                        )
    # train_dataset, _ = randomSplit
    
    """val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]))"""
    
    model = models.__dict__['resnet50'](pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    return train_dataset, model, optimizer
# end of load_train_objs ft

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset) # Sampler that restricts data loading to a subset of the dataset
    )
# end of prepare_dataloader ft

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pth"):
    # distributed data parallel
    ddp_setup(rank, world_size)
    
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    
    trainer = Trainer(model, train_data, optimizer, rank, save_every, snapshot_path)
    train_loss, train_acc = trainer.train(total_epochs)
    
    destroy_process_group() # Destroy a given process group, and deinitialize the distributed package
#


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = 2 #torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)