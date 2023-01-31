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
import time


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "115.145.134.134"
    os.environ["MASTER_PORT"] = "2132"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        print(torch.cuda.current_device(), gpu_id)

    def _run_batch(self, source, targets, loop, mean_loss):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        mean_loss.append(loss.item())
        loss.backward()
        self.optimizer.step()
        loop.set_postfix(loss=loss.item())

    def _run_epoch(self, epoch):
        loop = tqdm(self.train_data, leave=True) #진행률
        mean_loss = []

        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in tqdm(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets, loop, mean_loss)
            

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "/home/syon1203/DASH_Freshmen/resnet50_classification/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            #if self.gpu_id == 0 and epoch % self.save_every == 0:
                #self._save_checkpoint(epoch)
















def load_train_objs():
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    #transf_train = tr.Compose([tr.Resize((224, 224)),tr.ToTensor(),tr.RandomHorizontalFlip(), tr.Normalize(*stats, inplace=True)])

    transf_train = tr.Compose([tr.ToTensor(),tr.Normalize(*stats, inplace=True)])

    #train_set = ImageFolder('/home/data/Imagenet/train', transform=transf_train,target_transform=None)

    train_set = torchvision.datasets.FakeData(1000, (3, 32, 32), 10, transform=transf_train,target_transform=None)

    model = MD.ResNet50()  # load your model
    #model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

  
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)

    print("rank", rank,"world_size", world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=1 , type=int,help='Total epochs to train the model')
    parser.add_argument('--save_every',default=1, type=int,help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    # world_size = torch.cuda.device_count()

    world_size = 2

    if torch.cuda.device_count() < 2 :
        world_size = torch.cuda.device_count()
    #elif torch.cuda.device_count() == 0 :
    print(world_size, torch.cuda.device_count())
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)