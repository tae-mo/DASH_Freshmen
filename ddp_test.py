import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from resnet import *

from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
import torchvision.transforms as transforms
import torchvision

from arg_parse import get_args

from datautils import TrainDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '10.201.134.132'
    os.environ['MASTER_PORT'] = '8891'

    init_process_group("nccl", rank = rank, world_size = world_size)


def cleanup():
    destroy_process_group()


class Validator:
    def __init__(
        self,
        model: torch.nn.Module,
        valid_data: DataLoader,
        gpu_id: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.valid_data = valid_data
        self.model = DDP(model, device_ids = [gpu_id])

    # 배치 한번
    def _run_batch(self, source, targets):
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        print('validation loss : ', loss)

    # 에폭 한번
    def _run_valid(self):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Valid | Batchsize: {b_sz} | Steps: {len(self.valid_data)}")
        for source, targets in self.valid_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    # 전체 validation
    def validate(self):
        self._run_valid()



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int, 
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids = [gpu_id])
        self.validator = Validator(model, valid_data, gpu_id)

    # 배치 한번
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    # 에폭 한번
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at checkpoint.pt")

    def _run_validation(self):
        Validator.validate()

    # 전체 훈련
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            self._run_validation()
            # save_every의 에폭 간격으로 저장
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)





def load_train_objs():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = torchvision.datasets.ImageNet('/media/data1/data/Imagenet', split='train', transform=transform)
    val_set = torchvision.datasets.ImageNet('/media/data1/data/Imagenet', split='val', transform=transform)
    model = ResNet50()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shufflt=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    setup(rank,world_size)
    train_dataset, valid_dataset, model, optimizer=load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size=32)
    valid_data = prepare_dataloader(valid_dataset, batch_size=32)
    trainer = Trainer(model, train_data, valid_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    args = get_args()
    total_epochs = int(args.epochs)
    save_every = int(args.save_every)
    device = 0
    main(device, total_epochs, save_every)