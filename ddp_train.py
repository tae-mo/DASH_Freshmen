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
import argparse
from data_utils import TrainDataset
from score_utils import *
from tqdm import tqdm

train_path = '/media/data1/data/Imagenet'
valid_path = '/media/data1/data/Imagenet'
check_path = "/media/data1/geonho/resnet50_checkpoints/snapshot.pt"


def setup():
    init_process_group(backend="nccl")


def cleanup():
    destroy_process_group()


class Validator:
    def __init__(
        self,
        model: torch.nn.Module,
        valid_loader: DataLoader,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.gpu_id = self.gpu_id
        self.model = model.to(self.gpu_id)
        self.valid_loader = valid_loader
        self.model = DDP(model, device_ids = [self.gpu_id])

    # 배치 한번
    def _run_batch(self, source, targets):
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        return loss

    # 전체 validation
    def validate(self):
        b_sz = len(next(iter(self.valid_loader))[0])
        print(f"[GPU{self.gpu_id}] Valid | Batchsize: {b_sz} | Steps: {len(self.valid_loader)}")
        valid_bar = tqdm(self.valid_loader)
        for source, targets in valid_bar:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            valid_bar.set_description(desc = '[Validation]   cost: %.9f' % (
                loss,
            ))


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0

        self.cost = AverageMeter()

        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(model, device_ids = [self.gpu_id])
        self.validator = Validator(model, valid_loader)

    # 배치 한번
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets, reduction='mean')
        loss.backward()
        self.optimizer.step()
        return loss

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_ron = snapshot["EPOCHS_RUN"]
        print("Resuming training from snapshot at Epoch {self.epochs_run}")

    # 에폭 한번
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)

        train_bar = tqdm(self.train_loader)
        for source, targets in train_bar:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)

            num_data = len(source)

            loss = self._run_batch(source, targets)

            self.cost.update(loss, num_data)

            train_bar.set_description(desc = '[%d/%d]   cost: %.9f' % (
                epoch+1, self.epochs_run, loss/num_data,
            ))
        print(f'Epoch {epoch}, cost : {self.cost.avg}')
        
    # 체크포인트 저장
    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at", self.snapshot_path)

    # 검증 수행
    def _run_validation(self):
        self.validator.validate()

    # 전체 훈련
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):

            self._run_epoch(epoch)
            self._run_validation()

            # save_every의 에폭 간격으로 저장
            if self.gpu_id==0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)





def load_train_objs():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = torchvision.datasets.ImageNet(train_path, split='train', transform=transform)
    val_set = torchvision.datasets.ImageNet(valid_path, split='val', transform=transform)
    model = ResNet50()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-3)
    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, mode: str = 'Train'):
    if mode == 'Train':
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            sampler=DistributedSampler(dataset, shuffle=True)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
            sampler=DistributedSampler(dataset, shuffle=False)
        )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = check_path):
    setup()
    train_dataset, valid_dataset, model, optimizer=load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size, mode = 'Train')
    valid_data = prepare_dataloader(valid_dataset, batch_size, mode = 'Test')
    trainer = Trainer(model, train_data, valid_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    args = get_args()
    total_epochs = int(args.epochs)
    save_every = int(args.save_every)
    batch_size = int(args.batch_size)
    
    main(save_every, total_epochs, batch_size)


# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu ddp_train.py --epochs 20 --save_every 2 --batch_size 64





