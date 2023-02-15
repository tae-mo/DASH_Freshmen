import argparse
import os

import time
from datetime import datetime as dt
import random

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

from utils import *
from Train import *
from model.xception import Xception
from data import get_dataloader

def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_name", type=str, default="DeepFake",
                        choices=['DeepFake', 'DeepFakeDetection', 'Face2Face', 'FaceSwap', 'NeuralTextures'])
    parser.add_argument("--data_path", type=str, default='/media/data1/sangyong/df_datasets')
    parser.add_argument("--n_workers", type=int, default=4)
    #### train & test ####
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="xception", choices=["xception", "clrnet"])
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=5)
    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default='/media/data1/sangyong/deepfake_detection/save')
    parser.add_argument("--model_load_path", default=None)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument("--dist_backend", type=str, default='nccl')
    parser.add_argument("--use_wandb", default=False)
    args = parser.parse_args()
    
    if args.DDP:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = torch.cuda.device_count()
        args.batch_size = args.batch_size // torch.cuda.device_count()
    else:
        args.local_rank = 0
    args.save_name = f"[data-{args.data_name}]_[bs-{args.batch_size}]_"+\
                     f"[m-{args.model}]_[optim-{args.optimizer}]_[date-{dt.now().strftime('%Y%m%d')}]"
    args.save_dir = os.path.join(args.save_root_dir, args.save_name)
    args.model_save_dir = os.path.join(args.save_dir, "save_model")
    args.logger_path = os.path.join(args.save_dir, "log.txt")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    return args

def main(args, logger):
    if args.model == "xception":
        model = Xception(num_classes=2).cuda(args.local_rank)
    elif args.model == "clrnet":
        model = None
    
    train_loader, valid_loader, train_sampler = get_dataloader(args)
    
    if args.DDP:
        sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model backward pass에 연관되지 않은 parameter들을 mark해서 DDP가 해당 파라미터들의 gradient들을 영원히 기다리는 것을 방지 한다. 
        model = torch.nn.parallel.DistributedDataParallel(sync_bn_module, device_ids=[args.local_rank], find_unused_parameters=True) 

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr = args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr = args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"optimizer {args.optimzier} is not implemented. please change")
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 1
    best_acc = -1
    best_loss = float('inf')
    
    if args.model_load_path:
        logger.write(f"model load from {args.model_load_path}\n")
        if not args.DDP:
            checkpoint = torch.load(args.model_load_path)
        else:
            dist.barrier()
            checkpoint = torch.load(args.model_load_path, map_location={"cuda:0": f"cuda:{args.local_rank}"})
            
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.write(f"model is successfully loaded\n"
                     f"start epoch: {start_epoch}, best_acc: {best_acc}")
        
        del checkpoint
    
    for epoch in range(start_epoch, args.epochs):
        if args.DDP:
            train_sampler.set_epoch(epoch)
            
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        
        valid_loss, valid_acc = validate(valid_loader, model, criterion, args)
        
        if args.local_rank == 0:
            if valid_acc > best_acc:
                logger.write(f"Best accuracy: {best_acc:.4f} -> {valid_acc:.4f}\n")
                best_acc = valid_acc
                checkpoint_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc
                }
                model_save_path = os.path.join(args.model_save_dir, f"{epoch}_{best_acc}.pth")
                torch.save(checkpoint_dict, model_save_path)
            logger.write(f"[Epoch-{epoch}]_[Train accuracy-{train_acc}]_"
                          f"[Train loss-{train_loss:.4f}]_[Valid acc-{valid_acc}]_[Valid loss-{valid_loss}]\n")
            if valid_loss < best_loss:
                best_loss = valid_loss
            
            if args.use_wandb:
                wandb_msg = {"Train acc": train_acc,
                             "valid acc": valid_acc,
                             "Train loss": train_loss,
                             "valid loss": valid_loss}
                wandb.log(wandb_msg)
        logger.write(f"[Best accuracy-{best_acc}]_[Best loss-{best_loss}]")
        scheduler.step()
        # dist.barrier()
        
if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.logger_path)
    print_args(args, logger=logger)
    if args.use_wandb and args.local_rank ==0:
        wandb.init(project="Deepfake Detection", name=args.save_name, notes=args.save_name)
        wandb.config.update(args)
    if args.DDP:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend)
        logger.write(f'DDP using {args.world_size} GPUS\n')
    start_time = time.time()
    if args.data_name in ['DeepFake', 'DeepFakeDetection', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
        main(args, logger)
    else:
        raise NotImplementedError(f"data {args.data_name} is not implemented")
    
    logger.write(f"total time: {time.time() - start_time}")
    