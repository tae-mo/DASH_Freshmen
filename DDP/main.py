import os
import argparse
import wandb

import torch
import torch.nn as nn
import torch.distributed as dist

from datetime import datetime
# from sched import scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from prepare_dataset import pre_dset
from train import train, validate
from utils import save_ckpt

def parse_args():
    parser = argparse.ArgumentParser(description="Imagenet Training")
    
    ## Config
    parser.add_argument("--exp", type=str, default="./model_checkpoint") # checkpoint를 저장할 경로
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--data", type=str, default="imagenet")
    parser.add_argument("--local_rank", type=int, default=0) # GPU process ID
    
    ## Training
    parser.add_argument("--learning_rate", type=float, default=1e-4) # 학습률
    parser.add_argument("--epochs", type=int, default=100) # 학습 횟수
    parser.add_argument("--batch_size", type=int, default=32) # batch size
    parser.add_argument("--every", type=int, default=-1) # 학습 중간에 loss를 출력할 빈도 수
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--step_size", type=int, default=30) # StepLR의 step_size
    parser.add_argument("--gamma", type=float, default=1.0) # StepLR의 gamma
    
    ## Data loader
    parser.add_argument("--pin_memory", action='store_true') # 데이터를 CPU -> GPU로 옮길 때 사용할 memory(store_true: 언급 시 True를 저장)
    parser.add_argument("--num_workers", type=int, default=2) # DataLoder에서 사용할 CPU 코어 개수
    parser.add_argument("--shuffle", action='store_true') # DDP의 DistributedSampler에서 shuffle의 여부이며, Data Loader는 이와 반대로 지정(store_true: True를 저장)
    parser.add_argument("--imgsz", type=int, default=600) # RandomResizedCrop 시 crop output의 image size 지정

    ## Wandb
    parser.add_argument("--is_wandb", action='store_true')
    ## parser.add_argument("--project_name", type=str, default="ImageNet") # 나중에 프로젝트명을 더 아름답게 하기 위해 바꾸기
    parser.add_argument("--entity", type=str, default="classofficer") # 사용자명 or 팀명
    
    return parser.parse_args() # 입력받은 argument 리턴

def cleanup():
    dist.destroy_process_group() # 분산 학습 완료 후, 프로세스 초기화
    
def main(rank, world_size, args):
    torch.cuda.set_device(rank) # GPU의 각 프로세스 세팅
    
    train_loader, val_loader, train_len = pre_dset(rank, world_size, args) # 데이터 불러오기 및 전처리
    
    # 데이터가 흑백인지 컬러인지에 따라 입력채널을 다르게 함
    if args.data == 'mnist':
        input_channels = 1
    else:
        input_channels = 3
    
    # 모델 불러오기 및 적용하기
    if args.model == 'resnet':
        from model_layer.ResNet import ResNet, Bottleneck
        print('model: ResNet')
        model = ResNet(Bottleneck, [3,4,6,3], channels=input_channels).to(rank) # model 생성자
    elif args.model == 'vit':
        from model_layer.ViT import ViT
        print('model: ViT')
        model = ViT(image_size=args.imgsz, channels=input_channels).to(rank)
    elif args.model == 'xception':
        from model_layer.Xception import Xception
        print('model: Xception')
        model = Xception(num_classes=2, in_chans=3, drop_rate=0).to(rank)
    else:
        raise NotImplementedError(f'Your input model: {args.model}, please enter a valid model name(resnet, vit, xception)')
    model = DDP(model, device_ids=[rank], output_device=rank) # 병렬 처리를 위해 DDP에 model, process id를 넘겨줌   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # 최적화기법 및 learning rate 설정
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # learning rate를 step_size마다 gamma를 곱하여 감소시킴
    criterion = nn.CrossEntropyLoss()
    
    if args.is_wandb:
        # wandb.init(project=args.project_name, name=args.model, notes=' runed at ' + datetime.now().strftime('%Y-%m-%d %H:%M'), entity=args.entity)
        wandb.init(project=args.data, name=args.model, notes=' runed at ' + datetime.now().strftime('%Y-%m-%d %H:%M'), entity=args.entity)
        wandb.config.update(args)
        wandb.watch(model, criterion, log="all", log_freq=2)
    
    if rank == 0: print(f"Start Imagenet Training")
    best_acc, best_loss = 0., float("inf") # Best value 초기화 시 acc는 가장 낮은 값, loss는 무한으로 설정
    
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch) # epoch마다 DistributedSampler에게 현재 epoch을 계속 전달해야 함
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, rank, epoch, args, train_len) # Train
        val_acc, val_loss = validate(model, val_loader, criterion, rank, args) # validate
        if args.is_wandb:
            wandb.log({"train loss:": train_loss, "train acc": train_acc, "val loss": val_loss, "val acc": val_acc}) # logging to wandb
    
        ## reason of using ones_like: 
        ## the container's value should be on the same device with the value it will contain
        g_acc = [torch.ones_like(val_acc) for _ in range(world_size)]
        g_loss = [torch.ones_like(val_loss) for _ in range(world_size)]

        # DistributedDataParallel.all_gather: 모든 프로세스의 tensor 를 모든 프로세스의 tensor_list 에 복사
        dist.all_gather(g_acc, val_acc)
        dist.all_gather(g_loss, val_loss)

        if rank == 0: # 첫 번째 GPU process
            val_acc = torch.stack(g_acc, dim=0) # torch.stack: 새로운 차원에 Tensor를 붙임
            val_loss = torch.stack(g_loss, dim=0)
            val_acc, val_loss = val_acc.mean(), val_loss.mean()
            print(f"EPOCH {epoch} VALID: acc = {val_acc}, loss = {val_loss}") # EPOCH 당 accuracy 및 loss 출력
            if val_acc > best_acc: # best accuracy 찾기
                save_ckpt ({
                    "epoch": epoch+1, # 왜 1을 더해
                    "state_dict": model.module.state_dict(), # 학습된 모델 저장
                    "optimizer": optimizer.state_dict(), # optimizer 저장
                    "scheduler": scheduler.state_dict(), # scheduler 저장
                }, file_name=os.path.join(args.exp, f"best_acc.pth")) # best loss 저장
            if val_loss > best_loss: # best loss 찾기
                save_ckpt({
                    "epoch": epoch+1,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, file_name=os.path.join(args.exp, f"best_loss.pth"))
            save_ckpt({
                "epoch": epoch+1,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, file_name=os.path.join(args.exp, f"last.pth"))
        scheduler.step() # scheduler 업데이트
        # we do not need it when training, since DDP automatically does it for us (in loss.backward());
        # we do not need it when gathering data, since dist.all_gather_object does it for us;
        # we need it when enforcing execution order of codes, say one process loads the model that another process saves (I can hardly imagine this scenario is needed).
        dist.barrier() # 모든 프로세스가 동기화되도록 맞춰줌
    
    cleanup()
    
if __name__ == "__main__":
    args = parse_args() # argument 받기
    args.local_rank = int(os.environ['LOCAL_RANK']) # torchrun 사용을 위한 Local Rank 전달
    print(args)
    
    dist.init_process_group("nccl")
    
    if "./model_checkpoint" not in args.exp: # 입력 받은 경로에 해당 폴더가 없으면
        args.exp = os.path.join("/media/data1/kangjun/model_checkpoint", args.exp) # 경로를 이어붙임
    os.makedirs(args.exp, exist_ok=True) # exist_ok=True: 폴더가 없으면 자동 생성
    
    main(rank=args.local_rank, world_size=dist.get_world_size(), args=args) # main에 GPU의 process ID, process 수 및 args를 넘겨줌