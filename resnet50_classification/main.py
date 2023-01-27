import model as MD

import os 
import torch 

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group #initialize

from torch.utils.data import Dataset, DataLoader
import argparse

import torchvision.transforms as tr
#import util
from dataset import Imagenet

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from torchvision.datasets import ImageFolder

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



def ddp_setup(rank, world_size):
    #world size == total num of processes in a group
    #rank == identifier assigned to each process

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = ""
    init_process_group(backend="nccl",rank=rank, world_size=world_size)



# GPU Device 

gpu_id = '2' 

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) 

use_cuda = torch.cuda.is_available() 

print("GPU device " , use_cuda) 

device = torch.device('cuda' if use_cuda else 'cpu')

from PIL import Image
import matplotlib.pyplot as plt

# import matplotlib.image as mpimg
# import numpy as np
# img = mpimg.imread('/home/data/Imagenet/train/n01440764/n01440764_18.JPEG')
# plt.imshow(np.asarray(img))
# plt.show()


def main():
    parser = argparse.ArgumentParser(description='argparse')

    # dataset, model, batch size, epoch, learning rate
    parser.add_argument('--train', '-tr', required=False, default='/home/data/Imagenet/train', help='Root of Trainset')
    parser.add_argument('--test', '-ts', required=False, default='/home/data/Imagenet/validation', help='Root of Testset')
    parser.add_argument('--model', '-m', required=False, default='resnet50', help='Model')
    parser.add_argument('--batch', '-b', required=False, default=32, help='Batch Size')
    parser.add_argument('--epoch', '-e', required=False, default=50, help='Epoch')
    parser.add_argument('--lr', '-l', required=False, default=0.001, help='Learning Rate')

    args = parser.parse_args()

        # 입력받은 인자값 출력
    print(args.train)
    print(args.test)
    print(args.model)
    print(args.batch)
    print(args.epoch)
    print(args.lr)

    train_root = args.train
    test_root = args.test

    if args.model == 'resnet50':
        model = MD.ResNet50('Imagenet')

    # Data transforms (normalization & data augmentation)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transf_train = tr.Compose([tr.ToTensor(), tr.RandomCrop(32, padding=4, padding_mode='reflect'), #양 끝에 반사된 값 사용
                               tr.RandomHorizontalFlip(), tr.Normalize(*stats, inplace=True)]) #inplace = 변경된 스탯으로 덮어씀
    transf_test = tr.Compose([tr.ToTensor(), tr.Normalize(*stats, inplace=True)])

    trainset = ImageFolder('/home/data/Imagenet/train', transform=transf_train,target_transform=None)
    testset = ImageFolder('/home/data/Imagenet/validation', transform=transf_test)

    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False) 

    criterion = nn.CrossEntropyLoss() #crossentropy loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) #loss 계산시 gradient descent 를 미니배치를 이용함
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)
    #loss 향상하지 않으면 lr 을 factor배로 감소(0.5), 3epoch동안, threhold란 중요변화에만 초점 맞추기 위한 임계값

if __name__ == "__main__":
    main()