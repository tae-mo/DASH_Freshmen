import os
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms

from model.resnet import *

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
    # 확률값이 가장 높았던 클래스와 레이블의 실측값을 비교해서 불리언 배열을 얻고, 
    # 예측값이 실측값에 맏은 경우가 배치에서 얼마나 나왔는 지를 세어 합한다. 
    # (y_pred == y_true).to(torch.float).mean()
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
        os.mkdir(dir)
    return os.path.exists(dir), dir
#

def prepare_dataloader(dataset: Dataset, batch_size: int, scale=1):
    origin_sz = int(len(dataset))
    use_sz = int(origin_sz* scale)
    dataset, _ = random_split(dataset, [use_sz, origin_sz-use_sz])
    train, test = random_split(dataset, [int(use_sz*0.8), use_sz-int(use_sz*0.8)])
    return DataLoader(
        train,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train) # Sampler that restricts data loading to a subset of the dataset
    ), DataLoader(
        test,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test) # Sampler that restricts data loading to a subset of the dataset
    )
# end of prepare_dataloader ft



def load_train_objs(model_depth, mode='train'):
    assert model_depth in [0, 18, 34, 50, 101, 152], f'ResNet{model_depth}: Unknown architecture!'
    
    data_path = "/home/data/Imagenet"
    traindir = os.path.join(data_path, mode)
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
    
    if model_depth == 0 :
        model = models.__dict__['resnet50'](pretrained=True)
    else : 
        model = myResnet(num_layers=model_depth, block=Block)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    return train_dataset, model, optimizer
# end of load_train_objs ft



def print_once(gpu_id, message):
    if gpu_id == 0:
        print(message)
#









