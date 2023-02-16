# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from model.xception import Xception as XC

# dataset and transformation
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt
#%matplotlib inline

# utils
import numpy as np
from torchsummary import summary
import time
import copy

df_dir = '/media/data2/eunju/df_datasets/'


class Trainer():
    def __init__(self, loader:DataLoader, model_type: str, save_every: int, lr:float):
        self.loader = loader
        self.save_every=save_every
        
        if model_type == "xception":
            self.model = XC(num_classes=2)
        
        
if not os.path.exists(df_dir):
    print("where")
else : 
    print("data found")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(3, 3, 299, 299).to(device)
    model = XC(num_classes=2).to(device)
    output = model(x)
    print('output size:', output.size())
    
