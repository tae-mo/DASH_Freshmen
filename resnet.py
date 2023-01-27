
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np 

# n_out = ((n_in + 2p - k) / s) + 1

class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        #skip connection : optional (downsample)
        self.downsample = downsample
        # self.num_layers = num_layers
        self.expansion = 4 if num_layers > 34 else 1 # ResNet50, 101, and 152 include additional layers of 1x1 kernels
        self.relu = nn.ReLU()
        
        # 1x1 kernels - front part of bottleneck block
        self.conv_pre = self._convReLU(in_channels, out_channels, kernel_siz=1, stride=1, padding=0)
        
        # 3x3 kernels
        if self.num_layers > 34:    # for ResNet50, 101, connect input  to  first (1x1)kernel then 
            self.conv33 = self._convReLU(out_channels, out_channels, kernel_siz=3, stride=1, padding=1)
        else:                       # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv33 = self._convReLU(in_channels, out_channels, kernel_siz=3, stride=stride, padding=1)
        
        # 1x1 kernels - back part of bottleneck block
        self.conv_post = self._conv(out_channels, out_channels * self.expansion, kernel_siz=1, stride=1, padding=0)
    #
        
    def _conv(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                nn.BatchNorm2d(out_channels))
        return conv
    #

    def _convReLU(self, in_channels, out_channels, kernel_size, stride, padding):
        conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU() )
        return conv1
    #
    
    def forward(self, x):
        identity = x                # original source
        if self.num_layers > 34: 
            x = self.conv_pre(x)    # 1x1 kernels
        x = self.conv33(x)          # 3x3 kernels
        x = self.conv_post(x)       # 1x1 kernels

        #The block (as shown in the architecture) contains a skip connection that is an optional parameter ( downsample )
        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity               # skip connet
        x = self.relu(x)
        
        return x
    #
    
# end of ResidualBlock ################################################################################################

class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture!'
        super(ResNet, self).__init__()
        self.expansion = 4 if num_layers > 34 else 1 # ResNet50, 101, and 152 include additional layers of 1x1 kernels
        
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
            
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self._make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)       # 7*7, 64, stride2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 3*3 maxpool, stride2

        x = self.layer1(x)      # conv layer 1 (3, 56, 56) >> (64, 56, 56)
        x = self.layer2(x)      # conv layer 2 (64, 56, 56) >> (128, 28, 28)
        x = self.layer3(x)      # conv layer 3 (128, 28, 28) >> (256, 14, 14)
        x = self.layer4(x)      # conv layer 4 (256, 14, 14) >> (512, 7, 7)
        
        x = self.avgpool(x)     # 
        x = x.reshape(x.shape[0], -1)   # x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
            nn.BatchNorm2d(intermediate_channels*self.expansion)
            )
        
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)