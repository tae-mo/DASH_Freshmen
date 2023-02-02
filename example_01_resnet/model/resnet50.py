
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np 

# n_out = ((n_in + 2p - k) / s) + 1


#######################################################################################
# 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = self._conv(in_channels=in_channels, out_channels=out_channels, kernel_siz=3, stride=stride, padding=1)
        self.conv2 = self._convReLU(in_channels=in_channels, out_channels=out_channels, kernel_siz=3, stride=1, padding=1)

        #skip connection : optional (downsample)
        self.downsample = downsample
        
        self.relu = nn.ReLU()
        self.out_channels = out_channels
    #
    
    def _conv(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                            nn.BatchNorm2d(out_channels)
                            )
        return conv

    def _convReLU(self, in_channels, out_channels, kernel_size, stride, padding):
        conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                            )
        return conv1

    def forward(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        #The block (as shown in the architecture) contains a skip connection that is an optional parameter ( downsample )
        if self.downsample:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out
    #
# end of ResidualBlock

# Resnet 50 architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        # (3, 224, 224) >> (3, 112,  112)
        self.conv1 = self._convReLU(3, 64, kernel_siz=7, stride=2, padding=3)       # 1
        
        # (3, 112,  112) >> (3, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)       # 0 + 1 = 1
        
        # (3, 56, 56) >> (64, 56, 56)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)            # 3*3 + 1 = 10
        # (64, 56, 56) >> (128, 28, 28)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)           # 3*4 + 10 = 22
        # (128, 28, 28) >> (256, 14, 14)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)           # 3*6 + 22 = 40
        # (256, 14, 14) >> (512, 7, 7)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)           # 3*3 + 40 = 49
        
        self.avgpool = nn.AvgPool2d(7, stride=1)                                    # 0 + 49 = 49
        
        self.fc = nn.Linear(512, num_classes)                                       # 1 + 49 = 50
    #
    
    def _convReLU(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                            )
        return conv

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        # 
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    #
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    #
# end of ResNet
