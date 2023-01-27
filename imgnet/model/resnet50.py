import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.expansion = 4
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        
        # 하위 layer로 내려갈때 downsampling(1/2줄이기)하면서 feature 수가 달라짐 그러므로 skip connection을 할 수 있게 보정
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    def forward(self, x):
        out = self.residual_function(x)
        out = out + self.shortcut(x)
        return self.relu(out)
    

class ResNet(nn.Module):
    def __init__(self, block, layers=[3,4,6,3], in_channels=3, num_classes=1000):
        self.num_classes = num_classes
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        )        
        self.conv2 = self._make_layer(block, 64, layers[0])
        self.conv3 = self._make_layer(block, 128, layers[1], downsampling=True)
        self.conv4 = self._make_layer(block, 256, layers[2], downsampling=True)
        self.conv5 = self._make_layer(block, 512, layers[3], downsampling=True)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, downsampling=False):
        if downsampling is True:
            stride = 2
        else:
            stride = 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
