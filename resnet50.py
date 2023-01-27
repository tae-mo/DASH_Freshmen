import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion),
        )
        self.shortcut=nn.Sequential()
        self.relu = nn.ReLU()



class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        



class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        
