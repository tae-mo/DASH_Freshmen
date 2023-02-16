# import package
import torch
import torch.nn as nn


# Depthwise Separable Convolution
class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seperable = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.seperable(x)
        return x
    
#! 1 EnrtyFlow
class EntryFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2_residual = nn.Sequential(
            SeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv2_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.conv3_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeparableConv(256, 256),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, padding=0),
            nn.BatchNorm2d(256)
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(256, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv(728, 728),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1, stride=2, padding=0),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_residual(x) + self.conv2_shortcut(x)
        x = self.conv3_residual(x) + self.conv3_shortcut(x)
        x = self.conv4_residual(x) + self.conv4_shortcut(x)
        return x
    
#! 2 MiddleFlow
class MiddleFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(728, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv(728, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv(728, 728),
            nn.BatchNorm2d(728)
        )

        self.conv_shortcut = nn.Sequential()

    def forward(self, x):
        return self.conv_shortcut(x) + self.conv_residual(x)
 
#! 3 ExitFlow
class ExitFlow(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(728, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            SeparableConv(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv1_shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2, padding=0),
            nn.BatchNorm2d(1024)
        )

        self.conv2 = nn.Sequential(
            SeparableConv(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeparableConv(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        x = self.conv1_residual(x) + self.conv1_shortcut(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        return x
    
# Xception
class Xception(nn.Module):
    def __init__(self, num_classes=2, init_weights=True):
        super().__init__()
        self.init_weights = init_weights

        self.entry = EntryFlow()
        self.middle = self._make_middle_flow()
        self.exit = ExitFlow()

        self.linear = nn.Linear(2048, num_classes)

        # weights initialization
        if self.init_weights:
            # self._initialize_weights()
            pass


    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_middle_flow(self):
        middle = nn.Sequential()
        for i in range(8):
            middle.add_module('middle_block_{}'.format(i), MiddleFlow())
        return middle

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init_kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init_constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init_constant_(m.weight, 1)
                nn.init_bias_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init_normal_(m.weight, 0, 0.01)
                nn.init_constant_(m.bias, 0)