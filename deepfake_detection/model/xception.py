import torch
import torch.nn as nn


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, strides=1, padding=0, dilation=1, bias=False):
        super(SeparableConv, self).__init__()
        
        # https://gaussian37.github.io/dl-pytorch-conv2d/ dilation, groups 설명
        self.pointwiseconv = nn.Conv2d(in_channels, in_channels, kernel_size, strides, padding, dilation, groups=in_channels, bias=bias)
        self.depthwiseconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
    
    def forward(self,x):
        x = self.pointwiseconv(x)
        x = self.depthwiseconv(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, repeat, strides=1, start_with_relu=True, sizeup_first=True):
        super(Block, self).__init__()
        
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False), # bias를 False를 두는 이유???
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        layers = []
        
        channels = in_channels
        if sizeup_first:
            layers.append(nn.ReLU(inplace=True)) # inplace=True 하면, inplace 연산을 수행함, inplace 연산은 결과값을 새로운 변수에 값을 저장하는 대신 기존의 데이터를 대체하는것을 의미 메모리적 이득
            layers.append(SeparableConv(in_channels, out_channels, kernel_size=3, strides=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            channels = out_channels
            
        for i in range(repeat-1):
            layers.append(nn.ReLU(inplace=True)) # 근데 왜 굳이 해야 해? 안하면 안 될 정도?
            layers.append(SeparableConv(channels, channels, kernel_size=3, strides=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            
        if not sizeup_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(channels, out_channels, kernel_size=3, strides=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            
        if not start_with_relu:
            layers = layers[1:]
        else:
            layers[0] = nn.ReLU(inplace=True) # 다시 한번 확실하게 ReLU로 시작하게 하자
                    
        if strides != 1:
            layers.append(nn.MaxPool2d(3, strides, 1)) # padding을 1로 둠으로서 image shape을 논문과 같이 19X19로 만듦. 근데 self.skip에서 18X18로 나오는거 같은데 어케되는거지
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, img):
        x = self.layers(img)
        x = x + self.skip(img)
        
        return x
    
class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()
        
        self.num_classes = num_classes
        
        # Entry flow
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.module2 = Block(64, 128, 2, 2, False, True)
        self.module3 = Block(128, 256, 2, 2, True, True)
        self.module4 = Block(256, 728, 2, 2, True, True)
        
        # Middle flow
        self.module5_12 = nn.Sequential(*[Block(728,728,3,1,True,True) for _ in range(8)])
        
        # Exit flow
        self.module13 = Block(728, 1024, 2, 2, True, False)
        self.module14 = nn.Sequential(
            SeparableConv(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # classifier
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, img):
        x = self.module1(img)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5_12(x)
        x = self.module13(x)
        x = self.module14(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x