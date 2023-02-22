import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1
        ):
        super(SeparableConv2d, self).__init__()
        
        self.conv1 = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=in_channels,
                bias=False
            )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
    
class Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            reps,
            strides=1,
            start_with_relu=True,
            grow_first=True
        ): # reps: SeparableConv를 만들 개수,  
        super(Block, self).__init__()
        
        if out_channels != in_channels or strides != 1: # Skip Connection이 있어야 한다면
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
            
        rep = []
        for i in range(reps): # 모듈 생성
            if grow_first: # Exit flow의 first module를 제외한 나머지 module
                inc = in_channels if i == 0 else out_channels # 첫 reps이면 입력 채널을 in_channels로, 그외는 입력 채널을 out_channels 
                outc = out_channels
            else: # Exit flow의 first module
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels # reps의 마지막인 경우 출력 채널을 out_channels로 지정
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))
        
        if not start_with_relu: # start_with_relu가 False인 경우, 모듈 시작 시 ReLU를 무시
            rep = rep[1:]
        else: # start_with_relu가 True인 경우, 모듈 시작 시 ReLU를 사용
            rep[0] = nn.ReLU(inplace=False)
            
        if strides != 1: # MaxPooling 필요한 모듈에 삽입
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)
        
    def forward(self, inp):
        x = self.rep(inp) # ReLU, SeparableConv, MaxPooling
        
        if self.skip is not None: # skip을 정의하였으면
            skip = self.skip(inp)
            skip = self.skipbn(skip) # Skip의 BatchNorm
        else:
            skip = inp
            
        x += skip
        return x
        
class Xception(nn.Module):
    def __init__(self, num_classes=2, in_chans=3, drop_rate=0.):
        super(Xception, self).__init__()
        
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        # in_channels, outchannels, rps, strides, start_with_relu, grow_first
        self.entry_flow = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 2, 0 , bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            Block(64, 128, 2, 2, start_with_relu=False,),
            Block(128, 256, 2, 2),
            Block(256, 728, 2, 2,),
        )
        
        middle_layer = []
        for i in range(8):
            middle_layer.append(Block(728, 728, 3, 1,))
        self.middle_flow = nn.Sequential(*middle_layer)
        
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
        self.exit_flow = nn.Sequential(
            Block(728, 1024, 3, 2, grow_first=False),
            
            SeparableConv2d(1024, 1536, 3, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            
            SeparableConv2d(1536, 2048, 3, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Linear(2048, self.num_classes)
        
    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = torch.flatten(x, 1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x