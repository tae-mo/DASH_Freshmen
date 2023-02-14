import torch
import torch.nn as nn
from torch import Tensor
'''
필요한 레이어
1. 3x3 conv ==> function
2. 1x1 conv ==> function
3. 7x7 conv(stride 2)
4. 3x3 maxpool(stride 2)
5. ReLU
6. Batch normalization
7. average pool
8. 1000 FC(클래스가 1000개이므로)
9. softmax

클래스 구현
1. BasicBlock
2. Bottleneck
3. ResNet
'''

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d: # '->'는 리턴의 형식을 표시하기 위한 주석
    return nn.Conv2d(
        in_planes, # 입력 필터개수
        out_planes, # 출력 필터개수
        kernel_size=3, # 필터 사이즈
        stride=stride, # 스트라이드
        padding=dilation, # 패딩
        groups=groups, # input과 output의 connection을 제어
        bias=False, # 출력에 bias를 추가(디폴트는 True)
        dilation=dilation, # 커널 원소간의 거리, 클수록 더 넓은 범위
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )

class BasicBlock(nn.Module):
    expansion: int = 1 # 차원 증가 시 확장 계수
    def __init__( # 생성자
        self,
        inplanes: int, # 입력 채널 수
        planes: int, # 출력 채널 수
        stride: int = 1,
        downsample = None, # downsampling 
        groups: int = 1, # input과 output의 connection을 제어
        base_width: int = 64,
        dilation: int = 1, # 커널 원소간의 거리, 클수록 더 넓은 범위
        norm_layer = None, # 배치 정규화 레이어 지정
    ) -> None: # 리턴 없음
        super(BasicBlock, self).__init__() # 부모 클래스인 nn.Module을 초기화
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 배치 정규화를 하여 스케일 및 시프트 연산
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64 Okey?") # 근데 이럴거면 처음부터 파라미터를 안받고 값을 fixed 시키면 안되나?
        if dilation > 1:
            raise NotImplementedError("dilation >1 not supported in BasicLbock")
        
        # 지역변수
        self.conv1 = conv3x3(inplanes, planes, stride) # 3x3 convolution, ver. stride
        self.bn1 = norm_layer(planes) # batch_norm
        self.relu = nn.ReLU(inplace=True) # act_fnc
        self.conv2 = conv3x3(planes, planes) # 3x3 convolution, ver
        self.bn2 = norm_layer(planes) # batch_norm
        self.downsample = downsample # downsampling
        self.stride = stride
        
    def forward(self, x: Tensor) -> Tensor: # 입력 및 리턴: Tensor
        identity = x
        
        # 3x3
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None: # downsample을 지정한 경우
            identity = self.downample(x)
            
        out += identity # input 값과 output 값을 더함(residual learning)
        
        return self.relu(out) # 마지막으로 활성화 함수를 통과한 후 넘겨 줌
        
        
class Bottleneck(nn.Module):
    expansion: int = 4 # 차원 증가 시 확장 계수
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1, # input과 output의 connection을 제어
        base_width: int = 64,
        dilation: int = 1, # 커널 원소간의 거리, 클수록 더 넓은 범위
        norm_layer = None,
    ) -> None:
        super(Bottleneck, self).__init__() # 부모 클래스인 nn.Module을 초기화
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 배치 정규화를 하여 스케일 및 시프트 연산
        width = int(planes * (base_width / 64.0)) * groups # 무슨 식이징....
        
        self.conv1 = conv1x1(inplanes, width) # 1x1 convolution
        self.bn1 = norm_layer(width) # batch_norm
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # 3x3 convolution -> downsample
        self.bn2 = norm_layer(width) # batch_norm
        self.conv3 = conv1x1(width, planes * self.expansion) #1 x1 convolution -> 채널 수 복구
        self.bn3 = norm_layer(planes * self.expansion) # batch_norm
        self.relu = nn.ReLU(inplace=True) # act_fnc
        self.downsample = downsample # downsampling
        self.stride = stride
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        # 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity # input 값과 output 값을 더함(residual learning)
        
        return self.relu(out)
       
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers, # layer를 int로 이루어진 list로 받음
        num_classes: int = 1000,
        zero_init_residual: bool = False, # 잔차 분기의 마지막 BN을 0으로 초기화할지 여부
        groups: int = 1, # input과 output의 connection을 제어
        width_per_group: int = 64, # Width of residual blocks
        replace_stride_with_dilation = None, # stride를 dilation으로 대체할지
        norm_layer = None,
        channels = 3,
    )-> None: # 지금 생각해보니 init은 웬만하면 리턴을 안했었지
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 배치 정규화를 하여 스케일 및 시프트 연산
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}" # f-string: 문자열 안에 중괄호를 사용하여 변수 값을 삽입 가능
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # input layer
        self.bn1 = norm_layer(self.inplanes) # batch_norm
        self.relu = nn.ReLU(inplace=True) # act_func
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 입력층에서 은닉층으로 가기 전 max_pool
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules(): # 가중치 초기화
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # He 초기값
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1) # weight을 1로
                nn.init.constant_(m.bias, 0) # bias를 0으로
        
        # 잔차 분기의 마지막 BN을 0으로 초기화 한다는 데.... 몰라서 패스
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer( # 넘겨받은 파라미터에 맞게 레이어 생성
        self,
        block, # basic or bottle
        planes: int, # input
        blocks: int, # layer 깊이
        stride: int = 1, # stride는 conv3_x, conv4_x, conv5_x가 2로 주어짐
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer # batch_norm
        downsample = None # down_sampling
        previous_dilation = self.dilation # 커널 원소간의 거리, 클수록 더 넓은 범위
        if dilate:
            self.dilation *= stride # dilate를 위해 stride마다 dilation 적용하기 위한 연산
            stride = 1 # 연산을 하였기 때문에 1로 바꿔줌
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            
        layers = [] # 해당 Bottleneck에 맞는 레이어를 만듦
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
