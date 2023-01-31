import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch.optim as optim 
from tqdm import tqdm
import torch.nn as nn

import PracResNet

learning_rate = 0.001
num_epoch = 10
batch_size = 200

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = dset.ImageNet('../../../media/data1/data/Imagenet', split='train', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = dset.ImageNet('../../../media/data1/data/Imagenet', split='val', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:3" if torch.cuda.is_available() else None)
print('Now Device: ', device)

model = PracResNet.ResNet(PracResNet.Bottleneck, [3,4,6,3],).to(device)
print('My model:\n','#'*100,"\n",model)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

loss_arr = []

for i in range(num_epoch): # epoch만큼 학습 반복
    train_bar = tqdm(trainloader)
    running_results = {'batch_sizes':0, 'loss':0}
    model.train()
    for j,[image,label] in enumerate(train_bar): # train set으로 학습하기
        running_results['batch_sizes'] += batch_size
        x = image.to(device) # 입력 데이터를 GPU에 적재
        y_= label.to(device) # 라벨 데이터를 GPU에 적재
        
        optimizer.zero_grad() # 학습마다 기울기를 0으로 초기화
        output = model.forward(x) # 순전파한 후, 결과를 저장
        loss = loss_func(output,y_) # 예측값과 결과값의 loss를 구함
        loss.backward() # 가중치와 편향을 계산
        optimizer.step() # 가중치 업데이트
        
        running_results['loss'] += loss.item() * batch_size
        
        train_bar.set_description(desc = '[%d/%d] Loss: %.4f' % (
            i, num_epoch, running_results['loss'] / running_results['batch_sizes'],
        ))
#         if j % 1000 == 0:
#             print(i, ":", loss)
#             loss_arr.append(loss.cpu().detach().numpy())