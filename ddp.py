import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from .resnet import ResNet
import imagenetDataSet


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '115.145.134.134'
    os.environ['MASTER_PORT'] = '22'

    # 작업 그룹 초기화
    dist.init_process_group("nvcc", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
    
def train_model(rank, args):
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, args.world_size)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    # instantiate the model and transfer it to the GPU
    model = ResNet(classes=1000).to(rank)
    
    # wraps the network around distributed package
    model = DDP(model, device_ids=[rank])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Preparing the training data
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    
    transforms_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    training_set = Dataset(root='./data', train=True, transform=transforms_train)

    # torch.distributed's own data loader method which loads the data such that they are non-overlapping and
    # exclusive to each process
    train_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=training_set,
                                                                         num_replicas=args.world_size, rank=rank)
    trainLoader = torch.utils.data.DataLoader(dataset=training_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True,
                                              sampler=train_data_sampler)

    # Preparing the testing data
    transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    testing_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)

    test_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testing_set,
                                                                        num_replicas=args.world_size, rank=rank)
    testLoader = torch.utils.data.DataLoader(dataset=testing_set, batch_size = args.batch_size,
                                             shuffle = False, num_workers=4, pin_memory=True,
                                             sampler=test_data_sampler)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Training
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        accuracy = 0
        total = 0
        for idx, (inputs, labels) in enumerate(trainLoader):
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            total += labels.size(0)
            _, prediction = outputs.max(1)
            accuracy += prediction.eq(labels).sum().item()

        if rank == 0:
            print("Epoch: {}, Loss: {}, Training Accuracy: {}". format(epoch+1, loss.item(), accuracy/total))

    print("Training DONE!!!")
    print()
    print('Testing BEGINS!!')

    # Testing
    test_loss, test_acc, total = 0, 0, 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testLoader):
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, prediction = outputs.max(1)
            total += labels.size(0)
            test_acc += prediction.eq(labels).sum().item()

    # this condition ensures that processes do not trample each other and corrupt the files by overwriting
    if rank == 0:
        print("Loss: {}, Testing Accuracy: {}".format(loss.item(), test_acc / total))
        # Saving the model
        testAccuracy = 100*test_acc/total
        state = {'model': model.state_dict(), 'test_accuracy': testAccuracy, 'num_epochs' : args.n_epochs}
        if not os.path.exists('./models'):
            os.mkdir('./models')
        torch.save(state, './models/cifar10ResNet101.pth')

    cleanup()