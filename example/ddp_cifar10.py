import os
# import sys
# import tempfile
import torch
import argparse
import torchvision
# import imagenetDataSet
# import csv
import random
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import tqdm
# import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# from .resnet import ResNet


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '115.145.134.134'
    os.environ['MASTER_PORT'] = '22'

    # 작업 그룹 초기화
    dist.init_process_group("nvcc", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
'''    
def train_model(rank, train_loader, model, optim, criterion, device, epoch, batch_size, world_size):
    
    model.trian()
    
    if epoch == 0:
        print("start train {}".format(len(train_loader)))
        
    accuracies = []
    losses = []
    
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, world_size)
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
#
'''
def parse_args():
    parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
    parser.add_argument("--local_rank", '-r', type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--epoch', '-e', type=int, default=15)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    parser.add_argument('--random_seed', '-s', default=0, type=int)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default="saved_models")
    parser.add_argument("--model_filename", type=str, help="Model filename.", default="resnet_dist")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    
    # parser.add_argument('--train', '-t', default=True)
    # parser.add_argument('--project_name', '-n', default='crack_seg')
    # parser.add_argument('--data_loader', '-d', default='mini')
    # parser.add_argument('--no_cuda', '-c', default=False)
    # parser.add_argument('--keep', '-k', default='not_use_best')
    # parser.add_argument('—save_model', '-s', action='store_true', default=False) lr

    return parser.parse_args()


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def main():
    
    args = parse_args()
    local_rank = args.local_rank
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    random_seed = args.random_seed
    model_dir = args.model_dir
    model_filename = args.model_filename
    resume = args.resume
    
    # Create directories outside the PyTorch program
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    
    set_random_seeds(random_seed=random_seed)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if device_id == 0:
    #     print    
    
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    print(dist.is_available())
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=2)
    
    model = torchvision.models.resnet50(pretrained=False)
    
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    
    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))
        
    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform) 
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    
    
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
        
        # Save and evaluate model routinely
        if local_rank == 0:
            accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
            torch.save(ddp_model.state_dict(), model_filepath + '_' + epoch + ".pth")
            print("-" * 75)
            print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
            print("-" * 75)

        ddp_model.train()

        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description(f'Train Epoch {epoch}')
        
        for idx, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tq.update(batch_size)
        tq.close()
main()