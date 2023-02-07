import torchvision.transforms as tr
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


def dataloader(batch_size: int, args):
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #imagenet mean std 

    #transf_train = tr.Compose([tr.Resize((224, 224)),tr.ToTensor(),tr.RandomHorizontalFlip(), tr.Normalize(*stats, inplace=True)])

    transforms = tr.Compose([
        tr.Resize((224, 224)),
        tr.RandomHorizontalFlip(),
        #tr.RandomVerticalFlip(),
        tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        tr.ToTensor(),
        tr.Normalize(*stats, inplace=True)])

    #train_set = torchvision.datasets.FakeData(1000, (3, 224, 224), 10, transform=transforms,target_transform=None) #dummy
    #test_set = torchvision.datasets.FakeData(200, (3, 224, 224), 10, transform=transforms,target_transform=None) #dummy

    train_set = ImageFolder(args.train, transform=transforms,target_transform=None)
    test_set = ImageFolder(args.test, transform=transforms,target_transform=None)


    train_sampler=DistributedSampler(dataset=train_set,shuffle=True) #Default true
    test_sampler=DistributedSampler(dataset=test_set,shuffle=False) 

    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size, 
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  sampler=train_sampler)
    test_dataloader = DataLoader(test_set,
                                  batch_size=batch_size, 
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  sampler=test_sampler)

    return train_dataloader, test_dataloader
   
