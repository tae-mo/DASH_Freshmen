
import os
import matplotlib.image as img
from torch.utils.data import Dataset, DataLoader
import torch
import glob
from PIL import Image
from torchvision import transforms
import torchvision.transforms as tr
from torch.utils.data.distributed import DistributedSampler


class customDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        if train:
            self.fake_path = path + '/train/fake'
            self.real_path = path + '/train/real'
        else:
            self.fake_path = path + '/test/fake'
            self.real_path = path + '/test/real'
        #"/home/data/deepfake_privacy/FF_original/FaceForensics++/DeepFake/train/fake/000_003/"
        self.fake_img_list = glob.glob(self.fake_path + '/*/*.png')
        self.real_img_list = glob.glob(self.real_path + '/*/*.png')

        self.transform = transform

        self.img_list = self.fake_img_list + self.real_img_list

        self.Image_list = []  # 바뀐부분!!!!!!!
        for img_path in self.img_list:  # 바뀐부분!!!!!!!
            self.Image_list.append(Image.open(img_path))  # 바뀐부분!!!!!!!
        
        # print(len(self.Image_list))

        # exit()

        self.class_list = [0] * len(self.fake_img_list) + [1] * len(self.real_img_list) 

        #print(len(self.class_list))
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.Image_list[idx]  # 바뀐부분!!!!!!!
        #label = self.class_list[idx]
        label = self.class_list[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def dataloader(batch_size: int, args):
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #imagenet mean std 

    #transf_train = tr.Compose([tr.Resize((224, 224)),tr.ToTensor(),tr.RandomHorizontalFlip(), tr.Normalize(*stats, inplace=True)])

    transforms = tr.Compose([
        tr.Resize((299, 299)),
        tr.RandomHorizontalFlip(),
        #tr.RandomVerticalFlip(),
        #tr.ColorJitter(),
        tr.ToTensor(),
        tr.Normalize(*stats, inplace=True)])

    #train_set = torchvision.datasets.FakeData(1000, (3, 224, 224), 10, transform=transforms,target_transform=None) #dummy
    #test_set = torchvision.datasets.FakeData(200, (3, 224, 224), 10, transform=transforms,target_transform=None) #dummy

    train_set = customDataset(args.train, train=True, transform=transforms)
    test_set = customDataset(args.test, train=False, transform=transforms)
                       

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
   

# if __name__ == "__main__":
#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )

#     dataset = customDataset(path="/home/data/deepfake_privacy/FF_original/FaceForensics++/DeepFake", train=True, transform=transform)
#     dataloader = DataLoader(dataset=dataset,
#                         batch_size=1,
#                         shuffle=True,
#                         drop_last=False)

#     for epoch in range(2):
#         print(f"epoch : {epoch} ")
#         for batch in dataloader:
#             img, label = batch
#             # print(img.size(), label)        