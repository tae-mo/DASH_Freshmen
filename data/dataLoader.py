import torch
import os
import glob
import torchvision.transforms as tf
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader



def ImageNetDataLoader(split='train', batch_size=16, num_workers = 2):
    path = '/home/data/Imagenet/' + split
    # classes, class_to_idx = self._find_classes(self.path)
    # samples, file_names = make_dataset(self.root, class_to_idx, extensions, is_valid_file)

    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    
    train_transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean, std),
        tf.RandomHorizontalFlip(0.5),  # 좌우반전 
        tf.RandomVerticalFlip(0.5),    # 상하반전 
        tf.Resize((1024, 1024))
    ])
    
    test_transforms = tf.Compose([
        tf.ToTensor(),
        tf.Resize((1024, 1024))
    ])
    
    if split == 'train' : 
        imageNetDatasets = datasets.ImageFolder(
            root=path,
            transform=train_transforms
        )
    else :
        imageNetDatasets = datasets.ImageFolder(
            root=path,
            transform=test_transforms
        )
    
    dataLoader = DataLoader(imageNetDatasets, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataLoader




'''
class ImageNetDataSet(Dataset):
    def __init__(self, split = 'train'):
        super().__init__()
        self.path = '/home/data/Imagenet/' + split
        classes, class_to_idx = self._find_classes(self.path)
        # samples, file_names = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2023, 0.1994, 0.2010]
        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean, std),
            tf.RandomHorizontalFlip(0.5),  # 좌우반전 
            tf.RandomVerticalFlip(0.5),    # 상하반전 
            tf.Resize((1024, 1024))
        ])
        
        self.trainSet = datasets.ImageFolder(
            root=self.path,
            transform=self.transforms
        )
    #
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        

# end of DataSet




trainset = ImageNetDataSet(split='train')
print(len(trainset))'''