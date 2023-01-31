import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = len(data)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]