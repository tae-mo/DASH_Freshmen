import os
import matplotlib.image as img
from torch.utils.data import Dataset
import torch

def match_image(root):
    images = []
    a = os.listdir(root)

    if '.DS_Store' in a: #예외처리
        a.remove('.DS_Store')

    for i, label in enumerate(a):
        for j in os.listdir(os.path.join(root, label)):
            image = img.imread(os.path.join(root, label, j))
            images.append((i, image))  # 이미지를 어펜드할 필요 없음. 수정 필요

    print("finished loading dataset")

    return images


class Imagenet(Dataset):
    def __init__(self, root, transform=None):
        super(Imagenet, self).__init__()
        self.root = root
        self.labels = os.walk(root).__next__()[1]
        self.images = match_image(root)
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        label, image = self.images[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)