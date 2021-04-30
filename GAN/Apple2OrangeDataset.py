import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset


class Apple2OrangeDataset(Dataset):

    def __init__(self, train_or_test=True, apple_or_orange=True):
        split_path = 'train' if train_or_test else 'test'
        domain_path = 'A' if apple_or_orange else 'B'
        base_path = './dataset/' + split_path + domain_path + '/'
        self.imgs = [base_path + file_name for file_name in os.listdir(base_path)]
        self.len = len(self.imgs)
        self.transforms = torchvision.transforms.Compose([
            lambda file_path: Image.open(file_path).convert('RGB'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将取值范围变换到[-1,+1]
        ])

    def __getitem__(self, index):
        img = self.transforms(self.imgs[index])
        return img

    def __len__(self):
        return self.len
