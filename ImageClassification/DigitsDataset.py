import matplotlib.pyplot as plt
import pandas as pd
import torchvision
from torch.utils.data import Dataset


class DigitsDataset(Dataset):

    def __init__(self, is_train=True):
        path = './dataset/' + ('labelsTrain.csv' if is_train else 'labelsTest.csv')
        xy = pd.read_csv(path)  # 使用numpy读取数据
        self.file = xy.loc[:, 'file']
        self.label = xy.loc[:, 'label']
        self.len = xy.shape[0]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        label = self.label[index]
        img = self.transforms(plt.imread(self.file[index]))
        return img, label

    def __len__(self):
        return self.len
